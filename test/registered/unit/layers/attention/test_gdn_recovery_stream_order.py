"""CPU ordering coverage for the gdn_mtp_cache_mode=none recovery stream.

RecoverSSM commits the accepted SSM state from a side stream so the recovery hides
behind the next step's draft compute. Everything in this file is about who has to
wait for that writer, on which stream, and when:

  * the HiCache L2 legs (``cache_controller.start_writing`` / ``start_loading``,
    which is the ``hybrid_cache_controller.write`` -> ``l2_transfer`` ->
    ``pool_host/mamba.backup`` path) run on their own streams and fence on
    ``write_fence_stream`` == the model runner's forward stream. The side-stream
    recovery is NOT ordered into that stream until the next target forward plans its
    metadata, so a backup submitted earlier would read -- and publish, under a valid
    key -- rows its own recovery is still writing. The backend therefore publishes
    its completion event as an extra producer, and the legs join it at submit time,
    ahead of the submission whose finish event is the ACK the host publication and
    slot free already wait on.
  * a deferred device CoW/clear runs on the forward stream, and on the eager path
    BEFORE the forward plans its metadata (``ModelRunner._forward_raw``), so the join
    has to sit at that call site and cannot rely on the metadata one.
  * whoever retires a backend drains it, because the incoming backend's pending flag
    is not the outgoing recovery's.

These are seam and order tests over fakes. They prove the joins are wired and
ordered; they do not prove the joins are sufficient on real hardware (see
``test/registered/attention/unittests/gdn/test_recover_ssm_sm120.py``), and no test
here touches a device.
"""

import contextlib
import inspect
import sys
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import hybrid_linear_attn_backend as hybrid_module
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.linear.kernels import gdn_flashinfer as fi_module
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor import model_runner as model_runner_module
from sglang.srt.speculative.spec_utils import _recover_ssm_track_unreachable
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Recorder:
    """Records the (operation, argument) pairs a fake stream/engine receives."""

    def __init__(self, log):
        self.log = log

    def wait_stream(self, stream):
        self.log.append(("wait_stream", stream))

    def wait_event(self, event):
        self.log.append(("wait_event", event))

    def record(self, *args):
        self.log.append(("record", args))


class _FakeEngine:
    def __init__(self, log):
        self.log = log
        self.device_to_host_stream = _Recorder(log)
        self.host_to_device_stream = _Recorder(log)
        self.submitted = []

    def submit_device_to_host(self, transfers):
        self.log.append(("submit_d2h", len(transfers)))
        completion = SimpleNamespace(
            start_event="start", finish_event="finish-d2h", timing_enabled=False
        )
        self.submitted.append(("d2h", transfers))
        return completion

    def submit_host_to_device(self, transfers, **kwargs):
        self.log.append(("submit_h2d", len(transfers)))
        completion = SimpleNamespace(
            start_event="start", finish_event="finish-h2d", timing_enabled=False
        )
        self.submitted.append(("h2d", transfers, kwargs))
        return completion


def _controller(*, producer_events):
    """A HiCacheController with just what the two submit paths read."""
    log = []
    ctrl = HiCacheController.__new__(HiCacheController)
    ctrl.l2_transfer_engine = _FakeEngine(log)
    ctrl.forward_stream = "forward-stream"
    ctrl.write_fence_stream = "forward-stream"
    ctrl.load_fence_stream = "forward-stream"
    ctrl.write_fence_events = list(producer_events)
    ctrl.load_fence_events = list(producer_events)
    ctrl.mem_pool_host = SimpleNamespace(
        layout="page_first", can_use_write_back_jit=True, size_per_token=64
    )
    ctrl.mem_pool_device = "device-pool"
    ctrl.device = "cpu"
    ctrl.io_backend = "kernel"
    ctrl.mem_pool_device_allocator = SimpleNamespace(
        set_hicache_transfer_done_event=lambda key, event: log.append(
            ("ack-event", key[1], event)
        )
    )
    ctrl.ack_write_queue = []
    ctrl.ack_load_queue = []
    ctrl.layer_done_counter = SimpleNamespace(
        events=[
            SimpleNamespace(
                start_event=SimpleNamespace(record=lambda: None), complete=lambda: None
            )
        ],
        update_producer=lambda: 0,
    )
    ctrl.transfer_layer_id_max = 1
    ctrl.write_queue = []
    ctrl.load_queue = []
    ctrl.log = log
    return ctrl


def _operation():
    from sglang.srt.managers.cache_controller import CacheOperation

    return CacheOperation(
        torch.arange(4, dtype=torch.int64),
        torch.arange(100, 104, dtype=torch.int64),
        -1,
        None,
    )


class _FakeStreamCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeStashBuf:
    """One of the four stable recovery index/step buffers, watching its fill."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def fill_(self, value):
        self.log.append(("fill", self.name, value))
        return self


class TestWarmupCaptureJoinsTheCallerStream(unittest.TestCase):
    """capture_recovery_graphs hands off the way the serving path does.

    The four stable index/step buffers are zero-filled on the CALLER stream, and the
    per-layer stash and the state pool they read were written there by the verify
    capture; the first warmup launch runs on the RECOVERY stream. Only the
    producer->consumer join (record the join event on the producer, make the consumer
    wait it) orders them: a synchronize AFTER the launch is not protection -- the
    launch is already queued and can read uninitialized indices or commit state into
    a row the producer has not finished writing -- and a graph's own later sync is far
    too late. Same gap in the accepted source; it matters on this launch path.
    """

    def _capture(self):
        log = []
        fake = _FakeCuda(log)
        backend, _, _ = _init_hybrid(
            recover_ssm=True,
            pool_event=_FakeEvent("pool", log),
            log=log,
            fake=fake,
        )
        del log[:]
        for name, buf in (
            ("state", "_rec_state_idx_buf"),
            ("steps", "_rec_acc_steps_buf"),
            ("track_idx", "_rec_track_idx_buf"),
            ("track_steps", "_rec_track_steps_buf"),
        ):
            setattr(backend, buf, _FakeStashBuf(log, name))
        temporal = torch.zeros(4, 2, 2, 2, dtype=torch.bfloat16)
        sidecar = backend.linear_attn_backend
        sidecar._no_cache_stash = {0: {"conv_dims": (1, 1, 1, 1, 1, 1, 1)}}
        sidecar.state_pool = SimpleNamespace(size=4)
        sidecar.state_layer_cache = lambda layer_id: SimpleNamespace(temporal=temporal)
        backend._fi_recovery_launch = lambda n, *a, **k: log.append(("launch", n))
        fi_kernel = object()
        gdn_kernels = types.ModuleType("flashinfer.gdn_kernels")
        leaf = types.ModuleType("flashinfer.gdn_kernels.gdn_decode_bf16_state")
        leaf.gated_delta_rule_mtp = lambda **kwargs: None
        with (
            patch.object(hybrid_module, "get_exec") as published,
            patch.object(fi_module, "fi_recovery_kernel", return_value=fi_kernel),
            patch.object(torch.cuda, "Stream", fake.Stream),
            patch.object(torch.cuda, "Event", fake.Event),
            patch.object(torch.cuda, "stream", fake.stream),
            patch.object(torch.cuda, "graph", fake.graph),
            patch.object(torch.cuda, "CUDAGraph", fake.CUDAGraph),
            patch.object(torch.cuda, "graph_pool_handle", fake.graph_pool_handle),
            patch.dict(
                sys.modules,
                {
                    "flashinfer": types.ModuleType("flashinfer"),
                    "flashinfer.gdn_kernels": gdn_kernels,
                    "flashinfer.gdn_kernels.gdn_decode_bf16_state": leaf,
                },
            ),
        ):
            published.return_value = SimpleNamespace(
                mamba=SimpleNamespace(enable_mamba_extra_buffer=False)
            )
            backend.capture_recovery_graphs([4, 2])
        return backend, log

    def test_the_join_sits_between_the_fills_and_the_first_launch(self):
        backend, log = self._capture()
        kinds = [entry[0] for entry in log]
        self.assertEqual(kinds.count("fill"), 4, "every stable buffer is initialized")
        first_launch = kinds.index("launch")
        joins = [
            i
            for i, entry in enumerate(log)
            if entry[0] == "stream_wait" and entry[1] is backend._recovery_join_event
        ]
        self.assertEqual(len(joins), 1, "one handoff at the warmup boundary")
        self.assertLess(joins[0], first_launch)
        records = [i for i, k in enumerate(kinds) if k == "record"]
        # The producer side of the pair: recorded on the caller stream (no explicit
        # stream argument -> the current stream, same contract as the serving path),
        # after every buffer fill and before the launch that reads them.
        self.assertEqual(len(records), 1)
        self.assertEqual(log[records[0]][2], None)
        self.assertLess(kinds.index("record"), joins[0])
        self.assertLess(max(i for i, k in enumerate(kinds) if k == "fill"), records[0])
        # And the handoff is a wait, not a sync of the consumer after the fact.
        self.assertNotIn("stream_sync", kinds[:first_launch])

    def test_both_buckets_are_captured_after_that_one_join(self):
        backend, log = self._capture()
        self.assertEqual(
            backend._rec_capture_bs,
            [2, 4],
            "capture must succeed (its own failure fallback clears the buckets and "
            "downgrades to eager recovery, which would hide a broken fake)",
        )
        self.assertEqual(len(backend._rec_graphs), 2)
        launches = [e[1] for e in log if e[0] == "launch"]
        # Two launches per bucket: the warm-compile pass outside capture, then the
        # launch inside it. Largest bucket first so the smaller graphs reuse the pool.
        self.assertEqual(launches, [4, 4, 2, 2])
        kinds = [entry[0] for entry in log]
        self.assertEqual(
            kinds[:6], ["fill", "fill", "fill", "fill", "record", "stream_wait"]
        )


class TestRecoveryFencesTheL2Legs(unittest.TestCase):
    def setUp(self):
        self.recovery_event = object()
        self.ctrl = _controller(producer_events=[self.recovery_event])

    def test_write_leg_joins_the_producer_before_submitting(self):
        self.ctrl.write_queue.append(_operation())
        HiCacheController.start_writing(self.ctrl)
        kinds = [entry[0] for entry in self.ctrl.log]
        self.assertEqual(
            kinds,
            ["wait_stream", "wait_event", "submit_d2h", "ack-event"],
            "the D2H leg must wait the forward-stream fence, then the side-stream "
            "state producer, and only then submit -- a wait queued after the copy "
            "cannot order it",
        )
        self.assertEqual(self.ctrl.log[1], ("wait_event", self.recovery_event))
        # The completion that host publication / slot free wait on is the one the
        # submission returned, i.e. it is recorded after both waits.
        self.assertEqual(self.ctrl.log[3][:2], ("ack-event", "write"))
        self.assertEqual(self.ctrl.log[3][2], "finish-d2h")
        self.assertEqual(len(self.ctrl.ack_write_queue), 1)
        # Rows stay pinned until that ACK is reaped.
        self.assertTrue(self.ctrl.has_inflight_device_transfers())

    def test_load_leg_joins_the_producer_before_submitting(self):
        # A load-back WRITES the same state rows; ordering it after the pending
        # recovery keeps the two writers from interleaving on one slot.
        self.ctrl.load_queue.append(_operation())
        producer_id = HiCacheController.start_loading(self.ctrl)
        kinds = [entry[0] for entry in self.ctrl.log]
        self.assertEqual(
            kinds, ["wait_stream", "wait_event", "submit_h2d", "ack-event"]
        )
        self.assertEqual(self.ctrl.log[1], ("wait_event", self.recovery_event))
        # The join must not shadow the layer producer event the layer gate is
        # driven from: a load whose on_layer_done callback was replaced by the loop
        # variable never releases its layer.
        direction, _transfers, submit_kwargs = self.ctrl.l2_transfer_engine.submitted[0]
        self.assertEqual(direction, "h2d")
        self.assertEqual(len(self.ctrl.ack_load_queue), 1)
        self.assertIs(
            submit_kwargs["on_layer_done"],
            self.ctrl.layer_done_counter.events[producer_id].complete,
        )
        # The ACK that gates host publication carries the load leg's own completion.
        self.assertEqual(self.ctrl.ack_load_queue[0].finish_event, "finish-h2d")

    def test_no_producer_events_leaves_the_legs_unchanged(self):
        ctrl = _controller(producer_events=[])
        ctrl.write_queue.append(_operation())
        HiCacheController.start_writing(ctrl)
        self.assertEqual(
            [entry[0] for entry in ctrl.log], ["wait_stream", "submit_d2h", "ack-event"]
        )


class _FakeEvent:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait(self, stream=None):
        self.log.append(("join", self.name, stream))

    def record(self, stream=None):
        self.log.append(("record", self.name, stream))


def _hybrid(*, log, pool_event, recovery_mode=True):
    """A HybridLinearAttnBackend with only the recovery bookkeeping attached."""
    backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
    backend._recover_ssm = recovery_mode
    backend._recovery_stream = None
    backend._recovery_event = pool_event
    backend._recovery_join_event = None
    backend._recovery_stream_ctx = None
    backend._recovery_event_pending = False
    backend.linear_attn_backend = SimpleNamespace(state_pool=SimpleNamespace())
    if pool_event is not None:
        backend.linear_attn_backend.state_pool.state_recovery_event = pool_event
    backend._log = log
    return backend


class TestStateProducerPublication(unittest.TestCase):
    def test_the_state_pool_owns_the_published_event(self):
        # Every backend serving that pool records the SAME object, so the single
        # startup registration on the controller cannot be orphaned when an adaptive
        # width state or a PD role switch replaces the backend underneath it.
        log = []
        event = _FakeEvent("pool", log)
        first = _hybrid(log=log, pool_event=event)
        second = _hybrid(log=log, pool_event=event)
        self.assertEqual(first.state_producer_events(), [event])
        self.assertEqual(second.state_producer_events(), [event])
        self.assertIs(
            first.state_producer_events()[0], second.state_producer_events()[0]
        )

    def test_nothing_to_fence_without_the_recovery_mode(self):
        log = []
        backend = _hybrid(log=log, pool_event=None, recovery_mode=False)
        self.assertEqual(backend.state_producer_events(), [])
        # And the base contract every other backend inherits is a no-op pair, so
        # the call sites do not need isinstance checks.
        plain = AttentionBackend.__new__(AttentionBackend)
        self.assertEqual(plain.state_producer_events(), [])
        self.assertIsNone(plain.drain_pending_recovery())

    def test_drain_joins_once_on_the_current_stream(self):
        log = []
        event = _FakeEvent("pool", log)
        backend = _hybrid(log=log, pool_event=event)
        backend._recovery_event_pending = True
        backend.drain_pending_recovery()
        # No stream argument: Event.wait() resolves the CURRENT stream in C++, which
        # is the forward stream at every documented call site (and avoids building a
        # Python Stream wrapper on the bs=1 latency path).
        self.assertEqual(log, [("join", "pool", None)])
        self.assertFalse(backend._recovery_event_pending)
        backend.drain_pending_recovery()
        self.assertEqual(len(log), 1, "a drained recovery must not re-join")

    def test_metadata_preparation_cannot_consume_the_join(self):
        """None of the metadata-preparation variants may order or consume the join.

        `init_forward_metadata_out_graph` is the exact function the metadata glue
        graph captures (MetadataGlueGraph.run) and then replays for every later
        step: the wait is either recorded into a graph where it waits on an event
        recorded outside it (so it orders nothing), or skipped because the replay
        runs no Python at all. Either way the pending flag was cleared by something
        that never established the ordering, and the next verify read an SSM pool
        the recovery was still committing and overwrote a conv/a-b stash it was
        still reading. Preparation must leave the join to the dispatch seam.
        """
        log = []
        backend = _hybrid(log=log, pool_event=_FakeEvent("pool", log))
        backend.attn_backend_list = []
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_draft_extend_v2=lambda: False)
        )

        for name, call in (
            ("eager prep", lambda: backend.init_forward_metadata(forward_batch)),
            (
                "out-of-graph prep (what the glue captures)",
                lambda: backend.init_forward_metadata_out_graph(forward_batch),
            ),
            (
                "out-of-graph prep during capture",
                lambda: backend.init_forward_metadata_out_graph(
                    forward_batch, in_capture=True
                ),
            ),
            (
                "in-graph prep",
                lambda: backend.init_forward_metadata_in_graph(forward_batch),
            ),
        ):
            with self.subTest(name):
                backend._recovery_event_pending = True
                log.clear()
                call()
                self.assertEqual(log, [], "preparation must issue no cross-stream wait")
                self.assertTrue(
                    backend._recovery_event_pending,
                    "preparation must not clear the flag before the forward owns it",
                )

        backend.drain_pending_recovery()
        self.assertEqual(log, [("join", "pool", None)])
        self.assertFalse(backend._recovery_event_pending)

    def test_full_mode_pool_carries_no_recovery_event(self):
        # The event is allocated by the mode that records it, so a full-mode launch
        # pays nothing for a fence it never uses.
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateDType,
            Mamba2StateShape,
        )

        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=64,
            n_groups=1,
            num_heads=2,
            head_dim=32,
            state_size=32,
            conv_kernel=4,
        )
        params = Mamba2CacheParams(
            shape=shape,
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
            layers=[0],
        )
        # device="cpu" short-circuits the gate before the mode is consulted, and the
        # mode itself arrives as a constructor argument, so a pool needs no published
        # runtime context to exist. The CUDA half of the gate is covered on hardware
        # (test_recover_ssm_sm120.test_state_pool_publishes_the_recovery_fence).
        for mode in ("full", "none"):
            pool = MambaPool(
                size=2,
                spec_state_size=2,
                cache_params=params,
                mamba_layer_ids=[0],
                device="cpu",
                speculative_num_draft_tokens=4,
                gdn_mtp_cache_mode=mode,
            )
            self.assertIsNone(pool.state_recovery_event, mode)


class TestCowClearJoinSite(unittest.TestCase):
    """The eager forward's deferred CoW/clear runs before its metadata plan.

    A behavioural test would need a whole model forward, so this pins the call order
    at the one seam that matters, in the source the runner actually executes: it
    fails if a refactor moves the COW/clear ahead of the join (which is the race the
    join exists for -- a finished request's slot, still being committed by the
    recovery, handed to this batch and zeroed or copied under it).
    """

    def _forward_raw_source(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        return inspect.getsource(ModelRunner._forward_raw)

    def test_join_precedes_the_deferred_cow_and_clear(self):
        src = self._forward_raw_source()
        join = src.index("drain_pending_recovery()")
        cow = src.index("_maybe_execute_deferred_mamba_cow_and_clear(forward_batch)")
        self.assertLess(join, cow)

    def test_the_join_is_not_gated_on_the_mode_at_the_call_site(self):
        # The base class owns the no-op, so the hot path is one virtual call and the
        # runner never re-derives "is this a RecoverSSM launch".
        src = self._forward_raw_source()
        call = "self.attn_backend.drain_pending_recovery()"
        lines = [
            line.strip()
            for line in src.splitlines()
            if call in line and not line.strip().startswith("#")
        ]
        self.assertEqual(lines, [call], "one join call, and it is a call")

    def test_the_join_precedes_the_graph_dispatch_that_returns_early(self):
        """The eager-only position was the bug: the cuda-graph branch returns first.

        Behaviour is pinned in TestDispatchSeamJoinsBeforeGraphReplay; this is the
        structural guard against the call being moved back below the branch.
        """
        src = self._forward_raw_source()
        join = src.index("drain_pending_recovery()")
        dispatch = src.index("if can_run_graph:")
        self.assertLess(join, dispatch)
        self.assertLess(join, src.index("prepare_for_forward"))

    def test_ownership_is_not_duplicated_into_the_prep_methods(self):
        """The join lives in exactly one place; prep keeps no copy of it."""
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend as H,
        )

        for method in (
            H.init_forward_metadata,
            H.init_forward_metadata_out_graph,
            H.init_forward_metadata_in_graph,
        ):
            with self.subTest(method.__name__):
                self.assertNotIn("_wait_recovery_if_pending", inspect.getsource(method))
        self.assertIn(
            "_wait_recovery_if_pending",
            inspect.getsource(H.drain_pending_recovery),
        )


class _FakeCapturedGraph:
    def __init__(self, log):
        self.log = log

    def replay(self):
        self.log.append(("glue-graph-replay",))


class _GlueReplayDecodeRunner:
    """The decode cuda-graph runner's uncaptured host half, replaying a glue graph
    that was captured on an earlier step."""

    def __init__(self, log, glue, backend, key):
        self.log, self.glue, self.backend, self.key = log, glue, backend, key

    def can_run_graph(self, forward_batch):
        return True

    def execute(self, forward_batch, pp_proxy_tensors=None):
        # load_batch -> _init_replay_attn_metadata -> the captured-prep fast path.
        self.glue.run(self.backend, SimpleNamespace(batch=forward_batch), self.key)
        self.log.append(("model-graph-launch",))
        return "logits"


class TestDispatchSeamJoinsBeforeGraphReplay(unittest.TestCase):
    """The actual target-compute boundary, with an already-captured metadata replay.

    Real ModelRunner._forward_raw, real HybridLinearAttnBackend drain, real
    MetadataGlueGraph in its replayed state. The join must land on the compute
    stream before the graph is launched -- the graph branch returns early, so a
    join placed anywhere below the branch never runs on this path at all.
    """

    def _forward_raw(self, *, pending):
        from sglang.srt.model_executor.runner.metadata_glue_graph import (
            MetadataGlueGraph,
        )

        log = []
        event = _FakeEvent("pool", log)
        backend = _hybrid(log=log, pool_event=event)
        backend._recovery_event_pending = pending
        glue = MetadataGlueGraph(device="cuda")
        key = ("target-verify", 4)
        # Already captured on an earlier step: run() must take the replay branch.
        glue._states[key] = {
            "warmups": glue.NUM_WARMUP,
            "graph": _FakeCapturedGraph(log),
            "meta": [(backend, "metadata-of-capture-key")],
        }
        runner = model_runner_module.ModelRunner.__new__(
            model_runner_module.ModelRunner
        )
        runner.device = "cuda"
        runner.model = SimpleNamespace(shared_experts_config=None)
        runner.attn_backend = backend
        runner.hisparse_coordinator = None
        runner.decode_cuda_graph_runner = _GlueReplayDecodeRunner(
            log, glue, backend, key
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_cuda_graph=lambda: True,
                is_cpu_graph=lambda: False,
                is_decode=lambda: False,
            )
        )
        fake_model = SimpleNamespace(
            override=lambda shared_experts: contextlib.nullcontext()
        )
        with (
            patch.object(model_runner_module, "get_model", return_value=fake_model),
            patch.object(model_runner_module, "has_forward_context", return_value=True),
            patch.object(
                model_runner_module, "get_forward_context", return_value="ctx"
            ),
            patch.object(
                model_runner_module,
                "forward_context",
                lambda ctx, forward_batch: contextlib.nullcontext(),
            ),
        ):
            out = runner._forward_raw(forward_batch, pp_proxy_tensors=None)
        return log, backend, glue, out

    def test_the_seam_join_orders_the_graph_that_replays_captured_prep(self):
        log, backend, glue, out = self._forward_raw(pending=True)
        self.assertEqual(
            log,
            [("join", "pool", None), ("glue-graph-replay",), ("model-graph-launch",)],
        )
        self.assertFalse(backend._recovery_event_pending)
        self.assertEqual(glue.replays, 1)
        self.assertEqual(glue.eager, 0, "the captured prep must not have run eagerly")
        self.assertEqual(
            backend.forward_metadata,
            "metadata-of-capture-key",
            "the replay must still reinstall the captured metadata objects",
        )
        self.assertTrue(out.can_run_graph)

    def test_a_step_with_nothing_pending_costs_one_noop_call(self):
        log, backend, glue, _ = self._forward_raw(pending=False)
        self.assertEqual(
            log, [("glue-graph-replay",), ("model-graph-launch",)], "no stray wait"
        )
        self.assertFalse(backend._recovery_event_pending)

    def test_the_prep_the_glue_graph_captures_issues_no_join(self):
        """Both glue branches: the prep it records, and the replay that replaces it.

        The old placement cleared the pending flag in the warm-up/capture calls and
        then ordered nothing at all on every replayed step, so from the second step
        onward the recovery was never joined on the path that reads the pool.
        """
        from sglang.srt.model_executor.runner.metadata_glue_graph import (
            MetadataGlueGraph,
        )

        log = []
        glue = MetadataGlueGraph(device="cuda")
        backend = _hybrid(log=log, pool_event=_FakeEvent("pool", log))
        backend.attn_backend_list = []
        backend._recovery_event_pending = True
        calls = []

        class _Probe:
            attn_backend_list = []

            def init_forward_metadata_out_graph(self, fb_view, in_capture=False):
                calls.append(fb_view)
                backend.init_forward_metadata_out_graph(fb_view, in_capture=in_capture)

        for step in range(glue.NUM_WARMUP):
            with self.subTest(warmup=step):
                glue.run(_Probe(), f"fb-view-{step}", "k")
                self.assertEqual(log, [], "the prep the glue records issues no wait")
                self.assertTrue(
                    backend._recovery_event_pending,
                    "planning/capture must leave the join to the forward",
                )
        self.assertEqual(calls, ["fb-view-0", "fb-view-1"])

        # Now in its captured state: only reinstall + device-op replay, no Python
        # prep, no wait -- and the flag must still be there for the dispatch seam.
        glue._states["k"] = {
            "warmups": glue.NUM_WARMUP,
            "graph": _FakeCapturedGraph(log),
            "meta": [(backend, "metadata-of-capture-key")],
        }
        glue.run(_Probe(), "fb-view-2", "k")
        self.assertEqual(
            log,
            [("glue-graph-replay",)],
            "the replay issues no wait, so it cannot be the join",
        )
        self.assertEqual(calls, ["fb-view-0", "fb-view-1"], "no Python prep ran")
        self.assertTrue(backend._recovery_event_pending)
        backend.drain_pending_recovery()
        self.assertEqual(log[-1], ("join", "pool", None))


class _Req:
    def __init__(self, committed_len):
        self.kv = SimpleNamespace(kv_committed_len=committed_len)


def _batch(committed_lens):
    from sglang.srt.runtime_context import get_context

    batch = SimpleNamespace(reqs=[_Req(n) for n in committed_lens])
    return get_context(), batch


class TestBoundaryPlanElision(unittest.TestCase):
    """none-mode may drop the track plan only when a crossing is UNREACHABLE."""

    INTERVAL = 64

    def _unreachable(self, committed_lens, *, mode, draft_tokens=4):
        from sglang.srt.runtime_context import get_context

        _, batch = _batch(committed_lens)
        with get_context().override_server_args(
            gdn_mtp_cache_mode=mode,
            mamba_track_interval=self.INTERVAL,
            speculative_algorithm="NEXTN",
            speculative_num_draft_tokens=draft_tokens,
        ):
            return _recover_ssm_track_unreachable(batch)

    def test_only_none_mode_opts_in(self):
        self.assertTrue(self._unreachable([8], mode="none"))
        self.assertFalse(self._unreachable([8], mode="full"))

    def test_far_from_a_boundary_the_sweep_is_dropped(self):
        # 8 + 2*4 stays inside interval 0, so no checkpoint can be missed.
        self.assertTrue(self._unreachable([8, 40], mode="none"))

    def test_the_window_absorbs_the_committed_len_lag(self):
        # A request 2 draft tokens short of the boundary may still cross once
        # kv_committed_len catches up under overlap: the plan must survive.
        self.assertFalse(self._unreachable([self.INTERVAL - 2 * 4 + 1], mode="none"))
        # Exactly at the margin, and clearly past it.
        self.assertFalse(self._unreachable([self.INTERVAL - 2 * 4], mode="none"))
        self.assertTrue(self._unreachable([self.INTERVAL - 2 * 4 - 1], mode="none"))

    def test_any_crossing_request_keeps_the_plan(self):
        self.assertFalse(self._unreachable([8, self.INTERVAL - 3], mode="none"))

    def test_without_a_draft_window_nothing_is_elided(self):
        # A verify with no draft width has no boundary sweep to skip, and dropping
        # the plan on the strength of a None bound would be a guess.
        self.assertFalse(self._unreachable([8], mode="none", draft_tokens=None))


class _FakeStream:
    def __init__(self, log):
        self.log = log

    def wait_event(self, event):
        self.log.append(("stream_wait", event))

    def synchronize(self):
        self.log.append(("stream_sync",))


class _FakeCuda:
    """Stand-ins for the three torch.cuda factories the recovery helpers use."""

    def __init__(self, log):
        self.log = log
        self.streams = 0
        self.events = 0
        self.created_streams = []

    def Stream(self):
        self.streams += 1
        stream = _FakeStream(self.log)
        self.created_streams.append(stream)
        return stream

    def Event(self):
        self.events += 1
        return _FakeEvent(f"event{self.events}", self.log)

    def stream(self, stream):
        self.log.append(("ctx", stream))
        # The capture loop ENTERS this, so it has to be a context manager, not a
        # placeholder value -- a bare object here gets swallowed by the method's
        # capture-failure fallback and the test would silently see no launches.
        return _FakeStreamCtx()

    def graph_pool_handle(self):
        self.log.append(("pool-handle",))
        return "graph-pool"

    def CUDAGraph(self):
        return SimpleNamespace(replay=lambda: self.log.append(("replay",)))

    @contextmanager
    def graph(self, *args, **kwargs):
        self.log.append(("capture-begin",))
        yield
        self.log.append(("capture-end",))


def _linear_sidecar(*, recover_ssm, pool_event):
    sidecar = SimpleNamespace(
        needs_cpu_seq_lens=False,
        supports_prefill_graph_extend=lambda: False,
        prefill_graph_max_seqs=8,
        state_pool=SimpleNamespace(),
    )
    if recover_ssm:
        sidecar._recover_ssm = True
    if pool_event is not None:
        sidecar.state_pool.state_recovery_event = pool_event
    return sidecar


def _init_hybrid(*, recover_ssm, pool_event, cuda=True, log=None, fake=None):
    """Run the real HybridLinearAttnBackend.__init__ over fake CUDA primitives.

    `log`/`fake` can be supplied so a later step (a capture, a drain) records into the
    same journal as the stream and event objects created here -- the objects hold the
    log they were built with.
    """
    log = [] if log is None else log
    fake = _FakeCuda(log) if fake is None else fake
    full = SimpleNamespace(
        token_to_kv_pool="kv",
        req_to_token_pool="reqs",
        runner_ids="runner-ids",
        needs_cpu_seq_lens=False,
        use_captured_forward_metadata_for_breakable_cuda_graph=False,
    )
    sidecar = _linear_sidecar(recover_ssm=recover_ssm, pool_event=pool_event)
    with (
        patch.object(hybrid_module, "is_cuda", return_value=cuda),
        patch.object(hybrid_module, "check_cuda_graph_backend", lambda *a, **k: False),
        patch.object(torch.cuda, "Stream", fake.Stream),
        patch.object(torch.cuda, "Event", fake.Event),
        patch.object(torch.cuda, "stream", fake.stream),
    ):
        backend = HybridLinearAttnBackend(full, sidecar, full_attn_layers=[0])
    return backend, fake, log


class TestRecoveryStreamIsInitializedLast(unittest.TestCase):
    """__init__ must finish the field block before creating the side stream.

    The helper assigns every recovery attribute, so a call placed inside the block
    either reads an attribute that does not exist yet (AttributeError at startup,
    for every CUDA none-mode launch) or has its result overwritten by the
    initialization below it -- which would silently restore the lazy-create path and
    hand the HiCache fence an event that changes identity.
    """

    def test_recovery_mode_creates_the_shared_event_after_the_fields_exist(self):
        log = []
        pool_event = _FakeEvent("pool", log)
        backend, fake, _ = _init_hybrid(recover_ssm=True, pool_event=pool_event)
        self.assertEqual(fake.streams, 1, "the side stream is created once, at init")
        self.assertEqual(fake.created_streams, [backend._recovery_stream])
        self.assertIs(backend._recovery_event, pool_event)
        self.assertIsInstance(backend._recovery_join_event, _FakeEvent)
        self.assertIsNotNone(backend._recovery_stream_ctx)
        # One event only: the pool owns the completion event, so the backend must
        # not allocate (and later publish) a private one.
        self.assertEqual(fake.events, 1)
        self.assertEqual(backend.state_producer_events(), [pool_event])
        self.assertFalse(backend._recovery_event_pending)

    def test_the_published_event_survives_a_second_backend_on_the_same_pool(self):
        log = []
        pool_event = _FakeEvent("pool", log)
        first, _, _ = _init_hybrid(recover_ssm=True, pool_event=pool_event)
        second, _, _ = _init_hybrid(recover_ssm=True, pool_event=pool_event)
        self.assertEqual(first.state_producer_events(), second.state_producer_events())
        self.assertIs(
            first.state_producer_events()[0], second.state_producer_events()[0]
        )

    def test_a_wrapper_that_does_not_recover_creates_no_stream(self):
        # Publication follows the POOL, not the wrapper: if that pool can be written
        # by a none-mode recovery (some other state's backend, or this launch's own
        # later switch), the fence must already be joined -- an extra join is
        # harmless, a missing one is the torn-backup bug. What a non-recovering
        # wrapper must not do is allocate a side stream it never uses.
        pool_event = _FakeEvent("pool", [])
        backend, fake, _ = _init_hybrid(recover_ssm=False, pool_event=pool_event)
        self.assertEqual(fake.streams, 0)
        self.assertEqual(fake.events, 0)
        self.assertIsNone(backend._recovery_stream)
        self.assertEqual(backend.state_producer_events(), [pool_event])

    def test_off_cuda_creates_nothing_and_publishes_nothing(self):
        # A pool with no CUDA state has no event to publish (MambaPool gates the
        # allocation on the device), so the whole mechanism stays inert.
        backend, fake, _ = _init_hybrid(recover_ssm=True, pool_event=None, cuda=False)
        self.assertEqual(fake.streams, 0)
        self.assertIsNone(backend._recovery_stream)
        self.assertEqual(backend.state_producer_events(), [])


class _FakeCapture:
    def __init__(self, capture_bs):
        self.eager_runner = "eager"
        self.prefill = SimpleNamespace(runner="prefill-graph")
        self.decode = SimpleNamespace(
            runner=SimpleNamespace(capture_bs=list(capture_bs))
        )
        self.memory_usage = {}
        self.time_usage = {}


@contextmanager
def _graph_capture_env(*, mode, scale_joiner, capture_bs):
    """Run the real ModelRunner.init_cuda_graphs with the capture layer faked."""
    calls = []
    published = SimpleNamespace(
        moe=SimpleNamespace(is_ep_scale_joiner=scale_joiner),
        mamba=SimpleNamespace(gdn_mtp_cache_mode=mode),
    )

    def capture_cuda_graphs(**kwargs):
        calls.append(("capture", kwargs))
        return _FakeCapture(capture_bs)

    with (
        patch.object(model_runner_module, "get_exec", return_value=published),
        patch.object(
            model_runner_module,
            "get_parallel",
            return_value=SimpleNamespace(ep_join_rank_offset=0, tp_size=1),
        ),
        patch.object(model_runner_module, "capture_cuda_graphs", capture_cuda_graphs),
        patch.object(
            model_runner_module,
            "prebuild_deepep_v2_buffer",
            side_effect=lambda model: calls.append(("prebuild", model)),
        ),
        patch.object(
            model_runner_module,
            "finalize_cuda_graph_capture",
            side_effect=lambda runner: calls.append(("finalize",)),
        ),
    ):
        yield calls


def _runner(*, mode, device="cuda", draft=False, backend=None, elastic=False):
    from sglang.srt.model_executor.model_runner import ModelRunner

    runner = ModelRunner.__new__(ModelRunner)
    runner.device = device
    runner.is_draft_worker = draft
    runner.model = "model"
    runner.attn_backend = backend
    runner.decode_cuda_graph_capture_bs = []
    runner.decode_cuda_graph_runner = None
    runner._elastic_enabled = elastic
    runner._elastic_cuda_graph_enabled = lambda: elastic
    runner._validate_elastic_cuda_graph_recapture = lambda: None
    runner._finalize_elastic_ep_joiner = lambda size: runner.calls.append(
        ("joiner", size)
    )
    runner._recapture_elastic_cuda_graphs = lambda: (
        runner.calls.append(("recapture",))
        or setattr(runner, "decode_cuda_graph_runner", SimpleNamespace(capture_bs=[3]))
    )
    return runner


def _hybrid_backend_recorder():
    backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
    backend.captured = []
    backend.capture_recovery_graphs = lambda bs: backend.captured.append(list(bs))
    return backend


class TestInitCudaGraphsKeepsTheExistingTail(unittest.TestCase):
    """The recovery capture is an addition to init_cuda_graphs, not a new home.

    The elastic-joiner finalization and the deepep buffer prebuild that end that
    method read its locals and run on every launch; moving them into the
    self-guarded helper means they are skipped for full mode and every draft worker
    (and NameError on the joiner path). Both are asserted here, with the recovery
    capture checked at the same time.
    """

    def _run(
        self, *, mode, scale_joiner=False, backend=None, device="cuda", draft=False
    ):
        runner = _runner(
            mode=mode, device=device, draft=draft, backend=backend, elastic=scale_joiner
        )
        with _graph_capture_env(
            mode=mode, scale_joiner=scale_joiner, capture_bs=[1, 2, 4]
        ) as calls:
            runner.calls = calls
            ModelRunner_init = type(runner).init_cuda_graphs
            ModelRunner_init(runner)
        return runner, calls

    def test_full_mode_still_prebuilds_and_captures_no_recovery_graphs(self):
        backend = _hybrid_backend_recorder()
        runner, calls = self._run(mode="full", backend=backend)
        self.assertIn(("prebuild", "model"), calls)
        self.assertEqual(backend.captured, [])

    def test_a_draft_worker_still_prebuilds(self):
        runner, calls = self._run(mode="none", draft=True)
        self.assertIn(("prebuild", "model"), calls)

    def test_none_mode_captures_against_the_live_decode_buckets(self):
        backend = _hybrid_backend_recorder()
        runner, calls = self._run(mode="none", backend=backend)
        self.assertIn(("prebuild", "model"), calls)
        self.assertEqual(backend.captured, [[1, 2, 4]])

    def test_elastic_deferred_flow_finalizes_then_captures_the_recaptured_buckets(self):
        backend = _hybrid_backend_recorder()
        runner, calls = self._run(mode="none", scale_joiner=True, backend=backend)
        kinds = [call[0] for call in calls]
        self.assertEqual(
            kinds,
            ["capture", "recapture", "finalize", "joiner", "prebuild"],
            "the joiner path must keep its existing order, with the recovery "
            "capture added last",
        )
        # The buckets come from the runner produced by the recapture, not from the
        # empty snapshot taken before it.
        self.assertEqual(backend.captured, [[3]])

    def test_a_non_hybrid_backend_captures_nothing_and_breaks_nothing(self):
        runner, calls = self._run(mode="none", backend=SimpleNamespace())
        self.assertIn(("prebuild", "model"), calls)

    def test_off_cuda_captures_nothing(self):
        backend = _hybrid_backend_recorder()
        runner, calls = self._run(mode="none", backend=backend, device="cpu")
        self.assertEqual(backend.captured, [])
        self.assertIn(("prebuild", "model"), calls)


if __name__ == "__main__":
    unittest.main()
