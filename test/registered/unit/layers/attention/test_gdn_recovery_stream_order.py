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

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.memory_pool import MambaPool
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
        events=[SimpleNamespace(start_event=SimpleNamespace(record=lambda: None),
                               complete=lambda: None)],
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

    def test_metadata_plans_join_before_the_forward_reads_the_pool(self):
        log = []
        backend = _hybrid(log=log, pool_event=_FakeEvent("pool", log))
        backend._recovery_event_pending = True
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_draft_extend_v2=lambda: False)
        )
        backend.attn_backend_list = []
        backend.init_forward_metadata(forward_batch)
        self.assertEqual(log, [("join", "pool", None)])

        backend._recovery_event_pending = True
        log.clear()
        backend.init_forward_metadata_out_graph(forward_batch)
        self.assertEqual(log, [("join", "pool", None)])

        # During capture the wrapper must not enqueue a cross-stream wait into the
        # graph being captured.
        backend._recovery_event_pending = True
        log.clear()
        backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)
        self.assertEqual(log, [])
        self.assertTrue(backend._recovery_event_pending)

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
        import inspect

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
        line = next(
            line for line in src.splitlines() if "drain_pending_recovery" in line
        )
        self.assertEqual(line.strip(), "self.attn_backend.drain_pending_recovery()")


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


if __name__ == "__main__":
    unittest.main()
