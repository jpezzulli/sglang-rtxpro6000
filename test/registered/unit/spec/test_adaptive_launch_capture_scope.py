"""CPU-only checks on the launch-width scoping of the initial adaptive
CUDA-graph capture, including the review-1 shared-buffer boundary.

``Scheduler.init_all_cuda_graphs`` captures the launch state first --
``tp_worker.init_cuda_graphs`` for the target verify graphs, then
``draft_worker.init_cuda_graphs`` for the draft decode / draft-extend graphs --
and only afterwards does ``EAGLEWorkerV2`` register that state and run
``AdaptiveController.init_states``. init_states skips the registered launch
width, so the per-width ``cuda_graph_bs_for_step`` pruning reached only the
widths built later: under a W4/W8 policy where W8 (steps=7) is reachable only
in the C1 bucket, the launch width still captured its full C1-C6 graph set.

The scoping must land on the ACTUAL decode-capture boundaries and nowhere
else. ``GraphSharedOutput.create_for_model_runner`` sizes the process-shared
logits buffer from ``ModelRunner.max_decode_logits_rows`` -- the full-bucket
maximum -- BEFORE the decode runner is built, and EagerRunner then allocates
the fixed-max static buffer onto that shared pool; a launch-wide pruned config
under-provisions both (48 rows became 8 under BS [1..6] at W8, and W4/C6's
24-row get_logits_buffer then asserts), and the shared buffer can never be
widened after captured graphs point at it. So the pruned config is QUEUED by
``prepare_adaptive_launch_capture`` and consumed only by the target verify
boundary in ``cuda_graph_setup.capture_cuda_graphs`` and the initial
``EagleDraftWorker`` draft decode / draft-extend capture; the published config
is full canonical again before ``init_states`` builds the narrower widths, and
the scheduler clears the queue on every exit path. The GDN recovery graphs stay
tied to the target runner's real (pruned) capture_bs, which is what they bake.

These checks run against a real published ``RuntimeContext``, the real
``Scheduler.init_all_cuda_graphs`` control flow, the real
``GraphSharedOutput`` / ``ModelRunner.max_decode_logits_rows`` provisioning
and the real capture-list consumers (``get_batch_sizes_to_capture``,
``check_cuda_graph_backend``); no mocks stand in for what the decode runners
and the shared logits buffer read.
"""

import contextlib
import json
import os
import shutil
import tempfile
import types
import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor import cuda_graph_config as cgc
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    get_batch_sizes_to_capture,
)
from sglang.srt.runtime_context import get_context, get_exec, get_parallel
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

CANONICAL_BS = [1, 2, 4, 8]
# W4/W8 at topk=1: the launch width is W8 (steps=7); W4 is steps=3.
LAUNCH_STEPS = 7
NARROW_STEPS = 3
# A contradictory legacy alias, as in the sibling capture-config checks: the
# scoped boundary may only ever move the canonical leaf.
STALE_LEGACY_BS = [8, 16, 32]

# C1 is the only bucket whose slot lists the launch width.
W8_REACHABLE_AT_C1 = {
    "1": {"candidate_steps": [NARROW_STEPS, LAUNCH_STEPS]},
    "2": {"candidate_steps": [NARROW_STEPS]},
    "4": {"candidate_steps": [NARROW_STEPS]},
    "8": {"candidate_steps": [NARROW_STEPS]},
}
# Asymmetric table: the launch width is reachable at C1 AND C2, so the scoped
# launch capture must be [1, 2] -- a hard-coded [1] would over-prune.
W8_REACHABLE_AT_C1_C2 = {
    "1": {"candidate_steps": [NARROW_STEPS, LAUNCH_STEPS]},
    "2": {"candidate_steps": [NARROW_STEPS, LAUNCH_STEPS]},
    "4": {"candidate_steps": [NARROW_STEPS]},
    "8": {"candidate_steps": [NARROW_STEPS]},
}
# Every bucket can reach every width: nothing to prune.
ALL_AT_EVERY_BUCKET = {
    bs: {"candidate_steps": [NARROW_STEPS, LAUNCH_STEPS]} for bs in ("1", "2", "4", "8")
}


def _published_config(backend=Backend.FULL, bs=None):
    return CudaGraphConfig.from_dict(
        {
            "decode": {
                "backend": backend,
                "max_bs": 256,
                "bs": list(CANONICAL_BS if bs is None else bs),
            }
        }
    )


@contextlib.contextmanager
def _aligned():
    with get_parallel().override(attn_tp_size=1, attn_cp_size=1):
        yield


def _capture_seen():
    """What a decode graph runner sizes its capture list from, right now.

    Same read DecodeCudaGraphRunner and EagleDraftWorker._capture_cuda_graphs
    perform (DISABLED branch included).
    """
    with _aligned():
        if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            return None
        runner_stub = SimpleNamespace(req_to_token_pool=SimpleNamespace(size=64))
        return get_batch_sizes_to_capture(runner_stub)[0]


def _decode_capture_scope():
    """The real boundary helper, resolved at call time.

    On a tree without it (the whole-startup-scope revision) this degrades to
    the plain window that tree effectively ran around EVERYTHING -- so the
    shared-output regression below goes RED by under-provisioning the real
    GraphSharedOutput, not by an import-time attribute error.
    """
    fn = getattr(cgc, "adaptive_launch_capture_scope", None)
    return contextlib.nullcontext() if fn is None else fn()


def _published_bs():
    cfg = get_exec().graph.cuda_graph_config
    return None if cfg is None else list(cfg.decode.bs)


def _clear_pending_scope():
    # Late-bound: a tree without the queue helper has nothing to clear.
    clear = getattr(cgc, "clear_adaptive_launch_capture_scope", None)
    if clear is not None:
        clear()


def _pending_scope():
    return getattr(cgc, "_adaptive_launch_capture_scope", None)


def _state(steps):
    return SpecRuntimeState(
        speculative_num_steps=steps,
        speculative_num_draft_tokens=steps + 1,
        draft_attn_backend=SimpleNamespace(name=f"draft-decode-{steps}"),
        cuda_graph_runner=SimpleNamespace(name=f"draft-graph-{steps}"),
        target_attn_backend=SimpleNamespace(name=f"target-{steps}"),
        target_graph_runner=SimpleNamespace(name=f"target-graph-{steps}"),
        draft_extend_attn_backend=SimpleNamespace(name=f"draft-extend-{steps}"),
        cuda_graph_runner_for_draft_extend=SimpleNamespace(
            name=f"extend-graph-{steps}"
        ),
    )


def _target_runner_stub():
    """Enough ModelRunner for the REAL max_decode_logits_rows (the rows the
    process-shared logits buffer provisions), at the launch W8 width."""
    runner = SimpleNamespace(
        device="cpu",
        req_to_token_pool=SimpleNamespace(size=64),
        decode_num_tokens_per_req=lambda: LAUNCH_STEPS + 1,
    )
    runner.max_decode_logits_rows = types.MethodType(
        ModelRunner.max_decode_logits_rows, runner
    )
    return runner


class _LaunchWorker(EAGLEWorkerV2):
    """Enough EAGLEWorkerV2 to drive the REAL startup init_cuda_graphs (its
    super() resolves to BaseSpecWorker for real too), including the real state
    registration and the real per-width capture window, without a model on a
    GPU."""

    _decode_graph_capture_bs = EAGLEWorkerV2._decode_graph_capture_bs
    _validate_adaptive_widths = EAGLEWorkerV2._validate_adaptive_widths
    _override_worker_state = EAGLEWorkerV2._override_worker_state

    def __init__(self, config_path, seen, applied=None, raise_in_build=False):
        self.seen = seen
        self.shared_rows_seen = []
        self.shared = None
        self.speculative_num_steps = LAUNCH_STEPS
        self.speculative_num_draft_tokens = LAUNCH_STEPS + 1
        self.topk = 1
        self._raise_in_build = raise_in_build

        def capture_draft():
            # The real EagleDraftWorker.init_cuda_graphs order: the draft
            # TpModelWorker provisioning (decode capture off) happens BEFORE
            # the draft decode / draft-extend capture boundary.
            self.seen.append(("draft-provisioning", _published_bs()))
            with _decode_capture_scope():
                # EagleDraftWorker._capture_cuda_graphs builds the draft
                # decode and the draft-extend graphs in this window.
                self.seen.append(("draft-decode", _capture_seen()))
                self.seen.append(("draft-extend", _capture_seen()))

        self._draft_worker = SimpleNamespace(
            speculative_num_steps=LAUNCH_STEPS,
            speculative_num_draft_tokens=LAUNCH_STEPS + 1,
            draft_tp_context=lambda group: contextlib.nullcontext(),
            draft_runner=SimpleNamespace(
                tp_group=object(),
                draft_attn_backend=None,
                attn_backend=None,
                token_to_kv_pool=SimpleNamespace(),
            ),
            draft_attn_backend=SimpleNamespace(name="draft-decode-launch"),
            draft_extend_attn_backend=SimpleNamespace(name="draft-extend-launch"),
            cuda_graph_runner=SimpleNamespace(name="draft-graph-launch"),
            cuda_graph_runner_for_draft_extend=SimpleNamespace(
                name="extend-graph-launch"
            ),
            _rebuild_topk1_chain_buffers=lambda: None,
            init_cuda_graphs=capture_draft,
        )
        self._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(name="target-launch"),
                decode_cuda_graph_runner=SimpleNamespace(name="target-graph-launch"),
                token_to_kv_pool=SimpleNamespace(),
            )
        )
        self.applied = [] if applied is None else applied
        self.adaptive_controller = (
            AdaptiveController(self, config_path=config_path)
            if config_path is not None
            else None
        )

    def apply_runtime_state(self, state):
        self.applied.append(state.speculative_num_steps)

    def init_target_cuda_graphs(self, raise_after=False):
        """The real cuda_graph_setup.capture_cuda_graphs order, with the real
        shared-output provisioning: GraphSharedOutput and the eager/prefill
        fixed-max buffers are sized off the published (FULL) config; only the
        decode-graph boundary consumes the queued launch capture scope."""
        with _aligned():
            self.shared = GraphSharedOutput.create_for_model_runner(
                _target_runner_stub()
            )
        self.shared_rows_seen.append(0 if self.shared is None else self.shared.max_rows)
        # The decode-graph boundary is the only scoped part of the target
        # worker's capture (cuda_graph_setup.capture_cuda_graphs' decode
        # branch); the provisioning above stays on the published config.
        with _decode_capture_scope():
            self.seen.append(("target", _capture_seen()))
        # ... and the boundary is transient: the published list is full again
        # for everything after the decode capture (EagerRunner hooks, the
        # draft worker's provisioning, init_states).
        self.seen.append(("target_after", _published_bs()))
        if raise_after:
            raise RuntimeError("target capture blew up")

    def build_adaptive_runtime_state(
        self, speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs=None
    ):
        # The real per-width scoping machinery; inside it sits the width's
        # _capture_cuda_graphs and its DecodeCudaGraphRunner.
        with self._override_worker_state(
            speculative_num_steps,
            speculative_num_draft_tokens,
            cuda_graph_bs=cuda_graph_bs,
        ):
            self.seen.append((f"build{speculative_num_steps}", _capture_seen()))
            if self._raise_in_build:
                raise RuntimeError("width capture blew up")
            return _state(speculative_num_steps)


class _StartupCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._dir = tempfile.mkdtemp(prefix="adaptive-launch-cfg-")
        self.addCleanup(shutil.rmtree, self._dir, True)
        shared = GraphSharedOutput._process_shared
        GraphSharedOutput._process_shared = None
        self.addCleanup(setattr, GraphSharedOutput, "_process_shared", shared)
        self.addCleanup(_clear_pending_scope)

    def _publish(self, cfg):
        override = get_context().override_server_args(
            cuda_graph_config=cfg, cuda_graph_bs_decode=STALE_LEGACY_BS
        )
        override.install()
        self.addCleanup(override.restore)
        return cfg

    def _config_path(self, table):
        path = os.path.join(self._dir, "candidates.json")
        with open(path, "w") as f:
            json.dump(table, f)
        return path

    def _run_startup(self, worker, raise_in_target=False):
        """Drive the real scheduler startup-capture control flow."""
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(
                init_cuda_graphs=lambda: worker.init_target_cuda_graphs(
                    raise_after=raise_in_target
                )
            ),
            draft_worker=worker,
        )
        Scheduler.init_all_cuda_graphs(scheduler)

    def _seen(self, seen, phase):
        return [bs for name, bs in seen if name == phase]


class TestLaunchWidthScopedToPolicy(_StartupCase):
    """W8 reachable only at C1: the initial decode captures are BS1-only,
    W4 keeps every bucket, and the shared provisioning stays full."""

    def setUp(self):
        super().setUp()
        self.cfg = self._publish(_published_config())
        self.applied = []
        self.seen = []
        self.worker = _LaunchWorker(
            self._config_path(W8_REACHABLE_AT_C1), self.seen, applied=self.applied
        )
        self._run_startup(self.worker)

    def test_shared_logits_provisioned_from_full_buckets(self):
        # The review-1 regression: max_decode_logits_rows under the FULL
        # canonical list is max(bs) * W8 tokens = 8 * 8 = 64 rows. A whole-
        # startup pruned scope provisions 1 * 8 = 8 rows here (red at the
        # whole-startup-scope revision) and W4's later 8-bucket * 4-token
        # get_logits_buffer(24/32-row) caller then asserts on a buffer that
        # cannot be resized after the captured graphs bind it.
        self.assertEqual(self.worker.shared_rows_seen, [64])
        buffer = self.worker.shared.get_logits_buffer(16, rows=(LAUNCH_STEPS + 1) * 4)
        self.assertEqual(tuple(buffer.shape), (32, 16))

    def test_initial_target_verify_capture_sees_only_reachable_buckets(self):
        # RED on the exact base: unscoped, the launch state captures ALL of
        # C1-C6 (this assertion fails with the full list).
        self.assertEqual(
            self._seen(self.seen, "target"),
            [[1]],
            f"W8 target capture saw {self._seen(self.seen, 'target')}",
        )

    def test_initial_draft_captures_see_only_reachable_buckets(self):
        self.assertEqual(self._seen(self.seen, "draft-decode"), [[1]])
        self.assertEqual(self._seen(self.seen, "draft-extend"), [[1]])

    def test_full_config_restored_before_draft_provisioning_and_init_states(self):
        # The scope is only the decode boundary: the draft worker's own
        # provisioning runs on the published list, and so does everything
        # between the two boundaries.
        self.assertEqual(self._seen(self.seen, "draft-provisioning"), [CANONICAL_BS])
        self.assertEqual(self._seen(self.seen, "target_after"), [CANONICAL_BS])
        self.assertEqual(_pending_scope(), None)

    def test_narrower_width_still_captures_every_reachable_bucket(self):
        # The routing table says every bucket can step down to W4; pruning
        # the launch width must not prune the shared routing bucket list.
        self.assertEqual(self._seen(self.seen, f"build{NARROW_STEPS}"), [CANONICAL_BS])
        params = self.worker.adaptive_controller.params
        self.assertEqual(params.cuda_graph_bs_for_step(NARROW_STEPS), CANONICAL_BS)
        self.assertEqual(
            self.worker.adaptive_controller.launch_cuda_graph_bs, CANONICAL_BS
        )

    def test_published_config_and_legacy_alias_survive_the_window(self):
        after = get_exec().graph.cuda_graph_config
        self.assertIs(after, self.cfg, "published config was mutated in place")
        self.assertEqual(after.decode.bs, CANONICAL_BS)
        self.assertEqual(after.decode.backend, Backend.FULL)
        self.assertEqual(get_exec().graph.cuda_graph_bs_decode, STALE_LEGACY_BS)

    def test_the_boundary_records_its_provenance(self):
        sources = [src for src, _ in get_context().overrides_log()]
        self.assertIn("adaptive_spec.launch_capture", sources)
        self.assertIn("adaptive_spec.launch_capture_restore", sources)

    def test_launch_state_is_the_active_state_after_init(self):
        # init_states still ends on the launch width with the registered
        # (now BS1-only) graphs, not on a rebuilt state.
        self.assertEqual(self.applied, [LAUNCH_STEPS])


class TestLaunchWidthScopedAsymmetric(_StartupCase):
    """The subset comes from the policy, not a hard-coded [1]: a launch width
    reachable at C1 and C2 captures exactly those two buckets."""

    def test_launch_capture_follows_the_two_bucket_slot(self):
        self._publish(_published_config())
        seen = []
        worker = _LaunchWorker(self._config_path(W8_REACHABLE_AT_C1_C2), seen)
        self._run_startup(worker)
        self.assertEqual(self._seen(seen, "target"), [[1, 2]])
        self.assertEqual(self._seen(seen, "draft-decode"), [[1, 2]])
        self.assertEqual(self._seen(seen, f"build{NARROW_STEPS}"), [CANONICAL_BS])
        self.assertEqual(worker.shared_rows_seen, [64])


class TestNoScopingWhenNothingToPruneOrNoAdaptive(_StartupCase):
    def test_all_buckets_reachable_launches_an_ordinary_window(self):
        cfg = self._publish(_published_config())
        seen = []
        worker = _LaunchWorker(self._config_path(ALL_AT_EVERY_BUCKET), seen)
        self._run_startup(worker)
        self.assertEqual(self._seen(seen, "target"), [CANONICAL_BS])
        self.assertEqual(self._seen(seen, "draft-decode"), [CANONICAL_BS])
        # An equal bucket set must not queue a scope, so nothing re-published
        # the config at all.
        sources = [src for src, _ in get_context().overrides_log()]
        self.assertNotIn("adaptive_spec.launch_capture", sources)
        self.assertIs(get_exec().graph.cuda_graph_config, cfg)

    def test_graph_disabled_launch_captures_nothing_as_before(self):
        cfg = self._publish(_published_config(backend=Backend.DISABLED))
        seen = []
        worker = _LaunchWorker(self._config_path(W8_REACHABLE_AT_C1), seen)
        self._run_startup(worker)
        # DISABLED wins over any bucket list: every window reads the same
        # "no graphs" answer the startup consumers already give, and with no
        # decode buckets there is no shared logits buffer to size at all.
        for name in ("target", "draft-decode", "draft-extend", f"build{NARROW_STEPS}"):
            self.assertEqual(self._seen(seen, name), [None], name)
        self.assertEqual(worker.shared_rows_seen, [0])
        sources = [src for src, _ in get_context().overrides_log()]
        self.assertNotIn("adaptive_spec.launch_capture", sources)
        self.assertIs(get_exec().graph.cuda_graph_config, cfg)

    def test_nonadaptive_worker_keeps_the_full_startup_capture(self):
        # Fixed-width adaptive-off EAGLE: same window, same full list, no
        # override provenance anywhere.
        cfg = self._publish(_published_config())
        seen = []
        worker = _LaunchWorker(None, seen)
        self._run_startup(worker)
        self.assertEqual(self._seen(seen, "target"), [CANONICAL_BS])
        self.assertEqual(self._seen(seen, "draft-decode"), [CANONICAL_BS])
        self.assertEqual(self._seen(seen, "draft-extend"), [CANONICAL_BS])
        self.assertEqual(worker.shared_rows_seen, [64])
        sources = [src for src, _ in get_context().overrides_log()]
        self.assertNotIn("adaptive_spec.launch_capture", sources)
        self.assertIs(get_exec().graph.cuda_graph_config, cfg)

    def test_other_speculative_worker_is_not_touched(self):
        # A draft worker without an adaptive plan at all (DFLASH / Frozen-KV
        # MTP / 27B-DFlash2): init_all_cuda_graphs must capture both workers
        # with the published buckets, exactly as before.
        cfg = self._publish(_published_config())
        seen = []
        draft = SimpleNamespace(
            init_cuda_graphs=lambda: seen.append(("draft", _capture_seen()))
        )
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(
                init_cuda_graphs=lambda: seen.append(("target", _capture_seen()))
            ),
            draft_worker=draft,
        )
        Scheduler.init_all_cuda_graphs(scheduler)
        self.assertEqual(self._seen(seen, "target"), [CANONICAL_BS])
        self.assertEqual(self._seen(seen, "draft"), [CANONICAL_BS])
        self.assertIs(get_exec().graph.cuda_graph_config, cfg)

    def test_no_draft_worker_captures_the_target_alone(self):
        self._publish(_published_config())
        seen = []
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(
                init_cuda_graphs=lambda: seen.append(("target", _capture_seen()))
            ),
            draft_worker=None,
        )
        Scheduler.init_all_cuda_graphs(scheduler)
        self.assertEqual(self._seen(seen, "target"), [CANONICAL_BS])


class TestRestoreAfterCaptureErrors(_StartupCase):
    def test_failed_draft_capture_restores_the_published_config(self):
        cfg = self._publish(_published_config())
        seen = []
        worker = _LaunchWorker(self._config_path(W8_REACHABLE_AT_C1), seen)

        def exploding_draft():
            with _decode_capture_scope():
                seen.append(("draft-decode", _capture_seen()))
            raise RuntimeError("draft capture blew up")

        worker._draft_worker.init_cuda_graphs = exploding_draft
        with self.assertRaises(RuntimeError):
            self._run_startup(worker)
        self.assertIs(get_exec().graph.cuda_graph_config, cfg)
        self.assertEqual(get_exec().graph.cuda_graph_config.decode.bs, CANONICAL_BS)
        self.assertEqual(get_exec().graph.cuda_graph_bs_decode, STALE_LEGACY_BS)
        # The queue is gone even though no boundary finished after it.
        self.assertEqual(_pending_scope(), None)
        # Review-1 P1: the shared logits buffer was provisioned BEFORE the
        # blew-up draft capture, from the FULL buckets, and stays usable.
        self.assertEqual(worker.shared_rows_seen, [64])

    def test_failed_target_capture_restores_and_clears_the_queue(self):
        cfg = self._publish(_published_config())
        seen = []
        worker = _LaunchWorker(self._config_path(W8_REACHABLE_AT_C1), seen)
        with self.assertRaises(RuntimeError):
            self._run_startup(worker, raise_in_target=True)
        self.assertIs(get_exec().graph.cuda_graph_config, cfg)
        self.assertEqual(_pending_scope(), None)

    def test_failed_width_capture_restores_the_published_config(self):
        cfg = self._publish(_published_config())
        seen = []
        worker = _LaunchWorker(
            self._config_path(W8_REACHABLE_AT_C1), seen, raise_in_build=True
        )
        with self.assertRaises(RuntimeError):
            self._run_startup(worker)
        self.assertIs(get_exec().graph.cuda_graph_config, cfg)
        self.assertEqual(get_exec().graph.cuda_graph_bs_decode, STALE_LEGACY_BS)

    def test_a_candidate_wider_than_the_launch_fails_before_any_capture(self):
        # Fail loud at startup, BEFORE the first graph is captured -- shared
        # provisioning included: launching at W4 with W8 as a candidate cannot
        # resize what the captured graphs already point at.
        cfg = self._publish(_published_config())
        seen = []
        worker = _LaunchWorker(self._config_path(W8_REACHABLE_AT_C1), seen)
        # Rebuild the controller so its launch-width snapshot matches the
        # W4 launch.
        worker.speculative_num_steps = NARROW_STEPS
        worker.speculative_num_draft_tokens = NARROW_STEPS + 1
        worker._draft_worker.speculative_num_steps = NARROW_STEPS
        worker._draft_worker.speculative_num_draft_tokens = NARROW_STEPS + 1
        worker.adaptive_controller = AdaptiveController(
            worker, config_path=self._config_path(W8_REACHABLE_AT_C1)
        )
        with self.assertRaises(ValueError):
            self._run_startup(worker)
        self.assertEqual(seen, [], "nothing was captured once the table was refused")
        self.assertEqual(worker.shared_rows_seen, [])
        self.assertIs(get_exec().graph.cuda_graph_config, cfg)

    def test_a_leaked_queue_fails_loud_on_the_next_launch(self):
        self._publish(_published_config())
        seen = []
        worker = _LaunchWorker(self._config_path(W8_REACHABLE_AT_C1), seen)
        prepare = getattr(worker, "prepare_adaptive_launch_capture", None)
        self.assertIsNotNone(
            prepare, "adaptive launch capture must be planned, not wrapped"
        )
        prepare()
        self.assertNotEqual(_pending_scope(), None)
        with self.assertRaises(RuntimeError):
            prepare()
        _clear_pending_scope()


if __name__ == "__main__":
    unittest.main()
