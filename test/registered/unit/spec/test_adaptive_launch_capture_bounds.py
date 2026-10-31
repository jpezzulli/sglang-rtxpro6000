"""CPU-only checks for the launch and capture bounds of adaptive widths.

The launch width is the capacity every other bound inherits from: the flat
``--speculative-num-steps`` / ``--speculative-num-draft-tokens`` leaves size the
process-shared logits buffer, the eager fixed-max buffers, the QSA pending
index-K ring and the request reservation once, before any width is chosen at
run time. Three mechanics are pinned here:

* a candidate wider than the launch width is REFUSED ahead of the first capture
  (``AdaptiveController.validate_candidates``, reached from both
  ``plan_launch_capture`` and ``init_states``), never silently resized under
  graphs that already bind the old buffers;
* the launch width's start-up capture is pruned to the buckets its policy can
  actually route to, but only AT the decode-capture boundaries: the queued
  ``adaptive_launch_capture_scope`` swaps the canonical
  ``cuda_graph_config[decode].bs`` for one boundary and restores the published
  object on every exit path, so provisioning still sees the FULL list. The
  per-candidate build prunes the same canonical leaf, not the legacy
  ``cuda_graph_bs_decode`` alias that the resolution pipeline folded away;
* the QSA pending ring's own span model is asked, per pool, whether it covers
  every width the run can reach, so a step-down reuses the launch-maximum
  allocation instead of aliasing two live verify positions onto one slot.

All evidence is CPU-only: real capture, replay and VRAM numbers are GPU work.
"""

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.qsa.metadata import pending_ring_groups_required
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    PhaseConfig,
    adaptive_launch_capture_scope,
    check_cuda_graph_backend,
    clear_adaptive_launch_capture_scope,
    scoped_capture_cuda_graph_config,
    set_adaptive_launch_capture_scope,
)
from sglang.srt.runtime_context import get_context, get_exec
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.adaptive_spec_params import AdaptiveSpeculativeParams
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

SHIPPED_CONFIG = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        *[os.pardir] * 4,
        "configs",
        "experimental",
        "flashnext-native",
        "adaptive-next-v3.0.2.json",
    )
)
FULL_BUCKETS = [1, 2, 4, 8]


class _StartupCase(CustomTestCase):
    """A published config, canonical decode buckets, and a worker stand-in."""

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(speculative_algorithm="EAGLE")
        override.install()
        self.addCleanup(override.restore)
        self.addCleanup(clear_adaptive_launch_capture_scope)
        self.published = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL, bs=list(FULL_BUCKETS))
        )
        get_context().override("_StartupCase", cuda_graph_config=self.published)
        self._dir = tempfile.mkdtemp(prefix="adaptive-capture-")
        self.addCleanup(shutil.rmtree, self._dir, True)

    def _config_path(self, table: dict) -> str:
        path = os.path.join(self._dir, "cfg.json")
        with open(path, "w") as f:
            json.dump(table, f)
        return path

    def _worker(self, launch_steps: int, cfg_path: str, pools=None):
        """An EAGLEWorkerV2 with only what the launch-capture path reads.

        The states are recorded, not built: no CUDA on this host.
        """
        worker = object.__new__(EAGLEWorkerV2)
        worker.speculative_num_steps = launch_steps
        worker.speculative_num_draft_tokens = launch_steps + 1
        worker.adaptive_controller = None
        worker._draft_worker = SimpleNamespace(
            speculative_num_steps=launch_steps,
            speculative_num_draft_tokens=launch_steps + 1,
            draft_attn_backend=object(),
            cuda_graph_runner=object(),
            draft_extend_attn_backend=object(),
            cuda_graph_runner_for_draft_extend=object(),
            _rebuild_topk1_chain_buffers=lambda: None,
            draft_runner=SimpleNamespace(
                token_to_kv_pool=(pools or {}).get("draft", object()),
                draft_attn_backend=object(),
                attn_backend=object(),
            ),
        )
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=object(),
                decode_cuda_graph_runner=object(),
                token_to_kv_pool=(pools or {}).get("target", object()),
            )
        )
        if cfg_path is not None:
            worker.adaptive_controller = AdaptiveController(
                worker,
                AdaptiveSpeculativeParams(
                    initial_steps=launch_steps, cfg_path=cfg_path
                ),
            )
        return worker

    def _recorded_states(self, worker):
        """Capture the per-width prunes instead of building real states."""
        requests = {}

        def build(
            speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs=None
        ):
            requests[speculative_num_steps] = cuda_graph_bs
            return SpecRuntimeState(
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                draft_attn_backend=None,
                cuda_graph_runner=None,
                target_attn_backend=None,
                target_graph_runner=None,
                draft_extend_attn_backend=None,
                cuda_graph_runner_for_draft_extend=None,
            )

        worker.build_adaptive_runtime_state = build
        return requests

    def _decode_bs_now(self):
        cfg = get_exec().graph.cuda_graph_config
        return list(cfg.decode.bs) if cfg.decode.bs else None


class TestLaunchWidthCaptureScope(_StartupCase):
    def test_the_launch_capture_is_pruned_to_the_buckets_the_width_can_reach(self):
        worker = self._worker(7, SHIPPED_CONFIG)
        states = self._recorded_states(worker)
        controller = worker.adaptive_controller

        worker.prepare_adaptive_launch_capture()
        # The queued scope, as the target verify and draft decode boundaries
        # see it: only BS1, because C1 is the only bucket steps 7 can route to.
        with adaptive_launch_capture_scope():
            self.assertEqual(self._decode_bs_now(), [1])
        # Restored for the next provisioning step and for init_states, which
        # must size the narrow state against the FULL list it retained.
        self.assertIs(get_exec().graph.cuda_graph_config, self.published)
        self.assertEqual(controller.launch_cuda_graph_bs, FULL_BUCKETS)
        controller.init_states(cuda_graph_bs=controller.launch_cuda_graph_bs)
        self.assertEqual(states, {7: [1], 3: FULL_BUCKETS})
        self.assertEqual(worker.speculative_num_steps, 7)

    def test_the_wider_than_launch_table_is_refused_before_any_capture(self):
        worker = self._worker(3, SHIPPED_CONFIG)
        states = self._recorded_states(worker)
        with self.assertRaises(ValueError) as ctx:
            worker.prepare_adaptive_launch_capture()
        self.assertIn("7", str(ctx.exception))
        self.assertEqual(states, {}, "no graph was captured for a refused table")
        self.assertEqual(self._decode_bs_now(), FULL_BUCKETS)

    def test_no_scope_is_queued_when_every_bucket_is_reachable(self):
        worker = self._worker(7, self._config_path({"1": {"candidate_steps": [3, 7]}}))
        worker.prepare_adaptive_launch_capture()
        with adaptive_launch_capture_scope():
            self.assertEqual(self._decode_bs_now(), FULL_BUCKETS)
        # ...and a non-adaptive (fixed width) launch asks for nothing either.
        fixed = self._worker(3, None)
        fixed.prepare_adaptive_launch_capture()
        with adaptive_launch_capture_scope():
            self.assertEqual(self._decode_bs_now(), FULL_BUCKETS)

    def test_a_leaked_queue_fails_loud_on_the_next_launch(self):
        worker = self._worker(7, SHIPPED_CONFIG)
        worker.prepare_adaptive_launch_capture()
        with self.assertRaisesRegex(RuntimeError, "already pending"):
            set_adaptive_launch_capture_scope(scoped_capture_cuda_graph_config([1]))
        clear_adaptive_launch_capture_scope()
        worker.prepare_adaptive_launch_capture()

    def test_the_bucket_list_is_read_from_the_canonical_config(self):
        worker = self._worker(7, SHIPPED_CONFIG)
        self.assertEqual(worker._decode_graph_capture_bs(), FULL_BUCKETS)
        # A stale legacy alias must not shadow the canonical list -- the
        # resolution pipeline folds it in once and override does not sync back.
        get_context().override("_test", cuda_graph_bs_decode=[4])
        self.assertEqual(worker._decode_graph_capture_bs(), FULL_BUCKETS)
        get_context().override(
            "_test",
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend=Backend.DISABLED, bs=list(FULL_BUCKETS))
            ),
        )
        self.assertIsNone(worker._decode_graph_capture_bs())


class TestPerStateCaptureWindow(_StartupCase):
    def test_the_prune_reaches_the_capture_list_consumer(self):
        worker = self._worker(7, SHIPPED_CONFIG)
        with worker._override_worker_state(3, 4, cuda_graph_bs=[2, 4, 8]):
            self.assertEqual(self._decode_bs_now(), [2, 4, 8])
            self.assertFalse(check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED))
        self.assertIs(get_exec().graph.cuda_graph_config, self.published)

    def test_an_empty_prune_disables_the_decode_phase_for_the_window(self):
        worker = self._worker(7, SHIPPED_CONFIG)
        with worker._override_worker_state(7, 8, cuda_graph_bs=[]):
            self.assertTrue(check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED))
            self.assertEqual(self._decode_bs_now(), None)
        self.assertFalse(check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED))
        self.assertIs(get_exec().graph.cuda_graph_config, self.published)
        self.assertFalse(get_exec().graph.disable_cuda_graph)

    def test_the_window_restores_the_published_config_after_a_failed_capture(self):
        worker = self._worker(7, SHIPPED_CONFIG)
        with self.assertRaises(RuntimeError):
            with worker._override_worker_state(3, 4, cuda_graph_bs=[2]):
                raise RuntimeError("capture failed")
        self.assertIs(get_exec().graph.cuda_graph_config, self.published)

    def test_the_scoped_config_never_mutates_or_carries_the_published_prefill(self):
        scoped = scoped_capture_cuda_graph_config([1])
        self.assertEqual(self.published.decode.bs, FULL_BUCKETS)
        self.assertEqual(scoped.prefill, self.published.prefill)


class TestLaunchMaximumPersistentCapacity(_StartupCase):
    """The QSA pending ring must cover every width the run can reach."""

    RATIO = 4  # small enough that the shipped W4 and W8 windows differ

    def _pools(self, target_groups, draft_groups):
        return {
            "target": SimpleNamespace(
                qsa_num_groups=target_groups, qsa_compress_ratio=self.RATIO
            ),
            "draft": SimpleNamespace(
                qsa_num_groups=draft_groups, qsa_compress_ratio=self.RATIO
            ),
        }

    def test_the_launch_maximum_allocation_serves_every_narrower_width(self):
        # The ring is sized off the widest window the table can reach (the W8
        # launch maximum), which every narrower width then reuses.
        widest = pending_ring_groups_required(draft_tokens=8, compress_ratio=self.RATIO)
        worker = self._worker(7, SHIPPED_CONFIG, pools=self._pools(widest, widest))
        worker._validate_adaptive_widths()  # no raise

    def test_a_width_above_the_ring_capacity_is_refused_at_start_up(self):
        # A ring sized for the W4 window only: the W8 candidate the table can
        # select would alias two live verify positions onto one slot.
        narrow = pending_ring_groups_required(draft_tokens=4, compress_ratio=self.RATIO)
        worker = self._worker(7, SHIPPED_CONFIG, pools=self._pools(narrow, narrow))
        with self.assertRaises(ValueError) as ctx:
            worker._validate_adaptive_widths()
        self.assertIn("QSA pending index-key ring", str(ctx.exception))

    def test_the_draft_ring_counts_even_when_the_target_ring_is_wide_enough(self):
        wide = pending_ring_groups_required(draft_tokens=8, compress_ratio=self.RATIO)
        narrow = pending_ring_groups_required(draft_tokens=4, compress_ratio=self.RATIO)
        worker = self._worker(7, SHIPPED_CONFIG, pools=self._pools(wide, narrow))
        with self.assertRaises(ValueError) as ctx:
            worker._validate_adaptive_widths()
        self.assertIn("draft", str(ctx.exception))

    def test_a_pool_without_a_pending_ring_is_not_measured(self):
        # Tokenwise QSA and the 27B DFlash2 draft have no pending index-K ring.
        worker = self._worker(7, SHIPPED_CONFIG, pools=self._pools(None, None))
        worker._validate_adaptive_widths()

    def test_no_controller_means_no_adaptive_width_to_check(self):
        worker = self._worker(3, None)
        worker._validate_adaptive_widths()


if __name__ == "__main__":
    unittest.main()
