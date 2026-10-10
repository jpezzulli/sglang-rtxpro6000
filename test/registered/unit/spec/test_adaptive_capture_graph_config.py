"""CPU-only checks that adaptive state capture reads and scopes the CANONICAL
CUDA-graph config, not the legacy alias leaves.

The current runners size their capture lists from
``get_exec().graph.cuda_graph_config[decode].bs`` and ask
``check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)``; the legacy
``cuda_graph_bs_decode`` / ``disable_cuda_graph`` leaves are folded into that
config once at startup and ``RuntimeContext.override`` does not synchronize
them back. So an adaptive per-width build that overrides the legacy leaves
captures the *published* buckets for every width (a requested ``[1]`` silently
became ``[1, 2, 4, 8]``), an empty prune keeps ``backend=full`` (the width
builds graph runners over a list nothing pruned), and an adaptive init that
reads the legacy leaf routes C1 off ``None`` -- skipping the BS1 prerequisite
even though ``cuda_graph_bs_for_step`` itself prunes correctly.

These checks run against a real published ``RuntimeContext`` and the real
capture-list consumer ``get_batch_sizes_to_capture``; no mocks stand in for
what the decode runner reads.
"""

from types import SimpleNamespace

from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    get_batch_sizes_to_capture,
)
from sglang.srt.runtime_context import get_context, get_exec, get_parallel
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

CANONICAL_BS = [1, 2, 4, 8]
# A deliberately contradictory legacy alias: the canonical launch below says
# 1/2/4/8, a stale legacy leaf says something else; the capture must follow
# the canonical config, never the alias.
STALE_LEGACY_BS = [8, 16, 32]


def _published_config() -> CudaGraphConfig:
    return CudaGraphConfig.from_dict(
        {"decode": {"backend": Backend.FULL, "max_bs": 256, "bs": list(CANONICAL_BS)}}
    )


class _CaptureWorkerStub:
    """Enough EAGLEWorkerV2 for _override_worker_state / _decode_graph_capture_bs."""

    _override_worker_state = EAGLEWorkerV2._override_worker_state

    def _decode_graph_capture_bs(self):
        # Resolved at call time: a tree without the canonical-config repair
        # fails THIS check with a named reason, not at import.
        fn = getattr(EAGLEWorkerV2, "_decode_graph_capture_bs", None)
        if fn is None:
            raise AssertionError(
                "EAGLEWorkerV2._decode_graph_capture_bs is missing: adaptive "
                "init still reads the legacy cuda_graph_bs_decode alias"
            )
        return fn(self)

    def __init__(self):
        self.speculative_num_steps = 7
        self.speculative_num_draft_tokens = 8
        self._draft_worker = SimpleNamespace(
            speculative_num_steps=7,
            speculative_num_draft_tokens=8,
            draft_attn_backend=None,
            draft_extend_attn_backend=None,
            draft_runner=SimpleNamespace(draft_attn_backend=None, attn_backend=None),
            cuda_graph_runner=None,
            cuda_graph_runner_for_draft_extend=None,
            _rebuild_topk1_chain_buffers=lambda: None,
        )


class _PublishedCase(CustomTestCase):
    def _publish(self, cuda_graph_config, legacy_bs):
        override = get_context().override_server_args(
            cuda_graph_config=cuda_graph_config,
            cuda_graph_bs_decode=legacy_bs,
        )
        published = override.install()
        self.addCleanup(override.restore)
        return published

    def _probe_window(self, worker, cuda_graph_bs):
        """Run the real capture window; record what the capture-list consumers
        see INSIDE it (this is where DecodeCudaGraphRunner and the draft
        _capture_cuda_graphs read their buckets)."""
        seen = {}

        # Single-rank widths scoped through the sanctioned parallel override;
        # everything the capture list reads on the config tier is the real
        # published bag.
        with get_parallel().override(attn_tp_size=1, attn_cp_size=1):
            with worker._override_worker_state(7, 8, cuda_graph_bs=cuda_graph_bs):
                # This is the branch the width build itself takes: a disabled
                # decode phase never reaches get_batch_sizes_to_capture.
                seen["decode_disabled"] = check_cuda_graph_backend(
                    Phase.DECODE, Backend.DISABLED
                )
                if not seen["decode_disabled"]:
                    runner_stub = SimpleNamespace(
                        req_to_token_pool=SimpleNamespace(size=64)
                    )
                    seen["capture_bs"] = get_batch_sizes_to_capture(runner_stub)[0]
                seen["legacy_bs"] = get_exec().graph.cuda_graph_bs_decode
                seen["prefill_backend"] = (
                    get_exec().graph.cuda_graph_config.prefill.backend
                )
        return seen


class TestCaptureWindowOnCanonicalConfig(_PublishedCase):
    def test_pruned_buckets_reach_the_real_capture_list_consumer(self):
        self._publish(_published_config(), STALE_LEGACY_BS)
        seen = self._probe_window(_CaptureWorkerStub(), [1])
        # The state asked for BS1 only; get_batch_sizes_to_capture is exactly
        # what its DecodeCudaGraphRunner will capture.
        self.assertEqual(seen["capture_bs"], [1])
        self.assertFalse(seen["decode_disabled"])
        # The contradictory legacy alias was not rewritten either: the scoped
        # override lives on the canonical leaf only.
        self.assertEqual(seen["legacy_bs"], STALE_LEGACY_BS)

    def test_empty_prune_disables_decode_for_the_window(self):
        cfg = _published_config()
        self._publish(cfg, STALE_LEGACY_BS)
        seen = self._probe_window(_CaptureWorkerStub(), [])
        # This is the consumer the width build branches on; "bs=[]" alone with
        # backend=full still says "graphs on" and the build would capture a
        # runner over an emptied list.
        self.assertTrue(seen["decode_disabled"])
        # Only the decode phase is scoped.
        self.assertEqual(seen["prefill_backend"], _published_config().prefill.backend)

    def test_the_published_config_and_legacy_leaves_survive_the_window(self):
        cfg = _published_config()
        self._publish(cfg, STALE_LEGACY_BS)
        self._probe_window(_CaptureWorkerStub(), [1])
        self._probe_window(_CaptureWorkerStub(), [])
        after = get_exec().graph.cuda_graph_config
        self.assertIs(after, cfg, "published config was mutated in place")
        self.assertEqual(after.decode.bs, CANONICAL_BS)
        self.assertEqual(after.decode.backend, Backend.FULL)
        self.assertEqual(get_exec().graph.cuda_graph_bs_decode, STALE_LEGACY_BS)
        self.assertFalse(get_exec().graph.disable_cuda_graph)

    def test_the_window_restores_the_config_after_a_failed_capture(self):
        cfg = _published_config()
        self._publish(cfg, STALE_LEGACY_BS)
        worker = _CaptureWorkerStub()
        with self.assertRaises(RuntimeError):
            with worker._override_worker_state(7, 8, cuda_graph_bs=[1]):
                raise RuntimeError("capture blew up")
        self.assertIs(get_exec().graph.cuda_graph_config, cfg)
        self.assertEqual(get_exec().graph.cuda_graph_config.decode.bs, CANONICAL_BS)
        self.assertEqual(get_exec().graph.cuda_graph_bs_decode, STALE_LEGACY_BS)


class TestAdaptiveInitReadsCanonicalBs(_PublishedCase):
    def test_canonical_only_launch_without_a_legacy_leaf_still_prunes(self):
        # The reproduced routing defect: --cuda-graph-config set the canonical
        # decode list, the legacy leaf stayed None, and the adaptive init read
        # the alias -> C1 got wide states with no BS1 prerequisite.
        self._publish(_published_config(), None)
        worker = _CaptureWorkerStub()
        self.assertEqual(worker._decode_graph_capture_bs(), CANONICAL_BS)

    def test_stale_legacy_leaf_does_not_shadow_the_canonical_list(self):
        cfg = CudaGraphConfig.from_dict(
            {"decode": {"backend": Backend.FULL, "bs": [1]}}
        )
        self._publish(cfg, STALE_LEGACY_BS)
        self.assertEqual(_CaptureWorkerStub()._decode_graph_capture_bs(), [1])

    def test_disabled_decode_has_no_bucket_list(self):
        cfg = _published_config()
        cfg.decode.backend = Backend.DISABLED
        self._publish(cfg, CANONICAL_BS)
        self.assertIsNone(_CaptureWorkerStub()._decode_graph_capture_bs())

    def test_unresolved_config_has_no_bucket_list(self):
        self._publish(None, STALE_LEGACY_BS)
        self.assertIsNone(_CaptureWorkerStub()._decode_graph_capture_bs())


if __name__ == "__main__":
    import unittest

    unittest.main()
