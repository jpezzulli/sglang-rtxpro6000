"""CPU-only regression: every extra adaptive target graph gets its own warmup.

``BaseRunner.warmup`` (kernel warmup + flashinfer autotune) runs ONCE per model
runner, gated by ``model_runner._kernel_warmed_up``. The adaptive controller builds
one ``SpecRuntimeState`` per candidate width and captures a fresh target
``DecodeCudaGraphRunner`` for each, but those runners share the launch state's
model runner -- so only the target graph the server launched with was ever tuned,
and every other width captured whatever tactic the autotuner happened to leave
behind for the launch width (the draft autotuner does not tune the target).

``adaptive_target_graph_warmup`` opts a candidate state's capture back in: it
clears the run-once gate and binds ``model_runner.attn_backend`` to the backend
those graphs are about to be captured against -- the dummy forward reads it off
the runner, so warming up against the live state's backend would tune the wrong
graph metadata -- restoring both, including the absence of the attribute, in
``finally``. It is opt-in (SGLANG_ADAPTIVE_TARGET_AUTOTUNE=1) and keeps the
existing warmup disable/determinism gates: the context alone tunes nothing, and
``should_run_flashinfer_autotune`` still says the last word. Nothing here changes
what is captured, which buffers are private, or who owns the resulting graphs.

All evidence is CPU-only (no CUDA on this host): that per-width tuning produces
different tactics is a GPU-window claim.
"""

import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.speculative import adaptive_runtime_state as ars_module
from sglang.srt.speculative import eagle_worker_v2 as eagle_module
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

FLAG = "SGLANG_ADAPTIVE_TARGET_AUTOTUNE"
BASE_RUNNER = "sglang.srt.model_executor.runner.base_runner"


def adaptive_target_graph_warmup(model_runner, attn_backend):
    # Resolved at call time: a tree without the mechanism fails these checks by
    # name, instead of failing the module (and every other check in it) at import.
    fn = getattr(ars_module, "adaptive_target_graph_warmup", None)
    if fn is None:
        raise AssertionError(
            "adaptive_runtime_state.adaptive_target_graph_warmup is missing: "
            "extra target graphs are captured with the run-once gate still shut"
        )
    return fn(model_runner, attn_backend)


class TestAdaptiveTargetGraphWarmup(CustomTestCase):
    def test_disabled_keeps_prior_behaviour_including_constructor_writes(self):
        for value in ("", "0", "true"):
            with self.subTest(value=value), patch.dict(os.environ, {FLAG: value}):
                base, candidate = object(), object()
                mr = SimpleNamespace(attn_backend=base, _kernel_warmed_up=True)
                with adaptive_target_graph_warmup(mr, candidate):
                    # No rebinding, no re-arming: the run-once gate stays shut.
                    self.assertIs(mr.attn_backend, base)
                    self.assertTrue(mr._kernel_warmed_up)
                    mr._kernel_warmed_up = "constructor-owned"
                self.assertEqual(mr._kernel_warmed_up, "constructor-owned")

    def test_enabled_binds_the_candidate_then_restores(self):
        with patch.dict(os.environ, {FLAG: "1"}):
            base, candidate = object(), object()
            mr = SimpleNamespace(attn_backend=base, _kernel_warmed_up=True)
            with adaptive_target_graph_warmup(mr, candidate):
                self.assertIs(mr.attn_backend, candidate)
                self.assertFalse(mr._kernel_warmed_up)
                mr._kernel_warmed_up = True  # what warmup() leaves behind
            self.assertIs(mr.attn_backend, base)
            self.assertTrue(mr._kernel_warmed_up)

    def test_a_failed_capture_restores_both_attributes(self):
        with patch.dict(os.environ, {FLAG: "1"}):
            base = object()
            mr = SimpleNamespace(attn_backend=base, _kernel_warmed_up=True)
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                with adaptive_target_graph_warmup(mr, object()):
                    raise RuntimeError("capture failed")
            self.assertIs(mr.attn_backend, base)
            self.assertTrue(mr._kernel_warmed_up)

    def test_a_runner_that_never_had_the_gate_keeps_not_having_it(self):
        with patch.dict(os.environ, {FLAG: "1"}):
            mr = SimpleNamespace(attn_backend=object())
            with adaptive_target_graph_warmup(mr, object()):
                self.assertFalse(mr._kernel_warmed_up)
            self.assertFalse(hasattr(mr, "_kernel_warmed_up"))

    def test_the_real_warmup_retunes_once_per_state_and_keeps_its_own_gates(self):
        from sglang.srt.model_executor.runner.base_runner import BaseRunner

        for enabled, allowed in ((False, True), (True, False), (True, True)):
            with self.subTest(enabled=enabled, allowed=allowed):
                base, candidate = object(), object()
                mr = SimpleNamespace(
                    attn_backend=base,
                    _kernel_warmed_up=True,
                    device="cuda",
                    ps=SimpleNamespace(pp_size=1),
                )
                seen = []
                runner = SimpleNamespace(
                    model_runner=mr,
                    _pre_initialize_flashinfer_allreduce_workspace=Mock(),
                    _pre_initialize_fi_a2a_workspace=Mock(),
                    _autotune_buffers=lambda: (object(), 1),
                    _flashinfer_autotune=lambda **kw: seen.append(mr.attn_backend),
                )
                with (
                    patch.dict(os.environ, {FLAG: "1" if enabled else "0"}),
                    patch(
                        BASE_RUNNER + ".should_run_flashinfer_autotune",
                        return_value=allowed,
                    ),
                    patch(BASE_RUNNER + ".maybe_flashinfer_autotune_extend"),
                ):
                    # Two constructions inside one warmup window: the gate is
                    # cleared once for the state, not once per replay.
                    with adaptive_target_graph_warmup(mr, candidate):
                        BaseRunner.warmup(runner)
                        BaseRunner.warmup(runner)
                self.assertEqual(seen, [candidate] if enabled and allowed else [])
                self.assertIs(mr.attn_backend, base)
                self.assertTrue(mr._kernel_warmed_up)


class _CandidateTargetGraphRunner:
    """DecodeCudaGraphRunner stand-in: captures against whatever the runner says.

    Mirrors the real construction order (capture() -> warmup() -> a dummy forward
    that reads model_runner.attn_backend and the run-once gate) and records both,
    so the test can see what the capture was armed against.
    """

    def __init__(self, model_runner, **kwargs):
        self.captured_backend = model_runner.attn_backend
        self.was_armed = not getattr(model_runner, "_kernel_warmed_up", False)
        model_runner._kernel_warmed_up = True  # BaseRunner.warmup's bookkeeping
        self.capture_bs = [1]


def _model_runner(*, warmed):
    """Target model runner whose live backend is the launch state's."""
    return SimpleNamespace(
        attn_backend=SimpleNamespace(name="target-launch"),
        init_new_workspace=False,
        _kernel_warmed_up=warmed,
        _get_attention_backend=lambda init_new_workspace=False: SimpleNamespace(
            name="target-candidate"
        ),
        maybe_capture_gdn_recovery_graphs=lambda **kwargs: None,
    )


class _WorkerStub:
    """Enough EAGLEWorkerV2 to build one extra state without a GPU.

    The capture mechanics this state owns (private input buffers, the draft
    backend's fresh workspace, graph ownership) are theirs to test; here they are
    held still so a failure points at the warmup opt-in and nowhere else.
    """

    build_adaptive_runtime_state = EAGLEWorkerV2.build_adaptive_runtime_state

    def __init__(self):
        self.device = "cpu"
        self.gpu_id = 0
        self._additional_graph_memory_usage = {}
        self._additional_graph_time_usage = {}
        self._target_worker = SimpleNamespace(model_runner=_model_runner(warmed=True))
        self._draft_worker = SimpleNamespace(
            draft_attn_backend=SimpleNamespace(name="draft"),
            cuda_graph_runner=SimpleNamespace(name="draft-graph"),
            draft_extend_attn_backend=SimpleNamespace(name="extend"),
            cuda_graph_runner_for_draft_extend=SimpleNamespace(name="extend-graph"),
            init_attention_backend=lambda: None,
            _configure_qsa_mtp_index_share=lambda: None,
            _capture_cuda_graphs=lambda: None,
        )

    def _override_worker_state(self, *args, **kwargs):
        return contextlib.nullcontext()

    def _private_capture_scope(self):
        return contextlib.nullcontext()

    def _own_draft_extend_backend(self):
        return None


class TestBuildAdaptiveRuntimeStateCallSite(CustomTestCase):
    """The mechanism only exists if the extra state's capture goes through it."""

    def _build(self, worker, *, enabled, runner_cls=_CandidateTargetGraphRunner):
        with (
            patch.dict(os.environ, {FLAG: "1" if enabled else ""}),
            patch.object(
                eagle_module, "check_cuda_graph_backend", lambda *a, **k: False
            ),
            patch.object(eagle_module, "get_available_gpu_memory", lambda *a: 0.0),
            patch.object(eagle_module, "DecodeCudaGraphRunner", runner_cls),
        ):
            return worker.build_adaptive_runtime_state(
                speculative_num_steps=3, speculative_num_draft_tokens=4
            )

    def test_the_opt_in_rearms_the_capture_of_the_states_own_target_graph(self):
        worker = _WorkerStub()
        state = self._build(worker, enabled=True)
        self.assertTrue(
            state.target_graph_runner.was_armed,
            "the extra target graph captured with the run-once gate still shut",
        )
        self.assertIs(
            state.target_graph_runner.captured_backend, state.target_attn_backend
        )
        # Restored: the launch state's backend, and its already-tuned status.
        mr = worker._target_worker.model_runner
        self.assertEqual(mr.attn_backend.name, "target-launch")
        self.assertTrue(mr._kernel_warmed_up)

    def test_without_the_opt_in_the_build_is_exactly_as_before(self):
        worker = _WorkerStub()
        state = self._build(worker, enabled=False)
        self.assertFalse(state.target_graph_runner.was_armed)
        self.assertEqual(
            state.target_graph_runner.captured_backend.name, "target-launch"
        )
        mr = worker._target_worker.model_runner
        self.assertEqual(mr.attn_backend.name, "target-launch")
        self.assertTrue(mr._kernel_warmed_up)

    def test_a_graph_capture_failure_leaves_no_state_behind(self):
        class _Boom(_CandidateTargetGraphRunner):
            def __init__(self, model_runner, **kwargs):
                raise RuntimeError("capture failed")

        worker = _WorkerStub()
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            self._build(worker, enabled=True, runner_cls=_Boom)
        mr = worker._target_worker.model_runner
        self.assertEqual(mr.attn_backend.name, "target-launch")
        self.assertTrue(mr._kernel_warmed_up)


if __name__ == "__main__":
    unittest.main()
