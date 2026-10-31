"""CPU-only checks for the shipped Next adaptive width policy and for WHEN a
width change is allowed to reach the worker.

``configs/experimental/flashnext-native/adaptive-next-v3.0.2.json`` is an
ordinary file for the existing ``--speculative-adaptive-config`` flag -- no
Penny-specific policy lives in the runtime. The batch-size floor 1 lists BOTH
shipped widths (steps 3 and 7, i.e. W4 and W8 at topk=1) and carries the
anti-oscillation hold (a 40 completed-feedback-batch hold after a switch,
doubled per direction reversal, capped at 80), while floor 2 lists only steps
3: every concurrent batch runs W4, immediately, whatever the C1 slot is holding.

Three mechanics are pinned here:

* the shipped numbers, read back through the real loader and path, and the
  damping the hold produces on a cycling acceptance stream (the shape that
  flipped the native controller 47 times in 72 s of acceptance-only feedback);
* a verify completion feeds the policy of the batch it came from and cannot
  reconfigure the worker. The width is applied by the next actual forward, from
  that forward's own batch size (``activate_step_by_batch``). A stale C1
  completion landing while a newer C4 batch is in flight must not hand that
  batch wide speculation;
* a tier that omits the hold keys (every other configuration, including the
  defaults and the single-candidate C>=2 tier) keeps the switch-on-every-decision
  policy, and a single-candidate tier has no re-decision to delay.

All evidence is CPU-only (no CUDA on this host): real capture, replay, numerics
and any hardware/performance claim remain GPU-window work.
"""

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.runtime_context import get_context, get_spec
from sglang.srt.speculative import adaptive_spec_params
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

# The shipped configuration, resolved from the repository layout and consumed by
# nothing but the existing loader -- these checks fail if the file moves, stops
# being valid JSON, or if the width policy drifts back into the runtime.
V302_CONFIG = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        *[os.pardir] * 4,
        "configs",
        "experimental",
        "flashnext-native",
        "adaptive-next-v3.0.2.json",
    )
)

CANDIDATE_STEPS = [3, 7]  # the two shipped C1 widths: W4 and W8 at topk=1
FAST = {"ema_alpha": 1.0, "warmup_batches": 0, "update_interval": 1}
# The hold the shipped C1 tier is required to carry.
C1_HOLD = {"switch_grace_batches": 40, "grace_backoff": 2.0, "max_grace_batches": 80}


def _state(steps: int) -> SpecRuntimeState:
    return SpecRuntimeState(
        speculative_num_steps=steps,
        speculative_num_draft_tokens=steps + 1,
        draft_attn_backend=None,
        cuda_graph_runner=None,
        target_attn_backend=SimpleNamespace(name=f"target-{steps}"),
        target_graph_runner=None,
        draft_extend_attn_backend=None,
        cuda_graph_runner_for_draft_extend=None,
    )


class _WorkerStub:
    """Enough AdaptiveSpecWorker to watch which width reaches the worker, and when."""

    def __init__(self, launch_steps: int):
        self.speculative_num_steps = launch_steps
        self.events: list[str] = []
        self.capture_requests: dict[int, list[int] | None] = {}

    def build_adaptive_runtime_state(
        self, speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs=None
    ):
        self.capture_requests[speculative_num_steps] = cuda_graph_bs
        return _state(speculative_num_steps)

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        if state.speculative_num_steps == self.speculative_num_steps:
            return
        self.events.append(f"apply:{state.speculative_num_steps}")
        self.speculative_num_steps = state.speculative_num_steps


class _ConfigCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._dir = tempfile.mkdtemp(prefix="adaptive-next-")

    def tearDown(self):
        shutil.rmtree(self._dir, ignore_errors=True)

    def _temp_config(self, cfg: dict) -> str:
        path = os.path.join(self._dir, "cfg.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
        return path

    def _controller(
        self, cfg, launch_steps=7, cuda_graph_bs=(1, 2, 4, 8), initial_steps=None
    ):
        """A controller over the real params loader, with the real states built."""
        cfg_path = self._temp_config(cfg) if isinstance(cfg, dict) else cfg
        worker = _WorkerStub(launch_steps)
        controller = AdaptiveController(
            worker,
            adaptive_spec_params.AdaptiveSpeculativeParams(
                initial_steps=(
                    launch_steps if initial_steps is None else initial_steps
                ),
                cfg_path=cfg_path,
            ),
        )
        controller.init_states(cuda_graph_bs=list(cuda_graph_bs))
        return controller, worker

    @staticmethod
    def _alternating(batches: int) -> list[list[int]]:
        return [[0] if i % 2 else [3] for i in range(1, batches + 1)]

    def _policy_switches(self, controller, feedback, batch_size=1):
        """Width changes the policy decides on, as (completed batch, steps)."""
        switches = []
        last = controller.params.get_steps_for_batch(batch_size)
        for i, counts in enumerate(feedback, start=1):
            controller.on_verify_complete(counts, batch_size=batch_size)
            steps = controller.params.get_steps_for_batch(batch_size)
            if steps != last:
                switches.append((i, steps))
                last = steps
        return switches


class TestShippedV302Policy(_ConfigCase):
    def test_the_shipped_file_is_plain_json_with_the_accepted_table(self):
        with open(V302_CONFIG) as f:
            shipped = json.load(f)
        # The one non-numeric key is prose, not a batch-size tier.
        self.assertEqual(sorted(k for k in shipped if k.isdigit()), ["1", "2"])
        self.assertEqual(shipped["1"]["candidate_steps"], CANDIDATE_STEPS)
        self.assertEqual(shipped["2"]["candidate_steps"], [3])
        self.assertEqual(shipped["1"]["up_hysteresis"], 0.0)
        self.assertEqual(shipped["1"]["down_hysteresis"], -0.25)
        self.assertEqual(shipped["1"]["ceiling_coeff"], 0)
        for key, value in C1_HOLD.items():
            self.assertEqual(shipped["1"][key], value, key)
        # The C>=2 tier carries no hold keys: one candidate, nothing to delay.
        self.assertNotIn("switch_grace_batches", shipped["2"])
        self.assertEqual(
            adaptive_spec_params.resolve_candidate_steps_from_config(V302_CONFIG),
            CANDIDATE_STEPS,
            "the launch maximum is the widest candidate: steps 7 / 8 draft tokens",
        )

    def test_the_shipped_widths_are_w4_or_w8_at_c1_and_w4_at_every_concurrency(self):
        params = adaptive_spec_params.AdaptiveSpeculativeParams(7, V302_CONFIG)
        self.assertEqual(params._slots[1].candidate_steps, CANDIDATE_STEPS)
        self.assertEqual(params._slots[1].switch_grace_batches, 40)
        self.assertEqual(params._slots[2].switch_grace_batches, 0)
        # steps + 1 is the draft width, so C1 runs W4 or W8 and C>1 runs W4.
        self.assertEqual({s + 1 for s in params._slots[1].candidate_steps}, {4, 8})
        for batch_size in (2, 3, 4, 6, 8, 32, 256):
            self.assertEqual(params.get_steps_for_batch(batch_size), 3)

    def test_the_shipped_hold_damps_a_cycling_workload_without_freezing_it(self):
        # The workload whose acceptance cycles between rejecting and fully
        # accepting every 10 batches -- 400 completed feedback batches -- is the
        # shape that ping-ponged the native controller (39 re-decisions on the
        # exact base). On the shipped file it is 5, spaced by the 40-batch hold
        # and its 80-batch reversal cap, and both widths are still selected.
        params = adaptive_spec_params.AdaptiveSpeculativeParams(7, V302_CONFIG)
        stream = ([[7]] * 10 + [[0]] * 10) * 20
        trace = []
        last = params.get_steps_for_batch(1)
        for i, counts in enumerate(stream, start=1):
            params.on_verify_complete(counts, batch_size=1)
            steps = params.get_steps_for_batch(1)
            if steps != last:
                trace.append((i, steps))
                last = steps
        self.assertEqual(trace, [(20, 3), (65, 7), (155, 3), (245, 7), (335, 3)])
        self.assertLessEqual(
            len(trace), len(stream) // C1_HOLD["switch_grace_batches"] + 1
        )
        self.assertEqual({steps for _, steps in trace}, set(CANDIDATE_STEPS))
        gaps = [b - a for (a, _), (b, _) in zip(trace, trace[1:])]
        self.assertTrue(all(gap >= C1_HOLD["switch_grace_batches"] for gap in gaps))
        self.assertEqual(params._slots[1]._grace, 80.0, "the reversal hold is capped")

    def test_the_same_stream_flips_every_batch_without_the_hold(self):
        # Same table with the keys omitted: the base behaviour the hold exists to
        # damp, so this file's C1 entry is what makes the shipped policy quiet.
        params = adaptive_spec_params.AdaptiveSpeculativeParams(
            7,
            self._temp_config(
                {
                    "1": {"candidate_steps": CANDIDATE_STEPS},
                    "2": {"candidate_steps": [3]},
                }
            ),
        )
        stream = ([[7]] * 10 + [[0]] * 10) * 20
        trace = []
        last = params.get_steps_for_batch(1)
        for i, counts in enumerate(stream, start=1):
            params.on_verify_complete(counts, batch_size=1)
            steps = params.get_steps_for_batch(1)
            if steps != last:
                trace.append((i, steps))
                last = steps
        self.assertGreater(len(trace), 30, "39 re-decisions without the hold")
        self.assertEqual(params._slots[1].switch_grace_batches, 0)


class TestWidthAppliesAtTheForwardBoundary(_ConfigCase):
    def _held_controller(self, cuda_graph_bs=(1, 2, 4, 8)):
        return self._controller(
            {
                "1": {
                    "candidate_steps": CANDIDATE_STEPS,
                    **FAST,
                    **C1_HOLD,
                },
                "2": {"candidate_steps": [3]},
            },
            cuda_graph_bs=cuda_graph_bs,
        )

    def test_an_alternating_c1_stream_holds_one_width_between_holds(self):
        controller, _ = self._held_controller()
        # Without the hold every one of these 200 batches flips the width. With
        # it: one switch, a 40-batch hold, the first reversal doubled to 80, the
        # next capped at 80 -- and the width still moves at batch 123, so the
        # hold slows adaptation instead of freezing it.
        switches = self._policy_switches(controller, self._alternating(200))
        self.assertEqual(switches, [(1, 3), (42, 7), (123, 3)])
        self.assertEqual(controller.params._slots[1]._grace, 80.0, "hold is capped")

    def test_the_c1_hold_never_delays_the_w4_routing_of_a_concurrent_batch(self):
        controller, worker = self._held_controller()
        for counts in self._alternating(60):
            controller.on_verify_complete(counts, batch_size=1)
            self.assertEqual(controller.params.get_steps_for_batch(2), 3)
            self.assertEqual(controller.params.get_steps_for_batch(8), 3)
            # Every concurrent C>=2 forward runs W4, immediately, whatever C1 is
            # holding -- and the first one is what applies the narrow state.
            controller.activate_step_by_batch(4)
        self.assertEqual(worker.events, ["apply:3"])
        self.assertGreater(controller.params._slots[1]._grace_until, 60)

    def test_a_stale_c1_completion_cannot_widen_the_in_flight_c4_batch(self):
        controller, worker = self._controller(
            {
                "1": {"candidate_steps": CANDIDATE_STEPS, **FAST},
                "2": {"candidate_steps": [3]},
            }
        )
        self.assertEqual(worker.speculative_num_steps, 7)  # launched at the widest

        # C1 settles at W4 and the next actual batch -- a C4 one -- starts.
        controller.on_verify_complete([0], batch_size=1)
        controller.activate_step_by_batch(1)
        self.assertEqual(worker.events, ["apply:3"])
        controller.activate_step_by_batch(4)
        self.assertEqual(worker.speculative_num_steps, 3)

        # The OLDER C1 batch's result lands now, after the C4 forward started,
        # and its policy wants W8.
        before = list(worker.events)
        controller.on_verify_complete([3], batch_size=1)
        self.assertEqual(controller.params.get_steps_for_batch(1), 7)
        self.assertEqual(
            worker.events, before, "a completion callback reconfigured the worker"
        )
        self.assertEqual(worker.speculative_num_steps, 3)

        # The C4 batch keeps decoding (overlap: further of its own iterations);
        # its width comes from its own batch size, not the pending C1 policy.
        controller.activate_step_by_batch(4)
        controller.on_verify_complete([7], batch_size=1)
        self.assertEqual(controller.params.get_steps_for_batch(1), 7)
        self.assertEqual(worker.events, before, "C4 was forced wide by a stale C1")
        self.assertEqual(worker.speculative_num_steps, 3)

        # The next ACTUAL C1 forward is where the wide state is applied.
        controller.activate_step_by_batch(1)
        self.assertEqual(worker.events, before + ["apply:7"])
        self.assertEqual(worker.speculative_num_steps, 7)

    def test_a_stale_c2_completion_cannot_narrow_the_in_flight_c1_batch(self):
        controller, worker = self._controller(
            {
                "1": {"candidate_steps": CANDIDATE_STEPS, **FAST},
                "2": {"candidate_steps": CANDIDATE_STEPS, **FAST},
            }
        )
        # A C1 batch is replaying its W8 graph.
        controller.activate_step_by_batch(1)
        self.assertEqual(worker.speculative_num_steps, 7)

        # A delayed completion of an earlier C2 batch downshifts the C>=2 slot.
        before = list(worker.events)
        controller.on_verify_complete([0, 0], batch_size=2)
        self.assertEqual(controller.params.get_steps_for_batch(2), 3)
        self.assertEqual(worker.events, before, "a completion callback repointed")
        self.assertEqual(worker.speculative_num_steps, 7)

        # The wide C1 batch runs on; the narrow width lands at the next C2 forward.
        controller.activate_step_by_batch(1)
        self.assertEqual(worker.events, before)
        controller.activate_step_by_batch(2)
        self.assertEqual(worker.events, before + ["apply:3"])
        self.assertEqual(worker.speculative_num_steps, 3)

    def test_slot_state_is_independent_per_tier(self):
        # The identical reversal reaches a C>=2 slot with no hold keys on its own
        # next batch, while the C1 slot stays inside its hold.
        controller, _ = self._controller(
            {
                "1": {"candidate_steps": CANDIDATE_STEPS, **FAST, **C1_HOLD},
                "2": {"candidate_steps": CANDIDATE_STEPS, **FAST},
            }
        )
        controller.on_verify_complete([0], batch_size=1)
        self.assertEqual(controller.params.get_steps_for_batch(1), 3)

        for _ in range(3):
            controller.on_verify_complete([0, 0], batch_size=2)
            self.assertEqual(controller.params.get_steps_for_batch(2), 3)
            self.assertEqual(controller.params._slots[2]._grace_until, 0)
            controller.on_verify_complete([3, 3], batch_size=2)
            self.assertEqual(controller.params.get_steps_for_batch(2), 7)

        self.assertEqual(controller.params.get_steps_for_batch(1), 3)
        self.assertEqual(controller.params._slots[1]._grace_until, 41)

    def test_the_same_observations_give_the_same_transitions_on_every_rank(self):
        # TP2: each rank runs its own controller off the shared acceptance
        # observations, so two ranks must walk identical widths with no extra
        # collective and no per-rank timing input.
        cfg = {
            "1": {"candidate_steps": CANDIDATE_STEPS, **FAST},
            "2": {"candidate_steps": CANDIDATE_STEPS, **FAST},
        }
        script = [(1, [3]), (1, [7]), (4, [0, 0, 0, 0]), (1, [0]), (4, [3, 3, 3, 3])]
        traces = []
        for _ in range(2):
            controller, worker = self._controller(cfg)
            trace = [worker.speculative_num_steps]
            for batch_size, counts in script:
                controller.activate_step_by_batch(batch_size)
                controller.on_verify_complete(counts, batch_size=batch_size)
                trace.append((batch_size, worker.speculative_num_steps))
                controller.activate_step_by_batch(batch_size)
                trace.append((batch_size, worker.speculative_num_steps))
            trace.append(worker.events)
            traces.append(trace)
        self.assertEqual(traces[0], traces[1])


class TestCapturedBucketGeometry(_ConfigCase):
    def test_the_shipped_table_prunes_the_states_to_their_reachable_buckets(self):
        _, worker = self._controller(V302_CONFIG)
        # C1 can settle wide or narrow, so both states keep BS1 and the narrow
        # state also captures every concurrent bucket.
        self.assertEqual(worker.capture_requests[7], [1])
        self.assertEqual(worker.capture_requests[3], [1, 2, 4, 8])

    def test_a_configured_bs1_bucket_keeps_c1_on_the_wide_candidate(self):
        params = adaptive_spec_params.AdaptiveSpeculativeParams(7, V302_CONFIG)
        params.set_cuda_graph_bs([1, 2, 4, 8])
        self.assertIn(7, params._route(1).candidate_steps)
        self.assertEqual(params.get_steps_for_batch(1), 7)
        self.assertEqual(params.get_steps_for_batch(2), 3)

    def test_without_a_bs1_bucket_c1_pads_into_the_narrow_slot(self):
        # No BS1 graph: the wide state has no bucket to capture, and a C1 batch
        # replays the BS2 graph, so it must route as C>=2 (W4) instead of
        # selecting wide shapes nothing captured.
        params = adaptive_spec_params.AdaptiveSpeculativeParams(7, V302_CONFIG)
        params.set_cuda_graph_bs([2, 4, 8])
        self.assertEqual(params.cuda_graph_bs_for_step(7), [])
        self.assertEqual(params.cuda_graph_bs_for_step(3), [2, 4, 8])
        self.assertEqual(params.get_steps_for_batch(1), 3)


class TestFixedWidthAndNonAdaptiveUnaffected(_ConfigCase):
    def setUp(self):
        super().setUp()
        # apply_runtime_state syncs the resolved spec leaves, which needs a
        # published config; the dummy ServerArgs is not read by the policy.
        override = get_context().override_server_args(speculative_algorithm="EAGLE")
        override.install()
        self.addCleanup(override.restore)

    def _hook_worker(self, cfg_path: str):
        """The real worker hooks over the real controller and an explicit table."""
        worker = object.__new__(EAGLEWorkerV2)
        worker.speculative_num_steps = 7
        worker.speculative_num_draft_tokens = 8
        worker._draft_worker = SimpleNamespace(
            speculative_num_steps=7,
            speculative_num_draft_tokens=8,
            draft_attn_backend=object(),
            cuda_graph_runner=object(),
            draft_extend_attn_backend=object(),
            cuda_graph_runner_for_draft_extend=object(),
            draft_runner=SimpleNamespace(
                draft_attn_backend=object(), attn_backend=object()
            ),
            _rebuild_topk1_chain_buffers=lambda: None,
        )
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=object(), decode_cuda_graph_runner=object()
            )
        )
        worker.adaptive_controller = AdaptiveController(
            worker, adaptive_spec_params.AdaptiveSpeculativeParams(7, cfg_path)
        )
        # The states are stand-ins (no CUDA on this host); the real
        # apply_runtime_state is what these checks are about.
        worker.build_adaptive_runtime_state = (
            lambda speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs=None: (
                _state(  # noqa: E501
                    speculative_num_steps
                )
            )
        )
        worker.adaptive_controller.init_states(cuda_graph_bs=[1, 2, 4, 8])
        return worker

    def test_the_worker_hooks_route_feedback_and_width_by_batch_size(self):
        # A same-shaped table with the EMA knobs pinned, so one rejected batch
        # is one decision: what matters here is WHO applies that decision.
        worker = self._hook_worker(
            self._temp_config(
                {
                    "1": {"candidate_steps": CANDIDATE_STEPS, **FAST},
                    "2": {"candidate_steps": [3]},
                }
            )
        )

        # The result processor feeds the batch's own accept counts and size.
        EAGLEWorkerV2.on_verify_complete_cpu(worker, [0], batch_size=1)
        self.assertEqual(worker.adaptive_controller.params.get_steps_for_batch(1), 3)
        self.assertEqual(worker.speculative_num_steps, 7, "a callback repointed it")

        # The next forward's own batch size is what selects and applies a width.
        EAGLEWorkerV2.activate_step_by_batch(worker, 4)
        self.assertEqual(worker.speculative_num_steps, 3)
        # The resolved leaves move with the state, so every later reader of the
        # flat spec config sees the width that is actually live.
        self.assertEqual(get_spec().speculative_num_steps, 3)
        self.assertEqual(get_spec().speculative_num_draft_tokens, 4)
        EAGLEWorkerV2.activate_step_by_batch(worker, 4)
        self.assertEqual(worker.speculative_num_steps, 3, "idempotent per batch")

    def test_a_worker_without_a_controller_ignores_both_hooks(self):
        # Fixed W4 / non-adaptive: the result processor still calls the hook,
        # and nothing about the worker's width may move.
        class _Fixed:
            speculative_num_steps = 3
            speculative_num_draft_tokens = 4
            adaptive_controller = None

        fixed = _Fixed()
        EAGLEWorkerV2.on_verify_complete_cpu(fixed, [3, 3], batch_size=2)
        EAGLEWorkerV2.activate_step_by_batch(fixed, 2)
        BaseSpecWorker.on_verify_complete_cpu(fixed, [3], batch_size=1)
        BaseSpecWorker.activate_step_by_batch(fixed, 4)
        self.assertEqual(
            (fixed.speculative_num_steps, fixed.speculative_num_draft_tokens), (3, 4)
        )


if __name__ == "__main__":
    unittest.main()
