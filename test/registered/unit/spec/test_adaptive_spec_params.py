import json
import math
import tempfile
import unittest

from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    AdaptiveStepSlot,
    resolve_candidate_steps_from_config,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")
register_xpu_ci(est_time=10, suite="stage-a-test-1-gpu-xpu")


class TestAdaptiveStepSlot(CustomTestCase):
    def _make_params_from_config(self, initial_steps: int, config: dict):
        return AdaptiveStepSlot(initial_steps=initial_steps, cfg=config)

    def test_initial_steps_snaps_to_middle_when_missing(self):
        params = self._make_params_from_config(2, {"candidate_steps": [1, 3, 7]})

        self.assertEqual(params.candidate_steps, [1, 3, 7])
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.0)

    def test_update_respects_warmup_and_interval(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 1,
                "update_interval": 2,
            },
        )

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_empty_batches_do_not_consume_warmup_or_shift_steps(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 1,
                "update_interval": 1,
            },
        )

        self.assertFalse(params.update([]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.0)

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_update_scales_up_across_candidates(self):
        params = self._make_params_from_config(
            1,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
            },
        )

        self.assertTrue(params.update([1, 1]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([3, 3]))
        self.assertEqual(params.current_steps, 7)

    def test_update_can_scale_down_across_candidates_in_one_recompute(self):
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
            },
        )

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_exact_rise_threshold_does_not_upshift(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
            },
        )

        self.assertFalse(params.update([2, 3]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.5)

        self.assertTrue(params.update([3, 3]))
        self.assertEqual(params.current_steps, 7)

    def test_exact_drop_threshold_does_downshift(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": 0.0,
                "up_hysteresis": 0.5,
            },
        )

        self.assertTrue(params.update([0, 1]))
        self.assertEqual(params.current_steps, 1)
        self.assertEqual(params.ema_accept_len, 0.5)

    def test_hysteresis_can_prevent_premature_upshift(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.75,
            },
        )

        self.assertFalse(params.update([3, 3]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([4, 4]))
        self.assertEqual(params.current_steps, 7)

    def test_down_hysteresis_can_prevent_premature_downshift(self):
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": -0.75,
            },
        )

        self.assertFalse(params.update([2, 2]))
        self.assertEqual(params.current_steps, 7)

        self.assertTrue(params.update([1, 1]))
        self.assertEqual(params.current_steps, 3)

    def test_multi_batch_sequence_can_ramp_up_then_back_down(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 0.5,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
                "down_hysteresis": 0.0,
            },
        )

        self.assertTrue(params.update([4, 4]))
        self.assertEqual(params.current_steps, 7)
        self.assertEqual(params.ema_accept_len, 3.0)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 1.5)

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 0.75)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)
        self.assertEqual(params.ema_accept_len, 0.375)

    def test_zero_step_mixed_slot_drops_probes_and_rechecks(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [0, 3],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": 0.0,
            },
        )

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 0)
        self.assertEqual(params.ema_accept_len, 0.0)

        self.assertTrue(params.update([3, 3]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 0.0)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 0)
        self.assertEqual(params.ema_accept_len, 0.0)

    def test_ceiling_coeff_caps_steps(self):
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "ceiling_coeff": 1.0,
            },
        )
        # Force low ema to trigger ceiling
        params.ema_accept_len = 1.0
        self.assertTrue(params.update([1, 1]))
        # ceiling = ceil(1.0 * 1.0) = 1, target capped to 1
        self.assertEqual(params.current_steps, 1)

    def test_ceiling_disabled_by_default(self):
        params = self._make_params_from_config(3, {"candidate_steps": [1, 3, 7]})
        self.assertEqual(params.ceiling_coeff, 0)


class TestSwitchGraceHold(CustomTestCase):
    """Post-switch hold and reversal backoff, ported from the donor controller.

    Donor source: https://github.com/aiueo52/sglang-rtxpro6000 snapshot
    5105985, ``adaptive_confidence.py`` (grace setup 373-385, completed-feedback
    gate 512-518, switch bookkeeping 624-632). The base policy re-decides on
    every update, so an acceptance stream that alternates between the two C1
    candidates flips the width on every single batch. The hold gates only the
    re-decision: the EMA keeps learning, and the first decision after the hold
    still follows sustained acceptance. Empty feedback is not a completed batch
    and advances neither the warmup, the interval, nor the hold.
    """

    # The two shipped C1 widths, with the EMA knobs pinned so each case is a
    # hand-checkable line of arithmetic instead of a decay curve.
    _FAST = {
        "candidate_steps": [3, 7],
        "ema_alpha": 1.0,
        "warmup_batches": 0,
        "update_interval": 1,
    }

    def _slot(self, initial_steps: int, **hold):
        return AdaptiveStepSlot(initial_steps=initial_steps, cfg={**self._FAST, **hold})

    @staticmethod
    def _alternating(batches: int) -> list[list[int]]:
        """Rejected batch on odd batches, W4-sized acceptance on even ones."""
        return [[0] if i % 2 else [3] for i in range(1, batches + 1)]

    def _run(self, slot: AdaptiveStepSlot, feedback: list[list[int]]):
        """Switches as (completed-batch index, new steps)."""
        switches = []
        for counts in feedback:
            if slot.update(counts):
                switches.append((slot._batch_count, slot.current_steps))
        return switches

    def test_omitting_the_keys_keeps_the_base_flip_every_batch_policy(self):
        slot = self._slot(7)
        self.assertEqual(slot.switch_grace_batches, 0)
        self.assertEqual(slot.grace_backoff, 2.0)
        self.assertEqual(slot.max_grace_batches, 80)

        switches = self._run(slot, self._alternating(10))
        # The reported pathology in miniature: no hold at all, a reversal every
        # batch, 10 switches in 10 completed feedback batches.
        self.assertEqual([steps for _, steps in switches], [3, 7] * 5)

    def test_the_hold_gates_the_reversal_and_grows_capped_on_reversals(self):
        slot = self._slot(
            7, switch_grace_batches=4, grace_backoff=2.0, max_grace_batches=8
        )

        switches = self._run(slot, self._alternating(20))

        # One switch, then a 4-batch hold, then 8, then the 8-batch cap. 3
        # switches in 20 batches against 20 without the hold, and the width is
        # still moving at batch 15, so the hold slowed the controller instead of
        # freezing it.
        self.assertEqual(switches, [(1, 3), (6, 7), (15, 3)])
        self.assertEqual(slot._grace, 8.0, "the reversal hold is capped")
        self.assertEqual(slot.ema_accept_len, 3.0, "the EMA kept updating")

    def test_sustained_acceptance_is_honoured_the_batch_the_hold_expires(self):
        slot = self._slot(7, switch_grace_batches=4)

        self.assertTrue(slot.update([0]))  # 7 -> 3, hold through batch 5
        self.assertEqual(slot.current_steps, 3)

        for _ in range(3):
            self.assertFalse(slot.update([7, 7]))
        self.assertEqual(slot.ema_accept_len, 7.0, "the hold did not stop learning")
        self.assertEqual(slot.current_steps, 3, "no re-decision inside the hold")

        self.assertTrue(slot.update([7, 7]))  # batch 5: the hold expired
        self.assertEqual(slot.current_steps, 7)

    def test_empty_feedback_advances_neither_the_hold_nor_the_counters(self):
        slot = self._slot(7, switch_grace_batches=4)

        self.assertTrue(slot.update([0]))
        for _ in range(10):
            self.assertFalse(slot.update([]))
        self.assertEqual(slot._batch_count, 1)
        self.assertEqual(slot._grace_until, 5)

        self.assertFalse(slot.update([3]))  # batch 2: still inside the hold
        self.assertEqual(slot.current_steps, 3)

    def test_same_direction_switches_keep_the_initial_hold(self):
        # Three candidates, so the controller really can step 1 -> 3 -> 7: two
        # SAME-direction switches in a row, then reversals. Both up-switches land
        # on consecutive batches, which only the reset-to-initial-hold branch
        # allows, and only the direction changes grow the hold -- up to its cap.
        slot = self._slot(
            1,
            candidate_steps=[1, 3, 7],
            switch_grace_batches=1,
            grace_backoff=2.0,
            max_grace_batches=3,
        )
        stream = [[2], [6], [1], [3], [3], [0], [0], [0]]

        self.assertEqual(
            self._run(slot, stream), [(1, 3), (2, 7), (3, 3), (5, 7), (8, 1)]
        )
        self.assertEqual(slot._grace, 3.0, "the reversal hold stops at the cap")

    def test_invalid_hold_settings_are_rejected(self):
        invalid = (
            {"switch_grace_batches": -1},
            {"switch_grace_batches": 1.5},
            {"switch_grace_batches": "40"},
            {"switch_grace_batches": True},
            {"max_grace_batches": -1},
            {"max_grace_batches": 0.5},
            {"grace_backoff": 0.5},
            {"grace_backoff": math.inf},
            {"grace_backoff": "2x"},
            {"switch_grace_batches": 40, "max_grace_batches": 10},
        )
        for cfg in invalid:
            with self.subTest(**cfg):
                with self.assertRaises(ValueError):
                    self._slot(7, **cfg)

    def test_a_valid_hold_below_the_cap_is_accepted(self):
        slot = self._slot(
            7, switch_grace_batches=40, grace_backoff=2.0, max_grace_batches=80
        )
        self.assertEqual(slot.switch_grace_batches, 40)
        self.assertEqual(slot.max_grace_batches, 80)


class TestAdaptiveSpeculativeParams(CustomTestCase):
    def test_default_config_loads(self):
        params = AdaptiveSpeculativeParams(initial_steps=3)
        self.assertEqual(params._bs_list, [1, 8, 32, 64])
        self.assertEqual(params._slots[1].candidate_steps, [1, 3, 5, 7])
        self.assertEqual(params._slots[8].candidate_steps, [0, 1, 3])
        self.assertEqual(params._slots[32].candidate_steps, [0, 1])
        self.assertEqual(params._slots[64].candidate_steps, [0])

    def test_config_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 5], "up_hysteresis": 0.3},
                    "32": {"candidate_steps": [1, 2]},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=5, cfg_path=f.name)
        self.assertEqual(params._bs_list, [1, 32])
        # Slots are built straight from the config; the launch flag never pollutes
        # them. initial_steps just selects the smallest slot's starting step.
        self.assertEqual(params._slots[1].candidate_steps, [1, 5])
        self.assertEqual(params._slots[1].current_steps, 5)
        self.assertEqual(params._slots[1].up_hysteresis, 0.3)
        self.assertEqual(params._slots[32].candidate_steps, [1, 2])

    def test_launch_flag_not_injected_into_slots(self):
        # initial_steps lives only in a larger slot. It must NOT be merged into
        # any other slot's candidates: slots come straight from the config.
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 5]},
                    "8": {"candidate_steps": [1, 3, 7]},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=7, cfg_path=f.name)
        self.assertEqual(params._slots[1].candidate_steps, [1, 5])
        self.assertEqual(params._slots[8].candidate_steps, [1, 3, 7])
        # The slot that does not own initial_steps starts at its own median.
        self.assertEqual(params._slots[1].current_steps, 5)
        self.assertEqual(params._slots[8].current_steps, 7)

    def test_invalid_config_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"not_a_bs": "bad"}, f)
            f.flush()
            with self.assertRaises(ValueError):
                AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)

    def test_invalid_steps_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": "bad"}}, f)
            f.flush()
            with self.assertRaises(ValueError):
                AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)

    def test_empty_steps_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": []}}, f)
            f.flush()
            with self.assertRaises(ValueError):
                AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)

    def test_unusable_hold_settings_raise_through_the_loader(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {
                        "candidate_steps": [3, 7],
                        "switch_grace_batches": 40,
                        "max_grace_batches": 10,
                    }
                },
                f,
            )
            f.flush()
            with self.assertRaises(ValueError):
                AdaptiveSpeculativeParams(initial_steps=7, cfg_path=f.name)

    def test_global_hysteresis_inherited(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "up_hysteresis": 0.5,
                    "1": {"candidate_steps": [1, 3]},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)
        self.assertEqual(params._slots[1].up_hysteresis, 0.5)

    def test_entry_hysteresis_overrides_global(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "up_hysteresis": 0.5,
                    "1": {"candidate_steps": [1, 3], "up_hysteresis": 0.1},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)
        self.assertEqual(params._slots[1].up_hysteresis, 0.1)


class TestBatchSizeRouting(CustomTestCase):
    """BS-aware routing: batch size selects the slot, CUDA-graph BS pads first."""

    def _params(self):
        # Slots: bs=1 -> [1,3,5,7], bs=8 -> [0,1,3], bs=32 -> [0,1].
        return AdaptiveSpeculativeParams(initial_steps=3)

    def test_routes_to_floor_slot_without_cuda_graph(self):
        params = self._params()
        # A batch maps to the largest slot BS <= batch (floor), capped at the top slot.
        self.assertEqual(params._route(1).candidate_steps, [1, 3, 5, 7])
        self.assertEqual(params._route(7).candidate_steps, [1, 3, 5, 7])
        self.assertEqual(params._route(8).candidate_steps, [0, 1, 3])
        self.assertEqual(params._route(31).candidate_steps, [0, 1, 3])
        self.assertEqual(params._route(32).candidate_steps, [0, 1])
        self.assertEqual(params._route(1000).candidate_steps, [0])

    def test_cuda_graph_bs_pads_batch_up_before_routing(self):
        params = self._params()
        params.set_cuda_graph_bs([4, 8, 16, 32])
        # bs=5 pads up to the captured graph BS 8 -> slot bs=8.
        self.assertEqual(params._route(5).candidate_steps, [0, 1, 3])
        # bs=17 pads up to 32 -> slot bs=32.
        self.assertEqual(params._route(17).candidate_steps, [0, 1])
        # A batch larger than every captured BS keeps its own value -> top slot.
        self.assertEqual(params._route(100).candidate_steps, [0])

    def test_cuda_graph_bs_for_step_prunes_unreachable_graphs(self):
        params = self._params()
        params.set_cuda_graph_bs([4, 8, 16, 32])
        # step=1 is reachable from every slot.
        self.assertEqual(params.cuda_graph_bs_for_step(1), [4, 8, 16, 32])
        # step=3 lives in the bs=1 and bs=8 slots: graphs 4,8,16 floor into them.
        self.assertEqual(params.cuda_graph_bs_for_step(3), [4, 8, 16])
        # step=5 lives only in the bs=1 slot: only graph BS 4 floors into it.
        self.assertEqual(params.cuda_graph_bs_for_step(5), [4])
        # step=7 lives only in the bs=1 slot: only graph BS 4 floors into it.
        self.assertEqual(params.cuda_graph_bs_for_step(7), [4])

    def test_cuda_graph_bs_for_step_returns_none_when_disabled(self):
        params = self._params()
        self.assertIsNone(params.cuda_graph_bs_for_step(7))
        params.set_cuda_graph_bs(None)
        self.assertIsNone(params.cuda_graph_bs_for_step(7))

    def test_observe_verify_feeds_the_routed_slot(self):
        params = self._params()
        # Drive the bs=1 slot up with perfect acceptance; the bs=32 slot is
        # untouched and stays at its single candidate step.
        for _ in range(40):
            params.on_verify_complete([7, 7, 7], batch_size=1)
        self.assertGreater(params.get_steps_for_batch(1), 1)
        self.assertEqual(params.get_steps_for_batch(32), 1)


class TestResolveCandidateSteps(CustomTestCase):
    def test_default_config(self):
        steps = resolve_candidate_steps_from_config()
        self.assertEqual(steps, [0, 1, 3, 5, 7])

    def test_config_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": [2, 4]}}, f)
            f.flush()
            steps = resolve_candidate_steps_from_config(cfg_path=f.name)
        self.assertEqual(steps, [2, 4])

    def test_unions_and_dedups_across_slots(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 5]},
                    "8": {"candidate_steps": [3, 5, 7]},
                },
                f,
            )
            f.flush()
            steps = resolve_candidate_steps_from_config(cfg_path=f.name)
        self.assertEqual(steps, [1, 3, 5, 7])


if __name__ == "__main__":
    unittest.main()
