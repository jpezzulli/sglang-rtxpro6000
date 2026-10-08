"""CPU guards for opt-in confidence-policy transition configuration.

Ported with the controller itself from https://github.com/aiueo52/sglang-rtxpro6000,
branch flash-next-fast, snapshot 5105985116eb00dea8e6138aabeb5363387cb9de (Apache
License 2.0; see the header of the ported module). The legacy-score twin in here is
what pins that our port keeps the donor selection arithmetic: absent the new config
keys, the decision trajectory must match it exactly.
"""

import random
import unittest

from sglang.srt.speculative.adaptive_confidence import ConfidenceStepSlot
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class LegacyConfidenceStepSlot(ConfidenceStepSlot):
    """Frozen pre-WA3 selection arithmetic for exact default parity checks."""

    def best_steps(self):
        best, best_tps = self.current_steps, -1.0
        for s in self.candidate_steps:
            tps = self.value(s)
            if s == self.current_steps:
                tps *= 1.0 + self.switch_margin
            elif s > self.current_steps:
                tps /= 1.0 + self.up_margin
            if tps > best_tps:
                best, best_tps = s, tps
        return best, best_tps


class TestConfidenceTransitionConfig(unittest.TestCase):
    def slot(self, current=7, values=None, **extra):
        cfg = dict(
            candidate_steps=[3, 7, 15],
            switch_margin=0.12,
            up_margin=0.30,
            switch_grace_batches=40,
            grace_backoff=2.0,
            warmup_batches=15,
            update_interval=10,
        )
        cfg.update(extra)
        slot = ConfidenceStepSlot(current, cfg)
        if values is not None:
            slot.value = values.__getitem__
        return slot

    def test_absent_keys_preserve_exact_legacy_scores(self):
        rng = random.Random(31415)
        for candidates in ([3, 15], [3, 7, 15], [15]):
            for current in candidates:
                cfg = dict(
                    candidate_steps=candidates, switch_margin=0.12, up_margin=0.30
                )
                new = ConfidenceStepSlot(current, cfg)
                old = LegacyConfidenceStepSlot(current, cfg)
                for _ in range(200):
                    values = {s: rng.random() for s in candidates}
                    new.value = old.value = values.__getitem__
                    self.assertEqual(new.best_steps(), old.best_steps())

    def test_default_trajectory_matches_legacy(self):
        rng = random.Random(2718)
        cfg = dict(
            candidate_steps=[3, 15],
            switch_margin=0.12,
            up_margin=0.30,
            switch_grace_batches=40,
            grace_backoff=2.0,
            tail_bias=0.75,
            warmup_batches=15,
            update_interval=10,
            weight_alpha=0.05,
            rate_alpha=0.1,
            min_bucket_samples=20,
        )
        new = ConfidenceStepSlot(15, cfg)
        old = LegacyConfidenceStepSlot(15, cfg)
        for i in range(2000):
            confidence = [0.9999 if i % 400 < 200 else 0.5]
            accepted = [new.current_steps if i % 400 < 200 else rng.randrange(3)]
            new.observe_confidence(confidence)
            old.observe_confidence(confidence)
            self.assertEqual(new.update(accepted), old.update(accepted))
            self.assertEqual(vars(new), vars(old))
            self.assertEqual(new.best_steps(), old.best_steps())

    def test_down_margin_does_not_change_upward_threshold(self):
        slot = self.slot(values={3: 1.029, 7: 1.0, 15: 1.455}, down_margin=0.03)
        self.assertEqual(slot.best_steps()[0], 7)
        slot.value = {3: 1.031, 7: 1.0, 15: 1.0}.__getitem__
        self.assertEqual(slot.best_steps()[0], 3)
        slot.value = {3: 0.5, 7: 1.0, 15: 1.457}.__getitem__
        self.assertEqual(slot.best_steps()[0], 15)

    def test_destination_down_margins_separate_code_and_prose(self):
        margins = {"3": 0.03, "7": 0.12}
        slot = self.slot(15, {3: 0.5, 7: 1.11, 15: 1.0}, down_margin=margins)
        self.assertEqual(slot.best_steps()[0], 15)
        slot.value = {3: 0.5, 7: 1.13, 15: 1.0}.__getitem__
        self.assertEqual(slot.best_steps()[0], 7)
        slot.current_steps = 7
        slot.value = {3: 1.04, 7: 1.0, 15: 0.5}.__getitem__
        self.assertEqual(slot.best_steps()[0], 3)

    def test_unmapped_down_destination_uses_legacy_margin(self):
        slot = self.slot(15, {3: 0.5, 7: 1.11, 15: 1.0}, down_margin={"3": 0.03})
        self.assertEqual(slot.best_steps()[0], 15)

    def test_adjacent_promotion_prevents_skipping_middle(self):
        values = {3: 1.0, 7: 1.2, 15: 2.0}
        slot = self.slot(3, values, adjacent_only_promotion=True, up_margin=0)
        self.assertEqual(slot.best_steps()[0], 7)
        slot.current_steps = 7
        self.assertEqual(slot.best_steps()[0], 15)
        slot.current_steps = 15
        slot.value = {3: 3.0, 7: 1.0, 15: 1.0}.__getitem__
        self.assertEqual(slot.best_steps()[0], 3)  # downshifts may still skip

    def test_adjacent_promotion_must_clear_margin(self):
        slot = self.slot(
            3, {3: 1.0, 7: 1.1, 15: 3.0}, adjacent_only_promotion=True, up_margin=0
        )
        self.assertEqual(slot.best_steps()[0], 3)

    def test_adjacent_false_is_inert_and_two_candidates_still_promote(self):
        values = {3: 1.0, 7: 1.1, 15: 3.0}
        self.assertEqual(
            self.slot(3, values, adjacent_only_promotion=False).best_steps()[0], 15
        )
        self.assertEqual(
            self.slot(
                3,
                {3: 1.0, 15: 3.0},
                candidate_steps=[3, 15],
                adjacent_only_promotion=True,
            ).best_steps()[0],
            15,
        )

    def test_reversal_grace_cap_and_decision_hold(self):
        slot = self.slot(15, max_grace_batches=80)
        for target, expected in [(7, 40), (15, 80), (7, 80), (15, 80)]:
            slot._pmn[slot.current_steps] = 1
            slot.value = lambda s, target=target: 10.0 if s == target else 1.0
            self.assertTrue(slot._recompute())
            self.assertEqual(slot._grace, expected)
            self.assertEqual(slot._grace_until, slot._batch_count + expected)
        slot.value = {3: 1.0, 7: 10.0, 15: 1.0}.__getitem__
        for _ in range(84):
            self.assertFalse(slot.update([0]))
        self.assertTrue(slot.update([0]))  # first interval after the grace ends
        self.assertEqual(slot.current_steps, 7)
        slot.value = {3: 10.0, 7: 1.0, 15: 1.0}.__getitem__
        self.assertTrue(slot._recompute())
        self.assertEqual(slot._grace, 40)  # continuing downward resets backoff
        self.assertEqual(slot.max_grace_batches, 80)
        self.assertEqual(self.slot().max_grace_batches, 2000)

    def test_invalid_new_values_rejected(self):
        for bad in (-1, float("nan"), {"3": float("inf")}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.slot(down_margin=bad)
        with self.assertRaises(ValueError):
            self.slot(adjacent_only_promotion="false")


if __name__ == "__main__":
    unittest.main()
