"""CPU-only regression: a confidence/accept observation belongs to its own forward.

The C1 policy takes two kinds of observation, both delivered asynchronously: the
position-0 draft confidence (the only measurement that exists *before* a width is
chosen) and the accepted draft counts of a finished verify. The consumer reads the
pinned ring without waiting on a stream, and ``event_loop_overlap`` launches
forward N+1 (scheduler.py:1846) before it completes N (:1860) -- so by the time a
batch's accept counts reach ``batch_result_processor``, the NEXT forward has
already been launched and has already made its own width decision. Neither
observation may therefore be interpreted against whatever the worker or the policy
happens to hold at delivery time. Base did exactly that, three times, from one
root cause:

* ``ConfidenceStepSlot.update`` filtered and credited with ``current_steps``, the
  *live* policy width. A batch produced at S=7 that landed after the policy moved
  to S=15 reported "7 accepted out of 15, boundary never reached" instead of a
  fully accepted 7-chain -- so the hazard the upshift extrapolates from was
  poisoned by a chain that width never ran, and a downshift threw the accepted
  sample away entirely;
* the bucket a result is credited to was picked from a queue by position, and the
  entry was only consumed when the counts survived the width filter. Under
  overlap that is doubly wrong: the queued entries belong to forwards that are
  still in flight (N+1's confidence was recorded before N's counts arrived, and
  rid alone cannot tell those two forwards apart), and a rejected result left its
  entry behind to shift every later association by one permanently;
* ``ConfidenceChannel.latest_position0`` answered "newest landed, else last
  cached" with no identity at all, and its invalidation only reset the read
  pointer -- so an old C1 sample survived a new request (prefill stages nothing)
  or a C1 -> C>=2 -> C1 round trip, and could even be *revived* after the
  invalidation by the copy staged behind it.

Each observation is now paired with the launch that produced it, from metadata
that already exists: the result's immutable ``speculative_num_draft_tokens``
stride (a width only on the topk=1 chain path, where stride == steps + 1) and the
scheduler's ``forward_iter``, which ``ScheduleBatch.copy`` carries to the queued
result so the decision and its result name the same number. Delivery discipline is
unchanged: no forced synchronisation, no C>=2 sampling work, and a completion
callback still never activates a runtime state.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.speculative import base_spec_worker as base_spec_module
from sglang.srt.speculative.adaptive_confidence import (
    ConfidenceChannel,
    ConfidenceStepSlot,
)
from sglang.srt.speculative.adaptive_runtime_state import AdaptiveController
from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    AdaptiveStepSlot,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

CANDIDATES = [3, 7, 15]
CONFIDENCE_ENV = {"SGLANG_ADAPTIVE_POLICY": "confidence"}


def _sync_is_a_bug():
    raise AssertionError("the confidence hot path synchronised a stream")


class _FakeEvent:
    """torch.cuda.Event stand-in: landing is scripted, waiting is forbidden."""

    def __init__(self, done):
        self.done = done

    def query(self):
        return self.done

    def record(self):
        pass

    def synchronize(self):
        _sync_is_a_bug()


def _slot(initial=7, **extra):
    cfg = dict(
        candidate_steps=CANDIDATES,
        warmup_batches=10000,  # keep the decision machinery out of the way
        update_interval=1,
        switch_grace_batches=0,
    )
    cfg.update(extra)
    return ConfidenceStepSlot(initial, cfg)


def _observe(slot, confidences, forward_id=None):
    """Report one C1 width decision; empty confidences = an explicit no-sample."""
    try:
        slot.observe_confidence(confidences, forward_id)
    except TypeError as exc:
        raise AssertionError(f"observe_confidence cannot carry a forward id: {exc}")


def _update(slot, counts, steps=None, forward_id=None):
    """Credit one result, resolving the producing-batch contract at call time so
    a tree without the fix fails this check with a reason."""
    try:
        return slot.update(counts, steps=steps, forward_id=forward_id)
    except TypeError as exc:
        raise AssertionError(f"update() cannot attribute a producing batch: {exc}")


class TestProducingWidthAttribution(CustomTestCase):
    def test_an_upshift_in_flight_credits_the_chain_that_actually_ran(self):
        slot = _slot(7)
        _observe(slot, [0.99], 101)
        # The policy moved up while this S=7 batch was on the GPU.
        slot.current_steps = 15
        _update(slot, [7], steps=7, forward_id=101)

        b = slot._bucket(0.99)
        # A fully accepted 7-chain is uncensored at its own boundary. Credited to
        # the live S=15 instead, the same sample said "7 of 15 accepted, boundary
        # never reached" and told the next decision to come down.
        self.assertEqual(slot._m[b][7], 7.0)
        self.assertEqual(slot._sat[b][7], 1.0)
        self.assertEqual(slot._satn[b][7], 1)
        self.assertEqual(slot._psat[7], 1.0)
        # S=15 never drafted this chain, so it owns no observation of it.
        self.assertEqual(slot._satn[b][15], 0)
        self.assertEqual(slot._m[b][15], 0.0)

    def test_a_downshift_in_flight_keeps_the_wide_samples_wide(self):
        slot = _slot(15)
        _observe(slot, [1.0], 201)
        slot.current_steps = 3
        # A perfect 15-chain is the evidence for staying wide; filtering it out
        # against the narrow live width threw it away and skewed S=3 low.
        _update(slot, [15], steps=15, forward_id=201)

        b = slot._bucket(1.0)
        self.assertEqual(slot._sat[b][15], 1.0)
        self.assertEqual(slot._satn[b][15], 1)
        self.assertEqual(slot._m[b][3], 3.0, "min(15,3): downshift stays exact")
        self.assertEqual(slot._satn[b][3], 0, "S=3 never ran this chain")

    def test_counts_above_their_own_width_are_still_rejected(self):
        slot = _slot(3)
        _observe(slot, [0.99], 301)
        _update(slot, [9], steps=3, forward_id=301)
        b = slot._bucket(0.99)
        self.assertEqual(slot._mn[b][3], 0, "a 9-accept cannot come from a 3-chain")

    def test_an_unregistered_width_credits_means_but_invents_no_boundary(self):
        # A stride that is not one of this policy's candidates (a state built
        # elsewhere, a table that changed) still measures the means it can, but
        # must not create a saturation bucket keyed on a width never offered.
        slot = _slot(7)
        _observe(slot, [0.99], 401)
        _update(slot, [5], steps=5, forward_id=401)
        self.assertEqual(slot._pmn[3], 1)
        self.assertEqual(slot._psatn.get(5, 0), 0)

    def test_the_live_width_still_decides_when_no_metadata_arrives(self):
        slot = _slot(7)
        _observe(slot, [0.99])
        slot.update([7])  # the pre-fix call shape stays valid
        self.assertEqual(slot._satn[slot._bucket(0.99)][7], 1)


class TestForwardPairing(CustomTestCase):
    """Two C1 forwards of ONE request are in flight together; rid cannot split
    them, so the association has to be the launch itself."""

    def test_overlapped_forwards_of_one_request_do_not_cross(self):
        # forward 101 drafts chain A, forward 102 is launched (and observes its
        # own confidence) before 101's accept counts come back.
        slot = _slot(7)
        _observe(slot, [0.99], 101)
        _observe(slot, [0.10], 102)
        _update(slot, [7], steps=7, forward_id=101)

        high, low = slot._bucket(0.99), slot._bucket(0.10)
        self.assertEqual(
            slot._mn[high][7], 1, "the earlier result took the later bucket"
        )
        self.assertEqual(slot._mn[low][7], 0)

        _update(slot, [0], steps=7, forward_id=102)
        self.assertEqual(slot._mn[low][7], 1, "the later result lost its own bucket")
        self.assertEqual(slot._m[low][7], 0.0)
        self.assertEqual(slot._mn[high][7], 1, "the association shifted by one")

    def test_a_rejected_result_still_consumes_its_own_decision(self):
        slot = _slot(15)
        _observe(slot, [1.0], 101)
        # Counts no width this policy ran could have produced: unusable, but the
        # decision entry is this forward's and must not linger for the next one.
        _update(slot, [99], steps=3, forward_id=101)
        _observe(slot, [0.1], 102)
        _update(slot, [0], steps=3, forward_id=102)

        high, low = slot._bucket(1.0), slot._bucket(0.1)
        self.assertEqual(slot._mn[low][3], 1, "the rejected batch stole the low sample")
        self.assertEqual(slot._mn[high][3], 0)

    def test_a_decision_with_no_sample_credits_pooled_only(self):
        # An explicit absence, not somebody else's bucket: the copy had not landed
        # when this decision was taken, so nothing may claim a confidence for it.
        slot = _slot(7)
        _observe(slot, [], 101)
        _update(slot, [7], steps=7, forward_id=101)
        self.assertEqual(slot._pmn[7], 1, "the accept counts themselves are real")
        self.assertEqual(
            [n for b in slot._mn for n in b.values()],
            [0] * len(slot._mn) * len(CANDIDATES),
            "an unmeasured forward borrowed a bucket",
        )
        self.assertEqual(slot._psatn[7], 1)
        self.assertEqual(
            [n for b in slot._satn for n in b.values()],
            [0] * (len(slot._satn) * len(CANDIDATES)),
        )
        # ... and it did not consume the previous forward's pending entry either.
        _observe(slot, [0.99], 102)
        _update(slot, [7], steps=7, forward_id=102)
        self.assertEqual(slot._mn[slot._bucket(0.99)][7], 1)

    def test_a_result_with_no_decision_of_its_own_takes_no_bucket(self):
        # E.g. a batch whose decision was made at C>=2 and which arrived at C1's
        # slot: unmatched means unmatched.
        slot = _slot(7)
        _observe(slot, [0.99], 101)  # outstanding, its result has not arrived
        _update(slot, [7], steps=7, forward_id=999)
        self.assertEqual(slot._pmn[7], 1)
        self.assertEqual(
            [n for b in slot._mn for n in b.values()],
            [0] * (len(slot._mn) * len(CANDIDATES)),
        )
        # The outstanding decision is still there for its own forward.
        _update(slot, [7], steps=7, forward_id=101)
        self.assertEqual(slot._mn[slot._bucket(0.99)][7], 1)

    def test_empty_feedback_consumes_and_credits_nothing(self):
        slot = _slot(7)
        _observe(slot, [0.99], 101)
        self.assertFalse(_update(slot, [], steps=7, forward_id=101))
        self.assertEqual(slot._pmn[7], 0)
        self.assertEqual(slot._mn[slot._bucket(0.99)][7], 0)
        # Not consumed: the same forward's real result still finds its bucket.
        _update(slot, [7], steps=7, forward_id=101)
        self.assertEqual(slot._mn[slot._bucket(0.99)][7], 1)

    def test_a_stale_decision_entry_is_not_reused_by_a_later_forward(self):
        slot = _slot(7)
        _observe(slot, [0.99], 101)  # its result never arrives (retract/abort)
        _observe(slot, [0.10], 102)
        _update(slot, [0], steps=7, forward_id=102)
        low = slot._bucket(0.10)
        self.assertEqual(slot._mn[low][7], 1)
        self.assertEqual(slot._mn[slot._bucket(0.99)][7], 0)


class TestRingIdentity(CustomTestCase):
    """A staged sample belongs to one chain, and only to that chain."""

    def _channel(self, done, keys, cached=None, cached_id=None):
        # Built without __init__: this host has no driver to pin memory against.
        channel = ConfidenceChannel.__new__(ConfidenceChannel)
        channel.max_bs = 4
        channel.RING = len(done)
        channel._p0_host = [
            torch.full((4,), float(i), dtype=torch.float32) for i in range(len(done))
        ]
        channel._p0_event = [_FakeEvent(d) for d in done]
        channel._p0_bs = [4] * len(done)
        channel._p0_key = list(keys)
        channel._p0_write = 0
        channel._epoch = 0
        channel._p0_epoch = [0] * len(done)
        channel._p0_read = len(done) - 1
        channel._p0_cached = cached
        channel._p0_cached_id = cached_id
        return channel

    def test_a_sample_staged_for_another_request_is_not_reused(self):
        channel = self._channel([True], ["other-req"])
        self.assertIsNone(channel.latest_position0("this-req"))
        # A consumer with no identity to check keeps the donor's answer.
        self.assertEqual(channel.latest_position0(), [0.0] * 4)

    def test_the_cached_value_is_not_reusable_across_requests(self):
        channel = self._channel(
            [False], ["other-req"], cached=[0.97] * 4, cached_id=(0, "other-req")
        )
        self.assertIsNone(channel.latest_position0("this-req"))
        self.assertEqual(channel.latest_position0(), [0.97] * 4)

    def test_invalidate_drops_staged_and_cached_without_syncing(self):
        channel = self._channel([True], ["req-a"])
        self.assertEqual(channel.latest_position0("req-a"), [0.0] * 4)
        channel.invalidate_position0()
        self.assertIsNone(channel.latest_position0("req-a"))
        self.assertIsNone(channel.latest_position0())

    def test_a_pending_write_after_invalidation_cannot_revive_the_old_sample(
        self,
    ):
        # C1 -> C>=2 -> C1, same request throughout: the C>=2 pass drafts chains
        # it never stages, so the landed C1 sample is expired. Staging the next
        # C1 sample must not let the reader walk back and resurrect it.
        channel = self._channel([True, False], ["req-a", "req-a"])
        channel._p0_write = 1  # the next C1 draft-extend lands in slot 1
        channel.invalidate_position0()  # the C>=2 pass in between
        # Staging the next sample is what used to revive the expired one: the
        # reader walks back over the whole ring, and slot 0's copy had landed.
        with patch("torch.cuda.is_current_stream_capturing", return_value=False):
            channel.record_position0(torch.full((4,), 1.0), "req-a")
        self.assertIsNone(
            channel.latest_position0("req-a"),
            "the expired sample came back once a newer copy was pending",
        )
        channel._p0_event[1] = _FakeEvent(True)
        self.assertEqual(channel.latest_position0("req-a"), [1.0] * 4)

    def test_invalidate_expires_slots_repeatedly_not_just_once(self):
        channel = self._channel([True] * 3, ["req-a"] * 3)
        channel.invalidate_position0()
        for _ in range(3):  # every scan after the boundary stays empty
            self.assertIsNone(channel.latest_position0("req-a"))

    def test_a_concurrent_pass_discards_the_pending_c1_sample(self):
        staged, dropped = [], []

        class _Ring:
            def record_position0(self, p0, request_key=None):
                staged.append((p0.shape[0], request_key))

            def invalidate_position0(self):
                dropped.append(1)

        worker = SimpleNamespace(_conf_channel=_Ring())
        logits = torch.zeros((1, 6))
        EagleDraftWorker._record_position0_confidence(worker, logits, 1, "req-a")
        self.assertEqual(staged, [(1, "req-a")], "the C1 pass staged nothing")
        self.assertEqual(dropped, [])

        # C>=2: no top-1 over the draft vocabulary, and no leftover C1 sample.
        EagleDraftWorker._record_position0_confidence(worker, logits, 4, None)
        self.assertEqual(staged, [(1, "req-a")], "a C>=2 batch paid for top-1")
        self.assertEqual(dropped, [1], "the stale C1 sample survived the C>=2 pass")

    def test_no_channel_means_no_producer_work_at_all(self):
        worker = SimpleNamespace(_conf_channel=None)
        EagleDraftWorker._record_position0_confidence(worker, torch.zeros((1, 4)), 1)
        EagleDraftWorker._record_position0_confidence(worker, torch.zeros((4, 4)), 4)

    def test_the_worker_reports_a_decision_for_its_own_c1_forward_only(self):
        reads, observed, activated = [], [], []
        channel = SimpleNamespace(
            latest_position0=lambda key=None: (reads.append(key) or [0.42])
        )
        worker = SimpleNamespace(
            adaptive_controller=SimpleNamespace(
                observe_confidence=lambda conf, bs, forward_id=None: observed.append(
                    (conf, bs, forward_id)
                ),
                activate_step_by_batch=lambda bs: activated.append(bs),
            ),
            _draft_worker=SimpleNamespace(_conf_channel=channel),
        )
        EAGLEWorkerV2.activate_step_by_batch(worker, 4, None, 55)
        self.assertEqual(reads, [], "a C>=2 decision read the confidence ring")
        self.assertEqual(observed, [], "a C>=2 batch reported a confidence decision")
        self.assertEqual(activated, [4])

        EAGLEWorkerV2.activate_step_by_batch(worker, 1, "req-a", 56)
        self.assertEqual(reads, ["req-a"], "ring read without the request's ownership")
        self.assertEqual(
            observed,
            [([0.42], 1, 56)],
            "the decision must be paired with the forward that made it",
        )
        self.assertEqual(activated, [4, 1], "width selected after the read")

    def test_a_c1_forward_with_nothing_landed_reports_the_absence(self):
        landed = [None]
        channel = SimpleNamespace(latest_position0=lambda key=None: landed.pop(0))
        observed = []
        worker = SimpleNamespace(
            adaptive_controller=SimpleNamespace(
                observe_confidence=lambda conf, bs, forward_id=None: observed.append(
                    (conf, bs, forward_id)
                ),
                activate_step_by_batch=lambda bs: None,
            ),
            _draft_worker=SimpleNamespace(_conf_channel=channel),
        )
        with patch("torch.cuda.synchronize", side_effect=AssertionError("sync")):
            EAGLEWorkerV2.activate_step_by_batch(worker, 1, "req-a", 57)
        self.assertEqual(
            observed, [(None, 1, 57)], "the absence was not recorded for this forward"
        )


class TestProducingMetadataPlumbing(CustomTestCase):
    def _controller(self, seen):
        return SimpleNamespace(
            on_verify_complete=lambda counts, batch_size=None, steps=None, forward_id=None: seen.append(
                (tuple(counts), batch_size, steps, forward_id)
            )
        )

    def test_the_result_stride_becomes_the_producing_width(self):
        seen = []
        worker = SimpleNamespace(adaptive_controller=self._controller(seen), topk=1)
        EAGLEWorkerV2.on_verify_complete_cpu(
            worker, [7], batch_size=1, num_draft_tokens=8, forward_id=101
        )
        self.assertEqual(seen, [((7,), 1, 7, 101)])

    def test_a_tree_draft_is_not_assumed_to_be_a_chain(self):
        seen = []
        worker = SimpleNamespace(adaptive_controller=self._controller(seen), topk=4)
        EAGLEWorkerV2.on_verify_complete_cpu(
            worker, [3], batch_size=2, num_draft_tokens=12, forward_id=101
        )
        # stride - 1 means nothing for a tree, so the counts stay unattributed
        # and the slot reads them against its live width as before.
        self.assertEqual(seen, [((3,), 2, None, 101)])

    def test_the_controller_forwards_the_producing_metadata(self):
        seen = []

        class _Params:
            def on_verify_complete(
                self, counts, batch_size, steps=None, forward_id=None
            ):
                seen.append((tuple(counts), batch_size, steps, forward_id))

            def observe_confidence(self, confidences, batch_size, forward_id=None):
                seen.append(("observe", confidences, batch_size, forward_id))

        controller = AdaptiveController(
            SimpleNamespace(
                speculative_num_steps=15,
                build_adaptive_runtime_state=lambda **kwargs: None,
                apply_runtime_state=lambda state: None,
            )
        )
        controller.params = _Params()
        controller.on_verify_complete([3], 1, steps=3, forward_id=101)
        controller.observe_confidence([0.5], 1, 101)
        self.assertEqual(
            seen,
            [((3,), 1, 3, 101), ("observe", [0.5], 1, 101)],
            "a completion callback must only observe",
        )
        self.assertEqual(controller.worker.speculative_num_steps, 15)

    def test_the_processor_reports_the_result_not_the_worker(self):
        # A W8 result processed after the worker moved on: the hook must be told
        # what the RESULT ran at, and which launch it belongs to.
        calls = []
        proc = SimpleNamespace(
            model_worker=SimpleNamespace(
                on_verify_complete_cpu=lambda counts, **kwargs: calls.append(
                    (list(counts), kwargs)
                )
            ),
            advance_grammar_fsm=lambda *a, **k: None,
        )
        req = SimpleNamespace(
            is_retracted=False,
            finished=lambda: False,
            grammar=None,
            kv_committed_len=0,
            spec_verify_ct=0,
            spec_num_correct_drafts=0,
            update_spec_correct_drafts_histogram=lambda n: None,
        )
        result = SimpleNamespace(
            next_token_ids=torch.arange(10, 18),
            accept_lens=torch.tensor([8]),
            block_accept_lens=None,
            cap_lens=None,
            speculative_num_draft_tokens=8,
            grammar_advanced=True,
        )
        predict = SchedulerBatchResultProcessor._resolve_spec_v2_tokens(
            proc, result, SimpleNamespace(reqs=[req], forward_iter=101)
        )

        self.assertEqual(predict, [list(range(10, 18))])
        self.assertEqual(
            calls,
            [([7], {"batch_size": 1, "num_draft_tokens": 8, "forward_id": 101})],
            "the hook lost the producing batch's identity",
        )

    def test_the_pairing_helpers_use_existing_batch_metadata(self):
        # Resolved at call time so a tree without the pairing fails by name
        # instead of failing the module (and every other check) at import.
        key_of = getattr(base_spec_module, "chain_request_key", None)
        id_of = getattr(base_spec_module, "chain_forward_id", None)
        if key_of is None or id_of is None:
            self.fail("base_spec_worker lost the producing-batch helpers")
        req = lambda rid: SimpleNamespace(rid=rid)  # noqa: E731
        self.assertEqual(key_of(SimpleNamespace(reqs=[req("a")])), "a")
        self.assertIsNone(key_of(SimpleNamespace(reqs=[req("a"), req("b")])))
        self.assertIsNone(key_of(SimpleNamespace(reqs=[])))
        self.assertIsNone(key_of(SimpleNamespace()))
        # forward_iter is stamped by Scheduler.run_batch and survives the copy
        # the result processor receives -- that is the whole pairing argument.
        self.assertEqual(id_of(SimpleNamespace(reqs=[req("a")], forward_iter=42)), 42)
        self.assertIsNone(id_of(SimpleNamespace(reqs=[req("a")])))


class TestGenericPoliciesUnaffected(CustomTestCase):
    def test_the_ema_slot_keeps_its_behaviour_with_the_new_arguments(self):
        cfg = {
            "candidate_steps": [1, 3, 7],
            "ema_alpha": 1.0,
            "warmup_batches": 0,
            "update_interval": 1,
        }
        attributed = AdaptiveStepSlot(3, dict(cfg))
        self.assertTrue(attributed.update([0, 0], steps=7, forward_id=101))
        plain = AdaptiveStepSlot(3, dict(cfg))
        plain.update([0, 0])
        self.assertEqual(vars(attributed), vars(plain), "the generic EMA rule moved")

    def test_an_ema_launch_still_has_nowhere_to_put_a_sample(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_ADAPTIVE_POLICY", None)
            params = AdaptiveSpeculativeParams(initial_steps=15, cfg_path=None)
        self.assertFalse(params._confidence)
        slot = params._slots[1]
        before = vars(slot)
        # No confidence state exists to attribute a sample to, and reporting a
        # decision (present or absent) must not disturb the EMA slot -- or raise.
        params.observe_confidence([0.9], 1, 101)
        self.assertEqual(vars(slot), before)
        params.on_verify_complete([0], 1, steps=15, forward_id=101)
        self.assertEqual(slot.ema_accept_len, 0.8 * (slot.current_steps - 1))

    def test_confidence_slots_receive_the_producing_width_via_the_params(self):
        with patch.dict(os.environ, CONFIDENCE_ENV):
            params = AdaptiveSpeculativeParams(initial_steps=15, cfg_path=None)
            slot = params._slots[1]
        self.assertIsInstance(slot, ConfidenceStepSlot)
        # The default table's C1 candidates are [1, 3, 7], so a launch at 15
        # snaps the live width to 3 while the batch that produced these counts
        # ran the widest one.
        self.assertEqual(slot.current_steps, 3)
        params.observe_confidence([0.99], 1, 101)
        params.on_verify_complete([7], 1, steps=7, forward_id=101)
        b = slot._bucket(0.99)
        self.assertEqual(slot._sat[b][7], 1.0)
        self.assertEqual(slot._satn[b][7], 1)
        self.assertEqual(slot._satn[b][3], 0, "the live width claimed a foreign chain")

    def test_a_nonadaptive_or_dflash_worker_ignores_the_new_metadata(self):
        # BaseSpecWorker's default (which the 27B DFlash worker inherits) must
        # still accept the call the result processor makes, and move nothing.
        worker = SimpleNamespace(
            speculative_num_steps=3, speculative_num_draft_tokens=4
        )
        self.assertIsNone(
            BaseSpecWorker.on_verify_complete_cpu(
                worker, [1], batch_size=1, num_draft_tokens=4, forward_id=101
            )
        )
        self.assertEqual(
            (worker.speculative_num_steps, worker.speculative_num_draft_tokens), (3, 4)
        )


if __name__ == "__main__":
    unittest.main()
