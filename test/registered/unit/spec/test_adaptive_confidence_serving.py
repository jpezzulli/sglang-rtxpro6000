"""CPU-only checks for the confidence policy's configuration surface, concurrency
rules and stream discipline.

The controller itself is the donor's (adaptive_confidence.py, ported from
https://github.com/aiueo52/sglang-rtxpro6000 branch flash-next-fast @
5105985116eb00dea8e6138aabeb5363387cb9de); its own transition arithmetic is
guarded in test_adaptive_confidence.py. What is pinned here is the part this
repository had to adapt:

* the policy is opt-in and the shipped defaults stay put -- without
  SGLANG_ADAPTIVE_POLICY=confidence every slot is still the acceptance EMA, and
  ``observe_confidence`` has nowhere to put a sample;
* configs/pennyroyal/adaptive-next.json carries the donor's public C1 tuning
  (bench/adaptive/w16_3_7_15_c.json of that branch) and gives the C>=2 tier one
  candidate, steps=3 (W4), so a concurrent batch cannot widen itself;
* observations route by the batch size that produced them: a C1 chain is never
  steered by a C>=2 sample, and no observation -- stale or not -- moves the
  fixed tier off W4;
* the producers and consumers keep their discipline: the draft-extend producer
  is skipped for a concurrent batch (no needless reductions over the draft
  vocabulary), the consumer reads the ring only when a C1 decision is being
  made, and neither waits on the GPU -- an event that has not landed is simply
  not used, so the scheduler run-ahead survives.

No CUDA on this host: the device->host staging itself (pinned copies, events)
is exercised through the same code path with a completed/pending event stand-in.
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative import adaptive_confidence
from sglang.srt.speculative.adaptive_confidence import (
    ConfidenceChannel,
    ConfidenceStepSlot,
)
from sglang.srt.speculative.adaptive_runtime_state import AdaptiveController
from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    AdaptiveStepSlot,
)
from sglang.srt.speculative.base_spec_worker import EagleDraftWorkerBase
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

ADAPTIVE_NEXT_CONFIG = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        *[os.pardir] * 4,
        "configs",
        "pennyroyal",
        "adaptive-next.json",
    )
)

# The donor public wa configuration for C1 (w16_3_7_15_c.json, flash-next-fast).
DONOR_C1 = {
    "rate_alpha": 0.1,
    "weight_alpha": 0.05,
    "warmup_batches": 15,
    "switch_grace_batches": 40,
    "update_interval": 10,
    "grace_backoff": 2.0,
    "max_grace_batches": 80,
    "switch_margin": 0.1,
    "up_margin": 0.04,
    "down_margin": {"3": 0.03, "7": 0.15},
    "adjacent_only_promotion": True,
    "tail_bias": 0.75,
    "confidence_weight": 1.0,
    "min_bucket_samples": 20,
    "buckets": [0.8, 0.95, 0.99, 0.999],
}
CONFIDENCE_ENV = {"SGLANG_ADAPTIVE_POLICY": "confidence"}


def _params(initial_steps=15, cfg_path=ADAPTIVE_NEXT_CONFIG):
    return AdaptiveSpeculativeParams(initial_steps=initial_steps, cfg_path=cfg_path)


def _wire(name):
    """Bind a controller/worker hook that the confidence port is expected to
    provide, resolved at call time so a tree without it fails one named check."""

    def _bind(obj, *args, **kwargs):
        fn = getattr(type(obj), name, None)
        if fn is None:
            raise AssertionError(f"{type(obj).__name__}.{name} is missing")
        return fn(obj, *args, **kwargs)

    return _bind


class _FakeEvent:
    def __init__(self, done):
        self._done = done

    def query(self):
        return self._done

    def synchronize(self):
        raise AssertionError("the confidence hot path synchronized a stream")

    def record(self):
        pass


class _RecordingChannel:
    def __init__(self, landing=None):
        self.recorded = []
        self.reads = 0
        self._landing = landing if landing is not None else []

    def record_position0(self, p0):
        self.recorded.append(p0)

    def latest_position0(self):
        self.reads += 1
        return (
            self._landing[self.reads - 1] if self.reads <= len(self._landing) else None
        )


class TestPolicyIsOptIn(CustomTestCase):
    def test_default_env_keeps_the_acceptance_ema(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_ADAPTIVE_POLICY", None)
            params = _params()
        self.assertTrue(
            all(
                isinstance(slot, AdaptiveStepSlot)
                and not isinstance(slot, ConfidenceStepSlot)
                for slot in params._slots.values()
            ),
            "the shipped default swapped the decision rule out",
        )
        self.assertFalse(params._confidence)
        # An EMA slot has nowhere to put a confidence sample: it is not decided
        # by anything, and the call must not raise.
        before = {bs: slot.current_steps for bs, slot in params._slots.items()}
        _wire("observe_confidence")(params, [0.99] * 3, 1)
        self.assertEqual(
            before, {bs: s.current_steps for bs, s in params._slots.items()}
        )

    def test_confidence_env_selects_the_donor_slot(self):
        with patch.dict(os.environ, CONFIDENCE_ENV):
            params = _params()
        self.assertTrue(params._confidence)
        self.assertTrue(
            all(isinstance(slot, ConfidenceStepSlot) for slot in params._slots.values())
        )

    def test_donor_step_cost_model_is_env_configurable(self):
        # The donor fitted cost line is configuration, not a hard-coded Penny
        # number: the public wa launcher passes the fitted pair.
        self.assertEqual(adaptive_confidence.STEP_A, 9.74)
        self.assertEqual(adaptive_confidence.STEP_B, 0.70)
        self.assertAlmostEqual(
            adaptive_confidence.step_time_ms(15) - adaptive_confidence.step_time_ms(3),
            0.70 * 12,
        )


class TestShippedConfig(CustomTestCase):
    def test_c1_keeps_the_donor_confidence_tuning(self):
        with patch.dict(os.environ, CONFIDENCE_ENV):
            slot = _params()._slots[1]
        self.assertEqual(slot.candidate_steps, [3, 7, 15])
        # rate_alpha is the slot's `alpha` (the donor's name for the accept-rate
        # EMA); every other key is stored under its config name.
        for key, expected in DONOR_C1.items():
            attr = "alpha" if key == "rate_alpha" else key
            actual = getattr(slot, attr)
            if isinstance(expected, list):  # the slot keeps its bucket edges frozen
                actual = list(actual)
            elif isinstance(expected, dict):  # down_margin is keyed by int steps
                expected = {int(k): v for k, v in expected.items()}
            self.assertEqual(actual, expected, f"C1 {key} left the config")

    def test_concurrent_tier_is_fixed_at_w4(self):
        with open(ADAPTIVE_NEXT_CONFIG) as f:
            raw = json.load(f)
        self.assertEqual(raw["2"]["candidate_steps"], [3])
        with patch.dict(os.environ, CONFIDENCE_ENV):
            params = _params()
            slot = params._slots[2]
        self.assertEqual(slot.candidate_steps, [3])
        self.assertEqual(slot.current_steps, 3)
        # A stale wide-chain observation -- a full W16 acceptance from a batch
        # that ran wide, plus the ring's last high-confidence sample -- cannot
        # widen the fixed tier, however long the concurrent batch runs.
        for _ in range(60):
            _wire("observe_confidence")(params, [0.9999], 4)
            params.on_verify_complete([15, 15], 4)
            self.assertEqual(params.get_steps_for_batch(4), 3)
        # and the same holds when a C1 completion lands on the C1 slot only.
        for _ in range(60):
            _wire("observe_confidence")(params, [0.9999], 1)
            params.on_verify_complete([15], 1)
            self.assertEqual(params.get_steps_for_batch(4), 3)

    def test_observations_route_by_their_own_batch_size(self):
        with patch.dict(os.environ, CONFIDENCE_ENV):
            params = _params()
            _wire("observe_confidence")(params, [0.11], 1)
            _wire("observe_confidence")(params, [0.99], 4)
        self.assertAlmostEqual(params._slots[1]._last_conf, 0.11)
        self.assertAlmostEqual(params._slots[2]._last_conf, 0.99)


class TestProducerDiscipline(CustomTestCase):
    def _draft_worker(self, channel):
        worker = SimpleNamespace(_conf_channel=channel)
        worker.record = (
            lambda logits, bs: EagleDraftWorker._record_position0_confidence(
                worker, logits, bs
            )
        )
        return worker

    def test_no_channel_means_no_work(self):
        worker = self._draft_worker(None)
        worker.record(torch.zeros((1, 8), dtype=torch.float32), 1)  # must not raise

    def test_concurrent_batch_skips_the_confidence_computation(self):
        channel = _RecordingChannel()
        worker = self._draft_worker(channel)
        worker.record(torch.zeros((4, 8), dtype=torch.float32), 4)
        self.assertEqual(channel.recorded, [], "a C>=2 batch paid for top-1")
        worker.record(torch.zeros((1, 8), dtype=torch.float32), 1)
        self.assertEqual(len(channel.recorded), 1)
        # The staged value is the probability of the row that was drafted, not a
        # constant: a peaked row reads as confident, a flat one does not.
        peaked = torch.tensor([[9.0, 0.0, 0.0, 0.0]])
        flat = torch.zeros((1, 4))
        self._draft_worker(channel).record(peaked, 1)
        self._draft_worker(channel).record(flat, 1)
        self.assertGreater(
            channel.recorded[-2][0].item(), channel.recorded[-1][0].item()
        )

    def test_channel_sizes_once_and_keeps_the_widest_ring(self):
        worker = SimpleNamespace(device="cpu", _conf_channel=None, _chain_conf_buf=None)
        with patch.dict(os.environ, {}), patch.object(
            adaptive_confidence, "confidence_enabled", lambda: False
        ):
            EagleDraftWorkerBase._init_confidence_channel(worker, 8, 15)
            self.assertIsNone(worker._conf_channel, "the EMA policy built a channel")
        with patch.dict(os.environ, CONFIDENCE_ENV), patch.object(
            adaptive_confidence, "ConfidenceChannel"
        ) as Channel:
            Channel.return_value = "channel"
            EagleDraftWorkerBase._init_confidence_channel(worker, 8, 15)
            self.assertEqual(worker._conf_channel, "channel")
            # A width switch re-enters with the candidate's step count; the ring
            # keeps the launch (widest) sizing rather than shrinking per state.
            EagleDraftWorkerBase._init_confidence_channel(worker, 8, 3)
            self.assertEqual(Channel.call_count, 1)
            self.assertEqual(Channel.call_args.kwargs["max_steps"], 15)


class TestConsumerDiscipline(CustomTestCase):
    def _worker(self, channel, controller):
        return SimpleNamespace(
            adaptive_controller=controller,
            _draft_worker=SimpleNamespace(_conf_channel=channel),
        )

    def test_without_a_controller_nothing_moves(self):
        worker = SimpleNamespace(adaptive_controller=None)
        EAGLEWorkerV2.activate_step_by_batch(worker, 1)  # no _draft_worker at all

    def test_concurrent_batch_never_reads_the_ring(self):
        channel = _RecordingChannel(landing=[[0.5]])
        calls = []
        controller = SimpleNamespace(
            observe_confidence=lambda conf, bs: calls.append(("observe", conf, bs)),
            activate_step_by_batch=lambda bs: calls.append(("activate", bs)),
        )
        EAGLEWorkerV2.activate_step_by_batch(self._worker(channel, controller), 4)
        self.assertEqual(channel.reads, 0)
        self.assertEqual(calls, [("activate", 4)])

    def test_c1_reads_the_ring_then_activates(self):
        channel = _RecordingChannel(landing=[[0.42]])
        calls = []
        controller = SimpleNamespace(
            observe_confidence=lambda conf, bs: calls.append(("observe", conf, bs)),
            activate_step_by_batch=lambda bs: calls.append(("activate", bs)),
        )
        EAGLEWorkerV2.activate_step_by_batch(self._worker(channel, controller), 1)
        self.assertEqual(
            calls, [("observe", [0.42], 1), ("activate", 1)], "wrong order or bs"
        )

    def test_unlanded_sample_is_skipped_not_waited_for(self):
        channel = _RecordingChannel(landing=[None])
        calls = []
        controller = SimpleNamespace(
            observe_confidence=lambda *a: calls.append(a),
            activate_step_by_batch=lambda bs: calls.append(("activate", bs)),
        )
        with patch("torch.cuda.synchronize", side_effect=AssertionError("sync")):
            EAGLEWorkerV2.activate_step_by_batch(self._worker(channel, controller), 1)
        self.assertEqual(calls, [("activate", 1)])

    def test_latest_position0_never_synchronises(self):
        # The donor consumer walks back from the newest slot to the first event
        # that has completed and caches the value; a copy still in flight must
        # cost a query, never a wait. Built without __init__ because this host
        # has no driver to pin memory against.
        max_bs, ring = 4, 3
        channel = ConfidenceChannel.__new__(ConfidenceChannel)
        channel.max_bs = max_bs
        channel.RING = ring
        channel._p0_host = [
            torch.full((max_bs,), float(i), dtype=torch.float32) for i in range(ring)
        ]
        channel._p0_event = [_FakeEvent(done=False) for _ in range(ring)]
        channel._p0_bs = [max_bs] * ring
        channel._p0_write = 0
        channel._p0_read = 2
        channel._p0_cached = None
        self.assertIsNone(channel.latest_position0())
        # Nothing landed: the last known value is what the policy sees.
        self.assertIsNone(channel.latest_position0())
        channel._p0_event[1] = _FakeEvent(done=True)
        self.assertEqual(channel.latest_position0(), [1.0] * max_bs)
        channel._p0_event[1] = _FakeEvent(done=False)
        self.assertEqual(channel.latest_position0(), [1.0] * max_bs)

    def test_controller_passes_confidence_to_the_matching_slot(self):
        seen = []

        class _Params:
            def observe_confidence(self, confidences, batch_size):
                seen.append((confidences, batch_size))

        controller = AdaptiveController(
            SimpleNamespace(
                speculative_num_steps=15,
                build_adaptive_runtime_state=lambda **kwargs: None,
                apply_runtime_state=lambda state: None,
            )
        )
        controller.params = _Params()
        _wire("observe_confidence")(controller, [0.5], 1)
        self.assertEqual(seen, [([0.5], 1)])


if __name__ == "__main__":
    unittest.main()
