"""Draft-confidence side channel and the confidence/throughput step policy (C1).

Why a side channel
------------------
The shipped adaptive controller (``adaptive_spec_params.AdaptiveStepSlot``)
picks the chain length from one number: an EMA of ``accept_lens - 1``.  That
statistic is (a) purely historical and (b) not on the same scale as the
quantity the decision actually needs, which is *throughput*:

    tokens/s(S) = (1 + E[accepted | S]) / step_time(S)

This module supplies the two missing ingredients.

1. ``ConfidenceChannel`` makes the draft model's own top-1 probability
   available on the host.  For topk=1 drafting neither producer computed it
   (both wrote a constant 1.0), so this is new information, not a new
   smoothing of the old information.  The probability of chain position 0 --
   the token ``_draft_extend_for_decode`` selected at the end of the previous
   iteration -- is the only confidence available *before* the step-count
   decision is taken, because the whole draft loop runs inside one captured
   CUDA graph (see specs/C1_LOG.md section 1).

2. ``ConfidenceStepSlot`` turns observations into a throughput comparison
   instead of a threshold test.  Modelling per-position acceptance as flat after
   position 2 (a modelling assumption), a chain of length S behaves
   like S iid Bernoulli(r) trials in series:

       E[accepted | S] = sum_{k=1..S} r^k = r (1 - r^S) / (1 - r)

   A *single* per-position acceptance rate ``r`` therefore explains the
   observed mean at whichever S is currently live, and -- crucially --
   predicts the mean at the S that is *not* live.  The EMA has no such
   transfer: it measures a quantity whose scale changes with every switch,
   which is why the shipped config needs the asymmetric "re-seed on step-down
   only" hack.  Checked against the 2026-09-05 fixed-profile measurements:

       workload    E@15   -> r     -> predicted E@3   measured E@3
       code-edit    9.87    0.930      2.60            2.86
       agent-loop   3.82    0.797      1.95            2.29
       prose-en     1.83    0.645      1.33            1.51
       prose-ja     1.55    0.605      1.21            1.31

   (systematically ~15% low, because acceptance is *not* iid at positions 0-1;
   ``position_bias`` absorbs that.)

``r`` is tracked per confidence bucket, so a request whose next token the
draft is sure about is allowed a long chain even while the running average
says otherwise -- which is exactly the code-edit failure mode: its accept
distribution at S=15 is bimodal (full chains interleaved with 0/1), so the
*mean* dips under a fixed threshold even though most steps still want 15.
"""

from __future__ import annotations

import collections
import logging
import math
import os
from typing import Optional

import torch

# Ported from https://github.com/aiueo52/sglang-rtxpro6000 branch flash-next-fast,
# snapshot 5105985116eb00dea8e6138aabeb5363387cb9de, where it is a new file under
# the project's Apache License 2.0 (upstream MODIFICATIONS.md carries the Aiueo52
# copyright notice and the list of modified upstream files). The C1 decision rule,
# the env configuration surface (SGLANG_ADAPTIVE_POLICY / _TRACE / _DEBUG /
# SGLANG_ADAPTIVE_STEP_A / _STEP_B) and the donor's fitted default step costs are
# its own, unchanged; no synchronisation is added to the hot path. The batch-size
# gating of the producers/consumer lives with the worker that owns them
# (eagle_worker_v2), not here.
#
# Two local corrections, both in the delivery path rather than the arithmetic:
# an accept-count sample is credited to the width and the launch that produced it
# (ConfidenceStepSlot.update / _paired_bucket, keyed on the scheduler's immutable
# forward id) and a staged confidence is only readable by the request whose chain
# it describes, within the current invalidation generation (ConfidenceChannel
# request_key / invalidate_position0). The donor inherited the pre-overlap
# assumption that "current" still describes the batch being reported; with an
# overlapped schedule it does not. See test_adaptive_feedback_attribution.py.
logger = logging.getLogger(__name__)


def _env_policy() -> str:
    return os.environ.get("SGLANG_ADAPTIVE_POLICY", "ema") or "ema"


def _env_trace() -> str:
    return os.environ.get("SGLANG_ADAPTIVE_TRACE", "")


def confidence_policy_enabled() -> bool:
    """The runtime policy needs the position-0 probability (eager, cheap)."""
    return _env_policy() == "confidence"


def chain_trace_enabled() -> bool:
    """Offline analysis needs every chain position (writes inside the draft graph)."""
    return bool(_env_trace())


def confidence_enabled() -> bool:
    return confidence_policy_enabled() or chain_trace_enabled()


def top1_prob(logits: torch.Tensor) -> torch.Tensor:
    """softmax(logits).max(-1) without materialising the softmax.

    ``logsumexp`` + ``amax`` are two reductions over the draft's hot vocabulary
    (49152 wide); at the batch sizes this server runs they cost a few
    microseconds against a 12-20 ms step.
    """
    f = logits.float()
    return (f.amax(dim=-1) - torch.logsumexp(f, dim=-1)).exp()


# Sentinel for a consumer that has no request identity to pair against.
_ANY_REQUEST = object()


class ConfidenceChannel:
    """Async device->host staging for draft confidences.

    Two producers, both optional:
      * ``record_position0`` -- eager, at draft-extend, one value per request.
        This is what the policy reads, and it is ready a whole iteration before
        it is needed.
      * ``record_chain`` -- the full ``(bs, steps)`` matrix the Triton
        postprocess kernel filled during the draft.  Trace-only.

    Copies go out non-blocking on the current stream into a ring of pinned
    buffers; each slot carries an event so a reader never has to guess.  Nothing
    here ever synchronises the producing stream.

    A staged sample is a statement about ONE chain -- the one the producer had
    just drafted -- so it is paired with the request that owns that chain
    (``request_key``) and dropped outright by any pass that could not have
    produced it (``invalidate_position0``). Handing "the newest value" to
    whoever asks is what let a finished request's, or a concurrent batch's,
    confidence steer an unrelated C1 decision.
    """

    RING = 16

    def __init__(self, device: str, max_bs: int, max_steps: int):
        self.device = device
        self.max_bs = max_bs
        self.max_steps = max_steps
        self._p0_host = [
            torch.empty(max_bs, dtype=torch.float32, pin_memory=True)
            for _ in range(self.RING)
        ]
        self._p0_event = [torch.cuda.Event() for _ in range(self.RING)]
        self._p0_bs = [0] * self.RING
        # Which request's chain each staged sample describes, and which
        # invalidation generation it was staged under. The ring otherwise hands
        # out "the newest value" to whoever asks -- and, because the reader
        # scans backwards over every slot, a slot that survived an invalidation
        # can be revived by the copy staged after it.
        self._p0_key: list[Optional[str]] = [None] * self.RING
        self._p0_epoch = [0] * self.RING
        self._epoch = 0
        self._p0_write = 0
        self._p0_read = -1  # index of the newest slot with a copy in flight
        self._p0_cached: Optional[list[float]] = None
        self._p0_cached_id: Optional[tuple] = None

        self._chain_host = None
        self._chain_event = None
        self._chain_write = 0
        self._chain_pending: collections.deque = collections.deque()
        if chain_trace_enabled():
            self._chain_host = [
                torch.empty((max_bs, max_steps), dtype=torch.float32, pin_memory=True)
                for _ in range(self.RING)
            ]
            self._chain_event = [torch.cuda.Event() for _ in range(self.RING)]

    # -- producers ---------------------------------------------------------
    def record_position0(
        self, p0: torch.Tensor, request_key: Optional[str] = None
    ) -> None:
        bs = p0.shape[0]
        if bs == 0 or bs > self.max_bs or torch.cuda.is_current_stream_capturing():
            return
        slot = self._p0_write
        self._p0_host[slot][:bs].copy_(p0.view(-1), non_blocking=True)
        self._p0_event[slot].record()
        self._p0_bs[slot] = bs
        self._p0_key[slot] = request_key
        self._p0_epoch[slot] = self._epoch
        self._p0_read = slot
        self._p0_write = (slot + 1) % self.RING

    def invalidate_position0(self) -> None:
        """Expire every staged chain confidence; host-side bookkeeping, no sync.

        A staged sample is the probability of one specific chain's first token.
        A decode pass that drafted without staging one (the C>=2 gate) therefore
        consumed it: what the ring holds predates tokens the request has already
        committed, so the next C1 decision must see "no sample", not that value.

        Advancing the generation -- rather than only the read pointer and cache
        -- is what makes that stick: a reader walks back over the whole ring, so
        a slot left at the previous generation cannot be pulled back into use by
        the copy staged after the invalidation.
        """
        self._epoch += 1
        self._p0_read = -1
        self._p0_cached = None
        self._p0_cached_id = None

    def record_chain(self, chain: torch.Tensor, bs: int, steps: int) -> None:
        if (
            self._chain_host is None
            or bs == 0
            or bs > self.max_bs
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        slot = self._chain_write
        self._chain_host[slot][:bs, :steps].copy_(chain[:bs, :steps], non_blocking=True)
        self._chain_event[slot].record()
        self._chain_write = (slot + 1) % self.RING
        self._chain_pending.append((slot, bs, steps))

    # -- consumers ---------------------------------------------------------
    def latest_position0(self, request_key=_ANY_REQUEST) -> Optional[list[float]]:
        """Newest position-0 confidences whose copy has ALREADY landed.

        *request_key* restricts the answer -- and the cached fallback below --
        to the sample staged for that request; pass nothing to keep the donor's
        "newest landed value" behaviour where no identity is available.

        Never synchronises.  The scheduler deliberately runs the CPU ahead of
        the GPU (that is why the adaptive controller is fed from
        ``batch_result_processor`` after ``accept_lens`` is already on the
        host, rather than from the worker hot path), so waiting on an event
        here would collapse the run-ahead and cost far more than the decision
        is worth.  Instead: walk back from the newest slot to the first one
        whose event has completed, and cache the value.  In practice that is
        one or two iterations old, which is inside the lag the controller
        already tolerates -- ``update()`` samples arrive late by the same
        mechanism.  Returns the last known value when nothing new has landed.
        """
        newest = self._p0_read
        if newest < 0:
            return self._cached_for(request_key)
        for k in range(self.RING):
            slot = (newest - k) % self.RING
            if self._p0_bs[slot] == 0 or self._p0_epoch[slot] != self._epoch:
                continue
            if request_key is not _ANY_REQUEST and self._p0_key[slot] != request_key:
                continue
            if self._p0_event[slot].query():
                self._p0_cached = self._p0_host[slot][: self._p0_bs[slot]].tolist()
                self._p0_cached_id = (self._epoch, self._p0_key[slot])
                return self._p0_cached
        return self._cached_for(request_key)

    def _cached_for(self, request_key) -> Optional[list[float]]:
        """The last value handed out, if it still belongs to this reader's chain."""
        if request_key is _ANY_REQUEST:
            return self._p0_cached
        return (
            self._p0_cached
            if self._p0_cached_id == (self._epoch, request_key)
            else None
        )

    def pop_chain(self) -> Optional[tuple[list[list[float]], int]]:
        """Oldest recorded chain, paired FIFO with the verify results."""
        if not self._chain_pending:
            return None
        slot, bs, steps = self._chain_pending.popleft()
        self._chain_event[slot].synchronize()
        return self._chain_host[slot][:bs, :steps].tolist(), steps


class ChainTracer:
    """One line per verified decode step: the input to the offline simulator."""

    def __init__(self, path: str):
        # Line buffered on purpose.  The offline split into per-workload
        # segments is driven by `wc -l` on this file between fnbench
        # invocations, so a 64 KB buffer (~300 rows at steps=15) silently
        # shifts every boundary and each segment ends up a blend of two
        # workloads -- which is exactly what happened to the first w16 trace
        # (its per-segment mean acceptance disagreed with fnbench's own
        # acc: 8.94/4.01/1.81/2.68 traced vs 7.30/2.16/1.45/4.19 measured).
        self._f = open(path, "a", buffering=1)
        self._n = 0

    def write(self, steps: int, bs: int, chain, accepted: list[int]) -> None:
        import json
        import time

        self._f.write(
            json.dumps(
                {
                    "i": self._n,
                    # Wall clock, so the offline split into per-workload
                    # segments can key off the idle gap between fnbench
                    # invocations instead of trusting a line count.
                    "t": round(time.time(), 3),
                    "steps": steps,
                    "bs": bs,
                    "p": (
                        [[round(x, 5) for x in row] for row in chain]
                        if chain is not None
                        else None
                    ),
                    "a": accepted,
                }
            )
            + "\n"
        )
        self._n += 1

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

# ms/iteration = STEP_A + STEP_B * S.  Fitted to the 2026-09-05 fnbench
# end-to-end step times (tokens/acc), which is the quantity being maximised:
#   W4  (S=3):  11.7-12.2 ms      W16 (S=15): 19.9-21.9 ms
# The pure-decode profiler numbers quoted in the C1 brief (10.2 / 17.3 ms at
# T=4 / T=16) have the same slope but a smaller intercept; they omit the
# scheduler/detokenizer overhead that fnbench throughput does contain.  Both
# put the S=15 step at ~1.7x the S=3 step, which is all the decision uses.
STEP_A = float(os.environ.get("SGLANG_ADAPTIVE_STEP_A", "9.74"))
STEP_B = float(os.environ.get("SGLANG_ADAPTIVE_STEP_B", "0.70"))


def step_time_ms(steps: int) -> float:
    return STEP_A + STEP_B * steps


def expected_accept(r: float, steps: int) -> float:
    """E[accepted drafts] for a length-*steps* chain at per-position rate *r*."""
    if steps <= 0:
        return 0.0
    r = min(max(r, 1e-4), 0.999999)
    return r * (1.0 - r**steps) / (1.0 - r)


def invert_accept(mean_accept: float, steps: int) -> float:
    """The *r* whose length-*steps* chain accepts *mean_accept* drafts.

    Monotone in r, so a fixed-count bisection is exact enough and has no
    convergence branch to get wrong in a hot path (30 iterations of bisection
    on [0,1) is ~1e-9).
    """
    if steps <= 0 or mean_accept <= 0.0:
        return 0.0
    if mean_accept >= steps - 1e-6:
        return 0.999999
    lo, hi = 0.0, 0.999999
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if expected_accept(mid, steps) < mean_accept:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class ConfidenceStepSlot:
    """Throughput-maximising step choice from directly observable statistics.

    Drop-in for ``AdaptiveStepSlot``: same ``current_steps`` / ``candidate_steps``
    / ``update()`` contract, plus ``observe_confidence()``.

    It picks ``argmax_S sum_b w_b (1 + E[accepted | b, S]) / step_time(S)``.
    Everything interesting is in how ``E[accepted | b, S]`` is obtained, and the
    two directions are not symmetric:

    **Downward (S below the live chain) is exact, with no model at all.**  A
    topk=1 draft is a greedy chain and the target accepts a *prefix* of it, and
    the first S' tokens of a length-S chain are the same tokens a length-S'
    chain would have drafted.  So a step observed at S tells you exactly what
    S' < S would have accepted: ``min(accepted, S')``.  Checked against the
    2026-09-06 traces -- ``mean(min(a_15, 3))`` on the W16 run versus the mean
    accepted actually measured on the separate W4 run:

        code-edit  2.66 vs 2.76 (+4%)      prose-en   1.49 vs 1.47 (-1%)
        prose-ja   1.14 vs 1.35 (+18%)     agent-loop 2.09 vs 2.10 (+0%)

    **Upward is censored, and this is where the confidence earns its keep.**
    At S=3 an accepted count of 3 could mean "3" or "would have been 14"; the
    mean saturates and carries no information about how much chain is being
    left on the table.  What survives censoring is the *hazard* at the
    boundary: ``r = P(a >= S) / P(a >= S-1)``.  Assuming per-position acceptance is flat
    after position 2 (a modelling assumption), that one rate extends the
    chain:

        E[accepted | T] = E[min(a, S)] + P(a >= S) * r (1 - r^(T-S)) / (1 - r)

    Measured on the W4 traces predicting the W16 runs: +14%, -8%, +17%, +9%.
    Optimistic, so ``tail_bias`` discounts it and ``switch_margin`` covers the
    rest.  The upward estimate is ALWAYS recomputed from the live state rather
    than read from a remembered EMA of the other state: a remembered value goes
    stale exactly when it matters, i.e. when the workload has changed.

    **Why buckets.**  The estimates above are conditioned on the position-0
    confidence bucket and recombined over the recent bucket occupancy ``w_b``
    rather than pooled.  Pooling asks about a workload's mean; the mean is a
    bad summary of code-edit, whose accept distribution at S=15 is spread
    across 0..15 with a 30-40% spike at the full chain, and a threshold test on
    it (what the shipped EMA slot does) is answering the wrong question.  The
    confidence is what makes the sub-populations separable: on the 2026-09-06
    W16 traces the mean accepted mostly rises across the five buckets (one dip)
    -- prose-en 1.29 / 2.15 / 3.58 / 3.20 / 3.45, agent-loop 2.10 / 3.73 /
    4.56 / 4.74 / 5.18, code-edit 2.93 / 4.68 / 8.83 / 9.95 / 11.70 -- with
    corr(p0, accepted) between 0.37 and 0.53.

    A switch leaves the draft state cold for a batch or two, so decisions are
    held for a grace window; the per-step confidence is therefore NOT used to
    pick a step count per step.  It is used to decompose the window.
    """

    # Position-0 confidence bucket edges, packed against 1.0 because that is
    # where the mass is (code-edit at W4: 10th percentile 0.966, median 1.000).
    BUCKETS = (0.8, 0.95, 0.99, 0.999)

    def __init__(self, initial_steps: int, cfg: dict):
        candidates = sorted(set(cfg["candidate_steps"]))
        assert candidates, "candidate_steps must have at least 1 value"
        self.candidate_steps = candidates
        self.current_steps = (
            initial_steps
            if initial_steps in candidates
            else candidates[len(candidates) // 2]
        )

        self.alpha = float(cfg.get("rate_alpha", 0.05))
        self.weight_alpha = float(cfg.get("weight_alpha", 0.02))
        self.update_interval = int(cfg.get("update_interval", 4))
        self.warmup_batches = int(cfg.get("warmup_batches", 15))
        self.switch_grace_batches = int(cfg.get("switch_grace_batches", 20))
        # Anti-oscillation. When two candidates are genuinely close, small
        # estimate noise flips the argmax and the controller ping-pongs at the
        # crossover (measured offline on agent-loop: 24 switches over 3000
        # batches, -3% against just staying at W4). Each REVERSAL of the
        # previous switch's direction multiplies the grace window, so a true
        # tie converges to holding one state -- which costs almost nothing,
        # precisely because it is a tie -- while a real preference change still
        # gets through on the first decision.
        self.grace_backoff = float(cfg.get("grace_backoff", 2.0))
        self.max_grace_batches = int(cfg.get("max_grace_batches", 2000))
        self._grace = self.switch_grace_batches
        self._last_dir = 0
        self.switch_margin = float(cfg.get("switch_margin", 0.05))
        # Optional downward thresholds, independent of the upward incumbent
        # bonus. A scalar applies to all downshifts; a mapping is keyed by
        # destination steps (e.g. {"3": .03, "7": .12}). Missing destinations
        # retain switch_margin. Leave the legacy score arithmetic untouched
        # when the key is absent.
        down_margin = cfg.get("down_margin")
        self.down_margin = (
            {int(s): float(v) for s, v in down_margin.items()}
            if isinstance(down_margin, dict)
            else float(down_margin) if down_margin is not None else None
        )
        margins = (
            self.down_margin.values()
            if isinstance(self.down_margin, dict)
            else [self.down_margin] if self.down_margin is not None else []
        )
        if any(not math.isfinite(v) or v < 0 for v in margins):
            raise ValueError("down_margin must contain finite non-negative values")
        self.adjacent_only_promotion = cfg.get("adjacent_only_promotion", False)
        if not isinstance(self.adjacent_only_promotion, bool):
            raise ValueError("adjacent_only_promotion must be a boolean")
        # Asymmetric on purpose, because the two estimates are not equally
        # trustworthy. Going DOWN uses min(accepted, S), which is exact. Going
        # UP uses the hazard extrapolation, which measured 8-17% optimistic --
        # and on the 2026-09-06 server A/B that optimism cost agent-loop 8.6%:
        # it predicted E@15 = 5.53 (323 tok/s) against a true ~3.5, stepped up
        # mid-benchmark and had to come back 5 s later. A candidate whose
        # estimate is extrapolated must therefore clear a wider margin. It
        # separates cleanly: at S=3 code-edit predicts +88% for the long chain
        # while agent-loop and the prose workloads predict within +/-5%.
        self.up_margin = float(cfg.get("up_margin", 0.30))
        # The hazard extrapolation runs 8-17% optimistic against measurement.
        self.tail_bias = float(cfg.get("tail_bias", 0.9))
        # 0 pools every step into one estimate (option 3: a better statistic,
        # no confidence); 1 trusts the buckets, shrunk by their sample count.
        self.confidence_weight = float(cfg.get("confidence_weight", 1.0))
        self.min_bucket_samples = int(cfg.get("min_bucket_samples", 20))
        self.buckets = tuple(cfg.get("buckets", self.BUCKETS))

        nb = len(self.buckets) + 1
        self._nb = nb
        cands = self.candidate_steps
        # Per bucket: EMA of min(accepted, c) for every candidate c, updated
        # only from samples that were NOT censored at c (i.e. live S >= c).
        self._m = [{c: 0.0 for c in cands} for _ in range(nb)]
        self._mn = [{c: 0 for c in cands} for _ in range(nb)]
        # Per bucket, per live S: EMA of P(a >= S) and P(a >= S-1), the two
        # numbers the boundary hazard needs.
        self._sat = [{c: 0.0 for c in cands} for _ in range(nb)]
        self._satp = [{c: 0.0 for c in cands} for _ in range(nb)]
        self._satn = [{c: 0 for c in cands} for _ in range(nb)]
        self._w = [1.0 / nb] * nb
        # Pooled twins, used before a bucket has samples and when
        # confidence_weight is 0.
        self._pm = {c: 0.0 for c in cands}
        self._pmn = {c: 0 for c in cands}
        self._psat = {c: 0.0 for c in cands}
        self._psatp = {c: 0.0 for c in cands}
        self._psatn = {c: 0 for c in cands}

        self._batch_count = 0
        self._grace_until = 0
        self._next_bucket = nb // 2
        self._last_conf: Optional[float] = None
        # (forward_id, bucket_or_None) per reported C1 decision; one popped per
        # verify result routed to this slot. Bounded by the overlap run-ahead.
        self._inflight_bucket: collections.deque = collections.deque(maxlen=8)
        self._dbg = os.environ.get("SGLANG_ADAPTIVE_DEBUG", "") == "1"

    # -- inputs ------------------------------------------------------------
    def _bucket(self, conf: float) -> int:
        i = 0
        for edge in self.buckets:
            if conf < edge:
                return i
            i += 1
        return i

    def observe_confidence(
        self, confidences: list[float], forward_id: Optional[int] = None
    ) -> None:
        """Record the bucket that the width decision for this forward used.

        Called once per C1 decode forward, with an empty list when the ring had
        nothing landed: that is an explicit "no confidence this time" decision,
        recorded so the verify result of THIS forward can consume exactly the
        decision taken for it (see ``_paired_bucket``). The key is the
        scheduler's forward id -- a request id cannot tell two in-flight
        forwards of one request apart.
        """
        if not confidences:
            self._inflight_bucket.append((forward_id, None))
            return
        conf = sum(confidences) / len(confidences)
        self._last_conf = conf
        self._next_bucket = self._bucket(conf)
        self._inflight_bucket.append((forward_id, self._next_bucket))

    def _paired_bucket(self, forward_id: Optional[int]) -> Optional[int]:
        """Consume the decision entry recorded for this verify result's forward.

        The entry is taken whether or not the counts can then be credited: a
        result left unpaired shifted every later observation by one, so a
        rejected batch credited the NEXT batch's confidence. No match -- or a
        decision that had no sample -- yields None, and the caller credits the
        pooled twins only. Falling back to "the current bucket" instead is what
        let a result silently borrow another iteration's confidence.
        """
        for i in range(len(self._inflight_bucket)):
            if self._inflight_bucket[i][0] == forward_id:
                bucket = self._inflight_bucket[i][1]
                del self._inflight_bucket[i]
                return bucket
        return None

    @staticmethod
    def _ema(cur: float, n: int, x: float, a: float) -> float:
        return x if n == 0 else (1 - a) * cur + a * x

    def update(
        self,
        num_correct_drafts_per_req: list[int],
        steps: Optional[int] = None,
        forward_id: Optional[int] = None,
    ) -> bool:
        """Credit one batch's accept counts. Returns True if params changed.

        *steps* is the draft width the batch actually ran at and *forward_id* the
        launch it ran at; both come from the result, not from this slot, because
        under overlap either may have moved on since the forward (see
        ``AdaptiveController.on_verify_complete``). They default to the live
        width and to no bucket for callers with no per-result metadata.
        """
        if not num_correct_drafts_per_req:
            return False
        S = self.current_steps if steps is None else steps
        if S > 0:
            # Staleness guard, per the width that produced the sample: a count
            # above the chain that was drafted cannot come from it.
            fresh = [n for n in num_correct_drafts_per_req if n <= S]
            b = self._paired_bucket(forward_id)
            if fresh:
                a = self.alpha
                for c in self.candidate_steps:
                    if c > S:
                        continue  # censored at the chain this batch drafted
                    v = sum(min(x, c) for x in fresh) / len(fresh)
                    if b is not None:
                        self._m[b][c] = self._ema(self._m[b][c], self._mn[b][c], v, a)
                        self._mn[b][c] += 1
                    self._pm[c] = self._ema(self._pm[c], self._pmn[c], v, a)
                    self._pmn[c] += 1
                if S in self._psat:  # a width this policy can actually run
                    sat = sum(x >= S for x in fresh) / len(fresh)
                    satp = sum(x >= S - 1 for x in fresh) / len(fresh)
                    if b is not None:
                        self._sat[b][S] = self._ema(
                            self._sat[b][S], self._satn[b][S], sat, a
                        )
                        self._satp[b][S] = self._ema(
                            self._satp[b][S], self._satn[b][S], satp, a
                        )
                        self._satn[b][S] += 1
                    self._psat[S] = self._ema(self._psat[S], self._psatn[S], sat, a)
                    self._psatp[S] = self._ema(self._psatp[S], self._psatn[S], satp, a)
                    self._psatn[S] += 1
                    if b is not None:
                        aw = self.weight_alpha
                        for k in range(self._nb):
                            self._w[k] = (1 - aw) * self._w[k] + (aw if k == b else 0.0)

        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            return False
        if self._batch_count < self._grace_until:
            return False
        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False
        return self._recompute()

    # -- estimates ---------------------------------------------------------
    def _extrapolate(self, base: float, sat: float, satp: float, k: int) -> float:
        """Extend a chain by *k* positions past a boundary with survival *sat*."""
        if sat <= 0.0 or k <= 0:
            return base
        r = min(0.999, sat / satp) if satp > 0 else 0.9
        tail = k if r >= 1.0 else r * (1.0 - r**k) / (1.0 - r)
        return base + self.tail_bias * sat * tail

    def _pooled_estimate(self, c: int) -> float:
        S = self.current_steps
        if c <= S:
            return self._pm[c] if self._pmn[c] else float(max(0, c - 1))
        if not self._pmn.get(S) or not self._psatn.get(S):
            return float(max(0, S - 1))
        return self._extrapolate(self._pm[S], self._psat[S], self._psatp[S], c - S)

    def _bucket_estimate(self, b: int, c: int) -> float:
        pooled = self._pooled_estimate(c)
        S = self.current_steps
        n = self._mn[b].get(S, 0)
        if self.confidence_weight <= 0.0 or n == 0:
            return pooled
        if c <= S:
            est = self._m[b][c] if self._mn[b][c] else pooled
        elif self._satn[b].get(S):
            est = self._extrapolate(
                self._m[b][S], self._sat[b][S], self._satp[b][S], c - S
            )
        else:
            est = pooled
        # Shrink toward the pooled estimate until the bucket has enough samples.
        w = self.confidence_weight * min(1.0, n / max(1, self.min_bucket_samples))
        return w * est + (1 - w) * pooled

    def expected_accept_for(self, steps: int) -> float:
        """E[accepted] at *steps*, integrated over the confidence mixture."""
        if self.confidence_weight <= 0.0:
            return self._pooled_estimate(steps)
        tot = sum(self._w) or 1.0
        return sum(
            (self._w[b] / tot) * self._bucket_estimate(b, steps)
            for b in range(self._nb)
            if self._w[b] > 1e-4
        )

    def value(self, steps: int) -> float:
        return (1.0 + self.expected_accept_for(steps)) / step_time_ms(steps)

    def best_steps(self) -> tuple[int, float]:
        best, best_tps = self.current_steps, -1.0
        next_up = (
            next((s for s in self.candidate_steps if s > self.current_steps), None)
            if self.adjacent_only_promotion
            else None
        )
        for s in self.candidate_steps:
            if self.adjacent_only_promotion and s > self.current_steps and s != next_up:
                continue
            tps = self.value(s)
            if s == self.current_steps:
                # Hysteresis: a challenger must clear the incumbent, which pays
                # for the cold draft state a switch leaves behind.
                tps *= 1.0 + self.switch_margin
            elif s > self.current_steps:
                # ... and clear it by more when its own estimate is the
                # extrapolation rather than an exact observation.
                tps /= 1.0 + self.up_margin
            elif self.down_margin is not None:
                margin = (
                    self.down_margin.get(s)
                    if isinstance(self.down_margin, dict)
                    else self.down_margin
                )
                if margin is not None:
                    # Compare against the same incumbent score while requiring
                    # only (1 + down_margin) for this exact, downward estimate.
                    tps *= (1.0 + self.switch_margin) / (1.0 + margin)
            if tps > best_tps:
                best, best_tps = s, tps
        return best, best_tps

    def _recompute(self) -> bool:
        if not self._pmn.get(self.current_steps):
            return False
        target, _ = self.best_steps()
        if self._dbg:
            logger.info(
                "[adaptive-conf] steps=%d batch=%d conf=%s w=%s E=%s tps=%s -> %d",
                self.current_steps,
                self._batch_count,
                f"{self._last_conf:.3f}" if self._last_conf is not None else "-",
                [round(x, 3) for x in self._w],
                {
                    c: round(self.expected_accept_for(c), 2)
                    for c in self.candidate_steps
                },
                {c: round(self.value(c) * 1000) for c in self.candidate_steps},
                target,
            )
        if target == self.current_steps:
            return False
        old = self.current_steps
        e_old = self.expected_accept_for(old)
        e_new = self.expected_accept_for(target)
        v_old = self.value(old) * 1000
        v_new = self.value(target) * 1000
        direction = 1 if target > old else -1
        if direction == -self._last_dir:
            self._grace = min(
                self.max_grace_batches, max(1.0, self._grace) * self.grace_backoff
            )
        else:
            self._grace = float(self.switch_grace_batches)
        self._last_dir = direction
        self.current_steps = target
        self._grace_until = self._batch_count + int(self._grace)
        logger.info(
            "Adaptive spec params updated (confidence): steps %d -> %d "
            "(E@%d=%.2f -> %.0f tok/s, E@%d=%.2f -> %.0f tok/s, conf=%s)",
            old,
            target,
            old,
            e_old,
            v_old,
            target,
            e_new,
            v_new,
            f"{self._last_conf:.3f}" if self._last_conf is not None else "-",
        )
        return True
