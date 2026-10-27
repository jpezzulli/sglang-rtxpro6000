"""Finished requests leave the running batch before admission on every step.

Port of sgl-project/sglang#44351 (upstream head 5c5454a20255604c26706f631cd30ba11d187069,
merged as 3b5ae4ea86f6f791e018919b27f1008e470e85e1). Batches are fakes, the scheduler
method under test is the real Scheduler.get_next_batch_to_run.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import NextBatchPlan
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Req:
    def __init__(self):
        self.done = False

    def finished(self):
        return self.done


class _Batch:
    """Stand-in for ScheduleBatch with the surface get_next_batch_to_run touches."""

    def __init__(self, reqs, extend, batch_is_full=False, prefill_only=False):
        self.reqs = list(reqs)
        self.forward_mode = SimpleNamespace(is_extend=lambda: extend)
        self.chunked_req = None
        self.is_prefill_only = prefill_only
        self.batch_is_full = batch_is_full

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, chunked_req_to_exclude=None, keep_indices=None):
        exclude = chunked_req_to_exclude or []
        self.reqs = [r for r in self.reqs if not r.finished() and r not in exclude]

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)


class TestRunningBatchFinished(unittest.TestCase):
    def _scheduler(self, seen, admission=None):
        s = Scheduler.__new__(Scheduler)
        s.disaggregation_mode = DisaggregationMode.NULL
        s.process_pending_chunked_abort = Mock()
        s._abort_on_waiting_timeout = Mock()
        s._abort_on_running_timeout = Mock()
        s.enable_fpm = False
        s.dllm_config = None
        s.dllm_manager = None
        s.chunked_req = None
        s.enable_hisparse = False
        s.require_mlp_sync = False
        s._should_defer_prefill = Mock(return_value=False)
        s.dp_attn_adapter = Mock()
        s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = lambda b, **_: b
        s.update_running_batch = Mock(side_effect=lambda b: b)
        s._arm_prefill_decode_interval = Mock()
        s.ngram_embedding_manager = Mock()
        s.ngram_embedding_manager.prepare_for_forward.side_effect = lambda b, **_: b

        if admission is None:

            def admission(running_batch):
                seen.append((list(running_batch.reqs), running_batch.batch_is_full))
                # Chunked prefill keeps going: every step runs a prefill, never a decode.
                prefill = _Batch([], extend=True)
                return NextBatchPlan(batch_to_run=prefill, running_batch=running_batch)

        s.get_new_batch_prefill = admission
        return s

    def test_request_finished_at_prefill_leaves_before_next_admission(self):
        seen = []
        s = self._scheduler(seen)
        live, short = _Req(), _Req()
        running = _Batch([live], extend=False, batch_is_full=True)

        # Overlap: the prefill that runs `short` merges before its result
        # marks `short` finished. Nothing has finished yet, so the
        # batch_is_full latch must survive this step.
        plan = s.get_next_batch_to_run(
            running_batch=running, last_batch=_Batch([short], extend=True)
        )
        self.assertEqual(seen[-1], ([live, short], True))

        short.done = True
        s.get_next_batch_to_run(
            running_batch=plan.running_batch, last_batch=plan.batch_to_run
        )
        reqs, batch_is_full = seen[-1]
        self.assertEqual(reqs, [live])
        self.assertFalse(batch_is_full)
        s.update_running_batch.assert_not_called()

    def test_empty_prefill_only_batch_reopens_admission_when_a_row_frees(self):
        """Capacity 0 -> 1 with an empty prefill-only running batch.

        A prefill-only request held the last req_to_token row (streaming session,
        retracted chunk); its row is freed later. An empty running batch never
        reaches update_running_batch, the only other place that clears the latch,
        so admission must be re-consulted every step like it is on base: otherwise
        the flag stays shut after the row is gone and the waiting request starves.
        """
        seen = []
        free_rows = [0]

        def admission(running_batch):
            # Mirrors _get_new_batch_prefill_raw's two gates: a latched batch is
            # not retried at all, and with no allocatable row it re-latches.
            if running_batch.batch_is_full:
                seen.append("shut")
            elif free_rows[0] <= 0:
                seen.append("blocked_no_row")
                running_batch.batch_is_full = True
            else:
                seen.append("admitted")
            return NextBatchPlan(batch_to_run=None, running_batch=running_batch)

        s = self._scheduler(seen, admission=admission)
        running = _Batch([], extend=False, batch_is_full=True, prefill_only=True)
        plan = s.get_next_batch_to_run(running_batch=running, last_batch=None)
        self.assertEqual(seen[-1], "blocked_no_row")

        # A session row is freed / TTL reaped: nothing else changed, so the next
        # step must try admission again and take the waiting request.
        free_rows[0] = 1
        s.get_next_batch_to_run(
            running_batch=plan.running_batch, last_batch=plan.batch_to_run
        )
        self.assertEqual(seen[-1], "admitted")

    def test_empty_running_batch_reopens_admission_when_a_row_frees(self):
        """Same release, ordinary (non-prefill-only) empty running batch.

        The rows that latched admission can be handed back from outside the batch:
        closing an idle streaming session runs Scheduler.close_session ->
        StreamingSession.release_session, which frees the req_to_token row and the
        held KV and touches no latch (streaming_session.py:439-472). Nothing
        shrinks the already-empty running batch, so a latch that only clears on a
        shrink or on an empty *prefill-only* batch starves the waiting queue
        forever; an empty batch has no request left to leave either way.
        """
        seen = []
        free_rows = [0]

        def admission(running_batch):
            if running_batch.batch_is_full:
                seen.append("shut")
            elif free_rows[0] <= 0:
                seen.append("blocked_no_row")
                running_batch.batch_is_full = True
            else:
                seen.append("admitted")
            return NextBatchPlan(batch_to_run=None, running_batch=running_batch)

        s = self._scheduler(seen, admission=admission)
        running = _Batch([], extend=False, batch_is_full=True)
        plan = s.get_next_batch_to_run(running_batch=running, last_batch=None)
        self.assertEqual(seen[-1], "blocked_no_row")

        # A session row is freed: nothing else changed, so the next step must try
        # admission again and take the waiting request.
        free_rows[0] = 1
        s.get_next_batch_to_run(
            running_batch=plan.running_batch, last_batch=plan.batch_to_run
        )
        self.assertEqual(seen[-1], "admitted")


if __name__ == "__main__":
    unittest.main()
