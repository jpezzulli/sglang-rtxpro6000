"""Deferred chunked-prefill aborts must follow requests between queues."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeReq:
    def __init__(self, rid):
        self.rid = rid
        self.req_pool_idx = 1
        self.to_finish = None
        self._finished = False

    def finished(self):
        return self._finished


def _make_scheduler(req, running_reqs):
    sched = Scheduler.__new__(Scheduler)
    sched.chunked_req = None
    sched._pending_chunked_abort_req = req
    sched.waiting_queue = []
    sched.dllm_config = None
    sched.grammar_manager = Mock()
    sched.disaggregation_mode = None
    sched.enable_hicache_storage = False
    sched.mm_receiver = None
    sched.ps = SimpleNamespace(pp_size=1)
    sched.running_batch = SimpleNamespace(reqs=running_reqs)
    sched.last_batch = None
    return sched


class TestPendingChunkedAbortRace(CustomTestCase):
    def test_req_left_chunked_slot_is_aborted(self):
        req = _FakeReq("zombie")
        sched = _make_scheduler(req, [req])
        sched.process_pending_chunked_abort()
        self.assertIsNotNone(req.to_finish)
        self.assertIsNone(sched._pending_chunked_abort_req)

    def test_finished_or_released_req_only_clears_marker(self):
        for finished in (False, True):
            with self.subTest(finished=finished):
                req = _FakeReq("done")
                req._finished = finished
                req.req_pool_idx = 1 if finished else None
                sched = _make_scheduler(req, [])
                sched.abort_request = Mock()
                sched.process_pending_chunked_abort()
                sched.abort_request.assert_not_called()
                self.assertIsNone(sched._pending_chunked_abort_req)


if __name__ == "__main__":
    unittest.main()
