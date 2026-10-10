import logging
import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import compute_spec_context_reserve
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerInitReqMaxNewTokens(unittest.TestCase):
    """Property tests for Scheduler.init_req_max_new_tokens.

    Rules enforced when clipping a request's max_new_tokens:
      1. context: input_len + max_new_tokens + spec_context_reserve < max_req_len
      2. admission budget (PrefillAdder):
         ceil_page(input_len) + max_new_tokens + page_size < max_total_num_tokens
      3. env limit: <= SGLANG_MAX_NEW_TOKENS_LIMIT when set and positive
      4. never above the requested value
      5. min_new_tokens <= max_new_tokens afterwards

    Each case asserts all rules hold and the result is tight: one more token
    would violate a rule or exceed the request. Over-long inputs degenerate to
    max_new_tokens = 0 and are rejected by later admission checks.
    """

    @classmethod
    def setUpClass(cls):
        # Silence the per-request capping warning; the sweep triggers it a lot.
        cls._scheduler_logger = logging.getLogger("sglang.srt.managers.scheduler")
        cls._old_level = cls._scheduler_logger.level
        cls._scheduler_logger.setLevel(logging.ERROR)

    @classmethod
    def tearDownClass(cls):
        cls._scheduler_logger.setLevel(cls._old_level)

    def setUp(self):
        # The scheduler scales the budget by the live DCP size
        # (`get_parallel().attn_dcp_size`), so the double states a topology
        # rather than publishing a config it does not otherwise need.
        cm = get_parallel().override(attn_dcp_size=1)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    def _new_scheduler(
        self,
        max_req_len: int = 128,
        max_total_num_tokens: int = 1024,
        page_size: int = 1,
        spec_context_reserve: int = 0,
    ) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_req_len = max_req_len
        scheduler.spec_context_reserve = spec_context_reserve
        scheduler.max_total_num_tokens = max_total_num_tokens
        scheduler.page_size = page_size
        scheduler.max_new_tokens_limit = envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get()
        return scheduler

    def _new_req(self, max_new_tokens, input_len: int = 8, min_new_tokens: int = 0):
        return SimpleNamespace(
            rid="test-req",
            origin_input_ids=[0] * input_len,
            sampling_params=SimpleNamespace(
                max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens
            ),
        )

    def _init_and_check(self, scheduler, req) -> int:
        """Run init_req_max_new_tokens, then assert all admission rules hold
        and the result is tight. Returns the resulting max_new_tokens."""
        requested = req.sampling_params.max_new_tokens
        scheduler.init_req_max_new_tokens(req)
        max_new_tokens = req.sampling_params.max_new_tokens

        input_len = len(req.origin_input_ids)
        page_size = scheduler.page_size
        paged_input_len = -(-input_len // page_size) * page_size
        limit = scheduler.max_new_tokens_limit
        limit_active = limit is not None and limit > 0

        def satisfies_rules(candidate: int) -> bool:
            context_ok = (
                input_len + candidate + scheduler.spec_context_reserve
                < scheduler.max_req_len
            )
            budget_ok = (
                paged_input_len + candidate + page_size < scheduler.max_total_num_tokens
            )
            limit_ok = not limit_active or candidate <= limit
            requested_ok = requested is None or candidate <= requested
            return context_ok and budget_ok and limit_ok and requested_ok

        self.assertGreaterEqual(max_new_tokens, 0)
        if max_new_tokens > 0:
            self.assertTrue(satisfies_rules(max_new_tokens))
        self.assertFalse(satisfies_rules(max_new_tokens + 1))
        self.assertLessEqual(req.sampling_params.min_new_tokens, max_new_tokens)
        return max_new_tokens

    def test_limit_disabled_by_default(self):
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(None):
            scheduler = self._new_scheduler()
            req = self._new_req(max_new_tokens=64)
            self.assertEqual(self._init_and_check(scheduler, req), 64)

    def test_limit_clips_explicit_request(self):
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(16):
            scheduler = self._new_scheduler()
            req = self._new_req(max_new_tokens=64)
            self.assertEqual(self._init_and_check(scheduler, req), 16)

    def test_limit_applies_when_request_unset(self):
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(16):
            scheduler = self._new_scheduler()
            req = self._new_req(max_new_tokens=None)
            self.assertEqual(self._init_and_check(scheduler, req), 16)

    def test_non_positive_limit_is_ignored(self):
        for limit in (0, -1):
            with self.subTest(limit=limit):
                with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(limit):
                    scheduler = self._new_scheduler()
                    req = self._new_req(max_new_tokens=64)
                    self.assertEqual(self._init_and_check(scheduler, req), 64)

    def test_context_rule_binds_tighter_than_limit(self):
        max_req_len, input_len = 32, 20
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(16):
            scheduler = self._new_scheduler(max_req_len=max_req_len)
            req = self._new_req(max_new_tokens=64, input_len=input_len)
            self.assertEqual(
                self._init_and_check(scheduler, req), max_req_len - input_len - 1
            )

    def test_budget_rule_binds_tighter_than_limit(self):
        max_total_num_tokens, page_size, input_len = 24, 4, 8
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(32):
            scheduler = self._new_scheduler(
                max_total_num_tokens=max_total_num_tokens, page_size=page_size
            )
            req = self._new_req(max_new_tokens=64, input_len=input_len)
            paged_input_len = -(-input_len // page_size) * page_size
            self.assertEqual(
                self._init_and_check(scheduler, req),
                max_total_num_tokens - paged_input_len - page_size - 1,
            )

    def test_min_new_tokens_clamped_to_limit(self):
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(16):
            scheduler = self._new_scheduler()
            req = self._new_req(max_new_tokens=64, min_new_tokens=32)
            self.assertEqual(self._init_and_check(scheduler, req), 16)
            self.assertEqual(req.sampling_params.min_new_tokens, 16)

    def test_admission_rules_sweep(self):
        for page_size in (1, 4, 16):
            for input_len in (1, 8, 100):
                for requested in (None, 0, 5, 64, 1 << 20):
                    for limit in (None, 0, 16, 1 << 20):
                        for max_req_len, max_total_num_tokens in (
                            (128, 1024),
                            (32, 24),
                            (128, 24),
                        ):
                            for spec_context_reserve in (0, 8):
                                with self.subTest(
                                    page_size=page_size,
                                    input_len=input_len,
                                    requested=requested,
                                    limit=limit,
                                    max_req_len=max_req_len,
                                    max_total_num_tokens=max_total_num_tokens,
                                    spec_context_reserve=spec_context_reserve,
                                ):
                                    with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(
                                        limit
                                    ):
                                        scheduler = self._new_scheduler(
                                            max_req_len=max_req_len,
                                            max_total_num_tokens=max_total_num_tokens,
                                            page_size=page_size,
                                            spec_context_reserve=spec_context_reserve,
                                        )
                                        req = self._new_req(
                                            max_new_tokens=requested,
                                            input_len=input_len,
                                        )
                                        self._init_and_check(scheduler, req)

    # EAGLE chain: 3 draft steps, topk 1, 4 verify tokens.
    def test_no_spec_keeps_full_context(self):
        with get_context().override_server_args(speculative_algorithm=None):
            self.assertEqual(compute_spec_context_reserve(enable_overlap=True), 0)

    def test_dflash_is_never_reserved(self):
        """Only the EAGLE family stores drafts in output slots; DFlash (T8 here)
        keeps the whole context."""
        with get_context().override_server_args(
            speculative_algorithm="DFLASH",
            speculative_num_steps=8,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=8,
        ):
            for enable_overlap in (True, False):
                with self.subTest(enable_overlap=enable_overlap):
                    self.assertEqual(
                        compute_spec_context_reserve(enable_overlap=enable_overlap), 0
                    )

    def test_spec_reserve_doubles_only_under_overlap(self):
        for num_steps, num_draft_tokens in ((3, 4), (5, 6)):
            with get_context().override_server_args(
                speculative_algorithm="EAGLE",
                speculative_num_steps=num_steps,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=num_draft_tokens,
            ):
                with self.subTest(num_draft_tokens=num_draft_tokens):
                    self.assertEqual(
                        compute_spec_context_reserve(enable_overlap=True),
                        2 * num_draft_tokens,
                    )
                    self.assertEqual(
                        compute_spec_context_reserve(enable_overlap=False),
                        num_draft_tokens,
                    )

    def test_spec_overlap_tail_step_stays_within_context(self):
        """A request run to the length cap under EAGLE + overlap: the tail step's
        accept-then-verify KV length must not exceed context_len."""
        context_len, input_len = 4096, 25
        max_req_len = context_len - 1  # TpModelWorker.get_worker_info

        def tail_kv_len(max_new_tokens, num_draft_tokens):
            # The last unfinished step accepts up to num_draft_tokens (bonus
            # included, its KV not yet written); the tail step verifies that many.
            return input_len + max_new_tokens + 2 * num_draft_tokens - 2

        # 6 draft tokens: reserving num_draft_tokens once is not enough there.
        for num_steps, num_draft_tokens in ((3, 4), (5, 6)):
            with get_context().override_server_args(
                speculative_algorithm="EAGLE",
                speculative_num_steps=num_steps,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=num_draft_tokens,
            ):
                reserve = compute_spec_context_reserve(enable_overlap=True)
            for requested in (None, 1 << 20, context_len - 4 - input_len):
                with self.subTest(
                    num_draft_tokens=num_draft_tokens, requested=requested
                ):
                    scheduler = self._new_scheduler(
                        max_req_len=max_req_len,
                        max_total_num_tokens=1 << 20,
                        spec_context_reserve=reserve,
                    )
                    req = self._new_req(max_new_tokens=requested, input_len=input_len)
                    max_new_tokens = self._init_and_check(scheduler, req)
                    self.assertLessEqual(
                        tail_kv_len(max_new_tokens, num_draft_tokens), context_len
                    )

        # Without the reserve the tail step overruns the context (the bug).
        scheduler = self._new_scheduler(
            max_req_len=max_req_len, max_total_num_tokens=1 << 20
        )
        req = self._new_req(max_new_tokens=None, input_len=input_len)
        self.assertGreater(
            tail_kv_len(self._init_and_check(scheduler, req), 4), context_len
        )

    def test_release_profile_cap_keeps_lookahead_in_512k_context(self):
        """The 512K-context release profile (page 64, TP1, pool larger than the
        context) at the cap with EAGLE 3/1/4: 8 slots stay free under overlap."""
        context_len, page_size, input_len, num_draft_tokens = 524288, 64, 2047, 4
        with get_context().override_server_args(
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=num_draft_tokens,
        ):
            reserve = compute_spec_context_reserve(enable_overlap=True)
        self.assertEqual(reserve, 8)
        for requested in (None, 1 << 30):
            with self.subTest(requested=requested):
                scheduler = self._new_scheduler(
                    max_req_len=context_len - 1,
                    max_total_num_tokens=1 << 21,
                    page_size=page_size,
                    spec_context_reserve=reserve,
                )
                req = self._new_req(max_new_tokens=requested, input_len=input_len)
                max_new_tokens = self._init_and_check(scheduler, req)
                self.assertEqual(
                    max_new_tokens, context_len - 1 - input_len - 1 - reserve
                )
                self.assertLessEqual(
                    input_len + max_new_tokens + 2 * num_draft_tokens - 2, context_len
                )

    def test_extreme_boundary_never_hands_out_a_negative_allowance(self):
        """Once the reserve consumes the whole tail the request is clamped to 0
        output tokens, never to a negative allowance, and min_new_tokens follows.
        """
        context_len, page_size, reserve = 4096, 64, 8
        for input_len in (context_len - 3, context_len - 2, context_len + 8):
            with self.subTest(input_len=input_len):
                scheduler = self._new_scheduler(
                    max_req_len=context_len - 1,
                    max_total_num_tokens=1 << 21,
                    page_size=page_size,
                    spec_context_reserve=reserve,
                )
                req = self._new_req(
                    max_new_tokens=None, input_len=input_len, min_new_tokens=8
                )
                self.assertEqual(self._init_and_check(scheduler, req), 0)
                self.assertEqual(req.sampling_params.min_new_tokens, 0)


if __name__ == "__main__":
    unittest.main()
