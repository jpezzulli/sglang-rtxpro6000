"""
Unit tests for rid_to_state cleanup in TokenizerManager.

Verifies that request IDs are properly removed from rid_to_state after
completion or abort, allowing resubmission with the same rid without
triggering "Duplicate request ID detected" errors.

Covers:
  - _handle_abort_req cleans up rid_to_state
  - _handle_batch_output cleans up rid_to_state on finished requests
  - _init_req_state rejects duplicate rids
  - Resubmission succeeds after cleanup
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import msgspec

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    AbortReq,
    BatchStrOutput,
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    ReqState,
    TokenizerManager,
)
from sglang.srt.observability.req_time_stats import (  # noqa: E402
    APIServerReqTimeStats,
)
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


_NOT_FINISHED = object()  # Sentinel: request has not finished yet

# ---------------------------------------------------------------------------
# Per-request field defaults for BatchStrOutput construction.
# Categorised by value shape so that _make_batch_str_output can assign
# type-appropriate defaults without hardcoding every field name.
# When a field is renamed upstream, the old name simply won't appear in
# msgspec.structs.fields() and the new name will fall through to the
# pattern-matching or safe fallback — no test breakage.
# ---------------------------------------------------------------------------

_PER_REQUEST_INT_FIELDS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "retraction_counts",
        # Speculative-decoding int-scalar fields (current and historical names)
        "spec_verify_ct",
        "spec_accepted_drafts",
        "spec_num_correct_drafts",
    }
)

_PER_REQUEST_FLOAT_FIELDS = frozenset(
    {
        "output_token_entropy_val",
    }
)

_PER_REQUEST_NESTED_LIST_FIELDS = frozenset(
    {
        "output_ids",
        # Logprob fields
        "input_token_logprobs_val",
        "input_token_logprobs_idx",
        "output_token_logprobs_val",
        "output_token_logprobs_idx",
        "input_top_logprobs_val",
        "input_top_logprobs_idx",
        "output_top_logprobs_val",
        "output_top_logprobs_idx",
        "input_token_ids_logprobs_val",
        "input_token_ids_logprobs_idx",
        "output_token_ids_logprobs_val",
        "output_token_ids_logprobs_idx",
        # Speculative-decoding histogram fields (current and historical names)
        "spec_acceptance_histogram",
        "spec_correct_drafts_histogram",
    }
)

_PER_REQUEST_OPTIONAL_FIELDS = frozenset(
    {
        "output_hidden_states",
        "routed_experts",
        "indexer_topk",
        "placeholder_tokens_idx",
        "placeholder_tokens_val",
    }
)


def _make_tokenizer_manager(case) -> TokenizerManager:
    """Create a TokenizerManager with mocked dependencies, bypassing __init__.

    The config it reads comes from the bags, so the stand-in needs a published
    config rather than attributes on a mock.
    """
    override = get_context().override_server_args(speculative_algorithm=None)
    override.install()
    case.addCleanup(override.restore)
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm._config_updates = []
    tm.server_args.enable_trace = False
    tm.server_args.enable_metrics = False
    tm.server_args.enable_lora = False
    tm.server_args.speculative_algorithm = None
    tm.server_args.incremental_streaming_output = False
    tm.server_args.skip_tokenizer_init = False
    tm.server_args.batch_notify_size = 1
    tm.server_args.weight_version = "1"
    tm.server_args.crash_dump_folder = ""
    tm.server_args.dp_size = 1
    tm.disaggregation_mode = "none"
    tm.rid_to_state = {}
    tm.enable_metrics = False
    tm.enable_trace = False
    tm.enable_lora = False
    tm.incremental_streaming_output = False
    tm.allow_auto_truncate = False
    tm.skip_tokenizer_init = False
    tm.dump_requests_folder = ""
    tm.crash_dump_folder = ""
    tm.send_to_scheduler = MagicMock()
    return tm


def _make_req_state(rid: str = "test_rid") -> ReqState:
    """Create a minimal ReqState for testing."""
    obj = Mock(spec=GenerateReqInput)
    obj.rid = rid
    obj.stream = False
    obj.return_logprob = False
    obj.lora_path = None
    obj.log_metrics = False
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=APIServerReqTimeStats(),
    )


def _make_abort_req(rid: str, abort_message: str = "Aborted") -> AbortReq:
    """Create an AbortReq for testing."""
    return AbortReq(
        rid=rid,
        abort_all=False,
        finished_reason={"type": "abort", "message": abort_message},
        abort_message=abort_message,
    )


def _make_batch_str_output(rid: str, finished_reason=None) -> BatchStrOutput:
    """Create a minimal BatchStrOutput for a single request.

    Uses struct field introspection so that new or renamed fields in
    BatchStrOutput don't break this test.  Only the fields that matter for
    test logic (rids, finished_reasons, output_strs) are set explicitly;
    all others receive type-appropriate defaults based on naming patterns.
    Fields with class-level defaults are left alone automatically.
    """
    if finished_reason is _NOT_FINISHED:
        fr = None
    elif finished_reason is None:
        fr = {"type": "length"}
    else:
        fr = finished_reason

    kwargs = {}
    for f in msgspec.structs.fields(BatchStrOutput):
        if f.name == "rids":
            kwargs[f.name] = [rid]
        elif f.name == "finished_reasons":
            kwargs[f.name] = [fr]
        elif f.name == "output_strs":
            kwargs[f.name] = ["hello"]
        elif f.name in _PER_REQUEST_INT_FIELDS:
            kwargs[f.name] = [0]
        elif f.name in _PER_REQUEST_FLOAT_FIELDS:
            kwargs[f.name] = [0.0]
        elif f.name in _PER_REQUEST_NESTED_LIST_FIELDS:
            kwargs[f.name] = [[]]
        elif f.name in _PER_REQUEST_OPTIONAL_FIELDS:
            kwargs[f.name] = [None]
        # Fields with class defaults — skip, let the default be used
        elif (
            f.default is not msgspec.NODEFAULT
            or f.default_factory is not msgspec.NODEFAULT
        ):
            continue
        # Unknown required field — provide a safe per-request default.
        # Most BatchStrOutput fields are per-request lists; [[]] works for
        # List[List[...]] and is unlikely to crash on [i] indexing for
        # List[int] either (the inner [] just means "no data").
        else:
            kwargs[f.name] = [[]]

    return BatchStrOutput(**kwargs)


class TestRidToStateCleanupOnAbort(CustomTestCase):
    """Test that _handle_abort_req removes rid from rid_to_state."""

    def test_abort_removes_rid_from_state(self):
        """After _handle_abort_req, rid should be removed from rid_to_state."""
        tm = _make_tokenizer_manager(self)
        rid = "abort_test_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        self.assertNotIn(rid, tm.rid_to_state)

    def test_abort_allows_resubmit_same_rid(self):
        """After abort, _init_req_state should accept the same rid again."""
        tm = _make_tokenizer_manager(self)
        rid = "resubmit_after_abort_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        # Resubmit with the same rid — should not raise
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)

    def test_abort_sets_finished_and_notifies(self):
        """_handle_abort_req should mark state as finished and set the event."""
        tm = _make_tokenizer_manager(self)
        rid = "abort_notify_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        self.assertTrue(state.finished)
        self.assertTrue(state.event.is_set())
        self.assertEqual(len(state.out_list), 1)
        self.assertEqual(
            state.out_list[0]["meta_info"]["finish_reason"]["type"], "abort"
        )


class TestRidToStateCleanupOnBatchOutput(CustomTestCase):
    """Test that _handle_batch_output removes rid from rid_to_state on completion."""

    def test_batch_output_removes_rid_on_finish(self):
        """When a request finishes in _handle_batch_output, rid should be removed."""
        tm = _make_tokenizer_manager(self)
        rid = "batch_finish_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        batch_output = _make_batch_str_output(rid)
        asyncio.run(tm._handle_batch_output(batch_output))

        self.assertNotIn(rid, tm.rid_to_state)

    def test_batch_output_allows_resubmit_after_finish(self):
        """After a request finishes, the same rid can be resubmitted."""
        tm = _make_tokenizer_manager(self)
        rid = "batch_resubmit_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        batch_output = _make_batch_str_output(rid)
        asyncio.run(tm._handle_batch_output(batch_output))

        # Resubmit with the same rid — should not raise
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)

    def test_batch_output_keeps_rid_when_not_finished(self):
        """When a request is not yet finished, rid should remain in rid_to_state."""
        tm = _make_tokenizer_manager(self)
        rid = "batch_ongoing_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        # finished_reason=_NOT_FINISHED means the request is still ongoing
        batch_output = _make_batch_str_output(rid, finished_reason=_NOT_FINISHED)
        asyncio.run(tm._handle_batch_output(batch_output))

        self.assertIn(rid, tm.rid_to_state)


class TestInitReqStateDuplicateDetection(CustomTestCase):
    """Test that _init_req_state raises ValueError for duplicate rids."""

    def test_duplicate_rid_raises_error(self):
        """_init_req_state should raise ValueError if rid already exists."""
        tm = _make_tokenizer_manager(self)
        rid = "duplicate_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None

        with self.assertRaises(ValueError) as ctx:
            tm._init_req_state(obj)
        self.assertIn("Duplicate request ID", str(ctx.exception))

    def test_unique_rid_succeeds(self):
        """_init_req_state should succeed with a unique rid."""
        tm = _make_tokenizer_manager(self)
        rid = "unique_rid"

        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None

        tm._init_req_state(obj)
        self.assertIn(rid, tm.rid_to_state)


class TestResubmitAfterCompletion(CustomTestCase):
    """End-to-end test: complete a request, then resubmit with the same rid."""

    def test_complete_then_resubmit_same_rid(self):
        """A request that completes normally should allow resubmission with the same rid."""
        tm = _make_tokenizer_manager(self)
        rid = "complete_resubmit_rid"

        # Phase 1: simulate a request in rid_to_state, then complete it
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        batch_output = _make_batch_str_output(rid, finished_reason={"type": "length"})
        asyncio.run(tm._handle_batch_output(batch_output))

        # rid should be cleaned up
        self.assertNotIn(rid, tm.rid_to_state)

        # Phase 2: resubmit with the same rid — should succeed
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)

    def test_abort_then_resubmit_same_rid(self):
        """An aborted request should allow resubmission with the same rid."""
        tm = _make_tokenizer_manager(self)
        rid = "abort_resubmit_rid"

        # Phase 1: simulate a request, then abort it
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        self.assertNotIn(rid, tm.rid_to_state)

        # Phase 2: resubmit with the same rid — should succeed
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)


class _DummyAsyncCM:
    """Reusable no-op async context manager (stands in for an RW lock)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _make_tm_for_generate(case) -> TokenizerManager:
    """Augment the mocked TokenizerManager with what generate_request needs."""
    tm = _make_tokenizer_manager(case)
    tm.server_args.language_only = False
    tm.server_args.tokenizer_worker_num = 1
    tm.server_args.enable_strict_thinking = False
    tm.auto_create_handle_loop = Mock()
    tm._set_default_priority = Mock()
    tm.request_logger = Mock()
    tm.tokenizer = None
    tm.is_pause = False
    tm.is_pause_cond = asyncio.Condition()
    tm.model_update_lock = Mock()
    tm.model_update_lock.reader_lock = _DummyAsyncCM()
    tm._validate_and_resolve_lora = AsyncMock(return_value=None)
    return tm


def _make_generate_obj(rid, is_single):
    obj = MagicMock(spec=GenerateReqInput)
    obj.routed_dp_rank = None
    obj.is_single = is_single
    obj.rid = rid
    obj.received_time = 0.0
    obj.external_trace_header = None
    obj.bootstrap_room = None
    obj.max_thinking_tokens = None
    obj.normalize_batch_and_arguments = Mock()
    if not is_single:
        obj.__getitem__.side_effect = lambda i: Mock()
    return obj


class TestReleaseReqStatesOnFailure(CustomTestCase):
    """Local undelivered state is dropped; scheduler-owned state is retained."""

    def test_discard_single(self):
        tm = _make_tokenizer_manager(self)
        rid = "d_single"
        tm.rid_to_state[rid] = _make_req_state(rid)
        tm._release_req_states_on_failure([rid])
        self.assertNotIn(rid, tm.rid_to_state)

    def test_discard_batch_removes_all(self):
        tm = _make_tokenizer_manager(self)
        rids = ["d0", "d1", "d2"]
        for r in rids:
            tm.rid_to_state[r] = _make_req_state(r)
        tm._release_req_states_on_failure(rids)
        for r in rids:
            self.assertNotIn(r, tm.rid_to_state)

    def test_discard_ignores_already_removed(self):
        """Popping a rid that is no longer present must not raise."""
        tm = _make_tokenizer_manager(self)
        tm.rid_to_state["p1"] = _make_req_state("p1")
        tm._release_req_states_on_failure(["p1", "already_gone"])
        self.assertNotIn("p1", tm.rid_to_state)

    def test_partial_batch_aborts_dispatched_and_drops_pending(self):
        for fail_abort in (False, True):
            with self.subTest(fail_abort=fail_abort):
                tm = _make_tokenizer_manager(self)
                tm._dispatch_to_scheduler = Mock()
                live = _make_req_state("live")
                live.dispatched = True
                tm.rid_to_state.update(live=live, pending=_make_req_state("pending"))
                if fail_abort:
                    tm._dispatch_to_scheduler.side_effect = RuntimeError("send failed")
                    with self.assertLogs(level="ERROR"):
                        tm._release_req_states_on_failure(["live", "pending", "gone"])
                else:
                    tm._release_req_states_on_failure(["live", "pending", "gone"])
                self.assertEqual(list(tm.rid_to_state), ["live"])
                self.assertEqual(live.abort_sent, not fail_abort)
                self.assertEqual(
                    tm._dispatch_to_scheduler.call_args.args[0].rid, "live"
                )


class TestParallelStreamTaskCleanup(CustomTestCase):
    def test_failing_choice_cancels_and_closes_sibling_waiters(self):
        tm = _make_tokenizer_manager(self)

        async def drive():
            sibling_closed = asyncio.Event()

            async def failing_choice():
                await asyncio.sleep(0)
                raise RuntimeError("choice failed")
                yield  # pragma: no cover

            async def blocked_choice():
                try:
                    await asyncio.Event().wait()
                    yield  # pragma: no cover
                finally:
                    sibling_closed.set()

            stream = tm._stream_batch_responses(
                [failing_choice(), blocked_choice()],
                ["choice-0", "choice-1"],
            )
            with self.assertRaisesRegex(RuntimeError, "choice failed"):
                await stream.__anext__()
            self.assertTrue(sibling_closed.is_set())

        asyncio.run(drive())

    def test_failing_non_stream_choice_cancels_and_closes_sibling_waiters(self):
        tm = _make_tokenizer_manager(self)

        async def drive():
            sibling_closed = asyncio.Event()

            async def failing_choice():
                await asyncio.sleep(0)
                raise RuntimeError("choice failed")
                yield  # pragma: no cover

            async def blocked_choice():
                try:
                    await asyncio.Event().wait()
                    yield  # pragma: no cover
                finally:
                    sibling_closed.set()

            with self.assertRaisesRegex(RuntimeError, "choice failed"):
                await tm._collect_batch_responses([failing_choice(), blocked_choice()])
            self.assertTrue(sibling_closed.is_set())

        asyncio.run(drive())


class TestGenerateRequestCleanupOnDispatchFailure(CustomTestCase):
    """generate_request must not leak rid_to_state when dispatch fails.

    Regression guard: _init_req_state creates rid_to_state entries up front,
    and the only remover is the scheduler-response path. A failure before the
    request reaches the scheduler (e.g. input-length validation rejecting an
    over-context request) used to leak those entries permanently.
    """

    def test_single_failure_before_dispatch_cleans_up(self):
        tm = _make_tm_for_generate(self)
        rid = "single_overlen"
        obj = _make_generate_obj(rid, is_single=True)
        # Simulate over-length rejection during tokenization/validation.
        tm._tokenize_one_request = AsyncMock(side_effect=ValueError("input too long"))
        tm._send_one_request = Mock()

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaises(ValueError):
            asyncio.run(drive())

        # Got past _init_req_state (which created the entry) ...
        tm._tokenize_one_request.assert_awaited_once()
        tm._send_one_request.assert_not_called()
        # ... and the entry was cleaned up rather than leaked.
        self.assertNotIn(rid, tm.rid_to_state)

    def test_batch_failure_before_dispatch_cleans_up_all(self):
        tm = _make_tm_for_generate(self)
        rids = ["b0", "b1", "b2"]
        obj = _make_generate_obj(list(rids), is_single=False)

        # One over-length sub-request makes the whole batch dispatch raise.
        async def _boom(*args, **kwargs):
            raise ValueError("input too long")
            yield  # pragma: no cover  (marks this an async generator)

        tm._handle_batch_request = _boom

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaises(ValueError):
            asyncio.run(drive())

        # All sub-request entries created by _init_req_state are cleaned up.
        for r in rids:
            self.assertNotIn(r, tm.rid_to_state)

    def test_thinking_budget_rejects_runtime_without_strict_thinking(self):
        tm = _make_tm_for_generate(self)
        obj = GenerateReqInput(
            text="hello",
            rid="thinking-budget",
            sampling_params={},
            max_thinking_tokens=32,
        )

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaisesRegex(ValueError, "--enable-strict-thinking"):
            asyncio.run(drive())

        self.assertFalse(tm.rid_to_state)


class TestDispatchedRequestCleanup(CustomTestCase):
    @patch(
        "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
        side_effect=lambda obj: obj,
    )
    def test_parallel_pending_abort_send_failure_retains_generated_state(self, _wrap):
        tm = _make_tm_for_generate(self)
        tm.cuda_vmm_feature_transport = Mock()
        tm.cuda_vmm_feature_transport.prepare_for_dispatch.return_value = ["published"]
        obj = GenerateReqInput(text=["hello"], rid=["parent"], sampling_params={"n": 2})
        generated, aborted = [], []
        failed = False

        def dispatch(request):
            nonlocal failed
            if isinstance(request, AbortReq):
                if not failed:
                    failed = True
                    raise RuntimeError("abort send failed")
                aborted.append(request.rid)
            else:
                generated.append(request.rid)

        tm._dispatch_to_scheduler = Mock(side_effect=dispatch)

        async def tokenize(request):
            tm.abort_request("parent")
            return MagicMock(
                rid=request.rid, mm_inputs=None, sampling_params=MagicMock()
            )

        tm._tokenize_one_request = tokenize

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaisesRegex(RuntimeError, "abort send failed"):
            asyncio.run(drive())
        self.assertEqual(len(generated), 1)
        self.assertEqual(aborted, generated)
        self.assertEqual(list(tm.rid_to_state), generated)
        state = tm.rid_to_state[generated[0]]
        self.assertTrue(state.dispatched)
        self.assertTrue(state.abort_sent)
        self.assertFalse(state.abort_pending)
        tm.cuda_vmm_feature_transport.cancel_for_dispatch.assert_not_called()
        tm._handle_abort_req(_make_abort_req(generated[0]))
        self.assertFalse(tm.rid_to_state)

    @patch(
        "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
        side_effect=lambda obj: obj,
    )
    def test_parallel_pending_abort_preserves_other_group(self, _wrap):
        tm = _make_tm_for_generate(self)
        tm.cuda_vmm_feature_transport = Mock()
        tm.cuda_vmm_feature_transport.prepare_for_dispatch.return_value = []
        tm._dispatch_to_scheduler = Mock()
        obj = GenerateReqInput(
            text=["hello", "world"],
            rid=["cancelled", "normal"],
            sampling_params={"n": 2},
        )
        parents = {}

        async def tokenize(request):
            parents[request.rid] = tm.rid_to_state[request.rid]
            if request.rid == "cancelled":
                tm.abort_request(request.rid)
            return MagicMock(
                rid=request.rid, mm_inputs=None, sampling_params=MagicMock()
            )

        async def response(request, raw_request):
            tm.rid_to_state.pop(request.rid)
            yield {"text": "", "meta_info": {"id": request.rid}}

        tm._tokenize_one_request = tokenize
        tm._wait_one_response = response

        async def drive():
            return [result async for result in tm.generate_request(obj)]

        results = asyncio.run(drive())
        self.assertEqual(len(results[0]), 4)
        aborted = [
            call.args[0].rid
            for call in tm._dispatch_to_scheduler.call_args_list
            if isinstance(call.args[0], AbortReq)
        ]
        self.assertEqual(aborted, parents["cancelled"].parallel_sample_rids)
        self.assertEqual(len(aborted), 3)
        self.assertTrue(set(aborted).isdisjoint(parents["normal"].parallel_sample_rids))
        self.assertFalse(tm.rid_to_state)

    @patch(
        "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
        side_effect=lambda obj: obj,
    )
    def test_parallel_pending_abort_follows_regenerated_request_ids(self, _wrap):
        for phase in ("tokenization", "prefix_wait", "prefix_complete"):
            with self.subTest(phase=phase):
                tm = _make_tm_for_generate(self)
                tm.cuda_vmm_feature_transport = Mock()
                tm.cuda_vmm_feature_transport.prepare_for_dispatch.return_value = []
                tm._dispatch_to_scheduler = Mock()
                obj = GenerateReqInput(
                    text=["hello"], rid=["parent"], sampling_params={"n": 2}
                )
                waited = []

                async def tokenize(request):
                    if phase == "tokenization":
                        tm.abort_request("parent")
                    return MagicMock(
                        rid=request.rid, mm_inputs=None, sampling_params=MagicMock()
                    )

                async def response(request, raw_request):
                    is_prefix = not waited
                    waited.append(request.rid)
                    if is_prefix and phase == "prefix_wait":
                        tm.abort_request("parent")
                    # Prefix completion can race with abort delivery. The
                    # group's cancellation must still reach later choices.
                    tm.rid_to_state.pop(request.rid)
                    if is_prefix and phase == "prefix_complete":
                        tm.abort_request("parent")
                    yield {"text": "", "meta_info": {"id": request.rid}}

                tm._tokenize_one_request = tokenize
                tm._wait_one_response = response

                async def drive():
                    return [result async for result in tm.generate_request(obj)]

                result = asyncio.run(drive())
                self.assertEqual(len(result[0]), 2)
                sent = [
                    call.args[0] for call in tm._dispatch_to_scheduler.call_args_list
                ]
                generated = [
                    message.rid for message in sent if not isinstance(message, AbortReq)
                ]
                aborted = [
                    message.rid for message in sent if isinstance(message, AbortReq)
                ]
                self.assertEqual(len(generated), 3)
                self.assertNotIn("parent", generated)
                self.assertEqual(
                    aborted, generated[1:] if phase == "prefix_complete" else generated
                )
                for rid in aborted:
                    self.assertLess(
                        next(
                            i
                            for i, message in enumerate(sent)
                            if not isinstance(message, AbortReq) and message.rid == rid
                        ),
                        next(
                            i
                            for i, message in enumerate(sent)
                            if isinstance(message, AbortReq) and message.rid == rid
                        ),
                    )
                self.assertFalse(tm.rid_to_state)

    @patch(
        "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
        side_effect=lambda obj: obj,
    )
    def test_abort_while_tokenizing_is_effective_after_dispatch(self, _wrap):
        tm = _make_tm_for_generate(self)
        tm.cuda_vmm_feature_transport = Mock()
        tm.cuda_vmm_feature_transport.prepare_for_dispatch.return_value = []
        tm._dispatch_to_scheduler = Mock()
        obj = _make_generate_obj("pending", is_single=True)
        obj.return_prompt_token_ids = False

        async def drive():
            tokenizing, resume = asyncio.Event(), asyncio.Event()

            async def tokenize(request):
                tokenizing.set()
                await resume.wait()
                return MagicMock(rid=request.rid, mm_inputs=None)

            tm._tokenize_one_request = tokenize
            task = asyncio.create_task(tm.generate_request(obj).__anext__())
            try:
                await tokenizing.wait()
                tm.abort_request(obj.rid)
                tm.abort_request(obj.rid)
                self.assertFalse(tm.rid_to_state[obj.rid].abort_sent)
                tm._dispatch_to_scheduler.assert_not_called()
                resume.set()
                for _ in range(100):
                    await asyncio.sleep(0)
                    if tm.rid_to_state[obj.rid].dispatched:
                        break
                self.assertTrue(tm.rid_to_state[obj.rid].dispatched)
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            sent = [call.args[0] for call in tm._dispatch_to_scheduler.call_args_list]
            self.assertEqual(len(sent), 2)
            self.assertNotIsInstance(sent[0], AbortReq)
            self.assertIsInstance(sent[1], AbortReq)
            self.assertEqual(sent[1].rid, obj.rid)
            self.assertIn(obj.rid, tm.rid_to_state)

        asyncio.run(drive())

    @patch(
        "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
        side_effect=lambda obj: obj,
    )
    def test_pending_abort_single_and_batch_dispatch_order_and_retry(self, _wrap):
        for batch in (False, True):
            for fail_abort in (False, True):
                with self.subTest(batch=batch, fail_abort=fail_abort):
                    tm = _make_tokenizer_manager(self)
                    tm.cuda_vmm_feature_transport = Mock()
                    tm.cuda_vmm_feature_transport.prepare_for_dispatch.return_value = [
                        "published"
                    ]
                    tm.enable_metrics = True
                    tm.metrics_collector = MagicMock()
                    rids = ["a", "b"] if batch else ["a"]
                    tokenized = [
                        MagicMock(
                            spec=TokenizedGenerateReqInput,
                            rid=rid,
                            mm_inputs=None,
                            time_stats=MagicMock(),
                        )
                        for rid in rids
                    ]
                    for rid in rids:
                        tm.rid_to_state[rid] = _make_req_state(rid)
                    delivered = []
                    failed = False

                    def dispatch(request):
                        nonlocal failed
                        if isinstance(request, AbortReq):
                            self.assertTrue(
                                all(tm.rid_to_state[rid].dispatched for rid in rids)
                            )
                            if fail_abort and not failed:
                                failed = True
                                raise RuntimeError("abort send failed")
                            delivered.append(("abort", request.rid))
                        else:
                            delivered.append(("generate", list(rids)))

                    tm._dispatch_to_scheduler = Mock(side_effect=dispatch)
                    for rid in rids:
                        tm.abort_request(rid)
                        tm.abort_request(rid)
                    tm._dispatch_to_scheduler.assert_not_called()
                    send = tm._send_batch_request if batch else tm._send_one_request
                    payload = tokenized if batch else tokenized[0]
                    if fail_abort:
                        with self.assertRaisesRegex(RuntimeError, "abort send failed"):
                            send(payload)
                        self.assertTrue(
                            all(tm.rid_to_state[rid].dispatched for rid in rids)
                        )
                        self.assertTrue(
                            all(not tm.rid_to_state[rid].abort_sent for rid in rids)
                        )
                        tm.metrics_collector.observe_one_aborted_request.assert_not_called()
                        tm._release_req_states_on_failure(rids)
                    else:
                        send(payload)
                    tm._release_req_states_on_failure(rids)
                    self.assertEqual(
                        delivered,
                        [("generate", rids)] + [("abort", rid) for rid in rids],
                    )
                    self.assertEqual(
                        tm.metrics_collector.observe_one_aborted_request.call_count,
                        len(rids),
                    )
                    self.assertTrue(
                        all(tm.rid_to_state[rid].abort_sent for rid in rids)
                    )
                    tm.cuda_vmm_feature_transport.cancel_for_dispatch.assert_not_called()

    @patch(
        "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
        side_effect=lambda obj: obj,
    )
    def test_pending_abort_failed_generation_send_remains_pending(self, _wrap):
        tm = _make_tokenizer_manager(self)
        tm.cuda_vmm_feature_transport = Mock()
        tm.cuda_vmm_feature_transport.prepare_for_dispatch.return_value = ["published"]
        tm._dispatch_to_scheduler = Mock()
        state = _make_req_state("pending")
        tm.rid_to_state["pending"] = state
        tm.abort_request("pending")
        tm._dispatch_to_scheduler.assert_not_called()
        tm._dispatch_to_scheduler.side_effect = RuntimeError("generation send failed")
        with self.assertRaisesRegex(RuntimeError, "generation send failed"):
            tm._send_one_request(MagicMock(rid="pending", mm_inputs=None))
        self.assertFalse(state.dispatched)
        self.assertFalse(state.abort_sent)
        tm.cuda_vmm_feature_transport.cancel_for_dispatch.assert_called_once_with(
            ["published"]
        )
        tm._release_req_states_on_failure(["pending"])
        self.assertNotIn("pending", tm.rid_to_state)

    @patch(
        "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
        side_effect=lambda obj: obj,
    )
    def test_cancel_after_dispatch_aborts_and_retains_state(self, _wrap):
        tm = _make_tm_for_generate(self)
        tm.cuda_vmm_feature_transport = Mock()
        tm.cuda_vmm_feature_transport.prepare_for_dispatch.return_value = []
        tm._dispatch_to_scheduler = Mock()
        rid = "disconnected"
        obj = _make_generate_obj(rid, is_single=True)
        obj.return_prompt_token_ids = False
        obj.return_logprob = False
        obj.log_metrics = False
        obj.lora_path = None
        tokenized = MagicMock(rid=rid, mm_inputs=None)
        tm._tokenize_one_request = AsyncMock(return_value=tokenized)

        async def drive():
            task = asyncio.create_task(tm.generate_request(obj).__anext__())
            for _ in range(100):
                await asyncio.sleep(0)
                if tm._dispatch_to_scheduler.called:
                    break
            self.assertTrue(tm._dispatch_to_scheduler.called)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(drive())
        aborts = [
            call.args[0]
            for call in tm._dispatch_to_scheduler.call_args_list
            if isinstance(call.args[0], AbortReq)
        ]
        self.assertEqual([req.rid for req in aborts], [rid])
        self.assertIn(rid, tm.rid_to_state)
        self.assertTrue(tm.rid_to_state[rid].dispatched)
        tm._handle_abort_req(_make_abort_req(rid))
        self.assertNotIn(rid, tm.rid_to_state)

    def test_abort_deduplicates_and_retries_failed_send(self):
        tm = _make_tokenizer_manager(self)
        state = _make_req_state("retry")
        state.dispatched = True
        tm.rid_to_state["retry"] = state
        tm._dispatch_to_scheduler = Mock(side_effect=RuntimeError("send failed"))
        tm.enable_metrics = True
        tm.metrics_collector = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "send failed"):
            tm.abort_request("retry")
        tm.metrics_collector.observe_one_aborted_request.assert_not_called()
        tm._dispatch_to_scheduler.side_effect = None
        tm.abort_request("retry")
        tm.abort_request("retry")
        self.assertEqual(tm._dispatch_to_scheduler.call_count, 2)
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_unknown_and_empty_rids_keep_public_abort_guard(self):
        tm = _make_tokenizer_manager(self)
        tm._dispatch_to_scheduler = Mock()
        with get_context().override_server_args(tokenizer_worker_num=1):
            tm.abort_request("missing")
            tm.abort_request("")
            tm._dispatch_to_scheduler.assert_not_called()

    def test_parallel_sampling_failure_cleans_generated_rid(self):
        tm = _make_tm_for_generate(self)
        obj = GenerateReqInput(text=["hello"], rid=["base"], sampling_params={"n": 2})
        tokenized = MagicMock(mm_inputs=None)
        tm._tokenize_one_request = AsyncMock(return_value=tokenized)
        tm._send_one_request = Mock(side_effect=RuntimeError("dispatch failed"))

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
            asyncio.run(drive())
        self.assertFalse(tm.rid_to_state)

    @patch(
        "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
        side_effect=lambda obj: obj,
    )
    def test_dispatch_marks_single_and_batch_only_after_success(self, _wrap):
        for batch in (False, True):
            for fail in (False, True):
                with self.subTest(batch=batch, fail=fail):
                    tm = _make_tokenizer_manager(self)
                    tm.cuda_vmm_feature_transport = Mock()
                    tm.cuda_vmm_feature_transport.prepare_for_dispatch.return_value = []
                    tm._dispatch_to_scheduler = Mock(
                        side_effect=RuntimeError("send failed") if fail else None
                    )
                    tokenized = [
                        MagicMock(rid=rid, mm_inputs=None) for rid in ("a", "b")
                    ]
                    for obj in tokenized:
                        tm.rid_to_state[obj.rid] = _make_req_state(obj.rid)
                    send = tm._send_batch_request if batch else tm._send_one_request
                    send_obj = tokenized if batch else tokenized[0]
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "send failed"):
                            send(send_obj)
                    else:
                        send(send_obj)
                    self.assertEqual(tm.rid_to_state["a"].dispatched, not fail)
                    self.assertEqual(
                        tm.rid_to_state["b"].dispatched, batch and not fail
                    )

    def test_normal_completion_does_not_abort(self):
        tm = _make_tm_for_generate(self)
        obj = _make_generate_obj("completed", is_single=True)
        obj.return_prompt_token_ids = False
        tm._tokenize_one_request = AsyncMock(return_value=MagicMock(rid=obj.rid))
        tm._send_one_request = Mock(
            side_effect=lambda tokenized: tm._mark_state_dispatched(tokenized.rid)
        )
        tm._dispatch_to_scheduler = Mock()

        async def finished(*args):
            tm.rid_to_state.pop(obj.rid)
            yield {"text": "done"}

        tm._wait_one_response = finished

        async def drive():
            self.assertEqual(
                [result async for result in tm.generate_request(obj)],
                [{"text": "done"}],
            )

        asyncio.run(drive())
        tm._dispatch_to_scheduler.assert_not_called()
        self.assertFalse(tm.rid_to_state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
