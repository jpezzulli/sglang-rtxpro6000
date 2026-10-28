"""Defer omitted Responses limits until the final tokenized input is known."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from test_responses_codex import fixture, qwen_serving, request
from utils import engine_chunk

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class ResponsesCompletionBudgetTest(unittest.TestCase):
    def setUp(self):
        # Publish ordinary CPU-test defaults for the real manager's config
        # readers, then restore the previous in-process context after each test.
        config = get_context().override_server_args()
        config.install()
        self.addCleanup(config.restore)

    def _run_request(self, *, image, reserve, max_output_tokens=None, kv_bound=False):
        serving = qwen_serving(is_multimodal=image)
        inputs = [{"role": "user", "content": "Describe the result."}]
        if image:
            inputs.extend(
                [
                    {
                        "type": "function_call",
                        "name": "view_image",
                        "arguments": '{"path":"/tmp/example.png"}',
                        "call_id": "call_view_image",
                    },
                    fixture("codex_view_image_output.json")["item"],
                ]
            )
        req = request(
            input=inputs,
            **(
                {"max_output_tokens": max_output_tokens}
                if max_output_tokens is not None
                else {}
            )
        )
        serving.tokenizer_manager.num_reserved_tokens = reserve
        _, _, prompts, _ = asyncio.run(
            serving._make_request(req, None, serving.tokenizer_manager.tokenizer)
        )
        raw_ids = (
            serving.tokenizer_manager.tokenizer.encode(prompts[0])
            if image
            else prompts[0]
        )
        # The image expansion matches the extra 569 positions seen in the live
        # failure. Text uses a short tail to cover the old minimum-512 budget.
        expanded_ids = [0] * (len(raw_ids) + (569 if image else 0))
        context_len = len(raw_ids) + (1024 if image else 32)
        serving.tokenizer_manager.model_config.context_len = context_len

        manager = TokenizerManager.__new__(TokenizerManager)
        manager.context_len = context_len
        manager.num_reserved_tokens = reserve
        manager.allow_auto_truncate = False
        manager.validate_total_tokens = True
        manager.is_generation = True
        manager.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[]), vocab_size=256
        )
        manager.tokenizer = serving.tokenizer_manager.tokenizer
        manager._tokenize_texts = AsyncMock(return_value=(raw_ids, None))
        manager.preferred_sampling_params = None
        manager.sampling_params_class = SamplingParams
        manager.max_req_input_len = context_len - 1
        manager.rid_to_state = {
            req.request_id: SimpleNamespace(time_stats=APIServerReqTimeStats())
        }
        processor = AsyncMock(
            return_value=SimpleNamespace(
                input_ids=expanded_ids, token_type_ids=None, mm_items=[]
            )
        )
        manager.mm_processor = (
            SimpleNamespace(
                prefer_tokenized_input=False, process_mm_data_async=processor
            )
            if image
            else None
        )

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_req_len = context_len - 1
        scheduler.spec_context_reserve = reserve
        scheduler.page_size = 64
        scheduler.max_new_tokens_limit = None
        paged_input = -(-len(expanded_ids) // scheduler.page_size) * scheduler.page_size
        scheduler.max_total_num_tokens = (
            paged_input + scheduler.page_size + 16 if kv_bound else context_len * 2
        )

        captured = {
            "raw_count": len(raw_ids),
            "expanded_count": len(expanded_ids),
            "context_len": context_len,
            "processor": processor,
            "scheduler": scheduler,
        }

        async def generate(adapted, raw_request):
            captured["adapted"] = adapted
            captured["initial_limit"] = adapted.sampling_params["max_new_tokens"]
            # Tokenization and vision encoding are controlled CPU doubles;
            # manager validation, SamplingParams construction/verification and
            # scheduler initialization below are the real production methods.
            tokenized = await manager._tokenize_one_request(adapted)
            captured["tokenized"] = tokenized
            scheduler_request = SimpleNamespace(
                rid=req.request_id,
                origin_input_ids=tokenized.input_ids,
                sampling_params=tokenized.sampling_params,
            )
            with get_parallel().override(attn_dcp_size=1):
                scheduler.init_req_max_new_tokens(scheduler_request)
            captured["final_limit"] = scheduler_request.sampling_params.max_new_tokens
            yield engine_chunk("accepted", finish=True)

        # Keep the production non-Harmony generation loop too: SimpleContext
        # never enters the Harmony-only built-in tool rebudgeting branch.
        serving.tokenizer_manager.generate_request = generate
        return asyncio.run(serving.create_responses(req)), captured

    def test_omitted_limit_uses_final_input_and_scheduler_bounds(self):
        for image in (False, True):
            for reserve in (0, 8):
                for kv_bound in (False, True):
                    with self.subTest(image=image, reserve=reserve, kv_bound=kv_bound):
                        result, captured = self._run_request(
                            image=image, reserve=reserve, kv_bound=kv_bound
                        )
                        self.assertEqual(
                            getattr(result, "status", None),
                            "completed",
                            getattr(result, "body", None),
                        )
                        self.assertIsNone(captured["initial_limit"])
                        self.assertIsNone(result.max_output_tokens)
                        self.assertEqual(
                            len(captured["tokenized"].input_ids),
                            captured["expanded_count"],
                        )
                        if image:
                            captured["processor"].assert_awaited_once()
                            self.assertTrue(captured["adapted"].image_data)
                            self.assertIn("<|image_pad|>", captured["adapted"].text)
                        scheduler = captured["scheduler"]
                        paged_input = (
                            -(-captured["expanded_count"] // scheduler.page_size)
                            * scheduler.page_size
                        )
                        expected = min(
                            scheduler.max_req_len
                            - captured["expanded_count"]
                            - 1
                            - reserve,
                            scheduler.max_total_num_tokens
                            - paged_input
                            - scheduler.page_size
                            - 1,
                        )
                        self.assertEqual(captured["final_limit"], expected)
                        self.assertGreater(captured["final_limit"], 0)

    def test_explicit_small_limit_keeps_existing_conversion(self):
        for image in (False, True):
            for reserve in (0, 8):
                with self.subTest(image=image, reserve=reserve):
                    result, captured = self._run_request(
                        image=image, reserve=reserve, max_output_tokens=16
                    )
                    self.assertEqual(
                        getattr(result, "status", None),
                        "completed",
                        getattr(result, "body", None),
                    )
                    self.assertEqual(captured["initial_limit"], 14)
                    self.assertEqual(captured["final_limit"], 14)
                    self.assertEqual(result.max_output_tokens, 16)

    def test_explicit_oversized_limit_still_fails_total_validation(self):
        for image in (False, True):
            for reserve in (0, 8):
                with self.subTest(image=image, reserve=reserve):
                    result, captured = self._run_request(
                        image=image, reserve=reserve, max_output_tokens=100000
                    )
                    self.assertEqual(result.status_code, 400)
                    self.assertIn(b"Requested token count exceeds", result.body)
                    self.assertIsInstance(captured["initial_limit"], int)
                    self.assertNotIn("tokenized", captured)

    def test_protocol_none_budget_and_existing_explicit_bounds(self):
        self.assertIsNone(request().to_sampling_params()["max_new_tokens"])
        self.assertEqual(
            request(max_output_tokens=100).to_sampling_params()["max_new_tokens"], 98
        )
        self.assertEqual(
            request(max_output_tokens=1000).to_sampling_params(default_max_tokens=128)[
                "max_new_tokens"
            ],
            126,
        )


if __name__ == "__main__":
    unittest.main()
