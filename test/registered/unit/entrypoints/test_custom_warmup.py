import json
import unittest

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import warmup
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingTokenizerManager:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []
        self.yield_count = 0

    async def generate_request(self, request, raw_request):
        self.requests.append((request, raw_request))
        for response in self.responses:
            self.yield_count += 1
            yield response


class TestStructuredOutputWarmup(unittest.IsolatedAsyncioTestCase):
    async def test_remains_opt_in(self):
        manager = _RecordingTokenizerManager([])

        await warmup.execute_warmups("null", [], manager)

        self.assertEqual(manager.requests, [])

    async def test_registered_warmup_sends_one_bounded_schema_request_and_drains(self):
        manager = _RecordingTokenizerManager(
            [
                {"text": "{", "meta_info": {"finish_reason": None}},
                {
                    "text": '{"topic":"colors","items":["blue"]}',
                    "meta_info": {"finish_reason": {"type": "stop"}},
                },
            ]
        )

        await warmup.execute_warmups("null", ["structured_output"], manager)

        self.assertEqual(manager.yield_count, 2)
        self.assertEqual(len(manager.requests), 1)
        request, raw_request = manager.requests[0]
        self.assertIsNone(raw_request)
        self.assertIsNone(request.bootstrap_room)
        self.assertIsNone(request.bootstrap_host)
        self.assertLessEqual(request.sampling_params["max_new_tokens"], 32)
        self.assertEqual(request.sampling_params["temperature"], 0.0)
        schema = json.loads(request.sampling_params["json_schema"])
        self.assertEqual(schema["type"], "object")
        self.assertEqual(set(schema["required"]), {"topic", "items"})

    async def test_disaggregation_request_uses_existing_fake_bootstrap_convention(self):
        manager = _RecordingTokenizerManager(
            [{"text": "{}", "meta_info": {"finish_reason": {"type": "stop"}}}]
        )

        await warmup.execute_warmups("prefill", ["structured_output"], manager)

        request, _ = manager.requests[0]
        self.assertEqual(request.bootstrap_room, 0)
        self.assertEqual(request.bootstrap_host, FAKE_BOOTSTRAP_HOST)

    async def test_terminal_generator_error_fails_startup(self):
        error_responses = (
            {"error": {"message": "grammar compilation failed"}},
            {
                "text": "",
                "meta_info": {
                    "finish_reason": {
                        "type": "abort",
                        "message": "grammar compilation failed",
                        "status_code": 500,
                    }
                },
            },
        )

        for response in error_responses:
            with self.subTest(response=response):
                manager = _RecordingTokenizerManager([response])
                with self.assertRaisesRegex(RuntimeError, "grammar compilation failed"):
                    await warmup.execute_warmups("null", ["structured_output"], manager)


if __name__ == "__main__":
    unittest.main()
