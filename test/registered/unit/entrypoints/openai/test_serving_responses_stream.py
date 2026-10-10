import asyncio
import unittest
from unittest.mock import Mock, patch

from utils import (
    StreamFixture,
    engine_chunk,
    event_payloads,
    event_types,
    find_completed_event,
    make_serving,
)

from sglang.srt.entrypoints.openai.protocol import (
    RequestResponseMetadata,
    ResponsesRequest,
    ResponsesResponse,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class NonHarmonyStreamTestCase(CustomTestCase):
    def test_reasoning_content_part_lifecycle_matches_terminal_output(self):
        for finish_reason, status, suffix in (
            ({"type": "stop"}, "completed", "</think>answer"),
            ({"type": "length"}, "incomplete", ""),
            (
                {"type": "abort", "status_code": 503, "message": "Worker unavailable"},
                "failed",
                "",
            ),
        ):
            with self.subTest(status=status):
                serving = make_serving()
                serving.reasoning_parser = "qwen3"
                serving.tool_call_parser = None
                request = ResponsesRequest(
                    model="x", input="hi", stream=True, store=False
                )
                chunks = [engine_chunk("<think>wo"), engine_chunk("<think>work")]
                terminal_chunk = engine_chunk("<think>work" + suffix, 4, finish=True)
                terminal_chunk["meta_info"]["finish_reason"] = finish_reason
                chunks.append(terminal_chunk)
                events = StreamFixture(serving, request, require_reasoning=True).run(
                    chunks
                )
                payloads = event_payloads(events)
                self.assertEqual(payloads[-1]["response"]["status"], status)
                self.assertEqual(payloads[-1]["type"], f"response.{status}")
                output = payloads[-1]["response"]["output"]
                reasoning = output[0]
                self.assertEqual(reasoning["type"], "reasoning")
                item_events = [
                    p for p in payloads if p.get("item_id") == reasoning["id"]
                ]
                types = [p["type"] for p in item_events]
                self.assertEqual(types[0], "response.content_part.added")
                self.assertEqual(
                    types[-2:],
                    ["response.reasoning_text.done", "response.content_part.done"],
                )
                self.assertEqual(types.count("response.content_part.added"), 1)
                self.assertEqual(types.count("response.content_part.done"), 1)
                self.assertEqual(
                    item_events[0]["part"], {"type": "reasoning_text", "text": ""}
                )
                self.assertEqual(item_events[-1]["part"], reasoning["content"][0])
                self.assertEqual(item_events[-2]["text"], "work")
                self.assertEqual(
                    "".join(
                        p["delta"]
                        for p in item_events
                        if p["type"] == "response.reasoning_text.delta"
                    ),
                    "work",
                )
                self.assertTrue(
                    all(
                        p["output_index"] == 0 and p["content_index"] == 0
                        for p in item_events
                    )
                )
                done = [
                    p["item"]
                    for p in payloads
                    if p["type"] == "response.output_item.done"
                ]
                self.assertEqual(done, output)
                self.assertEqual(
                    [p["sequence_number"] for p in payloads], list(range(len(payloads)))
                )

    def test_reasoning_summary_keeps_its_own_part_events(self):
        serving = make_serving()
        serving.reasoning_parser = "qwen3"
        serving.tool_call_parser = None
        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
            reasoning={"summary": "auto"},
        )
        events = StreamFixture(serving, request, require_reasoning=True).run(
            [engine_chunk("<think>work</think>answer", 4, finish=True)]
        )
        payloads = event_payloads(events)
        reasoning = find_completed_event(events)["response"]["output"][0]
        parts = [p for p in payloads if p.get("item_id") == reasoning["id"]]
        self.assertEqual(
            [p["type"] for p in parts],
            [
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.done",
                "response.reasoning_summary_part.done",
            ],
        )
        self.assertEqual(parts[-1]["part"], reasoning["summary"][0])
        self.assertEqual(
            reasoning["content"], [{"type": "reasoning_text", "text": "work"}]
        )

    def test_reasoning_parser_uses_processed_reasoning_state(self):
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
        ) as parser_cls:
            parser_cls.return_value.parse_stream_chunk.return_value = (None, "done")
            fixture = StreamFixture(serving, request, require_reasoning=True)
            fixture.run([engine_chunk("done", 1, finish=True)])

        self.assertTrue(parser_cls.call_args.kwargs["force_reasoning"])

    def test_emits_typed_sse_events_in_order(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)
        fixture = StreamFixture(serving, request)
        events = fixture.run(
            [
                engine_chunk("Hel", 1),
                engine_chunk("Hello", 2),
                engine_chunk("Hello world", 4, finish=True),
            ]
        )

        types = event_types(events)
        self.assertEqual(types[0], "response.created")
        self.assertEqual(types[1], "response.in_progress")
        for ev in (
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ):
            self.assertIn(ev, types)
        self.assertEqual(types[-1], "response.completed")

        seqs = [p["sequence_number"] for p in event_payloads(events)]
        self.assertEqual(seqs, list(range(len(seqs))))

    def test_required_tool_choice_emits_function_call_events(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
            tool_choice="required",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
        )
        payload = '[{"name": "get_weather", "parameters": {"city": "Beijing"}}]'

        chunks = []
        sent = 0
        while sent < len(payload):
            sent += min(8, len(payload) - sent)
            chunks.append(
                engine_chunk(payload[:sent], sent, finish=sent == len(payload))
            )

        fixture = StreamFixture(serving, request)
        events = fixture.run(chunks)
        types = event_types(events)

        self.assertIn("response.function_call_arguments.delta", types)
        self.assertIn("response.function_call_arguments.done", types)
        self.assertIn("response.output_item.added", types)
        self.assertIn("response.output_item.done", types)
        self.assertNotIn("response.output_text.delta", types)

        added_kinds = [
            payload["item"]["type"]
            for payload in event_payloads(events)
            if payload.get("type") == "response.output_item.added"
        ]
        self.assertIn("function_call", added_kinds)

    def test_final_output_preserves_text_tool_text_order(self):
        from sglang.srt.function_call.core_types import (
            StreamingParseResult,
            ToolCallItem,
        )

        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = "qwen3_coder"

        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
        )

        scripted = [
            StreamingParseResult(normal_text="I'll check.", calls=[]),
            StreamingParseResult(
                normal_text="",
                calls=[
                    ToolCallItem(
                        tool_index=0,
                        name="get_weather",
                        parameters='{"city": "Beijing"}',
                    )
                ],
            ),
            StreamingParseResult(normal_text="It's sunny.", calls=[]),
        ]
        chunks = [
            engine_chunk(" " * 3, 3),
            engine_chunk(" " * 10, 10),
            engine_chunk(" " * 14, 14, finish=True),
        ]

        script_iter = iter(scripted)

        def fake_parse_stream_chunk(delta):
            sp = next(script_iter)
            return sp.normal_text, sp.calls

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            parser_cls.return_value.detector.supports_structural_tag.return_value = True
            parser_cls.return_value.parse_stream_chunk.side_effect = (
                fake_parse_stream_chunk
            )
            parser_cls.return_value.parse_stream_end.return_value = ("", [])
            fixture = StreamFixture(serving, request)
            events = fixture.run(chunks)

        completed = find_completed_event(events)
        output = completed["response"]["output"]
        kinds = [item["type"] for item in output]
        self.assertEqual(kinds, ["message", "function_call", "message"])
        self.assertEqual(output[0]["content"][0]["text"], "I'll check.")
        self.assertEqual(output[1]["name"], "get_weather")
        self.assertEqual(output[2]["content"][0]["text"], "It's sunny.")

    def test_reasoning_parser_flushed_at_stream_end(self):
        """Bug regression: the stream loop never drained text the reasoning
        parser held back as a possible marker prefix, so a response whose text
        genuinely ends with e.g. "<|e" lost that tail on /v1/responses (chat
        flushes via parse_stream_end; responses did not)."""
        serving = make_serving()
        serving.reasoning_parser = "muse"
        serving.tool_call_parser = None

        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)
        text = (
            " to=self<|message|>think<|eom|>"
            "<|start|>assistant to=user<|message|>Answer<|e"
        )
        fixture = StreamFixture(serving, request)
        events = fixture.run(
            [
                engine_chunk(text[:30], 4),
                engine_chunk(text, 9, finish=True),
            ]
        )

        streamed = "".join(
            p["delta"]
            for ev, p in zip(event_types(events), event_payloads(events))
            if ev == "response.output_text.delta"
        )
        self.assertEqual(streamed, "Answer<|e")


class MultiToolCallStreamingOrderTestCase(CustomTestCase):
    """The wire order of message / function_call items across tool-call deltas."""

    def setUp(self):
        from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

        self.serving = make_serving()
        self.serving.tool_call_parser = "qwen3_coder"
        self.serving.reasoning_parser = None

        det = Qwen3CoderDetector()
        s, e = det.tool_call_start_token, det.tool_call_end_token
        fp, fe = det.tool_call_prefix, det.function_end_token
        pp, pe = det.parameter_prefix, det.parameter_end_token
        self.weather = f"{s}{fp}get_weather>{pp}city>Beijing{pe}{fe}{e}"
        self.time = f"{s}{fp}get_time>{pp}tz>UTC{pe}{fe}{e}"
        # a prefix of ``weather`` that stops mid-arguments
        self.weather_head = f"{s}{fp}get_weather>{pp}city>Beij"

    def _seq(self, texts, *names):
        """Stream cumulative ``texts`` (last one final) and return (type, payload)."""
        request = ResponsesRequest(
            model="x",
            input="weather and time",
            store=False,
            tools=[
                {"type": "function", "name": n, "parameters": {"type": "object"}}
                for n in names
            ],
        )
        chunks = [engine_chunk(t) for t in texts]
        chunks.append(engine_chunk(texts[-1], finish=True))
        return StreamFixture(self.serving, request).run_seq(chunks)

    @staticmethod
    def _added(seq):
        return [
            (p["output_index"], p["item"].get("type"))
            for t, p in seq
            if t == "response.output_item.added"
        ]

    @staticmethod
    def _done_calls(seq):
        return [
            p["item"]
            for t, p in seq
            if t == "response.output_item.done"
            and p["item"].get("type") == "function_call"
        ]

    def test_prior_tool_call_done_before_next_added(self):
        full = self.weather + "\n" + self.time
        seq = self._seq(
            [self.weather, self.weather + "\n", full], "get_weather", "get_time"
        )

        def position(pred):
            return next(i for i, (t, p) in enumerate(seq) if pred(t, p))

        done0 = position(
            lambda t, p: t == "response.output_item.done" and p["output_index"] == 0
        )
        added1 = position(
            lambda t, p: t == "response.output_item.added" and p["output_index"] == 1
        )
        self.assertLess(done0, added1)

        items = self._done_calls(seq)
        self.assertEqual(sorted(i["name"] for i in items), ["get_time", "get_weather"])

    def test_prose_before_tool_call_keeps_message_first(self):
        """Prose and a tool-call start in one delta: the message item must come
        first, since the prose preceded the call."""
        # One delta spanning prose + the whole call, as spec decoding or
        # --stream-interval > 1 produces.
        seq = self._seq(["Let me check." + self.weather], "get_weather")

        added = self._added(seq)
        message_index = next(i for i, kind in added if kind == "message")
        call_index = next(i for i, kind in added if kind == "function_call")
        self.assertLess(message_index, call_index)

        # The call must not be split across two items by the reordering.
        self.assertEqual(len([k for _, k in added if k == "function_call"]), 1)

    def test_call_tail_prose_and_next_call_in_one_delta(self):
        """One delta closing tool1, carrying prose, and opening tool2 needs both
        orders at once: tool1's trailing "}" must be drained before the prose
        closes every open item, and tool2 must land after the message."""
        seq = self._seq(
            [self.weather_head, self.weather + "Here you go." + self.time],
            "get_weather",
            "get_time",
        )

        items = self._done_calls(seq)
        # No duplicate item invented for the already-closed call, and no call
        # left nameless by being reopened from an args-only fragment.
        self.assertEqual(len(items), 2)
        self.assertTrue(all(i["name"] for i in items))
        self.assertEqual(items[0]["arguments"], '{"city": "Beijing"}')


TERMINAL_CASES = (
    ({"type": "stop"}, "completed", None),
    ({"type": "length"}, "incomplete", None),
    (
        {"type": "abort", "status_code": 503, "message": "Worker unavailable"},
        "failed",
        "Worker unavailable",
    ),
    (
        {"type": "abort", "status_code": 400, "message": "Bad request"},
        "failed",
        "Bad request",
    ),
)


class TerminalStreamStatusTestCase(CustomTestCase):
    """The last SSE event and ``response.status`` must agree: clients read the
    event name as the outcome, so a length cap or an engine abort wrapped in
    ``response.completed``/``completed`` reads as a finished answer."""

    def _stream(self, serving, request, finish_reason, text="partial answer"):
        chunk = engine_chunk(text, 3, finish=True)
        chunk["meta_info"]["finish_reason"] = finish_reason
        return StreamFixture(serving, request).run([chunk])

    def test_terminal_event_matches_status_for_stop_length_and_abort(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        for finish_reason, status, message in TERMINAL_CASES:
            with self.subTest(status=status):
                request = ResponsesRequest(
                    model="x", input="hi", stream=True, store=True
                )
                events = self._stream(serving, request, finish_reason)
                types = event_types(events)
                payloads = event_payloads(events)

                terminal_names = (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                )
                self.assertEqual(types[-1], f"response.{status}")
                self.assertEqual(sum(t in terminal_names for t in types), 1)
                terminal = payloads[-1]["response"]
                self.assertEqual(terminal["status"], status)

                # partial output, usage and ids survive the non-completed outcome
                self.assertEqual([i["type"] for i in terminal["output"]], ["message"])
                self.assertEqual(
                    terminal["output"][0]["content"][0]["text"], "partial answer"
                )
                done_item = next(
                    p["item"]
                    for p in payloads
                    if p["type"] == "response.output_item.done"
                )
                self.assertEqual(done_item["id"], terminal["output"][0]["id"])
                self.assertEqual(terminal["usage"]["output_tokens"], 3)
                self.assertEqual(terminal["usage"]["input_tokens"], 5)

                if status == "incomplete":
                    self.assertEqual(
                        terminal["incomplete_details"], {"reason": "max_output_tokens"}
                    )
                if status == "failed":
                    self.assertEqual(
                        terminal["error"],
                        {"code": "server_error", "message": message},
                    )

                # sequence numbering stays contiguous across the new event
                self.assertEqual(
                    [p["sequence_number"] for p in payloads],
                    list(range(len(payloads))),
                )
                # and the stored response reports the same outcome
                stored = serving.response_store[request.request_id]
                self.assertEqual(stored.status, status)
                self.assertEqual(stored.id, request.request_id)

    def test_cancelled_stored_response_is_not_overwritten(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None
        request = ResponsesRequest(model="x", input="hi", stream=True, store=True)
        cancelled = ResponsesResponse.from_request(
            request,
            sampling_params={},
            model_name="x",
            created_time=0,
            output=[],
            status="cancelled",
            usage=None,
        )
        serving.response_store[request.request_id] = cancelled

        events = self._stream(
            serving, request, {"type": "abort", "status_code": 503, "message": "boom"}
        )

        self.assertEqual(event_types(events)[-1], "response.failed")
        self.assertEqual(event_payloads(events)[-1]["response"]["status"], "failed")
        self.assertIs(serving.response_store[request.request_id], cancelled)

    def test_unhandled_stream_error_keeps_failed_terminal_event(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None
        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)

        async def boom_generator():
            yield engine_chunk("so far", 1)
            raise RuntimeError("parser exploded")

        async def collect():
            out = []
            async for chunk in serving.responses_stream_generator_non_harmony(
                request,
                sampling_params={},
                result_generator=boom_generator(),
                model_name="x",
                tokenizer=Mock(),
                request_metadata=RequestResponseMetadata(request_id=request.request_id),
                require_reasoning=False,
            ):
                out.append(chunk)
            return out

        events = asyncio.run(collect())
        payloads = event_payloads(events)
        self.assertEqual(payloads[-1]["type"], "response.failed")
        self.assertEqual(payloads[-1]["response"]["status"], "failed")
        self.assertEqual(
            payloads[-1]["response"]["error"]["message"], "parser exploded"
        )


if __name__ == "__main__":
    unittest.main()
