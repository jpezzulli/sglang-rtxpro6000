import asyncio
import hashlib
import json
import pathlib
import re
import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from fastapi.responses import JSONResponse  # noqa: E402
from jinja2 import Environment  # noqa: E402
from transformers.utils.chat_template_utils import _compile_jinja_template  # noqa: E402

from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicCountTokensRequest,
    AnthropicMessage,
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import (  # noqa: E402
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from sglang.srt.entrypoints.openai.serving_chat import (  # noqa: E402
    normalize_assistant_tool_call_arguments,
    normalize_tool_content,
)
from sglang.srt.parser.jinja_template_utils import (  # noqa: E402
    detect_jinja_template_content_format,
    process_content_for_template_format,
)
from sglang.srt.parser.template_detection import (  # noqa: E402
    detect_inline_system_support,
    detect_tool_reference_support,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
_PENNYROYAL_CONFIGS = _REPO_ROOT / "configs" / "pennyroyal"


def _pennyroyal_chat_template() -> str:
    """Bytes of the chat template the Pennyroyal launchers pin.

    ``configs/pennyroyal/chat-template.sh`` is the single shared pin every
    launcher sources, so both supported profiles render through this one
    template. Resolving the path through the helper keeps this test from
    hardcoding a template version, and reading the file (rather than copying
    it in) keeps the template itself unchanged.
    """
    helper = (_PENNYROYAL_CONFIGS / "chat-template.sh").read_text(encoding="utf-8")
    pinned = re.search(r'pwd\)/([^"]+\.jinja)"', helper)
    assert pinned, f"no pinned jinja template found in chat-template.sh: {helper}"
    return (_PENNYROYAL_CONFIGS / pinned.group(1)).read_text(encoding="utf-8")


def _render_chat_request(
    template_text: str, chat_request: ChatCompletionRequest, *, tokenizer=None
) -> str:
    """Render a converted request the way the chat-completions route does."""
    dump = chat_request.model_dump(exclude_none=True)
    content_format = detect_jinja_template_content_format(template_text)
    messages = []
    for message in dump["messages"]:
        normalize_assistant_tool_call_arguments(message)
        if message.get("content") is None:
            message["content"] = ""
        processed = process_content_for_template_format(
            message, content_format, [], [], [], []
        )
        processed["content"] = normalize_tool_content(
            processed["role"], processed.get("content")
        )
        messages.append(processed)
    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            messages,
            chat_template=template_text,
            tools=dump.get("tools") or [],
            add_generation_prompt=True,
            tokenize=False,
        )
    return _compile_jinja_template(template_text).render(
        messages=messages,
        tools=dump.get("tools") or [],
        add_generation_prompt=True,
    )


class _FakeOpenAIServingChat:
    native_reasoning_history = False

    def __init__(self, stream_lines=None, chat_template=None):
        self.stream_lines = stream_lines or []
        self.apply_reasoning_calls: list[bool] = []
        self.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template=chat_template)
        )

    def supports_native_reasoning_history(self):
        return self.native_reasoning_history

    def _generate_chat_stream(self, adapted_request, processed_request, raw_request):
        async def _gen():
            for line in self.stream_lines:
                yield line

        return _gen()

    def apply_reasoning_enabled(self, chat_request, enabled):
        self.apply_reasoning_calls.append(enabled)

    def wrap_reasoning_history(self, text):
        return f"<think>\n{text}\n</think>"


class _FakeNonStreamingErrorOpenAI:
    """Returns a configurable error response from the OpenAI handler."""

    def __init__(self, status_code=400, body=None, content=None):
        self._status_code = status_code
        self._body = body
        self._content = content

    def _validate_request(self, chat_request):
        return None

    def _convert_to_internal_request(self, chat_request, raw_request):
        return SimpleNamespace(), chat_request

    async def _handle_non_streaming_request(
        self, adapted_request, processed_request, raw_request
    ):
        if self._body is not None:
            # Build a response object exposing raw bytes via `.body`.
            return SimpleNamespace(status_code=self._status_code, body=self._body)
        return JSONResponse(
            status_code=self._status_code,
            content=self._content
            or {
                "error": {
                    "type": "invalid_request_error",
                    "message": "context length exceeded",
                }
            },
        )


class _FakeNonStreamingOpenAI:
    """Returns a configurable ChatCompletionResponse from the OpenAI handler."""

    def __init__(self, response):
        self._response = response

    def _validate_request(self, chat_request):
        return None

    def _convert_to_internal_request(self, chat_request, raw_request):
        return SimpleNamespace(), chat_request

    async def _handle_non_streaming_request(
        self, adapted_request, processed_request, raw_request
    ):
        return self._response


def _chunk(choices=None, usage=None):
    data = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": choices or [],
    }
    if usage is not None:
        data["usage"] = usage
    return f"data: {json.dumps(data)}\n\n"


def _choice(delta, finish_reason=None):
    return {
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
    }


async def _collect_anthropic_events(serving, anthropic_request):
    events = []
    async for sse in serving._generate_anthropic_stream(
        adapted_request=object(),
        processed_request=object(),
        anthropic_request=anthropic_request,
        raw_request=object(),
    ):
        for line in sse.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


class TestAnthropicServing(unittest.TestCase):
    # Renders system at any position (GLM/Kimi/Qwen3) → can pass through.
    INLINE_SYSTEM_TEMPLATE = (
        "{%- for message in messages %}"
        "{{- message.role }}: {{ message.content }}\n"
        "{%- endfor %}"
    )
    GLM_TOOL_RESULT_TEMPLATE = """
{%- for message in messages if message.role == "tool" -%}
{%- if loop.first -%}<|observation|>{%- endif -%}
{%- if message.content is string -%}
<tool_response>{{ message.content }}</tool_response>
{%- elif message.content.0.type == "tool_reference" -%}
<tool_response><tools>
{%- for reference in message.content -%}
{%- for tool in tools if tool.function.name == reference.name -%}
{{ tool.function.name }}
{%- endfor -%}
{%- endfor -%}
</tools></tool_response>
{%- endif -%}
{%- endfor -%}
"""

    def _serving(self, stream_lines=None, chat_template=None):
        return AnthropicServing(_FakeOpenAIServingChat(stream_lines, chat_template))

    def _anthropic_request(self, **overrides):
        data = {
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
        data.update(overrides)
        return AnthropicMessagesRequest.model_validate(data)

    def _tool_result_request(self, content, tools=None):
        overrides = {
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": content,
                        }
                    ],
                }
            ],
        }
        if tools is not None:
            overrides["tools"] = tools
        return self._anthropic_request(**overrides)

    def test_stream_closes_tool_block_before_text_delta(self):
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"query"',
                                        },
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "type": "function",
                                        "function": {"arguments": ': "sglang"}'},
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk([_choice({"content": "done"})]),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )

        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        block_events = [
            (event["type"], event.get("content_block", {}).get("type"))
            for event in events
            if event["type"].startswith("content_block")
        ]

        self.assertEqual(
            block_events,
            [
                ("content_block_start", "tool_use"),
                ("content_block_delta", None),
                ("content_block_delta", None),
                ("content_block_stop", None),
                ("content_block_start", "text"),
                ("content_block_delta", None),
                ("content_block_stop", None),
            ],
        )

        text_delta = [
            event
            for event in events
            if event["type"] == "content_block_delta"
            and event["delta"].get("type") == "text_delta"
        ][0]
        self.assertEqual(text_delta["index"], 1)

    def test_stream_reasoning_content_uses_thinking_block(self):
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"reasoning_content": "think first"})]),
                _chunk([_choice({"content": "answer"})]),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )

        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        content_events = [
            event for event in events if event["type"].startswith("content_block")
        ]

        self.assertEqual(content_events[0]["content_block"]["type"], "thinking")
        # Signature is absent (None and excluded) — never emit empty
        # string, which would fail downstream Anthropic signature verifiers.
        self.assertNotIn("signature", content_events[0]["content_block"])
        self.assertEqual(content_events[1]["delta"]["type"], "thinking_delta")
        self.assertEqual(content_events[1]["delta"]["thinking"], "think first")
        # No empty signature_delta event between thinking_delta and content_block_stop.
        self.assertEqual(content_events[2]["type"], "content_block_stop")
        self.assertEqual(content_events[3]["content_block"]["type"], "text")
        # Confirm no signature_delta event was emitted in the entire stream.
        sig_deltas = [
            event
            for event in events
            if event["type"] == "content_block_delta"
            and event.get("delta", {}).get("type") == "signature_delta"
        ]
        self.assertEqual(sig_deltas, [])

    def test_stream_usage_subtracts_cache_read_and_omits_final_input_tokens(self):
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "total_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        final_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"content": "hi"})], usage=usage),
                _chunk([], usage=final_usage),
                "data: [DONE]\n\n",
            ]
        )

        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        message_start = [event for event in events if event["type"] == "message_start"][
            0
        ]
        message_delta = [event for event in events if event["type"] == "message_delta"][
            0
        ]

        self.assertEqual(message_start["message"]["usage"]["input_tokens"], 6)
        self.assertEqual(
            message_start["message"]["usage"]["cache_read_input_tokens"], 4
        )
        self.assertNotIn("input_tokens", message_delta["usage"])
        self.assertEqual(message_delta["usage"]["output_tokens"], 2)

    def test_non_streaming_usage_subtracts_cache_read_tokens(self):
        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            }
        )

        anthropic_response = self._serving()._convert_response(response)

        self.assertEqual(anthropic_response.usage.input_tokens, 6)
        self.assertEqual(anthropic_response.usage.output_tokens, 2)
        self.assertEqual(anthropic_response.usage.cache_read_input_tokens, 4)

    def test_tool_result_search_result_content_is_flattened(self):
        request = AnthropicMessagesRequest.model_validate(
            {
                "model": "test-model",
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call_1",
                                "content": [
                                    {
                                        "type": "search_result",
                                        "title": "SGLang docs",
                                        "source": "https://docs.sglang.ai",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": "Anthropic API notes",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )

        chat_request = self._serving()._convert_to_chat_completion_request(request)
        tool_message = [
            msg
            for msg in chat_request.model_dump()["messages"]
            if msg["role"] == "tool"
        ][0]

        self.assertIn("SGLang docs", tool_message["content"])
        self.assertIn("https://docs.sglang.ai", tool_message["content"])
        self.assertIn("Anthropic API notes", tool_message["content"])

    def test_mixed_tool_reference_content_preserves_part_order(self):
        request = self._tool_result_request(
            [
                {"type": "text", "text": "Tool loaded: Bash"},
                {"type": "tool_reference", "tool_name": "Bash"},
                {"type": "text", "text": "Ready"},
            ]
        )

        chat_request = self._serving()._convert_to_chat_completion_request(request)
        messages = chat_request.model_dump(exclude_none=True)["messages"]

        self.assertEqual([message["role"] for message in messages], ["tool"] * 3)
        self.assertEqual(
            [message["tool_call_id"] for message in messages], ["call_1"] * 3
        )
        self.assertEqual(messages[0]["content"], "Tool loaded: Bash")
        self.assertEqual(
            messages[1]["content"],
            [{"type": "tool_reference", "name": "Bash"}],
        )
        self.assertEqual(messages[2]["content"], "Ready")

    def test_mixed_tool_reference_content_renders_text_and_schema(self):
        template = Environment().from_string(self.GLM_TOOL_RESULT_TEMPLATE)
        request = self._tool_result_request(
            [
                {"type": "text", "text": "Tool loaded: Bash"},
                {"type": "tool_reference", "tool_name": "Bash"},
            ],
            tools=[
                {
                    "name": "Bash",
                    "description": "Run a shell command",
                    "input_schema": {"type": "object", "properties": {}},
                    "defer_loading": True,
                }
            ],
        )

        chat_request = self._serving()._convert_to_chat_completion_request(request)
        payload = chat_request.model_dump(exclude_none=True)
        prompt = template.render(messages=payload["messages"], tools=payload["tools"])

        self.assertIn("<tool_response>Tool loaded: Bash</tool_response>", prompt)
        self.assertIn("<tools>Bash</tools>", prompt)
        self.assertEqual(prompt.count("<|observation|>"), 1)

    def test_reference_only_tool_result_remains_one_message(self):
        request = self._tool_result_request(
            [
                {"type": "tool_reference", "tool_name": "Bash"},
                {"type": "tool_reference", "tool_name": "Read"},
            ]
        )

        chat_request = self._serving()._convert_to_chat_completion_request(request)
        messages = chat_request.model_dump(exclude_none=True)["messages"]

        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0]["content"],
            [
                {"type": "tool_reference", "name": "Bash"},
                {"type": "tool_reference", "name": "Read"},
            ],
        )

    def test_builtin_web_search_tool_without_schema_is_skipped(self):
        request = AnthropicMessagesRequest.model_validate(
            {
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "search sglang"}],
                "tools": [{"name": "web_search", "type": "web_search_20250305"}],
                "tool_choice": {"type": "auto"},
            }
        )

        chat_request = self._serving()._convert_to_chat_completion_request(request)

        self.assertIsNone(chat_request.tools)
        self.assertEqual(chat_request.tool_choice, "none")

    def test_custom_tool_without_schema_is_rejected(self):
        # With the discriminated union, an AnthropicCustomTool variant must
        # carry an input_schema. The check fires at request-parse time
        # (Pydantic raises ValidationError, a subclass of ValueError).
        with self.assertRaisesRegex(ValueError, "input_schema"):
            AnthropicMessagesRequest.model_validate(
                {
                    "model": "test-model",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "call a tool"}],
                    "tools": [{"name": "custom_without_schema"}],
                }
            )

    def test_non_streaming_openai_error_response_is_forwarded(self):
        serving = AnthropicServing(_FakeNonStreamingErrorOpenAI())
        chat_request = ChatCompletionRequest(
            model="test-model",
            max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
        )
        anthropic_request = self._anthropic_request(stream=False)

        response = asyncio.run(
            serving._handle_non_streaming(chat_request, anthropic_request, object())
        )
        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["message"], "context length exceeded")

    # ------------------------------------------------------------------
    # Edge-case coverage added in the review-fix pass
    # ------------------------------------------------------------------

    def test_stream_text_then_tool_use_closes_text_block(self):
        """Text deltas followed by tool_use must close the text block before opening tool_use index."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": "Hello"})]),
                _chunk([_choice({"content": " world"})]),
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "search",
                                            "arguments": '{"q":"hello"}',
                                        },
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk([_choice({}, finish_reason="tool_calls")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        block_events = [
            (event["type"], event.get("content_block", {}).get("type"))
            for event in events
            if event["type"].startswith("content_block")
        ]
        # text block (start, delta, delta, stop) then tool_use (start, delta, stop)
        self.assertEqual(block_events[0], ("content_block_start", "text"))
        text_stop_idx = next(
            i for i, ev in enumerate(block_events) if ev == ("content_block_stop", None)
        )
        tool_start_idx = next(
            i
            for i, ev in enumerate(block_events)
            if ev == ("content_block_start", "tool_use")
        )
        self.assertLess(text_stop_idx, tool_start_idx)

    def test_stream_tool_use_without_arguments_is_not_empty_completion(self):
        """A zero-argument tool call is valid content even without input_json_delta."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "ping", "arguments": ""},
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk([_choice({}, finish_reason="tool_calls")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )

        self.assertFalse(any(event["type"] == "error" for event in events))
        tool_start = next(
            event
            for event in events
            if event["type"] == "content_block_start"
            and event["content_block"]["type"] == "tool_use"
        )
        self.assertEqual(tool_start["content_block"]["name"], "ping")
        message_delta = next(
            event for event in events if event["type"] == "message_delta"
        )
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

    def test_stream_no_usage_chunk_emits_error_event(self):
        """Stream that yields only [DONE] (no content delta) must surface as an error event."""
        serving = self._serving(["data: [DONE]\n\n"])
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [event["type"] for event in events]
        # Sequence: message_start, error, message_stop
        self.assertEqual(types, ["message_start", "error", "message_stop"])
        error_event = events[1]
        self.assertEqual(error_event["error"]["type"], "api_error")
        self.assertIn("no content", error_event["error"]["message"].lower())

    def test_cache_read_exceeds_prompt_tokens_clamps_to_zero(self):
        """When cached_tokens > prompt_tokens, input_tokens clamps to 0 instead of going negative."""
        usage = {
            "prompt_tokens": 4,
            "completion_tokens": 0,
            "total_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 10},
        }
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"content": "ok"})], usage=usage),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        message_start = next(e for e in events if e["type"] == "message_start")
        usage_out = message_start["message"]["usage"]
        self.assertEqual(usage_out["input_tokens"], 0)
        self.assertEqual(usage_out["cache_read_input_tokens"], 10)

    def test_usage_without_prompt_tokens_details(self):
        """Usage object without prompt_tokens_details must omit cache_read_input_tokens cleanly."""
        usage = {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5}
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"content": "ok"})], usage=usage),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        message_start = next(e for e in events if e["type"] == "message_start")
        usage_out = message_start["message"]["usage"]
        self.assertEqual(usage_out["input_tokens"], 5)
        self.assertNotIn("cache_read_input_tokens", usage_out)

    def test_non_streaming_error_with_non_json_body(self):
        """Non-JSON upstream error body falls back to body[:500] as the message (for 4xx)."""
        serving = AnthropicServing(
            _FakeNonStreamingErrorOpenAI(
                status_code=400,
                body=b"<html>upstream gateway rejected: bad payload</html>",
            )
        )
        chat_request = ChatCompletionRequest(
            model="test-model",
            max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
        )
        anthropic_request = self._anthropic_request(stream=False)
        response = asyncio.run(
            serving._handle_non_streaming(chat_request, anthropic_request, object())
        )
        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertIn("upstream gateway rejected", payload["error"]["message"])

    def test_non_streaming_error_5xx_scrubs_message(self):
        """5xx errors always return a generic message regardless of upstream payload."""
        for status_code, expected_type in [
            (500, "api_error"),
            (502, "api_error"),
            (503, "overloaded_error"),
            (504, "api_error"),
        ]:
            serving = AnthropicServing(
                _FakeNonStreamingErrorOpenAI(
                    status_code=status_code,
                    body=b'{"error":{"message":"sensitive internals: /opt/secret","type":"internal"}}',
                )
            )
            chat_request = ChatCompletionRequest(
                model="test-model",
                max_tokens=16,
                messages=[{"role": "user", "content": "hello"}],
            )
            anthropic_request = self._anthropic_request(stream=False)
            response = asyncio.run(
                serving._handle_non_streaming(chat_request, anthropic_request, object())
            )
            payload = json.loads(response.body)
            self.assertEqual(
                response.status_code,
                status_code,
                f"status {status_code} should be preserved",
            )
            self.assertEqual(
                payload["error"]["type"],
                expected_type,
                f"status {status_code} should map to {expected_type}",
            )
            self.assertEqual(
                payload["error"]["message"],
                "Internal server error",
                f"status {status_code} must scrub the message; got {payload['error']['message']!r}",
            )

    def test_non_streaming_response_includes_thinking_block(self):
        """When the OpenAI response carries reasoning_content, the Anthropic response has a thinking block first."""
        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "the answer is 4",
                            "reasoning_content": "2 + 2 = 4",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                },
            }
        )
        anthropic_response = self._serving()._convert_response(response)
        # thinking block first, then text block
        self.assertEqual(anthropic_response.content[0].type, "thinking")
        self.assertEqual(anthropic_response.content[0].thinking, "2 + 2 = 4")
        self.assertEqual(anthropic_response.content[1].type, "text")
        self.assertEqual(anthropic_response.content[1].text, "the answer is 4")

    def test_request_thinking_disabled_invokes_apply_reasoning_enabled(self):
        """``thinking={"type": "disabled"}`` must flip the reasoning toggle off."""
        serving = self._serving()
        request = self._anthropic_request(thinking={"type": "disabled"}, stream=False)
        serving._convert_to_chat_completion_request(request)
        self.assertEqual(serving.openai_serving_chat.apply_reasoning_calls, [False])

    def test_request_thinking_enabled_with_budget_tokens_logs_warning(self):
        """A supplied ``budget_tokens`` on ``enabled`` is accepted but we log
        a WARNING because the local backend has no equivalent hard-cap knob —
        the budget is not enforced."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(
            thinking={"type": "enabled", "budget_tokens": 2048}, stream=False
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.serving", level=logging.WARNING
        ) as log:
            serving._convert_to_chat_completion_request(request)
        self.assertEqual(serving.openai_serving_chat.apply_reasoning_calls, [True])
        self.assertTrue(
            any("budget_tokens=2048" in r and "not enforced" in r for r in log.output),
            f"expected unenforced-budget warning: {log.output}",
        )

    def test_request_thinking_enabled_without_budget_tokens_is_accepted(self):
        """Issue 124 compat: ``enabled`` with ``budget_tokens`` absent or null
        parses and enables reasoning through the same path. The backend never
        enforced a budget, so a missing one must not 400 — and we leave the
        budget unset instead of fabricating a number."""
        serving = self._serving()
        for thinking in (
            {"type": "enabled"},
            {"type": "enabled", "budget_tokens": None},
        ):
            with self.subTest(thinking=thinking):
                request = self._anthropic_request(thinking=thinking, stream=False)
                self.assertIsNone(request.thinking.budget_tokens)
                serving._convert_to_chat_completion_request(request)
        self.assertEqual(
            serving.openai_serving_chat.apply_reasoning_calls, [True, True]
        )

    def test_request_thinking_enabled_budget_below_min_is_rejected(self):
        """SDK doc: ``budget_tokens`` must be >= 1024."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._anthropic_request(
                thinking={"type": "enabled", "budget_tokens": 512}, stream=False
            )
        self.assertIn("1024", str(ctx.exception))

    def test_request_thinking_disabled_with_display_is_rejected(self):
        """SDK ``ThinkingConfigDisabledParam`` has no ``display`` field."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._anthropic_request(
                thinking={"type": "disabled", "display": "omitted"}, stream=False
            )
        self.assertIn("display", str(ctx.exception))

    def test_request_thinking_disabled_with_budget_is_rejected(self):
        """SDK ``ThinkingConfigDisabledParam`` has no ``budget_tokens`` field."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._anthropic_request(
                thinking={"type": "disabled", "budget_tokens": 2048}, stream=False
            )
        self.assertIn("budget_tokens", str(ctx.exception))

    def test_request_thinking_adaptive_with_budget_is_rejected(self):
        """SDK ``ThinkingConfigAdaptiveParam`` has no ``budget_tokens`` field."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._anthropic_request(
                thinking={"type": "adaptive", "budget_tokens": 2048}, stream=False
            )
        self.assertIn("budget_tokens", str(ctx.exception))

    def test_request_thinking_adaptive_is_treated_as_enabled(self):
        """Claude 4.7 ``thinking.type='adaptive'`` (the SDK default for
        unknown models) must be accepted and routed to ``apply_reasoning_enabled(True)``.
        """
        serving = self._serving()
        request = self._anthropic_request(thinking={"type": "adaptive"}, stream=False)
        serving._convert_to_chat_completion_request(request)
        self.assertEqual(serving.openai_serving_chat.apply_reasoning_calls, [True])

    def test_request_thinking_display_omitted_logs_warning_but_still_enables(self):
        """``thinking.display='omitted'`` is accepted; reasoning stays on
        because we cannot suppress reasoning text from the OpenAI stream.
        A supplied ``budget_tokens`` is still validated normally."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(
            thinking={
                "type": "enabled",
                "budget_tokens": 1024,
                "display": "omitted",
            },
            stream=False,
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.serving", level=logging.WARNING
        ) as log:
            serving._convert_to_chat_completion_request(request)
        self.assertEqual(serving.openai_serving_chat.apply_reasoning_calls, [True])
        self.assertTrue(any("omitted" in r for r in log.output))

    def test_request_output_config_effort_maps_to_reasoning_effort(self):
        """``output_config.effort`` rows map onto ``reasoning_effort``."""
        for anthropic_effort, openai_effort in [
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "max"),  # OpenAI Literal has no xhigh
            ("max", "max"),
        ]:
            with self.subTest(anthropic_effort=anthropic_effort):
                serving = self._serving()
                request = self._anthropic_request(
                    output_config={"effort": anthropic_effort}, stream=False
                )
                chat_request = serving._convert_to_chat_completion_request(request)
                self.assertEqual(chat_request.reasoning_effort, openai_effort)

    def test_request_output_config_task_budget_is_logged_not_enforced(self):
        """``task_budget`` is a soft hint; ``max_tokens`` is the hard cap."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(
            output_config={"task_budget": {"type": "tokens", "total": 32768}},
            stream=False,
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.serving", level=logging.INFO
        ) as log:
            chat_request = serving._convert_to_chat_completion_request(request)
        # max_tokens is untouched
        self.assertEqual(chat_request.max_tokens, 16)
        self.assertTrue(any("task_budget" in r and "32768" in r for r in log.output))

    def test_request_betas_is_accepted_and_logged(self):
        """``betas`` is accepted and logged; the local backend has no beta system."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(betas=["thinking-2025-08-04"], stream=False)
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.serving", level=logging.INFO
        ) as log:
            serving._convert_to_chat_completion_request(request)
        self.assertTrue(any("thinking-2025-08-04" in r for r in log.output))

    def test_assistant_thinking_history_is_rewrapped_for_chat_template(self):
        """Past-turn thinking blocks get re-emitted via wrap_reasoning_history."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "ponder"},
                        {"type": "text", "text": "hello"},
                    ],
                },
                {"role": "user", "content": "again"},
            ],
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        # ``ChatCompletionRequest.messages`` is a list of Pydantic
        # ChatCompletionMessage*Param instances; access via attributes.
        assistant_msg = next(m for m in chat_request.messages if m.role == "assistant")
        content = assistant_msg.content
        # Reasoning history sits in front; the thinking block itself is dropped
        # from the prompt so its text is not duplicated.
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict):
                    texts.append(part.get("text", ""))
                else:
                    texts.append(getattr(part, "text", "") or "")
        else:
            texts = [content]
        joined = "\n".join(texts)
        self.assertIn("<think>", joined)
        self.assertIn("ponder", joined)
        self.assertNotIn("<think>\nponder\n</think>\nponder", joined)

    def test_assistant_thinking_history_uses_native_reasoning_content(self):
        """Channel-framing encoders take thinking history as ``reasoning_content``.

        Splicing the detector's markers into content would nest a reasoning block
        inside the content channel and leave the real one empty, training the
        model to emit raw markers as visible text.
        """

        class _NativeOpenAI(_FakeOpenAIServingChat):
            native_reasoning_history = True

            def wrap_reasoning_history(self, text):
                raise AssertionError("must not rewrap for a native-history encoder")

        serving = AnthropicServing(_NativeOpenAI())
        request = self._anthropic_request(
            stream=False,
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "ponder"},
                        {"type": "text", "text": "hello"},
                    ],
                },
                {"role": "user", "content": "again"},
            ],
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        assistant_msg = next(m for m in chat_request.messages if m.role == "assistant")
        self.assertEqual(assistant_msg.reasoning_content, "ponder")
        self.assertEqual(assistant_msg.content, "hello")

    def test_thinking_only_turn_keeps_native_reasoning_content(self):
        """An assistant turn that is only thinking still carries its reasoning."""

        class _NativeOpenAI(_FakeOpenAIServingChat):
            native_reasoning_history = True

        serving = AnthropicServing(_NativeOpenAI())
        request = self._anthropic_request(
            stream=False,
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "ponder"}],
                },
                {"role": "user", "content": "again"},
            ],
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        assistant_msg = next(m for m in chat_request.messages if m.role == "assistant")
        self.assertEqual(assistant_msg.reasoning_content, "ponder")
        self.assertEqual(assistant_msg.content, "")

    def test_redacted_thinking_history_is_rejected(self):
        """``redacted_thinking`` cannot be rendered by local parsers."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": "opaque"},
                    ],
                },
            ],
        )
        with self.assertRaises(ValueError):
            serving._convert_to_chat_completion_request(request)

    def test_stream_text_then_thinking_closes_text_block(self):
        """Text deltas followed by reasoning_content must close the text block before opening thinking."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": "Direct"})]),
                _chunk([_choice({"reasoning_content": "but wait, let me think"})]),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        block_events = [
            (event["type"], event.get("content_block", {}).get("type"))
            for event in events
            if event["type"].startswith("content_block")
        ]
        # text (start, delta, stop) then thinking (start, delta, stop)
        self.assertEqual(block_events[0], ("content_block_start", "text"))
        text_stop_idx = next(
            i for i, ev in enumerate(block_events) if ev == ("content_block_stop", None)
        )
        thinking_start_idx = next(
            i
            for i, ev in enumerate(block_events)
            if ev == ("content_block_start", "thinking")
        )
        self.assertLess(text_stop_idx, thinking_start_idx)

    def test_stream_consecutive_tool_calls_get_separate_blocks(self):
        """Two tool_use calls in sequence must occupy distinct content_block indices."""
        serving = self._serving(
            [
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_a",
                                        "function": {
                                            "name": "alpha",
                                            "arguments": '{"x":1}',
                                        },
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 1,
                                        "id": "call_b",
                                        "function": {
                                            "name": "beta",
                                            "arguments": '{"y":2}',
                                        },
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk([_choice({}, finish_reason="tool_calls")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        starts = [
            (e["index"], e["content_block"]["name"])
            for e in events
            if e["type"] == "content_block_start"
        ]
        stops = [e["index"] for e in events if e["type"] == "content_block_stop"]
        deltas = [
            (e["index"], e["delta"].get("partial_json"))
            for e in events
            if e["type"] == "content_block_delta"
            and e["delta"].get("type") == "input_json_delta"
        ]
        # Each tool gets its own start, its own stop, and its own
        # argument delta — without the fix, beta's args were appended
        # to alpha's index 0 block.
        self.assertEqual(starts, [(0, "alpha"), (1, "beta")])
        self.assertEqual(stops, [0, 1])
        self.assertEqual(deltas, [(0, '{"x":1}'), (1, '{"y":2}')])

    def test_stream_finish_chunk_with_payload_emits_delta(self):
        """A chunk carrying both finish_reason and content must not drop the content."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant"})]),
                _chunk([_choice({"content": "last token"}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        text_deltas = [
            e["delta"]["text"]
            for e in events
            if e["type"] == "content_block_delta"
            and e["delta"].get("type") == "text_delta"
        ]
        self.assertEqual(text_deltas, ["last token"])
        # And stop_reason still travels via message_delta
        message_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "end_turn")

    def test_stream_empty_completion_with_finish_reason_emits_message_delta(self):
        """An empty stream with a finish_reason is a legitimate stop, not api_error."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant"})]),
                _chunk([_choice({}, finish_reason="length")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [e["type"] for e in events]
        self.assertIn("message_start", types)
        self.assertIn("message_delta", types)
        self.assertIn("message_stop", types)
        self.assertNotIn("error", types)
        message_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "max_tokens")

    def test_stream_no_finish_no_content_still_emits_api_error(self):
        """Backend that drops both content and finish_reason is genuinely broken."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant"})]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [e["type"] for e in events]
        self.assertIn("error", types)
        err = next(e for e in events if e["type"] == "error")
        self.assertEqual(err["error"]["type"], "api_error")

    def test_stream_upstream_error_envelope_is_forwarded(self):
        """OpenAI handler streaming-error JSON must surface real type/message."""
        upstream_error = {
            "error": {
                "object": "error",
                "message": "context length exceeded",
                "type": "BadRequestError",
                "code": 400,
            }
        }
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant"})]),
                f"data: {json.dumps(upstream_error)}\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        err = next(e for e in events if e["type"] == "error")
        self.assertEqual(err["error"]["type"], "invalid_request_error")
        self.assertEqual(err["error"]["message"], "context length exceeded")
        # message_stop must still close the stream
        self.assertEqual(events[-1]["type"], "message_stop")

    def test_stream_parse_failure_closes_open_content_block(self):
        """Unparsable mid-stream chunk must still close any open content_block."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": "first"})]),
                "data: {not-json\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [e["type"] for e in events]
        # Sequence: message_start, content_block_start, content_block_delta,
        # content_block_stop, error, message_stop
        self.assertIn("content_block_start", types)
        self.assertEqual(
            types.count("content_block_stop"),
            types.count("content_block_start"),
            f"unbalanced block events: {types}",
        )
        self.assertIn("error", types)
        self.assertEqual(types[-1], "message_stop")

    def test_stream_pre_first_chunk_value_error_emits_envelope(self):
        """ValueError before any chunk must yield a clean Anthropic error sequence."""

        class _RaisingOpenAI(_FakeOpenAIServingChat):
            def _generate_chat_stream(
                self, adapted_request, processed_request, raw_request
            ):
                async def _gen():
                    raise ValueError("tokenization failed")
                    yield  # pragma: no cover

                return _gen()

        serving = AnthropicServing(_RaisingOpenAI())
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "message_start")
        self.assertIn("error", types)
        err = next(e for e in events if e["type"] == "error")
        self.assertEqual(err["error"]["type"], "invalid_request_error")
        self.assertIn("tokenization failed", err["error"]["message"])
        self.assertEqual(types[-1], "message_stop")

    def test_server_tool_only_with_tool_choice_any_raises_400(self):
        """A request with only server-side tools cannot honor tool_choice=any."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            tool_choice={"type": "any"},
        )
        with self.assertRaises(ValueError) as ctx:
            serving._convert_to_chat_completion_request(request)
        self.assertIn("tool_choice", str(ctx.exception))

    def test_tool_choice_named_custom_tool_is_resolved(self):
        """tool_choice={type:'tool', name:'X'} where X is a custom tool wires through."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            tools=[
                {
                    "type": "custom",
                    "name": "lookup",
                    "input_schema": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "lookup"},
        )
        # Must not AttributeError: Tool.function is a Pydantic model, not a
        # dict — access must be via .name, never .get("name").
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(chat_request.tool_choice.type, "function")
        self.assertEqual(chat_request.tool_choice.function.name, "lookup")

    def test_tool_choice_named_unknown_tool_raises_400(self):
        """tool_choice={type:'tool', name:'X'} where X is missing must raise."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            tools=[
                {
                    "type": "custom",
                    "name": "lookup",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            tool_choice={"type": "tool", "name": "nonexistent"},
        )
        with self.assertRaises(ValueError) as ctx:
            serving._convert_to_chat_completion_request(request)
        self.assertIn("nonexistent", str(ctx.exception))

    def test_convert_response_non_streaming_empty_content_keeps_block(self):
        """Empty-string completion must still produce a content list of len 1."""
        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 0,
                    "total_tokens": 5,
                },
            }
        )
        serving = self._serving()
        anthropic_response = serving._convert_response(response)
        self.assertEqual(len(anthropic_response.content), 1)
        self.assertEqual(anthropic_response.content[0].type, "text")
        self.assertEqual(anthropic_response.content[0].text, "")

    def test_error_response_does_not_leak_exception_name(self):
        """``error.type`` must stay in Anthropic's documented literal set."""
        serving = self._serving()
        response = serving._error_response(
            status_code=500,
            error_type="api_error",
            message="Internal server error",
            exception_name="KeyError",
        )
        body = json.loads(bytes(response.body).decode())
        self.assertEqual(body["error"]["type"], "api_error")

    def test_user_message_text_tool_text_preserves_order(self):
        """User message [text, tool_result, text] must stay user→tool→user on the wire."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_x",
                            "content": "ok",
                        },
                        {"type": "text", "text": "second"},
                    ],
                }
            ],
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        # chat_request.messages items are Pydantic ChatCompletionMessage*Param
        # variants — use attribute access, not subscripts.
        roles = [m.role for m in chat_request.messages]
        self.assertEqual(roles, ["user", "tool", "user"])
        self.assertEqual(chat_request.messages[0].content, "first")
        self.assertEqual(chat_request.messages[2].content, "second")

    def test_empty_text_assistant_turn_preserves_role_alternation(self):
        """Assistant turn with only empty text must NOT vanish from the wire."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            messages=[
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                {"role": "user", "content": "u2"},
            ],
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        roles = [m.role for m in chat_request.messages]
        # Without the fix this collapses to ['user', 'user'] and breaks
        # strict role-alternation chat templates (qwen, llama, mistral).
        self.assertEqual(roles, ["user", "assistant", "user"])

    def test_in_messages_system_merged_when_template_requires_first(self):
        """When the chat template rejects mid-conversation ``role: "system"``
        (e.g. Qwen's system-first guard), the converter folds the inline
        system turn into the leading system block so the template doesn't
        400. The request object itself is no longer mutated — detection runs
        in the serving layer on conversion."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "Reply with exactly: OK"},
                {"role": "user", "content": "go"},
            ],
        )
        self.assertIsNone(request.system)
        self.assertEqual([m.role for m in request.messages], ["user", "system", "user"])
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(
            [m.role for m in chat_request.messages], ["system", "user", "user"]
        )
        self.assertEqual(chat_request.messages[0].content, "Reply with exactly: OK")

    def test_in_messages_system_merged_with_top_level_when_merge(self):
        """On the merge path, a top-level ``system`` field and a mid-conversation
        system turn are joined into the leading system block; top-level text
        comes first."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            system="You are terse.",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "One word only."},
                {"role": "user", "content": "go"},
            ],
        )
        self.assertEqual(request.system, "You are terse.")
        self.assertEqual([m.role for m in request.messages], ["user", "system", "user"])
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(
            [m.role for m in chat_request.messages], ["system", "user", "user"]
        )
        self.assertEqual(
            chat_request.messages[0].content, "You are terse.\nOne word only."
        )

    def test_in_messages_system_passed_through_when_template_allows_inline(self):
        """When the chat template renders ``role: "system"`` at any position
        (GLM / Kimi / Qwen3), the inline system turn stays at its original
        position — preserving the prefix cache and the request's structure."""
        serving = self._serving(chat_template=self.INLINE_SYSTEM_TEMPLATE)
        self.assertFalse(serving._merge_inline_system)
        request = self._anthropic_request(
            stream=False,
            system="You are terse.",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "Reply with exactly: OK"},
                {"role": "user", "content": "go"},
            ],
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(
            [m.role for m in chat_request.messages],
            ["system", "user", "system", "user"],
        )
        self.assertEqual(chat_request.messages[0].content, "You are terse.")
        self.assertEqual(chat_request.messages[2].content, "Reply with exactly: OK")

    def test_top_level_system_only_is_unchanged(self):
        """A request with only the top-level ``system`` field (no in-messages
        system turn) is unaffected on both detection paths: the system field is
        preserved verbatim and the dialogue order is untouched. Guards the
        common multi-turn path against regressions."""
        for template in (None, self.INLINE_SYSTEM_TEMPLATE):
            serving = self._serving(chat_template=template)
            request = self._anthropic_request(
                stream=False,
                system="You are a helpful assistant.",
                messages=[
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Hello Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
            )
            self.assertEqual(request.system, "You are a helpful assistant.")
            self.assertEqual(
                [m.role for m in request.messages], ["user", "assistant", "user"]
            )
            chat_request = serving._convert_to_chat_completion_request(request)
            self.assertEqual(
                [m.role for m in chat_request.messages],
                ["system", "user", "assistant", "user"],
            )
            self.assertEqual(
                chat_request.messages[0].content, "You are a helpful assistant."
            )

    def test_constructed_message_objects_merged_on_merge_path(self):
        """Requests built programmatically with ``AnthropicMessage`` objects
        (e.g. ``handle_count_tokens``) also get inline system folded into the
        leading block on the merge path."""
        serving = self._serving()
        request = AnthropicMessagesRequest(
            model="m",
            max_tokens=8,
            messages=[
                AnthropicMessage(role="user", content="hi"),
                AnthropicMessage(role="system", content="be terse"),
                AnthropicMessage(role="user", content="go"),
            ],
        )
        self.assertEqual([m.role for m in request.messages], ["user", "system", "user"])
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(
            [m.role for m in chat_request.messages], ["system", "user", "user"]
        )
        self.assertEqual(chat_request.messages[0].content, "be terse")

    def test_thinking_history_drop_on_missing_detector(self):
        """Replaying a thinking block on a non-reasoning model should not 400."""

        class _NoDetectorOpenAI(_FakeOpenAIServingChat):
            def wrap_reasoning_history(self, text):
                raise ValueError("no reasoning detector is configured")

        serving = AnthropicServing(_NoDetectorOpenAI())
        request = self._anthropic_request(
            stream=False,
            messages=[
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "I think..."}],
                },
                {"role": "user", "content": "follow-up"},
            ],
        )
        # Must convert successfully; the thinking block is silently dropped.
        chat_request = serving._convert_to_chat_completion_request(request)
        roles = [m.role for m in chat_request.messages]
        self.assertIn("user", roles)
        # The assistant turn was rendered (as empty placeholder) so
        # alternation is preserved.
        self.assertIn("assistant", roles)

    def test_stop_reason_content_filter_falls_back_with_warning(self):
        """Unmapped OpenAI finish_reasons default to 'end_turn' + log a warning.

        ``content_filter`` and ``abort`` have no entry in STOP_REASON_MAP
        because Anthropic's ``stop_reason`` Literal (end_turn/max_tokens/
        stop_sequence/tool_use) has no perfect target. The fallback must
        produce a spec-valid stop_reason and a WARNING so operators don't
        silently lose the safety/abort signal.
        """
        import logging

        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "content_filter",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )
        serving = self._serving()
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.serving", level=logging.WARNING
        ) as log:
            anthropic_response = serving._convert_response(response)
        self.assertEqual(anthropic_response.stop_reason, "end_turn")
        self.assertTrue(
            any("content_filter" in rec for rec in log.output),
            f"expected a warning mentioning the unmapped finish_reason: {log.output}",
        )


class _CountTokensOpenAI(_FakeOpenAIServingChat):
    """OpenAI handler stub for ``/v1/messages/count_tokens``."""

    def __init__(self, chat_template=None):
        super().__init__(chat_template=chat_template)
        self.tokenizer_manager.model_config = SimpleNamespace(is_multimodal=False)
        self.converted: ChatCompletionRequest | None = None

    def _process_messages(self, chat_request, is_multimodal):
        self.converted = chat_request
        _render_chat_request(
            self.tokenizer_manager.tokenizer.chat_template, chat_request
        )
        return SimpleNamespace(prompt_ids=[1, 2, 3])


class TestNestedToolReferenceCompatibility(unittest.TestCase):
    """Nested ``tool_reference`` blocks against the real pinned Pennyroyal template.

    The template is loaded as opaque bytes through the shared launcher pin (no
    version is hardcoded here, no template is edited) and rendered with the same
    content processing the chat-completions route applies. Assertions stay on
    converted structures, indexes and hashes so a failure never prints a prompt.
    """

    ISSUE_137_PAYLOAD = {
        "model": "pennyroyal",
        "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "tool_reference", "tool_name": "y"}],
                    },
                    {"type": "text", "text": "ok"},
                ],
            },
        ],
    }

    @classmethod
    def setUpClass(cls):
        cls.template = _pennyroyal_chat_template()

    def _serving(self, chat_template=None):
        return AnthropicServing(_FakeOpenAIServingChat(chat_template=chat_template))

    def _user_tool_turn(self, content, **overrides):
        """The reporter conversation with ``content`` as the tool answer."""
        payload = json.loads(json.dumps(self.ISSUE_137_PAYLOAD))
        payload["messages"][2]["content"][0]["content"] = content
        payload.update(overrides)
        return AnthropicMessagesRequest.model_validate(payload)

    def _messages(self, chat_request):
        return chat_request.model_dump(exclude_none=True)["messages"]

    def test_reporter_payload_renders_on_pinned_template(self):
        request = AnthropicMessagesRequest.model_validate(self.ISSUE_137_PAYLOAD)
        chat_request = self._serving(self.template)._convert_to_chat_completion_request(
            request
        )
        # Precondition: this template takes structured content parts, so the
        # reference part really does reach ``render_content``.
        self.assertEqual(detect_jinja_template_content_format(self.template), "openai")
        messages = self._messages(chat_request)
        self.assertEqual(
            [message["role"] for message in messages],
            ["user", "assistant", "tool", "user"],
        )
        self.assertEqual(
            messages[2],
            {"role": "tool", "tool_call_id": "t1", "content": "Tool reference: y"},
        )
        prompt = _render_chat_request(self.template, chat_request)
        self.assertTrue(
            "Tool reference: y" in prompt,
            "flattened reference is missing from the rendered prompt",
        )
        # The referenced tool has no schema anywhere in this request, and the
        # adapter must not invent one for it.
        self.assertNotIn("tools", chat_request.model_dump(exclude_none=True))

    def test_reference_names_order_and_tool_call_association_survive(self):
        request = self._user_tool_turn(
            [
                {"type": "text", "text": "before-marker"},
                {"type": "tool_reference", "name": "alpha-marker"},
                {"type": "tool_reference", "tool_name": "beta-marker"},
                {"type": "text", "text": "after-marker"},
            ],
        )
        messages = self._messages(
            self._serving(self.template)._convert_to_chat_completion_request(request)
        )
        # One tool answer carrying every part in place, then the turn's own text.
        self.assertEqual(
            [message["role"] for message in messages],
            ["user", "assistant", "tool", "user"],
        )
        self.assertEqual(
            messages[2],
            {
                "role": "tool",
                "tool_call_id": "t1",
                "content": [
                    {"type": "text", "text": "before-marker"},
                    {"type": "text", "text": "Tool reference: alpha-marker"},
                    {"type": "text", "text": "Tool reference: beta-marker"},
                    {"type": "text", "text": "after-marker"},
                ],
            },
        )
        prompt = _render_chat_request(
            self.template,
            ChatCompletionRequest.model_validate(
                {
                    "model": "pennyroyal",
                    "max_tokens": 8,
                    "messages": messages,
                }
            ),
        )
        order = [
            prompt.index(marker)
            for marker in (
                "before-marker",
                "Tool reference: alpha-marker",
                "Tool reference: beta-marker",
                "after-marker",
            )
        ]
        self.assertEqual(order, sorted(order), "reference parts lost their order")
        self.assertLess(order[-1], prompt.index("ok"), "user text lost its position")

    def test_capable_template_keeps_structured_reference_expansion(self):
        """The distinction is what the selected template can render, not the
        adapter's own preference: a capable template keeps the structured part
        and expands the supplied schema."""
        capable = TestAnthropicServing.GLM_TOOL_RESULT_TEMPLATE.replace(
            "{{ tool.function.name }}", "{{ tool.function | tojson }}"
        )
        self.assertTrue(detect_tool_reference_support(capable))
        self.assertFalse(detect_tool_reference_support(self.template))
        request = self._user_tool_turn(
            [{"type": "tool_reference", "tool_name": "Bash"}],
            tools=[
                {
                    "name": "Bash",
                    "description": "Run a shell command",
                    "input_schema": {"type": "object", "properties": {}},
                    "defer_loading": True,
                }
            ],
        )
        chat_request = self._serving(capable)._convert_to_chat_completion_request(
            request
        )
        self.assertEqual(
            self._messages(chat_request)[2]["content"],
            [{"type": "tool_reference", "name": "Bash"}],
        )
        prompt = _render_chat_request(capable, chat_request)
        expanded_schema = prompt.split("<tools>", 1)[1].split("</tools>", 1)[0]
        self.assertEqual(
            json.loads(expanded_schema), chat_request.tools[0].function.model_dump()
        )
        # The fallback does not remove supplied schemas: the pinned template
        # already renders them even when it cannot render a reference part.
        fallback = self._serving(self.template)._convert_to_chat_completion_request(
            request
        )
        self.assertEqual(fallback.tools, chat_request.tools)
        fallback_prompt = _render_chat_request(self.template, fallback)
        self.assertTrue("Run a shell command" in fallback_prompt)
        self.assertTrue("Tool reference: Bash" in fallback_prompt)

    def test_native_expansion_uses_normalized_tool_history(self):
        capable = (
            TestDetectToolReferenceSupport.NORMALIZED_HISTORY_GUARD
            + TestDetectToolReferenceSupport.NORMALIZED_TOOLS_GUARD
            + TestAnthropicServing.GLM_TOOL_RESULT_TEMPLATE.replace(
                "{{ tool.function.name }}", "{{ tool.function | tojson }}"
            )
        )
        request = self._user_tool_turn(
            [
                {"type": "text", "text": "before-one"},
                {"type": "text", "text": "before-two"},
                {"type": "tool_reference", "tool_name": "Bash"},
            ],
            tools=[
                {
                    "name": "Bash",
                    "description": "Supplied schema survives native rendering",
                    "input_schema": {"type": "object", "properties": {}},
                    "defer_loading": True,
                }
            ],
        )
        converted = self._serving(capable)._convert_to_chat_completion_request(request)
        self.assertEqual(
            self._messages(converted)[3]["content"],
            [{"type": "tool_reference", "name": "Bash"}],
        )
        prompt = _render_chat_request(capable, converted)
        self.assertTrue("before-one before-two" in prompt)
        schema = prompt.split("<tools>", 1)[1].split("</tools>", 1)[0]
        self.assertEqual(json.loads(schema), converted.tools[0].function.model_dump())

    def test_native_expansion_receives_tokenizer_special_tokens(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast

        capable = "{{ bos_token + 'prefix' }}" + (
            TestAnthropicServing.GLM_TOOL_RESULT_TEMPLATE.replace(
                "{{ tool.function.name }}", "{{ tool.function | tojson }}"
            )
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                WordLevel({"[UNK]": 0, "[BOS]": 1}, unk_token="[UNK]")
            ),
            bos_token="[BOS]",
            unk_token="[UNK]",
            chat_template=capable,
        )
        openai_serving = _FakeOpenAIServingChat(chat_template=capable)
        openai_serving.tokenizer_manager.tokenizer = tokenizer
        request = self._user_tool_turn(
            [{"type": "tool_reference", "tool_name": "Bash"}],
            tools=[
                {
                    "name": "Bash",
                    "description": "Schema expanded with tokenizer context",
                    "input_schema": {"type": "object", "properties": {}},
                    "defer_loading": True,
                }
            ],
        )
        # The real tokenizer renderer supports this template independently of
        # whether the adapter's capability probe recognizes it.
        native = self._serving()._convert_to_chat_completion_request(request)
        prompt = _render_chat_request(capable, native, tokenizer=tokenizer)
        self.assertTrue(prompt.startswith(tokenizer.bos_token + "prefix"))
        schema = prompt.split("<tools>", 1)[1].split("</tools>", 1)[0]
        self.assertEqual(json.loads(schema), native.tools[0].function.model_dump())

        converted = AnthropicServing(
            openai_serving
        )._convert_to_chat_completion_request(request)
        self.assertEqual(
            self._messages(converted)[2]["content"],
            [{"type": "tool_reference", "name": "Bash"}],
        )
        self.assertEqual(
            _render_chat_request(capable, converted, tokenizer=tokenizer), prompt
        )

    def test_unaffected_requests_render_the_same_prompt(self):
        """Text/image answers and the supplied tools are untouched, whether or
        not the selected template renders references."""
        request = self._user_tool_turn(
            [
                {"type": "text", "text": "plain answer"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "aGVsbG8=",
                    },
                },
            ],
            tools=[
                {
                    "name": "y",
                    "input_schema": {"type": "object", "properties": {}},
                    "defer_loading": True,
                }
            ],
        )
        flattening = self._serving(self.template)._convert_to_chat_completion_request(
            request
        )
        structured = self._serving(
            TestAnthropicServing.GLM_TOOL_RESULT_TEMPLATE
        )._convert_to_chat_completion_request(request)
        self.assertEqual(
            flattening.model_dump(exclude_none=True),
            structured.model_dump(exclude_none=True),
        )
        hashes = {
            hashlib.sha256(
                _render_chat_request(self.template, converted).encode()
            ).hexdigest()
            for converted in (flattening, structured)
        }
        self.assertEqual(
            len(hashes), 1, "rendered prompt changed for a request with no reference"
        )
        dumped = flattening.model_dump(exclude_none=True)
        self.assertEqual([tool["function"]["name"] for tool in dumped["tools"]], ["y"])
        self.assertEqual(
            [tool["function"]["parameters"] for tool in dumped["tools"]],
            [{"type": "object", "properties": {}}],
        )

    def test_document_and_mcp_blocks_stay_validation_errors(self):
        for block in (
            {"type": "unknown", "data": "do not serialize unsupported blocks"},
            {"type": "document", "source": {"type": "text", "data": "x"}},
            {"type": "mcp_tool_use", "id": "t2", "name": "m", "input": {}},
            {"type": "mcp_tool_result", "tool_use_id": "t2", "content": "x"},
        ):
            with self.subTest(block=block["type"]):
                with self.assertRaises(ValidationError):
                    self._user_tool_turn([block])
                with self.assertRaises(ValidationError):
                    AnthropicMessagesRequest.model_validate(
                        {
                            "model": "pennyroyal",
                            "max_tokens": 8,
                            "messages": [{"role": "user", "content": [block]}],
                        }
                    )

    def test_count_tokens_conversion_shares_the_correction(self):
        openai_serving = _CountTokensOpenAI(chat_template=self.template)
        serving = AnthropicServing(openai_serving)
        count_request = AnthropicCountTokensRequest.model_validate(
            {
                key: value
                for key, value in self.ISSUE_137_PAYLOAD.items()
                if key != "max_tokens"
            }
        )
        response = asyncio.run(serving.handle_count_tokens(count_request, object()))
        self.assertEqual(json.loads(response.body), {"input_tokens": 3})
        self.assertEqual(
            [
                message["content"]
                for message in openai_serving.converted.model_dump(exclude_none=True)[
                    "messages"
                ]
                if message["role"] == "tool"
            ],
            ["Tool reference: y"],
        )

    def test_every_pennyroyal_profile_sources_the_shared_pin(self):
        sourcing = {
            path.name
            for path in _PENNYROYAL_CONFIGS.glob("serve-*.sh")
            if "chat-template.sh" in path.read_text(encoding="utf-8")
        }
        self.assertTrue(
            any("flash-next" in name for name in sourcing)
            and any("27b" in name for name in sourcing),
            f"both supported profiles must source the shared template pin: {sorted(sourcing)}",
        )


class TestDetectToolReferenceSupport(unittest.TestCase):
    """Template capability for nested ``tool_reference`` parts."""

    NORMALIZED_HISTORY_GUARD = """
{%- for message in messages -%}
{%- if message.role == 'assistant' -%}
{%- for call in message.tool_calls | default([]) -%}
{%- for key, value in call.function.arguments.items() -%}
{{- key }}={{ value }}
{%- endfor -%}
{%- endfor -%}
{%- elif message.role == 'tool' and message.content is not string
    and message.content[0].type == 'text' -%}
{{- raise_exception('ordinary tool text must be a string') }}
{%- endif -%}
{%- endfor -%}
"""

    NORMALIZED_TOOLS_GUARD = """
{%- for tool in tools -%}
{%- if tool.function.defer_loading is not sameas true
    or tool.function.strict is not sameas false
    or tool.function.description is not string -%}
{{- raise_exception('tool schema defaults must be normalized') }}
{%- endif -%}
{%- endfor -%}
"""

    EXPANDING = """
{%- for message in messages -%}
{%- if message.role == 'system' and 'image' in message.content -%}
{{- raise_exception('system message cannot hold an image') }}
{%- endif -%}
{%- if message.role == 'tool' and message.content is not string -%}
{%- for part in message.content -%}
{%- if part.type == 'tool_reference' -%}
{%- for tool in tools if tool.function.name == part.name -%}
[ref:{{ tool.function.name }}]
{%- endfor -%}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{%- endfor -%}
"""

    def test_expanding_template_is_supported(self):
        # ``raise_exception`` above is a global the real renderer provides; a
        # probe environment without it would call this template broken.
        self.assertTrue(detect_tool_reference_support(self.EXPANDING))

    def test_normalized_tool_history_is_supported(self):
        self.assertTrue(
            detect_tool_reference_support(
                self.NORMALIZED_HISTORY_GUARD + self.EXPANDING
            )
        )

    def test_normalized_deferred_tool_schema_is_supported(self):
        deferred_expansion = self.EXPANDING.replace(
            "tool.function.name == part.name",
            "tool.function.name == part.name and tool.function.defer_loading",
        )
        self.assertTrue(
            detect_tool_reference_support(
                self.NORMALIZED_TOOLS_GUARD + deferred_expansion
            )
        )

    def test_name_independent_placeholder_is_not_supported(self):
        # Emits a fixed marker for any reference: the name is lost either way,
        # so the caller's text fallback is the better prompt.
        placeholder = self.EXPANDING.replace("[ref:{{ tool.function.name }}]", "[ref]")
        self.assertFalse(detect_tool_reference_support(placeholder))

    def test_silently_dropped_reference_is_not_supported(self):
        dropping = (
            "{%- for message in messages -%}"
            "{%- if message.content is string -%}{{ message.content }}{%- endif -%}"
            "{%- endfor -%}"
        )
        self.assertFalse(detect_tool_reference_support(dropping))

    def test_raising_template_is_not_supported(self):
        raising = "{{ raise_exception('no tool answers here') }}"
        self.assertFalse(detect_tool_reference_support(raising))

    def test_renderer_globals_and_extensions_are_available(self):
        timing = self.EXPANDING + "{{ strftime_now('%f') }}"
        self.assertTrue(detect_tool_reference_support(timing))
        self.assertTrue(
            detect_tool_reference_support(
                "{% generation %}" + self.EXPANDING + "{% endgeneration %}"
            )
        )

    def test_serialized_references_are_not_native_support(self):
        for content in ("message.content", "message.content | tojson"):
            with self.subTest(content=content):
                self.assertFalse(
                    detect_tool_reference_support(
                        "{% for message in messages %}{{ " + content + " }}{% endfor %}"
                    )
                )

    def test_globally_rendered_schema_names_are_not_native_support(self):
        globally_listed = "{% for tool in tools %}{{ tool.function.name }}{% endfor %}"
        self.assertFalse(detect_tool_reference_support(globally_listed))
        self.assertTrue(detect_tool_reference_support(globally_listed + self.EXPANDING))

    def test_failed_control_does_not_establish_support(self):
        rejects_text = (
            "{% for message in messages if message.role == 'tool' %}"
            "{% if message.content is string %}"
            "{{ raise_exception('text is unsupported') }}{% endif %}{% endfor %}"
        )
        self.assertFalse(detect_tool_reference_support(rejects_text + self.EXPANDING))

    def test_no_template_is_not_supported(self):
        self.assertFalse(detect_tool_reference_support(None))
        self.assertFalse(detect_tool_reference_support(""))

    def test_pinned_pennyroyal_template_is_not_supported(self):
        self.assertFalse(detect_tool_reference_support(_pennyroyal_chat_template()))


class TestDetectInlineSystemSupport(unittest.TestCase):
    """Chat-template detection for mid-conversation system messages (#28883)."""

    def test_guarded_template_not_supported(self):
        guarded = (
            "{%- for message in messages %}"
            "{%- if message.role == 'system' and not loop.first %}"
            "{{- raise_exception('system must be first') }}"
            "{%- endif %}"
            "{%- endfor %}"
        )
        self.assertFalse(detect_inline_system_support(guarded))

    def test_inline_template_supported(self):
        inline = (
            "{%- for message in messages %}"
            "{{- message.role }}: {{ message.content }}\n"
            "{%- endfor %}"
        )
        self.assertTrue(detect_inline_system_support(inline))

    def test_silent_drop_template_not_supported(self):
        # Renders only the leading system; silently ignores later system turns.
        silent_drop = (
            "{%- if messages[0].role == 'system' %}"
            "{{ messages[0].content }}\n"
            "{%- endif %}"
            "{%- for message in messages %}"
            "{%- if message.role in ('user', 'assistant') %}"
            "{{ message.role }}: {{ message.content }}\n"
            "{%- endif %}"
            "{%- endfor %}"
        )
        self.assertFalse(detect_inline_system_support(silent_drop))

    def test_no_template_not_supported(self):
        self.assertFalse(detect_inline_system_support(None))
        self.assertFalse(detect_inline_system_support(""))


if __name__ == "__main__":
    unittest.main()
