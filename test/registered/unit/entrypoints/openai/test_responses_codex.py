"""Responses wire regressions for ordinary Codex tools and callbacks."""

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from utils import StreamFixture, engine_chunk, make_serving

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest, ResponsesResponse
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def function(name="run", **kwargs):
    return {
        "type": "function",
        "name": name,
        "parameters": {"type": "object"},
        **kwargs,
    }


def namespace(name, *tools):
    return {"type": "namespace", "name": name, "tools": list(tools)}


def request(**kwargs):
    return ResponsesRequest(
        **{"model": "pennyroyal", "input": "Do the work", "store": False, **kwargs}
    )


def qwen_serving(*, is_multimodal=False):
    """Use the pinned template unchanged through HF's real compiler/context."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast

    root = Path(__file__).resolve().parents[5]
    template = (root / "configs/pennyroyal/templates/froggeric-v22.5.jinja").read_text()
    serving = make_serving(is_multimodal=is_multimodal)
    # A lossless local byte vocabulary exercises the real HF rendering API
    # without requiring model weights or downloading a tokenizer in CPU CI.
    backend = Tokenizer(
        BPE({c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}, [])
    )
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = ByteLevelDecoder()
    serving.tokenizer_manager.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template=template,
    )
    serving.template_manager.chat_template_name = None
    serving.template_manager.jinja_template_content_format = "openai"
    serving.default_chat_template_kwargs = {}
    serving.tool_call_parser = "qwen3_coder"
    return serving


def qwen_call(name, parameters):
    return (
        "<tool_call>\n<function="
        + name
        + ">\n"
        + "".join(
            f"<parameter={key}>\n{value}\n</parameter>\n"
            for key, value in parameters.items()
        )
        + "</function>\n</tool_call>"
    )


def fixture(name):
    return json.loads((Path(__file__).parent / "fixtures" / name).read_text())


def install_engine_reply(serving, text, captured):
    async def generate(
        request_id, request_prompt, adapted_request, sampling_params, context, **kwargs
    ):
        captured.update(
            request=adapted_request, prompt=request_prompt, sampling=sampling_params
        )
        context.append_output(engine_chunk(text, finish=True))
        yield context

    # Only generation is mocked; protocol, template, media extraction, sampling
    # conversion and serialization all run their production paths.
    serving._generate_with_builtin_tools = generate


class CodexBoundaryTest(CustomTestCase):
    def test_namespaces_reach_chat_without_colliding_with_flat_names(self):
        serving = make_serving()
        req = request(
            tools=[
                function(),
                namespace("one", function()),
                namespace("two", function()),
                function("one.run"),
            ]
        )
        tools = serving._response_tools_to_chat_tools(req)
        self.assertEqual(len(tools), 4)
        self.assertEqual(len({t.function.name for t in tools}), 4)

    def test_custom_tool_becomes_string_input_function(self):
        serving = make_serving()
        req = request(
            tools=[
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "format": {
                        "type": "grammar",
                        "syntax": "lark",
                        "definition": "start: /.+/",
                    },
                }
            ]
        )
        tools = serving._response_tools_to_chat_tools(req)
        self.assertEqual(len(tools), 1)
        self.assertEqual(
            tools[0].function.parameters["properties"]["input"]["type"], "string"
        )
        self.assertIn("start: /.+/", tools[0].function.description)

    def test_malformed_supported_declarations_raise(self):
        for tool in (
            {"type": "custom"},
            {"type": "namespace", "tools": []},
            {"type": "namespace", "name": "x"},
            namespace("x", {"type": "custom"}),
            namespace("x", {"type": "namespace", "name": "y", "tools": []}),
        ):
            with self.subTest(tool=tool), self.assertRaises(ValueError):
                request(tools=[tool])

    def test_inline_inventory_stays_between_history_items_including_empty(self):
        serving = make_serving()
        req = request()
        req.input = [
            {"role": "user", "content": "before"},
            {
                "type": "additional_tools",
                "tools": [namespace("mcp", function("search"))],
            },
            {"role": "assistant", "content": "middle"},
            {"type": "additional_tools", "tools": []},
            {"role": "user", "content": "after"},
        ]
        messages = serving._construct_input_messages(req)
        self.assertEqual(
            [m["role"] for m in messages],
            ["user", "system", "assistant", "system", "user"],
        )
        self.assertEqual(len(messages[1]["tools"]), 1)
        self.assertEqual(messages[3]["tools"], [])

    def test_tool_result_content_array_keeps_image_and_text_order(self):
        serving = make_serving()
        parts = [
            {"type": "input_text", "text": "first"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            {"type": "input_text", "text": "last"},
        ]
        for kind in ("function_call_output", "custom_tool_call_output"):
            with self.subTest(kind=kind):
                normalized = serving._normalize_response_message_for_chat(
                    {"type": kind, "call_id": "call_a", "output": parts}
                )
                self.assertEqual(
                    [p["type"] for p in normalized["content"]],
                    ["text", "image_url", "text"],
                )
                self.assertEqual(normalized["tool_call_id"], "call_a")

    def test_stored_continuation_replays_reasoning_and_call_before_result(self):
        serving = make_serving()
        old = request()
        output = [
            {
                "id": "rs_a",
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "think"}],
            },
            {
                "id": "fc_a",
                "type": "function_call",
                "name": "search",
                "namespace": "mcp",
                "arguments": '{"query":"x"}',
                "call_id": "call_a",
            },
        ]
        previous = ResponsesResponse.from_request(
            old, {}, "pennyroyal", 0, output, "completed", None
        )
        serving.msg_store[previous.id] = [{"role": "user", "content": "before"}]
        req = request(previous_response_id=previous.id)
        req.input = [
            {"type": "function_call_output", "call_id": "call_a", "output": "found"}
        ]
        messages = serving._construct_input_messages(req, previous)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool"])
        self.assertEqual(messages[1]["reasoning_content"], "think")
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call_a")
        self.assertEqual(messages[2]["tool_call_id"], "call_a")

    def test_namespace_wire_identity_full_stream_and_stateless_replay(self):
        serving = qwen_serving()
        req = request(
            tools=[
                function(),
                namespace("one", function()),
                namespace("two", function()),
                function("one.run"),
            ]
        )
        tools = serving._response_tools_to_chat_tools(req)
        for tool, identity in zip(
            tools,
            (
                {"name": "run"},
                {"name": "run", "namespace": "one"},
                {"name": "run", "namespace": "two"},
                {"name": "one.run"},
            ),
        ):
            with self.subTest(identity=identity):
                model_text = qwen_call(tool.function.name, {})
                full = serving._make_response_output_items(
                    req,
                    model_text,
                    serving.tokenizer_manager.tokenizer,
                    require_reasoning=False,
                )
                self.assertEqual(len(full), 1)
                self.assertEqual(
                    {k: full[0].model_dump()[k] for k in identity}, identity
                )
                seq = StreamFixture(serving, req).run_seq(
                    [engine_chunk(model_text, finish=True)]
                )
                for kind in ("response.output_item.added", "response.output_item.done"):
                    wire = next(p["item"] for t, p in seq if t == kind)
                    self.assertEqual({k: wire[k] for k in identity}, identity)
                final = seq[-1][1]["response"]["output"][0]
                self.assertEqual({k: final[k] for k in identity}, identity)
                replay = request(
                    tools=req.tools,
                    input=[
                        {"role": "user", "content": "before"},
                        final,
                        {
                            "type": "function_call_output",
                            "call_id": final["call_id"],
                            "output": "done",
                        },
                    ],
                )
                # JSON round trip is where SDK TypedDicts used to drop namespace.
                replay = ResponsesRequest.model_validate_json(replay.model_dump_json())
                messages = serving._construct_input_messages(replay)
                self.assertEqual(
                    messages[1]["tool_calls"][0]["function"]["name"], tool.function.name
                )
                self.assertEqual(messages[2]["tool_call_id"], final["call_id"])

    def test_custom_exact_input_full_stream_and_callback(self):
        serving = qwen_serving()
        payload = '*** Begin Patch\n+quote " slash \\ emoji 😀\n*** End Patch'
        req = request(
            tools=[
                namespace(
                    "functions", {"type": "custom", "name": "exec"}, function("wait")
                )
            ]
        )
        text = (
            qwen_call("functions.exec", {"input": payload})
            + "\n"
            + qwen_call("functions.wait", {})
        )
        full = serving._make_response_output_items(
            req, text, serving.tokenizer_manager.tokenizer, require_reasoning=False
        )
        self.assertEqual(
            [item.type for item in full], ["custom_tool_call", "function_call"]
        )
        self.assertEqual(full[0].input, payload)
        chunks = [engine_chunk(text[:i]) for i in range(1, len(text), 7)] + [
            engine_chunk(text, finish=True)
        ]
        seq = StreamFixture(serving, req).run_seq(chunks)
        self.assertNotIn("response.failed", [t for t, _ in seq])
        deltas = "".join(
            p["delta"] for t, p in seq if t == "response.custom_tool_call_input.delta"
        )
        self.assertEqual(deltas, payload)
        self.assertEqual(
            next(
                p["input"]
                for t, p in seq
                if t == "response.custom_tool_call_input.done"
            ),
            payload,
        )
        call = next(
            p["item"]
            for t, p in seq
            if t == "response.output_item.done"
            and p["item"]["type"] == "custom_tool_call"
        )
        self.assertEqual(
            (call["name"], call["namespace"], call["input"]),
            ("exec", "functions", payload),
        )
        replay = request(
            tools=req.tools,
            input=[
                {"role": "user", "content": "before"},
                call,
                {
                    "type": "custom_tool_call_output",
                    "call_id": call["call_id"],
                    "output": "executed",
                },
            ],
        )
        messages = serving._construct_input_messages(
            ResponsesRequest.model_validate_json(replay.model_dump_json())
        )
        self.assertEqual(
            json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]),
            {"input": payload},
        )
        self.assertEqual(messages[2]["content"], "executed")

    def test_inline_tools_visible_only_at_their_history_position(self):
        serving = qwen_serving()
        req = request(
            tools=[function("initial")],
            input=[
                {"role": "user", "content": "before_inventory"},
                {
                    "type": "additional_tools",
                    "tools": [namespace("later", function("search"))],
                },
                {"role": "assistant", "content": "middle_marker"},
                {"type": "additional_tools", "tools": []},
                {"role": "user", "content": "after_inventory"},
            ],
            tool_choice={"type": "function", "namespace": "later", "name": "search"},
        )
        messages, _, _, _ = asyncio.run(
            serving._make_request(req, None, serving.tokenizer_manager.tokenizer)
        )
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        chat = ChatCompletionRequest(
            model="x",
            messages=messages,
            tools=serving._response_tools_to_chat_tools(req.tools),
        )
        processed = serving._process_messages(chat, False)
        prompt = serving.tokenizer_manager.tokenizer.apply_chat_template(
            messages,
            tools=[t.model_dump() for t in chat.tools],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertLess(prompt.index("before_inventory"), prompt.index("later.search"))
        self.assertLess(prompt.index("later.search"), prompt.index("middle_marker"))
        self.assertLess(
            prompt.index("middle_marker"), prompt.index("Additional tools:\n[]")
        )
        self.assertEqual(
            [t.function.name for t in serving._effective_tools(chat)],
            ["initial", "later.search"],
        )
        self.assertTrue(processed.prompt_ids)

    def test_plaintext_agent_message_keeps_author_recipient_and_parts(self):
        req = request(
            input=[
                {
                    "type": "agent_message",
                    "author": "reviewer",
                    "recipient": "main",
                    "content": [
                        {"type": "input_text", "text": "first"},
                        {"type": "input_text", "text": "second"},
                    ],
                }
            ]
        )
        messages = make_serving()._construct_input_messages(req)
        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": "[agent message from reviewer to main]\nfirst\nsecond",
                }
            ],
        )

    def test_opaque_agent_message_fails_loudly(self):
        req = request(
            input=[
                {
                    "type": "agent_message",
                    "content": [
                        {"type": "encrypted_content", "encrypted_content": "opaque"}
                    ],
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "plaintext"):
            make_serving()._construct_input_messages(req)

    def test_captured_mcp_inventory_reaches_qwen_with_original_schemas(self):
        req = ResponsesRequest.model_validate(
            fixture("codex_namespaced_mcp.json")["request"]
        )
        serving = qwen_serving(is_multimodal=True)
        _, prompts, _, _ = asyncio.run(
            serving._make_request(req, None, serving.tokenizer_manager.tokenizer)
        )
        tools = serving._response_tools_to_chat_tools(req)
        self.assertEqual(len(tools), 3)
        by_name = {tool.function.name: tool.function for tool in tools}
        self.assertEqual(
            by_name["mcp__tavily.tavily_search"].parameters["required"], ["query"]
        )
        self.assertEqual(
            by_name["mcp__tavily.tavily_extract"].parameters["properties"]["urls"],
            req.tools[1].tools[0]["parameters"]["properties"]["urls"],
        )
        for name in by_name:
            self.assertIn(name, prompts[0])
        self.assertIn('"query": {"type": "string"', prompts[0])

    def test_captured_github_union_schema_survives_namespace_adapter(self):
        tool = fixture("codex_github_union.json")["tool"]
        req = request(tools=[tool])
        adapted = make_serving()._response_tools_to_chat_tools(req)
        self.assertEqual(len(adapted), 1)
        self.assertEqual(
            adapted[0].function.name, "mcp__codex_apps__github._search_issues"
        )
        self.assertEqual(adapted[0].function.parameters, tool["tools"][0]["parameters"])
        types = adapted[0].function.parameters["properties"]["repository_full_name"][
            "anyOf"
        ]
        self.assertEqual([arm["type"] for arm in types], ["string", "array", "null"])

    def test_image_only_callback_reaches_engine_media_handoff(self):
        original = fixture("codex_view_image_output.json")["item"]
        serving = qwen_serving(is_multimodal=True)
        captured = {}
        install_engine_reply(serving, "green", captured)
        req = request(
            input=[
                {"role": "user", "content": "Describe the viewed image."},
                {
                    "type": "function_call",
                    "name": "view_image",
                    "arguments": '{"path":"/tmp/example.png"}',
                    "call_id": "call_view_image",
                },
                original,
            ]
        )
        result = asyncio.run(serving.create_responses(req))
        self.assertEqual(result.status, "completed")
        self.assertEqual(
            [image.url for image in captured["request"].image_data],
            [original["output"][0]["image_url"]],
        )
        self.assertEqual(captured["request"].modalities, ["image"])
        self.assertIn("<|image_pad|>", captured["request"].text)
        self.assertIn("<tool_response>", captured["request"].text)
        # The exact same wire callback must fail before generation on text-only
        # models instead of becoming an empty successful tool message.
        text_serving = qwen_serving()
        rejected = asyncio.run(text_serving.create_responses(req))
        self.assertEqual(rejected.status_code, 400)
        self.assertIn(b"image_url", rejected.body)
        text_serving.tokenizer_manager.generate_request.assert_not_called()

    def test_opt_in_autoreview_schema_and_assessment_full_and_stream(self):
        import jsonschema
        from utils import collect_stream_events, event_payloads

        schema = {
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "enum": ["allow", "deny"]},
                "risk_level": {"type": "string"},
                "user_authorization": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["outcome"],
            "additionalProperties": False,
        }
        assessment = {
            "outcome": "deny",
            "risk_level": "medium",
            "user_authorization": "not established",
            "rationale": "The requested action exceeds the approved scope.",
        }
        for stream in (False, True):
            with self.subTest(stream=stream):
                serving = qwen_serving(is_multimodal=True)
                captured = {}
                install_engine_reply(serving, json.dumps(assessment), captured)
                req = request(
                    stream=stream,
                    reasoning={"effort": "xhigh"},
                    tools=[function("exec_command")],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "assessment",
                            "schema": schema,
                        }
                    },
                )
                result = asyncio.run(serving.create_responses(req))
                if stream:
                    payloads = event_payloads(
                        asyncio.run(collect_stream_events(result))
                    )
                    result = payloads[-1]["response"]
                else:
                    result = result.model_dump()
                self.assertEqual(
                    json.loads(captured["sampling"]["json_schema"]), schema
                )
                self.assertEqual(result["status"], "completed")
                self.assertEqual(req.reasoning.effort, "xhigh")
                self.assertIn("xhigh", captured["request"].text)
                self.assertEqual(result["text"]["format"]["schema"], schema)
                decoded = json.loads(result["output"][0]["content"][0]["text"])
                jsonschema.validate(decoded, schema)
                self.assertEqual(decoded, assessment)

    def test_named_and_required_inline_custom_tools_use_existing_parser(self):
        for choice in (
            "required",
            {"type": "custom", "namespace": "functions", "name": "exec"},
        ):
            with self.subTest(choice=choice):
                serving = qwen_serving()
                captured = {}
                install_engine_reply(
                    serving,
                    qwen_call("functions.exec", {"input": "print(42)"}),
                    captured,
                )
                req = request(
                    tool_choice=choice,
                    input=[
                        {"role": "user", "content": "Run the code."},
                        {
                            "type": "additional_tools",
                            "tools": [
                                namespace(
                                    "functions", {"type": "custom", "name": "exec"}
                                )
                            ],
                        },
                    ],
                )
                result = asyncio.run(serving.create_responses(req))
                self.assertEqual(result.status, "completed")
                self.assertEqual(result.output[0].type, "custom_tool_call")
                self.assertEqual(result.output[0].input, "print(42)")
                self.assertIn("functions.exec", captured["sampling"]["structural_tag"])
                if isinstance(choice, dict):
                    self.assertEqual(result.tool_choice, choice)

    def test_inline_declarations_validate_even_in_mixed_replay(self):
        for bad in (
            {"type": "namespace", "name": "x"},
            {"type": "custom"},
            {"type": "web_search"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                request(
                    input=[
                        {"role": "user", "content": "before"},
                        {"type": "additional_tools", "tools": [bad]},
                    ]
                )

    def test_custom_input_prefix_decodes_split_escapes(self):
        from sglang.srt.entrypoints.openai.responses_adapters import (
            decode_custom_tool_input_prefix,
        )

        payload = '\n"\\😀\tend'
        arguments = json.dumps({"input": payload}, ensure_ascii=True)
        previous = ""
        for i in range(len(arguments) + 1):
            decoded = decode_custom_tool_input_prefix(arguments[:i])
            self.assertTrue(decoded.startswith(previous))
            self.assertTrue(payload.startswith(decoded))
            previous = decoded
        self.assertEqual(previous, payload)

    def test_stored_inline_custom_workflow_keeps_inventory_and_callback(self):
        serving = qwen_serving(is_multimodal=True)
        captured = {}
        install_engine_reply(
            serving, qwen_call("functions.exec", {"input": "print(42)"}), captured
        )
        first = request(
            store=True,
            input=[
                {"role": "user", "content": "Run the code."},
                {
                    "type": "additional_tools",
                    "tools": [
                        namespace("functions", {"type": "custom", "name": "exec"})
                    ],
                },
            ],
        )
        first_response = asyncio.run(serving.create_responses(first))
        self.assertEqual(first_response.output[0].type, "custom_tool_call")
        call_id = first_response.output[0].call_id
        install_engine_reply(serving, "done", captured)
        second = request(
            store=True,
            previous_response_id=first_response.id,
            input=[
                {"type": "custom_tool_call_output", "call_id": call_id, "output": "42"}
            ],
        )
        second_response = asyncio.run(serving.create_responses(second))
        self.assertEqual(second_response.status, "completed")
        self.assertIn("functions.exec", captured["request"].text)
        self.assertIn("print(42)", captured["request"].text)
        self.assertIn("42", captured["request"].text)
        self.assertEqual(
            serving.msg_store[second_response.id][1]["type"], "additional_tools"
        )
        third = request(previous_response_id=second_response.id, input="Continue")
        third_response = asyncio.run(serving.create_responses(third))
        self.assertEqual(third_response.status, "completed")
        self.assertEqual(captured["request"].text.count("<function=functions.exec>"), 1)

    def test_ordinary_text_prompt_matches_direct_chat_rendering(self):
        serving = qwen_serving(is_multimodal=True)
        req = request(
            instructions="Keep the exact prose.",
            input='plain "text"\nwith punctuation: % and .',
        )
        _, prompts, _, _ = asyncio.run(
            serving._make_request(req, None, serving.tokenizer_manager.tokenizer)
        )
        expected = serving.tokenizer_manager.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": req.instructions},
                {"role": "user", "content": req.input},
            ],
            tools=None,
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertEqual(prompts[0].encode(), expected.encode())

    def test_malformed_replayed_call_identity_returns_400(self):
        for kind in ("function_call", "custom_tool_call"):
            for identity in (
                {},
                {"name": None},
                {"name": ""},
                {"name": 3},
                {"name": ["run"]},
                {"name": "run", "namespace": 3},
            ):
                with self.subTest(kind=kind, identity=identity):
                    serving = make_serving()
                    req = request(
                        input=[
                            {
                                "type": kind,
                                "call_id": "call_bad",
                                "arguments": "{}",
                                "input": "payload",
                                **identity,
                            }
                        ]
                    )
                    result = asyncio.run(serving.create_responses(req))
                    self.assertEqual(result.status_code, 400)
                    self.assertIn(
                        b"namespace" if "namespace" in identity else b"name",
                        result.body,
                    )
                    serving.tokenizer_manager.generate_request.assert_not_called()

    def test_malformed_plaintext_agent_content_returns_400(self):
        for fields in (
            {"content": ["plaintext"]},
            {"content": "plaintext"},
            {"content": [{"type": "input_text", "text": 3}]},
            {"content": [{"type": "input_text", "text": "hello"}], "agent": "reviewer"},
        ):
            with self.subTest(fields=fields):
                serving = make_serving()
                result = asyncio.run(
                    serving.create_responses(
                        request(input=[{"type": "agent_message", **fields}])
                    )
                )
                self.assertEqual(result.status_code, 400)
                self.assertIn(b"agent_message", result.body)
                serving.tokenizer_manager.generate_request.assert_not_called()

    def test_custom_replay_rejects_nonstring_input(self):
        for payload in (None, 0, ["code"], {"code": "print(42)"}):
            with self.subTest(payload=payload):
                serving = make_serving()
                result = asyncio.run(
                    serving.create_responses(
                        request(
                            input=[
                                {
                                    "type": "custom_tool_call",
                                    "call_id": "call_bad",
                                    "name": "exec",
                                    "input": payload,
                                }
                            ]
                        )
                    )
                )
                self.assertEqual(result.status_code, 400)
                self.assertIn(b"string input", result.body)
                serving.tokenizer_manager.generate_request.assert_not_called()

    def test_inline_rebinding_uses_latest_schema_and_keeps_other_members(self):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        old = function(
            "A",
            description="old_schema",
            parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        )
        new = function(
            "A",
            description="new_schema",
            parameters={"type": "object", "properties": {"value": {"type": "integer"}}},
        )
        serving = qwen_serving(is_multimodal=True)
        captured = {}
        install_engine_reply(serving, qwen_call("N.A", {"value": "7"}), captured)
        req = request(
            input=[
                {"role": "user", "content": "before"},
                {
                    "type": "additional_tools",
                    "tools": [namespace("N", old, function("B"))],
                },
                {"role": "assistant", "content": "between"},
                {"type": "additional_tools", "tools": [namespace("N", new)]},
                {"role": "user", "content": "after"},
            ]
        )
        result = asyncio.run(serving.create_responses(req))
        self.assertEqual(result.status, "completed")
        self.assertEqual(json.loads(result.output[0].arguments), {"value": 7})
        prompt = captured["request"].text
        self.assertLess(prompt.index("old_schema"), prompt.index("between"))
        self.assertLess(prompt.index("between"), prompt.index("new_schema"))
        self.assertLess(prompt.index("new_schema"), prompt.index("after"))
        messages = serving._construct_input_messages(req)
        current = serving._effective_tools(
            ChatCompletionRequest(model="x", messages=messages)
        )
        self.assertEqual([tool.function.name for tool in current], ["N.A", "N.B"])
        self.assertEqual(
            current[0].function.parameters["properties"]["value"]["type"], "integer"
        )

    def test_restored_inline_tool_and_stored_rebinding_are_accepted(self):
        old = function(
            "A",
            description="original_schema",
            parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        )
        serving = qwen_serving(is_multimodal=True)
        captured = {}
        install_engine_reply(serving, "ready", captured)
        first = request(
            store=True,
            input=[
                {"role": "user", "content": "before"},
                {"type": "additional_tools", "tools": [namespace("N", old)]},
                {"role": "user", "content": "restore"},
                {"type": "additional_tools", "tools": [namespace("N", old)]},
            ],
        )
        first_response = asyncio.run(serving.create_responses(first))
        self.assertEqual(first_response.status, "completed")
        new = function(
            "A",
            description="revised_schema",
            parameters={"type": "object", "properties": {"value": {"type": "integer"}}},
        )
        second = request(
            previous_response_id=first_response.id,
            input=[
                {"type": "additional_tools", "tools": [namespace("N", new)]},
                {"role": "user", "content": "call A"},
            ],
        )
        install_engine_reply(serving, qwen_call("N.A", {"value": "7"}), captured)
        second_response = asyncio.run(serving.create_responses(second))
        self.assertEqual(second_response.status, "completed")
        self.assertEqual(json.loads(second_response.output[0].arguments), {"value": 7})
        self.assertEqual(captured["request"].text.count("original_schema"), 2)
        self.assertLess(
            captured["request"].text.rindex("original_schema"),
            captured["request"].text.index("revised_schema"),
        )

    def test_duplicate_identity_inside_single_inventory_returns_400(self):
        for tools in (
            [function(), function()],
            [namespace("N", function("A"), function("A"))],
            [namespace("N", function("A")), namespace("N", function("A"))],
        ):
            with self.subTest(tools=tools):
                serving = make_serving()
                result = asyncio.run(
                    serving.create_responses(
                        request(
                            input=[
                                {"role": "user", "content": "before"},
                                {"type": "additional_tools", "tools": tools},
                            ]
                        )
                    )
                )
                self.assertEqual(result.status_code, 400)

    def _stream_custom_arguments(self, arguments, *, close_during_decode=False):
        from sglang.srt.function_call.core_types import ToolCallItem

        serving = make_serving()
        serving.tool_call_parser = "qwen3_coder"
        req = request(tools=[{"type": "custom", "name": "exec"}])
        parser = Mock()
        parser.detector.supports_structural_tag.return_value = True
        parser.parse_stream_chunk.return_value = (
            "",
            [ToolCallItem(tool_index=0, name="exec", parameters=arguments)],
        )
        parser.parse_stream_end.return_value = ("", [])
        chunks = [engine_chunk("tool arguments")]
        if close_during_decode:
            parser.parse_stream_chunk.side_effect = [
                parser.parse_stream_chunk.return_value,
                ("following text", []),
            ]
            chunks.append(engine_chunk("tool arguments and following text"))
        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser",
            return_value=parser,
        ):
            return StreamFixture(serving, req).run_seq(chunks)

    def test_custom_prefix_ignores_nested_and_nonstring_input_fields(self):
        from sglang.srt.entrypoints.openai.responses_adapters import (
            decode_custom_tool_input_prefix,
        )

        self.assertEqual(
            decode_custom_tool_input_prefix(
                '{"other":{"input":"nested"},"input":"real"}'
            ),
            "real",
        )
        self.assertEqual(
            decode_custom_tool_input_prefix('{"input":123,"other":"value"}'), ""
        )

    def test_custom_fallback_without_streamed_conflict_matches_completed_input(self):
        from sglang.srt.entrypoints.openai.responses_adapters import (
            decode_custom_tool_input,
        )

        for arguments in (
            '{"other":{"input":"nested"},"input":"real"}',
            '{"input":123,"other":"value"}',
            '{"payload":"freeform"}',
            "bare payload",
        ):
            with self.subTest(arguments=arguments):
                seq = self._stream_custom_arguments(arguments)
                self.assertEqual(seq[-1][0], "response.completed")
                displayed = "".join(
                    payload["delta"]
                    for kind, payload in seq
                    if kind == "response.custom_tool_call_input.delta"
                )
                executed = next(
                    payload["item"]["input"]
                    for kind, payload in seq
                    if kind == "response.output_item.done"
                )
                self.assertEqual(displayed, executed)
                self.assertEqual(executed, decode_custom_tool_input(arguments))

    def test_conflicting_custom_input_fails_before_any_completed_call(self):
        for arguments in ('{"input":"prefix"', '{"input":"first","input":"last"}'):
            for close_during_decode in (False, True):
                with self.subTest(
                    arguments=arguments, close_during_decode=close_during_decode
                ):
                    seq = self._stream_custom_arguments(
                        arguments, close_during_decode=close_during_decode
                    )
                    self.assertEqual(seq[-1][0], "response.failed")
                    self.assertEqual(seq[-1][1]["response"]["status"], "failed")
                    self.assertNotIn(
                        "response.custom_tool_call_input.done",
                        [kind for kind, _ in seq],
                    )
                    self.assertNotIn(
                        "response.output_item.done", [kind for kind, _ in seq]
                    )
                    self.assertNotIn("response.completed", [kind for kind, _ in seq])


if __name__ == "__main__":
    unittest.main()
