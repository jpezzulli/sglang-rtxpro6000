"""Namespaced tools and inline inventories, as the model actually sees them.

A namespace declaration has to reach every parser's tool inventory, not only the
one whose template renders namespaces, and the spelling a prompt advertises has
to be the spelling the parser answers and the response item reports. A request
can also widen its own inventory partway through the history it replays; that
inline declaration rebinds the leaf instead of merging two schemas together.
"""

import asyncio
import json
import sys
import unittest

from jinja2 import Environment
from utils import StreamFixture, engine_chunk, event_payloads, make_serving

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ResponsesRequest
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

QUERY = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

# Muse's documented contract: a namespace member is addressed as
# ``<namespace>.<member>`` and an ordinary tool by its own name.
MUSE_TEMPLATE = (
    "{%- for tool in tools or [] -%}"
    "{%- if tool.type == 'namespace' -%}"
    "NS[{{ tool.name }}]"
    "{%- for member in tool.tools -%}"
    " CALL({{ tool.name }}.{{ member.name or member.function.name }})"
    "{%- endfor -%}"
    "{%- else -%}"
    " CALL({{ tool.function.name }})"
    "{%- endif -%}"
    "{%- endfor -%}"
)

MUSE_CALL = (
    " to={name}<|message|><atem:function_calls>"
    '<atem:invoke name="{name}">'
    '<atem:parameter name="query">news</atem:parameter>'
    "</atem:invoke></atem:function_calls><|eot|>"
)


def function(name="search", **kwargs):
    return {"type": "function", "name": name, "parameters": QUERY, **kwargs}


def namespace(name, *members):
    return {
        "type": "namespace",
        "name": name,
        "description": f"{name} tools",
        "tools": [
            member if isinstance(member, dict) else function(member) for member in members
        ],
    }


def request(tools=(), **kwargs):
    payload = {"model": "x", "input": "go", "tools": list(tools), "store": False}
    payload.update(kwargs)
    return ResponsesRequest(**payload)


def serving_for(parser, chat_template=None):
    serving = make_serving()
    serving.reasoning_parser = None
    serving.tool_call_parser = parser
    serving.template_manager.chat_template_name = None
    serving._prompt_text_round_trip_is_lossy = False
    if chat_template is not None:
        template = Environment().from_string(chat_template)
        serving.tokenizer_manager.tokenizer.apply_chat_template.side_effect = (
            lambda messages, **kwargs: template.render(
                messages=messages, tools=kwargs.get("tools")
            )
        )
    return serving


def qwen_call(name, arguments):
    """One real Qwen3-Coder markup call, built without literal delimiter tokens."""

    def tag(body):
        return chr(60) + body + chr(62)

    newline = chr(10)
    parameters = "".join(
        tag("parameter=" + key) + newline + str(value) + newline + tag("/parameter") + newline
        for key, value in arguments.items()
    )
    return (
        tag("tool_call")
        + newline
        + tag("function=" + name)
        + newline
        + parameters
        + tag("/function")
        + newline
        + tag("/tool_call")
    )


def function_calls(items):
    return [
        item.model_dump(exclude_none=True) for item in items if item.type == "function_call"
    ]


class QwenNamespaceTestCase(CustomTestCase):
    def test_namespace_reaches_a_plain_parser_as_one_callable_per_member(self):
        tools = serving_for("qwen3_coder")._response_tools_to_chat_tools(
            request([namespace("browser", "search"), function("lookup")])
        )
        self.assertEqual(
            [tool.function.name for tool in tools], ["browser__search", "lookup"]
        )
        self.assertIn("browser tools", tools[0].function.description)
        self.assertIn("query", tools[0].function.parameters["properties"])

    def test_a_namespace_member_cannot_collide_with_an_ordinary_name(self):
        tools = serving_for("qwen3_coder")._response_tools_to_chat_tools(
            request(
                [
                    function("browser__search"),
                    namespace("browser", "search"),
                    namespace("other", "search"),
                ]
            )
        )
        names = [tool.function.name for tool in tools]
        self.assertEqual(len(names), len(set(names)), names)
        self.assertIn("browser__search", names)

    def test_named_selection_and_parsed_call_share_the_bound_spelling(self):
        serving = serving_for("qwen3_coder")
        req = request(
            [namespace("browser", "search")],
            tool_choice={"type": "function", "namespace": "browser", "name": "search"},
        )
        choice = serving._chat_tool_choice(
            req.effective_tool_choice(), serving._tool_identities(req)
        )
        self.assertEqual(choice, {"type": "function", "function": {"name": "browser__search"}})

        (call,) = function_calls(
            serving._make_response_output_items(
                req,
                qwen_call("browser__search", {"query": "news"}),
                tokenizer=None,
                require_reasoning=False,
            )
        )
        self.assertEqual((call.get("namespace"), call["name"]), ("browser", "search"))

    def test_a_streamed_namespaced_call_reports_the_declaration(self):
        serving = serving_for("qwen3_coder")
        req = request([namespace("browser", "search")], stream=True)
        raw = qwen_call("browser__search", {"query": "news"})
        events = StreamFixture(serving, req).run(
            [engine_chunk(raw[:20], 20), engine_chunk(raw, 2, finish=True)]
        )
        reported = [
            payload["item"]
            for payload in event_payloads(events)
            if payload["type"] == "response.output_item.done"
            and payload["item"]["type"] == "function_call"
        ]
        self.assertTrue(reported)
        for item in reported:
            self.assertEqual((item.get("namespace"), item["name"]), ("browser", "search"))

    def test_replayed_namespaced_call_uses_the_bound_spelling(self):
        messages = serving_for("qwen3_coder")._construct_input_messages(
            request(
                [namespace("browser", "search")],
                input=[
                    {"role": "user", "content": "before"},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "namespace": "browser",
                        "name": "search",
                        "arguments": "{}",
                    },
                ],
            )
        )
        replayed = [
            call["function"]["name"]
            for message in messages
            for call in message.get("tool_calls") or []
        ]
        self.assertEqual(replayed, ["browser__search"])


class MuseRenderedNameTestCase(CustomTestCase):
    """What the template composes has to be what the parser answers."""

    def _prompt(self, serving, req):
        chat_request = ChatCompletionRequest(model="x", messages=[{"role": "user", "content": "go"}])
        chat_request._template_tools = serving._response_tools_to_template_tools(req)
        tools = [
            tool.model_dump(exclude_none=True)
            for tool in serving._response_tools_to_chat_tools(req)
        ]
        return serving.tokenizer_manager.tokenizer.apply_chat_template(
            chat_request.messages,
            tools=chat_request._template_tools or tools,
            add_generation_prompt=True,
        )

    def test_uncontested_member_is_rendered_under_the_namespaced_spelling(self):
        serving = serving_for("muse", MUSE_TEMPLATE)
        req = request([namespace("browser", "search")])
        prompt = self._prompt(serving, req)
        self.assertIn("NS[browser] CALL(browser.search)", prompt)
        (call,) = function_calls(
            serving._make_response_output_items(
                req,
                MUSE_CALL.format(name="browser.search"),
                tokenizer=None,
                require_reasoning=False,
            )
        )
        self.assertEqual((call.get("namespace"), call["name"]), ("browser", "search"))

    def test_contested_member_is_rendered_under_the_spelling_it_routes_to(self):
        serving = serving_for("muse", MUSE_TEMPLATE)
        req = request([namespace("browser", "search"), function("browser.search")])
        prompt = self._prompt(serving, req)
        advertised = sorted(
            name for name in ("browser.search", "browser%2Esearch") if f"CALL({name})" in prompt
        )
        # The ordinary tool keeps its own name and the member is offered under the
        # one spelling the parser will answer, without a second namespace block.
        self.assertEqual(advertised, ["browser%2Esearch", "browser.search"], prompt)
        self.assertNotIn("NS[browser]", prompt)
        (call,) = function_calls(
            serving._make_response_output_items(
                req,
                MUSE_CALL.format(name="browser%2Esearch"),
                tokenizer=None,
                require_reasoning=False,
            )
        )
        self.assertEqual((call.get("namespace"), call["name"]), ("browser", "search"))

    def test_a_flat_lookalike_still_routes_to_the_ordinary_tool(self):
        serving = serving_for("muse", MUSE_TEMPLATE)
        req = request([namespace("browser", "search"), function("browser.search")])
        (call,) = function_calls(
            serving._make_response_output_items(
                req,
                MUSE_CALL.format(name="browser.search"),
                tokenizer=None,
                require_reasoning=False,
            )
        )
        self.assertIsNone(call.get("namespace"))
        self.assertEqual(call["name"], "browser.search")


class AdditionalToolsInventoryTestCase(CustomTestCase):
    def test_inventory_stays_at_its_history_position_including_empty(self):
        serving = serving_for("qwen3_coder")
        req = request()
        req.input = [
            {"role": "user", "content": "before"},
            {"type": "additional_tools", "tools": [namespace("mcp", function("search"))]},
            {"role": "assistant", "content": "middle"},
            {"type": "additional_tools", "tools": []},
            {"role": "user", "content": "after"},
        ]
        messages = serving._construct_input_messages(req)
        self.assertEqual(
            [message["role"] for message in messages],
            ["user", "system", "assistant", "system", "user"],
        )
        self.assertEqual(
            [tool["function"]["name"] for tool in messages[1]["tools"]], ["mcp__search"]
        )
        self.assertEqual(messages[3]["tools"], [])
        self.assertEqual(messages[3]["content"], "Additional tools:\n[]")

    def test_inline_inventory_is_offered_to_the_parser_and_the_renderer(self):
        serving = serving_for("qwen3_coder")
        req = request(
            [function("initial")],
            input=[
                {"role": "user", "content": "before"},
                {"type": "additional_tools", "tools": [namespace("later", function("search"))]},
                {"role": "user", "content": "after"},
            ],
            tool_choice={"type": "function", "namespace": "later", "name": "search"},
        )
        messages, _, _, _ = asyncio.run(
            serving._make_request(req, None, serving.tokenizer_manager.tokenizer)
        )
        chat = ChatCompletionRequest(
            model="x", messages=messages, tools=serving._response_tools_to_chat_tools(req)
        )
        self.assertEqual(
            [tool.function.name for tool in serving._effective_tools(chat)],
            ["initial", "later__search"],
        )

    def test_rebinding_uses_the_latest_declaration_and_keeps_its_siblings(self):
        req = request(
            input=[
                {"role": "user", "content": "before"},
                {
                    "type": "additional_tools",
                    "tools": [
                        namespace("N", function("A", description="old_schema"), function("B"))
                    ],
                },
                {"type": "additional_tools", "tools": [namespace("N", function("A", description="new_schema"))]},
            ]
        )
        serving = serving_for("qwen3_coder")
        tools = {
            tool.function.name: tool for tool in serving._response_tools_to_chat_tools(req)
        }
        self.assertEqual(set(tools), {"N__A", "N__B"})
        self.assertIn("new_schema", tools["N__A"].function.description)
        self.assertIn("old_schema", tools["N__B"].function.description)
        inventories = [
            message["content"]
            for message in serving._construct_input_messages(req)
            if "tools" in message
        ]
        self.assertEqual(len(inventories), 2)
        self.assertIn("old_schema", inventories[0])
        self.assertIn("new_schema", inventories[1])

    def test_duplicate_identity_in_one_inventory_is_a_bad_request(self):
        for tools in (
            [function(), function()],
            [namespace("N", function("A"), function("A"))],
            [namespace("N", function("A")), namespace("N", function("A"))],
        ):
            with self.subTest(tools=tools):
                result = asyncio.run(
                    serving_for("qwen3_coder").create_responses(
                        request(
                            input=[
                                {"role": "user", "content": "before"},
                                {"type": "additional_tools", "tools": tools},
                            ]
                        )
                    )
                )
                self.assertEqual(result.status_code, 400, getattr(result, "body", None))

    def test_unsupported_inline_declaration_is_a_bad_request(self):
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


class AgentMessageTestCase(unittest.TestCase):
    MESSAGE = {
        "type": "agent_message",
        "author": "coder",
        "recipient": "reviewer",
        "content": [
            {"type": "input_text", "text": "first"},
            {"type": "input_text", "text": "second"},
        ],
    }

    def normalize(self, **overrides):
        return OpenAIServingResponses._normalize_response_message_for_chat(
            {**self.MESSAGE, **overrides}
        )

    def test_plaintext_message_keeps_author_recipient_and_parts(self):
        self.assertEqual(
            self.normalize(),
            {"role": "user", "content": "[agent message from coder to reviewer]\nfirst\nsecond"},
        )

    def test_agent_name_in_the_nested_object_is_equivalent(self):
        message = self.normalize(author=None, agent={"agent_name": "coder"})
        self.assertIn("from coder", message["content"])

    def test_opaque_content_fails_loudly(self):
        with self.assertRaises(ValueError):
            self.normalize(content=[{"type": "output_text", "text": "x"}])

    def test_missing_plaintext_fails_loudly(self):
        with self.assertRaises(ValueError):
            self.normalize(content=[])


if __name__ == "__main__":
    sys.exit(unittest.main())
