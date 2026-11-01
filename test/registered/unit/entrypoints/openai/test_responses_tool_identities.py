"""Flat and namespaced callable identities, and the history that outlives them.

One request can declare ``search`` under the namespace ``browser`` and, at the
same time, an ordinary tool literally named ``browser.search``. The chat
function-call path has one flat name space, so the derived spelling has to be
distinct, reversible, and the same one the prompt, a named ``tool_choice``, a
parsed call and a replayed transcript all use. Replayed history can also name a
tool the request no longer declares; that identity keeps a spelling of its own
without becoming callable.
"""

import sys
import unittest

from utils import make_serving

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest, ResponseTool
from sglang.srt.entrypoints.openai.responses_adapters import (
    decode_custom_tool_input_prefix,
    encode_custom_tool_input,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}}
QUERY = {**SCHEMA, "required": ["query"]}


def namespace(name, *members, description="Browse"):
    return {
        "type": "namespace",
        "name": name,
        "description": description,
        "tools": [
            member
            if isinstance(member, dict)
            else {"name": member, "parameters": QUERY}
            for member in members
        ],
    }


def function(name):
    return {"type": "function", "name": name, "parameters": QUERY}


def request(tools, **kwargs):
    payload = {"model": "x", "input": "go", "tools": tools, "store": False}
    payload.update(kwargs)
    return ResponsesRequest(**payload)


def muse_serving():
    serving = make_serving()
    serving.tool_call_parser = "muse"
    return serving


def offered_names(serving, req):
    return [tool.function.name for tool in serving._response_tools_to_chat_tools(req)]


class IdentityBindingTestCase(unittest.TestCase):
    """The binding table alone, no serving fixture in the way."""

    def setUp(self):
        from sglang.srt.entrypoints.openai import responses_adapters

        self.tool_identities = responses_adapters.tool_identities

    def test_flat_declaration_wins_the_spelling_a_namespace_would_flatten_to(self):
        identities = self.tool_identities(
            [
                ResponseTool.model_validate(namespace("browser", "search")),
                ResponseTool.model_validate(function("browser.search")),
            ],
            separator=".",
        )
        # The ordinary name is never rewritten...
        self.assertEqual(identities.model_name("browser.search"), "browser.search")
        # ...so the namespace member takes a spelling of its own, and both route
        # back to the declaration they came from.
        member = identities.model_name("search", "browser")
        self.assertNotEqual(member, "browser.search")
        self.assertEqual(
            identities.wire_identity(member), {"name": "search", "namespace": "browser"}
        )
        self.assertEqual(
            identities.wire_identity("browser.search"), {"name": "browser.search"}
        )

    def test_bindings_ignore_inventory_and_history_order_and_are_injective(self):
        tools = [
            ResponseTool.model_validate(namespace("a", "b__c", "c")),
            ResponseTool.model_validate(namespace("a__b", "c")),
            ResponseTool.model_validate(function("a__b__c")),
        ]
        spellings = {}
        for shuffle in (tools, tools[::-1]):
            identities = self.tool_identities(shuffle)
            for tool in shuffle:
                if tool.type == "namespace":
                    for member in tool.tools:
                        spellings[(tool.name, member["name"])] = identities.model_name(
                            member["name"], tool.name
                        )
                else:
                    spellings[(None, tool.name)] = identities.model_name(tool.name)
            self.assertEqual(len(set(spellings.values())), len(spellings), spellings)
            for identity, spelling in spellings.items():
                expected = {"name": identity[1]}
                if identity[0]:
                    expected["namespace"] = identity[0]
                self.assertEqual(identities.wire_identity(spelling), expected)

    def test_retired_identity_keeps_a_spelling_and_stays_uncallable(self):
        identities = self.tool_identities(
            [ResponseTool.model_validate(namespace("browser", "search"))],
            replayed=[("browser", "lookup"), ("files", "search")],
            separator=".",
        )
        retired = identities.model_name("lookup", "browser")
        self.assertEqual(retired, "browser.lookup")
        self.assertFalse(identities.declares("lookup", "browser"))
        # History-only spellings are never the destination of a parsed call.
        self.assertIsNone(identities.wire_identity(retired))
        self.assertNotIn(retired, identities.names)
        # A retired member cannot take the spelling of a callable one.
        self.assertNotEqual(
            identities.model_name("search", "files"),
            identities.model_name("search", "browser"),
        )

    def test_an_unknown_identity_still_renders_without_stealing_a_spelling(self):
        identities = self.tool_identities(
            [ResponseTool.model_validate(function("browser.search"))], separator="."
        )
        self.assertEqual(
            identities.model_name("search", "browser"),
            identities.model_name("search", "browser"),
        )
        self.assertNotIn(identities.model_name("search", "browser"), identities.names)


class NamespaceServingTestCase(CustomTestCase):
    def test_contested_namespace_member_is_offered_under_its_own_name(self):
        req = request(
            [namespace("browser", "search"), function("browser.search")],
            tool_choice="required",
        )
        names = offered_names(muse_serving(), req)
        self.assertEqual(len(set(names)), 2, names)
        self.assertIn("browser.search", names)

    def test_named_choice_reaches_the_parser_as_the_advertised_spelling(self):
        """A selection of a namespace member names the tool the prompt offered,
        not the bare leaf the caller wrote."""
        serving = muse_serving()
        req = request(
            [namespace("browser", "search")],
            tool_choice={"type": "function", "namespace": "browser", "name": "search"},
        )
        offered = offered_names(serving, req)
        choice = serving._chat_tool_choice(
            req.effective_tool_choice(), serving._tool_identities(req)
        )
        self.assertEqual(choice, {"type": "function", "function": {"name": offered[0]}})
        self.assertFalse(
            serving._uses_required_tool_json(
                req,
                serving._response_tools_to_chat_tools(req),
                serving._parser_tool_choice(req, req.effective_tool_choice()),
            )
        )

    def test_a_flat_lookalike_choice_is_not_a_namespace_member(self):
        import asyncio

        result = asyncio.run(
            muse_serving().create_responses(
                request(
                    [namespace("browser", "search")],
                    tool_choice={"type": "function", "name": "browser.search"},
                )
            )
        )
        self.assertEqual(result.status_code, 400)

    def test_parsed_call_reports_the_declaration_it_came_from(self):
        req = request([namespace("browser", "search"), function("browser.search")])
        serving = muse_serving()
        names = offered_names(serving, req)
        member = next(name for name in names if name != "browser.search")
        raw = (
            " to=" + member + "<|message|><atem:function_calls>"
            '<atem:invoke name="' + member + '">'
            '<atem:parameter name="query">news</atem:parameter>'
            "</atem:invoke></atem:function_calls><|eot|>"
        )
        items = serving._make_response_output_items(
            req, raw, tokenizer=None, require_reasoning=False
        )
        dumped = [item.model_dump(exclude_none=True) for item in items]
        (item,) = [i for i in dumped if i["type"] == "function_call"]
        self.assertEqual(
            (item.get("namespace"), item["name"]), ("browser", "search"), dumped
        )
        # The lookalike stays an ordinary call, whichever name the model used.
        flat = serving._make_response_output_items(
            req,
            raw.replace(member, "browser.search"),
            tokenizer=None,
            require_reasoning=False,
        )
        (flat_item,) = [
            i.model_dump(exclude_none=True) for i in flat if i.type == "function_call"
        ]
        self.assertEqual(
            (flat_item.get("namespace"), flat_item["name"]), (None, "browser.search")
        )

    def test_replayed_retired_call_keeps_its_own_spelling(self):
        req = request(
            [namespace("browser", "search"), function("browser.search")],
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
        messages = muse_serving()._construct_input_messages(req)
        replayed = [
            call["function"]["name"]
            for message in messages
            for call in message.get("tool_calls") or []
        ]
        self.assertEqual(len(replayed), 1)
        # The transcript names the member, so it cannot be rendered as the flat
        # tool that happens to spell the same.
        self.assertNotEqual(replayed[0], "browser.search")
        self.assertIn(replayed[0], offered_names(muse_serving(), req))


class ReplayIdentityTestCase(CustomTestCase):
    """Replayed call items under the table the current request binds.

    A name in history is an identity, not a string to concatenate. When the
    request's current declarations already hold that flat spelling, the retired
    tool has to come back under the escaped name reserved for it -- otherwise the
    transcript advertises the callable tool for a call that was aimed at another
    one, in both the function and the custom shapes, and whether the history was
    sent inline or is a stored continuation.
    """

    def _serving(self, parser):
        serving = make_serving()
        serving.tool_call_parser = parser
        serving.reasoning_parser = None
        return serving

    def _replayed_names(self, serving, req):
        return [
            call["function"]["name"]
            for message in serving._construct_input_messages(req)
            for call in message.get("tool_calls") or []
        ]

    def test_a_retired_flat_call_is_not_rendered_as_the_current_member(self):
        for parser, retired, escaped in (
            ("qwen3_coder", "browser__search", "browser%5F%5Fsearch"),
            ("muse", "browser.search", "browser%2Esearch"),
        ):
            with self.subTest(parser=parser):
                serving = self._serving(parser)
                req = request(
                    [namespace("browser", "search")],
                    input=[
                        {"role": "user", "content": "before"},
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": retired,
                            "arguments": '{"query": "old"}',
                        },
                    ],
                )
                self.assertEqual(self._replayed_names(serving, req), [escaped])
                identities = serving._tool_identities(
                    req, serving._replayed_tool_identities(req)
                )
                # The retired tool keeps a spelling and stays uncallable, and the
                # current member is the only thing offered under the flat name.
                self.assertIsNone(identities.wire_identity(escaped))
                self.assertEqual(offered_names(serving, req), [retired])

    def test_a_retired_flat_custom_call_keeps_its_payload_and_spelling(self):
        serving = self._serving("qwen3_coder")
        req = request(
            [namespace("browser", "search")],
            input=[
                {"role": "user", "content": "before"},
                {
                    "type": "custom_tool_call",
                    "call_id": "call_1",
                    "name": "browser__search",
                    "input": "echo \"a\tb",
                },
            ],
        )
        (name,) = self._replayed_names(serving, req)
        self.assertEqual(name, "browser%5F%5Fsearch")
        message = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "browser__search",
                "input": "echo \"a\tb",
            },
            identities=serving._tool_identities(
                req, serving._replayed_tool_identities(req)
            ),
        )
        (call,) = message["tool_calls"]
        self.assertEqual(
            call["function"]["arguments"], encode_custom_tool_input('echo "a\tb')
        )

    def test_a_replayed_call_for_a_declared_flat_tool_keeps_its_name(self):
        serving = self._serving("qwen3_coder")
        req = request(
            [function("lookup")],
            input=[
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{}",
                }
            ],
        )
        self.assertEqual(self._replayed_names(serving, req), ["lookup"])

    def test_a_stored_continuation_binds_the_names_it_rendered(self):
        serving = self._serving("muse")
        serving.msg_store["resp_prior"] = [
            {"role": "user", "content": "first"},
            {
                "type": "function_call",
                "call_id": "a",
                "name": "browser.search",
                "arguments": "{}",
            },
            {
                "type": "custom_tool_call",
                "call_id": "b",
                "namespace": "browser",
                "name": "search",
                "input": "pwd",
            },
        ]
        req = request([], input="again", previous_response_id="resp_prior")
        # Nothing is callable now, so neither retired name may take the other's
        # spelling: the two items the store rendered stay distinct.
        self.assertEqual(offered_names(serving, req), [])
        self.assertEqual(
            self._replayed_names(serving, req),
            ["browser.search", "browser%2Esearch"],
        )

    def test_a_stored_retired_name_does_not_steal_the_current_member_spelling(self):
        serving = self._serving("muse")
        serving.msg_store["resp_prior"] = [
            {
                "type": "function_call",
                "call_id": "a",
                "name": "browser.search",
                "arguments": "{}",
            }
        ]
        req = request(
            [namespace("browser", "search")],
            input="again",
            previous_response_id="resp_prior",
        )
        self.assertEqual(offered_names(serving, req), ["browser.search"])
        self.assertEqual(self._replayed_names(serving, req), ["browser%2Esearch"])


class CustomToolPayloadTestCase(unittest.TestCase):
    def test_nested_input_field_is_not_the_payload(self):
        self.assertEqual(
            decode_custom_tool_input_prefix(
                '{"other":{"input":"nested"},"input":"real"}'
            ),
            "real",
        )

    def test_a_nonstring_input_field_has_no_payload_to_stream(self):
        self.assertEqual(
            decode_custom_tool_input_prefix('{"input":123,"other":"value"}'), ""
        )

    def test_every_prefix_of_an_escaped_payload_decodes_to_a_prefix(self):
        payload = '\n"\\U0001f600\tend'
        arguments = '{"input": ' + __import__("json").dumps(payload) + "}"
        previous = ""
        for end in range(len(arguments) + 1):
            decoded = decode_custom_tool_input_prefix(arguments[:end])
            self.assertTrue(decoded.startswith(previous), (end, decoded))
            self.assertTrue(payload.startswith(decoded), (end, decoded))
            previous = decoded
        self.assertEqual(previous, payload)


if __name__ == "__main__":
    sys.exit(unittest.main())
