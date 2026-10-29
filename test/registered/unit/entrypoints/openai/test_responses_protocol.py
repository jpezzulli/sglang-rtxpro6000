import json
import unittest
from typing import get_args, get_origin

from openai.types.responses import (
    ResponseCustomToolCall,
    ResponseFunctionToolCall,
    ResponseReasoningItem,
)
from utils import make_serving  # noqa: F401 — bootstrap import

from sglang.srt.entrypoints.openai.protocol import (
    PromptTokensDetails,
    ResponsesRequest,
    ResponsesResponse,
    UsageInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


def _in_progress_response(request: ResponsesRequest) -> ResponsesResponse:
    return ResponsesResponse.from_request(
        request,
        sampling_params={},
        model_name="x",
        created_time=0,
        output=[],
        status="in_progress",
        usage=None,
    )


class ResponsesRequestTestCase(CustomTestCase):
    def test_function_tool_accepted(self):
        request = ResponsesRequest(
            model="x",
            input="call the tool",
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                    "strict": True,
                }
            ],
            store=False,
        )

        self.assertEqual(request.tools[0].type, "function")
        self.assertEqual(request.tools[0].name, "lookup")
        self.assertTrue(request.tools[0].strict)

    def test_function_tool_requires_name(self):
        with self.assertRaises(ValueError):
            ResponsesRequest(
                model="x", input="hi", tools=[{"type": "function"}], store=False
            )
        with self.assertRaises(ValueError):
            ResponsesRequest(
                model="x",
                input="hi",
                tools=[{"type": "function", "name": ""}],
                store=False,
            )

    def test_extended_tool_types_accepted(self):
        for tool_type in (
            "web_search",
            "web_search_preview",
            "code_interpreter",
            "file_search",
            "image_generation",
            "computer_use_preview",
            "local_shell",
            "mcp",
            "custom",
            "namespace",
        ):
            request = ResponsesRequest(
                model="x",
                input="hi",
                tools=[
                    {
                        "type": tool_type,
                        **(
                            {"name": "named"}
                            if tool_type in ("custom", "namespace")
                            else {}
                        ),
                        **({"tools": []} if tool_type == "namespace" else {}),
                    }
                ],
                store=False,
            )
            self.assertEqual(request.tools[0].type, tool_type)

    def test_namespace_tool_carries_inner_tools_list(self):
        request = ResponsesRequest(
            model="x",
            input="hi",
            tools=[
                {
                    "type": "namespace",
                    "name": "codex",
                    "tools": [
                        {"type": "function", "name": "apply_patch"},
                        {"type": "function", "name": "shell"},
                    ],
                }
            ],
            store=False,
        )
        self.assertEqual(request.tools[0].type, "namespace")
        self.assertEqual(len(request.tools[0].tools), 2)
        self.assertEqual(request.tools[0].tools[0]["name"], "apply_patch")


class ResponsesSamplingParamsTestCase(CustomTestCase):
    def test_processed_stop_and_tool_constraint_propagate(self):
        request = ResponsesRequest(model="x", input="call the tool", store=False)
        params = request.to_sampling_params(
            default_max_tokens=128,
            default_params={},
            stop=["</s>"],
            tool_call_constraint=("json_schema", {"type": "object"}),
        )
        self.assertEqual(params["stop"], ["</s>"])
        self.assertEqual(params["json_schema"], '{"type": "object"}')

    def test_constraint_conflict_raises(self):
        request = ResponsesRequest(model="x", input="hi", store=False)
        with self.assertRaises(ValueError):
            request.to_sampling_params(
                default_max_tokens=128,
                default_params={"json_schema": '{"type": "object"}'},
                tool_call_constraint=("json_schema", {"type": "object"}),
            )

    def test_structural_tag_with_model_dump(self):
        class _FakeStructuralTag:
            def model_dump(self, by_alias=False):
                return {"type": "structural_tag"}

        request = ResponsesRequest(model="x", input="hi", store=False)
        params = request.to_sampling_params(
            default_max_tokens=128,
            default_params={},
            tool_call_constraint=("structural_tag", _FakeStructuralTag()),
        )
        self.assertEqual(params["structural_tag"], '{"type": "structural_tag"}')

    def test_text_format_maps_to_json_schema_constraint(self):
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        for fmt, expected in [
            ({"type": "json_schema", "name": "p", "schema": schema}, schema),
            ({"type": "json_object"}, {"type": "object"}),
        ]:
            req = ResponsesRequest(
                model="x", input="hi", store=False, text={"format": fmt}
            )
            params = req.to_sampling_params(default_max_tokens=128, default_params={})
            self.assertEqual(json.loads(params["json_schema"]), expected, fmt)
        plain = ResponsesRequest(
            model="x", input="hi", store=False, text={"format": {"type": "text"}}
        ).to_sampling_params(default_max_tokens=128, default_params={})
        self.assertNotIn("json_schema", plain)

    def test_text_format_conflicts_with_tool_constraint(self):
        request = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "p",
                    "schema": {"type": "object"},
                }
            },
        )
        # The message must name text.format: it is the only source of the
        # conflict a /v1/responses caller can actually set.
        with self.assertRaisesRegex(ValueError, r"text\.format"):
            request.to_sampling_params(
                default_max_tokens=128,
                default_params={},
                tool_call_constraint=("json_schema", {"type": "object"}),
            )


class IncludeOutputLogprobsTestCase(CustomTestCase):
    def test_detected_only_for_logprobs_include(self):
        def has(include):
            return ResponsesRequest(
                model="x", input="hi", store=False, include=include
            ).is_include_output_logprobs()

        self.assertTrue(has(["message.output_text.logprobs"]))
        self.assertFalse(has(None))
        self.assertFalse(has(["reasoning.encrypted_content"]))


class ThinkingControlTestCase(CustomTestCase):
    def test_effort_none_disables_thinking(self):
        ctk = ResponsesRequest(
            model="x", input="hi", store=False, reasoning={"effort": "none"}
        ).chat_template_kwargs
        self.assertEqual((ctk["enable_thinking"], ctk["thinking"]), (False, False))

    def test_thinking_untouched_otherwise(self):
        # grammar-constrained requests keep thinking on; ReasonerGrammarBackend
        # defers the grammar past </think> so the two coexist.
        for kw in (
            {"reasoning": {"effort": "medium"}},
            {"text": {"format": {"type": "text"}}},
            {
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "p",
                        "schema": {"type": "object"},
                    }
                }
            },
            {"text": {"format": {"type": "json_object"}}},
            {"tool_choice": "required"},
            {"tool_choice": {"type": "function", "name": "f"}},
            {},
        ):
            req = ResponsesRequest(model="x", input="hi", store=False, **kw)
            self.assertIsNone(req.chat_template_kwargs, kw)

    def test_explicit_chat_template_kwargs_preserved(self):
        req = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            chat_template_kwargs={"enable_thinking": True},
        )
        self.assertTrue(req.chat_template_kwargs["enable_thinking"])
        # a lone toggle fills the other family's key rather than being ignored
        self.assertIs(req.chat_template_kwargs["thinking"], True)

    def test_single_thinking_key_is_mirrored(self):
        """A lone toggle for one model family fills the other family's key,
        before --default-chat-template-kwargs can re-enable thinking."""
        for value in (True, False):
            for key in ("thinking", "enable_thinking"):
                with self.subTest(key=key, value=value):
                    req = ResponsesRequest(
                        model="x",
                        input="hi",
                        store=False,
                        chat_template_kwargs={key: value},
                    )
                    self.assertEqual(
                        req.chat_template_kwargs,
                        {"thinking": value, "enable_thinking": value},
                    )

    def test_single_thinking_key_mirroring_does_not_mutate_caller_dict(self):
        """A reused options dict must not come back pre-filled with the other
        family's key, or flipping the toggle yields a conflicting request."""
        for value in (True, False):
            for key in ("thinking", "enable_thinking"):
                with self.subTest(key=key, value=value):
                    options = {key: value}
                    req = ResponsesRequest(
                        model="x",
                        input="hi",
                        store=False,
                        chat_template_kwargs=options,
                    )
                    self.assertEqual(
                        req.chat_template_kwargs,
                        {"thinking": value, "enable_thinking": value},
                    )
                    self.assertEqual(
                        options,
                        {key: value},
                        "normalizer wrote through to the caller's dict",
                    )

                    options[key] = not value
                    second = ResponsesRequest(
                        model="x",
                        input="hi",
                        store=False,
                        chat_template_kwargs=options,
                    )
                    self.assertEqual(
                        second.chat_template_kwargs,
                        {"thinking": not value, "enable_thinking": not value},
                    )
                    self.assertEqual(options, {key: not value})

    def test_single_thinking_key_mirroring_leaves_conflicts_and_nonbooleans(self):
        req = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            chat_template_kwargs={"thinking": False, "enable_thinking": True},
        )
        self.assertEqual(
            req.chat_template_kwargs, {"thinking": False, "enable_thinking": True}
        )
        req = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            chat_template_kwargs={"thinking": "auto", "other": 1},
        )
        self.assertEqual(req.chat_template_kwargs, {"thinking": "auto", "other": 1})
        req = ResponsesRequest(model="x", input="hi", store=False)
        self.assertIsNone(req.chat_template_kwargs)


class ResponsesResponseFromRequestTestCase(CustomTestCase):
    def test_requested_text_format_is_echoed(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        request = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            text={"format": {"type": "json_schema", "name": "p", "schema": schema}},
        )
        response = ResponsesResponse.from_request(
            request,
            sampling_params={},
            model_name="x",
            created_time=0,
            output=[],
            status="completed",
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        self.assertEqual(response.text["format"]["type"], "json_schema")

    def test_parallel_tool_calls_false_preserved(self):
        request = ResponsesRequest(
            model="x", input="hi", parallel_tool_calls=False, store=False
        )
        response = ResponsesResponse.from_request(
            request,
            sampling_params={},
            model_name="x",
            created_time=0,
            output=[],
            status="completed",
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        self.assertFalse(response.parallel_tool_calls)

    def test_incomplete_status_sets_incomplete_details(self):
        request = ResponsesRequest(model="x", input="hi", store=False)
        incomplete = ResponsesResponse.from_request(
            request,
            sampling_params={},
            model_name="x",
            created_time=0,
            output=[],
            status="incomplete",
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        self.assertEqual(incomplete.status, "incomplete")
        self.assertEqual(incomplete.incomplete_details, {"reason": "max_output_tokens"})

        completed = ResponsesResponse.from_request(
            request,
            sampling_params={},
            model_name="x",
            created_time=0,
            output=[],
            status="completed",
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        self.assertIsNone(completed.incomplete_details)

    def test_usage_serialized_in_responses_shape(self):
        request = ResponsesRequest(model="x", input="hi", store=False)
        resp = ResponsesResponse.from_request(
            request,
            sampling_params={},
            model_name="x",
            created_time=0,
            output=[],
            status="completed",
            usage=UsageInfo(
                prompt_tokens=11,
                completion_tokens=102,
                total_tokens=113,
                reasoning_tokens=7,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=3),
            ),
        )
        usage = resp.model_dump()["usage"]
        self.assertEqual(usage["input_tokens"], 11)
        self.assertEqual(usage["output_tokens"], 102)
        self.assertEqual(usage["total_tokens"], 113)
        self.assertEqual(usage["output_tokens_details"]["reasoning_tokens"], 7)
        self.assertEqual(usage["input_tokens_details"]["cached_tokens"], 3)
        # Chat-style keys must be gone.
        self.assertNotIn("prompt_tokens", usage)
        self.assertNotIn("completion_tokens", usage)

    def test_only_sdk_known_efforts_echoed_so_streaming_event_validates(self):
        import openai.types.responses as ort

        # OpenAI's Reasoning.effort literal is narrower than our tier list:
        # "none" is a request-side extension and xhigh/max postdate it. Echoing
        # one of those raises a ValidationError in the typed event below, which
        # the stream generator builds before its try block -- the connection
        # then dies without a single SSE byte.
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            resp = _in_progress_response(
                ResponsesRequest(
                    model="x", input="hi", store=False, reasoning={"effort": effort}
                )
            )
            expected = (
                effort if effort in ("minimal", "low", "medium", "high") else None
            )
            self.assertEqual(resp.reasoning["effort"], expected, effort)
            ort.ResponseCreatedEvent(
                type="response.created", sequence_number=0, response=resp.model_dump()
            )


class InputItemStringIdTestCase(CustomTestCase):
    """A response.output item replayed into input (string id + content) must
    keep its content rather than collapse to an item-reference."""

    def test_string_id_dropped_only_for_content_items(self):
        norm = ResponsesRequest._normalize_input_item_for_validation
        kept = norm(
            {
                "id": "msg_x",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            }
        )
        self.assertNotIn("id", kept)
        self.assertTrue(kept["content"])
        # int ids and bare item-references are left alone
        self.assertEqual(norm({"id": 123, "content": [{"type": "text"}]})["id"], 123)
        ref = {"type": "item_reference", "id": "msg_ref"}
        self.assertEqual(norm(ref), ref)

    def test_request_accepts_replayed_output_item(self):
        # Construction must not raise (the bug returned 400).
        ResponsesRequest(
            model="x",
            store=False,
            input=[
                {
                    "id": "msg_x",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ],
        )


class ToolChoiceObjectFormTestCase(CustomTestCase):
    def test_echoed_choice_is_what_the_server_honors(self):
        import openai.types.responses as ort

        named = {"type": "function", "name": "get_weather"}
        # Object forms other than a named function cannot be forced through the
        # tool-call parser, so the response must echo the "auto" we actually
        # run -- which also keeps it inside the SDK's ToolChoice union that the
        # typed event below validates against.
        for tool_choice, expected in (
            ("auto", "auto"),
            ("required", "required"),
            ("none", "none"),
            (named, named),
            ({"type": "function", "function": {"name": "get_weather"}}, named),
            ({"type": "web_search"}, "auto"),
            ({"type": "mcp", "server_label": "s"}, "auto"),
        ):
            req = ResponsesRequest(
                model="x", input="hi", store=False, tool_choice=tool_choice
            )
            self.assertEqual(req.effective_tool_choice(), expected, tool_choice)
            resp = _in_progress_response(req)
            self.assertEqual(resp.tool_choice, expected, tool_choice)
            ort.ResponseCreatedEvent(
                type="response.created", sequence_number=0, response=resp.model_dump()
            )


def _fields(item):
    """One input item as the serving layer reads it: SDK objects dump, wire
    dicts pass through untouched."""
    if hasattr(item, "model_dump"):
        item = item.model_dump(exclude_none=True)
    return item


def _call(item_type="function_call", name="run", namespace="one", **extra):
    item = {
        "type": item_type,
        "call_id": "call_1",
        "name": name,
        "namespace": namespace,
    }
    item["input" if item_type == "custom_tool_call" else "arguments"] = "{}"
    item.update(extra)
    return item


def _union_members(annotation):
    for arg in get_args(annotation):
        yield arg
        yield from _union_members(arg)


class ResponsesInputItemUnionTestCase(CustomTestCase):
    """``input`` must select the open dict arm before the SDK TypedDict arms.

    Smart union selection lets a TypedDict arm win over the dict arm, and
    TypedDicts silently discard fields they do not declare -- ``namespace``
    first, anything newer after it. The loss happened inside request
    validation, before any adapter or renderer saw the item (pydantic 2.14 /
    pydantic-core 2.50 regression over openai 2.6.1).
    """

    def _roundtrip(self, request):
        return ResponsesRequest.model_validate_json(request.model_dump_json())

    def _assert_fields_survive(self, item):
        """Every field the client sent must reach the serving layer, both
        through direct construction and through a JSON revalidation (stored
        response replay). Null padding a coerced SDK object adds is not field
        loss; a dropped field is."""
        request = ResponsesRequest(model="x", store=False, input=[item])
        for stage, revalidated in (
            ("validate", request),
            ("roundtrip", self._roundtrip(request)),
        ):
            fields = _fields(revalidated.input[0])
            for key, value in item.items():
                self.assertEqual(
                    fields.get(key, "<dropped>"), value, (stage, key, item)
                )

    def test_namespaced_function_call_keeps_its_identity(self):
        self._assert_fields_survive(_call())

    def test_namespaced_custom_tool_call_keeps_its_identity(self):
        self._assert_fields_survive(
            _call("custom_tool_call", name="exec", namespace="functions", input="code")
        )

    def test_fields_the_sdk_does_not_declare_yet_survive(self):
        self._assert_fields_survive(_call(future_field={"x": 1}))

    def test_constructed_sdk_objects_are_still_accepted(self):
        function_call = ResponseFunctionToolCall(
            type="function_call",
            call_id="call_1",
            name="run",
            arguments="{}",
            namespace="one",
        )
        custom_call = ResponseCustomToolCall(
            type="custom_tool_call",
            call_id="call_2",
            name="exec",
            input="code",
            namespace="functions",
        )
        reasoning = ResponseReasoningItem(
            id="rs_1",
            type="reasoning",
            summary=[],
            content=[{"type": "reasoning_text", "text": "think"}],
        )
        request = ResponsesRequest(
            model="x", store=False, input=[function_call, custom_call, reasoning]
        )
        for revalidated in (request, self._roundtrip(request)):
            first, second, third = map(_fields, revalidated.input)
            self.assertEqual(first["namespace"], "one", first)
            self.assertEqual(second["namespace"], "functions", second)
            self.assertEqual(second["input"], "code", second)
            self.assertEqual(third["content"][0]["text"], "think", third)

    def test_the_input_union_still_offers_every_sdk_item_arm(self):
        # protocol.py binds ResponseInputOutputItem twice; the alias defined
        # after ResponsesRequest is narrower and must not become the field's.
        members = list(
            _union_members(ResponsesRequest.model_fields["input"].annotation)
        )
        self.assertTrue(any(get_origin(m) is dict for m in members), members)
        for arm in (ResponseFunctionToolCall, ResponseCustomToolCall):
            self.assertIn(arm, members)

    def test_ordinary_wire_items_still_validate(self):
        self.assertEqual(ResponsesRequest(model="x", input="hello").input, "hello")
        request = ResponsesRequest(
            model="x",
            store=False,
            input=[
                {"role": "user", "content": "before"},
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "done",
                },
                _call(),
                {
                    "type": "additional_tools",
                    "tools": [{"type": "function", "name": "run", "parameters": {}}],
                },
            ],
        )
        self.assertEqual(len(request.input), 4)
        self.assertEqual(_fields(request.input[3])["tools"][0]["name"], "run")
        self.assertEqual(len(self._roundtrip(request).input), 4)

    def test_invalid_namespace_is_retained_for_the_http_400_path(self):
        # /v1/responses rejects a non-string namespace itself; validation must
        # keep the offending value so that rejection names it instead of
        # reporting a missing one.
        request = ResponsesRequest(model="x", store=False, input=[_call(namespace=3)])
        self.assertEqual(_fields(request.input[0]).get("namespace", "<dropped>"), 3)
        self.assertEqual(
            _fields(self._roundtrip(request).input[0]).get("namespace", "<dropped>"), 3
        )

    def test_retired_identities_that_would_collide_stay_distinct(self):
        # (a, b__c) and (a__b, c) flatten to the same spelling without a
        # namespace, so history replay cannot rely on the pair surviving.
        retired = [
            _call(name="b__c", namespace="a"),
            _call(name="c", namespace="a__b", call_id="call_2"),
        ]
        request = self._roundtrip(
            ResponsesRequest(model="x", store=False, input=retired)
        )
        self.assertEqual(
            [
                (_fields(i).get("namespace", "<dropped>"), _fields(i).get("name"))
                for i in request.input
            ],
            [("a", "b__c"), ("a__b", "c")],
        )


if __name__ == "__main__":
    unittest.main()
