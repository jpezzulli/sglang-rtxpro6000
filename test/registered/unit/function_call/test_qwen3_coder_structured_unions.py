"""Structured-union regressions adapted from upstream SGLang PR #44124."""

import json
from unittest.mock import patch

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

STRING = {"type": "string"}
STRING_ARRAY = {"type": "array", "items": STRING}
LOCATION = {
    "type": "object",
    "properties": {"city": STRING},
    "required": ["city"],
}
GITHUB_REPOSITORIES = {"anyOf": [STRING, STRING_ARRAY, {"type": "null"}]}


@pytest.fixture(params=[None, 1, 7, 10000], ids=["complete", "char", "seven", "whole"])
def width(request):
    return request.param


def tools_for(schema, definitions=None):
    return [
        Tool(
            function=Function(
                name="record",
                parameters={
                    "type": "object",
                    "properties": {"value": schema},
                    "$defs": definitions or {},
                },
            )
        )
    ]


def tool_call(value, name="record"):
    return (
        f"<tool_call><function={name}><parameter=value>{value}"
        "</parameter></function></tool_call>"
    )


def parse(source, tools, width, forward=False):
    detector = Qwen3CoderDetector()
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(forward):
        if width is None:
            result = detector.detect_and_parse(source, tools)
            normal, calls = result.normal_text, result.calls
        else:
            normal, calls = "", []
            for offset in range(0, len(source), width):
                result = detector.parse_streaming_increment(
                    source[offset : offset + width], tools
                )
                normal += result.normal_text
                calls.extend(result.calls)
            result = detector.finish(tools)
            normal += result.normal_text
            calls.extend(result.calls)

    reconstructed = {}
    for call in calls:
        if call.name:
            assert call.tool_index not in reconstructed
            reconstructed[call.tool_index] = {"name": call.name, "arguments": ""}
        assert call.tool_index in reconstructed, "arguments without a tool name"
        reconstructed[call.tool_index]["arguments"] += call.parameters
    assert list(reconstructed) == list(range(len(reconstructed)))
    return normal, [
        (call["name"], json.loads(call["arguments"])) for call in reconstructed.values()
    ]


def check_value(schema, raw, expected, width, definitions=None):
    assert parse(tool_call(raw), tools_for(schema, definitions), width) == (
        "",
        [("record", {"value": expected})],
    )


def test_github_string_array_null_regression(width):
    check_value(
        GITHUB_REPOSITORIES,
        '["owner/one","owner/two"]',
        ["owner/one", "owner/two"],
        width,
    )


def test_array_only_control(width):
    check_value(
        STRING_ARRAY,
        '["owner/one","owner/two"]',
        ["owner/one", "owner/two"],
        width,
    )


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
@pytest.mark.parametrize("string_first", [True, False])
@pytest.mark.parametrize(
    "structured,value",
    [
        (LOCATION, {"city": "Paris"}),
        (STRING_ARRAY, ["owner/one", "owner/two"]),
        ({"type": "array", "items": LOCATION}, [{"city": "Paris"}]),
    ],
)
def test_structured_string_unions(keyword, string_first, structured, value, width):
    branches = [STRING, structured] if string_first else [structured, STRING]
    check_value({keyword: branches}, json.dumps(value), value, width)


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
@pytest.mark.parametrize(
    "raw,expected", [(' ["owner/one"] ', ["owner/one"]), ("null", None)]
)
def test_nullable_arrays_remain_compatible(keyword, raw, expected, width):
    check_value({keyword: [STRING_ARRAY, {"type": "null"}]}, raw, expected, width)
    check_value(GITHUB_REPOSITORIES, raw, expected, width)


@pytest.mark.parametrize(
    "schema,raw",
    [
        (STRING, '{"city":"Paris"}'),
        (STRING, '["owner/one"]'),
        (GITHUB_REPOSITORIES, "  owner/one  "),
        (GITHUB_REPOSITORIES, '"owner/one"'),
        (GITHUB_REPOSITORIES, '["owner/one"'),
        ({"anyOf": [STRING, LOCATION]}, '{"city":'),
        ({"anyOf": [{"type": "number"}, STRING]}, "250"),
        ({"anyOf": [{"type": "boolean"}, STRING]}, "true"),
        ({"anyOf": [STRING, {"type": "null"}]}, "{}"),
    ],
)
def test_string_values_remain_strings(schema, raw, width):
    check_value(schema, raw, raw, width)


@pytest.mark.parametrize(
    "schema,raw",
    [
        ({"anyOf": [STRING, LOCATION]}, '{"country":"France"}'),
        ({"anyOf": [STRING, LOCATION]}, '{"city":123}'),
        (GITHUB_REPOSITORIES, "[123]"),
        ({"anyOf": [STRING, {**STRING_ARRAY, "minItems": 1}]}, "[]"),
        ({"oneOf": [STRING, LOCATION, {"type": "object"}]}, '{"city":"Paris"}'),
    ],
)
def test_structures_require_schema_match(schema, raw, width):
    check_value(schema, raw, raw, width)


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"city":"Paris"}', {"city": "Paris"}),
        ('{"city":123}', '{"city":123}'),
        ('{"country":"France"}', '{"country":"France"}'),
    ],
)
def test_references_use_enclosing_schema(keyword, raw, expected, width):
    definitions = {
        "City": STRING,
        "Location": {**LOCATION, "properties": {"city": {"$ref": "#/$defs/City"}}},
    }
    schema = {keyword: [STRING, {"$ref": "#/$defs/Location"}]}
    check_value(schema, raw, expected, width, definitions)


@pytest.mark.parametrize(
    "branch,raw",
    [
        ({"$ref": "#/$defs/missing"}, "{}"),
        ({"$id": "urn:recursive", "$ref": "urn:recursive"}, "{}"),
        ({"type": "map"}, "{}"),
        ({"type": "array", "items": [STRING]}, '["Paris"]'),
    ],
)
def test_unresolved_or_invalid_schemas_preserve_strings(branch, raw, width):
    check_value({"anyOf": [STRING, branch]}, raw, raw, width)


def test_external_references_never_fetch(width):
    with patch(
        "urllib.request.urlopen", side_effect=AssertionError("Unexpected schema fetch")
    ) as urlopen:
        schema = {"anyOf": [STRING, {"$ref": "https://example.invalid/schema"}]}
        check_value(schema, "{}", "{}", width)
        urlopen.assert_not_called()


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e309", "-1e309"])
@pytest.mark.parametrize("container", ['{{"number":{}}}', "[{}]"])
def test_nonfinite_numbers_remain_strings(number, container, width):
    schema = {"anyOf": [STRING, {"type": "object"}, {"type": "array"}]}
    raw = container.format(number)
    check_value(schema, raw, raw, width)


def test_finite_numbers_and_quoted_nonfinite_names_are_structured(width):
    schema = {"anyOf": [STRING, {"type": "array"}]}
    check_value(
        schema,
        '[1.5,1e3,"NaN","Infinity"]',
        [1.5, 1000.0, "NaN", "Infinity"],
        width,
    )


def test_plain_union_text_skips_json_decoding():
    detector = Qwen3CoderDetector()
    with patch("sglang.srt.function_call.qwen3_coder_detector.json.loads") as loads:
        raw = "  owner/one  "
        assert (
            detector._convert_param_value(
                raw, "value", {"value": GITHUB_REPOSITORIES}, "record"
            )
            == raw
        )
        loads.assert_not_called()


@pytest.mark.parametrize("forward", [False, True])
def test_union_conversion_preserves_example_and_unknown_tool_guards(forward, width):
    raw = '["owner/one","owner/two"]'
    example = "Example:\n```xml\n" + tool_call(raw) + "\n```\n"
    unknown = tool_call(raw, name="unknown")
    source = example + unknown + tool_call(raw)
    expected_calls = [("record", {"value": ["owner/one", "owner/two"]})]
    if forward:
        expected_calls.insert(0, ("unknown", {"value": raw}))
    assert parse(source, tools_for(GITHUB_REPOSITORIES), width, forward) == (
        example + ("" if forward else unknown),
        expected_calls,
    )
