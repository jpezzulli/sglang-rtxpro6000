"""Wrapper/name validation without losing rejected markup or later calls."""

import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

TOOLS = [
    Tool(
        function=Function(
            name="run_commands",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        )
    )
]


def function(name, value="echo hi"):
    return f"<function={name}><parameter=command>{value}</parameter></function>"


def wrapped(name, value="echo hi"):
    return f"<tool_call>\n{function(name, value)}\n</tool_call>"


def parse(text, width, forward=False):
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(forward):
        parser = FunctionCallParser(TOOLS, "qwen3_coder")
        if width is None:
            normal, calls = parser.parse_non_stream(text)
        else:
            normal, calls = "", []
            for i in range(0, len(text), width):
                content, increment = parser.parse_stream_chunk(text[i : i + width])
                normal += content
                calls.extend(increment)
            content, increment = parser.parse_stream_end()
            normal += content
            calls.extend(increment)
    reconstructed = {}
    for call in calls:
        if call.name:
            assert call.tool_index not in reconstructed
            reconstructed[call.tool_index] = {"name": call.name, "arguments": ""}
        assert call.tool_index in reconstructed, "argument fragment without a tool name"
        reconstructed[call.tool_index]["arguments"] += call.parameters
    assert list(reconstructed) == list(range(len(reconstructed)))
    return normal, [
        (call["name"], json.loads(call["arguments"])) for call in reconstructed.values()
    ]


@pytest.mark.parametrize("width", [None, 1, 2, 5, 13, 10000])
@pytest.mark.parametrize(
    "markup",
    [
        function("run_commands"),
        function("unknown"),
        wrapped("run_command"),
        "<tool_call></function></tool_call>",
    ],
)
def test_rejected_markup_remains_literal_text(markup, width):
    source = "Example:\n```xml\n" + markup + "\n```\nNot an invocation.\n"
    assert parse(source, width) == (source, [])


@pytest.mark.parametrize("width", [None, 1, 2, 5, 13, 10000])
def test_unknown_then_valid_keeps_text_and_exact_arguments(width):
    rejected = "Example: " + wrapped("run_command") + "\nThat was an example.\n"
    source = rejected + wrapped("run_commands", 'a "quote" and \\ slash')
    assert parse(source, width) == (
        rejected,
        [("run_commands", {"command": 'a "quote" and \\ slash'})],
    )


@pytest.mark.parametrize("width", [None, 1, 7, 10000])
def test_explicit_forward_unknown_policy_still_requires_wrapper(width):
    source = wrapped("run_command")
    assert parse(source, width, forward=True) == (
        "",
        [("run_command", {"command": "echo hi"})],
    )
    bare = function("run_command")
    assert parse(bare, width, forward=True) == (bare, [])


@pytest.mark.parametrize("width", [None, 1, 7, 10000])
def test_valid_parallel_and_fully_wrapped_declared_quotation_remain_calls(width):
    # This correction validates syntax/name, not Markdown or the model's intent.
    source = (
        "Example:\n```xml\n"
        + wrapped("run_commands")
        + wrapped("run_commands", "second")
    )
    assert parse(source, width) == (
        "Example:\n```xml\n",
        [
            ("run_commands", {"command": "echo hi"}),
            ("run_commands", {"command": "second"}),
        ],
    )


@pytest.mark.parametrize("width", [None, 1, 7, 10000])
def test_bare_markup_before_unclosed_wrapper_is_not_harvested(width):
    bare = function("run_commands", "bare")
    source = bare + "<tool_call><function=unknown></function>"
    assert parse(source, width) == (source, [])


@pytest.mark.parametrize("width", [None, 1, 2, 7, 10000])
def test_unknown_then_valid_in_same_wrapper_does_not_poison_json(width):
    unknown = function("unknown")
    source = "<tool_call>" + unknown + function("run_commands") + "</tool_call>"
    assert parse(source, width) == (
        "<tool_call>" + unknown + "</tool_call>",
        [("run_commands", {"command": "echo hi"})],
    )


@pytest.mark.parametrize("width", [1, 2, 7, 10000])
def test_incomplete_unknown_and_stray_parameters_do_not_emit_fragments(width):
    source = (
        "<tool_call><function=unknown><parameter=command>unfinished</tool_call>"
        + wrapped("run_commands")
    )
    normal, calls = parse(source, width)
    assert normal == source[: source.index("<tool_call>", 1)]
    assert calls == [("run_commands", {"command": "echo hi"})]


@pytest.mark.parametrize(
    "source",
    [
        "<tool_call>",
        "<tool_call>\n<function=",
        "<tool_call><parameter=command>value</parameter></tool_call>",
    ],
)
def test_stream_end_preserves_unrecognized_or_incomplete_wrapper(source):
    assert parse(source, 1) == (source, [])


@pytest.mark.parametrize("width", [None, 1, 7, 10000])
def test_reasoning_parser_excludes_thought_markup_before_tool_parser(width):
    # A full wrapper intentionally ends Qwen reasoning, even without </think>.
    # Bare markup in a closed thinking section must remain reasoning instead.
    thought = "Consider " + function("run_commands", "not a call")
    literal = "Quoted " + wrapped("unknown") + "\n"
    source = "<think>" + thought + "</think>" + literal + wrapped("run_commands")
    reasoning_parser = ReasoningParser(model_type="qwen3", stream_reasoning=True)
    tool_parser = FunctionCallParser(TOOLS, "qwen3_coder")
    if width is None:
        reasoning, content = reasoning_parser.parse_non_stream(source)
        normal, calls = tool_parser.parse_non_stream(content)
    else:
        reasoning, normal, calls = "", "", []
        for i in range(0, len(source), width):
            thought_chunk, content = reasoning_parser.parse_stream_chunk(
                source[i : i + width]
            )
            reasoning += thought_chunk or ""
            if content:
                text, increments = tool_parser.parse_stream_chunk(content)
                normal += text
                calls.extend(increments)
        thought_chunk, content = reasoning_parser.parse_stream_end()
        reasoning += thought_chunk or ""
        if content:
            text, increments = tool_parser.parse_stream_chunk(content)
            normal += text
            calls.extend(increments)
        text, increments = tool_parser.parse_stream_end()
        normal += text
        calls.extend(increments)
    assert reasoning == thought
    assert normal == literal
    assert [call.name for call in calls if call.name] == ["run_commands"]
    assert json.loads("".join(call.parameters for call in calls)) == {
        "command": "echo hi"
    }


@pytest.mark.parametrize("width", [1, 7, 10000])
def test_implicit_reasoning_close_still_validates_unknown_name(width):
    literal = wrapped("unknown")
    source = "<think>Consider " + literal
    reasoning_parser = ReasoningParser(model_type="qwen3", stream_reasoning=True)
    tool_parser = FunctionCallParser(TOOLS, "qwen3_coder")
    normal, calls = "", []
    for i in range(0, len(source), width):
        _, content = reasoning_parser.parse_stream_chunk(source[i : i + width])
        if content:
            text, increments = tool_parser.parse_stream_chunk(content)
            normal += text
            calls.extend(increments)
    text, increments = tool_parser.parse_stream_end()
    assert normal + text == literal
    assert calls + increments == []
