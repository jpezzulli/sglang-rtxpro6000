"""Markdown fenced-code boundaries for the Qwen3 Coder tool parser."""

import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

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


def wrapped(value="echo hi"):
    return (
        "<tool_call>\n"
        f"<function=run_commands><parameter=command>{value}</parameter></function>\n"
        "</tool_call>"
    )


def parse(source, width):
    parser = FunctionCallParser(TOOLS, "qwen3_coder")
    if width is None:
        normal, increments = parser.parse_non_stream(source)
    else:
        normal, increments = "", []
        for start in range(0, len(source), width):
            text, calls = parser.parse_stream_chunk(source[start : start + width])
            normal += text
            increments.extend(calls)
        text, calls = parser.parse_stream_end()
        normal += text
        increments.extend(calls)

    calls_by_index = {}
    for call in increments:
        if call.name:
            calls_by_index[call.tool_index] = {"name": call.name, "arguments": ""}
        calls_by_index[call.tool_index]["arguments"] += call.parameters
    calls = [
        (call["name"], json.loads(call["arguments"]))
        for call in calls_by_index.values()
    ]
    return normal, calls


@pytest.mark.parametrize("width", [None, 1, 2, 7, 10000])
@pytest.mark.parametrize(
    ("opening", "closing"),
    [
        ("```xml", "```"),
        ("   ````text", "  `````"),
        ("~~~", "~~~"),
        (" ~~~~ example", "   ~~~~~"),
    ],
)
def test_declared_calls_in_complete_fences_remain_literal(opening, closing, width):
    source = f"Example:\n{opening}\n{wrapped()}\n{closing}\nNot an invocation.\n"
    assert parse(source, width) == (source, [])


@pytest.mark.parametrize("width", [None, 1, 3, 11, 10000])
def test_unfinished_fence_remains_literal_through_end_of_stream(width):
    source = "Example:\n```xml\n" + wrapped() + "\nno closing delimiter"
    assert parse(source, width) == (source, [])


@pytest.mark.parametrize("width", [None, 1, 2, 9, 10000])
def test_shorter_delimiter_does_not_close_fence(width):
    source = (
        "````xml\n"
        + wrapped("first example")
        + "\n```\n"
        + wrapped("second example")
        + "\n````\n"
    )
    assert parse(source, width) == (source, [])


@pytest.mark.parametrize("width", [None, 1, 5, 17, 10000])
def test_real_calls_before_and_after_fenced_example_are_preserved(width):
    example = "\nExample:\n```xml\n" + wrapped("not a call") + "\n```\nDone.\n"
    source = wrapped("before") + example + wrapped("after")
    assert parse(source, width) == (
        example,
        [
            ("run_commands", {"command": "before"}),
            ("run_commands", {"command": "after"}),
        ],
    )


@pytest.mark.parametrize("width", [None, 1, 4, 13, 10000])
def test_markdown_fences_inside_real_arguments_are_not_parser_boundaries(width):
    command = "Use `printf`:\n```sh\nprintf 'hello'\n```"
    assert parse(wrapped(command), width) == (
        "",
        [("run_commands", {"command": command})],
    )


@pytest.mark.parametrize("width", [None, 1, 7, 10000])
def test_four_space_indent_and_inline_backticks_keep_legacy_tool_semantics(width):
    source = "Inline `example`: " + wrapped("inline")
    normal, calls = parse(source, width)
    assert normal == "Inline `example`: "
    assert calls == [("run_commands", {"command": "inline"})]

    indented = "    ```xml\n" + wrapped("four spaces") + "\n    ```\n"
    normal, calls = parse(indented, width)
    assert normal == "    ```xml\n\n    ```\n"
    assert calls == [("run_commands", {"command": "four spaces"})]


def test_streaming_fence_body_progress_only_holds_ambiguous_close_prefix():
    parser = FunctionCallParser(TOOLS, "qwen3_coder")

    text, calls = parser.parse_stream_chunk("```xml\n")
    assert (text, calls) == ("```xml\n", [])

    long_line = "ordinary fenced body " + "x" * 8192
    text, calls = parser.parse_stream_chunk(long_line)
    assert (text, calls) == (long_line, [])

    literal_call = wrapped("still an example")
    text, calls = parser.parse_stream_chunk(literal_call)
    assert (text, calls) == (literal_call, [])

    # A possible delimiter is the only tail held across chunks.
    text, calls = parser.parse_stream_chunk("\n  ``")
    assert (text, calls) == ("\n", [])
    text, calls = parser.parse_stream_chunk("`\nAfter.\n")
    assert (text, calls) == ("  ```\nAfter.\n", [])


def test_streaming_midline_backticks_do_not_hide_later_real_fence():
    parser = FunctionCallParser(TOOLS, "qwen3_coder")
    prefix = "Inline marker: "
    assert parser.parse_stream_chunk(prefix) == (prefix, [])

    remainder = "``` is prose\n```xml\n" + wrapped() + "\n```\n"
    text, calls = parser.parse_stream_chunk(remainder)
    assert (text, calls) == (remainder, [])
    assert parser.parse_stream_end() == ("", [])


@pytest.mark.parametrize("width", [None, 1, 2, 11, 10000])
@pytest.mark.parametrize("line_end", ["\r", "\r\n"])
def test_cr_line_endings_preserve_complete_fenced_examples(line_end, width):
    source = (
        f"Example:{line_end}```xml{line_end}"
        + wrapped("EXAMPLE_ONLY")
        + f"{line_end}```{line_end}Done"
    )
    assert parse(source, width) == (source, [])


@pytest.mark.parametrize("width", [None, 1, 7, 10000])
def test_bare_cr_unfinished_fence_remains_literal(width):
    source = "Example:\r~~~xml\r" + wrapped("EXAMPLE_ONLY") + "\rstill code"
    assert parse(source, width) == (source, [])


def test_streaming_crlf_split_across_increments_preserves_exact_text():
    chunks = [
        "Example:\r",
        "\n```xml\r",
        "\n",
        wrapped("EXAMPLE_ONLY"),
        "\r",
        "\n```\r",
        "\nDone",
    ]
    parser = FunctionCallParser(TOOLS, "qwen3_coder")
    normal, calls = "", []
    for chunk in chunks:
        text, increments = parser.parse_stream_chunk(chunk)
        normal += text
        calls.extend(increments)
    text, increments = parser.parse_stream_end()
    normal += text
    calls.extend(increments)

    assert normal == "".join(chunks)
    assert calls == []


@pytest.mark.parametrize("width", [None, 1, 5, 10000])
def test_cr_and_crlf_inside_real_arguments_are_not_normalized(width):
    command = "first\rsecond\r\nthird"
    assert parse(wrapped(command), width) == (
        "",
        [("run_commands", {"command": command})],
    )
