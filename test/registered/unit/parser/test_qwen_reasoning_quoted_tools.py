"""Quoted tool examples must not become calls when reasoning is streamed."""

import pytest

from sglang.srt.parser.reasoning_parser import (
    BaseReasoningFormatDetector,
    Qwen3Detector,
    _DeepSeekV3Detector,
    _FlintDetector,
    _MimoDetector,
    _PoolsideV1Detector,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CALL = "<tool_call><function=ping></function></tool_call>"
JSON_CALL = '<tool_call>{"name":"ping","arguments":{}}</tool_call>'


def feed(text, size, **kwargs):
    detector = Qwen3Detector(**kwargs)
    results = [
        detector.parse_streaming_increment(text[i : i + size])
        for i in range(0, len(text), size)
    ]
    results.append(detector.finish())
    return (
        "".join(r.reasoning_text for r in results),
        "".join(r.normal_text for r in results),
    )


@pytest.mark.parametrize(
    "wrapper", ["`{}`", "``{}``", "\n```xml\n{}\n```\n", "\n~~~xml\n{}\n~~~\n"]
)
@pytest.mark.parametrize("call", [CALL, JSON_CALL])
def test_quoted_call_stays_reasoning_across_chunks(wrapper, call):
    reasoning = "Discuss " + wrapper.format(call) + " literally. Continue reasoning."
    text = "<think>" + reasoning + "</think>Final answer."
    for stream in (True, False):
        for size in (1, 2, 7, len(text)):
            assert feed(text, size, stream_reasoning=stream) == (
                reasoning,
                "Final answer.",
            )
    parsed = Qwen3Detector().detect_and_parse(text)
    assert (parsed.reasoning_text, parsed.normal_text) == (reasoning, "Final answer.")


@pytest.mark.parametrize(
    "wrapper", ["`{}`", "``{}``", "\n```\n{}\n```\n", "\n~~~\n{}\n~~~\n"]
)
def test_real_implicit_call_after_example_still_works(wrapper):
    reasoning = "Example " + wrapper.format(CALL) + "\nNow act.\n"
    text = "<think>" + reasoning + CALL
    for size in (1, 2, 7, len(text)):
        assert feed(text, size) == (reasoning, CALL)


def test_unfinished_example_flushes_as_reasoning():
    reasoning = "Example `" + CALL
    for size in (1, 7):
        assert feed("<think>" + reasoning, size) == (reasoning, "")


def test_previous_reasoning_preserves_open_code_span():
    previous = "<think>Discuss `"
    text = CALL + "` literally.</think>Answer."
    for size in (1, 7):
        assert feed(
            text, size, continue_final_message=True, previous_content=previous
        ) == (CALL + "` literally.", "Answer.")


@pytest.mark.parametrize(
    "reasoning",
    [
        r"An escaped \` is prose. ",
        "``A single ` inside code: " + CALL + "``\n",
        "\n````xml\n```\n" + CALL + "\n````\n",
        "\n~~~xml\n~~~ not a closing fence\n" + CALL + "\n~~~\n",
    ],
)
def test_delimiter_rules_preserve_later_real_calls(reasoning):
    for size in (1, 2, 7):
        assert feed(reasoning + CALL, size, force_reasoning=True) == (reasoning, CALL)


@pytest.mark.parametrize("size", [1, 7, 10000])
def test_reasoning_to_tool_pipeline_executes_only_real_call(size):
    from sglang.srt.entrypoints.openai.protocol import Function, Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    tools = [Tool(function=Function(name="ping", parameters={"type": "object"}))]
    reasoner = Qwen3Detector(force_reasoning=True)
    parser = FunctionCallParser(tools, "qwen3_coder")
    reasoning = "Example `" + CALL + "`. Now call it.\n"
    source = reasoning + CALL
    names = []
    emitted_reasoning = ""
    for start in range(0, len(source), size):
        result = reasoner.parse_streaming_increment(source[start : start + size])
        emitted_reasoning += result.reasoning_text
        _, calls = parser.parse_stream_chunk(result.normal_text)
        names.extend(call.name for call in calls if call.name)
    result = reasoner.finish()
    emitted_reasoning += result.reasoning_text
    _, calls = parser.parse_stream_chunk(result.normal_text)
    names.extend(call.name for call in calls if call.name)
    _, calls = parser.parse_stream_end()
    names.extend(call.name for call in calls if call.name)
    assert emitted_reasoning == reasoning
    assert names == ["ping"]


def test_incomplete_quoted_wrapper_does_not_hide_explicit_reasoning_end():
    reasoning = "Example `<tool_call><function=foo>` done"
    normal = "Answer " + CALL
    text = "<think>" + reasoning + "</think>" + normal
    for size in (1, 7, len(text)):
        assert feed(text, size) == (reasoning, normal)
    result = Qwen3Detector().detect_and_parse(text)
    assert (result.reasoning_text, result.normal_text) == (reasoning, normal)


@pytest.mark.parametrize(
    "factory", [_DeepSeekV3Detector, _FlintDetector, _MimoDetector, _PoolsideV1Detector]
)
def test_shared_qwen_envelope_keeps_native_unconditional_tool_interruption(factory):
    reasoning = "Use the native wire format.\n"
    native = "<tool_call>native-payload"
    for size in (1, 7, 1000):
        detector = factory(force_reasoning=True)
        results = [
            detector.parse_streaming_increment((reasoning + native)[i : i + size])
            for i in range(0, len(reasoning + native), size)
        ]
        results.append(detector.finish())
        assert "".join(r.reasoning_text for r in results) == reasoning
        assert "".join(r.normal_text for r in results) == native
    result = factory(force_reasoning=True).detect_and_parse(reasoning + native)
    assert (result.reasoning_text, result.normal_text) == (reasoning, native)


@pytest.mark.parametrize("prefix", ["before", "before\n"])
def test_base_line_eligibility_survives_emitted_prefix_and_split_marker(prefix):
    source = prefix + "<tool_call>native-payload"
    for size in (1, 7, len(source)):
        detector = BaseReasoningFormatDetector(
            "<think>",
            "</think>",
            force_reasoning=True,
            tool_start_token="<tool_call>",
            tool_start_at_line_start=True,
        )
        results = [
            detector.parse_streaming_increment(source[i : i + size])
            for i in range(0, len(source), size)
        ]
        results.append(detector.finish())
        expected = (
            (prefix, "<tool_call>native-payload")
            if prefix.endswith("\n")
            else (source, "")
        )
        assert (
            "".join(r.reasoning_text for r in results),
            "".join(r.normal_text for r in results),
        ) == expected


@pytest.mark.parametrize(
    "normal",
    [
        "Answer with a literal <think> marker.",
        "<tool_call><function=echo><parameter=text>literal <think> marker"
        "</parameter></function></tool_call>",
    ],
)
def test_prefilled_reasoning_preserves_openers_in_later_content_and_payload(normal):
    reasoning = "Done reasoning.\n"
    boundary = "" if normal.startswith("<tool_call>") else "</think>"
    source = reasoning + boundary + normal
    for size in (1, 7, len(source)):
        assert feed(source, size, force_reasoning=True) == (reasoning, normal)
