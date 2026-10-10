"""Aggregate tool-call parser counters: one event per logical parse event.

Covers the shared parser used by both supported Penny profiles (the
qwen3_coder tool-call parser), streaming and non-streaming, plus the
metrics-collector export path. Fixtures assemble the detector's own marker
constants instead of hard-coding structural text.
"""

import json
import os
import re
import subprocess
import sys
from collections import Counter
from typing import Dict, List, Optional

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.function_call.utils import (
    TOOL_PARSER_ACCEPTED,
    TOOL_PARSER_ARGUMENT_CONVERSION_FAILED,
    TOOL_PARSER_INCOMPLETE,
    TOOL_PARSER_PARSE_ERROR,
    TOOL_PARSER_UNKNOWN_TOOL,
    get_tool_parser_event_sink,
    set_tool_parser_event_sink,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=45, suite="base-a-test-cpu")

_MARKERS = Qwen3CoderDetector()
TOOL_OPEN = _MARKERS.tool_call_start_token
TOOL_CLOSE = _MARKERS.tool_call_end_token
FUNC_OPEN = _MARKERS.tool_call_prefix
FUNC_CLOSE = _MARKERS.function_end_token
PARAM_OPEN = _MARKERS.parameter_prefix
PARAM_CLOSE = _MARKERS.parameter_end_token

WEATHER = "get_weather"
SEARCH = "search"

TOOLS = [
    Tool(
        function=Function(
            name=WEATHER,
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                    "flags": {"type": "object"},
                },
            },
        )
    ),
    Tool(
        function=Function(
            name=SEARCH,
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
    ),
]

# Shared corpus for the counter checks and for CPU replay measurements, so the
# same work is exercised with instrumentation disabled and enabled.


def call(function_name: str, **params) -> str:
    body = "".join(
        f"{PARAM_OPEN}{name}>{value}{PARAM_CLOSE}" for name, value in params.items()
    )
    return f"{FUNC_OPEN}{function_name}>{body}{FUNC_CLOSE}"


def wrap(*calls: str) -> str:
    return "".join(f"{TOOL_OPEN}\n{item}\n{TOOL_CLOSE}" for item in calls)


CORPUS = (
    wrap(call(WEATHER, location="Paris", days="3")),
    wrap(call(SEARCH, query="rain"), call(WEATHER, location="Oslo", days="2")),
    "Prose with " + wrap(call("not_a_tool", query="x")) + " quoted\n```\nx\n```",
)


@pytest.fixture
def events() -> List[str]:
    recorded: List[str] = []
    set_tool_parser_event_sink(recorded.append)
    yield recorded
    set_tool_parser_event_sink(None)


def counts(recorded: List[str]) -> Dict[str, int]:
    return dict(Counter(recorded))


def parse_stream(text: str, width: int, tools: Optional[List[Tool]] = TOOLS):
    """Feed SSE-sized fragments through the streaming API, then terminate."""
    parser = FunctionCallParser(tools, "qwen3_coder")
    normal_chunks, calls = [], []
    for start in range(0, len(text), width):
        chunk_normal, chunk_calls = parser.parse_stream_chunk(
            text[start : start + width]
        )
        normal_chunks.append(chunk_normal)
        calls.extend(chunk_calls)
    end_normal, end_calls = parser.parse_stream_end()
    normal_chunks.append(end_normal)
    calls.extend(end_calls)
    return "".join(normal_chunks), calls


def parse_text(text: str, tools: Optional[List[Tool]] = TOOLS):
    return FunctionCallParser(tools, "qwen3_coder").parse_non_stream(text)


def test_single_call_streamed_in_fragments_counts_once(events):
    text = wrap(call(WEATHER, location="Paris", days="3"))
    normal, calls = parse_stream(text, width=1)

    assert counts(events) == {TOOL_PARSER_ACCEPTED: 1}
    assert len(calls) > 1  # name plus argument fragments, i.e. SSE chunks
    assert json.loads("".join(c.parameters for c in calls if c.parameters)) == {
        "location": "Paris",
        "days": 3,
    }
    assert normal == ""


def test_multiple_calls_count_once_each_streamed_and_not(events):
    text = wrap(call(SEARCH, query="rain"), call(WEATHER, location="Oslo", days="2"))

    normal_stream, calls_stream = parse_stream(text, width=3)
    assert counts(events) == {TOOL_PARSER_ACCEPTED: 2}
    assert [c.name for c in calls_stream if c.name] == [SEARCH, WEATHER]

    events.clear()
    normal_plain, calls_plain = parse_text(text)
    assert counts(events) == {TOOL_PARSER_ACCEPTED: 2}
    assert [(c.name, json.loads(c.parameters)) for c in calls_plain] == [
        (SEARCH, {"query": "rain"}),
        (WEATHER, {"location": "Oslo", "days": 2}),
    ]
    assert normal_plain == normal_stream


def test_quoted_markup_in_prose_is_not_an_attempt(events):
    fenced = "\n".join(
        [
            "Example:",
            "```text",
            wrap(call(WEATHER, location="Berlin", days="1")),
            "```",
            "That is all.",
        ]
    )
    normal, calls = parse_text(fenced)

    assert counts(events) == {}
    assert calls == []
    assert TOOL_OPEN in normal and FUNC_OPEN in normal

    events.clear()
    normal_stream, calls_stream = parse_stream(fenced, width=2)
    assert counts(events) == {}
    assert calls_stream == []
    assert normal_stream == normal


@pytest.mark.parametrize("prefix", ["", "Some answer. "])
def test_wrapper_without_function_is_prose_not_an_attempt(events, prefix):
    """A wrapper alone (its closing markup present or absent) is never a call."""
    for text in (
        f"{prefix}{TOOL_OPEN}\n{TOOL_CLOSE}",
        f"{prefix}{TOOL_OPEN}\n",
        f"{prefix}{TOOL_OPEN}",
    ):
        events.clear()
        normal, calls = parse_text(text)
        assert counts(events) == {}, text
        assert calls == [], text
        assert normal == text, text

        events.clear()
        normal_stream, calls_stream = parse_stream(text, width=2)
        assert counts(events) == {}, text
        assert calls_stream == [], text
        assert normal_stream == text, text


def test_function_without_its_closing_markup_counts_incomplete_alike(events):
    """The same unterminated call shape counts once in both parsing modes."""
    cases = (
        # Function closed by the wrapper instead of its own markup.
        f"{TOOL_OPEN}{FUNC_OPEN}{WEATHER}>{PARAM_OPEN}location>Rome{PARAM_CLOSE}{TOOL_CLOSE}",
        # Response ended inside the function body.
        f"{TOOL_OPEN}{FUNC_OPEN}{WEATHER}>{PARAM_OPEN}location>Rome",
    )
    for text in cases:
        events.clear()
        _, calls = parse_text(text)
        nonstream = counts(events)
        assert [c.name for c in calls] == [WEATHER], text
        assert nonstream == {
            TOOL_PARSER_ACCEPTED: 1,
            TOOL_PARSER_INCOMPLETE: 1,
        }, text

        events.clear()
        _, calls_stream = parse_stream(text, width=3)
        assert counts(events) == nonstream, text
        assert [c.name for c in calls_stream if c.name] == [WEATHER], text


def test_incomplete_is_counted_only_at_termination(events):
    truncated = (
        f"{TOOL_OPEN}\n{FUNC_OPEN}{WEATHER}>{PARAM_OPEN}location>Paris{PARAM_CLOSE}"
    )

    parser = FunctionCallParser(TOOLS, "qwen3_coder")
    for start in range(0, len(truncated), 4):
        parser.parse_stream_chunk(truncated[start : start + 4])
    # The name may already have been accepted mid-stream; nothing may be
    # classified incomplete until the response actually terminates.
    assert TOOL_PARSER_INCOMPLETE not in counts(events)

    parser.parse_stream_end()
    assert counts(events)[TOOL_PARSER_INCOMPLETE] == 1
    parser.parse_stream_end()
    assert counts(events)[TOOL_PARSER_INCOMPLETE] == 1


def test_complete_call_records_no_incomplete(events):
    text = wrap(call(WEATHER, location="Perth", days="1"))
    parse_stream(text, width=2)
    assert counts(events) == {TOOL_PARSER_ACCEPTED: 1}


def test_truncated_call_after_complete_call_is_still_counted(events):
    complete = wrap(call(SEARCH, query="rain"))
    unfinished = f"{TOOL_OPEN}{FUNC_OPEN}{WEATHER}>{PARAM_OPEN}location>Rome"

    normal, calls = parse_text(complete + unfinished)
    assert normal == unfinished
    assert [c.name for c in calls] == [SEARCH]
    assert counts(events) == {
        TOOL_PARSER_ACCEPTED: 1,
        TOOL_PARSER_INCOMPLETE: 1,
    }

    events.clear()
    parse_stream(complete + unfinished, width=3)
    assert counts(events)[TOOL_PARSER_INCOMPLETE] == 1


@pytest.mark.parametrize("forward_unknown", [False, True])
def test_unknown_trailing_attempt_is_counted(events, forward_unknown):
    text = wrap(call(SEARCH, query="rain")) + f"{TOOL_OPEN}{FUNC_OPEN}unknown>"
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(forward_unknown):
        for parse in (parse_text, lambda value: parse_stream(value, width=3)):
            events.clear()
            parse(text)
            assert counts(events)[TOOL_PARSER_UNKNOWN_TOOL] == 1
            assert counts(events).get(TOOL_PARSER_INCOMPLETE, 0) == int(
                forward_unknown
            )


@pytest.mark.parametrize("next_function", [SEARCH, "unknown"])
def test_later_call_does_not_hide_an_earlier_incomplete_call(events, next_function):
    text = f"{TOOL_OPEN}{FUNC_OPEN}{WEATHER}>{TOOL_CLOSE}" + wrap(
        call(next_function, query="rain")
    )
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        parser = FunctionCallParser(TOOLS, "qwen3_coder")
        for character in text:
            parser.parse_stream_chunk(character)
        assert TOOL_PARSER_INCOMPLETE not in counts(events)
        parser.parse_stream_end()
        assert counts(events)[TOOL_PARSER_INCOMPLETE] == 1
        parser.parse_stream_end()
        assert counts(events)[TOOL_PARSER_INCOMPLETE] == 1


def test_unknown_tool_rejected_and_forwarded(events):
    text = wrap(call("not_a_tool", location="Rome"))

    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        normal, calls = parse_text(text)
        assert counts(events) == {TOOL_PARSER_UNKNOWN_TOOL: 1}
        assert calls == []
        assert FUNC_OPEN in normal

        events.clear()
        normal_stream, calls_stream = parse_stream(text, width=3)
        assert counts(events) == {TOOL_PARSER_UNKNOWN_TOOL: 1}
        assert calls_stream == []
        assert normal_stream == normal

    # Forwarding keeps the call, so the unknown-tool event overlaps accepted.
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        events.clear()
        _, calls = parse_text(text)
        assert counts(events) == {
            TOOL_PARSER_UNKNOWN_TOOL: 1,
            TOOL_PARSER_ACCEPTED: 1,
        }
        assert [c.name for c in calls] == ["not_a_tool"]


def test_conversion_fallback_counts_once_per_parameter(events):
    text = wrap(call(WEATHER, location="Lima", days="three", flags="not-a-json-object"))

    _, calls = parse_text(text)
    assert counts(events) == {
        TOOL_PARSER_ACCEPTED: 1,
        TOOL_PARSER_ARGUMENT_CONVERSION_FAILED: 2,
    }
    assert json.loads(calls[0].parameters) == {
        "location": "Lima",
        "days": "three",
        "flags": "not-a-json-object",
    }

    events.clear()
    _, calls_stream = parse_stream(text, width=5)
    assert counts(events) == {
        TOOL_PARSER_ACCEPTED: 1,
        TOOL_PARSER_ARGUMENT_CONVERSION_FAILED: 2,
    }
    assert json.loads("".join(c.parameters for c in calls_stream if c.parameters)) == {
        "location": "Lima",
        "days": "three",
        "flags": "not-a-json-object",
    }


def test_parse_error_recorded_once_and_output_unchanged(events, monkeypatch):
    text = wrap(call(WEATHER, location="Cairo", days="4"))

    def boom(self, func_name, tools):
        raise RuntimeError("boom")

    monkeypatch.setattr(Qwen3CoderDetector, "_iter_unquoted_tool_spans", boom)
    normal, calls = parse_text(text)
    assert counts(events) == {TOOL_PARSER_PARSE_ERROR: 1}
    assert calls == []
    assert normal == text


def test_streaming_parse_error_recorded_once_and_still_raises(events):
    """A set literal in a declared object parameter breaks JSON serialization.

    The streaming path must keep raising as before, but count the event.
    """
    text = wrap(call(WEATHER, location="Cairo", flags="{1}"))

    parser = FunctionCallParser(TOOLS, "qwen3_coder")
    with pytest.raises(TypeError):
        for start in range(0, len(text), 2):
            parser.parse_stream_chunk(text[start : start + 2])

    assert counts(events)[TOOL_PARSER_PARSE_ERROR] == 1

    events.clear()
    normal, calls = parse_text(text)
    assert counts(events) == {
        TOOL_PARSER_ARGUMENT_CONVERSION_FAILED: 1,
        TOOL_PARSER_PARSE_ERROR: 1,
    }
    assert calls == []
    assert normal == text


def test_sink_unset_is_silent():
    assert get_tool_parser_event_sink() is None
    text = wrap(call(WEATHER, location="Perth", days="bad"))
    normal, calls = parse_text(text)
    assert [c.name for c in calls] == [WEATHER]
    assert json.loads(calls[0].parameters)["days"] == "bad"
    assert normal == ""


_EXPORT_SCRIPT = """
import json
import sys
from prometheus_client import REGISTRY, generate_latest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.utils import get_tool_parser_event_sink
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

publish(ServerArgs(model_path="dummy"), role="tokenizer")
labels = {"model_name": "qwen-test", "engine_type": "text"}
collector = TokenizerMetricsCollector(labels=labels, server_args=None)

sink = get_tool_parser_event_sink()
assert sink.__self__ is collector, sink
assert sink.__func__ is TokenizerMetricsCollector.record_tool_parser_event, sink

tools = [
    Tool(
        function=Function(
            name=name,
            parameters={"type": "object", "properties": properties},
        )
    )
    for name, properties in json.loads(sys.argv[2]).items()
]
text = sys.argv[1]

# Drive real parses: one non-streaming, one streamed in fragments, plus an
# unterminated streamed call and a call whose own closing markup is missing.
FunctionCallParser(tools, "qwen3_coder").parse_non_stream(text)
parser = FunctionCallParser(tools, "qwen3_coder")
for start in range(0, len(text), 2):
    parser.parse_stream_chunk(text[start : start + 2])
parser.parse_stream_end()

sys.stdout.write(generate_latest(REGISTRY).decode())
"""


def _repo_python_path() -> str:
    """Source root of this checkout (the directory holding the sglang package)."""
    parts = os.path.abspath(__file__).split(os.sep)
    for depth in range(len(parts) - 1, 0, -1):
        candidate = os.sep.join(parts[:depth] + ["python"])
        if os.path.isfile(os.sep.join([candidate, "sglang", "__init__.py"])):
            return candidate
    raise AssertionError("could not locate the sglang source root")


def test_counter_exported_by_real_prometheus():
    """Exposition check in a subprocess, so the default registry stays clean."""
    pytest.importorskip("prometheus_client")
    text = wrap(
        call(WEATHER, location="Paris", days="3"),
        f"{FUNC_OPEN}{WEATHER}>{PARAM_OPEN}location>Rome{PARAM_CLOSE}",
    )
    schemas = {WEATHER: TOOLS[0].function.parameters["properties"]}
    proc = subprocess.run(
        [sys.executable, "-c", _EXPORT_SCRIPT, text, json.dumps(schemas)],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, [_repo_python_path(), os.environ.get("PYTHONPATH")])
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    exposition = proc.stdout

    # A tuple help string (regression) makes generate_latest fail outright.
    assert re.search(
        r"^# HELP sglang:tool_parser_events_total \S", exposition, re.MULTILINE
    ), exposition
    assert (
        "# TYPE sglang:tool_parser_events_total counter" in exposition
    ), exposition
    for outcome, expected in (
        (TOOL_PARSER_ACCEPTED, 4),  # two calls, parsed once each way
        (TOOL_PARSER_INCOMPLETE, 2),  # the unterminated call, both modes
    ):
        match = re.search(
            r'^sglang:tool_parser_events_total\{[^}]*outcome="'
            + re.escape(outcome)
            + r'"[^}]*\} (\S+)$',
            exposition,
            re.MULTILINE,
        )
        assert match is not None, (outcome, exposition)
        assert float(match.group(1)) == float(expected), (outcome, exposition)
    assert "outcome=\"parse_error\"" not in exposition, exposition
    assert "outcome=\"unknown_tool\"" not in exposition, exposition
