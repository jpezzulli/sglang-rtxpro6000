"""Aggregate observations for the qwen3_coder parser used by Penny profiles.

Covers streaming/non-streaming logical-event accounting, sink lifecycle,
observer failure isolation and real Prometheus export. Existing shared-base
hooks remain available to other parsers, but this suite does not claim complete
accounting for every parser (including qwen25). Counters use fixed outcomes and
existing collector labels, not caller data or tool-execution success rates.
"""

import json
import os
import re
import subprocess
import sys
import types
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
    TOOL_PARSER_EVENT_OUTCOMES,
    TOOL_PARSER_INCOMPLETE,
    TOOL_PARSER_PARSE_ERROR,
    TOOL_PARSER_UNKNOWN_TOOL,
    get_tool_parser_event_sink,
    record_tool_parser_event,
    set_tool_parser_event_sink,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

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

LABELS = {"model_name": "qwen-test", "engine_type": "text"}


def call(function_name: str, **params) -> str:
    body = "".join(
        f"{PARAM_OPEN}{name}>{value}{PARAM_CLOSE}" for name, value in params.items()
    )
    return f"{FUNC_OPEN}{function_name}>{body}{FUNC_CLOSE}"


def wrap(*calls: str) -> str:
    return "".join(f"{TOOL_OPEN}\n{item}\n{TOOL_CLOSE}" for item in calls)


# Shared corpus for the counter checks and for output-equivalence replay, so
# the same work is exercised with the sink unset and installed.
CORPUS = (
    wrap(call(WEATHER, location="Paris", days="3")),
    wrap(call(SEARCH, query="rain"), call(WEATHER, location="Oslo", days="2")),
    "Prose with " + wrap(call("not_a_tool", query="x")) + " quoted\n```\nx\n```",
)


@pytest.fixture(autouse=True)
def _sink_restored():
    yield
    set_tool_parser_event_sink(None)


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


def corpus_results():
    return [
        parse_text(CORPUS[0]),
        parse_text(CORPUS[1]),
        parse_text(CORPUS[2]),
        parse_stream(CORPUS[0], width=1),
        parse_stream(CORPUS[1], width=3),
        parse_stream(CORPUS[2], width=2),
        parse_stream(CORPUS[1] + CORPUS[2], width=7),
    ]


# ---------------------------------------------------------------------------
# Sink lifecycle
# ---------------------------------------------------------------------------


def test_sink_defaults_to_unset_and_round_trips():
    assert get_tool_parser_event_sink() is None

    def observe(outcome: str) -> None:
        observed.append(outcome)

    observed: List[str] = []
    set_tool_parser_event_sink(observe)
    assert get_tool_parser_event_sink() is observe
    set_tool_parser_event_sink(None)
    assert get_tool_parser_event_sink() is None


def test_recorder_is_a_no_op_while_the_sink_is_unset():
    for outcome in TOOL_PARSER_EVENT_OUTCOMES:
        record_tool_parser_event(outcome)


@pytest.mark.parametrize("outcome", TOOL_PARSER_EVENT_OUTCOMES)
def test_recorder_delivers_each_fixed_outcome_unchanged(events, outcome):
    record_tool_parser_event(outcome)
    assert events == [outcome]
    assert outcome in TOOL_PARSER_EVENT_OUTCOMES


def test_outcome_vocabulary_is_five_bounded_fixed_strings():
    assert TOOL_PARSER_EVENT_OUTCOMES == (
        "accepted",
        "incomplete",
        "unknown_tool",
        "argument_conversion_failed",
        "parse_error",
    )
    assert len(set(TOOL_PARSER_EVENT_OUTCOMES)) == 5
    assert all(isinstance(o, str) and len(o) <= 64 for o in TOOL_PARSER_EVENT_OUTCOMES)


def test_events_carry_only_the_fixed_vocabulary(events):
    for text in CORPUS:
        parse_text(text)
        parse_stream(text, width=2)
    assert set(events) <= set(TOOL_PARSER_EVENT_OUTCOMES), events
    assert all(len(event) <= 64 for event in events), events


def test_recorder_swallows_a_failing_sink():
    calls: List[str] = []

    def exploding_sink(outcome: str) -> None:
        calls.append(outcome)
        raise RuntimeError("observer is down")

    set_tool_parser_event_sink(exploding_sink)
    for outcome in (
        TOOL_PARSER_ACCEPTED,
        TOOL_PARSER_INCOMPLETE,
        TOOL_PARSER_UNKNOWN_TOOL,
        TOOL_PARSER_ARGUMENT_CONVERSION_FAILED,
        TOOL_PARSER_PARSE_ERROR,
    ):
        record_tool_parser_event(outcome)
    assert calls == list(TOOL_PARSER_EVENT_OUTCOMES)


# ---------------------------------------------------------------------------
# Parser output must not depend on observation
# ---------------------------------------------------------------------------


def test_enabled_and_disabled_sinks_parse_identically():
    with_sink = corpus_results()
    set_tool_parser_event_sink(lambda outcome: None)
    without_sink = corpus_results()
    set_tool_parser_event_sink(None)
    assert with_sink == without_sink


def test_failing_sink_does_not_change_parser_output_or_raise():
    def exploding_sink(outcome: str) -> None:
        raise RuntimeError("observer is down")

    set_tool_parser_event_sink(exploding_sink)
    assert corpus_results() == corpus_results()


def test_sink_unset_is_silent():
    assert get_tool_parser_event_sink() is None
    text = wrap(call(WEATHER, location="Perth", days="bad"))
    normal, calls = parse_text(text)
    assert [c.name for c in calls] == [WEATHER]
    assert json.loads(calls[0].parameters)["days"] == "bad"
    assert normal == ""


# ---------------------------------------------------------------------------
# Collector registration and export
# ---------------------------------------------------------------------------


class _RecordingBoundMetric:
    def __init__(self, metric: "_RecordingMetric", labels: Dict[str, str]) -> None:
        self.metric = metric
        self.labels = labels

    def inc(self, value: float = 1) -> None:
        self.metric.increments.append((self.labels, value))

    def observe(self, value: float) -> None:
        self.metric.observations.append((self.labels, value))

    def set(self, value: float) -> None:
        self.metric.sets.append((self.labels, value))


class _RecordingMetric:
    """prometheus_client-compatible metric that keeps the labels it was given."""

    def __init__(
        self, *args, name=None, documentation="", labelnames=(), **kwargs
    ) -> None:
        self.name = name if name is not None else args[0]
        self.documentation = documentation
        self.labelnames = tuple(labelnames)
        self.increments: List = []
        self.observations: List = []
        self.sets: List = []

    def labels(self, *values, **labels) -> _RecordingBoundMetric:
        if values:
            labels = dict(zip(self.labelnames, values, strict=True))
        return _RecordingBoundMetric(self, labels)

    def inc(self, value: float = 1) -> None:
        self.increments.append(({}, value))

    def observe(self, value: float) -> None:
        self.observations.append(({}, value))

    def set(self, value: float) -> None:
        self.sets.append(({}, value))


@pytest.fixture
def collector():
    """A real collector built against recording metrics and a published config."""
    from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    prom = types.SimpleNamespace(
        Counter=_RecordingMetric,
        Gauge=_RecordingMetric,
        Histogram=_RecordingMetric,
        Summary=_RecordingMetric,
    )
    original = sys.modules.get("prometheus_client")
    sys.modules["prometheus_client"] = prom
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    try:
        yield TokenizerMetricsCollector(labels=dict(LABELS), server_args=None)
    finally:
        if original is None:
            del sys.modules["prometheus_client"]
        else:
            sys.modules["prometheus_client"] = original
        reset_context()


def test_counter_registered_with_bounded_labels_and_installs_the_sink(collector):
    counter = collector.tool_parser_events_total
    assert counter.name == "sglang:tool_parser_events_total"
    # A non-string help text makes generate_latest fail at scrape time.
    assert isinstance(counter.documentation, str)
    assert counter.documentation
    assert counter.labelnames == (*LABELS.keys(), "outcome")

    sink = get_tool_parser_event_sink()
    assert sink is not None and sink.__self__ is collector
    assert sink.__func__ is type(collector).record_tool_parser_event


def test_collector_export_carries_only_outcome_and_its_own_labels(collector):
    counter = collector.tool_parser_events_total
    for outcome in TOOL_PARSER_EVENT_OUTCOMES:
        record_tool_parser_event(outcome)
        record_tool_parser_event(outcome)

    assert counter.increments == [
        ({**LABELS, "outcome": outcome}, 1)
        for outcome in TOOL_PARSER_EVENT_OUTCOMES
        for _ in range(2)
    ]
    assert all(set(labels) == {*LABELS, "outcome"} for labels, _ in counter.increments)


def test_collector_sink_failure_does_not_reach_the_parser(collector):
    record_tool_parser_event(TOOL_PARSER_ACCEPTED)
    collector.tool_parser_events_total = types.SimpleNamespace(
        labels=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("scrape broke"))
    )
    for outcome in TOOL_PARSER_EVENT_OUTCOMES:
        record_tool_parser_event(outcome)
    assert get_tool_parser_event_sink() is not None


_EXPORT_SCRIPT = """
import sys
from prometheus_client import REGISTRY, generate_latest

from sglang.srt.function_call.utils import (
    TOOL_PARSER_EVENT_OUTCOMES,
    get_tool_parser_event_sink,
    record_tool_parser_event,
)
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

publish(ServerArgs(model_path="dummy"), role="tokenizer")
labels = {"model_name": "qwen-test", "engine_type": "text"}
collector = TokenizerMetricsCollector(labels=labels, server_args=None)

sink = get_tool_parser_event_sink()
assert sink.__self__ is collector, sink
assert sink.__func__ is TokenizerMetricsCollector.record_tool_parser_event, sink

for outcome in TOOL_PARSER_EVENT_OUTCOMES:
    record_tool_parser_event(outcome)

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
    proc = subprocess.run(
        [sys.executable, "-c", _EXPORT_SCRIPT],
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
    assert "# TYPE sglang:tool_parser_events_total counter" in exposition, exposition
    from prometheus_client.parser import text_string_to_metric_families

    samples = [
        sample
        for family in text_string_to_metric_families(exposition)
        for sample in family.samples
        if sample.name == "sglang:tool_parser_events_total"
    ]
    assert len(samples) == len(TOOL_PARSER_EVENT_OUTCOMES)
    by_outcome = {sample.labels["outcome"]: sample for sample in samples}
    for outcome in TOOL_PARSER_EVENT_OUTCOMES:
        match = re.search(
            r'^sglang:tool_parser_events_total\{[^}]*outcome="'
            + re.escape(outcome)
            + r'"[^}]*\} (\S+)$',
            exposition,
            re.MULTILINE,
        )
        assert match is not None, (outcome, exposition)
        assert float(match.group(1)) == 1.0, (outcome, exposition)
        assert by_outcome[outcome].labels == {**LABELS, "outcome": outcome}
    for leak in ("get_weather", "search", "Paris", "model_prompt"):
        assert leak not in exposition, exposition


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
            # Current Qwen parsing emits unknown names regardless of this flag;
            # observe its unfinished attempt without restoring the older filter.
            assert counts(events).get(TOOL_PARSER_INCOMPLETE, 0) == 1


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


def test_unknown_tool_accounting_preserves_existing_passthrough(events):
    text = wrap(call("not_a_tool", location="Rome"))
    # The pinned parser accepts this name with either flag. Instrumentation
    # must observe that behavior, not restore the donor's older rejection path.
    for forward_unknown in (False, True):
        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(forward_unknown):
            for parse in (parse_text, lambda value: parse_stream(value, width=3)):
                events.clear()
                normal, calls = parse(text)
                assert counts(events) == {
                    TOOL_PARSER_UNKNOWN_TOOL: 1,
                    TOOL_PARSER_ACCEPTED: 1,
                }
                assert normal == ""
                assert [c.name for c in calls if c.name] == ["not_a_tool"]


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
