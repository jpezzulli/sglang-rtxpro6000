"""Qwen3-Coder tool markup that is an example, not a call.

Literal markup outside a ``<|im_start|>`` wrapper, markup inside a Markdown
fence, and a call the response cut off all stay text: none of them may be
harvested into a call, dropped, or closed with a fabricated delimiter.
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

START = "<tool_call>"
END = "</tool_call>"
OPEN = "<function="
CLOSE = "</function>"
P_OPEN = "<parameter="
P_CLOSE = "</parameter>"


def tool(name="record", value_schema=None):
    return Tool(
        function=Function(
            name=name,
            parameters={
                "type": "object",
                "properties": {"value": value_schema or {"type": "string"}},
            },
        )
    )


def call(name, value):
    return f"{START}{OPEN}{name}>{P_OPEN}value>{value}{P_CLOSE}{CLOSE}{END}"


def streamed(source, tools, width):
    """Reassemble (normal_text, calls) from fixed-width stream chunks."""
    detector = Qwen3CoderDetector()
    normal = ""
    names, arguments = {}, {}
    for offset in range(0, len(source), width):
        result = detector.parse_streaming_increment(
            source[offset : offset + width], tools
        )
        normal += result.normal_text
        for item in result.calls:
            if item.name:
                names[item.tool_index] = item.name
                arguments.setdefault(item.tool_index, "")
            else:
                arguments[item.tool_index] += item.parameters
    normal += detector.finish(tools).normal_text
    return normal, [(names[i], arguments[i]) for i in sorted(names)]


class LiteralMarkupTestCase(CustomTestCase):
    WIDTHS = (None, 1, 3, 7, 64)

    def assert_literal(self, source, tools):
        for width in self.WIDTHS:
            with self.subTest(width=width):
                if width is None:
                    result = Qwen3CoderDetector().detect_and_parse(source, tools)
                    normal, calls = result.normal_text, result.calls
                else:
                    normal, calls = streamed(source, tools, width)
                    calls = [name for name, _ in calls]
                self.assertEqual(calls, [], normal)
                self.assertEqual(normal, source)

    def test_bare_markup_outside_a_wrapper_is_not_a_call(self):
        self.assert_literal(f"Use {OPEN}record>{P_OPEN}value>x{P_CLOSE}{CLOSE}\n", [])

    def test_fenced_example_is_returned_verbatim(self):
        self.assert_literal(
            "Try:\n```xml\n" + call("record", "x") + "\n```\n", [tool()]
        )

    def test_unclosed_fence_holds_apparent_calls_to_the_end(self):
        self.assert_literal("Docs:\n```\n" + call("record", "x") + "\n", [tool()])

    def test_wrapper_body_may_quote_the_delimiters(self):
        """A real wrapper owns its body, so Markdown inside an argument does not
        change what is a call."""
        quoted = "```\nnot a fence\n```"
        source = call("record", quoted)
        for width in (None, 1, 7):
            with self.subTest(width=width):
                if width is None:
                    result = Qwen3CoderDetector().detect_and_parse(source, [tool()])
                    calls = [(c.name, json.loads(c.parameters)) for c in result.calls]
                    normal = result.normal_text
                else:
                    normal, calls = streamed(source, [tool()], width)
                    calls = [(name, json.loads(args)) for name, args in calls]
                self.assertEqual(calls, [("record", {"value": quoted})])
                self.assertEqual(normal, "")

    def test_parameter_wrapper_markers_do_not_consume_following_prose(self):
        for value in (
            "literal " + START + " marker",
            "literal " + START + "example" + END + " marker",
            "literal " + OPEN + "example> marker",
        ):
            source = "BEFORE" + call("record", value) + "TAIL"
            for width in self.WIDTHS:
                with self.subTest(value=value, width=width):
                    if width is None:
                        result = Qwen3CoderDetector().detect_and_parse(source, [tool()])
                        normal = result.normal_text
                        calls = [(c.name, c.parameters) for c in result.calls]
                    else:
                        normal, calls = streamed(source, [tool()], width)
                    self.assertEqual(normal, "BEFORETAIL")
                    self.assertEqual(
                        [(name, json.loads(args)) for name, args in calls],
                        [("record", {"value": value})],
                    )


class IncompleteCallTestCase(CustomTestCase):
    def test_later_stray_parameter_close_does_not_extend_an_incomplete_call(self):
        first = START + OPEN + "record>" + P_OPEN + "value>one" + CLOSE + END
        between = "BETWEEN " + P_CLOSE + " "
        source = first + between + call("record", "two") + "TAIL"
        result = Qwen3CoderDetector().detect_and_parse(source, [tool()])
        self.assertEqual(result.normal_text, between + "TAIL")
        self.assertEqual(
            [(c.name, json.loads(c.parameters)) for c in result.calls],
            [("record", {"value": "one"}), ("record", {"value": "two"})],
        )

    def test_missing_parameter_close_does_not_consume_a_later_call(self):
        first = START + OPEN + "record>" + P_OPEN + "value>one" + CLOSE + END
        source = "BEFORE" + first + "BETWEEN" + call("record", "two") + "TAIL"
        result = Qwen3CoderDetector().detect_and_parse(source, [tool()])
        self.assertEqual(result.normal_text, "BEFOREBETWEENTAIL")
        self.assertEqual(
            [(c.name, json.loads(c.parameters)) for c in result.calls],
            [("record", {"value": "one"}), ("record", {"value": "two"})],
        )

    def test_trailing_wrapper_stays_prose_after_a_complete_call(self):
        done = call("record", "one")
        source = done + "\n" + START + OPEN + "record>" + P_OPEN + "value>two"
        result = Qwen3CoderDetector().detect_and_parse(source, [tool()])
        self.assertEqual(
            [(c.name, json.loads(c.parameters)) for c in result.calls],
            [("record", {"value": "one"})],
        )
        self.assertEqual(result.normal_text, source[len(done) :])

    def test_an_unclosed_call_keeps_its_partial_arguments(self):
        """Neither prose nor a fabricated closing quote: the call is incomplete
        and the response, not the parser, decides what that means."""
        source = START + OPEN + "record>" + P_OPEN + "value>par"
        for width in (1, 5, 64):
            with self.subTest(width=width):
                normal, calls = streamed(source, [tool()], width)
                self.assertEqual(normal, "")
                # The call is announced, and nothing is invented for it: no
                # fabricated closing quote, and no partial argument resurfacing
                # as prose.
                self.assertEqual(calls, [("record", "")])

    def test_empty_wrapper_comes_back_as_text(self):
        for width in (1, 8, 64):
            with self.subTest(width=width):
                normal, calls = streamed(START + END, [tool()], width)
                self.assertEqual(calls, [])
                self.assertEqual(normal, START + END)

    def test_a_declared_string_argument_streams_in_chunks(self):
        """A long string value arrives in many deltas and is emitted as it comes,
        without holding the whole call back for its closing markup."""
        value = "z" * 200
        normal, calls = streamed(call("record", value), [tool()], 7)
        self.assertEqual(normal, "")
        self.assertEqual(calls, [("record", json.dumps({"value": value}))])
        # Streamed in pieces, not held for the whole value.
        detector = Qwen3CoderDetector()
        fragments = [
            len(result.calls)
            for result in (
                detector.parse_streaming_increment(
                    call("record", value)[i : i + 7], [tool()]
                )
                for i in range(0, len(call("record", value)), 7)
            )
        ]
        self.assertGreater(sum(1 for count in fragments if count), 2)


if __name__ == "__main__":
    unittest.main()
