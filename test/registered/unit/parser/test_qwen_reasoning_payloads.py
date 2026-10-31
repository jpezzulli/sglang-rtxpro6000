"""Accepted Qwen payload classification; upstream interfaces stay intact."""

from sglang.srt.parser.reasoning_parser import Qwen3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestQwenAcceptedPayloads(CustomTestCase):
    def setUp(self):
        self.detector = Qwen3Detector()

    @staticmethod
    def _feed(text, chunk_size, stream_reasoning=True):
        detector = Qwen3Detector(stream_reasoning=stream_reasoning)
        reasoning = normal = ""
        for i in range(0, len(text), chunk_size):
            result = detector.parse_streaming_increment(text[i : i + chunk_size])
            reasoning += result.reasoning_text
            normal += result.normal_text
        result = detector.finish()
        return reasoning + result.reasoning_text, normal + result.normal_text

    def test_detect_and_parse_without_thinking(self):
        """Test parsing without thinking (enable_thinking=False case)."""
        text = "Direct answer without thinking."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.reasoning_text, "")

    def test_streaming_tool_tag_mention_stays_in_reasoning(self):
        text = (
            '<think>The rule says "output the <tool_call> block immediately."'
            "</think>answer"
        )
        expected = (
            'The rule says "output the <tool_call> block immediately."',
            "answer",
        )

        for chunk_size in (1, 2, 7, 11, 1000):
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(self._feed(text, chunk_size), expected)

    def test_buffered_stream_tool_tag_mention_stays_buffered(self):
        detector = Qwen3Detector(stream_reasoning=False)
        first = detector.parse_streaming_increment("<think>quote <tool_call>")
        self.assertEqual((first.reasoning_text, first.normal_text), ("", ""))

        second = detector.parse_streaming_increment(" block</think>answer")
        self.assertEqual(second.reasoning_text, "quote <tool_call> block")
        self.assertEqual(second.normal_text, "answer")

    def test_non_streaming_tool_tag_classification(self):
        quoted = self.detector.detect_and_parse("<think>quote <tool_call> block")
        self.assertEqual(quoted.reasoning_text, "quote <tool_call> block")
        self.assertEqual(quoted.normal_text, "")

        coder = self.detector.detect_and_parse(
            "<think>reason<tool_call><function=get_weather>"
        )
        self.assertEqual(coder.reasoning_text, "reason")
        self.assertEqual(coder.normal_text, "<tool_call><function=get_weather>")

        json_call = self.detector.detect_and_parse(
            '<think>reason<tool_call>{"name":"get_weather","arguments":{}}'
        )
        self.assertEqual(json_call.reasoning_text, "reason")
        self.assertEqual(
            json_call.normal_text,
            '<tool_call>{"name":"get_weather","arguments":{}}',
        )

    def test_streaming_tool_tag_followed_by_prose_markup_stays_in_reasoning(self):
        """A following ``<`` is not enough to prove this is a tool call."""
        cases = (
            ("write <tool_call> <em>literally</em>", "answer"),
            ("emit a closed <tool_call></tool_call> block", "done"),
        )
        for reasoning, answer in cases:
            text = f"<think>{reasoning}</think>{answer}"
            for chunk_size in (1, 5, 13, 1000):
                with self.subTest(reasoning=reasoning, chunk_size=chunk_size):
                    self.assertEqual(self._feed(text, chunk_size), (reasoning, answer))

    def test_malformed_documentary_tool_prefix_stays_in_reasoning(self):
        cases = (
            "Document <tool_call>{not JSON} literally.",
            "Document <tool_call><function= syntax literally.",
            "Document <tool_call><function=> literally.",
            'Document <tool_call>{"name": not JSON} literally.',
            'Document <tool_call>{"name":"get_weather","arguments": not JSON}.',
        )
        for reasoning in cases:
            text = f"<think>{reasoning}</think>answer"
            for chunk_size in (1, 2, 7, 19, 1000):
                with self.subTest(reasoning=reasoning, chunk_size=chunk_size):
                    self.assertEqual(
                        self._feed(text, chunk_size), (reasoning, "answer")
                    )

    def test_streaming_qwen3_coder_tool_still_interrupts_reasoning(self):
        text = (
            "<think>use the tool\n<tool_call>\n"
            "<function=get_weather>\n</function>\n</tool_call>"
        )
        expected = (
            "use the tool\n",
            "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
        )

        for chunk_size in (1, 2, 7, 11, 1000):
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(self._feed(text, chunk_size), expected)

    def test_streaming_json_tool_still_interrupts_reasoning(self):
        text = (
            '<think>use the tool\n<tool_call>\n{"name":"get_weather",'
            '"arguments":{}}\n</tool_call>'
        )
        expected = (
            "use the tool\n",
            '<tool_call>\n{"name":"get_weather","arguments":{}}\n</tool_call>',
        )

        for chunk_size in (1, 5, 13, 1000):
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(self._feed(text, chunk_size), expected)

    def test_json_tool_member_order_and_supported_argument_keys(self):
        payloads = (
            '{"arguments":{"city":"Paris"},"name":"get_weather"}',
            '{"parameters":{"city":"Paris"},"name":"get_weather"}',
        )
        for payload in payloads:
            text = f"<think>use the tool\n<tool_call>{payload}</tool_call>"
            expected = ("use the tool\n", f"<tool_call>{payload}</tool_call>")
            for chunk_size in (1, 5, 17, 1000):
                with self.subTest(payload=payload, chunk_size=chunk_size):
                    self.assertEqual(self._feed(text, chunk_size), expected)

    def test_json_zero_argument_tool_still_interrupts_reasoning(self):
        payload = '{"name":"ping"}'
        text = f"<think>use the tool\n<tool_call>{payload}</tool_call>"
        expected = ("use the tool\n", f"<tool_call>{payload}</tool_call>")

        parsed = self.detector.detect_and_parse(text)
        self.assertEqual((parsed.reasoning_text, parsed.normal_text), expected)
        for stream_reasoning in (True, False):
            for chunk_size in (1, 5, 17, 1000):
                with self.subTest(
                    stream_reasoning=stream_reasoning, chunk_size=chunk_size
                ):
                    self.assertEqual(
                        self._feed(text, chunk_size, stream_reasoning), expected
                    )

    def test_truncated_json_tool_prefix_flushes_as_reasoning(self):
        text = '<think>document <tool_call>{"name":"weather","arguments":{"city":"Pa'
        expected = (text.removeprefix("<think>"), "")

        parsed = self.detector.detect_and_parse(text)
        self.assertEqual((parsed.reasoning_text, parsed.normal_text), expected)
        for stream_reasoning in (True, False):
            for chunk_size in (1, 5, 17, 1000):
                with self.subTest(
                    stream_reasoning=stream_reasoning, chunk_size=chunk_size
                ):
                    self.assertEqual(
                        self._feed(text, chunk_size, stream_reasoning), expected
                    )

    def test_json_tool_literal_end_token_still_interrupts_reasoning(self):
        payload = '{"name":"echo","arguments":{"text":"literal </think> marker"}}'
        text = f"<think>use the tool\n<tool_call>{payload}</tool_call>"
        expected = ("use the tool\n", f"<tool_call>{payload}</tool_call>")

        parsed = self.detector.detect_and_parse(text)
        self.assertEqual((parsed.reasoning_text, parsed.normal_text), expected)
        for stream_reasoning in (True, False):
            for chunk_size in (1, 5, 17, 1000):
                with self.subTest(
                    stream_reasoning=stream_reasoning, chunk_size=chunk_size
                ):
                    self.assertEqual(
                        self._feed(text, chunk_size, stream_reasoning), expected
                    )

    def test_function_tool_literal_end_token_still_interrupts_reasoning(self):
        payload = (
            "<function=echo>\n<parameter=text>literal </think> marker"
            "</parameter>\n</function>"
        )
        text = f"<think>use the tool\n<tool_call>{payload}</tool_call>"
        expected = ("use the tool\n", f"<tool_call>{payload}</tool_call>")

        parsed = self.detector.detect_and_parse(text)
        self.assertEqual((parsed.reasoning_text, parsed.normal_text), expected)
        for stream_reasoning in (True, False):
            for chunk_size in (1, 5, 17, 1000):
                with self.subTest(
                    stream_reasoning=stream_reasoning, chunk_size=chunk_size
                ):
                    self.assertEqual(
                        self._feed(text, chunk_size, stream_reasoning), expected
                    )

    def test_second_json_tool_literal_end_token_still_interrupts_reasoning(self):
        calls = (
            '<tool_call>{"name":"ping"}</tool_call>\n'
            '<tool_call>{"name":"echo","arguments":'
            '{"text":"literal </think> marker"}}</tool_call>'
        )
        text = f"<think>use the tools\n{calls}"
        expected = ("use the tools\n", calls)

        parsed = self.detector.detect_and_parse(text)
        self.assertEqual((parsed.reasoning_text, parsed.normal_text), expected)
        for stream_reasoning in (True, False):
            self.assertEqual(
                self._feed(text, 1000, stream_reasoning),
                expected,
                msg=f"stream_reasoning={stream_reasoning}",
            )

    def test_second_function_tool_literal_end_token_still_interrupts_reasoning(self):
        calls = (
            "<tool_call><function=ping>\n</function>\n</tool_call>\n"
            "<tool_call><function=echo>\n<parameter=text>literal </think> marker"
            "</parameter>\n</function>\n</tool_call>"
        )
        text = f"<think>use the tools\n{calls}"
        expected = ("use the tools\n", calls)

        parsed = self.detector.detect_and_parse(text)
        self.assertEqual((parsed.reasoning_text, parsed.normal_text), expected)
        for stream_reasoning in (True, False):
            self.assertEqual(
                self._feed(text, 1000, stream_reasoning),
                expected,
                msg=f"stream_reasoning={stream_reasoning}",
            )

    def test_completed_json_example_before_real_end_stays_reasoning(self):
        example = '<tool_call>{"name":"ping"}</tool_call>'
        reasoning = f"Example: {example} but do not call it."
        text = f"<think>{reasoning}</think>Answer"
        expected = (reasoning, "Answer")

        parsed = self.detector.detect_and_parse(text)
        self.assertEqual((parsed.reasoning_text, parsed.normal_text), expected)
        for stream_reasoning in (True, False):
            with self.subTest(stream_reasoning=stream_reasoning):
                self.assertEqual(self._feed(text, 1000, stream_reasoning), expected)

    def test_completed_function_example_before_real_end_stays_reasoning(self):
        example = "<tool_call><function=ping>\n</function>\n</tool_call>"
        reasoning = f"Example: {example} but do not call it."
        text = f"<think>{reasoning}</think>Answer"
        expected = (reasoning, "Answer")

        parsed = self.detector.detect_and_parse(text)
        self.assertEqual((parsed.reasoning_text, parsed.normal_text), expected)
        for stream_reasoning in (True, False):
            with self.subTest(stream_reasoning=stream_reasoning):
                self.assertEqual(self._feed(text, 1000, stream_reasoning), expected)

    def test_two_completed_examples_before_real_end_stay_reasoning(self):
        examples = (
            '<tool_call>{"name":"ping"}</tool_call> and '
            "<tool_call><function=echo>\n<parameter=text>literal </think> marker"
            "</parameter>\n</function>\n</tool_call>"
        )
        reasoning = f"Examples: {examples} but do not call them."
        text = f"<think>{reasoning}</think>Answer"
        expected = (reasoning, "Answer")

        parsed = self.detector.detect_and_parse(text)
        self.assertEqual((parsed.reasoning_text, parsed.normal_text), expected)
        for stream_reasoning in (True, False):
            self.assertEqual(
                self._feed(text, 1000, stream_reasoning),
                expected,
                msg=f"stream_reasoning={stream_reasoning}",
            )

    def test_explicit_end_before_json_tool_remains_authoritative(self):
        payload = '{"name":"ping"}'
        text = f"<think>done reasoning</think><tool_call>{payload}</tool_call>"
        expected = ("done reasoning", f"<tool_call>{payload}</tool_call>")

        parsed = self.detector.detect_and_parse(text)
        self.assertEqual((parsed.reasoning_text, parsed.normal_text), expected)
        for stream_reasoning in (True, False):
            for chunk_size in (1, 5, 17, 1000):
                with self.subTest(
                    stream_reasoning=stream_reasoning, chunk_size=chunk_size
                ):
                    self.assertEqual(
                        self._feed(text, chunk_size, stream_reasoning), expected
                    )

    def test_malformed_json_mention_does_not_hide_later_real_call(self):
        quoted = '{"name":"fake","arguments": not JSON}'
        real = '{"arguments":{},"name":"get_weather"}'
        text = (
            f"<think>document <tool_call>{quoted} first\n<tool_call>{real}</tool_call>"
        )
        expected = (
            f"document <tool_call>{quoted} first\n",
            f"<tool_call>{real}</tool_call>",
        )
        for chunk_size in (1, 7, 23, 1000):
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(self._feed(text, chunk_size), expected)

    def test_quoted_tag_before_real_tool_does_not_hide_interrupt(self):
        text = (
            "<think>the <tool_call> block is required\n"
            "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>"
        )
        expected = (
            "the <tool_call> block is required\n",
            "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
        )

        for chunk_size in (1, 7, 23, 1000):
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(self._feed(text, chunk_size), expected)
