"""Formulas for the host-cache wall and resume report. No GPU and no server."""

import importlib.util
import io
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

_SCRIPT = Path(__file__).resolve().parent / "hicache_mamba_host_report.py"
_SPEC = importlib.util.spec_from_file_location("hicache_mamba_host_report", _SCRIPT)
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)


def _line(
    when,
    input_len,
    cached,
    output,
    queue,
    prefill,
    post,
    forward,
    rid="a" * 32,
    entry=None,
):
    stamp = "" if entry is None else f", entry_time={entry}"
    return (
        f"[{when}] ReqTimeStats(rid={rid}, "
        f"input_len={input_len}, cached_input_len={cached}, output_len={output}, "
        f"attempts=0, type=unified): queue_duration={queue}ms, "
        f"initial_prefill_elapsed={prefill}ms, post_prefill_elapsed={post}ms, "
        f"forward_duration={forward}ms{stamp}"
    )


def _match(when, rid, full_kv, device, host_hit):
    return (
        f"[{when}] mamba match rid={rid} session=abc prefix= input=100 "
        f"full_kv={full_kv} device={device} host_hit={host_hit} mamba_host=0 "
        f"branching=None node=1 mamba_used=1 mamba_total=272"
    )


class ReportTests(unittest.TestCase):
    def test_empty_log_is_zero(self):
        summary = report.summarize(report.parse_requests(["not a request"]))
        self.assertEqual(summary["n"], 0)
        self.assertEqual(summary["out_tok_s"], 0)
        self.assertEqual(summary["trunc_pct"], 0.0)

    def test_one_request_counts_its_own_forward(self):
        # 100 tokens over a 10s forward. The clock is that forward, so the
        # wall rate is 10 and the GPU is busy for the whole span.
        lines = [_line("2026-09-29 13:00:10", 100, 0, 100, 0, 0, 10000, 10000)]
        summary = report.summarize(report.parse_requests(lines))
        self.assertEqual(summary["span_s"], 10.0)
        self.assertEqual(summary["out_tok_s"], 10)
        self.assertEqual(summary["mean_req_tok_s"], 10)
        self.assertEqual(summary["busy_pct"], 100.0)

    def test_wall_speed_counts_idle_time_between_finishes(self):
        # Each request decodes for 1s. The span runs from the first forward
        # start to the second finish, 11s, and 9s of that is idle.
        lines = [
            _line("2026-09-29 13:00:00", 100, 0, 100, 0, 0, 1000, 1000),
            _line("2026-09-29 13:00:10", 100, 0, 100, 0, 0, 1000, 1000, rid="b" * 32),
        ]
        summary = report.summarize(report.parse_requests(lines))
        self.assertEqual(summary["span_s"], 11.0)
        self.assertEqual(summary["out_tok_s"], 18)
        self.assertEqual(summary["mean_req_tok_s"], 100)
        self.assertEqual(summary["decode_tok_s"], 100)
        self.assertEqual(summary["busy_pct"], 18.2)
        self.assertEqual(summary["peak_inflight"], 1)

    def test_overlapping_decodes_share_one_clock(self):
        # Finishes 5s apart. Each forward is 10s, so the span is 15s and the
        # two forwards overlap. Wall speed counts that shared clock once.
        # Mean request speed stays at 10, because each request still took 10s.
        lines = [
            _line("2026-09-29 13:00:10", 100, 0, 100, 0, 0, 10000, 10000),
            _line("2026-09-29 13:00:15", 100, 0, 100, 0, 0, 10000, 10000, rid="b" * 32),
        ]
        summary = report.summarize(report.parse_requests(lines))
        self.assertEqual(summary["span_s"], 15.0)
        self.assertEqual(summary["out_tok_s"], 13)
        self.assertEqual(summary["mean_req_tok_s"], 10)
        self.assertEqual(summary["decode_tok_s"], 13)
        self.assertEqual(summary["peak_inflight"], 2)
        self.assertEqual(summary["busy_pct"], 100.0)

    def test_prefill_rate_uses_the_union_of_prefill_intervals(self):
        # Uncached 200 + 500 over 1.0s + 0.5s of prefill, with no overlap.
        lines = [
            _line("2026-09-29 12:00:10", 200, 0, 1, 0, 1000, 10, 1000),
            _line("2026-09-29 12:00:20", 500, 0, 1, 0, 500, 10, 500, rid="b" * 32),
        ]
        summary = report.summarize(report.parse_requests(lines))
        self.assertEqual(summary["prefill_tok_s"], 467)
        self.assertEqual(summary["uncached_tokens"], 700)
        self.assertEqual(summary["med_pre_s"], 0.75)
        self.assertEqual(summary["p90_pre_s"], 0.95)

    def test_first_mamba_match_keeps_kv_beside_resume(self):
        deep = "b" * 32
        collapsed = "c" * 32
        short = "d" * 32
        lines = [
            _match("2026-09-29 12:00:01", deep, 80000, 0, 78000),
            # A later line for the same request must not replace the first match.
            _match("2026-09-29 12:00:20", deep, 80000, 0, 4096),
            _line("2026-09-29 12:00:10", 82000, 78000, 10, 0, 100, 100, 200, rid=deep),
            _match("2026-09-29 12:05:00", collapsed, 60000, 0, 4096),
            _line(
                "2026-09-29 12:05:10",
                61000,
                4096,
                5,
                0,
                100,
                100,
                200,
                rid=collapsed,
            ),
            # Under 8192 tokens, a short resume is a cold prompt, not a collapse.
            _match("2026-09-29 12:10:00", short, 4000, 4000, 0),
            _line("2026-09-29 12:10:10", 4000, 4000, 1, 0, 10, 10, 20, rid=short),
            "not a request",
        ]
        summary = report.summarize(report.parse_requests(lines))
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["n_match"], 3)
        # 78000 + 4096 + 4000 cached, over 82000 + 61000 + 4000 prompt tokens.
        self.assertEqual(summary["hit"], 58.6)
        # 78000 + 4096 + 4000 resume, over 80000 + 60000 + 4000 KV.
        self.assertEqual(summary["mamba_of_kv"], 59.8)
        # Only the 60k match resumed under half of its KV. The 4k match is too short.
        self.assertEqual(summary["trunc_pct"], 50.0)

    def test_entry_time_sets_the_finish_when_the_bracket_is_coarse(self):
        lines = [
            _line(
                "2026-09-29 13:00:00",
                10,
                0,
                50,
                0,
                0,
                0,
                0,
                entry=1_700_000_000,
            ),
            _line(
                "2026-09-29 13:00:00",
                10,
                0,
                50,
                0,
                0,
                0,
                0,
                rid="b" * 32,
                entry=1_700_000_010,
            ),
        ]
        summary = report.summarize(report.parse_requests(lines))
        self.assertEqual(summary["span_s"], 10.0)
        self.assertEqual(summary["out_tok_s"], 10)

    def test_main_reads_a_log_file_and_stdin(self):
        text = _line("2026-09-29 13:00:00", 100, 80, 10, 0, 100, 100, 200) + "\n"
        with NamedTemporaryFile("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                code = report.main([handle.name])
        self.assertEqual(code, 0)
        self.assertIn("n: 1\n", stdout.getvalue())
        self.assertIn("hit: 80.0\n", stdout.getvalue())

        stdout = io.StringIO()
        with patch("sys.stdin", io.StringIO(text)), patch("sys.stdout", stdout):
            code = report.main([])
        self.assertEqual(code, 0)
        self.assertIn("output_tokens: 10\n", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
