#!/usr/bin/env python3
"""Summarize SGLang ReqTimeStats and decode telemetry from a server log."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


REQ_RE = re.compile(
    r"^(?:\[)?(?P<timestamp>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]? "
    r"ReqTimeStats\(rid=(?P<request_id>[0-9a-f]+), "
    r"input_len=(?P<input_len>\d+), cached_input_len=(?P<cached_input_len>\d+), "
    r"output_len=(?P<output_len>\d+), attempts=(?P<attempts>\d+), "
    r"type=(?P<request_type>[^)]+)\): queue_duration=(?P<queue_ms>[0-9.]+)ms, "
    r"initial_prefill_elapsed=(?P<prefill_ms>[0-9.]+)ms, "
    r"post_prefill_elapsed=(?P<post_prefill_ms>[0-9.]+)ms, "
    r"forward_duration=(?P<forward_ms>[0-9.]+)ms, entry_time=(?P<entry_time>[0-9.]+)"
)

DECODE_RE = re.compile(
    r"Decode batch, #running-req: (?P<running_requests>\d+), "
    r"#full token: (?P<full_tokens>\d+).*?mamba num: (?P<mamba_states>\d+).*?"
    r"accept len: (?P<accept_length>[0-9.]+), accept rate: (?P<accept_rate>[0-9.]+), "
    r"cuda graph: (?P<cuda_graph>\w+), gen throughput \(token/s\): (?P<throughput>[0-9.]+)"
)

CONTEXT_BANDS = (
    (0, 2_000, "0-2K"),
    (2_000, 64_000, "2-64K"),
    (64_000, 100_000, "64-100K"),
    (100_000, 150_000, "100-150K"),
    (150_000, 262_144, "150-262K"),
    (262_144, 300_000, "262-300K"),
    (300_000, 325_000, "300-325K"),
    (325_000, 340_000, "325-340K"),
    (340_000, 360_000, "340-360K"),
    (360_000, 524_289, "360-524K"),
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty sequence")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_requests(rows: list[dict]) -> dict:
    if not rows:
        return {"request_count": 0}
    effective = [row["effective_decode_tps"] for row in rows]
    output_tokens = sum(row["output_len"] for row in rows)
    post_seconds = sum(row["post_prefill_ms"] for row in rows) / 1000
    return {
        "request_count": len(rows),
        "output_tokens": output_tokens,
        "effective_decode_tps": {
            "minimum": min(effective),
            "p25": percentile(effective, 0.25),
            "median": statistics.median(effective),
            "mean": statistics.mean(effective),
            "p75": percentile(effective, 0.75),
            "maximum": max(effective),
            "token_weighted_over_summed_post_prefill_time": output_tokens
            / post_seconds,
        },
    }


def summarize_telemetry(rows: list[dict]) -> dict:
    if not rows:
        return {"sample_count": 0}
    throughput = [row["throughput"] for row in rows]
    return {
        "sample_count": len(rows),
        "throughput_tps": {
            "minimum": min(throughput),
            "p25": percentile(throughput, 0.25),
            "median": statistics.median(throughput),
            "mean": statistics.mean(throughput),
            "p75": percentile(throughput, 0.75),
            "maximum": max(throughput),
        },
        "mean_accept_length": statistics.mean(
            row["accept_length"] for row in rows
        ),
        "mean_accept_rate": statistics.mean(row["accept_rate"] for row in rows),
    }


def parse_log(path: Path) -> tuple[list[dict], list[dict]]:
    requests: list[dict] = []
    telemetry: list[dict] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if match := REQ_RE.search(line):
            row = match.groupdict()
            for key in ("input_len", "cached_input_len", "output_len", "attempts"):
                row[key] = int(row[key])
            for key in (
                "queue_ms",
                "prefill_ms",
                "post_prefill_ms",
                "forward_ms",
                "entry_time",
            ):
                row[key] = float(row[key])
            row["line_number"] = line_number
            row["uncached_input_len"] = row["input_len"] - row["cached_input_len"]
            row["effective_decode_tps"] = row["output_len"] / (
                row["post_prefill_ms"] / 1000
            )
            row["end_time"] = row["entry_time"] + (
                row["queue_ms"] + row["forward_ms"]
            ) / 1000
            requests.append(row)
        if match := DECODE_RE.search(line):
            row = match.groupdict()
            for key in ("running_requests", "full_tokens", "mamba_states"):
                row[key] = int(row[key])
            for key in ("accept_length", "accept_rate", "throughput"):
                row[key] = float(row[key])
            row["cuda_graph"] = row["cuda_graph"].lower() == "true"
            row["line_number"] = line_number
            telemetry.append(row)

    for row in requests:
        row["overlapping_completed_request_intervals"] = sum(
            1
            for other in requests
            if other is not row
            and max(row["entry_time"], other["entry_time"])
            < min(row["end_time"], other["end_time"])
        )
    return requests, telemetry


def build_summary(requests: list[dict], telemetry: list[dict]) -> dict:
    context_requests = {}
    context_nonoverlap = {}
    telemetry_single = {}
    for lower, upper, label in CONTEXT_BANDS:
        selected = [row for row in requests if lower <= row["input_len"] < upper]
        context_requests[label] = summarize_requests(selected)
        context_nonoverlap[label] = summarize_requests(
            [
                row
                for row in selected
                if row["overlapping_completed_request_intervals"] == 0
            ]
        )
        telemetry_single[label] = summarize_telemetry(
            [
                row
                for row in telemetry
                if row["running_requests"] == 1
                and lower <= row["full_tokens"] < upper
            ]
        )

    return {
        "measurement_definition": (
            "completed-request effective decode = output_len / "
            "(post_prefill_elapsed_ms / 1000)"
        ),
        "completed_requests": summarize_requests(requests),
        "completed_requests_without_detected_interval_overlap": summarize_requests(
            [
                row
                for row in requests
                if row["overlapping_completed_request_intervals"] == 0
            ]
        ),
        "completed_requests_with_detected_interval_overlap": summarize_requests(
            [
                row
                for row in requests
                if row["overlapping_completed_request_intervals"] > 0
            ]
        ),
        "completed_by_context": context_requests,
        "nonoverlapping_completed_by_context": context_nonoverlap,
        "single_running_request_telemetry_by_context": telemetry_single,
        "telemetry_by_running_request_count": {
            str(count): summarize_telemetry(
                [row for row in telemetry if row["running_requests"] == count]
            )
            for count in sorted({row["running_requests"] for row in telemetry})
        },
        "strongest_instantaneous_windows": sorted(
            telemetry, key=lambda row: row["throughput"], reverse=True
        )[:10],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "timestamp",
        "request_id",
        "input_len",
        "cached_input_len",
        "uncached_input_len",
        "output_len",
        "queue_ms",
        "prefill_ms",
        "post_prefill_ms",
        "forward_ms",
        "effective_decode_tps",
        "overlapping_completed_request_intervals",
        "line_number",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    requests, telemetry = parse_log(args.log)
    summary = build_summary(requests, telemetry)
    if args.csv_out:
        write_csv(args.csv_out, requests)
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not args.csv_out and not args.json_out:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
