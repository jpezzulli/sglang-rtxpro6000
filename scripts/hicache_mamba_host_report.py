#!/usr/bin/env python3
"""Wall speed and Mamba resume quality from an SGLang server log.

Run it on a log from before the host-cache change and again on a log from
after, using a window where the GPU is busy and about four requests are in
flight. A short run on an empty cache will not show the collapse this is
meant to catch. That collapse shows up after the host pool is full and a
new session starts.

The log timestamp on a ReqTimeStats line is completion, to the second.
``entry_time`` plus queue plus forward is the precise finish when it is
present. The span runs from the first forward start to the last finish.
Idle time inside that span counts. A single request's rate is its output
divided by its own forward, which includes prefill.

``cached_input_len`` is the Mamba-truncated prefix, so ``hit`` is the resume
the request could use. The first ``mamba match`` line for a request keeps
the KV length (``full_kv``) beside the Mamba resume (``device`` + ``host_hit``).
Later lines for the same rid are ignored.

Fields:

- out_tok_s: output tokens / wall span. The span starts at the first forward
  and ends at the last finish. This is the wall output speed.
- mean_req_tok_s: mean of each request's output / its own forward. Prefill
  counts. Queue and time between requests do not. This falls as more
  requests run together, because each decode gets slower.
- busy_pct: share of that span with at least one forward running.
- mean_inflight, peak_inflight: concurrency over that span.
- hit: cached_input_len / input_len, summed across requests.
- med_pre_s, p90_pre_s: initial_prefill_elapsed, in seconds.
- mamba_of_kv: (device + host_hit) / full_kv, summed across matched requests.
- trunc_pct: among matches with full_kv of at least 8192, the share whose
  resume was under half of that KV length.
- prefill_tok_s: uncached tokens / union of prefill intervals.
- decode_tok_s: output tokens / union of decode intervals. Overlapping
  requests share one clock.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone

_LINE = re.compile(
    r"\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*ReqTimeStats\("
    r"rid=(?P<rid>[0-9a-f]+), input_len=(?P<input>\d+), cached_input_len=(?P<cached>\d+), "
    r"output_len=(?P<out>\d+).*?\): queue_duration=(?P<q>[\d.]+)ms, "
    r"initial_prefill_elapsed=(?P<pre>[\d.]+)ms, post_prefill_elapsed=(?P<post>[\d.]+)ms, "
    r"forward_duration=(?P<fwd>[\d.]+)ms(?:, entry_time=(?P<entry>[\d.]+))?"
)

_MATCH = re.compile(
    r"mamba match rid=(?P<rid>[0-9a-f]+) session=\S+ prefix=\S* "
    r"input=\d+ full_kv=(?P<full_kv>\d+) device=(?P<device>\d+) "
    r"host_hit=(?P<host_hit>\d+)"
)

# Shorter than this, a shallow resume is a cold prompt, not a collapsed checkpoint.
_DEEP_KV = 8192

_FIELDS = (
    "n",
    "span_s",
    "out_tok_s",
    "mean_req_tok_s",
    "busy_pct",
    "mean_inflight",
    "peak_inflight",
    "hit",
    "med_pre_s",
    "p90_pre_s",
    "mamba_of_kv",
    "trunc_pct",
    "prefill_tok_s",
    "decode_tok_s",
    "n_match",
    "output_tokens",
    "uncached_tokens",
)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    frac = index - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Length of the union of [start, end] intervals."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for left, right in ordered[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    return total + (end - start)


def _occupancy(
    intervals: list[tuple[float, float]], start: float, end: float
) -> tuple[float, int, float]:
    """Mean concurrency, peak, and busy percent of [start, end]."""
    span = end - start
    if span <= 0:
        return 0.0, 0, 0.0
    events: list[tuple[float, int]] = []
    for left, right in intervals:
        left = max(left, start)
        right = min(right, end)
        if right > left:
            events.append((left, 1))
            events.append((right, -1))
    if not events:
        return 0.0, 0, 0.0
    # A request ending as another starts is a handoff, not a peak of two.
    events.sort(key=lambda item: (item[0], item[1]))
    current = 0
    peak = 0
    area = 0.0
    busy = 0.0
    prev = start
    for when, delta in events:
        dt = when - prev
        if dt > 0 and current > 0:
            area += current * dt
            busy += dt
        current += delta
        peak = max(peak, current)
        prev = when
    return round(area / span, 2), peak, round(100 * busy / span, 1)


def _phases(
    row: dict,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Forward, prefill, and decode intervals. The log timestamp is completion."""
    finish = row["t"]
    forward_start = finish - row["forward_ms"] / 1000
    prefill_end = forward_start + row["prefill_ms"] / 1000
    decode_start = finish - row["post_ms"] / 1000
    return (forward_start, finish), (forward_start, prefill_end), (decode_start, finish)


def parse_mamba_matches(lines) -> dict[str, dict]:
    """First mamba-match line per rid. Resume length is device plus host hit."""
    found: dict[str, dict] = {}
    for line in lines:
        match = _MATCH.search(line)
        if not match:
            continue
        rid = match.group("rid")
        if rid in found:
            continue
        found[rid] = {
            "full_kv": int(match.group("full_kv")),
            "resume": int(match.group("device")) + int(match.group("host_hit")),
        }
    return found


def parse_requests(lines) -> list[dict]:
    """Return one dict per ReqTimeStats line, in log order."""
    matches = parse_mamba_matches(lines)
    rows = []
    for line in lines:
        match = _LINE.search(line)
        if not match:
            continue
        fields = match.groupdict()
        when = datetime.strptime(fields["ts"], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
        # The bracket time is whole seconds. entry_time is the precise arrival,
        # so arrival + queue + forward is the real completion.
        entry = float(fields["entry"]) if fields.get("entry") else 0.0
        if entry > 1_000_000_000:
            finish = entry + (float(fields["q"]) + float(fields["fwd"])) / 1000
        else:
            finish = when.timestamp()
        rid = fields["rid"]
        hit = matches.get(rid)
        rows.append(
            {
                "t": finish,
                "rid": rid,
                "input": int(fields["input"]),
                "cached": int(fields["cached"]),
                "output": int(fields["out"]),
                "prefill_ms": float(fields["pre"]),
                "post_ms": float(fields["post"]),
                "forward_ms": float(fields["fwd"]),
                "full_kv": None if hit is None else hit["full_kv"],
                "resume": None if hit is None else hit["resume"],
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    """One window covering every parsed request."""
    if not rows:
        return {key: 0 for key in _FIELDS}

    prompt = sum(row["input"] for row in rows)
    cached = sum(row["cached"] for row in rows)
    output = sum(row["output"] for row in rows)
    uncached = prompt - cached
    # Count the forward that produced the tokens, not just the gap between
    # completions. One request then reports output / its forward, and 0% busy
    # is no longer what a full forward looks like.
    start = min(row["t"] - row["forward_ms"] / 1000 for row in rows)
    end = max(row["t"] for row in rows)
    wall = end - start
    prefill_iv = []
    decode_iv = []
    forwards = []
    for row in rows:
        forward, prefill, decode = _phases(row)
        forwards.append(forward)
        if prefill[1] > prefill[0]:
            prefill_iv.append(prefill)
        if decode[1] > decode[0]:
            decode_iv.append(decode)
    prefill_busy = _union_seconds(prefill_iv)
    decode_busy = _union_seconds(decode_iv)
    prefill_s = [row["prefill_ms"] / 1000 for row in rows]
    mean, peak, busy = _occupancy(forwards, start, end if wall else start)
    request_rates = [
        row["output"] / (row["forward_ms"] / 1000)
        for row in rows
        if row["forward_ms"] > 0
    ]

    paired = [row for row in rows if row.get("full_kv") is not None]
    kv_sum = sum(float(row["full_kv"]) for row in paired)
    resume_sum = sum(float(row["resume"]) for row in paired)
    deep = [row for row in paired if row["full_kv"] >= _DEEP_KV]
    trunc = sum(1 for row in deep if row["resume"] < 0.5 * row["full_kv"])

    return {
        "n": len(rows),
        "span_s": round(wall, 1),
        "out_tok_s": round(output / wall) if wall else 0,
        "mean_req_tok_s": (
            round(sum(request_rates) / len(request_rates)) if request_rates else 0
        ),
        "busy_pct": busy,
        "mean_inflight": mean,
        "peak_inflight": peak,
        "hit": round(100 * cached / prompt, 1) if prompt else 0.0,
        "med_pre_s": round(_percentile(prefill_s, 0.5), 2),
        "p90_pre_s": round(_percentile(prefill_s, 0.9), 2),
        "mamba_of_kv": round(100 * resume_sum / kv_sum, 1) if kv_sum else 0.0,
        "trunc_pct": round(100 * trunc / len(deep), 1) if deep else 0.0,
        "prefill_tok_s": round(uncached / prefill_busy) if prefill_busy else 0,
        "decode_tok_s": round(output / decode_busy) if decode_busy else 0,
        "n_match": len(paired),
        "output_tokens": output,
        "uncached_tokens": uncached,
    }


def format_report(summary: dict) -> str:
    lines = [f"{key}: {summary[key]}" for key in _FIELDS]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "log",
        nargs="?",
        help="Server log path. Reads stdin when omitted.",
    )
    args = parser.parse_args(argv)
    if args.log:
        with open(args.log, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    else:
        lines = sys.stdin.readlines()
    summary = summarize(parse_requests(lines))
    sys.stdout.write(format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
