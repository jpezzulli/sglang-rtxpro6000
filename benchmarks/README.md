# Benchmark evidence

This directory contains compact public evidence. It deliberately excludes raw
request bodies, private prompts, conversation text, complete hidden reasoning,
model weights, and cache contents.

## Agentic sample

`agentic/2026-08-24-agentic-server.sanitized.log` is a normal-use SGLang server
sample from 15:56:11 through 18:18:51 EDT. HTTP access-log lines containing the
private client address were removed; request timing, prefill/decode telemetry,
acceptance, CUDA-graph, Mamba, and warning lines were retained.

The current medium reasoning/tool suite ran earlier, from 14:52:48 through
14:56:07 EDT. It does not overlap this sample. The 124 parsed completions are
therefore treated as real agentic workload rather than qualification traffic.

Reproduce the derived artifacts with only Python's standard library:

```bash
python scripts/pennyroyal/analyze_req_time_stats.py \
  benchmarks/agentic/2026-08-24-agentic-server.sanitized.log \
  --csv-out benchmarks/agentic/2026-08-24-req-time-stats.csv \
  --json-out benchmarks/agentic/2026-08-24-summary.json
```

Completed-request effective decode is exactly:

```text
output_len / (post_prefill_elapsed_ms / 1000)
```

The JSON reports, for every context band, both the per-request median and
token-weighted effective decode `sum(output_tokens) / sum(post_prefill_seconds)`.
Request overlap is estimated from logged entry time plus queue/forward elapsed
spans. It separates obvious concurrent interference but is not a transactional
server scheduler trace. Decode telemetry does not carry request IDs, so
acceptance is summarized by full-token and active-request bands rather than
silently assigned to individual concurrent requests.

## Controlled and reasoning summaries

- `controlled/2026-08-21-summary.json` is a compact extraction of the final
  clean-runtime performance and xhigh reasoning qualification.
- `controlled/2026-08-21-community-tp1-baseline.json` records the current
  public RTX6KPRO TP1 official-FP8/MTP3 baseline and calculated directional
  deltas.
- `reasoning/2026-08-24-medium-summary.json` is a compact extraction of the
  medium-effort reasoning and tool qualification.

Detailed public test definitions, result records, and methodology live in
[`jpezzulli/pennyroyal-validation`](https://github.com/jpezzulli/pennyroyal-validation)
at validation revision `d0c86c40222eacfd8c39c2db3439b07065916c1a` for these captures. Local raw
SSE/request artifacts were used to verify the summaries but are not duplicated
here because this repository is a runtime distribution, not a prompt corpus.
