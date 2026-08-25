# Results

These are several different measurement classes. They are intentionally not
collapsed into one throughput number.

## Measurement definitions

- **Completed-request effective decode:** `output_len / (post_prefill_elapsed_ms / 1000)` from SGLang `ReqTimeStats`. This excludes the initial prefill span but includes scheduler waits, re-prefill after retraction, and interference during the post-prefill phase.
- **Per-request median:** median of those completed-request rates. It describes the typical request, independent of output length.
- **Token-weighted effective decode:** `sum(output_tokens) / sum(post_prefill_seconds)`. It gives longer and slower generations their actual time weight.
- **Instantaneous server throughput:** SGLang's periodic `gen throughput` telemetry. This is a short window and is not sustained completed-request throughput.
- **Concurrent aggregate throughput:** total output tokens divided by the group makespan, or SGLang telemetry while the specified number of requests is simultaneously active.
- **Controlled prefill:** prompt tokens divided by time to first token. It is separate from decode.

## Headline

On one RTX PRO 6000, ordinary short-context agentic requests had a **165.25
tok/s per-request median** in the supplied two-hour log. Across non-overlapping
requests below 150K input tokens, the median was approximately **163 tok/s**.
The same live workload slowed with deep context: non-overlapping 300-360K
requests had a **112.50 tok/s median** and **114.44 tok/s token-weighted rate**;
the 340-360K band was **105.45 median / 101.69 token-weighted tok/s**.

The fastest completed request was **244.24 tok/s**. Favorable DFlash2 telemetry
windows reached **258-300 tok/s**, but those are not presented as sustained
request throughput.

## Real agentic completed requests

The sanitized server sample spans 2026-08-24 15:56:11-18:18:51 EDT and contains
124 completed requests, 85,156 output tokens, and contexts from 183 to 350,195
input tokens. The medium reasoning/tool suite ended at 14:56:07, one hour
before this sample began; none of these 124 completions belongs to that suite.

The primary context curve below excludes seven requests whose lifecycle spans
overlapped another completed request in the same sample. It does not pretend
that those requests did not happen; the full observed table follows.

### Non-overlapping request intervals

| Input context | Requests | Output tokens | Median effective decode | Token-weighted effective decode |
|---|---:|---:|---:|---:|
| 0-2K | 34 | 12,791 | 165.68 tok/s | 175.46 tok/s |
| 64-100K | 5 | 1,250 | 168.01 tok/s | 151.46 tok/s |
| 100-150K | 14 | 13,901 | 133.25 tok/s | 130.36 tok/s |
| 150-262K | 2 | 605 | 117.96 tok/s | 116.10 tok/s |
| 300-325K | 21 | 19,944 | 110.81 tok/s | 116.69 tok/s |
| 325-340K | 18 | 11,263 | 133.03 tok/s | 130.98 tok/s |
| 340-360K | 23 | 14,415 | 105.45 tok/s | 101.69 tok/s |
| **All non-overlapping** | **117** | **74,169** | **133.21 tok/s** | **125.36 tok/s** |

There were no completed requests in the 2-64K or 262-300K bands after overlap
filtering. The 325-340K band being faster than 300-325K is real; acceptance and
output mix vary enough that context does not create a perfectly monotonic
curve. The deepest 340K+ band still shows a clear practical slowdown.

### Full observed sample, including interference

| Input context | Requests | Output tokens | Median effective decode | Token-weighted effective decode |
|---|---:|---:|---:|---:|
| 0-2K | 36 | 13,118 | 165.25 tok/s | 148.35 tok/s |
| 2-64K | 1 | 4,123 | 43.02 tok/s | 43.02 tok/s |
| 64-100K | 5 | 1,250 | 168.01 tok/s | 151.46 tok/s |
| 100-150K | 15 | 16,922 | 130.25 tok/s | 124.36 tok/s |
| 150-262K | 3 | 676 | 112.29 tok/s | 111.06 tok/s |
| 300-325K | 21 | 19,944 | 110.81 tok/s | 116.69 tok/s |
| 325-340K | 18 | 11,263 | 133.03 tok/s | 130.98 tok/s |
| 340-360K | 25 | 17,860 | 103.81 tok/s | 74.83 tok/s |
| **All observed** | **124** | **85,156** | **131.31 tok/s** | **102.57 tok/s** |

The seven overlapping requests had a 68.37 tok/s median and 46.05 tok/s
token-weighted rate. This is not an engine-only speed estimate; it records real
interference. One 28.7K-input/4,123-output request fell to 43.02 tok/s while two
large request intervals overlapped it. A 345K-input request reached only 20.79
tok/s under similar overlap. Another 135.8K-input, 3,021-output request completed
at 102.62 tok/s while acceptance remained mostly low and other work entered.

Representative longer clean completions include:

| Input | Cached input | Output | Effective decode | Note |
|---:|---:|---:|---:|---|
| 308,818 | 308,416 | 4,981 | 126.73 tok/s | long completion, small appended delta |
| 310,640 | 308,800 | 5,768 | 130.71 tok/s | largest output in sample |
| 325,481 | 324,992 | 3,668 | 143.41 tok/s | favorable 325K session segment |
| 129,247 | 127,040 | 4,429 | 121.58 tok/s | long 129K completion |
| 136,999 | 135,744 | 1,018 | 244.24 tok/s | near-perfect DFlash2 acceptance windows |
| 348,154 | 347,904 | 1,746 | 91.79 tok/s | deep-context slowdown |

## DFlash2 acceptance and telemetry

Acceptance varies materially with workload. Across single-running-request
telemetry, the median instantaneous throughput was 106.22 tok/s and mean
acceptance length was 3.81. The strongest window was 300.16 tok/s at acceptance
length 7.75 / rate 0.96. The 244.24 tok/s completed request contained three
successive windows at 258.26, 276.34, and 278.40 tok/s with acceptance lengths
7.40, 7.95, and 7.97.

At 340-360K full-token telemetry, the single-request median was 90.10 tok/s,
with mean acceptance length 3.32 and acceptance rate 0.332. High acceptance is
a major contributor to the fastest windows, but lower-acceptance work still
completes usefully, commonly around 100-140 tok/s at long context.

The raw sanitized log, every parsed request, and machine-generated summaries
are in [benchmarks/agentic](benchmarks/agentic/).

## Controlled performance qualification

### Primary public TP1 baseline

The primary baseline is the current public
[`local-inference-lab/rtx6kpro` Qwen3.8-27B catalog](https://github.com/local-inference-lab/rtx6kpro/blob/7ceba1df33bb9c76060a251bba2d851b4f37a485/models/qwen38-27b.md)
at commit `7ceba1df33bb9c76060a251bba2d851b4f37a485`. This avoids choosing one of
this project's already-custom Ferrari or DSpARK stages as the denominator.

Its closest TP1 row uses official FP8 weights, vLLM Gilded Gnosis r31, MTP3,
FP8 KV, a 48 GiB native KV-offload tier, prefix caching, four sequences, and
one 600 W RTX PRO 6000. The catalog marks it `research-only`: one community
capture without repeat variance or quality-gate receipts.

| Metric | Community TP1 official-FP8/MTP3 | This DFlash2 distribution | Directional change |
|---|---:|---:|---:|
| C1 at/near empty context | 77.8 tok/s | 108.75 tok/s | **+39.8%** |
| C4 at/near empty context | 292.7 tok/s | 390.23 tok/s | **+33.3%** |
| 64K prefill | 5,877 tok/s | 6,163 tok/s | **+4.9%** |

The catalog's complete C1 curve is 77.8 at ctx0, 76.2 at 16K, 73.9 at 32K,
81.7 at 64K, and 69.8 at 128K. Its C4 curve is 292.7, 296.5, 279.7, and
264.7 through 64K. This distribution's real 100-150K agentic work reached
133.25 tok/s median and 130.36 tok/s token-weighted without detected overlap,
but that is a natural-workload observation rather than the catalog's fixed C1
method and is not presented as a percentage speedup.

This is still not a strict A/B: the checkpoint, runtime, speculative method,
power limit, client harness, output duration, and cache/offload configuration
differ. The catalog also cites an isolated 106.9 tok/s MTP result for a W8A8
checkpoint under another measurement; that number is close to this runtime's
108.75 controlled C1 and is intentionally not hidden or merged into the
official-FP8 matrix.

Ferrari and DSpARK remain useful local chronology, but both were already custom
runtime work and are not used as the publication baseline.

### This distribution's controlled results

| Workload | Context | Concurrency | Effort | Output | Result | Notes |
|---|---:|---:|---|---:|---:|---|
| Prefill | 63,906 | 1 | xhigh request | 28 | 6,163.07 prompt tok/s | 10.369 s TTFT |
| Prefill + needles | 489,921 | 1 | xhigh request | 203 | 1,618.31 prompt tok/s | all 3 needle locations exact |
| Decode ceiling | 136 prompt | 1 | xhigh | 1,024 | 108.75 tok/s | post-first-token; 107.52 makespan |
| Decode ceilings | 136 each | 4 | xhigh | 4 x 1,024 | 390.23 tok/s aggregate | per-stream 120.79/99.78/102.82/104.26 |
| Reasoning active decode | case-dependent | 3 | xhigh | 70,770 group tokens | 396.78 tok/s server median | active-sample mean 400.96; acceptance 3.98/0.426 |
| Reasoning active decode | case-dependent | 3 | medium | suite-dependent | 486.19 tok/s server median | mean 483.29; acceptance 4.657/0.522 |

The 1x/4x decode ceiling test is deliberately simple and output-length-bound;
it is not a substitute for completed natural responses. The three-request
reasoning figures are server telemetry while exactly three requests were
active, not per-request decode rates.

## Quality and tool behavior

| Qualification | Result |
|---|---:|
| xhigh reasoning | 98.26 / 100 dimension-weighted |
| medium reasoning | 95.807 / 100 dimension-weighted |
| Medium completion-token reduction vs xhigh | 80.4% |
| Medium summed-request-time reduction vs xhigh | 84.74% |
| Medium completion tokens / summed request second | 160.21 tok/s |
| xhigh completion tokens / summed request second | 124.74 tok/s |
| Medium tool suite | 27/30 automatic; 30/30 exact calls; 29/30 reviewed semantic |

Medium and xhigh used different concurrency schedules and generated very
different token distributions. This is a measured operational tradeoff, not a
claim that medium is universally equivalent to xhigh.

## HiCache/NIXL behavior

The selected persistence path demonstrated exact long-prefix restoration,
configuration-specific namespace isolation, rollback to a prior namespace,
and concurrent restore. Representative process-restart results included:

- 518,528 tokens restored with a six-token tail in 14.64 seconds, all needles exact;
- an identical 60K namespace restart restored 60,032 tokens in 3.16 seconds;
- switching page size selected another directory; switching back reused the original;
- three approximately 60K requests restored after restart in 6.92 seconds;
- 375K/200K/200K concurrent restore reused 775,168 tokens and computed only 320 tail/page-rounding tokens.

These results establish prefix persistence behavior. They do not explain or
cause DFlash2's GPU-resident decode throughput.
