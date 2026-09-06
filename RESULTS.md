# Results

This document preserves the complete useful 27B/DFlash2 measurement record and
adds Flash-Next as a separate configuration of the same SGLang source. Metrics
with different timing windows are never added together or presented as one
interchangeable throughput number.

## Measurement definitions

- **Cold prefill:** prompt tokens divided by time to first token with no matching
  radix or persistent prefix.
- **Completed-request effective decode:** output tokens divided by SGLang's
  post-prefill elapsed span. This excludes initial prefill but includes waits,
  re-prefill after retraction, and interference after the prefill boundary.
- **Per-request median:** median of completed-request effective decode rates.
- **Token-weighted effective decode:** `sum(output_tokens) /
  sum(post_prefill_seconds)`.
- **Instantaneous server throughput:** SGLang periodic `gen throughput`; a short
  telemetry window, not completed-request throughput.
- **Concurrent aggregate:** all group output tokens divided by synchronized
  group makespan.
- **Restored-prefix effective prefill:** input tokens divided by TTFT after
  NIXL restoration. It is not cold model prefill.

## Pennyroyal 2.3 — Flash-Next FR-Spec

September 5–6, 2026. Same Flash-Next ModelOpt NVFP4 checkpoint across arms,
one RTX PRO 6000, TP1; executable `836206a0ad`, unchanged v2.1.2 wheel.
Native NEXTN 3/1/4, 65,536-ID FR map, full target vocabulary and unchanged
acceptance policy. Context 524,288, KV 824,384, page 64, C4, 24 Mamba slots,
BF16 GDN, graphs and 32 GiB HiCache/NIXL remain enabled.

Adoption credit: [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120).
The token map was independently generated; no external throughput headline
or complete external patch stack is presented as a Penny measurement.

### Controlled decode and prefill

Ordinary Chat Completions, ordinary warmups, 1,024 server-reported completion
tokens per stream. C1 is **post-first-token** throughput; C4 is aggregate
**synchronized whole-request makespan**, including first-token delay.

| Arm | Samples C1/C4 | C1 median, tok/s | C1 whole-request median, tok/s | C4 aggregate median, tok/s |
|---|---:|---:|---:|---:|
| Initial v2.1.2 baseline | 3/3 | 156.79 | 151.17 | 417.92 |
| Initial FR-Spec | 3/3 | 165.33 | 157.16 | 447.83 |
| v2.1.2 baseline after reboot | 3/3 | 147.15 | 140.88 | 427.91 |
| v2.3 FR-Spec | 6/6 | **171.93** | **164.40** | **447.04** |

The v2.3 configuration improved C1 by **9.7–16.8%** and C4 by **4.5–7.0%**
against the two baseline medians. The initial FR trial measured +5.4%/+7.2%
against the initial baseline. These are sequential session comparisons with
material baseline variance and an intervening reboot, not randomized paired
confidence intervals. All sample-level rates and per-stream/acceptance/memory
observations are in the [numeric record](https://github.com/jpezzulli/pennyroyal-validation/blob/main/results/qwen38-flash-next-frspec-20260905.json).

| Fresh-namespace prefill | Baseline TTFT | FR-Spec TTFT | Correctness |
|---|---:|---:|---|
| 63,864 prompt tokens | 4.969 s | 4.886 s | Exact READY |
| One 489,879-token prompt | 61.070 s | 61.319 s | All three separated needles exact |

Cold prefill is essentially unchanged. It is not mixed with decode. A later
clean six-sample confirmation measured FR-only C1 **173.05** and C4 **450.94**;
adding the held BF16 low-M kernel measured **172.13**/**451.55** (−0.53%/+0.13%).
That kernel is excluded from 2.3. Those follow-up arms are not replacements
for the frozen baseline/FR measurements or a new cold-prefill A/B.

### Qualification and persistence

- Full current reasoning: two warmups, nine measured requests in the
  `three-user-1-3-3-1` schedule; **98.52/100**, 131,770 completion tokens,
  all natural stops and no fatal/loop caps. Full case/dimension breakdown
  and grading limitations are in the detailed report.
- Tools: 27/30 literal, 29/30 exact calls/arguments, 30/30 parseable, zero
  runtime errors. Actual workflows: **29 clean plus one redundant read-only
  call**, no consequential execution failure. Two answer-discipline caveats
  are separately disclosed; no invented 30/30 automatic result.
- Vision and sealed agentic passed. The natural 3,072-token response contained
  all direct IDs; the original automatic failure is retained because the
  collector expected the wrong field. Collector fixes were not bundled.
- Both supported profiles passed smoke, 64K/490K prefill, C1/C4 decode, and
  actual NIXL restart restoration, with their original graph/pool capacities.
- FR restart restored **489,856 tokens** with three exact needles. Concurrent
  paired 64K/490K restoration completed correctly in **9.396 s**, with 553,664
  NIXL-hit tokens plus 553,664 device-hit tokens and four Mamba states restored.
  Duplicate prefixes reused admitted state; this is not four independent
  full disk transfers. Two dependent 490K conversation turns also passed.
- Final recovery after the excluded-kernel A/B again passed 9/9 smoke and
  489,856-token NIXL restoration. Full capacity and idle state were verified.

Detailed model results and interpretation belong in the
[validation report](https://github.com/jpezzulli/pennyroyal-validation/blob/main/results/qwen38-flash-next-frspec-20260905.md).
The raw local evidence and first attempts remain preserved separately.
No new 27B speed gain, shared-kernel promotion, or transactional-cache claim.

## Earlier Flash-Next campaign

Source: `64ecd64924fee338e3bf846a32167cd604186827`.

| Workload | Context / concurrency | Result | Correctness / note |
|---|---|---:|---|
| Cold prefill | ~64K, C1 | 10,103.70 tok/s | normal chat request |
| Cold prefill + needles | ~490K, C1 | 7,872.15 tok/s | 3/3 exact needles |
| Decode | 1 x 1,024 output | 171.09 tok/s | post-first-token |
| Decode | 4 x 1,024 output | 427.54 tok/s aggregate | synchronized batch makespan |
| Native NEXTN acceptance | final suite | 2.58 mean accepted / 52.74% | live speculative telemetry |
| Reasoning | full suite | 97.49/100 | 139,863 completion tokens |
| Tools | 30 invocations | 30/30 exact calls and semantic success | legacy literal checker was 27/30 |
| Vision | complete request | passed | 1,024-token validation |
| Sealed agentic control | controlled request | 148.80 tok/s | normal API path |
| Natural decode | 3,072 output | 162.05 tok/s | normal response |

Flash-Next QSA sparse decode resolved to XQA inside FlashInfer's wrapper for
this campaign. These measured results remain valid; they are not evidence for
TRTLLM-Gen on SM120, and no matched end-to-end backend percentage is claimed.

### Four-request timing reconciliation

The reported individual streams were 115.13, 127.56, 126.64, and 122.96
tok/s. Each uses 1,023 post-first-token tokens over its own 8.02-8.89 second
decode interval. The aggregate instead uses all 4,096 completion tokens over
the synchronized 9.580393-second batch, including TTFT and the slowest tail:

```text
4096 / 9.580393 = 427.539869 tok/s
```

The stream values are useful latency observations but are not additive. The
427.54 figure is the matched aggregate measurement.

### Flash-Next real agentic sample

| Measurement | Result |
|---|---:|
| Completed requests | 96 |
| Output tokens | 100,666 |
| Token-weighted effective decode | 139.54 tok/s |
| Per-request median | 153.48 tok/s |
| Arithmetic mean | 155.05 tok/s |
| Sustained completed-request peak | 218.84 tok/s |
| Instantaneous C1 peak | 246.99 tok/s |
| Brief instantaneous C4 peak | 543.07 tok/s |
| 90K-279K lane | 138.66 weighted / 148.41 median tok/s |

The first telemetry sample after a large prefill was excluded because its
window mixed prefill or idle time with decode. The sample showed no high-
context cliff; acceptance variation caused most visible throughput movement.

### Flash-Next persistent restoration

| Prefix | Restored | Recomputed | Effective restored-prefix rate | Result |
|---|---:|---:|---:|---|
| ~64K | 63,808 | 56 | not used as a cold-prefill comparison | coherent response |
| ~490K | 489,856 | 23 | 62,040.60 tok/s | 3/3 needles exact |

## Independent TP=2 FP8 Validation

These are third-party results reported by Reddit user H3PO, not Penny's TP=1
benchmark campaign. H3PO used the published runtime with a dual-RTX PRO 6000
TP=2/EP=2 deployment without NVLink, the
`Qwen/Qwen3.8-Flash-Next-FP8` checkpoint, FP8 KV, native NEXTN MTP, and a
StackOverflow-derived coding corpus. The complete report and surrounding
diagnostic chain are in [H3PO's Reddit
comment](https://www.reddit.com/r/BlackwellPerformance/comments/1w04xb7/comment/p6ek6e8/).

The decisive configuration result was removing
`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`, an optional setting absent from Penny's
qualified launcher. Native MTP then booted and sustained generation. This did
not require disabling CUDA graphs, MTP, QSA, GDN, TP=2/EP=2, or FP8. The
earlier `custom_all_reduce.cuh` graph-capture failure was separate and had
already been resolved by removing
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

The headline observations are:

- C1 prefill: **13,414.47 ± 82.74 tok/s** on the coding corpus.
- Aggregate decode scaled from **200.93 ± 12.87 tok/s at C1** to
  **295.88 ± 29.07 at C2**, **330.76 ± 22.04 at C4**, and
  **339.34 ± 11.14 at C6**.
- `mem-fraction-static=0.95` yielded a **3,182,848-token** KV pool, described by
  H3PO as approximately six 512K contexts.
- The high-concurrency prefill rows have very large variance and visible
  scheduler effects. They are retained below as raw evidence, not promoted as
  comparative headline results.

Third-party benchmark shape, with the private endpoint anonymized:

```bash
~/.venv/bin/llama-benchy \
  --base-url http://<server>:6000/v1 \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --served-model-name Qwen3.8-Flash-Next-FP8-TP2 \
  --pp 4096 --tg 1024 --latency-mode generation \
  --concurrency 1 2 4 6 --runs 3 --depth 0 \
  --book-url local://coding.txt
```

Raw table as reported:

| Model | Test | t/s (total) | t/s (req) | Peak t/s | Peak t/s (req) | TTFR (ms) | Est. PPT (ms) | E2E TTFT (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3.8-Flash-Next-FP8 | pp4096 (c1) | 13414.47 ± 82.74 | 13414.47 ± 82.74 |  |  | 408.69 ± 1.89 | 305.43 ± 1.89 | 408.69 ± 1.89 |
| Qwen/Qwen3.8-Flash-Next-FP8 | tg1024 (c1) | 200.93 ± 12.87 | 200.93 ± 12.87 | 201.33 ± 12.81 | 201.33 ± 12.81 |  |  |  |
| Qwen/Qwen3.8-Flash-Next-FP8 | pp4096 (c2) | 10770.55 ± 20.76 | 6723.66 ± 491.07 |  |  | 715.87 ± 44.74 | 612.61 ± 44.74 | 715.87 ± 44.74 |
| Qwen/Qwen3.8-Flash-Next-FP8 | tg1024 (c2) | 295.88 ± 29.07 | 165.99 ± 9.30 | 333.00 ± 4.90 | 166.50 ± 9.18 |  |  |  |
| Qwen/Qwen3.8-Flash-Next-FP8 | pp4096 (c4) | 2806.18 ± 790.40 | 4119.82 ± 2284.63 |  |  | 2260.41 ± 2428.25 | 2157.14 ± 2428.25 | 2260.41 ± 2428.25 |
| Qwen/Qwen3.8-Flash-Next-FP8 | tg1024 (c4) | 330.76 ± 22.04 | 144.41 ± 10.83 | 426.33 ± 8.18 | 144.75 ± 10.83 |  |  |  |
| Qwen/Qwen3.8-Flash-Next-FP8 | pp4096 (c6) | 2848.10 ± 249.01 | 2981.79 ± 2520.70 |  |  | 4068.86 ± 3489.61 | 3965.60 ± 3489.61 | 4068.86 ± 3489.61 |
| Qwen/Qwen3.8-Flash-Next-FP8 | tg1024 (c6) | 339.34 ± 11.14 | 135.70 ± 15.21 | 471.33 ± 56.84 | 136.11 ± 15.11 |  |  |  |

This independently extends the integration evidence across:

- TP=1 NVFP4 and TP=2 FP8;
- different checkpoints;
- different hardware configurations;
- different workloads; and
- another operator's deployment.

It is useful upstream evidence that the integration is not confined to
Penny's exact TP=1 shape. It is not a controlled TP=1-versus-TP=2 comparison,
and the third-party result has not been reproduced locally.

## Qwen3.8-27B/DFlash2 current-launcher confirmation

This 2026-08-26 campaign qualified the current 24-slot, five-state path setting
at runtime-line source `8e197ed3af`. It is separate from the earlier dated
performance release below.

| Property | Result |
|---|---:|
| Target/draft FP8 KV capacity | 1,118,784 tokens each |
| Mamba slots / state-path cap | 24 / 5 |
| Maximum observed Mamba entries | 10 |
| 64K prefill | 6,169.18 tok/s |
| ~490K prefill | 1,616.29 tok/s; 3/3 needles exact |
| 1x decode | 108.93 tok/s |
| 4x aggregate decode | 375.81 tok/s |
| 4x individual post-first-token | 98.69 / 103.00 / 96.51 / 105.61 tok/s |
| DFlash2 acceptance during C4 | 2.92 mean accepted / 27.67% |
| Reasoning | 96.92/100 across 50,986 completion tokens |
| Reasoning token-weighted effective decode | 156.70 tok/s |
| Tool selection and arguments | 30/30 |
| Reviewed response discipline | 29/30; no tool-use or security failure |

## Qwen3.8-27B/DFlash2 dated performance release

Tag: `qwen38-dflash2-pro6000-20260824`. This is the established 27B result set
and is retained exactly rather than overwritten by the later allocation.

### Controlled results

| Workload | Context | Concurrency | Effort | Output | Result | Notes |
|---|---:|---:|---|---:|---:|---|
| Prefill | 63,906 | 1 | xhigh request | 28 | 6,163.07 prompt tok/s | 10.369 s TTFT |
| Prefill + needles | 489,921 | 1 | xhigh request | 203 | 1,618.31 prompt tok/s | 3/3 exact |
| Decode ceiling | 136 prompt | 1 | xhigh | 1,024 | 108.75 tok/s | post-first-token; 107.52 makespan |
| Decode ceiling | 136 each | 4 | xhigh | 4 x 1,024 | 390.23 tok/s aggregate | group makespan |
| Reasoning active decode | case-dependent | 3 | xhigh | 70,770 group tokens | 396.78 server tok/s median | telemetry while C3 active |
| Reasoning active decode | case-dependent | 3 | medium | suite-dependent | 486.19 server tok/s median | acceptance 4.657 / 0.522 |

The four individual decode rates were 120.79, 99.78, 102.82, and 104.26 tok/s.
As with Flash-Next, these are post-first-token per-stream windows, while 390.23
uses the synchronized group makespan; they are not additive.

### Public TP1 directional baseline

The cited community baseline is the
[`local-inference-lab/rtx6kpro` Qwen3.8-27B catalog](https://github.com/local-inference-lab/rtx6kpro/blob/7ceba1df33bb9c76060a251bba2d851b4f37a485/models/qwen38-27b.md)
at commit `7ceba1df33bb9c76060a251bba2d851b4f37a485`. Its closest TP1 row used the
official FP8 checkpoint, vLLM Gilded Gnosis r31, MTP3, FP8 KV, prefix caching,
and one RTX PRO 6000.

| Metric | Community official-FP8/MTP3 | DFlash2 dated release | Directional change |
|---|---:|---:|---:|
| C1 near empty context | 77.8 tok/s | 108.75 tok/s | +39.8% |
| C4 near empty context | 292.7 tok/s | 390.23 tok/s | +33.3% |
| 64K prefill | 5,877 tok/s | 6,163 tok/s | +4.9% |

This is not a strict A/B. Checkpoint, runtime, speculation, power, client,
prompt, duration, and cache/offload configuration differ. No “fastest” claim is
made.

### Reasoning and tools

| Qualification | Result |
|---|---:|
| xhigh reasoning | 98.26/100 dimension-weighted |
| medium reasoning | 95.807/100 dimension-weighted |
| Medium completion-token reduction vs xhigh | 80.4% |
| Medium summed-request-time reduction vs xhigh | 84.74% |
| Medium completion tokens / summed request second | 160.21 tok/s |
| xhigh completion tokens / summed request second | 124.74 tok/s |
| Medium tool suite | 27/30 automatic; 30/30 exact calls; 29/30 reviewed semantic |

The two automatic tool misses caused by wording/checker mismatch were retained
as raw checker outcomes; the reviewed result distinguishes those from the one
substantive response-discipline miss.

### Real agentic completed requests

The sanitized sample spans 2026-08-24 15:56:11-18:18:51 EDT and contains 124
completed requests, 85,156 output tokens, and 183-350,195 input tokens. It
begins one hour after the reasoning/tool suite ended.

#### Non-overlapping request intervals

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

#### Full sample, including interference

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

Seven overlapping requests measured 68.37 tok/s median and 46.05 tok/s
weighted. That is valid user experience under interference, not an isolated
engine-speed estimate.

### DFlash2 acceptance

Across single-running-request telemetry, median instantaneous throughput was
106.22 tok/s and mean accepted length was 3.81. The strongest window reached
300.16 tok/s at 7.75 accepted tokens and 0.96 acceptance. The fastest completed
request reached 244.24 tok/s and contained successive 258.26, 276.34, and
278.40 tok/s telemetry windows. At 340-360K, single-request telemetry measured
90.10 tok/s median with 3.32 mean accepted length and 0.332 acceptance.

### 27B HiCache/NIXL qualification

- 518,528 tokens restored with a six-token tail in 14.64 seconds; all needles
  exact.
- Identical 60K configuration restored 60,032 tokens after restart in 3.16
  seconds.
- Changing page size selected another namespace; restoring the original setting
  selected and reused the original namespace.
- Three approximately 60K requests restored concurrently after restart in 6.92
  seconds.
- 375K/200K/200K concurrent restoration reused 775,168 tokens and computed only
  320 tail/page-rounding tokens.

These results establish prefix persistence. They do not explain or cause
DFlash2's GPU-resident decode rate.

The dated tag retains the sanitized raw 27B agentic log and machine-generated
summaries. The current maintained harness and published result catalog are in
[`jpezzulli/pennyroyal-validation`](https://github.com/jpezzulli/pennyroyal-validation).
