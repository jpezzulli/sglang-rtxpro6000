# Results

This document preserves the complete useful 27B/DFlash2 measurement record and
adds Flash-Next as a separate configuration of the same SGLang source. Metrics
with different timing windows are never added together or presented as one
interchangeable throughput number.

## Measurement definitions

- **Cold prefill:** processing a prompt with no matching radix or persistent
  prefix. Timing is explicitly labelled: **server prefill throughput** divides
  prompt tokens by SGLang's initial-prefill span; **client-TTFT throughput**
  divides them by time to first token, including other request processing.
  Earlier v2.3 and dated tables use the client-TTFT definition.
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

## Pennyroyal v2.5.0 — Optional online FP8

Measured September 11, 2026, on one RTX PRO 6000, TP1. The fresh immediately
preceding option-off reference and v2.5.0 online-FP8 arm used the same compatible
Flash-Next checkpoint artifact with RAM PLE, FR-Spec, native NEXTN, 524,288-
token context, 824,384 KV tokens, page size 64, 24 Mamba slots, CUDA graphs and
32 GiB HiCache/NIXL. The v2.5.0 source also includes startup-only structured-
output warmup and loader-lifetime maintenance, so the measurements do not
attribute every difference solely to online FP8. The precision option itself
changes selected projection storage and compute; it does not change the
checkpoint's NVFP4 expert, router or FP8 PLE-table formats, BF16 GDN state,
FP8 KV, or FR-Spec map and scale alignment.

### Short single-request decode

All six requests returned HTTP 200 and exactly 1,024 completion tokens. Rates
are client-observed generation after the first token, excluding TTFT.

| Precision option | Three post-first-token samples | Median | Median TTFT | Median whole-request rate |
|---|---|---:|---:|---:|
| Off | 161.465, 163.069, 155.835 tok/s | **161.465 tok/s** | 0.341 s | 153.066 tok/s |
| Online FP8 | 212.808, 193.004, 207.124 tok/s | **207.124 tok/s** | 0.428 s | 187.777 tok/s |

Observed median change was **+28.28%** post-first-token and **+22.68%** over
whole-request makespan. Median TTFT was 0.087 seconds higher. Three samples per
arm are not a distribution estimate.

### 128K and 490K decode

The C1 rows used exactly 1,024 output tokens. All retrieval checks found the
three expected keys in every C1 and C4 response.

| Context/test | Option off | Online FP8 | Observed change | Timing detail |
|---|---:|---:|---:|---|
| 128K C1 post-first-token | 154.703 tok/s | **195.634 tok/s** | **+26.46%** | TTFT 10.111 → 12.906 s; whole request 16.755 → 18.145 s |
| 490K C1 post-first-token | 149.135 tok/s | **172.644 tok/s** | **+15.76%** | TTFT 61.747 → 63.296 s; whole request 68.640 → 69.254 s |
| 128K C4 concurrent aggregate | 367.127 tok/s | **422.037 tok/s** | **+14.96%** | 15,725 vs 15,898 output tokens; 42.833 vs 37.670 s batch wall |
| 490K C4 concurrent aggregate | 236.678 tok/s | **329.010 tok/s** | **+39.01%** | 15,455 vs 15,127 output tokens; 65.300 vs 45.977 s batch wall |

The long C1 results are single observations. The C4 requests ended naturally
at either `stop` or a 4,096-token cap, so their differing total output counts
make them contextual throughput observations, not fixed-output matched decode
rates. All four streams overlapped for 30.152 seconds at 128K and 24.850
seconds at 490K in the online-FP8 arm. Most importantly, **cold TTFT did not
improve** in the matching long samples.

### VRAM, compatibility, and functional checks

The online-FP8 boot reported **7.52 GiB available after CUDA-graph capture**;
the earlier matching option-off boot reported **3.66 GiB**. The observed
difference is about 3.86 GiB. The 7.52 GiB value is the total left after graphs,
not a claim that 7 GiB was newly freed, and it does not qualify a larger token
pool. Both measured arms retained the same 824,384-token pool.

Representative checks covered real SM120 FP8 kernels and graph replay,
reasoning, schema/tool use, vision, a sealed agent workflow, long-context
retrieval, and identical-restart NIXL restoration. The saved 490K request
restored/prefetched 489,984 tokens, loaded two Mamba states, and returned all
three expected keys.

The fully exercised compatible artifact scored **95.75/100** in one blinded
reasoning review, with no fatal cap. This is one local validation-suite sample,
not an independently preregistered benchmark or evidence of improvement over
an older precision. It was not a RadixArk result.

With online FP8, the model GPU was also selected for media preprocessing as
logical `cuda:0`, with no second GPU visible. The 824,384-token pool and graphs
were retained, and all ten selected media checks passed: ordinary/large JPEG,
ten images, two concurrent ten-image requests, static MP4 frames, and three
successive image-history turns at 208,021, 208,078 and 208,135 input tokens.
Minimum sampled free GPU memory was 1,897 MiB. This is direct coverage for those
shapes, not a universal media/concurrency headroom guarantee.

One cold synthetic request with 393,223 input tokens recovered **64/64 opaque
records exactly**, with no missing, wrong, duplicate, or unexpected keys. It
used 2,754 completion tokens, reached first token in 44.523 seconds, and
completed in 53.491 seconds. This uniform synthetic archive tests dense exact
recall, not ordinary agentic work.

The public
[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
checkpoint separately passed the identical 64/64 exact-recall check. Its client
TTFT was 47.296 seconds and whole wall time 55.086 seconds. Only this focused
compatibility check was run on RadixArk; the performance, reasoning, tools and
vision results above must not be attributed to that checkpoint.

The ordinary tool suite produced semantically correct/exact call lists in
29/30 workflows; one workflow added a redundant delegation and unsupported
cross-check claims. A separate quoted-markup probe had 10/10 transport success
but only 7/10 correct behavior: three of six quoted fully wrapped examples
became unwanted calls. Parser code was unchanged, so these observations do not
establish online-FP8 causality. This is not a clean-all-tools claim.

### Explicit 1,000,000-token capacity option

The FR-Spec launcher's `MAX_TOTAL_TOKENS=1000000` option was qualified with
online FP8, RAM PLE, CPU media preprocessing and only the RTX PRO 6000 visible.
The runtime allocated the requested 1,000,000-token pool, retained 524,288-
token context and captured target-verify, draft-decode and draft-extend graphs.
Available GPU memory after graph capture was 5.21 GiB, compared with 7.52 GiB
for the standard 824,384-token online-FP8 boot.

Nine warmup/schema/tool requests passed, as did exact 64K READY and 490K
three-needle retrieval, one fixed-1,024-output C1 request, and four concurrent
fixed-1,024-output C4 streams with four requests observed running together.
All ten selected CPU-media checks also passed: ordinary/large JPEG, ten images,
two concurrent ten-image requests, static MP4 frames, and three successive
image-history turns at 208,021, 208,078 and 208,135 input tokens. Minimum
sampled free GPU memory during the media window was 1,187 MiB.

The first C1 sample measured 147.41 tok/s post-first-token; the first C4 sample
measured 303.73 tok/s aggregate. Three subsequent warmed C1 samples were
200.2128, 207.2144 and 206.8993 tok/s, for a **206.90 tok/s median**; their TTFT
values were 0.475, 0.345 and 0.338 seconds. The warmed median is close to the
earlier 207.12 tok/s standard-pool observation, but these runs are not a new
matched A/B and establish no speed benefit from the larger pool.

The option increases GPU KV capacity, not the 524,288-token served-context
limit. It was not qualified with media preprocessing on the model GPU, and the
minimum sampled headroom is not a guarantee for arbitrary inputs.

## Pennyroyal v2.5.0 — Optional NVMe PLE

Measured September 10–11, 2026, with the same qualified Flash-Next shape:
524,288-token context, 824,384 KV tokens, page size 64, FR-Spec/native NEXTN,
graphs and 32 GiB HiCache/NIXL. This comparison predates the online-FP8 arm;
do not combine their rates as if they were one matched experiment.

The external FP8 table was 51,200,245,760 bytes (47.683944702 GiB). NVMe mode
replaced that fixed pinned-table residency with bounded reader buffers and
reclaimable filesystem cache. Host `MemAvailable` was about 121–123 GiB after
NVMe workload checks versus about 67 GiB after the separate RAM comparison—a
54–56 GiB observed difference, not a claim that all of it came from PLE.

Three four-request runs used exactly 1,024 completion tokens per stream:

| PLE placement | Run 1 | Run 2 | Run 3 | Median aggregate |
|---|---:|---:|---:|---:|
| RAM | 428.90 tok/s | 286.16 tok/s | 434.92 tok/s | **428.90 tok/s** |
| NVMe | 369.73 tok/s | 259.86 tok/s | 391.95 tok/s | **369.73 tok/s** |

Aggregate throughput includes TTFT and synchronized makespan. All 24 streams
returned HTTP 200 and exactly 1,024 completion tokens. Both modes had a slower
second sample with roughly five-second TTFT. The runs occurred in separate
operational comparisons with different cache/JIT histories; they were not a
randomized or fully controlled A/B. The data does not support a universal
no-speed-cost claim for NVMe PLE.

NVMe mode passed cold 63,888-token prefill at 14,749.59 server tok/s and cold
489,903-token prefill at 8,644.77 server tok/s, with exact READY and all three
needles respectively. It also passed warmups, schema/tools, image and long
image-history requests, CUDA-graph capture, and identical-restart NIXL reuse.
Four concurrent saved 64K/490K requests restored 553,728 storage/prefetch/KV
tokens and four Mamba states. These checks establish function on the exercised
configuration; short host-wide samples do not establish that NVMe eliminates
memory compaction.

### Combined online FP8 and NVMe PLE

A later run selected both options with the standard 824,384-token pool,
524,288-token context, CPU media preprocessing and one visible RTX PRO 6000.
Post-graph free GPU memory was 7.64 GiB. Nine warmup/schema/tool requests,
64K READY, 490K three-needle retrieval, fixed-output C1/C4, and all ten selected
image/video checks passed.

The final source also passed identical-restart restoration of four saved
64K/490K requests: 553,728 storage-hit, prefetched and KV load-back tokens,
four Mamba states, and 8,068,005,952 load-back bytes. All four replies were
correct, with 9.90 seconds synchronized makespan. Logs recorded 63,872- and
489,856-token storage-prefetch completions. These are aggregate counters
across two prefixes, not a single request exceeding the context limit.

| Combined-option sample | Run 1 | Run 2 | Run 3 | Median |
|---|---:|---:|---:|---:|
| Warmed C1 post-first-token, 1,024 outputs | 172.3843 | 175.7067 | 171.5150 | **172.38 tok/s** |
| Warmed C4 synchronized aggregate, four × 1,024 outputs | 448.3009 | 442.1500 | 438.2573 | **442.15 tok/s** |

Each C4 run reached four simultaneously running requests; TTFT was about
0.67–0.71 seconds. Before those warmups, the first basic C1 sample was
123.08 tok/s post-first-token, and the first C4 aggregate was 254.35 tok/s with
roughly 7.6-second TTFT. The first-use values are retained because startup/JIT/
cache state materially affected observed throughput.

This is not a fresh controlled RAM-versus-NVMe A/B. In particular, the earlier
207.12 tok/s online-FP8/RAM-PLE C1 result was collected in a different window;
the numbers do not establish a causal speed ratio between PLE placements.

## Pennyroyal v2.4.0 — Prefill and maintenance

Measured September 9, 2026, on one RTX PRO 6000, TP1, with the same Flash-Next
ModelOpt NVFP4 checkpoint before and after the update. Both runs retained
524,288-token context, 824,384 KV tokens, page64/chunk4096, native NEXTN with
FR-Spec, and HiCache/NIXL. Requests used ordinary Chat Completions after warmups.

| Cold prompt | Before: server prefill tok/s | v2.4.0: server prefill tok/s | Observed throughput increase | Before: client TTFT | v2.4.0: client TTFT |
|---|---:|---:|---:|---:|---:|
| 63,864 tokens | 13,506 | **14,842** | **9.90%** | 5.637 s | 5.242 s |
| 489,879 tokens | 8,464 | **8,773** | **3.65%** | 62.314 s | 60.613 s |

Server initial-prefill spans were 4.728560 → 4.302790 seconds and
57.875780 → 55.836900 seconds. Percentages use unrounded values. The first
request returned exact READY; the single long prompt returned all three
separated needles. These are single cold-prefill observations at each length,
not guaranteed gains or a repeated distribution. The optimization also covers
prefill work for new suffixes, but no warm-prefill speedup percentage was measured.

The latest three 1,024-output-token decode runs measured **181.72 tok/s C1
median** and **446.49 tok/s four-request aggregate median**. C1 excludes
time to first token; aggregate uses synchronized whole-batch makespan. Decode
varied across runs/boots, and this release claims **no decode improvement**.
The v2.3 six-sample results below remain unchanged historical measurements.

Both supported profiles passed startup, prefill, single/concurrent decode,
disconnect cleanup and identical-restart NIXL restoration. Cached long-prefix
extension also completed correctly. Context and token-pool capacities were
retained. [CHANGES.md](CHANGES.md#v240--maintenance-and-faster-flash-next-prefill)
lists the adapted fixes; [LIMITATIONS.md](LIMITATIONS.md#v240-scope) bounds
these observations.

## Pennyroyal v2.3 — Flash-Next FR-Spec

Measured September 5–6, 2026, on one RTX PRO 6000 at TP1. The tests used the
same Flash-Next ModelOpt NVFP4 checkpoint, SGLang executable `836206a0ad`,
and dependencies for the baseline and FR-Spec configurations.

FR-Spec uses a 65,536-token draft vocabulary while keeping full target
verification and the existing acceptance policy. The configuration retains
524,288-token context, 824,384 KV tokens, page size 64, four concurrent
requests, 24 BF16 Mamba slots, CUDA graphs, and 32 GiB HiCache/NIXL.

Source credit: [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120).
Pennyroyal uses a separately generated token map; the numbers below are
measurements from this workstation.

### Decode throughput

Tests used the ordinary Chat Completions API after warmups, with 1,024
server-reported completion tokens per stream. Single-request decode excludes
time to first token. Four-request aggregate divides all output tokens by the
total synchronized batch duration, including time to first token.

| Configuration | Samples per metric | Single-request decode, median | Single-request whole-request rate, median | Four-request aggregate, median |
|---|---:|---:|---:|---:|
| v2.1.2 baseline, run 1 | 3 | 156.79 tok/s | 151.17 tok/s | 417.92 tok/s |
| v2.3 FR-Spec, initial measurements | 3 | 165.33 tok/s | 157.16 tok/s | 447.83 tok/s |
| v2.1.2 baseline, run 2 | 3 | 147.15 tok/s | 140.88 tok/s | 427.91 tok/s |
| **v2.3 FR-Spec** | **6** | **171.93 tok/s** | **164.40 tok/s** | **447.04 tok/s** |

The six-sample v2.3 medians are **9.7–16.8% higher for single-request decode**
and **4.5–7.0% higher for four-request aggregate** than the two baseline
medians. Initial FR-Spec measurements showed +5.4% and +7.2%, respectively,
against baseline run 1.

Tests ran sequentially, and the baseline runs were on separate boots.
The range reflects measured variation, not a randomized confidence interval
or a guaranteed speedup. Sample-level timings, per-stream rates, acceptance,
and memory observations are in the
[numeric results](https://github.com/jpezzulli/pennyroyal-validation/blob/main/results/qwen38-flash-next-frspec-20260905.json).

### Cold prefill

| Prompt | v2.1.2 speed | v2.3 FR-Spec speed | v2.1.2 time to first token | v2.3 time to first token | Result |
|---|---:|---:|---:|---:|---|
| 63,864 tokens | 12,853 tok/s | **13,070 tok/s** | 4.969 s | 4.886 s | Exact READY |
| One 489,879-token prompt | 8,022 tok/s | **7,989 tok/s** | 61.070 s | 61.319 s | All three separated needles exact |

Both tests used fresh cache namespaces. Speeds use server-reported input token
counts divided by unrounded client time to first token; displayed values are
rounded. Cold prefill was essentially unchanged by FR-Spec. These are the v2.3
qualification measurements, not a speed increase attributed to v2.3.1 maintenance.

### Functional validation

| Test | Result |
|---|---|
| Reasoning | 98.52/100; nine measured requests, all natural stops |
| Tools | 29 clean workflows and one redundant read-only call; zero runtime errors |
| Vision | Passed the spatial and claim-separation checks |
| Agent workflow | Passed the six-turn, five-tool release-note task |
| Natural decode | 3,072 tokens and token IDs returned; automatic check affected by a collector field-name mismatch |
| Long-context continuation | Two dependent turns on a restored 490K conversation passed |
| Both supported profiles | Startup, prefill, single/concurrent decode, and NIXL restart restoration passed |

Tool scoring was 27/30 literal, 29/30 exact calls/arguments, and 30/30
parseable responses. Two response-quality issues are documented separately.
The [detailed validation report](https://github.com/jpezzulli/pennyroyal-validation/blob/main/results/qwen38-flash-next-frspec-20260905.md)
contains the reasoning breakdown, tool review, and evaluator limitations.

### NIXL restoration

Restart testing restored **489,856 tokens** and recovered all three needles.
A group of four requests, comprising two copies each of the 64K and 490K
prompts, completed correctly in **9.396 seconds**. Counters showed 553,664
NIXL-hit tokens, 553,664 device-hit tokens, and four restored Mamba states.
Duplicate requests reused restored prefixes; the result does not represent
four independent full transfers from storage.

Both the Flash-Next and 27B/DFlash2 profiles retained their existing context,
graph settings, and KV capacities. No 27B throughput improvement is claimed.

## Flash-Next agentic session — September 6, 2026

An agentic-use log covered 12:04:59–12:11:32 EDT. It contains 41
completed requests, with one request running at a time in the decode samples.
For sustained-generation reporting, only responses with **at least 1,024
output tokens** are included; short replies and periodic throughput readings
that span idle or prefill time are not used in the calculation.

| Measurement | Result |
|---|---:|
| Included completed requests | 9 |
| Input context for included requests | 202,815–279,824 tokens |
| Generated output | 22,535 tokens |
| Summed server post-prefill time | 145.00612 s |
| Sustained generation rate | **155.4 tok/s** |
| Per-request median | 159.9 tok/s |

The sustained rate is total output divided by summed server post-prefill time.
It excludes initial prefill and time between requests; it is not whole-session
throughput or a controlled release comparison. This observation does not
establish a speedup from the maintenance fix.

One request separately processed **119,382 uncached input tokens in 10.38742 s**,
or **11,493 tok/s**, while reusing an 83,456-token prefix. This is additional
prefill on an existing prefix, not the cold 64K benchmark or proof of a fresh
NIXL disk restore.

## Earlier Flash-Next campaign

Source: `64ecd64924fee338e3bf846a32167cd604186827`.

This campaign used `lactd` with the workstation card's fan curve active and
the following power/clock profile:

```yaml
power_cap: 450.0
min_core_clock: 210
max_core_clock: 2750
gpu_clock_offsets:
  0: 1000
mem_clock_offsets:
  0: 2000
```

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

## Qwen3.8-27B/DFlash2 September 10 qualification

Measured September 10, 2026, during v2.4.1 maintenance qualification at source
`1048ef671ff9`. The FP8 target and DFlash2 ran on one RTX PRO 6000, TP1, with
524,288-token context, 1,118,784-token target and draft KV pools, 24 Mamba
slots, and HiCache/NIXL enabled. Requests explicitly used medium reasoning
effort. These are observations, not a new performance-improvement claim.

| Workload | Speed | Timing / result |
|---|---:|---|
| 64K cold prefill — 63,888 input tokens | **6,570 tok/s** | 9.72427 s server prefill; 10.468 s client TTFT; exact READY |
| 490K cold prefill — 489,903 input tokens | **1,646 tok/s** | 297.71330 s server prefill; 302.320 s client TTFT; all three needles found |
| Single-request decode — 1,024 output tokens | **108.31 tok/s** | Median of three post-first-token rates |
| Four simultaneous requests — 1,024 output tokens each | **374.98 tok/s aggregate** | Median of three synchronized batch makespan rates |

Each prefill row is one cold request with zero cached input tokens, matched
to its server request-time record. Rates divide authoritative input counts by
the server's initial-prefill span, not client TTFT. The 490K request contained
all three separated needles in one prompt.

C1 samples were 108.93575, 105.57627 and 108.31228 tok/s; C4 aggregate samples
were 371.35032, 374.98499 and 377.01625 tok/s. All three runs are retained.
C1 divides 1,023 tokens by the post-first-token interval; C4 divides all
4,096 completion tokens by synchronized batch makespan, including TTFT and
the slowest response. The per-stream rates are not added together. The earlier
dated 27B measurements below remain historical results, not comparison controls.

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
