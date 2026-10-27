# Qwen3.8 SGLang benchmarks on one RTX PRO 6000

This record contains local benchmarks for Qwen3.8 Flash-Next NVFP4 with
native NEXTN MTP or optional DFlash2, and Qwen3.8-27B FP8 with DFlash2, on
one NVIDIA RTX PRO 6000 Blackwell 96 GB GPU (SM120). Dated sections retain
their measured configurations. External results identify their contributors,
hardware and scope.

**Looking for other users' experience?** [Community results](COMMUNITY-RESULTS.md)
keeps credited external field reports separate from our measurements, including
week-long agentic use, Max-Q, both model profiles and multi-GPU deployments.
The older H3PO raw table remains at its existing link.

## Penny Royal 3.1.3 (October 27, 2026)

Measured on the 3.1.3 candidate source `ef3258749b` (runtime-code parent
`1a6d208276`) on one RTX PRO 6000 Blackwell (TP1), 524,288-token context, on the
two profiles this candidate carries day to day: the maintained Flash-Next
native-MTP profile and 27B FP8 with DFlash2. Means are elapsed-weighted over complete native decode
intervals, peaks are the highest eligible native sample from the same windows,
reasoning is included, and C4/C6 figures are full-batch aggregates. Rates use
the actual launched configuration, named in each row.

### Decode rates

| Profile / workload | Mean tok/s | Sampled peak tok/s | Intervals | Configuration |
|---|---:|---:|---:|---|
| Flash-Next native MTP, single request | **209.75** | **299.44** | 127 | `PENNY_MTP_POLICY=adaptive`, xhigh, 6 requests / 36 Mamba slots, 32 decimal GB HiCache |
| Flash-Next native MTP, full C4 aggregate | **510.02** | **691.71** | 188 | as above |
| Flash-Next native MTP, full C6 aggregate | **461.90** | **572.34** | 111 | as above |
| 27B DFlash2, single request, fixed W8 | **116.99** | **205.02** | 128 | xhigh, 4 requests / 24 Mamba slots, 96 decimal GB HiCache, FP32 Mamba state, 2,048-token draft window, 1,466,304-token automatically sized KV pool |
| 27B DFlash2, full C4 aggregate (accepted retry) | **404.3877** | **538.55** | 135 | as above; 159.876 eligible seconds |

The 27B C4 row is the accepted retry of the first attempt on October 27, 2026:
all four requests completed naturally, 72,007 generated tokens between them.
That run's 201.86-second client batch wall time is a separate end-to-end
measurement and is not a native decode rate. The public width-policy default is
still `tiered` (block 8 at one request, block 4 at two or more) and the
Flash-Next rows here were measured with `adaptive` selected locally, so no row
here is a matched-baseline comparison and no speed-up is claimed. The 27B rows
use the same fixed block-8 shape as the 3.1 record.

| Cold prompt time to first token | Flash-Next native MTP | 27B DFlash2 |
|---|---:|---:|
| 63,906 tokens | 4.50 s | 10.61 s |
| 489,921 tokens | 59.05 s | 296.77 s |

Client-measured first token on cold prompts, from the same launches. The dated
prefill-throughput rows in the 3.1 section below stay as recorded; these are the
request-side costs of this candidate's launches.

### Correctness, cache and behaviour on this candidate

- Integrated CPU suite: 409 tests and 404 subtests passed, with 1 optional S3
  check skipped. Each of the six fixes was also reviewed individually, and its
  patch matched the accepted contribution exactly.
- The Flash-Next native-MTP profile and the 27B DFlash2 profile each passed 4/4
  API probes, a sealed agentic run (6 turns, 5 tool calls), image checks, and
  correct 63,906-token and 489,921-token retrieval. The optional Flash-Next
  DFlash2 profile was not requalified on this candidate.
- Fresh-process restore was correct on both of those profiles: each returned
  553,792 NIXL-backed KV tokens and 4 Mamba states, the 27B profile additionally
  4,096 sliding-window attention tokens, and both restores after a restart are
  valid. The repeat replays of the same prompt answered from the device cache
  rather than four independent disk loads, so disk-path evidence is the two
  restore runs, not the replay count.
- The narrow fixed-W4 QSA path passed its real CUDA regression (production QSA
  shape, retained-prefix residues of 1, 2 and 3 tokens, partial acceptance,
  fused compression) and the 9/9 serving checks. `fixed-w4` is the operator-selected
  width policy; the public default remains `tiered`.
- Reasoning: all 36 measured requests completed naturally, with no confirmed
  loops and extended reasoning not penalized. Blind grading scored 95.36/100
  sequentially and 97.61/100 across three concurrent users on Flash-Next native
  MTP, and 98.67/100 and 99.23/100 on 27B DFlash2. These are this suite's
  scores; the 3.1-era figures in the sections below come from a different suite
  run and the two are not directly comparable.
- Tool calls: Flash-Next native MTP 30/30 parseable, 29/30 strict-matcher, 26/30
  automatic. The four automatic misses are scorer/date substring behaviour and
  mock-router behaviour, not model call failures. 27B DFlash2 30/30 scheduled,
  exact and parseable, 28/30 automatic: the two failures are the invalid-date test
  cases, where the model issued the correct `validate_date` calls and then spent
  its entire 512-token output ceiling on reasoning, so the final answer came back
  with an empty visible body; one invoice request also returned unwanted
  explanatory prose alongside a correct call. The separate agentic and vision
  checks passed.

### Recorded exceptions

- The first 27B C4 screen on October 27, 2026 stopped with an Xid 109 GPU freeze.
  The unchanged retry completed and is the accepted row. The cause is unproven and
  nothing in 3.1.3 fixes it, so no freeze fix is claimed here; the detailed
  incident note is kept internally.
- Under host-cache pressure, state was evicted rather than retained purely on the
  host. Host checkpoint reclamation and the NIXL disk-budget cleaner were both in
  play, so pure-host retention was not isolated on this candidate. Both restart
  restores above are valid.
- No NVMe PLE serving qualification was run for this release. The published
  image was separately verified on 2026-10-27: the existing-image check passed
  with GPUs hidden and network none, and a quick actual-GPU serving smoke passed
  on the Flash-Next profile with the public `tiered` default (TP1, 4 requests /
  24 slots, 524,288-token context, 32 GB HiCache with NIXL, RAM preprocessing,
  diagnostics off) - health and model discovery, one exact READY answer with a
  stop finish, and one tool-call turn returning the exact `lookup_alpha` (key
  ALPHA) and `lookup_bravo` (key BRAVO) calls as valid JSON with a `tool_calls`
  finish and no prose. Native was then restored byte-for-byte and passed the
  same two checks at `adaptive`, 6 requests / 36 slots and 4,000-token
  preprocessing. This is image-serving evidence only: not a new throughput
  qualification and not a 27B container boot. The bounded 20-second stop used
  SIGKILL, so graceful container stop is not claimed.

## Penny Royal 3.1

Penny Royal 3.1.0. Recorded October 2026
on one RTX PRO 6000 Blackwell (TP1). The maintained default Flash-Next profile
stays native NEXTN MTP with FR-Spec (.97 static memory fraction; 4 requests /
24 Mamba slots unless overridden; `next-plain` is the same without the FR-Spec
map at .981), with the verify width fixed by active concurrency - block 8 for
one running request, block 4 for two or more - from
`configs/pennyroyal/adaptive-next.json`, whose SHA-256 `3edaa87d66f535caae0bddb8f457f03fce0f6dc1f4e8091b20495a0dc9da41fb` is pinned
by the launch recipes. The DFlash2 configurations below record the optional
profile: our OrcaRouter ModelOpt NVFP4 checkpoint with the separate
`incoai/Qwen3.8-Flash-Next-DFlash2` draft, FR-Spec on the bundled
65,536-ID map, width routed by active batch size (block 6 for one request,
block 4 for two or more), draft tokens verified with FlashInfer's
SM120 kernels (`--linear-attn-verify-backend flashinfer`), 524,288-token
context, 6 requests / 36 Mamba slots, `.95` static memory fraction, automatic
KV pool sizing, RAM PLE, 32 decimal GB HiCache, and NIXL. The 27B profile
keeps its FP8 target, its own DFlash2 draft at fixed block 8, 4 requests /
24 slots, `.92`, and 96 decimal GB HiCache. TP2 is part of the optional
DFlash2 capability: it is expected to work; it has not been tested on
two-GPU hardware. Local numbers here are one inference card. DFlash2 offers
more token-pool capacity; we are still evaluating performance across
concurrency levels.

### Presentation workflow (agentic), optional profile

The original ten-slide presentation ran end to end on the final candidate at
runtime source `a60f4428f7aae4fd843cfb6f7cbda94d0aee7b82`, on the optional
DFlash2 profile with FlashInfer verification, C1 with the routed block-6 width
throughout, at the explicit Medium reasoning effort:

| Measurement | Result |
|---|---:|
| Native elapsed-weighted generation mean | **217.19 tok/s** |
| Native sampled peak | **448.70 tok/s** |
| Model turns / matched server requests | 70 / 70 |
| Complete measured decode intervals | 1,075 over 570.391832 seconds |
| Server output tokens, including reasoning | 132,873 |
| Prompt growth across the workflow | 42,527 to 206,219 tokens |
| Total workflow time, including tools and rendering | 13 minutes 44 seconds |

All turns matched the retained workflow timestamps one-to-one; no competing
requests or width switches occurred. Native speed includes reasoning and
excludes prefill, tool waits, idle, each request's first reporting interval,
and unlogged final tails. The workflow completed the normal presentation
tools, the deck rendered, and the previewed deck displayed correctly. The
earlier recorded presentation averaged 189.27 tok/s on a different
configuration; these were not matched A/B runs.

### Current decode rates

| Workload | Mean tok/s | Sampled peak tok/s | Source / configuration |
|---|---:|---:|---|
| Native MTP, single request, C1 W8 | **197.87** | **345.30** | `fe5f02c`, native MTP + FR-Spec, C1 W8 |
| Native MTP, full C4 aggregate, W4 | **555.11** | **731.06** | `fe5f02c`, concurrent W4 tier |
| Native MTP, full C6 aggregate, W4 | **472.79** | **528.49** | `fe5f02c`, concurrent W4 tier |
| Native MTP, coding workflow (agentic), C1 W8 | **252.83** | **501.63** | `fe5f02c`, medium verified every turn |
| Native MTP, presentation workflow (agentic), C1 W8 | **216.23** | **525.80** | `fe5f02c`, explicit medium reasoning |
| DFlash2 + FlashInfer, single request, C1 W6 | **241.49** | **339.53** | `a60f442`, optional profile, routed W6 |
| DFlash2 + FlashInfer, full C6 aggregate | **504.80** | **568.90** | `a60f442`, concurrent W4 tier |
| DFlash2 + FlashInfer, presentation workflow (agentic), C1 W6 | **217.19** | **448.70** | `a60f442`, explicit medium reasoning |

The normal decode screens used 128-token inputs and a 1,024-token output
ceiling, three repeats with seeds 42/43/44, xhigh reasoning, temperature 0.8,
top-p 0.95, top-k 40 and min-p 0.05. Rates include reasoning and use complete
active-decode intervals, excluding the initial idle/prefill interval and
unlogged tails. Full-C rates also exclude batch fill and drain.

The FlashInfer C1 screen contains 21 complete intervals over 10.687586
seconds; full C6 contains 17 over 20.495504 seconds.

The separate medium coding attempt on the optional DFlash2 profile measured
259.38 / 439.28 tok/s but ended unfinished. It is retained as first-pass
timing evidence, not a completed coding benchmark.

### Final native MTP screens and workflows (October 21, 2026)

Measured on the merged final source
`fe5f02ccacb50d3aea10bea2044a9816292e120f` (service invocation
`87586b6cfec84b60af6ef6de17f78715`): native NEXTN MTP with FR-Spec at the
maintained default routing (block 8 at one running request, block 4 at two or
more), `.97` static memory fraction at 6 requests / 36 Mamba slots, 524,288-
token context, automatic KV pool sizing (the boot resolved a 1,081,792-token
target pool), 32-decimal-GB HiCache with NIXL, diagnostics off, one RTX PRO
6000 with an A4000 preprocessing sidecar. The decode screens used one-wave
random 128-input / 1,024-output prompts, three repeats per concurrency at
seeds 42/43/44, normal non-greedy sampling (temperature 0.8, top-p 0.95,
top-k 40, min-p 0.05), xhigh reasoning, `ignore_eos` false, warmup 0. All
nine runs were isolated with no errors. Means are elapsed-weighted complete
active-decode intervals; peaks are the highest eligible logged rate from the
same windows; reasoning is included and each request's first
idle/prefill-crossing rate, unlogged tails, tools and waits are excluded.
Full-C means use only adjacent interval endpoints at the configured
concurrency with no intervening prefill; fill and drain are excluded and
aggregates are never divided by C.

Single-request screen: 197.87 / 345.30 tok/s over 25 intervals spanning
12.97803 seconds. Full C4 aggregate: 555.11 / 731.06 over 28 intervals
spanning 18.475742 seconds. Full C6 aggregate: 472.79 / 528.49 over 26
intervals spanning 29.326962 seconds. These supersede the earlier native C1
screen recorded at `a60f442` (227.54 / 375.90, 22 intervals over 11.171146
seconds, measured in a test profile whose concurrent tier was W6); that row
remains dated single-request evidence.

The real coding workflow (log-triage CLI, tests, offline HTML) ran at medium
reasoning verified every turn: 252.83 / 501.63 tok/s, 27 model turns matched
to 27 server requests, 38,402 server output tokens including reasoning, 256
complete C1 intervals over 133.632276 seconds, prompt growth 9,300 to 51,591
tokens, block 8 throughout, one natural completion with no correction passes.
The workflow's 16 existing tests passed independently.

The real ten-slide presentation workflow (agentic) ran through ordinary
OWUI/PPTMaster tooling at explicitly assigned medium: 216.23 / 525.80 tok/s,
80 model turns matched to 80 server requests, 133,430 server output tokens
including reasoning, 1,015 complete C1 intervals over 558.962449 seconds,
prompt growth 42,534 to 238,546 tokens, block 8 throughout. The deck delivered
10 slides with 10 notes plus PDF and preview (one preview inspection); total
workflow time including tools and rendering was 824.132 seconds (13 minutes 44
seconds), separate from decode. No other requests or runtime errors occurred.
Native width was W8 throughout both workflows.

### Earlier 3.1 decode screens on the Triton path

| Workload | Mean tok/s | Sampled peak tok/s | Source / configuration |
|---|---:|---:|---|
| Presentation workflow (agentic), C1 W6 | 194.56 | 421.35 | `a60f442`, DFlash2 + FR-Spec, Triton verify, routed W6 |
| Coding flow, C1 W6 | 235.60 | 417.50 | `5ca3f39`, FR-Spec W6 |
| Normal single request, C1 W6 | 228.44 | 336.20 | DFlash2 + FR-Spec, Triton verify, C1 W6 |
| Regular C1 W4 | 214.18 | 306.44 | `5ca3f39`, FR-Spec fixed W4 |
| Full six-request C6, aggregate | 566.14 | 657.52 | `5ca3f39`, FR-Spec W4, 15 full-C6 intervals |
| Matched Triton control, C1 W6 | 227.37 | 346.82 | `a60f442`, verifier comparison |
| Matched Triton control, full C6 W4 aggregate | 503.41 | 581.67 | `a60f442`, verifier comparison |

The earlier 566.14 full-C6 mean supersedes the drain-inclusive 522.69 summary
for that screen. It is a separate run from the matched verifier comparison.
The 235.60 / 417.50 coding-flow timing was accepted; its final handoff record
was not retained.

Before FlashInfer verification, the same optional profile and routing measured
194.56 / 421.35 on the completed five-slide xhigh presentation (121 model
turns matched one-to-one, 1,676 complete decode intervals over 909.385
seconds, 192,349 output tokens including reasoning, prompt growth 42,637 to
253,267 tokens, 33 minutes 25 seconds total, with independent arithmetic
verification) and 228.44 / 336.20 on a normal single request. The five-slide and ten-slide runs differ in slide count and
reasoning effort; neither is presented as a regression or an improvement of
the other.

### Cold prefill

| Prompt | Flash-Next tok/s | 27B tok/s |
|---|---:|---:|
| 63,906 tokens | 15,365.38 | 6,421.59 |
| 489,921 tokens | 8,715.05 | 1,677.10 |

Server initial-prefill timing on cold, uncached prompts. The Flash-Next
column was recorded on the DFlash2 profile during its qualification and
remains accurate there; no native-MTP prefill re-screen exists for this candidate.
Prefill is not speculative.

### 27B DFlash2 rates

| Workload | Mean tok/s | Sampled peak tok/s | Source / configuration |
|---|---:|---:|---|
| Regular C1 W8 | 133.06 | 232.07 | `27f9eaa`, fixed W8 |
| Full four-request C4, aggregate | 459.74 | 570.73 | `27f9eaa`, 14 full-C4 intervals |

The corrected full-C4 mean supersedes the earlier drain-inclusive 386.26
tok/s summary; a contaminated first C1 run remains excluded. The September 10
and August 24 27B tables below stay as dated history.

### Correctness and cache

Both profiles completed the recorded reasoning, tool, vision, long-context,
and RAM/NIXL restore checks on the qualified sources. Current reasoning-suite
scores are 93.46/100 (Flash-Next, recorded on the DFlash2 qualification) and
96.94/100 (27B), graded on actual
answers; extended or repetitive reasoning is not penalized, and `ask_user`
is treated as a valid tool choice. The recorded model and tool deviations
stand. The optional DFlash2 candidate boot resolved the observed
1,210,944-token shared target KV pool with automatic sizing; the maintained
native-MTP profile reports its own resolved pool size at startup. The
accumulated `.95`-fraction evidence includes 49 successful image requests.
The fixed-width routing (PR100) passed three source reviews, 139 focused
checks plus 24 subtests, pinned lint, and two live
C1→C2→C1→C6→C1 cycles with eight switches and 122
correctly routed CUDA-graph decode samples, followed by normal-answer and
tool checks; that evidence was recorded on the DFlash2 profile at `a60f442`.
The three configuration changes (PR102/PR103/PR104) were reviewed and
checked together: 231 CPU tests plus 69 subtests passed with
2 environment skips, and pinned lint passed; this is configuration
validation, not a new native hardware campaign. FlashInfer verification was
checked by the four existing SM120 RecoverSSM checks at each of W4 and W6
(accepted-prefix state output, the 63/64/65 boundary fixture, 24-cycle
continuation/prefix replay, CUDA-graph recovery) and one focused W4/B6
mixed-accept and graph-replay check, at .04 atol/rtol; live smoke passed C1
memory, C6 arithmetic, post-concurrency continuation, a tool call and a tool
result. The TP2 capture support (PR89) passed 21 focused capture
tests with pinned lint and an independent review.

[Machine-readable release record](release-3.1.json) includes the exact source
identities, settings, validation scope, and published image digest for
[Penny Royal 3.1.0](https://github.com/jpezzulli/sglang-rtxpro6000/releases/tag/pennyroyal-v3.1.0).
The 3.0 and earlier entries below remain dated history with their original
labels and timing methods.

## Penny Royal 3.0

Measured October 10, 2026, on one RTX PRO 6000 Blackwell (TP1). Flash-Next used
our OrcaRouter ModelOpt NVFP4 checkpoint, FR-Spec with adaptive NEXTN MTP, a 524,288-token
context limit and a 1,067,072-token GPU pool. RAM PLE and HiCache were enabled;
image preprocessing used an A4000 sidecar. TP2 remains a supported deployment;
these local numbers are from one inference card.

| Workload | Native mean tok/s | Sampled peak tok/s |
|---|---:|---:|
| Coding, 21 model turns / 21,193 output tokens | 262.32 | 518.18 |
| Presentation, 113 model turns / 164,934 output tokens | 239.53 | 543.72 |
| Regular C1 | 218.63 | 326.56 |
| Configured C4, aggregate, includes fill/drain | 489.35 | 636.71 |
| Configured C6, aggregate, includes fill/drain | 433.54 | 606.48 |

The regular screens used three repeats, 128 input tokens and 1,024 output
tokens per request, temperature 0.8, top-p 0.95, top-k 40, min-p 0.05 and xhigh
reasoning. Those October 10 C4/C6 rows include batch fill and drain, and full-
batch samples were not retained for them, so they are not used as full-
concurrency baselines. Use the corrected 3.0.2
full-batch reference below for that. The completed coding workflow used medium reasoning. The
presentation ran through the normal presentation tools with unchanged client
settings; its exact client overrides were not retained. These are separate
workloads, not a matched reasoning-effort comparison.

The coding task produced a log-triage CLI, tests and an HTML report; all ten
tests and its fixture command passed. The presentation produced ten editable
slides with notes. Both rates combine generation over model turns, including
reasoning; tool and rendering time are separate.

| Cold prefill | Prompt tokens | Native mean input tok/s | TTFT | Recall |
|---|---:|---:|---:|---|
| 64K | 63,921 | 17,235.90 | 4.71 s | Correct |
| 490K | 489,915 | 8,839.75 | 60.37 s | All three keys |

**Correctness and cache:** all nine reasoning scenarios completed naturally;
independent blind grading scored 95.69/100. The tool suite returned 30 parseable
responses with no runtime errors: 28 automatic passes and two confirmed
fixture-classifier false failures. Those two runs did not exercise the intended
schedule fixture. Image input passed. Cache churn restored 63,744 KV tokens
and two Mamba states from RAM, with all three facts correct and a subsequent
device-cache hit. This was RAM restoration, not a new NIXL restart test.

The 27B profile retains its dated RC3 generation/tool/image checks and previous
published performance; no new full 27B campaign or speed measurement was run
for this final Flash-Next configuration. The accepted native source is
`b48b70f3f80494d78a778fd4d26958ede343075f`; the local running checkout adds
private diagnostic instrumentation. Publication and image-build identity are
tracked separately.

[Machine-readable release record](release-3.0.json) includes exact settings,
validation scope and measured memory stages. Earlier measurements below keep
their original dates and timing methods.

### Corrected 3.0.2 full-batch reference

The published 3.0.2 current-RC concurrency baselines were recomputed on
October 21, 2026 from the retained raw samples of the same runs on source
`81f006db32978519d09302b9984016d6e1bfd48b`, invocation
`4568fcb7ce014906950e7fb4b2480214`:

| Workload | Full-batch mean tok/s | Sampled peak tok/s |
|---|---:|---:|
| Full C4 aggregate, block 4 | 532.05 | 644.96 |
| Full C6 aggregate, block 4 | 477.53 | 705.98 |

Full C4 covers 71 intervals over 47.652033 seconds; full C6 covers 63
intervals over 71.398232 seconds. The means exclude batch fill and drain and
the peaks are unchanged from the original record. Protocol and settings are as
originally recorded: the older queued 12/18-request ShareGPT protocol, 128
input / 1,024 output tokens, three repeats, normal non-greedy sampling with
xhigh, private diagnostics on. The corrected values replace the drain-inclusive
means 496.12604 and 457.50336 in
[release-3.0.2.json](release-3.0.2.json); no new tests were run.

### Recorded comparison with 2.5.3

| Recorded measurement | 2.5.3 | 3.0 | Change |
|---|---:|---:|---:|
| Regular C1 native mean | 186.79 tok/s | 218.63 tok/s | +17.0% |
| Regular C1 sampled peak | 265.71 tok/s | 326.56 tok/s | +22.9% |
| GPU token pool at C6 / 524K context | 1,039,040 | 1,067,072 | +2.7% |

The 2.5.3 C1 reference is the October 6 three-run pool with the restored
450W LACT profile, accepted source `69583da7c7`, and the same Orca checkpoint.
It used medium reasoning, temperature 1.0, top-p 0.95 and a 1,024-token output
limit. The 3.0 October 10 regular runs used xhigh, temperature 0.8, top-p 0.95,
top-k 40 and min-p 0.05. Both use native elapsed-weighted complete reporting
intervals. These are comparisons of the saved release configurations, not an
isolated measurement of one code change. The percentage describes the recorded
C1 rates; it is not a claim that every workload or concurrency improved by 17%.

### Public code-edit benchmark

| Run | Mean native tok/s | Sampled peak tok/s | Sampling |
|---|---:|---:|---|
| Accepted 3.0 adaptive screen, October 10 | **491.88** | **574.16** | Temperature 0.8, top-p 0.95, top-k 40, min-p 0.05 |
| Saved 2.5.3 eight-prompt code-edit set, October 6 | 336.57 | — | Greedy |

The 3.0 screen uses the public token-maxxing code-edit prompt, SHA-256
`3599ff10297e679d6409e686220814bcdbd3f4cf2226fda740f6f56856a38907`,
with 6,076 input tokens, a 2,400-token output cap, C1 and inherited medium
reasoning. Its native mean is elapsed-weighted over eight complete reporting
intervals; mean logged acceptance length was 6.56. This is a generation-speed
screen, separate from the completed coding workflow with its tools and tests.

It ran on the accepted attention/adaptive implementation at a 1,048,576-token
pool, before the final increase to 1,067,072. Runtime source was
`5874a958019467c44a53f4b0c194d9f137629f0a`; the final accepted core differs
only in comments/test recipe, with production AST equivalence already checked.
The 2.5.3 reference used the public eight-prompt held-out set and different
sampling, so the rows provide benchmark context rather than a matched
percentage improvement.

A mechanical edit repeats much of the supplied code, making the next tokens
easier for the draft head to predict. That explains why this score is higher
than real coding or agent workflows. We publish both so readers can compare
like workloads instead of treating every number labelled “coding” as the same
test.

## How we measure current Penny performance

Our headline **tok/s is SGLang's own reported generation throughput**: the
`gen throughput` rate in its decode logs. We report a **mean and sampled peak**
for the workload. Client elapsed time and time to first token answer different
questions and are labelled separately.

### Mean and peak

- **Mean:** use complete active-decode reporting intervals and weight each
  native rate by the time it covers: `sum(rate * interval_seconds) /
  sum(interval_seconds)`. Combine repeated runs by adding those numerators and
  denominators, not by averaging their rates.
- **Sampled peak:** the highest native reported rate in the same measured
  window. Keep the reporting cadence consistent between compared runs. This is
  a sampled interval rate, not a theoretical or instantaneous per-token maximum.
- Exclude warmup and the first interval crossing idle/prefill, plus incomplete
  intervals at measurement boundaries. Apply the same rule to both builds.
  Include all remaining active intervals; do not select only the fast part.
- Use logs from the measured service invocation and workload. A stale metrics
  gauge sampled repeatedly does not provide new throughput intervals. If the
  run has no valid native interval, report the metric as unavailable.
- Internal token/time counters can be useful, but their timing scope must match
  before they replace this calculation. For example, a `decode_moments` rate
  based on summed scheduler-step time must not silently replace the native log
  rate. Isolated CUDA/kernel timings describe the kernel, not serving throughput.

### Concurrency, sampling and agentic work

**C1** means one generating request. **C4/C6 aggregate** means native throughput
across the running batch, not that rate for every user. Record configured and
observed concurrency. A finite batch naturally fills and drains; keep those
active intervals in its mean. A separate sustained-full-batch measurement must
be named as such and use the same method on both builds.

Use the workload's normal **non-greedy sampling**, with matching temperature,
top-p/top-k/min-p and reasoning settings. Our current reasoning screens use
**xhigh**. Record the actual values, model/checkpoint, prompt or workload,
context length, output limit, cache state and runtime configuration. Keep these
comparable when calculating an improvement. A greedy reproduction or correctness
check is useful evidence, but it is not the normal-sampling performance result.

For **real coding and presentation workflows**, combine native active-decode
intervals across all model turns. Reasoning tokens are generated tokens too;
count accepted output once, not speculative proposals or reasoning a second
time. Tool execution, rendering, idle gaps and user waits do not belong in
decode time. Record task completion separately. A synthetic agent transcript
does not measure tool use, and a repetitive code-edit prompt does not stand in
for the real coding task. Label token-capped speed screens separately from
completed agentic tasks.

Report **prefill throughput** separately using native server prefill timing and
the uncached tokens it actually covers. Prompt tokens divided by client TTFT is
a different measurement. Likewise, total output divided by request or task
wall time is an end-to-end rate; do not label it native decode tok/s or compare
it directly with the headline rate.

Use the existing validation runners. Keep compact results and their measurement
settings; full responses and traces are not required for a speed result.
These definitions do not require rerunning completed qualification.

## Other timing methods in historical results

Dated tables retain their original measurements; client and wall-time rates
have not been relabelled as current native-generation rates.


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
preceding option-off reference and v2.5.0 online-FP8 arm used the same RadixArk
Flash-Next ModelOpt NVFP4 checkpoint with RAM PLE, FR-Spec, native NEXTN, 524,288-
token context, 824,384 KV tokens, page size 64, 24 Mamba slots, CUDA graphs,
and 32 GiB HiCache/NIXL. The v2.5.0 source also includes startup-only
structured-output warmup and loader-lifetime maintenance, so the table compares
the complete release states. Online FP8 changes selected projection storage
and compute. NVFP4 expert, router, FP8 PLE-table, BF16 GDN state, FP8 KV, and
FR-Spec formats remain unchanged.

### Short single-request decode

All six requests returned HTTP 200 and exactly 1,024 completion tokens. Rates
are client-observed generation after the first token, excluding TTFT.

| Precision option | Three post-first-token samples | Median | Median TTFT | Median whole-request rate |
|---|---|---:|---:|---:|
| Off | 161.465, 163.069, 155.835 tok/s | **161.465 tok/s** | 0.341 s | 153.066 tok/s |
| Online FP8 | 212.808, 193.004, 207.124 tok/s | **207.124 tok/s** | 0.428 s | 187.777 tok/s |

Observed median change was **+28.28%** post-first-token and **+22.68%** over
whole-request makespan. Median TTFT was 0.087 seconds higher. Each arm contains
three samples, which is too few to estimate a distribution.

### 128K and 490K decode

The C1 rows used exactly 1,024 output tokens. All retrieval checks found the
three expected keys in every C1 and C4 response.

| Context/test | Option off | Online FP8 | Observed change | Timing detail |
|---|---:|---:|---:|---|
| 128K C1 post-first-token | 154.703 tok/s | **195.634 tok/s** | **+26.46%** | TTFT 10.111 → 12.906 s; whole request 16.755 → 18.145 s |
| 490K C1 post-first-token | 149.135 tok/s | **172.644 tok/s** | **+15.76%** | TTFT 61.747 → 63.296 s; whole request 68.640 → 69.254 s |
| 128K C4 concurrent aggregate | 367.127 tok/s | **422.037 tok/s** | **+14.96%** | 15,725 vs 15,898 output tokens; 42.833 vs 37.670 s batch wall |
| 490K C4 concurrent aggregate | 236.678 tok/s | **329.010 tok/s** | **+39.01%** | 15,455 vs 15,127 output tokens; 65.300 vs 45.977 s batch wall |

The long C1 results are single observations. The C4 requests ended at either
`stop` or a 4,096-token cap and produced different token totals, so those rows
measure contextual aggregate throughput rather than fixed-output decode. All
four streams overlapped for 30.152 seconds at 128K and 24.850 seconds at 490K
in the online-FP8 arm. **Cold TTFT did not improve** in the matching long
samples.

### VRAM, compatibility, and functional checks

The online-FP8 boot reported **7.52 GiB available after CUDA-graph capture**;
the earlier matching option-off boot reported **3.66 GiB**. The observed
difference is about 3.86 GiB. Both arms used the same 824,384-token pool, so
this comparison measures headroom at that capacity.

Representative checks covered real SM120 FP8 kernels and graph replay,
reasoning, schema/tool use, vision, a sealed agent workflow, long-context
retrieval, and identical-restart NIXL restoration. The saved 490K request
restored/prefetched 489,984 tokens, loaded two Mamba states, and returned all
three expected keys.

The RadixArk checkpoint scored **95.75/100** in one blinded local
validation-suite review, with no fatal cap. The review did not compare
precision modes.

With online FP8, the model GPU was also selected for media preprocessing as
logical `cuda:0`, with no second GPU visible. The 824,384-token pool and graphs
were retained, and all ten selected media checks passed: ordinary/large JPEG,
ten images, two concurrent ten-image requests, static MP4 frames, and three
successive image-history turns at 208,021, 208,078 and 208,135 input tokens.
Minimum sampled free GPU memory was 1,897 MiB. Dimensions and concurrency
determine the headroom required for other shapes.

One cold synthetic request with 393,223 input tokens recovered **64/64 opaque
records exactly**, with no missing, wrong, duplicate, or unexpected keys. It
used 2,754 completion tokens, reached first token in 44.523 seconds, and
completed in 53.491 seconds. This uniform synthetic archive tests dense exact
recall, not ordinary agentic work.

A separate run of the pinned public
[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
revision also passed the 64/64 exact-recall check. Its client TTFT was 47.296
seconds and whole wall time 55.086 seconds.

The ordinary tool suite produced semantically correct/exact call lists in
29/30 workflows; one workflow added a redundant delegation and unsupported
cross-check claims. A separate quoted-markup probe had 10/10 transport success
but only 7/10 correct behavior: three of six quoted fully wrapped examples
became unwanted calls. Parser code was unchanged, so these observations do not
isolate online FP8 as the cause.

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

The option increases GPU KV capacity. Served context remains 524,288 tokens,
and the test used CPU media preprocessing. Input shape and concurrency affect
the minimum required headroom.

## Pennyroyal v2.5.0 — Optional NVMe PLE

Measured September 10–11, 2026, with the same Flash-Next shape:
524,288-token context, 824,384 KV tokens, page size 64, FR-Spec/native NEXTN,
graphs, and 32 GiB HiCache/NIXL. This comparison predates the online-FP8 arm
and uses the original precision path.

The external FP8 table was 51,200,245,760 bytes (47.683944702 GiB). NVMe mode
replaced that fixed pinned-table residency with bounded reader buffers and
reclaimable filesystem cache. Host `MemAvailable` was about 121–123 GiB after
NVMe workload checks versus about 67 GiB after the separate RAM comparison, a
54–56 GiB observed difference. Process state, cache, and host activity also
differed between snapshots.

Three four-request runs used exactly 1,024 completion tokens per stream:

| PLE placement | Run 1 | Run 2 | Run 3 | Median aggregate |
|---|---:|---:|---:|---:|
| RAM | 428.90 tok/s | 286.16 tok/s | 434.92 tok/s | **428.90 tok/s** |
| NVMe | 369.73 tok/s | 259.86 tok/s | 391.95 tok/s | **369.73 tok/s** |

Aggregate throughput includes TTFT and synchronized makespan. All 24 streams
returned HTTP 200 and exactly 1,024 completion tokens. Both modes had a slower
second sample with roughly five-second TTFT. The runs occurred in separate
operational comparisons with different cache/JIT histories rather than a
randomized, fully controlled A/B. NVMe had a lower median in this comparison.

NVMe mode passed cold 63,888-token prefill at 14,749.59 server tok/s and cold
489,903-token prefill at 8,644.77 server tok/s, with exact READY and all three
needles respectively. It also passed warmups, schema/tools, image and long
image-history requests, CUDA-graph capture, and identical-restart NIXL reuse.
Four concurrent saved 64K/490K requests restored 553,728 storage/prefetch/KV
tokens and four Mamba states. The short host-wide samples measure available
memory during this configuration; they do not isolate every source of memory
compaction.

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
489,856-token storage-prefetch completions. These aggregate counters span two
prefixes; each request remains within the context limit.

| Combined-option sample | Run 1 | Run 2 | Run 3 | Median |
|---|---:|---:|---:|---:|
| Warmed C1 post-first-token, 1,024 outputs | 172.3843 | 175.7067 | 171.5150 | **172.38 tok/s** |
| Warmed C4 synchronized aggregate, four × 1,024 outputs | 448.3009 | 442.1500 | 438.2573 | **442.15 tok/s** |

Each C4 run reached four simultaneously running requests; TTFT was about
0.67–0.71 seconds. Before those warmups, the first basic C1 sample was
123.08 tok/s post-first-token, and the first C4 aggregate was 254.35 tok/s with
roughly 7.6-second TTFT. The first-use values are retained because startup/JIT/
cache state materially affected observed throughput.

The earlier 207.12 tok/s online-FP8/RAM-PLE C1 result came from a different
measurement window. It cannot serve as the RAM arm of a matched PLE-placement
comparison.

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
