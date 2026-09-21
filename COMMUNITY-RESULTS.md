# Community results for Qwen3.8 on RTX PRO 6000

These reports cover public Pennyroyal deployments of Qwen3.8-27B/DFlash2 and
Qwen3.8 Flash-Next. They include NVIDIA RTX PRO 6000 Blackwell results,
sustained agent work, different GPU variants, both model profiles, and
multi-GPU deployments. Thank you to everyone who shared their configurations,
measurements, and problems.

**These are independent users' results, not our measurements. We have not
independently reproduced them.** Hardware, versions, workloads, and timing
methods differ. Each report keeps its own context; this is not a leaderboard
or a promise of the same performance on another system. Our own dated tests
remain in [RESULTS.md](RESULTS.md).

- [kazimirek: more than a week of agentic work on a 300 W Max-Q](#kazimirek--a-week-of-agentic-work-on-max-q)
- [WonderRico: both profiles on GSM8K and Automation Bench](#wonderrico--both-profiles-on-gsm8k-and-automation-bench)
- [StockSpecialist1707: a larger single-GPU pool and online FP8](#stockspecialist1707--single-gpu-capacity-and-online-fp8)
- [hpiguyTR: RAM versus NVMe PLE over OCuLink](#hpiguytr--ram-versus-nvme-ple-over-oculink)
- [H3PO: two-GPU Flash-Next FP8 and NVFP4](#h3po--two-gpu-flash-next)
- [AntigravityAI: Flash-Next with 62 GB host RAM](#antigravityai--flash-next-with-62-gb-host-ram)
- [untcoder2: TP2 with GPU-resident PLE](#untcoder2--tp2-with-gpu-resident-ple)

## kazimirek — a week of agentic work on Max-Q

**Reported September 12, 2026** by
[u/kazimirek](https://www.reddit.com/user/kazimirek/).
[Original field report](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9cn6h6/).

**Pennyroyal version:** v2.1.1, source `fb1216c6c`.

> Your runtime has been running a real workload for me for over a week now and it just works

**4.8 billion prompt tokens with a reported 96% prefix-cache hit rate**, alongside
**55 million generated tokens**, is the headline from this week-long workload.
The author used Claude Code through LiteLLM: one orchestrator and up to three
subagents, contexts usually **150K–236K**, with images in the mix.

The cache reuse matters as much as the decode speed: long agent conversations
repeatedly carry earlier context forward. Reusing that work avoids processing
the entire prefix from scratch on each turn. These are reported operational
totals, not 4.8 billion tokens of fresh prefill computation or a measured
cache-on/cache-off speedup.

### Reported configuration

| Item | Configuration |
|---|---|
| Hardware | One RTX PRO 6000 Blackwell Max-Q, 300 W variant; 125 GB system RAM; TP1 |
| Runtime | Pennyroyal v2.1.1, source `fb1216c6c` |
| Target | `RadixArk/Qwen3.8-Flash-Next-NVFP4` |
| Context and GPU KV | 262K context; approximately 524K GPU KV tokens; FP8 E4M3 KV |
| Host cache | 20 GB HiCache tier, reported as approximately 1.06M tokens |
| Concurrency and state | Four running requests; 40 Mamba slots |
| Prefill and speculation | 4K chunks; NEXTN MTP, three steps / four draft tokens |
| Other settings | PIL image preprocessing; expandable segments |

### Reported performance over the run

These are **medians of scheduler-log throughput samples**, grouped by running
request count. They are not medians of completed-request latency or our
synchronized fixed-output benchmark.

| Running requests | Reported median throughput |
|---|---:|
| 1 | 118 tok/s |
| 2 | 207 tok/s aggregate |
| 3 | 272 tok/s aggregate |
| 4 | 326 tok/s aggregate |

The author also reported C1 mean 125 tok/s and p90 174; C4 p90 379 tok/s;
approximately 200 tok/s aggregate overall, with peaks around 520; and MTP
acceptance around 0.42. Prefill was reported at approximately **9.8K tok/s per
chunk while decoding concurrently**, and **16K tok/s when not decoding**.
Those prefill figures are not a specified-length cold-prefill test.

**Operational notes:** the author described reliable ongoing use and reported
no server errors in the workload summary, but also described two issues they
had encountered: streaming tool-call corruption and image-preprocessing OOM
when GPU memory was full. They pointed to later maintenance work; our
[changelog](CHANGES.md) records the actual fixes and release history. We do
not turn the report into a claim of an incident-free week. The complete log,
counter-collection method, and incident timing were not supplied in the comment.
This report predates their planned v2.5 test and is not Max-Q qualification of
v2.5's online FP8 option.

## WonderRico — both profiles on GSM8K and Automation Bench

**Reported September 12, 2026** by
[u/WonderRico](https://www.reddit.com/user/WonderRico/), using the updated
Pennyroyal repository. The comments include repeated scheduler-log samples
for **27B FP8 with FP8 KV**, then **Flash-Next NVFP4 with FP8 KV and
`SGLANG_SM120_ONLINE_MXFP8` enabled**.

**Pennyroyal version:** v2.5.0. An exact source SHA was not supplied.

| Model and workload | Author's throughput summary | Range of posted C4 samples | Posted C4 acceptance length |
|---|---|---:|---:|
| [27B FP8, GSM8K](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9cv7a2/) | Up to about 800 tok/s sustained | 743–819 tok/s aggregate | 6.11–6.49 |
| [27B FP8, Automation Bench](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9cwdf9/) | About 500–600 tok/s | 491–670 tok/s aggregate | 4.95–5.82 |
| [Flash-Next, GSM8K](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9esqd5/) | About 480–580 tok/s sustained | 476–557 tok/s aggregate | 3.22–3.40 |
| [Flash-Next, Automation Bench](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9ewf4k/) | About 400–500 tok/s | 399–494 tok/s aggregate | 2.75–3.18 |

The ranges above are rounded from the posted lines with **exactly four running
requests**. Lines with three running requests are not included in those C4
ranges. The author's whole-run summary and the range of a posted excerpt are
different observations, which is why both are shown.

GSM8K used very short, one-turn prompts. Automation Bench was described as
more agentic; its log excerpts show substantially larger active token counts.
Those counts are batch totals, not per-request context lengths. The samples
show CUDA graphs active and queued work keeping the server busy. Scheduler
throughput windows are not whole-request or synchronized-batch measurements.

**Hardware inference:** the successful exact-SM120 online-FP8 run, Flash-Next
memory footprint, achieved throughput, and surrounding RTX PRO 6000 discussion
strongly indicate an RTX PRO 6000 Blackwell-class configuration. The author did
not state the exact Workstation, Server, or Max-Q variant or GPU count, so those
remain unspecified.

**Scope:** these comments also do not specify an exact target repository for
27B, source SHA, full launcher, power setting, or host RAM. No benchmark
accuracy scores or full-run exports were included, so these logs establish
reported throughput and acceptance, not reasoning/tool-quality parity or a
direct comparison with our longer-context tests. In a separate
[configuration discussion](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9c961f/),
the author preferred RAM PLE because SSD PLE's decode tradeoff was too large
for their use; no matched numeric comparison was supplied there.

## StockSpecialist1707 — single-GPU capacity and online FP8

**Reported September 12, with follow-up September 13, 2026** by
[u/StockSpecialist1707](https://www.reddit.com/user/StockSpecialist1707/).
[Original report](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9e3xao/).

**Pennyroyal version:** v2.5.0, as stated by the author.

The author used the v2.5.0 Flash-Next FR-Spec recipe with online FP8 on a
**single RTX PRO 6000 Blackwell Workstation Edition, 96 GB, TP1**, driver
595.84 and CUDA 13.2, with no user-imposed GPU power cap. Their system has two
cards, but the model was pinned to one by UUID and the second was idle for
these measurements. **This is not a TP2 result.** The full checkpoint ID and
host RAM were not stated in the comment; other settings were described as stock.

### Pool capacity

| `MAX_TOTAL_TOKENS` setting | Reported resulting pool | Reported `available_gpu_mem` | Reported 1,024-token test wall time, median of three |
|---|---:|---:|---:|
| Unset — recipe default | 824,384 | 3.66 GB | 3.86 s |
| 1,048,576 | 1,048,576 | 4.67 GB | Not reported |
| 1,310,720 | 1,114,304, clamped by server | 3.81 GB | 3.79 s |

The reported larger pool is approximately **35% above 824,384**. Served context
remained **524,288 with factor-2 YaRN**. Pool capacity is shared storage, not a
larger per-request context window or a test of filling that entire pool.
The author had not tested needles above 490K on this configuration.

**Recipe clarification:** the original comment calls the unset row automatic
sizing. In the published v2.5.0 FR-Spec recipe, unset `MAX_TOTAL_TOKENS`
actually selects the explicit **824,384 default cap**. The report's pool and
free-memory numbers are preserved, but they do not by themselves establish an
estimator defect. A clamped startup allocation also does not guarantee enough
transient memory for every later workload.
The author subsequently [acknowledged this correction](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9fvkzl/)
and returned with the sustained tests below.

### Reported optimization comparison

| Configuration label from the report | Test wall time, median of three | Reported mean server decode |
|---|---:|---:|
| v2.5 plain | 4.91 s | 212 tok/s |
| + FR-Spec | 4.44 s | 225 tok/s |
| + online FP8 | 3.86 s | 272 tok/s |

The same prompt was used for three runs of each configuration; the report also
mentions a 360 tok/s peak. These wall times and server means are not our
post-first-token metric. Complete request payloads and server usage were not
provided, so we do not derive an additional token rate from the wall times or
claim a statistically established zero decode cost for the larger pool.

The author identifies the system as BESTIA, managed with Prometeus, and
discloses that Claude drafted the write-up from their logs. They offered a
future two-card sweep; no result from that proposed sweep is included here.

### Follow-up: sustained testing at 1,114,304 KV tokens

In the [September 13 follow-up](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9i76fb/),
the author kept v2.5.0, FR-Spec, online FP8 and TP1, requesting 1,310,720 tokens
and receiving the same **1,114,304-token pool**.

| Reported workload | Reported result |
|---|---|
| 20 back-to-back generations, 4,096 output tokens | Mean 16.2 s; range 15.1–17.2 s; no timing drift |
| 40 reuses of a cached approximately 100K prefix | 6–7 s each after the first |
| 60 fresh approximately 100K prefills, shuffled to avoid prefix hits | 17–18 s each; 60/60 HTTP 200 |
| 15 rounds of four submitted approximately 100K requests | Steady-state 61–63 s per round; peak 542,080 tokens / 49% pool usage |
| 10 rounds of four submitted approximately 240K requests, 8,192 output tokens each | 93–107 s per round; 40/40 HTTP 200; peak 57% pool usage |

The author reported **zero OOMs, retractions or aborts**, with the service
remaining active throughout. During the approximately 100K concurrent test,
they reported 12–13K tok/s prefill and 405 tok/s aggregate decode with two
requests running. These are their reported timings and scheduler rates, not
our fixed-output benchmark measurements.

**What this establishes:** sustained operation under the reported workload,
beyond startup capacity and three short runs. Four submitted requests did not
necessarily mean four running together: the author observed queuing in the
240K test, and occupancy peaked at 57%, not the intended 85–90%. This does not
establish near-full-pool concurrency, retrieval correctness from HTTP status,
or headroom for arbitrary image workloads. Served context remained 524,288;
the follow-up does not supply a new needle-test result. The larger pool is an
independent tested configuration, **not a change to Pennyroyal's default**.

## hpiguyTR — RAM versus NVMe PLE over OCuLink

**Reported September 13, 2026** by
[u/hpiguyTR](https://www.reddit.com/user/hpiguyTR/).
[Original field report](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9humyi/).

**Pennyroyal version:** not explicitly stated. The report appears in the v2.5
release thread and compares RAM/NVMe paths; no exact tag or source SHA is given.

The author used an RTX PRO 6000 in an **OCuLink eGPU enclosure with four PCIe
lanes**, attached to a Minisforum MS-S1 Max: Ryzen AI Max+ 395, Radeon 8060S
iGPU and 128 GB LPDDR5x-8000. A separate image model shared host memory through
the iGPU. Both PLE paths used the same checkpoint and FP8 setting; the exact
checkpoint ID and complete serving configuration were not supplied.

The practical gain was memory: the author measured approximately **99.5 GB
with RAM PLE versus 35 GB with NVMe PLE**, using cgroup anonymous plus shared
memory rather than total `memory.current`, which also includes file cache.
That let the image model remain resident beside the LLM. These are reported
whole-runtime footprints, not a measurement of the PLE table alone or proof
that all NVMe/file-cache memory disappears.

### Reported decode comparison

Short context, 1,024 output tokens; the comment does not specify the timing
denominator or repeat count.

| Concurrent requests | RAM PLE | NVMe PLE |
|---|---:|---:|
| 1 | 222 tok/s | 224 tok/s |
| 2 | 349 tok/s aggregate | 323 tok/s aggregate |
| 3 | 460 tok/s aggregate | 454 tok/s aggregate |
| 4 | 411 tok/s aggregate | 403 tok/s aggregate |

At **128K context, one stream and 1,024 output tokens**, the author reported
176 tok/s with RAM versus 164 tok/s with NVMe, approximately 7% lower.
They explicitly could not separate the NVMe cost from the four-lane eGPU
connection. Load times were 255 s for RAM and 223 s for NVMe with warm JIT
caches; teardown was 10 s for either path.

Both paths also selected the expected tools and arguments in a synthetic
agentic prompt containing tool definitions and conversation history, without
phantom calls. This is a reported functional spot check, not a full tool suite.

**Integration note:** the author initially bypassed the launch scripts and
only set the environment option, leaving the RAM path active and encountering
an OOM. Their working setup also required the reader plugin/import path,
prepared model overlay and removal of the RAM-PLE flag. Use the documented
[NVMe PLE setup](NVME-PLE.md) rather than treating the environment variable
alone as a complete integration.

## H3PO — two-GPU Flash-Next

**Reported August 28, 2026** by
[u/H3PO](https://www.reddit.com/user/H3PO/).

**Pennyroyal version:** not specified; the August 28 branch was used, without
a release tag or SHA identified in the benchmark comments.

The deployment used **two RTX PRO 6000s, TP2/EP2, without NVLink**, FP8 KV,
and native NEXTN MTP. It was a Docker deployment built from the branch, not
evidence that the repository's inherited Dockerfile reproduces our native
qualified setup.

### FP8 target

The [FP8 report](https://www.reddit.com/r/BlackwellPerformance/comments/1w04xb7/comment/p6ek6e8/)
used `Qwen/Qwen3.8-Flash-Next-FP8` and a StackOverflow-derived coding corpus.
The `llama-benchy` invocation used 4,096 prompt tokens, 1,024 generated tokens,
generation-latency mode and three runs at each concurrency. Reported
`mem-fraction-static=0.95` produced a 3,182,848-token KV pool. The shared
configuration used 524,288 context; the much larger pool is not context proof.

| Workload | Reported total throughput |
|---|---:|
| C1 prefill, 4,096 tokens | 13,414.47 ± 82.74 tok/s |
| C1 decode | 200.93 ± 12.87 tok/s |
| C2 decode | 295.88 ± 29.07 tok/s aggregate |
| C4 decode | 330.76 ± 22.04 tok/s aggregate |
| C6 decode | 339.34 ± 11.14 tok/s aggregate |

The original wide table, including per-request rates, peaks and high-variance
concurrent-prefill rows, remains in the
[retained TP2 result](RESULTS.md#independent-tp2-fp8-validation). The `±`
notation is preserved as supplied; these are not relabeled as our medians.

### NVFP4 target follow-up

H3PO subsequently reported the same benchmark against
`RadixArk/Qwen3.8-Flash-Next-NVFP4`, with FP8 KV and MTP, cards at **600 W**,
and `mem-fraction-static=0.90`. The author reported reducing the memory
fraction after a Mamba runtime OOM on multimodal requests with the earlier
configuration, and an available KV capacity of 5,787,136 tokens.
[Original follow-up](https://www.reddit.com/r/BlackwellPerformance/comments/1w04xb7/comment/p6go1j3/).

| Workload | Reported total throughput |
|---|---:|
| C1 prefill, 4,096 tokens | 13,576.99 ± 819.88 tok/s |
| C1 decode | 204.10 ± 10.60 tok/s |
| C2 decode | 338.99 ± 5.68 tok/s aggregate |
| C4 decode | 386.74 ± 41.13 tok/s aggregate |
| C8 decode | 427.46 ± 24.02 tok/s aggregate |
| C11 decode | 459.94 ± 13.08 tok/s aggregate |

**Configuration lessons and limits:** removing the optional
`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` setting allowed native MTP to work.
A separate custom-allreduce graph-capture failure had been resolved by
removing `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` from this TP2 setup.
The [diagnostic exchange](https://www.reddit.com/r/BlackwellPerformance/comments/1w04xb7/comment/p6dtvoh/)
keeps those two problems distinct. Do not generalize that allocator setting
to our qualified TP1 recipe. Neither table supplies MTP acceptance or a
full reasoning/tool/media qualification; high-concurrency prefill was noisy.
This is useful independent multi-GPU evidence, not a controlled comparison
with our TP1 results or a claim that all TP2/TP4 combinations have been tested.

## AntigravityAI — Flash-Next with 62 GB host RAM

**Reported September 10, 2026** by [Eddy / AntigravityAI](https://github.com/AntigravityAI).
[Original report (#2)](https://github.com/jpezzulli/sglang-rtxpro6000/issues/2) ·
[Campaign and follow-up](https://github.com/AntigravityAI/Qwen3.8-Flash-Next-on-RTX-PRO-6000).

**Pennyroyal version:** v2.3.0, source `836206a0ad`, with the author's changes.

One **RTX PRO 6000 96 GB with about 62 GB usable host RAM** drove this work:
the full RAM-PLE plus 32 GB HiCache recipe did not fit. The author used an
SSD-Stream split checkpoint with a roughly 51 GB PLE table on NVMe and two
320 MiB staging buffers, then also explored a quantized, pinned-RAM PLE path.

Across the campaign, the reported KV pool grew from **374,208 to 1,111,168
tokens**, with native NEXTN MTP, FP8 KV and a configured **1,048,576-token
context** using factor-4 YaRN. This combined allocator, loading, memory-accounting,
precision and Mamba-capacity changes. The largest individual reported jump was
528,640 KV tokens after collecting temporary loading tensors before pool sizing.

The author passed a **786,432-input-token cold long-context test** on an earlier
configuration and reported **99.99% prefix reuse** on a follow-up. The final
1.11M-pool configuration had boot and short warmup checks; those results should
not be read as a completed 1M-input test. Their later capacity work also tested
seven images back-to-back and showed why increasing the pool further could
leave too little room for a second image.

This is useful low-RAM and capacity-tuning work, especially the NVMe PLE path.
The settings and gains belong to the author's modified configuration, rather
than a drop-in change to Pennyroyal's defaults.

## untcoder2 — TP2 with GPU-resident PLE

**Reported September 16, 2026** by [untcoder2](https://github.com/untcoder2).
[Original report (#9)](https://github.com/jpezzulli/sglang-rtxpro6000/issues/9) ·
[Full write-up](https://github.com/untcoder2/qwen38-flash-next-nvfp4-sm120-tp2).

**Pennyroyal version:** v2.5, as identified by the author; no exact source SHA
specified in the report.

The deployment used **two RTX PRO 6000 Blackwell Workstation cards without
NVLink**, 90 GB host RAM, `next-plain`, online MXFP8, and an NVFP4 checkpoint
with a transplanted BF16 MTP block. PLE stayed in GPU memory. The configured
context was **786,432 tokens**, with a **2,254,464-token KV pool**, 48 Mamba
slots and an eight-request limit.

### Reported single-request decode

| Configuration | Median tok/s | p90 tok/s | Acceptance length |
|---|---:|---:|---:|
| One card, `next-plain`, temperature 0 | 259.1 | 298.9 | 3.06 |
| Two cards, temperature 0 | **308.3** | 339.1 | 3.01 |
| Two cards, temperature 1.0 / top-p 0.95 / top-k 20 | 274.7 | 313.4 | 2.91 |

These are scheduler decode samples from **one running request**, after warmup
and four 1,000-token generations. Samples below 20 tok/s were excluded as
batch tails. They are not HTTP completion rates or eight-request aggregate
measurements. The author reports **20/20** on a small verifiable-answer test
set. [Measurement details](https://github.com/untcoder2/qwen38-flash-next-nvfp4-sm120-tp2/blob/main/docs/MEASUREMENTS.md).

**Setup findings:** this host needed `--disable-custom-all-reduce` to pass
startup, `--no-ple-offload-embedding` for GPU-resident PLE, a reduced static
memory fraction of 0.88, and 48 Mamba slots to admit eight requests. NIXL was
disabled after backend initialization failed, so this is not a persistence
result. Concurrent throughput and long-uptime stability of the new TP2 setup
were not yet measured.

The author also found that full-vocabulary drafting beat the bundled FR-Spec
map on their tested Russian and English prompts. Changing from froggeric-v22.5
to the checkpoint template barely changed speed but changed tool formatting.
Both are useful reminders to measure the actual language and agent workload.

## Share a field report

Reports of ordinary use are welcome alongside benchmarks. Include your
Pennyroyal version/source, target and draft models, hardware and power settings,
context and pool sizes, concurrency, relevant launcher changes, and how you
measured the result. Say whether rates are per request or aggregate, and
whether they are medians, peaks, scheduler samples, or completed-request rates.
Include problems and tradeoffs as well as successes. Remove credentials,
private prompts and personal data before sharing logs through
[GitHub issues](https://github.com/jpezzulli/sglang-rtxpro6000/issues) or a public
discussion. A public report link and your preferred attribution make it easier
to credit your work accurately.
