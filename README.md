# SGLang for Qwen3.8 on one NVIDIA RTX PRO 6000 Blackwell (SM120)

> Pennyroyal is an optimized SGLang-derived runtime with qualified single-GPU
> launch configurations for Qwen3.8-27B FP8 with DFlash2 and Qwen3.8
> Flash-Next NVFP4 with native NEXTN and FR-Spec.

This repository contains the complete SGLang-derived source used on one
NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB, SM120, TP=1). Both
model configurations run from the same patched source.

**Get running:** [Use the prebuilt container](docker/pennyroyal/README.md)
for the shortest setup, or [build and install natively](BUILD.md).
Both paths support the two model profiles below.
An optional [terminal setup utility **(beta)**](CONFIGURE.md) walks through your model,
GPU, and cache settings; manual configuration remains available.

Using an AI assistant to set this up? Give it [llms.txt](llms.txt)
for the short setup map and links to the detailed instructions.

**Evidence:** [Benchmarks and measurement definitions](RESULTS.md) ·
[Community-reported results](COMMUNITY-RESULTS.md) ·
[Validation suite and published reports](https://github.com/jpezzulli/pennyroyal-validation).

## Prebuilt Docker image for Qwen3.8 on RTX PRO 6000

**Pennyroyal has a prebuilt Docker image** for both
Qwen3.8 Flash-Next NVFP4/NEXTN and Qwen3.8-27B FP8/DFlash2, with HiCache and
NIXL. No local SGLang build is required; model files and caches stay in mounted
host directories. Native installation remains supported.

**[Docker and Compose setup](docker/pennyroyal/README.md)** ·
Image: `ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.2` ·
[Container build status](https://github.com/jpezzulli/sglang-rtxpro6000/actions/workflows/pennyroyal-container.yml).
Images build automatically when a release is published. Model weights are
downloaded separately.

<a id="models-and-launch-recipes"></a>

## Qualified model profiles and launch recipes

The two profiles below are equally supported with **524,288-token context,
HiCache and NIXL** on one RTX PRO 6000. [RESULTS.md](RESULTS.md) identifies the
exact checkpoints used for each benchmark campaign.

| Profile | Precision | Speculative decoding | Reference models | Run |
|---|---|---|---|---|
| **Qwen3.8 Flash-Next** | NVFP4 target with FP8 KV | Native NEXTN MTP with FR-Spec | [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4); no separate draft | [Recommended FR-Spec recipe](RUN.md#launch-flash-next-with-fr-spec) or [non-FR alternative](RUN.md#launch-flash-next-without-fr-spec-alternative) |
| **Qwen3.8-27B** | FP8 target and KV | DFlash2 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) with [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) | [27B/DFlash2 recipe](RUN.md#launch-27b-with-dflash2) |

For Flash-Next, our [OrcaRouter Uncensored ModelOpt NVFP4 conversion](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4)
is also available by community request and works with the same Pennyroyal launch recipe.

**Swift 27B FP8 is preliminarily validated** with DFlash2: it passed reasoning
validation and used fewer tokens overall in our comparison.
See [Swift 27B validation and per-case timings](SWIFT-27B.md).

[Online FP8](FP8.md) and [NVMe-backed PLE](NVME-PLE.md) are independent
Flash-Next options.

The retained public 27B measurements used
[orcarouter/Qwen3.8-27B-Uncensored-FP8](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8),
an uncensored/abliterated derivative of the official checkpoint. That
provenance is material to behavioral results. Compatibility still depends on
matching architecture, quantization, tokenizer, and speculative-decoding
requirements.

FR-Spec credit goes directly to Gabriel's
[`gabrielolympie/sglang-flashnext-sm120`](https://github.com/gabrielolympie/sglang-flashnext-sm120).
Both recipes pin the unmodified
[Froggeric v22.5 template](configs/pennyroyal/templates/README.md).

The v2.5.0 online-FP8 implementation is directly inspired by
[`mratsim/sglang-qwen38fn-sm120-turbo`](https://github.com/mratsim/sglang-qwen38fn-sm120-turbo/tree/94a68214b77514bc26ef78cee4c01f128162d09b)
at commit `94a68214b77514bc26ef78cee4c01f128162d09b`, especially patches
`0003`, `0007`, and `0008`. Optional NVMe PLE adapts Garner McCloud's
[SSD Stream v0.2.0](https://github.com/garnermccloud/sglang-ssd-stream/tree/176a522ef9d6dbb5056ae1f467fe49af0f1258a5),
with the upstream license and NOTICE retained. Detailed boundaries and credit
are in [FP8.md](FP8.md#credit) and [NVME-PLE.md](NVME-PLE.md#source-and-credit).

<a id="current-release--pennyroyal-v251"></a>

## Current release — Pennyroyal v2.5.2

**v2.5.2 brings lower memory overhead, cache fixes, and easier setup.**

- Flash-Next uses less temporary GPU memory during loading and long prefills,
  and avoids oversized pinned-RAM allocations for its PLE table.
- HiCache reclaims only the space it needs. Choose your HiCache RAM size and
  set a NIXL disk-cache budget, including smaller caches on shared systems.
- The optional **[beta configurator](CONFIGURE.md)** provides numbered
  choices, explanations, and saved settings for native and
  [container installs](docker/pennyroyal/README.md).
- TP2 work adds shared FR-Spec weights, separate NIXL namespaces and startup
  guidance. **TP2 verification is pending from
  [u/StockSpecialist1707](https://www.reddit.com/user/StockSpecialist1707/).**

The existing model profiles, C6 option and performance tables remain.
See [changes and upstream credits](CHANGES.md#v252--memory-cache-and-setup-maintenance)
or [get started](docker/pennyroyal/README.md).

<a id="current-release--pennyroyal-v250"></a>

## v2.5.0 options — online FP8 and NVMe PLE

**v2.5.0** adds two optional Flash-Next capabilities. Existing precision and
RAM-backed PLE remain the defaults:

- **Online FP8** converts selected otherwise-BF16 projections, HyperConnection
  mix weights and the output head during model load. The measured C1 decode
  improvement was **15.8–28.3%**, with about **3.86 GiB more post-graph
  available VRAM**. Enable it with `SGLANG_SM120_ONLINE_MXFP8=true`.
- **NVMe-backed PLE** streams the approximately **47.68 GiB** FP8 PLE table
  from a prepared local SSD snapshot. Select it with
  `PENNY_PLE_BACKEND=nvme` to remove the table's fixed pinned-RAM residency.

Both options retain the qualified 524,288-token context and 824,384-token
Flash-Next pool in their tested configurations. Online FP8 and NVMe PLE are
independent choices: either, both, or neither can be selected. Read
[Online FP8](FP8.md), [NVMe PLE](NVME-PLE.md), and
[v2.5.0 changes and credits](CHANGES.md#v250--optional-online-fp8-and-nvme-ple)
before enabling them.

The FR-Spec recipe keeps 824,384 as its default. An explicit
`MAX_TOTAL_TOKENS=1000000` option was also qualified with online FP8, RAM PLE,
CPU media preprocessing and one visible GPU. It increases KV capacity; served
context remains 524,288 tokens, and no speed comparison was run.

This release also performs one bounded structured-output warmup before the API
becomes ready and releases a temporary Flash-Next PLE loader reference at the
end of model loading.

Media preprocessing still defaults to CPU. The public recipes also accept
`cuda:0` on the model GPU or `cuda:1` on a second visible GPU. The v2.5.0
single-GPU `cuda:0` check retained the 824,384-token pool and passed all ten
selected image/video scenarios with online FP8; see
[RUN.md](RUN.md#cpu-model-gpu-or-secondary-gpu-media-preprocessing) for the
headroom tradeoff.

Earlier v2.4.1/v2.4.0 work covered grammar, streaming logprobs, metadata
transfer, prefill, cancellation, reasoning effort and tool markup. See
[CHANGES.md](CHANGES.md) for release history and [RESULTS.md](RESULTS.md) for
the dated measurements.

### Flash-Next v2.5.0 online-FP8 performance at a glance

**Beyond our machine:** [Community results](COMMUNITY-RESULTS.md) collects
reports from independent users, including more than a week of Max-Q agentic use:
**4.8 billion prompt tokens with a reported 96% prefix-cache hit rate**, plus
**55 million generated tokens**. It also covers both-model benchmarks and TP2.
Special thanks to [u/StockSpecialist1707](COMMUNITY-RESULTS.md#stockspecialist1707--single-gpu-capacity-and-online-fp8)
for independently stress-testing v2.5 with a **1.11M-token KV pool** and sharing
the workload results and limits, not just a successful boot.
These are contributor-reported observations, not our own qualification.

Measured September 11, 2026, on one RTX PRO 6000, TP1, with FR-Spec,
524,288-token context, 824,384 KV tokens, RAM-backed PLE and HiCache/NIXL
retained. Online FP8 was compared with a fresh immediately preceding option-off
reference using the same serving shape and RadixArk ModelOpt NVFP4 checkpoint.
The v2.5.0 arm also includes startup-only grammar warmup and loader-lifetime
maintenance. Online FP8 changes precision; the table compares the complete
source states.

| Single-request workload | Option off | Online FP8 | Observed change |
|---|---:|---:|---:|
| Short, 1,024 output tokens | 161.47 tok/s | **207.12 tok/s** | **+28.3%** |
| 128K context, 1,024 output tokens | 154.70 tok/s | **195.63 tok/s** | **+26.5%** |
| 490K context, 1,024 output tokens | 149.14 tok/s | **172.64 tok/s** | **+15.8%** |

Rates are client-observed post-first-token decode. The short values are
three-run medians; 128K and 490K are single observations. Cold TTFT did not
improve in the long samples. Online FP8 also left 7.52 GiB available after
graph capture versus 3.66 GiB in the earlier matching boot—about 3.86 GiB more,
not 7 GiB newly added capacity. Full precision boundaries and caveats are in
[FP8.md](FP8.md); detailed timings are in
[RESULTS.md](RESULTS.md#pennyroyal-v250--optional-online-fp8).

The RadixArk checkpoint was used for the full performance and quality campaign,
including a focused **64/64 exact** opaque-record recall request. See
[FP8.md](FP8.md#qualified-scope) for the coverage and format requirements.

## Retained v2.4.0 Flash-Next prefill measurements

Measured September 9, 2026, with **v2.4.0 and FR-Spec on one RTX PRO 6000, TP1**,
keeping 524,288-token context, 824,384 KV tokens, and HiCache/NIXL enabled.
The table records that configuration and date; speeds vary with workload and
system state.

| Workload | Speed | Timing / result |
|---|---:|---|
| 64K cold prefill — 63,864 input tokens | **14,842 tok/s** | Server prefill rate; 5.242 s to first token; exact READY |
| 490K cold prefill — 489,879 input tokens | **8,773 tok/s** | Server prefill rate; 60.613 s to first token; all three needles found |
| Single-request decode — 1,024 output tokens | **181.72 tok/s** | Median of three runs; excludes time to first token |
| Four simultaneous requests — 1,024 output tokens each | **446.49 tok/s aggregate** | Median of three runs; includes time to first token and the slowest response |

Prefill rates divide input tokens by the server's initial-prefill time; client
time to first token includes other request processing. The 4–10% gain compares
that same server timing window before and after v2.4.0. Older tables use TTFT.
Each cold-prefill row is one observation, and warm-prefill speed was not
measured separately. Decode varies with workload and acceptance. The
four-request figure is aggregate throughput across the group.
See [RESULTS.md](RESULTS.md#pennyroyal-v240--prefill-and-maintenance) for the
comparison; earlier v2.3 measurements remain there as dated history.

**In real agentic use:** a session on September 6 sustained
**155 tok/s** across nine responses of at least 1,024 output tokens, with
roughly **203K–280K input context**. Those responses generated 22,535 tokens;
the rate excludes prefill, time between requests, and short replies.
[Session details and method](RESULTS.md#flash-next-agentic-session--september-6-2026).

## 27B FP8 / DFlash2 performance at a glance

Measured September 10, 2026, during **v2.4.1 qualification on one RTX PRO 6000,
TP1**, keeping 524,288-token context, 1,118,784-token target and draft KV pools,
and HiCache/NIXL enabled. Speeds describe that workload and system state.

| Workload | Speed | Timing / result |
|---|---:|---|
| 64K cold prefill — 63,888 input tokens | **6,570 tok/s** | Server prefill rate; 10.468 s to first token; exact READY |
| 490K cold prefill — 489,903 input tokens | **1,646 tok/s** | Server prefill rate; 302.320 s to first token; all three needles found |
| Single-request decode — 1,024 output tokens | **108.31 tok/s** | Median of three runs; excludes time to first token |
| Four simultaneous requests — 1,024 output tokens each | **374.98 tok/s aggregate** | Median of three runs; includes time to first token and the slowest response |

Prefill uses the server's initial-prefill time, with one cold observation at
each length. Decode values are three-run medians. This campaign measured the
v2.4.1 configuration without an earlier-release comparison.
[Results and timing details](RESULTS.md#qwen38-27bdflash2-september-10-qualification).

## Broader model support without HiCache/NIXL

**Without HiCache and NIXL, many more SGLang-supported models and speculative
configurations can run and may benefit from applicable performance
optimizations.** Those speedups come from the applicable runtime paths;
HiCache/NIXL controls prefix persistence.

HiCache/NIXL requires model- and speculative-method-specific cache layouts
and complete state-restoration support. Different MTP or draft designs can
require additional integration to save and restore target KV, draft KV, and
recurrent or other model-specific state together.

The two recipes above are Pennyroyal's ready-to-run profiles. Other models use
the normal SGLang target/draft configuration, kernels, memory sizing, and
launch settings.

See [RUN.md](RUN.md) for the launchers and the
[configuration matrix](#qualified-configuration-matrix) for dtypes, context,
cache capacity, and backends.

Flash-Next support was built at model release and tested on this machine across
524K context, multimodal input, reasoning, tools, agentic workloads, CUDA-graph
recovery, and persistent prefix restoration. Hardware- and model-specific
paths are documented in [BACKENDS.md](BACKENDS.md).

## Hardware, storage, and first-start expectations

The qualified launchers use a 96 GB GPU and substantial host RAM and disk.
Flash-Next uses a 47.68 GiB PLE table in addition to its
default 32 GB HiCache tier; RAM placement is the default and the optional
[NVMe PLE mode](NVME-PLE.md) trades lower fixed host residency for SSD I/O.
The 27B recipe defaults to a 96 GB HiCache tier. Both sizes are adjustable.
Process, filesystem, driver
and page-cache overhead add to those configured values. See
[RUN.md](RUN.md#host-memory-and-first-start) for operating guidance.

Allow disk space for the target checkpoint, the 27B draft when applicable,
compiler/JIT caches, and persistent NIXL namespaces. NIXL cleaner percentages
apply to the whole selected filesystem. Representation-specific namespaces
remain until they are removed by the operator or storage policy.

The first launch can appear quiet while checkpoint identities are hashed, then
spend substantial time compiling kernels and capturing CUDA graphs. Wait for
the server-ready log and verify the API; a cold namespace will not restore an
older prefix. Choose the [native build](BUILD.md) or the dedicated
[Pennyroyal Docker image](docker/pennyroyal/README.md). The separate Docker
material inherited from upstream is not the qualified Pennyroyal recipe.

## v2.3 — Faster Flash-Next decoding with FR-Spec

Pennyroyal v2.3 adds FR-Spec to Flash-Next, improving decode throughput while
keeping **524,288-token context, 824,384 KV tokens, CUDA graphs, and
HiCache/NIXL persistence**. Qwen3.8-27B with DFlash2 remains supported and
unchanged.

Thanks to Gabriel's
[`gabrielolympie/sglang-flashnext-sm120`](https://github.com/gabrielolympie/sglang-flashnext-sm120)
for the reduced draft-vocabulary optimization used here. Pennyroyal uses
SGLang's FR-Spec support with a 65,536-token draft map generated for this
release. The draft model scores fewer tokens; the target model keeps its
full vocabulary, verification, and acceptance policy.

Against two v2.1.2 baseline runs on the same Flash-Next checkpoint, v2.3
measured **9.7–16.8% faster single-request decode** and **4.5–7.0% higher
four-request aggregate throughput**. Each request generated 1,024 tokens;
the comparison uses medians from six v2.3 samples and three samples per
baseline run. The baselines ran on separate boots, so the range includes
run-to-run variation. Cold 64K/490K prefill was essentially unchanged.

The [retained v2.4.0 table above](#retained-v240-flash-next-prefill-measurements)
gives the later observations. [RESULTS.md](RESULTS.md#pennyroyal-v23--flash-next-fr-spec)
preserves the v2.3 baseline comparison and measurement definitions.

Reasoning, tools, vision, agent workflows, long-context continuation, and
NIXL restart restoration were tested. Detailed scores and evaluator
limitations are in the
[validation report](https://github.com/jpezzulli/pennyroyal-validation/blob/main/results/qwen38-flash-next-frspec-20260905.md).

v2.3 uses the same executable and dependencies as v2.1.2.
Start with the [FR-Spec launch guide](RUN.md#launch-flash-next-with-fr-spec);
the non-FR Flash-Next launcher remains available.

## Previous v2.1.2 — maintenance release

v2.1.2 added two correctness fixes for both supported profiles—27B
FP8/DFlash2 and Flash-Next NVFP4/native MTP:

- [#37962](https://github.com/sgl-project/sglang/pull/37962): skip token-ID
  synchronization when the selected group has only one GPU. This avoids an
  unnecessary NCCL operation and its possible late GPU-memory allocation,
  including during grammar-constrained JSON output.
- [#37408](https://github.com/sgl-project/sglang/pull/37408): keep separator
  whitespace between streamed Qwen3 Coder tool calls from overtaking pending
  JSON arguments. Genuine prose and its word spaces are preserved.

That update kept launch settings and dependencies unchanged. Test coverage is in
[CHANGES.md](CHANGES.md#version-212-maintenance-only).

## Validation suite

The canonical test suite used to qualify these runtimes—including reasoning,
tool calling, long-context needles, vision, and concurrent decode—is maintained
in [`jpezzulli/pennyroyal-validation`](https://github.com/jpezzulli/pennyroyal-validation).
This repository contains the runtime source, configuration, and measured
results; `pennyroyal-validation` contains the reusable tests and result catalog.

## Why this runtime exists

Pennyroyal changes the runtime in these areas:

- DFlash2 integration for Qwen3.8-27B, with the selector inside the draft CUDA
  graph, fused draft KV materialization, and fixed-width TRTLLM-MHA/XQA target
  verification on SM120.
- Native NEXTN MTP for Flash-Next with QSA index sharing, recovery graphs, and
  `gdn_mtp_cache_mode=none`.
- FlashInfer GDN decode and prefill for Flash-Next, plus an SM120-only
  FlashInfer WY output/recovery path for RecoverSSM.
- QSA sparse attention: Triton sparse-GQA prefill, the FlashInfer QSA wrapper
  resolving to XQA on SM120, and `sgl-kernel` top-k.
- FlashInfer CUTLASS MoE for the Flash-Next target and native-MTP layer.
- Optional exact-SM120 online FP8 for otherwise-BF16 Flash-Next projections,
  with explicit precision boundaries and fail-loud validation.
- Optional NVMe-backed PLE for operators who prefer lower fixed host-RAM
  residency and accept the measured storage-performance tradeoff.
- Correct three-axis multimodal mRoPE in the fused QK RMSNorm+RoPE kernel.
- Target, draft, verify, prefill, and accepted-state recovery CUDA graphs.
- FP8 KV and 524,288-token factor-2 YaRN for both launch configurations.
- HiCache/NIXL persistence for complete hybrid state—not KV alone—including
  DFlash2 side state or Flash-Next GDN, Qwen4 PLE, and compressed QSA state.
- Restart restoration, representation-specific FILE namespaces, and radix-
  prefix reuse followed by long continuation.
- Measured reasoning, tool-calling, vision, long-context needle, controlled
  decode, and real agentic behavior.

## Building on Pennyroyal

I'm thrilled to see people run Pennyroyal, adapt parts of it, or use it as the
starting point for their own runtime, deployment guide, benchmark, or
optimization. That is why the complete source, recipes, evidence, and known
limits are public.

If you publish work that uses or builds on Pennyroyal, please link the
canonical repository and identify the release tag or commit you started from:

- [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000)
- Current release: [`pennyroyal-v2.5.2`](https://github.com/jpezzulli/sglang-rtxpro6000/releases/tag/pennyroyal-v2.5.2)

A short note distinguishing the Pennyroyal source or technique from your own
changes helps readers reproduce the lineage, understand what you improved, and
find both projects.

## Published identity

The former `jpezzulli/qwen38-dflash2-pro6000` URL redirects here. It was the
earlier name of this repository and uses the same source history.

| Item | Value |
|---|---|
| Canonical branch | `pennyroyal-main-sm120-final` |
| Current executable source | Recorded in [PROVENANCE.md](PROVENANCE.md#source-identity) |
| Current release | **v2.5.2** |
| Git tag | `pennyroyal-v2.5.2` |
| Initial unified dated tag | `sglang-rtxpro6000-20260827` |
| Earlier 27B dated tag | `qwen38-dflash2-pro6000-20260824` |
| Upstream integration base | `e7e78940168f3ba65c762a6f82fd8bc5b6ee04e3` |
| Qualification dependency base | `0.5.19.dev492+g836206a0a`; exact updated Python/JIT source in [PROVENANCE.md](PROVENANCE.md) |
| Python / PyTorch | `3.12.13` / `2.13.0+cu130` |
| CUDA / compiler | CUDA `13.3` (NVCC `13.3.73`) / GCC `15.3.1` |
| FlashInfer / NIXL | `0.6.17` / `1.4.0` |
| GPU / driver | RTX PRO 6000 96 GB, SM120 / `610.57.04` |

Install v2.5.2 from the updated source using [BUILD.md](BUILD.md). Older wheels
do not contain these changes. [PROVENANCE.md](PROVENANCE.md) records the exact
source identity; [CHANGES.md](CHANGES.md) records release and upstream history.

## Qualified configuration matrix

The matrix separates representation, datatype, backend, and capacity. “BF16
runtime dtype” covers unquantized tensors; quantized kernels retain the formats
listed in their own rows.

| Property | Qwen3.8-27B + DFlash2 | Qwen3.8 Flash-Next |
|---|---|---|
| Target checkpoint | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) recipe reference; retained measurements used the [orcarouter derivative](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8) | [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) |
| Target weight format | block FP8 E4M3, 128x128 blocks | ModelOpt NVFP4, group size 16 on selected Linear modules |
| Quantized-path activations | dynamic FP8 E4M3 | NVFP4 input activations on selected Linear modules |
| Runtime dtype for unquantized tensors | BF16; includes excluded layers and BF16 `lm_head` | Default: BF16 for otherwise-unquantized tensors. Optional online FP8 converts eligible transformer linears, HC mix weights and `lm_head`; BF16 GDN state and the existing NVFP4 expert, router and FP8 PLE-table formats remain unchanged. |
| Target KV datatype | FP8 E4M3 | FP8 E4M3 |
| Speculative KV datatype | DFlash2 draft: FP8 E4M3 | native-MTP: FP8 E4M3 |
| Recurrent/GDN SSM state | FP32 | BF16 |
| Convolution state | BF16 | BF16 |
| MoE backend | Not applicable: dense FP8 feed-forward layers | FlashInfer CUTLASS for target and native MTP |
| Target attention | FlashInfer prefill; TRTLLM-MHA/XQA decode and fixed-width verify | QSA Triton sparse prefill; FlashInfer QSA wrapper resolving to XQA for sparse decode; general attention FlashInfer |
| Linear/GDN attention | Triton decode, prefill, and state-writing verify | FlashInfer decode/prefill; WY output-only verify/recovery in `none` mode |
| Speculative backend | DFlash2, 8 draft tokens, 2,048-token window | native NEXTN, 3 steps, top-k 1, 4 draft tokens |
| Draft vocabulary | unchanged DFlash2 | 65,536-ID FR-Spec map introduced in v2.3; full target vocabulary unchanged |
| Served context | 524,288, factor-2 YaRN target and draft | 524,288, factor-2 YaRN |
| KV page size | 64 | 64 |
| Recipe default GPU KV capacity | 1,118,784 target and draft tokens | 824,384 target and native-MTP tokens |
| Mamba capacity | 24 slots; maximum 5 retained states/path | 24 slots by default; optional C6 uses 36; `extra_buffer`, tracking interval 64 |
| HiCache/NIXL | 96 GB configured host tier; target KV + Mamba/GDN + DFlash2 state | 32 GB configured host tier; packed target/native-MTP KV + GDN + PLE + QSA keys. PLE table placement is RAM by default or optional NVMe. |
| Maximum active requests | 4 | 4 by default; [optional 6](RUN.md#optional-six-request-flash-next-profile) |

The current 27B launcher uses the 24-slot/five-state setting qualified on
2026-08-26. The 2026-08-24 performance release used 16 slots, a three-state
path cap, and a 1,194,496-token KV pool. [RESULTS.md](RESULTS.md) keeps the two
campaigns and their allocations separate.

## Resolved backends: Qwen3.8-27B/DFlash2

| Component or phase | Resolved implementation | How selected or enabled | Evidence |
|---|---|---|---|
| Target prefill attention | FlashInfer | explicit hybrid prefill backend | 64K and 490K controlled prefill |
| Target decode attention | TRTLLM-MHA with XQA | explicit decode backend; SM120 supported path | controlled and agentic decode |
| Fixed-width target verification | TRTLLM-MHA/XQA with packed causal mask | local `0f159cd545` | graph-metadata regression; 1x/4x decode |
| Linear/GDN decode, prefill, verify | Triton | resolved linear backend | startup configuration and long-context suite |
| Target feed-forward layers | Dense FP8 MLP | 27B model architecture; no routed-expert MoE | model configuration and source |
| DFlash2 draft attention | FlashInfer | explicit draft backend | draft-runner startup line |
| Draft local convolution | DFlash2 local-convolution path | merged upstream PR #35371 | DFlash2 startup and decode |
| Candidate selection | folded into draft CUDA graph | upstream DFlash2 integration | graph-capture startup line |
| DFlash fused KV materialization | enabled, 5 layers / 8 KV heads / 128 head dim | upstream DFlash2 integration | allocation/startup line |
| Multimodal attention | `triton_attn` | automatic | startup log; vision smoke |
| Multimodal rotary | fused three-axis mRoPE | local `64ecd64924`, PR #35744 | numerical kernel tests and vision |
| Sampling / grammar | FlashInfer / XGrammar | automatic | startup log; reasoning and tools |
| HiCache transfer | NIXL POSIX, io_uring, O_DIRECT | explicit storage backend | restart and concurrent restoration |
| Persistent representation | target KV + Mamba/GDN + DFlash2 sidecar | local NIXL integration | namespace and restore qualification |

The selected 27B target keeps `lm_head` in BF16. The quantized-head selector
from upstream PR #35496 is present in the base but was not selected for these
measurements. The 27B model is dense and does not use Flash-Next's MoE routing
corrections. The measured stack contains no separate local DeepGEMM SM120 patch.

## Resolved backends: Qwen3.8 Flash-Next

| Component or phase | Resolved implementation | How selected or enabled | Evidence |
|---|---|---|---|
| Target GDN decode | `FlashInferGDNKernel` | narrow phase override | startup dispatcher; 1x/4x decode |
| Target GDN prefill | `FlashInferGDNKernel` | narrow phase override | 64K/490K prefill and needles |
| Native-MTP draft decode | `FlashInferGDNKernel` | shared GDN dispatch | draft CUDA graphs and live decode |
| Native-MTP draft extend | `FlashInferGDNKernel` | shared GDN dispatch | draft extend graph capture |
| Active GDN target verification | FlashInfer WY output-only | local `280825c3e2`, `none` mode only | direct state parity and graph replay |
| Ordinary/tree state-writing verification | `TritonGDNKernel` | Triton remains selected for state-writing modes | non-`none` source dispatch and focused tests |
| Accepted-state recovery | FlashInfer WY output-only | narrow SM120 RecoverSSM route | recovery graphs BS 1-4; long continuation |
| QSA sparse prefill | Triton sparse GQA | model path + FP8 tile fix `95da38fb3b` | 64K/490K exact qualification |
| QSA sparse decode | FlashInfer QSA wrapper resolving to XQA | SM120 wrapper dispatch `c1da0eef56`, narrowed to exact SM120 by `8de07a058f`; FlashInfer selects XQA on SM120 | direct backend probe and live decode |
| QSA top-k | `sgl-kernel` | automatic | startup resolution |
| QSA MTP index sharing | enabled | model path | startup line and shared-index tests |
| Target MoE | FlashInfer CUTLASS | automatic ModelOpt-NVFP4 resolution | startup resolution and live tests |
| Native-MTP MoE | FlashInfer CUTLASS | automatic speculative resolution | startup resolution and live tests |
| HyperConnection Mix | persistent Triton Mix | automatic SM120 fallback | HC numerical tests; FlashInfer gate retained |
| Optional online-FP8 linears | FlashInfer CUTLASS MXFP8 for eligible transformer projections; row-wise FP8 for HC mix and output head | `SGLANG_SM120_ONLINE_MXFP8=true`, exact SM120 only | real-kernel numerics, graph replay, load invariants and live C1/C4 checks |
| HyperConnection Combine | fused SGLang CUDA | Qwen4 model path | kernel tests and full suite |
| Multimodal attention | `triton_attn` | automatic | full 1,024-token vision validation |
| Sampling / grammar | FlashInfer / XGrammar | automatic | startup log; reasoning and tool suite |
| HiCache transfer | NIXL POSIX, io_uring, O_DIRECT | explicit storage backend | 64K/490K restart restoration |
| Persistent representation | packed target/native-MTP KV + complete GDN/PLE siblings + compressed QSA keys | local hybrid-pool integration | restart restore, needles, continuation |

<a id="sm120-enablement-was-scoped-not-indiscriminate"></a>

## How Pennyroyal selects SM120 paths

Several upstream checks treated SM100 as the only eligible architecture even
when an underlying kernel already compiled for SM120. Pennyroyal handles four
cases:

1. **Explicit phase override.** Automatic GDN selection used an SM100 helper and
   therefore did not select FlashInfer on SM120. FlashInfer 0.6.17 already
   supported the BF16 decode and prefill operations used here. The launcher
   selects only those phases with `--linear-attn-decode-backend flashinfer` and
   `--linear-attn-prefill-backend flashinfer`. Long prefill, exact needles,
   decode, and vision/reasoning qualification cover the resulting runtime.

2. **SM120 source integration.** RecoverSSM needed FlashInfer WY output-only
   state output and accepted-state recovery. FlashInfer already contained an
   `sm_120a` implementation, but SGLang did not route SM120 into it. Commit
   `280825c3e2` enables only the output/recovery route used by
   `gdn_mtp_cache_mode=none` and preserves Qwen4 PLE accepted-state commit
   ordering. Direct tests cover all accepted lengths for four draft tokens,
   batch sizes 1-4, mixed acceptance, positions 63/64/65, `extra_buffer`, prefix
   restoration, retraction, long continuation, and captured-graph replay.

3. **QSA wrapper with XQA.** Commit `c1da0eef56` lets
   SM120 enter FlashInfer's page-aligned QSA decode wrapper, while `8de07a058f`
   adopts upstream #36806's exact `(12, 0)` gate so SM121/GB10 is not admitted.
   On SM120 the wrapper resolves to XQA. Pennyroyal's SM120 measurements use
   that backend, separately from the upstream SM100 TRTLLM-Gen results.

4. **Incompatible gates retained.** Forced TRTLLM-Gen fails with
   `Unsupported architecture`. Bypassing the guard and loading the exact
   BF16-Q/FP8-KV/H256/P64 cubin directly returns
   `CUDA_ERROR_NO_BINARY_FOR_GPU (209)`: the binary targets SM100 and contains
   the `sm10x_tcgen05` instruction family unavailable on SM120. Supporting it
   would require kernel/compiler adaptation upstream of SGLang; see
   [TensorRT-LLM #11799](https://github.com/NVIDIA/TensorRT-LLM/issues/11799)
   and [FlashInfer #3628](https://github.com/flashinfer-ai/flashinfer/issues/3628).
   Ordinary FlashInfer state-writing GDN verification also remains gated and
   uses Triton, while FlashInfer HyperConnection Mix remains SM100-only and
   falls back to the qualified persistent Triton Mix.

Other corrections follow the same scope. `0f159cd545` supplies XQA's packed
mask for fixed-width DFlash2 verification. `64ecd64924` extends the fused
rotary kernel to real three-axis mRoPE with height and width positions.

## Memory recovery and automatic KV sizing

| Flash-Next configuration | Intermediate SSM | Mamba slots | GPU KV capacity |
|---|---:|---:|---:|
| BF16 before RecoverSSM | 1.05 GiB | 24 | 745,600 tokens |
| RecoverSSM `none` mode | 0 | 24 | 824,384 tokens |

The sequence matters:

1. BF16 reduced recurrent-state bytes per slot.
2. Automatic Mamba sizing then expanded from 21 to 49 slots and consumed much
   of that saving.
3. The workload needed far fewer slots; 24 remained above observed demand and
   returned the rest of the budget to KV.
4. RecoverSSM removed the separate 1.05 GiB intermediate speculative SSM pool.
5. With no forced KV-token count, the measured KV pool increased by 78,784
   tokens—from 745,600 to 824,384, or 10.6%.

The physical pool is removed in `280825c3e2`, but the current
`kv_cache_configurator.py::_handle_max_mamba_cache` estimator only omits that
reserve for ReplaySSM, not RecoverSSM `none`. This runtime has no separate
allocator correction. The qualified launcher increases
`--mem-fraction-static` from `.97` to `.981`, approximately the recovered GPU
fraction, so automatic KV sizing can consume the physically freed memory. The
824,384-token result was measured, not hard-coded, but the estimator debt is
real and documented.

## Performance and qualification

The [v2.5.0 table above](#flash-next-v250-online-fp8-performance-at-a-glance)
shows the online-FP8 C1 observations; the
[retained v2.4.0 table](#retained-v240-flash-next-prefill-measurements) preserves
the earlier prefill measurements. The
[earlier Flash-Next campaign](RESULTS.md#earlier-flash-next-campaign),
including its source, clock settings, and quality results, remains in
RESULTS.md. The 27B and third-party sections below use different models,
hardware, and methods.

### Independent TP=2 FP8 validation

H3PO independently validated this runtime on a dual-GPU TP=2/EP=2 FP8
deployment with native MTP after removing the optional overlap-plan-stream
setting. The raw third-party table, corpus description, capacity result, and
scope limits are preserved in [RESULTS.md](RESULTS.md#independent-tp2-fp8-validation).
See [Community results](COMMUNITY-RESULTS.md#h3po--two-gpu-flash-next) for
H3PO's NVFP4 follow-up and reports from other operators.

### Qwen3.8-27B FP8/DFlash2 — August 24, 2026 campaign

Measured for the `qwen38-dflash2-pro6000-20260824` release on one RTX PRO
6000 at TP1. These results remain associated with that dated release.

| Test | Result |
|---|---:|
| 64K prefill | 6,163.07 tok/s |
| 489,921-token prefill | 1,618.31 tok/s; 3/3 exact needles |
| 1x 1,024-token decode | 108.75 tok/s after first token |
| 4x 1,024-token decode | 390.23 tok/s aggregate |
| Reasoning, xhigh / medium | 98.26 / 95.807 |
| Medium tool suite | 30/30 tool selections and arguments; 29/30 reviewed response discipline |

The cited public TP1 official-FP8/MTP3 community capture is directional rather
than a strict A/B because checkpoint, runtime, speculation, power, harness,
output duration, and cache configuration differ.

The August 26 confirmation of the 24-slot/five-state configuration retained
exact long-context needles, reached a 96.92 reasoning score across 50,986
completion tokens, and used at most 10 of 24 Mamba entries.

### 27B real agentic context behavior

The dated 124-request sample covered 85,156 output tokens and inputs from 183
to 350,195 tokens. [RESULTS.md](RESULTS.md) preserves the context-band tables,
overlap treatment, and distinction between completed-request and instantaneous
telemetry measurements.

Detailed definitions, complete 27B context bands, both 27B campaigns, and
persistence evidence are in [RESULTS.md](RESULTS.md).

## HiCache/NIXL persistence

HiCache/NIXL provides reusable prefix state across GPU/host eviction and
service restart. GPU-resident decode remains on the normal model path:

```text
GPU radix state
  -> page-first HiCache host RAM, kernel I/O, write-through
  -> NIXL POSIX FILE storage, io_uring + O_DIRECT
```

Flash-Next restart evidence:

- 63,808 of 63,864 input tokens restored; 56 recomputed.
- 489,856 of 489,879 restored; 23 recomputed.
- all three needles exact after restart.

The earlier 27B work also demonstrated 518,528 restored tokens with a six-token
tail in 14.64 seconds, 60,032-token namespace reuse after restart, A→B→A page-
size namespace rollback, three concurrent ~60K restores, and a 775,168-token
three-request concurrent restore with only 320 tail/page-rounding tokens
computed.

Two portable NIXL fixes from this project remain open upstream: PR #36520
isolates overlapping path registrations, and PR #36524 bounds bounce-backed
hybrid transfers. Flash-Next adds complete PLE/GDN/QSA sibling-state persistence
and load/COW ordering. [MEMORY-AND-PERSISTENCE.md](MEMORY-AND-PERSISTENCE.md)
documents the representation namespace and actual failure boundaries.

## Build and launch

There is one native build procedure and two supported model profiles. Flash-Next
provides FR-Spec and non-FR launchers:

```bash
git clone --branch pennyroyal-v2.5.2 --single-branch \
  https://github.com/jpezzulli/sglang-rtxpro6000.git pennyroyal
cd pennyroyal
# Follow BUILD.md for the fresh or existing-environment native install.
```

Set `REPO_ROOT`, `CACHE_BASE`, `NIXL_STORAGE_BASE`, and the checkpoint paths,
then choose one recipe:

```bash
configs/pennyroyal/serve-flash-next-frspec.sh
configs/pennyroyal/serve-qwen38-27b-dflash2.sh
# Alternative Flash-Next recipe without FR-Spec:
configs/pennyroyal/serve-flash-next.sh
```

Do not run them simultaneously on one GPU. [BUILD.md](BUILD.md) records exact
dependencies and [RUN.md](RUN.md) provides startup assertions, ordinary
OpenAI-compatible smoke requests, and cold/radix/NIXL cache distinctions.

## Source changes and upstream work

The source line starts with the 2026-08-24 27B release, then adds the unified
runtime and Flash-Next work. v2.5.2 builds on the v2.5.1/v2.5.0 stack;
[PROVENANCE.md](PROVENANCE.md) records the exact release source. Major groups
are:

- Qwen3.8-27B/DFlash2: independent target/draft overrides, fixed-width XQA mask,
  NVCC host-compiler identity, request-span observability, and NIXL correctness.
- Flash-Next: Qwen4 model support, QSA/HC/PLE/native-MTP integration, SM120 QSA
  decode, FP8 QSA prefill, RecoverSSM, complete hybrid persistence, and mRoPE.
- v2.1 upstream sync: exact-SM120 QSA routing from #36806 and the Mamba radix
  ghost-node/speculative tracking correction from #35821, adapted to Penny's
  fused CUDA and KDA accepted-state paths.
- v2.1.1 maintenance: accumulated additive sampling penalties are applied
  correctly during 27B DFlash2 verification; shared HiCache JIT transfers are
  bound to their torch copy streams and load-back waits for in-flight forward
  writes for both supported models.
- v2.1.2 maintenance: skip one-rank sampler synchronization (#37962) and
  preserve streamed tool-call framing around separator whitespace (#37408).
  The graph-lifetime trial is reverted and is not part of this release's
  runtime behavior.
- v2.3: FR-Spec reduces draft-vocabulary work for Flash-Next. The token map
  and launcher use the same runtime and kernels as v2.1.2.
- v2.3.1 maintenance: align paired Triton GDN verification and accepted-state
  recovery with ordinary decode's gate rounding, adapted from #36014.
- v2.4.0: reduce Flash-Next prefill preparation, bound short QSA extensions,
  correct routing dependency order, and improve shared request cancellation,
  explicit reasoning-effort handling, and tool-markup parsing.
- v2.5.0: add exact-SM120 opt-in online FP8 for selected Flash-Next weights,
  optional NVMe-backed PLE, structured-output startup warmup, and bounded
  Flash-Next loader-lifetime cleanup. The 27B path remains option-off.
- v2.5.1: improve host-cache checkpoint retention, handle temporary Mamba-slot
  exhaustion and unaligned QSA chunk prefixes, fix reasoning/tool-marker
  handling, and expose optional C6 capacity settings.
- Project upstream submissions include #36520, #36524, and #35584; dated
  status records are in CHANGES.md.
- Closed project submission #35583 remains in runtime history. Transient
  ragged/DSpARK PR #35586 is documented outside the active source.
- Related upstream work: #30967, #35371, #35496, #35744, #35821, #36497,
  #36644, and #36806.

[CHANGES.md](CHANGES.md) lists every material current commit, the earlier dated
release hashes, exact PR links/status/heads, affected execution paths, and test
coverage.

## Scope and limitations

- One RTX PRO 6000, TP=1, exact checkpoint families, and the recorded CUDA,
  PyTorch, FlashInfer, compiler, NIXL, and SGLang source.
- SM120 paths were qualified narrowly; unsupported-by-default is not treated as
  proof of incompatibility, but no architecture gate was globally deleted.
- Benchmark values describe the recorded system and workload.
- Flash-Next and 27B results remain separate; neither model's features or
  numbers are silently attributed to the other.
- The earlier Radix Flash-Next reasoning/tool/vision campaign ran at
  `64ecd64924`. The v2.1.1 source at `fb1216c6c4` received focused CPU/GPU
  DFlash, HiCache, hybrid/NIXL, and unified-load-back tests plus ordinary
  64K/490K/1K/C4 and post-restart NIXL restoration on both supported models.
  It relied on the earlier reasoning and tool results; those suites were not
  rerun for the narrow update.
- v2.1.2 received focused sampler/parser tests and both-profile JSON,
  incremental parallel-tool, prefill, C1/C4 decode, and NIXL restart-restore
  regressions. Full reasoning, tools, and vision were not repeated for that
  maintenance update; its historical performance tables were unchanged.
- v2.3 testing covered the full Flash-Next suite and both profiles' startup,
  prefill, single/concurrent decode, and NIXL restoration. Reasoning scored
  98.52/100; tools completed 29 clean workflows plus one redundant read-only
  call. [Detailed results](RESULTS.md#pennyroyal-v23--flash-next-fr-spec)
  distinguish tool execution from response-quality and evaluator limitations.
  The 27B checks were regressions; no speed comparison was run.
- NIXL cleaner thresholds use whole-filesystem occupancy percentages. They do
  not set a directory byte quota.
- Online FP8 did not improve cold long-context TTFT in the matching samples.
  The v2.5.0 tool campaign retained observed quoted-markup misses; v2.5.1 adds
  the parser corrections described in [CHANGES.md](CHANGES.md#v251--cache-and-parser-maintenance).
- NVMe PLE reduced fixed host residency but was slower than RAM PLE in the
  separate repeated C4 comparison. It trades fixed host residency for SSD I/O
  and lower measured throughput in that comparison.

See [LIMITATIONS.md](LIMITATIONS.md) for measurement and reproducibility detail.

## Documentation map

- [BUILD.md](BUILD.md) — one native build and dependency identity.
- [RUN.md](RUN.md) — native launch paths for both profiles, the non-FR
  Flash-Next alternative, startup checks, and optional settings.
- [CONFIGURE.md](CONFIGURE.md) — what the beta setup configurator does, how to
  start it for native and container use, and where its settings are saved.
- [FP8.md](FP8.md) — online-FP8 precision boundary, activation, evidence and
  limits.
- [NVME-PLE.md](NVME-PLE.md) — optional reader installation, overlay
  preparation, launch settings and measured RAM/speed tradeoff.
- [BACKENDS.md](BACKENDS.md) — source-backed resolved backend and SM120 tables.
- [RESULTS.md](RESULTS.md) — controlled, reasoning, tools, context, agentic, and
  persistence measurements for both models.
- [MEMORY-AND-PERSISTENCE.md](MEMORY-AND-PERSISTENCE.md) — GPU allocation and
  hybrid state representation.
- [CHANGES.md](CHANGES.md) — cumulative release and upstream history.
- [PROVENANCE.md](PROVENANCE.md) — source, package, checkpoint, and tag identity.
- [LIMITATIONS.md](LIMITATIONS.md) — evidence, portability, and reproducibility boundaries.
- [`jpezzulli/pennyroyal-validation`](https://github.com/jpezzulli/pennyroyal-validation)
  — maintained public validation harness and result catalog.

## Engineering history

The experimental path—including DFlash2 work and the HiCache/Mooncake/NIXL
detour—is described at
[msoexpert.com](https://msoexpert.com/articles/qwen38-dflash2-rtx-pro-6000/).
Mooncake was evaluated and removed. The runtime now uses NIXL POSIX for
persistent prefix storage.

## License

The SGLang-derived source remains under Apache-2.0. Model checkpoints and
third-party runtimes retain their own licenses.
