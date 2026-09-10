# One SGLang runtime, two Qwen3.8 configurations, one RTX PRO 6000

> One optimized SGLang runtime for a single RTX PRO 6000, with qualified
> launch configurations for both Qwen3.8-27B/DFlash2 and Qwen3.8 Flash-Next.

This repository contains the complete SGLang-derived source used on one
NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB, SM120, TP=1). It is
not two builds: both model configurations run from the same patched source.

**Get running:** [Fresh install](BUILD.md#fresh-install) ·
[Update an existing install](BUILD.md#update-an-existing-install) ·
[Choose a model](#models-and-launch-recipes) · [Launch](RUN.md).

**v2.4.1** is optional maintenance: more efficient grammar-history handling,
correct streaming logprobs, pinned metadata transfers, and clearer setup.
Image decoding/preprocessing defaults to CPU, with an optional secondary GPU;
[configuration and tradeoffs](RUN.md#cpu-or-secondary-gpu-media-preprocessing).
The separate NumPy huge-page-advice default is also overridable.
If v2.4.0 works well for you, there is no urgent need to update. No additional
serving-speed gain is claimed. [Changes and upstream credits](CHANGES.md#v241--maintenance-and-easier-setup).

The preceding v2.4.0 release reduced Flash-Next prefill overhead, with
approximately **4–10% higher throughput observed in cold-prefill tests at 64K
and 490K**. Cached-prefix extension and NIXL restart restoration remained
verified. Both profiles retained their full context, token pools, speculative
decoding, and HiCache/NIXL configurations.

That v2.4.0 update also corrected disconnected-request cleanup, explicit Chat
reasoning-effort overrides, and bare/undeclared tool-markup handling. It was an
optional update, not an urgent upgrade or a new decode-speedup claim.
[Changes and upstream credits](CHANGES.md#v240--maintenance-and-faster-flash-next-prefill) ·
[Build/update instructions](BUILD.md).

## Models and launch recipes

The two recipes below are equally supported with **HiCache and NIXL** on a
single RTX PRO 6000. The linked targets are the public reference checkpoints
for their recipes; that designation is not a claim that every checkpoint in
this section was separately benchmarked.

| Recipe | Reference target | Speculative decoding and launcher |
|---|---|---|
| **Flash-Next NVFP4** | [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | Native NEXTN MTP is included. Start with the recommended [FR-Spec launcher](configs/pennyroyal/serve-flash-next-frspec.sh); the [non-FR launcher](configs/pennyroyal/serve-flash-next.sh) remains available. No separate draft download. |
| **27B FP8** | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | Use the [27B launcher](configs/pennyroyal/serve-qwen38-27b-dflash2.sh) with the separate [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) draft checkpoint. |

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

### Flash-Next performance at a glance

Measured September 9, 2026, with **v2.4.0 and FR-Spec on one RTX PRO 6000, TP1**,
keeping 524,288-token context, 824,384 KV tokens, and HiCache/NIXL enabled.
These are dated measurements, not guaranteed speeds.

| Workload | Speed | Timing / result |
|---|---:|---|
| 64K cold prefill — 63,864 input tokens | **14,842 tok/s** | Server prefill rate; 5.242 s to first token; exact READY |
| 490K cold prefill — 489,879 input tokens | **8,773 tok/s** | Server prefill rate; 60.613 s to first token; all three needles found |
| Single-request decode — 1,024 output tokens | **181.72 tok/s** | Median of three runs; excludes time to first token |
| Four simultaneous requests — 1,024 output tokens each | **446.49 tok/s aggregate** | Median of three runs; includes time to first token and the slowest response |

Prefill rates divide input tokens by the server's initial-prefill time; client
time to first token also includes other request processing. The 4–10% gain
compares that same server timing window before and after v2.4.0, not the
TTFT-based rates in older tables. These are single cold-prefill observations;
warm-prefill speedup was not separately measured. Decode varies with workload
and acceptance, and no decode improvement is claimed. The four-request figure
is combined throughput, not each response's speed.
See [RESULTS.md](RESULTS.md#pennyroyal-v240--prefill-and-maintenance) for the
comparison; earlier v2.3 measurements remain there as dated history.

**In real agentic use:** a session on September 6 sustained
**155 tok/s** across nine responses of at least 1,024 output tokens, with
roughly **203K–280K input context**. Those responses generated 22,535 tokens;
the rate excludes prefill and time between requests. Short replies are excluded
from this sustained-generation figure. This is an observed session, not a
controlled benchmark. [Session details](RESULTS.md#flash-next-agentic-session--september-6-2026).

### Broader model support without HiCache/NIXL

**Without HiCache and NIXL, many more SGLang-supported models and speculative
configurations can run and may benefit from applicable performance
optimizations.** Potential speed improvements come from those optimizations,
not automatically from disabling HiCache/NIXL.

HiCache/NIXL requires model- and speculative-method-specific cache layouts
and complete state-restoration support. Different MTP or draft designs can
require additional integration to save and restore target KV, draft KV, and
recurrent or other model-specific state together.

The supported recipes above are therefore not an exhaustive model
support list. Other models still need compatible target/draft configurations,
supported kernels, sufficient memory, and appropriate launch settings.

See [RUN.md](RUN.md) for the launchers and the
[configuration matrix](#qualified-configuration-matrix) for dtypes, context,
cache capacity, and backends.

Flash-Next is day-one engineering. It has extensive qualification on this exact
machine—including 524K context, multimodal input, reasoning, tools, agentic
workloads, CUDA-graph recovery, and persistent prefix restoration—but it may
still contain rough edges or hardware/model-specific assumptions.

### Hardware, storage, and first-start expectations

The qualified launchers use a 96 GB GPU, but they also need substantial host
RAM and disk. Flash-Next offloads PLE embeddings to the host in addition to its
configured 32 GiB HiCache tier; 27B configures a 96 GiB HiCache tier. Process,
filesystem, and driver overhead are additional. These are measured settings,
not minimum-RAM claims.

Allow disk space for the target checkpoint, the 27B draft when applicable,
compiler/JIT caches, and persistent NIXL namespaces. NIXL cleaner percentages
apply to the whole selected filesystem, and old representation-specific
namespaces are preserved rather than silently deleted.

The first launch can appear quiet while checkpoint identities are hashed, then
spend substantial time compiling kernels and capturing CUDA graphs. Wait for
the server-ready log and verify the API; a cold namespace will not restore an
older prefix. The qualified path is the native build in [BUILD.md](BUILD.md).
Docker material inherited from upstream is not a qualified Pennyroyal recipe.

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
baseline run. The baselines were on separate boots, so the range reflects
observed variation, not a guaranteed speedup. Cold 64K/490K prefill was
essentially unchanged by FR-Spec.

The [performance table above](#flash-next-performance-at-a-glance) gives the
later v2.4.0 observations. [RESULTS.md](RESULTS.md#pennyroyal-v23--flash-next-fr-spec)
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

The important work is architectural, not merely a collection of launch flags:

- DFlash2 integration for Qwen3.8-27B, with the selector inside the draft CUDA
  graph, fused draft KV materialization, and fixed-width TRTLLM-MHA/XQA target
  verification on SM120.
- Native NEXTN MTP for Flash-Next with QSA index sharing, recovery graphs, and
  `gdn_mtp_cache_mode=none`.
- FlashInfer GDN decode and prefill for Flash-Next, plus a deliberately narrow
  SM120 FlashInfer WY output-only RecoverSSM path.
- QSA sparse attention: Triton sparse-GQA prefill, the FlashInfer QSA wrapper
  resolving to XQA on SM120, and `sgl-kernel` top-k.
- FlashInfer CUTLASS MoE for the Flash-Next target and native-MTP layer.
- Correct three-axis multimodal mRoPE in the fused QK RMSNorm+RoPE kernel.
- Target, draft, verify, prefill, and accepted-state recovery CUDA graphs.
- FP8 KV and 524,288-token factor-2 YaRN for both launch configurations.
- HiCache/NIXL persistence for complete hybrid state—not KV alone—including
  DFlash2 side state or Flash-Next GDN, Qwen4 PLE, and compressed QSA state.
- Restart restoration, representation-specific FILE namespaces, and radix-
  prefix reuse followed by long continuation.
- Measured reasoning, tool-calling, vision, long-context needle, controlled
  decode, and real agentic behavior.

## Published identity

| Item | Value |
|---|---|
| Canonical branch | `pennyroyal-main-sm120-final` |
| Current executable source | `cf811a8c5988dc87941c1442fdc8ba574a0400f7` |
| Current release | **v2.4.1** |
| Git tag | `pennyroyal-v2.4.1` |
| Initial unified dated tag | `sglang-rtxpro6000-20260827` |
| Earlier 27B dated tag | `qwen38-dflash2-pro6000-20260824` |
| Upstream integration base | `e7e78940168f3ba65c762a6f82fd8bc5b6ee04e3` |
| Qualification dependency base | `0.5.19.dev492+g836206a0a` with the updated v2.4.1 Python/JIT source |
| Python / PyTorch | `3.12.13` / `2.13.0+cu130` |
| CUDA / compiler | CUDA `13.3` (NVCC `13.3.73`) / GCC `15.3.1` |
| FlashInfer / NIXL | `0.6.17` / `1.4.0` |
| GPU / driver | RTX PRO 6000 96 GB, SM120 / `610.57.04` |

Install v2.4.1 from the updated source using [BUILD.md](BUILD.md); retaining
an older wheel alone does not apply these changes. Dependency versions and
model settings are unchanged. Source lineage and upstream relationships are in
[CHANGES.md](CHANGES.md) and [PROVENANCE.md](PROVENANCE.md).

## Qualified configuration matrix

Rows intentionally separate representation, datatype, backend, and capacity.
The BF16 runtime dtype is not presented as a claim that every quantized kernel
executes or accumulates entirely in BF16.

| Property | Qwen3.8-27B + DFlash2 | Qwen3.8 Flash-Next |
|---|---|---|
| Target checkpoint | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) recipe reference; retained measurements used the [orcarouter derivative](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8) | [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) |
| Target weight format | block FP8 E4M3, 128x128 blocks | ModelOpt NVFP4, group size 16 on selected Linear modules |
| Quantized-path activations | dynamic FP8 E4M3 | NVFP4 input activations on selected Linear modules |
| Runtime dtype for unquantized tensors | BF16; includes excluded layers and BF16 `lm_head` | BF16; includes ignored layers, native MTP, PLE, QSA/GDN state-facing tensors, and vision |
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
| Current GPU KV capacity | 1,118,784 target and draft tokens | 824,384 target and native-MTP tokens |
| Mamba capacity | 24 slots; maximum 5 retained states/path | 24 slots; `extra_buffer`, tracking interval 64 |
| HiCache/NIXL | 96 GB configured host tier; target KV + Mamba/GDN + DFlash2 state | 32 GB configured host tier; packed target/native-MTP KV + GDN + PLE + QSA keys |
| Maximum active requests | 4 | 4 |

The current 27B launcher uses the 24-slot/five-state setting qualified on
2026-08-26. The earlier 2026-08-24 performance release used 16 slots, a
three-state path cap, and a 1,194,496-token KV pool. Both campaigns are retained
in [RESULTS.md](RESULTS.md) rather than silently merging their allocations.

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
from upstream PR #35496 is present in the base but is not the source of this
checkpoint's measured speed. The 27B model is dense; Flash-Next's MoE routing
corrections do not apply to it. No separate local DeepGEMM SM120 patch is claimed.

## Resolved backends: Qwen3.8 Flash-Next

| Component or phase | Resolved implementation | How selected or enabled | Evidence |
|---|---|---|---|
| Target GDN decode | `FlashInferGDNKernel` | narrow phase override | startup dispatcher; 1x/4x decode |
| Target GDN prefill | `FlashInferGDNKernel` | narrow phase override | 64K/490K prefill and needles |
| Native-MTP draft decode | `FlashInferGDNKernel` | shared GDN dispatch | draft CUDA graphs and live decode |
| Native-MTP draft extend | `FlashInferGDNKernel` | shared GDN dispatch | draft extend graph capture |
| Active GDN target verification | FlashInfer WY output-only | local `280825c3e2`, `none` mode only | direct state parity and graph replay |
| Ordinary/tree state-writing verification | `TritonGDNKernel` | architecture gate deliberately preserved | non-`none` source dispatch and focused tests |
| Accepted-state recovery | FlashInfer WY output-only | narrow SM120 RecoverSSM route | recovery graphs BS 1-4; long continuation |
| QSA sparse prefill | Triton sparse GQA | model path + FP8 tile fix `95da38fb3b` | 64K/490K exact qualification |
| QSA sparse decode | FlashInfer QSA wrapper resolving to XQA | SM120 wrapper dispatch `c1da0eef56`, narrowed to exact SM120 by `8de07a058f`; FlashInfer selects XQA on SM120 | direct backend probe and live decode |
| QSA top-k | `sgl-kernel` | automatic | startup resolution |
| QSA MTP index sharing | enabled | model path | startup line and shared-index tests |
| Target MoE | FlashInfer CUTLASS | automatic ModelOpt-NVFP4 resolution | startup resolution and live tests |
| Native-MTP MoE | FlashInfer CUTLASS | automatic speculative resolution | startup resolution and live tests |
| HyperConnection Mix | persistent Triton Mix | automatic SM120 fallback | HC numerical tests; FlashInfer gate retained |
| HyperConnection Combine | fused SGLang CUDA | Qwen4 model path | kernel tests and full suite |
| Multimodal attention | `triton_attn` | automatic | full 1,024-token vision validation |
| Sampling / grammar | FlashInfer / XGrammar | automatic | startup log; reasoning and tool suite |
| HiCache transfer | NIXL POSIX, io_uring, O_DIRECT | explicit storage backend | 64K/490K restart restoration |
| Persistent representation | packed target/native-MTP KV + complete GDN/PLE siblings + compressed QSA keys | local hybrid-pool integration | restart restore, needles, continuation |

## SM120 enablement was scoped, not indiscriminate

Several upstream checks treated SM100 as the only eligible architecture even
when an underlying kernel already compiled for SM120. This runtime distinguishes
three cases:

1. **Explicit phase override.** Automatic GDN selection used an SM100 helper and
   therefore did not select FlashInfer on SM120. FlashInfer 0.6.17 already
   supported the BF16 decode and prefill operations used here. The launcher
   selects only those phases with `--linear-attn-decode-backend flashinfer` and
   `--linear-attn-prefill-backend flashinfer`. Long prefill, exact needles,
   decode, and vision/reasoning qualification cover the resulting runtime.

2. **Narrow source integration.** RecoverSSM needed FlashInfer WY output-only
   state output and accepted-state recovery. FlashInfer already contained an
   `sm_120a` implementation, but SGLang did not route SM120 into it. Commit
   `280825c3e2` enables only the output/recovery route used by
   `gdn_mtp_cache_mode=none` and preserves Qwen4 PLE accepted-state commit
   ordering. Direct tests cover all accepted lengths for four draft tokens,
   batch sizes 1-4, mixed acceptance, positions 63/64/65, `extra_buffer`, prefix
   restoration, retraction, long continuation, and captured-graph replay.

3. **QSA wrapper enabled, backend distinguished.** Commit `c1da0eef56` lets
   SM120 enter FlashInfer's page-aligned QSA decode wrapper, while `8de07a058f`
   adopts upstream #36806's exact `(12, 0)` gate so SM121/GB10 is not admitted.
   On SM120 the wrapper resolves to XQA, an existing supported implementation;
   it does not select TRTLLM-Gen. No upstream SM100 throughput claim is
   attributed to Penny's SM120 results.

4. **Incompatible gates deliberately retained.** Forced TRTLLM-Gen fails with
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

Other bounded corrections follow the same rule. `0f159cd545` supplies XQA's
packed mask for fixed-width DFlash2 verification rather than enabling a new
global attention backend. `64ecd64924` extends the fused rotary kernel to real
three-axis mRoPE rather than dropping the fused path or ignoring H/W positions.

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
reserve for ReplaySSM, not RecoverSSM `none`. There is no separate allocator
correction commit in this runtime. The qualified launcher increases
`--mem-fraction-static` from `.97` to `.981`, approximately the recovered GPU
fraction, so automatic KV sizing can consume the physically freed memory. The
824,384-token result was measured, not hard-coded, but the estimator debt is
real and documented.

## Performance and qualification

The [Flash-Next table above](#flash-next-performance-at-a-glance) shows the
v2.4.0 observations. The [earlier Flash-Next campaign](RESULTS.md#earlier-flash-next-campaign),
including its source, clock settings, and quality results, remains in
RESULTS.md as historical evidence. The separate 27B and third-party results
below are not a matched A/B comparison with Flash-Next.

### Independent TP=2 FP8 validation

H3PO independently validated this runtime on a dual-GPU TP=2/EP=2 FP8
deployment with native MTP after removing the optional overlap-plan-stream
setting. The raw third-party table, corpus description, capacity result, and
scope limits are preserved in [RESULTS.md](RESULTS.md#independent-tp2-fp8-validation).

### Qwen3.8-27B FP8/DFlash2 — August 24, 2026 campaign

Measured for the `qwen38-dflash2-pro6000-20260824` release on one RTX PRO
6000 at TP1. These are retained 27B measurements, not a new v2.4.0 benchmark.

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

HiCache/NIXL does not cause GPU-resident decode speed. It provides disposable,
reusable prefix state across GPU/host eviction and service restart:

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
git clone --branch pennyroyal-v2.4.1 --single-branch \
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

The cumulative history starts with the 2026-08-24 27B release, then layers the
unified runtime and Flash-Next work on the same source line. The current active
stack contains 35 runtime commits above its integration base, excluding
documentation and recipe-only commits. Major groups are:

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
- Project upstream submissions include #36520, #36524, and #35584; dated
  status records are in CHANGES.md.
- Closed project submissions retained in runtime history: #35583; transient
  ragged/DSpARK PR #35586 is documented but not in the active source.
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
- Benchmark observations are not guarantees for another system.
- Flash-Next and 27B results remain separate; neither model's features or
  numbers are silently attributed to the other.
- The earlier Radix Flash-Next reasoning/tool/vision campaign ran at
  `64ecd64924`. The v2.1.1 source at `fb1216c6c4` received focused CPU/GPU
  DFlash, HiCache, hybrid/NIXL, and unified-load-back tests plus ordinary
  64K/490K/1K/C4 and post-restart NIXL restoration on both supported models.
  The full reasoning and tool suites were not rerun for this narrow update.
- v2.1.2 received focused sampler/parser tests and both-profile JSON,
  incremental parallel-tool, prefill, C1/C4 decode, and NIXL restart-restore
  regressions. Full reasoning, tools, and vision were not repeated for that
  maintenance update; its historical performance tables were unchanged.
- v2.3 testing covered the full Flash-Next suite and both profiles' startup,
  prefill, single/concurrent decode, and NIXL restoration. Reasoning scored
  98.52/100; tools completed 29 clean workflows plus one redundant read-only
  call. [Detailed results](RESULTS.md#pennyroyal-v23--flash-next-fr-spec)
  distinguish tool execution from response-quality and evaluator limitations.
  No 27B speed improvement is claimed.
- NIXL cleaner thresholds are whole-filesystem occupancy percentages, not an
  absolute directory byte quota.

See [LIMITATIONS.md](LIMITATIONS.md) for measurement and reproducibility detail.

## Documentation map

- [BUILD.md](BUILD.md) — one native build and dependency identity.
- [RUN.md](RUN.md) — the two canonical launch configurations.
- [BACKENDS.md](BACKENDS.md) — source-backed resolved backend and SM120 tables.
- [RESULTS.md](RESULTS.md) — controlled, reasoning, tools, context, agentic, and
  persistence measurements for both models.
- [MEMORY-AND-PERSISTENCE.md](MEMORY-AND-PERSISTENCE.md) — GPU allocation and
  hybrid state representation.
- [CHANGES.md](CHANGES.md) — cumulative release and upstream history.
- [PROVENANCE.md](PROVENANCE.md) — source, package, checkpoint, and tag identity.
- [LIMITATIONS.md](LIMITATIONS.md) — what the evidence does and does not prove.
- [`jpezzulli/pennyroyal-validation`](https://github.com/jpezzulli/pennyroyal-validation)
  — maintained public validation harness and result catalog.

## Engineering history

The experimental path—including DFlash2 work and the HiCache/Mooncake/NIXL
detour—is described at
[msoexpert.com](https://msoexpert.com/articles/qwen38-dflash2-rtx-pro-6000/).
Mooncake is not a fallback in this runtime; it was rejected and removed.

## License

The SGLang-derived source remains under Apache-2.0. Model checkpoints and
third-party runtimes retain their own licenses.
