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

**Stay updated:** choose **Watch → Custom → Releases** at the top of this
repository to get notified about new Pennyroyal releases.

Using an AI assistant to set this up? Give it [llms.txt](llms.txt)
for the short setup map and links to the detailed instructions.

**Evidence:** [Benchmarks and measurement definitions](RESULTS.md) ·
[Community-reported results](COMMUNITY-RESULTS.md) ·
[Validation suite and published reports](https://github.com/jpezzulli/pennyroyal-validation).

## Hardware, storage, and first-start expectations

The tested setup is **Linux with one RTX PRO 6000 Blackwell, 96 GB VRAM**.
Plan for host RAM as well as GPU memory:

| Profile | Default host-memory use, before loading and runtime overhead |
|---|---|
| Flash-Next | 32 GB HiCache plus approximately 47.68 GiB for the RAM-backed PLE table |
| 27B/DFlash2 | 96 GB HiCache |

These are configured allocations, not minimum system specifications. Leave
room for model loading, the operating system, and other applications. You can
[choose a smaller HiCache](RUN.md#choose-hicache-ram-size); Flash-Next also has
an [NVMe PLE option](NVME-PLE.md) that trades fixed RAM use for SSD I/O.

Allow disk space for model weights, the 27B draft if selected, compiled
kernels, and the persistent conversation cache. The
[NIXL disk budget](RUN.md#limit-nixl-disk-use) can limit cache growth, but is
a cleanup target rather than a hard quota.

The first start can take many minutes while weights are checked and loaded,
kernels compile, and CUDA graphs are captured. Wait for the server-ready log,
then check the API as described in the [launch guide](RUN.md#smoke-through-the-normal-api).

## Prebuilt Docker image for Qwen3.8 on RTX PRO 6000

The **v2.5.2 image is available** at
`ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.2`. It includes the runtime,
CUDA build tools, and NIXL for both supported profiles. You supply the NVIDIA
driver, model files, and writable cache folders on the host.

Follow the [Docker and Compose guide](docker/pennyroyal/README.md) to prepare
those folders, configure your model, and start the server. No local SGLang
build is needed.

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

<a id="v250-options--online-fp8-and-nvme-ple"></a>

## Optional Flash-Next features

- **[Online FP8](FP8.md)** converts selected weights during model loading.
  The measured decode improvements and precision tradeoffs are below.
- **[NVMe-backed PLE](NVME-PLE.md)** moves the large PLE table from fixed
  host RAM to a prepared SSD snapshot. It saves RAM, with a throughput cost
  in our repeated concurrent-request comparison.

Either option can be used on its own. Both are off by default; the linked
guides explain how to enable them and what was tested. For request capacity
and media preprocessing, see the [launch options](RUN.md).

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

<a id="v23--faster-flash-next-decoding-with-fr-spec"></a>
<a id="previous-v212--maintenance-release"></a>

## Earlier releases

The [release history](CHANGES.md) covers earlier maintenance and feature work.
The [v2.3 FR-Spec comparison](RESULTS.md#pennyroyal-v23--flash-next-fr-spec)
preserves its original baselines, measurements, and qualification results.

## Validation suite

The canonical test suite used to qualify these runtimes—including reasoning,
tool calling, long-context needles, vision, and concurrent decode—is maintained
in [`jpezzulli/pennyroyal-validation`](https://github.com/jpezzulli/pennyroyal-validation).
This repository contains the runtime source, configuration, and measured
results; `pennyroyal-validation` contains the reusable tests and result catalog.

## Why this runtime exists

Pennyroyal brings long-context, thinking-enabled agentic workloads to one
RTX PRO 6000: faster token generation, tool use and multimodal input, and
conversation state that can survive eviction from GPU memory or a server
restart. The two model profiles share one runtime and have their own tested
launch recipes.

The work includes SM120 attention and speculative-decoding paths, memory
savings, and persistence of the models' complete state. See
[the backend reference](BACKENDS.md) and
[memory and persistence](MEMORY-AND-PERSISTENCE.md) for how it works.

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

The current release is **[Pennyroyal v2.5.2](https://github.com/jpezzulli/sglang-rtxpro6000/releases/tag/pennyroyal-v2.5.2)**,
tag `pennyroyal-v2.5.2`. The default branch is `pennyroyal-main-sm120-final`.
[PROVENANCE.md](PROVENANCE.md) records source and checkpoint identities;
[BUILD.md](BUILD.md#qualified-environment) lists the tested dependencies.
The former `jpezzulli/qwen38-dflash2-pro6000` URL redirects here.

<a id="qualified-configuration-matrix"></a>
<a id="resolved-backends-qwen38-27bdflash2"></a>
<a id="resolved-backends-qwen38-flash-next"></a>
<a id="sm120-enablement-was-scoped-not-indiscriminate"></a>
<a id="how-pennyroyal-selects-sm120-paths"></a>
<a id="memory-recovery-and-automatic-kv-sizing"></a>

## Technical reference

- [Configuration matrix](BACKENDS.md#qualified-configuration-matrix): precision,
  context, cache capacity, and speculative settings for both profiles.
- [Resolved backends](BACKENDS.md): attention, GDN, MoE, and the narrowly
  selected SM120 paths, including the architecture gates we keep.
- [GPU allocation and persistence](MEMORY-AND-PERSISTENCE.md): RecoverSSM,
  automatic KV sizing, host state, and NIXL representation namespaces.

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

Choose one installation path:

1. **[Docker and Compose](docker/pennyroyal/README.md)** for the prebuilt image.
2. **[Native installation](BUILD.md)** to build and run from source.

Then use the **[beta configurator](CONFIGURE.md)** to save and check your
model, GPU, and cache settings. Start the server with the launch command it
gives you. Both installation guides also provide a manual path. Run one model
profile at a time on a single GPU.

## Source changes and upstream work

[CHANGES.md](CHANGES.md) records the release history, individual fixes,
upstream PRs, and verification scope. [PROVENANCE.md](PROVENANCE.md) records
the source lineage. These include the DFlash2 and native-MTP integrations,
SM120 kernels, cache restoration, parser fixes, and memory work.

## Scope and limitations

The published measurements belong to the stated model, release, hardware,
and workload. They are not promises for every SGLang model or GPU. Both
supported profiles have long-context, reasoning, tool, and media evidence;
[RESULTS.md](RESULTS.md) separates the campaigns and their limitations.

Community TP2 results are identified separately from local single-GPU tests.
The v2.5.2 TP2 changes still await external verification. Online FP8 did not
improve cold long-context time to first token in the matching samples, and
NVMe PLE was slower than RAM PLE in the repeated concurrent-request comparison.

See [LIMITATIONS.md](LIMITATIONS.md) for the evidence boundaries and
[CHANGES.md](CHANGES.md) for the checks performed for each maintenance release.

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
