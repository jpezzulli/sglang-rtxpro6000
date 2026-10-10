# SGLang for Qwen3.8 on one NVIDIA RTX PRO 6000 Blackwell (SM120)

> Pennyroyal is an optimized SGLang-derived runtime with single-GPU
> launch configurations for Qwen3.8-27B FP8 with DFlash2 and Qwen3.8
> Flash-Next NVFP4 with native NEXTN and FR-Spec.

This repository contains the complete SGLang-derived source used on one
NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB, SM120, TP=1). Both
model configurations run from the same patched source.

**Get running:** [Use the prebuilt container](docker/pennyroyal/README.md)
for the shortest setup, or [build and install natively](BUILD.md).
Both paths support the two model profiles below.
The manual instructions in both guides are confirmed working. An optional
[terminal setup utility **(beta)**](CONFIGURE.md) is open for testing;
please [report any issues](https://github.com/jpezzulli/sglang-rtxpro6000/issues).

**Stay updated:** choose **Watch → Custom → Releases** at the top of this
repository to get notified about new Pennyroyal releases.

Using an AI assistant to set this up? Give it [llms.txt](llms.txt)
for the short setup map and links to the detailed instructions.

**Results:** [Benchmarks and measurement definitions](RESULTS.md) ·
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

The v2.5.3 release image is named
`ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.3`. Images are published after the
release build passes its checks. The image includes the runtime,
CUDA build tools, and NIXL for both supported profiles. You supply the NVIDIA
driver, model files, and writable cache folders on the host.

Follow the [Docker and Compose guide](docker/pennyroyal/README.md) to prepare
those folders, configure your model, and start the server. No local SGLang
build is needed.

<a id="models-and-launch-recipes"></a>

<a id="qualified-model-profiles-and-launch-recipes"></a>

## Model profiles and launch recipes

The two profiles below are equally supported with **524,288-token context,
HiCache and NIXL** on one RTX PRO 6000.

I personally run Flash-Next at **C=6 (six concurrent requests)**, with
**524,288-token context** and an observed token pool of **1,039,040 tokens**.

| Profile | Precision | Speculative decoding | Reference models | Run |
|---|---|---|---|---|
| **Qwen3.8 Flash-Next** | NVFP4 target with FP8 KV | Native NEXTN MTP with FR-Spec | [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4); no separate draft | [Recommended FR-Spec recipe](RUN.md#launch-flash-next-with-fr-spec) or [non-FR alternative](RUN.md#launch-flash-next-without-fr-spec-alternative) |
| **Qwen3.8-27B** | FP8 target and KV | DFlash2 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) with [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) | [27B/DFlash2 recipe](RUN.md#launch-27b-with-dflash2) |

For Flash-Next, our [OrcaRouter Uncensored ModelOpt NVFP4 conversion](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4)
is also available by community request and works with the same Pennyroyal launch recipe.

**Swift 27B FP8 works with DFlash2.** It passed the reasoning suite and used
fewer tokens overall in our comparison.
See [Swift 27B validation and per-case timings](SWIFT-27B.md).

[Online FP8](FP8.md) is selected automatically for eligible Flash-Next
launches on exact SM120; [NVMe-backed PLE](NVME-PLE.md) stays an independent
opt-in.

The public 27B measurements used
[orcarouter/Qwen3.8-27B-Uncensored-FP8](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8),
an uncensored/abliterated derivative of the official checkpoint.

FR-Spec credit goes directly to Gabriel's
[`gabrielolympie/sglang-flashnext-sm120`](https://github.com/gabrielolympie/sglang-flashnext-sm120).
Both recipes pin the unmodified
[Froggeric v22.5 template](configs/pennyroyal/templates/README.md).

The v2.5.0 online-FP8 implementation is directly inspired by
[`mratsim/sglang-qwen38fn-sm120-turbo`](https://github.com/mratsim/sglang-qwen38fn-sm120-turbo/tree/94a68214b77514bc26ef78cee4c01f128162d09b)
at commit `94a68214b77514bc26ef78cee4c01f128162d09b`, especially patches
`0003`, `0007`, and `0008`. Optional NVMe PLE adapts Garner McCloud's
[SSD Stream v0.2.0](https://github.com/garnermccloud/sglang-ssd-stream/tree/176a522ef9d6dbb5056ae1f467fe49af0f1258a5),
with the upstream license and NOTICE retained. Implementation details and credit
are in [FP8.md](FP8.md#credit) and [NVME-PLE.md](NVME-PLE.md#source-and-credit).

<a id="current-release--pennyroyal-v251"></a>

## Current release — Pennyroyal v2.5.3

**v2.5.3 fixes the Flash-Next checkpoint bug and includes the BF16 NVMe PLE hotfix.**

- Corrects the native MTP checkpoint selection behind the phantom-token and
  file-writing problem reported in [issue #17](https://github.com/jpezzulli/sglang-rtxpro6000/issues/17).
- Includes the fix for Swift's BF16 NVMe PLE staging error, so v2.5.3 does not
  need the separate patch.
- Streams long string tool arguments as they are generated, improves cache
  persistence, and adds startup warnings when the cache filesystem is under pressure.
- Adds parallel encoding for eligible long prompts and a small DFlash2
  prefill-copy improvement.

The two model profiles, setup utility and existing performance tables remain.
See [changes and upstream credits](CHANGES.md#v253--agentic-correctness-and-cache-maintenance)
or [get started](docker/pennyroyal/README.md).

<a id="current-release--pennyroyal-v250"></a>

<a id="v250-options--online-fp8-and-nvme-ple"></a>

## Flash-Next precision and placement

- **[Online FP8](FP8.md)** converts eligible weights during model loading and
  engages automatically on exact SM120; a saved private flag opts out. The
  measured decode improvements and precision tradeoffs are below.
- **[NVMe-backed PLE](NVME-PLE.md)** moves the large PLE table from fixed
  host RAM to a prepared SSD snapshot. It saves RAM, with a throughput cost
  in our repeated concurrent-request comparison.

Either option can be used on its own. Both are off by default; the linked
guides explain how to enable them and what was tested. For request capacity
and media preprocessing, see the [launch options](RUN.md).

<a id="performance-and-qualification"></a>
<a id="performance-and-testing"></a>

## Performance

Here are the latest published benchmarks for each model on one RTX PRO 6000.
Both use 524,288-token context with HiCache and NIXL enabled.

<a id="flash-next-v250-online-fp8-performance-at-a-glance"></a>

### Qwen3.8 Flash-Next NVFP4 with online FP8

Measured September 11, 2026, on v2.5.0 with FR-Spec, RAM-backed PLE and an
824,384-token KV pool, using the RadixArk ModelOpt NVFP4 checkpoint.

| Single-request workload | Online FP8 off | Online FP8 on | Improvement |
|---|---:|---:|---:|
| Short, 1,024 output tokens | 161.47 tok/s | **207.12 tok/s** | **+28.3%** |
| 128K context, 1,024 output tokens | 154.70 tok/s | **195.63 tok/s** | **+26.5%** |
| 490K context, 1,024 output tokens | 149.14 tok/s | **172.64 tok/s** | **+15.8%** |

These are decode rates after the first token. The short result is a three-run
median; the long-context rows are single runs. Online FP8 also freed about
**3.86 GiB** after graph capture. Eligible launches get it automatically; see [online FP8](FP8.md).

<a id="27b-fp8--dflash2-performance-at-a-glance"></a>

### Qwen3.8-27B FP8 with DFlash2

Measured September 10, 2026, on v2.4.1 with 1,118,784-token target and draft
KV pools, using the OrcaRouter FP8 checkpoint linked above.

| Workload | Speed | Result |
|---|---:|---|
| 64K cold prefill — 63,888 input tokens | **6,570 tok/s** | 10.468 s to first token; exact READY |
| 490K cold prefill — 489,903 input tokens | **1,646 tok/s** | 302.320 s to first token; all three needles found |
| Single-request decode — 1,024 output tokens | **108.31 tok/s** | After the first token |
| Four simultaneous requests — 1,024 output tokens each | **374.98 tok/s aggregate** | Combined throughput, including time to first token |

Prefill uses the server's initial-prefill time. Decode results are three-run
medians; the four-request rate measures the whole group through its last response.

<a id="retained-v240-flash-next-prefill-measurements"></a>
<a id="qwen38-27b-fp8dflash2--august-24-2026-campaign"></a>
<a id="27b-real-agentic-context-behavior"></a>
<a id="earlier-releases"></a>
<a id="v23--faster-flash-next-decoding-with-fr-spec"></a>
<a id="previous-v212--maintenance-release"></a>

**[Benchmark history and detailed results](RESULTS.md)** includes earlier
releases, prefill comparisons, real agentic workloads, and cache-restoration
results for both models.

<a id="independent-tp2-fp8-validation"></a>

**[Community results](COMMUNITY-RESULTS.md)** covers other users' setups,
including Max-Q and two-GPU runs. Thanks to u/StockSpecialist1707 for the
1.11M-token KV-pool testing and H3PO for the TP2 results.

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
- Current release: [`pennyroyal-v2.5.3`](https://github.com/jpezzulli/sglang-rtxpro6000/releases/tag/pennyroyal-v2.5.3)

A short note distinguishing the Pennyroyal source or technique from your own
changes helps readers reproduce the lineage, understand what you improved, and
find both projects.

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

## HiCache/NIXL persistence

HiCache/NIXL provides reusable prefix state across GPU/host eviction and
service restart. GPU-resident decode remains on the normal model path:

```text
GPU radix state
  -> page-first HiCache host RAM, kernel I/O, write-through
  -> NIXL POSIX FILE storage, io_uring + O_DIRECT
```

For restart and cache-restoration measurements on both models, see
[the benchmark history](RESULTS.md). [Memory and persistence](MEMORY-AND-PERSISTENCE.md)
explains cache namespaces and how restoration works.

## Build and launch

Choose one installation path:

1. **[Docker and Compose](docker/pennyroyal/README.md)** for the prebuilt image.
2. **[Native installation](BUILD.md)** to build and run from source.

The manual setup and launch instructions in both guides are confirmed working.
Follow your chosen guide to configure and start the server. Run one model
profile at a time on a single GPU.

The optional **[beta configurator](CONFIGURE.md)** is open for testing. It
walks through model, GPU, and cache settings and generates a launch command.
If you try it, please [report any issues](https://github.com/jpezzulli/sglang-rtxpro6000/issues).

## Source changes and upstream work

[CHANGES.md](CHANGES.md) records the release history, individual fixes,
upstream PRs, and tests. [PROVENANCE.md](PROVENANCE.md) records
the source lineage. These include the DFlash2 and native-MTP integrations,
SM120 kernels, cache restoration, parser fixes, and memory work.

<a id="scope-and-limitations"></a>

## Choosing your settings

Online FP8 improved decode speed in our tests; cold long-context time to first
token stayed about the same. NVMe PLE saves host RAM, but was slower than RAM
PLE with concurrent requests. The linked guides explain how to choose and
enable each option.

See [RESULTS.md](RESULTS.md) for measurements, [Community results](COMMUNITY-RESULTS.md)
for other users' setups, and [LIMITATIONS.md](LIMITATIONS.md) for known issues.

## Documentation map

- [BUILD.md](BUILD.md) — one native build and dependency identity.
- [RUN.md](RUN.md) — native launch paths for both profiles, the non-FR
  Flash-Next alternative, startup checks, and optional settings.
- [CONFIGURE.md](CONFIGURE.md) — what the beta setup configurator does, how to
  start it for native and container use, and where its settings are saved.
- [FP8.md](FP8.md) — how online FP8 selects itself, understand its precision
  changes, and see the results.
- [NVME-PLE.md](NVME-PLE.md) — optional reader installation, overlay
  preparation, launch settings and measured RAM/speed tradeoff.
- [BACKENDS.md](BACKENDS.md) — source-backed resolved backend and SM120 tables.
- [RESULTS.md](RESULTS.md) — controlled, reasoning, tools, context, agentic, and
  persistence measurements for both models.
- [MEMORY-AND-PERSISTENCE.md](MEMORY-AND-PERSISTENCE.md) — GPU allocation and
  hybrid state representation.
- [CHANGES.md](CHANGES.md) — cumulative release and upstream history.
- [PROVENANCE.md](PROVENANCE.md) — source, package, checkpoint, and tag identity.
- [LIMITATIONS.md](LIMITATIONS.md) — known issues and hardware considerations.
- [`jpezzulli/pennyroyal-validation`](https://github.com/jpezzulli/pennyroyal-validation)
  — maintained public validation harness and result catalog.

## License

The SGLang-derived source remains under Apache-2.0. Model checkpoints and
third-party runtimes retain their own licenses.
