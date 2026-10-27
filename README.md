<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/penny-royal/logos/logo-horizontal-dark.svg">
    <img src="assets/brand/penny-royal/logos/logo-horizontal-light.svg" alt="Penny Royal" width="480">
  </picture>
</p>

# Penny Royal — Qwen3.8 on RTX PRO 6000

Run long conversations, coding agents, and tool-heavy workflows on your own
hardware. Penny Royal is an SGLang-derived runtime tuned for **NVIDIA RTX PRO
6000 Blackwell (SM120), with one card (TP1) or two (TP2)**.

**Persistent conversation caching is at the heart of Penny.** HiCache and NIXL
keep reusable state in system RAM and on NVMe, beyond what fits on the GPU,
so long-running sessions can pick up from cached history.

**524K context · HiCache + NIXL · FR-Spec · FlashInfer verification · DFlash2**

[Get started](#build-and-launch) · [Conversation cache](#hicache-and-nixl) ·
[Performance](#performance) ·
[3.1.3 release notes](RELEASE_NOTES_3.1.3.md) · [Community results](COMMUNITY-RESULTS.md)

**Penny Royal 3.1:** Flash-Next runs native NEXTN MTP with FR-Spec as the
maintained default profile, with automatic KV pool sizing replacing the preset
cap; Flash-Next DFlash2 with a separate draft model, fixed concurrency
width routing (block 6 for one request, block 4 for two or more) and
FlashInfer draft-token verification is an **experimental / alpha** profile,
fully available to opt into.

**3.1.3:** fixes reasoning controls, cache restores, scheduling and `fixed-w4` correctness. [Release notes](RELEASE_NOTES_3.1.3.md).

<a id="dflash2-maturity"></a>

> **DFlash2:** 27B is established and qualified; Flash-Next DFlash2 is alpha. Both need extra host RAM for the draft cache — **24.83 GB extra** in our 27B run. [RAM guidance](CONFIGURATION.md#capacity-and-memory).

<a id="hicache-and-nixl"></a>

## Built for long conversations and agents

Every tool call adds another turn. A coding session can build up hundreds of
thousands of tokens of history, and processing that history again costs time.

Penny keeps reusable conversation state across three layers:

| Where | What it does |
|---|---|
| **GPU memory** | Keeps active conversations ready for the next turn. |
| **System RAM — HiCache** | Holds reusable state beyond what fits on the GPU and brings it back when needed. |
| **NVMe storage — NIXL** | Keeps cached state on disk so matching conversation history can be restored, including after a server restart with the same configuration. |

When your client sends the next turn, Penny reuses the matching cached history
and processes the new material.

Qwen's hybrid models need their recurrent state restored alongside the
attention cache. Penny handles that complete state, including Flash-Next's
additional PLE and QSA state, so restored conversations can continue correctly.

Penny restored a **490K-token conversation** from cache and processed just
**five new tokens**, skipping almost half a million tokens of repeated prefill.

Both model profiles use this hierarchy. **NIXL disk caching is on by default**;
you can disable it while keeping GPU and RAM caching. The
[run guide](RUN.md#choose-hicache-ram-size) covers sizing and setup, and
[memory and persistence](MEMORY-AND-PERSISTENCE.md#hicache-and-nixl) explains
how the state is stored and restored.

## What's new in 3.1

Flash-Next keeps **native NEXTN MTP with FR-Spec as the maintained default**,
and its verify width is now **fixed by active concurrency** - block 8 for a
single running request, block 4 for two or more - so the width is identical at
every boot. **DFlash2 with FR-Spec** and its own draft model,
`incoai/Qwen3.8-Flash-Next-DFlash2`, is an **experimental / alpha** opt-in
profile with the same concurrency-routed widths (block 6 at one request, block
4 at two or more), and it now verifies accepted draft tokens with
**FlashInfer's SM120 kernel**.
The **KV pool is sized automatically** from profiled memory instead of a preset
cap; an explicit `MAX_TOTAL_TOKENS` remains available. Scheduler admission,
stop-regex inside speculative blocks and mixed-probability requests are fixed.
See the
[3.1 release notes](RELEASE_NOTES_3.1.md) for the full change list and the
[configuration reference](CONFIGURATION.md) for every supported setting.

<a id="models-and-launch-recipes"></a>
<a id="qualified-model-profiles-and-launch-recipes"></a>

## Two model profiles

Both profiles support **524,288-token context**, tool calling, reasoning,
images, HiCache RAM prefix caching, and optional NIXL disk caching.

| Model | Weights | Speculative decoding | Checkpoints |
|---|---|---|---|
| **Qwen3.8 Flash-Next** | NVFP4 experts, automatic FP8 dense layers on SM120 | Maintained native NEXTN MTP with FR-Spec is the default, width routed by batch size (block 8 at C1, block 4 at C2+); **experimental / alpha** DFlash2 opt-in profile (separate draft model, FR-Spec, block 6 at C1 / block 4 at C2+) verifies draft tokens with FlashInfer | [Our OrcaRouter ModelOpt conversion](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4) · [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) · [Flash-Next DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) |
| **Qwen3.8-27B** | FP8 | DFlash2 (established and qualified integration), with a separate draft model at fixed block 8 | [Qwen FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) · [DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) |

I personally run Flash-Next on **one RTX PRO 6000 at C=6**, with **524,288-token
context** and the KV pool sized automatically from profiled memory; the final
3.1 measurements resolved a shared **target KV pool** of **1,081,792 tokens**
on the maintained native-MTP profile and **1,210,944 tokens** on the optional
DFlash2 profile of this system. The profile reports the resolved number at
startup.
That pool is shared across active requests and cached prefixes.

### How draft width is routed

Both Flash-Next speculation paths read their verification width from a bundled
fixed policy, keyed only by how many requests are generating together: the
maintained native-MTP recipes from `configs/pennyroyal/adaptive-next.json`,
and the optional DFlash2 profile (NVFP4 target with the separate
`incoai/Qwen3.8-Flash-Next-DFlash2` draft and FR-Spec) from
`configs/pennyroyal/adaptive-dflash2.json`:

| Requests generating together | Native MTP (`next`, `next-plain`) | DFlash2 (`next-dflash2`, optional) |
|---|---|---|
| **C=1** | Fixed eight-token blocks. | Fixed six-token blocks. |
| **C=2 or more** | Fixed four-token blocks. | Fixed four-token blocks. |

One candidate width per tier means the width is a lookup, so it is identical
between boots, and a single request gets the wider draft without making every
request in a busy batch pay that extra draft work. On the maintained native-MTP
recipes that policy is a choice — `PENNY_MTP_POLICY=tiered` (the table above),
`adaptive` (the v3.0.2 policy, which may move between four and eight tokens at
one request) or `fixed-w4` (four tokens at every concurrency) — and it is one
setting on every launch route: `./configure-penny` saves it, a launch folder
carries it in its startup script, and
`PENNY_MTP_POLICY=fixed-w4 ./run-penny` or the same variable in front of
`docker compose up` decides it while nothing is saved
([details](CONFIGURATION.md#speculation-width-policy)). **C is request concurrency;
TP is the number of inference GPUs.**

**FR-Spec rides along at every concurrency.** FR-Spec narrows the vocabulary
searched by the draft head; routing selects how many tokens it drafts. The
target model keeps its full vocabulary. The maintained default is the
[FR-Spec native-MTP recipe](RUN.md#launch-flash-next-with-fr-spec), with the
[non-FR recipe](RUN.md#launch-flash-next-without-fr-spec-alternative) also
selectable - without the map, and with the same block 8 / block 4 routing; the
experimental / alpha [DFlash2 recipe](RUN.md#launch-flash-next-with-dflash2) -
which verifies its draft tokens with FlashInfer's SM120 kernel - remains
available as an opt-in. DFlash2 offers more token-pool capacity; we are still
evaluating performance across concurrency levels. TP2 for the DFlash2
profile is expected to work; it has not been tested on two-GPU hardware.

For 27B alternatives, see the [Swift 27B results](SWIFT-27B.md) and the
[OrcaRouter FP8 checkpoint](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8).

<a id="performance-and-qualification"></a>
<a id="performance-and-testing"></a>
<a id="flash-next-v250-online-fp8-performance-at-a-glance"></a>

## Performance

### Flash-Next 3.1

**Native MTP default: 252.83 tok/s mean / 501.63 peak on a real coding
workflow, and 216.23 / 525.80 on the ten-slide presentation (agentic), single
request at block 8.**

Measured on one RTX PRO 6000 Blackwell, using our OrcaRouter ModelOpt NVFP4
checkpoint, FR-Spec, a 524K context limit, and normal non-greedy sampling. The
native rows are the final October 21 screens on the merged source
(`fe5f02c`); the DFlash2 rows keep their `a60f442` labels. Every rate is
SGLang's native generation rate, including reasoning; prefill, tool execution,
and idle time are separate.

| Profile / workload | Mean tok/s | Sampled peak tok/s |
|---|---:|---:|
| Native MTP (maintained default), single request, C1 W8 | **197.87** | **345.30** |
| Native MTP, full four-request batch, aggregate, C4 W4 | **555.11** | **731.06** |
| Native MTP, full six-request batch, aggregate, C6 W4 | **472.79** | **528.49** |
| Native MTP, coding workflow (agentic), C1 W8 | **252.83** | **501.63** |
| Native MTP, presentation workflow (agentic), C1 W8 | **216.23** | **525.80** |
| DFlash2 + FlashInfer (optional profile), single request, C1 W6 | **241.49** | **339.53** |
| DFlash2 + FlashInfer, six concurrent requests, aggregate, C6 W4 | **504.80** | **568.90** |
| DFlash2 + FlashInfer, presentation workflow (agentic), C1 W6 | **217.19** | **448.70** |

Both native workflows completed end to end at block 8 with one request
generating, and the finished presentation deck previewed.
[Current results](RESULTS.md#penny-royal-31) records the sources, intervals
and settings behind every number, plus the dated earlier screens.

| Cold prompt (Flash-Next DFlash2) | Mean prefill tok/s |
|---|---:|
| 63,906 tokens | **15,365.38** |
| 489,921 tokens | **8,715.05** |

The Flash-Next DFlash2 reasoning qualification scored **93.46/100**. See
[results and measurement settings](RESULTS.md#penny-royal-31)
for the tool checks, cache restore, sampling details, and earlier results.

[RESULTS.md](RESULTS.md#penny-royal-313-october-27-2026) adds the dated decode
and time-to-first-token rows measured on the 3.1.3 candidate on both qualified
profiles, with the launched configuration and interval counts. The Flash-Next
rows were measured with `adaptive` selected locally while the public width-policy
default stays `tiered`, so they are not a matched-baseline comparison and carry
no speed-up claim.

<a id="27b-fp8--dflash2-performance-at-a-glance"></a>

### Qwen3.8-27B FP8 with DFlash2

Current recorded performance on one RTX PRO 6000, fixed block 8:

| Workload | Mean tok/s | Sampled peak tok/s |
|---|---:|---:|
| Single request | **133.06** | **232.07** |
| Four concurrent requests, aggregate | **459.74** | **570.73** |

| Cold prompt | Mean prefill tok/s |
|---|---:|
| 63,906 tokens | 6,421.59 |
| 489,921 tokens | 1,677.10 |

The reasoning suite scored **96.94/100** on this profile.

[Benchmark history](RESULTS.md) keeps the earlier records — the October 10 3.0
screens, the 2.5.3 comparison, the public code-edit benchmark, and the dated
27B campaigns — with their original labels and timing methods.

<a id="independent-tp2-fp8-validation"></a>

Running two cards? [Community results](COMMUNITY-RESULTS.md) covers TP2 setups
and measurements from other users.

## Build and launch

Choose the installation path that suits your machine:

| Path | What you do |
|---|---|
| **[Prebuilt container](docker/pennyroyal/README.md)** | Download the image and launch files, edit your host configuration, and run it. Models and caches stay on the host. |
| **[Native installation](BUILD.md)** | Build the runtime, set your model and cache paths, and use the [native startup recipe](RUN.md). |

Both paths work with ordinary configuration files. The optional
**[beta configurator](CONFIGURE.md)** helps write those files and prints the
launch command. It is open for testing; please [report issues](https://github.com/jpezzulli/sglang-rtxpro6000/issues)
if you try it. The complete list of supported settings, defaults, interactions
and restart/cache effects is the [configuration reference](CONFIGURATION.md).

### Your container configuration stays outside the image

```text
your-host/
  run.sh                              image, GPU selection, ports and mounts
  config/start-flash-next-frspec.sh   model and SGLang startup settings
  config/nixl-posix-frspec.toml        optional disk-cache settings
```

Prepare those files before starting. `run.sh` starts Docker and mounts your
configuration directory read-only at `/config`; the container runs the startup
file from that mount. Models and writable caches are mounted separately.
Edit the host files and restart to apply changes. You can download a newer
image while keeping your own configuration—no container rebuild required.

The [container guide](docker/pennyroyal/README.md) provides the files and
commands; the beta configurator can generate the same ordinary files for you.

### Hardware and memory

The inference target is **RTX PRO 6000 Blackwell, TP1 or TP2**. An Ampere or
Ada Lovelace GPU can be used separately for image preprocessing; it is not an
extra tensor-parallel inference card. CPU preprocessing is also available. The
build is narrowed to RTX Blackwell inference and Ampere/Ada sidecar image
processing — other GPU build targets are trimmed, which keeps the release
slimmer and easier to download.

| Profile | Default host cache | Additional model-related host memory |
|---|---|---|
| Flash-Next | 32 GB HiCache | About 48 GiB for RAM-backed PLE |
| Flash-Next/DFlash2 | 32 GB HiCache | The draft's own host cache, on top of the target's budget |
| 27B/DFlash2 | 96 GB HiCache | Draft cache plus loading and runtime allocations |

A DFlash2 draft keeps a host cache of its own that is **additional to** the
HiCache figure above, so size host RAM from what a boot reports rather than
from the HiCache setting alone. The recorded 27B launch spent 79.45 GB of
target KV plus 16.63 GB of Mamba state - the 96 GB target/Mamba budget - and a
further 24.83 GB of draft sliding-window cache, about 120.91 GB of caches
before other process memory. This release documents that bill; it does not
reduce it. See [limitations](LIMITATIONS.md) and the
[configuration reference](CONFIGURATION.md#capacity-and-memory).

Leave room for the operating system and model loading. You can
[change the RAM cache size](RUN.md#choose-hicache-ram-size), or move Flash-Next's
PLE table to an SSD with [NVMe PLE](NVME-PLE.md).

NIXL disk caching is **on by default and optional**. Switching it off keeps GPU
and RAM prefix caching. NVMe PLE is a separate choice. The [launch guide](RUN.md)
explains cache sizing, GPU selection, concurrency, and startup.

The first start compiles kernels and captures CUDA graphs. Wait for the
server-ready message, then [check the API](RUN.md#smoke-through-the-normal-api).

## Built for the way we use it

Penny Royal started as my home runtime on thegrid. I wanted long conversations
and useful coding agents on the hardware I already owned. The tuning follows
that workload: keep several requests moving, make tool calls usable, and avoid
reprocessing a long conversation when its state can be restored.

The name comes from Penny Royal in Neal Asher's Polity books.

Thanks to **[aiueo52](https://github.com/aiueo52)** for the Flash-Next optimization
work and discussion, **[LandOfLemons](https://github.com/LandOfLemons)** for the
session-cache and WSL2 contributions, **[cube4elements](https://github.com/cube4elements)**
for the image-request crash fix, and **[untcoder2](https://github.com/untcoder2)**,
**[StockSpecialist1707](https://www.reddit.com/user/StockSpecialist1707/)**, and
**[palves](https://github.com/palves)** for field testing and setup feedback.
For 3.1, **[@jhsmith409](https://github.com/jhsmith409)** and
**[@LocalhostRonin](https://github.com/LocalhostRonin)** contributed directly.
[Penny Royal](mailto:Pennyroyal@agentmail.to) did almost all the coding across
these releases, including the ports and integration work. [John](https://github.com/jpezzulli),
Penny and Codex worked through the architecture, implementation, tuning,
verification-backend choice and debugging together, with plenty of back and forth
over what was actually worth keeping.

## Documentation

| I want to… | Read |
|---|---|
| Install or upgrade | [Native build](BUILD.md) · [Container](docker/pennyroyal/README.md) |
| Choose launch, cache, GPU, and context settings | [Run guide](RUN.md) · [Configuration reference](CONFIGURATION.md) |
| Understand conversation caching | [HiCache and NIXL](#hicache-and-nixl) · [Memory and persistence](MEMORY-AND-PERSISTENCE.md#hicache-and-nixl) |
| Try guided setup | [Beta configurator](CONFIGURE.md) |
| Understand the FP8 and SSD options | [Online FP8](FP8.md) · [NVMe PLE](NVME-PLE.md) |
| See measurements | [Results and history](RESULTS.md) · [Community reports](COMMUNITY-RESULTS.md) |
| Read the implementation details | [Backends](BACKENDS.md) · [Memory and persistence](MEMORY-AND-PERSISTENCE.md) |
| Check changes and source lineage | [3.1.3 notes](RELEASE_NOTES_3.1.3.md) · [Machine-readable notes](release-3.1.3.json) · [History](CHANGES.md) · [Provenance](PROVENANCE.md) |
| Check known issues | [Limitations](LIMITATIONS.md) |
| Use an AI assistant for setup | [llms.txt](llms.txt) |

Our [validation suite](https://github.com/jpezzulli/pennyroyal-validation)
contains the reasoning, tool, image, context, and performance tests.

To hear about new releases, choose **Watch → Custom → Releases** on GitHub.
If you build on Penny Royal, please link this repository and the version you
started from. Feedback, contributions, and real workload reports are welcome.

<p align="center">
  <img src="assets/brand/penny-royal/artwork/penny-royal-crystal-transparent.png" alt="Penny Royal crystal artwork" width="240">
</p>

## License

The SGLang-derived source is Apache-2.0. Model checkpoints and third-party
components retain their own licenses.

<!-- Preserve existing README links while detailed material lives on its own page. -->
<a id="current-release--pennyroyal-v251"></a>
<a id="current-release--pennyroyal-v250"></a>
<a id="current-release--pennyroyal-v253"></a>
<a id="v250-options--online-fp8-and-nvme-ple"></a>
<a id="retained-v240-flash-next-prefill-measurements"></a>
<a id="qwen38-27b-fp8dflash2--august-24-2026-campaign"></a>
<a id="27b-real-agentic-context-behavior"></a>
<a id="earlier-releases"></a>
<a id="v23--faster-flash-next-decoding-with-fr-spec"></a>
<a id="previous-v212--maintenance-release"></a>
<a id="qualified-configuration-matrix"></a>
<a id="resolved-backends-qwen38-27bdflash2"></a>
<a id="resolved-backends-qwen38-flash-next"></a>
<a id="sm120-enablement-was-scoped-not-indiscriminate"></a>
<a id="how-pennyroyal-selects-sm120-paths"></a>
<a id="memory-recovery-and-automatic-kv-sizing"></a>
<a id="scope-and-limitations"></a>
<a id="documentation-map"></a>

[Current release notes](RELEASE_NOTES_3.1.3.md) · [Benchmark history](RESULTS.md) ·
[Configuration reference](CONFIGURATION.md) · [Known issues](LIMITATIONS.md)

<!-- Compatibility anchors for earlier versions of this guide. -->
<a id="broader-model-support-without-hicachenixl"></a>
<a id="building-on-pennyroyal"></a>
<a id="choosing-your-settings"></a>
<a id="flash-next-precision-and-placement"></a>
<a id="hardware-storage-and-first-start-expectations"></a>
<a id="hicachenixl-persistence"></a>
<a id="model-profiles-and-launch-recipes"></a>
<a id="prebuilt-docker-image-for-qwen38-on-rtx-pro-6000"></a>
<a id="qwen38-flash-next-nvfp4-with-online-fp8"></a>
<a id="sglang-for-qwen38-on-one-nvidia-rtx-pro-6000-blackwell-sm120"></a>
<a id="source-changes-and-upstream-work"></a>
<a id="technical-reference"></a>
<a id="validation-suite"></a>
<a id="why-this-runtime-exists"></a>
