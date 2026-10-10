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

**524K context · FR-Spec + adaptive MTP · Persistent conversation cache**

[Get started](#build-and-launch) · [Performance](#performance) ·
[3.0 release notes](RELEASE_NOTES_3.0.md) · [Community results](COMMUNITY-RESULTS.md)

## What's new in 3.0

Flash-Next gets adaptive MTP at **C=1**, automatic FP8 and small-batch kernels,
and more room for cached tokens. There are also fixes for session state, agent
API responses, image requests, and interrupted builds.

**Container configuration now lives on your host.** Edit your startup file,
model paths and mounts, then run the downloaded image. Updating the image does
not replace your configuration.

**The release went on a diet.** We kept RTX Blackwell for inference and
Ampere/Ada Lovelace for image-processing sidecars, and trimmed the other GPU
build targets. The result is a slimmer release for easier downloads. CPU image
processing remains available.

**Penny Royal did almost all the coding for this release.** John, Penny and
Codex worked through the architecture, tuning and debugging together, down to
fused kernels, memory layouts and milliseconds per decode cycle.
See the [draft release notes](RELEASE_NOTES_3.0.md) for the full change list.

<a id="models-and-launch-recipes"></a>
<a id="qualified-model-profiles-and-launch-recipes"></a>

## Two model profiles

Both profiles support **524,288-token context**, tool calling, reasoning,
images, RAM prefix caching, and optional NIXL disk caching.

| Model | Weights | Speculative decoding | Checkpoints |
|---|---|---|---|
| **Qwen3.8 Flash-Next** | NVFP4 experts, automatic FP8 dense layers on SM120 | Native NEXTN MTP with FR-Spec; adaptive four/eight-token windows | [Our OrcaRouter ModelOpt conversion](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4) · [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) |
| **Qwen3.8-27B** | FP8 | DFlash2, with a separate draft model | [Qwen FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) · [DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) |

I personally run Flash-Next on **one RTX PRO 6000 at C=6**, with **524,288-token
context** and an observed **1,067,072-token GPU pool**. That pool is shared
across active requests and cached prefixes.

### How adaptive MTP works

Flash-Next's MTP head is part of the checkpoint; no separate draft model is
loaded. **Only C=1 uses adaptive draft width.**

| Requests generating together | Draft behavior |
|---|---|
| **C=1** | Switch between four- and eight-token windows as draft acceptance changes. |
| **C=2 or more** | Fixed four-token windows; speculative decoding stays on. |

This lets one request benefit from wider drafts without making every request
in a busy batch pay that extra draft work. Configuring capacity for C=6 does
not disable adaptation: a lone active request can still use it. **C is request
concurrency; TP is the number of inference GPUs.**

**FR-Spec remains enabled alongside adaptive MTP.** FR-Spec narrows the
vocabulary searched by the draft head; adaptation changes how many tokens it
drafts. At C2 and above, FR-Spec stays on while draft width stays at four.
The target model keeps its full vocabulary. Use the
[Flash-Next FR-Spec recipe](RUN.md#launch-flash-next-with-fr-spec); the
[non-FR recipe](RUN.md#launch-flash-next-without-fr-spec-alternative) is the
optional alternative.

For 27B alternatives, see the [Swift 27B results](SWIFT-27B.md) and the
[OrcaRouter FP8 checkpoint](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8).

<a id="performance-and-qualification"></a>
<a id="performance-and-testing"></a>
<a id="flash-next-v250-online-fp8-performance-at-a-glance"></a>

## Performance

### Flash-Next 3.0

**Real coding: 262 tok/s mean, 518 peak. Presentation workflow: 240 mean, 544 peak.**

Measured on one RTX PRO 6000 Blackwell, using our OrcaRouter ModelOpt NVFP4
checkpoint, FR-Spec plus adaptive MTP, a 524K context limit, and normal non-greedy sampling.

| Workload | Mean tok/s | Sampled peak tok/s |
|---|---:|---:|
| Real coding workflow | **262.32** | **518.18** |
| Presentation workflow | **239.53** | **543.72** |
| Single request | 218.63 | 326.56 |
| Four concurrent requests, aggregate | 489.35 | 636.71 |
| Six concurrent requests, aggregate | 433.54 | 606.48 |

These are **SGLang's native generation rates**, including reasoning. Prefill,
tool execution, and idle time are separate. The coding task ran for 21 model
turns; the presentation task ran for 113. The concurrent rows are combined
throughput, not the speed of each request.

| Cold prompt | Mean prefill tok/s | Time to first token |
|---|---:|---:|
| 64K | **17,235.90** | 4.71 s |
| 490K | **8,839.75** | 60.37 s |

The reasoning suite scored **95.69/100**. See [results and measurement settings](RESULTS.md#penny-royal-30)
for the tool checks, cache restore, sampling details, and earlier results.

### Compared with 2.5.3

Our saved native single-request runs went from **186.79 to 218.63 tok/s mean**,
a **17.0% higher recorded rate**. The GPU token pool grew from **1,039,040 to
1,067,072 tokens** at the same 524K context limit and C6 capacity.

Those runs used each release's recorded settings: medium reasoning in the
October 6 baseline and xhigh in the October 10 release run. [Comparison details](RESULTS.md#recorded-comparison-with-253)
keep the sampling and source records together.

### The public code-edit benchmark

For comparison with the token-maxxing results, we also ran the public
**code-edit prompt**: **491.88 tok/s mean and 574.16 sampled peak** in the
accepted adaptive screen, using normal non-greedy sampling. The saved 2.5.3
code-edit set averaged **336.57 tok/s** with greedy sampling.

This workload asks for a mechanical rewrite of supplied code. Its predictable
output is well suited to wider speculation. The **real coding workflow above**
is a separate task with tool calls, implementation and tests. [Code-edit details](RESULTS.md#public-code-edit-benchmark)
identify the prompt, output limit and measured configuration.

<a id="27b-fp8--dflash2-performance-at-a-glance"></a>

### Qwen3.8-27B FP8 with DFlash2

Latest recorded performance, from the September 10 v2.4.1 run on one RTX PRO
6000 with the OrcaRouter FP8 checkpoint:

| Workload | Recorded speed |
|---|---:|
| 64K cold prefill | 6,570 tok/s |
| 490K cold prefill | 1,646 tok/s |
| Single-request decode, after the first token | 108.31 tok/s |
| Four requests, aggregate including time to first token | 374.98 tok/s |

[Benchmark history](RESULTS.md) keeps these dated measurements and their
original timing methods.

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
if you try it.

### Your container configuration stays outside the image

```text
your-host/
  run.sh                         image, GPU selection, ports and mounts
  config/start-flash-next-frspec.sh  model and SGLang startup settings
  config/nixl-posix-frspec.toml      optional disk-cache settings
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
extra tensor-parallel inference card. CPU preprocessing is also available.

| Profile | Default host cache | Additional model-related host memory |
|---|---|---|
| Flash-Next | 32 GB HiCache | About 48 GiB for RAM-backed PLE |
| 27B/DFlash2 | 96 GB HiCache | Loading and runtime allocations |

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
[Penny Royal](mailto:Pennyroyal@agentmail.to) did almost all the coding for 3.0,
including the ports and integration work. [John](https://github.com/jpezzulli),
Penny and Codex worked through the design, implementation, tuning and debugging
together, with plenty of back and forth over what was actually worth keeping.

## Documentation

| I want to… | Read |
|---|---|
| Install or upgrade | [Native build](BUILD.md) · [Container](docker/pennyroyal/README.md) |
| Choose launch, cache, GPU, and context settings | [Run guide](RUN.md) |
| Try guided setup | [Beta configurator](CONFIGURE.md) |
| Understand the FP8 and SSD options | [Online FP8](FP8.md) · [NVMe PLE](NVME-PLE.md) |
| See measurements | [Results and history](RESULTS.md) · [Community reports](COMMUNITY-RESULTS.md) |
| Read the implementation details | [Backends](BACKENDS.md) · [Memory and persistence](MEMORY-AND-PERSISTENCE.md) |
| Check changes and source lineage | [3.0 notes](RELEASE_NOTES_3.0.md) · [Machine-readable notes](release-3.0.json) · [History](CHANGES.md) · [Provenance](PROVENANCE.md) |
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

[Current release notes](RELEASE_NOTES_3.0.md) · [Benchmark history](RESULTS.md) ·
[Configuration reference](BACKENDS.md) · [Known issues](LIMITATIONS.md)

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
