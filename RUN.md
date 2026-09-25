# Run Qwen3.8 with SGLang on one RTX PRO 6000 Blackwell

This is the native launch guide for Pennyroyal. The same source runs
Qwen3.8-27B FP8 with DFlash2 or Qwen3.8 Flash-Next NVFP4 with native NEXTN and
FR-Spec on one NVIDIA RTX PRO 6000 Blackwell 96 GB GPU (SM120). Both profiles
serve an OpenAI-compatible model named `pennyroyal`.

Build the release with [BUILD.md](BUILD.md) first. Container users should
follow the separate [Docker and Compose guide](docker/pennyroyal/README.md).
Pennyroyal does not include a native systemd service file.

## Configure and run

The optional setup assistant is **beta**. The existing direct launchers remain
available below if you prefer manual configuration.

After building Pennyroyal and downloading your model, run this from the checkout:

```bash
./configure-penny --native
./run-penny --check
./run-penny
```

Setup asks for the model, cache locations, and GPU, then shows your choices
before saving. Next needs one target checkpoint; 27B also needs its DFlash2
draft. Leave advanced settings alone to use the normal recipe defaults.
Create any missing cache directories shown by the check before starting.

Settings are saved in `~/.config/pennyroyal/pennyroyal.env`. Rerun setup to
change them, or edit the file directly. `./run-penny --show-config` shows the
selected configuration without loading a model. For separate saved profiles,
pass `--config /absolute/path/to/next.env` to both setup and launch.

The setup utility needs Python 3 but no model packages or GPU to validate
configuration. It does not install dependencies, download models, start the
server, or delete caches. The normal launcher still checks the runtime and
checkpoint when starting. Stop and restart the server to apply changes.

Prefer shell exports or an existing service? The direct launchers below still
work; the setup utility is optional.

## Common setup

Set the repository, compiler-cache, and persistent-cache roots. The launchers
validate these inputs and create their cache directories:

```bash
export REPO_ROOT=/path/to/pennyroyal
export CACHE_BASE=/path/to/compiler-and-runtime-caches
export NIXL_STORAGE_BASE=/path/to/persistent-nixl-root
cd "$REPO_ROOT"
```

The scripts use `$REPO_ROOT/.venv/bin/sglang` and its Python interpreter. Set
both overrides if the environment lives elsewhere:

```bash
export SGLANG_EXE=/path/to/venv/bin/sglang
export PYTHON=/path/to/venv/bin/python
```

If NIXL is installed under a prefix outside the system linker path, add its
`lib64` directory:

```bash
export NIXL_PREFIX=/path/to/nixl-prefix
export LD_LIBRARY_PATH="$NIXL_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Review the sample NIXL watermarks before first use. The storage directory must
be writable, and its filesystem must support the configured O_DIRECT/io_uring
path.

Both profiles use the bundled
[Froggeric v22.5 template](configs/pennyroyal/templates/README.md) and default
to CPU image preprocessing. The template hash, preprocessing device, and
backend are part of the NIXL namespace. Representation changes select a new
cache identity while leaving older namespaces intact.

The launch commands below use the defaults. Set any
[precision or PLE](#optional-flash-next-precision-and-ple-placement),
[request/KV-capacity](#optional-six-request-flash-next-profile),
[media-device](#cpu-model-gpu-or-secondary-gpu-media-preprocessing), or
[host-control](#optional-host-and-launcher-controls) overrides before running
the launcher. Stop and restart the server after changing them.

## Launch Flash-Next with FR-Spec

This is the recommended Flash-Next recipe. Native NEXTN MTP is part of the
target checkpoint, so only the target model is required:

```bash
cd "$REPO_ROOT"
export TARGET_MODEL=/path/to/RadixArk-Qwen3.8-Flash-Next-NVFP4
"$REPO_ROOT/configs/pennyroyal/serve-flash-next-frspec.sh"
```

The launcher verifies the bundled 65,536-ID FR-Spec map and reference tokenizer,
then includes the map hash in the NIXL namespace. The reduced draft vocabulary
leaves the target vocabulary and acceptance policy unchanged. Gabriel's
[`gabrielolympie/sglang-flashnext-sm120`](https://github.com/gabrielolympie/sglang-flashnext-sm120)
is the source of the reduced-vocabulary work.

The extra BF16 draft head uses 320 MiB. The recipe defaults to 824,384 KV
tokens, 524,288-token context, four concurrent requests, and 24 Mamba slots.
During startup, confirm the token map, reduced draft head, graphs, and state
pools in the resolved server output.

## Launch 27B with DFlash2

Use the official Qwen reference target, or the measured public alternative in
[BUILD.md](BUILD.md#reference-and-measured-checkpoints), with the separate
DFlash2 draft:

```bash
cd "$REPO_ROOT"
export TARGET_MODEL=/path/to/Qwen3.8-27B-FP8
export DRAFT_MODEL=/path/to/incoai-Qwen3.8-27B-DFlash2
"$REPO_ROOT/configs/pennyroyal/serve-qwen38-27b-dflash2.sh"
```

| Setting | 27B/DFlash2 value |
|---|---|
| Target weights / activations | Block FP8 E4M3 with dynamic FP8 on quantized paths; BF16 for unquantized tensors |
| Target and draft KV | FP8 E4M3 |
| Recurrent / convolution state | FP32 GDN SSM / BF16 convolution |
| DFlash2 shape | 8 draft tokens; 2,048-token window |
| Attention | TRTLLM-MHA/XQA target decode; FlashInfer target prefill and draft attention; Triton GDN |
| State and host cache | 24 Mamba slots; 5 retained states per path; 96 GB HiCache by default; NIXL POSIX |

The target is a dense FP8 model; Flash-Next's routed-expert settings do not
apply.

Run one profile at a time on a single GPU.

### Unknown tool names

The qualified launchers default `SGLANG_FORWARD_UNKNOWN_TOOLS=true`. A native
tool call whose name is absent from the request's tool definitions reaches the
API consumer's executor, which can return an error for the model to correct and
retry. Markdown fenced tool examples remain text, and forwarding does not
execute anything by itself. Set `SGLANG_FORWARD_UNKNOWN_TOOLS=false` before a
native launch, or in the Compose `.env`, to opt out.

## Smoke through the normal API

Wait for SGLang to report readiness, then check health and send a normal chat
request:

```bash
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"pennyroyal",
    "messages":[{"role":"user","content":"Explain why 17 is prime."}],
    "max_tokens":256,
    "stream":false
  }'
```

Run several ordinary thinking-enabled warmups before measuring. Let requests
finish or cancel them through the protocol; terminal signals do not exercise
normal request cleanup.

## Host memory and first start

| Profile | Configured host cache | Additional host use |
|---|---:|---|
| Flash-Next | 32 GB HiCache | Approximately 47.68 GiB for RAM-backed PLE |
| 27B/DFlash2 | 96 GB HiCache | Model loading and runtime overhead |

Process, driver, filesystem, and page-cache memory are additional. NVMe-backed
PLE removes Flash-Next's fixed table residency and uses SSD I/O plus reclaimable
filesystem cache. The values above describe the configured profiles rather
than minimum host specifications.

Allow disk space for the target checkpoint, the 27B draft when selected,
compiler/JIT caches, and NIXL objects. On first use, the launcher prints its
profile, paths, and cache roots before deriving the namespace. A checkpoint
without usable Hugging Face download metadata requires hashing its weight
files, which can be quiet for a while. Kernel compilation and CUDA graph
capture follow. The server is ready when the startup log reports readiness and
the API smoke test succeeds.

One built-in JSON-schema warmup runs before readiness. It exercises a stable
grammar/mask path; other schemas, long prefill, media, and concurrency shapes
warm when used.

This page covers the qualified native launchers. For the qualified container
path, use [`docker/pennyroyal`](docker/pennyroyal/README.md); other Docker files
inherited from upstream serve their upstream purposes.

## Startup checks

Each recipe prints the requested profile, memory, cache, and media settings
before starting SGLang. SGLang reports the actual KV capacity and request
admission after memory profiling. Verify those resolved values rather than the
request summary alone.

NIXL may log an open/registration error for `/nonexistent-nixl-probe`, followed
by `path-mode FILE registration active`. That pair is the expected capability
probe. Errors naming a real cache path need attention.

Transformers may warn that the original Flash-Next configuration contains
`mrope_interleaved` and `mrope_section` fields unknown to its default RoPE
reader. SGLang retains the fields and selects the configured factor-2
YaRN/mRoPE implementation. Keep them in the checkpoint configuration.

For Flash-Next, confirm:

- 824,384 target/native-MTP KV tokens by default, or the resolved value for an
  explicit cap, with FP8 E4M3 dtypes;
- four running requests and 24 Mamba slots by default, or six requests and 36
  slots for the optional C6 profile, with zero intermediate speculative SSM;
- `FlashInferGDNKernel` decode/prefill and
  `none-mode WY output-only` verification/recovery;
- QSA sparse decode through FlashInfer's wrapper resolving to XQA on SM120,
  `sgl-kernel` top-k, and MTP index sharing;
- target and native-MTP MoE resolved to FlashInfer CUTLASS;
- recovery graphs for batch sizes 1-4 by default, or 1-6 with C6; and
- attached KV, Mamba/PLE, and QSA HiCache pools.

With online FP8, also confirm MXFP8 projection signatures and the row-wise HC
mix and output-head weights. With NVMe PLE, confirm the prepared-table checksum,
plugin registration, SSD reader, and separate NIXL namespace. An explicit
`MAX_TOTAL_TOKENS` value is a request; the resolved KV capacity is the result.

For 27B, confirm:

- target and draft FP8 E4M3 pools;
- `Initialized DFLASH draft runner` with eight tokens and window 2,048;
- fused KV materialization;
- FlashInfer prefill/draft and TRTLLM-MHA target decode/verify;
- a dense FP8 target;
- 24 Mamba slots, five retained states per path, and 1,118,784 KV tokens; and
- target prefill, target verify, and draft verify graph capture.

[BACKENDS.md](BACKENDS.md) maps each resolved implementation and its evidence.

## Distinguish cache paths

### Choose HiCache RAM size

Set `PENNY_HICACHE_SIZE_GB` to the amount of system RAM to use for conversation
cache. It accepts whole GB starting at `1`; leave it blank or unset for the
existing defaults of 32 GB for Next and 96 GB for 27B. Pool alignment and runtime
overhead are additional, as are model loading and RAM-backed PLE.

Smaller settings leave less room for reused conversation state. They do not
disable HiCache or NIXL, and they do not limit the GPU KV pool.

### Limit NIXL disk use

Set `SGLANG_HICACHE_NIXL_MAX_CACHE_GB=200` in your saved config, Compose `.env`,
or environment before launching to give the active NIXL cache a 200 GiB
budget. The setup utility asks about this too. Zero or unset disables it.

This is a periodic cleanup target, not a hard disk quota: writes can briefly
exceed it, and cleanup aims for 90% of the budget. Existing filesystem-space
watermarks still apply. It covers the active cache namespace, not old caches
left by other versions or configurations. No model files are removed.

### Cache reuse

- **Cold prefill:** no matching GPU radix prefix and no matching NIXL object;
  logs show most tokens as newly computed.
- **Radix reuse:** the same server process retains the prefix in GPU/host state;
  logs show cached tokens without a service restart.
- **Persistent NIXL restore:** restart the service without deleting the selected
  namespace, submit the identical serialized prefix, and verify HiCache/NIXL
  load plus a large restored count and a small recomputed suffix.

A restored-prefix effective rate is different from cold prefill. The namespace
must remain representation-compatible; the helper selects a new directory
after relevant configuration or source changes.

## Optional Flash-Next precision and PLE placement

The two v2.5.0 options are independent. The default recipe uses the original
checkpoint precision and RAM-backed PLE:

```bash
unset SGLANG_SM120_ONLINE_MXFP8
export PENNY_PLE_BACKEND=ram
```

Enable exact-SM120 online FP8 with a literal `true`:

```bash
export SGLANG_SM120_ONLINE_MXFP8=true
```

This converts eligible otherwise-BF16 transformer projections, HC mix weights,
and the output head during loading. NVFP4 experts, routers, and PLE remain in
their checkpoint formats; GDN state remains BF16, KV remains FP8, and FR-Spec
alignment is unchanged. Read [FP8.md](FP8.md) before enabling it.

To stream the PLE table from a prepared local SSD overlay:

```bash
export PENNY_PLE_BACKEND=nvme
export PENNY_PLE_NVME_MODEL=/path/on/local-nvme/flash-next-ple
```

NVMe mode requires the isolated reader and prepared overlay in
[NVME-PLE.md](NVME-PLE.md). Explicit NVMe errors stop startup. Either PLE
placement can be combined with online FP8.

`nixl-posix-frspec.toml` uses 85%/80% cleaner watermarks for a dedicated cache
filesystem. Review them for your storage.

The [map builder](scripts/pennyroyal/frspec/build_token_map.py) supports a
different tokenizer or corpus. The bundled map defines the qualified FR-Spec
profile; a new map gets its own validation and cache namespace. See
[PROVENANCE.md](PROVENANCE.md#v23-fr-spec-provenance) for hashes and source
details.

## Optional 1,000,000-token Flash-Next KV pool

`MAX_TOTAL_TOKENS` accepts a positive page-64-aligned override:

```bash
export MAX_TOTAL_TOKENS=1000000
```

This v2.5.0 setting was tested with online FP8, RAM PLE, CPU media
preprocessing, and only the RTX PRO 6000 visible. The runtime allocated
1,000,000 KV tokens, retained 524,288-token context, captured the normal
graphs, and passed warmup/schema/tools, 64K/490K retrieval, fixed-output C1/C4,
JPEG and static video checks, and three image-history turns around 208K
context. Post-graph free memory was 5.21 GiB; the lowest media sample was
1,187 MiB.

The option changes KV capacity. Context remains 524,288 tokens, and no speed
comparison was run. Model-GPU media preprocessing was not tested with this
pool. Unset `MAX_TOTAL_TOKENS` to return to the 824,384-token FR-Spec default.
The non-FR recipe uses automatic sizing when unset and accepts the same kind of
page-aligned override. See
[RESULTS.md](RESULTS.md#explicit-1000000-token-capacity-option) for timings.

## Optional six-request Flash-Next profile

For longer concurrent conversations, Flash-Next can use six running requests
and a larger shared KV pool while keeping 524,288 tokens per request. The
FR-Spec defaults remain four requests, 24 Mamba slots, and 824,384 KV tokens. Set the
C6 values before launching either Flash-Next recipe:

```bash
export SGLANG_SM120_ONLINE_MXFP8=true
export SGLANG_MM_PREPROCESS_DEVICE=cpu
export MAX_RUNNING_REQUESTS=6
export MAX_MAMBA_CACHE_SIZE=36
export MAX_TOTAL_TOKENS=1048576
```

With online FP8, RAM PLE, and A4000 media preprocessing, the RTX PRO 6000
profiled 1,034,176 shared KV tokens from the requested 1,048,576. This is one
shared pool, not six independent 524K contexts or a 1M per-request context.
SGLang may clamp the request to the capacity it profiles on another system.

CPU preprocessing is the single-GPU default. A secondary GPU is another way
to keep media preprocessing off the model GPU; see
[CPU, model-GPU, or secondary-GPU media preprocessing](#cpu-model-gpu-or-secondary-gpu-media-preprocessing).
The C6 profile does not require a dual-socket system or NUMA configuration.
More Mamba slots and larger CUDA graphs also consume VRAM, so confirm the
resolved token pool, six-request admission, 36 Mamba slots, and batch 1-6
graphs in the startup output.

## Launch Flash-Next without FR-Spec (alternative)

This recipe keeps the same target, context, KV/state pools, native NEXTN,
HiCache, and NIXL configuration while omitting the reduced-vocabulary map:

```bash
cd "$REPO_ROOT"
export TARGET_MODEL=/path/to/RadixArk-Qwen3.8-Flash-Next-NVFP4
"$REPO_ROOT/configs/pennyroyal/serve-flash-next.sh"
```

| Setting | Flash-Next value |
|---|---|
| Target | ModelOpt NVFP4 weights and activations on selected Linear modules |
| Other tensors | BF16 excluded/unquantized tensors and recurrent state |
| Target/native-MTP KV | FP8 E4M3 |
| Speculation and context | Native NEXTN; 524,288-token YaRN context |
| State and host cache | 24 Mamba slots; RecoverSSM `none`; 32 GiB HiCache; NIXL POSIX |
| Linear attention | Explicit FlashInfer GDN decode/prefill |

The online-FP8 and PLE-placement options also apply to this recipe.

## CPU, model-GPU, or secondary-GPU media preprocessing

The recipes default to `SGLANG_MM_PREPROCESS_DEVICE=cpu` with the PIL backend.
This keeps JPEG decoding and preprocessing off the model GPU. No second GPU is
required. For direct `sglang serve` use, the image-processor backend alone does
not select the preprocessing device.

To use the model GPU for JPEG decoding, resize, normalization, and patch
assembly:

```bash
export CUDA_VISIBLE_DEVICES=0
export SGLANG_MM_PREPROCESS_DEVICE=cuda:0
```

This spends some serving headroom. With v2.5.0 online FP8 and only the RTX PRO
6000 visible, `cuda:0` retained the 824,384-token pool and passed ten selected
media scenarios: ordinary and large JPEGs, ten images, two concurrent ten-image
requests, static MP4 frames, and three image-history turns around 208K context.
The lowest sampled free GPU memory was 1,897 MiB. Image dimensions and
concurrency change the requirement. Test headroom before combining model-GPU
preprocessing with a larger `MAX_TOTAL_TOKENS` value.

To use a second CUDA GPU for preprocessing:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export SGLANG_MM_PREPROCESS_DEVICE=cuda:1
```

Either CUDA choice selects the Torchvision backend. Indices refer to the
visible list: `cuda:0` is the model GPU and `cuda:1` is the second visible GPU.
With `cuda:1`, the model and vision encoder remain on `cuda:0` at TP1. Features
pass through host memory using the default CPU transport; this option does not
provide cross-device CUDA IPC/VMM or split the model. GPU UUIDs provide stable
device selection in `CUDA_VISIBLE_DEVICES`.

Direct `sglang serve` use also needs the matching
`--image-processor-backend pil` or `torchvision`. Without the environment
option, upstream automatic device selection applies. Invalid devices and a
CUDA selection paired with a PIL-only backend fail at startup. Restart the
server after changing the option.

Supported Qwen video-frame tensor preprocessing can use the selected GPU.
Video decoding, sampling, and initial frame preparation remain on CPU. The two
profiles do not support audio input. GPU JPEG decoding retains the CPU fallback
for unsupported image formats; an out-of-memory error still fails the request.

CPU mode uses host RAM and CPU time. A secondary GPU uses auxiliary VRAM and
adds transfers. Peak usage depends on dimensions, image count, and concurrency.
Backend pixel differences select a distinct persistence identity; older cache
directories remain intact.

## Optional host and launcher controls

The launchers default `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 4. Set explicit
values to retain a deliberate host-thread configuration:

```bash
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
```

`PENNY_BUILD_JOBS` controls the separate four-job build/JIT budget; NVCC uses
one thread by default. Existing per-tool overrides take precedence. These
settings affect compilation, not inference concurrency.

The launchers set `NUMPY_MADVISE_HUGEPAGE=0` to avoid observed huge-page
compaction stalls during CPU image-array allocation. It changes NumPy
allocations in the launched processes; system huge-page policy, GPU pools, and
cache contents are untouched. Set `NUMPY_MADVISE_HUGEPAGE=1` before startup to
restore NumPy's huge-page requests. NumPy reads the setting at import, so a
change requires a server restart. A host-wide `always` policy can still supply
huge pages.

For Pennyroyal's thinking-enabled agentic use, the launcher defaults to medium
reasoning effort. Chat Completions `reasoning_effort` takes precedence over
that default. With Froggeric v22.5, `high`, `xhigh`, and `max` select the same
xhigh instruction, which can change answer length relative to medium.
Responses API precedence is unchanged.

To change the launcher default itself (PR#18), export
`PENNY_REASONING_EFFORT=none|minimal|low|medium|high|xhigh|max` before
starting a recipe; unset or empty keeps the qualified medium. The recipes
build `--default-chat-template-kwargs` from that single value, the server
never reads the variable, and an invalid tier stops the launch.

Natively, `TP_SIZE=2` makes the Next recipes claim GPUs 0..TP_SIZE-1 when
`CUDA_VISIBLE_DEVICES` is unset, but it does not grant GPU access: the
launch aborts with the visible-device count if fewer devices are visible
than TP (plus a dedicated `SGLANG_MM_PREPROCESS_DEVICE=cuda:N`) requires.
In Docker/Compose the same `TP_SIZE=2` also needs the existing
`deploy.resources.reservations.devices` entry edited to name both GPU ids,
for example `device_ids: ["0", "1"]`; see docker/pennyroyal/README.md.
