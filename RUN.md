# Run Penny Royal on RTX PRO 6000 Blackwell

This is the native launch guide for Penny Royal. The same source runs
Qwen3.8 Flash-Next NVFP4 with native NEXTN MTP and FR-Spec, the same target
with the separate DFlash2 draft, or Qwen3.8-27B FP8 with its own DFlash2
draft, on NVIDIA RTX PRO 6000 Blackwell (SM120), TP1 or TP2. The Flash-Next
native NEXTN MTP path is the maintained default and 27B's DFlash2 path is
established and qualified; the Flash-Next DFlash2 path is an
**experimental / alpha** opt-in that stays fully available with its existing
recipe (see the [DFlash2 maturity note](README.md#dflash2-maturity)).
Both profiles serve an OpenAI-compatible model named `pennyroyal`.

Build the release with [BUILD.md](BUILD.md) first. Container users should
follow the separate [container guide](docker/pennyroyal/README.md).
Pennyroyal does not include a native systemd service file. This page covers
launching; [CONFIGURATION.md](CONFIGURATION.md) is the complete reference for
every supported setting, its default, and its restart and cache effects.

## Configure and run

Use the ordinary startup scripts below: set your model, GPU and cache paths,
then run the recipe. Settings can live in your own shell startup file or the
provided environment-file format. The same runtime supports RTX PRO 6000
Blackwell at TP1 or TP2; optional Ampere/Ada sidecar image processing is separate
from the inference cards.

Start with [Common setup](#common-setup), then choose
[Flash-Next with FR-Spec](#launch-flash-next-with-fr-spec) — the default newly
generated setup — [27B with DFlash2](#launch-27b-with-dflash2), or one of the
optional native-MTP and DFlash2 recipes,
[Flash-Next with DFlash2](#launch-flash-next-with-dflash2) and the
[non-FR alternative](#launch-flash-next-without-fr-spec-alternative).
Stop and restart the server after changing configuration.

The optional [beta configurator](CONFIGURE.md) can write the settings for you.
It creates a new timestamped file and prints the exact check and launch
commands for that file. Use those commands rather than assuming it overwrote
`~/.config/pennyroyal/pennyroyal.env`.

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

**Maintained native MTP, the default.** This recipe remains what new setups start
on and keeps its measured model, memory and capacity settings; the verify-width
routing described below is new in 3.1. The opt-in
[Flash-Next with DFlash2](#launch-flash-next-with-dflash2) is a separate choice.
Native NEXTN MTP is part of the target checkpoint, so only the target model is
required:

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

The verify width is fixed by active concurrency through the bundled policy
`configs/pennyroyal/adaptive-next.json`: block 8 for a single running request,
block 4 for every batch of two or more, one candidate per tier, so the width
is identical between boots and the acceptance EMA cannot move it. The recipes
read the file through the shared width selector
`configs/pennyroyal/mtp-policy.sh`, hash what it selects into the NIXL namespace,
and launch at the widest routed block
(`--speculative-num-steps 7 --speculative-num-draft-tokens 8`).

That policy is a choice, and choosing it is one setting:
`PENNY_MTP_POLICY`.

```bash
PENNY_MTP_POLICY=fixed-w4 ./run-penny     # block 4 at every concurrency
PENNY_MTP_POLICY=adaptive ./run-penny     # the v3.0.2 policy at one request
PENNY_MTP_POLICY=tiered   ./run-penny     # the qualified default, as above
```

The same variable works on the recipe directly
(`PENNY_MTP_POLICY=fixed-w4 "$REPO_ROOT/configs/pennyroyal/serve-flash-next.sh"`)
and on Compose (`compose.yaml` forwards it), and `./configure-penny` can save the
choice instead — a container save writes it into the generated startup script's
own `PENNY_MTP_POLICY=` settings line, which is what `./run.sh` then launches.
One value names the policy file and the launch width that can carry it, so there
is no JSON to copy and no flag pair to keep in step; the precedence is the one
that has always applied here, a saved value beating the environment, the
selected policy's hash is part of the NIXL namespace, and a change takes effect
at the next start. See
[CONFIGURATION.md](CONFIGURATION.md#speculation-width-policy) for what each
choice runs and launches.

The extra BF16 draft head uses 320 MiB. The recipe asks for no KV pool size:
SGLang profiles the free GPU memory and fits the pool inside the static memory
fraction the launch sets, before the graphs are captured. The qualified
single-GPU boot reported 824,384 KV tokens, and that launch had asked for
exactly that cap; the capacity an uncapped FR-Spec boot reaches here has not
been measured. Context is 524,288 tokens with four concurrent requests and 24
Mamba slots. During startup, confirm the token map, reduced draft head, graphs,
and state pools in the resolved server output.

## Launch 27B with DFlash2

> The 27B DFlash2 integration is **established and qualified**: it launches and
> serves as this profile's speculative path, and the October 27 checks on the
> 3.1.3 candidate covered reasoning, tools, images, long-context retrieval and a
> fresh-process cache restore on it. What still needs explicit validation on your
> hardware is memory, not maturity - the draft keeps its own host cache **in
> addition to** the 96 GB HiCache budget below (the recorded launch: 79.45 GB
> target KV plus 16.63 GB Mamba state, and a further 24.83 GB of draft
> sliding-window cache, about 120.91 GB of caches before other process memory).
> The #115 decode-checkpoint arithmetic stays covered by its CPU regression; an
> end-to-end restore is not a grid-scatter unit regression, and no isolated GPU
> check of that path exists on either DFlash2 profile (see the
> [DFlash2 maturity note](README.md#dflash2-maturity)).

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

The launchers default `SGLANG_FORWARD_UNKNOWN_TOOLS=true`. A native
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

This page covers the native launchers. For the container
path, use [`docker/pennyroyal`](docker/pennyroyal/README.md); other Docker files
inherited from upstream serve their upstream purposes.

## WSL2 host-memory workaround

For WSL2, enable the pinned-memory workaround before launching either profile
directly with its recipe:

```bash
export SGLANG_HICACHE_TORCH_PINNED_ALLOC=true
```

This addresses illegal memory accesses during HiCache transfers caused by
WSL2's handling of CUDA host-memory pointers. It is off by default; leave it
off on native Linux. If you use `run-penny` with a saved configuration, set
`SGLANG_HICACHE_TORCH_PINNED_ALLOC=true` in that file: saved settings take
precedence over shell exports. In the beta configurator, choose **WSL2
host-memory workaround** under **Advanced**. For Docker Compose, set the same variable in
your `.env` file using the [container instructions](docker/pennyroyal/README.md#wsl2).

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

- the resolved target/native-MTP KV token count with FP8 E4M3 dtypes; the
  824,384-token figure is a capped boot's request, not an automatic value;
- four running requests and 24 Mamba slots by default, or six requests and 36
  slots for the optional C6 profile, with zero intermediate speculative SSM;
- `FlashInferGDNKernel` decode/prefill and
  `none-mode WY output-only` verification/recovery;
- QSA sparse decode through FlashInfer's wrapper resolving to XQA on SM120,
  `sgl-kernel` top-k, and MTP index sharing;
- target and native-MTP MoE resolved to FlashInfer CUTLASS;
- recovery graphs for batch sizes 1-4 by default, or 1-6 with C6; and
- attached KV, Mamba/PLE, and QSA HiCache pools.

With online FP8 active, also confirm the row-wise FP8 projection signatures
and the row-wise HC mix and output-head weights. With NVMe PLE, confirm the
prepared-table checksum, plugin registration, SSD reader, and separate NIXL
namespace. An explicit
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

Both Flash-Next recipes export `FLASHINFER_GDN_FP16_ACCUM_MMA=1`, which selects
the FP16-accumulate MMA mode of the patched FlashInfer SM12x delta-rule prefill
kernels (see
[FlashInfer SM120 source integration](BUILD.md#flashinfer-sm120-source-integration)).
The mode is part of the qualified Next profile: export
`FLASHINFER_GDN_FP16_ACCUM_MMA=0` before launch, or change the line in the
recipe or the mounted startup script, to run the kernels exactly as released.
FlashInfer's own default stays off, the 27B/DFlash2 recipe never sets the
variable, and its Triton GDN path and numerics are unchanged. The resolved mode is
one of the NIXL namespace fields, so the two modes use separate persistent roots:
nothing is deleted, and the root you leave behind stays as ordinary user-owned
cache.

Online FP8 is the automatic default of the Flash-Next recipes on exact SM120:
a fresh or default launch needs no kernel switch. The saved private override
`SGLANG_SM120_ONLINE_MXFP8=false` pins the original checkpoint path (and
`=true` forces the conversion where eligibility would stay off); read
[FP8.md](FP8.md) before using either. PLE placement stays a separate, explicit
choice; the default recipe uses RAM-backed PLE:

```bash
export PENNY_PLE_BACKEND=ram
```

This converts eligible otherwise-BF16 transformer projections, HC mix weights,
and the output head during loading. NVFP4 experts, routers, and PLE remain in
their checkpoint formats; GDN state remains BF16, KV remains FP8, and FR-Spec
alignment is unchanged.

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
different tokenizer or corpus. The bundled map defines the default FR-Spec
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

The option changes KV capacity; context remains 524,288 tokens. Use CPU media
preprocessing with this pool to leave more GPU memory for requests.
Leave `MAX_TOTAL_TOKENS` unset or blank and both Flash-Next recipes take the
pool size SGLang profiles from the free memory, within the static memory
fraction the recipe sets. The 824,384-token figure in the records below is the
older FR-Spec boot that asked for exactly that cap; an uncapped capacity has
not been measured here. See
[RESULTS.md](RESULTS.md#explicit-1000000-token-capacity-option) for timings.

## Optional six-request Flash-Next profile

For longer concurrent conversations, Flash-Next can use six running requests
and a larger shared KV pool while keeping 524,288 tokens per request. The
FR-Spec defaults remain four requests and 24 Mamba slots with the KV pool
sized automatically. Set the C6 values before launching either Flash-Next
recipe:

```bash
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

**Native MTP without FR-Spec**, like the FR-Spec recipe above, stays maintained and
selectable, just not the default. This recipe keeps the
same target, context, KV/state pools, native NEXTN, HiCache, and NIXL
configuration while omitting the reduced-vocabulary map:

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
| Speculation and context | Native NEXTN, width fixed by concurrency (block 8 at C1, block 4 at C2+; the same pinned policy as the FR-Spec recipe); 524,288-token YaRN context |
| State and host cache | 24 Mamba slots; RecoverSSM `none`; 32 GiB HiCache; NIXL POSIX |
| Linear attention | Explicit FlashInfer GDN decode/prefill, FP16-accumulate MMA mode |

The online-FP8 and PLE-placement options also apply to this recipe.

## Launch Flash-Next with DFlash2

`PENNYROYAL_PROFILE=next-dflash2` is the optional Flash-Next DFlash2 choice;
select it explicitly (or keep a saved file that names it) to use it. This
profile is **experimental / alpha** - the label belongs to it alone, not to
27B's qualified DFlash2 path (see the
[DFlash2 maturity note](README.md#dflash2-maturity)); native NEXTN MTP remains
the default profile. It runs the same accepted NVFP4 target and the bundled FR-Spec
64K token map against the separate
`incoai/Qwen3.8-Flash-Next-DFlash2` draft — its own download, next to the
target:

```bash
hf download incoai/Qwen3.8-Flash-Next-DFlash2 \
  --local-dir /path/to/incoai-Qwen3.8-Flash-Next-DFlash2
```

The FR-Spec map and the width-routing policy are already in this repository
(`configs/pennyroyal/frspec/flash-next-64k.pt` and
`configs/pennyroyal/adaptive-dflash2.json`), so the container path needs no extra
files beyond the launch scripts. The profile now uses FlashInfer for target
linear-attention verification (`--linear-attn-verify-backend flashinfer`);
both the native recipe and the container startup files select it
automatically.

```bash
cd "$REPO_ROOT"
export TARGET_MODEL=/path/to/RadixArk-Qwen3.8-Flash-Next-NVFP4
export DRAFT_MODEL=/path/to/incoai-Qwen3.8-Flash-Next-DFlash2
"$REPO_ROOT/configs/pennyroyal/serve-flash-next-dflash2.sh"
```

| Setting | Flash-Next/DFlash2 value |
|---|---|
| Target | ModelOpt NVFP4 weights and activations on selected Linear modules |
| Target output head | Rowwise FP8 E4M3 with one FP32 scale per channel (the online-FP8 conversion the target's own logits path uses; the draft reuses it) |
| Draft weights | BF16, unquantized (`--speculative-draft-model-quantization unquant`) |
| Speculation | DFLASH at steps 5 / 6 draft tokens, adaptive with the bundled fixed-width policy `configs/pennyroyal/adaptive-dflash2.json`: block 6 for a single running request, block 4 for every batch of two or more; the FR-Spec map `configs/pennyroyal/frspec/flash-next-64k.pt` (65,536 rows) |
| CUDA graphs | Prefill CUDA graphs disabled (`--disable-prefill-cuda-graph`); decode buckets stay on the runtime's own spec-decode list, so the single-request width keeps its bucket and the wider concurrent batches keep theirs |
| Target features | Completed layers 5, 15, 25, 35 and 45, averaged over the four hyper-connection streams (5 x 2,560 into the draft's `fc`) |
| Draft context | 2,048-token trained window; 524,288-position extension from this draft's own `rope_theta` 10000 and 262,144-token base |
| Target verification | FlashInfer linear-attention verify (`--linear-attn-verify-backend flashinfer`, fixed in the recipe; no operator toggle) |
| State and host cache | 36 Mamba slots for 6 requests; RecoverSSM `none`; 32 GB HiCache; NIXL POSIX |

This is the qualified Flash-Next path on RTX PRO 6000 at TP1: resident rowwise
FP8, the FR-Spec map, FlashInfer verification of accepted draft tokens, and the
fixed 6-at-C1 / 4-at-C2-and-above routing described above. DFlash2 offers more
token-pool capacity; we are still evaluating performance across concurrency
levels. The model card's own numbers were taken against the official FP8 target
on one GB300 at block 4, so they still say nothing about this NVFP4, SM120,
six-request stack — read your resolved token pool and request admission in the
startup output on your first boot. TP1 is the qualified default; TP2 is expected
to work; it has not been tested on two-GPU hardware, and the launcher's device
guard still refuses a TP size the visible GPUs cannot serve.

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

Both Flash-Next recipes export `FLASHINFER_GDN_FP16_ACCUM_MMA=1`, the accepted
FP16-accumulate MMA mode of the patched FlashInfer SM12x delta-rule prefill
kernels ([BUILD.md](BUILD.md#flashinfer-sm120-source-integration)). Export
`FLASHINFER_GDN_FP16_ACCUM_MMA=0` before startup to opt out; in a container
setup, saving or exporting that value reaches the generated launch the same way
the other forwarded knobs do. FlashInfer's own default is off, the 27B/DFlash2
recipe never sets it, and no other profile's numerics change. The recipes export
it before the server starts, so changing it requires a restart, and the resolved
mode is part of the NIXL cache identity.

For Pennyroyal's thinking-enabled agentic use, the launcher defaults to medium
reasoning effort. Chat Completions `reasoning_effort` takes precedence over
that default. With Froggeric v22.5, `high`, `xhigh`, and `max` select the same
xhigh instruction, which can change answer length relative to medium.
Responses API precedence is unchanged.

To change the launcher default itself (PR#18), export
`PENNY_REASONING_EFFORT=none|minimal|low|medium|high|xhigh|max` before
starting a recipe; unset or empty keeps the default medium. The recipes
build `--default-chat-template-kwargs` from that single value, the server
never reads the variable, and an invalid tier stops the launch.

TP2 verification is pending from
[u/StockSpecialist1707](https://www.reddit.com/user/StockSpecialist1707/).
Natively, `TP_SIZE=2` makes the Next recipes claim GPUs 0..TP_SIZE-1 when
`CUDA_VISIBLE_DEVICES` is unset, but it does not grant GPU access: the
launch aborts with the visible-device count if fewer devices are visible
than TP (plus a dedicated `SGLANG_MM_PREPROCESS_DEVICE=cuda:N`) requires.
In Docker/Compose the same `TP_SIZE=2` also needs the existing
`deploy.resources.reservations.devices` entry edited to name both GPU ids,
for example `device_ids: ["0", "1"]`; see docker/pennyroyal/README.md.

<!-- Compatibility anchors for earlier versions of this guide. -->
<a id="run-qwen38-with-sglang-on-one-rtx-pro-6000-blackwell"></a>
