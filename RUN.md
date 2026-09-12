# Run

Both model profiles use the same built source and expose the model as
`pennyroyal` on an OpenAI-compatible endpoint. Set the paths below before
running a launcher; systemd service files are not included.

## Common setup

Set the common paths first, then run from the checked-out release root. The
launchers create the cache directories after validating their inputs:

```bash
export REPO_ROOT=/path/to/pennyroyal
export CACHE_BASE=/path/to/compiler-and-runtime-caches
export NIXL_STORAGE_BASE=/path/to/persistent-nixl-root
cd "$REPO_ROOT"
```

The scripts expect the SGLang executable at `$REPO_ROOT/.venv/bin/sglang`.
If the environment lives elsewhere, set both paths before launch:

```bash
export SGLANG_EXE=/path/to/venv/bin/sglang
export PYTHON=/path/to/venv/bin/python
```

The launchers default `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 4. Operators
can set `PENNY_BUILD_JOBS` before launch to change the separate four-job
build/JIT budget; NVCC defaults to one thread. Existing per-tool build
overrides take precedence. These are compilation limits, not inference limits.

The launchers default `NUMPY_MADVISE_HUGEPAGE=0` to mitigate huge-page
compaction stalls during CPU image-array allocation. This affects NumPy allocations
in the launched processes, not system-wide huge-page policy, GPU pools or
cache contents. Large CPU-array workloads can benefit from huge pages on other
hosts; set `NUMPY_MADVISE_HUGEPAGE=1` before launch to opt back in. The setting
is read when NumPy imports, so changing it requires restarting the server.
This removes NumPy's huge-page requests; it does not prohibit huge pages under
an administrator's global `always` policy or eliminate unrelated memory pressure.

For inference-library CPU threads, operators can retain deliberate host-thread
settings instead:

```bash
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
```

If NIXL was installed under a prefix not known to the system linker, also add
its `lib64` directory:

```bash
export NIXL_PREFIX=/path/to/nixl-prefix
export LD_LIBRARY_PATH="$NIXL_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Review the sample NIXL watermarks before first use. The storage directory must
be writable and the filesystem must support the configured O_DIRECT/io_uring
path.

Both profiles use the bundled [Froggeric v22.5 template](configs/pennyroyal/templates/README.md)
and default to CPU image decoding/preprocessing; model inference stays on GPU.
The template hash, preprocessing device and backend are included in the NIXL namespace, so these
launchers start a separate cache identity without deleting older caches.

### CPU, model-GPU, or secondary-GPU media preprocessing

The recipes default to `SGLANG_MM_PREPROCESS_DEVICE=cpu` with the PIL backend.
This keeps JPEG decoding off the model GPU too; `--image-processor-backend pil`
alone does not do that. No second GPU is required.

To use the model GPU for JPEG decoding, image resize, normalization and patch
assembly, choose `cuda:0`:

```bash
export CUDA_VISIBLE_DEVICES=0
export SGLANG_MM_PREPROCESS_DEVICE=cuda:0
```

This is simple but spends some of the model GPU's serving headroom. In v2.5.0
qualification with online FP8 and only the RTX PRO 6000 visible, `cuda:0`
retained the 824,384-token pool and passed ten selected media scenarios:
ordinary and large JPEGs, ten images, two concurrent ten-image requests, a
static MP4 frame path, and three successive image-history turns around 208K
context. Minimum sampled free GPU memory was 1,897 MiB. That sample is not a
promise for arbitrary image dimensions or concurrency. Do not combine this
placement with a larger experimental `MAX_TOTAL_TOKENS` value without a new
headroom and workload check.

To use a second CUDA GPU for JPEG decoding, image resize, normalization and
patch assembly:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export SGLANG_MM_PREPROCESS_DEVICE=cuda:1
```

Either CUDA choice selects the Torchvision backend. Indices refer to the
visible list: `cuda:0` is the model GPU and `cuda:1` is the second visible GPU.
With `cuda:1`, the model and vision encoder remain on `cuda:0` at TP1; this
does not split the model. Features pass through host memory using the default
CPU feature transport; cross-device CUDA IPC/VMM is not supported by this
option. A GPU UUID list can be used in `CUDA_VISIBLE_DEVICES` for stable
placement.

For direct `sglang serve` use, also specify the matching
`--image-processor-backend pil` or `torchvision`. With the environment option
unset, upstream automatic device selection is retained. Invalid/unavailable
devices or a CUDA selection with a PIL-only backend fail clearly. Set the
option before startup and restart to change it.

Supported Qwen video-frame tensor preprocessing can use the selected GPU;
video decoding, sampling and initial frame preparation remain CPU work.
Neither qualified profile supports audio input. Other models/media processors
need separate validation; this option does not add modalities. GPU JPEG decode
retains the existing CPU fallback for unsupported images, but preprocessing
OOM does not silently redirect work to the model GPU.

CPU mode uses host RAM and CPU time. A secondary GPU uses auxiliary VRAM and
adds transfers; peak usage depends on dimensions, image count and concurrency.
Decoder/backend pixel values may differ slightly, so the recipes select a
distinct persistence identity. Older cache directories are retained.

Build/install the selected release source before starting the server. A source
change selects a fresh NIXL namespace, so matching prefixes initially start
cold. Let the helper choose the directory; do not point new source at an old
representation's cache. Older namespaces are not deleted automatically.

Explicit Chat Completions reasoning-effort requests now take precedence over
launcher defaults; the normal medium default is unchanged. With the bundled
Froggeric v22.5 template, `high`, `xhigh`, and `max` select the same xhigh
instruction. A client whose override was previously ignored may therefore
observe different answer length. Responses API precedence is unchanged.

## Launch Flash-Next with FR-Spec

This is the recommended Next recipe. Use the RadixArk reference checkpoint.
Native NEXTN MTP is already included, so
there is no separate draft download:

```bash
cd "$REPO_ROOT"
export TARGET_MODEL=/path/to/RadixArk-Qwen3.8-Flash-Next-NVFP4
"$REPO_ROOT/configs/pennyroyal/serve-flash-next-frspec.sh"
```

The launcher adds `--speculative-token-map` with the bundled 65,536-ID map. It
verifies the map and reference tokenizer before startup and includes the map
hash in the NIXL cache namespace. Target vocabulary and acceptance policy are
unchanged. Direct credit for the reduced draft-vocabulary work belongs to
Gabriel's
[`gabrielolympie/sglang-flashnext-sm120`](https://github.com/gabrielolympie/sglang-flashnext-sm120).

The extra BF16 draft head uses 320 MiB. The primary qualified recipe defaults
to a cap of 824,384 KV tokens while retaining 524,288-token context, four
concurrent requests, and 24 Mamba slots. Confirm `speculative_token_map` in the
resolved server arguments, the reduced draft head, and the normal graph/state-
pool checks below.

`MAX_TOTAL_TOKENS` accepts a positive page-64-aligned override for experiments:

```bash
export MAX_TOTAL_TOKENS=1000000
```

The 1,000,000-token option was qualified with online FP8, RAM PLE, CPU media
preprocessing and only the RTX PRO 6000 visible. It retained 524,288-token
context, captured the normal graphs, and passed warmup/schema/tools, 64K/490K
retrieval, fixed-output C1/C4, ordinary/large/concurrent image checks, static
MP4 frames and three successive image-history turns around 208K context.
Post-graph free memory was 5.21 GiB; the minimum sampled during media work was
1,187 MiB. This is a capacity option, not a larger-context or speedup claim.
It has not been qualified with model-GPU media preprocessing. Unset the
variable to return to the 824,384-token FR-Spec default. The non-FR recipe
retains automatic sizing when unset and also accepts an explicit positive
page-aligned override. Detailed timing is in
[RESULTS.md](RESULTS.md#explicit-1000000-token-capacity-option).

### Optional Flash-Next precision and PLE placement

The two v2.5.0 options are independent. The ordinary recipe uses the original
checkpoint precision and RAM-backed PLE:

```bash
unset SGLANG_SM120_ONLINE_MXFP8
export PENNY_PLE_BACKEND=ram
```

Enable exact-SM120 online FP8 with a literal `true`:

```bash
export SGLANG_SM120_ONLINE_MXFP8=true
```

This converts eligible otherwise-BF16 transformer projections, HC mix weights
and the output head during loading. It preserves NVFP4 experts/routers/PLE,
BF16 GDN state, FP8 KV and FR-Spec alignment. Read [FP8.md](FP8.md) before
selecting it.

To stream the PLE table from a prepared local SSD overlay:

```bash
export PENNY_PLE_BACKEND=nvme
export PENNY_PLE_NVME_MODEL=/path/on/local-nvme/flash-next-ple
```

NVMe mode requires the isolated reader and prepared overlay described in
[NVME-PLE.md](NVME-PLE.md). RAM remains the default; explicit NVMe errors fail
startup rather than falling back. Either PLE placement can be combined with
online FP8.

`nixl-posix-frspec.toml` uses 85%/80% cleaner watermarks for a dedicated cache
filesystem. Review those thresholds for your storage. The non-FR and 27B
launchers remain available with their existing configurations.

The [map builder](scripts/pennyroyal/frspec/build_token_map.py) is included
for users who need a different tokenizer or corpus. Use the bundled map to
use the qualified FR-Spec profile; a newly generated map requires its own
validation and cache namespace. See
[PROVENANCE.md](PROVENANCE.md#v23-fr-spec-provenance) for hashes and source
details.

## Launch 27B with DFlash2

Use the official Qwen reference target, or the measured public alternative
identified in [BUILD.md](BUILD.md#reference-and-measured-checkpoints), plus the
separate DFlash2 draft:

```bash
cd "$REPO_ROOT"
export TARGET_MODEL=/path/to/Qwen3.8-27B-FP8
export DRAFT_MODEL=/path/to/incoai-Qwen3.8-27B-DFlash2
"$REPO_ROOT/configs/pennyroyal/serve-qwen38-27b-dflash2.sh"
```

The target uses block-FP8 E4M3 weights and dynamic FP8 activations on quantized
paths, with BF16 for unquantized tensors. Target/draft KV are FP8 E4M3; GDN SSM
state is FP32 and convolution state BF16. The shape uses eight DFlash2 draft
tokens, a 2,048-token draft window, TRTLLM-MHA/XQA target decode, FlashInfer
target prefill/draft attention, dense FP8 feed-forward layers and Triton GDN,
24 Mamba slots, five retained states per path, 96 GiB HiCache, and NIXL POSIX
persistence.

## Launch Flash-Next without FR-Spec (alternative)

The non-FR recipe keeps the same target model, context, pools, native NEXTN,
HiCache, and NIXL configuration while omitting only the reduced-vocabulary map:

```bash
cd "$REPO_ROOT"
export TARGET_MODEL=/path/to/RadixArk-Qwen3.8-Flash-Next-NVFP4
"$REPO_ROOT/configs/pennyroyal/serve-flash-next.sh"
```

The qualified Flash-Next shape uses ModelOpt NVFP4 weights and input
activations on selected Linear modules, BF16 for excluded/unquantized tensors
and recurrent state, FP8 E4M3 target/native-MTP KV, native NEXTN, 524K YaRN,
24 Mamba slots, RecoverSSM `none`, explicit FlashInfer GDN decode/prefill,
32 GiB HiCache, and NIXL POSIX persistence. The v2.5.0 online-FP8 and PLE-
placement options above apply to this recipe too. This is not a claim that
every kernel computes in BF16.

## Host memory and first start

The configured HiCache tier consumes host memory: 32 GiB for Flash-Next and
96 GiB for 27B. RAM-backed Flash-Next additionally pins an approximately
47.68 GiB PLE table. Optional NVMe PLE removes that fixed table residency but
uses SSD I/O and reclaimable filesystem cache; observed available-memory gains
are not entirely attributable to the table. Both profiles need additional
process and driver overhead. These are measured configurations, not minimum-
host-RAM specifications.

On first use, allow disk for model weights, the 27B draft when selected,
compiler/JIT caches, and NIXL objects. The launcher prints its selected profile,
paths, and cache roots, then warns before namespace derivation. With checkpoints
that lack usable Hugging Face download metadata, namespace derivation hashes
the weight files and may be quiet for a while. Kernel JIT compilation and CUDA
graph capture follow and can also take substantial time. A server is ready only
after the startup log reports readiness and the API smoke below succeeds.

The launchers run one bounded built-in JSON-schema warmup before announcing API
readiness. This verifies one structured-output grammar/mask path; it does not
precompile arbitrary schemas or warm every long-prefill or concurrency shape.

These launchers document the qualified native build. Docker files inherited
from upstream are not a qualified Pennyroyal deployment recipe.

## Startup checks

NIXL may log an open/registration error for `/nonexistent-nixl-probe`, followed
by `path-mode FILE registration active`. That specific pair is the expected
capability probe; errors for real cache paths still need attention.

With the Next recipe, Transformers may warn that default RoPE does not
recognize `mrope_interleaved` and `mrope_section` while reading the original
checkpoint config. SGLang retains those fields and selects the configured
factor-2 YaRN/mRoPE implementation; do not remove them to silence the warning.

For Flash-Next, confirm log lines for:

- 824,384 target/native-MTP KV tokens and FP8 E4M3 dtypes;
- 24 Mamba slots and zero intermediate speculative SSM;
- `FlashInferGDNKernel` decode/prefill and
  `none-mode WY output-only` verification/recovery;
- QSA sparse decode through FlashInfer's wrapper resolving to XQA on SM120,
  `sgl-kernel` top-k, and MTP index sharing;
- target and native-MTP MoE resolved to FlashInfer CUTLASS;
- recovery graphs for batch sizes 1-4;
- attached KV, Mamba/PLE, and QSA HiCache pools.

When online FP8 is selected, also confirm that the option is enabled, eligible
projection signatures resolved to the MXFP8 backend, and the expected row-wise
HC mix and output-head weights were installed. When NVMe PLE is selected,
confirm the prepared-table checksum, plugin registration, SSD reader and
separate NIXL namespace. In every case verify the **actual** KV capacity; a
requested cap alone is not evidence that it was retained.

For 27B, confirm:

- target and draft FP8 E4M3 pools;
- `Initialized DFLASH draft runner` with eight tokens and window 2,048;
- fused KV materialization;
- FlashInfer prefill/draft and TRTLLM-MHA target decode/verify;
- a dense FP8 target, not a routed-expert MoE model;
- 24 Mamba slots, five retained states/path, and 1,118,784 KV tokens;
- target prefill, target verify, and draft verify graph capture.

## Smoke through the normal API

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

Run several ordinary thinking-enabled warmups before measuring. Do not abort a
request with terminal signals; use protocol-level cancellation or let it finish.

## Distinguish cache paths

- **Cold prefill:** no matching GPU radix prefix and no matching NIXL object;
  logs show most tokens as newly computed.
- **Radix reuse:** same server process retains the prefix in GPU/host state;
  logs show cached tokens without a service restart.
- **Persistent NIXL restore:** restart the service without deleting the selected
  namespace, submit the identical serialized prefix, and verify HiCache/NIXL
  load plus a large restored/cached count and only a small recomputed suffix.

Do not label a restored-prefix effective rate as cold prefill. The persistent
namespace must remain representation-compatible; the helper deliberately
selects a new directory after relevant configuration or source changes.
