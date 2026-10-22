#!/usr/bin/env bash
# Flash-Next NVFP4 with the Qwen3.8-Flash-Next-DFlash2 draft: 524K context,
# HiCache and the NIXL FILE tier, with the FR-Spec 64K token map and fixed
# concurrency width routing (block 6 at one request, block 4 at two or more).
#
# Disk tier: on (the qualified default) adds the NIXL FILE backend under
# $NIXL_STORAGE_BASE. off keeps the GPU radix cache and the host-RAM HiCache
# tier and drops only the disk tier: no NIXL root, config, namespace derivation
# or storage-backend argument, and no data removal.
NIXL="${NIXL:-on}"
case "$NIXL" in
  on|off) ;;
  *) echo "NIXL must be on or off; got '$NIXL'" >&2; exit 1 ;;
esac
# Draft settings carried by this recipe come from the DFlash2 checkpoint it is
# trained against (incoai/Qwen3.8-Flash-Next-DFlash2, a DFlash2DraftModel of
# 5 layers, width 2560 and the target's full 248320-token vocabulary):
#   dflash_config.target_layer_ids = [5,15,25,35,45] completed target layers,
#   dflash_config.mask_token_id    = 248077,
#   sliding_window                 = 2048,   dflash_config.block_size = 8,
#   rope_parameters.rope_theta     = 10000,  max_position_embeddings = 262144.
# They are written down rather than probed, exactly like the 27B recipe, and a
# draft that does not match them is refused by the runtime: the target-feature
# capture fails loud on a feature-width mismatch, and a wrong mask token id
# drafts nothing. The verify width is not an operator knob on this profile: it
# is routed per batch size by the bundled policy below, and the launch carries
# the widest routed width (see DRAFT_TOKENS / SPEC_STEPS). No
# --speculative-dflash-block-size is passed, so the checkpoint's own block-size
# field never conflicts with the routed widths.
set -euo pipefail
# Avoid synchronous RAM compaction for transient NumPy image arrays.
# Operators can opt back into NumPy huge-page advice; this is not a kernel policy.
export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
export SGLANG_MM_PREPROCESS_DEVICE="${SGLANG_MM_PREPROCESS_DEVICE:-cpu}"
export SGLANG_FORWARD_UNKNOWN_TOOLS="${SGLANG_FORWARD_UNKNOWN_TOOLS:-true}"
case "$SGLANG_MM_PREPROCESS_DEVICE" in
  cpu) IMAGE_PROCESSOR_BACKEND=pil ;;
  cuda:*) IMAGE_PROCESSOR_BACKEND=torchvision ;;
  *) echo "Choose SGLANG_MM_PREPROCESS_DEVICE=cpu or cuda:N" >&2; exit 1 ;;
esac
# Accepted FlashInfer GDN fix: run the patched SM12x delta-rule prefill kernels
# in FP16-accumulate MMA mode. This is the qualified default of the Flash-Next
# profiles, exported before Python imports FlashInfer. Export
# FLASHINFER_GDN_FP16_ACCUM_MMA=0 before launch to opt out; FlashInfer's own
# default stays off and the 27B/DFlash2 recipe never sets it.
export FLASHINFER_GDN_FP16_ACCUM_MMA="${FLASHINFER_GDN_FP16_ACCUM_MMA:-1}"

# The mode changes the GDN computation, so it is cache identity: the resolved
# value (not the raw string) goes into the NIXL namespace fields below, and
# anything but the literal 1 is the released FP32-accumulate representation.
GDN_FP16_ACCUM_MMA=off
if [[ "$FLASHINFER_GDN_FP16_ACCUM_MMA" == 1 ]]; then GDN_FP16_ACCUM_MMA=on; fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
SGLANG_EXE="${SGLANG_EXE:-$REPO_ROOT/.venv/bin/sglang}"
PYTHON="${PYTHON:-$(dirname "$SGLANG_EXE")/python}"
TARGET_MODEL="${TARGET_MODEL:?Set TARGET_MODEL to the Flash-Next checkpoint}"
DRAFT_MODEL="${DRAFT_MODEL:?Set DRAFT_MODEL to the Flash-Next DFlash2 checkpoint}"
CACHE_BASE="${CACHE_BASE:?Set CACHE_BASE to the durable compiler-cache root}"
required_executables=("$SGLANG_EXE" "$PYTHON")
if [[ "$NIXL" == on ]]; then
  NIXL_STORAGE_BASE="${NIXL_STORAGE_BASE:?Set NIXL_STORAGE_BASE to the FILE cache root}"
  NIXL_CONFIG="${NIXL_CONFIG:-$SCRIPT_DIR/nixl-posix.toml}"
  NAMESPACE_HELPER="$REPO_ROOT/scripts/pennyroyal/derive_namespace.py"
  required_executables+=("$NAMESPACE_HELPER")
fi
# Host-RAM HiCache tier: --hicache-size counts decimal GB (SGLang sizes the
# host pool at size * 1e9 bytes), not GiB. An unset PENNY_HICACHE_SIZE_GB keeps
# this recipe's qualified default; a chosen value must be a positive integer.
HICACHE_SIZE_GB="${PENNY_HICACHE_SIZE_GB:-32}"
if [[ ! "$HICACHE_SIZE_GB" =~ ^[1-9][0-9]*$ ]]; then
  echo "PENNY_HICACHE_SIZE_GB must be a positive integer number of GB, got '$HICACHE_SIZE_GB'" >&2
  exit 1
fi
# The draft verifies a whole block per request, so this profile admits six
# requests into thirty-six recurrent-state slots (the documented Flash-Next
# capacity pair) instead of the smaller NEXTN default. An operator choice still
# wins: the shared helper validates what it is handed.
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-6}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-36}"
export MAX_RUNNING_REQUESTS MAX_MAMBA_CACHE_SIZE
source "$SCRIPT_DIR/chat-template.sh"
source "$SCRIPT_DIR/request-capacity.sh"
source "$SCRIPT_DIR/reasoning-effort.sh"
source "$SCRIPT_DIR/tp-devices.sh"
# Name the release and its source before any preflight step can fail.
source "$SCRIPT_DIR/startup-summary.sh"
pennyroyal_version_banner

CONTEXT_LENGTH=524288
PAGE_SIZE=64
# TP1 is the qualified default. TP2 is expected to work; it has not been tested
# on two-GPU hardware, and the TP guard below still refuses a TP size the
# visible devices cannot serve.
TP_SIZE="${TP_SIZE:-1}"
if [[ ! "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "TP_SIZE must be a positive integer" >&2
  exit 1
fi
COMPUTE_DTYPE=bfloat16
KV_DTYPE=fp8_e4m3
DRAFT_KV_DTYPE=fp8_e4m3
MAMBA_SSM_DTYPE=bfloat16
MAMBA_CONV_DTYPE=bfloat16
MAMBA_TRACK_INTERVAL=64
PREFILL_CHUNK_SIZE=4096
LINEAR_ATTN_VERIFY_BACKEND=flashinfer
SPECULATIVE_ALGORITHM=DFLASH
# Verification width is routed, not tuned: the bundled fixed-width policy gives
# block 6 (steps 5) to a single-request decode batch and block 4 (steps 3) to
# every batch of two or more. The launch itself always asks for the widest
# routed width, because a routed width above the launch width is refused, not
# resized, under captured graphs. Decode graph buckets are left to the runtime's
# own spec-decode list, which covers the reachable batches including one; naming
# a narrower list here would leave the wider concurrency tiers without graphs.
# The policy is a runtime asset read from the repository, never copied to a host
# directory, and its hash is namespace identity below.
SPEC_STEPS=5
DRAFT_TOKENS=6
# The qualified Flash-Next FR-Spec map: 65,536 target-token rows. No checksum
# gate here — the file is a bundled repository asset, and its hash below is
# what keeps a swapped map out of this namespace rather than a startup refusal.
TOKEN_MAP="$SCRIPT_DIR/frspec/flash-next-64k.pt"
ADAPTIVE_CONFIG="$SCRIPT_DIR/adaptive-dflash2.json"
# The draft attends over the window the checkpoint was trained with, and its
# mask token id is part of what it drafts, so both are namespace identity here.
DRAFT_WINDOW_SIZE=2048
DRAFT_MASK_TOKEN_ID=248077
DFLASH_TARGET_LAYER_IDS=5,15,25,35,45
for path in "${required_executables[@]}"; do
  [[ -x "$path" ]] || { echo "Required executable missing: $path" >&2; exit 1; }
done
if [[ "$NIXL" == on ]]; then
  [[ -r "$NIXL_CONFIG" ]] || { echo "NIXL config missing: $NIXL_CONFIG" >&2; exit 1; }
fi
[[ -f "$TARGET_MODEL/config.json" && -f "$TARGET_MODEL/model.safetensors.index.json" ]] || {
  echo "Incomplete target checkpoint: $TARGET_MODEL" >&2
  exit 1
}
[[ -f "$DRAFT_MODEL/config.json" ]] || {
  echo "Incomplete draft checkpoint: $DRAFT_MODEL" >&2
  exit 1
}
[[ -f "$TOKEN_MAP" ]] || { echo "FR-Spec map missing: $TOKEN_MAP" >&2; exit 1; }
[[ -f "$ADAPTIVE_CONFIG" ]] || { echo "Width-routing policy missing: $ADAPTIVE_CONFIG" >&2; exit 1; }
read -r TOKEN_MAP_SHA _ < <(sha256sum "$TOKEN_MAP")
read -r ADAPTIVE_SHA _ < <(sha256sum "$ADAPTIVE_CONFIG")
mkdir -p "$CACHE_BASE"/{huggingface,torch,torchinductor,triton,cuda,flashinfer,sglang/jit}
if [[ "$NIXL" == on ]]; then
  mkdir -p "$NIXL_STORAGE_BASE"
fi

# Default to as many consecutive GPUs as TP requires (one scheduler process
# per visible device); an explicit CUDA_VISIBLE_DEVICES still wins.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TP_SIZE - 1)))"
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
export CC="${CC:-/usr/bin/gcc-15}" CXX="${CXX:-/usr/bin/g++-15}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6 8.9 12.0+PTX}"
export PENNY_BUILD_JOBS="${PENNY_BUILD_JOBS:-4}"
export MAX_JOBS="${MAX_JOBS:-$PENNY_BUILD_JOBS}" CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$PENNY_BUILD_JOBS}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$PENNY_BUILD_JOBS}"
export FLASHINFER_NINJA_JOBS="${FLASHINFER_NINJA_JOBS:-$PENNY_BUILD_JOBS}" FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-$PENNY_BUILD_JOBS}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HOME="$CACHE_BASE/huggingface" XDG_CACHE_HOME="$CACHE_BASE"
export TORCH_HOME="$CACHE_BASE/torch" TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor"
export TRITON_CACHE_DIR="$CACHE_BASE/triton" CUDA_CACHE_PATH="$CACHE_BASE/cuda"
export FLASHINFER_WORKSPACE_BASE="$CACHE_BASE/flashinfer"
export SGLANG_CACHE_DIR="$CACHE_BASE/sglang" SGLANG_JIT_CACHE_DIR="$CACHE_BASE/sglang/jit"
# TileLang's compiled kernels belong with the other durable caches: a container
# UID with no passwd entry gets HOME=/, so TileLang's ~ default lands on an
# unwritable /.tilelang and the import aborts (#65). An explicit value wins.
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$CACHE_BASE/tilelang}"
mkdir -p "$TILELANG_CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_NUMA_BIND_V2=false SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_MAMBA_CONV_DTYPE="$MAMBA_CONV_DTYPE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
# The startup summary would otherwise report every DFLASH launch as the 27B one.
export PENNYROYAL_PROFILE_LABEL="Flash-Next NVFP4 + DFlash2 (DFLASH, FR-Spec, routed widths)"

# TP selects a topology, it never grants GPUs: refuse to launch when the
# requested ranks (plus a dedicated cuda:N preprocessor) exceed what is
# visible, instead of letting NCCL fail or the request be ignored. Runs
# after the durable cache environment so its $PYTHON probe sees the same
# cache locations as every later Python invocation.
pennyroyal_check_tp_devices "$TP_SIZE" "$SGLANG_MM_PREPROCESS_DEVICE"

# NVMe preflight imports Torch, Triton, FlashInfer and SGLang. Activate their
# durable cache locations before selecting the optional backend.
source "$SCRIPT_DIR/ple-backend.sh"
# No KV-token preset: unset or blank MAX_TOTAL_TOKENS leaves the pool to the
# sizing SGLang profiles from the free memory, within the static memory fraction
# below. A deliberately exported cap stays opt-in and is still validated.
configure_max_total_tokens

# The target keeps the qualified Flash-Next long-context extension: the same
# YaRN override, theta and 262144-token base the other Flash-Next recipes pass.
# Nothing here retunes the target's rotary settings for the draft.
TARGET_OVERRIDES='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}'
# The draft's own 524K position extension, scaled from *this* draft's
# rope_theta 10000 and 262144-token base -- not the 27B draft's theta 1e7. The
# factor is the context length over that base, exactly as the 27B recipe states
# its pair. This makes the draft able to address 524K positions; it is not
# evidence that drafting is correct at that depth (the sliding window bounds its
# own attention, and long-context acceptance still has to be measured).
DRAFT_OVERRIDES='{"max_position_embeddings":524288,"rope_parameters":{"rope_type":"yarn","rope_theta":10000,"factor":2.0,"original_max_position_embeddings":262144}}'
SGLANG_REV="$(git -C "$REPO_ROOT" rev-parse --short=10 HEAD)"
PENNY_IDENTITY_PROBE="$("$PYTHON" -c 'import sys; import torch; from sglang.srt.utils import resolve_mm_preprocess_device; resolve_mm_preprocess_device(); from sglang.kernels.ops.gemm.sm120_online_fp8 import launch_precision; print(torch.__version__); print(launch_precision(sys.argv[1]))' "$TARGET_MODEL")"
TORCH_VERSION="$(printf '%s\n' "$PENNY_IDENTITY_PROBE" | sed -n 1p)"
# The namespace must name the EFFECTIVE precision, resolved before derivation
# through the same eligibility logic the runtime applies: automatic and
# explicit-on share the rowwise_fp8 identity (the legacy "true" named the old
# mixed-MXFP8 caches), while off and auto-off keep the untouched-weight
# identity. A probe that cannot answer falls back to the saved explicit choice.
# This matters more here than on the NEXTN recipes: the DFlash2 candidate
# projection is the rowwise-FP8 head, so a launch that lost the conversion is a
# different runtime, not a slower one.
ONLINE_FP8_PRECISION="$(printf '%s\n' "$PENNY_IDENTITY_PROBE" | sed -n 2p)"
case "$ONLINE_FP8_PRECISION" in
  rowwise_fp8|false) ;;
  *) if [[ "${SGLANG_SM120_ONLINE_MXFP8:-}" == true ]]; then ONLINE_FP8_PRECISION=rowwise_fp8; else ONLINE_FP8_PRECISION=false; fi ;;
esac
echo "Media preprocessing: $SGLANG_MM_PREPROCESS_DEVICE ($IMAGE_PROCESSOR_BACKEND); model GPU: cuda:0"
printf 'Pennyroyal profile: Flash-Next NVFP4 + DFlash2 / FR-Spec (routed block 6 at C1, block 4 at C2+)\n  runtime: %s\n  target: %s\n  draft: %s\n  token map: %s\n  width policy: %s\n  cache root: %s\n  NIXL root: %s\n' \
  "$SGLANG_EXE" "$TARGET_MODEL" "$DRAFT_MODEL" "$TOKEN_MAP" "$ADAPTIVE_CONFIG" "$CACHE_BASE" "${NIXL_STORAGE_BASE:-off}"
if [[ "$NIXL" != on ]]; then
  echo "NIXL disk tier: off; GPU radix cache and host-RAM HiCache only"
  HICACHE_STORAGE_ARGS=()
else
  echo "Deriving NIXL namespace; checkpoint identity hashing may take time..."
  NIXL_STORAGE="$("$NAMESPACE_HELPER" \
    --base-root "$NIXL_STORAGE_BASE" \
    --slug "qwen3_8_flash_next_frspec_524k_dflash2_${SGLANG_REV}" \
    --git-repo "$REPO_ROOT" \
    --model "target=$TARGET_MODEL" \
    --model "draft=$DRAFT_MODEL" \
    --field "chat_template_sha256=$CHAT_TEMPLATE_SHA" \
    --field "online_mxfp8=$ONLINE_FP8_PRECISION" \
    --field "image_processor_backend=$IMAGE_PROCESSOR_BACKEND" \
    --field "mm_preprocess_device=$SGLANG_MM_PREPROCESS_DEVICE" \
    --field "context_length=$CONTEXT_LENGTH" \
    --field "tp_size=$TP_SIZE" \
    --field "page_size=$PAGE_SIZE" \
    --field "compute_dtype=$COMPUTE_DTYPE" \
    --field "target_kv_dtype=$KV_DTYPE" \
    --field "draft_kv_dtype=$DRAFT_KV_DTYPE" \
    --field "speculative_algorithm=$SPECULATIVE_ALGORITHM" \
    --field "speculative_num_steps=$SPEC_STEPS" \
    --field "speculative_num_draft_tokens=$DRAFT_TOKENS" \
    --field "speculative_eagle_topk=1" \
    --field "speculative_adaptive=true" \
    --field "speculative_adaptive_policy_sha256=$ADAPTIVE_SHA" \
    --field "draft_token_map_sha256=$TOKEN_MAP_SHA" \
    --field "prefill_cuda_graph=disabled" \
    --field "dflash_target_layer_ids=$DFLASH_TARGET_LAYER_IDS" \
    --field "dflash_mask_token_id=$DRAFT_MASK_TOKEN_ID" \
    --field "draft_window_size=$DRAFT_WINDOW_SIZE" \
    --field "draft_quantization=unquant" \
    --field "gdn_mtp_cache_mode=none" \
    --field "gdn_fp16_accum_mma=$GDN_FP16_ACCUM_MMA" \
    --field "hicache_io_backend=kernel" \
    --field "hicache_mem_layout=page_first" \
    --field "mamba_ssm_dtype=$MAMBA_SSM_DTYPE" \
    --field "mamba_conv_dtype=$MAMBA_CONV_DTYPE" \
    --field "max_mamba_cache_size=$MAX_MAMBA_CACHE_SIZE" \
    --field "max_running_requests=$MAX_RUNNING_REQUESTS" \
    --field "mamba_radix_cache_strategy=extra_buffer" \
    --field "mamba_track_interval=$MAMBA_TRACK_INTERVAL" \
    --field "linear_attn_decode_backend=flashinfer" \
    --field "linear_attn_prefill_backend=flashinfer" \
    --field "linear_attn_verify_backend=$LINEAR_ATTN_VERIFY_BACKEND" \
    --field "ple_offload_embedding=$PLE_OFFLOAD_EMBEDDING" \
    "${PLE_NAMESPACE_ARGS[@]}" \
    --field "qsa_compressed_hicache=true" \
    --field "chunked_prefill_size=$PREFILL_CHUNK_SIZE" \
    --field "target_model_overrides=$TARGET_OVERRIDES" \
    --field "draft_model_overrides=$DRAFT_OVERRIDES" \
    --field "torch_version=$TORCH_VERSION" \
    --field "cuda_arch=12.0")"
  export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR="$NIXL_STORAGE"
  echo "NIXL FILE namespace: $NIXL_STORAGE"
  # With the disk tier off only these three storage-backend arguments
  # disappear; the GPU radix cache and the host-RAM tier above them stay.
  HICACHE_STORAGE_ARGS=(--hicache-storage-backend nixl \
    --hicache-storage-prefetch-policy timeout \
    --hicache-storage-backend-extra-config "@$NIXL_CONFIG")
fi

launch_args=(serve \
  --warmups=structured_output \
  --model-path "$TARGET_MODEL" \
  --load-format safetensors \
  --served-model-name pennyroyal \
  --host 0.0.0.0 --port 8001 --tp "$TP_SIZE" \
  --dtype "$COMPUTE_DTYPE" --quantization modelopt_fp4 --kv-cache-dtype "$KV_DTYPE" \
  --mem-fraction-static 0.95 \
  "${TOKEN_CAP_ARGS[@]}" \
  --context-length "$CONTEXT_LENGTH" --json-model-override-args "$TARGET_OVERRIDES" \
  --page-size "$PAGE_SIZE" --max-running-requests "$MAX_RUNNING_REQUESTS" --sleep-on-idle \
  --chunked-prefill-size "$PREFILL_CHUNK_SIZE" \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" --gdn-mtp-cache-mode none \
  --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer \
  --linear-attn-verify-backend "$LINEAR_ATTN_VERIFY_BACKEND" \
  --mamba-track-interval "$MAMBA_TRACK_INTERVAL" \
  --enable-hierarchical-cache --hicache-size "$HICACHE_SIZE_GB" --hicache-host-memory-mode cache \
  --hicache-write-policy write_through --hicache-io-backend kernel \
  --hicache-mem-layout page_first "${HICACHE_STORAGE_ARGS[@]}" \
  "${PLE_ARGS[@]}" --trust-remote-code \
  --chat-template "$CHAT_TEMPLATE" --image-processor-backend "$IMAGE_PROCESSOR_BACKEND" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-request-time-stats-logging --enable-metrics --enable-cache-report \
  --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS" \
  --speculative-algorithm "$SPECULATIVE_ALGORITHM" \
  --speculative-num-steps "$SPEC_STEPS" \
  --speculative-adaptive --speculative-adaptive-config "$ADAPTIVE_CONFIG" \
  --speculative-token-map "$TOKEN_MAP" \
  --disable-prefill-cuda-graph \
  --speculative-draft-model-path "$DRAFT_MODEL" \
  --speculative-draft-load-format safetensors \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-model-override-args "$DRAFT_OVERRIDES" \
  --speculative-num-draft-tokens "$DRAFT_TOKENS" \
  --speculative-draft-window-size "$DRAFT_WINDOW_SIZE" \
  --speculative-draft-kv-cache-dtype "$DRAFT_KV_DTYPE" \
  --watchdog-timeout 1800)
pennyroyal_startup_summary "${launch_args[@]}"
exec "$SGLANG_EXE" "${launch_args[@]}"
