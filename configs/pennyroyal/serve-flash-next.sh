#!/usr/bin/env bash
set -euo pipefail
# Avoid synchronous RAM compaction for transient NumPy image arrays.
# Operators can opt back into NumPy huge-page advice; this is not a kernel policy.
export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
export SGLANG_MM_PREPROCESS_DEVICE="${SGLANG_MM_PREPROCESS_DEVICE:-cpu}"
case "$SGLANG_MM_PREPROCESS_DEVICE" in
  cpu) IMAGE_PROCESSOR_BACKEND=pil ;;
  cuda:*) IMAGE_PROCESSOR_BACKEND=torchvision ;;
  *) echo "Choose SGLANG_MM_PREPROCESS_DEVICE=cpu or cuda:N" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
SGLANG_EXE="${SGLANG_EXE:-$REPO_ROOT/.venv/bin/sglang}"
PYTHON="${PYTHON:-$(dirname "$SGLANG_EXE")/python}"
TARGET_MODEL="${TARGET_MODEL:?Set TARGET_MODEL to the Flash-Next checkpoint}"
CACHE_BASE="${CACHE_BASE:?Set CACHE_BASE to the durable compiler-cache root}"
# Disk tier: on (the qualified default) adds the NIXL FILE backend under
# $NIXL_STORAGE_BASE. off keeps the GPU radix cache and the host-RAM HiCache
# tier and drops only the disk tier: no NIXL root, config, namespace derivation
# or storage-backend argument, and no data removal. Same choice the mounted
# docker/pennyroyal/launch/config/start-flash-next.sh script makes; the NVMe
# PLE reader below stays independent of it.
NIXL="${NIXL:-on}"
case "$NIXL" in
  on|off) ;;
  *) echo "NIXL must be on or off; got '$NIXL'" >&2; exit 1 ;;
esac
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
# Optional fixed native NEXTN verify window. Unset keeps the qualified W4
# default; the only published choices are 4 and 8. This is the recipe face
# of the existing width controls (speculative_num_steps = width - 1,
# speculative_eagle_topk = 1, speculative_num_draft_tokens = width): the
# QSA pending index-K ring takes its group count from the same resolved
# draft-token bound and the CUDA-graph static widths follow the server's
# derivation, so there is no second width knob. W8 is not GPU-accepted
# yet: RAM PLE only (ple-backend.sh refuses nvme+W8), and it stays out of
# the configurator until acceptance.
SPEC_WIDTH="${PENNY_SPEC_WIDTH:-4}"
case "$SPEC_WIDTH" in
  4|8) ;;
  *) echo "PENNY_SPEC_WIDTH must be 4 or 8; got '$SPEC_WIDTH'" >&2; exit 1 ;;
esac
SPEC_STEPS=$((SPEC_WIDTH - 1))
source "$SCRIPT_DIR/chat-template.sh"
source "$SCRIPT_DIR/request-capacity.sh"
source "$SCRIPT_DIR/reasoning-effort.sh"
source "$SCRIPT_DIR/tp-devices.sh"

CONTEXT_LENGTH=524288
PAGE_SIZE=64
# TP1 is the qualified default; TP_SIZE=2 opts into the two-GPU experimental
# path (the NIXL namespace hashes tp_size, so the two topologies can never
# share one cache root).
TP_SIZE="${TP_SIZE:-1}"
if [[ ! "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "TP_SIZE must be a positive integer" >&2
  exit 1
fi
COMPUTE_DTYPE=bfloat16
KV_DTYPE=fp8_e4m3
MAMBA_SSM_DTYPE=bfloat16
MAMBA_CONV_DTYPE=bfloat16
MAMBA_TRACK_INTERVAL=64
PREFILL_CHUNK_SIZE=4096
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
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_NUMA_BIND_V2=false SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_MAMBA_CONV_DTYPE="$MAMBA_CONV_DTYPE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

# TP selects a topology, it never grants GPUs: refuse to launch when the
# requested ranks (plus a dedicated cuda:N preprocessor) exceed what is
# visible, instead of letting NCCL fail or the request be ignored. Runs
# after the durable cache environment so its $PYTHON probe sees the same
# cache locations as every later Python invocation.
pennyroyal_check_tp_devices "$TP_SIZE" "$SGLANG_MM_PREPROCESS_DEVICE"

# NVMe preflight imports Torch, Triton, FlashInfer and SGLang. Activate their
# durable cache locations before selecting the optional backend.
source "$SCRIPT_DIR/ple-backend.sh"
configure_max_total_tokens

TARGET_OVERRIDES='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}'
SGLANG_REV="$(git -C "$REPO_ROOT" rev-parse --short=10 HEAD)"
TORCH_VERSION="$("$PYTHON" -c 'import torch; from sglang.srt.utils import resolve_mm_preprocess_device; resolve_mm_preprocess_device(); print(torch.__version__)')"
echo "Media preprocessing: $SGLANG_MM_PREPROCESS_DEVICE ($IMAGE_PROCESSOR_BACKEND); model GPU: cuda:0"
printf 'Pennyroyal profile: Flash-Next (native NEXTN, no FR-Spec)\n  runtime: %s\n  target: %s\n  cache root: %s\n  NIXL root: %s\n' \
  "$SGLANG_EXE" "$TARGET_MODEL" "$CACHE_BASE" "${NIXL_STORAGE_BASE:-off}"
if [[ "$NIXL" != on ]]; then
  echo "NIXL disk tier: off; GPU radix cache and host-RAM HiCache only"
  HICACHE_STORAGE_ARGS=()
else
  echo "Deriving NIXL namespace; checkpoint identity hashing may take time..."
  NIXL_STORAGE="$("$NAMESPACE_HELPER" \
    --base-root "$NIXL_STORAGE_BASE" \
    --slug "qwen3_8_flash_next_524k_nextn_${SGLANG_REV}" \
    --git-repo "$REPO_ROOT" \
    --model "target=$TARGET_MODEL" \
    --field "chat_template_sha256=$CHAT_TEMPLATE_SHA" \
    --field "online_mxfp8=$SGLANG_SM120_ONLINE_MXFP8" \
    --field "image_processor_backend=$IMAGE_PROCESSOR_BACKEND" \
    --field "mm_preprocess_device=$SGLANG_MM_PREPROCESS_DEVICE" \
    --field "context_length=$CONTEXT_LENGTH" \
    --field "tp_size=$TP_SIZE" \
    --field "page_size=$PAGE_SIZE" \
    --field "compute_dtype=$COMPUTE_DTYPE" \
    --field "target_kv_dtype=$KV_DTYPE" \
    --field "speculative_algorithm=NEXTN" \
    --field "speculative_num_steps=$SPEC_STEPS" \
    --field "speculative_eagle_topk=1" \
    --field "speculative_num_draft_tokens=$SPEC_WIDTH" \
    --field "speculative_draft_quantization=unquant" \
    --field "gdn_mtp_cache_mode=none" \
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
    --field "ple_offload_embedding=$PLE_OFFLOAD_EMBEDDING" \
  "${PLE_NAMESPACE_ARGS[@]}" \
    --field "qsa_compressed_hicache=true" \
    --field "chunked_prefill_size=$PREFILL_CHUNK_SIZE" \
    --field "target_model_overrides=$TARGET_OVERRIDES" \
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
  --mem-fraction-static 0.981 \
  "${TOKEN_CAP_ARGS[@]}" \
  --context-length "$CONTEXT_LENGTH" --json-model-override-args "$TARGET_OVERRIDES" \
  --page-size "$PAGE_SIZE" --max-running-requests "$MAX_RUNNING_REQUESTS" --sleep-on-idle \
  --chunked-prefill-size "$PREFILL_CHUNK_SIZE" \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" --gdn-mtp-cache-mode none \
  --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer \
  --mamba-track-interval "$MAMBA_TRACK_INTERVAL" \
  --enable-hierarchical-cache --hicache-size "$HICACHE_SIZE_GB" --hicache-host-memory-mode cache \
  --hicache-write-policy write_through --hicache-io-backend kernel \
  --hicache-mem-layout page_first "${HICACHE_STORAGE_ARGS[@]}" \
  "${PLE_ARGS[@]}" --trust-remote-code \
  --chat-template "$CHAT_TEMPLATE" --image-processor-backend "$IMAGE_PROCESSOR_BACKEND" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-request-time-stats-logging --enable-metrics \
  --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS" \
  --speculative-algorithm NEXTN --speculative-num-steps "$SPEC_STEPS" \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens "$SPEC_WIDTH" \
  --speculative-draft-model-quantization unquant --watchdog-timeout 1800)
source "$SCRIPT_DIR/startup-summary.sh"
pennyroyal_startup_summary "${launch_args[@]}"
exec "$SGLANG_EXE" "${launch_args[@]}"
