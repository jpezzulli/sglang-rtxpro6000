#!/usr/bin/env bash
# Qwen3.8-27B FP8 with the DFlash2 draft and the NIXL FILE tier: startup
# settings for the prebuilt container image.
#
# This is an ordinary SGLang startup script. Edit your settings below, then
# start it: ../run.sh --startup config/start-27b-dflash2.sh. The launcher
# mounts this directory read-only at /config and the image's existing
# entrypoint runs 'exec bash /config/<this file>'.
#
# Nothing here is copied out of the image: the SGLang runtime and virtualenv,
# the shared launcher helpers, the pinned chat template and the NIXL POSIX
# build all stay inside it and are reached through REPO_ROOT.
#
# The launch flags, namespace fields and pinned assets below are the ones the
# qualified configs/pennyroyal/serve-qwen38-27b-dflash2.sh recipe uses; keep
# them in step if you change one, and review the watermarks in
# nixl-posix.toml for your own filesystem. This profile has no PLE offload and
# therefore needs no io_uring permission of its own.
set -euo pipefail

# --- Your settings ----------------------------------------------------------
# Quote a value that contains a space.
# Container paths: this directory's host counterpart is mounted at /models.
# Both checkpoints are required: the FP8 target plus its DFlash2 draft.
TARGET_MODEL="/models/Qwen3.8-27B-FP8"
DRAFT_MODEL="/models/Qwen3.8-27B-DFlash2"
# Host-RAM HiCache tier in decimal GB (SGLang sizes the pool at GB * 1e9).
HICACHE_SIZE_GB=96
# cpu keeps vision work off the model GPU; cuda:N needs that extra device.
export SGLANG_MM_PREPROCESS_DEVICE=cpu
# Disk tier: on (the qualified default) adds the NIXL FILE backend under
# /nixl. off keeps the GPU radix cache and the host-RAM HiCache tier above and
# drops only the disk tier: no NIXL root, config, namespace derivation or
# storage-backend argument, and no data removal. run.sh decides separately
# whether /nixl is mounted; the two choices have to agree.
NIXL=on
# ----------------------------------------------------------------------------

export NUMPY_MADVISE_HUGEPAGE=0
# The launcher may forward the operator's saved choice; unset keeps the
# qualified default, exactly like configs/pennyroyal/serve-flash-next.sh.
export SGLANG_FORWARD_UNKNOWN_TOOLS="${SGLANG_FORWARD_UNKNOWN_TOOLS:-true}"
case "$SGLANG_MM_PREPROCESS_DEVICE" in
  cpu) IMAGE_PROCESSOR_BACKEND=pil ;;
  cuda:*) IMAGE_PROCESSOR_BACKEND=torchvision ;;
  *) echo "Choose SGLANG_MM_PREPROCESS_DEVICE=cpu or cuda:N" >&2; exit 1 ;;
esac

CONFIG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/opt/pennyroyal}"
IMAGE_CONFIGS="$REPO_ROOT/configs/pennyroyal"
SGLANG_EXE="${SGLANG_EXE:-$REPO_ROOT/.venv/bin/sglang}"
PYTHON="${PYTHON:-$(dirname "$SGLANG_EXE")/python}"
# /cache and /nixl are the container mount points run.sh publishes.
CACHE_BASE="${CACHE_BASE:?Set CACHE_BASE to the durable compiler-cache root}"
case "$NIXL" in
  on|off) ;;
  *) echo "NIXL must be on or off; got '$NIXL'" >&2; exit 1 ;;
esac
required_executables=("$SGLANG_EXE" "$PYTHON")
writable_dirs=("$CACHE_BASE")
# Disk tier off needs no NIXL root, no NIXL config and no namespace
# helper: the settings block above stays the only source of truth.
if [[ "$NIXL" == on ]]; then
  NIXL_STORAGE_BASE="${NIXL_STORAGE_BASE:?Set NIXL_STORAGE_BASE to the FILE cache root}"
  NIXL_CONFIG="${NIXL_CONFIG:-$CONFIG_DIR/nixl-posix.toml}"
  NAMESPACE_HELPER="$REPO_ROOT/scripts/pennyroyal/derive_namespace.py"
  required_executables+=("$NAMESPACE_HELPER")
  writable_dirs+=("$NIXL_STORAGE_BASE")
fi

source "$IMAGE_CONFIGS/chat-template.sh"
source "$IMAGE_CONFIGS/reasoning-effort.sh"
source "$IMAGE_CONFIGS/tp-devices.sh"

CONTEXT_LENGTH=524288
PAGE_SIZE=64
TP_SIZE=1
COMPUTE_DTYPE=bfloat16
TARGET_KV_DTYPE=fp8_e4m3
DRAFT_KV_DTYPE=fp8_e4m3
MAMBA_SSM_DTYPE=float32
MAMBA_CONV_DTYPE=bfloat16
MAMBA_TRACK_INTERVAL=256
DRAFT_TOKENS=8
DRAFT_WINDOW_SIZE=2048
if [[ ! "$HICACHE_SIZE_GB" =~ ^[1-9][0-9]*$ ]]; then
  echo "HICACHE_SIZE_GB must be a positive integer number of GB, got '$HICACHE_SIZE_GB'" >&2
  exit 1
fi

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
[[ -f "$DRAFT_MODEL/config.json" && -f "$DRAFT_MODEL/model.safetensors" ]] || {
  echo "Incomplete draft checkpoint: $DRAFT_MODEL" >&2
  exit 1
}
# The image entrypoint checks these for its built-in profiles; the exec path
# leaves the roots this script actually uses to the script itself.
for directory in "${writable_dirs[@]}"; do
  if [[ ! -d "$directory" || ! -w "$directory" ]]; then
    echo "Mount a writable directory at $directory for container UID $(id -u)." >&2
    exit 1
  fi
done
mkdir -p "$CACHE_BASE"/{huggingface,torch,torchinductor,triton,cuda,flashinfer,sglang/jit}
if [[ "$NIXL" == on ]]; then
  mkdir -p "$NIXL_STORAGE_BASE"
fi

# Docker/NVIDIA select the permitted host GPUs. The exec entrypoint hands the
# launch to this script, so keep every granted device visible here instead of
# narrowing to the model's own ranks: an optional cuda:N media processor needs
# a device outside 0..TP_SIZE-1, and the TP guard below reads the same view.
# An explicit CUDA_VISIBLE_DEVICES still wins.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$("$PYTHON" -c 'import torch; print(",".join(map(str, range(torch.cuda.device_count()))))')"
  [[ -n "$CUDA_VISIBLE_DEVICES" ]] || {
    echo 'No NVIDIA GPU is visible. Check the host Container Toolkit and GPU selection.' >&2
    exit 1
  }
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
export SGLANG_PREP_IN_CUDA_GRAPH=1 SGLANG_MAMBA_CONV_DTYPE="$MAMBA_CONV_DTYPE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

# TP stays 1 here, but the guard still turns a dedicated cuda:N preprocessor
# outside the model range into an actionable failure, never a silent ignore.
pennyroyal_check_tp_devices "$TP_SIZE" "$SGLANG_MM_PREPROCESS_DEVICE"

TARGET_OVERRIDES='{"text_config":{"max_position_embeddings":524288,"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}'
DRAFT_OVERRIDES='{"max_position_embeddings":524288,"rope_parameters":{"rope_type":"yarn","rope_theta":10000000,"factor":2.0,"original_max_position_embeddings":262144}}'
SGLANG_REV="$(git -C "$REPO_ROOT" rev-parse --short=10 HEAD)"
TORCH_VERSION="$("$PYTHON" -c 'import torch; from sglang.srt.utils import resolve_mm_preprocess_device; resolve_mm_preprocess_device(); print(torch.__version__)')"
echo "Media preprocessing: $SGLANG_MM_PREPROCESS_DEVICE ($IMAGE_PROCESSOR_BACKEND); model GPU: cuda:0"
printf 'Pennyroyal profile: Qwen3.8-27B FP8 + DFlash2\n  runtime: %s\n  target: %s\n  draft: %s\n  cache root: %s\n  NIXL root: %s\n' \
  "$SGLANG_EXE" "$TARGET_MODEL" "$DRAFT_MODEL" "$CACHE_BASE" "${NIXL_STORAGE_BASE:-off}"
if [[ "$NIXL" == on ]]; then
  echo "Deriving NIXL namespace; checkpoint identity hashing may take time..."
  NIXL_STORAGE="$("$NAMESPACE_HELPER" \
    --base-root "$NIXL_STORAGE_BASE" \
    --slug "qwen3_8_27b_524k_dflash2_${SGLANG_REV}" \
    --git-repo "$REPO_ROOT" \
    --model "target=$TARGET_MODEL" --model "draft=$DRAFT_MODEL" \
    --field "chat_template_sha256=$CHAT_TEMPLATE_SHA" \
    --field "image_processor_backend=$IMAGE_PROCESSOR_BACKEND" \
    --field "mm_preprocess_device=$SGLANG_MM_PREPROCESS_DEVICE" \
    --field "context_length=$CONTEXT_LENGTH" --field "tp_size=$TP_SIZE" \
    --field "page_size=$PAGE_SIZE" --field "compute_dtype=$COMPUTE_DTYPE" \
    --field "target_kv_dtype=$TARGET_KV_DTYPE" --field "draft_kv_dtype=$DRAFT_KV_DTYPE" \
    --field "draft_quantization=unquant" --field "speculative_algorithm=DFLASH" \
    --field "speculative_attention_mode=decode" --field "draft_tokens=$DRAFT_TOKENS" \
    --field "draft_window_size=$DRAFT_WINDOW_SIZE" --field "hicache_io_backend=kernel" \
    --field "hicache_mem_layout=page_first" --field "mamba_ssm_dtype=$MAMBA_SSM_DTYPE" \
    --field "mamba_conv_dtype=$MAMBA_CONV_DTYPE" --field "max_mamba_cache_size=24" \
    --field "mamba_max_states_per_path=5" --field "mamba_track_interval=$MAMBA_TRACK_INTERVAL" \
    --field "mamba_radix_cache_strategy=extra_buffer_lazy" \
    --field "prefill_attention_backend=flashinfer" \
    --field "decode_attention_backend=trtllm_mha" --field "linear_attention_backend=triton" \
    --field "chunked_prefill_size=2048" --field "max_prefill_tokens=2048" \
    --field "target_model_overrides=$TARGET_OVERRIDES" --field "draft_model_overrides=$DRAFT_OVERRIDES" \
    --field "torch_version=$TORCH_VERSION" --field "cuda_arch=12.0")"
  export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR="$NIXL_STORAGE"
  echo "NIXL FILE namespace: $NIXL_STORAGE"
else
  echo "NIXL disk tier: off; GPU radix cache and host-RAM HiCache only"
fi

# With the disk tier off only these three storage-backend arguments
# disappear; the GPU radix cache and the host-RAM tier above them stay.
HICACHE_STORAGE_ARGS=()
if [[ "$NIXL" == on ]]; then
  HICACHE_STORAGE_ARGS=(--hicache-storage-backend nixl \
    --hicache-storage-prefetch-policy timeout \
    --hicache-storage-backend-extra-config "@$NIXL_CONFIG")
fi

launch_args=(serve \
  --warmups=structured_output \
  --model-path "$TARGET_MODEL" --load-format safetensors \
  --served-model-name pennyroyal --host 0.0.0.0 --port 8001 --tp "$TP_SIZE" \
  --dtype "$COMPUTE_DTYPE" --kv-cache-dtype "$TARGET_KV_DTYPE" \
  --context-length "$CONTEXT_LENGTH" --page-size "$PAGE_SIZE" \
  --json-model-override-args "$TARGET_OVERRIDES" \
  --max-running-requests 4 --sleep-on-idle --mem-fraction-static 0.92 \
  --max-mamba-cache-size 24 --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
  --mamba-radix-cache-strategy extra_buffer_lazy --mamba-max-states-per-path 5 \
  --mamba-track-interval "$MAMBA_TRACK_INTERVAL" \
  --chunked-prefill-size 2048 --max-prefill-tokens 2048 \
  --attention-backend flashinfer --decode-attention-backend trtllm_mha \
  --trust-remote-code --chat-template "$CHAT_TEMPLATE" \
  --image-processor-backend "$IMAGE_PROCESSOR_BACKEND" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-request-time-stats-logging --enable-metrics \
  --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS" \
  --enable-hierarchical-cache --hicache-size "$HICACHE_SIZE_GB" --hicache-host-memory-mode cache \
  --hicache-write-policy write_through --hicache-io-backend kernel \
  --hicache-mem-layout page_first "${HICACHE_STORAGE_ARGS[@]}" \
  --speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT_MODEL" \
  --speculative-draft-load-format safetensors \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-model-override-args "$DRAFT_OVERRIDES" \
  --speculative-num-draft-tokens "$DRAFT_TOKENS" \
  --speculative-draft-window-size "$DRAFT_WINDOW_SIZE" \
  --speculative-attention-mode decode \
  --speculative-draft-attention-backend flashinfer \
  --speculative-draft-kv-cache-dtype "$DRAFT_KV_DTYPE" \
  --watchdog-timeout 1800)
source "$IMAGE_CONFIGS/startup-summary.sh"
pennyroyal_startup_summary "${launch_args[@]}"
exec "$SGLANG_EXE" "${launch_args[@]}"
