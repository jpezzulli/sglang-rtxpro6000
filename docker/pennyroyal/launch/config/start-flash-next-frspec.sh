#!/usr/bin/env bash
# Pennyroyal Flash-Next with FR-Spec and the NIXL FILE tier (the default
# profile of the released image): startup settings for the prebuilt container.
#
# This is an ordinary SGLang startup script. Edit your settings below, then
# start it: ./run.sh (this file is the launcher's default). The launcher mounts
# this directory read-only at /config and the image's existing entrypoint runs
# 'exec bash /config/<this file>'.
#
# Nothing here is copied out of the image: the SGLang runtime and virtualenv,
# the shared launcher helpers, the pinned chat template, the pinned FR-Spec
# token map and the NIXL POSIX build all stay inside it and are reached through
# REPO_ROOT (the image's /opt/pennyroyal).
#
# The launch flags, namespace fields and pinned assets below are the ones the
# qualified configs/pennyroyal/serve-flash-next-frspec.sh recipe uses; keep
# them in step if you change one, and review the watermarks in
# nixl-posix-frspec.toml for your own filesystem.
set -euo pipefail

# --- Your settings ----------------------------------------------------------
# Quote a value that contains a space.
# Container paths: this directory's host counterpart is mounted at /models.
TARGET_MODEL="/models/RadixArk-Qwen3.8-Flash-Next-NVFP4"
# Host-RAM HiCache tier in decimal GB (SGLang sizes the pool at GB * 1e9).
HICACHE_SIZE_GB=32
# TP1 is the qualified topology; TP_SIZE=2 also needs run.sh --gpu 0,1.
TP_SIZE=1
# RAM keeps the original checkpoint and PLE path; nvme needs the prepared
# snapshot plus run.sh --nvme-ple (see tools/ple_nvme/README.md).
PENNY_PLE_BACKEND=ram
# cpu keeps vision work off the model GPU; cuda:N needs that extra device.
export SGLANG_MM_PREPROCESS_DEVICE=cpu
# Disk tier: on (the qualified default) adds the NIXL FILE backend under
# /nixl. off keeps the GPU radix cache and the host-RAM HiCache tier above and
# drops only the disk tier: no NIXL root, config, namespace derivation or
# storage-backend argument, and no data removal. run.sh decides separately
# whether /nixl is mounted; the two choices have to agree.
NIXL=on
# Flash-Next request/state capacity; the qualified pair is 4 requests/24 slots.
MAX_RUNNING_REQUESTS=4
MAX_MAMBA_CACHE_SIZE=24
# Optional proposal-only precision for the FR-Spec draft head: off keeps the
# shared rowwise-FP8/BF16 head, nvfp4 prepares the selected hot rows as NVFP4
# W4A16 while the unchanged target head keeps verifying every proposal.
SGLANG_FR_SPEC_PROPOSAL_HEAD_PRECISION=off
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

# The runtime reads the precision from the environment and the namespace field
# records it, so both come from this one value; keep the literal in step with
# NVFP4_NAMESPACE_LABEL in python/sglang/srt/speculative/proposal_head.py.
export SGLANG_FR_SPEC_PROPOSAL_HEAD_PRECISION="${SGLANG_FR_SPEC_PROPOSAL_HEAD_PRECISION:-off}"
PROPOSAL_HEAD_NAMESPACE_ARGS=()
case "$SGLANG_FR_SPEC_PROPOSAL_HEAD_PRECISION" in
  off) ;;
  nvfp4)
    PROPOSAL_HEAD_NAMESPACE_ARGS=(
      --field "fr_spec_proposal_head_precision=nvfp4_w4a16_marlin")
    ;;
  *)
    echo "SGLANG_FR_SPEC_PROPOSAL_HEAD_PRECISION must be off or nvfp4" >&2
    exit 1
    ;;
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
  NIXL_CONFIG="${NIXL_CONFIG:-$CONFIG_DIR/nixl-posix-frspec.toml}"
  NAMESPACE_HELPER="$REPO_ROOT/scripts/pennyroyal/derive_namespace.py"
  required_executables+=("$NAMESPACE_HELPER")
  writable_dirs+=("$NIXL_STORAGE_BASE")
fi

source "$IMAGE_CONFIGS/chat-template.sh"
source "$IMAGE_CONFIGS/request-capacity.sh"
source "$IMAGE_CONFIGS/reasoning-effort.sh"
source "$IMAGE_CONFIGS/tp-devices.sh"

# Pin the qualified map and tokenizer: a different ID mapping changes draft
# proposals and must never silently reuse this representation's cache namespace.
TOKEN_MAP="$IMAGE_CONFIGS/frspec/flash-next-64k.pt"

CONTEXT_LENGTH=524288
PAGE_SIZE=64
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
[[ -f "$TARGET_MODEL/tokenizer.json" ]] || {
  echo "Target tokenizer missing: $TARGET_MODEL/tokenizer.json" >&2
  exit 1
}
[[ -f "$TOKEN_MAP" ]] || { echo "FR-Spec map missing: $TOKEN_MAP" >&2; exit 1; }
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
export SGLANG_MAMBA_CONV_DTYPE="$MAMBA_CONV_DTYPE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

# TP selects a topology, it never grants GPUs: refuse to launch when the
# requested ranks (plus a dedicated cuda:N preprocessor) exceed what is
# visible, instead of letting NCCL fail or the request be ignored.
pennyroyal_check_tp_devices "$TP_SIZE" "$SGLANG_MM_PREPROCESS_DEVICE"

# NVMe preflight imports Torch, Triton, FlashInfer and SGLang. Activate their
# durable cache locations before selecting the optional backend.
source "$IMAGE_CONFIGS/ple-backend.sh"
configure_max_total_tokens 824384

TARGET_OVERRIDES='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}'
printf 'Pennyroyal profile: Flash-Next FR-Spec\n  runtime: %s\n  target: %s\n  token map: %s\n  cache root: %s\n  NIXL root: %s\n' \
  "$SGLANG_EXE" "$TARGET_MODEL" "$TOKEN_MAP" "$CACHE_BASE" "${NIXL_STORAGE_BASE:-off}"
echo "Verifying the pinned FR-Spec map and tokenizer..."
read -r TOKEN_MAP_SHA _ < <(sha256sum "$TOKEN_MAP")
[[ "$TOKEN_MAP_SHA" == becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25 ]] || {
  echo "FR-Spec map does not match the bundled checksum" >&2; exit 1;
}
read -r TOKENIZER_SHA _ < <(sha256sum "$TARGET_MODEL/tokenizer.json")
[[ "$TOKENIZER_SHA" == 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3 ]] || {
  echo "Target tokenizer differs from the qualified FR-Spec tokenizer" >&2; exit 1;
}
SGLANG_REV="$(git -C "$REPO_ROOT" rev-parse --short=10 HEAD)"
TORCH_VERSION="$("$PYTHON" -c 'import torch; from sglang.srt.utils import resolve_mm_preprocess_device; resolve_mm_preprocess_device(); print(torch.__version__)')"
echo "Media preprocessing: $SGLANG_MM_PREPROCESS_DEVICE ($IMAGE_PROCESSOR_BACKEND); model GPU: cuda:0"
if [[ "$NIXL" == on ]]; then
  echo "Deriving NIXL namespace; checkpoint identity hashing may take time..."
  NIXL_STORAGE="$("$NAMESPACE_HELPER" \
    --base-root "$NIXL_STORAGE_BASE" \
    --slug "qwen3_8_flash_next_frspec_524k_nextn_${SGLANG_REV}" \
    --git-repo "$REPO_ROOT" \
    --model "target=$TARGET_MODEL" \
    --field "chat_template_sha256=$CHAT_TEMPLATE_SHA" \
    --field "image_processor_backend=$IMAGE_PROCESSOR_BACKEND" \
    --field "mm_preprocess_device=$SGLANG_MM_PREPROCESS_DEVICE" \
    --field "draft_token_map_sha256=$TOKEN_MAP_SHA" \
    --field "online_mxfp8=$SGLANG_SM120_ONLINE_MXFP8" \
    --field "context_length=$CONTEXT_LENGTH" \
    --field "tp_size=$TP_SIZE" \
    --field "page_size=$PAGE_SIZE" \
    --field "compute_dtype=$COMPUTE_DTYPE" \
    --field "target_kv_dtype=$KV_DTYPE" \
    --field "speculative_algorithm=NEXTN" \
    --field "speculative_num_steps=3" \
    --field "speculative_eagle_topk=1" \
    --field "speculative_num_draft_tokens=4" \
    --field "speculative_draft_quantization=unquant" \
    "${PROPOSAL_HEAD_NAMESPACE_ARGS[@]}" \
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
  --model-path "$TARGET_MODEL" \
  --load-format safetensors \
  --served-model-name pennyroyal \
  --host 0.0.0.0 --port 8001 --tp "$TP_SIZE" \
  --dtype "$COMPUTE_DTYPE" --quantization modelopt_fp4 --kv-cache-dtype "$KV_DTYPE" \
  --mem-fraction-static 0.981 \
  "${TOKEN_CAP_ARGS[@]}" --warmups=structured_output \
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
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-draft-model-quantization unquant \
  --speculative-token-map "$TOKEN_MAP" --watchdog-timeout 1800)
source "$IMAGE_CONFIGS/startup-summary.sh"
pennyroyal_startup_summary "${launch_args[@]}"
exec "$SGLANG_EXE" "${launch_args[@]}"
