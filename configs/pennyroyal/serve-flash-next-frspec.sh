#!/usr/bin/env bash
# Pennyroyal 2.3: existing native-MTP FR-Spec, qualified with the full Penny context/cache envelope.
# Adoption credit: https://github.com/gabrielolympie/sglang-flashnext-sm120
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
SGLANG_EXE="${SGLANG_EXE:-$REPO_ROOT/.venv/bin/sglang}"
PYTHON="${PYTHON:-$(dirname "$SGLANG_EXE")/python}"
TARGET_MODEL="${TARGET_MODEL:?Set TARGET_MODEL to the Flash-Next checkpoint}"
CACHE_BASE="${CACHE_BASE:?Set CACHE_BASE to the durable compiler-cache root}"
NIXL_STORAGE_BASE="${NIXL_STORAGE_BASE:?Set NIXL_STORAGE_BASE to the FILE cache root}"
NIXL_CONFIG="${NIXL_CONFIG:-$SCRIPT_DIR/nixl-posix-frspec.toml}"
NAMESPACE_HELPER="$REPO_ROOT/scripts/pennyroyal/derive_namespace.py"

# Pin the qualified map and tokenizer: a different ID mapping changes draft
# proposals and must never silently reuse this representation's cache namespace.
TOKEN_MAP="$SCRIPT_DIR/frspec/flash-next-64k.pt"
read -r TOKEN_MAP_SHA _ < <(sha256sum "$TOKEN_MAP")
[[ "$TOKEN_MAP_SHA" == becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25 ]] || {
  echo "FR-Spec map differs from the qualified 2.3 artifact" >&2; exit 1;
}
read -r TOKENIZER_SHA _ < <(sha256sum "$TARGET_MODEL/tokenizer.json")
[[ "$TOKENIZER_SHA" == 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3 ]] || {
  echo "Target tokenizer differs from the qualified FR-Spec tokenizer" >&2; exit 1;
}

CONTEXT_LENGTH=524288
PAGE_SIZE=64
TP_SIZE=1
COMPUTE_DTYPE=bfloat16
KV_DTYPE=fp8_e4m3
MAMBA_SSM_DTYPE=bfloat16
MAMBA_CONV_DTYPE=bfloat16
MAMBA_TRACK_INTERVAL=64
PREFILL_CHUNK_SIZE=4096

for path in "$SGLANG_EXE" "$PYTHON" "$NAMESPACE_HELPER"; do
  [[ -x "$path" ]] || { echo "Required executable missing: $path" >&2; exit 1; }
done
[[ -r "$NIXL_CONFIG" ]] || { echo "NIXL config missing: $NIXL_CONFIG" >&2; exit 1; }
[[ -f "$TARGET_MODEL/config.json" && -f "$TARGET_MODEL/model.safetensors.index.json" ]] || {
  echo "Incomplete target checkpoint: $TARGET_MODEL" >&2
  exit 1
}
mkdir -p "$CACHE_BASE"/{huggingface,torch,torchinductor,triton,cuda,flashinfer,sglang/jit}
mkdir -p "$NIXL_STORAGE_BASE"

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
export CC="${CC:-/usr/bin/gcc-15}" CXX="${CXX:-/usr/bin/g++-15}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-24}" CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-24}"
export FLASHINFER_NINJA_JOBS="${FLASHINFER_NINJA_JOBS:-24}" FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-4}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-24}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HOME="$CACHE_BASE/huggingface" XDG_CACHE_HOME="$CACHE_BASE"
export TORCH_HOME="$CACHE_BASE/torch" TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor"
export TRITON_CACHE_DIR="$CACHE_BASE/triton" CUDA_CACHE_PATH="$CACHE_BASE/cuda"
export FLASHINFER_WORKSPACE_BASE="$CACHE_BASE/flashinfer"
export SGLANG_CACHE_DIR="$CACHE_BASE/sglang" SGLANG_JIT_CACHE_DIR="$CACHE_BASE/sglang/jit"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_NUMA_BIND_V2=false SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_MAMBA_CONV_DTYPE="$MAMBA_CONV_DTYPE"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

TARGET_OVERRIDES='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}'
SGLANG_REV="$(git -C "$REPO_ROOT" rev-parse --short=10 HEAD)"
TORCH_VERSION="$("$PYTHON" -c 'import torch; print(torch.__version__)')"
NIXL_STORAGE="$($NAMESPACE_HELPER \
  --base-root "$NIXL_STORAGE_BASE" \
  --slug "qwen3_8_flash_next_frspec_524k_nextn_${SGLANG_REV}" \
  --git-repo "$REPO_ROOT" \
  --model "target=$TARGET_MODEL" \
  --field "draft_token_map_sha256=$TOKEN_MAP_SHA" \
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
  --field "gdn_mtp_cache_mode=none" \
  --field "hicache_io_backend=kernel" \
  --field "hicache_mem_layout=page_first" \
  --field "mamba_ssm_dtype=$MAMBA_SSM_DTYPE" \
  --field "mamba_conv_dtype=$MAMBA_CONV_DTYPE" \
  --field "max_mamba_cache_size=24" \
  --field "mamba_radix_cache_strategy=extra_buffer" \
  --field "mamba_track_interval=$MAMBA_TRACK_INTERVAL" \
  --field "linear_attn_decode_backend=flashinfer" \
  --field "linear_attn_prefill_backend=flashinfer" \
  --field "ple_offload_embedding=true" \
  --field "qsa_compressed_hicache=true" \
  --field "chunked_prefill_size=$PREFILL_CHUNK_SIZE" \
  --field "target_model_overrides=$TARGET_OVERRIDES" \
  --field "torch_version=$TORCH_VERSION" \
  --field "cuda_arch=12.0")"
export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR="$NIXL_STORAGE"
echo "NIXL FILE namespace: $NIXL_STORAGE"

exec "$SGLANG_EXE" serve \
  --model-path "$TARGET_MODEL" \
  --load-format safetensors \
  --served-model-name pennyroyal \
  --host 0.0.0.0 --port 8001 --tp "$TP_SIZE" \
  --dtype "$COMPUTE_DTYPE" --quantization modelopt_fp4 --kv-cache-dtype "$KV_DTYPE" \
  --mem-fraction-static 0.981 \
  --context-length "$CONTEXT_LENGTH" --json-model-override-args "$TARGET_OVERRIDES" \
  --page-size "$PAGE_SIZE" --max-running-requests 4 --sleep-on-idle \
  --chunked-prefill-size "$PREFILL_CHUNK_SIZE" \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
  --max-mamba-cache-size 24 --gdn-mtp-cache-mode none \
  --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer \
  --mamba-track-interval "$MAMBA_TRACK_INTERVAL" \
  --enable-hierarchical-cache --hicache-size 32 --hicache-host-memory-mode cache \
  --hicache-write-policy write_through --hicache-io-backend kernel \
  --hicache-mem-layout page_first --hicache-storage-backend nixl \
  --hicache-storage-prefetch-policy timeout \
  --hicache-storage-backend-extra-config "@$NIXL_CONFIG" \
  --ple-offload-embedding --trust-remote-code \
  --chat-template "$TARGET_MODEL/chat_template.jinja" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-request-time-stats-logging --enable-metrics \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}' \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-draft-model-quantization unquant \
  --speculative-token-map "$TOKEN_MAP" --watchdog-timeout 1800
