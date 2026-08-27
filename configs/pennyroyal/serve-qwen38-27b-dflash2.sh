#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
SGLANG_EXE="${SGLANG_EXE:-$REPO_ROOT/.venv/bin/sglang}"
PYTHON="${PYTHON:-$(dirname "$SGLANG_EXE")/python}"
TARGET_MODEL="${TARGET_MODEL:?Set TARGET_MODEL to the Qwen3.8-27B target checkpoint}"
DRAFT_MODEL="${DRAFT_MODEL:?Set DRAFT_MODEL to the DFlash2 checkpoint}"
CACHE_BASE="${CACHE_BASE:?Set CACHE_BASE to the durable compiler-cache root}"
NIXL_STORAGE_BASE="${NIXL_STORAGE_BASE:?Set NIXL_STORAGE_BASE to the FILE cache root}"
NIXL_CONFIG="${NIXL_CONFIG:-$SCRIPT_DIR/nixl-posix.toml}"
NAMESPACE_HELPER="$REPO_ROOT/scripts/pennyroyal/derive_namespace.py"

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

for path in "$SGLANG_EXE" "$PYTHON" "$NAMESPACE_HELPER"; do
  [[ -x "$path" ]] || { echo "Required executable missing: $path" >&2; exit 1; }
done
[[ -r "$NIXL_CONFIG" ]] || { echo "NIXL config missing: $NIXL_CONFIG" >&2; exit 1; }
[[ -f "$TARGET_MODEL/config.json" && -f "$TARGET_MODEL/model.safetensors.index.json" ]] || {
  echo "Incomplete target checkpoint: $TARGET_MODEL" >&2
  exit 1
}
[[ -f "$DRAFT_MODEL/config.json" && -f "$DRAFT_MODEL/model.safetensors" ]] || {
  echo "Incomplete draft checkpoint: $DRAFT_MODEL" >&2
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
export SGLANG_PREP_IN_CUDA_GRAPH=1 SGLANG_MAMBA_CONV_DTYPE="$MAMBA_CONV_DTYPE"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

TARGET_OVERRIDES='{"text_config":{"max_position_embeddings":524288,"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}'
DRAFT_OVERRIDES='{"max_position_embeddings":524288,"rope_parameters":{"rope_type":"yarn","rope_theta":10000000,"factor":2.0,"original_max_position_embeddings":262144}}'
SGLANG_REV="$(git -C "$REPO_ROOT" rev-parse --short=10 HEAD)"
TORCH_VERSION="$("$PYTHON" -c 'import torch; print(torch.__version__)')"
NIXL_STORAGE="$($NAMESPACE_HELPER \
  --base-root "$NIXL_STORAGE_BASE" \
  --slug "qwen3_8_27b_524k_dflash2_${SGLANG_REV}" \
  --git-repo "$REPO_ROOT" \
  --model "target=$TARGET_MODEL" --model "draft=$DRAFT_MODEL" \
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

exec "$SGLANG_EXE" serve \
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
  --trust-remote-code --chat-template "$TARGET_MODEL/chat_template.jinja" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-request-time-stats-logging --enable-metrics \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}' \
  --enable-hierarchical-cache --hicache-size 96 --hicache-host-memory-mode cache \
  --hicache-write-policy write_through --hicache-io-backend kernel \
  --hicache-mem-layout page_first --hicache-storage-backend nixl \
  --hicache-storage-prefetch-policy timeout \
  --hicache-storage-backend-extra-config "@$NIXL_CONFIG" \
  --speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT_MODEL" \
  --speculative-draft-load-format safetensors \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-model-override-args "$DRAFT_OVERRIDES" \
  --speculative-num-draft-tokens "$DRAFT_TOKENS" \
  --speculative-draft-window-size "$DRAFT_WINDOW_SIZE" \
  --speculative-attention-mode decode \
  --speculative-draft-attention-backend flashinfer \
  --speculative-draft-kv-cache-dtype "$DRAFT_KV_DTYPE" \
  --watchdog-timeout 1800
