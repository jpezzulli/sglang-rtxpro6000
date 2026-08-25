#!/usr/bin/env bash
set -euo pipefail

# Required local paths. The defaults are examples, not host-independent facts.
SGLANG_ROOT="${SGLANG_ROOT:-/opt/sglang}"
TARGET_MODEL="${TARGET_MODEL:-/srv/models/hf/orcarouter-Qwen3.8-27B-Uncensored-FP8}"
DRAFT_MODEL="${DRAFT_MODEL:-/srv/models/hf/incoai-Qwen3.8-27B-DFlash2}"
CACHE_BASE="${CACHE_BASE:-/srv/cache/sglang-runtime}"
NIXL_STORAGE_BASE="${NIXL_STORAGE_BASE:-/srv/cache/sglang_nixl}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-pennyroyal}"
LISTEN_HOST="${LISTEN_HOST:-0.0.0.0}"
LISTEN_PORT="${LISTEN_PORT:-8001}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NIXL_CONFIG="$REPO_ROOT/configs/pennyroyal/nixl-posix.toml"
NIXL_NAMESPACE_HELPER="$REPO_ROOT/scripts/pennyroyal/derive_namespace.py"

CONTEXT_LENGTH=524288
PAGE_SIZE=64
TP_SIZE=1
TARGET_COMPUTE_DTYPE=bfloat16
TARGET_KV_DTYPE=fp8_e4m3
DRAFT_KV_DTYPE=fp8_e4m3
DRAFT_QUANTIZATION=unquant
DRAFT_TOKENS=8
DRAFT_WINDOW_SIZE=2048
HICACHE_IO_BACKEND=kernel
HICACHE_MEM_LAYOUT=page_first
MAMBA_SSM_DTYPE=float32
MAMBA_CONV_DTYPE=bfloat16

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC="${CC:-/usr/bin/gcc-15}"
export CXX="${CXX:-/usr/bin/g++-15}"
export CUDAHOSTCXX="$CXX"
export TORCH_CUDA_ARCH_LIST=12.0

export MAX_JOBS=24
export CMAKE_BUILD_PARALLEL_LEVEL=24
export FLASHINFER_NINJA_JOBS=24
export FLASHINFER_NVCC_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=24

export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$CACHE_BASE/huggingface"
export XDG_CACHE_HOME="$CACHE_BASE"
export TORCH_HOME="$CACHE_BASE/torch"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor"
export TRITON_CACHE_DIR="$CACHE_BASE/triton"
export CUDA_CACHE_PATH="$CACHE_BASE/cuda"
export FLASHINFER_WORKSPACE_BASE="$CACHE_BASE/flashinfer"
export SGLANG_CACHE_DIR="$CACHE_BASE/sglang"
export SGLANG_JIT_CACHE_DIR="$CACHE_BASE/sglang/jit"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_PREP_IN_CUDA_GRAPH=1
export SGLANG_NUMA_BIND_V2=false
export SGLANG_MAMBA_CONV_DTYPE="$MAMBA_CONV_DTYPE"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

TARGET_OVERRIDES='{
  "text_config": {
    "max_position_embeddings": 524288,
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "rope_type": "yarn",
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25,
      "factor": 2.0,
      "original_max_position_embeddings": 262144
    }
  }
}'

DRAFT_OVERRIDES='{
  "max_position_embeddings": 524288,
  "rope_parameters": {
    "rope_type": "yarn",
    "rope_theta": 10000000,
    "factor": 2.0,
    "original_max_position_embeddings": 262144
  }
}'

SGLANG_REV="$(git -C "$SGLANG_ROOT/src" rev-parse --short=10 HEAD)"
TORCH_VERSION="$("$SGLANG_ROOT/.venv/bin/python" -c 'import torch; print(torch.__version__)')"
NIXL_STORAGE="$(
  "$NIXL_NAMESPACE_HELPER" \
    --base-root "$NIXL_STORAGE_BASE" \
    --slug "qwen3_8_27b_524k_dflash2_$SGLANG_REV" \
    --git-repo "$SGLANG_ROOT/src" \
    --model "target=$TARGET_MODEL" \
    --model "draft=$DRAFT_MODEL" \
    --field "context_length=$CONTEXT_LENGTH" \
    --field "tp_size=$TP_SIZE" \
    --field "page_size=$PAGE_SIZE" \
    --field "target_compute_dtype=$TARGET_COMPUTE_DTYPE" \
    --field "target_kv_dtype=$TARGET_KV_DTYPE" \
    --field "draft_kv_dtype=$DRAFT_KV_DTYPE" \
    --field "draft_quantization=$DRAFT_QUANTIZATION" \
    --field "speculative_algorithm=DFLASH" \
    --field "speculative_attention_mode=decode" \
    --field "draft_tokens=$DRAFT_TOKENS" \
    --field "draft_window_size=$DRAFT_WINDOW_SIZE" \
    --field "hicache_io_backend=$HICACHE_IO_BACKEND" \
    --field "hicache_mem_layout=$HICACHE_MEM_LAYOUT" \
    --field "mamba_ssm_dtype=$MAMBA_SSM_DTYPE" \
    --field "mamba_conv_dtype=$MAMBA_CONV_DTYPE" \
    --field "mamba_radix_cache_strategy=extra_buffer_lazy" \
    --field "prefill_attention_backend=flashinfer" \
    --field "decode_attention_backend=trtllm_mha" \
    --field "linear_attention_backend=triton" \
    --field "chunked_prefill_size=2048" \
    --field "max_prefill_tokens=2048" \
    --field "target_model_overrides=$TARGET_OVERRIDES" \
    --field "draft_model_overrides=$DRAFT_OVERRIDES" \
    --field "torch_version=$TORCH_VERSION" \
    --field "cuda_arch=12.0"
)"
export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR="$NIXL_STORAGE"
echo "NIXL FILE namespace: $NIXL_STORAGE"

exec "$SGLANG_ROOT/.venv/bin/sglang" serve \
  --model-path "$TARGET_MODEL" \
  --load-format safetensors \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$LISTEN_HOST" \
  --port "$LISTEN_PORT" \
  --tp "$TP_SIZE" \
  --dtype "$TARGET_COMPUTE_DTYPE" \
  --kv-cache-dtype "$TARGET_KV_DTYPE" \
  --context-length "$CONTEXT_LENGTH" \
  --page-size "$PAGE_SIZE" \
  --json-model-override-args "$TARGET_OVERRIDES" \
  --max-running-requests 4 \
  --sleep-on-idle \
  --max-mamba-cache-size 16 \
  --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-max-states-per-path 3 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 2048 \
  --attention-backend flashinfer \
  --decode-attention-backend trtllm_mha \
  --trust-remote-code \
  --chat-template "$TARGET_MODEL/chat_template.jinja" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-request-time-stats-logging \
  --enable-metrics \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}' \
  --enable-hierarchical-cache \
  --hicache-size 96 \
  --hicache-host-memory-mode cache \
  --hicache-write-policy write_through \
  --hicache-io-backend "$HICACHE_IO_BACKEND" \
  --hicache-mem-layout "$HICACHE_MEM_LAYOUT" \
  --hicache-storage-backend nixl \
  --hicache-storage-prefetch-policy timeout \
  --hicache-storage-backend-extra-config "@$NIXL_CONFIG" \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT_MODEL" \
  --speculative-draft-load-format safetensors \
  --speculative-draft-model-quantization "$DRAFT_QUANTIZATION" \
  --speculative-draft-model-override-args "$DRAFT_OVERRIDES" \
  --speculative-num-draft-tokens "$DRAFT_TOKENS" \
  --speculative-draft-window-size "$DRAFT_WINDOW_SIZE" \
  --speculative-attention-mode decode \
  --speculative-draft-attention-backend flashinfer \
  --speculative-draft-kv-cache-dtype "$DRAFT_KV_DTYPE" \
  --watchdog-timeout 1800
