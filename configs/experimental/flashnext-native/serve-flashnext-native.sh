#!/usr/bin/env bash
set -euo pipefail
# Experimental native bring-up of Flash-Next (Qwen4-Exp) on this upstream sglang
# checkout, staged from a minimal boot to the native-MTP/C6/HiCache
# configuration. Derived from the Pennyroyal v3.1.x serve-flash-next.sh recipe,
# mapped onto this base's CLI; portable by construction: every host, model and
# cache location is a variable, and nothing here hardcodes a machine path.
#
#   BRINGUP=1  minimal boot: model, NVFP4, YaRN, template and parsers only.
#              No speculative decode, no HiCache, no NIXL. The smoke gate.
#   BRINGUP=0  the full configuration below (the default).
#
# Full configuration (the v3.1.4 shapes, mapped to this base):
#   native NEXTN MTP with a selectable batch-tiered width policy (MTP_POLICY:
#   tiered / fixed-w4 / adaptive; the accepted adaptive table runs W4/W8 at
#   batch 1 behind the post-switch hold and W4 from batch 2, all launched at
#   W8 = steps 7 / 8 draft tokens), C6 with 36 recurrent-state slots, 524288
#   context via the YaRN override of a 262144-native checkpoint, page 64,
#   chunked prefill 4096, KV fp8_e4m3, HiCache with a write-through host tier
#   and the optional NIXL POSIX FILE tier under NIXL_STORAGE_BASE.
#
# Differences this recipe states instead of hiding (see the boot banners):
#   width policy      this base implements candidate_steps, the hysteresis knobs
#                     and the v3.0.2 switch_grace_batches/grace_backoff/
#                     max_grace_batches hold, so MTP_POLICY=adaptive honours the
#                     accepted table. What this base cannot claim yet is
#                     qualification: no matched-policy capture, replay or
#                     throughput measurement has been done here, so every
#                     policy -- tiered, fixed-w4 and adaptive alike -- is
#                     announced as mapped-and-configured, never as matched or
#                     qualified against the live adaptive-policy service.
#   FR-Spec           TOKEN_MAP is an optional input (the qualified 64K map is
#                     a baseline artifact, not shipped here). Unset means a
#                     full-vocabulary draft baseline, not a matched FR-Spec run.
#   --gdn-mtp-cache-mode none   not carried: this base has no such flag; its
#                     GDN MTP verify keeps the intermediate h-state cache (the
#                     v3.1.4 'full' behavior). Parity is a cache-work item.
#   media processing  SGLANG_MM_PREPROCESS_DEVICE is carried: unset keeps the
#                     model/platform-selected behavior, cpu parks image loading
#                     and fast preprocessing on the CPU, and cuda:N parks it on
#                     that logical device (a secondary GPU, e.g. an A4000,
#                     after CUDA_VISIBLE_DEVICES remapping). cuda:N requires
#                     the fast image processor backend, so the recipe's default
#                     switches to torchvision for it and a pinned pil backend
#                     fails at launch. Actual device offload is only claimed
#                     once a runtime check on the serving host proves it.
#
# Flag mapping from the v3.1.4 recipe (verified against this base's argument
# surface by test_serve_flashnext_native.py):
#   --tp            renamed here to --tp-size (alias --tensor-parallel-size)
#   everything else carried with its v3.1.4 spelling.
#
# Environment:
#   REPO_ROOT, TARGET_MODEL (required), CACHE_BASE (required),
#   PORT (8001), SERVED_MODEL_NAME (pennyroyal),
#   SGLANG_EXE ($REPO_ROOT/.venv/bin/sglang) or PYTHON,
#   NIXL=on|off with NIXL_STORAGE_BASE + NIXL_CONFIG (full mode; BRINGUP=1
#   forces the disk tier off and never reads those settings),
#   PENNY_HICACHE_SIZE_GB (32; decimal GB, full mode), MEM_FRACTION_STATIC (.97),
#   CONTEXT_LENGTH (524288), MODEL_OVERRIDE_JSON (default: the YaRN override),
#   MAX_RUNNING_REQUESTS (6) MAX_MAMBA_CACHE_SIZE (36) MAX_TOTAL_TOKENS (unset),
#   TP_SIZE (1), CHAT_TEMPLATE (optional pinned file),
#   IMAGE_PROCESSOR_BACKEND (pil, or torchvision when
#   SGLANG_MM_PREPROCESS_DEVICE=cuda:N; auto|torchvision|pil stays the real
#   backend knob), SGLANG_MM_PREPROCESS_DEVICE (unset|cpu|cuda:N; see above),
#   TOKEN_MAP (full mode; unset = full-vocabulary
#   draft and only validated when speculative decode actually launches),
#   PENNY_REASONING_EFFORT (medium default; OpenAI tiers),
#   MTP_POLICY=tiered|fixed-w4|adaptive (full mode), FLASHINFER_GDN_FP16_ACCUM_MMA (1; see below),
#   SGLANG_SM120_ONLINE_MXFP8 (unset|true|false; NIXL identity uses effective
#   resident precision, so equivalent automatic/explicit modes share a namespace).
#
# Accepted FlashInfer GDN mode: run the patched SM12x delta-rule prefill kernels
# in FP16-accumulate MMA mode (scripts/pennyroyal/flashinfer/install.py); this
# is the qualified default of every Flash-Next profile, exported before Python
# imports FlashInfer. Export FLASHINFER_GDN_FP16_ACCUM_MMA=0 before launch to
# opt out; FlashInfer's own default stays off.
# It changes the GDN computation, so the resolved value is part of the derived
# NIXL namespace below; anything but the literal 1 is the FP32-accumulate
# (stock-representation) cache identity.
export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
MM_PREPROCESS_DEVICE="${SGLANG_MM_PREPROCESS_DEVICE:-}"
case "$MM_PREPROCESS_DEVICE" in
  ""|cpu) MM_BACKEND_DEFAULT=pil ;;
  cuda:*)
    [[ "$MM_PREPROCESS_DEVICE" =~ ^cuda:[0-9]+$ ]] || { echo "SGLANG_MM_PREPROCESS_DEVICE must be cpu or a logical cuda:N such as cuda:1; got '$MM_PREPROCESS_DEVICE'" >&2; exit 1; }
    MM_BACKEND_DEFAULT=torchvision ;;
  *) echo "SGLANG_MM_PREPROCESS_DEVICE must stay unset (model/platform-selected) or select cpu or cuda:N; got '$MM_PREPROCESS_DEVICE'" >&2; exit 1 ;;
esac
# bash treats an empty export as unset; the Python resolver reads "" as an
# explicit invalid value, so make the child environment agree with the guard.
if [[ -z "$MM_PREPROCESS_DEVICE" ]]; then unset SGLANG_MM_PREPROCESS_DEVICE; else export SGLANG_MM_PREPROCESS_DEVICE="$MM_PREPROCESS_DEVICE"; fi
IMAGE_PROCESSOR_BACKEND="${IMAGE_PROCESSOR_BACKEND:-$MM_BACKEND_DEFAULT}"
case "$IMAGE_PROCESSOR_BACKEND" in
  auto|torchvision) ;;
  pil)
    if [[ "$MM_PREPROCESS_DEVICE" == cuda:* ]]; then
      echo "SGLANG_MM_PREPROCESS_DEVICE=$MM_PREPROCESS_DEVICE preprocesses on a CUDA device and needs the fast image processor backend; set IMAGE_PROCESSOR_BACKEND=auto or torchvision, not pil" >&2; exit 1
    fi ;;
  *) echo "IMAGE_PROCESSOR_BACKEND must be auto, torchvision or pil; got '$IMAGE_PROCESSOR_BACKEND'" >&2; exit 1 ;;
esac
export FLASHINFER_GDN_FP16_ACCUM_MMA="${FLASHINFER_GDN_FP16_ACCUM_MMA:-1}"
GDN_FP16_ACCUM_MMA=off
if [[ "$FLASHINFER_GDN_FP16_ACCUM_MMA" == 1 ]]; then GDN_FP16_ACCUM_MMA=on; fi
# Normalize the online-FP8 escape hatch and state what will be resident.
# What joins the NIXL identity
# is the EFFECTIVE precision (online_fp8_identity.py, on this checkpoint's
# metadata, this launch's dtype and the device this recipe makes visible), not
# the requested mode: automatic and explicit true are the same row-wise E4M3
# weights, and automatic-off and explicit false are the same untouched BF16
# checkpoint, so the requested mode would split equivalent launches into
# separate KV namespaces. It stays in the boot banner as a diagnostic.
ONLINE_FP8_MODE=auto
case "${SGLANG_SM120_ONLINE_MXFP8:-}" in
  "") SGLANG_SM120_ONLINE_MXFP8=""; export SGLANG_SM120_ONLINE_MXFP8 ;;
  false|true) ONLINE_FP8_MODE="$SGLANG_SM120_ONLINE_MXFP8"; export SGLANG_SM120_ONLINE_MXFP8 ;;
  *) echo "SGLANG_SM120_ONLINE_MXFP8 must be true or false (unset/blank = automatic)" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
SGLANG_EXE="${SGLANG_EXE:-$REPO_ROOT/.venv/bin/sglang}"
PYTHON="${PYTHON:-$(dirname "$SGLANG_EXE")/python}"
TARGET_MODEL="${TARGET_MODEL:?Set TARGET_MODEL to the Flash-Next checkpoint}"
CACHE_BASE="${CACHE_BASE:?Set CACHE_BASE to the durable compiler-cache root}"
PORT="${PORT:-8001}"
# The maintained clients and the regression suite address this id; keep it.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-pennyroyal}"
BRINGUP="${BRINGUP:-0}"
NIXL="${NIXL:-on}"
case "$NIXL" in on|off) ;; *) echo "NIXL must be on or off; got '$NIXL'" >&2; exit 1 ;; esac
if [[ "$BRINGUP" == 1 ]]; then NIXL=off; fi
TP_SIZE="${TP_SIZE:-1}"
if [[ ! "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then echo "TP_SIZE must be a positive integer" >&2; exit 1; fi
CONTEXT_LENGTH="${CONTEXT_LENGTH:-524288}"
# The checkpoint's native 262144 context, extended by the approved YaRN
# override (factor 2.0). Kept in its own variable: a ${VAR:-{json}} default
# re-splits the expansion once the caller sets VAR and appends stray braces.
# This JSON is cache identity: it is hashed into the NIXL namespace below, so
# a changed rope policy never reuses another one's KV pages.
MODEL_OVERRIDE_DEFAULT='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}'
MODEL_OVERRIDE_JSON="${MODEL_OVERRIDE_JSON:-$MODEL_OVERRIDE_DEFAULT}"
PAGE_SIZE="${PAGE_SIZE:-64}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-.97}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-4096}"
COMPUTE_DTYPE=bfloat16
KV_DTYPE=fp8_e4m3
MAMBA_SSM_DTYPE=bfloat16
MAMBA_CONV_DTYPE=bfloat16
MAMBA_TRACK_INTERVAL="${MAMBA_TRACK_INTERVAL:-64}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-6}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-36}"
for capacity_name in MAX_RUNNING_REQUESTS MAX_MAMBA_CACHE_SIZE; do
  if [[ ! ${!capacity_name} =~ ^[1-9][0-9]*$ ]]; then echo "$capacity_name must be a positive integer" >&2; exit 1; fi
done
TOKEN_CAP_ARGS=()
if [[ -n "${MAX_TOTAL_TOKENS:-}" ]]; then
  if [[ ! "$MAX_TOTAL_TOKENS" =~ ^[0-9]+$ ]] || (( 10#$MAX_TOTAL_TOKENS <= 10#$PAGE_SIZE )); then
    echo "MAX_TOTAL_TOKENS must be a positive integer" >&2; exit 1
  fi
  if (( 10#$MAX_TOTAL_TOKENS % 10#$PAGE_SIZE != 0 )); then
    echo "MAX_TOTAL_TOKENS must be aligned to page size $PAGE_SIZE" >&2; exit 1
  fi
  TOKEN_CAP_ARGS=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi
# The approved native-MTP width policy, the FR-Spec token map and the host-tier
# size are FULL-mode inputs only: a BRINGUP=1 smoke boot must not require or
# hash them, so their validation lives inside the full-mode branch below.

CHAT_TEMPLATE_ARGS=()
if [[ -n "${CHAT_TEMPLATE:-}" ]]; then
  [[ -r "$CHAT_TEMPLATE" ]] || { echo "Chat template missing: $CHAT_TEMPLATE" >&2; exit 1; }
  read -r CHAT_TEMPLATE_SHA _ < <(sha256sum "$CHAT_TEMPLATE")
  CHAT_TEMPLATE_ARGS=(--chat-template "$CHAT_TEMPLATE")
else
  CHAT_TEMPLATE_SHA=checkpoint-default
fi
# Launcher-only reasoning default (the qualified medium tier); an explicit
# per-request reasoning_effort still wins at serving time.
PENNY_REASONING_EFFORT_TIERS="none minimal low medium high xhigh max"
PENNY_REASONING_EFFORT_TRIMMED="$(printf '%s' "${PENNY_REASONING_EFFORT:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
PENNY_REASONING_EFFORT_NORMALIZED="medium"
if [[ -n "$PENNY_REASONING_EFFORT_TRIMMED" ]]; then
  PENNY_REASONING_EFFORT_LOWER="$(printf '%s' "$PENNY_REASONING_EFFORT_TRIMMED" | tr '[:upper:]' '[:lower:]')"
  for tier in $PENNY_REASONING_EFFORT_TIERS; do
    if [[ "$PENNY_REASONING_EFFORT_LOWER" == "$tier" ]]; then PENNY_REASONING_EFFORT_NORMALIZED="$tier"; break; fi
  done
  if [[ "$PENNY_REASONING_EFFORT_NORMALIZED" == "medium" && "$PENNY_REASONING_EFFORT_LOWER" != "medium" ]]; then
    echo "PENNY_REASONING_EFFORT must be one of: $PENNY_REASONING_EFFORT_TIERS; got '$PENNY_REASONING_EFFORT_TRIMMED'" >&2; exit 1
  fi
fi
DEFAULT_CHAT_TEMPLATE_KWARGS="{\"enable_thinking\":true,\"preserve_thinking\":true,\"reasoning_effort\":\"${PENNY_REASONING_EFFORT_NORMALIZED}\"}"

if [[ "$NIXL" == on ]]; then
  NIXL_STORAGE_BASE="${NIXL_STORAGE_BASE:?Set NIXL_STORAGE_BASE to the FILE cache root}"
  NIXL_CONFIG="${NIXL_CONFIG:-$SCRIPT_DIR/nixl-posix.toml}"
  NAMESPACE_HELPER="$REPO_ROOT/scripts/pennyroyal/derive_namespace.py"
  FP8_IDENTITY_HELPER="$SCRIPT_DIR/online_fp8_identity.py"
  [[ -r "$NIXL_CONFIG" ]] || { echo "NIXL config missing: $NIXL_CONFIG" >&2; exit 1; }
  [[ -f "$NAMESPACE_HELPER" ]] || { echo "Namespace helper missing: $NAMESPACE_HELPER" >&2; exit 1; }
  [[ -f "$FP8_IDENTITY_HELPER" ]] || { echo "FP8 identity helper missing: $FP8_IDENTITY_HELPER" >&2; exit 1; }
fi
[[ -f "$TARGET_MODEL/config.json" && -f "$TARGET_MODEL/model.safetensors.index.json" ]] || {
  echo "Incomplete target checkpoint: $TARGET_MODEL" >&2; exit 1
}
for path in "$SGLANG_EXE" "$PYTHON"; do
  [[ -x "$path" ]] || { echo "Required executable missing: $path" >&2; exit 1; }
done

mkdir -p "$CACHE_BASE"/{huggingface,torch,torchinductor,triton,cuda,flashinfer,sglang/jit,tilelang}
if [[ "$NIXL" == on ]]; then mkdir -p "$NIXL_STORAGE_BASE"; fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TP_SIZE - 1)))"
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
# Compiler and cache environment for any first-use JIT (FlashInfer AOT, Triton,
# TorchInductor, TileLang). CC/CXX/CUDA_HOME stay the caller's unless set here;
# a host that knows its toolchain should export them.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6 8.9 12.0+PTX}"
BUILD_JOBS="${PENNY_BUILD_JOBS:-${MAX_JOBS:-4}}"
export MAX_JOBS="$BUILD_JOBS" CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$BUILD_JOBS}"
export FLASHINFER_NINJA_JOBS="${FLASHINFER_NINJA_JOBS:-$BUILD_JOBS}" FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-$BUILD_JOBS}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$CACHE_BASE/huggingface" XDG_CACHE_HOME="$CACHE_BASE"
export TORCH_HOME="$CACHE_BASE/torch" TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor"
export TRITON_CACHE_DIR="$CACHE_BASE/triton" CUDA_CACHE_PATH="$CACHE_BASE/cuda"
export FLASHINFER_WORKSPACE_BASE="$CACHE_BASE/flashinfer"
export SGLANG_CACHE_DIR="$CACHE_BASE/sglang" SGLANG_JIT_CACHE_DIR="$CACHE_BASE/sglang/jit"
# TileLang's cache belongs with the other durable caches: a container UID with
# no passwd entry gets HOME=/, so its ~ default lands on an unwritable
# /.tilelang and the import aborts. An explicit value wins.
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$CACHE_BASE/tilelang}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_NUMA_BIND_V2=false SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_MAMBA_CONV_DTYPE="$MAMBA_CONV_DTYPE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

SGLANG_REV="$(git -C "$REPO_ROOT" rev-parse --short=10 HEAD)"
TORCH_VERSION="$("$PYTHON" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"

if [[ -n "$MM_PREPROCESS_DEVICE" ]]; then
  MM_DEVICE_FIELD=(--field "mm_preprocess_device=$MM_PREPROCESS_DEVICE")
else
  MM_DEVICE_FIELD=()
fi

launch_args=(serve
  --warmups=structured_output
  --model-path "$TARGET_MODEL"
  --load-format safetensors
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0 --port "$PORT" --tp-size "$TP_SIZE"
  --dtype "$COMPUTE_DTYPE" --quantization modelopt_fp4 --kv-cache-dtype "$KV_DTYPE"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  "${TOKEN_CAP_ARGS[@]}"
  --context-length "$CONTEXT_LENGTH" --json-model-override-args "$MODEL_OVERRIDE_JSON"
  --page-size "$PAGE_SIZE" --max-running-requests "$MAX_RUNNING_REQUESTS" --sleep-on-idle
  --chunked-prefill-size "$PREFILL_CHUNK_SIZE"
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype "$MAMBA_SSM_DTYPE"
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE"
  --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer
  --mamba-track-interval "$MAMBA_TRACK_INTERVAL"
  "${CHAT_TEMPLATE_ARGS[@]}" --image-processor-backend "$IMAGE_PROCESSOR_BACKEND"
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  --enable-request-time-stats-logging --enable-metrics
  --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS"
  --trust-remote-code --watchdog-timeout 1800)

# On SM120 this base resolves modelopt_fp4 with moe_runner_backend=auto to
# flashinfer_cutlass before any trtllm-gen fallback (arg_groups/overrides.py:
# the trtllm-gen MoE kernels are SM100-only, and the generic ModelOpt
# auto->flashinfer_trtllm resolution is guarded by the SM100 platform check).
# The resolution is logged at boot; the smoke check reads the log rather than
# re-deriving it here, and DeepGEMM FP8 is not an NVFP4 replacement.

if [[ "$BRINGUP" != 1 ]]; then
  # Host-RAM HiCache tier: --hicache-size counts decimal GB (the host pool
  # sizes itself at host_size * 1e9 bytes), not GiB.
  HICACHE_SIZE_GB="${PENNY_HICACHE_SIZE_GB:-32}"
  if [[ ! "$HICACHE_SIZE_GB" =~ ^[1-9][0-9]*$ ]]; then
    echo "PENNY_HICACHE_SIZE_GB must be a positive integer number of GB, got '$HICACHE_SIZE_GB'" >&2; exit 1
  fi
  # The approved native-MTP width policies ship as ordinary
  # --speculative-adaptive-config files beside this script. A candidate wider
  # than the launch width is refused, not resized, under captured graphs, so
  # the policy and the launch maximum are selected together; the default
  # spec-decode capture list keeps the BS1 bucket that every wide (steps 7) C1
  # tier needs -- without it a batch of one pads into the C>=2 tier and stays W4.
  MTP_POLICY="${MTP_POLICY:-tiered}"
  case "$MTP_POLICY" in
    tiered)   MTP_POLICY_FILE=adaptive-next.json;          MTP_SPEC_STEPS=7; MTP_DRAFT_TOKENS=8 ;;
    fixed-w4) MTP_POLICY_FILE=adaptive-next-fixed-w4.json; MTP_SPEC_STEPS=3; MTP_DRAFT_TOKENS=4 ;;
    # The accepted v3.0.2 table: C1 chooses W4 or W8 behind the 40-batch
    # post-switch hold (doubled per reversal, capped at 80); C>=2 is fixed W4.
    # Same launch maximum as the tiered table, so the same W8 capture bounds.
    adaptive) MTP_POLICY_FILE=adaptive-next-v3.0.2.json;   MTP_SPEC_STEPS=7; MTP_DRAFT_TOKENS=8 ;;
    *) echo "MTP_POLICY must be tiered, fixed-w4 or adaptive; got '$MTP_POLICY'" >&2; exit 1 ;;
  esac
  ADAPTIVE_CONFIG="$SCRIPT_DIR/$MTP_POLICY_FILE"
  [[ -r "$ADAPTIVE_CONFIG" ]] || { echo "MTP width policy is missing or unreadable: $ADAPTIVE_CONFIG" >&2; exit 1; }
  read -r ADAPTIVE_SHA _ < <(sha256sum "$ADAPTIVE_CONFIG")
  # FR-Spec reduced-vocabulary drafts: TOKEN_MAP points at an existing
  # qualified map (a baseline runtime asset; this recipe does not ship or
  # fabricate one). The map changes draft proposals, so its hash joins the
  # NIXL identity below and an unset map is announced as the full-vocabulary
  # baseline instead of being silently presented as a matched FR-Spec launch.
  TOKEN_MAP_ARGS=()
  if [[ -n "${TOKEN_MAP:-}" ]]; then
    [[ -r "$TOKEN_MAP" ]] || { echo "FR-Spec token map is not readable: $TOKEN_MAP" >&2; exit 1; }
    read -r TOKEN_MAP_SHA _ < <(sha256sum "$TOKEN_MAP")
    TOKEN_MAP_ARGS=(--speculative-token-map "$TOKEN_MAP")
  else
    TOKEN_MAP_SHA=full-vocabulary
    echo "FR-Spec: off (full-vocabulary NEXTN draft baseline, not the qualified reduced-vocabulary run)"
  fi
  echo "MTP width policy: $MTP_POLICY ($MTP_POLICY_FILE, launch W$((MTP_SPEC_STEPS + 1))); the selected table's hold/backoff keys are honoured on this base, and every width switch lands at a batch boundary inside the captured buckets --" \
    "this launch is stated as mapped, not as matched or qualified (no matched-policy capture, replay or throughput measurement on this base yet)."
  launch_args+=(
    --enable-hierarchical-cache --hicache-size "$HICACHE_SIZE_GB" --hicache-host-memory-mode cache
    --hicache-write-policy write_through --hicache-io-backend kernel
    --hicache-mem-layout page_first
    --speculative-algorithm NEXTN --speculative-num-steps "$MTP_SPEC_STEPS"
    --speculative-eagle-topk 1 --speculative-num-draft-tokens "$MTP_DRAFT_TOKENS"
    --speculative-adaptive --speculative-adaptive-config "$ADAPTIVE_CONFIG"
    --speculative-draft-model-quantization unquant "${TOKEN_MAP_ARGS[@]}")
  if [[ "$NIXL" == on ]]; then
    echo "Deriving NIXL namespace; checkpoint identity hashing may take time..."
    # Resident representation, resolved from the checkpoint, this dtype and the
    # visible device -- so an explicit true on hardware this build cannot serve
    # aborts the launch here, before a namespace is claimed for it.
    if ! ONLINE_FP8_PRECISION="$("$PYTHON" "$FP8_IDENTITY_HELPER" \
        --model "$TARGET_MODEL" --dtype "$COMPUTE_DTYPE" --requested "$ONLINE_FP8_MODE")"; then
      echo "Cannot name the online-FP8 precision of this launch (requested: $ONLINE_FP8_MODE)" >&2
      exit 1
    fi
    echo "Online FP8: resident=$ONLINE_FP8_PRECISION requested=$ONLINE_FP8_MODE"
    NIXL_STORAGE="$("$PYTHON" "$NAMESPACE_HELPER" \
      --base-root "$NIXL_STORAGE_BASE" \
      --slug "flashnext_${CONTEXT_LENGTH}_nextn_${SGLANG_REV}" \
      --git-repo "$REPO_ROOT" \
      --model "target=$TARGET_MODEL" \
      --field "chat_template_sha256=$CHAT_TEMPLATE_SHA" \
      --field "image_processor_backend=$IMAGE_PROCESSOR_BACKEND" \
      "${MM_DEVICE_FIELD[@]}" \
      --field "context_length=$CONTEXT_LENGTH" \
      --field "tp_size=$TP_SIZE" \
      --field "page_size=$PAGE_SIZE" \
      --field "compute_dtype=$COMPUTE_DTYPE" \
      --field "target_kv_dtype=$KV_DTYPE" \
      --field "model_override_json=$MODEL_OVERRIDE_JSON" \
      --field "speculative_algorithm=NEXTN" \
      --field "speculative_num_steps=$MTP_SPEC_STEPS" \
      --field "speculative_eagle_topk=1" \
      --field "speculative_num_draft_tokens=$MTP_DRAFT_TOKENS" \
      --field "speculative_adaptive=true" \
      --field "speculative_adaptive_policy_sha256=$ADAPTIVE_SHA" \
      --field "speculative_draft_quantization=unquant" \
      --field "speculative_token_map_sha256=$TOKEN_MAP_SHA" \
      --field "online_fp8_precision=$ONLINE_FP8_PRECISION" \
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
      --field "chunked_prefill_size=$PREFILL_CHUNK_SIZE" \
      --field "torch_version=$TORCH_VERSION" \
      --field "cuda_arch=12.0")"
    export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR="$NIXL_STORAGE"
    echo "NIXL FILE namespace: $NIXL_STORAGE"
    launch_args+=(--hicache-storage-backend nixl
      --hicache-storage-prefetch-policy timeout
      --hicache-storage-backend-extra-config "@$NIXL_CONFIG")
  else
    echo "NIXL disk tier: off; GPU radix cache and host-RAM HiCache only"
  fi
else
  echo "BRINGUP=1: minimal boot (no HiCache, no NEXTN, no NIXL disk tier)"
fi

printf 'flashnext-native profile: %s\n  served model id: %s\n  runtime: %s\n  target: %s\n  cache root: %s\n  NIXL root: %s\n' \
  "$([[ "$BRINGUP" == 1 ]] && echo bringup || echo "$([[ "$NIXL" == on ]] && echo full+nixl || echo full)")" \
  "$SERVED_MODEL_NAME" "$SGLANG_EXE" "$TARGET_MODEL" "$CACHE_BASE" "${NIXL_STORAGE:-off}"
if [[ -n "$MM_PREPROCESS_DEVICE" ]]; then
  echo "Media preprocessing: parked on $MM_PREPROCESS_DEVICE (image processor backend: $IMAGE_PROCESSOR_BACKEND)"
else
  echo "Media preprocessing: model/platform-selected (image processor backend: $IMAGE_PROCESSOR_BACKEND)"
fi
exec "$SGLANG_EXE" "${launch_args[@]}"
