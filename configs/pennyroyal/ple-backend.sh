#!/usr/bin/env bash
# Shared by the two Flash-Next recipes, not the 27B recipe.
# RAM mode leaves its original arguments, import path and namespace unchanged.
PENNY_PLE_BACKEND="${PENNY_PLE_BACKEND:-ram}"
PLE_ARGS=(--ple-offload-embedding)
PLE_NAMESPACE_ARGS=()
PLE_OFFLOAD_EMBEDDING=true

configure_max_total_tokens() {
  local default_cap="${1:-}"
  local token_cap="${MAX_TOTAL_TOKENS:-$default_cap}"
  TOKEN_CAP_ARGS=()
  if [[ -z "$token_cap" ]]; then
    return
  fi
  if [[ ! "$token_cap" =~ ^[0-9]+$ ]] || (( 10#$token_cap <= 0 )); then
    echo "MAX_TOTAL_TOKENS must be a positive integer" >&2
    return 1
  fi
  if (( 10#$token_cap % PAGE_SIZE != 0 )); then
    echo "MAX_TOTAL_TOKENS must be aligned to page size $PAGE_SIZE" >&2
    return 1
  fi
  TOKEN_CAP_ARGS=(--max-total-tokens "$token_cap")
}

SGLANG_SM120_ONLINE_MXFP8="${SGLANG_SM120_ONLINE_MXFP8:-false}"
case "$SGLANG_SM120_ONLINE_MXFP8" in
  false|true) export SGLANG_SM120_ONLINE_MXFP8 ;;
  *)
    echo "SGLANG_SM120_ONLINE_MXFP8 must be true or false" >&2
    exit 1
    ;;
esac

case "$PENNY_PLE_BACKEND" in
  ram) ;;
  nvme)
    : "${PENNY_PLE_NVME_MODEL:?Set PENNY_PLE_NVME_MODEL to the prepared NVMe snapshot}"
    PENNY_PLE_PLUGIN_DIR="${PENNY_PLE_PLUGIN_DIR:-$REPO_ROOT/.ple-nvme}"
    [[ -f "$PENNY_PLE_PLUGIN_DIR/sglang_ssd_stream/plugin.py" ]] || {
      echo "Optional NVMe reader missing; see tools/ple_nvme/README.md" >&2; exit 1;
    }
    # Do not enable arbitrary installed plugins or their replacement runtimes.
    if [[ -n "${SGLANG_PLUGINS:-}" && "$SGLANG_PLUGINS" != ssd_stream ]]; then
      echo "NVMe PLE recipe cannot combine unqualified SGLANG_PLUGINS" >&2; exit 1
    fi
    export PENNY_PLE_BACKEND SGLANG_PLUGINS=ssd_stream
    export PYTHONPATH="$PENNY_PLE_PLUGIN_DIR:$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
    echo "Checking NVMe PLE artifact and reader compatibility..." >&2
    PLE_MANIFEST_SHA="$(CUDA_VISIBLE_DEVICES='' "$PYTHON" \
      "$REPO_ROOT/scripts/pennyroyal/check_ple_nvme.py" \
      --source "$TARGET_MODEL" --prepared "$PENNY_PLE_NVME_MODEL")"
    [[ "$PLE_MANIFEST_SHA" =~ ^[0-9a-f]{64}$ ]] || {
      echo "NVMe PLE preflight did not return a valid identity" >&2; exit 1;
    }
    TARGET_MODEL="$PENNY_PLE_NVME_MODEL"
    PLE_ARGS=()
    PLE_OFFLOAD_EMBEDDING=false
    PLE_NAMESPACE_ARGS=(--field "ple_backend=nvme"
      --field "ple_manifest_sha256=$PLE_MANIFEST_SHA"
      --field "ple_reader_version=0.2.0+pennyroyal2")
    ;;
  *) echo "PENNY_PLE_BACKEND must be ram or nvme" >&2; exit 1 ;;
esac
