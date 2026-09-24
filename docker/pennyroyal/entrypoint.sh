#!/usr/bin/env bash
set -euo pipefail
export REPO_ROOT=/opt/pennyroyal
export SGLANG_EXE="$REPO_ROOT/.venv/bin/sglang"
export PYTHON="$REPO_ROOT/.venv/bin/python"
export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export CACHE_BASE="${CACHE_BASE:-/cache}"
export NIXL_STORAGE_BASE="${NIXL_STORAGE_BASE:-/nixl}"
export PENNY_PLE_PLUGIN_DIR="$REPO_ROOT/.ple-nvme"

# Compose forwards optional knobs with empty defaults; an empty passthrough
# must behave exactly like an unset variable (qualified defaults, untouched
# NCCL env), so normalize them away before the recipes read them.
for optional in PENNY_REASONING_EFFORT TP_SIZE NCCL_P2P_DISABLE; do
  if [[ -z ${!optional:-} ]]; then unset "$optional"; fi
done

profile="${1:-next}"
if (( $# )); then shift; fi
case "$profile" in
  --help|-h)
    printf '%s\n' \
      'Pennyroyal container' \
      '  next        Flash-Next NVFP4 + FR-Spec/native NEXTN (default)' \
      '  next-plain  Flash-Next NVFP4 + native NEXTN without FR-Spec' \
      '  27b         Qwen3.8-27B FP8 + DFlash2' \
      '  --check     Check installed packages and POSIX plugin without a GPU' \
      '  exec CMD    Run an explicit utility command inside the image' \
      'Set TARGET_MODEL; 27b also needs DRAFT_MODEL. Mount model directories' \
      'read-only and writable /cache and /nixl directories. See the container guide.'
    exit 0 ;;
  --check) exec "$PYTHON" "$REPO_ROOT/docker/pennyroyal/check_install.py" ;;
  exec)
    (( $# )) || { echo 'exec requires a command' >&2; exit 2; }
    exec "$@" ;;
  next) recipe=serve-flash-next-frspec.sh ;;
  next-plain) recipe=serve-flash-next.sh ;;
  27b) recipe=serve-qwen38-27b-dflash2.sh ;;
  *) echo "Unknown profile: $profile (use --help)" >&2; exit 2 ;;
esac
(( $# == 0 )) || { echo 'Configure the recipes through environment variables; see --help.' >&2; exit 2; }
for directory in "$CACHE_BASE" "$NIXL_STORAGE_BASE"; do
  if [[ ! -d "$directory" || ! -w "$directory" ]]; then
    echo "Mount a writable directory at $directory for container UID $(id -u)." >&2
    exit 1
  fi
done
# Docker/NVIDIA select the permitted host GPUs. Preserve all of those logical
# devices so an optional cuda:1 media processor is not hidden by the recipes'
# native single-GPU default. An explicit CUDA_VISIBLE_DEVICES still wins.
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  CUDA_VISIBLE_DEVICES="$("$PYTHON" -c 'import torch; print(",".join(map(str, range(torch.cuda.device_count()))))')"
  [[ -n "$CUDA_VISIBLE_DEVICES" ]] || {
    echo 'No NVIDIA GPU is visible. Check the host Container Toolkit and GPU selection.' >&2
    exit 1
  }
  export CUDA_VISIBLE_DEVICES
fi
exec bash "$REPO_ROOT/configs/pennyroyal/$recipe"
