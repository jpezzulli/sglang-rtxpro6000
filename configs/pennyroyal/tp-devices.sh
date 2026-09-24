#!/usr/bin/env bash
# Shared Pennyroyal launcher guard for TP vs. visible GPUs (PR#18 TP plumbing);
# source from the recipes after TP_SIZE validation, CUDA_VISIBLE_DEVICES setup,
# and the durable cache environment exports -- and before the next $PYTHON
# invocation (e.g. the NVMe preflight), which must be the first Python the
# launch runs. TP_SIZE selects a topology, it never grants GPU
# access: Docker/Compose exposes only the GPUs named by
# deploy.resources.reservations.devices (one GPU by default), so a request for
# more ranks than there are visible inference devices must stop the launch
# here -- with the exact file and fragment to edit -- instead of failing deep
# inside NCCL or, worse, silently ignoring the requested TP. Sizing accounts
# for the existing preprocessing knob: model ranks take the logical devices
# 0..TP_SIZE-1, and SGLANG_MM_PREPROCESS_DEVICE=cuda:N outside that range is a
# dedicated extra device. Uses the same $PYTHON/torch view of
# CUDA_VISIBLE_DEVICES the server will see. Fail-closed guidance only: sets no
# environment, reserves no device, and adds no orchestration layer.
pennyroyal_check_tp_devices() {
  local tp_size="$1" preprocess="${2:-cpu}" required device_count preprocess_index
  required="$tp_size"
  if [[ "$preprocess" == cuda:* ]]; then
    preprocess_index="${preprocess#cuda:}"
    if [[ "$preprocess_index" =~ ^[0-9]+$ ]] && (( preprocess_index >= tp_size )); then
      required=$(( preprocess_index + 1 ))
    fi
  fi
  device_count="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
  if ! [[ "$device_count" =~ ^[0-9]+$ ]]; then
    echo "Could not read the visible CUDA device count from $PYTHON; refusing to launch TP_SIZE=$tp_size." >&2
    exit 1
  fi
  if (( device_count < required )); then
    echo "TP_SIZE=$tp_size needs at least $required visible CUDA device(s), but only $device_count is visible (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}', SGLANG_MM_PREPROCESS_DEVICE=$preprocess)." >&2
    echo "TP_SIZE does not grant GPU access. For Docker/Compose, edit the existing reservation under deploy.resources.reservations.devices in docker/pennyroyal/compose.yaml to name every GPU id; the complete item is (see the README's TP_SIZE section):" >&2
    echo '            - driver: nvidia' >&2
    echo '              device_ids: ["0", "1"]' >&2
    echo '              capabilities: [gpu]' >&2
    echo "Natively, export a CUDA_VISIBLE_DEVICES listing $required devices instead. Lower TP_SIZE or move the preprocessor onto a model GPU (cuda:N with N < TP_SIZE) to stay within $device_count device(s)." >&2
    exit 1
  fi
}
