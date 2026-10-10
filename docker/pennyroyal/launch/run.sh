#!/usr/bin/env bash
# Prebuilt-image launcher for Pennyroyal: download this file plus the config
# directory, edit the settings, and start the server. No image rebuild, no
# source checkout, no host Python and no Compose are involved.
#
# Division of labour:
#   run.sh        this file: image, GPU, port and the three mount choices
#   config/*.sh   the SGLang startup script (model, runtime and cache settings),
#                 run by the image's own entrypoint inside the container
#   config/*.toml operational NIXL settings, used when the NIXL tier is enabled
#
# The directory holding the selected startup script is mounted read-only at
# /config (a directory, not a single file, so an editor that writes a new file
# and renames it is still visible on the next start), and the image runs
# `pennyroyal exec bash /config/<name>`, which is the entrypoint's existing
# exec capability. The image's SGLang, virtualenv, helper scripts, pinned chat
# template, pinned FR-Spec map and NIXL build all stay inside the image; the
# startup script reaches them through REPO_ROOT (default /opt/pennyroyal).
#
# Usage: ./run.sh [option]...
# Every option also has a setting above; the option wins.
#   --startup PATH             startup script to run (default: config/start-flash-next-frspec.sh)
#   --nixl-config PATH         NIXL config to use; omit to keep the one your
#                              startup script names inside the config directory
#   --no-nixl                  skip the /nixl mount, for a startup script whose
#                              NIXL setting is off (no root, config or backend).
#                              The two choices are separate and must agree.
#   --nvme-ple                 keep the io_uring-permitting seccomp setting for the
#                              independent NVMe PLE reader while NIXL is off
#   --image NAME               container image (default: the released v2.5.3 image)
#   --port HOST_PORT           published server port (default: 8001)
#   --gpu IDS                  GPU id list, e.g. 0 or 0,1 (default: 0)
#   --models DIR               host models root, mounted read-only at /models
#   --cache DIR                host durable cache root, mounted writable at /cache
#   --nixl-root DIR            host NIXL root, mounted writable at /nixl
#   --user UID:GID             identity owning the writable directories (default: 1000:1000)
#   --help, -h                 this text
#
# Optional recipe knobs (PENNY_HICACHE_SIZE_GB, TP_SIZE, PENNY_PLE_BACKEND,
# PENNY_REASONING_EFFORT, MAX_TOTAL_TOKENS, ...) are read by the startup script
# from the container environment. This example forwards none of them: set them
# in the startup script, or add '-e NAME=value' to the docker arguments below.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# --- Your host settings ------------------------------------------------------
IMAGE="${PENNYROYAL_IMAGE:-ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.3}"
STARTUP="${PENNYROYAL_STARTUP:-$SCRIPT_DIR/config/start-flash-next-frspec.sh}"
# Empty means: use the NIXL config the selected startup script names itself.
NIXL_CONFIG="${PENNYROYAL_NIXL_CONFIG:-}"
HOST_MODELS_ROOT="${HOST_MODELS_ROOT:-/srv/models}"
HOST_CACHE_BASE="${HOST_CACHE_BASE:-/var/cache/pennyroyal}"
HOST_NIXL_STORAGE_BASE="${HOST_NIXL_STORAGE_BASE:-/srv/pennyroyal-nixl}"
PORT="${PENNYROYAL_PORT:-8001}"
GPU="${NVIDIA_GPU:-0}"
RUN_AS="${PENNYROYAL_USER:-1000:1000}"
NIXL=on
NVME_PLE=off
# -----------------------------------------------------------------------------

usage() {
  sed -n '/^# Prebuilt-image launcher/,/^set -euo/p' "${BASH_SOURCE[0]}" |
    sed -e '/^set -euo/d' -e 's/^# \{0,1\}//'
}

# A --volume value without a leading '/' is a named volume, not a host
# directory, so every host path is resolved after it has been checked.
host_path() {
  (cd -- "$1" && pwd)
}

while (( $# )); do
  case "$1" in
    --startup) STARTUP="${2:?--startup needs a path}"; shift 2 ;;
    --startup=*) STARTUP="${1#*=}"; shift ;;
    --nixl-config) NIXL_CONFIG="${2:?--nixl-config needs a path}"; shift 2 ;;
    --nixl-config=*) NIXL_CONFIG="${1#*=}"; shift ;;
    --no-nixl) NIXL=off; shift ;;
    --nvme-ple) NVME_PLE=on; shift ;;
    --image) IMAGE="${2:?--image needs a name}"; shift 2 ;;
    --image=*) IMAGE="${1#*=}"; shift ;;
    --port) PORT="${2:?--port needs a number}"; shift 2 ;;
    --port=*) PORT="${1#*=}"; shift ;;
    --gpu) GPU="${2:?--gpu needs a device id}"; shift 2 ;;
    --gpu=*) GPU="${1#*=}"; shift ;;
    --models) HOST_MODELS_ROOT="${2:?--models needs a path}"; shift 2 ;;
    --models=*) HOST_MODELS_ROOT="${1#*=}"; shift ;;
    --cache) HOST_CACHE_BASE="${2:?--cache needs a path}"; shift 2 ;;
    --cache=*) HOST_CACHE_BASE="${1#*=}"; shift ;;
    --nixl-root) HOST_NIXL_STORAGE_BASE="${2:?--nixl-root needs a path}"; shift 2 ;;
    --nixl-root=*) HOST_NIXL_STORAGE_BASE="${1#*=}"; shift ;;
    --user) RUN_AS="${2:?--user needs UID:GID}"; shift 2 ;;
    --user=*) RUN_AS="${1#*=}"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1 (./run.sh --help)" >&2; exit 2 ;;
  esac
done

command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 1; }
# Fail here, on the host, instead of letting the container fall back to the
# image's default profile with a startup file the operator never chose.
[[ -f "$STARTUP" && -r "$STARTUP" ]] || {
  echo "Startup script is missing or unreadable: $STARTUP" >&2
  exit 1
}
[[ -d "$HOST_MODELS_ROOT" && -r "$HOST_MODELS_ROOT" ]] || {
  echo "Models root is missing or unreadable: $HOST_MODELS_ROOT" >&2
  exit 1
}
[[ -d "$HOST_CACHE_BASE" && -w "$HOST_CACHE_BASE" ]] || {
  echo "Cache root is missing or not writable: $HOST_CACHE_BASE" >&2
  exit 1
}
HOST_MODELS_ROOT="$(host_path "$HOST_MODELS_ROOT")"
HOST_CACHE_BASE="$(host_path "$HOST_CACHE_BASE")"

docker_args=(
  run --rm --init
  --user "$RUN_AS"
  # The device list needs literal double quotes: docker reads --gpus as CSV, so
  # an unquoted device=0,1 arrives as device=0 plus the count 1.
  --gpus "\"device=$GPU\""
  --publish "$PORT:8001"
  --shm-size 16g
  --ulimit memlock=-1:-1
  --stop-timeout 120
)

# Mount the script's directory, not the file, and keep every custom location
# working the same way: its own directory lands on /config (or /nixl-config).
startup_dir="$(cd -- "$(dirname -- "$STARTUP")" && pwd)"
docker_args+=(--volume "$startup_dir:/config:ro" --volume "$HOST_MODELS_ROOT:/models:ro")
docker_args+=(--volume "$HOST_CACHE_BASE:/cache")

if [[ "$NIXL" == on ]]; then
  [[ -d "$HOST_NIXL_STORAGE_BASE" && -w "$HOST_NIXL_STORAGE_BASE" ]] || {
    echo "NIXL root is missing or not writable: $HOST_NIXL_STORAGE_BASE (use --no-nixl to skip it)" >&2
    exit 1
  }
  HOST_NIXL_STORAGE_BASE="$(host_path "$HOST_NIXL_STORAGE_BASE")"
  docker_args+=(--volume "$HOST_NIXL_STORAGE_BASE:/nixl")
  if [[ -n "$NIXL_CONFIG" ]]; then
    [[ -f "$NIXL_CONFIG" && -r "$NIXL_CONFIG" ]] || {
      echo "NIXL config is missing or unreadable: $NIXL_CONFIG" >&2
      exit 1
    }
    nixl_dir="$(cd -- "$(dirname -- "$NIXL_CONFIG")" && pwd)"
    if [[ "$nixl_dir" == "$startup_dir" ]]; then
      docker_args+=(-e "NIXL_CONFIG=/config/$(basename -- "$NIXL_CONFIG")")
    else
      docker_args+=(--volume "$nixl_dir:/nixl-config:ro"
        -e "NIXL_CONFIG=/nixl-config/$(basename -- "$NIXL_CONFIG")")
    fi
  fi
  # NIXL POSIX uses io_uring, which Docker's default seccomp profile commonly
  # blocks; the NVMe PLE reader needs the same permission without NIXL.
  docker_args+=(--security-opt seccomp=unconfined)
elif [[ "$NVME_PLE" == on ]]; then
  docker_args+=(--security-opt seccomp=unconfined)
fi

exec docker "${docker_args[@]}" "$IMAGE" exec bash "/config/$(basename -- "$STARTUP")"
