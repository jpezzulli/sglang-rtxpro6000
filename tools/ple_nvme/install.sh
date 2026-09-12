#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
DESTINATION="${PENNY_PLE_PLUGIN_DIR:-$REPO_ROOT/.ple-nvme}"
[[ ! -e "$DESTINATION" ]] || {
  echo "Destination exists: $DESTINATION; select a new PENNY_PLE_PLUGIN_DIR" >&2; exit 1;
}
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
# Isolated import directory: no modifications to SGLang's Python environment.
# uv/maturin resolve build dependencies separately; runtime deps come from Penny.
uv pip install --python "$PYTHON" --target "$DESTINATION" --no-deps "$SCRIPT_DIR/ssd_stream"
echo "Optional reader installed at $DESTINATION"
