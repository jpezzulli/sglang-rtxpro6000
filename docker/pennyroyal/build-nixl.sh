#!/usr/bin/env bash
# Build the POSIX wheel against the image's installed Torch, without a second
# isolated Torch/CUDA download. No GPU is needed; the fixed target list covers
# the supported RTX SM86/89/120 family, matching scripts/pennyroyal/build-env.sh.
set -euo pipefail
build_root="$(mktemp -d /tmp/penny-nixl.XXXXXXXX)"
cleanup() {
  status=$?
  # Hosted builders disappear after failure; retain the actual compiler error
  # in Actions output before removing this script's temporary build directory.
  if (( status != 0 )) && [[ -f "$build_root/meson/meson-logs/meson-log.txt" ]]; then
    tail -n 100 "$build_root/meson/meson-logs/meson-log.txt" >&2
  fi
  rm -rf -- "$build_root"
  exit "$status"
}
trap cleanup EXIT
git clone https://github.com/ai-dynamo/nixl.git "$build_root/source"
git -C "$build_root/source" checkout --detach aecbc3846d92c34c7507a58d776e1fda50ff4fba
cd "$build_root/source"
python contrib/tomlutil.py --wheel-name nixl-cu13 pyproject.toml
uv build --wheel --no-build-isolation --python /opt/pennyroyal/.venv/bin/python \
  --out-dir "$build_root/dist" \
  -Cbuilddir="$build_root/meson" \
  -Csetup-args=-Dbuildtype=release \
  -Csetup-args=-Denable_plugins=POSIX \
  -Csetup-args=-Dbuild_tests=false \
  -Csetup-args=-Dbuild_examples=false \
  -Csetup-args=-Dwith_trace=false \
  -Csetup-args=-Dnixl_cuda_arch_list=86,89,120 \
  -Csetup-args=-Dinstall_headers=false
uv pip install --python /opt/pennyroyal/.venv/bin/python --no-deps \
  "$build_root"/dist/nixl_cu13-1.4.0-*.whl \
  "$build_root/meson/src/bindings/python/nixl-meta/nixl-1.4.0-py3-none-any.whl"
