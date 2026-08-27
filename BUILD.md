# Build

This is a native, non-Docker SM120 build. The exact qualified dependency set is
part of the runtime contract; newer versions should be requalified rather than
assumed equivalent.

## Qualified environment

| Component | Version |
|---|---|
| OS | Fedora 44, x86-64 |
| GPU | RTX PRO 6000 Blackwell Workstation Edition, 96 GB, SM120 |
| NVIDIA driver | `610.57.04` |
| Python | `3.12.13` |
| CUDA / NVCC | `13.3` / `13.3.73` |
| GCC / Rust | `15.3.1` / `1.97.1` |
| PyTorch | `2.13.0+cu130` |
| FlashInfer | `0.6.17` |
| NIXL | `1.4.0` |
| `sglang-kernel` | `0.4.6.post1` |
| Triton / XGrammar | `3.7.1` / `0.2.1` |

The installed SGLang wheel built from source commit `64ecd64924` had SHA-256
`67aed4edcda3cba59903790cf8e4d3834064d46a7f19276affb1a1d42d9e293c`.

## Native build

Check out this release tag and create an isolated Python 3.12 environment:

```bash
uv python install 3.12.13
uv venv --python 3.12.13 .venv
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC=/usr/bin/gcc-15
export CXX=/usr/bin/g++-15
export CUDAHOSTCXX=/usr/bin/g++-15
export TORCH_CUDA_ARCH_LIST=12.0

export MAX_JOBS=24
export CMAKE_BUILD_PARALLEL_LEVEL=24
export FLASHINFER_NINJA_JOBS=24
export FLASHINFER_NVCC_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=24

uv pip install --prerelease=allow \
  --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ \
  --no-build-isolation \
  -e python
```

The NVCC compiler-pin change makes `CXX` part of compilation as well as the JIT
fingerprint and host link. Do not omit the GCC 15 variables while expecting the
published JIT identity.

Verify the installed identity:

```bash
python - <<'PY'
import importlib.metadata as m
for name in ("sglang", "torch", "flashinfer-python", "nixl-cu13"):
    print(name, m.version(name))
PY
nvcc --version
gcc-15 --version
```

## NIXL POSIX

The qualified native NIXL build used upstream commit
`aecbc3846d92c34c7507a58d776e1fda50ff4fba`, release mode, SM120, and the
POSIX plugin. It is not vendored here.

```bash
git clone https://github.com/ai-dynamo/nixl.git
cd nixl
git checkout aecbc3846d92c34c7507a58d776e1fda50ff4fba
python -m pip install tomlkit
python -m pip install .
./contrib/tomlutil.py --wheel-name nixl-cu13 pyproject.toml
meson setup build-posix --buildtype=release \
  --prefix=/opt/nvidia/nvda_nixl --libdir=lib64 \
  -Denable_plugins=POSIX -Dnixl_cuda_arch_list=120
ninja -C build-posix -j24 install
python -m pip install build-posix/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl
```

The recorded `nixl-cu13==1.4.0` wheel SHA-256 was
`b2d618bc9593bf78120b44f9d573af8807e83716ae8c539b0f1532cc55a55ad8`.

## Build verification

Run the focused tests named in [CHANGES.md](CHANGES.md), then perform a real
startup. A healthy `/health` response alone is insufficient: check resolved
backends, KV dtypes, state-pool allocations, and CUDA-graph capture in logs as
described in [RUN.md](RUN.md).
