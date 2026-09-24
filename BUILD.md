# Build SGLang for NVIDIA RTX PRO 6000 Blackwell (SM120)

These instructions build and install the Pennyroyal SGLang source used by the
qualified Qwen3.8-27B/DFlash2 and Qwen3.8 Flash-Next recipes on one 96 GB RTX
PRO 6000. Choose [Fresh install](#fresh-install) for a new environment or
[Update an existing install](#update-an-existing-install) for a working setup.
Both install the same source for the two profiles.

For a prebuilt environment, use the
[Pennyroyal container guide](docker/pennyroyal/README.md). Its GitHub
Actions-built image includes the toolchain and NIXL POSIX plugin; models and
writable caches remain in host directories. The rest of this page covers the
native installation.

## Prerequisites

Use Linux with an RTX PRO 6000 Blackwell (SM120), a working NVIDIA driver,
Git, `uv`, CUDA 13.3, GCC/G++ 15 and Rust. These instructions install SGLang,
not the driver or operating-system toolchain. The
[tested versions](#qualified-environment) are listed below.

The launch recipes require substantial host RAM and disk space; see
[host memory and first start](RUN.md#host-memory-and-first-start).
HiCache/NIXL also requires the separate [NIXL POSIX installation](#nixl-posix).

## Fresh install

Run this sequence in Bash. It creates a new checkout and Python environment,
installs the build tools, then installs SGLang and its dependencies once:

```bash
git clone --branch pennyroyal-v2.5.1 --single-branch \
  https://github.com/jpezzulli/sglang-rtxpro6000.git pennyroyal
cd pennyroyal

uv python install 3.12.13
uv venv --python 3.12.13 .venv
source .venv/bin/activate
uv pip install pip "setuptools>=61.0" "setuptools-rust>=1.10" \
  "setuptools-scm>=8.0" wheel build

source scripts/pennyroyal/build-env.sh

uv pip install --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ \
  --no-build-isolation -e python
```

SGLang is now installed as an editable package from this checkout. Keep the
checkout in place while using this environment.

Next, complete [NIXL POSIX](#nixl-posix) if it is not already installed,
[download your checkpoints](#reference-and-measured-checkpoints), then follow
[RUN.md](RUN.md#configure-and-run) to save your settings and start the server.
The build helper uses four parallel jobs by default. Set `PENNY_BUILD_JOBS`
before sourcing it to change that; compiler paths can also be overridden.

## Update an existing install

Use this path for an existing Pennyroyal environment, such as v2.5.0. Stop any
server using the checkout first. Start with a clean checkout:
`git status --short` must be empty; save your own changes before switching tags.
Replace the path below with your checkout. The public remote is assumed to be
named `origin`.

```bash
cd /path/to/pennyroyal
git fetch origin tag pennyroyal-v2.5.1
git switch --detach pennyroyal-v2.5.1
source .venv/bin/activate

source scripts/pennyroyal/build-env.sh

uv pip install --no-build-isolation --no-deps -e python
```

This updates SGLang without re-resolving the existing dependencies.
v2.5.1 reuses the v2.5.0 PyTorch, `sglang-kernel`, FlashInfer, and NIXL
dependencies. If build tools are missing, install the bootstrap packages from
the fresh-install sequence, then retry the final command.

If you use NVMe PLE, also refresh the [isolated reader](#optional-nvme-ple-reader)
for the new source version. The prepared PLE overlay can be reused.

Restart using your existing model paths and the [launch guide](RUN.md).
The namespace helper chooses a fresh NIXL cache identity for changed source;
do not manually point it at an older namespace.

## NIXL POSIX

The native NIXL build used upstream commit
`aecbc3846d92c34c7507a58d776e1fda50ff4fba`, release mode, SM120, and the
POSIX plugin. Fetch it separately from its upstream repository.

Its source build requires Linux, a C++20 compiler, CMake, Meson, Ninja,
`pkg-config`, and the POSIX plugin's Linux AIO development package
(`libaio-devel` on Fedora or `libaio-dev` on Debian/Ubuntu). The pinned NIXL
source can build liburing through its Meson wrap when a system copy is absent.
The configuration below enables io_uring. Choose an install prefix writable by
the installing operator; `/opt/nvidia/nvda_nixl` was the qualified prefix but
normally requires administrator preparation.

```bash
cd /path/to/build-workspace
git clone https://github.com/ai-dynamo/nixl.git
cd nixl
git checkout aecbc3846d92c34c7507a58d776e1fda50ff4fba
export NIXL_PREFIX=/opt/nvidia/nvda_nixl
python -m pip install tomlkit
python -m pip install .
./contrib/tomlutil.py --wheel-name nixl-cu13 pyproject.toml
meson setup build-posix --buildtype=release \
  --prefix="$NIXL_PREFIX" --libdir=lib64 \
  -Denable_plugins=POSIX -Dnixl_cuda_arch_list=120
ninja -C build-posix -j "${PENNY_BUILD_JOBS:-4}" install
python -m pip install build-posix/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl
export LD_LIBRARY_PATH="$NIXL_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Persist that library path in the account or service environment used to run
Pennyroyal when the prefix is not already covered by the system dynamic-linker
configuration. The NIXL storage root itself must be writable by that same
runtime identity and reside on a filesystem suitable for the selected
O_DIRECT/io_uring configuration.

The recorded `nixl-cu13==1.4.0` wheel SHA-256 was
`b2d618bc9593bf78120b44f9d573af8807e83716ae8c539b0f1532cc55a55ad8`.

## Reference and measured checkpoints

Weights are distributed separately. Download the reference targets and the
27B draft as needed:

```bash
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --local-dir /path/to/flash-next

hf download Qwen/Qwen3.8-27B-FP8 \
  --local-dir /path/to/qwen38-27b-target

hf download incoai/Qwen3.8-27B-DFlash2 \
  --revision adde41d8fde3a75dc905a7df0bd5088d2a44b5a1 \
  --local-dir /path/to/qwen38-27b-draft
```

The retained 27B performance and behavior campaign used this public
alternative target:

```bash
hf download orcarouter/Qwen3.8-27B-Uncensored-FP8 \
  --revision 9228df5c6c9c509e1019f83b4e085cf643118bac \
  --local-dir /path/to/qwen38-27b-alternative-target
```

The orcarouter checkpoint is an uncensored/abliterated derivative. Its
provenance matters for reasoning, refusal, and other behavioral results.
Review each model card and license.

## Optional NVMe PLE reader

Skip this section when using the default RAM-backed PLE. NVMe-backed PLE needs
an additional isolated reader and a prepared local-SSD model overlay:

```bash
cd /path/to/pennyroyal
PYTHON="$PWD/.venv/bin/python" bash tools/ple_nvme/install.sh

.venv/bin/python scripts/pennyroyal/prepare_ple_nvme.py \
  --source /path/to/original-flash-next-checkpoint \
  --output /path/on/local-nvme/flash-next-ple
```

The reader installs under `.ple-nvme` by default and does not alter the main
environment. It requires Rust/Cargo and `uv`; build tools may download
dependencies. The prepared overlay requires approximately 48 GiB plus
filesystem overhead and retains links to the original immutable checkpoint.
The installer preserves existing reader directories. When upgrading from
v2.5.0, install the reader from this checkout into a new location and use it
for launch:

```bash
export PENNY_PLE_PLUGIN_DIR="$PWD/.ple-nvme-v251"
PYTHON="$PWD/.venv/bin/python" bash tools/ple_nvme/install.sh
```

The reader checks Pennyroyal's source signatures, which changed with the QSA
fix. The prepared overlay and table format are unchanged; reuse the existing
overlay when its normal integrity checks pass. Container images already
include the matching reader.

See [NVME-PLE.md](NVME-PLE.md) for overrides, launch variables, integrity
checks, upstream license/NOTICE credit and the measured memory/performance
tradeoff. Online FP8 needs no separate package; it is a runtime opt-in described
in [FP8.md](FP8.md).

## Build verification

Check the installed package identities from the activated environment:

```bash
python - <<'PY'
import importlib.metadata as m
for name in ("sglang", "torch", "flashinfer-python", "nixl-cu13"):
    print(name, m.version(name))
PY
nvcc --version
gcc-15 --version
```

Then start a real profile with [RUN.md](RUN.md). Confirm the resolved backends,
KV dtypes, state pools, and CUDA graphs before measuring. `/health` verifies the
API process; RUN lists the functional startup checks. Focused source regressions
are recorded in [CHANGES.md](CHANGES.md).

## Advanced details

### Qualified environment

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

These are the tested dependency versions. A fresh install can resolve newer
versions of unpinned packages, so compare the installed environment when exact
reproduction matters. The native procedure was reconstructed from the working
installation; the container build supplies the clean automated build path.

### Compiler and build jobs

The examples and launchers default to four build jobs and one NVCC thread.
For a larger machine, set `export PENNY_BUILD_JOBS=8` before running either
sequence or launching. On a memory-constrained host, use 1 or 2 instead.
Individual tool settings such as `MAX_JOBS` take precedence when already set;
unset old overrides if you want the shared budget to apply.

The qualification host used 24 jobs and four NVCC threads; that is a reference
measurement, not the public default. These limits control compilation, not
inference threads or GPU token-pool sizes. Keep the GCC 15 compiler variables
consistent: `CXX` participates in both NVCC host-compiler selection and the
JIT fingerprint.

### Optional wheel packaging

The install sequences above are sufficient to run SGLang. Use this only if
you also want a standalone wheel, for example to move the package out of an
editable checkout. Activate the build environment and retain the compiler
settings above:

```bash
cd /path/to/pennyroyal/python
WHEEL_OUT="$(mktemp -d)"
python -m build --wheel --no-isolation --outdir "$WHEEL_OUT"
python -m pip install --force-reinstall --no-deps "$WHEEL_OUT"/sglang-*.whl
```

The fresh output directory avoids accidentally selecting an older wheel in
`python/dist`. The generated package version depends on the checked-out Git
revision. Model weights and separately installed runtime dependencies are not
bundled in this wheel.

### Source and earlier wheels

The exact v2.5.1 executable source is recorded in [PROVENANCE.md](PROVENANCE.md).
This release uses updated Python/JIT sources on the existing dependency stack;
no new prebuilt wheel is distributed. The optional NVMe reader is a separate
isolated install and is not included in the main SGLang wheel.

The earlier v2.1.2/v2.3 wheel from source `836206a0ad` has SHA-256
`96cb28701ac6f2ad1523e5607218f4fa68d9f9bb26041d6f36f36e2a39362542`.
It does not contain the later maintenance changes. See
[PROVENANCE.md](PROVENANCE.md) for source history.
