# Build SGLang for NVIDIA RTX PRO 6000 Blackwell (SM120)

These instructions build and install the Pennyroyal SGLang source used by the
Qwen3.8-27B/DFlash2 and Qwen3.8 Flash-Next recipes on one 96 GB RTX
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
Git, `uv`, CUDA 13.3, GCC/G++ 15, Rust and Ninja. The FlashInfer packaging step
below runs Ninja from the build environment, so the bootstrap line of the fresh
sequence installs it. These instructions install SGLang,
not the driver or operating-system toolchain. The
[tested versions](#qualified-environment) are listed below.

The launch recipes require substantial host RAM and disk space; see
[host memory and first start](RUN.md#host-memory-and-first-start).
HiCache/NIXL also requires the separate [NIXL POSIX installation](#nixl-posix).

## Fresh install

Run this sequence in Bash. It creates a new checkout and Python environment,
installs the build tools, then installs SGLang and its dependencies once:

```bash
# Substitute the tag of the release you are installing; the tag that carries
# this packaging is chosen when that release is published, so it is not named
# here. The release's own notes give it.
RELEASE_REF='<release-tag>'

git clone --branch "$RELEASE_REF" --single-branch \
  https://github.com/jpezzulli/sglang-rtxpro6000.git pennyroyal
cd pennyroyal

uv python install 3.12.13
uv venv --python 3.12.13 .venv
source .venv/bin/activate
uv pip install pip "setuptools>=61.0" "setuptools-rust>=1.10" \
  "setuptools-scm>=8.0" wheel build ninja

source scripts/pennyroyal/build-env.sh

uv pip install --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ \
  --no-build-isolation -e python

.venv/bin/python scripts/pennyroyal/flashinfer/install.py
```

Any release at or after the FlashInfer SM120 packaging carries the step below;
older tags do not, and neither does the released `setup1` configuration update,
so use the release that includes this source. SGLang is installed as an editable
package from this checkout. Keep the checkout in place while using this
environment.

The last command packages the accepted FlashInfer SM120 source into the
FlashInfer installation of that same environment and compiles the fused-MoE
module from it; see
[FlashInfer SM120 source integration](#flashinfer-sm120-source-integration).
It compiles once per environment, needs the CUDA 13.3 compiler and Ninja, and
uses the shared job budget below. Users of the prebuilt image run nothing: the
image contains the module its own build compiled from these sources.

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
RELEASE_REF='<release-tag>'   # the same release ref as above
git fetch origin tag "$RELEASE_REF"
git switch --detach "$RELEASE_REF"
source .venv/bin/activate

source scripts/pennyroyal/build-env.sh

uv pip install --no-build-isolation --no-deps -e python

# This source changes one dependency, and --no-deps will not install it for you:
# FlashInfer. The accepted source needs the pin, and the pin refuses a mismatched
# JIT cache family while importing, so align the whole family before the step.
uv pip install --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ \
  'flashinfer-python[cu13]==0.7.0.post1'
uv pip install --no-deps --index-url https://flashinfer.ai/whl/cu130 \
  'flashinfer-jit-cache==0.7.0.post1+cu130' \
  'flashinfer-jit-cache-sm120f==0.7.0.post1+cu130'

.venv/bin/python scripts/pennyroyal/flashinfer/install.py
```

This updates SGLang without re-resolving the existing dependencies: the release
keeps the PyTorch, `sglang-kernel` and NIXL packages of the v2.5.x dependency
base, and only the FlashInfer family moves, to the pin this source names
(`flashinfer-python[cu13]==0.7.0.post1`; the older tags here have 0.6.17). The JIT
cache family is the one optional part of the block: uninstalling it works too, and
FlashInfer then compiles the other kernels into its own cache on first use. The
packaging step is safe to repeat, and rerunning it after any later FlashInfer
install or upgrade is what keeps the environment packaged rather than stock. If it
names a FlashInfer that is not the pin, or a cache family that does not match, the
alignment commands above are the remedy. The step itself needs only the
`flashinfer-python` distribution and this checkout.

If you use NVMe PLE, also refresh the [isolated reader](#optional-nvme-ple-reader)
for the new source version. The prepared PLE overlay can be reused.

Restart using your existing model paths and the [launch guide](RUN.md).
The namespace helper chooses a fresh NIXL cache identity for changed source;
do not manually point it at an older namespace.

## NIXL POSIX

The native NIXL build used upstream commit
`aecbc3846d92c34c7507a58d776e1fda50ff4fba`, release mode, the supported RTX
SM86/89/120 targets, and the POSIX plugin. Fetch it separately from its
upstream repository.

Its source build requires Linux, a C++20 compiler, CMake, Meson, Ninja,
`pkg-config`, and the POSIX plugin's Linux AIO development package
(`libaio-devel` on Fedora or `libaio-dev` on Debian/Ubuntu). The pinned NIXL
source can build liburing through its Meson wrap when a system copy is absent.
The configuration below enables io_uring. Choose an install prefix writable by
the installing operator; `/opt/nvidia/nvda_nixl` was the build prefix but
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
  -Denable_plugins=POSIX -Dnixl_cuda_arch_list=86,89,120
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

The 27B performance and reasoning tests used this public
alternative target:

```bash
hf download orcarouter/Qwen3.8-27B-Uncensored-FP8 \
  --revision 9228df5c6c9c509e1019f83b4e085cf643118bac \
  --local-dir /path/to/qwen38-27b-alternative-target
```

The orcarouter checkpoint is an uncensored/abliterated derivative.
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
.venv/bin/python scripts/pennyroyal/flashinfer/install.py --check
```

The last command prints the accepted source commits, the installed SM120
module path, size and SHA-256, and the state of the optional flashinfer-cubin
payload (pruned, or absent when the wheel is not installed). It fails on a
stock FlashInfer source tree, on an unexpected FlashInfer version, on an
environment where the module was never built, and on a cubin wheel that still
carries the unsupported trtllm-gen payload the packaging step prunes; rerun
the step itself to prune it.

Then start a real profile with [RUN.md](RUN.md). Confirm the resolved backends,
KV dtypes, state pools, and CUDA graphs before measuring. `/health` verifies the
API process; RUN lists the functional startup checks. Focused source regressions
are recorded in [CHANGES.md](CHANGES.md).

## Advanced details

<a id="qualified-environment"></a>

### Build environment

| Component | Version |
|---|---|
| OS | Fedora 44, x86-64 |
| GPU | RTX PRO 6000 Blackwell Workstation Edition, 96 GB, SM120 |
| NVIDIA driver | `610.57.04` |
| Python | `3.12.13` |
| CUDA / NVCC | `13.3` / `13.3.73` |
| GCC / Rust | `15.3.1` / `1.97.1` |
| PyTorch | `2.13.0+cu130` |
| FlashInfer | `0.7.0.post1`, plus the accepted SM120 source below |
| NIXL | `1.4.0` |
| `sglang-kernel` | `0.4.6.post1` |
| Triton / XGrammar | `3.7.1` / `0.2.1` |

These are the versions used for the release. A fresh native install can
resolve newer versions of unpinned packages. Use the release container for
the packaged environment.

### Compiler and build jobs

The examples and launchers default to four build jobs and one NVCC thread.
For a larger machine, set `export PENNY_BUILD_JOBS=8` before running either
sequence or launching. On a memory-constrained host, use 1 or 2 instead.
Individual tool settings such as `MAX_JOBS` take precedence when already set;
unset old overrides if you want the shared budget to apply.

Our machine used 24 jobs and four NVCC threads. These limits control compilation, not
inference threads or GPU token-pool sizes. Keep the GCC 15 compiler variables
consistent: `CXX` participates in both NVCC host-compiler selection and the
JIT fingerprint. The FlashInfer JIT build reads its host compiler from `CC`, so
export `CUDAHOSTCXX` as well if you want a specific one there -- the packaging
step maps it onto `CC` for that build; see
[FlashInfer SM120 source integration](#flashinfer-sm120-source-integration).

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

<a id="flashinfer-sm120-source-integration"></a>

### FlashInfer SM120 source integration

`scripts/pennyroyal/flashinfer/install.py` is the one packaging step behind both
sequences above and the container image build. It takes the released
`flashinfer_python-0.7.0.post1` wheel sources, applies two already accepted
patches plus one build-only compatibility guard, and builds the SM120 CUTLASS
fused-MoE module from the result:

| Input | Contents | Attribution |
|---|---|---|
| `patches/moe-source.patch` | Three commits through `2a4d8d3a9501bf3b3fe3b78d7c6bad38bfc76064`; two C++ headers | Penny `<Pennyroyal@agentmail.to>`, port of the `aiueo52/flash-next-rtxpro6000` donor patches at `524af49abcca` and `e0fa9fa9fc3c` |
| `patches/gdn-source.patch` | `0b0ba4c2b18173303b46dd8ec381735e1615b313`; four Python files | aa24aa `<2496788660@qq.com>`, upstream FlashInfer #6227 |
| `patches/asan-include-compat.patch` | `c84ae2ff261d08bb212f5d72867185876d9d71e7`; one stock `.cu` file | Penny `<Pennyroyal@agentmail.to>`, local to this packaging; not an upstream change |

The mailboxes are kept byte-for-byte; only their install path is adapted. The
MoE mailbox names the `csrc/` tree of the FlashInfer source repository, which
the wheel ships as `flashinfer/data/csrc`, and the GDN mailbox names the
importable `flashinfer/` package. Both apply cleanly to stock 0.7.0.post1
sources (FlashInfer base `946200de1ae94fc93fdd0926f0a13afd1fa7f0f1`, wheel
`flashinfer_python-0.7.0.post1-py3-none-any.whl`, SHA-256
`c7adf826568d61fc1b7d3aadd4cae387a35a138bfc08e2deca2f34ecfa280716`).
`accepted-sources.json` records the digest of every file before and after the
patch, which is how the step tells stock, accepted and unexpected source apart
and why a repeat run is a no-op instead of a double patch. The module is
compiled with `FLASHINFER_CUDA_ARCH_LIST=12.0f` -- `TORCH_CUDA_ARCH_LIST` alone
is ignored here -- through the JIT spec's own Ninja build, and installed at
`flashinfer/data/aot/fused_moe_120/fused_moe_120.so` inside the package, the
first place FlashInfer looks. The stock provider wheels can stay for the other
kernels; the loader prefers the package-local module. No kernel source is
redesigned, and no host binary or private overlay is copied into the image.

After the module is installed the same step prunes the optional
`flashinfer-cubin` wheel through the sibling `prune_cubins.py`: it removes only
the pinned 0.7.0.post1 trtllm-gen cubins whose pinned runners dispatch on SM100,
SM103 or SM107 alone (`fmhaSm100a|fmhaSm100f|fmhaSm103a|fmhaSm107a...`,
`..._sm100a|_sm100f|_sm103a|_sm107a.cubin`), which no SM86/89/120 target can
load, and drops exactly those lines from the distribution RECORD; deep-gemm
cubins, `checksums.txt`, metadata and licenses are retained untouched. A wheel
at another version, a changed payload or an unexpected layout fails before the
first deletion, a repeat run changes nothing, and `--check` verifies the result
read-only (`prune_cubins.py --site DIR` runs the step on its own).

The build keeps your job budget (`MAX_JOBS`) and your `CXX`, and it honours
`CUDAHOSTCXX` for nvcc's host compiler. That honouring is a mapping, not a
pass-through: FlashInfer builds nvcc's `-ccbin` from `CC` and never reads
`CUDAHOSTCXX`, so a host compiler named only in `CUDAHOSTCXX` would be ignored and
nvcc would bind whatever `CC` the environment carried. The step sets `CC` from
`CUDAHOSTCXX` in the build subprocess environment alone, which leaves your shell
and the C++/link compiler as they were, and keeps FlashInfer's own `CC` behaviour
when `CUDAHOSTCXX` is unset. A build failure names the host compiler that was
really used and repeats the build's own output, labelled `stdout`/`stderr` and
trimmed to the last 40 lines of each, because FlashInfer merges the compiler log
into stderr normally but leaves it on stdout, with only its own traceback on
stderr, when `FLASHINFER_JIT_VERBOSE=1` is inherited; either way the fatal
diagnostic survives underneath Ninja's `build stopped` summary. This is what lets
`CC=/usr/bin/gcc CUDAHOSTCXX=/usr/bin/g++-15` mean what it says on a CUDA 13.3
host whose default GCC is too new for the toolkit.

The third input is not one of the fixes. Stock
`flashinfer/data/csrc/nv_internal/cpp/common/memoryUtils.cu` asks for
`<sanitizer/asan_interface.h>` unconditionally while using its macros only under
the ASAN detection the file itself makes a few lines later, and compiler packages
do not universally install that header: Fedora 44's gcc-15.3.1 package, which is
the same package set the image installs, does not, and the SM120 compile stops at
`common_memoryUtils.cuda.o`. The guard asks for the header under the condition the
macros are used under, so a non-ASAN build needs nothing new while an ASAN build
still requires, and still receives, the real header. It is carried as its own
mailbox so the two accepted ones stay as reviewed, and it comes out when the pin
ships the guard upstream or the build environment is defined to always install the
sanitizer development headers.

The MoE changes are the accepted fused-routing prologue and the two folded
finalize/expansion passes, with `FLASHINFER_MOE_FUSED_PROLOGUE=0` as their
kill switch. The GDN change is opt-in: the FP16-accumulate MMA mode of the
patched SM12x prefill kernels is enabled by the two Next launch recipes and the
two Next startup scripts through `FLASHINFER_GDN_FP16_ACCUM_MMA=1`, which does
not change FlashInfer's own default, the online-FP8 choice or the 27B profile.

### Source and earlier wheels

The exact executable source of each release is recorded in
[PROVENANCE.md](PROVENANCE.md). Releases that carry the FlashInfer SM120
integration above use the pinned 0.7.0.post1 dependency set; no new prebuilt
wheel is distributed. The optional NVMe reader is a separate
isolated install and is not included in the main SGLang wheel.

The earlier v2.1.2/v2.3 wheel from source `836206a0ad` has SHA-256
`96cb28701ac6f2ad1523e5607218f4fa68d9f9bb26041d6f36f36e2a39362542`.
It does not contain the later maintenance changes. See
[PROVENANCE.md](PROVENANCE.md) for source history.
