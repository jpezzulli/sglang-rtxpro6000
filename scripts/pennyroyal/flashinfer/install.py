#!/usr/bin/env python3
"""Package the accepted FlashInfer SM120 source into one installed environment.

Both Pennyroyal installation paths run this single step after their FlashInfer
dependencies: the fresh and update sequences in BUILD.md and the container
build in docker/pennyroyal/Dockerfile.  It patches the installed
flashinfer-python sources in place under patches/ -- the accepted MoE and GDN
mailboxes plus one build-only compatibility guard -- builds the SM120 fused-MoE
module from those patched headers, and installs it at the package-local AOT path
that FlashInfer's loader checks before any provider wheel:

    <site-packages>/flashinfer/data/aot/fused_moe_120/fused_moe_120.so

Nothing here changes a kernel source, a FlashInfer default or a launch flag:
accepted-sources.json pins the accepted contents and attribution, and the
patched GDN mode stays opt-in through the Next recipes'
FLASHINFER_GDN_FP16_ACCUM_MMA default.

Usage, with the target environment's own Python:

    python scripts/pennyroyal/flashinfer/install.py               # install
    python scripts/pennyroyal/flashinfer/install.py --check       # verify it
    python scripts/pennyroyal/flashinfer/install.py --apply-only  # source half only
    python scripts/pennyroyal/flashinfer/install.py --record DIR  # refresh the pins

DIR is the `flashinfer` package directory of a fresh extraction of the pinned
wheel.  --apply-only, --record and --package exist for the focused CPU check and
for re-pinning after an approved FlashInfer bump; the installation paths use the
plain command.  It needs no GPU, no model and no download: the accepted source
is in this repository.  The build needs a CUDA compiler and keeps the existing
job-count controls (MAX_JOBS, FLASHINFER_NVCC_THREADS) and the caller's CXX.  For
nvcc's host compiler it honours CUDAHOSTCXX by handing FlashInfer that value as
CC, because FlashInfer reads CC for -ccbin and ignores CUDAHOSTCXX; without the
override its existing CC behaviour stands.  A failed build reports that effective
host compiler along with a bounded tail of the compiler's own output, because
Ninja's final line is only a summary and the fatal diagnostic sits above it.
FLASHINFER_CUDA_ARCH_LIST is set to the accepted SM120 family target for this
build because FlashInfer ignores TORCH_CUDA_ARCH_LIST here.  Set
FLASHINFER_WORKSPACE_BASE to keep the ninja objects between runs.

The SM120 compile also needs the compatibility guard: stock nv_internal
memoryUtils.cu asks for <sanitizer/asan_interface.h> even though it uses the ASAN
macros only under its own ASAN detection, so a toolchain whose compiler package
omits the sanitizer headers cannot build the module at all.  It is carried
separately, so the two accepted mailboxes stay exactly as reviewed, and it comes
out when the pin ships the guard or the build environment is defined to always
carry the headers.

An environment that predates this pin keeps a stale flashinfer-jit-cache shim
beside the new FlashInfer, and FlashInfer rejects that combination while
importing its JIT environment; the step names those wheels before the build
starts, with the aligned family recorded in accepted-sources.json.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "accepted-sources.json"
PREFIX = "FlashInfer SM120 packaging"

# Built in a fresh interpreter so the arch list is set before FlashInfer reads
# it. JitSpec.build() runs the Ninja graph and stops: build_and_load(), or
# build_jit_specs() with its skip_prebuilt default, would take the stock AOT
# module and silently compile nothing.
BUILD_ERROR_TAIL_LINES = 40

BUILD_CODE = """
import os
import pathlib

from flashinfer.jit import env as jit_env
from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module

expected = pathlib.Path(os.environ["PENNY_FLASHINFER_ACCEPTED_CSRC"]).resolve()
built_from = pathlib.Path(jit_env.FLASHINFER_CSRC_DIR).resolve()
if built_from != expected:
    raise SystemExit(
        f"FlashInfer compiles from {built_from}, not the patched source at {expected}"
    )
spec = gen_cutlass_fused_moe_sm120_module(use_fast_build=False)
spec.build()
print(pathlib.Path(spec.jit_library_path))
"""


def fail(message: str) -> RuntimeError:
    return RuntimeError(f"{PREFIX}: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def accepted_sources() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def locate_package() -> Path:
    """The flashinfer package of the interpreter running this script."""
    spec = find_spec("flashinfer")
    search = list(spec.submodule_search_locations or ()) if spec else []
    if not search:
        raise fail(f"flashinfer-python is not installed for {sys.executable}")
    return Path(search[0]).resolve()


def require_accepted_pin(source: dict, package: Path) -> str:
    """Read the version from the distribution beside the selected package."""
    version = ""
    for info in sorted(package.parent.glob("flashinfer_python-*.dist-info")):
        for line in (info / "METADATA").read_text().splitlines():
            if line.startswith("Version: "):
                version = line[len("Version: ") :]
    if not version:
        raise fail(f"{package} has no flashinfer_python .dist-info beside it")
    if version != source["flashinfer_python"]:
        raise fail(
            f"{package} is flashinfer-python {version}, not the accepted "
            f"{source['flashinfer_python']} source pin"
        )
    return version


def require_aligned_jit_cache(source: dict, package: Path) -> None:
    """Stop a stale JIT-cache shim before the build imports FlashInfer.

    flashinfer-python 0.7.0.post1 checks the flashinfer-jit-cache shim against
    its own version while importing flashinfer.jit.env, so the 0.6.17+cu130
    wheel that a v2.5.x environment keeps beside it aborts the compile with a
    version error from inside the build. The provider wheels only get skipped
    with a warning, so only the shim is refused here; it is genuinely optional,
    and without it FlashInfer compiles the other kernels on first use.
    """
    pin = source["flashinfer_python"]
    for info in sorted(package.parent.glob("flashinfer_jit_cache-*.dist-info")):
        version = info.name.removesuffix(".dist-info").rsplit("-", 1)[1]
        if version == pin or version.startswith(f"{pin}+"):
            continue
        raise fail(
            f"{info} is flashinfer-jit-cache {version}, which "
            f"flashinfer-python {pin} refuses at import time; install "
            f"'flashinfer-jit-cache=={source['flashinfer_jit_cache']}' and "
            f"'flashinfer-jit-cache-sm120f=={source['flashinfer_jit_cache']}' "
            "with --no-deps from https://flashinfer.ai/whl/cu130 as in the update "
            "block of BUILD.md, "
            "or uninstall it to compile the other kernels on first use"
        )


def _classify(root: Path, entry: dict) -> tuple[str, str]:
    """Tell the stock source, the carried source and anything else apart."""
    verdicts = set()
    for spec in entry["files"]:
        path = root / spec["path"]
        if not path.is_file():
            return (
                "unexpected",
                f"{path} is missing from the installed FlashInfer source",
            )
        actual = sha256_file(path)
        if actual == spec["accepted_sha256"]:
            verdicts.add("accepted")
        elif actual == spec["stock_sha256"]:
            verdicts.add("stock")
        else:
            return (
                "unexpected",
                f"{path} holds {actual}, which is neither the stock nor the "
                "accepted FlashInfer source",
            )
    if len(verdicts) > 1:
        return "unexpected", f"{root} holds a partly applied accepted patch"
    return (next(iter(verdicts)), "")


def _git_apply(entry: dict, root: Path) -> None:
    """Apply one mailbox to root, which is normally outside any Git checkout.

    Each entry names the root its paths are relative to.  The MoE mailbox names
    the source-repo csrc/ tree, which the wheel ships under flashinfer/data; the
    GDN mailbox and the compatibility guard name the importable flashinfer/
    package, one level up.  GIT_WORK_TREE stops git apply from
    adopting the surrounding Pennyroyal checkout and quietly skipping every
    file that is not inside it.
    """
    subprocess.run(
        ["git", "apply", "-p1", str(HERE / entry["patch"])],
        cwd=root,
        env={**os.environ, "GIT_WORK_TREE": str(root)},
        check=True,
    )


def apply_accepted_source(package: Path, source: dict) -> list[str]:
    """Patch the installed sources; already-patched source is accepted as-is."""
    statuses = []
    for entry in source["patches"]:
        root = (package / entry["root"]).resolve()
        verdict, detail = _classify(root, entry)
        if verdict == "unexpected":
            raise fail(detail)
        if verdict == "accepted":
            statuses.append(f"{entry['patch']}: already patched")
            continue
        _git_apply(entry, root)
        verdict, detail = _classify(root, entry)
        if verdict != "accepted":
            raise fail(
                f"{entry['patch']} did not produce the accepted source: {detail}"
            )
        statuses.append(f"{entry['patch']}: applied")
    return statuses


def nvcc_environment(source: dict, package: Path, environ: dict) -> dict:
    """The environment of the build subprocess, and of nothing else.

    FlashInfer picks nvcc's host compiler out of CC (jit/cpp_ext.py hands that
    value to -ccbin) and never reads CUDAHOSTCXX, so an explicit CUDA host
    compiler has to reach CC or it is silently ignored and nvcc binds whatever CC
    the environment happened to carry.  The mapping is applied to this copy only:
    the caller keeps its own CC, CXX stays in charge of the C++ extension and of
    the link step, and without CUDAHOSTCXX FlashInfer keeps the CC it already had.
    """
    env = dict(environ)
    env["FLASHINFER_CUDA_ARCH_LIST"] = source["cuda_arch_list"]
    env["PENNY_FLASHINFER_ACCEPTED_CSRC"] = str(package / "data" / "csrc")
    host_cxx = env.get("CUDAHOSTCXX")
    if host_cxx:
        env["CC"] = host_cxx
    return env


def nvcc_host_compiler(env: dict) -> str:
    """Name the host compiler nvcc is actually being told to use."""
    cc = env.get("CC")
    if not cc:
        return "nvcc default"
    return f"{cc} (CUDAHOSTCXX)" if env.get("CUDAHOSTCXX") == cc else f"{cc} (CC)"


def install_accepted_source(package: Path, source: dict) -> tuple[Path, str]:
    """Build the patched SM120 module and put it where the loader prefers it.

    Both guards are cheap and repeated by main() before it patches anything, so
    a direct caller of this function still cannot reach the import error.
    """
    require_accepted_pin(source, package)
    require_aligned_jit_cache(source, package)
    env = nvcc_environment(source, package, os.environ)
    persistent = env.get("FLASHINFER_WORKSPACE_BASE")
    stack = (
        nullcontext()
        if persistent
        else tempfile.TemporaryDirectory(prefix="pennyroyal-flashinfer-sm120-")
    )
    with stack as temp:
        workspace = Path(persistent or temp)
        if not persistent:
            env["FLASHINFER_WORKSPACE_BASE"] = str(workspace)
        try:
            result = subprocess.run(
                [sys.executable, "-c", BUILD_CODE],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            # Ninja's last line is only "build stopped: subcommand failed"; the
            # compiler diagnostic that matters sits above it, so a bounded tail of
            # the real output is reported rather than the final summary alone.
            lines = str(error.stderr or "").strip().splitlines()
            kept = lines[-BUILD_ERROR_TAIL_LINES:]
            excerpt = "\n".join(kept) if kept else "no compiler output"
            if len(lines) > len(kept):
                excerpt = f"[... {len(lines) - len(kept)} earlier lines ...]\n{excerpt}"
            raise fail(
                f"the {source['aot_module']} build failed with nvcc host compiler "
                f"{nvcc_host_compiler(env)}, MAX_JOBS="
                f"{env.get('MAX_JOBS', 'ninja default')}:\n{excerpt}"
            ) from error
        built = Path(result.stdout.strip().splitlines()[-1])
        module = source["aot_module"]
        if built.parent.name != module or built.name != f"{module}.so":
            raise fail(f"expected the build to emit {module}/{module}.so, got {built}")
        if not built.is_file() or built.stat().st_size == 0:
            raise fail(f"the SM120 build produced no {built}")
        destination = package / source["aot_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        shutil.copyfile(built, temporary)
        temporary.chmod(0o644)
        state = "reinstalled"
        if destination.is_file() and sha256_file(destination) == sha256_file(built):
            temporary.unlink()
            state = "unchanged"
        else:
            os.replace(temporary, destination)
    return destination, state


def check_installed(
    package: Path, source: dict, stock_modules: list[Path] | None = None
) -> dict:
    """Verify the accepted source and the package-local SM120 module are there.

    A stock FlashInfer install, or a stock provider wheel alone, fails here:
    every carried source set and the package-local module are required, so an
    install missing only the compatibility guard is named too.  Pass
    the provider wheels' own copies of the module in stock_modules so an install
    that merely copied the prebuilt kernel into place also fails.
    """
    version = require_accepted_pin(source, package)
    require_aligned_jit_cache(source, package)
    for entry in source["patches"]:
        root = (package / entry["root"]).resolve()
        verdict, detail = _classify(root, entry)
        if verdict != "accepted":
            raise fail(
                f"{detail or f'{root} is stock FlashInfer source'}; "
                f"{entry['patch']} ({entry['reason']}) is not installed, run "
                f"{__file__}"
            )
    module = package / source["aot_path"]
    if not module.is_file() or module.stat().st_size == 0:
        raise fail(
            f"{module} is absent or empty; the patched SM120 fused-MoE module "
            f"was never installed into this package (run {__file__})"
        )
    installed = sha256_file(module)
    for stock in stock_modules or ():
        if stock.is_file() and sha256_file(stock) == installed:
            raise fail(
                f"{module} is a copy of the stock provider kernel {stock}, not "
                "the module built from the accepted source"
            )
    return {
        "flashinfer": version,
        "package": str(package),
        "accepted_source": [
            {
                "patch": entry["patch"],
                "author": entry["author"],
                "commits": entry["commits"],
            }
            for entry in source["patches"]
        ],
        "sm120_module": str(module),
        "sm120_module_bytes": module.stat().st_size,
        "sm120_module_sha256": installed,
    }


def record(package: Path) -> None:
    """Repin the accepted digests from a fresh extraction of the pinned wheel."""
    source = accepted_sources()
    for entry in source["patches"]:
        root = (package / entry["root"]).resolve()
        for spec in entry["files"]:
            if sha256_file(root / spec["path"]) != spec["stock_sha256"]:
                raise fail(
                    f"--record needs the stock source of {spec['path']}; point it "
                    "at a fresh extraction of the pinned wheel"
                )
        _git_apply(entry, root)
        for spec in entry["files"]:
            spec["accepted_sha256"] = sha256_file(root / spec["path"])
    MANIFEST_PATH.write_text(json.dumps(source, indent=2) + "\n")
    print(f"repinned {MANIFEST_PATH.name} from {package}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="verify an installed environment"
    )
    parser.add_argument(
        "--apply-only", action="store_true", help="patch the sources without building"
    )
    parser.add_argument(
        "--package",
        type=Path,
        metavar="DIR",
        help="installed flashinfer package to act on (default: this interpreter's)",
    )
    parser.add_argument(
        "--record",
        type=Path,
        metavar="DIR",
        help="recompute the accepted digests from a fresh wheel extraction",
    )
    args = parser.parse_args(argv)
    source = accepted_sources()
    try:
        package = (
            args.record.resolve()
            if args.record
            else (args.package.resolve() if args.package else locate_package())
        )
        if args.record:
            record(package)
        elif args.check:
            print(json.dumps(check_installed(package, source), indent=2))
        else:
            require_accepted_pin(source, package)
            require_aligned_jit_cache(source, package)
            for status in apply_accepted_source(package, source):
                print(status)
            if not args.apply_only:
                jobs = os.environ.get("MAX_JOBS", "ninja default")
                print(
                    f"building {source['aot_module']} for "
                    f"FLASHINFER_CUDA_ARCH_LIST={source['cuda_arch_list']} "
                    f"(MAX_JOBS={jobs})"
                )
                module, state = install_accepted_source(package, source)
                print(
                    f"{state} {module} ({module.stat().st_size} bytes, "
                    f"sha256 {sha256_file(module)})"
                )
    except subprocess.CalledProcessError as error:
        print(
            f"{PREFIX}: a build step failed (exit {error.returncode}, "
            f"MAX_JOBS={os.environ.get('MAX_JOBS', 'ninja default')}); the "
            f"compiler output follows",
            file=sys.stderr,
        )
        if error.stderr:
            print(str(error.stderr).rstrip(), file=sys.stderr)
        return 1
    except (RuntimeError, OSError) as error:
        # Usually a read-only site-packages or a missing CUDA toolchain.
        print(f"{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
