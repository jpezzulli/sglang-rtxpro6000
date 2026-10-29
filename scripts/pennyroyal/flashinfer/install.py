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
    python scripts/pennyroyal/flashinfer/install.py --build-cache DIR  # reuse the compile

DIR is the `flashinfer` package directory of a fresh extraction of the pinned
wheel.  --apply-only, --record and --package exist for the focused CPU check and
for re-pinning after an approved FlashInfer bump; the installation paths use the
plain command.  It needs no GPU, no model and no download: the accepted source
is in this repository.  The build needs a CUDA compiler and keeps the existing
job-count controls (MAX_JOBS, FLASHINFER_NVCC_THREADS) and the caller's CXX.  For
nvcc's host compiler it honours CUDAHOSTCXX by handing FlashInfer that value as
CC, because FlashInfer reads CC for -ccbin and ignores CUDAHOSTCXX; without the
override its existing CC behaviour stands.  A failed build reports that effective
host compiler along with a bounded tail of each stream the build wrote to -- Ninja
merges the compiler output into stderr, and with FLASHINFER_JIT_VERBOSE=1
FlashInfer puts it on stdout and keeps only its traceback on stderr -- because
Ninja's final line is a summary and the fatal diagnostic sits above it.
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

The compile is the expensive part of this step, so it can be reused. With
--build-cache DIR (or PENNY_FLASHINFER_BUILD_CACHE=DIR) the step first binds the
module to the inputs that actually produce it: the Ninja file FlashInfer generated
for it, hashed with the build's own scratch directory blanked out so a run in a
different temporary directory is the same build; every source the build compiles,
including the instantiation files FlashInfer generates; the contents of every
include directory and linked library those flags name; the CUDA toolkit's bin, nvvm
and include trees; the host compiler, the header and startup-object directories it
reports, and every program and library it says it execs or links; the nvcc the
build file names, which FLASHINFER_NVCC can point outside the toolkit; and the
interpreter, torch and tvm-ffi binaries the module is built for and loaded by, next
to the accepted manifest, mailboxes and installer. A cache entry whose record
declares this fingerprint, whose own inputs hash back to that fingerprint and whose
bytes match the recorded digest is installed without entering the compile; an entry
that is missing, malformed, from another schema, recorded for another module,
shorter than it claims or holding other bytes is reported and compiled fresh, never
reused quietly. Anything the fingerprint could not look at -- an include directory
that is not there, a tool the driver cannot place, a source that does not exist --
is recorded as a gap in the identity and blocks reuse instead of being counted as a
fact about it. Without a cache directory nothing of this is computed and the build
is what it always was. On a miss the Ninja build directory is removed first, so
neither an output nor an object left in a persistent workspace can be relinked and
republished under a fingerprint that does not describe it. The
repository's own source revision is not one of those inputs, because this module
does not consume it: an API-only release restores the module in seconds instead of
paying 5,620 seconds of compilation again, while any real change to the kernel,
toolchain, flags or ABI recompiles. The published artifact is stored beside a record
carrying its digest and the whole input map, so a rebuild that should not have
happened is explainable by diffing two records.

After the module is in place the step prunes the pinned flashinfer-cubin wheel's
proven-unsupported payload through the sibling prune_cubins.py: the
sm100a/sm100f/sm103a/sm107a trtllm-gen cubins the pinned runners dispatch only
on datacenter SM100/103/107, which no RTX target (SM86/89/120) can ever load,
with the distribution RECORD kept consistent for the deletion. A site without
the optional wheel is reported and left alone; a wheel at another version or an
unexpected layout fails before anything is removed. --check verifies the same
invariant read-only, so an image that shipped the payload fails the check until
it is pruned; --apply-only keeps its source-half meaning and does not prune.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "accepted-sources.json"
INSTALLER_PATH = Path(__file__).resolve()
PREFIX = "FlashInfer SM120 packaging"

# The reusable-artifact cache: one directory per fingerprint, one record beside
# the module naming the artifact's own digest and every input it was bound to,
# and a `latest` pointer so a later build can say which input moved instead of
# just spending an hour again. CACHE_KEEP bounds the footprint to the newest
# artifacts, so a toolchain that flips back and forth does not recompile on
# every flip and the cache does not grow with every CUDA package refresh.
CACHE_SCHEMA = "pennyroyal-flashinfer-aot-v1"
CACHE_RECORD = "record.json"
CACHE_LATEST = "latest"
CACHE_KEEP = 2
# The cached module is stored under a name the repository's .dockerignore cannot
# match: the artifact reaches the image build through the build context, and a
# context pattern written to keep stray build products out of the image would
# silently drop a .so and leave the records alone.
CACHE_ARTIFACT_SUFFIX = ".aot"
BUILD_STATES = ("restored", "compiled")
CUDA_TOOLCHAIN_PARTS = ("bin", "nvvm", "include")
# The binaries the driver execs and links, named as it names them. Only the driver
# is asked where they are; nothing here guesses a location.
COMPILER_PROGRAMS = ("cc1plus", "collect2", "as", "ld")
COMPILER_LIBRARIES = ("libc.so", "libstdc++.so")
WORKSPACE_PLACEHOLDER = "!workspace"

# Built in a fresh interpreter so the arch list is set before FlashInfer reads
# it. JitSpec.build() runs the Ninja graph and stops: build_and_load(), or
# build_jit_specs() with its skip_prebuilt default, would take the stock AOT
# module and silently compile nothing. The reuse decision is this file's, so the
# fingerprint rules cannot drift apart from the ones the check reports.
BUILD_ERROR_TAIL_LINES = 40

BUILD_CODE = """
import importlib.util
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
packaging = importlib.util.spec_from_file_location(
    "pennyroyal_flashinfer_install", os.environ["PENNY_FLASHINFER_INSTALLER"]
)
step = importlib.util.module_from_spec(packaging)
packaging.loader.exec_module(step)
cache = os.environ.get("PENNY_FLASHINFER_BUILD_CACHE")
module, state = step.build_or_reuse(
    spec, step.accepted_sources(), pathlib.Path(cache) if cache else None
)
print(state, module)
"""


def fail(message: str) -> RuntimeError:
    return RuntimeError(f"{PREFIX}: {message}")


def cubin_pruner():
    """The cubin-payload helper that ships beside this step.

    Loaded from this file's own directory so both installation paths and the
    image check run the same pinned pruning, whatever sys.path says.
    """
    spec = importlib.util.spec_from_file_location(
        "penny_flashinfer_cubin_prune", HERE / "prune_cubins.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _tree_digest(root: Path) -> str:
    """The digest of the path and bytes of every file under one directory.

    What the build reads is what the module is bound to, so files are counted by
    content rather than by any version string. Byte-compiled files are skipped:
    they appear whenever anything imports the package, which is a property of the
    last run rather than of this build. A directory that is not there is counted as
    absent and an unreadable file as unreadable, so a missing or blocked input
    moves the digest instead of quietly dropping out of it.
    """
    if not root.is_dir():
        return "absent"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix == ".pyc" or "__pycache__" in path.parts:
            continue
        try:
            content = sha256_file(path)
        except OSError:
            content = "unreadable"
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(content.encode())
    return digest.hexdigest()


def _flag_values(ninja: str, *patterns: str) -> list[str]:
    """Flag values out of the generated ninja file, in the order they appear.

    The search runs over the whole text instead of line by line, because
    FlashInfer's join_multiline() wraps a long value as "$" followed by a newline:
    every flag is still one complete token on one side of that wrap, so the values
    can be read without reassembling lines and without depending on where the
    generator chose to break them.
    """
    found: list[str] = []
    for pattern in patterns:
        found.extend(
            value for value in re.findall(pattern, ninja) if value not in found
        )
    return found


def _binding(ninja: str, name: str) -> str:
    """One single-line ninja binding, or the empty string when there is none.

    A value ending in the generator's "$" continuation is deliberately not read
    here: the bindings asked for below (cuda_home, cxx, nvcc) are single tokens in
    every generated file, and a wrapped one would mean the step cannot say with
    certainty what the build names, which is a gap rather than a guess.
    """
    match = re.search(rf"(?m)^{name} = (.*)$", ninja)
    if not match:
        return ""
    value = match.group(1).strip()
    return "" if value.endswith("$") else value


def _expand(text: str, cuda_home: str) -> str:
    return text.replace("$cuda_home", cuda_home)


def _driver_queries(cxx: Path) -> tuple[list[str], dict[str, str], list[str]]:
    """What a compiler reports about itself: headers, its search path, its files.

    `-E -v` gives the directories it searches for headers, `-print-search-dirs`
    gives the programs it execs and the directory it keeps its startup objects and
    runtime library in, and the `-print-prog-name`/`-print-file-name` queries give
    the binaries it actually invokes and links. No version string is involved, and
    no location is guessed here: these are the answers the driver would use.
    """
    listed = subprocess.run(
        [str(cxx), "-E", "-v", "-x", "c++", os.devnull],
        check=True,
        capture_output=True,
        text=True,
    )
    directories = []
    started = False
    for line in (listed.stderr + listed.stdout).splitlines():
        if "search starts here:" in line:
            started = True
        elif "End of search list." in line:
            break
        elif started and line.strip() and line.strip() not in directories:
            directories.append(line.strip())
    search = subprocess.run(
        [str(cxx), "-print-search-dirs"], check=True, capture_output=True, text=True
    ).stdout
    programs: list[str] = []
    install: list[str] = []
    for line in search.splitlines():
        kind, _, value = line.partition(":")
        value = value.strip().lstrip("=").strip()
        if kind == "programs":
            programs = [entry for entry in value.split(":") if entry]
        elif kind == "install" and value:
            install = [value]
    answers = {}
    for flag, names in (
        ("-print-prog-name=", COMPILER_PROGRAMS),
        ("-print-file-name=", COMPILER_LIBRARIES),
    ):
        for name in names:
            answers[name] = subprocess.run(
                [str(cxx), flag + name], check=True, capture_output=True, text=True
            ).stdout.strip()
    return directories, answers, install + programs


def _resolved_answer(answer: str, search: list[str], cxx_parent: Path) -> Path | None:
    """Where the driver's own answer points, following its search path."""
    named = Path(answer)
    if named.is_absolute():
        return named if named.is_file() else None
    for directory in [Path(entry) for entry in search] + [cxx_parent]:
        candidate = directory / answer
        if candidate.is_file():
            return candidate.resolve()
    found = shutil.which(answer)
    return Path(found).resolve() if found else None


def _script_children(script: Path) -> list[Path]:
    """The libraries a linker script stands for; the script itself is not the input."""
    with script.open("rb") as handle:
        head = handle.read(512)
    if not head.startswith(b"/*") or b"(" not in head:
        return []
    children = []
    for token in re.findall(rb"[^\s()]+", head[head.index(b"(") :]):
        child = Path(os.fsdecode(token))
        if not child.is_absolute():
            child = script.with_name(os.fsdecode(token))
        if child.is_file() and child not in children:
            children.append(child)
    return children


def _compiler_inputs(cxx: Path, incomplete: list[str]) -> dict[str, str]:
    """The host compiler and everything it execs, resolves and links, by content.

    Every answer the compiler gives is hashed, including the directory holding its
    startup objects and libgcc; an answer that resolves to nothing is recorded as a
    gap in this toolchain's identity rather than counted as a fact about it, because
    a fingerprint that cannot see part of its own compiler must not claim reuse.
    """
    inputs = {str(cxx): sha256_file(cxx)}
    directories, answers, search = _driver_queries(cxx)
    for directory in directories:
        inputs[f"include:{directory}"] = _tree_digest(Path(directory))
    for directory in [entry for entry in search if Path(entry).is_dir()]:
        inputs[f"install:{directory}"] = _tree_digest(Path(directory))
    for name, answer in sorted(answers.items()):
        resolved = _resolved_answer(answer, search, cxx.parent)
        if resolved is None:
            incomplete.append(f"{cxx.name} answer {name}={answer!r} is not a file")
            continue
        inputs[str(resolved)] = sha256_file(resolved)
        for child in _script_children(resolved):
            inputs[str(child)] = sha256_file(child)
    return inputs


def _link_inputs(ninja: str, incomplete: list[str]) -> dict[str, str]:
    """The libraries the link step resolves -lNAME to, by content.

    The flags are read out of the whole build file rather than out of its ldflags
    binding, because that binding is one of the wrapped ones and only the tokens
    themselves are stable across how the generator broke the lines.
    """
    inputs: dict[str, str] = {}
    directories = _flag_values(ninja, r"(?:^|\s)-L(\S+)")
    for name in _flag_values(ninja, r"(?:^|\s)-l(\S+)"):
        for suffix in (".so", ".a"):
            found = next(
                (
                    Path(directory) / f"lib{name}{suffix}"
                    for directory in directories
                    if (Path(directory) / f"lib{name}{suffix}").is_file()
                ),
                None,
            )
            if found is not None:
                inputs[str(found)] = sha256_file(found)
                break
        else:
            incomplete.append(f"-l{name} is on no -L path")
    return inputs


def abi_dependency_files() -> dict[str, Path]:
    """The installed binaries whose contents decide this module's ABI.

    The interpreter the build runs on, the torch binary that answers the
    _GLIBCXX_USE_CXX11_ABI question the build asks it, and the tvm-ffi libraries
    that host the compiled module. Their version strings are not the fingerprint,
    their bytes are.
    """
    import torch
    import tvm_ffi

    answer = getattr(torch, "_C", None)
    library = getattr(answer, "__file__", None) or str(
        Path(torch.__file__).resolve().parent / "lib" / "libtorch_python.so"
    )
    if not Path(library).is_file():
        raise fail(f"{library} is not the installed torch C extension")
    ffi_root = Path(tvm_ffi.__file__).resolve().parent
    shared = sorted(path for path in ffi_root.rglob("*.so*") if path.is_file())
    if not shared:
        raise fail(f"{tvm_ffi.__file__} carries no shared library to fingerprint")
    found = {"python": Path(sys.executable).resolve(), "torch": Path(library).resolve()}
    for path in shared:
        found[f"tvm_ffi/{path.relative_to(ffi_root)}"] = path
    return found


def build_inputs(spec, environ: dict) -> dict:
    """The paths the build reads, with its scratch locations set aside.

    ``spec`` is FlashInfer's JitSpec for the module. Its sources are carried so
    each one is hashed by content, and the workspace roots are carried so they can
    be blanked out of the build-file digest: FlashInfer writes the generated
    instantiations and the Ninja file under a directory that changes with the
    temporary directory of a run, and a scratch path is not a build input. The
    generated sources themselves are still hashed, so their content is.
    """
    roots = {str(Path(spec.build_dir).parents[1])}
    base = environ.get("FLASHINFER_WORKSPACE_BASE")
    if base:
        roots.add(str(Path(base)))
    return {
        "workspace": sorted(roots, key=len, reverse=True),
        "sources": [str(path) for path in spec.sources],
    }


def _resolved(tool: str, cuda_home: str) -> Path:
    """Where a compiler named in the build file actually is."""
    path = Path(_expand(tool, cuda_home))
    found = path if path.is_absolute() else shutil.which(str(path))
    if not found or not Path(found).is_file():
        raise fail(f"{tool} is not an executable this build can fingerprint")
    return Path(found).resolve()


def _fingerprint_of(inputs: dict) -> str:
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def build_fingerprint(
    ninja: str,
    source: dict,
    inputs: dict,
    abi: dict[str, Path] | None = None,
) -> dict:
    """Bind the compiled module to the inputs that actually produced it.

    ``ninja`` is the build file FlashInfer generated for this module, hashed whole
    so every architecture, host compiler, include and link flag it carries counts
    as the build would have run it; the files and directories its flags name are
    then hashed by content, which covers the patched CUDA/C++ sources, the vendored
    headers, the CUDA toolkit, the host toolchain the driver execs and links, the
    linked CUDA libraries and the Python, torch and tvm-ffi binaries. The accepted
    manifest, the carried mailboxes and this installer are identities too. This
    repository's own source revision is deliberately none of that, because the
    module consumes nothing else from the checkout: that is what lets an API-only
    release reuse a compile of 5,620 seconds.

    ``incomplete`` lists anything the fingerprint could not look at. It travels with
    the digest so a record says what it does not know, and it blocks reuse.
    """
    roots = [str(root) for root in inputs.get("workspace") or ()]

    def neutralise(value: str) -> str:
        for root in sorted(roots, key=len, reverse=True):
            value = value.replace(root, WORKSPACE_PLACEHOLDER)
        return value

    text = neutralise(ninja)
    # Everything below reads paths out of the raw file and neutralises them only for
    # the digests: the workspace is blanked where it is hashed, not where it points.
    cuda_home = (
        _binding(ninja, "cuda_home") or os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    )
    # -I<path> and "-isystem <path>" each arrive as one flag, but "-ccbin <path>"
    # is two list entries, so the generator's wrap can land between the flag and its
    # argument: the optional separator token is part of reading the file as it is.
    includes = [
        _expand(value, cuda_home)
        for value in _flag_values(
            ninja, r"(?:^|\s)-I(\S+)", r"(?:^|\s)-isystem\s+(?:\$(?=\s)\s*)?(\S+)"
        )
    ]
    ccbin = _flag_values(ninja, r"(?:^|\s)-ccbin\s+(?:\$(?=\s)\s*)?(\S+)")
    incomplete: list[str] = []
    compilers: set[Path] = set()
    for value in [cuda_home, *includes, *ccbin]:
        if "$" in value:
            incomplete.append(f"unexpanded flag {value!r}")
    toolchain = {
        f"cuda-{part}": _tree_digest(Path(cuda_home, part))
        for part in CUDA_TOOLCHAIN_PARTS
    }
    for directory in [Path(cuda_home, part) for part in CUDA_TOOLCHAIN_PARTS]:
        if not directory.is_dir():
            incomplete.append(f"{directory} is not a directory")
    for name, value in (
        ("cxx", _binding(ninja, "cxx")),
        ("ccbin", ccbin[0] if ccbin else ""),
    ):
        if not value:
            continue
        try:
            compiler = _resolved(value, cuda_home)
        except RuntimeError as error:
            incomplete.append(f"{name}: {error}")
            continue
        if compiler in compilers:
            continue
        compilers.add(compiler)
        toolchain.update(_compiler_inputs(compiler, incomplete))
    toolchain.update(_link_inputs(_expand(ninja, cuda_home), incomplete))
    # The compiler the build invokes, as the build names it. FlashInfer takes it
    # from FLASHINFER_NVCC, so it is not necessarily the nvcc under CUDA_HOME whose
    # tree is hashed above, and an override the step cannot see is a gap.
    nvcc = _expand(_binding(ninja, "nvcc"), cuda_home)
    if not nvcc:
        incomplete.append("the build file does not name its nvcc on one line")
    else:
        try:
            invoked = _resolved(nvcc, cuda_home)
        except RuntimeError as error:
            incomplete.append(f"nvcc: {error}")
        else:
            toolchain[str(invoked)] = sha256_file(invoked)
    try:
        dependencies = abi if abi is not None else abi_dependency_files()
    except (ImportError, RuntimeError) as error:
        incomplete.append(f"abi: {error}")
        dependencies = {}
    sources = []
    for path in [Path(value) for value in inputs.get("sources") or ()]:
        if not path.is_file():
            incomplete.append(f"{path} is not a source file")
            continue
        sources.append([f"{path.parent.name}/{path.name}", sha256_file(path)])
    recorded = {
        "schema": CACHE_SCHEMA,
        "accepted": hashlib.sha256(
            json.dumps(source, sort_keys=True).encode()
        ).hexdigest(),
        "installer": sha256_file(INSTALLER_PATH),
        "patches": {
            entry["patch"]: sha256_file(HERE / entry["patch"])
            for entry in source["patches"]
        },
        "ninja": hashlib.sha256(text.encode()).hexdigest(),
        "includes": {
            neutralise(value): _tree_digest(Path(value)) for value in sorted(includes)
        },
        "sources": sorted(sources),
        "toolchain": toolchain,
        "abi": {
            label: sha256_file(path) for label, path in sorted(dependencies.items())
        },
        "incomplete": sorted(set(incomplete)),
    }
    return {"fingerprint": _fingerprint_of(recorded), "inputs": recorded}


def _artifact_dir(cache: Path, fingerprint: str) -> Path:
    return cache / CACHE_SCHEMA / fingerprint


def _load_record(path: Path) -> tuple[dict | None, str]:
    """A cache record, or why this file is not one.

    Any JSON value can sit at this path: an empty list, a bare string, a truncated
    object, bytes from another schema. None of them may raise their way past a
    rebuild, so the shape is checked here and everything malformed comes back as a
    reason to compile.
    """
    try:
        loaded = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, ValueError) as error:
        return None, f"{path} is not a readable record ({error})"
    if not isinstance(loaded, dict):
        return None, f"{path} holds {type(loaded).__name__}, not a record"
    return loaded, ""


def _last_used(cache: Path) -> tuple[str, dict] | None:
    """The fingerprint and inputs of the artifact this cache last worked with."""
    try:
        fingerprint = (cache / CACHE_LATEST).read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None
    record, _ = _load_record(_artifact_dir(cache, fingerprint) / CACHE_RECORD)
    if record is None:
        return None
    inputs = record.get("inputs")
    return str(record.get("fingerprint", fingerprint)), (
        inputs if isinstance(inputs, dict) else {}
    )


def _miss_reason(cache: Path, fingerprint: str, inputs: dict) -> str:
    """Why the cache cannot answer this build, in the names it is recorded by."""
    last = _last_used(cache)
    if last is None:
        return "no cached module"
    before, before_inputs = last
    if before == fingerprint:
        return "the cached module is incomplete or corrupt"
    moved = sorted(name for name in inputs if before_inputs.get(name) != inputs[name])
    return "build inputs changed since {}: {}".format(
        before[:12], ", ".join(moved or ["the cache schema"])
    )


def _restore(
    cache: Path, fingerprint: str, name: str, inputs: dict
) -> tuple[Path | None, str]:
    """The cached module for this fingerprint, or why there is no such thing.

    Every check is a refusal. The record must be an object that declares this
    schema, this module and this fingerprint, its own inputs must hash back to that
    fingerprint so a record cannot be pasted onto another build, and the artifact
    must be exactly as long and as digestible as the record says. Nothing that fails
    here is installed quietly.
    """
    directory = _artifact_dir(cache, fingerprint)
    record_path = directory / CACHE_RECORD
    if not record_path.is_file():
        return None, _miss_reason(cache, fingerprint, inputs)
    record, why = _load_record(record_path)
    if record is None:
        return None, why
    for field, wanted in (
        ("schema", CACHE_SCHEMA),
        ("fingerprint", fingerprint),
        ("module", name),
    ):
        if record.get(field) != wanted:
            return None, (
                f"{record_path} records {field}={record.get(field)!r}, not {wanted!r}"
            )
    if not isinstance(record.get("inputs"), dict):
        return (
            None,
            f"{record_path} records its inputs as {type(record.get('inputs')).__name__}",
        )
    if _fingerprint_of(record["inputs"]) != fingerprint:
        return (
            None,
            f"{record_path} inputs hash to {_fingerprint_of(record['inputs'])[:12]}, not its own fingerprint",
        )
    artifact = directory / (name + CACHE_ARTIFACT_SUFFIX)
    size = artifact.stat().st_size if artifact.is_file() else None
    if size is None or size != record.get("artifact_bytes"):
        return None, (
            f"{artifact} is "
            f"{'absent' if size is None else f'{size} bytes'}, the record says "
            f"{record.get('artifact_bytes')!r}"
        )
    actual = sha256_file(artifact)
    if actual != record.get("artifact_sha256"):
        return None, (
            f"{artifact} holds {actual}, the record says "
            f"{record.get('artifact_sha256')!r}"
        )
    return artifact, ""


def _note_latest(cache: Path, fingerprint: str) -> None:
    try:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / CACHE_LATEST).write_text(fingerprint + "\n")
    except OSError as error:
        print(f"{PREFIX}: could not update {cache}: {error}", file=sys.stderr)


def _prune(cache: Path, keep: set[str]) -> None:
    """Drop every artifact but the ones this cache can still come back to.

    The published fingerprint and the one the cache last used are the current
    toolchain and the previous one: an environment that flips back and forth between
    two of them keeps paying one compile instead of two, and nothing else is allowed
    to accumulate. CACHE_KEEP is that count.
    """
    try:
        directories = [path for path in (cache / CACHE_SCHEMA).iterdir()]
    except OSError:
        return
    for stale in (path for path in directories if path.name not in keep):
        shutil.rmtree(stale, ignore_errors=True)


def _publish(
    cache: Path, fingerprint: str, name: str, module: Path, inputs: dict
) -> None:
    """Carry this build's module to the next one, and keep the footprint bounded."""
    directory = _artifact_dir(cache, fingerprint)
    artifact = directory / (name + CACHE_ARTIFACT_SUFFIX)
    before = _last_used(cache)
    record = {
        "schema": CACHE_SCHEMA,
        "fingerprint": fingerprint,
        "module": name,
        "artifact_sha256": sha256_file(module),
        "artifact_bytes": module.stat().st_size,
        "inputs": inputs,
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        temporary = artifact.with_name(artifact.name + ".tmp")
        shutil.copyfile(module, temporary)
        temporary.chmod(0o644)
        os.replace(temporary, artifact)
        (directory / CACHE_RECORD).write_text(json.dumps(record, indent=2) + "\n")
    except OSError as error:
        print(
            f"{PREFIX}: could not publish {directory}: {error}; the next build "
            "will compile again",
            file=sys.stderr,
        )
        return
    _note_latest(cache, fingerprint)
    keep = [fingerprint] + ([before[0]] if before else [])
    _prune(cache, set(keep[:CACHE_KEEP]))


def build_or_reuse(
    spec,
    source: dict,
    cache: Path | None = None,
    inputs: dict | None = None,
    abi: dict[str, Path] | None = None,
) -> tuple[Path, str]:
    """The compiled SM120 module, from the fingerprinted cache or from nvcc.

    Without a cache directory this is the build exactly as it always was: write the
    Ninja file, run it, no fingerprint and no extra read of the environment. With
    one, the Ninja file is hashed first, because it is the record of the flags and
    the toolchain the build would run under, and skipping the compile is then the
    only difference from a cold build: the patched sources, the generated
    instantiations and the loader's expectations are the same either way.

    A refused entry is reported on stderr and compiled. On a miss the Ninja build
    directory is removed first, because Ninja is otherwise content to relink the
    objects its previous inputs left behind and this build would publish code it
    never compiled under its own new fingerprint; regenerating the build file is the
    build's own job. Without a cache directory none of this happens.
    """
    spec.write_ninja()
    destination = Path(spec.jit_library_path)
    if cache is None:
        spec.build()
        return destination, "compiled"
    printed = build_fingerprint(
        Path(spec.ninja_path).read_text(),
        source,
        inputs or build_inputs(spec, os.environ),
        abi,
    )
    fingerprint, manifest = printed["fingerprint"], printed["inputs"]
    gaps = ", ".join(manifest.get("incomplete") or ())
    if gaps:
        print(
            f"{PREFIX}: the build inputs are not fully identifiable ({gaps}); "
            "compiling",
            file=sys.stderr,
        )
    else:
        cached, why = _restore(cache, fingerprint, spec.name, manifest)
        if cached is None:
            print(f"{PREFIX}: {why}; compiling", file=sys.stderr)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            shutil.copyfile(cached, temporary)
            temporary.chmod(0o644)
            os.replace(temporary, destination)
            _note_latest(cache, fingerprint)
            return destination, "restored"
    # A miss means something that changes the code really changed, so the
    # intermediates from the previous inputs cannot be trusted to be relinked.
    shutil.rmtree(Path(spec.build_dir))
    spec.build()
    _publish(cache, fingerprint, spec.name, destination, manifest)
    return destination, "compiled"


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
    env["PENNY_FLASHINFER_INSTALLER"] = str(INSTALLER_PATH)
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


def _build_failure_message(
    source: dict, env: dict, error: subprocess.CalledProcessError
) -> str:
    """Say which compiler ran and keep its own output, bounded per stream.

    Ninja's final line is only a summary, so the tail of the real output is what
    matters.  Which stream holds it depends on FlashInfer: run_ninja pipes the
    merged build output into stderr normally, but with FLASHINFER_JIT_VERBOSE=1 it
    lets ninja write to the child's own stdout and raises with no output embedded,
    so the diagnostic is in stdout and only the FlashInfer traceback is left on
    stderr.  Both captured streams are therefore reported, each trimmed to the same
    tail bound.
    """
    parts = []
    for label, stream in (("stdout", error.output), ("stderr", error.stderr)):
        lines = str(stream or "").strip().splitlines()
        if not lines:
            continue
        kept = lines[-BUILD_ERROR_TAIL_LINES:]
        if len(lines) > len(kept):
            kept = [f"[... {len(lines) - len(kept)} earlier {label} lines ...]", *kept]
        parts.append(f"--- {label} ---\n" + "\n".join(kept))
    return (
        f"the {source['aot_module']} build failed with nvcc host compiler "
        f"{nvcc_host_compiler(env)}, MAX_JOBS="
        f"{env.get('MAX_JOBS', 'ninja default')}:\n"
        + ("\n".join(parts) if parts else "no compiler output")
    )


def install_accepted_source(
    package: Path, source: dict, cache: Path | None = None
) -> tuple[Path, str]:
    """Build the patched SM120 module and put it where the loader prefers it.

    Both guards are cheap and repeated by main() before it patches anything, so
    a direct caller of this function still cannot reach the import error. With a
    cache directory the compile itself is the only thing build_or_reuse may skip;
    everything else here runs the same, and the reported state says which of
    "restored", "compiled" or "unchanged" this run was.
    """
    require_accepted_pin(source, package)
    require_aligned_jit_cache(source, package)
    env = nvcc_environment(source, package, os.environ)
    if cache is not None:
        env["PENNY_FLASHINFER_BUILD_CACHE"] = str(cache)
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
            raise fail(_build_failure_message(source, env, error)) from error
        reported = result.stdout.strip().splitlines()[-1].split(" ", 1)
        if len(reported) != 2 or reported[0] not in BUILD_STATES:
            raise fail(
                f"the build step reported {reported} instead of a build state and "
                "the module path it wrote"
            )
        state, built = reported[0], Path(reported[1])
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
        "cubin_payload": cubin_pruner().check(package.parent),
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
    parser.add_argument(
        "--build-cache",
        type=Path,
        metavar="DIR",
        help=(
            "reuse and publish the compiled module in DIR "
            "(default: $PENNY_FLASHINFER_BUILD_CACHE, unset means always compile)"
        ),
    )
    args = parser.parse_args(argv)
    source = accepted_sources()
    declared = args.build_cache or (
        Path(os.environ["PENNY_FLASHINFER_BUILD_CACHE"])
        if os.environ.get("PENNY_FLASHINFER_BUILD_CACHE")
        else None
    )
    cache = declared.resolve() if declared else None
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
                    f"installing {source['aot_module']} for "
                    f"FLASHINFER_CUDA_ARCH_LIST={source['cuda_arch_list']} "
                    f"(MAX_JOBS={jobs}, "
                    f"build cache {cache if cache else 'disabled'})"
                )
                module, state = install_accepted_source(package, source, cache)
                print(
                    f"{state} {module} ({module.stat().st_size} bytes, "
                    f"sha256 {sha256_file(module)})"
                )
                payload = cubin_pruner().prune(package.parent)
                if payload["status"] == "absent":
                    print(
                        "flashinfer-cubin is not installed; no prebuilt cubin "
                        "payload to prune"
                    )
                else:
                    print(
                        f"pruned {payload['removed_files']} unsupported cubin "
                        f"files ({payload['removed_bytes']} bytes) and dropped "
                        f"{payload['record_entries_removed']} RECORD entries"
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
