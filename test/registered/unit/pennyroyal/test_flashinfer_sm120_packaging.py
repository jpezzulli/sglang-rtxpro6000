"""CPU-only packaging checks for the accepted FlashInfer SM120 source.

No GPU, no CUDA toolchain, no container runtime and no wheel download: the
packaging rules are exercised against synthetic installed-source trees, and the
carried mailboxes are checked against their pins and attribution. Compilation,
image build and GPU behaviour are qualified separately on the RTX PRO 6000 host;
nothing here claims them.
"""

import hashlib
import importlib.util
import inspect
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
PACKAGE_DIR = REPO / "scripts" / "pennyroyal" / "flashinfer"
INSTALLER = PACKAGE_DIR / "install.py"
MANIFEST = PACKAGE_DIR / "accepted-sources.json"
DOCKERFILE = REPO / "docker" / "pennyroyal" / "Dockerfile"
CHECK = REPO / "docker/pennyroyal/check_install.py"
BUILD_GUIDE = REPO / "BUILD.md"
NEXT_SCRIPTS = (
    REPO / "configs/pennyroyal/serve-flash-next.sh",
    REPO / "configs/pennyroyal/serve-flash-next-frspec.sh",
    REPO / "docker/pennyroyal/launch/config/start-flash-next.sh",
    REPO / "docker/pennyroyal/launch/config/start-flash-next-frspec.sh",
)
OTHER_SCRIPTS = (
    REPO / "configs/pennyroyal/serve-qwen38-27b-dflash2.sh",
    REPO / "docker/pennyroyal/launch/config/start-27b-dflash2.sh",
)
GDN_EXPORT = (
    'export FLASHINFER_GDN_FP16_ACCUM_MMA="${FLASHINFER_GDN_FP16_ACCUM_MMA:-1}"'
)
GDN_RESOLVE = (
    'if [[ "$FLASHINFER_GDN_FP16_ACCUM_MMA" == 1 ]]; then GDN_FP16_ACCUM_MMA=on; fi'
)
GDN_FIELD = '--field "gdn_fp16_accum_mma=$GDN_FP16_ACCUM_MMA" \\'
DIGEST = re.compile(r"\A[0-9a-f]{64}\Z")
COMPAT = "patches/asan-include-compat.patch"
COMPAT_PATH = "flashinfer/data/csrc/nv_internal/cpp/common/memoryUtils.cu"
# The stock and guarded bytes of that one file, repinned from the wheel.
COMPAT_STOCK_SHA256 = "b4d493f293db5b0294272ef6a3ad7821fbda3c42ce66ea57268da87f3f24e5ad"
COMPAT_ACCEPTED_SHA256 = (
    "a8c8945ed2c8a5ede9026127f5f923a27bb206ca532c12bed48c22e3df0f1ae8"
)
COMPAT_COMMIT = "c84ae2ff261d08bb212f5d72867185876d9d71e7"


def installer():
    spec = importlib.util.spec_from_file_location("penny_flashinfer_sm120", INSTALLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


GUARD_STOCK = "#include <unused.h>\nint guard;\n"
GUARD_ACCEPTED = "#if 0\n#include <unused.h>\n#endif\nint guard;\n"


def installed_tree(
    root: Path, contents: dict, version: str, guard: str = GUARD_STOCK
) -> Path:
    """A stand-in site-packages: the package, its metadata, the given sources.

    `guard` is the sibling file the second generated set covers, written one level
    above the package, where the compatibility guard's paths live.
    """
    package = root / "flashinfer"
    root.mkdir(parents=True, exist_ok=True)
    (root / "guard.cu").write_text(guard)
    for name, content in contents.items():
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    info = root / f"flashinfer_python-{version}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(f"Name: flashinfer-python\nVersion: {version}\n")
    return package


def synthetic_source(root: Path) -> dict:
    """The carried source sets, over generated files, pinned the accepted way.

    The real mailboxes cover seven FlashInfer files; every packaging rule below is
    per file and per set, so two-line stand-ins exercise the stock, accepted and
    unexpected states without a wheel download. The second stand-in is rooted one
    level up, like the carried compatibility guard, so the install-path adaptation
    of both roots is covered.
    """
    patch = root / "generated-source.patch"
    patch.write_text(
        "From 0000000000000000000000000000000000000001 Mon Sep 17 00:00:00 2001\n"
        "From: Test <test@example.com>\n"
        "Subject: [PATCH] test: the accepted generated source\n"
        "\n"
        "---\n"
        "diff --git a/kernel.py b/kernel.py\n"
        "--- a/kernel.py\n"
        "+++ b/kernel.py\n"
        "@@ -1,2 +1,2 @@\n"
        " # stock kernel\n"
        "-return 1\n"
        "+return 2\n"
        "-- \n"
        "2.56.0\n"
    )
    (root / "generated-guard.patch").write_text(
        "From 0000000000000000000000000000000000000002 Mon Sep 17 00:00:00 2001\n"
        "From: Penny <Pennyroyal@agentmail.to>\n"
        "Subject: [PATCH] test: the generated build-only guard\n"
        "\n"
        "---\n"
        "diff --git a/guard.cu b/guard.cu\n"
        "--- a/guard.cu\n"
        "+++ b/guard.cu\n"
        "@@ -1,2 +1,4 @@\n"
        "-#include <unused.h>\n"
        "+#if 0\n"
        "+#include <unused.h>\n"
        "+#endif\n"
        " int guard;\n"
        "-- \n"
        "2.56.0\n"
    )
    return {
        "flashinfer_python": "9.9.9",
        "flashinfer_jit_cache": "9.9.9+cu130",
        "cuda_arch_list": "12.0f",
        "aot_module": "fused_moe_120",
        "aot_path": "data/aot/fused_moe_120/fused_moe_120.so",
        "patches": [
            {
                "patch": str(patch),
                "root": ".",
                "author": "Test <test@example.com>",
                "commits": ["0" * 40],
                "attribution": "Test <test@example.com>, generated fix",
                "reason": "the generated accepted fix",
                "files": [
                    {
                        "path": "kernel.py",
                        "stock_sha256": digest("# stock kernel\nreturn 1\n"),
                        "accepted_sha256": digest("# stock kernel\nreturn 2\n"),
                    }
                ],
            },
            {
                "patch": str(root / "generated-guard.patch"),
                "root": "..",
                "attribution": "Penny <Pennyroyal@agentmail.to>, generated guard",
                "author": "Penny <Pennyroyal@agentmail.to>",
                "commits": ["d" * 40],
                "reason": "the generated include guard",
                "scope": "generated build-only guard",
                "files": [
                    {
                        "path": "guard.cu",
                        "stock_sha256": digest(GUARD_STOCK),
                        "accepted_sha256": digest(GUARD_ACCEPTED),
                    }
                ],
            },
        ],
    }


def raises(call, needle: str) -> None:
    try:
        call()
    except RuntimeError as error:
        assert needle in str(error), f"expected {needle!r} in {error}"
    else:
        raise AssertionError(f"nothing failed; expected {needle!r}")


def test_carried_mailboxes_are_the_accepted_source():
    source = source_manifest()
    assert source["flashinfer_python"] == "0.7.0.post1"
    assert source["flashinfer_jit_cache"] == "0.7.0.post1+cu130"
    assert source["cuda_arch_list"] == "12.0f"
    assert source["aot_path"] == "data/aot/fused_moe_120/fused_moe_120.so"
    assert (
        source["flashinfer_base_commit"] == "946200de1ae94fc93fdd0926f0a13afd1fa7f0f1"
    )
    expected = {
        "patches/moe-source.patch": {
            "commits": [
                "5e86c489f5759cb4006d3b1b5e7bdd14f7a581df",
                "b9fa8893102dc3bbcec92762230e7f1678644a4e",
                "2a4d8d3a9501bf3b3fe3b78d7c6bad38bfc76064",
            ],
            "paths": [
                "csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh",
                "csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h",
            ],
            "author": "Penny <Pennyroyal@agentmail.to>",
            "root": "data",
            # The donor lineage the accepted commits record.
            "lineage": [
                "aiueo52/flash-next-rtxpro6000",
                "524af49abcca66fcb4377ba8297022804535fccf",
            ],
        },
        "patches/gdn-source.patch": {
            "commits": ["0b0ba4c2b18173303b46dd8ec381735e1615b313"],
            "paths": [
                "flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm120.py",
                "flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py",
                "flashinfer/gdn_kernels/delta_rule_dsl/helpers.py",
                "flashinfer/gdn_prefill.py",
            ],
            "author": "aa24aa <2496788660@qq.com>",
            "root": "..",
            # Upstream FlashInfer #6227, cherry picked from the accepted commit.
            "lineage": ["c0771c79b7e2f2bc0edf4fdcb7c43b986a56707a", "#6227"],
        },
        COMPAT: {
            "commits": [COMPAT_COMMIT],
            "paths": [COMPAT_PATH],
            "author": "Penny <Pennyroyal@agentmail.to>",
            "root": "..",
            # Declared in the manifest for what it is: a local build guard.
            "lineage": ["not an upstream FlashInfer change"],
        },
    }
    by_name = {entry["patch"]: entry for entry in source["patches"]}
    assert set(by_name) == set(expected)
    for name, want in expected.items():
        entry = by_name[name]
        mailbox = (PACKAGE_DIR / name).read_text()
        assert entry["commits"] == want["commits"], name
        assert entry["author"] == want["author"], name
        assert entry["root"] == want["root"], name
        assert sorted(spec["path"] for spec in entry["files"]) == sorted(
            want["paths"]
        ), name
        assert f"From: {want['author']}" in mailbox, name
        for commit in want["commits"]:
            assert mailbox.count(f"From {commit} ") == 1, (name, commit)
        for token in want["lineage"]:
            # Donor and upstream ids live in the mailbox or in the manifest's
            # attribution line, never in a comment someone can drop silently.
            assert token in mailbox or token in entry["attribution"], (name, token)
        assert want["commits"] == entry["commits"], name
        for spec in entry["files"]:
            assert DIGEST.match(spec["stock_sha256"]), spec["path"]
            assert DIGEST.match(spec["accepted_sha256"]), spec["path"]
            assert spec["stock_sha256"] != spec["accepted_sha256"], spec["path"]
    # Scope guards: the accepted input is the six production files, nothing else.
    assert len(by_name["patches/moe-source.patch"]["files"]) == 2
    assert len(by_name["patches/gdn-source.patch"]["files"]) == 4
    compat = by_name[COMPAT]
    guard = compat["files"][0]
    assert guard["stock_sha256"] == COMPAT_STOCK_SHA256, guard
    assert guard["accepted_sha256"] == COMPAT_ACCEPTED_SHA256, guard
    # The two accepted fixes stay the fingerprinted pair; the guard travels with
    # them, last, and is not itself one of the Xid109 fixes.
    assert source["patches"].index(compat) == len(source["patches"]) - 1, source
    mailbox = (PACKAGE_DIR / COMPAT).read_text()
    # The preimage is the stock unconditional include that real build choked on,
    # and the fix is a pure addition around it: the guard restates the file's own
    # ASAN condition, so nothing in the file moves and nothing is stubbed out.
    assert " #include <sanitizer/asan_interface.h>\n+#endif" in mailbox, mailbox
    assert "+#if defined(__SANITIZE_ADDRESS__) || \\" in mailbox, mailbox
    assert (
        "+    (defined(__has_feature) && __has_feature(address_sanitizer))" in mailbox
    )
    assert not re.search(r"^-[^-]", mailbox, flags=re.M), mailbox
    assert "FLASHINFER_" not in mailbox, "the guard adds no new knob"
    # Build-only: one file, and the changed lines touch nothing but the guard.
    assert mailbox.count("diff --git") == 1, mailbox
    changed = "".join(
        line + "\n"
        for line in mailbox.splitlines()
        if line[:1] in "+-" and not line.startswith(("---", "+++", "-- "))
    )
    for token in ("cudaMemcpy", "__nv_bfloat16", "-fsanitize", "CUDAGen", "cudaStream"):
        assert token not in changed, token
    assert "sanitizer headers" in compat["scope"], compat["scope"]
    assert (
        "FLASHINFER_MOE_FUSED_PROLOGUE"
        in (PACKAGE_DIR / "patches/moe-source.patch").read_text()
    )
    assert (
        "FLASHINFER_GDN_FP16_ACCUM_MMA"
        in (PACKAGE_DIR / "patches/gdn-source.patch").read_text()
    )


def test_accepted_pin_matches_every_dependency_pin():
    source = source_manifest()
    version, cache = source["flashinfer_python"], source["flashinfer_jit_cache"]
    assert cache == f"{version}+cu130"
    assert (
        f'"flashinfer_python[cu13]=={version}"'
        in (REPO / "python/pyproject.toml").read_text()
    )
    dockerfile = DOCKERFILE.read_text()
    assert f"'flashinfer-jit-cache=={version}+cu130'" in dockerfile
    assert f"'flashinfer-jit-cache-sm120f=={version}+cu130'" in dockerfile
    assert f'"flashinfer-python": "{version}"' in CHECK.read_text()


def test_stock_source_is_patched_once_and_a_repeat_run_accepts_it(tmp_path):
    module = installer()
    source = synthetic_source(tmp_path)
    package = installed_tree(
        tmp_path / "site", {"kernel.py": "# stock kernel\nreturn 1\n"}, "9.9.9"
    )
    patch = Path(source["patches"][0]["patch"])
    guard = Path(source["patches"][1]["patch"])
    # A Pennyroyal checkout around the environment must not redirect git apply.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)

    assert module.apply_accepted_source(package, source) == [
        f"{patch}: applied",
        f"{guard}: applied",
    ]
    assert (package / "kernel.py").read_text() == "# stock kernel\nreturn 2\n"
    assert (package.parent / "guard.cu").read_text() == GUARD_ACCEPTED
    # The same command again is a no-op, not an error and not a double patch.
    assert module.apply_accepted_source(package, source) == [
        f"{patch}: already patched",
        f"{guard}: already patched",
    ]
    assert (package / "kernel.py").read_text() == "# stock kernel\nreturn 2\n"
    assert (package.parent / "guard.cu").read_text() == GUARD_ACCEPTED


def test_unexpected_source_and_wrong_pin_fail_before_building(tmp_path):
    module = installer()
    source = synthetic_source(tmp_path)
    package = installed_tree(
        tmp_path / "site", {"kernel.py": "# mine\nreturn 3\n"}, "9.9.9"
    )
    raises(
        lambda: module.apply_accepted_source(package, source),
        "neither the stock nor the accepted",
    )

    # The compatibility guard is its own set, one level above the package: a
    # third-party edit of that single file is refused the same way, before
    # anything is compiled.
    edited = installed_tree(
        tmp_path / "site-guard",
        {"kernel.py": "# stock kernel\nreturn 2\n"},
        "9.9.9",
        guard="#pragma once\n",
    )
    raises(
        lambda: module.apply_accepted_source(edited, source),
        f"guard.cu holds {digest('#pragma once' + chr(10))}",
    )

    # A partly applied state is a different answer from a repeat installation.
    source["patches"][0]["files"].append(
        {
            "path": "second.py",
            "stock_sha256": digest("second\n"),
            "accepted_sha256": digest("patched\n"),
        }
    )
    (package / "kernel.py").write_text("# stock kernel\nreturn 1\n")
    (package / "second.py").write_text("changed\n")
    raises(
        lambda: module.apply_accepted_source(package, source),
        "neither the stock nor the accepted",
    )

    for version in ("0.6.17", "0.7.1"):
        other = installed_tree(tmp_path / f"site-{version}", {"kernel.py": ""}, version)
        raises(
            lambda other=other: module.require_accepted_pin(source, other),
            "not the accepted 9.9.9 source pin",
        )
    missing = tmp_path / "site-nodist" / "flashinfer"
    missing.mkdir(parents=True)
    raises(
        lambda: module.require_accepted_pin(source, missing),
        "no flashinfer_python .dist-info",
    )


def test_installed_check_needs_the_accepted_source_and_the_package_local_module(
    tmp_path,
):
    module = installer()
    source = synthetic_source(tmp_path)
    package = installed_tree(
        tmp_path / "site", {"kernel.py": "# stock kernel\nreturn 1\n"}, "9.9.9"
    )
    stock_kernel = tmp_path / "provider" / "fused_moe_120.so"
    stock_kernel.parent.mkdir()
    stock_kernel.write_bytes(b"stock provider bytes")
    installed = package / source["aot_path"]
    installed.parent.mkdir(parents=True)
    installed.write_bytes(b"stock provider bytes")

    # Stock source with the provider wheel: what an ordinary 0.7.0.post1 install
    # looks like, and what used to satisfy the image check.
    raises(
        lambda: module.check_installed(package, source, [stock_kernel]),
        "is stock FlashInfer source",
    )
    # Patched fixes with the guard missing is its own answer, and it names the
    # file the compile would have died on.
    module.apply_accepted_source(package, source)
    (package.parent / "guard.cu").write_text(GUARD_STOCK)
    raises(
        lambda: module.check_installed(package, source, [stock_kernel]),
        "the generated include guard",
    )
    (package.parent / "guard.cu").write_text(GUARD_ACCEPTED)
    # Patched source, but the preferred path only holds a copy of the prebuilt
    # provider kernel: nothing was built from the accepted source.
    raises(
        lambda: module.check_installed(package, source, [stock_kernel]),
        "copy of the stock provider kernel",
    )
    installed.unlink()
    raises(
        lambda: module.check_installed(package, source, [stock_kernel]),
        "absent or empty",
    )

    installed.write_bytes(b"built from patched bytes")
    info = module.check_installed(package, source, [stock_kernel])
    assert info["sm120_module"] == str(installed)
    assert info["sm120_module_bytes"] == len(b"built from patched bytes")
    assert (
        info["sm120_module_sha256"]
        == hashlib.sha256(b"built from patched bytes").hexdigest()
    )
    assert info["flashinfer"] == "9.9.9"
    assert info["accepted_source"] == [
        {
            "patch": str(tmp_path / "generated-source.patch"),
            "author": "Test <test@example.com>",
            "commits": ["0" * 40],
        },
        {
            "patch": str(tmp_path / "generated-guard.patch"),
            "author": "Penny <Pennyroyal@agentmail.to>",
            "commits": ["d" * 40],
        },
    ]

    # Copying the stock prebuilt kernel into the preferred path is not a build,
    # and without a provider wheel to compare against the module is still required.
    installed.write_bytes(stock_kernel.read_bytes())
    raises(
        lambda: module.check_installed(package, source, [stock_kernel]),
        "copy of the stock provider kernel",
    )
    module.check_installed(package, source)


def test_the_image_check_refuses_an_unpackaged_flashinfer(tmp_path, monkeypatch):
    """docker/pennyroyal/check_install.py must not pass on a stock install."""
    installed_tree(tmp_path / "site", {"gdn_prefill.py": "# stock\n"}, "0.7.0.post1")
    spec = importlib.util.spec_from_file_location("penny_container_check", CHECK)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    importlib.invalidate_caches()
    try:
        check.check_flashinfer_sm120(REPO, [])
    except RuntimeError as error:
        assert "FlashInfer source" in str(error), str(error)
    else:
        raise AssertionError("the container check passed on a stock FlashInfer")


def test_build_targets_the_patched_sources_without_the_prebuilt_shortcut():
    code = installer().BUILD_CODE
    assert "gen_cutlass_fused_moe_sm120_module(use_fast_build=False)" in code
    assert "spec.build()" in code
    assert "print(pathlib.Path(spec.jit_library_path))" in code
    # The shortcuts that take the stock AOT module and compile nothing.
    for shortcut in ("build_and_load", "build_jit_specs", "skip_prebuilt"):
        assert shortcut not in code, shortcut
    # The patched headers of the selected package, not another source tree.
    assert "PENNY_FLASHINFER_ACCEPTED_CSRC" in code
    assert "FLASHINFER_CSRC_DIR" in code
    text = INSTALLER.read_text()
    assert 'env["FLASHINFER_CUDA_ARCH_LIST"] = source["cuda_arch_list"]' in text
    # The env the build runs under comes from the helper that maps the host
    # compiler, so the mapping cannot be bypassed by another call site.
    assert "env = nvcc_environment(source, package, os.environ)" in text
    assert source_manifest()["cuda_arch_list"] == "12.0f"
    for host_path in ("/home/", "thegrid", ".aot-overlay"):
        assert host_path not in text, host_path


def test_a_failed_build_keeps_the_compiler_diagnostic(tmp_path, monkeypatch):
    """Ninja's summary line is not the diagnostic, and the wrapper must not hide it.

    A real FlashInfer compile failure ends in "ninja: build stopped: subcommand
    failed."; the fatal error an operator needs is further up. The report has to
    carry that output and the effective host compiler together.
    """
    module = installer()
    source = synthetic_source(tmp_path)
    package = installed_tree(
        tmp_path / "site", {"kernel.py": "# stock kernel\nreturn 2\n"}, "9.9.9"
    )
    (package / source["aot_path"]).parent.mkdir(parents=True)
    monkeypatch.setenv("CUDAHOSTCXX", "/usr/bin/g++-15")
    monkeypatch.setenv("CC", "/usr/bin/gcc")
    monkeypatch.setenv("MAX_JOBS", "24")
    # The shape of the real failure: filler, the fatal include error, and the
    # unhelpful summary ninja prints last.
    lines = [
        f"[{n}/101] Building CUDA object CMakeFiles/dir/memoryUtils.cuda.o"
        for n in range(96)
    ]
    stderr = "\n".join(
        lines
        + [
            "/usr/lib/python3.12/site-packages/flashinfer/data/csrc/nv_internal/"
            "cpp/common/memoryUtils.cu:18:10: fatal error: "
            "sanitizer/asan_interface.h: No such file or directory",
            "   18 | #include <sanitizer/asan_interface.h>",
            "      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~",
            "compilation terminated.",
            "ninja: build stopped: subcommand failed.",
        ]
    )

    def failing_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], output="", stderr=stderr)

    monkeypatch.setattr(module.subprocess, "run", failing_run)
    raises(
        lambda: module.install_accepted_source(package, source),
        "fatal error: sanitizer/asan_interface.h: No such file or directory",
    )
    # Both facts are in the one message: the output that matters, and the
    # compiler that produced it. Truncation is bounded and says so.
    try:
        module.install_accepted_source(package, source)
    except RuntimeError as error:
        report = str(error)
    assert "/usr/bin/g++-15 (CUDAHOSTCXX)" in report, report
    assert "ninja: build stopped: subcommand failed." in report, report
    assert "MAX_JOBS=24" in report, report
    # The excerpt is the bounded tail, and it says how much came before it.
    tail = module.BUILD_ERROR_TAIL_LINES
    assert f"[... {len(lines) + 5 - tail} earlier lines ...]" in report, report
    assert report.splitlines()[-tail:] == stderr.splitlines()[-tail:], report
    # An empty failure still says so instead of reporting an empty line.
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, a[0], output="", stderr="")
        ),
    )
    raises(
        lambda: module.install_accepted_source(package, source), "no compiler output"
    )


def test_the_build_subprocess_gets_the_explicit_cuda_host_compiler(tmp_path):
    """CUDAHOSTCXX is what an operator means; CC is what FlashInfer reads.

    jit/cpp_ext.py turns CC into nvcc's -ccbin and never looks at CUDAHOSTCXX, so
    a supported host compiler named only in CUDAHOSTCXX used to be ignored and the
    build bound the environment's default compiler. The mapping may not leak out of
    the build subprocess, and it may not touch the C++/link compiler or the jobs.
    """
    module = installer()
    source = source_manifest()
    caller = {
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "CUDAHOSTCXX": "/usr/bin/g++-15",
        "MAX_JOBS": "24",
    }
    package = tmp_path / "flashinfer"

    env = module.nvcc_environment(source, package, caller)
    assert env["CC"] == "/usr/bin/g++-15", env
    assert env["CXX"] == "/usr/bin/g++", "the C++ and link compiler is untouched"
    assert env["MAX_JOBS"] == "24", "the job budget is untouched"
    assert env["FLASHINFER_CUDA_ARCH_LIST"] == source["cuda_arch_list"]
    assert env["PENNY_FLASHINFER_ACCEPTED_CSRC"] == str(package / "data" / "csrc")
    # The caller's own environment is preserved: an ordinary shell keeps its CC.
    assert caller == {
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "CUDAHOSTCXX": "/usr/bin/g++-15",
        "MAX_JOBS": "24",
    }, caller
    # The diagnostic names the compiler that is really in force, and says where it
    # came from, instead of quoting a variable FlashInfer ignores.
    assert module.nvcc_host_compiler(env) == "/usr/bin/g++-15 (CUDAHOSTCXX)"

    # Without the override the existing CC behaviour stands, and an empty
    # CUDAHOSTCXX is not a request to change compilers.
    for unset in (
        {k: v for k, v in caller.items() if k != "CUDAHOSTCXX"},
        {**caller, "CUDAHOSTCXX": ""},
    ):
        plain = module.nvcc_environment(source, package, unset)
        assert plain["CC"] == "/usr/bin/gcc", plain
    assert module.nvcc_host_compiler(plain) == "/usr/bin/gcc (CC)"
    assert module.nvcc_host_compiler({"CXX": "/usr/bin/g++"}) == "nvcc default"
    # The step wires the helper in, so the mapping cannot be bypassed by a caller.
    text = INSTALLER.read_text()
    assert "env = nvcc_environment(source, package, os.environ)" in text
    assert (
        "CUDAHOSTCXX=" not in text.split("def main")[1]
    ), "no ignored variable in the report"


def test_both_installation_paths_run_the_one_step():
    guide = BUILD_GUIDE.read_text()
    dockerfile = DOCKERFILE.read_text()
    fresh = guide.split("## Fresh install", 1)[1].split(
        "## Update an existing install", 1
    )[0]
    update = guide.split("## Update an existing install", 1)[1].split(
        "## NIXL POSIX", 1
    )[0]
    assert "scripts/pennyroyal/flashinfer/install.py" in fresh
    assert "scripts/pennyroyal/flashinfer/install.py" in update
    assert "--no-deps -e python" in update
    assert "flashinfer-python" in update  # the prerequisite is spelled out
    jit_cache = dockerfile.index("flashinfer-jit-cache-sm120f")
    packaging = dockerfile.index("scripts/pennyroyal/flashinfer/install.py")
    freeze = dockerfile.index("pip check")
    assert jit_cache < packaging < freeze
    assert "PENNY_BUILD_JOBS=4" in dockerfile and "MAX_JOBS=4" in dockerfile
    assert "check_flashinfer_sm120" in CHECK.read_text()


def test_a_stale_jit_cache_shim_stops_the_upgrade_before_the_build(tmp_path):
    """The 0.6.17 -> 0.7.0.post1 native sequence, checked on a stand-in site.

    flashinfer-python's own metadata does not pull the JIT-cache family, so an
    upgraded environment keeps the old shim, and FlashInfer aborts its import
    over the mismatch. The step has to name that package instead of dying inside
    the compile.
    """
    module = installer()
    source = synthetic_source(tmp_path)
    package = installed_tree(
        tmp_path / "site",
        {"kernel.py": "# stock kernel\nreturn 2\n"},
        "9.9.9",
        guard=GUARD_ACCEPTED,
    )
    (package / source["aot_path"]).parent.mkdir(parents=True)
    (package / source["aot_path"]).write_bytes(b"built")
    site = tmp_path / "site"

    def dist_info(name, version):
        info = site / f"{name}-{version}.dist-info"
        info.mkdir(exist_ok=True)
        (info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")

    # No cache family at all: genuinely optional, the step still accepts it.
    module.check_installed(package, source)

    # The stale shim a v2.5.x environment keeps beside the new FlashInfer.
    dist_info("flashinfer_jit_cache", "0.6.17+cu130")
    dist_info("flashinfer_jit_cache_sm120f", "0.6.17+cu130")
    for call in (
        lambda: module.check_installed(package, source),
        lambda: module.install_accepted_source(package, source),
    ):
        raises(call, "flashinfer-jit-cache 0.6.17+cu130")
    # The documented remedy: the same pinned family the image installs, which
    # replaces the old wheel. The provider may stay old -- FlashInfer skips an
    # incompatible provider with a warning -- because only the shim breaks the
    # import, and only the shim is refused here.
    shutil.rmtree(site / "flashinfer_jit_cache-0.6.17+cu130.dist-info")
    dist_info("flashinfer_jit_cache", "9.9.9+cu130")
    module.check_installed(package, source)


def test_the_documented_update_aligns_the_family_before_the_step():
    guide = BUILD_GUIDE.read_text()
    # The one fenced command block that touches the JIT-cache family is the
    # primary update sequence itself: an operator pastes one block, in order.
    blocks = [part for part in guide.split("```") if "flashinfer-jit-cache==" in part]
    assert len(blocks) == 1, blocks
    update = blocks[0]
    source = source_manifest()
    step = update.index("scripts/pennyroyal/flashinfer/install.py")
    pin = update.index(f"'flashinfer-python[cu13]=={source['flashinfer_python']}'")
    cache = update.index(f"'flashinfer-jit-cache=={source['flashinfer_jit_cache']}'")
    assert f"'flashinfer-jit-cache-sm120f=={source['flashinfer_jit_cache']}'" in update
    # The family is aligned before the step runs, so a v2.5.x upgrade works on
    # its first pass instead of dying on the stale shim inside the compile.
    assert update.index("--no-deps -e python") < pin < cache < step, update
    assert "--no-deps --index-url https://flashinfer.ai/whl/cu130" in update
    assert "#flashinfer-python" not in update
    # Both source selections name a release ref rather than an older tag, so the
    # step cannot be documented against a checkout that lacks it.
    for sequence in (
        guide.split("## Fresh install", 1)[1].split("## Update an existing install", 1)[
            0
        ],
        guide.split("## Update an existing install", 1)[1].split("## NIXL POSIX", 1)[0],
    ):
        assert "$RELEASE_REF" in sequence, sequence[:200]
        assert "scripts/pennyroyal/flashinfer/install.py" in sequence
    assert "--branch pennyroyal-v" not in guide
    assert "switch --detach pennyroyal-v" not in guide


def test_next_recipes_default_the_accepted_gdn_mode_and_27b_does_not():
    for script in NEXT_SCRIPTS:
        text = script.read_text()
        lines = text.splitlines()
        assert GDN_EXPORT in lines, script
        # Before the interpreter or the server starts, so FlashInfer reads it.
        first_start = min(
            index
            for index, line in enumerate(lines)
            if '"$PYTHON"' in line or '"$SGLANG_EXE"' in line
        )
        assert lines.index(GDN_EXPORT) < first_start, script
        # The mode changes the computation, so it has to change the persisted
        # identity too: resolved to the mode it actually selects, then named in
        # the recipe's existing namespace field list, before the helper runs.
        assert GDN_RESOLVE in lines, script
        assert GDN_FIELD in text, script
        assert text.index("GDN_FP16_ACCUM_MMA=off") < text.index(GDN_FIELD), script
        # It sits inside the existing field list, next to the other recurrent
        # state mode, in the one derivation call that selects the cache root.
        sibling = [
            i for i, line in enumerate(lines) if "gdn_mtp_cache_mode=none" in line
        ]
        field_at = [
            i for i, line in enumerate(lines) if line.strip() == GDN_FIELD.strip()
        ]
        assert len(sibling) == len(field_at) == 1, (script, sibling, field_at)
        assert field_at[0] == sibling[0] + 1, (
            script,
            lines[sibling[0]],
            lines[field_at[0]],
        )
        assert lines[field_at[0]].startswith(
            lines[sibling[0]][: -len(lines[sibling[0]].lstrip())]
        ), script
    for script in OTHER_SCRIPTS:
        text = script.read_text()
        assert "FLASHINFER_GDN_FP16_ACCUM_MMA" not in text, script
        assert "gdn_fp16_accum_mma" not in text, script
    # The accepted source keeps the mode off; no global FlashInfer default moved.
    assert (
        "FLASHINFER_GDN_FP16_ACCUM_MMA"
        not in (REPO / "python/pyproject.toml").read_text()
    )
    entrypoint = (REPO / "docker/pennyroyal/entrypoint.sh").read_text()
    assert "FLASHINFER_GDN_FP16_ACCUM_MMA" not in entrypoint


if __name__ == "__main__":  # pytest is the real driver; this is a smoke check
    for name, case in sorted(globals().items()):
        if not (name.startswith("test_") and inspect.isfunction(case)):
            continue
        with tempfile.TemporaryDirectory() as temp:
            case(Path(temp)) if inspect.signature(case).parameters else case()
        print("pass", name)
