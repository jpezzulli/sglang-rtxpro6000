"""CPU-only packaging checks for the accepted FlashInfer SM120 source.

No GPU, no CUDA toolchain, no container runtime and no wheel download: the
packaging rules are exercised against synthetic installed-source trees, and the
carried mailboxes are checked against their pins and attribution. Compilation,
image build and GPU behaviour are qualified separately on the RTX PRO 6000 host;
nothing here claims them. The reuse step is held to the same standard: the
fingerprint and the cache are exercised against fixture bytes -- a stand-in
CUDA tree, a stand-in compiler script and a stand-in build.ninja -- and a fake
JitSpec records whether the compile was dispatched. Nothing here runs nvcc, and
no test claims a real compiled binary or a real container layer.
"""

import hashlib
import importlib.util
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

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


def installer(*, shared_libraries=False):
    spec = importlib.util.spec_from_file_location("penny_flashinfer_sm120", INSTALLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not shared_libraries:
        # Most tests use text stand-ins for compiler executables. Exercise ELF
        # resolution separately with real binaries and in the image probe.
        module._program_libraries = lambda programs, incomplete: {}
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
    # The cubin payload invariant is reported with the rest of the check; a
    # site without the optional flashinfer-cubin wheel is an accepted state.
    assert info["cubin_payload"] == {"status": "absent"}
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
    assert "print(state, module)" in code
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
    # The compile itself is reached through the reuse helper, so no caller can
    # rejoin ninja without passing the fingerprint check.
    assert "step.build_or_reuse(" in code
    assert "step.accepted_sources()" in code
    assert "spec.build()" not in code, "the build code does not reach ninja itself"
    assert "def build_or_reuse" in text
    assert "spec.build()" in text.split("def build_or_reuse")[1]


def test_a_failed_build_keeps_the_compiler_diagnostic(tmp_path, monkeypatch):
    """Ninja's summary line is not the diagnostic, and the wrapper must not hide it.

    A real FlashInfer compile failure ends in "ninja: build stopped: subcommand
    failed."; the fatal error an operator needs is further up. Which stream it is on
    depends on FLASHINFER_JIT_VERBOSE, because run_ninja pipes the merged build
    output into stderr normally but lets ninja write to the child's own stdout, and
    raises with nothing embedded, when verbose. Both variants have to survive with
    the effective host compiler next to them.
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

    fatal = (
        "/usr/lib/python3.12/site-packages/flashinfer/data/csrc/nv_internal/"
        "cpp/common/memoryUtils.cu:18:10: fatal error: "
        "sanitizer/asan_interface.h: No such file or directory"
    )
    compiler_output = "\n".join(
        lines
        + [
            fatal,
            "   18 | #include <sanitizer/asan_interface.h>",
            "compilation terminated.",
        ]
    )
    summary = "ninja: build stopped: subcommand failed."
    # FlashInfer's own traceback, with no build output embedded, is all verbose
    # mode leaves on stderr.
    traceback_only = "\n".join(
        [
            "Traceback (most recent call last):",
            '  File "<string>", line 12, in <module>',
            "  File .../flashinfer/jit/cpp_ext.py, line 415, in run_ninja",
            "    raise RuntimeError(msg) from e",
            'RuntimeError: Ninja build failed."',
            summary,
        ]
    )
    tail = module.BUILD_ERROR_TAIL_LINES

    def report_for(output, stderr):
        def failing_run(*args, **kwargs):
            raise subprocess.CalledProcessError(
                1, args[0], output=output, stderr=stderr
            )

        monkeypatch.setattr(module.subprocess, "run", failing_run)
        try:
            module.install_accepted_source(package, source)
        except RuntimeError as error:
            return str(error)
        raise AssertionError("the build failure did not surface")

    # The default shape: FlashInfer captures the merged build output into stderr.
    merged = compiler_output + "\n" + summary
    report = report_for("", merged)
    assert fatal in report, report
    assert summary in report, report
    # Both facts are in the one message, and truncation is bounded and says so.
    assert "/usr/bin/g++-15 (CUDAHOSTCXX)" in report, report
    assert "MAX_JOBS=24" in report, report
    assert "--- stderr ---" in report, report
    assert f"[... {len(merged.splitlines()) - tail} earlier stderr lines ...]" in report
    assert report.splitlines()[-tail:] == merged.splitlines()[-tail:], report

    # The verbose shape: the build output went to the child's stdout, so stderr
    # alone would show a traceback and a summary and hide the fatal error.
    report = report_for(compiler_output + "\n" + summary, traceback_only)
    assert fatal in report, report
    assert "--- stdout ---" in report and "--- stderr ---" in report, report
    assert "RuntimeError: Ninja build failed." in report, report
    assert (
        report.index("--- stdout ---")
        < report.index(fatal)
        < report.index("--- stderr ---")
    )
    assert "/usr/bin/g++-15 (CUDAHOSTCXX)" in report, report
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


def test_the_launchers_default_the_generic_rtx_targets():
    """No unset native environment may reintroduce an SM120-only generic JIT.

    The six profile launchers carry the architecture-only compiler fallback; a
    user-set TORCH_CUDA_ARCH_LIST must still win, and the dedicated SM120
    fused-MoE target lives elsewhere (accepted-sources cuda_arch_list).
    """
    for script in (*NEXT_SCRIPTS, *OTHER_SCRIPTS):
        text = script.read_text()
        assert (
            'TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6 8.9 12.0+PTX}"' in text
        ), script
        assert ':-12.0}"' not in text, script
    assert source_manifest()["cuda_arch_list"] == "12.0f"


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
    freeze = dockerfile.index(
        "python scripts/pennyroyal/flashinfer/check_dependencies.py"
    )
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


# --- reusing the compiled SM120 module -------------------------------------

CACHE_SCHEMA = "pennyroyal-flashinfer-aot-v1"
CACHE_RECORD = "record.json"
CACHE_LATEST = "latest"
CACHE_SUFFIX = ".aot"
COMPILED = b"compiled by nvcc\n"
JOINED = " $\n    ".join  # flashinfer.jit.cpp_ext.join_multiline, verbatim


def build_fixture(root: Path, workspace: Path | None = None) -> SimpleNamespace:
    """One stand-in installed environment, as the fingerprint sees it.

    The CUDA tree, the headers, the linked libraries, the interpreter, torch and
    tvm-ffi are fixture files, and the host "compiler" is a script that answers the
    queries the fingerprint asks it -- including the bare names it must place from
    its own search path -- so the reading is the real one with no toolchain present.
    The build.ninja has the shape FlashInfer's generator writes: values wrapped with
    "$" followed by a newline, the -ccbin argument on the line after its flag, and
    the generated instantiations inside the build's scratch directory.
    """
    workspace = workspace or root / "workspace-a"

    def file(relative: str, payload: bytes, base: Path = root) -> Path:
        target = base / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    cuda = root / "cuda"
    fixture = SimpleNamespace(
        root=root,
        workspace=workspace,
        nvcc=file("bin/nvcc", b"nvcc image one\n", cuda),
        libdevice=file("nvvm/libdevice.10.bc", b"libdevice one\n", cuda),
        cuda_header=file("include/cuda_runtime.h", b"typedef int one;\n", cuda),
        cudart=file("lib64/libcudart.so", b"cudart one\n", cuda),
        cudadevrt=file("lib64/libcudadevrt.a", b"cudadevrt one\n", cuda),
        libcuda=file("lib64/stubs/libcuda.so", b"libcuda one\n", cuda),
        header=file("flashinfer-include/kernel.h", b"#pragma once\n"),
        csrc=file("flashinfer-csrc/moe.cuh", b"#pragma once\n"),
        python_header=file("python-include/Python.h", b"typedef int one;\n"),
        ffi_header=file("tvm-ffi-include/builtin.h", b"#pragma once\n"),
        cxx_include=file("c++-include/vector", b"#pragma once\n"),
        crt=file("gcc-15/crtbeginS.o", b"crt one\n"),
        libgcc=file("gcc-15/libgcc.a", b"libgcc one\n"),
        cc1plus=file("gcc-15/cc1plus", b"cc1plus one\n"),
        collect2=file("gcc-15/collect2", b"collect2 one\n"),
        assembler=file("gcc-15/as", b"as one\n"),
        ld=file("toolchain/ld", b"ld one\n"),
        libc=file("toolchain/libc.so.6", b"glibc one\n"),
        libstdcxx=file("toolchain/libstdc++.so.6", b"libstdc++ one\n"),
    )
    scripts = {
        "libc.so": file(
            "toolchain/libc.so", b"/* GNU ld script */\nGROUP ( libc.so.6 )\n"
        ),
        "libstdc++.so": file(
            "toolchain/libstdc++.so",
            b"/* GNU ld script */\nGROUP ( libstdc++.so.6 )\n",
        ),
    }
    fixture.abi = {
        "python": file("toolchain/python3.12", b"cpython one\n"),
        "torch": file("toolchain/libtorch_python.so", b"libtorch one\n"),
        "tvm_ffi/libtvm_ffi.so": file("toolchain/libtvm_ffi.so", b"tvm ffi one\n"),
    }
    fixture.generated = file("generated/120/moe.cu", b"// generated\n", workspace)
    fixture.sources = [
        file("flashinfer-csrc/moe_input.cu", b"// patched flashinfer source\n"),
        file("flashinfer-csrc/moe_fp8.cu", b"// patched flashinfer source\n"),
        fixture.generated,
    ]
    fixture.compiler = file("toolchain/g++-15", b"")
    fixture.ccbin_compiler = file("toolchain/host-c++", b"")
    driver = (
        "#!/bin/sh\n"
        'case "$1" in\n'
        "-print-search-dirs) echo 'install: %s'\n"
        '  echo "programs: =%s:%s"\n  ;;\n'
        % (root / "gcc-15", root / "toolchain", root / "gcc-15")
        + "".join(
            f"-print-prog-name={name}) echo {answer} ;;\n"
            for name, answer in (
                # Real g++ answers bare names for the programs it execs when they
                # are not next to it; only its own search path can place them.
                ("cc1plus", "cc1plus"),
                ("collect2", "collect2"),
                ("as", "as"),
                ("ld", fixture.ld),
            )
        )
        + "".join(
            f"-print-file-name={name}) echo {path} ;;\n"
            for name, path in scripts.items()
        )
        + "*) echo '#include <...> search starts here:' >&2\n"
        f"   echo {fixture.cxx_include.parent} >&2\n"
        "   echo 'End of search list.' >&2 ;;\n"
        "esac\n"
    )
    fixture.compiler.write_text(driver)
    fixture.ccbin_compiler.write_text(driver)
    fixture.compiler.chmod(0o755)
    fixture.ccbin_compiler.chmod(0o755)
    fixture.ninja = ninja_fixture(fixture)
    fixture.inputs = {
        "workspace": [str(workspace / "flashinfer"), str(workspace)],
        "sources": [str(path) for path in fixture.sources],
    }
    return fixture


def ninja_fixture(fixture: SimpleNamespace) -> str:
    """The build file FlashInfer would run, generated the way it generates one."""
    root = fixture.root
    common = [
        "-DPy_LIMITED_API=0x03090000",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
        f"-I{root / 'flashinfer-csrc'}",
        f"-I{fixture.workspace / 'generated'}",
        f"-isystem {root / 'flashinfer-include'}",
        "-isystem $cuda_home/include",
        f"-isystem {root / 'python-include'}",
        f"-isystem {root / 'tvm-ffi-include'}",
    ]
    cuda = [
        "-std=c++17",
        "--threads=1",
        "-use_fast_math",
        "-ccbin",
        str(fixture.ccbin_compiler),
        "$common_cflags",
        "--expt-relaxed-constexpr",
        "-static-global-template-stub=false",
        "-gencode=arch=compute_120f,code=sm_120f",
        "-DNDEBUG",
        "-O3",
    ]
    link = [
        "-shared",
        "-L$cuda_home/lib64",
        "-L$cuda_home/lib64/stubs",
        "-lcudart",
        "-lcuda",
        "-lcudadevrt",
    ]
    return "\n".join(
        [
            "ninja_required_version = 1.3",
            "name = fused_moe_120",
            f"cuda_home = {root / 'cuda'}",
            f"cxx = {fixture.compiler}",
            "nvcc = $cuda_home/bin/nvcc",
            "cxx_launcher =",
            "nvcc_launcher =",
            "",
            "common_cflags = " + JOINED(common),
            "cflags = $common_cflags -fPIC",
            "cuda_cflags = " + JOINED(cuda),
            "cuda_arch_flags = -gencode=arch=compute_120f,code=sm_120f",
            "ldflags = " + JOINED(link),
            "",
            "rule cuda_compile",
            "  command = $nvcc --generate-dependencies-with-compile -MF $out.d"
            " $cuda_cflags -c $in -o $out",
            "",
        ]
        + [
            f"build cached_ops/fused_moe_120/{path.stem}.cuda.o: cuda_compile {path}"
            for path in fixture.sources
        ]
        + ["", "build cached_ops/fused_moe_120/fused_moe_120.so: link objects", ""]
    )


class FakeSpec:
    """The slice of FlashInfer's JitSpec that the reuse step touches.

    ``compiled`` records every build call together with whether an output was
    already sitting there, which is how the tests tell a restored module from one
    that entered the compile step again, and a relink from a republished leftover.
    """

    name = "fused_moe_120"

    payload = COMPILED

    def __init__(self, workspace: Path, ninja: str, sources, compiled: list):
        self.ninja = ninja
        self.sources = sources
        self.compiled = compiled
        self.build_dir = workspace / "flashinfer/cached_ops/fused_moe_120"
        self.ninja_path = self.build_dir / "build.ninja"
        self.jit_library_path = self.build_dir / f"{self.name}.so"

    def write_ninja(self) -> None:
        self.ninja_path.parent.mkdir(parents=True, exist_ok=True)
        self.ninja_path.write_text(self.ninja)
        # Ninja leaves its objects next to the module it links, and it is content to
        # relink them without asking whether the inputs that made them changed.
        (self.build_dir / "moe_input.cuda.o").write_bytes(self.payload)

    def build(self, *args, **kwargs) -> None:
        self.compiled.append(
            (
                self.name,
                self.jit_library_path.is_file(),
                (self.build_dir / "moe_input.cuda.o").is_file(),
            )
        )
        self.jit_library_path.parent.mkdir(parents=True, exist_ok=True)
        self.jit_library_path.write_bytes(self.payload)


def reused(module, fixture, source, cache, workspace, report, payload=COMPILED):
    """Run the reuse step in one separate build environment, recording the calls."""
    compiled: list = []
    spec = FakeSpec(workspace, fixture.ninja, fixture.sources, compiled)
    spec.payload = payload
    with redirect_stderr(report):
        built, state = module.build_or_reuse(
            spec, source, cache, fixture.inputs, fixture.abi
        )
    return spec, built, state, compiled, report.getvalue()


def fingerprint_of(module, fixture, source, ninja=None):
    return module.build_fingerprint(
        ninja or fixture.ninja, source, fixture.inputs, fixture.abi
    )


def cache_keys(cache: Path) -> set:
    return {path.name for path in (cache / CACHE_SCHEMA).iterdir()}


def test_the_generated_build_file_is_read_as_generated(tmp_path):
    """The real generator wraps values with "$" and a newline; the reader must cope.

    A hand-written file with backslash continuations proves nothing about that, and
    the first version of this step died here: it reassembled the lines wrongly, took
    the trailing dollar sign as part of the flags, and then tried to fingerprint "$"
    as a compiler.
    """
    module = installer()
    fixture = build_fixture(tmp_path)
    source = synthetic_source(tmp_path)
    assert " $\n    -D_GLIBCXX_USE_CXX11_ABI=1" in fixture.ninja
    assert "-ccbin $\n" in fixture.ninja, "the flag ends a wrapped line"
    printed = fingerprint_of(module, fixture, source)
    assert printed["inputs"]["incomplete"] == [], printed["inputs"]["incomplete"]
    toolchain = printed["inputs"]["toolchain"]
    assert (
        toolchain[str(fixture.compiler.resolve())]
        == hashlib.sha256(fixture.compiler.read_bytes()).hexdigest()
    )
    # -ccbin sits on the line after its argument and was still followed, to a
    # compiler the cxx binding does not name.
    assert (
        toolchain[str(fixture.ccbin_compiler.resolve())]
        == hashlib.sha256(fixture.ccbin_compiler.read_bytes()).hexdigest()
    )
    assert not [value for value in printed["inputs"]["includes"] if "$" in value]
    assert all(
        value != "absent" for value in printed["inputs"]["includes"].values()
    ), printed["inputs"]["includes"]
    assert all(
        value != "absent" for value in printed["inputs"]["toolchain"].values()
    ), printed["inputs"]["toolchain"]
    # FLASHINFER_NVCC lets the build name any nvcc on the box, and the file says
    # which one: its bytes are an input even when they are not under CUDA_HOME.
    outside = tmp_path / "toolchain/nvcc-wrapper"
    outside.write_text("#!/bin/sh\necho nvcc one\n")
    outside.chmod(0o755)
    overridden = fixture.ninja.replace(
        "nvcc = $cuda_home/bin/nvcc", f"nvcc = {outside}"
    )
    second = module.build_fingerprint(overridden, source, fixture.inputs, fixture.abi)
    assert second["inputs"]["incomplete"] == [], second["inputs"]["incomplete"]
    assert (
        second["inputs"]["toolchain"][str(outside.resolve())]
        == hashlib.sha256(outside.read_bytes()).hexdigest()
    ), second["inputs"]["toolchain"]
    outside.write_text("#!/bin/sh\necho nvcc two\n")
    assert (
        module.build_fingerprint(overridden, source, fixture.inputs, fixture.abi)[
            "fingerprint"
        ]
        != second["fingerprint"]
    ), "the nvcc the build names is a build input"
    gone = module.build_fingerprint(
        overridden.replace(str(outside), str(tmp_path / "absent-nvcc")),
        source,
        fixture.inputs,
        fixture.abi,
    )
    assert any("nvcc" in gap for gap in gone["inputs"]["incomplete"]), gone["inputs"]
    # The container sets CXX and CUDAHOSTCXX to the same compiler; naming it twice
    # must not double-count it, and must not lose it either.
    same = printed["inputs"]["toolchain"]
    both = module.build_fingerprint(
        fixture.ninja.replace(str(fixture.ccbin_compiler), str(fixture.compiler)),
        source,
        fixture.inputs,
        fixture.abi,
    )
    assert both["inputs"]["incomplete"] == [], both["inputs"]["incomplete"]
    assert (
        both["inputs"]["toolchain"][str(fixture.compiler.resolve())]
        == same[str(fixture.compiler.resolve())]
    )
    assert both["fingerprint"] != printed["fingerprint"], "the flag text is an input"
    assert re.search(r"^[0-9a-f]{64}$", printed["fingerprint"])


def test_the_build_fingerprint_binds_the_actual_build_inputs(tmp_path):
    module = installer()
    root = tmp_path
    fixture = build_fixture(root)
    source = synthetic_source(root)
    installer_copy = root / "install-copy.py"
    installer_copy.write_bytes(INSTALLER.read_bytes())
    module.INSTALLER_PATH = installer_copy

    def fingerprint(ninja=None):
        return fingerprint_of(module, fixture, source, ninja)["fingerprint"]

    base = fingerprint()
    assert re.fullmatch(r"[0-9a-f]{64}", base), base

    def moved(path: Path, payload: bytes) -> None:
        before = path.read_bytes()
        path.write_bytes(payload)
        try:
            assert fingerprint() != base, f"{path} is a build input"
        finally:
            path.write_bytes(before)

    # The CUDA toolchain by content: nvcc, libdevice and the headers the build is
    # given, whatever nvcc --version still says about itself.
    moved(fixture.nvcc, b"nvcc image two\n")
    moved(fixture.libdevice, b"libdevice two\n")
    moved(fixture.cuda_header, b"typedef int two;\n")
    # The host toolchain by content, including what the driver only names.
    moved(fixture.compiler, b"#!/bin/sh\necho two\n")
    moved(fixture.cc1plus, b"cc1plus two\n")
    moved(fixture.collect2, b"collect2 two\n")
    moved(fixture.assembler, b"as two\n")
    moved(fixture.ld, b"ld two\n")
    moved(fixture.crt, b"crt two\n")
    moved(fixture.libgcc, b"libgcc two\n")
    moved(fixture.libc, b"glibc two\n")
    moved(fixture.libstdcxx, b"libstdc++ two\n")
    moved(fixture.cxx_include, b"#pragma once\n// two\n")
    # The kernel sources and headers, and the instantiation the generator wrote.
    moved(fixture.header, b"#pragma once\n// two\n")
    moved(fixture.csrc, b"#pragma once\n// two\n")
    moved(fixture.sources[0], b"// patched two\n")
    moved(fixture.generated, b"// generated two\n")
    # What the link step resolves -lNAME to, shared and archive alike.
    moved(fixture.cudart, b"cudart two\n")
    moved(fixture.cudadevrt, b"cudadevrt two\n")
    moved(fixture.libcuda, b"libcuda two\n")
    # The ABI of what the module is built into and loaded by.
    moved(fixture.python_header, b"typedef int two;\n")
    moved(fixture.ffi_header, b"#pragma once\n// two\n")
    for label in ("python", "torch", "tvm_ffi/libtvm_ffi.so"):
        moved(fixture.abi[label], b"another image\n")
    # The accepted patch set and this installer are identities too.
    moved(Path(source["patches"][0]["patch"]), b"a different mailbox\n")
    moved(installer_copy, b"a different installer\n")
    source["patches"][0]["files"][0]["accepted_sha256"] = "0" * 64
    assert fingerprint() != base, "the accepted manifest is a build input"
    source["patches"][0]["files"][0]["accepted_sha256"] = digest(
        "# stock kernel\nreturn 2\n"
    )
    # The effective flags, however the generator chose to break their lines.
    for label, before, after in (
        ("arch", "compute_120f", "compute_103f"),
        ("nvcc threads", "--threads=1", "--threads=4"),
        ("libstdc++ abi", "_GLIBCXX_USE_CXX11_ABI=1", "_GLIBCXX_USE_CXX11_ABI=0"),
        ("link set", "-lcudadevrt", ""),
        ("include set", "-isystem $cuda_home/include", ""),
        ("cuda home", f"cuda_home = {root / 'cuda'}", f"cuda_home = {root}"),
    ):
        assert fingerprint(fixture.ninja.replace(before, after)) != base, label

    # And what must not move it: an API-only release has no say here.
    (root / "flashinfer-include/__pycache__").mkdir(exist_ok=True)
    (root / "flashinfer-include/__pycache__/x.cpython-312.pyc").write_bytes(b"stale")
    aot = root / "flashinfer/data/aot/fused_moe_120"
    aot.mkdir(parents=True, exist_ok=True)
    (aot / "fused_moe_120.so").write_bytes(b"the installed module is an output\n")
    api = root / "python/sglang/srt/models/another_api.py"
    api.parent.mkdir(parents=True, exist_ok=True)
    api.write_text("# api only\n")
    os.environ["SOURCE_REVISION"] = "b" * 40
    try:
        assert fingerprint() == base, "the source revision is not a build input"
    finally:
        del os.environ["SOURCE_REVISION"]
    assert "SOURCE_REVISION" not in INSTALLER.read_text(), "no revision here at all"


def test_the_fingerprint_ignores_where_the_build_scratch_lives(tmp_path):
    """The same build in another temporary directory is the same build.

    The uncached path builds inside a fresh TemporaryDirectory, so a workspace path
    leaking into the digest would make every documented native --build-cache run
    miss, and would tie a container artifact to the one scratch directory that made
    it. Everything under the workspace is still hashed; only its location is not.
    """
    module = installer()
    source = synthetic_source(tmp_path)
    fixture = build_fixture(tmp_path)
    base = fingerprint_of(module, fixture, source)["fingerprint"]
    moved = build_fixture(tmp_path, workspace=tmp_path / "workspace-b")
    assert moved.generated != fixture.generated
    assert (
        fingerprint_of(module, moved, source)["fingerprint"] == base
    ), "the workspace location is not a build input"
    moved.generated.write_bytes(b"// generated two\n")
    assert fingerprint_of(module, moved, source)["fingerprint"] != base


def test_an_input_the_fingerprint_cannot_see_blocks_reuse(tmp_path):
    """A gap in the identity is reported and compiles; it is never a silent hit."""
    module = installer()
    fixture = build_fixture(tmp_path)
    source = synthetic_source(tmp_path)
    cache = tmp_path / "build-cache"
    report = io.StringIO()
    reused(module, fixture, source, cache, tmp_path / "run-one", report)
    assert cache_keys(cache)

    shutil.rmtree(fixture.nvcc.parent)
    gaps = fingerprint_of(module, fixture, source)["inputs"]["incomplete"]
    assert any("cuda/bin" in gap for gap in gaps), gaps
    _, _, state, compiled, said = reused(
        module, fixture, source, cache, tmp_path / "run-two", report
    )
    assert state == "compiled"
    assert "not fully identifiable" in said, said
    assert compiled and not compiled[0][1], "a stale output was republished"
    fixture.nvcc.parent.mkdir(parents=True, exist_ok=True)
    fixture.nvcc.write_bytes(b"nvcc image one\n")

    # An unplaceable tool is a gap, not a fact: the driver's own answer is all the
    # fingerprint has to go on, and it does not get to pretend otherwise.
    blind = tmp_path / "compiler-without-cc1plus"
    blind.write_text(
        fixture.compiler.read_text().replace(
            "-print-prog-name=cc1plus", "-print-prog-name=cc1plusX"
        )
    )
    blind.chmod(0o755)
    blind_ninja = fixture.ninja.replace(str(fixture.compiler), str(blind))
    gaps = module.build_fingerprint(blind_ninja, source, fixture.inputs, fixture.abi)[
        "inputs"
    ]["incomplete"]
    assert any("cc1plus" in gap for gap in gaps), gaps
    fixture.ninja = blind_ninja
    _, _, state, _, said = reused(
        module, fixture, source, cache, tmp_path / "run-three", report
    )
    assert state == "compiled" and "not fully identifiable" in said, said


def test_the_same_inputs_reuse_the_compiled_module_without_the_compile(tmp_path):
    module = installer()
    fixture = build_fixture(tmp_path)
    source = synthetic_source(tmp_path)
    cache = tmp_path / "build-cache"
    report = io.StringIO()

    _, built, state, compiled, _ = reused(
        module, fixture, source, cache, tmp_path / "run-one", report
    )
    assert state == "compiled"
    assert compiled == [("fused_moe_120", False, False)], "the first build reaches nvcc"
    assert built.read_bytes() == COMPILED
    fingerprint = (cache / CACHE_LATEST).read_text().strip()
    directory = cache / CACHE_SCHEMA / fingerprint
    # The artifact is not a .so: the repository's .dockerignore excludes *.so from
    # every build context, and a context filter that drops the binary while leaving
    # the records behind is exactly how a cache would start lying.
    assert (directory / ("fused_moe_120" + CACHE_SUFFIX)).read_bytes() == COMPILED
    assert not list(directory.glob("*.so")), list(directory.iterdir())
    record = json.loads((directory / CACHE_RECORD).read_text())
    assert record["schema"] == CACHE_SCHEMA and record["module"] == "fused_moe_120"
    assert record["fingerprint"] == fingerprint
    assert record["artifact_sha256"] == hashlib.sha256(COMPILED).hexdigest()
    assert record["artifact_bytes"] == len(COMPILED)
    assert record["inputs"]["toolchain"], "the inputs travel with the artifact"

    # A second, separate build environment with the same actual inputs: the compile
    # step is not entered, and the module is the recorded bytes.
    spec, built, state, compiled, said = reused(
        module, fixture, source, cache, tmp_path / "run-two", report
    )
    assert state == "restored", said
    assert compiled == [], "a cache hit must not enter the compile step"
    assert built.read_bytes() == COMPILED and spec.jit_library_path.is_file()

    # A real change to one input comes back as a compile and names the class of
    # input that moved, so a 90-minute rebuild is explainable from the log alone.
    fixture.generated.write_bytes(b"// generated two\n")
    fixture.inputs = {**fixture.inputs, "sources": [str(p) for p in fixture.sources]}
    _, _, state, compiled, said = reused(
        module, fixture, source, cache, tmp_path / "run-three", report
    )
    assert state == "compiled", said
    assert "sources" in said, said
    assert cache_keys(cache) == {
        fingerprint,
        (cache / CACHE_LATEST).read_text().strip(),
    }

    # Same paths, different bytes: the objects a previous compiler left behind must
    # not be relinked and published under the fingerprint of the new one.
    compiler_bytes = fixture.compiler.read_bytes()
    rebuilt = b"compiled by the new compiler\n"
    # A rebuilt toolchain package: the same path, working, different bytes.
    fixture.compiler.write_bytes(compiler_bytes + b"# rebuilt\n")
    _, _, state, compiled, said = reused(
        module,
        fixture,
        source,
        cache,
        tmp_path / "run-three-b",
        report,
        payload=rebuilt,
    )
    assert state == "compiled", said
    assert "toolchain" in said, said
    assert compiled == [
        ("fused_moe_120", False, False)
    ], "a fingerprint miss left Ninja intermediates in place: " + str(compiled)
    republished = (cache / CACHE_LATEST).read_text().strip()
    record = json.loads((cache / CACHE_SCHEMA / republished / CACHE_RECORD).read_text())
    assert record["artifact_sha256"] == hashlib.sha256(rebuilt).hexdigest(), record
    fixture.compiler.write_bytes(compiler_bytes)

    fixture.nvcc.write_bytes(b"nvcc image three\n")
    _, _, state, _, said = reused(
        module, fixture, source, cache, tmp_path / "run-four", report
    )
    assert state == "compiled", said
    assert "toolchain" in said, said
    assert len(cache_keys(cache)) == 2, cache_keys(cache)
    assert fingerprint not in cache_keys(cache), "the superseded artifact is pruned"
    assert (cache / CACHE_LATEST).read_text().strip() in cache_keys(cache)


def test_a_corrupt_or_foreign_cache_entry_is_never_reused(tmp_path):
    module = installer()
    fixture = build_fixture(tmp_path)
    source = synthetic_source(tmp_path)
    cache = tmp_path / "build-cache"
    report = io.StringIO()
    _, _, state, _, _ = reused(
        module, fixture, source, cache, tmp_path / "run-one", report
    )
    assert state == "compiled"
    directory = cache / CACHE_SCHEMA / (cache / CACHE_LATEST).read_text().strip()
    artifact = directory / ("fused_moe_120" + CACHE_SUFFIX)
    record_path = directory / CACHE_RECORD
    stored = json.loads(record_path.read_text())

    def refuses(description, break_cache):
        break_cache()
        report.seek(0)
        report.truncate()
        _, _, state, compiled, said = reused(
            module, fixture, source, cache, tmp_path / "again", report
        )
        assert state == "compiled", f"{description}: {said}"
        assert compiled, description
        assert said.strip(), f"{description}: said nothing"
        # The rejection heals itself: the rebuild republishes the artifact.
        _, _, state, compiled, said = reused(
            module, fixture, source, cache, tmp_path / "after", report
        )
        assert state == "restored", f"{description}: {said}"
        assert compiled == [], description
        stored.update(json.loads(record_path.read_text()))

    refuses("a truncated artifact", lambda: artifact.write_bytes(COMPILED[:-4]))
    refuses("an empty artifact", lambda: artifact.write_bytes(b""))
    refuses("a vanished artifact", lambda: artifact.unlink())
    refuses("no record at all", lambda: record_path.unlink(missing_ok=True))
    refuses("an unreadable record", lambda: record_path.write_text("{"))
    for malformed in ("[]", "null", '"just a string"', "[[]]", "{}", '{"inputs": []}'):
        refuses(
            f"a record holding {malformed}",
            lambda malformed=malformed: record_path.write_text(malformed),
        )
    for field, value in (
        ("fingerprint", "f" * 64),
        ("module", "fused_moe_103"),
        ("schema", "v0"),
        ("artifact_bytes", 1),
        ("artifact_sha256", "0" * 64),
    ):
        refuses(
            f"a record with {field}={value!r}",
            lambda field=field, value=value: record_path.write_text(
                json.dumps({**stored, field: value})
            ),
        )
    # The recorded inputs are bound to the fingerprint, so another build's inputs
    # cannot be pasted onto this record to make a stale entry look right.
    refuses(
        "inputs that are not this build's",
        lambda: record_path.write_text(
            json.dumps({**stored, "inputs": {**stored["inputs"], "ninja": "0" * 64}})
        ),
    )


def test_the_uncached_build_does_no_fingerprinting(tmp_path):
    """--build-cache is an addition; without it the step is what it always was."""
    module = installer()
    fixture = build_fixture(tmp_path)
    source = synthetic_source(tmp_path)
    compiled: list = []
    spec = FakeSpec(tmp_path / "workspace", fixture.ninja, fixture.sources, compiled)
    spec.build_dir.mkdir(parents=True, exist_ok=True)
    spec.jit_library_path.write_bytes(b"left over from a previous run\n")
    original = module.build_fingerprint

    def no_fingerprinting(*args, **kwargs):
        raise AssertionError("the uncached build fingerprinted something")

    module.build_fingerprint = no_fingerprinting
    report = io.StringIO()
    try:
        with redirect_stderr(report):
            built, state = module.build_or_reuse(spec, source)
    finally:
        module.build_fingerprint = original
    assert (state, compiled) == ("compiled", [("fused_moe_120", True, True)])
    assert built.read_bytes() == COMPILED
    assert report.getvalue() == "", report.getvalue()


def test_a_miss_cannot_publish_when_old_objects_cannot_be_removed(
    tmp_path, monkeypatch
):
    module = installer()
    fixture = build_fixture(tmp_path)
    source = synthetic_source(tmp_path)
    compiled: list = []
    spec = FakeSpec(tmp_path / "workspace", fixture.ninja, fixture.sources, compiled)

    def refuse_cleanup(path):
        raise PermissionError("cannot remove old objects")

    monkeypatch.setattr(module.shutil, "rmtree", refuse_cleanup)
    with pytest.raises(PermissionError, match="cannot remove old objects"):
        module.build_or_reuse(
            spec, source, tmp_path / "cache", fixture.inputs, fixture.abi
        )
    assert compiled == [], "a failed cleanup must not relink stale objects"
    assert not (tmp_path / "cache" / CACHE_LATEST).exists()


def test_the_build_subprocess_is_told_the_installer_and_the_cache(tmp_path):
    module = installer()
    source = source_manifest()
    env = module.nvcc_environment(source, tmp_path / "flashinfer", {})
    assert env["PENNY_FLASHINFER_INSTALLER"] == str(INSTALLER.resolve()), env
    assert "PENNY_FLASHINFER_BUILD_CACHE" not in env, "no cache, no key"

    seen = {}
    built = tmp_path / "jit/fused_moe_120/fused_moe_120.so"
    built.parent.mkdir(parents=True)
    built.write_bytes(b"nvcc made this\n")
    original_run = module.subprocess.run

    def recording_run(argv, **kwargs):
        seen.update(kwargs["env"])
        return subprocess.CompletedProcess(
            argv, 0, stdout=f"compiled {built}\n", stderr=""
        )

    module.subprocess.run = recording_run
    try:
        package = installed_tree(
            tmp_path / "site",
            {"kernel.py": "# stock kernel\nreturn 2\n"},
            "9.9.9",
            guard=GUARD_ACCEPTED,
        )
        destination, state = module.install_accepted_source(
            package, synthetic_source(tmp_path), tmp_path / "build-cache"
        )
    finally:
        module.subprocess.run = original_run  # the module object is the shared one
    assert seen["PENNY_FLASHINFER_BUILD_CACHE"] == str(tmp_path / "build-cache"), seen
    assert state == "compiled", state
    assert destination.read_bytes() == built.read_bytes()
    text = INSTALLER.read_text()
    assert "--build-cache" in text
    assert (
        "PENNY_FLASHINFER_BUILD_CACHE" in text
    ), "the environment is the container's knob"


def test_the_compiler_queries_are_the_ones_a_real_toolchain_answers(tmp_path):
    """The fixture script covers the reading; this covers a real compiler.

    A CPU runner with no C++ compiler has nothing to answer here and the step asks
    for nothing; on a host that has one, every query the fingerprint relies on has
    to resolve, because an unresolved answer blocks reuse instead of counting as a
    fact about the toolchain.
    """
    module = installer(shared_libraries=True)
    cxx = shutil.which(os.environ.get("CXX") or "c++")
    if cxx is None:
        return
    resolved = Path(cxx).resolve()
    directories, answers, search = module._driver_queries(resolved)
    assert directories and all(
        Path(entry).is_dir() for entry in directories
    ), directories
    assert set(answers) == {
        *module.COMPILER_PROGRAMS,
        *module.COMPILER_LIBRARIES,
    }, answers
    assert search and all(Path(entry).is_absolute() for entry in search), search
    incomplete: list = []
    inputs = module._compiler_inputs(resolved, incomplete)
    assert incomplete == [], incomplete
    assert str(resolved) in inputs
    assert any(key.startswith("include:") for key in inputs), inputs
    assert any(key.startswith("install:") for key in inputs), inputs
    for name, answer in sorted(answers.items()):
        placed = module._resolved_answer(answer, search, resolved.parent)
        assert placed is not None and str(placed) in inputs, (
            name,
            answer,
            sorted(inputs),
        )


def test_the_compiler_identity_includes_its_loaded_shared_libraries():
    module = installer(shared_libraries=True)
    cxx = shutil.which(os.environ.get("CXX") or "c++")
    if cxx is None or shutil.which("ldd") is None:
        pytest.skip("a C++ compiler and ldd are required")
    compiler = Path(cxx).resolve()
    _, answers, search = module._driver_queries(compiler)
    frontend = module._resolved_answer(answers["cc1plus"], search, compiler.parent)
    assert frontend is not None
    listed = subprocess.run(["ldd", str(frontend)], capture_output=True, text=True)
    assert listed.returncode == 0, listed.stderr
    dependencies = {
        str(Path(match).resolve())
        for match in re.findall(r"(?:=>\s+|^\s*)(/\S+)\s+\(", listed.stdout, re.M)
    }
    assert dependencies, listed.stdout
    incomplete = []
    recorded = module._compiler_inputs(compiler, incomplete)
    assert incomplete == [], incomplete
    assert dependencies <= recorded.keys(), dependencies - recorded.keys()


def test_changed_compiler_library_bytes_change_identity(tmp_path):
    module = installer(shared_libraries=True)
    cc = shutil.which("cc")
    if cc is None or shutil.which("ldd") is None:
        pytest.skip("a C compiler and ldd are required")
    library = tmp_path / "libcompiler_input.so"
    executable = tmp_path / "compiler"
    subprocess.run(
        [cc, "-x", "c", "-shared", "-fPIC", "-o", str(library), "-"],
        input="int compiler_input(void) { return 1; }\n",
        text=True,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        [
            cc,
            "-x",
            "c",
            "-",
            f"-L{tmp_path}",
            "-lcompiler_input",
            f"-Wl,-rpath,{tmp_path}",
            "-o",
            str(executable),
        ],
        input="extern int compiler_input(void); int main(void) { return compiler_input(); }\n",
        text=True,
        capture_output=True,
        check=True,
    )
    gaps = []
    before = module._program_libraries({executable}, gaps)
    assert gaps == [], gaps
    assert str(library) in before
    program_digest = module.sha256_file(executable)
    with library.open("ab") as handle:
        handle.write(b"updated library contents\n")
    after = module._program_libraries({executable}, gaps)
    assert gaps == [], gaps
    assert before[str(library)] != after[str(library)]
    assert module.sha256_file(executable) == program_digest
    library.unlink()
    module._program_libraries({executable}, gaps)
    assert any("not found" in gap for gap in gaps), gaps


def test_unresolved_toolchain_library_scan_prevents_reuse(tmp_path, monkeypatch):
    module = installer(shared_libraries=True)
    program = tmp_path / "compiler"
    program.write_bytes(b"\x7fELF fixture")

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("ldd unavailable")

    monkeypatch.setattr(module.subprocess, "run", unavailable)
    gaps = []
    assert module._program_libraries({program}, gaps) == {}
    assert any("ldd unavailable" in gap for gap in gaps)
    program.write_text("#!/bin/sh\nexec arbitrary-compiler\n")
    gaps = []
    assert module._program_libraries({program}, gaps) == {}
    assert any("non-ELF" in gap for gap in gaps)


def test_the_abi_inputs_are_the_installed_binaries_and_not_their_versions(tmp_path):
    """The container resolves torch and tvm-ffi for real; here they are stood in.

    The fingerprint cannot be honest about the ABI it binds the module to without
    those binaries, and an environment that cannot name one gets no reuse rather
    than a thinner fingerprint.
    """
    module = installer()
    fixture = build_fixture(tmp_path)
    source = synthetic_source(tmp_path)
    site = tmp_path / "abi-site"
    extension = site / "torch/lib/libtorch_python.so"
    extension.parent.mkdir(parents=True)
    (site / "torch/__init__.py").write_text("")
    extension.write_bytes(b"libtorch one\n")
    library = site / "tvm_ffi/lib/libtvm_ffi.so"
    library.parent.mkdir(parents=True)
    (site / "tvm_ffi/__init__.py").write_text("")
    library.write_bytes(b"tvm ffi one\n")
    # torch._C without __file__ is the older layout; the fallback is the same file.
    installed = {
        "torch": SimpleNamespace(
            _C=SimpleNamespace(), __file__=str(site / "torch/__init__.py")
        ),
        "tvm_ffi": SimpleNamespace(__file__=str(site / "tvm_ffi/__init__.py")),
    }
    saved = {name: module.sys.modules.get(name) for name in installed}
    module.sys.modules.update(installed)

    def fingerprint():
        return module.build_fingerprint(fixture.ninja, source, fixture.inputs)[
            "fingerprint"
        ]

    try:
        found = module.abi_dependency_files()
        assert found["python"] == module.Path(module.sys.executable).resolve(), found
        assert found["torch"] == extension.resolve(), found
        assert found["tvm_ffi/lib/libtvm_ffi.so"] == library.resolve(), found
        base = fingerprint()
        assert not fingerprint_of(module, fixture, source)["inputs"]["incomplete"]
        library.write_bytes(b"tvm ffi two\n")
        assert fingerprint() != base, "the loader that hosts the module is an input"
        library.write_bytes(b"tvm ffi one\n")
        extension.write_bytes(b"libtorch two\n")
        assert fingerprint() != base, "the torch ABI answer is an input"
        extension.write_bytes(b"libtorch one\n")
        library.unlink()
        raises(lambda: module.abi_dependency_files(), "no shared library")
        module.sys.modules["torch"] = None
        gaps = module.build_fingerprint(fixture.ninja, source, fixture.inputs)[
            "inputs"
        ]["incomplete"]
        assert any("abi" in gap for gap in gaps), gaps
    finally:
        for name, previous in saved.items():
            if previous is None:
                module.sys.modules.pop(name, None)
            else:
                module.sys.modules[name] = previous


def test_the_container_restores_the_module_before_it_can_recompile():
    dockerfile = DOCKERFILE.read_text()
    restore = dockerfile.index("COPY docker/pennyroyal/build-cache/")
    packaging = dockerfile.index("scripts/pennyroyal/flashinfer/install.py")
    export = dockerfile.index("FROM scratch AS flashinfer-artifact")
    assert restore < packaging < export, "restore, then build, then export"
    assert "PENNY_FLASHINFER_BUILD_CACHE=/tmp/penny-fi-cache" in dockerfile
    assert "COPY --from=build /tmp/penny-fi-cache/ /cache/" in dockerfile[export:]
    # Only the fingerprinted modules travel; the ninja objects still do not.
    assert dockerfile.count("/tmp/penny-fi-cache") == 3, dockerfile.count(
        "/tmp/penny-fi-cache"
    )
    assert "rm -rf /tmp/penny-fi-sm120" in dockerfile
    assert (REPO / "docker/pennyroyal/build-cache/.gitignore").is_file()


REAL_SPEC_PROBE = r"""
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

root, workspace, installer, manifest = map(pathlib.Path, sys.argv[1:5])
cuda = root / "cuda"
(cuda / "bin").mkdir(parents=True, exist_ok=True)
(cuda / "include").mkdir(exist_ok=True)
(cuda / "nvvm").mkdir(exist_ok=True)
nvcc = cuda / "bin" / "nvcc"
if not nvcc.exists():
    # A tiny CPU executable answers the generator's version query. No CUDA is
    # compiled, and the dependency scan can examine a real ELF executable.
    subprocess.run(
        [os.environ.get("CXX", "c++"), "-x", "c++", "-o", str(nvcc), "-"],
        input='#include <stdio.h>\nint main() { puts("Cuda compilation tools, release 13.0, V13.0.48"); }\n',
        text=True, capture_output=True, check=True,
    )
os.environ["CUDA_HOME"] = str(cuda)
os.environ["FLASHINFER_WORKSPACE_BASE"] = str(workspace)
source = json.loads(manifest.read_text())
os.environ["FLASHINFER_CUDA_ARCH_LIST"] = source["cuda_arch_list"]
try:
    from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module
except ImportError as error:
    print(json.dumps({"unavailable": str(error)}))
    raise SystemExit(0)

spec = gen_cutlass_fused_moe_sm120_module(use_fast_build=False)


def never_build(*args, **kwargs):
    raise AssertionError("the fingerprint compiled")


spec.build = never_build
spec.write_ninja()
ninja = pathlib.Path(spec.ninja_path).read_text()
for name in re.findall(r"(?:^|\s)-l(\S+)", ninja):
    for directory in re.findall(r"(?:^|\s)-L(\S+)", ninja):
        target = pathlib.Path(directory.replace("$cuda_home", str(cuda)))
        # Only the fixture CUDA tree may be stubbed, never an installed directory.
        assert target.resolve().is_relative_to(cuda.resolve()), target
        target.mkdir(parents=True, exist_ok=True)
        (target / ("lib" + name + ".so")).write_bytes(b"stub\n")
loader = importlib.util.spec_from_file_location("penny_flashinfer_sm120", installer)
step = importlib.util.module_from_spec(loader)
loader.loader.exec_module(step)
printed = step.build_fingerprint(
    ninja, source, step.build_inputs(spec, os.environ), step.abi_dependency_files()
)
print(
    json.dumps(
        {
            "fingerprint": printed["fingerprint"],
            "incomplete": printed["inputs"]["incomplete"],
            "includes": list(printed["inputs"]["includes"]),
            "sources": len(spec.sources),
            "ninja": str(spec.ninja_path),
            "workspace": str(workspace),
            "cuda": str(cuda),
        }
    )
)
"""


def test_the_build_context_actually_carries_the_build_cache(tmp_path):
    """The cache is restored through the build context, so the context decides.

    The repository .dockerignore excludes *.so, which is why this is built rather
    than asserted about: a scratch image out of a cache directory holding the step's
    real file names shows what the engine really sent, under the ignore files this
    repository really has. Only a box with no working engine skips; once the engine
    answers, a broken build, create or copy is a failure. No Penny Royal image is
    built here.
    """
    engine = next((tool for tool in ("podman", "docker") if shutil.which(tool)), None)
    if engine is None:
        pytest.skip("no container engine on this box")
    context = tmp_path / "context"
    cache = context / "docker/pennyroyal/build-cache"
    payload = cache / CACHE_SCHEMA / ("f" * 64)
    payload.mkdir(parents=True)
    (payload / ("fused_moe_120" + CACHE_SUFFIX)).write_bytes(b"artifact\n")
    (payload / CACHE_RECORD).write_text("{}\n")
    (cache / CACHE_LATEST).write_text("f" * 64 + "\n")
    (payload / "decoy.so").write_bytes(b"artifact\n")
    shutil.copyfile(REPO / ".dockerignore", context / ".dockerignore")
    shutil.copyfile(
        REPO / "docker/pennyroyal/Dockerfile.dockerignore",
        context / "docker/pennyroyal/Dockerfile.dockerignore",
    )
    # Named Dockerfile, so the adjacent Dockerfile.dockerignore really is the file
    # the engine looks for beside it. The artifact stage of the image puts the cache
    # under /cache, which is what the workflow copies out of a created container.
    (context / "docker/pennyroyal/Dockerfile").write_text(
        "FROM scratch\nCOPY docker/pennyroyal/build-cache/ /cache/\n"
    )
    image = f"pennyroyal-context-probe-{os.getpid()}"
    built = subprocess.run(
        [engine, "build", "-q", "-t", image, "-f", "docker/pennyroyal/Dockerfile", "."],
        cwd=context,
        capture_output=True,
        text=True,
    )
    if built.returncode != 0:
        reachable = subprocess.run([engine, "version"], capture_output=True)
        if reachable.returncode != 0:
            pytest.skip(f"{engine} cannot build here: {built.stderr.strip()[:200]}")
        raise AssertionError(
            f"{engine} is working but the probe build failed: {built.stderr}"
        )
    container = None
    try:
        # The artifact stage has no command, exactly like the real one, so a dummy
        # is required to create the container. It is never started.
        create = subprocess.run(
            [engine, "create", built.stdout.strip() or image, "true"],
            capture_output=True,
            text=True,
            check=True,
        )
        container = create.stdout.strip().splitlines()[-1]
        copied = subprocess.run(
            [engine, "cp", f"{container}:/cache/.", str(tmp_path / "extracted")],
            capture_output=True,
            text=True,
            check=True,
        )
        assert copied.returncode == 0
    finally:
        if container is not None:
            subprocess.run([engine, "rm", container], capture_output=True)
        subprocess.run([engine, "rmi", image], capture_output=True)
    sent = sorted(
        str(path.relative_to(tmp_path / "extracted"))
        for path in (tmp_path / "extracted").rglob("*")
        if path.is_file()
    )
    for name in (
        CACHE_LATEST,
        f"{CACHE_SCHEMA}/{'f' * 64}/{CACHE_RECORD}",
        f"{CACHE_SCHEMA}/{'f' * 64}/fused_moe_120{CACHE_SUFFIX}",
    ):
        assert name in sent, (engine, sent)


def test_a_real_flashinfer_spec_is_fingerprinted_without_compiling(tmp_path):
    """The generated spec of an installed FlashInfer, read without compiling.

    Both runs happen in a disposable subprocess with their own temporary workspace
    and their own CUDA tree, so nothing here can write into an installed directory
    or leave FlashInfer's environment changed for the rest of the suite, and the
    two fingerprints come from two processes that never shared an import. Writing
    the Ninja file is the whole interaction: a step that reached the build would
    fail the child.
    """
    root = tmp_path / "probe"
    root.mkdir()
    manifest = root / "accepted-sources.json"
    shutil.copyfile(MANIFEST, manifest)
    runs = []
    for name in ("workspace-one", "workspace-two"):
        workspace = tmp_path / name
        workspace.mkdir()
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                REAL_SPEC_PROBE,
                str(root),
                str(workspace),
                str(INSTALLER),
                str(manifest),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout.strip().splitlines()[-1])
        if "unavailable" in report:
            pytest.skip(f"flashinfer is not importable here: {report['unavailable']}")
        runs.append(report)
    first, second = runs
    assert first["sources"] > 10, runs
    assert first["incomplete"] == [], runs
    for report in runs:
        assert report["ninja"].startswith(report["workspace"]), report
        assert report["cuda"].startswith(str(root)), report
    assert first["ninja"] != second["ninja"], runs
    assert first["cuda"] == second["cuda"], runs
    assert first["fingerprint"] == second["fingerprint"], runs
    assert not [value for value in first["includes"] if "$" in value], runs


if __name__ == "__main__":  # pytest is the real driver; this is a smoke check
    for name, case in sorted(globals().items()):
        if not (name.startswith("test_") and inspect.isfunction(case)):
            continue
        with tempfile.TemporaryDirectory() as temp:
            case(Path(temp)) if inspect.signature(case).parameters else case()
        print("pass", name)
