"""CPU-only checks for the unsupported flashinfer-cubin payload pruning.

No GPU, no container and no wheel download: the rules are exercised against a
synthetic installed flashinfer-cubin whose paths and RECORD sizes mirror the
pinned 0.7.0.post1 distribution (cubins/<40-hex>/batched_gemm-*/, gemm-*/ and
fmha/trtllm-gen/, with the sm100a/sm100f/sm103a/sm107a trtllm-gen kernels that
the pinned runners dispatch only on SM100/103/107). The unsupported families
are removed, every other recorded byte is retained, and the distribution
RECORD stays consistent for what was deleted.
"""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
PRUNER = REPO / "scripts" / "pennyroyal" / "flashinfer" / "prune_cubins.py"
INSTALLER = REPO / "scripts" / "pennyroyal" / "flashinfer" / "install.py"

HEX = "1d145b82ac60add55ea213863523f12d63005651"
HEX2 = "2d6a5a029eefcc388ec0ceb87efb55d8bcce5c3c"
BMM = f"flashinfer_cubin/cubins/{HEX}/batched_gemm-09795a1-31ee4e5"
GEMM = f"flashinfer_cubin/cubins/{HEX}/gemm-b738138-25754e6"
FMHA = f"flashinfer_cubin/cubins/{HEX2}/fmha/trtllm-gen"
DEEP = f"flashinfer_cubin/cubins/{HEX2}/deep-gemm"

# rel path -> payload size; mirrors the pinned RECORD layout: unsupported
# trtllm-gen families, plus retained deep-gemm kernels, checksums, common files
# and an RTX-named file that an arch-name sweep must never touch.
FAMILIES = {
    f"{BMM}/Bmm_a_swiGlu_dynB_sm100f.cubin": 64,
    f"{BMM}/Bmm_a_swiGlu_dynB_sm100f.cubin.lock": 0,
    f"{BMM}/Bmm_a_relu2_bN_sm107a.cubin": 32,
    f"{BMM}/Bmm_a_relu2_bN_sm107a.cubin.lock": 0,
    f"{BMM}/Bmm_a_bias_sm103a.cubin": 16,
    f"{BMM}/Bmm_keep_sm80.cubin": 8,
    f"{BMM}/Bmm_keep_sm80.cubin.lock": 0,
    f"{GEMM}/Gemm_bf16_sm107a.cubin": 32,
    f"{GEMM}/Gemm_bf16_sm107a.cubin.lock": 0,
    f"{GEMM}/Gemm_bf16_sm100f.cubin": 64,
    f"{FMHA}/fmhaSm100fKernel_QkvBfloat16H128ForGen.cubin": 128,
    f"{FMHA}/fmhaSm100fKernel_QkvBfloat16H128ForGen.cubin.lock": 0,
    f"{FMHA}/fmhaSm107aKernel_Qe4m3H128ForGen.cubin": 256,
    f"{FMHA}/fmhaSm100aKernel_QBfloat16H128ForGen.cubin": 96,
    f"{FMHA}/fmhaSm103aKernel_QBfloat16H128ForGen.cubin": 96,
    f"{FMHA}/fmhaSm120fKernel_QkvBfloat16H128ForGen.cubin": 48,
    f"{FMHA}/checksums.txt": 12,
    f"{DEEP}/kernel.fp8_m_grouped_gemm.007404769193.cubin": 24,
    "flashinfer_cubin/__init__.py": 4,
}
UNSUPPORTED = {
    path
    for path in FAMILIES
    if re.search(r"(_sm10[037][af]?\.cubin|fmhaSm10[037][af]?Kernel)", path)
}
RETAINED = set(FAMILIES) - UNSUPPORTED


def pruner():
    spec = importlib.util.spec_from_file_location("penny_flashinfer_cubin_prune", PRUNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def installed_cubin(site: Path, version: str = "0.7.0.post1") -> Path:
    """A stand-in site-packages holding one installed flashinfer-cubin."""
    for rel, size in FAMILIES.items():
        target = site / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * size)
    info = site / f"flashinfer_cubin-{version}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: flashinfer-cubin\nVersion: {version}\n"
    )
    lines = [
        f"flashinfer_cubin/__init__.py,sha256=AAAA,{FAMILIES['flashinfer_cubin/__init__.py']}",
        f"flashinfer_cubin-VERSION.dist-info/METADATA,sha256=BBBB,60",
        f"flashinfer_cubin-VERSION.dist-info/RECORD,,",
    ]
    for rel, size in sorted(FAMILIES.items()):
        if rel == "flashinfer_cubin/__init__.py":
            continue
        lines.append(f"{rel},sha256=dl{_record_digest(rel)},{size}")
    (info / "RECORD").write_text("\n".join(lines) + "\n")
    return info


def _record_digest(rel: str) -> str:
    return hashlib.sha256(rel.encode()).hexdigest()[:16]


def record_lines(site: Path) -> list[str]:
    info = next(site.glob("flashinfer_cubin-*.dist-info"))
    return (info / "RECORD").read_text().splitlines()


def raises(call, needle: str) -> None:
    try:
        call()
    except RuntimeError as error:
        assert needle in str(error), f"expected {needle!r} in {error}"
    else:
        raise AssertionError(f"nothing failed; expected {needle!r}")


def test_prune_removes_only_the_proven_unsupported_families(tmp_path):
    module = pruner()
    site = tmp_path / "site"
    installed_cubin(site)
    before = record_lines(site)

    summary = module.prune(site)
    assert summary["status"] == "pruned"
    assert summary["removed_files"] == len(UNSUPPORTED)
    assert summary["removed_bytes"] == sum(FAMILIES[p] for p in UNSUPPORTED)
    assert summary["record_entries_removed"] == len(UNSUPPORTED)
    for rel in UNSUPPORTED:
        assert not (site / rel).exists(), rel
    for rel in RETAINED:
        assert (site / rel).is_file(), rel
        assert (site / rel).read_bytes() == b"x" * FAMILIES[rel], rel

    # RECORD keeps every retained identity line byte-for-byte, including the
    # RETAINED cubin hashes and sizes, and lists only removed paths gone.
    after = record_lines(site)
    kept = [line for line in before if not any(line.startswith(p + ",") for p in UNSUPPORTED)]
    assert sorted(after) == sorted(kept)
    for rel in RETAINED:
        if rel == "flashinfer_cubin/__init__.py":
            continue
        assert any(line.startswith(rel + ",sha256=dl") for line in after), rel
    assert any(line.startswith("flashinfer_cubin/__init__.py,sha256=AAAA,") for line in after)


def test_a_repeat_prune_changes_nothing(tmp_path):
    module = pruner()
    site = tmp_path / "site"
    installed_cubin(site)
    module.prune(site)
    info = next(site.glob("flashinfer_cubin-*.dist-info"))
    record = (info / "RECORD").read_bytes()
    again = module.prune(site)
    assert again["status"] == "pruned"
    assert again["removed_files"] == 0 and again["removed_bytes"] == 0
    assert (info / "RECORD").read_bytes() == record


def test_check_is_read_only_until_the_payload_is_pruned(tmp_path):
    module = pruner()
    site = tmp_path / "site"
    installed_cubin(site)
    files_before = sorted(p for p in site.rglob("*") if p.is_file())
    raises(
        lambda: module.check(site),
        "unsupported trtllm-gen cubin payload is still present",
    )
    assert sorted(p for p in site.rglob("*") if p.is_file()) == files_before
    module.prune(site)
    assert module.check(site)["status"] == "pruned"


def test_a_site_without_the_package_is_not_an_error(tmp_path):
    module = pruner()
    site = tmp_path / "site"
    site.mkdir()
    assert module.prune(site)["status"] == "absent"
    assert module.check(site)["status"] == "absent"


def test_an_unpinned_cubin_version_fails_before_any_removal(tmp_path):
    module = pruner()
    site = tmp_path / "site"
    installed_cubin(site, version="0.6.17")
    raises(lambda: module.prune(site), "not the pinned 0.7.0.post1")
    for rel in UNSUPPORTED:
        assert (site / rel).is_file(), rel


def test_tampered_and_unexpected_layouts_fail_before_any_removal(tmp_path):
    module = pruner()

    # A candidate whose bytes no longer match the RECORD size.
    site = tmp_path / "sizes"
    installed_cubin(site)
    (site / f"{BMM}/Bmm_a_swiGlu_dynB_sm100f.cubin").write_bytes(b"x" * 999)
    raises(lambda: module.prune(site), "size does not match its RECORD entry")
    for rel in UNSUPPORTED:
        assert (site / rel).exists(), rel

    # A candidate on disk that the RECORD does not know: an unknown file is
    # never deleted and never silently left inconsistent either.
    site = tmp_path / "extra"
    installed_cubin(site)
    extra = site / f"{GEMM}/Gemm_unknown_sm103a.cubin"
    extra.write_bytes(b"x" * 64)
    raises(lambda: module.prune(site), "not recorded in RECORD")
    assert extra.is_file()
    for rel in UNSUPPORTED:
        assert (site / rel).exists(), rel

    # A candidate that is a symlink: the tool never follows links out.
    site = tmp_path / "linked"
    installed_cubin(site)
    target = site / f"{GEMM}/Gemm_bf16_sm100f.cubin"
    payload = tmp_path / "elsewhere.cubin"
    payload.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(payload)
    raises(lambda: module.prune(site), "is a symlink")
    assert target.is_symlink()


def test_a_symlinked_payload_directory_cannot_lead_unlinkings_outside(tmp_path):
    """os.walk skips symlinked directories but unlink follows them: the whole
    directory chain of every recorded candidate is guarded, not just leaves."""
    module = pruner()
    site = tmp_path / "site"
    installed_cubin(site)
    before = record_lines(site)
    cubins = site / "flashinfer_cubin" / "cubins"
    external = tmp_path / "external"
    cubins.rename(external)
    cubins.symlink_to(external)
    raises(lambda: module.prune(site), "is a symlink")
    # The external payload and the RECORD are exactly as they were: nothing
    # outside the install was deleted and nothing inconsistent was recorded.
    assert (external / f"{HEX}/batched_gemm-09795a1-31ee4e5/"
            "Bmm_a_swiGlu_dynB_sm100f.cubin").is_file()
    assert record_lines(site) == before


def test_a_recorded_candidate_that_is_not_a_file_fails_closed(tmp_path):
    module = pruner()
    site = tmp_path / "site"
    installed_cubin(site)
    before = record_lines(site)
    target = site / f"{BMM}/Bmm_a_relu2_bN_sm107a.cubin"
    target.unlink()
    target.mkdir()
    raises(lambda: module.prune(site), "not a regular file")
    assert record_lines(site) == before
    for rel in UNSUPPORTED:
        if rel != f"{BMM}/Bmm_a_relu2_bN_sm107a.cubin":
            assert (site / rel).exists(), rel


def test_the_packaging_step_prunes_and_checks_the_payload():
    text = INSTALLER.read_text()
    assert "prune_cubins.py" in text
    assert "prune(package.parent)" in text
    assert "check(package.parent)" in text
    # --apply-only stays the source half: the pruning call is behind the build
    # guard, so a source-only application never touches the cubin payload.
    step = text[text.index("def main("):]
    assert step.index("if not args.apply_only:") < step.index("prune(package.parent)")


def _run_cli(site: Path, argv: list[str]) -> str:
    out = subprocess.run(
        [sys.executable, str(PRUNER), "--site", str(site), *argv],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise AssertionError(out.stdout + out.stderr)
    return out.stdout


def test_cli_apply_then_check_exit_clean(tmp_path):
    site = tmp_path / "site"
    installed_cubin(site)
    summary = json.loads(_run_cli(site, []))
    assert summary["status"] == "pruned" and summary["removed_files"] == len(UNSUPPORTED)
    assert json.loads(_run_cli(site, ["--check"]))["status"] == "pruned"


def test_the_pinned_unsupported_families_are_the_whole_trtllm_payload():
    """The two runner dispatchers prove sm100/103/107 names and nothing else.

    Guards the shipped pattern against a fixture that quietly diverges from
    the pinned distribution: every unsupported fixture name must be one of
    the four datacenter-only trtllm-gen families.
    """
    assert len(UNSUPPORTED) == 13
    for rel in UNSUPPORTED:
        assert re.search(r"[sS]m10[037][af]?(?![0-9])", rel), rel
