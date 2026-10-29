"""CPU-only image check; this does not claim model/GPU qualification."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def _step(root: Path, relative: str, name: str):
    """A packaging module of this checkout, loaded whatever sys.path says."""
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_source(root: Path, expected_revision: str = "") -> str:
    source = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if expected_revision and source != expected_revision:
        raise RuntimeError(
            f"Image revision {source} differs from requested {expected_revision}"
        )
    spec = importlib.util.find_spec("sglang")
    if (
        spec is None
        or not spec.origin
        or Path(spec.origin).resolve() != (root / "python/sglang/__init__.py").resolve()
    ):
        raise RuntimeError(f"SGLang import is not from the image source: {spec}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if dirty:
        raise RuntimeError(f"Image has modified tracked source: {dirty}")
    return source


def check_flashinfer_sm120(root: Path, stock_kernels: list[Path]) -> dict:
    """Require the accepted FlashInfer source and the package-local SM120 module.

    Stock flashinfer-python plus the stock provider wheels fail here: the
    packaging step must have patched the installed sources and installed the
    module it compiled from them. No GPU, no model and no FlashInfer import are
    involved.
    """
    spec = importlib.util.find_spec("flashinfer")
    locations = list(spec.submodule_search_locations or ()) if spec else []
    if not locations:
        raise RuntimeError("flashinfer is not importable from the image environment")
    packaging = importlib.util.spec_from_file_location(
        "penny_flashinfer_sm120",
        root / "scripts" / "pennyroyal" / "flashinfer" / "install.py",
    )
    installer = importlib.util.module_from_spec(packaging)
    packaging.loader.exec_module(installer)
    return installer.check_installed(
        Path(locations[0]).resolve(), installer.accepted_sources(), stock_kernels
    )


def check_baseline_parity(root: Path, source: str) -> dict:
    """Require the installed environment to be the qualified native baseline.

    The baseline is the one list of versions a release may install, so this
    reports drift instead of agreeing with a second dictionary of expectations:
    every shared dependency at its exact version including any local build tag,
    no native-only development tool, nothing unapproved, and Pennyroyal itself at
    the version derived from the revision this image was built from. No GPU is
    involved.
    """
    parity = _step(
        root, "scripts/pennyroyal/dependency_parity.py", "pennyroyal_dependency_parity"
    )
    manifest = parity.load()
    result = parity.check(manifest, parity.installed_versions(), source)
    problems = parity.report(result)
    if problems:
        raise RuntimeError("Dependency parity failed:\n  " + "\n  ".join(problems))
    return {
        "baseline": manifest["qualified"]["inspected"],
        "distributions": len(manifest["packages"]),
        "python": result["python"],
        "platform": result["platform"],
    }


def main():
    import nixl
    import torch

    root = Path("/opt/pennyroyal")
    source = check_source(root, os.environ.get("PENNY_EXPECTED_SOURCE_REVISION", ""))
    dependencies = check_baseline_parity(root, source)
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"Unexpected CUDA runtime: {torch.version.cuda}")
    import flashinfer_jit_cache

    # 0.7.0 splits the jit cache into arch-specific provider wheels behind a
    # shim; discover them through the provider entry points instead of the
    # removed get_jit_cache_dir().
    providers = flashinfer_jit_cache.get_jit_cache_providers()
    if not providers:
        raise RuntimeError("No FlashInfer jit-cache provider wheels are installed")
    moe_kernel = next(
        (
            kernel
            for provider in providers
            if (
                kernel := provider.jit_cache_dir / "fused_moe_120/fused_moe_120.so"
            ).is_file()
        ),
        None,
    )
    if moe_kernel is None or moe_kernel.stat().st_size == 0:
        raise RuntimeError("Prebuilt SM120 fused-MoE kernel is missing")
    # The image must carry the packaging step's own module, not the stock
    # provider one: the accepted sources and the package-local AOT file are
    # what the Next profiles actually load.
    sm120 = check_flashinfer_sm120(root, [moe_kernel])
    sys.path.insert(0, str(root / ".ple-nvme"))
    import sglang_ssd_stream._io  # noqa: F401

    for name in (
        "serve-flash-next-frspec.sh",
        "serve-flash-next.sh",
        "serve-qwen38-27b-dflash2.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(root / "configs/pennyroyal" / name)], check=True
        )
    agent = nixl.nixl_agent(
        "penny-container-check",
        nixl.nixl_agent_config(
            enable_prog_thread=False, enable_listen_thread=False, backends=["POSIX"]
        ),
    )
    if not {"FILE_SEG", "DRAM_SEG"} <= set(agent.get_backend_mem_types("POSIX")):
        raise RuntimeError("NIXL POSIX plugin is missing FILE/DRAM support")
    print(
        json.dumps(
            {
                "source": source,
                "dependencies": dependencies,
                "flashinfer_sm120": sm120,
                "posix_plugin": "available",
                "gpu_tested": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
