"""CPU-only image check; this does not claim model/GPU qualification."""

import importlib.metadata as metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


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


def main():
    import nixl
    import torch

    root = Path("/opt/pennyroyal")
    source = check_source(root, os.environ.get("PENNY_EXPECTED_SOURCE_REVISION", ""))
    expected = {
        "torch": "2.13.0+cu130",
        "torchvision": "0.28.0+cu130",
        "torchaudio": "2.11.0+cu130",
        "flashinfer-python": "0.6.17",
        "flashinfer-jit-cache": "0.6.17+cu130",
        "sglang-kernel": "0.4.6.post1+cu130",
        "triton": "3.7.1",
        "nixl": "1.4.0",
        "nixl-cu13": "1.4.0",
    }
    installed = {name: metadata.version(name) for name in expected}
    if installed != expected or torch.version.cuda != "13.0":
        raise RuntimeError(
            f"Unexpected CUDA package set: {installed}, CUDA={torch.version.cuda}"
        )
    import flashinfer_jit_cache

    moe_kernel = (
        Path(flashinfer_jit_cache.get_jit_cache_dir())
        / "fused_moe_120/fused_moe_120.so"
    )
    if not moe_kernel.is_file() or moe_kernel.stat().st_size == 0:
        raise RuntimeError("Prebuilt SM120 fused-MoE kernel is missing")
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
                "packages": installed,
                "posix_plugin": "available",
                "gpu_tested": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
