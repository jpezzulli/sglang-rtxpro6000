#!/usr/bin/env python3
"""Run pip check for the deliberately SM120-only FlashInfer cache install.

The pinned shim discovers installed providers via entry points; it does not import
every architecture, and its metadata nevertheless requires all six. Only the five
providers this GPU family cannot load are forgiven, together with the recorded
Humming pair -- 0.1.12 asked by the sglang metadata, 0.1.10 in the qualified
environment, upgrade deferred, the package unused by these profiles. A wrong
version, a missing SM120f provider, a missing dependency of anything installed (the
retained Humming wheel needs pyelftools, which is why pyelftools is not one of the
excluded development tools), and every unrelated error still fail, because the
accepted findings are literal rather than parsed out of the requirements. The image
runs this twice: before the SM120 compile can be spent, and once its own NIXL build
has joined the environment.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parity():
    """The dependency baseline beside this step, loaded from this file's own tree."""
    spec = importlib.util.spec_from_file_location(
        "pennyroyal_dependency_parity", HERE.parent / "dependency_parity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def only_accepted_findings(result: subprocess.CompletedProcess) -> bool:
    """Is every line pip check wrote one the qualified baseline explains?"""
    return result.returncode == 1 and parity().accepted_pip_check_output(
        result.stdout, result.stderr, parity().load()
    )


def main() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        capture_output=True,
    )
    if only_accepted_findings(result):
        print(
            "Dependency check passed; the unused FlashInfer cache providers and the"
            " deferred Humming pin are the only findings."
        )
        return 0
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
