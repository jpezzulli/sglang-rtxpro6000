#!/usr/bin/env python3
"""Run pip check for the deliberately SM120-only FlashInfer cache install."""

import subprocess
import sys

# The pinned shim discovers installed providers via entry points; it does not
# import every architecture. Its wheel metadata nevertheless requires all six.
# Keep that shim and the SM120f provider without pulling unused GPU binaries.
# Only these exact missing-provider messages are exempt: wrong versions, a
# missing SM120f provider, and every unrelated dependency error still fail.
PIN = "0.7.0.post1+cu130"
OMITTED_PROVIDERS = ("sm100a", "sm103a", "sm80", "sm89", "sm90a")


def only_omitted_providers(result: subprocess.CompletedProcess) -> bool:
    expected = {
        f"flashinfer-jit-cache {PIN} requires flashinfer-jit-cache-{arch}, "
        "which is not installed."
        for arch in OMITTED_PROVIDERS
    }
    lines = result.stdout.strip().splitlines()
    return (
        result.returncode == 1
        and not result.stderr.strip()
        and bool(lines)
        and all(line in expected for line in lines)
    )


def main() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        capture_output=True,
    )
    if only_omitted_providers(result):
        print("Dependency check passed; unused FlashInfer cache providers omitted.")
        return 0
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
