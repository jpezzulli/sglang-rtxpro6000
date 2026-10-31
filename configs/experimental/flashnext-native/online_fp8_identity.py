#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print the online-FP8 precision this launch will actually run in.

A persistence namespace has to be keyed on what is resident, not on which
Boolean asked for it: on an eligible Flash-Next checkpoint on SM120, an unset
(automatic) request and an explicit ``true`` resolve to the same row-wise E4M3
weights and scales, and automatic-off resolves to the same untouched BF16
checkpoint as an explicit ``false``. Hashing the requested mode splits those
equivalent launches into separate namespaces, so a launch that changes only the
spelling of the same representation pays for a cold re-read of a qualified tier.

This is the launch-side adapter for the runtime's own decision, not a second
policy: it feeds the checkpoint's metadata, the compute dtype this recipe pins,
and the capability of the device the recipe just made visible into the same
resolver (``resolve_precision``) the model build uses. It exists so a shell
recipe can ask that question without importing the serving package.

An explicit ``true`` on hardware this build cannot serve exits non-zero with the
resolver's message, which is the fail-loud contract the runtime applies at boot;
--requested is the recipe's already-normalized mode, so a value the recipe would
not vouch for cannot reach the resolver.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

#: This checkout's copy, for a worker where the serving package is not installed
#: (or is not importable in this interpreter). The resolver is pure torch.
_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "sglang"
    / "kernels"
    / "ops"
    / "gemm"
    / "sm120_online_fp8.py"
)

_REQUESTED = {"auto": None, "true": True, "false": False}


def _online_fp8_module():
    try:
        from sglang.kernels.ops.gemm import sm120_online_fp8

        return sm120_online_fp8
    except Exception:
        # Fall back to the copy this recipe ships with; if that cannot load
        # either, the launch has no resolver and must not guess an identity.
        spec = importlib.util.spec_from_file_location("sm120_online_fp8", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="checkpoint directory")
    parser.add_argument(
        "--dtype", required=True, help="the compute dtype this launch pins"
    )
    parser.add_argument(
        "--requested",
        choices=tuple(_REQUESTED),
        default="auto",
        help="the normalized SGLANG_SM120_ONLINE_MXFP8 mode",
    )
    args = parser.parse_args(argv)

    import torch

    dtype = getattr(torch, args.dtype, None)
    if dtype is None:
        print(f"unknown compute dtype: {args.dtype}", file=sys.stderr)
        return 2
    fp8 = _online_fp8_module()
    # launch_precision's own checkpoint reader, so config.json discovery stays
    # in one place and cannot drift from the runtime's.
    available = torch.cuda.is_available()
    try:
        precision = fp8.resolve_precision(
            _REQUESTED[args.requested],
            cuda_available=available,
            capability=torch.cuda.get_device_capability() if available else None,
            model_eligible=fp8.checkpoint_eligible(
                fp8._read_checkpoint_metadata(args.model), dtype
            ),
        )
    except RuntimeError as err:
        # An explicit request this build cannot serve: say which, and let the
        # recipe abort rather than claim a namespace for a missing representation.
        print(f"{err}", file=sys.stderr)
        return 1
    print(precision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
