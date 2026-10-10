"""Programmatic Dependent Launch (PDL) for the fork's own Triton kernels.

The decode step runs ~1850 kernels inside CUDA graphs and the GPU is idle for
only ~0.15 ms of the ~11.3 ms, so the cost is not the launch gap but each small
kernel's own ramp: block scheduling plus the first dependent loads. PDL lets
kernel N+1 start scheduling blocks and run its *independent* prologue (weight,
scale and index loads that do not depend on N's output) while N drains, and
block on ``griddepcontrol.wait`` only for the loads that really do depend on N.

Two things are needed per kernel:

* the launch carries ``launch_pdl=True`` (Triton turns that into
  ``cudaLaunchAttributeProgrammaticStreamSerialization`` on the launch config);
* the body calls :func:`pdl_wait` before its first dependent load and
  :func:`pdl_trigger` once the buffers the *next* kernel may overwrite have
  been read for the last time.

Both are gated on ``SGLANG_TRITON_PDL=1`` (default off). When off, ``PDL`` is
``False``, so the ``griddepcontrol`` instructions are not traced at all and
``launch_pdl=False`` is Triton's own default: the generated PTX and the launch
config are identical to the un-instrumented build.

The JIT CUDA kernels (``kernels/jit/csrc``) already do this on their own via
``PDLWaitPrimary`` / ``PDLTriggerSecondary`` / ``LaunchKernel::enable_pdl`` and
are not affected by this flag.
"""

from __future__ import annotations

import os

import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

__all__ = ["PDL", "triton_pdl_enabled", "pdl_wait", "pdl_trigger"]


def _detect() -> bool:
    if os.environ.get("SGLANG_TRITON_PDL", "0").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    try:
        import torch

        if getattr(torch.version, "hip", None) or getattr(torch.version, "musa", None):
            return False
        # griddepcontrol is sm_90 (Hopper) and later.
        major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
        return major >= 9
    except Exception:
        return False


#: ``True`` when ``SGLANG_TRITON_PDL=1`` and the device supports griddepcontrol.
#: Launch sites pass it as both ``launch_pdl=PDL`` and ``USE_PDL=PDL``.
PDL = _detect()


def triton_pdl_enabled() -> bool:
    return PDL


@triton.jit
def pdl_wait(USE_PDL: tl.constexpr):
    """Block until the preceding kernel in the stream has made its stores visible.

    Everything *before* this in the kernel may read only memory the preceding
    kernel does not write (weights, scales, constants, its own program ids).
    """
    if USE_PDL:
        gdc_wait()


@triton.jit
def pdl_trigger(USE_PDL: tl.constexpr):
    """Allow the next kernel in the stream to start scheduling its blocks.

    Only correct after the last read of every buffer the next kernel may
    overwrite; placing it at the very end of the body is always safe and still
    hides the next kernel's scheduling ramp behind this kernel's drain.
    """
    if USE_PDL:
        gdc_launch_dependents()
