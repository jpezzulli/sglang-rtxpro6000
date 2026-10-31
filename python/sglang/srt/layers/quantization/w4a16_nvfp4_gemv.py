"""W4A16 NVFP4 Triton GEMV dequant primitives, carried in for the draft MoE GEMV.

Direct port of the donor material needed by ``layers/moe/draft_moe_gemv.py``
(DG1, donor patch 0112 / upstream commit 714eba5ab6, donor snapshot
5105985116eb00dea8e6138aabeb5363387cb9de).  The donor's full file also holds
the dense skinny-GEMV kernels of patch 0089; only these two symbols are part
of the DG1 contract, so the rest of the file (and its ``w8a16_gemv`` scratch
dependency) is intentionally not carried here.

The weight format is the same NVFP4 the experts already use:

    W[n, k] = e2m1_code(n, k) * block_scale_e4m3(n, k // 16) * global_scale

stored as ``wq[N, K//2]`` uint8 (two 4-bit codes per byte, **low nibble = even k**),
``bs[N, K//16]`` uint8 holding the e4m3 bit patterns, and one fp32 global scale.
"""

import triton
import triton.language as tl

#: The bit trick decodes to `e2m1_value * 2^-14`; this undoes it. It is applied ONCE
#: per output in the epilogue, together with the global scale, rather than inside the k
#: loop -- the block scale stays the raw e4m3 value there, which keeps `fp4 * scale`
#: exactly representable in bf16 (2 + 4 significand bits against bf16's 8).
_FP4_TRICK = tl.constexpr(16384.0)


@triton.jit
def _unpack_dequant(b, BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr):
    """[BLOCK_N, BLOCK_KB] packed uint8 -> [BLOCK_N, 2*BLOCK_KB] bf16 = e2m1 * 2^-14.

    Low nibble is the even k, high nibble the odd k, matching the NVFP4 checkpoint
    layout.  ``tl.join`` puts them on a new trailing axis in exactly that order, so the
    reshape below is the identity permutation and costs no shuffle beyond the join.
    """
    v = b.to(tl.int32)
    lo = v & 0x0F
    hi = (v >> 4) & 0x0F
    # magnitude -> fp16 [11:9] (exponent low bits + mantissa top bit), sign -> [15].
    rl = ((lo << 9) & 0x0E00) | ((lo & 0x08) << 12)
    rh = ((hi << 9) & 0x0E00) | ((hi & 0x08) << 12)
    fl = rl.to(tl.uint16).to(tl.float16, bitcast=True)
    fh = rh.to(tl.uint16).to(tl.float16, bitcast=True)
    w = tl.join(fl, fh)  # [BLOCK_N, BLOCK_KB, 2], last axis = (even k, odd k)
    return tl.reshape(w, (BLOCK_N, 2 * BLOCK_KB)).to(tl.bfloat16)
