# Direct port of the donor's W8A16 GEMV, unmodified except where noted:
#   aiueo52/sglang-rtxpro6000 @5105985116eb00dea8e6138aabeb5363387cb9de,
#   python/sglang/srt/layers/quantization/w8a16_gemv.py (Apache-2.0), whose
#   per-shape tile table / in-launch split-K fixup and split-K scratch slots
#   come from aiueo52/flash-next-rtxpro6000 @524af49abcca66fcb4377ba8297022804535fccf
#   (patches 0026 and 0041). Public companion documentation:
#   aiueo52/flash-next-rtxpro6000 (W8A16).
#
# `_store_split`, `_w8a16_gemv_kernel`, the plan/fit tables, the scratch slots and
# `w8a16_gemv` are the donor's, byte for byte (PDL included). The fused
# gated-RMSNorm prologue is part of that byte-for-byte kernel/`_launch` (its
# `NORM_G`/`NORM_SIGMOID` constexprs), and the donor's `_norm_plan`,
# `w8a16_gemv_norm_gated_supported` and `w8a16_gemv_norm_gated` wrappers below
# are direct excerpts of donor ``w8a16_gemv.py:663-810`` @5105985116eb, used by
# the GDN out_proj caller (`models/qwen3_5.py`, SGLANG_NORM_INTO_GEMV). The
# deviations are:
#   * the other callers this base does not have are not carried: the fused
#     gate_up+SiLU kernel, so some donor comments name callers that do not exist
#     here.  `bf16_gemv` (+ its scale-of-one) is carried for the draft MTP entry
#     fusion (`models/qwen4_exp_mtp.py`, SGLANG_MTP_FC_GEMV, donor
#     qwen4_exp_mtp.py:32/280/331 @5105985).
#   * `_workspace_slot` raises instead of silently allocating during a capture;
#   * `prealloc`'s documented caller is this fork's rowwise-FP8 output heads
#     (`sglang.kernels.ops.gemm.sm120_online_fp8`), not `Fp8LinearMethod`;
#   * the fused-norm gate is `norm_into_gemv_enabled()`, a call-time tri-state
#     resolution (saved explicit SGLANG_NORM_INTO_GEMV wins; unset follows the
#     Flash-Next/SM120 default selection), not a raw import-time env read.
"""Triton W8A16 skinny GEMM: y[M,N] = x[M,K](bf16) @ W[N,K](fp8 e4m3)^T * scale[N] (fp32).
Targets decode/verify shapes (M <= 16). Weight-only FP8: no activation quantization.

Two properties of the decode shapes decide throughput here:

- Weight layout. Every caller hands over an [N, K]-CONTIGUOUS weight, stride (K, 1).
  Fp8LinearMethod quantizes to [N, K], stores the [K, N] transposed view on the layer,
  and transposes it back at the call site; bf16_gemv passes linear.weight directly.
  So K is the contiguous axis, the natural weight tile is [BLOCK_N, BLOCK_K], and
  tl.dot needs it transposed (W_KN False). The W_KN True path is for a genuinely
  [K, N]-contiguous weight, which no caller produces today.
- Grid size. One CTA per N block owning the whole K reduction launches 3..80 CTAs for
  the small-N linears (in_proj_ba, shared-expert gate_up/down, GDN out_proj, o_proj) on
  a 188-SM part, so those calls run at a few hundred GB/s no matter how well the inner
  loop is written. Splitting K across CTAs fixes that, and the reduction is folded into
  the same launch: every CTA writes an fp32 partial and bumps a per-N-block counter, and
  whoever observes the last increment sums the partials, scales, and writes bf16 y. The
  counter is reset to 0 by that same CTA, so no memset is needed and the kernel is safe
  to capture in a CUDA graph and replay.

M == 1 additionally wants USE_DOT False on every shape measured (2-23% faster): the
broadcast-multiply path keeps M_PAD at 1 instead of padding to a 16-row mma, which
shrinks both the accumulator and the split-K partials 16-fold.

The split-K scratch is one buffer per device, so two split-K calls must not overlap;
that holds for the decode path, which issues every linear on one stream.
"""

import functools
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.triton_pdl import PDL, pdl_trigger, pdl_wait
from sglang.srt.environ import envs


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


#: When a two-destination store's column split does not land on a BLOCK_N tile
#: boundary, the straddling CTA runs *both* stores and, on a 3-CTA launch like
#: GDN in_proj_ba, sets the whole kernel's latency (see the revert of
#: 633f70f5ff: branching inside the straddling CTA only made it worse -- the
#: straddler still does both stores and there is nothing to hide the extra
#: control flow behind). This flag re-grids instead: the [0, SPLIT_N) and
#: [SPLIT_N, N) column ranges get their own contiguous block ranges, so *no*
#: CTA straddles and every CTA has exactly one destination. Each weight row is
#: still read by exactly one CTA running the same k loop, so the output is
#: bit-identical -- only the (pid_n -> columns) map changes.
#:
#: Measured (server A/B 2026-09-06, in_proj_ba grid [3,1,1] -> [4,1,1], CUPTI
#: medians): W16 code-edit 7.82 -> 7.61 us/call, W16 prose-en 7.43 -> 7.23,
#: W4 code-edit 5.82 -> 5.84, W4 prose-en 5.61 -> 6.16. So it recovers ~0.2 us
#: of the 0.9 us the split costs at M=16 and is neutral-to-noise at M=4.
#: Below the 1 us/call bar -> RECOMMENDED OFF; kept because it is bit-exact and
#: is the right shape for any future two-destination split that is not
#: launch-bound at 3-4 CTAs.
BA_SPLIT_GRID = _env_flag("SGLANG_GEMV_BA_SPLIT_GRID")


@triton.jit
def _store_split(
    acc,
    y_ptr,
    y2_ptr,
    offs_m,
    offs_n,
    stride_ym,
    stride_yn,
    stride_y2m,
    m_mask,
    n_mask,
    seg1,
    SPLIT_N: tl.constexpr,
    SEG_NB0: tl.constexpr,
):
    """Write the rounded accumulator to `y`, or to `y`/`y2` split at column SPLIT_N.

    SPLIT_N == 0 is the default single-destination store, byte-for-byte the
    original epilogue. With SPLIT_N > 0 columns [0, SPLIT_N) land in `y` and
    columns [SPLIT_N, N) land in `y2` at column `n - SPLIT_N`; both stores see
    the same rounded value, so the numerics are identical either way.

    SEG_NB0 > 0 (the re-gridded form, see BA_SPLIT_GRID) means the caller has
    already restricted this CTA's columns to one side of the split and passes
    which side in `seg1`, so exactly one store is emitted.
    """
    # Round to bf16 first even when the destination is fp32: an fp32 destination
    # is only ever the shared logits buffer, whose old contents were a bf16
    # result widened by `.copy_()`. Rounding here keeps that bit-identical.
    val = acc.to(tl.bfloat16).to(y_ptr.dtype.element_ty)
    if SEG_NB0 > 0:
        if seg1:
            tl.store(
                y2_ptr
                + offs_m[:, None] * stride_y2m
                + (offs_n - SPLIT_N)[None, :] * stride_yn,
                val,
                mask=m_mask[:, None] & n_mask[None, :],
            )
        else:
            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                val,
                mask=m_mask[:, None] & n_mask[None, :],
            )
    elif SPLIT_N > 0:
        lo = offs_n < SPLIT_N
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            val,
            mask=m_mask[:, None] & n_mask[None, :] & lo[None, :],
        )
        n2 = tl.maximum(offs_n - SPLIT_N, 0)
        tl.store(
            y2_ptr + offs_m[:, None] * stride_y2m + n2[None, :] * stride_yn,
            val,
            mask=m_mask[:, None] & n_mask[None, :] & (offs_n >= SPLIT_N)[None, :],
        )
    else:
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            val,
            mask=m_mask[:, None] & n_mask[None, :],
        )


@triton.jit
def _w8a16_gemv_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    y2_ptr,
    ws_ptr,
    cnt_ptr,
    z_ptr,
    nw_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    stride_y2m,
    stride_zm,
    norm_eps,
    SPLIT_N: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_PAD: tl.constexpr,
    SPLITS: tl.constexpr,
    EVEN_K: tl.constexpr,
    W_KN: tl.constexpr,
    USE_DOT: tl.constexpr,
    USE_PDL: tl.constexpr = False,
    SEG_NB0: tl.constexpr = 0,
    NORM_G: tl.constexpr = 0,
    NORM_SIGMOID: tl.constexpr = False,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    # SEG_NB0 > 0: the n grid is two back-to-back segments, [0, SPLIT_N) in
    # blocks [0, SEG_NB0) and [SPLIT_N, N) in the rest, so no CTA straddles the
    # two-destination split and every CTA has a single store.
    seg1 = SEG_NB0 > 0 and pid_n >= SEG_NB0
    if SEG_NB0 > 0:
        n_lo = tl.where(seg1, SPLIT_N + (pid_n - SEG_NB0) * BLOCK_N, pid_n * BLOCK_N)
        n_hi = tl.where(seg1, N, SPLIT_N)
    else:
        n_lo = pid_n * BLOCK_N
        n_hi = N
    offs_n = n_lo + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, M_PAD)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < n_hi
    m_mask = offs_m < M
    if NORM_G > 0:
        # Gated RMSNorm folded into the A-load; the group is exactly one
        # BLOCK_K tile (asserted by the planner), so the row statistics are
        # local to the tile and nothing extra is read for them.
        norm_w = tl.load(nw_ptr + offs_k).to(tl.float32)
    # x comes from the previous kernel; the weight/scale loads below are all
    # inside the reduction loop, so the wait sits at the top and what PDL buys
    # here is the block-scheduling ramp overlapping the previous kernel's drain.
    pdl_wait(USE_PDL)
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    for k0 in range(pid_k * BLOCK_K, K, SPLITS * BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < K
        # W_KN: the weight is [K, N] contiguous, so the [BLOCK_K, BLOCK_N] tile is the
        # contiguous one and tl.dot takes it directly instead of transposing in smem.
        # Every current caller is [N, K] contiguous and takes the else branch.
        if W_KN:
            wp = w_ptr + kk[:, None] * stride_wk + offs_n[None, :] * stride_wn
            if EVEN_K:
                w = tl.load(wp, mask=n_mask[None, :], other=0.0)
            else:
                w = tl.load(wp, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        else:
            wp = w_ptr + offs_n[:, None] * stride_wn + kk[None, :] * stride_wk
            if EVEN_K:
                w = tl.load(wp, mask=n_mask[:, None], other=0.0)
            else:
                w = tl.load(wp, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        if USE_DOT:
            xp = x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk
            if EVEN_K:
                x = tl.load(xp, mask=m_mask[:, None], other=0.0)
            else:
                x = tl.load(xp, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            if NORM_G > 0:
                zp = z_ptr + offs_m[:, None] * stride_zm + kk[None, :] * stride_xk
                if EVEN_K:
                    zt = tl.load(zp, mask=m_mask[:, None], other=0.0).to(tl.float32)
                else:
                    zt = tl.load(
                        zp, mask=m_mask[:, None] & k_mask[None, :], other=0.0
                    ).to(tl.float32)
                xf = x.to(tl.float32)
                var = tl.sum(xf * xf, axis=1) / NORM_G
                rstd = tl.rsqrt(var + norm_eps)
                yn = xf * rstd[:, None] * norm_w[None, :]
                if NORM_SIGMOID:
                    yn = yn * tl.sigmoid(zt)
                else:
                    yn = yn * (zt * tl.sigmoid(zt))
                x = yn.to(tl.bfloat16)
            if W_KN:
                acc += tl.dot(x, w.to(tl.bfloat16), out_dtype=tl.float32)
            else:
                acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
        else:
            # M == 1: broadcast-multiply and reduce instead of padding to a 16-row mma.
            if EVEN_K:
                xv = tl.load(x_ptr + kk * stride_xk).to(tl.float32)
            else:
                xv = tl.load(x_ptr + kk * stride_xk, mask=k_mask, other=0.0).to(
                    tl.float32
                )
            if NORM_G > 0:
                if EVEN_K:
                    zv = tl.load(z_ptr + kk * stride_xk).to(tl.float32)
                else:
                    zv = tl.load(z_ptr + kk * stride_xk, mask=k_mask, other=0.0).to(
                        tl.float32
                    )
                rstd1 = tl.rsqrt(tl.sum(xv * xv, axis=0) / NORM_G + norm_eps)
                yv = xv * rstd1 * norm_w
                if NORM_SIGMOID:
                    yv = yv * tl.sigmoid(zv)
                else:
                    yv = yv * (zv * tl.sigmoid(zv))
                xv = yv.to(tl.bfloat16).to(tl.float32)
            if W_KN:
                acc += tl.sum(w.to(tl.float32) * xv[:, None], axis=0)[None, :]
            else:
                acc += tl.sum(w.to(tl.float32) * xv[None, :], axis=1)[None, :]

    if SPLITS == 1:
        if PER_CHANNEL:
            acc = acc * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        else:
            acc = acc * tl.load(s_ptr)
        _store_split(
            acc,
            y_ptr,
            y2_ptr,
            offs_m,
            offs_n,
            stride_ym,
            stride_yn,
            stride_y2m,
            m_mask,
            n_mask,
            seg1,
            SPLIT_N,
            SEG_NB0,
        )
        pdl_trigger(USE_PDL)
        return

    # Split-K fixup, in this same launch. `.cg` keeps the partials out of the
    # non-coherent L1 so the CTA that runs the reduction sees the other CTAs' writes.
    slot = offs_m[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    base = ws_ptr + (pid_n * SPLITS) * (M_PAD * BLOCK_N)
    tl.store(base + pid_k * (M_PAD * BLOCK_N) + slot, acc, cache_modifier=".cg")
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    if done == SPLITS - 1:
        tot = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
        for s in tl.static_range(SPLITS):
            tot += tl.load(base + s * (M_PAD * BLOCK_N) + slot, cache_modifier=".cg")
        if PER_CHANNEL:
            tot = tot * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        else:
            tot = tot * tl.load(s_ptr)
        _store_split(
            tot,
            y_ptr,
            y2_ptr,
            offs_m,
            offs_n,
            stride_ym,
            stride_yn,
            stride_y2m,
            m_mask,
            n_mask,
            seg1,
            SPLIT_N,
            SEG_NB0,
        )
        # Every increment for this N block has happened, so a plain store is enough to
        # leave the counter at 0 for the next launch / graph replay.
        tl.store(cnt_ptr + pid_n, 0)
    pdl_trigger(USE_PDL)


# Split-K scratch, allocated once per device and reused: the fixup CTA leaves the
# counters at 0, so nothing has to be cleared between calls. Sized for the largest
# plan _plan can emit (MAX_N_BLOCKS n blocks x MAX_SPLITS splits x 16 rows x 128 wide
# would be 8 M floats; the planner caps the product at _WS_FLOATS instead).
_WS_FLOATS = 1 << 21
_WS_COUNTERS = 4096
_MAX_SPLITS = 32
_WS = {}
# The scalar "scale" bf16_gemv passes for its (unquantized) weights.
_ONES = {}


# Split-K scratch slots. The decode path issues most linears on one stream, but the
# MoE block overlaps the shared expert with the router/routed experts on an alt
# stream (qwen2_moe.py), and two split-K launches that share one scratch and one
# counter array while running concurrently corrupt each other's partials (seen as
# garbage generations with SGLANG_ROUTER_GEMV=1). Work issued inside
# ``with scratch_slot(1):`` uses a second, independent scratch.
_N_SLOTS = 2
_SLOT = 0


class scratch_slot:
    """Context manager selecting the split-K scratch slot for launches inside it."""

    def __init__(self, slot: int):
        assert 0 <= slot < _N_SLOTS
        self.slot = slot
        self.prev = 0

    def __enter__(self):
        global _SLOT
        self.prev = _SLOT
        _SLOT = self.slot
        return self

    def __exit__(self, *exc):
        global _SLOT
        _SLOT = self.prev
        return False


def _workspace_slot(device, slot: int):
    key = (device, slot)
    ws = _WS.get(key)
    if ws is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            # Donor contract: the scratch outlives every graph that replays into
            # it. Allocating here would take memory from this graph's private pool.
            raise RuntimeError(
                "w8a16 gemv split-K scratch must be prealloc'd before CUDA graph "
                f"capture (device={device}, slot={slot})"
            )
        ws = (
            torch.empty(_WS_FLOATS, dtype=torch.float32, device=device),
            torch.zeros(_WS_COUNTERS, dtype=torch.int32, device=device),
        )
        _WS[key] = ws
    return ws


def _workspace(device):
    return _workspace_slot(device, _SLOT)


def prealloc(device) -> None:
    """Materialize the per-device scratch before any CUDA graph is captured.

    Warm-up runs at prefill widths, where every caller falls back to cuBLAS, so
    without this the split-K scratch (and ``bf16_gemv``'s scale-of-one) would
    first be allocated inside a capture and would live in that graph's private
    memory pool. Called from ``sglang.kernels.ops.gemm.sm120_online_fp8`` when
    the rowwise-FP8 output heads are installed.
    """
    for slot in range(_N_SLOTS):
        _workspace_slot(device, slot)
    if device not in _ONES:
        _ONES[device] = torch.ones(1, dtype=torch.float32, device=device)


_NUM_SMS = None


def _num_sms(device) -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS


def _m_bucket(M: int) -> int:
    return 1 if M == 1 else (4 if M <= 4 else 16)


# Measured on RTX PRO 6000 Blackwell Max-Q (sm_120, 188 SMs) with
# bench/w8a16v2/tune_v3.py + the head-to-head re-check in bench/w8a16v2/final_v3.py;
# the metric is the CUPTI kernel-duration median over a CUDA-graph replay whose
# weight working set is several times the 128 MB L2, i.e. the number the serving
# torch profile reports. Re-run it after a kernel or hardware change.
#
# key:   (weight element bytes, N is the contiguous weight axis, N, K)
# value: (BLOCK_N, BLOCK_K, SPLITS, USE_DOT, W_KN, num_warps, num_stages)
# W_KN None derives the tile orientation from the weight strides.
#
# `N is the contiguous weight axis` is False on every row: both callers pass an
# [N, K]-contiguous weight (see the module docstring). The pre-v3 table keyed the
# fp8 rows True, which no call ever matched, so the server silently ran _plan's
# fallback for all of them; the "vs" ratios below are against that fallback.
_BY_SHAPE = {
    # fp8, w8a16_gemv
    # shared-expert down: the single-split (32,128,1,...,4,4) bench winner measured 5.2us
    # in the W4 server (M=4) vs 4.0us for the split-5 fallback; M=16 keeps the winner below.
    (1, False, 2560, 640): (32, 128, 5, True, None, 4, 3),
    (1, False, 1280, 2560): (16, 128, 10, True, None, 2, 2),  # shared gate_up, 1.08x
    (1, False, 32768, 2560): (128, 256, 1, True, None, 8, 3),  # draft lm_head, 1.14x
    (1, False, 248320, 2560): (128, 256, 1, True, None, 8, 3),  # lm_head, 1.02x
    # bf16, bf16_gemv
    # GDN in_proj_ba: the bench's split-K winner (16,256,10,...,2,4) measured 9.2us in the
    # server vs 7.9us for this single-launch tile (2026-09-03 gemv3 trace), so keep it.
    (2, False, 96, 2560): (32, 512, 1, True, None, 4, 3),
    (2, False, 512, 2560): (16, 128, 10, True, None, 2, 4),  # MoE router gate, 1.04x
    # The 2560x6144 out_proj/o_proj and the 13312x2560 attention qkv are absent at
    # M > 1 because _plan's fallback already emits the fastest tile measured for them.
}
_BY_SHAPE_M16 = {
    # The _BY_SHAPE gate_up tile loses 12% at M=16; the qkvz tile below loses 1% at M=4.
    (1, False, 1280, 2560): (16, 64, 10, True, None, 4, 3),  # shared gate_up, 1.03x
    (1, False, 16384, 2560): (64, 128, 1, True, None, 4, 3),  # GDN in_proj_qkvz, 1.04x
    (1, False, 2560, 640): (
        32,
        128,
        1,
        True,
        None,
        4,
        4,
    ),  # shared-expert down (W16 server 4.1->3.8us)
    # attention qkv at M=16: both the wide (128,256,8w) tile and the (64,128) qkvz tile
    # measured 27.5us in the server vs 25.6us for the narrow single-split tile; pin it.
    (1, False, 13312, 2560): (32, 128, 1, True, None, 4, 3),
}
_BY_SHAPE_M1 = {
    # USE_DOT False (the broadcast path, M_PAD 1) is only legal at M == 1.
    (1, False, 2560, 640): (16, 256, 3, False, None, 4, 3),  # shared-expert down, 1.20x
    (1, False, 1280, 2560): (16, 128, 10, False, None, 2, 3),  # shared gate_up, 1.23x
    (1, False, 2560, 6144): (16, 256, 6, False, None, 4, 3),  # out_proj/o_proj, 1.05x
    (1, False, 13312, 2560): (32, 256, 1, False, None, 4, 3),  # attention qkv, 1.02x
    (1, False, 32768, 2560): (32, 256, 1, False, None, 4, 3),  # draft lm_head, 1.17x
    (2, False, 512, 2560): (16, 256, 10, False, None, 4, 3),  # MoE router gate, 1.15x
    (2, False, 2560, 2560): (16, 256, 5, False, None, 4, 2),  # MTP fc_*, 1.06x
}


def _expand(table, buckets):
    return {
        (wb, cn, m, N, K): cfg for (wb, cn, N, K), cfg in table.items() for m in buckets
    }


_TUNED = _expand(_BY_SHAPE, (1, 4, 16))
_TUNED.update(_expand(_BY_SHAPE_M16, (16,)))
_TUNED.update(_expand(_BY_SHAPE_M1, (1,)))


def _prev_pow2(v: int) -> int:
    return 1 << max(0, v.bit_length() - 1)


@functools.lru_cache(maxsize=512)
def _plan(M: int, N: int, K: int, contig_n: bool, w_bytes: int, sms: int):
    """(BLOCK_N, BLOCK_K, SPLITS, USE_DOT, W_KN, num_warps, num_stages) for one call."""
    tuned = _TUNED.get((w_bytes, contig_n, _m_bucket(M), N, K))
    if tuned is not None:
        return _fit(M, N, tuned)
    if contig_n:
        # A [K, N]-contiguous weight wants the widest N block it can keep the grid
        # full with, since 32 fp8 columns is only one 32 B sector per k. Untuned:
        # no caller produces this layout.
        block_n = 128 if N >= 2048 else (64 if N >= 512 else 32)
        return _fit(M, N, (block_n, 128, 1, True, None, 8, 3))
    # [N, K] contiguous. When N alone fills the machine, a long BLOCK_K keeps more
    # bytes in flight: 1.15x on the 32768x2560 draft head, 1.01-1.04x on the other
    # three big shapes. The M == 4 bucket is the exception -- there the wide tile
    # measured 1-2% slower than the [32, 128] one on both 2560-K shapes.
    if triton.cdiv(N, 128) * 2 >= sms and _m_bucket(M) != 4:
        if M == 1:
            return (32, 256, 1, False, None, 4, 3)
        # A [128, 256] bf16 tile needs 144 KB of smem, over the 99 KB sm_120 limit.
        return (128 if w_bytes == 1 else 64, 256, 1, True, None, 8, 3)
    use_dot = M > 1
    block_n = 16 if M == 1 else 32
    block_k, warps = 128, 4
    if triton.cdiv(N, 32) >= sms:
        return (block_n, block_k, 1, use_dot, None, warps, 3)
    # Aim for ~2 CTAs per SM, ~5 at M == 1 where an M_PAD 1 partial is 16x cheaper;
    # shrink BLOCK_K if K does not hold enough blocks for that.
    per_sm = 5 if M == 1 else 2
    want = min(_MAX_SPLITS, max(1, (per_sm * sms) // triton.cdiv(N, block_n)))
    block_k = min(block_k, max(64, _prev_pow2(max(1, K // want))))
    n_kb = triton.cdiv(K, block_k)
    splits = min(want, n_kb)
    # Prefer a split count that divides the k-block count so every CTA does equal work.
    for cand in range(min(n_kb, splits + 2), max(1, splits - 3), -1):
        if n_kb % cand == 0 and cand <= _MAX_SPLITS:
            splits = cand
            break
    return _fit(M, N, (block_n, block_k, splits, use_dot, None, warps, 3))


def _fit(M: int, N: int, cfg, n_blocks: Optional[int] = None):
    """Drop to SPLITS=1 if the plan would not fit the preallocated scratch."""
    block_n, block_k, splits, use_dot, w_kn, warps, stages = cfg
    if splits == 1:
        return cfg
    m_pad = 16 if (use_dot or M > 1) else 1
    if n_blocks is None:
        n_blocks = triton.cdiv(N, block_n)
    if (
        splits > _MAX_SPLITS
        or n_blocks > _WS_COUNTERS
        or n_blocks * splits * m_pad * block_n > _WS_FLOATS
    ):
        return (block_n, block_k, 1, use_dot, w_kn, warps, stages)
    return cfg


def _launch(x, w, s, y, M, N, K, per_channel, cfg, y2=None, split_n=0, norm=None):
    block_n, block_k, splits, use_dot, w_kn, num_warps, num_stages = cfg
    use_dot = use_dot or M > 1
    m_pad = 16 if use_dot else 1
    ws, cnt = _workspace(x.device)
    n_blocks = triton.cdiv(N, block_n)
    seg_nb0 = 0
    if BA_SPLIT_GRID and split_n > 0 and split_n % block_n != 0:
        # Re-grid so the two destinations own disjoint block ranges.
        nb0 = triton.cdiv(split_n, block_n)
        nb = nb0 + triton.cdiv(N - split_n, block_n)
        # The re-grid adds at most one block; only take it if the split-K
        # scratch (sized from cdiv(N, BLOCK_N) in _fit) still covers it.
        if splits == 1 or (
            nb <= _WS_COUNTERS and nb * splits * m_pad * block_n <= _WS_FLOATS
        ):
            seg_nb0, n_blocks = nb0, nb
    z, nw, norm_g, norm_eps, norm_sigmoid = (x, x, 0, 0.0, False)
    if norm is not None:
        z, nw, norm_g, norm_eps, norm_sigmoid = norm
    _w8a16_gemv_kernel[(n_blocks, splits)](
        x,
        w,
        s,
        y,
        y if y2 is None else y2,
        ws,
        cnt,
        z,
        nw,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        y.stride(0) if y2 is None else y2.stride(0),
        z.stride(0),
        norm_eps,
        SPLIT_N=split_n,
        PER_CHANNEL=per_channel,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        M_PAD=m_pad,
        SPLITS=splits,
        EVEN_K=K % block_k == 0,
        W_KN=(w.stride(0) == 1) if w_kn is None else w_kn,
        USE_DOT=use_dot,
        USE_PDL=PDL,
        SEG_NB0=seg_nb0,
        NORM_G=norm_g,
        NORM_SIGMOID=norm_sigmoid,
        launch_pdl=PDL,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


def _out_or_new(out, M: int, N: int, device):
    """Validate an optional destination, else allocate the usual bf16 one.

    `out` may be fp32 (the shared next-token logits buffer): the kernel still
    rounds the fp32 accumulator to bf16 before widening it, so writing straight
    into that buffer is bit-identical to the bf16 result plus `buffer.copy_()`.
    """
    if out is None:
        return torch.empty((M, N), dtype=torch.bfloat16, device=device)
    if (
        tuple(out.shape) != (M, N)
        or out.dtype not in (torch.bfloat16, torch.float32)
        or out.device != device
        or out.stride(1) != 1
    ):
        raise ValueError(
            f"w8a16 gemv out= must be a [{M}, {N}] row-contiguous bf16/fp32 CUDA "
            f"tensor on {device}; got {tuple(out.shape)} {out.dtype} {out.device}"
        )
    return out


def _split_dests(out, out2, split_n: int, M: int, N: int, device):
    """Validate a two-destination (`out` | `out2`) column split of the [M, N] result.

    Returns ``(y, y2, split_n)``; ``split_n == 0`` means the usual single
    destination. Column ``n < split_n`` goes to ``out``, ``n >= split_n`` to
    ``out2`` at column ``n - split_n``. The GEMV itself is untouched -- same
    k-loop, same fp32 accumulation order, same per-channel scale index, same
    bf16 rounding -- so the two destinations hold exactly the values a single
    [M, N] output would have held.
    """
    if out2 is None:
        return _out_or_new(out, M, N, device), None, 0
    if not (0 < split_n < N):
        raise ValueError(f"w8a16 gemv split_n must be in (0, {N}); got {split_n}")
    y = _out_or_new(out, M, split_n, device)
    y2 = _out_or_new(out2, M, N - split_n, device)
    if y.dtype != y2.dtype or y.stride(1) != 1 or y2.stride(1) != 1:
        raise ValueError("w8a16 gemv out/out2 must share a dtype and be row-contiguous")
    return y, y2, split_n


def w8a16_gemv(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: torch.Tensor,
    cfg=None,
    out: Optional[torch.Tensor] = None,
    out2: Optional[torch.Tensor] = None,
    split_n: int = 0,
):
    """x: [M,K] bf16; w: [N,K] fp8_e4m3 (any strides); scale: [N] or [N,1] or scalar fp32.

    With ``out2``, the N columns are split at ``split_n`` across two
    caller-owned destinations instead of one [M, N] buffer (see _split_dests).
    """
    M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K and M <= 16
    y, y2, split_n = _split_dests(out, out2, split_n, M, N, x.device)
    s = scale.reshape(-1)
    if s.dtype != torch.float32 or not s.is_contiguous():
        s = s.contiguous().float()
    if cfg is None:
        cfg = _plan(M, N, K, w.stride(0) == 1, w.element_size(), _num_sms(x.device))
    _launch(x, w, s, y, M, N, K, s.numel() > 1, cfg, y2, split_n)
    return y if y2 is None else (y, y2)


# ---------------------------------------------------------------------------
# Fused gated-RMSNorm prologue (SGLANG_NORM_INTO_GEMV=1)
#
# The GDN block normalises its attention output per (token, v-head) --
# `RMSNormGated(head_v_dim, norm_before_gate=True)`, i.e.
# `y = x * rsqrt(mean(x^2) + eps) * w * silu(z)` -- and immediately feeds the
# [tokens, num_v_heads * head_v_dim] result to the out_proj GEMV. That norm is
# 192 one-warp CTAs moving ~150 KB: pure launch + ramp, ~1.9 us, 36 times per
# verify step.
#
# The GEMV already re-reads the whole A row once per n block, and with
# BLOCK_K == head_v_dim each A tile is exactly one norm group, so the row
# statistics can be recomputed inside the tile from registers. The only extra
# traffic is the `z` tile (same size as the A tile, L2-resident at these
# widths) and the [head_v_dim] norm weight.
#
# Numerics: the fp32 sum of squares is re-associated (the GEMV tile has a
# different thread layout from the norm kernel's), so the result is within
# 1 ulp of bf16 rather than bit-identical -- everything else (fp32 math, the
# `silu` form, the bf16 rounding before the dot) is reproduced exactly.
# ---------------------------------------------------------------------------


def norm_into_gemv_enabled() -> bool:
    """Gate for the fused gated-RMSNorm prologue.

    One call-time resolution shared by both callers (``models/qwen3_5.py`` and
    ``w8a16_gemv_norm_gated_supported`` below), so import order can never pin
    one gate and not the other.  A saved explicit true/false is honored
    verbatim as the private escape hatch it always was; unset or blank follows
    the accepted Flash-Next/SM120 default selection
    (``sglang.kernels.ops.gemm.sm120_online_fp8.fast_paths_enabled``).
    """
    explicit = envs.SGLANG_NORM_INTO_GEMV.get()
    if explicit is None:
        from sglang.kernels.ops.gemm.sm120_online_fp8 import fast_paths_enabled

        return fast_paths_enabled()
    return explicit


# Tiles for the fused-norm variant, keyed (M bucket, N, K, group size). A CTA
# recomputes the norm for its k slice once per n block, so the right geometry is
# NOT the plain one: halving the n-block count (BLOCK_N 32 -> 64) halves that
# redundancy, and the split count comes down with it to keep ~2 CTAs/SM.
# Measured with bench/h1/h1_check.py sweep (CUPTI medians, 4x-L2 working set,
# RTX PRO 6000 Blackwell Max-Q); "plain" is the unfused GEMV on the server plan
# plus the standalone `_layer_norm_fwd_1pass_kernel`.
#
#   M    plain gemv + norm      fused (64, 128, 4, 8w)
#    4   12.54 + 1.31 = 13.85   13.44   (-0.41)
#   16   12.74 + 1.50 = 14.24   14.08   (-0.16)
#
# In the server the norm launch costs 1.88 us (W4) / 1.98 us (W16) rather than
# the 1.3-1.5 us it costs here, so the fold is worth more there. Server A/B
# (2026-09-06, out_proj grid [80,6,1] -> [40,4,1], CUPTI medians per call):
#
#   profile / workload   plain + norm            fused    delta
#   W4  code-edit        13.84 + 1.88 = 15.72    14.13    -1.59
#                                                14.50    -1.22
#   W4  prose-en         13.19 + 1.88 = 15.07    14.06    -1.01
#                                                15.26    +0.19
#   W16 code-edit        13.11 + 1.98 = 15.09    14.04    -1.05
#                                                14.01    -1.08
#   W16 prose-en         12.96 + 1.98 = 14.94    13.75    -1.19
#                                                14.30    -0.64
#
# (two independent server sessions per row) -> mean -0.95, median -1.06
# us/call over 36 GDN layers = -36 us/step, and 36 of the 1596 (W4) / 2211
# (W16) kernels per step disappear.
_FUSED_NORM_BY_SHAPE = {
    (4, 2560, 6144, 128): (64, 128, 4, True, None, 8, 4),
    (16, 2560, 6144, 128): (64, 128, 4, True, None, 8, 3),
}


@functools.lru_cache(maxsize=128)
def _norm_plan(M: int, N: int, K: int, G: int, w_bytes: int, sms: int):
    """A plan for the fused-norm GEMV: same as `_plan` but with BLOCK_K == G."""
    tuned = _FUSED_NORM_BY_SHAPE.get((_m_bucket(M), N, K, G))
    if tuned is not None:
        return _fit(M, N, tuned)
    block_n, block_k, splits, use_dot, w_kn, warps, stages = _plan(
        M, N, K, False, w_bytes, sms
    )
    if block_k == G:
        return (block_n, block_k, splits, use_dot, w_kn, warps, stages)
    # Keep roughly the same CTA count when the tile gets narrower/wider in K.
    n_kb = K // G
    want = max(1, min(_MAX_SPLITS, n_kb, round(splits * block_k / G)))
    divs = [d for d in range(1, min(n_kb, _MAX_SPLITS) + 1) if n_kb % d == 0]
    splits = min(divs, key=lambda d: abs(d - want))
    return _fit(M, N, (block_n, G, splits, use_dot, w_kn, warps, stages))


def w8a16_gemv_norm_gated_supported(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    group_size: int,
) -> bool:
    """Whether this call can take the fused gated-RMSNorm + GEMV path."""
    K = x.shape[-1] if x.dim() == 2 else -1
    return bool(
        norm_into_gemv_enabled()
        and x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and z.dtype == torch.bfloat16
        and 1 <= x.shape[0] <= 16
        and tuple(z.shape) == tuple(x.shape)
        and x.stride(1) == 1
        and z.stride(1) == 1
        and group_size > 0
        and K % group_size == 0
        and norm_weight.dtype in (torch.bfloat16, torch.float32)
        and norm_weight.numel() == group_size
        and norm_weight.is_contiguous()
        and w.dim() == 2
        and w.shape[1] == K
        and w.stride(1) == 1
        and scale.numel() == w.shape[0]
    )


def w8a16_gemv_norm_gated(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    group_size: int,
    eps: float,
    sigmoid_gate: bool = False,
    out: Optional[torch.Tensor] = None,
):
    """`(gated_rms_norm(x, z) @ w^T) * scale` in one launch.

    `x`/`z` are [M, K] bf16 with K a multiple of `group_size`; the norm treats
    each contiguous `group_size` slice of a row as its own RMS group, exactly
    as `RMSNormGated(group_size, norm_before_gate=True)` does on the
    [M * K / group_size, group_size] reshape the model applies.
    """
    M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K and M <= 16
    y = _out_or_new(out, M, N, x.device)
    s = scale.reshape(-1)
    if s.dtype != torch.float32 or not s.is_contiguous():
        s = s.contiguous().float()
    # The kernel upcasts the norm weight itself, exactly as the standalone
    # layer-norm kernel does, so pass it through untouched (no per-call cast
    # kernel would otherwise get baked into the CUDA graph).
    nw = norm_weight
    cfg = _norm_plan(M, N, K, group_size, w.element_size(), _num_sms(x.device))
    assert cfg[1] == group_size, "fused-norm GEMV needs BLOCK_K == group_size"
    _launch(
        x,
        w,
        s,
        y,
        M,
        N,
        K,
        s.numel() > 1,
        cfg,
        norm=(z, nw, group_size, float(eps), bool(sigmoid_gate)),
    )
    return y


def bf16_gemv(
    x: torch.Tensor,
    w: torch.Tensor,
    cfg=None,
    out: Optional[torch.Tensor] = None,
    out2: Optional[torch.Tensor] = None,
    split_n: int = 0,
):
    """Skinny BF16 GEMM y = x @ w^T for tiny-N linears where cuBLAS picks a poor kernel.

    Donor ``w8a16_gemv.py:826`` @5105985116eb (Apache-2.0), byte for byte.
    x: [M,K] bf16 (M<=16); w: [N,K] bf16. ``out2``/``split_n``: see w8a16_gemv.
    """
    M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K and M <= 16
    y, y2, split_n = _split_dests(out, out2, split_n, M, N, x.device)
    one = _ONES.get(x.device)
    if one is None:
        one = torch.ones(1, dtype=torch.float32, device=x.device)
        _ONES[x.device] = one
    if cfg is None:
        cfg = _plan(M, N, K, w.stride(0) == 1, w.element_size(), _num_sms(x.device))
    _launch(x, w, one, y, M, N, K, False, cfg, y2, split_n)
    return y if y2 is None else (y, y2)
