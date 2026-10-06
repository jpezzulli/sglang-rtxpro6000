# SPDX-License-Identifier: Apache-2.0
# Adapted from aiueo52/sglang-rtxpro6000 (flash-next-fast), commit
# 5105985116eb00dea8e6138aabeb5363387cb9de, file
# python/sglang/srt/layers/quantization/w8a16_gemv.py (Apache-2.0), including the
# per-shape tile table / in-launch split-K fixup of patch 0026 and the per-stream
# split-K scratch slots of patch 0041 of aiueo52/flash-next-rtxpro6000 @
# 524af49abcca66fcb4377ba8297022804535fccf.  Donor-only PDL, fused-norm,
# fused gate_up+SiLU, bf16-GEMV and two-destination-store paths are dropped: this
# file serves the resident rowwise-FP8 output heads only.
"""Low-row (M <= 16) W8A16 FP8 GEMV for the SM120 online-FP8 output heads.

``y[M,N](bf16) = x[M,K](bf16) @ w[N,K](fp8 e4m3)^T * scale[N](fp32)``.  Weight-only
FP8: the activation is never quantized and the FP32 accumulation contract of the
original ``_rowwise_fp8_gemv_kernel`` is unchanged.

Two donor properties carry over and both decide correctness, so they are kept
verbatim:

- Split-K is folded into the same launch: every CTA writes an fp32 partial and
  bumps a per-N-block counter; the CTA that observes the last increment sums the
  partials, applies the per-row scale and writes bf16 ``y``, then stores 0 back to
  the counter.  No memset between calls means the launch is safe to capture in a
  CUDA graph and replay (a replay re-runs the same increments to the same reset).
- That scratch/counter array is owned per (device, slot).  Two split-K launches
  that run concurrently on different streams and share one array corrupt each
  other's partials, so overlapping work takes an independent slot with
  ``with scratch_slot(1):`` and ``prealloc`` materializes every slot before the
  first capture — allocating inside a capture would hand the buffer to that
  graph's private pool and leave the other graphs pointing into freed memory.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl

#: Candidate gate, default OFF.  Off, the rowwise heads keep the original kernel.
GEMV_ENV = "SGLANG_FP8_W8A16_GEMV"

#: The donor kernel's own row limit; it is deliberately not raised.  Bigger
#: batches (C6's 24-row verification, prefill) stay on the original path.
MAX_ROWS = 16

_WS_FLOATS = 1 << 21
_WS_COUNTERS = 4096
_MAX_SPLITS = 32
_N_SLOTS = 2
_SLOT = 0
_WS: dict = {}


def lowrow_gemv_enabled() -> bool:
    """Whether the candidate GEMV may take a rowwise-FP8 head call."""
    return os.environ.get(GEMV_ENV, "0") == "1"


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
            raise RuntimeError(
                "SM120 low-row FP8 GEMV split-K scratch must be materialized by "
                "prealloc() before CUDA graph capture, not allocated into the "
                "graph's private memory pool"
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
    """Materialize every per-device scratch slot before any CUDA graph is captured."""
    for slot in range(_N_SLOTS):
        _workspace_slot(device, slot)


@triton.jit
def _w8a16_gemv_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    ws_ptr,
    cnt_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_PAD: tl.constexpr,
    SPLITS: tl.constexpr,
    EVEN_K: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, M_PAD)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    m_mask = offs_m < M
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    for k0 in range(pid_k * BLOCK_K, K, SPLITS * BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < K
        # The head weight is always [N, K]-contiguous, so the natural tile is
        # [BLOCK_N, BLOCK_K] and tl.dot takes it transposed.
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
            acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
        else:
            # M == 1: broadcast-multiply and reduce instead of padding to a
            # 16-row mma, which shrinks the accumulator and the split-K
            # partials 16-fold.  Only legal because M_PAD is 1 here.
            if EVEN_K:
                xv = tl.load(x_ptr + kk * stride_xk).to(tl.float32)
            else:
                xv = tl.load(x_ptr + kk * stride_xk, mask=k_mask, other=0.0).to(
                    tl.float32
                )
            acc += tl.sum(w.to(tl.float32) * xv[None, :], axis=1)[None, :]

    if SPLITS == 1:
        acc = acc * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            acc.to(y_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )
        return

    # Split-K fixup in this same launch.  `.cg` keeps the partials out of the
    # non-coherent L1 so the CTA that runs the reduction sees the others' writes.
    tile = M_PAD * BLOCK_N
    slot = offs_m[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    base = ws_ptr + (pid_n * SPLITS) * tile
    tl.store(base + pid_k * tile + slot, acc, cache_modifier=".cg")
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    if done == SPLITS - 1:
        tot = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
        for s in tl.static_range(SPLITS):
            tot += tl.load(base + s * tile + slot, cache_modifier=".cg")
        tot = tot * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            tot.to(y_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )
        # Every increment for this N block has happened, so a plain store leaves
        # the counter at 0 for the next launch / graph replay.
        tl.store(cnt_ptr + pid_n, 0)


def _prev_pow2(v: int) -> int:
    return 1 << max(0, v.bit_length() - 1)


def _m_bucket(M: int) -> int:
    return 1 if M == 1 else (4 if M <= 4 else 16)


# Donor-tuned tiles for the exact Flash-Next head shapes, keyed
# (M bucket, N, K) -> (BLOCK_N, BLOCK_K, SPLITS, USE_DOT, num_warps, num_stages),
# measured on RTX PRO 6000 Blackwell Max-Q (sm_120, 188 SMs) by the donor's
# bench/w8a16v2 tuning.  The generic planner below already emits the same tile for
# these shapes except for the draft head at M == 1 and the M == 4 bucket; a
# vocab-sharded (TP2) or hot-vocab head has a different N and takes the planner.
# Re-measure before trusting these on other parts.
_BY_SHAPE = {
    (1, 32768, 2560): (32, 256, 1, False, 4, 3),  # draft head, donor 1.17x
    (4, 32768, 2560): (128, 256, 1, True, 8, 3),
    (16, 32768, 2560): (128, 256, 1, True, 8, 3),
    (1, 248320, 2560): (128, 256, 1, True, 8, 3),
    (4, 248320, 2560): (128, 256, 1, True, 8, 3),
    (16, 248320, 2560): (128, 256, 1, True, 8, 3),  # target head, donor 1.02x
}


@functools.lru_cache(maxsize=512)
def _plan(M: int, N: int, K: int, sms: int):
    """(BLOCK_N, BLOCK_K, SPLITS, USE_DOT, num_warps, num_stages) for one call."""
    tuned = _BY_SHAPE.get((_m_bucket(M), N, K))
    if tuned is not None:
        return _fit(M, N, tuned)
    # N alone fills the machine: a long BLOCK_K keeps more bytes in flight.  The
    # M == 4 bucket is the exception, where the donor measured the narrow tile
    # faster on both 2560-K shapes.
    if triton.cdiv(N, 128) * 2 >= sms and _m_bucket(M) != 4:
        if M == 1:
            return (32, 256, 1, False, 4, 3)
        # A [128, 256] bf16 tile needs 144 KB of smem, over the 99 KB sm_120 limit.
        return (128, 256, 1, True, 8, 3)
    use_dot = M > 1
    block_n = 16 if M == 1 else 32
    block_k, warps = 128, 4
    if triton.cdiv(N, 32) >= sms:
        return (block_n, block_k, 1, use_dot, warps, 3)
    # Underfilled grid: split K.  Aim for ~2 CTAs per SM, ~5 at M == 1 where an
    # M_PAD 1 partial is 16x cheaper; shrink BLOCK_K if K has too few blocks.
    per_sm = 5 if M == 1 else 2
    want = min(_MAX_SPLITS, max(1, (per_sm * sms) // triton.cdiv(N, block_n)))
    block_k = min(block_k, max(64, _prev_pow2(max(1, K // want))))
    n_kb = triton.cdiv(K, block_k)
    splits = min(want, n_kb)
    # Prefer a split count that divides the k-block count so every CTA is equal.
    for cand in range(min(n_kb, splits + 2), max(1, splits - 3), -1):
        if n_kb % cand == 0 and cand <= _MAX_SPLITS:
            splits = cand
            break
    return _fit(M, N, (block_n, block_k, splits, use_dot, warps, 3))


def _fit(M: int, N: int, cfg):
    """Drop to SPLITS=1 if the plan would not fit the preallocated scratch."""
    block_n, block_k, splits, use_dot, warps, stages = cfg
    if splits == 1:
        return cfg
    m_pad = 16 if (use_dot or M > 1) else 1
    n_blocks = triton.cdiv(N, block_n)
    if (
        splits > _MAX_SPLITS
        or n_blocks > _WS_COUNTERS
        or n_blocks * splits * m_pad * block_n > _WS_FLOATS
    ):
        return (block_n, block_k, 1, use_dot, warps, stages)
    return cfg


def lowrow_gemv_supported(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> bool:
    """Whether this call fits the candidate GEMV's supported contract (fail-safe).

    Anything False here is routed to the untouched original implementation, so
    this is shape gating, not an approximation: wider batches (C6's 24 rows),
    non-rowwise weight layouts and unusual scales all keep the qualified path.
    """
    if not lowrow_gemv_enabled():
        return False
    if hidden_states.dim() != 2 or weight.dim() != 2:
        return False
    rows, columns = hidden_states.shape
    if not 1 <= rows <= MAX_ROWS:
        return False
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.float8_e4m3fn:
        return False
    if scale is None or weight.shape[1] != columns or scale.shape != (weight.shape[0],):
        return False
    if scale.dtype != torch.float32:
        return False
    # One CTA per N block owns a contiguous K run of the weight; a strided or
    # offset weight (a transposed view, a sharded copy) is the original kernel's.
    if weight.stride(1) != 1 or scale.stride(0) != 1:
        return False
    if hidden_states.stride(1) != 1:
        return False
    devices = {hidden_states.device, weight.device, scale.device}
    if len(devices) != 1:
        return False
    return True


_NUM_SMS: dict = {}


def _num_sms(device) -> int:
    sms = _NUM_SMS.get(device)
    if sms is None:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        _NUM_SMS[device] = sms
    return sms


def lowrow_fp8_gemv(
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Rowwise-FP8 skinny GEMM for the target/draft output heads. bf16 out."""
    M, K = hidden_2d.shape
    N = weight.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=hidden_2d.device)
    block_n, block_k, splits, use_dot, num_warps, num_stages = _plan(
        M, N, K, _num_sms(hidden_2d.device)
    )
    use_dot = use_dot or M > 1
    m_pad = 16 if use_dot else 1
    ws, cnt = _workspace(hidden_2d.device)
    _w8a16_gemv_kernel[(triton.cdiv(N, block_n), splits)](
        hidden_2d,
        weight,
        scale,
        out,
        ws,
        cnt,
        M,
        N,
        K,
        hidden_2d.stride(0),
        hidden_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        M_PAD=m_pad,
        SPLITS=splits,
        EVEN_K=K % block_k == 0,
        USE_DOT=use_dot,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
