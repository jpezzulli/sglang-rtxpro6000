# SPDX-License-Identifier: Apache-2.0
# Adapted from aiueo52/sglang-rtxpro6000 (flash-next-fast), commit
# 5105985116eb00dea8e6138aabeb5363387cb9de, file
# python/sglang/srt/layers/quantization/w8a16_gemv.py (Apache-2.0), including the
# per-shape tile table / in-launch split-K fixup of patch 0026 and the per-stream
# split-K scratch slots of patch 0041 of aiueo52/flash-next-rtxpro6000 @
# 524af49abcca66fcb4377ba8297022804535fccf.  Donor-only PDL, fused-norm,
# fused gate_up+SiLU, bf16-GEMV and two-destination-store paths are dropped; the
# resident rowwise-FP8 output heads and the stored block-MXFP8 dense projections
# share what is left, and the [1, 32] scale handling is Penny's, not donor code.
"""Low-row (M <= 16) W8A16 FP8 GEMV for SM120 rowwise-FP8 heads and dense MXFP8.

Two scale layouts share this one kernel and its split-K machinery:

- rowwise (the output heads): ``y[M,N](bf16) = x[M,K](bf16) @ w[N,K](fp8 e4m3)^T
  * scale[N](fp32)``;
- block-MXFP8 (``SF_GROUP = 32``, the eligible dense projections): ``weight`` is
  the *stored* MXFP8 tensor and ``scale`` is ``weight_scale_inv[N, K // 32]``
  uint8 UE8M0, so the per-row-per-32-column scale is folded into the weight tile
  inside the K loop and nothing is requantized.  ``SF_GROUP = 0`` is rowwise.

Weight-only FP8 in both cases: the activation is never quantized and the FP32
accumulation contract of the original ``_rowwise_fp8_gemv_kernel`` is unchanged.

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

The block-MXFP8 variant is Penny's adaptation, not donor code: the donor's
generic ``fp8.py`` dispatch only ever hands this kernel a per-output-channel
scale.  The per-element scale gather (``k // 32``) is what lets one K loop serve a
scale granularity the donor kernel never saw, and the UE8M0 decode is the same
exponent-bit trick the in-tree MXFP8 loaders use (``fp8.py``'s DeepGEMM branch,
``mxfp8_quant.from_mxfp8``).
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl

_UNSET = object()

#: Candidate gate, default OFF.  Off, the rowwise heads keep the original kernel.
GEMV_ENV = "SGLANG_FP8_W8A16_GEMV"

#: Separate candidate gate for the dense block-MXFP8 path, default OFF.  Kept
#: apart from GEMV_ENV so the output-head candidate and the dense one can be
#: reviewed and A/B'ed independently; off, every dense MXFP8 call stays on the
#: qualified W8A8 dispatch in srt/layers/quantization/fp8.py.
MX_GEMV_ENV = "SGLANG_FP8_MXFP8_W8A16_GEMV"

#: MXFP8 (OCP) fixes the weight block at [1, 32]: one UE8M0 byte per output row
#: per 32 K columns, stored as ``weight_scale_inv[N, K // 32]`` uint8.
MXFP8_SF = 32

#: The only part this candidate is qualified on, the same exact-SM120 rule as
#: sm120_online_fp8.configure_online_fp8.  The dense MXFP8 dispatch also runs on
#: other GPUs, so the candidate has to name its own hardware bound.
EXACT_SM120 = (12, 0)

#: The donor kernel's own row limit; it is deliberately not raised.  Bigger
#: batches (C6's 24-row verification, prefill) stay on the original path.
MAX_ROWS = 16

_WS_FLOATS = 1 << 21
_WS_COUNTERS = 4096
_MAX_SPLITS = 32
_N_SLOTS = 2
_SLOT = 0
_WS: dict = {}
_LAYER_SLOTS: dict = {}
_LAYER_ROTATION = 0
_CAPABILITIES: dict = {}


def lowrow_gemv_enabled() -> bool:
    """Whether the candidate GEMV may take a rowwise-FP8 head call."""
    return os.environ.get(GEMV_ENV, "0") == "1"


def mxfp8_gemv_enabled() -> bool:
    """Whether the dense block-MXFP8 candidate may take an eligible call."""
    return os.environ.get(MX_GEMV_ENV, "0") == "1"


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


def _slot_for_launch(device, owner) -> int:
    """Split-K scratch slot for one call site (the linear that issued the launch).

    Two dense launches can be resident at the same time — ``in_proj_qkvz`` runs on
    the main stream while ``in_proj_ba`` is captured on the alt stream — and
    launches that share one counter array corrupt each other's partials.  The slot
    is keyed per call site rather than per stream because a graph's capture stream
    changes with every captured batch size: a stream-keyed slot would quietly strip
    split-K from all but the first two graphs.  Adjacent registrations land on
    different slots, which is the case the fork needs.

    ponytail: 2 rotating slots and an adjacency assumption; give the scratch a real
    arena (or per-launch counter regions) if a second fork ever carries two dense
    linears at once.
    """
    global _LAYER_ROTATION
    key = (str(device), id(owner))
    slot = _LAYER_SLOTS.get(key)
    if slot is None:
        slot = _LAYER_ROTATION % _N_SLOTS
        _LAYER_ROTATION += 1
        _LAYER_SLOTS[key] = slot
    return slot


def ue8m0_to_float32(scale_bytes: torch.Tensor) -> torch.Tensor:
    """Decode stored UE8M0 bytes to fp32, i.e. 2**(byte - 127), via the exponent.

    The same decode ``fp8.py``'s DeepGEMM branch and ``mxfp8_quant.from_mxfp8``
    use, so the kernel, the loaders and the checks agree on one arithmetic.
    """
    return (scale_bytes.to(torch.int32) << 23).view(torch.float32)


def dequantize_mxfp8_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reference dequant of a stored block-MXFP8 weight: every fp8 value times
    its row's 32-column UE8M0 scale, in fp32, rounded once to ``dtype``.

    Power-of-two scaling keeps e4m3's 4-bit significand inside bf16's 8, so the
    bf16 result is the exact value the kernel's weight tile holds — the
    candidate's only deviation from this reference is the order of the fp32 K
    reduction (and, unlike the qualified W8A8 path, no activation is quantized).
    Reference path for checks and fallbacks, not a hot loop: it materializes
    ``weight_scale`` expanded 32x along K.
    """
    if weight.dim() != 2 or weight_scale.dim() != 2:
        raise TypeError("block-MXFP8 dequant needs 2-D weight and scale tensors")
    rows, columns = weight.shape
    if weight_scale.shape != (rows, columns // MXFP8_SF):
        raise ValueError(
            f"block-MXFP8 scale {tuple(weight_scale.shape)} does not match weight "
            f"{tuple(weight.shape)} at block [1,{MXFP8_SF}]"
        )
    descale = ue8m0_to_float32(weight_scale).repeat_interleave(MXFP8_SF, dim=1)
    return (weight.float() * descale).to(dtype)


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
    stride_sn,
    stride_sk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_PAD: tl.constexpr,
    SPLITS: tl.constexpr,
    EVEN_K: tl.constexpr,
    USE_DOT: tl.constexpr,
    SF_GROUP: tl.constexpr,
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
        if SF_GROUP:
            # Block-MXFP8: one UE8M0 byte per (row, SF_GROUP columns).  The byte
            # is gathered per element at kk // SF_GROUP, so BLOCK_K needs no
            # alignment with the scale group and a ragged K tail stays exact; the
            # gathered tile is N*K/32 unique bytes and stays in L1/L2.
            sp = (
                s_ptr
                + offs_n[:, None] * stride_sn
                + (kk[None, :] // SF_GROUP) * stride_sk
            )
            if EVEN_K:
                sb = tl.load(sp, mask=n_mask[:, None], other=0)
            else:
                sb = tl.load(sp, mask=n_mask[:, None] & k_mask[None, :], other=0)
            # fp32 * power-of-two is exact for the 4-bit e4m3 significand, and
            # bf16 holds that product exactly, so this is a dequant — not a
            # requant — of the stored weight.
            w = (
                w.to(tl.float32) * (sb.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            ).to(tl.bfloat16)
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
        if not SF_GROUP:
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
        if not SF_GROUP:
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
def _plan(M: int, N: int, K: int, sms: int, tuned: bool = True):
    """(BLOCK_N, BLOCK_K, SPLITS, USE_DOT, num_warps, num_stages) for one call.

    ``tuned`` selects the donor's head-shape table.  The dense block-MXFP8 caller
    passes False: those tiles were measured on the rowwise output heads, and a
    dense projection that happens to share an (N, K) is not that measurement.
    """
    table = _BY_SHAPE.get((_m_bucket(M), N, K)) if tuned else None
    if table is not None:
        return _fit(M, N, table)
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


def _device_capability(device):
    key = (str(device), device.type)
    capability = _CAPABILITIES.get(key)
    if capability is None:
        capability = (
            torch.cuda.get_device_capability(device) if device.type == "cuda" else ()
        )
        _CAPABILITIES[key] = capability
    return capability


def _launch(
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    cfg,
    sf_group: int,
    slot: int,
) -> None:
    """One (N / BLOCK_N, SPLITS) grid; the split-K fixup rides in the same launch."""
    M, K = hidden_2d.shape
    N = weight.shape[0]
    block_n, block_k, splits, use_dot, num_warps, num_stages = cfg
    ws, cnt = _workspace_slot(hidden_2d.device, slot)
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
        scale.stride(0) if scale.dim() == 2 else 0,
        scale.stride(1) if scale.dim() == 2 else 1,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        M_PAD=16 if (use_dot or M > 1) else 1,
        SPLITS=splits,
        EVEN_K=K % block_k == 0,
        USE_DOT=use_dot,
        SF_GROUP=sf_group,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def lowrow_fp8_gemv(
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Rowwise-FP8 skinny GEMM for the target/draft output heads. bf16 out."""
    M, K = hidden_2d.shape
    N = weight.shape[0]
    device = hidden_2d.device
    out = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    block_n, block_k, splits, use_dot, num_warps, num_stages = _plan(
        M, N, K, _num_sms(device)
    )
    use_dot = use_dot or M > 1
    _launch(
        hidden_2d,
        weight,
        scale,
        out,
        (block_n, block_k, splits, use_dot, num_warps, num_stages),
        0,
        _SLOT,
    )
    return out


def lowrow_mxfp8_gemv_supported(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    capability=_UNSET,
) -> bool:
    """Whether this call fits the dense block-MXFP8 candidate (fail-safe gate).

    False routes the call to the untouched MXFP8 dispatch, so this is a contract,
    not an approximation: the stored representation has to arrive as MXFP8 did —
    fp8 e4m3 ``weight[N, K]`` with K contiguous plus UE8M0 ``weight_scale_inv``
    bytes at ``[N, K // 32]``.  No requantization, no second weight format, no
    activation quantization.  The caller may pass ``capability`` to state the part
    it is asking about (the CPU checks do); otherwise it is read from the device.
    """
    if not mxfp8_gemv_enabled():
        return False
    if hidden_states.dim() != 2 or weight.dim() != 2 or weight_scale is None:
        return False
    if weight_scale.dim() != 2:
        return False
    rows, columns = hidden_states.shape
    # C6's 24-row verification, 48-row mixed batches and prefill all stay wider
    # than this and keep their ordinary kernels; MAX_ROWS is the donor limit.
    if not 1 <= rows <= MAX_ROWS:
        return False
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.float8_e4m3fn:
        return False
    if weight_scale.dtype != torch.uint8:
        return False
    n, k = weight.shape
    if k != columns or k % MXFP8_SF or weight_scale.shape != (n, k // MXFP8_SF):
        return False
    # The scale bytes must mean UE8M0, which is what the MXFP8 loaders promise.
    if not getattr(weight_scale, "format_ue8m0", False):
        return False
    if weight.stride(1) != 1 or weight_scale.stride(1) != 1:
        return False
    if hidden_states.stride(1) != 1:
        return False
    if len({hidden_states.device, weight.device, weight_scale.device}) != 1:
        return False
    if capability is _UNSET:
        capability = _device_capability(hidden_states.device)
    return tuple(capability) == EXACT_SM120


def lowrow_mxfp8_gemv(
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    owner: object = None,
) -> torch.Tensor | None:
    """Eligible dense block-MXFP8 linear at M <= MAX_ROWS; None = not eligible.

    Returns None instead of raising so the one call site in fp8.py can fall
    through to the qualified W8A8 dispatch in the same statement.  ``owner`` is the
    layer, used only to keep its launches off another layer's split-K scratch.
    """
    if not lowrow_mxfp8_gemv_supported(hidden_2d, weight, weight_scale):
        return None
    M, K = hidden_2d.shape
    N = weight.shape[0]
    device = hidden_2d.device
    block_n, block_k, splits, use_dot, num_warps, num_stages = _plan(
        M, N, K, _num_sms(device), False
    )
    use_dot = use_dot or M > 1
    out = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    _launch(
        hidden_2d,
        weight,
        weight_scale,
        out,
        (block_n, block_k, splits, use_dot, num_warps, num_stages),
        MXFP8_SF,
        _slot_for_launch(device, owner),
    )
    return out
