# Direct port of the donor's HC MIX2 (three-kernel per-branch norm + low-rank
# mix), unmodified except where noted:
#   aiueo52/sglang-rtxpro6000 @5105985116eb00dea8e6138aabeb5363387cb9de,
#   python/sglang/srt/layers/hc_mix2_triton.py (Apache-2.0). Lineage per the
#   public companion documentation aiueo52/flash-next-rtxpro6000
#   docs/optimizations.md C8 (`b20ce6eebb`, `f685bf517d`, `081e8d9b09`).
#
# `_hc_branch_stats_kernel` (K0), `_hc_down_kernel` (K1), `_hc_up_kernel` (K2)
# and `hc_norm_mix2` carry the donor's math, launch geometry (the pinned FP8
# sweep values, which is the only weight format this fork resident-quantises)
# and PDL ordering. The deviations, all deletions or a narrower gate:
#   * the FUSE_APPLY / FUSE_GATE / FUSE_SHARED stages and the `after_normed`
#     hook are not carried; they belong to the early-gate, combine_then_mix and
#     shared-expert items, which are out of scope here;
#   * only the donor's pinned `stats_mode="norm"` mode is carried, so K0 always
#     materialises `normed` and K1 always reads it. The `inv_rms` buffer, the
#     "stats"/"redundant" branches of K0/K1 and `SINGLE_TILE`/`READ_NORMED`/
#     `REDUNDANT_STATS` go with them (the launcher pins BLOCK_S >= HS, so K0 was
#     single-tile by construction), as do the arguments they alone consumed;
#   * `quantize_hc_mix2_weights_fp8` / `dequantize_hc_mix2_weight` /
#     `_maybe_build_mix2_fp8` are not carried: this fork already ships a second
#     resident copy of nothing. The mix weights arrive as E4M3 + one FP32 scale
#     per output row from `sglang.kernels.ops.gemm.sm120_online_fp8` and are
#     handed to the kernels as they stand, so no clamped re-quantisation
#     (`1e-12` there vs. this base's `1e-8` ingestion) ever happens;
#   * the donor's `HCMix2Config`/`SGLANG_HC_MIX2*` env sweeps are replaced by
#     the pinned `_FP8_*` constants plus an automatic gate: the caller adds no
#     switch, and `hc_norm_mix2_supported` narrows the path to this base's
#     exact-SM120 rowwise-FP8 HC4 / hidden 2560 / rank 320 shapes, to
#     contiguous CUDA BF16 activations of 1..16 rows, and to what the kernels
#     cannot guarantee (K1's atomics under deterministic inference, and TP > 1,
#     which keeps the existing path until hardware says otherwise).
"""Non-persistent HC per-branch norm + low-rank mix for decode-size batches.

The persistent variant in ``hc_mix_triton`` serializes the down projection, a
grid barrier, and the up projection inside one launch, so at M<=16 neither
weight stream ever runs at full rate. This module splits the same math into
three independent launches, each of which streams one thing:

* K0 (grid = M * hc) -- one CTA per (row, branch): the sum of squares, the
  ``inv_rms`` it implies, the bf16 ``normed`` row chunk, and a slice of the
  fp32 split-K workspace cleared for K1's atomics. Folding the clear in here
  is what keeps the graph free of a separate memset node.
* K1 (grid = k_groups x n_blocks) -- ``BLOCK_G`` chunks of ``BLOCK_K`` columns
  against the matching ``w_down`` tile, accumulated in registers, scaled by the
  per-n fp32 row scale and pushed to ``t_raw`` with one device-scope atomic per
  CTA.
* K2 (grid = j_blocks) -- ``silu(t_raw / hc)`` rounded to bf16, one ``tl.dot``
  per r-chunk against ``w_up``, scaled by the per-j fp32 row scale, then the
  sigmoid-gated mean over the hc branches.

A CTA's whole column range must lie inside one branch, hence the
``hs % (BLOCK_K * BLOCK_G) == 0`` requirement: one ``inv_rms`` per row suffices.

The two mix weights (6.5 MB each in bf16) are the whole cost of K1 and K2 at
decode widths, so they are carried as fp8 e4m3 with one fp32 scale per output
row; K1 scales its fp32 partial by the per-n scale before the atomic add and K2
scales the finished accumulator by the per-j scale, both exact w.r.t. the split.
``normed`` never touches the mix weights, so it is bit-identical either way.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.triton_pdl import PDL, pdl_trigger, pdl_wait

_HC_MIX2_MAX_ROWS = 16

# Donor FP8 geometry: bench_hc_mix2_fp8.py medians on the HC shapes. K1's wide
# n block is what fp8 buys -- the same 2 KB tile spans 64 lowrank columns
# instead of 16, so K1 needs a quarter of the CTAs with four k-chunks in flight.
_FP8_BLOCK_N = 64
_FP8_BLOCK_K = 128
_FP8_BLOCK_G = 4
_FP8_DOWN_WARPS = 8
_FP8_DOWN_STAGES = 4
_FP8_BLOCK_J = 16
_FP8_BLOCK_R = 64
_FP8_UP_WARPS = 4
_FP8_UP_STAGES = 4
# K0 (HCMix2Config defaults): one tile covers a whole branch, which is what
# keeps the normalised row in registers.
_STATS_BLOCK = 4096
_STATS_WARPS = 8

# The swept (hc_count, hidden_size, hc_lowrank) of this base's Flash-Next
# runtime. The gate pins them: this port carries the donor's measured shapes,
# not a general HC solver, and anything else keeps the existing path.
_HC_MIX2_HC = 4
_HC_MIX2_HS = 2560
_HC_MIX2_LOWRANK = 320
_EXACT_SM120 = (12, 0)


@triton.jit
def _hc_branch_stats_kernel(
    x_ptr,
    norm_w_ptr,
    normed_ptr,
    t_raw_ptr,
    num_tasks,
    zero_span,
    K,
    HS,
    eps,
    HC: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ZERO_BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    pid = tl.program_id(0)
    m = pid // HC
    c = pid % HC
    base = m * K + c * HS
    offs = tl.arange(0, BLOCK_S)
    # `x` is the previous kernel's output; nothing above this point touches
    # memory, so the CTAs can be scheduled while that kernel drains.
    pdl_wait(USE_PDL)

    # One branch fits in one tile: issue the x and weight loads together so
    # the weight latency overlaps the sum-of-squares reduction instead of
    # starting a second dependent round trip after it.
    mask_s = offs < HS
    x = tl.load(x_ptr + base + offs, mask=mask_s, other=0.0).to(tl.float32)
    w = tl.load(norm_w_ptr + c * HS + offs, mask=mask_s, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / HS + eps)
    nrm = (x * inv_rms * (1.0 + w)).to(normed_ptr.dtype.element_ty)
    tl.store(normed_ptr + base + offs, nrm, mask=mask_s)

    offs_z = tl.arange(0, ZERO_BLOCK)
    for z0 in range(pid * ZERO_BLOCK, zero_span, num_tasks * ZERO_BLOCK):
        idx = z0 + offs_z
        tl.store(t_raw_ptr + idx, 0.0, mask=idx < zero_span)
    # K1 atomically accumulates into t_raw; its own gdc_wait keeps its stores
    # behind this clear, so triggering here only lets it start scheduling.
    pdl_trigger(USE_PDL)


@triton.jit
def _hc_down_kernel(
    w_down_ptr,
    normed_ptr,
    t_raw_ptr,
    s_down_ptr,
    num_rows,
    K,
    LOWRANK,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_G: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    kg = tl.program_id(0)
    nb = tl.program_id(1)
    pdl_wait(USE_PDL)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows
    k0 = kg * (BLOCK_K * BLOCK_G)
    offs_k = tl.arange(0, BLOCK_K)

    n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n < LOWRANK
    acc = tl.zeros((ROWS, BLOCK_N), dtype=tl.float32)
    for g in tl.range(0, BLOCK_G, num_stages=NUM_STAGES):
        k = k0 + g * BLOCK_K + offs_k
        normed = tl.load(
            normed_ptr + offs_m[:, None] * K + k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        w_down = tl.load(
            w_down_ptr + n[:, None] * K + k[None, :], mask=mask_n[:, None], other=0.0
        )
        acc = tl.dot(normed, tl.trans(w_down.to(tl.bfloat16)), acc)
    # One fp32 scale per output row n. The split-K partials are summed with
    # atomics, and scaling is linear, so applying it here is exact.
    acc = acc * tl.load(s_down_ptr + n, mask=mask_n, other=0.0)[None, :]
    tl.atomic_add(
        t_raw_ptr + offs_m[:, None] * LOWRANK + n[None, :],
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
        sem="relaxed",
        scope="gpu",
    )
    pdl_trigger(USE_PDL)


@triton.jit
def _hc_up_kernel(
    normed_ptr,
    w_up_ptr,
    t_raw_ptr,
    out_ptr,
    s_up_ptr,
    num_rows,
    K,
    HS,
    LOWRANK,
    inv_hc,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    jb = tl.program_id(0)
    pdl_wait(USE_PDL)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows
    offs_j = tl.arange(0, BLOCK_J)
    offs_c = tl.arange(0, HC)
    j = jb * BLOCK_J + offs_j
    mask_j = j < HS
    cj = tl.reshape(offs_c[:, None] * HS + j[None, :], (HC * BLOCK_J,))
    mask_cj = tl.reshape(
        tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,)
    )

    acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
    offs_r = tl.arange(0, BLOCK_R)
    for r0 in tl.range(0, LOWRANK, BLOCK_R, num_stages=NUM_STAGES):
        r = r0 + offs_r
        mask_r = r < LOWRANK
        a = tl.load(
            t_raw_ptr + offs_m[:, None] * LOWRANK + r[None, :],
            mask=mask_m[:, None] & mask_r[None, :],
            other=0.0,
        )
        a = a * inv_hc
        t = (a * tl.sigmoid(a)).to(out_ptr.dtype.element_ty)
        w = tl.load(
            w_up_ptr + cj[:, None] * LOWRANK + r[None, :],
            mask=mask_cj[:, None] & mask_r[None, :],
            other=0.0,
        )
        acc = tl.dot(t, tl.trans(w.to(tl.bfloat16)), acc)

    # One fp32 scale per output column (the flattened c*HS + j row of w_up);
    # the whole r reduction for that column lives in this CTA, so scaling the
    # finished accumulator is exact.
    acc = acc * tl.load(s_up_ptr + cj, mask=mask_cj, other=0.0)[None, :]
    gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
    xg = tl.load(
        normed_ptr
        + offs_m[:, None, None] * K
        + offs_c[None, :, None] * HS
        + j[None, None, :],
        mask=mask_m[:, None, None] & mask_j[None, None, :],
        other=0.0,
    ).to(tl.float32)
    out = tl.sum(gate * xg, axis=1) * inv_hc
    tl.store(
        out_ptr + offs_m[:, None] * HS + j[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_j[None, :],
    )
    pdl_trigger(USE_PDL)


@functools.lru_cache(maxsize=None)
def _is_exact_sm120(device) -> bool:
    try:
        return torch.cuda.get_device_capability(device) == _EXACT_SM120
    except Exception:
        return False


_tp_size_cached = None


def _tp_size() -> int:
    """Tensor-parallel width, as configured. MIX2 stays off above 1 for now.

    Only a successful read is cached. The gate is consulted before activation
    eligibility, so a probe made while no ServerArgs is published must default
    for that call alone: latching 1 there would strand a later TP2 rank on
    MIX2 for the life of the process.
    """
    global _tp_size_cached
    if _tp_size_cached is not None:
        return _tp_size_cached
    try:
        from sglang.srt.runtime_context import get_server_args

        _tp_size_cached = int(get_server_args().tp_size)
    except Exception:
        # No runtime context yet (unit tests, tools, early probes): fall back
        # without leaving the cache set, so the next call reads the real one.
        return 1
    return _tp_size_cached


def _rowwise_fp8_pair(w_down: torch.Tensor, w_up: torch.Tensor, k: int, lowrank: int):
    """(s_down, s_up) of the resident rowwise FP8 pair, or None if this is not one.

    The scales are read off the live Parameter tensors the caller passed, never
    cached: a weight update replaces the Parameter and its scale attribute
    together, so checking and launching on the same two objects cannot pair a
    new weight with an old scale. A half-quantised pair raises, the same loud
    failure hc_mix_triton._rowwise_fp8_pair uses.
    """
    if w_down.dtype != torch.float8_e4m3fn and w_up.dtype != torch.float8_e4m3fn:
        # BF16 mix weights (online FP8 off) stay on the existing fused mix.
        return None
    from sglang.kernels.ops.gemm.sm120_online_fp8 import rowwise_scale_of

    s_down, s_up = rowwise_scale_of(w_down), rowwise_scale_of(w_up)
    if s_down is None or s_up is None:
        raise RuntimeError(
            "HC MIX2 requires the resident rowwise-FP8 mix weights to carry "
            "their FP32 row scales"
        )
    if (
        w_down.dtype != torch.float8_e4m3fn
        or w_up.dtype != torch.float8_e4m3fn
        or tuple(w_down.shape) != (lowrank, k)
        or tuple(w_up.shape) != (k, lowrank)
        or not all(w.is_contiguous() for w in (w_down, w_up))
        or not all(
            s.dim() == 1
            and s.dtype == torch.float32
            and s.numel() == n
            and s.is_contiguous()
            for s, n in ((s_down, lowrank), (s_up, k))
        )
    ):
        # K1 indexes s_down by lowrank row and K2 s_up by the flattened hc*hs
        # row, so a scale that does not line up with the weight is not MIX2.
        return None
    return s_down, s_up


def hc_norm_mix2_supported(
    hyper_input: torch.Tensor,
    norm_w: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
) -> bool:
    """Whether `hc_norm_mix2` can serve this call; anything else falls back.

    The donor's guard plus this fork's narrowing: exact SM120, the swept HC4 /
    2560 / 320 shapes, the resident rowwise-FP8 pair, and no atomics under
    deterministic inference (K1's split-K order is not reproducible across
    launches) or under tensor parallelism.
    """
    from sglang.srt.layers.hc_mix_triton import _deterministic_inference

    lowrank = _HC_MIX2_LOWRANK
    if _deterministic_inference() or _tp_size() > 1:
        return False
    if not (
        hyper_input.is_cuda
        and hyper_input.dim() == 2
        and hyper_input.dtype == torch.bfloat16
        and 1 <= hyper_input.shape[0] <= _HC_MIX2_MAX_ROWS
        and hyper_input.shape[1] == hc * hs
        and hyper_input.is_contiguous()
        and hc == _HC_MIX2_HC
        and hs == _HC_MIX2_HS
        and hs % (_FP8_BLOCK_K * _FP8_BLOCK_G) == 0
        and lowrank % _FP8_BLOCK_N == 0
        and _is_exact_sm120(hyper_input.device)
    ):
        return False
    if not (
        norm_w.is_cuda
        and norm_w.device == hyper_input.device
        and norm_w.dtype == torch.bfloat16
        and norm_w.is_contiguous()
        and norm_w.numel() == hc * hs
    ):
        # K0 reads the norm weight as [hc * hs] (per-branch); a shared-weight
        # norm is a different kernel and belongs to the existing path.
        return False
    if _rowwise_fp8_pair(w_down, w_up, hc * hs, lowrank) is None:
        return False
    return all(w.is_cuda and w.device == hyper_input.device for w in (w_down, w_up))


def hc_norm_mix2(
    hyper_input: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
    s_down: Optional[torch.Tensor] = None,
    s_up: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-branch Gemma RMSNorm followed by the gated low-rank mix.

    Returns ``(mixed[M, hs], normed[M, hc * hs])`` in the input dtype. ``t_raw``
    is accumulated with device-scope atomics, so the summation order of the
    down projection is not reproducible across launches.

    ``w_down``/``w_up`` are fp8 e4m3 (weight-only), in which case ``s_down``
    (one fp32 scale per lowrank row) and ``s_up`` (one per hc*hs row) are
    required; pass them to test the kernels against explicit scales, otherwise
    they are read off the resident Parameters. Every buffer is allocated per
    call, so two in-flight calls never share the split-K workspace and a
    captured graph keeps them in its own pool.
    """
    rows, k = hyper_input.shape
    lowrank = w_down.shape[0]
    if s_down is None or s_up is None:
        pair = _rowwise_fp8_pair(w_down, w_up, k, lowrank)
        if pair is None:
            raise RuntimeError(
                "HC MIX2 takes the resident rowwise-FP8 mix weights and their "
                "FP32 row scales; check hc_norm_mix2_supported first"
            )
        s_down, s_up = pair
    device = hyper_input.device
    rows_pad = _HC_MIX2_MAX_ROWS

    mixed = torch.empty((rows, hs), dtype=hyper_input.dtype, device=device)
    normed = torch.empty_like(hyper_input)
    t_raw = torch.empty((rows_pad, lowrank), dtype=torch.float32, device=device)
    num_tasks = rows * hc
    zero_span = rows * lowrank
    _hc_branch_stats_kernel[(num_tasks,)](
        hyper_input,
        norm_w,
        normed,
        t_raw,
        num_tasks,
        zero_span,
        k,
        hs,
        eps,
        HC=hc,
        BLOCK_S=_STATS_BLOCK,
        ZERO_BLOCK=max(64, triton.next_power_of_2(zero_span // num_tasks)),
        USE_PDL=PDL,
        launch_pdl=PDL,
        num_warps=_STATS_WARPS,
    )
    k_groups = k // (_FP8_BLOCK_K * _FP8_BLOCK_G)
    n_blocks = triton.cdiv(lowrank, _FP8_BLOCK_N)
    _hc_down_kernel[(k_groups, n_blocks)](
        w_down,
        normed,
        t_raw,
        s_down,
        rows,
        k,
        lowrank,
        ROWS=rows_pad,
        BLOCK_N=_FP8_BLOCK_N,
        BLOCK_K=_FP8_BLOCK_K,
        BLOCK_G=_FP8_BLOCK_G,
        NUM_STAGES=_FP8_DOWN_STAGES,
        USE_PDL=PDL,
        launch_pdl=PDL,
        num_warps=_FP8_DOWN_WARPS,
    )
    _hc_up_kernel[(triton.cdiv(hs, _FP8_BLOCK_J),)](
        normed,
        w_up,
        t_raw,
        mixed,
        s_up,
        rows,
        k,
        hs,
        lowrank,
        1.0 / hc,
        ROWS=rows_pad,
        HC=hc,
        BLOCK_J=_FP8_BLOCK_J,
        BLOCK_R=_FP8_BLOCK_R,
        NUM_STAGES=_FP8_UP_STAGES,
        USE_PDL=PDL,
        launch_pdl=PDL,
        num_warps=_FP8_UP_WARPS,
    )
    return mixed, normed
