"""Softmax top-k router on 32-bit packed keys (selected automatically for
eligible Flash-Next SM120 launches; SGLANG_ROUTER_FAST_TOPK stays the private
override).

Same weights and ids as the Triton router (``moe_fused_gate(..., scoring_func="softmax")``), bit for
bit, for the call ``fused_topk`` makes on CUDA: bf16 logits, no correction bias (``get_zero_bias``
below supplies the fp32 zeros this kernel's bias pointer wants, which is the same value the Triton
router keeps in registers), no shared experts /
groups / softcap / scaling, at most 512 experts, one warp per row.

``_router_triton_kernel`` picks each expert with three dependent warp reductions: the max of the
biased logits, the lowest lane among the maxima and a masked sum that fetches the winner's weight.
With bf16 logits and a zero bias, ``biased = float(logit) + 0.0`` is a bf16 value (-0.0 becomes
+0.0), so the order of the logits is the order of the high halves of their fp32 bits, and each logit
packs with its inverted lane into one int32,

    key = v << 9 | (511 - lane),  v = 2 * (order-preserving int16 of the high half)

One int32 max per pick (15 in-thread max.s32 and one redux.sync.max.s32) gives the winner and the
tie-break, and the winner's weight is recomputed from the float decoded out of the key with the
instructions the Triton router uses for every lane (fsub, fmul by log2e + ex2.approx, div.full).

Why the results are identical:
  * the row max, the row sum and the routed sum are the Triton router's code on the same layout,
    so they reduce in the same order;
  * key order is float order with ties broken by the lower lane; NaN, which the Triton router
    floors to -1e30 for the ranking, gets the odd v between bf16 0xF14A and 0xF149, the two
    neighbours of -1e30;
  * a NaN logit makes the row sum, so every weight of the row, NaN in both kernels.
Checked bit for bit by flash-next-bench ``bench/rt1/route_bench.py --check``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl


@triton.jit
def _router_softmax_fast32_kernel(
    scores_ptr,  # [M, N] bf16 router logits
    bias_ptr,  # [N] fp32 zeros
    out_weights_ptr,  # [M, K] fp32
    out_indices_ptr,  # [M, K] int32
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    USE_PDL: tl.constexpr,
    stride_sm,
    stride_sn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
):
    # ---- _router_triton_kernel, SCORING_FUNC == 2 (softmax), BLOCK_M == 1 ----
    pid = tl.program_id(0)
    offs_m = pid * 1 + tl.arange(0, 1)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    row_ptr = scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    mask2d = mask_m[:, None] & mask_n[None, :]
    scores = tl.load(row_ptr, mask=mask2d, other=0.0).to(tl.float32)

    logit = scores
    biased = logit + bias[None, :]
    biased = tl.where(mask_n[None, :], biased, -float("inf"))
    row_max = tl.max(biased, axis=1)[:, None]  # [1, 1]
    exp_row = tl.where(mask_n[None, :], tl.exp(biased - row_max), 0.0)
    row_sum = tl.sum(exp_row, axis=1)[:, None]  # [1, 1]

    biased = tl.where(mask_n[None, :], biased, -float("inf"))

    # ---- 32-bit keys ----
    bits = biased.to(tl.int32, bitcast=True)
    h = bits >> 16  # sign-extended high half (the bf16 pattern)
    s = h ^ ((h >> 15) & 0x7FFF)  # order-preserving int16
    # NaN -> the slot of the Triton router's -1e30 floor: 2 * int16order(0xF149) - 1
    v = tl.where(biased == biased, s * 2, -58005)
    lo = (BLOCK_N - 1) - offs_n  # lower lane -> larger key on ties
    keys = (v << 9) | lo[None, :]
    gone = tl.full([1, BLOCK_N], -2147483648, tl.int32)

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    selected_vals = tl.zeros([1, BLOCK_K], dtype=tl.float32)
    selected_idx = tl.zeros([1, BLOCK_K], dtype=tl.int32)
    cur = keys
    for k in tl.static_range(K):
        kmax = tl.max(cur, axis=1)[:, None]  # [1, 1] int32
        cur = tl.where(cur == kmax, gone, cur)
        win_lane = (BLOCK_N - 1) - (kmax & 511)
        wv = kmax >> 9
        ws = wv >> 1
        wh = ws ^ ((ws >> 15) & 0x7FFF)
        wb = (wh << 16).to(tl.float32, bitcast=True)
        wb = tl.where((wv & 1) != 0, -1e30, wb)
        win_activated = (
            tl.exp(wb - row_max) / row_sum  # the Triton router's activated[win]
        )
        slot = offs_k[None, :] == k
        selected_vals = tl.where(slot, win_activated, selected_vals)
        selected_idx = tl.where(slot, win_lane, selected_idx)

    routed_sum = tl.sum(tl.where(mask_k[None, :], selected_vals, 0.0), axis=1)[:, None]

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    if RENORMALIZE:
        norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)
        selected_vals = selected_vals / norm

    out_w_ptr = (
        out_weights_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk
    )
    out_i_ptr = (
        out_indices_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    )
    store_mask = mask_m[:, None] & mask_k[None, :]
    tl.store(out_w_ptr, selected_vals, mask=store_mask)
    tl.store(out_i_ptr, selected_idx, mask=store_mask)


def covered(scores: torch.Tensor, bias: torch.Tensor, topk: int) -> bool:
    """bf16 logits, fp32 bias, one warp per row (at most 512 experts, as the Triton router).

    The lower expert-count bound is the Triton router's own row tiling, not
    caution: ``moe_fused_gate`` packs ``BLOCK_M = max(1, min(4, 256 //
    BLOCK_N))`` rows per program, so below ``BLOCK_N == 256`` its axis-1 max,
    row sum and routed sum run in a different thread layout than this kernel's
    one-warp-per-row reduction, and the fp32 softmax weights are no longer
    guaranteed bit-identical.  Flash-Next's router (512 experts) is inside
    either way; an out-of-range width is not the call this kernel replaces.
    """
    return (
        scores.dim() == 2
        and scores.dtype == torch.bfloat16
        and bias.dim() == 1
        and bias.dtype == torch.float32
        and scores.size(1) == bias.size(0)
        and scores.size(1) <= 512
        and triton.next_power_of_2(scores.size(1)) >= 256
        and 0 < int(topk) <= scores.size(1)
    )


_ZERO_BIAS: dict[tuple[int, torch.device], torch.Tensor] = {}


def get_zero_bias(scores: torch.Tensor) -> Optional[torch.Tensor]:
    """The fp32 all-zero correction bias of one router call, or None if unavailable.

    ``fused_topk``'s softmax route hands the Triton router no bias at all (it
    keeps the zeros in registers), so the packed-key kernel -- which takes a
    bias pointer, like every other router caller -- needs one cached value.
    Returning ``None`` inside a CUDA graph capture keeps the caller on the
    Triton router rather than allocating a lifetime-long tensor from that
    graph's private pool.
    """
    key = (scores.size(1), scores.device)
    bias = _ZERO_BIAS.get(key)
    if bias is None:
        if scores.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return None
        bias = torch.zeros(key[0], dtype=torch.float32, device=scores.device)
        _ZERO_BIAS[key] = bias
    return bias


def route_softmax_fast(
    scores: torch.Tensor,
    zero_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """moe_fused_gate(scores, zero_bias, topk, "softmax", renormalize=renormalize) for an all-zero
    zero_bias (fused_topk's zero bias). The caller checks covered()."""
    M, N = scores.shape
    weights = torch.empty((M, topk), dtype=torch.float32, device=scores.device)
    indices = torch.empty((M, topk), dtype=torch.int32, device=scores.device)
    use_pdl = is_arch_support_pdl()
    extra = {"launch_pdl": True} if use_pdl else {}
    _router_softmax_fast32_kernel[(M,)](
        scores,
        zero_bias,
        weights,
        indices,
        M,
        N=N,
        K=topk,
        BLOCK_N=triton.next_power_of_2(N),
        BLOCK_K=triton.next_power_of_2(topk),
        RENORMALIZE=bool(renormalize),
        USE_PDL=use_pdl,
        stride_sm=scores.stride(0),
        stride_sn=scores.stride(1),
        stride_wm=weights.stride(0),
        stride_wk=weights.stride(1),
        stride_im=indices.stride(0),
        stride_ik=indices.stride(1),
        num_warps=1,
        **extra,
    )
    return weights, indices
