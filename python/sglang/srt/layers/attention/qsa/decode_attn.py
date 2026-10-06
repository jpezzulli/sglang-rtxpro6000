# SPDX-License-Identifier: Apache-2.0
# Adapted from aiueo52/sglang-rtxpro6000 (flash-next-fast), file
# python/sglang/srt/layers/attention/qsa/decode_attn.py (Apache-2.0) blob
# 29ff2e81a6e2960a956e94bcdb6f5c1325549e4b of donor commit 04f155ab921d9c2b20e137990088ac92bfaa2b4e
# ("Candidate 2026-10-02: merge the RS package (opus/rs3-bv) and XA1 on ST1",
# linearised side-branch subjects "QSA: opt-in split-KV Triton decode attention
# (SGLANG_OPT_TRITON_DECODE_ATTN)" and "QSA decode attention: scan only below
# seq_len on shared-tail-prefix rows"); donor parent
# cbe20cc00073a4363e988ab67f9b93ee62265b25, full snapshot
# 5105985116eb00dea8e6138aabeb5363387cb9de.
#
# Two deliberate deviations from the donor kernel:
#   - Programmatic Dependent Launch is dropped (this tree has no
#     sglang.kernels.triton_pdl): ordinary Triton launches only, no PDL
#     constexpr, no launch_pdl, so nothing depends on a guessed overlap.
#   - The shared-tail valid-prefix layout it assumes is built here by
#     QSAMTPSharedSparseIndices, not by the donor's ST1 lookup kernels.
#   - Q and K are not cast to F16 for the QK MMA: a finite BF16 query above the
#     F16 max (~65504) would become Inf and NaN the softmax, while the resident
#     BF16 path stays finite. The QK MMA runs on BF16 (E4M3 K converts exactly),
#     the P/V MMA keeps the donor's F16 path (P <= 1, E4M3 V <= 448).
"""Split-KV Triton decode attention over QSA sparse top-k rows.

Opt-in replacement (SGLANG_OPT_TRITON_DECODE_ATTN) for the valid-count, KV
compaction and trtllm-gen XQA decode sequence of QwenSparseAttnBackend. The
kernel reads the fp8 KV pool directly through the top-k logical indices and
req_to_token, so nothing is packed. A column is attended iff
0 <= index < seq_lens[row], checked per column: index-shared MTP draft rows
(frozen anchor selection plus the drafted tail) keep -1 holes mid-row, which the
packed path, attending only a valid prefix, mis-handles.

Grid (splits, kv heads, rows). A program covers one contiguous column chunk for
one kv head and all of its query heads (padded to 16 MMA rows): a BF16 QK MMA on
e4m3 K (exact conversion, query exponent range preserved) and an F16 P/V MMA on
e4m3 V (exact conversion), both with fp32 accumulation. Each program stores its
partial (acc / l in f16, lse = m + log2(l) in fp32); the last arriver per
(row, kv head), found with one acq_rel atomic, combines the partials in split
order and resets its counter to 0. The result is deterministic, and the
counters are zero again after every launch (CUDA graph replay safe).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634
_BLOCK_H = 16
_HEAD_DIM = 256
# The interpreter has no inline asm, and its low-precision dot is unreliable.
_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"

# (max rows, (num_splits, block_n, num_warps, kv_stages, v_asm)), first match
# wins. Measured on RTX PRO 6000 Max-Q (sm120, 188 SMs) at 2051 top-k columns,
# fp8 KV, 2 kv heads x 12 query heads x 256 (flash-next-bench bench/xa1, job4).
_LAUNCH_BY_ROWS = (
    (1, (22, 32, 4, 3, True)),
    (4, (22, 32, 4, 3, True)),
    (8, (11, 64, 4, 3, False)),
    (16, (11, 32, 4, 3, False)),
)
# Past the last bucket splits shrink as rows grow (see _launch_config), so a
# split launch never needs more than this many (row, split) partials per kv head.
_MAX_ROW_SPLITS = max(max_rows * config[0] for max_rows, config in _LAUNCH_BY_ROWS)


@triton.jit
def _e4m3_to_f16(x):
    # Opaque on purpose: V then goes through smem as f16 (ldmatrix operand)
    # instead of fp8 read byte by byte; a pure asm gets hoisted past the copy.
    return tl.inline_asm_elementwise(
        "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $2; cvt.rn.f16x2.e4m3x2 $0, lo; "
        "cvt.rn.f16x2.e4m3x2 $1, hi; }",
        "=r,=r,r",
        [x.to(tl.uint8, bitcast=True)],
        dtype=tl.float16,
        is_pure=False,
        pack=4,
    )


@triton.jit
def _combine_splits(
    partial_out_ptr,
    partial_lse_ptr,
    out_ptr,
    grp,
    offs_h,
    offs_d,
    hmask,
    GROUP: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    # Per-split lse loads land in the row slice of the [heads, D] layout:
    # the combine needs no smem and no barrier. Empty splits carry -1e30.
    lse_max = tl.full([BLOCK_H], -1.0e30, tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        lse = tl.load(
            partial_lse_ptr + (grp * NUM_SPLITS + s) * BLOCK_H + offs_h,
            cache_modifier=".cg",
        )
        lse_max = tl.maximum(lse_max, lse)
    den = tl.zeros([BLOCK_H], tl.float32)
    o = tl.zeros([BLOCK_H, D], tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        part = grp * NUM_SPLITS + s
        w = tl.exp2(
            tl.load(partial_lse_ptr + part * BLOCK_H + offs_h, cache_modifier=".cg")
            - lse_max
        )
        den += w
        o_s = tl.load(
            partial_out_ptr + (part * BLOCK_H + offs_h)[:, None] * D + offs_d[None, :],
            mask=hmask[:, None],
            other=0.0,
            cache_modifier=".cg",
        )
        o += o_s.to(tl.float32) * w[:, None]
    o = o / den[:, None]
    tl.store(
        out_ptr + (grp * GROUP + offs_h)[:, None] * D + offs_d[None, :],
        o.to(out_ptr.dtype.element_ty),
        mask=hmask[:, None],
    )


@triton.jit
def _store_split(
    acc,
    m_i,
    l_i,
    out_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    arrivals_ptr,
    grp,
    sid,
    qo_off,
    offs_h,
    offs_d,
    hmask,
    GROUP: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    o = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
    if NUM_SPLITS == 1:
        tl.store(out_ptr + qo_off, o.to(out_ptr.dtype.element_ty), mask=hmask[:, None])
    else:
        part = grp * NUM_SPLITS + sid
        tl.store(
            partial_out_ptr + (part * BLOCK_H + offs_h)[:, None] * D + offs_d[None, :],
            o.to(partial_out_ptr.dtype.element_ty),
            mask=hmask[:, None],
            cache_modifier=".cg",
        )
        lse = tl.where(l_i > 0, m_i + tl.log2(tl.where(l_i > 0, l_i, 1.0)), -1.0e30)
        tl.store(partial_lse_ptr + part * BLOCK_H + offs_h, lse, cache_modifier=".cg")
        tl.debug_barrier()
        arrived = tl.atomic_add(arrivals_ptr + grp, 1, sem="acq_rel", scope="gpu")
        if arrived == NUM_SPLITS - 1:
            _combine_splits(
                partial_out_ptr,
                partial_lse_ptr,
                out_ptr,
                grp,
                offs_h,
                offs_d,
                hmask,
                GROUP=GROUP,
                BLOCK_H=BLOCK_H,
                D=D,
                NUM_SPLITS=NUM_SPLITS,
            )
            tl.store(arrivals_ptr + grp, 0)


@triton.jit
def _qsa_decode_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    seq_lens_ptr,
    topk_ptr,
    req_to_token_ptr,
    row_req_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    arrivals_ptr,
    qk_scale,
    topk_stride,
    req_to_token_stride,
    NCOLS: tl.constexpr,
    HKV: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PREFIX_VALID: tl.constexpr,
    V_ASM: tl.constexpr,
    F32_DOT: tl.constexpr,
):
    sid = tl.program_id(0)
    kvh = tl.program_id(1)
    row = tl.program_id(2)
    grp = row * HKV + kvh
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    hmask = offs_h < GROUP
    qo_off = (grp * GROUP + offs_h)[:, None] * D + offs_d[None, :]
    hd_off = kvh * D + offs_d

    # Q stays in its own BF16: an F16 cast would turn a finite query above the
    # F16 max (~65504) into Inf and NaN the softmax, while the resident BF16
    # path stays finite. E4M3 K converts to BF16 exactly, so the QK MMA keeps
    # the query's exponent range; the P/V dot below stays F16 (P <= 1, E4M3 V
    # is at most 448).
    q = tl.load(q_ptr + qo_off, mask=hmask[:, None], other=0.0)
    length = tl.load(seq_lens_ptr + row)
    req = tl.load(row_req_ptr + row).to(tl.int64)
    # Indexer rows and tail-after-prefix shared rows keep every valid column in front,
    # all below seq_len; other index-shared rows have holes, so every column is scanned.
    ncols = tl.minimum(length, NCOLS) if PREFIX_VALID else NCOLS
    chunk = tl.cdiv(tl.cdiv(ncols, NUM_SPLITS), BLOCK_N) * BLOCK_N
    start = sid * chunk
    end = tl.minimum(start + chunk, ncols)

    m_i = tl.full([BLOCK_H], -1.0e30, tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)
    for n0 in range(start, end, BLOCK_N):
        cols = n0 + offs_n
        pos = tl.load(topk_ptr + row * topk_stride + cols, mask=cols < end, other=-1)
        valid = (pos >= 0) & (pos < length)
        slot = tl.load(
            req_to_token_ptr + req * req_to_token_stride + pos, mask=valid, other=0
        )
        # Invalid columns gather the pool's padding slot 0 and are masked in
        # the scores, so the K/V loads stay unmasked 16-byte cp.async.
        kv_off = slot.to(tl.int64)[:, None] * (HKV * D) + hd_off[None, :]
        k = tl.load(k_ptr + kv_off).to(tl.bfloat16)
        v = tl.load(v_ptr + kv_off)
        if V_ASM:
            v = _e4m3_to_f16(v)
        else:
            v = v.to(tl.float16)
        if F32_DOT:
            s = tl.dot(
                q.to(tl.float32), tl.trans(k.to(tl.float32)), input_precision="ieee"
            )
        else:
            s = tl.dot(q, tl.trans(k))
        s = tl.where(valid[None, :], s * qk_scale, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.where(valid[None, :], tl.exp2(s - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        if F32_DOT:
            acc += tl.dot(p, v.to(tl.float32), input_precision="ieee")
        else:
            acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new

    _store_split(
        acc,
        m_i,
        l_i,
        out_ptr,
        partial_out_ptr,
        partial_lse_ptr,
        arrivals_ptr,
        grp,
        sid,
        qo_off,
        offs_h,
        offs_d,
        hmask,
        GROUP=GROUP,
        BLOCK_H=BLOCK_H,
        D=D,
        NUM_SPLITS=NUM_SPLITS,
    )


def _launch_config(rows: int):
    for max_rows, config in _LAUNCH_BY_ROWS:
        if rows <= max_rows:
            return config
    # Arbitrary for rows past the measured buckets: keep the last bucket's
    # tile and shrink splits so the CTA count stays about the same.
    max_rows, (num_splits, block_n, num_warps, kv_stages, v_asm) = _LAUNCH_BY_ROWS[-1]
    num_splits = max(1, num_splits * max_rows // rows)
    return num_splits, block_n, num_warps, kv_stages, v_asm


class QSADecodeAttnWorkspace:
    """Split partials plus per-(row, kv head) arrival counters (zero between calls).

    Fixed size for any row count: about 3 MB for 2 kv heads x 256.

    Owned by the QwenSparseAttnBackend that launches the kernel, created on its
    first eager (warmup) call and never re-created: the partial and counter
    addresses are baked into every CUDA graph recorded afterwards, so a later
    allocation would leave the replayed graph pointing at freed memory.  The
    arrival counters self-reset (the last arriver stores 0) so neither launch
    nor replay needs a host-side clear, and one workspace serves all layers of
    its backend because launches on one stream are serialized.
    """

    def __init__(self, *, num_kv_heads: int, head_dim: int, device) -> None:
        self.max_parts = _MAX_ROW_SPLITS * num_kv_heads
        self.partial_out = torch.empty(
            self.max_parts * _BLOCK_H * head_dim, dtype=torch.float16, device=device
        )
        self.partial_lse = torch.empty(
            self.max_parts * _BLOCK_H, dtype=torch.float32, device=device
        )
        self.arrivals = torch.zeros(self.max_parts, dtype=torch.int32, device=device)


def qsa_decode_attention_supported(
    *,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    topk_indices: torch.Tensor,
) -> bool:
    """The measured shape only: bf16 q, fp8 e4m3 KV pool, head_dim 256, GQA <= 16."""
    if q.dim() != 3 or k_buffer.dim() != 3 or topk_indices.dim() != 2:
        return False
    rows, num_q_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    return (
        q.dtype == torch.bfloat16
        and k_buffer.dtype == torch.float8_e4m3fn
        and v_buffer.dtype == k_buffer.dtype
        and v_buffer.shape == k_buffer.shape
        and head_dim == _HEAD_DIM
        and k_buffer.shape[2] == head_dim
        and k_buffer.is_contiguous()
        and v_buffer.is_contiguous()
        and num_q_heads % num_kv_heads == 0
        and num_q_heads // num_kv_heads <= _BLOCK_H
        and topk_indices.shape[0] == rows
    )


def qsa_decode_attention(
    *,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    row_req_pool_indices: torch.Tensor,
    topk_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    workspace: QSADecodeAttnWorkspace,
    prefix_valid: bool,
) -> torch.Tensor:
    """Attention of q [rows, q heads, D] bf16 over the KV pool [slots, kv heads, D]
    fp8 at the top-k logical indices [rows, C] int32 of each row's request.

    prefix_valid: valid columns come first, all below seq_lens (indexer rows, and
    index-shared rows with the drafted tail after the valid prefix), so columns at
    or past seq_lens are skipped.
    """
    rows, num_q_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    num_splits, block_n, num_warps, kv_stages, v_asm = _launch_config(rows)
    if num_splits > 1 and rows * num_kv_heads * num_splits > workspace.max_parts:
        raise ValueError(
            f"QSA decode attention workspace too small: rows={rows}, splits={num_splits}"
        )
    q = q.contiguous()
    out = torch.empty_like(q)
    _qsa_decode_attn_kernel[(num_splits, num_kv_heads, rows)](
        q,
        k_buffer,
        v_buffer,
        out,
        seq_lens,
        topk_indices,
        req_to_token,
        row_req_pool_indices,
        workspace.partial_out,
        workspace.partial_lse,
        workspace.arrivals,
        sm_scale * _LOG2E,
        topk_indices.stride(0),
        req_to_token.stride(0),
        NCOLS=topk_indices.shape[1],
        HKV=num_kv_heads,
        GROUP=num_q_heads // num_kv_heads,
        BLOCK_H=_BLOCK_H,
        D=head_dim,
        NUM_SPLITS=num_splits,
        BLOCK_N=block_n,
        PREFIX_VALID=prefix_valid,
        V_ASM=v_asm and not _INTERPRET,
        F32_DOT=_INTERPRET,
        num_warps=num_warps,
        # The pipeliner splits num_stages over the topk -> slot -> K/V chain;
        # 3 * kv_stages - 2 leaves kv_stages buffers for the K/V tiles.
        num_stages=3 * kv_stages - 2,
    )
    return out
