# SPDX-License-Identifier: Apache-2.0
"""Row-wise (per-output-channel) FP8 projection for the SM120 output head.

Reads the resident E4M3 weight and its per-row FP32 scales exactly as
``sm120_online_fp8`` stored them -- no BF16 copy of the head, no second
representation.  Low row counts (decode, native-MTP verification) take a
split-N Triton GEMV, where the head is a memory-bound read of ``N * K`` codes
and the FP8 storage halves that traffic against BF16.  Wider row counts (long
prefill, big batches) take the chunked dequantized projection: the arithmetic is
identical, cuBLAS is the better kernel once the batch is large enough to be
compute-bound, and the chunk size bounds the transient operand instead of the
whole head.

Register this with the kernel registry when a second backend exists to select
between; with one implementation the caller's dtype/shape check is the dispatch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Above this many rows the GEMV's per-CTA weight re-read stops paying for
#: itself and the chunked cuBLAS projection wins.
_MAX_GEMV_ROWS = 32

#: Bytes of dequantized BF16 operand materialized per chunk (one pass per
#: ``_DEQUANT_TARGET_BYTES // (K * 4)`` output columns).
_DEQUANT_TARGET_BYTES = 64 * 1024 * 1024


@triton.jit
def _rowwise_fp8_gemv_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """output[m, n] = sum_k x[m, k] * w[n, k] * scale[n], one FP32 scale per row."""
    output_block = tl.program_id(0)
    rows_n = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    rows_m = tl.arange(0, BLOCK_M)
    mask_n = rows_n < N
    mask_m = rows_m < M
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, K, BLOCK_K):
        columns_k = start_k + tl.arange(0, BLOCK_K)
        mask_k = columns_k < K
        x = tl.load(
            x_ptr + rows_m[:, None] * stride_xm + columns_k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + rows_n[:, None] * stride_wn + columns_k[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        ).to(x_ptr.dtype.element_ty)
        accumulator += tl.dot(x, tl.trans(weight), out_dtype=tl.float32)
    scale = tl.load(scale_ptr + rows_n, mask=mask_n, other=0.0)
    result = accumulator * scale[None, :]
    tl.store(
        output_ptr + rows_m[:, None] * stride_om + rows_n[None, :] * stride_on,
        result.to(output_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _dequantized_projection(
    hidden_states: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    rows, columns = hidden_states.shape[0], weight.shape[0]
    output = torch.empty(
        (rows, columns), dtype=torch.bfloat16, device=hidden_states.device
    )
    block_rows = max(
        1,
        min(
            8192,
            _DEQUANT_TARGET_BYTES // max(1, weight.shape[1] * torch.float32.itemsize),
        ),
    )
    for start in range(0, columns, block_rows):
        stop = min(start + block_rows, columns)
        dense = (weight[start:stop].float() * scale[start:stop, None]).to(
            torch.bfloat16
        )
        output[:, start:stop] = hidden_states @ dense.T
    return output


def rowwise_fp8_linear(
    hidden_states: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """BF16 [M, N] logits from BF16 [M, K] activations and row-wise FP8 [N, K]."""
    rows, columns = hidden_states.shape[0], weight.shape[0]
    if not hidden_states.is_cuda or rows == 0 or rows > _MAX_GEMV_ROWS:
        return _dequantized_projection(hidden_states, weight, scale)
    output = torch.empty(
        (rows, columns), dtype=torch.bfloat16, device=hidden_states.device
    )
    _rowwise_fp8_gemv_kernel[(triton.cdiv(columns, 32),)](
        hidden_states,
        weight,
        scale,
        output,
        rows,
        columns,
        hidden_states.shape[1],
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=max(16, triton.next_power_of_2(rows)),
        BLOCK_N=32,
        BLOCK_K=128,
        num_warps=4,
        num_stages=4,
    )
    return output
