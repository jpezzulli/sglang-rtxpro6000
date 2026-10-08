"""Fused BF16 scale, FP8 cast, and NHD KV-cache scatter."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_kv_store_kernel(
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    loc_ptr,
    k_scale,
    v_scale,
    k_stride_tok,
    v_stride_tok,
    row_size: tl.constexpr,
    HAS_K_SCALE: tl.constexpr,
    HAS_V_SCALE: tl.constexpr,
    K_SCALE_IS_TENSOR: tl.constexpr,
    V_SCALE_IS_TENSOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < row_size
    # Local compatibility delta against the donor, the only one in this file.
    # Slot 0 of an NHD pool is the reserved CUDA-graph padding slot: the padded
    # batch is filled by ``out_cache_loc.zero_()``
    # (srt/model_executor/runner_utils/buffers.py:238), so a write destination of
    # 0 is padding, not data. The resident writer honours that by skipping it
    # (kernels/ops/kvcache/kvcache.py:67,84 ``reserved_skip_index=0``; same
    # predicate as kvcache/mla_buffer.py:40), and the donor's standalone kernel
    # has no such guard because it owns no reservation contract. Skipping it here
    # keeps the two writers byte-identical; every non-zero location, the scale
    # handling, the row layout and the launch geometry are untouched.
    slot = tl.load(loc_ptr + token).to(tl.int64)
    store_mask = mask & (slot != 0)
    dst = slot * row_size + offset
    k = tl.load(k_ptr + token * k_stride_tok + offset, mask=mask).to(tl.float32)
    v = tl.load(v_ptr + token * v_stride_tok + offset, mask=mask).to(tl.float32)
    # Match ``cache.div_(scale)`` exactly: a true fp32 division (not a
    # multiply by the reciprocal), rounded to BF16, then cast to FP8.
    # A 0-d scale *tensor* is first cast to the BF16 common dtype by torch's
    # type promotion (a python float scalar is not), so mirror that here.
    if HAS_K_SCALE:
        if K_SCALE_IS_TENSOR:
            k = k / tl.load(k_scale).to(tl.bfloat16).to(tl.float32)
        else:
            k = k / k_scale
    if HAS_V_SCALE:
        if V_SCALE_IS_TENSOR:
            v = v / tl.load(v_scale).to(tl.bfloat16).to(tl.float32)
        else:
            v = v / v_scale
    # div_ first materializes BF16, then Tensor.to performs the FP8 rounding.
    k_fp8 = k.to(tl.bfloat16).to(tl.float8e4nv)
    v_fp8 = v.to(tl.bfloat16).to(tl.float8e4nv)
    tl.store(k_cache_ptr + dst, k_fp8, mask=store_mask)
    tl.store(v_cache_ptr + dst, v_fp8, mask=store_mask)


def fp8_kv_store(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    loc: torch.Tensor,
    k_scale: float | torch.Tensor | None = None,
    v_scale: float | torch.Tensor | None = None,
) -> None:
    """Scale/cast BF16 K/V and scatter them into native E4M3 NHD buffers."""
    if cache_k.dim() != 3 or cache_v.shape != cache_k.shape:
        raise ValueError("K and V must have the same 3-D shape")
    tokens, heads, head_dim = cache_k.shape
    if not 1 <= tokens <= 64:
        raise ValueError("fused FP8 KV store requires 1 <= N <= 64")
    if loc.shape != (tokens,) or loc.dtype not in (torch.int32, torch.int64):
        raise ValueError("loc must be int32/int64 with shape [N]")
    tensors = (cache_k, cache_v, k_buffer, v_buffer, loc)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("fused FP8 KV store requires CUDA tensors")
    if any(tensor.device != cache_k.device for tensor in tensors):
        raise ValueError("all fused FP8 KV store tensors must be on one device")
    if cache_k.dtype != torch.bfloat16 or cache_v.dtype != torch.bfloat16:
        raise ValueError("fused FP8 KV store inputs must be BF16")
    if k_buffer.dtype != torch.float8_e4m3fn or v_buffer.dtype != k_buffer.dtype:
        raise ValueError("fused FP8 KV store destinations must be float8_e4m3fn")
    if k_buffer.dim() != 3 or v_buffer.shape != k_buffer.shape:
        raise ValueError("K/V destination buffers must have the same 3-D shape")
    if k_buffer.shape[1:] != (heads, head_dim):
        raise ValueError("source and destination KV row shapes must match")
    if any(not tensor.is_contiguous() for tensor in (k_buffer, v_buffer, loc)):
        raise ValueError("fused FP8 KV store destinations and loc must be contiguous")
    for name, src in (("cache_k", cache_k), ("cache_v", cache_v)):
        # Sources may be strided along the token axis (views of a fused
        # qkv buffer); each [H, D] row itself must be contiguous.
        if src.stride(2) != 1 or src.stride(1) != head_dim:
            raise ValueError(f"{name} rows must be contiguous [H, D]")

    def _check_scale(scale, name: str) -> None:
        if isinstance(scale, torch.Tensor):
            if (
                scale.numel() != 1
                or not scale.is_cuda
                or scale.device != cache_k.device
            ):
                raise ValueError(
                    f"{name} tensor must be a CUDA scalar on the KV device"
                )

    _check_scale(k_scale, "k_scale")
    _check_scale(v_scale, "v_scale")
    k_is_tensor = isinstance(k_scale, torch.Tensor)
    v_is_tensor = isinstance(v_scale, torch.Tensor)
    k_arg = k_scale if k_scale is not None else cache_k
    v_arg = v_scale if v_scale is not None else cache_v
    row_size = heads * head_dim
    _fp8_kv_store_kernel[(tokens, triton.cdiv(row_size, 256))](
        cache_k,
        cache_v,
        k_buffer,
        v_buffer,
        loc,
        k_arg,
        v_arg,
        cache_k.stride(0),
        cache_v.stride(0),
        row_size=row_size,
        HAS_K_SCALE=k_scale is not None,
        HAS_V_SCALE=v_scale is not None,
        K_SCALE_IS_TENSOR=k_is_tensor,
        V_SCALE_IS_TENSOR=v_is_tensor,
        BLOCK=256,
        num_warps=4,
    )
