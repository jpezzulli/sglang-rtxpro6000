# SPDX-License-Identifier: Apache-2.0
"""Bounded online-FP8 support for Qwen Flash-Next on exact SM120.

Large eligible projections use SGLang's existing MXFP8 quantization method.
This module owns the two weight-only exceptions: HyperConnection mix weights
and the language-model head.  Those weights carry one FP32 scale per output
row on the resident ``Parameter`` so target/draft sharing cannot separate the
quantized values from their scales.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
import triton
import triton.language as tl

_MAX_KERNEL_ROWS = 32
_DEQUANT_TARGET_BYTES = 64 * 1024 * 1024
_SCALE_ATTR = "_sm120_rowwise_scale"
_online_fp8_enabled = False


def configure_online_fp8(
    requested: bool,
    *,
    cuda_available: bool,
    capability: tuple[int, int] | None,
) -> bool:
    """Resolve the process switch, rejecting unsupported explicit requests."""
    global _online_fp8_enabled
    _online_fp8_enabled = False
    if not requested:
        return False
    if not cuda_available:
        raise RuntimeError("SGLANG_SM120_ONLINE_MXFP8 requires CUDA")
    if capability != (12, 0):
        raise RuntimeError(
            "SGLANG_SM120_ONLINE_MXFP8 requires exactly SM120; "
            f"detected compute capability {capability}"
        )
    _online_fp8_enabled = True
    return True


def online_fp8_enabled() -> bool:
    return _online_fp8_enabled


def quantize_rowwise_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D BF16 tensor with one FP32 scale per output row."""
    if weight.dim() != 2 or weight.dtype != torch.bfloat16:
        raise TypeError(
            "SM120 rowwise FP8 quantization requires a 2-D BF16 weight, got "
            f"shape={tuple(weight.shape)} dtype={weight.dtype}"
        )
    quantized = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scale = torch.empty(weight.shape[0], dtype=torch.float32, device=weight.device)
    rows_per_chunk = max(
        1, _DEQUANT_TARGET_BYTES // max(1, weight.shape[1] * torch.float32.itemsize)
    )
    for row in range(0, weight.shape[0], rows_per_chunk):
        block = weight[row : row + rows_per_chunk].float()
        block_scale = block.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 448.0
        quantized[row : row + rows_per_chunk] = (
            (block / block_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        )
        scale[row : row + rows_per_chunk] = block_scale.squeeze(1)
    return quantized, scale


def rowwise_scale_of(weight: torch.Tensor) -> torch.Tensor | None:
    return getattr(weight, _SCALE_ATTR, None)


def _require_rowwise_scale(weight: torch.Tensor) -> torch.Tensor:
    scale = rowwise_scale_of(weight)
    if weight.dtype != torch.float8_e4m3fn or scale is None:
        raise RuntimeError(
            "SM120 online FP8 weight is missing its rowwise scale metadata"
        )
    if scale.dim() != 1 or scale.shape[0] != weight.shape[0]:
        raise RuntimeError(
            "SM120 online FP8 rowwise scale shape does not match the weight: "
            f"weight={tuple(weight.shape)} scale={tuple(scale.shape)}"
        )
    return scale


def dequantize_rowwise_weight(
    weight: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    scale = _require_rowwise_scale(weight)
    return (weight.float() * scale[:, None]).to(dtype)


def _copy_parameter_attrs(source: torch.Tensor, destination: torch.Tensor) -> None:
    """Retain loader/sharding metadata when replacing a whole Parameter."""
    for name, value in vars(source).items():
        if name != _SCALE_ATTR:
            setattr(destination, name, value)


def _rowwise_parameter(
    weight: torch.Tensor, source: torch.Tensor
) -> torch.nn.Parameter:
    quantized, scale = quantize_rowwise_fp8(weight)
    parameter = torch.nn.Parameter(quantized, requires_grad=False)
    _copy_parameter_attrs(source, parameter)
    setattr(parameter, _SCALE_ATTR, scale)
    return parameter


def replace_linear_weight_rowwise_fp8(linear: torch.nn.Module) -> int:
    """Replace one resident BF16 linear weight; repeated calls are a no-op."""
    weight = getattr(linear, "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or weight.dim() != 2:
        raise RuntimeError("SM120 online FP8 requires a 2-D Parameter weight")
    if weight.dtype == torch.float8_e4m3fn and rowwise_scale_of(weight) is not None:
        return 0
    if weight.dtype != torch.bfloat16:
        raise RuntimeError(
            f"SM120 online FP8 expected a BF16 resident weight, got {weight.dtype}"
        )
    original_bytes = weight.numel() * weight.element_size()
    new_parameter = _rowwise_parameter(weight.data, weight)
    if hasattr(weight, "weight_loader"):
        # A later weight update must recompute both values and scales. Keeping
        # the old copy loader would silently write BF16 into FP8 storage while
        # leaving stale scale metadata behind.
        new_parameter.weight_loader = functools.partial(
            _ingest_rowwise_weight, linear, weight.device
        )
    linear.weight = new_parameter
    return original_bytes


def _ingest_rowwise_weight(
    linear: torch.nn.Module,
    target_device: torch.device | None,
    parameter: torch.Tensor,
    loaded_weight: torch.Tensor,
    *args,
    **kwargs,
) -> None:
    if parameter.shape != loaded_weight.shape:
        raise RuntimeError(
            "SM120 online FP8 checkpoint shape mismatch: "
            f"expected {tuple(parameter.shape)}, got {tuple(loaded_weight.shape)}"
        )
    if loaded_weight.dtype != torch.bfloat16:
        raise RuntimeError(
            "SM120 online FP8 HyperConnection ingest requires BF16 checkpoint "
            f"weights, got {loaded_weight.dtype}"
        )
    destination = target_device
    if destination is None:
        destination = torch.device("cuda", torch.cuda.current_device())
    # Quantize the checkpoint shard on CPU, then transfer only FP8 values and
    # row scales. This is why HC weights are born on meta: their full BF16 copy
    # never becomes resident device state.
    source = (
        loaded_weight
        if loaded_weight.device.type == "cpu"
        else loaded_weight.to(device="cpu")
    )
    quantized, scale = quantize_rowwise_fp8(source)
    new_parameter = torch.nn.Parameter(
        quantized.to(device=destination), requires_grad=False
    )
    _copy_parameter_attrs(parameter, new_parameter)
    setattr(new_parameter, _SCALE_ATTR, scale.to(device=destination))
    # Future reloads must call the same quantizing loader rather than copying
    # BF16 values directly into the resident FP8 tensor with stale scales.
    new_parameter.weight_loader = functools.partial(
        _ingest_rowwise_weight, linear, target_device
    )
    linear.weight = new_parameter


def attach_rowwise_ingest(
    linears, *, target_device: torch.device | None = None
) -> int:
    """Attach an all-or-nothing loader to meta-born BF16 linear weights."""
    checked = []
    for linear in linears:
        weight = getattr(linear, "weight", None)
        if not isinstance(weight, torch.nn.Parameter) or weight.dim() != 2:
            raise RuntimeError("SM120 online FP8 ingest requires 2-D Parameters")
        if weight.device.type != "meta" or weight.dtype != torch.bfloat16:
            raise RuntimeError(
                "SM120 online FP8 ingest requires meta-born BF16 weights, got "
                f"device={weight.device} dtype={weight.dtype}"
            )
        checked.append((linear, weight))
    for linear, weight in checked:
        weight.weight_loader = functools.partial(
            _ingest_rowwise_weight, linear, target_device
        )
    return len(checked)


def select_rowwise_weight_rows(
    weight: torch.Tensor, row_indices: torch.Tensor
) -> torch.Tensor:
    """Select draft-vocabulary rows without dropping matching scale metadata."""
    scale = _require_rowwise_scale(weight)
    selected_data = weight.index_select(0, row_indices)
    if isinstance(weight, torch.nn.Parameter):
        selected = torch.nn.Parameter(selected_data, requires_grad=False)
    else:
        selected = selected_data
    setattr(selected, _SCALE_ATTR, scale.index_select(0, row_indices))
    return selected


def convert_eligible_linears_to_mxfp8(
    root: torch.nn.Module,
    *,
    enabled: bool,
    method_factory: Callable[[], object],
    unquantized_method_type: type,
    excluded_module_type: type | tuple[type, ...],
) -> list[str]:
    """Install MXFP8 only on eligible BF16 linears outside excluded subtrees."""
    if not enabled:
        return []

    candidates: list[tuple[str, torch.nn.Module]] = []

    def visit(module: torch.nn.Module, prefix: str) -> None:
        for child_name, child in module.named_children():
            name = f"{prefix}.{child_name}" if prefix else child_name
            if child_name == "gate" or isinstance(child, excluded_module_type):
                continue
            weight = getattr(child, "weight", None)
            if (
                isinstance(
                    getattr(child, "quant_method", None), unquantized_method_type
                )
                and isinstance(weight, torch.nn.Parameter)
                and weight.dim() == 2
                and weight.dtype == torch.bfloat16
                and weight.shape[0] >= 128
                and weight.shape[1] >= 128
                and weight.shape[1] % 32 == 0
            ):
                candidates.append((name, child))
            visit(child, name)

    visit(root, "")
    if not candidates:
        return []
    method = method_factory()
    for _, module in candidates:
        module.quant_method = method
    return [name for name, _ in candidates]


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


def rowwise_fp8_lm_head_logits(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Project through the resident rowwise-FP8 lm_head without BF16 fallback."""
    scale = _require_rowwise_scale(weight)
    if hidden_states.device != weight.device or scale.device != weight.device:
        raise RuntimeError(
            "SM120 online FP8 lm_head weight/scale/input device mismatch"
        )
    if hidden_states.dtype != torch.bfloat16:
        hidden_states = hidden_states.bfloat16()
    original_shape = hidden_states.shape[:-1]
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    if hidden_2d.shape[1] != weight.shape[1]:
        raise RuntimeError(
            "SM120 online FP8 lm_head input width does not match its weight"
        )
    rows, columns = hidden_2d.shape[0], weight.shape[0]
    output = torch.empty(
        (rows, columns), dtype=torch.bfloat16, device=hidden_states.device
    )
    if rows <= _MAX_KERNEL_ROWS:
        _rowwise_fp8_gemv_kernel[(triton.cdiv(columns, 32),)](
            hidden_2d,
            weight,
            scale,
            output,
            rows,
            columns,
            hidden_2d.shape[1],
            hidden_2d.stride(0),
            hidden_2d.stride(1),
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
    else:
        block_rows = max(
            1,
            min(
                8192,
                _DEQUANT_TARGET_BYTES
                // max(1, weight.shape[1] * torch.float32.itemsize),
            ),
        )
        for start in range(0, columns, block_rows):
            stop = min(start + block_rows, columns)
            dense = (weight[start:stop].float() * scale[start:stop, None]).to(
                torch.bfloat16
            )
            output[:, start:stop] = hidden_2d @ dense.T
    return output.reshape(*original_shape, columns)
