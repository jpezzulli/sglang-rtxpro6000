# SPDX-License-Identifier: Apache-2.0
"""Online-FP8 support for Qwen Flash-Next on exact SM120.

Eligible Flash-Next launches select this path automatically: no launch flag
and no kernel question. An unset or blank ``SGLANG_SM120_ONLINE_MXFP8``
enables it only for a recognized Flash-Next (qwen4_exp) checkpoint on an
exact-SM120 CUDA device; anything unsupported simply keeps the original
paths. Saved explicit true/false values remain private compatibility/debug
escape hatches, and an explicit unsupported request still fails boot.

Large eligible projections carry one FP32 scale per output row (rowwise
weight-only FP8, the donor-compatible format); the same rowwise
representation covers HyperConnection mix weights and the language-model
head, so target/draft sharing cannot separate the quantized values from
their scales.
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

#: Donor hard limit of the W8A16 GEMV (``assert M <= 16``; below that the plan
#: table may pick the M_PAD 1 broadcast path).  ``SGLANG_FP8_W8A16_GEMV_MAX_M``
#: can only lower the row budget, never raise it past what the kernel promises.
_W8A16_GEMV_DONOR_MAX_M = 16

_MAX_KERNEL_ROWS = 32
_DEQUANT_TARGET_BYTES = 64 * 1024 * 1024
_SCALE_ATTR = "_sm120_rowwise_scale"
_online_fp8_enabled = False
_fast_paths_enabled = False

#: Cache-identity marker for the effective rowwise-FP8 representation.  It is
#: deliberately neither legacy Boolean: ``online_mxfp8=true`` named the old
#: mixed-MXFP8 namespaces, while ``false`` names the untouched-BF16 ones that
#: automatic-off and explicit-off still share.
ROWWISE_FP8_PRECISION = "rowwise_fp8"
_OFF_PRECISION = "false"

#: Same checkpoint-metadata families penny_config.py's MODEL_METADATA names.
_FLASH_NEXT_METADATA = (
    "qwen4expforconditionalgeneration",
    "qwen4_exp",
    "qwen4_exp_text",
)


def _resolve_switch(
    requested: bool | None,
    *,
    cuda_available: bool,
    capability: tuple[int, int] | None,
    model_eligible: bool,
) -> tuple[bool, bool]:
    """Return (online_fp8, shared fast paths) for one tri-state request.

    ``None`` (unset/blank) selects automatically: accepted kernels on an exact
    SM120 device for a recognized Flash-Next checkpoint, the original paths
    otherwise. ``False`` propagates the saved opt-out silently. ``True`` keeps
    the pre-existing fail-loud contract for explicit unsupported requests.
    """
    if requested is False:
        return False, False
    if requested is True:
        if not cuda_available:
            raise RuntimeError("SGLANG_SM120_ONLINE_MXFP8 requires CUDA")
        if capability != (12, 0):
            raise RuntimeError(
                "SGLANG_SM120_ONLINE_MXFP8 requires exactly SM120; "
                f"detected compute capability {capability}"
            )
        # The shared fast paths stay scoped to the Flash-Next family even
        # under an explicit request, so a stray true cannot bleed them onto
        # unrelated models (e.g. the generic 27B FP8 methods).
        return True, bool(model_eligible)
    automatic = bool(cuda_available and capability == (12, 0) and model_eligible)
    return automatic, automatic


def configure_online_fp8(
    requested: bool | None,
    *,
    cuda_available: bool,
    capability: tuple[int, int] | None,
    model_eligible: bool = False,
) -> bool:
    """Resolve the process switches, rejecting unsupported explicit requests.

    ``requested`` is the tri-state reading of SGLANG_SM120_ONLINE_MXFP8:
    None means automatic default selection, True/False are saved explicit
    choices.  Model eligibility comes from the loaded checkpoint metadata,
    never from a directory name.
    """
    global _online_fp8_enabled, _fast_paths_enabled
    _online_fp8_enabled, _fast_paths_enabled = _resolve_switch(
        requested,
        cuda_available=cuda_available,
        capability=capability,
        model_eligible=model_eligible,
    )
    return _online_fp8_enabled


def online_fp8_enabled() -> bool:
    return _online_fp8_enabled


def fast_paths_enabled() -> bool:
    """Whether the accepted shared low-row kernels may engage automatically.

    True exactly when the eligible Flash-Next/SM120 default selection (or an
    explicit request on an eligible model) turned the rowwise-FP8 bundle on.
    """
    return _fast_paths_enabled


def gated_by_fast_paths(explicit: bool | None) -> bool:
    """Shared tri-state for the accepted companion kernels.

    A saved explicit true/false wins verbatim (private escape hatch); unset or
    blank follows the resolved default selection, so the whole accepted bundle
    engages or retreats together and never leaks onto other models.
    """
    return _fast_paths_enabled if explicit is None else explicit


def _metadata_names(node: object, depth: int, names: set[str]) -> None:
    if node is None or depth > 1 or isinstance(node, (str, int, float, bool)):
        return
    if isinstance(node, dict):
        architectures = node.get("architectures")
        model_type = node.get("model_type")
        text_config = node.get("text_config")
    else:
        architectures = getattr(node, "architectures", None)
        model_type = getattr(node, "model_type", None)
        text_config = getattr(node, "text_config", None)
    for architecture in architectures or ():
        names.add(str(architecture).lower())
    if isinstance(model_type, str):
        names.add(model_type.lower())
    _metadata_names(text_config, depth + 1, names)


def flash_next_metadata(config: object) -> bool:
    """Whether checkpoint metadata names the Flash-Next (qwen4_exp) family.

    Accepts the loaded hf config object or a raw config.json mapping -- the
    actual model identity, never a filename.  Unknown architectures stay
    ineligible: they keep their original paths instead of guessing.
    """
    names: set[str] = set()
    _metadata_names(config, 0, names)
    return bool(names & frozenset(_FLASH_NEXT_METADATA))


def _metadata_flag_true(node: object, key: str, depth: int = 0) -> bool:
    if node is None or depth > 1 or isinstance(node, (str, int, float, bool)):
        return False
    if isinstance(node, dict):
        flagged = node.get(key)
        text_config = node.get("text_config")
    else:
        flagged = getattr(node, key, None)
        text_config = getattr(node, "text_config", None)
    return bool(flagged is True) or _metadata_flag_true(text_config, key, depth + 1)


def flash_next_eligible(model_config: object) -> bool:
    """Whether the ACTUAL loaded configuration can run the accepted
    rowwise-FP8 representation, reusing the accepted contracts verbatim -- no
    new support invented here:

    * Flash-Next (qwen4_exp) checkpoint metadata, the only family whose
      modules consume the conversion;
    * a BF16 compute dtype: ``replace_linear_weight_rowwise_fp8`` and the
      accepted dense conversion quantize the resident BF16 checkpoint weight
      and refuse any other dtype (float16 included), so automatic selection
      stays off rather than booting into that rejection;
    * untied input/lm_head weights: the accepted Qwen4Exp post-load contract
      rejects ``tie_word_embeddings=True`` under online FP8.
    """
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None or not flash_next_metadata(hf_config):
        return False
    if getattr(model_config, "dtype", None) is not torch.bfloat16:
        return False
    return not _metadata_flag_true(hf_config, "tie_word_embeddings")


def _read_checkpoint_metadata(model_path: str) -> dict | None:
    try:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def resolve_precision(
    requested: bool | None,
    *,
    cuda_available: bool,
    capability: tuple[int, int] | None,
    model_eligible: bool,
) -> str:
    """Effective-precision cache identity, resolved before startup.

    Automatic and explicit-on select the same accepted rowwise-FP8
    representation and must agree on one identity the legacy
    ``online_mxfp8=true`` mixed-MXFP8 namespaces cannot claim; automatic-off
    and explicit-off keep the untouched-weight identity they already had.
    An explicit request names the rowwise build even on hardware where the
    runtime will fail boot before any cache is touched.
    """
    if requested is True:
        return ROWWISE_FP8_PRECISION
    if (
        requested is None
        and cuda_available
        and capability == (12, 0)
        and model_eligible
    ):
        return ROWWISE_FP8_PRECISION
    return _OFF_PRECISION


def launch_precision(
    model_path: str, *, compute_dtype: torch.dtype = torch.bfloat16
) -> str:
    """Effective precision for one launch, from the same eligibility logic
    ``configure_online_fp8`` applies -- shared by the runtime default and the
    recipe/container identity probes, so public direct launches and recipe
    launches cannot disagree about which cache a run may read.  The recipe
    callers pin ``--dtype bfloat16`` in their own launch line, so the dtype
    half of the accepted representation contract is the keyword default; the
    tied-head half is read from the checkpoint's own config.json metadata."""
    metadata = _read_checkpoint_metadata(model_path)
    model_eligible = (
        flash_next_metadata(metadata)
        and compute_dtype is torch.bfloat16
        and not _metadata_flag_true(metadata, "tie_word_embeddings")
    )
    available = torch.cuda.is_available()
    return resolve_precision(
        envs.SGLANG_SM120_ONLINE_MXFP8.get(),
        cuda_available=available,
        capability=torch.cuda.get_device_capability() if available else None,
        model_eligible=model_eligible,
    )


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
    _prealloc_w8a16_gemv_scratch(weight.device)
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


def attach_rowwise_ingest(linears, *, target_device: torch.device | None = None) -> int:
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
    _prealloc_w8a16_gemv_scratch(weight.device)
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


def w8a16_gemv_enabled() -> bool:
    """Whether the accepted donor W8A16 output-head fast path may be considered.

    The eligible default selection turns the whole bundle on; a saved explicit
    true/false remains the private compatibility/debug override.  Larger rows
    and unsupported layouts keep the original kernel regardless."""
    return gated_by_fast_paths(envs.SGLANG_FP8_W8A16_GEMV.get())


def _prealloc_w8a16_gemv_scratch(device: torch.device) -> None:
    """Materialize the donor kernel's split-K scratch before warm-up/capture.

    Weight installation is the last hook that runs before the CUDA graphs are
    captured, and the scratch must not come from a graph's private memory pool.
    """
    if not w8a16_gemv_enabled():
        return
    from sglang.srt.layers.quantization.w8a16_gemv import prealloc

    prealloc(device)


def w8a16_gemv_supported(
    hidden: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor | None
) -> bool:
    """Whether `rowwise_fp8_lm_head_logits` may take the donor W8A16 GEMV.

    The donor's contract (`Fp8LinearMethod._w8a16_gemv_ok` in its fp8.py),
    restated for the resident rowwise representation: weight-only FP8 with a
    per-output-row fp32 scale, no activation quantization, a bf16 2-D
    activation of at most the donor's row limit, and an [N, K] weight whose K
    axis is contiguous, which is how the rowwise Parameter is stored.
    """
    if not w8a16_gemv_enabled() or scale is None or hidden.dim() != 2:
        return False
    max_rows = min(envs.SGLANG_FP8_W8A16_GEMV_MAX_M.get(), _W8A16_GEMV_DONOR_MAX_M)
    return (
        1 <= hidden.shape[0] <= max_rows
        and hidden.dtype is torch.bfloat16
        and hidden.stride(1) == 1
        and hidden.shape[1] == weight.shape[1]
        and weight.dtype is torch.float8_e4m3fn
        and weight.stride(1) == 1
        and scale.dtype is torch.float32
        and scale.shape == (weight.shape[0],)
        and scale.stride(0) == 1
        and scale.device == weight.device
        and hidden.device == weight.device
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
    # Accepted low-row candidate: the donor's W8A16 GEMV on the resident
    # weight/scale as they are, with no requantization and no second copy --
    # engaged automatically for the eligible default selection (or a saved
    # explicit SGLANG_FP8_W8A16_GEMV=1).  Everything outside its contract --
    # larger batches such as C6's 24-row verification and prefill, other
    # dtypes or layouts -- keeps the original kernel and dequantizing
    # fallback below, untouched for comparison.
    if w8a16_gemv_supported(hidden_2d, weight, scale):
        from sglang.srt.layers.quantization.w8a16_gemv import w8a16_gemv

        return w8a16_gemv(hidden_2d, weight, scale).reshape(*original_shape, columns)
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
