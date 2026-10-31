# SPDX-License-Identifier: Apache-2.0
"""Online row-wise FP8 for Flash-Next-style checkpoints on exact SM120.

``online`` is a one-time conversion of eligible BF16 checkpoint weights while
the model loads; generation runs on the converted weights and nothing is
re-quantized per step.  An unset or blank ``SGLANG_SM120_ONLINE_MXFP8`` selects
the path automatically -- only for a recognized Flash-Next (``qwen4_exp``)
checkpoint whose resident weights are BF16, on an exact-SM120 CUDA device.
Anything unsupported keeps the checkpoint's own precision layout; a saved
explicit ``true``/``false`` is honored verbatim as the compatibility/debug
escape hatch, and an explicit request on unsupported hardware fails the boot.

Selected representation: weight-only FP8 (E4M3) with one FP32 scale per output
channel.  Eligible ``LinearBase`` projections get that representation through
the existing ``Fp8LinearMethod`` non-serialized lifecycle, so their GEMM dispatch
is the tested per-token/per-channel FP8 path rather than a second kernel.  The
output head carries the same representation with the scale bound to the resident
tensor, because ``lm_head`` is not a ``LinearBase``: the scale travels with the
weight, so a draft that adopts or row-selects the target head cannot separate an
FP8 value from the scale that gives it meaning.  Reloading a converted head goes
through the loader that head was built with, so vocabulary padding and TP shard
placement survive, and the new codes and scales are written into the tensors that
are already resident rather than rebound underneath the draft's alias.

Already-quantized modules (NVFP4 / FP8 / W4A16 checkpoints), the router
``gate``, the PLE table and its projection, and everything outside the eligible
shape envelope keep their checkpoint representation: eligibility is read from
each module, never applied globally.  So do the HyperConnection mix weights --
they are plain ``nn.Linear`` state, and converting them would only pay off with
a kernel that reads row-wise FP8 mix weights directly; without one, an FP8 mix
weight costs decode time in re-materialized BF16 operands and saves nothing.
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable, Iterable
from typing import Any, Optional

import torch

#: Bytes of weight materialized per dequantization chunk, so a wide head never
#: needs a full BF16 copy of itself at once.
_DEQUANT_TARGET_BYTES = 64 * 1024 * 1024

#: Row-width/block floor inherited from the accepted Flash-Next conversion:
#: below it a projection is too small for the FP8 GEMM shapes to pay for the
#: added scale metadata, and K must stay a multiple of the FP8 tile width.
_MIN_FP8_OUTPUT_ROWS = 128
_MIN_FP8_INPUT_COLUMNS = 128
_FP8_COLUMN_MULTIPLE = 32

_SCALE_ATTR = "_sm120_rowwise_scale"
#: The loader the layer was built with, kept so a reload can still use it.
_NATIVE_LOADER_ATTR = "_sm120_native_weight_loader"
_online_fp8_enabled = False
#: The accepted small-batch companions of this selection (W8A16 GEMV, fused
#: gated norm, MTP FC GEMV, draft MoE GEMV, fast router).  One resolution per
#: launch, published next to the FP8 switch, so the bundle engages and retreats
#: as one decision instead of five flags that can disagree across imports.
_fast_paths_enabled = False

#: Hard row limit of the donor dense GEMV (``assert M <= 16`` in
#: ``sglang.srt.layers.quantization.w8a16_gemv``); the budget knob may only
#: shrink it, and the kernels' own low-row shapes stop at the C5 verify width.
_W8A16_GEMV_DONOR_MAX_M = 16

#: Cache-identity marker for the effective row-wise FP8 representation.  A
#: precision switch is a representation switch, so persistence namespaces must
#: be keyed on this value (what is actually resident), not on the requested
#: Boolean.  ``OFF_PRECISION`` names the untouched-checkpoint layout that
#: automatic-off and explicit-off share.
ROWWISE_FP8_PRECISION = "rowwise_fp8"
OFF_PRECISION = "false"

#: The checkpoint-metadata families of the Flash-Next (qwen4_exp) model family:
#: the only modules that consume this conversion.
_FLASH_NEXT_METADATA = (
    "qwen4expforconditionalgeneration",
    "qwen4_exp",
    "qwen4_exp_text",
)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def _resolve_switch(
    requested: Optional[bool],
    *,
    cuda_available: bool,
    capability: Optional[tuple[int, int]],
    model_eligible: bool,
) -> bool:
    """Resolve one tri-state request to whether the conversion is on.

    ``None`` (unset/blank) selects automatically: exact SM120 plus a recognized
    Flash-Next checkpoint.  ``False`` propagates the opt-out silently.  ``True``
    keeps the fail-loud contract for explicit requests this build cannot serve.
    """
    return _resolve_bundle(
        requested,
        cuda_available=cuda_available,
        capability=capability,
        model_eligible=model_eligible,
    )[0]


def _resolve_bundle(
    requested: Optional[bool],
    *,
    cuda_available: bool,
    capability: Optional[tuple[int, int]],
    model_eligible: bool,
) -> tuple[bool, bool]:
    """``(rowwise FP8, accepted small-batch fast paths)`` for one request.

    Automatic selection turns both on together, and both off together: they are
    the measured bundle, and a kernel fast path with no FP8 conversion behind it
    would either die on the first ``float8_e4m3fn`` load or force the very BF16
    dequantization this conversion exists to avoid.

    An explicit ``true`` still selects FP8 on an exact-SM120 machine that this
    metadata does not recognize as Flash-Next, and keeps the bundle off there:
    the fast-path tiles and the draft-MoE one-token GEMV were tuned and accepted
    for Flash-Next, so an explicit request on an unfamiliar model gets the
    already-tested FP8 GEMM dispatch it asked for, not a foreign kernel.

    ``False`` propagates the opt-out silently and takes the companions with it.
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
        return True, bool(model_eligible)
    automatic = bool(cuda_available and capability == (12, 0) and model_eligible)
    return automatic, automatic


def configure_online_fp8(
    requested: Optional[bool],
    *,
    cuda_available: bool,
    capability: Optional[tuple[int, int]],
    model_eligible: bool = False,
) -> bool:
    """Resolve and publish the process switch for one launch.

    ``requested`` is the tri-state reading of ``SGLANG_SM120_ONLINE_MXFP8``;
    ``model_eligible`` comes from the loaded checkpoint metadata, never from a
    directory or file name.  Calling it again with the same arguments is a
    no-op, so every layer of a shard-resolved launch sees one decision.
    """
    global _online_fp8_enabled, _fast_paths_enabled
    _online_fp8_enabled, _fast_paths_enabled = _resolve_bundle(
        requested,
        cuda_available=cuda_available,
        capability=capability,
        model_eligible=model_eligible,
    )
    return _online_fp8_enabled


def online_fp8_enabled() -> bool:
    return _online_fp8_enabled


def configure_for_model(model_config: object) -> bool:
    """``configure_online_fp8`` resolved from one model's own configuration.

    For a decision that must not wait for some other module's conversion hook:
    a draft (NEXTN) MoE block can be built before anything has published the
    bundle, and would otherwise leave the draft layers on the CUTLASS runner.
    The same arguments resolve to the same decision, so a caller that runs after
    the per-layer hook changes nothing; a caller that runs before it must treat
    this as arming only, never disarming, because a draft's own metadata is not
    the launch's eligibility evidence.  An explicit request this build cannot
    serve still fails the boot from here rather than later.
    """
    available = torch.cuda.is_available()
    hf_config = getattr(model_config, "hf_config", model_config)
    return configure_online_fp8(
        _env_request(),
        cuda_available=available,
        capability=torch.cuda.get_device_capability() if available else None,
        model_eligible=checkpoint_eligible(hf_config, torch.get_default_dtype()),
    )


def fast_paths_enabled() -> bool:
    """Whether unset companion fast-path switches follow the FP8 selection.

    Published by ``configure_online_fp8`` alongside the FP8 switch, so one
    launch's decision cannot half-apply across module import order.
    """
    return _fast_paths_enabled


def gated_by_fast_paths(requested: Optional[bool]) -> bool:
    """Resolution shared by every companion switch of the accepted bundle.

    A saved explicit ``true``/``false`` is honored verbatim as the private
    compatibility/debug escape hatch; unset or blank follows the default
    selection of the rowwise-FP8 bundle, so the whole bundle engages or retreats
    together and never leaks onto other models.
    """
    return fast_paths_enabled() if requested is None else requested


def w8a16_gemv_enabled() -> bool:
    """Whether a low-row FP8 GEMM may route through the donor W8A16 GEMV."""
    from sglang.srt.environ import envs

    return gated_by_fast_paths(envs.SGLANG_FP8_W8A16_GEMV.get())


def w8a16_gemv_row_budget() -> int:
    """Row budget, clamped to what the ported kernel promises (never above 16)."""
    from sglang.srt.environ import envs

    return max(1, min(envs.SGLANG_FP8_W8A16_GEMV_MAX_M.get(), _W8A16_GEMV_DONOR_MAX_M))


def w8a16_gemv_supported(
    hidden_states: torch.Tensor, weight: torch.Tensor, scale: Optional[torch.Tensor]
) -> bool:
    """Whether one resident rowwise-FP8 projection call takes the GEMV.

    The donor's contract (``Fp8LinearMethod._w8a16_gemv_ok`` in its fp8.py),
    restated for this candidate's two resident layouts: weight-only FP8 with one
    FP32 scale per output row, no activation quantization, a BF16 2-D activation
    within the row budget, and a weight whose K axis is contiguous -- which is
    how the row-wise ``Parameter`` is stored for the head, and what
    ``layer.weight.t()`` restores for a projection whose resident view is
    ``[K, N]``.

    Everything the kernel cannot serve stays on the path it already had instead
    of being approximated: the feature switch, the row budget (the C5 verify
    width of 6 is inside it, the C6 target's 24 rows are not), BF16 activations
    on the weight's device, and a scale that is exactly one FP32 value per output
    row in row order.
    """
    if scale is None or hidden_states.dim() != 2:
        return False
    return bool(
        w8a16_gemv_enabled()
        and _online_fp8_enabled
        and 1 <= hidden_states.shape[0] <= w8a16_gemv_row_budget()
        and hidden_states.dtype is torch.bfloat16
        and hidden_states.stride(1) == 1
        and hidden_states.shape[1] == weight.shape[1]
        and hidden_states.device == weight.device
        and weight.dtype is torch.float8_e4m3fn
        and weight.stride(1) == 1
        and scale.dtype is torch.float32
        and 1 <= scale.dim() <= 2
        and scale.is_contiguous()
        and scale.numel() == weight.shape[0]
        # [N], [N, 1] and [1, N] are the same row order; a square tensor is not.
        and (scale.dim() == 1 or min(scale.shape) == 1)
        and scale.device == weight.device
    )


def _prealloc_w8a16_gemv_scratch(device: torch.device) -> None:
    """Materialize the GEMV's split-K scratch outside any graph pool.

    Every hook here runs before CUDA graphs are captured (weight postprocess,
    ``post_load_weights``, draft head adoption), and the ported kernel refuses
    to allocate during a capture rather than tie a lifetime-long buffer to one
    graph's private pool.
    """
    if not w8a16_gemv_enabled():
        return
    from sglang.srt.layers.quantization.w8a16_gemv import prealloc

    prealloc(device)


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

    Accepts the loaded HF config object or a raw ``config.json`` mapping.
    Unknown architectures stay ineligible and keep their original paths,
    instead of guessing at a representation nobody asked for.
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
    return flagged is True or _metadata_flag_true(text_config, key, depth + 1)


def checkpoint_eligible(hf_config: object, compute_dtype: object) -> bool:
    """Whether one launch's actual configuration can run row-wise FP8.

    Three parts, all of them representation contracts rather than preferences:
    Flash-Next (qwen4_exp) metadata, because only that family's modules consume
    the conversion; a BF16 compute dtype, because the conversion quantizes a
    resident BF16 checkpoint weight and refuses any other dtype (float16
    included) -- automatic selection turns off instead of booting into that
    rejection; and untied input/lm_head weights, because a tied head would be
    converted for one consumer and read as BF16 by the other.
    """
    if not flash_next_metadata(hf_config):
        return False
    if compute_dtype is not torch.bfloat16:
        return False
    return not _metadata_flag_true(hf_config, "tie_word_embeddings")


def flash_next_eligible(model_config: object) -> bool:
    """``checkpoint_eligible`` for a served ``ModelConfig`` (or an HF config)."""
    hf_config = getattr(model_config, "hf_config", model_config)
    return checkpoint_eligible(hf_config, getattr(model_config, "dtype", None))


def resolve_precision(
    requested: Optional[bool],
    *,
    cuda_available: bool,
    capability: Optional[tuple[int, int]],
    model_eligible: bool,
) -> str:
    """Effective-precision identity for one launch, resolvable before startup.

    Automatic and explicit-on select the same representation and must agree on
    one identity; automatic-off and explicit-off keep the untouched-checkpoint
    identity they already had.  An explicit request names the FP8 build even on
    hardware where the runtime fails boot before touching any cache.
    """
    return (
        ROWWISE_FP8_PRECISION
        if _resolve_switch(
            requested,
            cuda_available=cuda_available,
            capability=capability,
            model_eligible=model_eligible,
        )
        else OFF_PRECISION
    )


def _read_checkpoint_metadata(model_path: str) -> Optional[dict]:
    try:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def launch_precision(
    model_path: str, *, compute_dtype: torch.dtype = torch.bfloat16
) -> str:
    """Effective precision of one launch, from the checkpoint's own metadata.

    Shared by the runtime default and any recipe/container identity probe, so
    two ways of starting the same checkpoint cannot disagree about which cache
    they are allowed to read.  ``compute_dtype`` defaults to the dtype the
    recipes pin in their own launch line; the head-tying half is read from the
    checkpoint.
    """
    metadata = _read_checkpoint_metadata(model_path)
    available = torch.cuda.is_available()
    return resolve_precision(
        _env_request(),
        cuda_available=available,
        capability=torch.cuda.get_device_capability() if available else None,
        model_eligible=checkpoint_eligible(metadata, compute_dtype),
    )


def _env_request() -> Optional[bool]:
    from sglang.srt.environ import envs

    return envs.SGLANG_SM120_ONLINE_MXFP8.get()


# --------------------------------------------------------------------------- #
# Row-wise FP8 weight representation
# --------------------------------------------------------------------------- #


def quantize_rowwise_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D BF16 tensor into E4M3 codes with one FP32 scale per row.

    Chunked so the FP32 working set stays bounded on a wide head, and clamped so
    an all-zero row cannot divide by zero.
    """
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


def rowwise_scale_of(weight: object) -> Optional[torch.Tensor]:
    return getattr(weight, _SCALE_ATTR, None)


def is_rowwise_fp8_weight(weight: object) -> bool:
    """Whether ``weight`` is this conversion's FP8 weight with its scale bound."""
    return (
        isinstance(weight, torch.Tensor)
        and weight.dtype is torch.float8_e4m3fn
        and rowwise_scale_of(weight) is not None
    )


def require_rowwise_scale(weight: torch.Tensor) -> torch.Tensor:
    """The row scale of an online-FP8 weight, or a loud failure.

    A float8 weight without its scale is not a format to guess about: the codes
    would be read as BF16-sized magnitudes.
    """
    scale = rowwise_scale_of(weight)
    if weight.dtype != torch.float8_e4m3fn or scale is None:
        raise RuntimeError("SM120 online FP8 weight is missing its rowwise scale")
    if scale.dim() != 1 or scale.shape[0] != weight.shape[0]:
        raise RuntimeError(
            "SM120 online FP8 rowwise scale does not match the weight: "
            f"weight={tuple(weight.shape)} scale={tuple(scale.shape)}"
        )
    return scale


def dequantize_rowwise_weight(
    weight: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Materialize a BF16 operand from resident row-wise FP8 storage."""
    scale = require_rowwise_scale(weight)
    return (weight.float() * scale[:, None]).to(dtype)


def _copy_parameter_attrs(source: torch.Tensor, destination: torch.Tensor) -> None:
    """Retain loader/sharding metadata when replacing a whole Parameter.

    This module's own attributes do not travel: the reload loader belongs to the
    head it writes, so a draft's row-selected copy carrying it would re-quantize
    the full target head against its own smaller shape.
    """
    for name, value in vars(source).items():
        if name not in (_SCALE_ATTR, _NATIVE_LOADER_ATTR, "weight_loader"):
            setattr(destination, name, value)


def _rowwise_parameter(
    weight: torch.Tensor, source: torch.Tensor
) -> torch.nn.Parameter:
    quantized, scale = quantize_rowwise_fp8(weight)
    parameter = torch.nn.Parameter(quantized, requires_grad=False)
    _copy_parameter_attrs(source, parameter)
    setattr(parameter, _SCALE_ATTR, scale)
    return parameter


def _bf16_load_stage(weight: torch.Tensor) -> torch.Tensor:
    """A transient BF16 copy of resident row-wise FP8 storage, chunk-bound.

    The native loader writes BF16 into the layout it knows -- shard offsets,
    padding rows -- so that layout is where the values have to be placed before
    anyone quantizes them.  Starting the stage from what is resident (rather than
    zeros) keeps a loader that is called once per shard from wiping the rows it
    was not asked about.  This is staging for one reload, not a BF16 shadow of
    the head: the memory this conversion freed stays freed.
    """
    scale = require_rowwise_scale(weight)
    stage = torch.empty(weight.shape, dtype=torch.bfloat16, device=weight.device)
    rows_per_chunk = max(
        1, _DEQUANT_TARGET_BYTES // max(1, weight.shape[1] * torch.float32.itemsize)
    )
    for row in range(0, weight.shape[0], rows_per_chunk):
        block = weight[row : row + rows_per_chunk]
        stage[row : row + rows_per_chunk] = (
            block.float() * scale[row : row + rows_per_chunk, None]
        )
    return stage


def _reload_rowwise_weight(
    linear: torch.nn.Module,
    parameter: torch.Tensor,
    loaded_weight: torch.Tensor,
    *args,
    **kwargs,
) -> None:
    """Weight-loader replacement that keeps the native layout contract.

    A served ``ParallelLMHead`` is a padded, per-rank shard of the checkpoint
    head, and the head's own ``weight_loader`` is what narrows the full
    ``[org_vocab, hidden]`` checkpoint tensor to this rank's rows and zeroes the
    padding.  An exact-shape quantizing replacement throws that away: the same
    tensor that loaded fine the first time arrives at a quantizer that only
    accepts an already-placed weight.  So the original loader still places the
    BF16 values -- into a stage buffer of the resident shape -- and only then are
    values and their row scales written together as FP8.

    The new codes and scales go into the tensors that are already resident, so a
    draft head adopted by ``set_embed_and_head``/``set_lm_head_from_target`` and a
    captured CUDA graph cannot be left holding the weights the target replaced.
    Without any of this, a reload would copy BF16 into FP8 storage and keep the
    previous row scales beside the new codes: a silently wrong magnitude.
    """
    if loaded_weight.dtype != torch.bfloat16:
        raise RuntimeError(
            "SM120 online FP8 reload requires a BF16 checkpoint weight, got "
            f"{loaded_weight.dtype}"
        )
    native = getattr(linear, _NATIVE_LOADER_ATTR, None)
    if not callable(native):
        raise RuntimeError(
            "SM120 online FP8 reload found no native weight loader on "
            f"{type(linear).__name__}: this load path cannot place the "
            "checkpoint's rows, refusing to guess the layout"
        )
    if getattr(native, "__self__", None) is parameter:
        # A loader bound to the parameter writes to itself and would ignore the
        # stage; this conversion is scoped to layer-bound loaders.
        raise RuntimeError(
            "SM120 online FP8 reload cannot delegate to a parameter-bound "
            f"weight loader ({native!r})"
        )
    if loaded_weight.device != parameter.device:
        loaded_weight = loaded_weight.to(parameter.device)
    stage = _bf16_load_stage(parameter)
    _copy_parameter_attrs(parameter, stage)
    native(stage, loaded_weight, *args, **kwargs)
    quantized, scale = quantize_rowwise_fp8(stage)
    parameter.data.copy_(quantized)
    require_rowwise_scale(parameter).copy_(scale)


def replace_linear_weight_rowwise_fp8(linear: torch.nn.Module) -> int:
    """Convert one resident BF16 linear weight in place; repeat calls no-op.

    Returns the byte count the replaced weight carried, or 0 when the weight is
    already row-wise FP8.  Anything else (a non-BF16 weight, a non-Parameter or
    a 1-D tensor) is a load path this conversion was not selected for, and fails
    loudly instead of being quietly left mixed with the converted modules.
    """
    weight = getattr(linear, "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or weight.dim() != 2:
        raise RuntimeError("SM120 online FP8 requires a 2-D Parameter weight")
    if is_rowwise_fp8_weight(weight):
        return 0
    if weight.dtype != torch.bfloat16:
        raise RuntimeError(
            f"SM120 online FP8 expected a BF16 resident weight, got {weight.dtype}"
        )
    original_bytes = weight.numel() * weight.element_size()
    new_parameter = _rowwise_parameter(weight.data, weight)
    original_loader = getattr(weight, "weight_loader", None)
    if callable(original_loader):
        # A later update goes through the loader this layer was built with, so
        # padding and shard placement survive alongside the new scales.
        setattr(linear, _NATIVE_LOADER_ATTR, original_loader)
        new_parameter.weight_loader = functools.partial(_reload_rowwise_weight, linear)
    linear.weight = new_parameter
    # Head conversion is the last hook before CUDA graphs are captured.
    _prealloc_w8a16_gemv_scratch(new_parameter.device)
    return original_bytes


def select_rowwise_weight_rows(
    weight: torch.Tensor, row_indices: torch.Tensor
) -> torch.Tensor:
    """Row-select an online-FP8 weight without dropping the matching scales.

    Used when a draft head is the target head restricted to a hot token set:
    the selection has to move the values and their per-row scales as one unit.
    """
    scale = require_rowwise_scale(weight)
    selected_data = weight.index_select(0, row_indices)
    selected: Any = (
        torch.nn.Parameter(selected_data, requires_grad=False)
        if isinstance(weight, torch.nn.Parameter)
        else selected_data
    )
    _copy_parameter_attrs(weight, selected)
    setattr(selected, _SCALE_ATTR, scale.index_select(0, row_indices))
    # A draft that adopts this view is still before any graph capture, and the
    # GEMV scratch is per-device and lifetime-long, so it comes with the head.
    _prealloc_w8a16_gemv_scratch(weight.device)
    return selected


# --------------------------------------------------------------------------- #
# Eligible-linear installation
# --------------------------------------------------------------------------- #


_ROWWISE_FP8_METHOD: Optional[Any] = None
#: Weight signatures already reported, so a 48-layer model logs each shape once.
_LOGGED_SIGNATURES: set[tuple] = set()


def _build_rowwise_fp8_method() -> Any:
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

    class OnlineRowwiseFp8LinearMethod(Fp8LinearMethod):
        """Upstream's online-FP8 linear, pinned to the row-wise outcome.

        Deliberately not ``block_quant``/MXFP8: with ``weight_block_size=None``
        and ``is_checkpoint_fp8_serialized=False``, the existing
        ``Fp8LinearMethod.process_weights_after_loading`` quantizes the loaded
        BF16 weight once over each whole row (one FP32 scale per output
        channel), stores the standard resident ``weight`` [K, N] plus
        ``weight_scale`` [N], and keeps the tested ``apply_fp8_linear``
        per-token/per-channel dispatch -- the same GEMM path an FP8 checkpoint
        would use.  This subclass only adds the invariants the Flash-Next
        conversion depends on, so a generic ``--quantization fp8`` user is
        untouched and this one cannot silently degrade to another layout.
        """

        def process_weights_after_loading(self, module) -> None:
            weight = getattr(module, "weight", None)
            # Fail loud BEFORE the base postprocess mutates anything: a repeat
            # postprocess would feed the resident FP8 back through the
            # quantizer (re-interpreting the [K, N] view as [N, K]) and a
            # non-BF16 weight never came from the checkpoint load this
            # conversion is scoped to.
            if (
                not isinstance(weight, torch.Tensor)
                or weight.dim() != 2
                or weight.dtype != torch.bfloat16
            ):
                raise RuntimeError(
                    "Flash-Next online rowwise FP8 expects the original 2-D "
                    "BF16 checkpoint weight before post-processing; got "
                    f"dtype={getattr(weight, 'dtype', None)} shape="
                    f"{tuple(getattr(weight, 'shape', ()))} for "
                    f"{type(module).__name__} (already quantized, or an "
                    "unexpected load path -- refusing to requantize)"
                )
            super().process_weights_after_loading(module)
            scale = getattr(module, "weight_scale", None)
            # ``weight`` is stored [K, N]; one scale per output channel means
            # exactly one value per column of that view.  Anything else (a
            # per-tensor fallback because this platform has no channelwise FP8
            # GEMM) is a different representation than the one selected.
            if (
                module.weight.dtype != torch.float8_e4m3fn
                or scale is None
                or scale.dtype != torch.float32
                or scale.numel() != module.weight.shape[1]
            ):
                raise RuntimeError(
                    "Flash-Next online rowwise-FP8 post-processing produced "
                    f"invalid weight/scale state for {type(module).__name__}"
                )
            if w8a16_gemv_enabled():
                # Warm-up runs at prefill widths, where every call falls back to
                # the FP8 GEMM, so without this the split-K scratch would first
                # be allocated inside a capture.
                _prealloc_w8a16_gemv_scratch(module.weight.device)
            signature = (
                type(module).__name__,
                tuple(module.weight.shape),
                tuple(scale.shape),
            )
            if signature not in _LOGGED_SIGNATURES:
                _LOGGED_SIGNATURES.add(signature)
                _logger_info(
                    "Flash-Next online rowwise FP8 projection ready: "
                    "module=%s weight(K,N)=%s scale=per-output-channel-fp32 %s",
                    *signature,
                )

        # -- accepted small-batch consumer (SGLANG_FP8_W8A16_GEMV) ----------- #
        #
        # Donor ``Fp8LinearMethod._w8a16_gemv_ok`` / ``apply`` /
        # ``apply_norm_gated`` (sglang-rtxpro6000 @5105985116eb fp8.py:986/1010/
        # 1031), bound to *these* modules only -- a generic ``--quantization fp8``
        # user's method is untouched -- and re-expressed against this candidate's
        # resident layout: ``weight`` is the [K, N] view of [N, K] FP8 storage and
        # ``weight_scale`` holds one FP32 scale per output channel, so the GEMV is
        # handed ``layer.weight.t()`` (an [N, K] weight with K contiguous, the one
        # layout the donor tile table was measured on) and the per-channel test is
        # against ``weight.shape[1]``, this view's output channels.
        #
        # Lifetime/scale invariants of the routed call: the E4M3 codes and their
        # FP32 row scales are the ones this postprocess produced, read from the
        # same resident tensors (no dequantized BF16 copy is materialized, which
        # is what the conversion freed), scaled inside the same kernel that reads
        # the codes.  A call the kernel cannot serve keeps the tested
        # ``apply_fp8_linear`` dispatch on those same tensors; it is never routed
        # through a BF16 cache and never through a GEMV past its row budget.

        def _w8a16_gemv_ok(self, layer, x) -> bool:
            """Whether ``apply`` would route this call through the GEMV.

            ``layer.weight`` is the ``[K, N]`` view of ``[N, K]`` FP8 storage and
            ``layer.weight_scale`` one FP32 scale per output channel, so the
            operand handed over is ``layer.weight.t()`` and the scale-count test
            inside ``w8a16_gemv_supported`` is this view's output-channel count.
            """
            return bool(
                # A fused quant producer can hand this fork's apply() a
                # (fp8, scale) tuple; the donor runtime never did, and the GEMV
                # is weight-only FP8 with no activation scale to read.
                isinstance(x, torch.Tensor)
                and not self.use_marlin
                and not self.block_quant
                and not self.use_mxfp8
                and w8a16_gemv_supported(x, layer.weight.t(), layer.weight_scale)
            )

        def apply(self, layer, x, bias=None):
            if self._w8a16_gemv_ok(layer, x):
                from sglang.srt.layers.quantization.w8a16_gemv import w8a16_gemv

                y = w8a16_gemv(x, layer.weight.t(), layer.weight_scale)
                if bias is not None:
                    y = y + bias
                return y
            return super().apply(layer, x, bias=bias)

        def apply_norm_gated(
            self,
            layer,
            x,
            z,
            norm_weight,
            group_size,
            eps,
            sigmoid_gate=False,
            bias=None,
        ):
            """``apply(layer, gated_rms_norm(x, z))`` in one launch, or ``None``.

            Each contiguous ``group_size`` slice of an ``x`` row is one RMS
            group, normalised, scaled by ``norm_weight``, gated by ``silu(z)``
            (or ``sigmoid(z)``) and rounded to BF16 before the same k-loop
            consumes it.  Numerics: the FP32 sum of squares is re-associated
            into the GEMV tile's thread layout, so the pre-round value can
            differ in the last FP32 bit; after the BF16 round the result is
            within 1 ulp of the separate norm + GEMV pair.  ``None`` means this
            call cannot take that path, and the caller keeps both kernels.
            """
            if bias is not None or not self._w8a16_gemv_ok(layer, x):
                return None
            from sglang.srt.layers.quantization.w8a16_gemv import (
                w8a16_gemv_norm_gated,
                w8a16_gemv_norm_gated_supported,
            )

            w = layer.weight.t()
            if not w8a16_gemv_norm_gated_supported(
                x, w, layer.weight_scale, z, norm_weight, group_size
            ):
                return None
            return w8a16_gemv_norm_gated(
                x,
                w,
                layer.weight_scale,
                z,
                norm_weight,
                group_size,
                eps,
                sigmoid_gate,
            )

    config = Fp8Config(
        is_checkpoint_fp8_serialized=False,
        activation_scheme="dynamic",
    )
    method = OnlineRowwiseFp8LinearMethod(config)
    if method.block_quant or method.use_mxfp8:
        raise RuntimeError(
            "Flash-Next online FP8 requires the row-wise (non-block) FP8 "
            "linear method; this build resolved a block-quantized one"
        )
    return method


def _logger_info(message: str, *args) -> None:
    import logging

    logging.getLogger(__name__).info(message, *args)


def rowwise_fp8_method() -> Any:
    """The single shared online row-wise FP8 linear method.

    One instance for every converted projection, exactly as one
    ``--quantization fp8`` config serves every FP8 linear: the method holds no
    per-layer state, and building it resolves the FP8 GEMM backend once instead
    of once per layer.
    """
    global _ROWWISE_FP8_METHOD
    if _ROWWISE_FP8_METHOD is None:
        _ROWWISE_FP8_METHOD = _build_rowwise_fp8_method()
    return _ROWWISE_FP8_METHOD


def unquantized_method_types() -> tuple[type, ...]:
    """The methods that own no representation of their own.

    A module carrying one of these has a BF16 (or fp16/fp32) weight that nothing
    else intends to reinterpret, so converting it is this selection's to do. A
    method that came from the checkpoint's quantization config -- FP8, NVFP4,
    W4A16, or an online-quant method that quantizes at post-load -- owns the
    weight's representation even while its storage is still BF16, and that
    post-load hook runs after this selection has finished.
    """
    from sglang.srt.layers.quantization.unquant import (
        UnquantizedEmbeddingMethod,
        UnquantizedLinearMethod,
    )

    # The embedding variant is a sibling, not a subclass: an lm_head can carry
    # either one depending on whether the checkpoint config named the prefix.
    return (UnquantizedLinearMethod, UnquantizedEmbeddingMethod)


def otherwise_unquantized(module: torch.nn.Module) -> bool:
    """Whether ``module`` is convertible under the rule the walker applies."""
    return isinstance(getattr(module, "quant_method", None), unquantized_method_types())


def install_rowwise_fp8_on_eligible_linears(
    root: torch.nn.Module,
    *,
    enabled: bool,
    method_factory: Callable[[], Any] = rowwise_fp8_method,
    unquantized_method_type: Optional[type] = None,
    excluded_module_type: type | tuple[type, ...] = (),
    excluded_module_names: Iterable[str] = ("gate",),
) -> list[str]:
    """Install row-wise FP8 on the eligible, otherwise-unquantized linears.

    Walks ``root``'s subtree and swaps the method of each ``LinearBase`` that is
    currently unquantized, holds a resident BF16 2-D weight and is large enough
    for the FP8 shapes.  Modules whose checkpoint already declared a
    representation (NVFP4/FP8/W4A16 methods, an FP8 PLE table) never satisfy the
    BF16+unquantized test, and the excluded names/subtrees are skipped without
    descending, so a router gate or a PLE block cannot be reached by a rename of
    something else.  Returns the dotted names of the converted modules.

    ``method_factory``, ``unquantized_method_type`` (a type or tuple of types)
    and ``excluded_module_type`` are parameters rather than imports so the
    selection rule stays testable without the serving runtime's quantization
    stack; leaving ``unquantized_method_type`` unset uses
    ``unquantized_method_types``, the same rule ``otherwise_unquantized`` applies
    to the output head.
    """
    if not enabled:
        return []
    if unquantized_method_type is None:
        unquantized_method_type = unquantized_method_types()
    if not excluded_module_type:
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        excluded_module_type = FusedMoE

    candidates: list[tuple[str, torch.nn.Module]] = []
    excluded_names = frozenset(excluded_module_names)

    def visit(module: torch.nn.Module, prefix: str) -> None:
        for child_name, child in module.named_children():
            name = f"{prefix}.{child_name}" if prefix else child_name
            if child_name in excluded_names:
                continue
            if isinstance(child, excluded_module_type):
                continue
            weight = getattr(child, "weight", None)
            if (
                isinstance(
                    getattr(child, "quant_method", None), unquantized_method_type
                )
                and isinstance(weight, torch.nn.Parameter)
                and weight.dim() == 2
                and weight.dtype == torch.bfloat16
                and weight.shape[0] >= _MIN_FP8_OUTPUT_ROWS
                and weight.shape[1] >= _MIN_FP8_INPUT_COLUMNS
                and weight.shape[1] % _FP8_COLUMN_MULTIPLE == 0
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


# --------------------------------------------------------------------------- #
# Output-head dispatch
# --------------------------------------------------------------------------- #


def rowwise_fp8_lm_head_logits(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Project through a resident row-wise FP8 lm_head, never via BF16 storage.

    The scale that gives the E4M3 codes their magnitude must be applied in the
    same pass, and the head's BF16 copy is precisely what the conversion
    removed: dequantizing it back to a resident parameter would restore the
    memory the conversion exists to free.  Small row counts (decode, MTP
    verification) use the low-row kernel; anything wider uses the chunked
    dequantized projection, which needs no extra resident state.
    """
    scale = require_rowwise_scale(weight)
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
    if w8a16_gemv_supported(hidden_2d, weight, scale):
        # Decode and MTP verification widths on the accepted low-row kernel: the
        # same resident E4M3 codes and the same FP32 row scales, applied in the
        # same kernel, with no BF16 materialization behind them.
        from sglang.srt.layers.quantization.w8a16_gemv import w8a16_gemv

        return w8a16_gemv(hidden_2d, weight, scale).reshape(
            *original_shape, weight.shape[0]
        )
    from sglang.kernels.ops.gemm.sm120_rowwise_fp8 import rowwise_fp8_linear

    return rowwise_fp8_linear(hidden_2d, weight, scale).reshape(
        *original_shape, weight.shape[0]
    )
