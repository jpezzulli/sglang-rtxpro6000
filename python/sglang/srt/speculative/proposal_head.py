# SPDX-License-Identifier: Apache-2.0
"""Optional proposal-only precision for the FR-Spec (hot-vocabulary) head.

FR-Spec proposes with a row-selection of the target's resident ``lm_head`` and
the *unchanged* full target head verifies every proposal, so the proposal head
may live at a lower precision than the verifier without weakening an accepted
token. ``SGLANG_SM120_ONLINE_MXFP8`` already gets the draft to rowwise FP8 --
but that is exactly the target head's own precision, so an FP8-only option
banks nothing here. This module adds one deliberately optional step further:
prepare the already-selected hot rows with the in-tree NVFP4 W4A16 (Marlin)
machinery.

    SGLANG_FR_SPEC_PROPOSAL_HEAD_PRECISION=nvfp4      # default: off

``off`` changes nothing at all: the draft keeps the shared rowwise-FP8 (or BF16)
head, the opt-in low-row W8A16 GEMV still applies to it, and every existing
default/TP2/27B path is byte-identical.

Prepared, not swapped. ``ModelOptNvFp4A16LinearMethod`` needs the whole module
contract -- packed E2M1 values, the FP8-E4M3 group(16) scales in Marlin's
permuted layout, the tensor-wide FP32 ``weight_global_scale``, the logical
``input_/output_size_per_partition`` geometry that Marlin pads around, and the
fixed-size ``workspace`` -- so preparation runs on the draft's own ``lm_head``
*after* ``hot_vocab``'s share, through the method's own
``process_weights_after_loading``. Those tensors are layer state (unlike the
rowwise-FP8 head, whose scale rides on the Parameter), which is why the
``.weight``-only target/draft share cannot carry them and why a weight swap
alone would meet the target's 248,320-row scales.

Ownership. The target's resident Parameter is only ever read; ``shared_tensors``
names the tensors this call must never rewrite (the target head, the shared
embedding) and it refuses rather than quantizing storage the target still uses.
Only the FR-Spec draft takes this path: ``EagleDraftWorker.init_lm_head`` calls
it when a ``--speculative-token-map`` is in play, and the 27B/DFlash and other
draft workers never import it. The target head, verifier, rejection/sampling
semantics, pinned token-map ids and ordering, hot logits width, NEXTN flow, FP8
KV and both model profiles are untouched.

Scale/packing invariant. Quantization runs *after* hot-id selection, on
contiguous raw rows, and only then are values and block scales packed and
permuted -- the Marlin/CUTLASS layouts are tiled, so a prepared FP4 buffer
cannot be indexed with token ids afterwards. When the source head is rowwise
FP8 its selected rows are dequantized to BF16 first (chunked, so the transient
for 65,536 x 2,560 stays near ``_DEQUANT_CHUNK_BYTES``), because a group scale
has to be derived from the row values themselves.

Graph invariant. ``layer.workspace`` is allocated during preparation, which
happens in ``init_lm_head`` -- before ``init_attention_backend`` captures the
draft graphs -- so it lives in the ordinary allocator rather than one graph's
private pool, and being sized by the device's SM count (not the batch) it keeps
one address and size across every captured batch size. ``apply_fp4_marlin_linear``
is a registered custom op with a fake impl, so the projection traces.

Dispatch is untouched: ``LogitsProcessor._compute_lm_head`` already routes a
layer whose prepared state matches its ``ModelOptNvFp4A16LinearMethod`` through
``quant_method.apply`` (see ``should_apply_lm_head_quant_method``), and the
rowwise-FP8 check ahead of it does not fire because the packed weight carries
no row scale. A LoRA-wrapped head is the exception -- it is dispatched through
``lm_head.forward`` before the quant method -- so this mode is scoped to the
non-LoRA FR-Spec draft the profile qualifies.

Size is the point, and it is an estimate, not a measurement: at the pinned
65,536-ID map and hidden width 2,560 the prepared head is ~80 MiB of packed E2M1
plus ~10 MiB of E4M3 group scales (~90 MiB, plus Marlin's small per-device
workspace) against the ~160.25 MiB the rowwise-FP8 head costs today. Whether the
smaller weight read (three NEXTN steps plus draft-extend per request) is worth
anything, and whether the two components below behave at this shape on the
qualified part, are GPU questions -- this fork has never run a dense FP4 Marlin
GEMM and the only in-tree online-NVFP4 packer defaults to FlashInfer's cute-dsl
backend, which this fork reserves for SM100:

* ``flashinfer.nvfp4_quantize(..., backend="cuda")`` at 65,536 x 2,560 on SM120;
* ``gptq_marlin_gemm`` with ``float4_e2m1f`` at the same shape, in a CUDA graph.

Both have prepared, unrun coverage in
``test/registered/kernels/ops/quantization/test_nvfp4_marlin.py``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs

OFF = "off"
NVFP4 = "nvfp4"
#: The exact representations this mode can install. ``off`` keeps the proposal
#: head at the target head's precision (BF16, or rowwise FP8 when the online
#: FP8 head is on); add a value here only with a prepared module contract.
SUPPORTED_PRECISIONS: Tuple[str, ...] = (OFF, NVFP4)

#: Identity strings shared with the FR-Spec launcher, which pins
#: ``--field "<PROPOSAL_HEAD_NAMESPACE_FIELD>=<NVFP4_NAMESPACE_LABEL>"`` only
#: when the mode is on. A mismatch fails closed at storage startup rather than
#: serving a quantized-proposal run out of a root derived as unquantized.
PROPOSAL_HEAD_NAMESPACE_FIELD = "fr_spec_proposal_head_precision"
NVFP4_NAMESPACE_LABEL = "nvfp4_w4a16_marlin"

# NVFP4 blocks are 16 wide and pack two E2M1 values per byte.
NVFP4_GROUP_SIZE = 16
_DEQUANT_CHUNK_BYTES = 64 * 1024 * 1024  # mirrors sm120_online_fp8's chunking


def proposal_head_precision() -> str:
    """The validated proposal precision; an unknown name fails at startup."""
    requested = envs.SGLANG_FR_SPEC_PROPOSAL_HEAD_PRECISION.get() or OFF
    requested = str(requested).strip().lower()
    if requested not in SUPPORTED_PRECISIONS:
        raise RuntimeError(
            "SGLANG_FR_SPEC_PROPOSAL_HEAD_PRECISION must be one of "
            f"{'|'.join(SUPPORTED_PRECISIONS)}, got {requested!r}"
        )
    return requested


def nvfp4_proposal_head_enabled() -> bool:
    return proposal_head_precision() == NVFP4


def namespace_field_value() -> Optional[str]:
    """The persisted-identity value for this run (``None`` while the mode is off)."""
    return NVFP4_NAMESPACE_LABEL if nvfp4_proposal_head_enabled() else None


def _source_rows_bf16(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.dtype]:
    """The draft's selected rows as raw BF16/FP16, values next to their scales.

    An SM120 rowwise-FP8 head is requantized from its dequantized rows -- the
    per-row scale is only an FP8 decode factor, and FP4 groups are 16 wide
    along K -- so data and row scale are reindexed together here.
    """
    from sglang.kernels.ops.gemm.sm120_online_fp8 import rowwise_scale_of

    scale = rowwise_scale_of(weight)
    if scale is None:
        if weight.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(
                "NVFP4 FR-Spec proposal head needs BF16/FP16 or SM120 rowwise "
                f"FP8 rows, got dtype {weight.dtype}"
            )
        return weight.contiguous(), weight.dtype
    if (
        weight.dtype != torch.float8_e4m3fn
        or scale.dim() != 1
        or (scale.shape[0] != weight.shape[0])
    ):
        raise RuntimeError(
            "NVFP4 FR-Spec proposal head received a rowwise-FP8 weight whose "
            f"scale does not describe it: weight={tuple(weight.shape)} "
            f"scale={tuple(scale.shape)}"
        )
    rows, columns = weight.shape
    chunk = max(
        1,
        _DEQUANT_CHUNK_BYTES // max(1, columns * torch.bfloat16.itemsize),
    )
    dequantized = torch.empty(
        (rows, columns), dtype=torch.bfloat16, device=weight.device
    )
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        dequantized[start:stop] = (
            weight[start:stop].float() * scale[start:stop, None]
        ).to(torch.bfloat16)
    return dequantized, torch.bfloat16


def _nvfp4_quantize(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack raw rows with the existing in-tree online-NVFP4 converter.

    ``ModelOptNvFp4OnlineFusedMoEMethod._quantize_weight_nvfp4`` owns the scale
    convention (``weight_scale_2 = amax / (448 * 6)``, the reciprocal of the
    encode scale FlashInfer consumes) and returns packed values plus linear
    layout ``[N, K/16]`` E4M3 block scales, which is exactly what
    ``prepare_nvfp4_layer_for_marlin`` permutes. The backend follows this
    fork's SM100/SM120 rule (``fp4_utils``): cute-dsl is the SM100 quantizer,
    SM120 takes FlashInfer's ``cuda`` backend.
    """
    from sglang.srt.layers.quantization.nvfp4_online import (
        ModelOptNvFp4OnlineFusedMoEMethod,
    )
    from sglang.srt.utils.common import is_sm100_supported

    return ModelOptNvFp4OnlineFusedMoEMethod._quantize_weight_nvfp4(
        weight,
        backend="cute-dsl" if is_sm100_supported() else "cuda",
    )


def _require_prepared_contract(lm_head: torch.nn.Module, method) -> None:
    """Fail unless the dispatcher would really use the prepared head.

    ``should_apply_lm_head_quant_method`` is the same predicate
    ``LogitsProcessor._compute_lm_head`` applies (it exists because a draft can
    share the target Parameter while keeping a stale quant method), so asking it
    here keeps the prepared-state contract in one place instead of duplicating
    the attribute list: a half-prepared head fails at startup rather than
    silently projecting a stale tensor through the unquantized path.
    """
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    if not should_apply_lm_head_quant_method(lm_head, method):
        raise RuntimeError(
            "NVFP4 FR-Spec proposal head is not fully prepared for the Marlin "
            "projection: the logits processor would fall back to the "
            f"unquantized path (weight dtype={lm_head.weight.dtype}, packed "
            "shape needs weight_scale/weight_global_scale/workspace and the "
            "input/output_size_per_partition geometry)"
        )


def prepare_nvfp4_proposal_head(
    lm_head: torch.nn.Module,
    *,
    shared_tensors: Sequence[torch.Tensor] = (),
    logger: Optional[object] = None,
) -> bool:
    """Convert one draft ``lm_head`` to NVFP4 W4A16 in place; False when off.

    ``lm_head`` must already hold the FR-Spec selection (``hot_vocab``'s
    assembled head). ``shared_tensors`` names the storage this call may not
    rewrite -- the target's resident head and the shared embedding -- because
    the draft head is derived from both and quantizing one of them in place
    would silently lower the *verifier's* precision too.
    """
    if not nvfp4_proposal_head_enabled():
        return False

    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptFp4Config,
        ModelOptNvFp4A16LinearMethod,
    )

    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        raise RuntimeError(
            "NVFP4 FR-Spec proposal head expects the draft's own 2-D lm_head "
            f"weight, got {type(weight).__name__}"
        )
    for shared in shared_tensors:
        if weight is shared:
            raise RuntimeError(
                "NVFP4 FR-Spec proposal head would quantize a tensor that is "
                "still shared with the target/embedding; the FR-Spec draft must "
                "own its selected rows (hot_vocab returns a fresh head)"
            )
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    if columns % NVFP4_GROUP_SIZE:
        raise RuntimeError(
            "NVFP4 FR-Spec proposal head needs a hidden width that is a "
            f"multiple of {NVFP4_GROUP_SIZE}, got {columns}"
        )

    source, params_dtype = _source_rows_bf16(weight)
    packed, group_scales, weight_scale_2 = _nvfp4_quantize(source)
    # The rows are prepared from resident BF16/FP8 storage, not read from a
    # packed checkpoint, so the config is constructed here (group 16, no
    # serialized FP4 payload) and handed to the existing method.
    config = ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=False, group_size=NVFP4_GROUP_SIZE
    )
    method = ModelOptNvFp4A16LinearMethod(config)
    lm_head.quant_config = config
    lm_head.params_dtype = params_dtype
    # The hot head is replicated full width on every rank (TP1 keeps its own
    # full-width head, TP2 assembles one), so the logical geometry Marlin pads
    # around is the whole hot vocab on each rank.
    lm_head.input_size_per_partition = columns
    lm_head.output_size_per_partition = rows
    lm_head.weight = torch.nn.Parameter(packed, requires_grad=False)
    lm_head.weight_scale = torch.nn.Parameter(group_scales, requires_grad=False)
    lm_head.weight_scale_2 = torch.nn.Parameter(
        weight_scale_2.reshape(1), requires_grad=False
    )
    # Derives weight_global_scale, permutes/pads the packed weight and scales,
    # and allocates the graph-stable workspace.
    method.process_weights_after_loading(lm_head)
    lm_head.quant_method = method
    _require_prepared_contract(lm_head, method)
    if logger is not None:
        logger.info(
            "FR-Spec proposal head: prepared the %d-row x %d hidden NVFP4 "
            "(W4A16/Marlin) draft head from the target's selected rows; the "
            "target head and full-vocab verifier keep their precision, and "
            "only proposals come from this copy.",
            rows,
            columns,
        )
    return True
