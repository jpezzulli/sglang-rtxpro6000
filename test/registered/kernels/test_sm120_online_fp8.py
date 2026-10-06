"""Actual-kernel numerics and CUDA-graph coverage for SM120 online FP8."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.gemm import sm120_w8a16_gemv
from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    configure_online_fp8,
    dequantize_rowwise_weight,
    replace_linear_weight_rowwise_fp8,
    rowwise_fp8_lm_head_logits,
    rowwise_scale_of,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")


def _is_exact_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


pytestmark = pytest.mark.skipif(
    not _is_exact_sm120(), reason="online FP8 kernels require exactly SM120"
)

HIDDEN_SIZE = 2560
VOCAB_ROWS = 1024
# The donor's Flash-Next head shapes; the candidate GEMV's tuned tiles are keyed
# on them, and a vocab-sharded (TP2) head takes the generic planner instead.
DRAFT_HEAD_ROWS = 32768
HC_COUNT = 4
HC_LOWRANK = 320

# Row counts: 1 decode, 4/12/16 the W4 C3/C4 verify widths, 24 C6 verification
# (which must stay on the original larger-row path), 33 the dequantizing fallback.
LM_HEAD_ROWS = [1, 4, 12, 16, 24, 33]
GEMV_CANDIDATES = [False, True]


@pytest.fixture(params=GEMV_CANDIDATES)
def gemv_candidate(request, monkeypatch) -> bool:
    """Run the coverage twice: original path and opt-in candidate GEMV."""
    if request.param:
        monkeypatch.setenv(sm120_w8a16_gemv.GEMV_ENV, "1")
    else:
        monkeypatch.delenv(sm120_w8a16_gemv.GEMV_ENV, raising=False)
    return bool(request.param)


def _randn(shape, *, seed: int, scale: float) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.randn(
            shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * scale
    )


def _assert_normalized_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_nrmse: float,
    min_cosine: float,
) -> None:
    assert actual.shape == expected.shape
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    error_rms = (actual_fp32 - expected_fp32).square().mean().sqrt().item()
    reference_rms = expected_fp32.square().mean().sqrt().item()
    nrmse = error_rms / max(reference_rms, 1e-8)
    cosine = F.cosine_similarity(
        actual_fp32.flatten(), expected_fp32.flatten(), dim=0
    ).item()
    assert nrmse <= max_nrmse, f"NRMSE {nrmse:.6f} exceeds {max_nrmse:.6f}"
    assert cosine >= min_cosine, f"cosine {cosine:.6f} is below {min_cosine:.6f}"


@pytest.fixture(scope="module")
def rowwise_lm_head_weight() -> torch.nn.Parameter:
    linear = nn.Linear(
        HIDDEN_SIZE,
        VOCAB_ROWS,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )
    linear.weight.data.copy_(
        _randn(
            (VOCAB_ROWS, HIDDEN_SIZE),
            seed=101,
            scale=1.0 / math.sqrt(HIDDEN_SIZE),
        )
    )
    replace_linear_weight_rowwise_fp8(linear)
    return linear.weight


def _rowwise_reference(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dense_weight = dequantize_rowwise_weight(weight, torch.bfloat16)
    return hidden.bfloat16() @ dense_weight.T


@pytest.mark.parametrize("rows", LM_HEAD_ROWS)
def test_rowwise_lm_head_matches_dequantized_bf16_reference(
    rowwise_lm_head_weight: torch.Tensor, rows: int, gemv_candidate: bool
) -> None:
    hidden = _randn((rows, HIDDEN_SIZE), seed=200 + rows, scale=0.25)
    # The candidate owns a call iff it is gated on AND the rows fit its own limit;
    # everything else (gate off, C6's 24 rows, the 33-row fallback) stays on the
    # original implementation and must still validate numerically.
    uses_candidate = gemv_candidate and rows <= sm120_w8a16_gemv.MAX_ROWS
    assert (
        sm120_w8a16_gemv.lowrow_gemv_supported(
            hidden, rowwise_lm_head_weight, rowwise_scale_of(rowwise_lm_head_weight)
        )
        is uses_candidate
    )

    actual = rowwise_fp8_lm_head_logits(hidden, rowwise_lm_head_weight)
    expected = _rowwise_reference(hidden, rowwise_lm_head_weight)

    _assert_normalized_error(actual, expected, max_nrmse=0.025, min_cosine=0.999)


@pytest.mark.parametrize("rows", LM_HEAD_ROWS)
def test_rowwise_lm_head_cuda_graph_replays_mutated_input(
    rowwise_lm_head_weight: torch.Tensor, rows: int, gemv_candidate: bool
) -> None:
    static_hidden = _randn((rows, HIDDEN_SIZE), seed=300 + rows, scale=0.25)

    # Compile/JIT and populate allocator state before capture.
    for _ in range(2):
        rowwise_fp8_lm_head_logits(static_hidden, rowwise_lm_head_weight)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = rowwise_fp8_lm_head_logits(static_hidden, rowwise_lm_head_weight)

    changed_hidden = _randn((rows, HIDDEN_SIZE), seed=400 + rows, scale=0.25)
    before = _rowwise_reference(static_hidden, rowwise_lm_head_weight)
    static_hidden.copy_(changed_hidden)
    expected = _rowwise_reference(changed_hidden, rowwise_lm_head_weight)
    assert not torch.equal(before, expected)

    graph.replay()
    torch.cuda.synchronize()
    actual = graph_output.clone()

    _assert_normalized_error(actual, expected, max_nrmse=0.025, min_cosine=0.999)


@pytest.fixture(scope="module")
def rowwise_draft_head_weight() -> torch.nn.Parameter:
    """A head at the donor's tuned draft shape (single-split tiles)."""
    linear = nn.Linear(
        HIDDEN_SIZE,
        DRAFT_HEAD_ROWS,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )
    linear.weight.data.copy_(
        _randn(
            (DRAFT_HEAD_ROWS, HIDDEN_SIZE),
            seed=102,
            scale=1.0 / math.sqrt(HIDDEN_SIZE),
        )
    )
    replace_linear_weight_rowwise_fp8(linear)
    return linear.weight


@pytest.mark.parametrize("rows", [1, 4, 12, 16])
def test_rowwise_head_shape_matches_dequantized_reference_and_replays(
    rowwise_draft_head_weight: torch.Tensor, rows: int, gemv_candidate: bool
) -> None:
    """Numerics and replay on the affected (tuned) head shape, no BF16 fallback."""
    hidden = _randn((rows, HIDDEN_SIZE), seed=800 + rows, scale=0.25)
    scale = rowwise_scale_of(rowwise_draft_head_weight)
    assert scale.shape == (DRAFT_HEAD_ROWS,)
    assert (
        sm120_w8a16_gemv.lowrow_gemv_supported(hidden, rowwise_draft_head_weight, scale)
        is gemv_candidate
    )

    for _ in range(2):
        rowwise_fp8_lm_head_logits(hidden, rowwise_draft_head_weight)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = rowwise_fp8_lm_head_logits(hidden, rowwise_draft_head_weight)
    hidden.copy_(_randn((rows, HIDDEN_SIZE), seed=900 + rows, scale=0.25))
    graph.replay()
    torch.cuda.synchronize()

    expected = _rowwise_reference(hidden, rowwise_draft_head_weight)
    _assert_normalized_error(graph_output, expected, max_nrmse=0.025, min_cosine=0.999)


def test_flashinfer_cutlass_mxfp8_linear_quantizes_and_applies() -> None:
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import (
        Fp8GemmRunnerBackend,
        Mxfp8DenseGemmBackend,
    )

    class Dense(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.weight = nn.Parameter(weight, requires_grad=False)

    original_backend = fp8_utils.FP8_GEMM_RUNNER_BACKEND
    fp8_utils.FP8_GEMM_RUNNER_BACKEND = Fp8GemmRunnerBackend.FLASHINFER_CUTLASS
    try:
        method = Fp8LinearMethod(
            Fp8Config(
                is_checkpoint_fp8_serialized=False,
                activation_scheme="dynamic",
                use_mxfp8=True,
            )
        )
        assert method.mxfp8_dense_backend is Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS

        original_weight = _randn((256, 256), seed=501, scale=1.0 / 16.0)
        layer = Dense(original_weight.clone())
        method.process_weights_after_loading(layer)
        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight_scale_inv.dtype == torch.uint8

        for rows in (4, 16):
            x = _randn((rows, 256), seed=500 + rows, scale=0.25)
            actual = method.apply(layer, x)
            expected = x.float() @ original_weight.float().T
            _assert_normalized_error(actual, expected, max_nrmse=0.12, min_cosine=0.99)
    finally:
        fp8_utils.FP8_GEMM_RUNNER_BACKEND = original_backend


# (N, K) at a Flash-Next dense-projection width: shared_expert / attention /
# linear-attention / MTP scale, small enough for CI, big enough that the
# underfilled grid the candidate exists for is the grid under test.
DENSE_PROJECTION = (1024, 2560)
DENSE_ROWS = [1, 4, 16, 24]  # 24 is C6 verification: never the candidate


@pytest.fixture(scope="module")
def mxfp8_dense_layer():
    """One online-quantized MXFP8 dense linear, in the stored representation."""
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import (
        Fp8GemmRunnerBackend,
        Mxfp8DenseGemmBackend,
    )

    class Dense(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.weight = nn.Parameter(weight, requires_grad=False)

    n, k = DENSE_PROJECTION
    original_backend = fp8_utils.FP8_GEMM_RUNNER_BACKEND
    fp8_utils.FP8_GEMM_RUNNER_BACKEND = Fp8GemmRunnerBackend.FLASHINFER_CUTLASS
    try:
        method = Fp8LinearMethod(
            Fp8Config(
                is_checkpoint_fp8_serialized=False,
                activation_scheme="dynamic",
                use_mxfp8=True,
            )
        )
        assert method.mxfp8_dense_backend is Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS
        layer = Dense(_randn((n, k), seed=1101, scale=1.0 / math.sqrt(k)).clone())
        method.process_weights_after_loading(layer)
        # The candidate's whole contract is this stored format, so pin it here:
        # fp8 e4m3 values with UE8M0 bytes at [N, K/32], row-major, on the layer.
        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight_scale_inv.dtype == torch.uint8
        assert layer.weight_scale_inv.shape == (n, k // sm120_w8a16_gemv.MXFP8_SF)
        assert layer.weight_scale_inv.format_ue8m0 is True
        assert layer.weight.stride(1) == 1 and layer.weight_scale_inv.stride(1) == 1
        yield method, layer
    finally:
        fp8_utils.FP8_GEMM_RUNNER_BACKEND = original_backend


def _stored_mxfp8_reference(layer: nn.Module) -> torch.Tensor:
    return sm120_w8a16_gemv.dequantize_mxfp8_weight(
        layer.weight, layer.weight_scale_inv, dtype=torch.float32
    )


@pytest.mark.parametrize("rows", DENSE_ROWS)
@pytest.mark.parametrize("gemv_candidate", [False, True])
def test_dense_mxfp8_linear_lowrow_gemv_matches_the_stored_weight(
    mxfp8_dense_layer, rows: int, gemv_candidate: bool, monkeypatch
) -> None:
    """The dense candidate against the dequantized stored weight, gate on and off.

    W8A16 by design: the activation stays BF16, so the candidate is compared with
    the dequantized MXFP8 weight, NOT claimed to be bitwise equal with the
    activation-quantized W8A8 dispatch that the same gate leaves untouched.
    """
    method, layer = mxfp8_dense_layer
    if gemv_candidate:
        monkeypatch.setenv(sm120_w8a16_gemv.MX_GEMV_ENV, "1")
    else:
        monkeypatch.delenv(sm120_w8a16_gemv.MX_GEMV_ENV, raising=False)
    assert sm120_w8a16_gemv.mxfp8_gemv_enabled() is gemv_candidate

    _n, k = DENSE_PROJECTION
    x = _randn((rows, k), seed=1200 + rows, scale=0.25)
    uses_candidate = gemv_candidate and rows <= sm120_w8a16_gemv.MAX_ROWS
    assert (
        sm120_w8a16_gemv.lowrow_mxfp8_gemv_supported(
            x, layer.weight, layer.weight_scale_inv
        )
        is uses_candidate
    )

    actual = method.apply(layer, x)
    expected = x.float() @ _stored_mxfp8_reference(layer).T
    if uses_candidate:
        # Only the fp32 reduction order and the bf16 output round differ.
        _assert_normalized_error(actual, expected, max_nrmse=0.005, min_cosine=0.9999)
    else:
        # Gate off (and C6's 24 rows): the qualified W8A8 dispatch, which also
        # quantizes the activation per 32-column group.
        _assert_normalized_error(actual, expected, max_nrmse=0.12, min_cosine=0.99)


@pytest.mark.parametrize("rows", [4, 16])
def test_dense_mxfp8_linear_lowrow_gemv_cuda_graph_replays(
    mxfp8_dense_layer, rows: int, monkeypatch
) -> None:
    """Split-K dense launches are replayable without a memset between calls."""
    method, layer = mxfp8_dense_layer
    monkeypatch.setenv(sm120_w8a16_gemv.MX_GEMV_ENV, "1")
    _n, k = DENSE_PROJECTION
    x = _randn((rows, k), seed=1300 + rows, scale=0.25)
    # Warm up and materialize the split-K scratch the way weight post-processing
    # does: allocating inside the capture would hand it to that graph's pool.
    sm120_w8a16_gemv.prealloc(x.device)
    for _ in range(2):
        method.apply(layer, x)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = method.apply(layer, x)
    x.copy_(_randn((rows, k), seed=1400 + rows, scale=0.25))
    expected = x.float() @ _stored_mxfp8_reference(layer).T

    # Three replays: the per-N-block counters must return themselves to 0, or the
    # second and third replay drift away from the first.
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        _assert_normalized_error(
            graph_output.clone(), expected, max_nrmse=0.005, min_cosine=0.9999
        )


@pytest.fixture(scope="module")
def rowwise_hyperconnection():
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    configure_online_fp8(
        True,
        cuda_available=True,
        capability=torch.cuda.get_device_capability(),
    )
    try:
        config = HyperConnectionConfig(
            hc_count=HC_COUNT,
            hidden_size=HIDDEN_SIZE,
            params_dtype=torch.bfloat16,
            hc_lowrank=HC_LOWRANK,
            hc_per_branch_norm=True,
        )
        with torch.device("cuda"):
            layer = GatedResidual(
                config, use_mix=True, use_combine=False, online_fp8=True
            )
        # Model construction runs under the configured BF16 default dtype.
        # Reproduce that for the norm parameter without touching FP8 mix weights.
        layer.hc_norm.to(dtype=torch.bfloat16)

        down = _randn(
            (HC_LOWRANK, HC_COUNT * HIDDEN_SIZE),
            seed=601,
            scale=1.0 / math.sqrt(HC_COUNT * HIDDEN_SIZE),
        )
        up = _randn(
            (HC_COUNT * HIDDEN_SIZE, HC_LOWRANK),
            seed=602,
            scale=1.0 / math.sqrt(HC_LOWRANK),
        )
        down_loader = layer.input_mix_weight_down.weight.weight_loader
        up_loader = layer.input_mix_weight_up.weight.weight_loader
        down_loader(layer.input_mix_weight_down.weight, down)
        up_loader(layer.input_mix_weight_up.weight, up)
        assert layer.input_mix_weight_down.weight.dtype == torch.float8_e4m3fn
        assert layer.input_mix_weight_up.weight.dtype == torch.float8_e4m3fn
        del down, up, down_loader, up_loader
        yield layer
    finally:
        configure_online_fp8(False, cuda_available=True, capability=(12, 0))


def _hyperconnection_reference(layer, hyper_input: torch.Tensor) -> torch.Tensor:
    normed = layer.hc_norm(hyper_input)
    down = dequantize_rowwise_weight(layer.input_mix_weight_down.weight, torch.bfloat16)
    up = dequantize_rowwise_weight(layer.input_mix_weight_up.weight, torch.bfloat16)
    mix = F.silu(F.linear(normed, down) / HC_COUNT)
    mix = torch.sigmoid(F.linear(mix, up)).unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
    return (mix * normed.unflatten(-1, (HC_COUNT, HIDDEN_SIZE))).mean(dim=-2)


@pytest.mark.parametrize("rows", [4, 16, 33])
def test_rowwise_hyperconnection_fused_and_prefill_paths(
    rowwise_hyperconnection, rows: int
) -> None:
    from sglang.srt.layers.hc_mix_triton import fused_hc_mix_supported

    hyper_input = _randn((rows, HC_COUNT * HIDDEN_SIZE), seed=700 + rows, scale=0.25)
    normed = rowwise_hyperconnection.hc_norm(hyper_input)
    uses_fused_kernel = fused_hc_mix_supported(
        normed,
        rowwise_hyperconnection.input_mix_weight_down.weight,
        rowwise_hyperconnection.input_mix_weight_up.weight,
    )
    assert uses_fused_kernel is (rows <= 16)

    actual, _ = rowwise_hyperconnection.mix(hyper_input)
    expected = _hyperconnection_reference(rowwise_hyperconnection, hyper_input)

    # Decode rows exercise the persistent fused kernel. At 33 rows the actual
    # GatedResidual path dequantizes transient BF16 operands for prefill.
    _assert_normalized_error(actual, expected, max_nrmse=0.06, min_cosine=0.995)
