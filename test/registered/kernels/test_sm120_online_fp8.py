"""Actual-kernel numerics and CUDA-graph coverage for SM120 online FP8."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    configure_online_fp8,
    dequantize_rowwise_weight,
    replace_linear_weight_rowwise_fp8,
    rowwise_fp8_lm_head_logits,
)
from sglang.test.ci.ci_register import register_cuda_ci
from torch import nn

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")


def _is_exact_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


pytestmark = pytest.mark.skipif(
    not _is_exact_sm120(), reason="online FP8 kernels require exactly SM120"
)

HIDDEN_SIZE = 2560
VOCAB_ROWS = 1024
HC_COUNT = 4
HC_LOWRANK = 320


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


@pytest.mark.parametrize("rows", [1, 4, 16, 33])
def test_rowwise_lm_head_matches_dequantized_bf16_reference(
    rowwise_lm_head_weight: torch.Tensor, rows: int
) -> None:
    hidden = _randn((rows, HIDDEN_SIZE), seed=200 + rows, scale=0.25)

    actual = rowwise_fp8_lm_head_logits(hidden, rowwise_lm_head_weight)
    expected = _rowwise_reference(hidden, rowwise_lm_head_weight)

    _assert_normalized_error(actual, expected, max_nrmse=0.025, min_cosine=0.999)


@pytest.mark.parametrize("rows", [1, 4, 16, 33])
def test_rowwise_lm_head_cuda_graph_replays_mutated_input(
    rowwise_lm_head_weight: torch.Tensor, rows: int
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
