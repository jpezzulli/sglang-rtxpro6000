# SPDX-License-Identifier: Apache-2.0

"""Real-kernel coverage for the accepted SM120 small-batch fast paths.

Everything here runs the ACTUAL Triton kernels on an ACTUAL exact-SM120 device.
The CPU suite next door pins which call is *selected*; this file is the evidence
that a selected call is numerically right, replays inside a CUDA graph, and
leaves its split-K scratch clean.  No CPU result is treated as proof of GPU
math, and the file skips (rather than passes) anywhere that is not SM120, which
is the hardware the donor tiles were measured and accepted for.

References are computed from the SAME resident FP8 bytes the kernel reads
(dequantized on the fly), so a wrong scale -- as opposed to a wrong rounding --
cannot hide inside the tolerance.

The dense projections are judged by the oracle of the path that actually ran.
The W8A16 GEMV takes the BF16 activation unchanged, so a BF16-A / dequantized-W
reference is its right oracle.  The reviewed fallback behind it is
``activation_scheme="dynamic"`` W8A8: ``apply_fp8_linear`` per-token-quantizes
the activation before the channelwise GEMM, and comparing THAT to a BF16-A
reference folds ~2.7% activation-quantization loss into a GEMM check (the donor
snapshot measured .0267 at M17 on SM120).  So the fallback's matrix math is
verified in FP64 from the bytes the unchanged platform encoder actually handed
it -- observed through a delegating spy, never recomputed by the oracle -- and
its residual against the BF16-A reference is reported under a separate loose
cap.  Neither gate is a model-quality result.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    configure_online_fp8,
    dequantize_rowwise_weight,
    install_rowwise_fp8_on_eligible_linears,
    replace_linear_weight_rowwise_fp8,
    rowwise_fp8_lm_head_logits,
    rowwise_scale_of,
    w8a16_gemv_supported,
)
from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8_utils as fp8_utils_module
from sglang.srt.layers.quantization import w8a16_gemv as w8a16_module
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=200, stage="base-b", runner_config="1-gpu-small")

HIDDEN = 2560
# The production GDN output projection: value_dim = num_v_heads * head_v_dim.
NUM_V_HEADS, HEAD_V_DIM = 48, 128
GDN_K = NUM_V_HEADS * HEAD_V_DIM  # 6144
VOCAB_ROWS = 1024
DRAFT_HEAD_ROWS = 32768  # the shape the donor tile table keys
HC_COUNT = 4

GEMV_ENV = "SGLANG_FP8_W8A16_GEMV"
NORM_ENV = "SGLANG_NORM_INTO_GEMV"
MTP_ENV = "SGLANG_MTP_FC_GEMV"
DONOR_MAX_M = 16
# 1 decode, 4/6/8/16 the verified widths inside the donor kernel's row limit;
# 17 just past it, 24 == the C6 target verification width, 33 a prefill chunk.
GEMV_ROWS = [1, 4, 6, 8, 16]
FALLBACK_ROWS = [17, 24, 33]


def _is_exact_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


pytestmark = pytest.mark.skipif(
    not _is_exact_sm120(), reason="these kernels are accepted on exactly SM120"
)


def _select_bundle():
    configure_online_fp8(
        None, cuda_available=True, capability=(12, 0), model_eligible=True
    )


@pytest.fixture(autouse=True)
def bundle_selected(monkeypatch):
    """Resolve the bundle as an eligible Flash-Next launch does, then undo it."""
    for name in (GEMV_ENV, NORM_ENV, MTP_ENV, "SGLANG_FP8_W8A16_GEMV_MAX_M"):
        monkeypatch.delenv(name, raising=False)
    _select_bundle()
    yield
    configure_online_fp8(False, cuda_available=False, capability=None)


@pytest.fixture
def gemv_calls():
    """Count REAL GEMV launches, at the one place every wrapper goes.

    The spy calls the real ``_launch``, so nothing here measures a mock.
    """
    calls: list = []
    real = w8a16_module._launch

    def spy(x, w, s, y, M, N, K, per_channel, cfg, *args, **kwargs):
        calls.append(
            {
                "m": M,
                "n": N,
                "k": K,
                "w_stride": tuple(w.stride()),
                "fused_norm": kwargs.get("norm") is not None,
            }
        )
        return real(x, w, s, y, M, N, K, per_channel, cfg, *args, **kwargs)

    with mock.patch.object(w8a16_module, "_launch", spy):
        yield calls


@pytest.fixture
def a8w8_encoder():
    """Observe the per-token activation encoder the W8A8 fallback calls.

    Delegating, so the operation still runs: the oracle below consumes the exact
    codes and scales the GEMM under test was handed, rather than a test-side
    clone of a platform kernel that could drift together with it.
    """
    observed: list = []
    real = fp8_utils_module.sglang_per_token_quant_fp8

    def spy(input_2d):
        qinput, x_scale = real(input_2d)
        observed.append((qinput, x_scale))
        return qinput, x_scale

    with mock.patch.object(fp8_utils_module, "sglang_per_token_quant_fp8", spy):
        yield observed


def _randn(shape, *, seed: int, scale: float) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        * scale
    )


def _assert_normalized_error(actual, expected, *, max_nrmse: float, min_cosine: float):
    assert actual.shape == expected.shape
    a, e = actual.float(), expected.float()
    nrmse = (a - e).square().mean().sqrt().item() / max(
        e.square().mean().sqrt().item(), 1e-8
    )
    cosine = F.cosine_similarity(a.flatten(), e.flatten(), dim=0).item()
    assert nrmse <= max_nrmse, f"NRMSE {nrmse:.6f} exceeds {max_nrmse:.6f}"
    assert cosine >= min_cosine, f"cosine {cosine:.6f} is below {min_cosine:.6f}"


def _counters(device):
    return w8a16_module._workspace_slot(device, 0)[1]


# --------------------------------------------------------------------------- #
# Output head
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module", params=["small", "table-keyed"])
def head_weight(request) -> torch.Tensor:
    _select_bundle()
    rows = VOCAB_ROWS if request.param == "small" else DRAFT_HEAD_ROWS
    linear = nn.Linear(HIDDEN, rows, bias=False, device="cuda", dtype=torch.bfloat16)
    linear.weight.data.copy_(
        _randn((rows, HIDDEN), seed=101, scale=1 / math.sqrt(HIDDEN))
    )
    replace_linear_weight_rowwise_fp8(linear)
    return linear.weight


def _head_reference(hidden, weight):
    return hidden.bfloat16() @ dequantize_rowwise_weight(weight, torch.bfloat16).t()


@pytest.mark.parametrize("rows", GEMV_ROWS + FALLBACK_ROWS)
@pytest.mark.parametrize("opt_in", [False, True])
def test_head_matches_the_dequantized_reference_within_its_budget(
    head_weight, rows, opt_in, gemv_calls
):
    with envs.SGLANG_FP8_W8A16_GEMV.override(opt_in):
        hidden = _randn((rows, HIDDEN), seed=200 + rows, scale=0.25)
        expect_gemv = opt_in and 1 <= rows <= DONOR_MAX_M
        assert (
            w8a16_gemv_supported(hidden, head_weight, rowwise_scale_of(head_weight))
            is expect_gemv
        )
        actual = rowwise_fp8_lm_head_logits(hidden, head_weight)
        _assert_normalized_error(
            actual,
            _head_reference(hidden, head_weight),
            max_nrmse=0.025,
            min_cosine=0.999,
        )
        assert (len(gemv_calls) == 1) is expect_gemv


@pytest.mark.parametrize("rows", [1, 6, 16, 24])
def test_head_replays_in_a_cuda_graph_with_mutated_input(head_weight, rows):
    static_hidden = _randn((rows, HIDDEN), seed=300 + rows, scale=0.25)
    counters = _counters(static_hidden.device)
    # The scratch a captured launch points at must predate the capture, and be
    # left clean by the fixup CTA, because no memset runs between replays.
    assert int(counters.abs().max().item()) == 0
    for _ in range(2):  # JIT + allocator warm-up, as the graph runners do
        rowwise_fp8_lm_head_logits(static_hidden, head_weight)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = rowwise_fp8_lm_head_logits(static_hidden, head_weight)

    changed = _randn((rows, HIDDEN), seed=400 + rows, scale=0.25)
    static_hidden.copy_(changed)
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    _assert_normalized_error(
        graph_out.clone(),
        _head_reference(changed, head_weight),
        max_nrmse=0.025,
        min_cosine=0.999,
    )
    assert int(counters.abs().max().item()) == 0


def test_head_scratch_is_refused_inside_a_capture():
    """Without the pre-load prealloc, the port must not allocate in a graph pool."""
    with (
        mock.patch.dict(w8a16_module._WS, {}, clear=True),
        mock.patch.object(torch.cuda, "is_current_stream_capturing", lambda: True),
    ):
        with pytest.raises(RuntimeError, match="before CUDA graph capture"):
            w8a16_module._workspace(torch.device("cuda:0"))


# --------------------------------------------------------------------------- #
# Dense projections, through the real converter and postprocess
# --------------------------------------------------------------------------- #


class _Projection(nn.Module):
    def __init__(self, out_features: int, in_features: int, *, seed: int):
        super().__init__()
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        self.weight = nn.Parameter(
            _randn(
                (out_features, in_features),
                seed=seed,
                scale=1 / math.sqrt(in_features),
            ),
            requires_grad=False,
        )
        self.quant_method = UnquantizedLinearMethod()
        self.output_size_per_partition = out_features
        self.input_size_per_partition = in_features
        self.logical_widths = [out_features]


@pytest.fixture(scope="module")
def dense_layers():
    """Target-shaped dense projections through the real converter + postprocess."""
    _select_bundle()
    root = nn.Module()
    root.linear_attn = nn.Module()
    root.linear_attn.out_proj = _Projection(HIDDEN, GDN_K, seed=11)
    root.self_attn = nn.Module()
    root.self_attn.o_proj = _Projection(HIDDEN, HIDDEN, seed=12)
    converted = install_rowwise_fp8_on_eligible_linears(
        root, enabled=True, excluded_module_type=()
    )
    assert sorted(converted) == ["linear_attn.out_proj", "self_attn.o_proj"], converted
    for name in converted:
        module = root.get_submodule(name)
        module.quant_method.process_weights_after_loading(module)
        # The one representation the conversion promises.
        assert module.weight.dtype is torch.float8_e4m3fn
        assert module.weight_scale.dtype is torch.float32
        assert module.weight_scale.numel() == module.weight.shape[1]
    return root


def _dense_reference(layer, x):
    """BF16-A / dequantized-W oracle: fp32 GEMM over the SAME resident FP8 codes
    and FP32 channel scales, rounded to BF16 once at the end.

    The right reference for the W8A16 path, which consumes the activation as it
    is; for the A8W8 fallback it only bounds activation-quantization loss.
    """
    weight = layer.weight.t().float()  # [N, K]
    scale = layer.weight_scale.float().reshape(-1)
    assert weight.shape[0] == scale.numel()
    return (x.float() @ (weight * scale[:, None]).t()).to(torch.bfloat16)


def _fp64_gemm_reference(quantized_a, a_scale, layer):
    """FP64 matrix math over the OBSERVED quantized A and the resident W bytes.

    The activation encoder is the unchanged platform operation that the
    ``a8w8_encoder`` spy observed, not a test-side clone of it; this recomputes
    only the accumulation -- (q_a * a_scale) @ (q_w * w_scale)^T in FP64, then the
    single final BF16 round -- so a wrong scale, a transposed operand or a dropped
    row scale cannot pass, and no reimplementation can drift with the kernel.
    """
    a = quantized_a.double() * a_scale.double()
    w = layer.weight.t().double() * layer.weight_scale.double().reshape(-1)[:, None]
    return (a @ w.t()).to(torch.bfloat16)


@pytest.mark.parametrize("rows", GEMV_ROWS + FALLBACK_ROWS)
@pytest.mark.parametrize("opt_in", [False, True])
def test_dense_projection_uses_the_gemv_only_within_its_budget(
    dense_layers, rows, opt_in, gemv_calls, a8w8_encoder
):
    with envs.SGLANG_FP8_W8A16_GEMV.override(opt_in):
        layer = dense_layers.self_attn.o_proj
        x = _randn((rows, HIDDEN), seed=500 + rows, scale=0.25)
        bf16_a_reference = _dense_reference(layer, x)
        actual = layer.quant_method.apply(layer, x)
        expect_gemv = opt_in and 1 <= rows <= DONOR_MAX_M
        assert len(gemv_calls) == (1 if expect_gemv else 0), (rows, opt_in, gemv_calls)
        # Both paths consume the resident FP8 bytes, never a BF16 copy.
        assert layer.weight.dtype is torch.float8_e4m3fn
        if expect_gemv:
            # W8A16: the BF16 activation reaches the kernel unchanged, so the
            # BF16-A oracle is the correct reference and nothing was quantized.
            _assert_normalized_error(
                actual, bf16_a_reference, max_nrmse=0.025, min_cosine=0.999
            )
            assert a8w8_encoder == []
            # [K, N] storage handed over as an [N, K] operand with K contiguous,
            # the one layout the donor tile table was measured on.
            assert (gemv_calls[0]["n"], gemv_calls[0]["k"]) == (HIDDEN, HIDDEN)
            assert gemv_calls[0]["w_stride"][1] == 1
            return

        # The reviewed fallback is dynamic W8A8: the online conversion leaves
        # layer.input_scale as None, so apply_fp8_linear runs the platform
        # per-token encoder once and hands its output to the channelwise GEMM.
        # Judge the GEMM from the bytes that actually went in -- 25x tighter than
        # any BF16-A bound can be, because the residual there belongs to
        # activation FP8 and not to the matrix multiply.
        assert len(a8w8_encoder) == 1, (
            "the rowwise-FP8 conversion declares activation_scheme='dynamic', so "
            "the fallback must per-token-quantize the activation exactly once"
        )
        qinput, x_scale = a8w8_encoder[0]
        assert qinput.dtype is torch.float8_e4m3fn
        assert tuple(qinput.shape) == (rows, HIDDEN)
        assert tuple(x_scale.shape) == (rows, 1), "one activation scale per row"
        assert x_scale.dtype is torch.float32
        _assert_normalized_error(
            actual,
            _fp64_gemm_reference(qinput, x_scale, layer),
            max_nrmse=1e-3,
            min_cosine=0.99999,
        )
        # Reported-only budget, loose on purpose: activation-FP8 loss against the
        # BF16-A reference, not a GEMM verdict and not a model-quality result.
        _assert_normalized_error(
            actual, bf16_a_reference, max_nrmse=0.05, min_cosine=0.998
        )


def test_bias_and_prequantized_activations_do_not_reach_the_gemv(
    dense_layers, gemv_calls
):
    layer = dense_layers.self_attn.o_proj
    x = _randn((4, HIDDEN), seed=601, scale=0.25)
    plain = layer.quant_method.apply(layer, x)
    biased = layer.quant_method.apply(
        layer, x, bias=_randn((HIDDEN,), seed=600, scale=0.01)
    )
    assert len(gemv_calls) == 2, (
        "a bias rides the GEMV result, it is not a reason to skip it"
    )
    assert biased.shape == plain.shape == (4, HIDDEN)

    quantized = torch.zeros(4, HIDDEN, dtype=torch.float8_e4m3fn, device="cuda")
    out = layer.quant_method.apply(layer, (quantized, torch.ones(4, 1, device="cuda")))
    assert out.shape == (4, HIDDEN)
    assert len(gemv_calls) == 2, (
        "a pre-quantized activation has no place in a W8A16 GEMV"
    )


# --------------------------------------------------------------------------- #
# Gated RMSNorm folded into the GEMV's A-load
# --------------------------------------------------------------------------- #


def _model_shaped_norm(activation: str) -> nn.Module:
    """The GDN's own norm: one RMS group per head, applied to a [rows, head_v_dim] view."""
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated

    norm = RMSNormGated(
        HEAD_V_DIM,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        dtype=torch.bfloat16,
        activation=activation,
    ).cuda()
    norm.weight.data.copy_(_randn((HEAD_V_DIM,), seed=700, scale=0.02) + 1.0)
    return norm


@pytest.mark.parametrize("rows", [4, 16])
@pytest.mark.parametrize("activation", ["swish", "sigmoid"])
def test_fused_norm_matches_norm_then_gemv(dense_layers, rows, activation, gemv_calls):
    layer = dense_layers.linear_attn.out_proj
    norm_weight = _randn((HEAD_V_DIM,), seed=700, scale=0.02) + 1.0
    x_head = _randn((rows * NUM_V_HEADS, HEAD_V_DIM), seed=800 + rows, scale=0.5)
    z_head = _randn(x_head.shape, seed=850 + rows, scale=1.0)

    fused = layer.quant_method.apply_norm_gated(
        layer,
        x_head.reshape(rows, GDN_K),
        z_head.reshape(rows, GDN_K),
        norm_weight,
        HEAD_V_DIM,
        1e-6,
        activation == "sigmoid",
    )
    assert fused is not None, "the exact GDN out_proj contract must take the fusion"
    assert len(gemv_calls) == 1 and gemv_calls[0]["fused_norm"]

    norm = _model_shaped_norm(activation)
    norm.weight.data.copy_(norm_weight)
    expected = layer.quant_method.apply(
        layer, norm(x_head, z_head).reshape(rows, GDN_K)
    )
    assert len(gemv_calls) == 2 and not gemv_calls[1]["fused_norm"]
    # The sum of squares is re-associated into the GEMV tile's thread layout, so
    # the pre-round value may differ in the last fp32 bit; after the bf16 round
    # the two paths agree far inside a bf16 ulp.
    _assert_normalized_error(fused, expected, max_nrmse=1e-3, min_cosine=0.99999)


@pytest.mark.parametrize("rows", [4, 16])
def test_fused_norm_caller_declines_every_layout_it_cannot_prove(dense_layers, rows):
    from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

    def fused(out_proj, norm, shape=None, x=None, z=None):
        fake = SimpleNamespace(head_v_dim=HEAD_V_DIM, out_proj=out_proj, norm=norm)
        return Qwen3_5GatedDeltaNet._norm_out_proj_fused(
            fake,
            x
            if x is not None
            else torch.zeros(
                rows * NUM_V_HEADS, HEAD_V_DIM, dtype=torch.bfloat16, device="cuda"
            ),
            z
            if z is not None
            else torch.zeros(
                rows * NUM_V_HEADS, HEAD_V_DIM, dtype=torch.bfloat16, device="cuda"
            ),
            shape if shape is not None else (rows, NUM_V_HEADS, HEAD_V_DIM),
        )

    layer = dense_layers.linear_attn.out_proj
    norm = _model_shaped_norm("swish")
    assert fused(layer, norm) is not None
    # A row-parallel shard: the fused kernel would compute a partial dot product.
    assert (
        fused(
            SimpleNamespace(
                quant_method=layer.quant_method,
                input_size=2 * GDN_K,
                input_size_per_partition=GDN_K,
                bias=None,
                use_decode_attn_tp=False,
            ),
            norm,
        )
        is None
    )
    for flag, value in (
        ("bias", torch.zeros(1, device="cuda")),
        ("use_decode_attn_tp", True),
    ):
        variant = SimpleNamespace(
            quant_method=layer.quant_method,
            input_size=GDN_K,
            input_size_per_partition=GDN_K,
            bias=None,
            use_decode_attn_tp=False,
        )
        setattr(variant, flag, value)
        assert fused(variant, norm) is None
    for attribute, value in (
        ("group_size", HEAD_V_DIM),
        ("norm_before_gate", False),
        ("activation", "gelu"),
    ):
        variant = _model_shaped_norm("swish")
        setattr(variant, attribute, value)
        assert fused(layer, variant) is None
    norm_bias = _model_shaped_norm("swish")
    # A registered parameter, not a plain attribute: the norm declares bias=None.
    norm_bias.register_parameter(
        "bias",
        nn.Parameter(torch.zeros(HEAD_V_DIM, dtype=torch.bfloat16, device="cuda")),
    )
    assert fused(layer, norm_bias) is None
    # A z that is not the norm's own view (DP-Attn padding, a strided prefill
    # layout) and a row count that is not tokens * num_v_heads.
    assert fused(layer, norm, shape=(rows, NUM_V_HEADS, HEAD_V_DIM, 1)) is None
    assert fused(layer, norm, shape=(rows * 2, NUM_V_HEADS, HEAD_V_DIM)) is None
    padded = torch.zeros(
        rows * NUM_V_HEADS + 8, HEAD_V_DIM, dtype=torch.bfloat16, device="cuda"
    )
    assert fused(layer, norm, x=padded, z=padded) is None


def test_fused_norm_honors_its_own_switch_and_the_row_budget(dense_layers):
    from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

    layer = dense_layers.linear_attn.out_proj
    fake = SimpleNamespace(
        head_v_dim=HEAD_V_DIM, out_proj=layer, norm=_model_shaped_norm("swish")
    )
    x = torch.zeros(4 * NUM_V_HEADS, HEAD_V_DIM, dtype=torch.bfloat16, device="cuda")
    shape = (4, NUM_V_HEADS, HEAD_V_DIM)
    assert Qwen3_5GatedDeltaNet._norm_out_proj_fused(fake, x, x, shape) is not None
    # The switch is re-read per call, so no import order can pin it away from the
    # GEMV's own guard, and the fusion honors the same row budget the GEMV does.
    with envs.SGLANG_NORM_INTO_GEMV.override(False):
        assert Qwen3_5GatedDeltaNet._norm_out_proj_fused(fake, x, x, shape) is None
    assert Qwen3_5GatedDeltaNet._norm_out_proj_fused(fake, x, x, shape) is not None
    wide = torch.zeros(
        24 * NUM_V_HEADS, HEAD_V_DIM, dtype=torch.bfloat16, device="cuda"
    )
    assert (
        Qwen3_5GatedDeltaNet._norm_out_proj_fused(
            fake, wide, wide, (24, NUM_V_HEADS, HEAD_V_DIM)
        )
        is None
    ), "24 rows (the C6 verification width) is past the kernel's limit"


# --------------------------------------------------------------------------- #
# Draft MTP entry fusion
# --------------------------------------------------------------------------- #

MTP_TOKEN_WIDTHS = [1, 4, 6, 16]


@pytest.fixture(scope="module")
def entry_fusion():
    """The entry fusion built through the production constructor API."""
    from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP

    _select_bundle()
    fusion = SimpleNamespace(hc_count=HC_COUNT, hidden_size=HIDDEN)
    config = SimpleNamespace(hidden_size=HIDDEN, rms_norm_eps=1e-6, hc_count=HC_COUNT)
    Qwen4ExpForCausalLMMTP._init_pre_fc_norms(fusion, config)
    Qwen4ExpForCausalLMMTP._init_linear_projections(fusion, config)
    for module in (
        fusion.pre_fc_norm_embedding,
        fusion.pre_fc_norm_hidden,
        fusion.fc_embedding,
        fusion.fc_hidden,
    ):
        module.to(device="cuda", dtype=torch.bfloat16)
    for name, width, seed in (
        ("pre_fc_norm_embedding", HIDDEN, 901),
        ("pre_fc_norm_hidden", HC_COUNT * HIDDEN, 902),
    ):
        norm = getattr(fusion, name)
        norm.weight.weight_loader(norm.weight, _randn((width,), seed=seed, scale=0.01))
    for name, seed in (("fc_embedding", 101), ("fc_hidden", 102)):
        getattr(fusion, name).weight.data.copy_(
            _randn((HIDDEN, HIDDEN), seed=seed, scale=1 / math.sqrt(HIDDEN))
        )
    return fusion


@pytest.mark.parametrize("tokens", MTP_TOKEN_WIDTHS)
@pytest.mark.parametrize("opt_in", [False, True])
def test_mtp_entry_fusion_matches_the_two_linear_path(entry_fusion, tokens, opt_in):
    from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP

    calls: list = []
    real = w8a16_module.bf16_gemv

    def spy(x, w, *args, **kwargs):
        calls.append(tuple(x.shape))
        return real(x, w, *args, **kwargs)

    embeds = _randn((tokens, HIDDEN), seed=950 + tokens, scale=0.25)
    hidden = _randn((tokens, HC_COUNT * HIDDEN), seed=960 + tokens, scale=0.25)
    with (
        envs.SGLANG_MTP_FC_GEMV.override(opt_in),
        mock.patch.object(w8a16_module, "bf16_gemv", spy),
    ):
        actual = Qwen4ExpForCausalLMMTP._fuse_residual_linear_shared(
            entry_fusion, embeds, hidden
        )
    # The hidden-side rows are tokens * hc, and BOTH entry GEMMs must fit.
    expect_gemv = opt_in and tokens * HC_COUNT <= DONOR_MAX_M
    assert calls == (
        [(tokens, HIDDEN), (tokens * HC_COUNT, HIDDEN)] if expect_gemv else []
    ), (
        tokens,
        opt_in,
        calls,
    )
    with envs.SGLANG_MTP_FC_GEMV.override(False):
        expected = Qwen4ExpForCausalLMMTP._fuse_residual_linear_shared(
            entry_fusion, embeds, hidden
        )
    assert actual.shape == expected.shape == (tokens, HC_COUNT * HIDDEN)
    _assert_normalized_error(actual, expected, max_nrmse=0.025, min_cosine=0.999)


def test_mtp_entry_fusion_replays_in_a_cuda_graph(entry_fusion):
    from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP

    embeds = _randn((4, HIDDEN), seed=970, scale=0.25)
    hidden = _randn((4, HC_COUNT * HIDDEN), seed=971, scale=0.25)
    for _ in range(2):
        Qwen4ExpForCausalLMMTP._fuse_residual_linear_shared(
            entry_fusion, embeds, hidden
        )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = Qwen4ExpForCausalLMMTP._fuse_residual_linear_shared(
            entry_fusion, embeds, hidden
        )
    embeds.copy_(_randn((4, HIDDEN), seed=972, scale=0.25))
    hidden.copy_(_randn((4, HC_COUNT * HIDDEN), seed=973, scale=0.25))
    graph.replay()
    torch.cuda.synchronize()
    with envs.SGLANG_MTP_FC_GEMV.override(False):
        expected = Qwen4ExpForCausalLMMTP._fuse_residual_linear_shared(
            entry_fusion, embeds, hidden
        )
    _assert_normalized_error(
        graph_out.clone(), expected, max_nrmse=0.025, min_cosine=0.999
    )
