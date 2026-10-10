"""Real-SM120 coverage for the donor-format dense path on the Qwen4-Exp candidate.

Runs the actual production pieces -- ``Qwen4ExpLayerExtensionMixin`` selection,
``Fp8LinearMethod`` postprocess (real Triton ``per_token_group_quant_fp8``), the
donor ``w8a16_gemv`` / ``w8a16_gemv_norm_gated`` Triton kernels and the
``apply_fp8_linear`` dispatch -- on the dense projection shapes of the model.
The two execution paths are judged by the oracle each one actually consumes:

  * the W8A16 fast path takes the BF16 activation unchanged, so it is compared
    against a BF16-A / dequantized-resident-W reference (fp32 GEMM over the
    same FP8 bytes and FP32 channel scales, never a self-reference of the
    kernel under test), and its execution is proven by wrapping the actual
    ``w8a16_gemv`` entry point;
  * every generic A8W8 fallback (M17/24/33 with the gate on, M4/16 with it
    off) first per-token-quantizes the activation, so comparing it to a BF16-A
    oracle would fold ~2.7% activation-quantization loss into a GEMM check.
    Instead the fallback's real GEMM is verified in FP64 from the *observed*
    quantized A/scales (unchanged platform encoder ``sglang_per_token_quant_fp8``
    observed through a delegating spy -- it is never called from the oracle,
    and ``apply_fp8_linear``/the GEMM under test are not either) plus the
    resident W bytes/scales; the encoder itself is a reused platform operation
    and only the matrix math is independently verified. Activation-
    quantization loss vs the BF16-A reference is reported under a separate
    loose cap: it is not a model-quality result;
  * the fused gated-norm output is compared against the standalone
    ``RMSNormGated`` + the plain GEMV on the same weights/scales, per the
    donor's within-1-ulp-of-bf16 contract;
  * CUDA-graph capture/replay with changed inputs exercises the prealloc
    installed by ``process_weights_after_loading``.

Performance is not claimed here; only that the intended path executes.
"""

from __future__ import annotations

import math
import os

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.gemm.sm120_online_fp8 import configure_online_fp8
from sglang.srt.layers.quantization import fp8 as fp8_module
from sglang.srt.layers.quantization import fp8_utils as fp8_utils_module
from sglang.srt.layers.quantization import w8a16_gemv as w8a16_gemv_module
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-small")

HIDDEN = 2560
# Production GDN output projection: value_dim = num_v_heads * head_v_dim.
NUM_V_HEADS, HEAD_V_DIM = 48, 128
GDN_K = NUM_V_HEADS * HEAD_V_DIM  # 6144
GEMV_ENV = "SGLANG_FP8_W8A16_GEMV"
NORM_ENV = "SGLANG_NORM_INTO_GEMV"
DONOR_MAX_M = 16
GEMV_ROWS = [1, 4, 12, 16]
FALLBACK_ROWS = [17, 24, 33]  # > donor limit, incl. the C6 target verify width


def _is_exact_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


pytestmark = pytest.mark.skipif(
    not _is_exact_sm120(), reason="donor-format dense path requires exactly SM120"
)


def _randn(shape, *, seed: int, scale: float) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        * scale
    )


class _CudaProjection(nn.Module):
    def __init__(self, out_features: int, in_features: int, *, seed: int):
        super().__init__()
        self.weight = nn.Parameter(
            _randn(
                (out_features, in_features),
                seed=seed,
                scale=1.0 / math.sqrt(in_features),
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
    from sglang.srt.models.qwen4_exp import Qwen4ExpLayerExtensionMixin

    saved = os.environ.get(GEMV_ENV)
    os.environ[GEMV_ENV] = "1"  # so process_weights preallocates the scratch
    configure_online_fp8(True, cuda_available=True, capability=(12, 0))
    try:
        root = nn.Module()
        root.linear_attn = nn.Module()
        root.linear_attn.out_proj = _CudaProjection(HIDDEN, GDN_K, seed=11)
        root.self_attn = nn.Module()
        root.self_attn.o_proj = _CudaProjection(HIDDEN, HIDDEN, seed=12)
        Qwen4ExpLayerExtensionMixin._maybe_convert_linears_to_mxfp8(root)
        converted = root._online_mxfp8_linears
        assert len(converted) == 2, converted  # positive selection, both blocks
        for name in converted:
            module = root.get_submodule(name)
            module.quant_method.process_weights_after_loading(module)
        yield root
    finally:
        configure_online_fp8(False, cuda_available=False, capability=None)
        if saved is None:
            os.environ.pop(GEMV_ENV, None)
        else:
            os.environ[GEMV_ENV] = saved


def _assert_normalized_error(actual, expected, *, max_nrmse, min_cosine):
    assert actual.shape == expected.shape
    a, e = actual.float(), expected.float()
    nrmse = (a - e).square().mean().sqrt().item() / max(
        e.square().mean().sqrt().item(), 1e-8
    )
    cosine = F.cosine_similarity(a.flatten(), e.flatten(), dim=0).item()
    assert nrmse <= max_nrmse, f"NRMSE {nrmse:.6f} exceeds {max_nrmse:.6f}"
    assert cosine >= min_cosine, f"cosine {cosine:.6f} is below {min_cosine:.6f}"


def _reference_logits(x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    """BF16-A / dequantized-W fp32 GEMM over the SAME resident FP8 bytes.

    The oracle for the W8A16 path (which consumes the BF16 activation as it
    is); for the A8W8 fallback it only bounds the activation-quantization
    loss, which is a property of activation FP8, not of the GEMM.
    """
    weight = layer.weight.t().float()  # [N, K]
    scale = layer.weight_scale.float().reshape(-1)
    assert weight.shape[0] == scale.numel()
    return (x.float() @ (weight * scale[:, None]).t()).to(torch.bfloat16)


def _uses_w8a16_fastpath(enabled: bool, rows: int) -> bool:
    """Routing truth table of ``Fp8LinearMethod.apply`` for this candidate."""
    return bool(enabled) and 1 <= rows <= DONOR_MAX_M


def _fp64_gemm_reference(
    q_a: torch.Tensor, x_scale: torch.Tensor, layer: nn.Module
) -> torch.Tensor:
    """FP64 matrix math over the observed quantized A (+ its scale) and resident W.

    The activation quantizer is the unchanged platform operation observed via
    the delegating spy (a reused platform operation, not re-implemented or
    cloned here); this independently recomputes only the GEMM:
    (q_a * x_scale) @ (w * w_scale)^T in FP64, then the single final BF16
    round.
    """
    a = q_a.double() * x_scale.double()
    w = layer.weight.t().double() * layer.weight_scale.double().reshape(-1)[:, None]
    return (a @ w.t()).to(torch.bfloat16)


@pytest.fixture
def gemv_env(monkeypatch, request):
    enabled, norm = request.param
    if enabled:
        monkeypatch.setenv(GEMV_ENV, "1")
    else:
        monkeypatch.delenv(GEMV_ENV, raising=False)
    if norm:
        monkeypatch.setenv(NORM_ENV, "1")
    else:
        monkeypatch.delenv(NORM_ENV, raising=False)
    return enabled, norm


@pytest.mark.parametrize(
    "rows, gemv_env",
    [(r, (True, False)) for r in GEMV_ROWS + FALLBACK_ROWS]
    + [(4, (False, False)), (16, (False, False))],
    # route the gemv_env parameter through the fixture of that name: passed
    # directly, pytest binds the tuple itself and the env monkeypatching
    # never runs, so the (False, ...) cases would inherit the module fixture's
    # GEMV=1 and the routing expectations below would be wrong.
    indirect=["gemv_env"],
)
def test_dense_projection_numerics_and_path_selection(
    dense_layers, rows, gemv_env, monkeypatch
):
    enabled, _ = gemv_env
    layer = dense_layers.linear_attn.out_proj
    x = _randn((rows, GDN_K), seed=200 + rows, scale=0.25)
    bf16_a_reference = _reference_logits(x, layer)

    gemv_calls, fallback_calls, encoder_calls = [], [], []
    real_gemv = w8a16_gemv_module.w8a16_gemv
    real_apply_fp8 = fp8_module.apply_fp8_linear
    real_encoder = fp8_utils_module.sglang_per_token_quant_fp8

    def spy_gemv(*args, **kwargs):
        gemv_calls.append(args)
        return real_gemv(*args, **kwargs)

    def spy_fallback(*args, **kwargs):
        fallback_calls.append(kwargs)
        return real_apply_fp8(*args, **kwargs)

    def spy_encoder(input_2d):
        qinput, x_scale = real_encoder(input_2d)
        # Observed, not recomputed: the oracle below consumes exactly the
        # bytes the unmodified platform encoder handed the GEMM under test.
        encoder_calls.append((qinput, x_scale))
        return qinput, x_scale

    monkeypatch.setattr(w8a16_gemv_module, "w8a16_gemv", spy_gemv)
    monkeypatch.setattr(fp8_module, "apply_fp8_linear", spy_fallback)
    monkeypatch.setattr(fp8_utils_module, "sglang_per_token_quant_fp8", spy_encoder)
    actual = layer.quant_method.apply(layer, x)

    fast = _uses_w8a16_fastpath(enabled, rows)
    # The fast path must actually execute (kernel wrapped, not assumed).
    assert len(gemv_calls) == (1 if fast else 0)
    assert len(fallback_calls) == (0 if fast else 1)

    if fast:
        # W8A16 consumes the BF16 activation unchanged: the BF16-A /
        # dequantized-W oracle is the right reference, and it passed at .025
        # on hardware -- no activation-quantization loss is involved here.
        _assert_normalized_error(
            actual, bf16_a_reference, max_nrmse=0.025, min_cosine=0.999
        )
        assert encoder_calls == []  # the A8W8 encoder never ran
        return

    # Generic A8W8 fallback (incl. gate-off M4/M16): verify GEMM correctness
    # independently of the platform encoder, at a bound 25x tighter than the
    # old BF16-A comparison -- measured on SM120 against this exact oracle:
    # all five fallback cases <=.000077 NRMSE (M24 .000018, M33 .000040,
    # gate-off M4 .000077, M16 .000043). The encoder bytes/scales are the
    # native operation's own, observed by the delegating spy above; no test-
    # side clone of the JIT encoder is trusted or compared against.
    assert len(encoder_calls) == 1
    q_actual, s_actual = encoder_calls[0]
    _assert_normalized_error(
        actual,
        _fp64_gemm_reference(q_actual, s_actual, layer),
        max_nrmse=1e-3,
        min_cosine=0.99999,
    )
    # Separate, reported-only budget: the residual vs the BF16-A reference is
    # activation-FP8 quantization loss (SM120-measured ~.0267 at M17), NOT a
    # GEMM-correctness failure and not a model-quality result. Kept loose on
    # purpose; the GEMM verdict is the FP64 check above.
    _assert_normalized_error(actual, bf16_a_reference, max_nrmse=0.05, min_cosine=0.998)
    if rows > DONOR_MAX_M:
        # larger M uses the resident rowwise FP8 through apply_fp8_linear
        assert fallback_calls[0]["weight"] is layer.weight
        assert fallback_calls[0]["weight"].dtype is torch.float8_e4m3fn


def test_graph_replay_uses_preallocated_scratch_and_tracks_inputs(dense_layers):
    layer = dense_layers.linear_attn.out_proj
    rows = 4
    x = _randn((rows, GDN_K), seed=301, scale=0.25)
    static_x = x.clone()

    # warm up (Triton JIT + plan caches) outside the capture.
    for _ in range(3):
        layer.quant_method.apply(layer, static_x)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = layer.quant_method.apply(layer, static_x)
    graph.replay()
    torch.cuda.synchronize()
    _assert_normalized_error(
        out.clone(), _reference_logits(x, layer), max_nrmse=0.025, min_cosine=0.999
    )

    # changed-input replay: the graph must read the buffer, not the old values
    new_x = _randn((rows, GDN_K), seed=302, scale=0.25)
    assert not torch.equal(new_x, x)
    static_x.copy_(new_x)
    graph.replay()
    torch.cuda.synchronize()
    _assert_normalized_error(
        out.clone(), _reference_logits(new_x, layer), max_nrmse=0.025, min_cosine=0.999
    )


@pytest.fixture
def norm_flag(monkeypatch):
    """Force the fused-norm gate on for this kernel check (call-time read)."""
    monkeypatch.setenv(NORM_ENV, "1")
    monkeypatch.setattr(w8a16_gemv_module, "norm_into_gemv_enabled", lambda: True)


def test_fused_norm_out_proj_matches_standalone_norm_and_projection(
    dense_layers, monkeypatch, norm_flag
):
    monkeypatch.setenv(GEMV_ENV, "1")
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated

    layer = dense_layers.linear_attn.out_proj
    method = layer.quant_method
    tokens = 4
    core = _randn((tokens * NUM_V_HEADS, HEAD_V_DIM), seed=401, scale=0.5)
    z = _randn((tokens * NUM_V_HEADS, HEAD_V_DIM), seed=402, scale=1.0)
    norm = RMSNormGated(
        HEAD_V_DIM,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        device="cuda",
        dtype=torch.bfloat16,
    )
    # nonzero, non-unit gate weights: y = x*rstd*w*silu(z) must move.
    norm.weight.data.copy_(
        torch.linspace(0.5, 1.5, HEAD_V_DIM, dtype=torch.bfloat16, device="cuda")
    )
    assert torch.count_nonzero(z) == z.numel()

    normalized = norm(core, z).reshape(tokens, GDN_K)
    expected = method.apply(  # same bytes/scales through the plain GEMV
        layer, normalized
    )
    actual = method.apply_norm_gated(
        layer,
        core.reshape(tokens, GDN_K),
        z.reshape(tokens, GDN_K),
        norm.weight.data,
        HEAD_V_DIM,
        1e-6,
    )
    assert actual is not None, "fused contract must hold on SM120"
    # The donor's fused path re-associates the fp32 sum of squares; the output
    # is within ~1 bf16 ulp of the two-launch reference -- not looser.
    _assert_normalized_error(actual, expected, max_nrmse=0.0125, min_cosine=0.9999)

    # sigmoid gate kind is preserved (NORM_SIGMOID branch of the real kernel).
    # Reference arithmetic matches the kernels' independent contract: the
    # norm computes x * rsqrt(mean(x^2) + eps) * weight * sigmoid(z) in FP32
    # and rounds to BF16 exactly once -- no intermediate bf16 rounds.
    core_fp32 = core.float()
    rstd = torch.rsqrt(core_fp32.pow(2).mean(-1, keepdim=True) + 1e-6)
    sigmoid_expected = (
        (core_fp32 * rstd * norm.weight.data.float() * torch.sigmoid(z.float()))
        .to(torch.bfloat16)
        .reshape(tokens, GDN_K)
    )
    actual_sig = method.apply_norm_gated(
        layer,
        core.reshape(tokens, GDN_K),
        z.reshape(tokens, GDN_K),
        norm.weight.data,
        HEAD_V_DIM,
        1e-6,
        sigmoid_gate=True,
    )
    assert actual_sig is not None
    ref_sig = method.apply(layer, sigmoid_expected)
    _assert_normalized_error(actual_sig, ref_sig, max_nrmse=0.0125, min_cosine=0.9999)

    # flag off -> decline to None (caller keeps the original norm + out_proj)
    monkeypatch.setattr(w8a16_gemv_module, "norm_into_gemv_enabled", lambda: False)
    assert (
        method.apply_norm_gated(
            layer,
            core.reshape(tokens, GDN_K),
            z.reshape(tokens, GDN_K),
            norm.weight.data,
            HEAD_V_DIM,
            1e-6,
        )
        is None
    )
    monkeypatch.setattr(w8a16_gemv_module, "norm_into_gemv_enabled", lambda: True)
    # larger than the donor budget -> None, and the row still computes via
    # the apply_fp8_linear fallback of apply().
    big = _randn((24 * NUM_V_HEADS, HEAD_V_DIM), seed=403, scale=0.5)
    assert (
        method.apply_norm_gated(
            layer,
            big.reshape(24, GDN_K),
            big.reshape(24, GDN_K),
            norm.weight.data,
            HEAD_V_DIM,
            1e-6,
        )
        is None
    )


def test_fused_norm_out_proj_graph_replay_tracks_norm_and_gate(
    dense_layers, monkeypatch, norm_flag
):
    monkeypatch.setenv(GEMV_ENV, "1")
    layer = dense_layers.linear_attn.out_proj
    method = layer.quant_method
    tokens = 4
    weight = torch.linspace(0.5, 1.5, HEAD_V_DIM, dtype=torch.bfloat16, device="cuda")
    x = _randn((tokens, GDN_K), seed=501, scale=0.5)
    z = _randn((tokens, GDN_K), seed=502, scale=1.0)
    static_x, static_z = x.clone(), z.clone()
    for _ in range(3):
        method.apply_norm_gated(layer, static_x, static_z, weight, HEAD_V_DIM, 1e-6)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = method.apply_norm_gated(
            layer, static_x, static_z, weight, HEAD_V_DIM, 1e-6
        )
    graph.replay()
    torch.cuda.synchronize()
    first = out.clone()

    new_x = _randn((tokens, GDN_K), seed=503, scale=0.5)
    new_z = _randn((tokens, GDN_K), seed=504, scale=1.0)
    static_x.copy_(new_x)
    static_z.copy_(new_z)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(first, out), "replay must reflect the changed inputs"
    # independent fp32 check of the fused result on the new inputs
    from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated

    normed = rms_norm_gated(
        x=new_x.reshape(tokens * NUM_V_HEADS, HEAD_V_DIM),
        weight=weight,
        bias=None,
        z=new_z.reshape(tokens * NUM_V_HEADS, HEAD_V_DIM),
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
        activation="swish",
    ).reshape(tokens, GDN_K)
    expected = _reference_logits(normed, layer)
    _assert_normalized_error(out, expected, max_nrmse=0.0125, min_cosine=0.9999)


if __name__ == "__main__":
    # CI's run_unittest_files launches `python3 <file> -f` (legacy unittest
    # failfast), which pytest would reject; translate it to -x, the same way
    # sglang.test.kernels.utils.multigpu_pytest_main does. Without this block a
    # registered file is merely imported and exits 0 with zero tests run.
    import sys as _sys

    _args = ["-x" if _arg == "-f" else _arg for _arg in _sys.argv[1:]]
    _sys.exit(_sys.modules["pytest"].main([__file__, "-v", *_args]))
