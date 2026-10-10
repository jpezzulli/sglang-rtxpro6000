"""CPU coverage for the donor-format dense path on the Qwen4-Exp candidate.

Pins the three functional contracts without a GPU:
  * selection/format: ``_maybe_convert_linears_to_mxfp8`` installs a
    non-``block_quant``/non-MXFP8 ``Fp8LinearMethod`` subclass on eligible
    BF16 transformer projections only (gate/router child, PLE, FusedMoE and
    already-quantized modules keep their original method), and the postprocess
    yields the donor's per-output-channel E4M3+FP32 resident state;
  * dispatch: the W8A16 GEMV fast path is opt-in, 2-D BF16 and M <= min(env,
    16); everything else (incl. the C6 target's 24 rows) falls back to the
    real-FP8 ``apply_fp8_linear`` dispatch on the same resident bytes;
  * the GDN gated-norm seam: ``apply_norm_gated`` and
    ``Qwen3_5GatedDeltaNet._norm_out_proj_fused`` decline to None on every
    contract break (TP2, bias, flag off, CPU tensors, quant methods without
    the scoped method -- e.g. the generic 27B FP8 ones).

``einops`` is stubbed only when the host lacks it (room101), mirroring how
``test_qwen4_exp_mtp_fc_gemv.py`` keeps registration executable without the
full serving wheel. Triton kernels are never launched here; the Triton entry
points are replaced by exact contract probes.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

try:
    import einops  # noqa: F401
except ModuleNotFoundError:
    _stub = types.ModuleType("einops")
    _stub.rearrange = lambda *a, **k: None
    _stub.repeat = lambda *a, **k: None
    sys.modules["einops"] = _stub

from sglang.kernels.ops.gemm import sm120_online_fp8 as online_mod
from sglang.kernels.ops.gemm.sm120_online_fp8 import configure_online_fp8
from sglang.srt.layers.quantization import fp8 as fp8_module
from sglang.srt.layers.quantization import w8a16_gemv as w8a16_gemv_module
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models import qwen3_5 as qwen3_5_module
from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpAttentionDecoderLayer,
    Qwen4ExpLayerExtensionMixin,
    Qwen4ExpLinearDecoderLayer,
    Qwen4ExpPLELayer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

HIDDEN = 2560
GDN_HEAD_DIM = 128
GEMV_ENV = "SGLANG_FP8_W8A16_GEMV"
GEMV_MAX_M_ENV = "SGLANG_FP8_W8A16_GEMV_MAX_M"
DONOR_MAX_M = 16


@pytest.fixture
def sm120_online():
    """The exact-SM120 online-FP8 process state the converter requires."""
    configure_online_fp8(True, cuda_available=True, capability=(12, 0))
    try:
        yield
    finally:
        configure_online_fp8(False, cuda_available=False, capability=None)


class _PlainLinear(nn.Module):
    """Minimal stand-in for a checkpoint-BF16, otherwise-unquantized linear."""

    def __init__(self, out_features: int, in_features: int):
        super().__init__()
        weight = (
            torch.arange(out_features * in_features, dtype=torch.float32)
            .remainder(9)
            .sub(4)
            .reshape(out_features, in_features)
            .to(torch.bfloat16)
            * 0.05
        )
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.quant_method = UnquantizedLinearMethod()
        self.output_size_per_partition = out_features
        self.input_size_per_partition = in_features
        self.logical_widths = [out_features]


def _eligible_tree():
    root = nn.Module()
    root.linear_attn = nn.Module()
    root.linear_attn.out_proj = _PlainLinear(HIDDEN, 6144)
    root.self_attn = nn.Module()
    root.self_attn.qkv_proj = _PlainLinear(13312, HIDDEN)
    root.self_attn.o_proj = _PlainLinear(HIDDEN, HIDDEN)
    root.mlp = nn.Module()
    # child named `gate` (the router) and already-quantized siblings are the
    # exclusions the accepted MXFP8 candidate also honored; they must survive.
    root.mlp.gate = _PlainLinear(512, HIDDEN)
    root.mlp.shared_expert_gate = _PlainLinear(1, HIDDEN)  # rows < 128: too small
    return root


def _convert(root):
    Qwen4ExpLayerExtensionMixin._maybe_convert_linears_to_mxfp8(root)
    return getattr(root, "_online_mxfp8_linears", [])


def test_target_and_draft_layers_share_the_selection_mixin():
    """The draft MTP model embeds Qwen4ExpModel layers; one selection point."""
    assert issubclass(Qwen4ExpAttentionDecoderLayer, Qwen4ExpLayerExtensionMixin)
    assert issubclass(Qwen4ExpLinearDecoderLayer, Qwen4ExpLayerExtensionMixin)


def test_selection_requires_online_fp8_and_keeps_exclusions(sm120_online):
    root = _eligible_tree()
    quantized = _PlainLinear(HIDDEN, HIDDEN)
    quantized.quant_method = Fp8LinearMethod(
        Fp8Config(is_checkpoint_fp8_serialized=True)
    )
    original_quantized_method = quantized.quant_method
    root.self_attn.v_proj = quantized

    # Online FP8 off: the untouched BF16 stays BF16.
    configure_online_fp8(False, cuda_available=True, capability=(12, 0))
    assert _convert(root) == []
    assert isinstance(root.self_attn.o_proj.quant_method, UnquantizedLinearMethod)
    configure_online_fp8(True, cuda_available=True, capability=(12, 0))

    # The wiring passes the accepted exclusions through to the converter.
    captured = {}
    real_convert = online_mod.convert_eligible_linears_to_mxfp8

    def spy(root_module, **kwargs):
        captured.update(kwargs)
        return real_convert(root_module, **kwargs)

    monkey_original = online_mod.convert_eligible_linears_to_mxfp8
    online_mod.convert_eligible_linears_to_mxfp8 = spy
    try:
        converted = _convert(root)
    finally:
        online_mod.convert_eligible_linears_to_mxfp8 = monkey_original

    assert converted == [
        "linear_attn.out_proj",
        "self_attn.qkv_proj",
        "self_attn.o_proj",
    ]
    assert captured["enabled"] is True
    assert captured["unquantized_method_type"] is UnquantizedLinearMethod
    excluded = captured["excluded_module_type"]
    assert Qwen4ExpPLELayer in excluded  # PLE storage stays as configured
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    assert FusedMoE in excluded
    # gate/router child and already-quantized/other-profile modules untouched.
    assert isinstance(root.mlp.gate.quant_method, UnquantizedLinearMethod)
    assert root.self_attn.v_proj.quant_method is original_quantized_method


def test_installed_method_is_rowwise_fp8_not_mxfp8(sm120_online, monkeypatch):
    root = _eligible_tree()
    _convert(root)
    method = root.self_attn.o_proj.quant_method
    assert isinstance(method, Fp8LinearMethod)
    assert type(method).__name__ == "CheckedOnlineRowwiseFp8LinearMethod"
    assert method.use_mxfp8 is False
    assert method.block_quant is False
    assert method.weight_block_size is None
    assert method.mxfp8_dense_backend is None
    assert method.quant_config.is_checkpoint_fp8_serialized is False
    assert method.w8a8_mxfp8_linear is None
    # target and draft each selected positive counts through this one method
    assert root.linear_attn.out_proj.quant_method is method


def _fake_rowwise_quant(weight, group_size):
    """CPU stand-in for the Triton per-token-group quantizer (row-group only).

    Returns what the real kernel contractually returns for group_size == K:
    e4m3 values [N, K] and one fp32 scale per row in a [N, 1] tensor.
    """
    assert group_size == weight.shape[-1]
    w = weight.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 448.0
    q = (w / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale


@pytest.fixture
def loaded_layer(monkeypatch, sm120_online):
    """A post-load layer in the resident rowwise format, quantizer stubbed."""
    monkeypatch.setattr(fp8_module, "per_token_group_quant_fp8", _fake_rowwise_quant)
    root = _eligible_tree()
    _convert(root)
    layer = root.self_attn.o_proj
    layer.quant_method.cutlass_fp8_supported = True  # the SM120 CUDA reality
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def test_postprocess_resident_format_is_rowwise_fp8(loaded_layer):
    weight, scale = loaded_layer.weight, loaded_layer.weight_scale
    assert weight.dtype is torch.float8_e4m3fn
    assert tuple(weight.shape) == (HIDDEN, HIDDEN)  # stored [K, N]
    assert weight.t().is_contiguous()  # [N, K]-contiguous storage
    assert scale.dtype is torch.float32
    assert scale.numel() == weight.shape[1]  # one fp32 scale per output channel
    assert loaded_layer.input_scale is None
    # started from the original BF16 values: dequantize matches within fp8 error
    # (weight is the [K, N] view, so dequant[k, n] ~ orig[n, k]).
    orig = loaded_layer_orig(loaded_layer).float()
    scale = orig.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 448.0
    ref = (orig / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
    dequant = weight.float() * scale.reshape(1, -1)
    torch.testing.assert_close(dequant, ref.t(), rtol=1e-6, atol=1e-6)


def loaded_layer_orig(layer):
    # reconstruct the fixture weight exactly as _PlainLinear built it
    return (
        torch.arange(HIDDEN * HIDDEN, dtype=torch.float32)
        .remainder(9)
        .sub(4)
        .reshape(HIDDEN, HIDDEN)
        .to(torch.bfloat16)
        * 0.05
    )


def test_repeat_postprocess_fails_loud_before_mutating_and_keeps_state(
    loaded_layer, monkeypatch
):
    """Review-1: the quantizer-stub silent requantize must now raise.

    Re-running the postprocess on the resident FP8 (dtype fp8, [K, N] view)
    used to feed the values back through the quantizer: the scales collapsed
    to 1.0 and the layout silently doubled-transposed. It must fail before
    the base postprocess touches the weight, leaving values and scales intact.
    """
    method = loaded_layer.quant_method
    weight_before, scale_before = loaded_layer.weight, loaded_layer.weight_scale
    weight_data = weight_before.data.clone()
    scale_data = scale_before.data.clone()

    with pytest.raises(RuntimeError, match="original 2-D BF16 checkpoint"):
        method.process_weights_after_loading(loaded_layer)

    assert loaded_layer.weight is weight_before
    assert loaded_layer.weight_scale is scale_before
    assert loaded_layer.weight.dtype is torch.float8_e4m3fn
    torch.testing.assert_close(loaded_layer.weight.data, weight_data)
    torch.testing.assert_close(loaded_layer.weight_scale.data, scale_data)
    assert scale_before.float().amax().item() < 1.0  # not the old 1.0 collapse

    # a fresh original-BF16 linear through the SAME method still quantizes
    fresh = _PlainLinear(HIDDEN, HIDDEN)
    fresh.quant_method = method
    fresh_orig = fresh.weight.data.clone()
    method.process_weights_after_loading(fresh)
    assert fresh.weight.dtype is torch.float8_e4m3fn
    dequant = fresh.weight.float() * fresh.weight_scale.float().reshape(1, -1)
    scale = fresh_orig.float().abs().amax(1, keepdim=True).clamp_min(1e-8) / 448.0
    ref = (fresh_orig.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(dequant, (ref * scale).t(), rtol=1e-6, atol=1e-6)


def test_postprocess_rejects_nonbf16_input_before_any_mutation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fp8_module,
        "per_token_group_quant_fp8",
        lambda *a, **k: calls.append(1) or (a[0], torch.ones(a[0].shape[0], 1)),
    )
    configure_online_fp8(True, cuda_available=True, capability=(12, 0))
    try:
        root = _eligible_tree()
        _convert(root)
        layer = root.self_attn.o_proj
        layer.weight = nn.Parameter(
            torch.zeros(HIDDEN, HIDDEN, dtype=torch.float32), requires_grad=False
        )
        with pytest.raises(RuntimeError, match="original 2-D BF16"):
            layer.quant_method.process_weights_after_loading(layer)
        assert layer.weight.dtype is torch.float32  # unchanged
        assert not hasattr(layer, "weight_scale")  # base postprocess never ran
        assert calls == []
    finally:
        configure_online_fp8(False, cuda_available=False, capability=None)


def test_postprocess_rejects_a_wrong_loaded_format(monkeypatch):
    monkeypatch.setattr(
        fp8_module,
        "per_token_group_quant_fp8",
        lambda weight, group_size: (
            weight.clone(),  # still BF16: an online quantizer that never ran
            torch.ones(weight.shape[0], 1, dtype=torch.float32),
        ),
    )
    configure_online_fp8(True, cuda_available=True, capability=(12, 0))
    try:
        root = _eligible_tree()
        _convert(root)
        layer = root.self_attn.o_proj
        layer.quant_method.cutlass_fp8_supported = True
        with pytest.raises(RuntimeError, match="rowwise-FP8 post-processing"):
            layer.quant_method.process_weights_after_loading(layer)
    finally:
        configure_online_fp8(False, cuda_available=False, capability=None)


# ---------------------------------------------------------------------------
# Fast/fallback dispatch through Fp8LinearMethod.apply
# ---------------------------------------------------------------------------


class _GemvProbe:
    def __init__(self):
        self.calls = []

    def install(self, monkeypatch, name="w8a16_gemv"):
        def probe(x, w, scale, cfg=None, out=None, out2=None, split_n=0):
            self.calls.append((x.shape, w.shape, w.stride(), tuple(scale.shape)))
            return torch.zeros((x.shape[0], w.shape[0]), dtype=torch.bfloat16)

        monkeypatch.setattr(w8a16_gemv_module, name, probe)
        return self


def _gemv_env(monkeypatch, on: bool, max_m=None):
    if on:
        monkeypatch.setenv(GEMV_ENV, "1")
    else:
        monkeypatch.delenv(GEMV_ENV, raising=False)
    if max_m is None:
        monkeypatch.delenv(GEMV_MAX_M_ENV, raising=False)
    else:
        monkeypatch.setenv(GEMV_MAX_M_ENV, str(max_m))


@pytest.mark.parametrize(
    "rows, expect_gemv",
    [
        (0, False),
        (1, True),
        (4, True),
        (16, True),
        (17, False),
        (24, False),
        (33, False),
    ],
)
def test_gemv_gate_respects_the_donor_row_budget(
    loaded_layer, monkeypatch, sm120_online, rows, expect_gemv
):
    _gemv_env(monkeypatch, on=True)
    probe = _GemvProbe().install(monkeypatch)

    def spy(*args, **kwargs):
        return torch.zeros((x_rows, HIDDEN), dtype=torch.bfloat16)

    monkeypatch.setattr(fp8_module, "apply_fp8_linear", spy)
    x_rows = rows
    method = loaded_layer.quant_method
    x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
    assert method._w8a16_gemv_ok(loaded_layer, x) is expect_gemv
    method.apply(loaded_layer, x)
    assert (len(probe.calls) == 1) is expect_gemv


def test_gemv_is_opt_in_and_falls_back_to_apply_fp8_linear(
    loaded_layer, monkeypatch, sm120_online
):
    _gemv_env(monkeypatch, on=False)
    probe = _GemvProbe().install(monkeypatch)
    seen = []

    def spy_apply_fp8_linear(*args, **kwargs):
        seen.append(kwargs)
        return torch.zeros((4, kwargs["weight"].shape[1]), dtype=torch.bfloat16)

    monkeypatch.setattr(fp8_module, "apply_fp8_linear", spy_apply_fp8_linear)
    out = loaded_layer.quant_method.apply(
        loaded_layer, torch.zeros(4, HIDDEN, dtype=torch.bfloat16)
    )
    assert probe.calls == [] and len(seen) == 1
    # the fallback consumes the resident fp8 bytes -- never a BF16 copy
    assert seen[0]["weight"] is loaded_layer.weight
    assert seen[0]["weight"].dtype is torch.float8_e4m3fn
    assert out.shape == (4, HIDDEN)


def test_large_m_c6_uses_real_fp8_not_the_gemv_or_a_dequant_cache(
    loaded_layer, monkeypatch, sm120_online
):
    _gemv_env(monkeypatch, on=True)
    probe = _GemvProbe().install(monkeypatch)
    seen = []

    def spy_apply_fp8_linear(*args, **kwargs):
        seen.append((args, kwargs))
        return torch.zeros((24, kwargs["weight"].shape[1]), dtype=torch.bfloat16)

    monkeypatch.setattr(fp8_module, "apply_fp8_linear", spy_apply_fp8_linear)
    x = torch.zeros(24, HIDDEN, dtype=torch.bfloat16)  # C6 target verification
    out = loaded_layer.quant_method.apply(loaded_layer, x)
    assert probe.calls == []
    assert len(seen) == 1 and seen[0][1]["weight"] is loaded_layer.weight


def test_env_can_only_shrink_the_row_budget(loaded_layer, monkeypatch, sm120_online):
    _gemv_env(monkeypatch, on=True, max_m=4)
    method = loaded_layer.quant_method
    x16 = torch.zeros(16, HIDDEN, dtype=torch.bfloat16)
    x4 = torch.zeros(4, HIDDEN, dtype=torch.bfloat16)
    assert method._w8a16_gemv_ok(loaded_layer, x4) is True
    assert method._w8a16_gemv_ok(loaded_layer, x16) is False
    _gemv_env(monkeypatch, on=True, max_m=64)
    assert method._w8a16_gemv_ok(loaded_layer, x16) is True
    assert (
        method._w8a16_gemv_ok(
            loaded_layer, torch.zeros(17, HIDDEN, dtype=torch.bfloat16)
        )
        is False
    )


def test_gate_declines_nonconforming_activations(
    loaded_layer, monkeypatch, sm120_online
):
    monkeypatch.setenv(GEMV_ENV, "1")
    method = loaded_layer.quant_method
    assert (
        method._w8a16_gemv_ok(loaded_layer, torch.zeros(4, HIDDEN)) is False
    )  # fp32 activation, not the BF16 contract
    assert (
        method._w8a16_gemv_ok(
            loaded_layer, torch.zeros(1, 4, HIDDEN, dtype=torch.bfloat16)
        )
        is False
    )  # 3-D activation: x.dim() == 2 is the donor contract
    monkeypatch.delenv(GEMV_ENV, raising=False)
    assert (
        method._w8a16_gemv_ok(
            loaded_layer, torch.zeros(4, HIDDEN, dtype=torch.bfloat16)
        )
        is False
    )  # opt-in off


@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setenv(GEMV_ENV, "1")


def test_gemv_call_signature_carries_the_resident_bytes(
    loaded_layer, monkeypatch, sm120_online, gate_on
):
    probe = _GemvProbe().install(monkeypatch)
    x = torch.zeros(4, HIDDEN, dtype=torch.bfloat16)
    bias = torch.ones(HIDDEN, dtype=torch.bfloat16)
    out = loaded_layer.quant_method.apply(loaded_layer, x, bias=bias)
    assert probe.calls == [((4, HIDDEN), (HIDDEN, HIDDEN), (HIDDEN, 1), (1, HIDDEN))]
    torch.testing.assert_close(out, bias.expand(4, HIDDEN))
    # bias semantics: no bias means the raw GEMV result
    out = loaded_layer.quant_method.apply(loaded_layer, x)
    assert torch.count_nonzero(out) == 0


def test_tuple_and_3d_activations_skip_the_gemv(
    loaded_layer, monkeypatch, sm120_online
):
    _gemv_env(monkeypatch, on=True)
    probe = _GemvProbe().install(monkeypatch)
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return torch.zeros(2, HIDDEN, dtype=torch.bfloat16)

    monkeypatch.setattr(fp8_module, "apply_fp8_linear", spy)
    qx = torch.zeros(2, HIDDEN, dtype=torch.float8_e4m3fn)
    loaded_layer.quant_method.apply(loaded_layer, (qx, torch.ones(1)))
    assert probe.calls == [] and len(seen) == 1


# ---------------------------------------------------------------------------
# apply_norm_gated (the scoped donor method on the dense GDN out_proj only)
# ---------------------------------------------------------------------------


def _norm_inputs(rows=4):
    x = torch.randn(rows, HIDDEN, dtype=torch.bfloat16)
    z = torch.randn(rows, HIDDEN, dtype=torch.bfloat16)
    w_norm = torch.linspace(0.5, 1.5, 128, dtype=torch.bfloat16)
    return x, z, w_norm


def test_apply_norm_gated_declines_off_contract(
    loaded_layer, sm120_online, monkeypatch
):
    _gemv_env(monkeypatch, on=True)
    method = loaded_layer.quant_method
    x, z, w_norm = _norm_inputs()
    # CPU activation: the real donor support helper requires x.is_cuda.
    assert method.apply_norm_gated(loaded_layer, x, z, w_norm, 128, 1e-6) is None
    # bias is not fused into the norm prologue.
    assert (
        method.apply_norm_gated(
            loaded_layer, x, z, w_norm, 128, 1e-6, bias=torch.zeros(HIDDEN)
        )
        is None
    )
    # env off kills the whole GEMV family.
    _gemv_env(monkeypatch, on=False)
    assert method.apply_norm_gated(loaded_layer, x, z, w_norm, 128, 1e-6) is None


def test_apply_norm_gated_forwards_the_full_contract(
    loaded_layer, sm120_online, monkeypatch
):
    _gemv_env(monkeypatch, on=True)
    monkeypatch.setattr(w8a16_gemv_module, "norm_into_gemv_enabled", lambda: True)
    # the real support helper is CPU-declining (x.is_cuda); its full truth
    # table is donor-AST-identical and covered on-GPU -- forward it here.
    monkeypatch.setattr(
        w8a16_gemv_module,
        "w8a16_gemv_norm_gated_supported",
        lambda *a, **k: True,
    )
    calls = []

    def spy(x, w, scale, z, norm_weight, group_size, eps, sigmoid_gate=False, out=None):
        calls.append(
            (
                tuple(x.shape),
                tuple(w.shape),
                w.dtype,
                w.stride(1),
                tuple(scale.shape),
                scale.dtype,
                tuple(z.shape),
                tuple(norm_weight.shape),
                group_size,
                eps,
                sigmoid_gate,
            )
        )
        return torch.zeros((x.shape[0], w.shape[0]), dtype=torch.bfloat16)

    monkeypatch.setattr(w8a16_gemv_module, "w8a16_gemv_norm_gated", spy)
    x, z, w_norm = _norm_inputs()
    out = loaded_layer.quant_method.apply_norm_gated(
        loaded_layer, x, z, w_norm, 128, 1e-6, sigmoid_gate=True
    )
    assert out is not None
    assert calls == [
        (
            (4, HIDDEN),
            (HIDDEN, HIDDEN),
            torch.float8_e4m3fn,
            1,
            (1, HIDDEN),
            torch.float32,
            (4, HIDDEN),
            (128,),
            128,
            1e-6,
            True,
        )
    ]


def test_generic_fp8_methods_have_no_scoped_norm_method():
    generic = Fp8LinearMethod(Fp8Config(is_checkpoint_fp8_serialized=True))
    assert not hasattr(generic, "apply_norm_gated")
    assert not hasattr(generic, "_w8a16_gemv_ok")


# ---------------------------------------------------------------------------
# Qwen3_5GatedDeltaNet._norm_out_proj_fused caller contract
# ---------------------------------------------------------------------------


def _fake_gdn(quant_method, *, tp_size=1, activation="swish", group_size=None):
    rows_heads = 8
    g = GDN_HEAD_DIM

    class _Norm:
        pass

    norm = _Norm()
    norm.group_size = group_size
    norm.norm_before_gate = True
    norm.bias = None
    norm.weight = torch.ones(g, dtype=torch.bfloat16)
    norm.eps = 1e-6
    norm.activation = activation
    out_proj = SimpleNamespace(
        quant_method=quant_method,
        tp_size=tp_size,
        bias=None,
        use_decode_attn_tp=False,
    )
    self = SimpleNamespace(
        head_v_dim=g,
        norm=norm,
        out_proj=out_proj,
    )
    core = torch.randn(rows_heads, g, dtype=torch.bfloat16)
    z = torch.randn(rows_heads, g, dtype=torch.bfloat16)
    z_shape_og = (2, 4, g)  # tokens=2, v_heads=4, head_v_dim
    return self, core, z, z_shape_og


class _RecordingNormMethod:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def apply_norm_gated(self, layer, x, z, w, group, eps, sigmoid):
        self.calls.append((x.shape, z.shape, group, eps, sigmoid))
        return self.result


def test_fused_caller_requires_flag_device_and_method(monkeypatch):
    rec = _RecordingNormMethod(result=torch.zeros(2, HIDDEN))
    self, core, z, og = _fake_gdn(rec)
    fused = Qwen3_5GatedDeltaNet._norm_out_proj_fused

    monkeypatch.setattr(qwen3_5_module, "_is_cuda", True)
    monkeypatch.setattr(w8a16_gemv_module, "norm_into_gemv_enabled", lambda: False)
    assert fused(self, core, z, og) is None and rec.calls == []
    monkeypatch.setattr(w8a16_gemv_module, "norm_into_gemv_enabled", lambda: True)
    monkeypatch.setattr(qwen3_5_module, "_is_cuda", False)
    assert fused(self, core, z, og) is None and rec.calls == []

    # generic (27B-style) quant method: no scoped method -> old norm+out_proj
    monkeypatch.setattr(qwen3_5_module, "_is_cuda", True)
    plain, core, z, og = _fake_gdn(Fp8LinearMethod(Fp8Config()))
    assert fused(plain, core, z, og) is None
    # quant_method without any apply_norm_gated attribute
    none_method, core, z, og = _fake_gdn(object())
    assert fused(none_method, core, z, og) is None


def test_norm_fusion_gate_tracks_default_selection_without_copied_flags(monkeypatch):
    # The two gates (models/qwen3_5.py and w8a16_gemv) resolve through one
    # call-time function: no import-order pinning, and an eligible automatic
    # selection enables the fusion without any operator-copied switch, while a
    # saved explicit false keeps the original separate norm + out_proj path.
    from sglang.kernels.ops.gemm.sm120_online_fp8 import configure_online_fp8

    monkeypatch.delenv("SGLANG_NORM_INTO_GEMV", raising=False)
    configure_online_fp8(False, cuda_available=False, capability=None)
    assert w8a16_gemv_module.norm_into_gemv_enabled() is False
    configure_online_fp8(
        None, cuda_available=True, capability=(12, 0), model_eligible=True
    )
    assert w8a16_gemv_module.norm_into_gemv_enabled() is True
    monkeypatch.setenv("SGLANG_NORM_INTO_GEMV", "0")
    assert w8a16_gemv_module.norm_into_gemv_enabled() is False
    monkeypatch.delenv("SGLANG_NORM_INTO_GEMV")
    configure_online_fp8(
        False, cuda_available=True, capability=(12, 0), model_eligible=True
    )
    assert w8a16_gemv_module.norm_into_gemv_enabled() is False
    monkeypatch.setenv("SGLANG_NORM_INTO_GEMV", "1")
    assert w8a16_gemv_module.norm_into_gemv_enabled() is True
    configure_online_fp8(False, cuda_available=False, capability=None)


def test_fused_caller_contract_guards(monkeypatch):
    monkeypatch.setattr(qwen3_5_module, "_is_cuda", True)
    monkeypatch.setattr(w8a16_gemv_module, "norm_into_gemv_enabled", lambda: True)
    fused = Qwen3_5GatedDeltaNet._norm_out_proj_fused

    ok = _RecordingNormMethod(result=torch.zeros(2, HIDDEN))
    self, core, z, og = _fake_gdn(ok)
    assert fused(self, core, z, og) is ok.result
    assert ok.calls == [((2, 512), (2, 512), GDN_HEAD_DIM, 1e-6, False)]

    tp2, core, z, og = _fake_gdn(_RecordingNormMethod(), tp_size=2)
    assert fused(tp2, core, z, og) is None  # never bypass RowParallel TP2 reduce
    tp2.out_proj.bias = torch.zeros(HIDDEN)
    assert fused(tp2, core, z, og) is None
    grouped, core, z, og = _fake_gdn(_RecordingNormMethod(), group_size=64)
    assert fused(grouped, core, z, og) is None
    gate3, core, z, og = _fake_gdn(_RecordingNormMethod(), activation="geglu")
    assert fused(gate3, core, z, og) is None
    self2, core, z, og = _fake_gdn(_RecordingNormMethod())
    permuted = core.transpose(0, 1).transpose(0, 1).contiguous()
    assert fused(self2, permuted, z, (2, 3, 4)) is None
    assert fused(self2, core[:-1], z[:-1], og) is None  # row/head mismatch
    sig = _RecordingNormMethod()
    self3, core, z, og = _fake_gdn(sig, activation="sigmoid")
    assert fused(self3, core, z, og) is sig.result
    assert sig.calls[-1][-1] is True


if __name__ == "__main__":
    # CI's run_unittest_files launches `python3 <file> -f` (legacy unittest
    # failfast), which pytest would reject; translate it to -x, the same way
    # sglang.test.kernels.utils.multigpu_pytest_main does, so the registered
    # invocation actually collects and runs this file instead of importing it
    # and exiting 0 with zero tests run.
    import sys as _sys

    _args = ["-x" if _arg == "-f" else _arg for _arg in _sys.argv[1:]]
    _sys.exit(_sys.modules["pytest"].main([__file__, "-v", *_args]))
