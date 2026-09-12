from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    attach_rowwise_ingest,
    configure_online_fp8,
    convert_eligible_linears_to_mxfp8,
    dequantize_rowwise_weight,
    online_fp8_enabled,
    replace_linear_weight_rowwise_fp8,
    rowwise_scale_of,
    select_rowwise_weight_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from torch import nn

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Unquantized:
    pass


class _Excluded(nn.Module):
    pass


class _Linear(nn.Module):
    def __init__(self, rows=128, columns=128, *, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(rows, columns, dtype=dtype))
        self.quant_method = _Unquantized()


def test_online_fp8_is_opt_in_and_exact_sm120():
    assert configure_online_fp8(False, cuda_available=False, capability=None) is False
    assert online_fp8_enabled() is False

    with pytest.raises(RuntimeError, match="requires CUDA"):
        configure_online_fp8(True, cuda_available=False, capability=None)
    with pytest.raises(RuntimeError, match="exactly SM120"):
        configure_online_fp8(True, cuda_available=True, capability=(12, 1))

    assert configure_online_fp8(True, cuda_available=True, capability=(12, 0)) is True
    assert online_fp8_enabled() is True
    configure_online_fp8(False, cuda_available=False, capability=None)


def test_environment_switch_defaults_off(monkeypatch):
    from sglang.srt.environ import envs

    monkeypatch.delenv("SGLANG_SM120_ONLINE_MXFP8", raising=False)
    assert envs.SGLANG_SM120_ONLINE_MXFP8.get() is False


def test_candidate_conversion_is_bounded_and_option_off_is_noop():
    root = nn.Module()
    root.proj = _Linear()
    root.small = _Linear(rows=96)
    root.gate = _Linear()
    root.experts = _Excluded()
    root.experts.proj = _Linear()
    root.wrong_dtype = _Linear(dtype=torch.float32)

    calls = []

    def factory():
        method = SimpleNamespace(kind="mxfp8")
        calls.append(method)
        return method

    assert (
        convert_eligible_linears_to_mxfp8(
            root,
            enabled=False,
            method_factory=factory,
            unquantized_method_type=_Unquantized,
            excluded_module_type=_Excluded,
        )
        == []
    )
    assert calls == []
    assert isinstance(root.proj.quant_method, _Unquantized)

    converted = convert_eligible_linears_to_mxfp8(
        root,
        enabled=True,
        method_factory=factory,
        unquantized_method_type=_Unquantized,
        excluded_module_type=_Excluded,
    )
    assert converted == ["proj"]
    assert root.proj.quant_method is calls[0]
    assert isinstance(root.small.quant_method, _Unquantized)
    assert isinstance(root.gate.quant_method, _Unquantized)
    assert isinstance(root.experts.proj.quant_method, _Unquantized)
    assert isinstance(root.wrong_dtype.quant_method, _Unquantized)


def test_rowwise_quantization_round_trip_and_replacement_are_idempotent():
    weight = torch.tensor(
        [[-4.0, -1.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.bfloat16,
    )
    linear = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
    linear.weight.data.copy_(weight)
    linear.weight.weight_loader = object()

    freed = replace_linear_weight_rowwise_fp8(linear)
    assert freed == weight.numel() * weight.element_size()
    assert linear.weight.dtype == torch.float8_e4m3fn
    assert rowwise_scale_of(linear.weight).shape == (2,)
    assert hasattr(linear.weight, "weight_loader")
    torch.testing.assert_close(
        dequantize_rowwise_weight(linear.weight),
        weight,
        rtol=0.03,
        atol=0.03,
    )
    assert replace_linear_weight_rowwise_fp8(linear) == 0

    reloaded = (weight * 3).contiguous()
    linear.weight.weight_loader(linear.weight, reloaded)
    torch.testing.assert_close(
        dequantize_rowwise_weight(linear.weight),
        reloaded,
        rtol=0.03,
        atol=0.03,
    )


def test_meta_ingest_installs_resident_fp8_parameter_and_scale():
    linear = nn.Linear(4, 2, bias=False, device="meta", dtype=torch.bfloat16)
    assert attach_rowwise_ingest([linear], target_device=torch.device("cpu")) == 1

    loaded = torch.tensor(
        [[-3.0, -1.0, 1.0, 3.0], [2.0, 2.0, 2.0, 2.0]],
        dtype=torch.bfloat16,
    )
    linear.weight.weight_loader(linear.weight, loaded)
    assert linear.weight.device.type == "cpu"
    assert linear.weight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        dequantize_rowwise_weight(linear.weight), loaded, rtol=0.03, atol=0.03
    )


def test_hot_token_selection_preserves_matching_rowwise_scales():
    linear = nn.Linear(4, 5, bias=False, dtype=torch.bfloat16)
    linear.weight.data.copy_(torch.arange(20).reshape(5, 4))
    replace_linear_weight_rowwise_fp8(linear)

    token_ids = torch.tensor([4, 1, 3])
    selected = select_rowwise_weight_rows(linear.weight, token_ids)
    assert isinstance(selected, nn.Parameter)
    assert not hasattr(selected, "weight_loader")
    torch.testing.assert_close(
        rowwise_scale_of(selected), rowwise_scale_of(linear.weight)[token_ids]
    )
    torch.testing.assert_close(
        dequantize_rowwise_weight(selected),
        dequantize_rowwise_weight(linear.weight)[token_ids],
    )


def test_logits_processor_prioritizes_rowwise_metadata_over_stale_quant_method(
    monkeypatch,
):
    from sglang.kernels.ops.gemm import sm120_online_fp8
    from sglang.srt.layers.logits_processor import LogitsProcessor

    linear = nn.Linear(4, 5, bias=False, dtype=torch.bfloat16)
    replace_linear_weight_rowwise_fp8(linear)
    expected = torch.randn(2, 5)
    calls = []

    def rowwise_logits(hidden_states, weight):
        calls.append((hidden_states, weight))
        return expected

    class _StaleQuantMethod:
        def apply(self, *args, **kwargs):
            raise AssertionError("stale draft quant method must not run")

    monkeypatch.setattr(sm120_online_fp8, "rowwise_fp8_lm_head_logits", rowwise_logits)
    processor = SimpleNamespace(use_fp32_lm_head=False, rl_on_policy_target=None)
    lm_head = SimpleNamespace(weight=linear.weight, quant_method=_StaleQuantMethod())
    hidden = torch.randn(2, 4)

    actual = LogitsProcessor._compute_lm_head(processor, hidden, lm_head)

    assert actual is expected
    assert calls == [(hidden, linear.weight)]
