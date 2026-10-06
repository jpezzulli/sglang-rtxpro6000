from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.ops.gemm import sm120_w8a16_gemv
from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    attach_rowwise_ingest,
    configure_online_fp8,
    convert_eligible_linears_to_mxfp8,
    dequantize_rowwise_weight,
    online_fp8_enabled,
    replace_linear_weight_rowwise_fp8,
    rowwise_fp8_lm_head_logits,
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


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _rowwise_head_weight(rows: int, columns: int):
    linear = nn.Linear(columns, rows, bias=False, dtype=torch.bfloat16)
    linear.weight.data.copy_(
        ((torch.arange(rows * columns).reshape(rows, columns) % 9) - 4).to(
            torch.bfloat16
        )
    )
    replace_linear_weight_rowwise_fp8(linear)
    return linear.weight


def _supported(hidden: torch.Tensor, weight: torch.Tensor, scale=None) -> bool:
    if scale is None:
        scale = rowwise_scale_of(weight)
    return sm120_w8a16_gemv.lowrow_gemv_supported(hidden, weight, scale)


def test_lowrow_gemv_is_opt_in_and_explicitly_shape_gated(monkeypatch):
    monkeypatch.delenv(sm120_w8a16_gemv.GEMV_ENV, raising=False)
    weight = _rowwise_head_weight(64, 32)
    assert sm120_w8a16_gemv.lowrow_gemv_enabled() is False
    assert _supported(torch.zeros(1, 32, dtype=torch.bfloat16), weight) is False

    monkeypatch.setenv(sm120_w8a16_gemv.GEMV_ENV, "1")
    assert sm120_w8a16_gemv.lowrow_gemv_enabled() is True
    assert sm120_w8a16_gemv.MAX_ROWS == 16
    for rows in (1, 4, 12, 16):
        assert _supported(torch.zeros(rows, 32, dtype=torch.bfloat16), weight) is True
    # 24 is C6 verification and 17/33 sit above the donor kernel's own limit.
    for rows in (0, 17, 24, 33, 64):
        assert _supported(torch.zeros(rows, 32, dtype=torch.bfloat16), weight) is False

    assert _supported(torch.zeros(4, 32), weight) is False  # fp32 activation
    assert _supported(torch.zeros(4, 16, dtype=torch.bfloat16), weight) is False
    scale = rowwise_scale_of(weight)
    transposed = weight.t().contiguous().t()
    assert transposed.stride(1) != 1
    assert (
        _supported(torch.zeros(4, 32, dtype=torch.bfloat16), transposed, scale)
        is False
    )
    assert (
        sm120_w8a16_gemv.lowrow_gemv_supported(
            torch.zeros(4, 32, dtype=torch.bfloat16), weight, None
        )
        is False
    )  # a scale-less tensor is never the rowwise head's
    assert _supported(
        torch.zeros(4, 32, dtype=torch.bfloat16), weight, scale[:, None]
    ) is False
    assert (
        _supported(
            torch.zeros(4, 32, dtype=torch.bfloat16),
            weight,
            scale.to(torch.bfloat16),
        )
        is False
    )
    assert (
        _supported(
            torch.zeros(4, 32, dtype=torch.bfloat16),
            weight,
            scale[:32].contiguous(),
        )
        is False
    )


def test_lowrow_gemv_scratch_is_per_slot_zeroed_and_capture_guarded(monkeypatch):
    monkeypatch.setattr(sm120_w8a16_gemv, "_WS", {})
    device = torch.device("cpu")
    sm120_w8a16_gemv.prealloc(device)
    ws0, counters0 = sm120_w8a16_gemv._workspace_slot(device, 0)
    ws1, counters1 = sm120_w8a16_gemv._workspace_slot(device, 1)

    assert ws0.data_ptr() != ws1.data_ptr()
    assert ws0.dtype == torch.float32 and ws0.numel() == sm120_w8a16_gemv._WS_FLOATS
    assert (
        counters0.dtype == torch.int32
        and counters0.numel() == sm120_w8a16_gemv._WS_COUNTERS
    )
    assert torch.count_nonzero(counters0) == 0
    assert torch.count_nonzero(counters1) == 0
    # Reused, never re-allocated: a captured launch keeps these addresses.
    assert sm120_w8a16_gemv._workspace_slot(device, 0)[0] is ws0
    with sm120_w8a16_gemv.scratch_slot(1):
        assert sm120_w8a16_gemv._workspace(device)[0] is ws1
    assert sm120_w8a16_gemv._workspace(device)[0] is ws0

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        sm120_w8a16_gemv._workspace_slot(torch.device("cuda"), 1)


def test_lowrow_gemv_plan_keeps_donor_head_tiles_and_fits_the_scratch():
    sms = 188
    # The donor's measured tiles for the exact head shapes, incl. the M == 1
    # broadcast path on the draft head and the M == 4 exception.
    assert sm120_w8a16_gemv._plan(1, 32768, 2560, sms) == (32, 256, 1, False, 4, 3)
    assert sm120_w8a16_gemv._plan(1, 248320, 2560, sms) == (128, 256, 1, True, 8, 3)
    for rows in (4, 12, 16):
        assert (
            sm120_w8a16_gemv._plan(rows, 32768, 2560, sms)
            == (128, 256, 1, True, 8, 3)
        )
        assert (
            sm120_w8a16_gemv._plan(rows, 248320, 2560, sms)
            == (128, 256, 1, True, 8, 3)
        )

    # A small (TP2-sharded or hot-vocab) head underfills the grid and takes split-K.
    _block_n, block_k, splits, use_dot, _warps, _stages = sm120_w8a16_gemv._plan(
        4, 1024, 2560, sms
    )
    assert 1 < splits <= sm120_w8a16_gemv._MAX_SPLITS
    assert splits <= _cdiv(2560, block_k) and use_dot is True
    _block_n, block_k, splits, use_dot, _warps, _stages = sm120_w8a16_gemv._plan(
        1, 1024, 2560, sms
    )
    assert 1 < splits <= sm120_w8a16_gemv._MAX_SPLITS and use_dot is False

    assert sm120_w8a16_gemv._fit(16, 32768, (128, 128, 32, True, 4, 3))[2] == 1
    assert sm120_w8a16_gemv._fit(1, 1_000_000, (32, 128, 8, False, 4, 3))[2] == 1
    assert sm120_w8a16_gemv._fit(4, 1024, (32, 128, 5, True, 4, 3))[2] == 5


def test_lm_head_dispatch_routes_low_rows_only_when_gated(monkeypatch):
    from sglang.kernels.ops.gemm import sm120_online_fp8

    weight = _rowwise_head_weight(64, 32)
    gemv_calls = []
    kernel_calls = []

    def fake_gemv(hidden_2d, _weight, _scale, out=None):
        gemv_calls.append(tuple(hidden_2d.shape))
        return torch.zeros(hidden_2d.shape[0], 64, dtype=torch.bfloat16)

    class _FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                kernel_calls.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(sm120_w8a16_gemv, "lowrow_fp8_gemv", fake_gemv)
    monkeypatch.setattr(
        sm120_online_fp8, "_rowwise_fp8_gemv_kernel", _FakeKernel()
    )

    # Gate off by default: the original kernel owns even the low-row calls.
    logits = rowwise_fp8_lm_head_logits(
        torch.zeros(16, 32, dtype=torch.bfloat16), weight
    )
    assert gemv_calls == [] and len(kernel_calls) == 1
    assert tuple(logits.shape) == (16, 64) and logits.dtype == torch.bfloat16

    monkeypatch.setenv(sm120_w8a16_gemv.GEMV_ENV, "1")
    logits = rowwise_fp8_lm_head_logits(
        torch.zeros(16, 32, dtype=torch.bfloat16), weight
    )
    assert gemv_calls == [(16, 32)] and len(kernel_calls) == 1
    assert tuple(logits.shape) == (16, 64) and logits.dtype == torch.bfloat16

    # C6's 24 rows keep the original implementation with the gate on.
    rowwise_fp8_lm_head_logits(torch.zeros(24, 32, dtype=torch.bfloat16), weight)
    assert gemv_calls == [(16, 32)] and len(kernel_calls) == 2
