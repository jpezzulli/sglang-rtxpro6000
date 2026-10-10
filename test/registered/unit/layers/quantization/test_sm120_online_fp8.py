from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.kernels.ops.gemm import sm120_online_fp8
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


def test_environment_switch_unset_or_blank_is_automatic(monkeypatch):
    from sglang.srt.environ import envs

    monkeypatch.delenv("SGLANG_SM120_ONLINE_MXFP8", raising=False)
    assert envs.SGLANG_SM120_ONLINE_MXFP8.get() is None
    monkeypatch.setenv("SGLANG_SM120_ONLINE_MXFP8", "")
    assert envs.SGLANG_SM120_ONLINE_MXFP8.get() is None
    for word in ("true", "TRUE", "1", "yes", "y"):
        monkeypatch.setenv("SGLANG_SM120_ONLINE_MXFP8", word)
        assert envs.SGLANG_SM120_ONLINE_MXFP8.get() is True
    for word in ("false", "0", "no", "n"):
        monkeypatch.setenv("SGLANG_SM120_ONLINE_MXFP8", word)
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


# ---------------------------------------------------------------------------
# Opt-in donor W8A16 GEMV over the resident rowwise-FP8 output heads.
# ---------------------------------------------------------------------------

GEMV_ENV = "SGLANG_FP8_W8A16_GEMV"
GEMV_MAX_M_ENV = "SGLANG_FP8_W8A16_GEMV_MAX_M"
# The donor kernel's own limit (w8a16_gemv asserts M <= 16).
DONOR_MAX_M = 16


def _rowwise_head(rows: int, columns: int):
    linear = nn.Linear(columns, rows, bias=False, dtype=torch.bfloat16)
    linear.weight.data.copy_(
        ((torch.arange(rows * columns).reshape(rows, columns) % 9) - 4).to(
            torch.bfloat16
        )
    )
    replace_linear_weight_rowwise_fp8(linear)
    return linear.weight


def _supported(hidden, weight, scale=...):
    if scale is ...:
        scale = rowwise_scale_of(weight)
    return sm120_online_fp8.w8a16_gemv_supported(hidden, weight, scale)


@pytest.fixture
def gemv_on(monkeypatch):
    from sglang.kernels.ops.gemm import sm120_online_fp8

    monkeypatch.setenv(GEMV_ENV, "1")
    monkeypatch.delenv(GEMV_MAX_M_ENV, raising=False)
    return sm120_online_fp8


def test_w8a16_gemv_output_head_path_is_opt_in(monkeypatch):
    from sglang.kernels.ops.gemm import sm120_online_fp8

    monkeypatch.delenv(GEMV_ENV, raising=False)
    weight = _rowwise_head(64, 32)
    assert sm120_online_fp8.w8a16_gemv_enabled() is False
    assert _supported(torch.zeros(1, 32, dtype=torch.bfloat16), weight) is False

    monkeypatch.setenv(GEMV_ENV, "1")
    assert sm120_online_fp8.w8a16_gemv_enabled() is True
    for rows in (1, 4, 12, DONOR_MAX_M):
        assert _supported(torch.zeros(rows, 32, dtype=torch.bfloat16), weight) is True
    # 0 rows, and 24 (C6 verification) / 33 (prefill) stay on the existing path.
    for rows in (0, 17, 24, 33, 64):
        assert _supported(torch.zeros(rows, 32, dtype=torch.bfloat16), weight) is False
    # The env can shrink the row budget, never raise it past the kernel's limit.
    monkeypatch.setenv(GEMV_MAX_M_ENV, "4")
    assert _supported(torch.zeros(4, 32, dtype=torch.bfloat16), weight) is True
    assert _supported(torch.zeros(8, 32, dtype=torch.bfloat16), weight) is False
    monkeypatch.setenv(GEMV_MAX_M_ENV, "64")
    assert _supported(torch.zeros(8, 32, dtype=torch.bfloat16), weight) is True
    assert _supported(torch.zeros(17, 32, dtype=torch.bfloat16), weight) is False


def test_w8a16_gemv_rejects_layouts_the_donor_kernel_does_not_read(gemv_on):
    weight = _rowwise_head(64, 32)
    scale = rowwise_scale_of(weight)
    hidden = torch.zeros(4, 32, dtype=torch.bfloat16)
    assert gemv_on.w8a16_gemv_supported(hidden, weight, scale) is True

    assert gemv_on.w8a16_gemv_supported(hidden, weight, None) is False
    assert _supported(hidden, weight, scale[:, None]) is False
    assert _supported(hidden, weight, scale.to(torch.bfloat16)) is False
    assert _supported(hidden, weight, scale[:32].contiguous()) is False
    # A pre-quantized / non-bf16 activation and a wrong hidden width.
    assert _supported(torch.zeros(4, 32), weight) is False
    assert _supported(torch.zeros(4, 16, dtype=torch.bfloat16), weight) is False
    # The donor kernel reads the weight along K with unit stride.
    column_major = weight.t().contiguous().t()
    assert column_major.stride(1) != 1
    assert _supported(hidden, column_major, scale) is False
    # 3-D activations are reshaped by the caller, so the gate sees 2-D only.
    assert (
        gemv_on.w8a16_gemv_supported(
            torch.zeros(2, 4, 32, dtype=torch.bfloat16), weight, scale
        )
        is False
    )


def test_w8a16_gemv_scratch_owns_its_slots(gemv_on, monkeypatch):
    from sglang.srt.layers.quantization import w8a16_gemv

    monkeypatch.setattr(w8a16_gemv, "_WS", {})
    device = torch.device("cpu")
    gemv_on._prealloc_w8a16_gemv_scratch(device)
    ws0, counters0 = w8a16_gemv._workspace_slot(device, 0)
    ws1, counters1 = w8a16_gemv._workspace_slot(device, 1)

    assert ws0.data_ptr() != ws1.data_ptr()
    assert ws0.dtype == torch.float32 and ws0.numel() == w8a16_gemv._WS_FLOATS
    assert counters0.dtype == torch.int32
    assert counters0.numel() == w8a16_gemv._WS_COUNTERS
    # The fixup CTA resets its own counter, so nothing may be dirty at the start.
    assert torch.count_nonzero(counters0) == 0
    assert torch.count_nonzero(counters1) == 0
    # Reused, never re-allocated: a captured launch keeps these addresses.
    assert w8a16_gemv._workspace_slot(device, 0)[0] is ws0
    with w8a16_gemv.scratch_slot(1):
        assert w8a16_gemv._workspace(device)[0] is ws1
    assert w8a16_gemv._workspace(device)[0] is ws0

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        w8a16_gemv._workspace_slot(torch.device("cuda:0"), 0)


def test_w8a16_gemv_plan_keeps_the_donor_head_tiles_and_scratch_limits():
    from sglang.srt.layers.quantization import w8a16_gemv

    sms = 188
    # The donor's measured draft-head tile (1.17x at M=1) and the generic
    # wide-N fallback it shares with the full target head.
    assert w8a16_gemv._plan(1, 32768, 2560, False, 1, sms) == (
        32,
        256,
        1,
        False,
        None,
        4,
        3,
    )
    assert w8a16_gemv._plan(4, 248320, 2560, False, 1, sms) == (
        128,
        256,
        1,
        True,
        None,
        8,
        3,
    )
    # At TP2 the local shard is 124160 rows, which the table does not key, and
    # K=2560 (8-block grid) falls through to the split-K planner instead.
    assert w8a16_gemv._plan(1, 124160, 2560, False, 1, sms)[2] == 1
    assert w8a16_gemv._plan(4, 124160, 2560, False, 1, sms) == (
        32,
        128,
        1,
        True,
        None,
        4,
        3,
    )
    # A split-K plan wider than the counter array must degrade to one split
    # rather than write past the scratch.
    wide = w8a16_gemv._fit(1, 248320, (16, 128, 10, False, None, 4, 3))
    assert wide[2] == 1
    assert w8a16_gemv._fit(1, 1024, (16, 128, 10, False, None, 4, 3))[2] == 10


def test_rowwise_lm_head_routes_to_the_donor_gemv_only_within_its_contract(
    gemv_on, monkeypatch
):
    from sglang.srt.layers.quantization import w8a16_gemv

    weight = _rowwise_head(64, 32)
    calls = []

    def fake_gemv(x, w, scale, cfg=None, out=None, out2=None, split_n=0):
        calls.append((x.shape, w.shape, tuple(scale.shape)))
        return torch.full((x.shape[0], w.shape[0]), 0.5, dtype=torch.bfloat16)

    monkeypatch.setattr(w8a16_gemv, "w8a16_gemv", fake_gemv)

    out = gemv_on.rowwise_fp8_lm_head_logits(
        torch.zeros(4, 32, dtype=torch.bfloat16), weight
    )
    assert out.shape == (4, 64) and out.dtype is torch.bfloat16
    assert calls == [((4, 32), (64, 32), (64,))]

    # 3-D activations are flattened for the GEMV and reshaped back.
    out = gemv_on.rowwise_fp8_lm_head_logits(
        torch.zeros(2, 3, 32, dtype=torch.bfloat16), weight
    )
    assert out.shape == (2, 3, 64)
    assert calls[-1] == ((6, 32), (64, 32), (64,))

    # Above the row budget the original dequantizing fallback runs instead.
    out = gemv_on.rowwise_fp8_lm_head_logits(
        torch.zeros(33, 32, dtype=torch.bfloat16), weight
    )
    assert out.shape == (33, 64)
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Default selection: eligible Flash-Next SM120 launches choose the accepted
# rowwise-FP8 bundle automatically; saved explicit values stay private
# escape hatches; unsupported automatic configurations keep original paths.
# ---------------------------------------------------------------------------


def test_default_selection_is_automatic_on_eligible_flash_next():
    assert (
        configure_online_fp8(
            None, cuda_available=True, capability=(12, 0), model_eligible=True
        )
        is True
    )
    assert online_fp8_enabled() is True
    assert sm120_online_fp8.fast_paths_enabled() is True
    configure_online_fp8(False, cuda_available=False, capability=None)


def test_unsupported_automatic_configurations_keep_the_original_path():
    for kwargs in (
        dict(cuda_available=False, capability=None, model_eligible=True),
        dict(cuda_available=True, capability=(9, 0), model_eligible=True),
        dict(cuda_available=True, capability=(12, 0), model_eligible=False),
    ):
        assert configure_online_fp8(None, **kwargs) is False
        assert online_fp8_enabled() is False
        assert sm120_online_fp8.fast_paths_enabled() is False


def test_saved_explicit_opt_out_stays_silent_and_isolates_the_bundle():
    # A saved false on a fully eligible machine must not raise, must keep the
    # original (unconverted, unshared-fast-path) routes, and must not be
    # overwritten by eligibility.
    assert (
        configure_online_fp8(
            False, cuda_available=True, capability=(12, 0), model_eligible=True
        )
        is False
    )
    assert online_fp8_enabled() is False
    assert sm120_online_fp8.fast_paths_enabled() is False
    configure_online_fp8(False, cuda_available=False, capability=None)


def test_explicit_unsupported_requests_still_fail_loud_despite_model_eligibility():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        configure_online_fp8(
            True, cuda_available=False, capability=None, model_eligible=True
        )
    with pytest.raises(RuntimeError, match="exactly SM120"):
        configure_online_fp8(
            True, cuda_available=True, capability=(12, 1), model_eligible=True
        )
    configure_online_fp8(False, cuda_available=False, capability=None)


def test_fast_path_gates_follow_the_resolved_decision(monkeypatch):
    from sglang.srt.layers.moe import topk
    from sglang.srt.layers.quantization import w8a16_gemv

    configure_online_fp8(
        None, cuda_available=True, capability=(12, 0), model_eligible=True
    )
    try:
        monkeypatch.delenv(GEMV_ENV, raising=False)
        assert sm120_online_fp8.w8a16_gemv_enabled() is True
        monkeypatch.setenv(GEMV_ENV, "0")
        assert sm120_online_fp8.w8a16_gemv_enabled() is False
        monkeypatch.delenv("SGLANG_ROUTER_FAST_TOPK", raising=False)
        assert topk._router_fast_topk_enabled() is True
        monkeypatch.setenv("SGLANG_ROUTER_FAST_TOPK", "false")
        assert topk._router_fast_topk_enabled() is False
        monkeypatch.delenv("SGLANG_NORM_INTO_GEMV", raising=False)
        assert w8a16_gemv.norm_into_gemv_enabled() is True
        monkeypatch.setenv("SGLANG_NORM_INTO_GEMV", "0")
        assert w8a16_gemv.norm_into_gemv_enabled() is False
    finally:
        configure_online_fp8(False, cuda_available=False, capability=None)


def test_gated_by_fast_paths_is_the_shared_tri_state():
    configure_online_fp8(
        None, cuda_available=True, capability=(12, 0), model_eligible=True
    )
    assert sm120_online_fp8.gated_by_fast_paths(None) is True
    assert sm120_online_fp8.gated_by_fast_paths(False) is False
    assert sm120_online_fp8.gated_by_fast_paths(True) is True
    configure_online_fp8(
        False, cuda_available=True, capability=(12, 0), model_eligible=True
    )
    assert sm120_online_fp8.gated_by_fast_paths(None) is False
    assert sm120_online_fp8.gated_by_fast_paths(True) is True


def test_flash_next_metadata_only_recognizes_the_next_family():
    assert (
        sm120_online_fp8.flash_next_metadata(
            {"architectures": ["Qwen4ExpForConditionalGeneration"]}
        )
        is True
    )
    assert (
        sm120_online_fp8.flash_next_metadata(
            {"model_type": "x", "text_config": {"model_type": "qwen4_exp_text"}}
        )
        is True
    )
    assert (
        sm120_online_fp8.flash_next_metadata(
            SimpleNamespace(
                architectures=["Qwen4ExpForConditionalGeneration"],
                model_type="qwen4_exp",
                text_config=None,
            )
        )
        is True
    )
    assert (
        sm120_online_fp8.flash_next_metadata(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {"model_type": "qwen3_5_text"},
            }
        )
        is False
    )
    assert sm120_online_fp8.flash_next_metadata({"model_type": "llama"}) is False
    assert sm120_online_fp8.flash_next_metadata(None) is False


def test_launch_precision_identity_follows_effective_precision(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.delenv("SGLANG_SM120_ONLINE_MXFP8", raising=False)
    resolve = sm120_online_fp8.resolve_precision
    # automatic and explicit-on share the accepted rowwise identity, distinct
    # from the legacy "true" namespace which meant the mixed-MXFP8 build.
    assert (
        resolve(True, cuda_available=True, capability=(12, 0), model_eligible=True)
        == "rowwise_fp8"
    )
    assert (
        resolve(None, cuda_available=True, capability=(12, 0), model_eligible=True)
        == "rowwise_fp8"
    )
    # auto-off and explicit-off keep the untouched-weight identity.
    assert (
        resolve(None, cuda_available=False, capability=None, model_eligible=True)
        == "false"
    )
    assert (
        resolve(None, cuda_available=True, capability=(12, 0), model_eligible=False)
        == "false"
    )
    assert (
        resolve(False, cuda_available=True, capability=(12, 0), model_eligible=True)
        == "false"
    )
    # The recipe/container identity probe decides with the same logic the
    # runtime applies, from the checkpoint's own config.json metadata.
    import json
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        for name, payload in (
            ("next", {"architectures": ["Qwen4ExpForConditionalGeneration"]}),
            ("dense", {"architectures": ["Qwen3_5ForConditionalGeneration"]}),
        ):
            os.mkdir(os.path.join(tmp, name))
            with open(os.path.join(tmp, name, "config.json"), "w") as handle:
                json.dump(payload, handle)
            assert sm120_online_fp8.launch_precision(os.path.join(tmp, name)) == (
                "rowwise_fp8" if name == "next" else "false"
            )
        monkeypatch.setenv("SGLANG_SM120_ONLINE_MXFP8", "false")
        assert sm120_online_fp8.launch_precision(os.path.join(tmp, "next")) == "false"
        monkeypatch.setenv("SGLANG_SM120_ONLINE_MXFP8", "true")
        assert (
            sm120_online_fp8.launch_precision(os.path.join(tmp, "dense"))
            == "rowwise_fp8"
        )
        # The recipe/container probe checks the same representation contract
        # the runtime applies: a tied-head checkpoint and a non-BF16 compute
        # dtype resolve to automatic-off before any namespace is derived, so
        # recipe precision stays consistent with the runtime decision.
        monkeypatch.delenv("SGLANG_SM120_ONLINE_MXFP8")
        os.mkdir(os.path.join(tmp, "tied"))
        with open(os.path.join(tmp, "tied", "config.json"), "w") as handle:
            json.dump(
                {
                    "architectures": ["Qwen4ExpForConditionalGeneration"],
                    "tie_word_embeddings": True,
                },
                handle,
            )
        assert sm120_online_fp8.launch_precision(os.path.join(tmp, "tied")) == "false"
        assert (
            sm120_online_fp8.launch_precision(
                os.path.join(tmp, "next"), compute_dtype=torch.float16
            )
            == "false"
        )


def test_scheduler_initialization_resolves_the_default_without_a_flag(monkeypatch):
    # The shared initialization boundary (scheduler -> initialize_bf16_gemm_
    # config) must decide automatically from the actual loaded checkpoint
    # metadata and the actual device, with no copied launch flags involved.
    from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config

    monkeypatch.delenv("SGLANG_SM120_ONLINE_MXFP8", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    server_args = SimpleNamespace(
        bf16_gemm_backend="torch", enable_deterministic_inference=False
    )
    next_config = SimpleNamespace(
        hf_config={"architectures": ["Qwen4ExpForConditionalGeneration"]}
    )
    dense_config = SimpleNamespace(
        hf_config={"architectures": ["Qwen3_5ForConditionalGeneration"]}
    )
    # On this CPU-only box the automatic path stays off for every model and
    # nothing raises; eligibility is what the SM120 contract pins.
    initialize_bf16_gemm_config(server_args, model_config=next_config)
    assert online_fp8_enabled() is False
    assert sm120_online_fp8.fast_paths_enabled() is False
    initialize_bf16_gemm_config(server_args, model_config=dense_config)
    assert online_fp8_enabled() is False
    # A saved explicit request keeps the fail-loud contract at this boundary.
    monkeypatch.setenv("SGLANG_SM120_ONLINE_MXFP8", "true")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        initialize_bf16_gemm_config(server_args, model_config=next_config)
    # A saved explicit false propagates unchanged: silent, original paths.
    monkeypatch.setenv("SGLANG_SM120_ONLINE_MXFP8", "false")
    initialize_bf16_gemm_config(server_args, model_config=next_config)
    assert online_fp8_enabled() is False


def _sm120_server_args():
    return SimpleNamespace(
        bf16_gemm_backend="torch", enable_deterministic_inference=False
    )


def test_automatic_selection_respects_the_loaded_representation(monkeypatch):
    # Review1 regression at the real initializer boundary with a mocked SM120:
    # an FP16 or tied-head Flash-Next config must stay automatic-OFF (the
    # accepted conversion rejects FP16 residents and the accepted Qwen4Exp
    # post-load rejects tied heads), while an explicit request still reaches
    # those accepted, useful failures instead of a silent downgrade.
    from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config

    monkeypatch.delenv("SGLANG_SM120_ONLINE_MXFP8", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )
    server_args = _sm120_server_args()
    next_hf = {"architectures": ["Qwen4ExpForConditionalGeneration"]}
    bf16 = SimpleNamespace(hf_config=next_hf, dtype=torch.bfloat16)
    fp16 = SimpleNamespace(hf_config=next_hf, dtype=torch.float16)
    tied = SimpleNamespace(
        hf_config={**next_hf, "tie_word_embeddings": True}, dtype=torch.bfloat16
    )
    nested_tied = SimpleNamespace(
        hf_config={
            **next_hf,
            "text_config": {"tie_word_embeddings": True},
        },
        dtype=torch.bfloat16,
    )

    initialize_bf16_gemm_config(server_args, model_config=bf16, device="cuda", gpu_id=0)
    assert online_fp8_enabled() is True and sm120_online_fp8.fast_paths_enabled()

    for unsupported in (fp16, tied, nested_tied):
        initialize_bf16_gemm_config(
            server_args, model_config=unsupported, device="cuda", gpu_id=0
        )
        assert online_fp8_enabled() is False
        assert sm120_online_fp8.fast_paths_enabled() is False

    # Explicit on FP16: the switch turns on (legacy contract) and the accepted
    # BF16-only conversion itself fails loudly with a useful message.
    monkeypatch.setenv("SGLANG_SM120_ONLINE_MXFP8", "true")
    initialize_bf16_gemm_config(server_args, model_config=fp16, device="cuda", gpu_id=0)
    assert online_fp8_enabled() is True
    linear = nn.Linear(4, 4, bias=False, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="BF16 resident weight"):
        replace_linear_weight_rowwise_fp8(linear)

    # Explicit false on the same unsupported configs: silent original paths.
    monkeypatch.setenv("SGLANG_SM120_ONLINE_MXFP8", "false")
    for unsupported in (fp16, tied):
        initialize_bf16_gemm_config(
            server_args, model_config=unsupported, device="cuda", gpu_id=0
        )
        assert online_fp8_enabled() is False


def test_device_resolution_probes_the_assigned_rank_device(monkeypatch):
    # Review1 regression: init runs before the ModelRunner selects the device,
    # and a heterogeneous host must not be decided by a GPU0 probe. GPU0 is
    # SM90 here, the scheduler-assigned GPU1 is SM120.
    from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config

    monkeypatch.delenv("SGLANG_SM120_ONLINE_MXFP8", raising=False)
    caps = {0: (9, 0), 1: (12, 0)}
    probed = []

    def fake_capability(device=None):
        probed.append(device)
        return caps[device if device is not None else 0]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", fake_capability)
    server_args = _sm120_server_args()
    bf16 = SimpleNamespace(
        hf_config={"architectures": ["Qwen4ExpForConditionalGeneration"]},
        dtype=torch.bfloat16,
    )

    initialize_bf16_gemm_config(server_args, model_config=bf16, device="cuda", gpu_id=1)
    assert (
        online_fp8_enabled() is True
    ), "assigned SM120 rank must not be decided by GPU0"
    assert probed and all(index == 1 for index in probed)

    probed.clear()
    initialize_bf16_gemm_config(server_args, model_config=bf16, device="cuda", gpu_id=0)
    assert online_fp8_enabled() is False, "inverse: no enabling on the wrong hardware"

    # Explicit request on the SM90 rank: fail loud, unchanged contract.
    monkeypatch.setenv("SGLANG_SM120_ONLINE_MXFP8", "true")
    with pytest.raises(RuntimeError, match="exactly SM120"):
        initialize_bf16_gemm_config(
            server_args, model_config=bf16, device="cuda", gpu_id=0
        )
    # Non-CUDA runtime device kind: explicit fail-loud, automatic silent-off.
    with pytest.raises(RuntimeError, match="requires CUDA"):
        initialize_bf16_gemm_config(
            server_args, model_config=bf16, device="cpu", gpu_id=1
        )
    monkeypatch.setenv("SGLANG_SM120_ONLINE_MXFP8", "")
    initialize_bf16_gemm_config(server_args, model_config=bf16, device="cpu", gpu_id=1)
    assert online_fp8_enabled() is False
