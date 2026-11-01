# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for the accepted SM120 small-batch fast paths' SELECTION.

What is pinned here is the part that decides which kernel a call goes to, and
the part that keeps a weight's representation and its scales together:

  * the rowwise-FP8 bundle publishes one decision, and every companion switch
    (W8A16 GEMV, fused gated norm, MTP FC GEMV, draft MoE GEMV, fast router)
    follows it when unset, while a saved explicit value stays honored;
  * each path's own layout/capability guards, so an unsupported call keeps the
    path it already had rather than being approximated by a foreign kernel;
  * the split-K scratch lifetime (preallocated before a capture, one slot per
    device, reused) that the weight-installation hooks are responsible for.

No CPU result here is evidence of GPU math or speed. The kernels' numerics are
covered by ``test/registered/kernels/test_sm120_small_batch_gemv.py``,
``test/registered/kernels/ops/moe/test_moe_router_softmax_fast.py`` and
``test/registered/moe/test_draft_moe_gemv.py``, all of which require real SM120.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

GEMV_ENV = "SGLANG_FP8_W8A16_GEMV"
GEMV_MAX_M_ENV = "SGLANG_FP8_W8A16_GEMV_MAX_M"
NORM_ENV = "SGLANG_NORM_INTO_GEMV"
MTP_ENV = "SGLANG_MTP_FC_GEMV"
DRAFT_ENV = "SGLANG_OPT_DRAFT_MOE_GEMV"
ROUTER_ENV = "SGLANG_ROUTER_FAST_TOPK"
ALL_FAST_ENVS = (GEMV_ENV, NORM_ENV, MTP_ENV, DRAFT_ENV, ROUTER_ENV)

#: Minimal checkpoint metadata that names the Flash-Next family.
FLASH_NEXT_METADATA = {"model_type": "qwen4_exp_text"}

# The donor kernel's own limit (w8a16_gemv asserts M <= 16), and the widths that
# must stay on the existing kernels: 0 rows, the C6 target's 24-row verification
# and a 33-row prefill chunk.
DONOR_MAX_M = 16
INSIDE = [1, 4, 6, 8, 12, 16]
OUTSIDE = [0, 17, 24, 33]


def _load_under_test():
    """The selection module by package, or by file when the runtime deps are absent.

    Same fallback the reviewed FP8 test uses: the conversion helpers are pure
    torch, so a CPU-only host can still exercise the selection invariants.
    """
    try:
        from sglang.kernels.ops.gemm import sm120_online_fp8 as module

        return module
    except ModuleNotFoundError:
        path = (
            Path(__file__)
            .resolve()
            .parents[5]
            .joinpath("python/sglang/kernels/ops/gemm/sm120_online_fp8.py")
        )
        spec = importlib.util.spec_from_file_location("sm120_online_fp8", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


fp8 = _load_under_test()


def _require_port(name: str):
    """Import a ported module, distinguishing an absent port from an absent extra.

    A missing ``sglang`` module of this port is the thing under test, so it
    reads as a failure (and as a failure on the exact base, where none of these
    modules exist yet).  A missing third-party dependency is a worker that
    cannot run this file at all, which reads as a skip.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as err:
        if err.name == name or (err.name or "").startswith(name + "."):
            pytest.fail(
                f"{name} is missing: the accepted fast-path port is not present"
            )
        pytest.skip(f"{name} needs an optional dependency ({err.name})")


w8a16 = _require_port("sglang.srt.layers.quantization.w8a16_gemv")
rfast = _require_port("sglang.kernels.ops.moe.moe_router_softmax_fast")
draft = _require_port("sglang.srt.layers.moe.draft_moe_gemv")
# Serving-runtime modules the gates live in; imported the same way so an
# incomplete worker skips instead of reporting a false pass.
topk_module = _require_port("sglang.srt.layers.moe.topk")
mtp_module = _require_port("sglang.srt.models.qwen4_exp_mtp")


# --------------------------------------------------------------------------- #
# The bundle: one decision, five companions
# --------------------------------------------------------------------------- #


def _reset():
    fp8.configure_online_fp8(False, cuda_available=False, capability=None)


@pytest.fixture(autouse=True)
def _cpu_device_and_clean_env(monkeypatch):
    """This file's coverage is CPU-only, from the untouched-checkpoint state.

    A sibling test module that leaves the process default device on CUDA must
    not decide where these tensors live: this file is exactly the selection
    coverage that has to run on a worker with no GPU at all.
    """
    monkeypatch.delenv("SGLANG_SM120_ONLINE_MXFP8", raising=False)
    for name in ALL_FAST_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(GEMV_MAX_M_ENV, raising=False)
    _reset()
    with torch.device("cpu"):
        yield
    _reset()


def test_unset_selection_publishes_the_bundle_and_its_fast_paths_together():
    assert (
        fp8.configure_online_fp8(
            None, cuda_available=True, capability=(12, 0), model_eligible=True
        )
        is True
    )
    assert fp8.online_fp8_enabled() is True
    assert fp8.fast_paths_enabled() is True


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(cuda_available=False, capability=None, model_eligible=True),
        dict(cuda_available=True, capability=(9, 0), model_eligible=True),
        dict(cuda_available=True, capability=(12, 1), model_eligible=True),
        dict(cuda_available=True, capability=(12, 0), model_eligible=False),
    ],
)
def test_unsupported_automatic_selection_keeps_both_off(kwargs):
    assert fp8.configure_online_fp8(None, **kwargs) is False
    assert fp8.online_fp8_enabled() is False
    assert fp8.fast_paths_enabled() is False


def test_saved_explicit_opt_out_is_silent_and_takes_the_companions():
    assert (
        fp8.configure_online_fp8(
            False, cuda_available=True, capability=(12, 0), model_eligible=True
        )
        is False
    )
    assert fp8.fast_paths_enabled() is False


def test_explicit_true_off_the_flash_next_family_gets_fp8_without_the_bundle():
    # The companion tiles were measured for Flash-Next's shapes; an explicit
    # request on an unrecognized exact-SM120 model gets the FP8 GEMM dispatch it
    # asked for and none of the foreign kernels.
    assert (
        fp8.configure_online_fp8(
            True, cuda_available=True, capability=(12, 0), model_eligible=False
        )
        is True
    )
    assert fp8.online_fp8_enabled() is True
    assert fp8.fast_paths_enabled() is False


def test_explicit_unsupported_requests_still_fail_loud():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        fp8.configure_online_fp8(
            True, cuda_available=False, capability=None, model_eligible=True
        )
    with pytest.raises(RuntimeError, match="exactly SM120"):
        fp8.configure_online_fp8(
            True, cuda_available=True, capability=(12, 1), model_eligible=True
        )


def test_configure_for_model_resolves_the_draft_worker_view(monkeypatch):
    # A draft (NEXTN) worker can build the draft model alone, so a draft MoE
    # block resolves the bundle from its own config instead of waiting for the
    # main model's conversion hook.
    seen = {}

    def spy(hf_config, compute_dtype):
        seen["hf_config"] = hf_config
        return True

    monkeypatch.setattr(fp8, "checkpoint_eligible", spy)
    monkeypatch.setattr(fp8.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fp8.torch.cuda, "get_device_capability", lambda: (12, 0))
    assert (
        fp8.configure_for_model(SimpleNamespace(hf_config=FLASH_NEXT_METADATA)) is True
    )
    # The served ModelConfig is unwrapped to the checkpoint metadata it carries,
    # which is the argument checkpoint_eligible accepts.
    assert seen["hf_config"] is FLASH_NEXT_METADATA
    assert fp8.fast_paths_enabled() is True
    # No CUDA: the automatic decision is off, and an explicit request still
    # fails the boot from here rather than at the first weight load.
    monkeypatch.setattr(fp8.torch.cuda, "is_available", lambda: False)
    assert fp8.configure_for_model(FLASH_NEXT_METADATA) is False
    assert fp8.fast_paths_enabled() is False
    monkeypatch.setenv("SGLANG_SM120_ONLINE_MXFP8", "true")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        fp8.configure_for_model(FLASH_NEXT_METADATA)


def test_cache_identity_is_unchanged_by_the_fast_paths():
    # The fast paths are a dispatch decision; the effective-precision identity
    # the reviewed FP8 selection publishes must not move with them.
    assert (
        fp8.resolve_precision(
            None,
            cuda_available=True,
            capability=(12, 0),
            model_eligible=True,
        )
        == fp8.ROWWISE_FP8_PRECISION
    )
    assert (
        fp8.resolve_precision(
            False,
            cuda_available=True,
            capability=(12, 0),
            model_eligible=True,
        )
        == fp8.OFF_PRECISION
    )


def test_gated_by_fast_paths_is_the_shared_tri_state():
    fp8.configure_online_fp8(
        None, cuda_available=True, capability=(12, 0), model_eligible=True
    )
    assert fp8.gated_by_fast_paths(None) is True
    assert fp8.gated_by_fast_paths(True) is True
    assert fp8.gated_by_fast_paths(False) is False
    fp8.configure_online_fp8(
        False, cuda_available=True, capability=(12, 0), model_eligible=True
    )
    assert fp8.gated_by_fast_paths(None) is False
    assert fp8.gated_by_fast_paths(True) is True


def test_every_companion_gate_follows_the_resolved_decision(monkeypatch):
    fp8.configure_online_fp8(
        None, cuda_available=True, capability=(12, 0), model_eligible=True
    )
    for name in ALL_FAST_ENVS:
        monkeypatch.delenv(name, raising=False)
    assert fp8.w8a16_gemv_enabled() is True
    assert w8a16.norm_into_gemv_enabled() is True
    assert topk_module._router_fast_topk_enabled() is True
    assert mtp_module._mtp_fc_gemv_supported(*_mtp_args()) is False  # CPU tensors
    with envs.SGLANG_MTP_FC_GEMV.override(True), monkeypatch.context() as patch:
        # The device half of the MTP gate is separate; pin the flag half here by
        # resolving it as the bundle would on an accepted device.
        patch.setattr(mtp_module, "_bf16_entry_gemm_target_ok", lambda *t: True)
        assert mtp_module._mtp_fc_gemv_supported(*_mtp_args()) is True
    for name in ALL_FAST_ENVS:
        monkeypatch.setenv(name, "0")
    assert fp8.w8a16_gemv_enabled() is False
    assert w8a16.norm_into_gemv_enabled() is False
    assert topk_module._router_fast_topk_enabled() is False
    with envs.SGLANG_MTP_FC_GEMV.override(False), monkeypatch.context() as patch:
        patch.setattr(mtp_module, "_bf16_entry_gemm_target_ok", lambda *t: True)
        assert mtp_module._mtp_fc_gemv_supported(*_mtp_args()) is False


def test_one_explicit_companion_does_not_move_the_others(monkeypatch):
    fp8.configure_online_fp8(
        None, cuda_available=True, capability=(12, 0), model_eligible=True
    )
    monkeypatch.setenv(GEMV_ENV, "0")
    assert fp8.w8a16_gemv_enabled() is False
    assert w8a16.norm_into_gemv_enabled() is True
    assert topk_module._router_fast_topk_enabled() is True
    # And an explicit true on a bundle that resolved off stays private to that
    # one path: the FP8 conversion is still not selected, so the GEMV has no
    # rowwise-FP8 weight to read.
    fp8.configure_online_fp8(False, cuda_available=True, capability=(12, 0))
    monkeypatch.setenv(GEMV_ENV, "1")
    assert fp8.w8a16_gemv_enabled() is True
    assert _supported(_head_weight(64, 32), rows=4) is False


# --------------------------------------------------------------------------- #
# W8A16 GEMV: row budget, layout guards, head dispatch, scratch lifetime
# --------------------------------------------------------------------------- #


def _head_weight(rows: int, columns: int) -> torch.Tensor:
    linear = nn.Linear(columns, rows, bias=False, dtype=torch.bfloat16)
    linear.weight.data.copy_(
        ((torch.arange(rows * columns).reshape(rows, columns) % 9) - 4).to(
            torch.bfloat16
        )
    )
    fp8.replace_linear_weight_rowwise_fp8(linear)
    return linear.weight


def _supported(weight: torch.Tensor, rows: int, hidden: torch.Tensor | None = None):
    if hidden is None:
        hidden = torch.zeros(rows, weight.shape[1], dtype=torch.bfloat16)
    return fp8.w8a16_gemv_supported(hidden, weight, fp8.rowwise_scale_of(weight))


@pytest.fixture
def gemv_on(monkeypatch):
    fp8.configure_online_fp8(
        None, cuda_available=True, capability=(12, 0), model_eligible=True
    )
    monkeypatch.delenv(GEMV_ENV, raising=False)
    monkeypatch.delenv(GEMV_MAX_M_ENV, raising=False)
    return fp8


def test_w8a16_gemv_is_opt_in_and_bounds_rows_by_the_donor_limit(gemv_on, monkeypatch):
    weight = _head_weight(64, 32)
    for rows in INSIDE:
        assert _supported(weight, rows) is True
    # 0 rows, the C6 24-row verification width and a 33-row prefill chunk stay
    # on the kernels this candidate already had.
    for rows in OUTSIDE:
        assert _supported(weight, rows) is False
    # The budget knob may only shrink what the kernel promises.
    monkeypatch.setenv(GEMV_MAX_M_ENV, "4")
    assert fp8.w8a16_gemv_row_budget() == 4
    assert _supported(weight, 4) is True
    assert _supported(weight, 8) is False
    monkeypatch.setenv(GEMV_MAX_M_ENV, "64")
    assert fp8.w8a16_gemv_row_budget() == DONOR_MAX_M
    assert _supported(weight, 16) is True
    assert _supported(weight, 17) is False
    monkeypatch.setenv(GEMV_MAX_M_ENV, "0")
    assert fp8.w8a16_gemv_row_budget() == 1
    monkeypatch.delenv(GEMV_ENV, raising=False)
    fp8.configure_online_fp8(False, cuda_available=True, capability=(12, 0))
    assert _supported(weight, 4) is False


def test_w8a16_gemv_refuses_layouts_the_donor_kernel_does_not_read(gemv_on):
    weight = _head_weight(64, 32)
    scale = fp8.rowwise_scale_of(weight)
    hidden = torch.zeros(4, 32, dtype=torch.bfloat16)
    assert fp8.w8a16_gemv_supported(hidden, weight, scale) is True
    # A scale that is absent, the wrong dtype, or the wrong count is a different
    # representation, not a detail to guess.  [N, 1] and [1, N] are the same row
    # order and are what the reused FP8 method itself stores, so they are the
    # same scale; a strided or square one is not a row-scale vector at all.
    assert fp8.w8a16_gemv_supported(hidden, weight, None) is False
    assert fp8.w8a16_gemv_supported(hidden, weight, scale.to(torch.bfloat16)) is False
    assert fp8.w8a16_gemv_supported(hidden, weight, scale[:32].contiguous()) is False
    assert fp8.w8a16_gemv_supported(hidden, weight, scale[:, None]) is True
    assert fp8.w8a16_gemv_supported(hidden, weight, scale[None, :]) is True
    assert fp8.w8a16_gemv_supported(hidden, weight, scale.double()) is False
    square = torch.ones(8, 8, dtype=torch.float32)
    assert fp8.w8a16_gemv_supported(hidden, weight, square) is False
    assert fp8.w8a16_gemv_supported(hidden, weight, scale.contiguous()[::2]) is False
    # A pre-quantized or wrong-width activation, and a 3-D activation (the gate
    # is on the 2-D operand the caller flattens to).
    assert fp8.w8a16_gemv_supported(hidden.float(), weight, scale) is False
    assert (
        fp8.w8a16_gemv_supported(
            torch.zeros(4, 16, dtype=torch.bfloat16), weight, scale
        )
        is False
    )
    assert (
        fp8.w8a16_gemv_supported(
            torch.zeros(2, 4, 32, dtype=torch.bfloat16), weight, scale
        )
        is False
    )
    # The kernel walks both operands along their last axis with unit stride.
    assert fp8.w8a16_gemv_supported(hidden.t().contiguous().t(), weight, scale) is False
    column_major = weight.t().contiguous().t()
    assert column_major.stride(1) != 1
    assert fp8.w8a16_gemv_supported(hidden, column_major, scale) is False
    # Activations or scales on another device than the resident weight.
    assert (
        fp8.w8a16_gemv_supported(
            hidden, weight, fp8.rowwise_scale_of(weight).to("meta")
        )
        is False
    )


def test_output_head_dispatch_stays_within_the_gemv_contract(gemv_on, monkeypatch):
    weight = _head_weight(64, 32)
    calls = []

    def fake_gemv(x, w, scale, cfg=None):
        calls.append((tuple(x.shape), tuple(w.shape), tuple(scale.shape)))
        return torch.full((x.shape[0], w.shape[0]), 0.5, dtype=torch.bfloat16)

    monkeypatch.setattr(w8a16, "w8a16_gemv", fake_gemv)

    out = gemv_on.rowwise_fp8_lm_head_logits(
        torch.zeros(4, 32, dtype=torch.bfloat16), weight
    )
    assert out.shape == (4, 64) and out.dtype is torch.bfloat16
    assert calls == [((4, 32), (64, 32), (64,))]

    # A 3-D activation is flattened for the kernel and reshaped back.
    out = gemv_on.rowwise_fp8_lm_head_logits(
        torch.zeros(2, 3, 32, dtype=torch.bfloat16), weight
    )
    assert out.shape == (2, 3, 64)
    assert calls[-1] == ((6, 32), (64, 32), (64,))

    # Past the row budget the reviewed low-row / chunked path runs instead, on
    # the same resident FP8 bytes.
    out = gemv_on.rowwise_fp8_lm_head_logits(
        torch.zeros(33, 32, dtype=torch.bfloat16), weight
    )
    assert out.shape == (33, 64)
    assert len(calls) == 2


def _dense_layer(rows: int = 64, columns: int = 32):
    """A post-processed resident FP8 linear: weight [K, N], one scale per N."""
    try:
        method = fp8.rowwise_fp8_method()
    except Exception:
        pytest.skip("the serving runtime's FP8 stack is unavailable")
    layer = SimpleNamespace(
        weight=nn.Parameter(
            torch.zeros(rows, columns, dtype=torch.float8_e4m3fn).t(),
            requires_grad=False,
        ),
        weight_scale=nn.Parameter(
            torch.ones(1, rows, dtype=torch.float32), requires_grad=False
        ),
        input_scale=None,
        bias=None,
        block_fp8_mxfp8_ready=False,
        output_size_per_partition=rows,
        input_size_per_partition=columns,
        logical_widths=[rows],
    )
    return method, layer


def test_dense_consumer_gate_and_fallback_on_the_resident_bytes(gemv_on, monkeypatch):
    method, layer = _dense_layer()
    assert layer.weight.dtype is torch.float8_e4m3fn
    # The transposed [K, N] view the FP8 method stores must come back as an
    # [N, K] K-contiguous operand for the kernel.
    assert layer.weight.t().stride(1) == 1
    calls = []
    seen = []

    def fake_gemv(x, w, scale, cfg=None):
        calls.append((tuple(x.shape), tuple(w.stride())))
        return torch.zeros((x.shape[0], w.shape[0]), dtype=torch.bfloat16)

    def fake_apply_fp8_linear(**kwargs):
        seen.append(kwargs)
        return torch.zeros(
            (kwargs["input"].shape[0], kwargs["weight"].shape[1]),
            dtype=torch.bfloat16,
        )

    monkeypatch.setattr(w8a16, "w8a16_gemv", fake_gemv)
    import sglang.srt.layers.quantization.fp8 as fp8_module

    monkeypatch.setattr(fp8_module, "apply_fp8_linear", fake_apply_fp8_linear)

    for rows in (1, 4, 16):
        assert method._w8a16_gemv_ok(layer, torch.zeros(rows, 32, dtype=torch.bfloat16))
        method.apply(layer, torch.zeros(rows, 32, dtype=torch.bfloat16))
    assert calls == [((1, 32), (32, 1)), ((4, 32), (32, 1)), ((16, 32), (32, 1))]

    # Outside the contract: the FP8 GEMM dispatch on the SAME resident bytes,
    # never a dequantized BF16 copy and never a GEMV past its row budget.
    for x in (
        torch.zeros(24, 32, dtype=torch.bfloat16),
        torch.zeros(33, 32, dtype=torch.bfloat16),
        torch.zeros(4, 32),
        (torch.zeros(4, 32, dtype=torch.float8_e4m3fn), torch.ones(4, 1)),
    ):
        assert method._w8a16_gemv_ok(layer, x) is False
        method.apply(layer, x)
    assert len(calls) == 3
    assert len(seen) == 4
    assert all(kwargs["weight"] is layer.weight for kwargs in seen)
    assert all(kwargs["weight"].dtype is torch.float8_e4m3fn for kwargs in seen)

    # A bias cannot ride the fused-norm path, and a CPU operand cannot ride the
    # fused norm at all: the caller keeps its separate norm kernel.
    assert (
        method.apply_norm_gated(
            layer,
            torch.zeros(4, 32, dtype=torch.bfloat16),
            torch.zeros(4, 32, dtype=torch.bfloat16),
            torch.ones(16, dtype=torch.bfloat16),
            16,
            1e-6,
        )
        is None
    )
    assert (
        method.apply_norm_gated(
            layer,
            torch.zeros(4, 32, dtype=torch.bfloat16),
            torch.zeros(4, 32, dtype=torch.bfloat16),
            torch.ones(16, dtype=torch.bfloat16),
            16,
            1e-6,
            bias=torch.zeros(64),
        )
        is None
    )


def test_fused_norm_guard_requires_a_cuda_operand_and_an_exact_group_grid(gemv_on):
    def args(x=None, z=None, group=16, norm_weight=None, w=None, scale=None):
        return (
            x if x is not None else torch.zeros(4, 32, dtype=torch.bfloat16),
            w if w is not None else torch.zeros(64, 32, dtype=torch.float8_e4m3fn),
            scale if scale is not None else torch.ones(64, dtype=torch.float32),
            z if z is not None else torch.zeros(4, 32, dtype=torch.bfloat16),
            norm_weight
            if norm_weight is not None
            else torch.ones(group, dtype=torch.bfloat16),
            group,
        )

    # CPU operands: the kernel cannot run, and there is no CPU fallback.
    assert w8a16.w8a16_gemv_norm_gated_supported(*args()) is False
    # K must be a multiple of the norm group, and the group weight must be one
    # contiguous vector of exactly that width.
    assert w8a16.w8a16_gemv_norm_gated_supported(*args(group=12)) is False
    assert (
        w8a16.w8a16_gemv_norm_gated_supported(
            *args(norm_weight=torch.ones(16, dtype=torch.bfloat16)[::2])
        )
        is False
    )
    # An x/z shape mismatch is a padded or broadcast operand, not one grid.
    assert (
        w8a16.w8a16_gemv_norm_gated_supported(
            *args(z=torch.zeros(2, 32, dtype=torch.bfloat16))
        )
        is False
    )
    # The gate itself: bundle on, explicit off wins.
    assert w8a16.norm_into_gemv_enabled() is True
    with envs.SGLANG_NORM_INTO_GEMV.override(False):
        assert w8a16.norm_into_gemv_enabled() is False


def test_split_scratch_is_preallocated_reused_and_never_built_in_a_capture(gemv_on):
    import unittest.mock as mock

    w8a16._WS.clear()
    try:
        device = torch.device("cpu")
        fp8._prealloc_w8a16_gemv_scratch(device)
        ws0, counters0 = w8a16._workspace_slot(device, 0)
        ws1, counters1 = w8a16._workspace_slot(device, 1)
        assert ws0.data_ptr() != ws1.data_ptr()
        assert ws0.dtype is torch.float32 and ws0.numel() == w8a16._WS_FLOATS
        assert counters0.dtype is torch.int32
        assert counters0.numel() == w8a16._WS_COUNTERS
        # The fixup CTA resets its own counter, so nothing may be dirty at the
        # start of the first launch.
        assert int(counters0.abs().sum()) == 0 and int(counters1.abs().sum()) == 0
        # Reused, never re-allocated: a captured launch keeps these addresses.
        assert w8a16._workspace_slot(device, 0)[0] is ws0
        with w8a16.scratch_slot(1):
            assert w8a16._workspace(device)[0] is ws1
        assert w8a16._workspace(device)[0] is ws0
        assert w8a16._ONES[device].item() == 1.0

        # Allocating during a capture would tie a lifetime-long buffer to one
        # graph's private pool, so the port raises instead.
        with mock.patch.object(torch.cuda, "is_current_stream_capturing", lambda: True):
            with pytest.raises(RuntimeError, match="before CUDA graph capture"):
                w8a16._workspace_slot(torch.device("cuda:0"), 0)
    finally:
        w8a16._WS.clear()


def test_weight_installation_hooks_materialize_the_scratch(gemv_on, monkeypatch):
    calls = []
    monkeypatch.setattr(
        fp8, "_prealloc_w8a16_gemv_scratch", lambda device: calls.append(device)
    )
    linear = nn.Linear(32, 64, bias=False, dtype=torch.bfloat16)
    fp8.replace_linear_weight_rowwise_fp8(linear)
    assert calls == [linear.weight.device], "head conversion is the last hook"

    # A repeat conversion is a no-op and does not touch the scratch again.
    calls.clear()
    assert fp8.replace_linear_weight_rowwise_fp8(linear) == 0
    assert calls == []

    # A draft adopting a hot-token slice of that head is also pre-capture.
    calls.clear()
    selected = fp8.select_rowwise_weight_rows(
        linear.weight, torch.arange(0, 8, dtype=torch.long)
    )
    assert calls == [linear.weight.device]
    assert fp8.rowwise_scale_of(selected).shape == (8,)
    assert torch.equal(
        fp8.rowwise_scale_of(selected), fp8.rowwise_scale_of(linear.weight)[:8]
    ), "values and the row scales that give them meaning move as one unit"
    # And the converted weight is still exactly one representation.
    assert selected.dtype is torch.float8_e4m3fn
    assert fp8.rowwise_scale_of(linear.weight) is fp8.rowwise_scale_of(linear.weight)


def test_scratch_is_left_alone_when_the_gemv_is_off(monkeypatch):
    monkeypatch.delenv(GEMV_ENV, raising=False)
    monkeypatch.setitem(w8a16._WS, torch.device("cpu"), ("sentinel", "sentinel"))
    try:
        fp8._prealloc_w8a16_gemv_scratch(torch.device("cpu"))
        assert w8a16._WS[torch.device("cpu")] == ("sentinel", "sentinel")
    finally:
        w8a16._WS.pop(torch.device("cpu"), None)


# --------------------------------------------------------------------------- #
# Draft MTP entry fusion
# --------------------------------------------------------------------------- #


def _mtp_args(tokens: int = 4, hidden: int = 32, hc: int = 2, dtype=torch.bfloat16):
    embeds = torch.zeros(tokens, hidden, dtype=dtype)
    weight = torch.zeros(hidden, hidden, dtype=dtype)
    rows = torch.zeros(tokens * hc, hidden, dtype=dtype)
    return embeds, weight, rows, weight.clone()


def test_mtp_fc_gemv_declines_cpu_and_mixed_dtype_and_oversized_rows():
    fp8.configure_online_fp8(
        None, cuda_available=True, capability=(12, 0), model_eligible=True
    )
    with envs.SGLANG_MTP_FC_GEMV.override(True):
        # CPU tensors: bf16_gemv has no CPU path at all.
        assert mtp_module._mtp_fc_gemv_supported(*_mtp_args()) is False
        # A BF16 embedding side with an FP32 hidden side keeps cuBLAS.
        embed, weight, rows, hidden_weight = _mtp_args()
        assert (
            mtp_module._mtp_fc_gemv_supported(
                embed, weight, rows, hidden_weight.float()
            )
            is False
        )
        assert (
            mtp_module._mtp_fc_gemv_supported(embed.float(), weight, rows, weight)
            is False
        )
        # A 3-D activation is not the kernel's contract either.
        assert (
            mtp_module._mtp_fc_gemv_supported(
                embed.unsqueeze(0), weight, rows, hidden_weight
            )
            is False
        )
        # rows = tokens * hc, so this bounds both GEMVs by the donor's 16.
        assert (
            mtp_module._mtp_fc_gemv_supported(
                *_mtp_args(tokens=8, hc=3)  # 24 rows, the C6 verification width
            )
            is False
        )
        # A column-major weight is not the layout the tiles were measured on.
        column_major = weight.t().contiguous().t()
        assert (
            mtp_module._mtp_fc_gemv_supported(embed, column_major, rows, weight)
            is False
        )


def test_mtp_fc_gemv_off_with_the_bundle_and_honoring_a_saved_explicit_off():
    # Bundle resolved off: the unset companion stays off even on an accepted
    # device, and so does an explicit false on one that is on.
    import unittest.mock as mock

    with mock.patch.object(mtp_module, "_bf16_entry_gemm_target_ok", lambda *t: True):
        fp8.configure_online_fp8(False, cuda_available=True, capability=(12, 0))
        with envs.SGLANG_MTP_FC_GEMV.override(False):
            assert mtp_module._mtp_fc_gemv_supported(*_mtp_args()) is False
        fp8.configure_online_fp8(
            None, cuda_available=True, capability=(12, 0), model_eligible=True
        )
        with envs.SGLANG_MTP_FC_GEMV.override(False):
            assert mtp_module._mtp_fc_gemv_supported(*_mtp_args()) is False


# --------------------------------------------------------------------------- #
# Packed-key softmax router
# --------------------------------------------------------------------------- #


def test_router_covered_matches_the_triton_router_s_own_tiling():
    bias = torch.zeros(512, dtype=torch.float32)
    assert rfast.covered(torch.zeros(4, 512, dtype=torch.bfloat16), bias, 10)
    assert rfast.covered(torch.zeros(1, 512, dtype=torch.bfloat16), bias, 1)
    # fp32 logits, a wider row than one warp, a bias that is not the fp32 zeros,
    # a mismatched expert count and a bad k are all not this call.
    assert not rfast.covered(torch.zeros(4, 512, dtype=torch.float32), bias, 10)
    assert not rfast.covered(
        torch.zeros(4, 1024, dtype=torch.bfloat16),
        torch.zeros(1024, dtype=torch.float32),
        10,
    )
    assert not rfast.covered(
        torch.zeros(4, 512, dtype=torch.bfloat16),
        torch.zeros(512, dtype=torch.bfloat16),
        10,
    )
    assert not rfast.covered(
        torch.zeros(4, 512, dtype=torch.bfloat16),
        torch.zeros(256, dtype=torch.float32),
        10,
    )
    assert not rfast.covered(torch.zeros(4, 512, dtype=torch.bfloat16), bias, 0)
    assert not rfast.covered(torch.zeros(4, 512, dtype=torch.bfloat16), bias, 513)
    assert not rfast.covered(torch.zeros(512, dtype=torch.bfloat16), bias, 10)
    # Below BLOCK_N == 256 the Triton router packs BLOCK_M > 1 rows per program,
    # so its axis-1 reductions no longer share this kernel's layout: out.  Above
    # 512 experts it spends more than one warp per row: also out.
    small = torch.zeros(256, dtype=torch.float32)
    assert not rfast.covered(torch.zeros(4, 128, dtype=torch.bfloat16), small, 10)
    assert rfast.covered(
        torch.zeros(4, 300, dtype=torch.bfloat16),
        torch.zeros(300, dtype=torch.float32),
        10,
    ), "300 experts still rounds up to BLOCK_N 512, one warp per row"
    assert rfast.covered(
        torch.zeros(4, 200, dtype=torch.bfloat16),
        torch.zeros(200, dtype=torch.float32),
        10,
    ), "BLOCK_N 256 is the router's own row-pack boundary"
    wide = torch.zeros(600, dtype=torch.float32)
    assert not rfast.covered(torch.zeros(4, 600, dtype=torch.bfloat16), wide, 10)


def test_router_zero_bias_is_cached_and_capture_safe():
    import unittest.mock as mock

    scores = torch.zeros(4, 512, dtype=torch.bfloat16)
    bias = rfast.get_zero_bias(scores)
    assert bias is not None and bias.dtype is torch.float32 and bias.shape == (512,)
    assert rfast.get_zero_bias(scores) is bias, "no per-call allocation or memset"
    rfast._ZERO_BIAS.clear()

    class _CudaScores:
        # Only .size(1) and .device are read; no CUDA context is needed to ask.
        device = torch.device("cuda:0")

        def size(self, dim):
            return 512

    with mock.patch.object(torch.cuda, "is_current_stream_capturing", lambda: True):
        # Under a capture the zeros must not come from the graph's private pool,
        # so the caller keeps the Triton router for this graph.
        assert rfast.get_zero_bias(_CudaScores()) is None
        assert rfast._ZERO_BIAS == {}
        # A CPU router call is never captured, so it needs no such guard.
        assert rfast.get_zero_bias(scores) is not None


# --------------------------------------------------------------------------- #
# Draft (NEXTN) NVFP4 MoE GEMV
# --------------------------------------------------------------------------- #

H, I, TOPK = 2560, 640, 10


def _moe_layer(**overrides):
    base = dict(
        moe_runner_config=SimpleNamespace(
            activation="silu",
            is_gated=True,
            apply_router_weight_on_input=False,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            swiglu_limit=None,
        ),
        moe_ep_size=1,
        moe_tp_size=1,
        should_fuse_routed_scaling_factor_in_topk=True,
        w13_weight=torch.empty((2, 2 * I, H // 2), dtype=torch.uint8),
        _nvfp4_layout_slot=None,
        _cutlass_swiglu_params=(None, None, None),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_draft_moe_shape_contract_is_the_swizzle_grid_not_a_guess():
    ok = draft.unsupported_reason(
        hidden_size=H, intermediate_size=I, cfg=draft.DEFAULT_CONFIG
    )
    assert ok is None
    for hidden, inter, why in (
        (2500, I, "multiples of 128"),
        (H, 600, "multiples of 128"),
    ):
        reason = draft.unsupported_reason(
            hidden_size=hidden, intermediate_size=inter, cfg=draft.DEFAULT_CONFIG
        )
        assert why in (reason or ""), (hidden, inter, reason)
    # A tile that cannot map onto the 128x4 swizzle is rejected too.
    odd = draft.DraftMoeGemvConfig(
        block_n1=12,
        block_k1=512,
        warps1=4,
        stages1=2,
        block_n2=16,
        block_k2=128,
        splits2=10,
        warps2=2,
        stages2=3,
    )
    assert "power of two" in (
        draft.unsupported_reason(hidden_size=H, intermediate_size=I, cfg=odd) or ""
    )
    tight = draft.DraftMoeGemvConfig(
        block_n1=8,
        block_k1=768,
        warps1=4,
        stages1=2,
        block_n2=16,
        block_k2=128,
        splits2=10,
        warps2=2,
        stages2=3,
    )
    assert "must divide" in (
        draft.unsupported_reason(hidden_size=H, intermediate_size=I, cfg=tight) or ""
    )


def test_draft_moe_layer_guards():
    assert draft._layer_unsupported_reason(_moe_layer()) is None
    layer = _moe_layer()
    layer.moe_runner_config.activation = "gelu"
    assert draft._layer_unsupported_reason(layer) == "needs gated silu"

    for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit", "swiglu_limit"):
        layer = _moe_layer()
        setattr(layer.moe_runner_config, name, 7.0)
        assert "plain silu(gate) * up" in draft._layer_unsupported_reason(layer)
    layer = _moe_layer()
    layer.moe_runner_config.apply_router_weight_on_input = True
    assert "plain silu(gate) * up" in draft._layer_unsupported_reason(layer)
    for key, why in (
        (dict(moe_tp_size=2), "moe_ep_size == moe_tp_size == 1"),
        (dict(moe_ep_size=2), "moe_ep_size == moe_tp_size == 1"),
        # This fork's CUTLASS quant info applies no factor of its own, so the
        # top-k weights must already carry the routed scaling factor.
        (
            dict(should_fuse_routed_scaling_factor_in_topk=False),
            "routed scaling factor folded into the top-k weights",
        ),
    ):
        assert why in draft._layer_unsupported_reason(_moe_layer(**key))
    assert draft.enable_draft_moe_gemv(nn.Linear(2, 2)) is False


def test_draft_moe_rejects_this_fork_s_extra_layer_layouts_per_call():
    class _Pair:
        is_pair = True
        pair = [None]

    assert draft._layer_layout_unsupported_reason(_moe_layer()) is None
    assert "paired W2" in (
        draft._layer_layout_unsupported_reason(
            _moe_layer(_nvfp4_layout_slot=(_Pair(), 0))
        )
        or ""
    )
    assert "SwiGLU" in (
        draft._layer_layout_unsupported_reason(
            _moe_layer(_cutlass_swiglu_params=(1.0, 1.0, None))
        )
        or ""
    )


def _dispatch(x, ids, wts, scale=None):
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    return StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=scale,
        topk_output=StandardTopKOutput(
            topk_weights=wts, topk_ids=ids, router_logits=None
        ),
    )


def test_draft_moe_only_selects_a_one_token_bf16_call():
    ids = torch.arange(TOPK, dtype=torch.int32).unsqueeze(0)
    wts = torch.full((1, TOPK), 1.0 / TOPK, dtype=torch.float32)
    x = torch.zeros(1, H, dtype=torch.bfloat16)
    assert draft._is_one_token_call(_dispatch(x, ids, wts)) is True
    # An FP4-dispatched activation carries its own scale, which this kernel does
    # not read: it must stay on the runner.
    assert (
        draft._is_one_token_call(_dispatch(x, ids, wts, scale=torch.ones(1, 1)))
        is False
    )
    assert draft._is_one_token_call(_dispatch(x.float(), ids, wts)) is False
    # A copy is cheaper than a strided read inside the kernel.
    strided = torch.zeros(1, 2 * H, dtype=torch.bfloat16)[:, ::2]
    assert not strided.is_contiguous()
    assert draft._is_one_token_call(_dispatch(strided, ids, wts)) is False
    wide_ids = torch.arange(2 * TOPK, dtype=torch.int32).unsqueeze(0)
    assert draft._is_one_token_call(_dispatch(x, wide_ids[:, ::2], wts)) is False
    wide_wts = torch.full((1, 2 * TOPK), 1.0 / TOPK, dtype=torch.float32)
    assert draft._is_one_token_call(_dispatch(x, ids, wide_wts[:, ::2])) is False
    # A multi-token call is a different shape of problem entirely.
    four_ids = ids.expand(4, TOPK).contiguous()
    assert (
        draft._is_one_token_call(
            _dispatch(
                x.expand(4, H).contiguous(), four_ids, wts.expand(4, TOPK).contiguous()
            )
        )
        is False
    )
    # And the runner itself refuses anything but the standard dispatcher, and a
    # layer whose layout it cannot read, before touching the scratch.
    runner = draft.DraftMoeGemvRunner()
    assert runner.maybe_apply(layer=_moe_layer(), dispatch_output=object()) is None
    assert (
        runner.maybe_apply(
            layer=_moe_layer(), dispatch_output=_dispatch(x.float(), ids, wts)
        )
        is None
    )
    assert (
        runner.maybe_apply(
            layer=_moe_layer(_cutlass_swiglu_params=(None, None, 7.0)),
            dispatch_output=_dispatch(x, ids, wts),
        )
        is None
    )
    # (The routed-scaling-factor rule is an ENABLE-time guard, not a per-call
    # one: a layer's top-k folding policy cannot change between steps. It is
    # pinned in test_draft_moe_layer_guards, and the workspace allocation that
    # would follow these refusals needs a real device.)


def test_draft_moe_wiring_is_gated_by_the_bundle(monkeypatch):
    fp8.configure_online_fp8(
        None, cuda_available=True, capability=(12, 0), model_eligible=True
    )
    monkeypatch.delenv(DRAFT_ENV, raising=False)
    assert fp8.gated_by_fast_paths(envs.SGLANG_OPT_DRAFT_MOE_GEMV.get()) is True
    monkeypatch.setenv(DRAFT_ENV, "0")
    assert fp8.gated_by_fast_paths(envs.SGLANG_OPT_DRAFT_MOE_GEMV.get()) is False
    fp8.configure_online_fp8(False, cuda_available=True, capability=(12, 0))
    monkeypatch.delenv(DRAFT_ENV, raising=False)
    assert fp8.gated_by_fast_paths(envs.SGLANG_OPT_DRAFT_MOE_GEMV.get()) is False


# --------------------------------------------------------------------------- #
# The A8W8 fallback's oracle, checkable without a GPU
# --------------------------------------------------------------------------- #


def _load_gpu_coverage_module():
    """The SM120 kernel file, for the reference helpers its GPU cases use.

    Imported by path because the test tree is not a package.  A missing file is
    the missing port; a missing third-party dependency is a worker that cannot
    run GPU coverage at all.
    """
    path = (
        Path(__file__)
        .resolve()
        .parents[5]
        .joinpath("test/registered/kernels/ops/gemm/test_sm120_small_batch_gemv.py")
    )
    if not path.exists():
        pytest.fail(
            f"{path.name} is missing: the accepted fast-path port is not present"
        )
    try:
        spec = importlib.util.spec_from_file_location("sm120_gemv_tests", path)
        module = importlib.util.module_from_spec(spec)
        # Registered before execution: inspect.getsource resolves a class back to
        # its file through sys.modules, and the coverage checks below read source.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except ModuleNotFoundError as err:
        sys.modules.pop(spec.name, None)
        pytest.skip(f"GPU coverage needs an optional dependency ({err.name})")
    return module


def test_the_w8a8_fallback_oracle_reads_both_scale_axes_independently():
    """The GPU fallback compares against test code, so that code needs a check.

    A reference that transposed an operand or broadcast the wrong scale axis
    would silently misjudge production on hardware and no selection test here
    would notice.  The two scale vectors are deliberately different, N != K, and
    the resident view is [K, N], so a swapped axis, a swapped transpose or a
    dropped channel scale cannot reproduce these hand-computed numbers.
    """
    tests = _load_gpu_coverage_module()

    class _Layer:
        pass

    layer = _Layer()
    # 2 output channels x 4 inputs, stored the way the conversion stores it.
    codes_nk = torch.tensor(
        [[1, 2, 4, 8], [16, 32, 64, 112]], dtype=torch.float8_e4m3fn
    )
    layer.weight = nn.Parameter(codes_nk.t().contiguous(), requires_grad=False)
    layer.weight_scale = torch.tensor([0.5, 4.0], dtype=torch.float32)
    q_a = torch.ones(2, 4, dtype=torch.float8_e4m3fn)
    a_scale = torch.tensor([[0.5], [2.0]], dtype=torch.float32)

    # Dequantized A rows are [0.5]*4 and [2]*4; dequantized W rows are
    # [0.5, 1, 2, 4] and [64, 128, 256, 448], so the dots are 0.5*7.5, 0.5*896,
    # 2*7.5 and 2*896 -- all exactly representable in BF16.  The two scale
    # vectors share no value, so applying one in the other's axis fails too.
    expected = torch.tensor([[3.75, 448.0], [15.0, 1792.0]], dtype=torch.bfloat16)
    assert torch.equal(tests._fp64_gemm_reference(q_a, a_scale, layer), expected)

    # And the BF16-A oracle that judges the W8A16 path agrees on the same resident
    # bytes and channel scales, for an activation that needs no quantizing.
    assert torch.equal(
        tests._dense_reference(
            layer, torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.bfloat16)
        ),
        expected[:1],
    )


def test_the_row_parallel_fake_carries_every_field_the_fused_caller_reads():
    """The fused-norm caller reads shard metadata straight off the out_proj.

    It does so with plain attribute access -- deliberately, so a layer it cannot
    fully reason about raises instead of quietly fusing through a half-known
    layout -- which means a test fake that omits one field fails on hardware
    rather than declining.  Pinned statically, so that class of mistake is CPU
    evidence: every field the consumer reads must be one the fixture assigns the
    way ``RowParallelLinear.__init__`` does.
    """
    import inspect
    import re

    qwen3_5 = _require_port("sglang.srt.models.qwen3_5")
    tests = _load_gpu_coverage_module()

    def code_of(obj):
        """Source with comments dropped, so a mention is not an assignment."""
        return "".join(
            "" if line.lstrip().startswith("#") else line.split("#", 1)[0] + "\n"
            for line in inspect.getsource(obj).splitlines(keepends=True)
        )

    consumer = code_of(qwen3_5.Qwen3_5GatedDeltaNet._norm_out_proj_fused)
    reads = set(re.findall(r"self\.out_proj\.(\w+)", consumer)) | set(
        re.findall(r'getattr\(self\.out_proj,\s*"(\w+)"', consumer)
    )
    assigned = set(re.findall(r"self\.(\w+)\s*=", code_of(tests._Projection)))
    assert reads, "the consumer stopped reading out_proj metadata"
    assert not reads - assigned, f"fake out_proj is missing {sorted(reads - assigned)}"
    # The refusal coverage is what keeps the guards honest, so the fake has to
    # declare the biased/sharded cases it is asked about, not just the happy one.
    assert {"bias", "use_decode_attn_tp"} <= assigned


def test_the_fused_norm_caller_decides_everything_it_can_on_cpu(monkeypatch):
    """The fused/fallback decision, up to the line that launches a kernel.

    Every guard ``_norm_out_proj_fused`` applies before the GEMV prologue is an
    attribute or shape test, so the refusals and the ``[tokens, num_v_heads x
    head_v_dim]`` reading are checkable without hardware -- the three SM120 cases
    that failed on a fake's missing metadata fail here first now.  Only the fused
    kernel's arithmetic is left to the GPU file.
    """
    qwen3_5 = _require_port("sglang.srt.models.qwen3_5")
    monkeypatch.setattr(qwen3_5, "_is_cuda", True)
    g, heads, tokens = 8, 4, 2

    class _Method:
        def __init__(self):
            self.calls = []

        def apply_norm_gated(self, *args):
            self.calls.append(args)
            return "fused"

    method = _Method()
    layer = SimpleNamespace(
        quant_method=method,
        input_size=g * heads,
        input_size_per_partition=g * heads,
        output_size=16,
        output_size_per_partition=16,
        logical_widths=[16],
        with_bias=False,
        bias=None,
        input_is_parallel=False,
        reduce_results=True,
        use_decode_attn_tp=False,
    )
    norm = SimpleNamespace(
        group_size=None,
        norm_before_gate=True,
        bias=None,
        weight=torch.ones(g),
        eps=1e-6,
        activation="swish",
    )
    fake = SimpleNamespace(head_v_dim=g, out_proj=layer, norm=norm)
    x = torch.zeros(tokens * heads, g, dtype=torch.bfloat16)
    shape = (tokens, heads, g)
    monkeypatch.setenv("SGLANG_NORM_INTO_GEMV", "1")

    fused = qwen3_5.Qwen3_5GatedDeltaNet._norm_out_proj_fused(fake, x, x, shape)
    assert fused == "fused", "a faithful single-rank biased-free out_proj fuses"
    _, a, z, norm_weight, group, eps, sigmoid = method.calls[0]
    # Both operands are re-viewed as one GEMV row per token, and one norm group
    # per contiguous head_v_dim slice of it.
    assert tuple(a.shape) == tuple(z.shape) == (tokens, heads * g)
    assert group == g and norm_weight.shape == (g,) and eps == 1e-6
    assert sigmoid is False

    def declines(target, **overrides):
        previous = {name: getattr(target, name, None) for name in overrides}
        try:
            for name, value in overrides.items():
                setattr(target, name, value)
            before = len(method.calls)
            assert (
                qwen3_5.Qwen3_5GatedDeltaNet._norm_out_proj_fused(fake, x, x, shape)
                is None
            ), overrides
            assert len(method.calls) == before, "a declined call reached the kernel"
        finally:
            for name, value in previous.items():
                setattr(target, name, value)

    declines(layer, input_size=g * heads * 2)  # a row-parallel shard: partial dot
    declines(layer, use_decode_attn_tp=True)
    declines(layer, bias=torch.zeros(1))
    declines(norm, group_size=g)
    declines(norm, norm_before_gate=False)
    declines(norm, activation="gelu")
    declines(norm, weight=torch.ones(g, 1))
    assert (
        qwen3_5.Qwen3_5GatedDeltaNet._norm_out_proj_fused(
            fake, x, x, (tokens, heads, g, 1)
        )
        is None
    )
    assert (
        qwen3_5.Qwen3_5GatedDeltaNet._norm_out_proj_fused(
            fake,
            torch.zeros(tokens * heads + 8, g, dtype=torch.bfloat16),
            torch.zeros(tokens * heads + 8, g, dtype=torch.bfloat16),
            shape,
        )
        is None
    ), "a padded or strided z breaks the one-group-per-row reading"

    norm.activation = "sigmoid"
    qwen3_5.Qwen3_5GatedDeltaNet._norm_out_proj_fused(fake, x, x, shape)
    assert method.calls[-1][-1] is True
    with envs.SGLANG_NORM_INTO_GEMV.override(False):
        assert (
            qwen3_5.Qwen3_5GatedDeltaNet._norm_out_proj_fused(fake, x, x, shape) is None
        ), "the switch is re-read per call"


# --------------------------------------------------------------------------- #
# Donor tile table / scratch limits the kernels were measured with
# --------------------------------------------------------------------------- #


def test_plan_keeps_the_donor_head_tiles_and_the_scratch_limit():
    sms = 188
    # The donor's measured draft-head tile, and the wide-N fallback the full
    # target head shares with it.
    assert w8a16._plan(1, 32768, 2560, False, 1, sms) == (32, 256, 1, False, None, 4, 3)
    assert w8a16._plan(4, 248320, 2560, False, 1, sms) == (
        128,
        256,
        1,
        True,
        None,
        8,
        3,
    )
    # A TP2 shard of the head is not in the table and must not split past the
    # counter array.
    assert w8a16._plan(4, 124160, 2560, False, 1, sms) == (32, 128, 1, True, None, 4, 3)
    # A split-K plan wider than the scratch degrades to one split rather than
    # writing past the buffer a captured launch points at.
    assert w8a16._fit(1, 248320, (16, 128, 10, False, None, 4, 3))[2] == 1
    assert w8a16._fit(1, 1024, (16, 128, 10, False, None, 4, 3))[2] == 10
    # The fused-norm plan keeps BLOCK_K at the norm group, which is the whole
    # reason the statistics can be recomputed inside the A tile.
    assert w8a16._norm_plan(4, 2560, 6144, 128, 1, sms)[1] == 128
    assert w8a16._norm_plan(16, 2560, 6144, 128, 1, sms)[1] == 128
