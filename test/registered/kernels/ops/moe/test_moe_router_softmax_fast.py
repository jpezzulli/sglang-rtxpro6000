# SPDX-License-Identifier: Apache-2.0

"""Focused checks for the packed-key softmax router (SGLANG_ROUTER_FAST_TOPK).

The donor claim is that this router returns the Triton router's
(``moe_fused_gate(..., scoring_func="softmax")``) weights and ids BIT FOR BIT for
the one call ``fused_topk`` makes on CUDA: bf16 logits, no correction bias (the
Triton router keeps its zeros in registers, this kernel reads a cached fp32 zero
vector -- the same value), no shared experts, no groups, no softcap or scaling,
at most 512 experts, one warp per row.  A tolerance would accept exactly the
errors this kernel exists to catch, so the GPU checks compare ids and weight bit
patterns.

``covered()`` is what keeps every other call on the Triton router, so its rules
are pinned on the CPU worker too, and the dispatch itself is pinned through
``fused_topk`` rather than by calling the kernel directly.

One documented exception to the bit-identity, kept as the donor wrote it: for a
row with more lanes at -inf than there are picks, the Triton router re-picks its
own lowest masked lane (its mask value and a real -inf logit are
indistinguishable) while the packed keys move on to the next lane.  bf16 router
logits do not produce -inf rows.
"""

from __future__ import annotations

import pytest
import torch

from sglang.kernels.ops.moe import moe_router_softmax_fast as rfast
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# The Flash-Next router shape (512 experts, top-10).
N, TOPK = 512, 10
ENV = "SGLANG_ROUTER_FAST_TOPK"


def _zero_bias(n: int = N, dtype=torch.float32) -> torch.Tensor:
    return torch.zeros(n, dtype=dtype)


def test_covered_matches_the_call_fused_topk_makes() -> None:
    bias = _zero_bias()
    assert rfast.covered(torch.zeros(4, N, dtype=torch.bfloat16), bias, TOPK)
    assert rfast.covered(torch.zeros(1, N, dtype=torch.bfloat16), bias, 1)
    # fp32 logits need the retired int64 key; a non-fp32 bias or a mismatched
    # expert count is not this call.
    assert not rfast.covered(torch.zeros(4, N, dtype=torch.float32), bias, TOPK)
    assert not rfast.covered(
        torch.zeros(4, N + 1, dtype=torch.bfloat16), _zero_bias(N + 1), TOPK
    )
    assert not rfast.covered(
        torch.zeros(4, N, dtype=torch.bfloat16), _zero_bias(N, torch.bfloat16), TOPK
    )
    assert not rfast.covered(
        torch.zeros(4, N, dtype=torch.bfloat16), _zero_bias(N // 2), TOPK
    )
    assert not rfast.covered(torch.zeros(4, N, dtype=torch.bfloat16), bias, 0)
    assert not rfast.covered(torch.zeros(4, N, dtype=torch.bfloat16), bias, N + 1)
    assert not rfast.covered(torch.zeros(N, dtype=torch.bfloat16), bias, TOPK)


def test_covered_stops_where_the_triton_router_changes_its_own_tiling() -> None:
    # moe_fused_gate uses BLOCK_N = next_pow2(N), BLOCK_M = max(1, min(4, 256 //
    # BLOCK_N)) and num_warps = 1 only up to BLOCK_N 512.  Below BLOCK_N 256 it
    # reduces several rows at once, so its fp32 row sum is no longer the same
    # order of additions as this kernel's; above 512 it spends 4 warps per row.
    assert rfast.covered(
        torch.zeros(2, 256, dtype=torch.bfloat16), _zero_bias(256), TOPK
    )
    assert rfast.covered(
        torch.zeros(2, 300, dtype=torch.bfloat16), _zero_bias(300), TOPK
    )
    assert not rfast.covered(
        torch.zeros(2, 128, dtype=torch.bfloat16), _zero_bias(128), TOPK
    )
    assert not rfast.covered(
        torch.zeros(2, 600, dtype=torch.bfloat16), _zero_bias(600), TOPK
    )


def test_zero_bias_is_cached_once_per_width():
    scores = torch.zeros(4, N, dtype=torch.bfloat16)
    bias = rfast.get_zero_bias(scores)
    assert bias is not None and bias.dtype is torch.float32 and bias.shape == (N,)
    assert float(bias.abs().max()) == 0.0
    assert rfast.get_zero_bias(scores) is bias, "no per-call allocation or memset"


_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _assert_bit_identical(fast_w, fast_i, prod_w, prod_i) -> None:
    assert torch.equal(fast_i, prod_i), f"ids differ:\n{fast_i[:4]}\n{prod_i[:4]}"
    nan_both = torch.isnan(fast_w) & torch.isnan(prod_w)
    diff = (fast_w.view(torch.int32) != prod_w.view(torch.int32)) & ~nan_both
    assert not bool(diff.any()), (
        f"weights differ in rows {diff.any(dim=1).nonzero()[:4]}"
    )


def _production_reference(logits: torch.Tensor):
    """Exactly what fused_topk's softmax branch does on CUDA today."""
    return moe_fused_gate(logits, None, TOPK, scoring_func="softmax", renormalize=True)


def _cases(rows: int, device: str):
    yield "randn", torch.randn(rows, N, device=device)
    yield (
        "spiky",
        torch.randn(rows, N, device=device)
        + 30 * (torch.rand(rows, N, device=device) < 0.01).float(),
    )
    yield "ties", torch.randint(-6, 6, (rows, N), device=device).float() * 0.5
    yield "all-equal", torch.full((rows, N), 0.75, device=device)
    zero = torch.zeros(rows, N, device=device)
    half = torch.rand(rows, N, device=device) < 0.5
    yield "zeros(+-0)", torch.where(half, zero, -zero)
    yield (
        "nan",
        torch.where(
            (torch.rand(rows, N, device=device) < 0.02)
            & (torch.rand(rows, 1, device=device) < 0.5),
            torch.full((rows, N), float("nan"), device=device),
            torch.randn(rows, N, device=device),
        ),
    )


@_cuda
@pytest.mark.parametrize("rows", [1, 4, 16, 128])
def test_matches_the_triton_router_bit_for_bit(rows: int) -> None:
    torch.manual_seed(0)
    bias = _zero_bias().to("cuda")
    for name, logits in _cases(rows, "cuda"):
        logits = logits.bfloat16().contiguous()
        prod_w, prod_i = _production_reference(logits)
        fast_w, fast_i = rfast.route_softmax_fast(logits, bias, TOPK, renormalize=True)
        torch.cuda.synchronize()
        try:
            _assert_bit_identical(fast_w, fast_i, prod_w, prod_i)
        except AssertionError as err:  # pragma: no cover - failure path
            raise AssertionError(f"{name} rows={rows}: {err}") from err
        # The zeros the Triton router keeps in registers and the zeros this
        # kernel reads must also agree with each other.
        with_bias_w, with_bias_i = moe_fused_gate(
            logits, bias, TOPK, scoring_func="softmax", renormalize=True
        )
        _assert_bit_identical(fast_w, fast_i, with_bias_w, with_bias_i)


@_cuda
def test_matches_under_cuda_graph_capture() -> None:
    """The decode path launches the router inside a captured graph."""
    torch.manual_seed(0)
    bias = _zero_bias().to("cuda")
    logits = torch.randn(128, N, device="cuda").bfloat16().contiguous()
    prod_w, prod_i = _production_reference(logits)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rfast.route_softmax_fast(logits, bias, TOPK, renormalize=True)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fast_w, fast_i = rfast.route_softmax_fast(logits, bias, TOPK, renormalize=True)
    graph.replay()
    torch.cuda.synchronize()
    _assert_bit_identical(fast_w, fast_i, prod_w, prod_i)

    # Replayed on new logits: the kernel reads its inputs, it does not bake them in.
    logits.copy_(torch.randn(128, N, device="cuda").bfloat16())
    prod_w, prod_i = _production_reference(logits)
    graph.replay()
    torch.cuda.synchronize()
    _assert_bit_identical(fast_w, fast_i, prod_w, prod_i)


@_cuda
@pytest.mark.parametrize("rows", [1, 6, 16, 24])
def test_fused_topk_dispatch_and_fallback(rows: int, monkeypatch) -> None:
    """The selection, through fused_topk: the fast router inside its contract, the
    Triton router everywhere else, and identical results either way."""
    from sglang.srt.layers.moe import topk as topk_module

    logits = torch.randn(rows, N, device="cuda").bfloat16().contiguous()
    hidden = torch.zeros(rows, 8, device="cuda", dtype=torch.bfloat16)

    def route(flag):
        monkeypatch.setenv(ENV, "1" if flag else "0")
        return topk_module.fused_topk(
            hidden_states=hidden,
            gating_output=logits,
            topk=TOPK,
            renormalize=True,
            scoring_func="softmax",
        )

    fast_w, fast_i = route(True)
    prod_w, prod_i = route(False)
    _assert_bit_identical(fast_w, fast_i, prod_w, prod_i)
    assert fast_w.dtype is torch.float32 and fast_i.dtype is torch.int32

    # fp32 logits are not the covered call: the Triton router must still run,
    # and it must still produce its own (unchanged) answer.
    monkeypatch.setenv(ENV, "1")
    fp32 = torch.randn(rows, N, device="cuda")
    got_w, got_i = topk_module.fused_topk(
        hidden_states=hidden,
        gating_output=fp32,
        topk=TOPK,
        renormalize=True,
        scoring_func="softmax",
    )
    ref_w, ref_i = moe_fused_gate(
        fp32, _zero_bias().to("cuda"), TOPK, scoring_func="softmax", renormalize=True
    )
    assert torch.equal(got_i, ref_i) and torch.equal(got_w, ref_w)


@_cuda
def test_fast_router_is_automatic_only_with_the_bundle_and_explicit_off_wins(
    monkeypatch,
) -> None:
    from sglang.kernels.ops.gemm.sm120_online_fp8 import configure_online_fp8
    from sglang.srt.layers.moe import topk as topk_module

    monkeypatch.delenv(ENV, raising=False)
    configure_online_fp8(
        None, cuda_available=True, capability=(12, 0), model_eligible=True
    )
    try:
        assert topk_module._router_fast_topk_enabled() is True
        monkeypatch.setenv(ENV, "0")
        assert topk_module._router_fast_topk_enabled() is False
    finally:
        configure_online_fp8(False, cuda_available=False, capability=None)
    monkeypatch.delenv(ENV, raising=False)
    assert topk_module._router_fast_topk_enabled() is False
