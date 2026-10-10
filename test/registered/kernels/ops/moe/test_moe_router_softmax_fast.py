"""Focused checks for the packed-key softmax router (``SGLANG_ROUTER_FAST_TOPK=1``).

The donor claim is that this router returns the Triton router's
(``moe_fused_gate(..., scoring_func="softmax")``) weights and ids bit for bit for
the one call ``fused_topk`` makes on CUDA: bf16 logits, fp32 zero bias, at most
512 experts, one warp per row. A tolerance would accept exactly the errors this
kernel exists to catch, so the GPU checks compare ids and weight bit patterns.
``covered()`` is what keeps every other call on the Triton router, so its rules
are pinned as well.

One documented exception to the bit-identity: for a row with more lanes at -inf
than there are picks, the Triton router re-picks its own lowest masked lane (its
mask value and a real -inf logit are indistinguishable) while the packed keys
move on to the next lane. Kept as the donor wrote it; bf16 router logits do not
produce -inf rows.
"""

from __future__ import annotations

import pytest
import torch

from sglang.kernels.ops.moe import moe_router_softmax_fast as rfast
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# The Flash-Next router shape (512 experts, top-10; RadixArk/Qwen3.8-Flash-Next-NVFP4).
N, TOPK = 512, 10


def _zeros_bias(n: int = N, dtype=torch.float32) -> torch.Tensor:
    return torch.zeros(n, dtype=dtype)


def test_covered_matches_the_donor_rules() -> None:
    bias = _zeros_bias()
    assert rfast.covered(torch.zeros(4, N, dtype=torch.bfloat16), bias, TOPK)
    assert rfast.covered(torch.zeros(1, N, dtype=torch.bfloat16), bias, 1)
    assert rfast.covered(torch.zeros(3, 5, dtype=torch.bfloat16), _zeros_bias(5), 5)
    # fp32 logits need the retired int64 key, a wider row needs more than one warp,
    # and a non-fp32 bias / mismatched N is not the fused_topk call.
    assert not rfast.covered(torch.zeros(4, N, dtype=torch.float32), bias, TOPK)
    assert not rfast.covered(
        torch.zeros(4, N + 1, dtype=torch.bfloat16), _zeros_bias(N + 1), TOPK
    )
    assert not rfast.covered(
        torch.zeros(4, N, dtype=torch.bfloat16),
        _zeros_bias(N, torch.bfloat16),
        TOPK,
    )
    assert not rfast.covered(
        torch.zeros(4, N, dtype=torch.bfloat16), _zeros_bias(N // 2), TOPK
    )
    assert not rfast.covered(torch.zeros(4, N, dtype=torch.bfloat16), bias, 0)
    assert not rfast.covered(torch.zeros(4, N, dtype=torch.bfloat16), bias, N + 1)
    assert not rfast.covered(torch.zeros(N, dtype=torch.bfloat16), bias, TOPK)


_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _assert_bit_identical(
    fast_w: torch.Tensor,
    fast_i: torch.Tensor,
    prod_w: torch.Tensor,
    prod_i: torch.Tensor,
) -> None:
    assert torch.equal(fast_i, prod_i), f"ids differ:\n{fast_i[:4]}\n{prod_i[:4]}"
    nan_both = torch.isnan(fast_w) & torch.isnan(prod_w)
    diff = (fast_w.view(torch.int32) != prod_w.view(torch.int32)) & ~nan_both
    bad_rows = diff.any(dim=1).nonzero()[:4]
    assert not bool(diff.any()), f"weights differ in rows {bad_rows}"


def _prod(logits: torch.Tensor, bias: torch.Tensor):
    return moe_fused_gate(logits, bias, TOPK, scoring_func="softmax", renormalize=True)


def _cases(rows: int, device: str):
    yield "randn", torch.randn(rows, N, device=device)
    yield "spiky", torch.randn(rows, N, device=device) + 30 * (
        torch.rand(rows, N, device=device) < 0.01
    ).float()
    yield "ties", torch.randint(-6, 6, (rows, N), device=device).float() * 0.5
    yield "all-equal", torch.full((rows, N), 0.75, device=device)
    zero = torch.zeros(rows, N, device=device)
    half = torch.rand(rows, N, device=device) < 0.5
    yield "zeros(+-0)", torch.where(half, zero, -zero)
    yield "nan", torch.where(
        (torch.rand(rows, N, device=device) < 0.02)
        & (torch.rand(rows, 1, device=device) < 0.5),
        torch.full((rows, N), float("nan"), device=device),
        torch.randn(rows, N, device=device),
    )


@_cuda
@pytest.mark.parametrize("rows", [1, 4, 16, 128])
def test_matches_triton_router_bit_for_bit(rows: int) -> None:
    torch.manual_seed(0)
    device, bias = "cuda", _zeros_bias().to("cuda")
    for name, logits in _cases(rows, device):
        logits = logits.bfloat16().contiguous()
        prod_w, prod_i = _prod(logits, bias)
        fast_w, fast_i = rfast.route_softmax_fast(logits, bias, TOPK, renormalize=True)
        torch.cuda.synchronize()
        _assert_bit_identical(fast_w, fast_i, prod_w, prod_i)


@_cuda
def test_matches_under_cuda_graph_capture() -> None:
    """The decode path launches the router inside a captured graph (PDL on)."""
    torch.manual_seed(0)
    device, bias = "cuda", _zeros_bias().to("cuda")
    logits = torch.randn(128, N, device=device).bfloat16().contiguous()
    prod_w, prod_i = _prod(logits, bias)

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

    # replayed on new logits: the kernel reads its inputs, it does not bake them in
    logits.copy_(torch.randn(128, N, device=device).bfloat16())
    prod_w, prod_i = _prod(logits, bias)
    graph.replay()
    torch.cuda.synchronize()
    _assert_bit_identical(fast_w, fast_i, prod_w, prod_i)


@_cuda
def test_opt_in_flag_selects_the_fast_router(monkeypatch) -> None:
    """``SGLANG_ROUTER_FAST_TOPK=1`` routes fused_topk's softmax call to the kernel."""
    from sglang.srt.layers.moe import topk as topk_module

    device, bias = "cuda", _zeros_bias().to("cuda")
    logits = torch.randn(64, N, device=device).bfloat16().contiguous()
    hidden = torch.zeros(logits.shape[0], 8, device=device, dtype=torch.bfloat16)

    def route(flag: bool, gating: torch.Tensor):
        monkeypatch.setattr(topk_module, "_router_fast_topk", flag)
        return topk_module.fused_topk(
            hidden_states=hidden,
            gating_output=gating,
            topk=TOPK,
            renormalize=True,
            scoring_func="softmax",
        )

    fast_w, fast_i = route(True, logits)
    prod_w, prod_i = route(False, logits)
    _assert_bit_identical(fast_w, fast_i, prod_w, prod_i)

    # uncovered logits (fp32) keep the Triton router even with the flag on
    fp32 = torch.randn(64, N, device=device)
    got_w, got_i = route(True, fp32)
    ref_w, ref_i = moe_fused_gate(
        fp32, bias, TOPK, scoring_func="softmax", renormalize=True
    )
    assert torch.equal(got_i, ref_i)
    assert torch.equal(got_w, ref_w)
