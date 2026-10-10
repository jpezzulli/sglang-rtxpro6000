"""Actual-kernel BF16 numerics and CUDA-graph coverage for the draft MTP
entry fusion GEMV (SGLANG_MTP_FC_GEMV) on exact SM120.

Mirrors the routing of ``models/qwen4_exp_mtp._fuse_residual_linear_shared``
at the donor-measured shape (hidden 2560, HC 4): token widths whose
hidden-side row count (tokens * HC) stays inside the donor's 16-row limit
run the Triton GEMV — asserted through a counting spy that still calls the
REAL ``bf16_gemv`` — while the C6 six-token (24-row) and wide-verify widths
fall back to the original two-Linear path (asserted by spy absence).  The
fixture builds the entry fusion with the PRODUCTION
``_init_pre_fc_norms``/``_init_linear_projections`` API (embedding norm
2560, hidden norm hc*hidden=10240) and moves it to CUDA/BF16, buffers
included; the CPU contract of that builder is pinned in
``test/registered/unit/models/test_qwen4_exp_mtp_fc_gemv.py``.  Two eager
warmups precede every capture, exactly like Eagle graph initialization, so
the split-K scratch and the scale-of-one materialize outside the graph
pool.  Nothing here runs without SM120 and no CPU result is treated as
GPU-math proof.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.models.qwen4_exp_mtp import (
    Qwen4ExpForCausalLMMTP,
    _mtp_fc_gemv_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")


def _is_exact_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


pytestmark = pytest.mark.skipif(
    not _is_exact_sm120(), reason="donor dense GEMV is accepted on exactly SM120"
)

HIDDEN_SIZE = 2560
HC_COUNT = 4
DONOR_MAX_M = 16
# 1/4 draft-decode widths (4/16 hidden-side rows, inside the limit), 6 == the
# C6 verification batch (24 rows, must fall back) and 16 wide verify (64 rows).
TOKEN_WIDTHS = [1, 4, 6, 16]

ENV = "SGLANG_MTP_FC_GEMV"


@pytest.fixture(params=[False, True])
def fc_gemv_candidate(request, monkeypatch) -> bool:
    """Run the coverage twice: the original path and the opt-in donor GEMV."""
    if request.param:
        monkeypatch.setenv(ENV, "1")
    else:
        monkeypatch.delenv(ENV, raising=False)
    return bool(request.param)


@pytest.fixture
def gemv_calls() -> list:
    """Count REAL bf16_gemv invocations; never replace the kernel."""
    from sglang.srt.layers.quantization import w8a16_gemv as w8a16_module

    calls: list = []
    real = w8a16_module.bf16_gemv

    def spy(x, w, *args, **kwargs):
        calls.append((tuple(x.shape), tuple(w.shape)))
        return real(x, w, *args, **kwargs)

    with mock.patch.object(w8a16_module, "bf16_gemv", spy):
        yield calls


def _randn(
    shape, *, seed: int, scale: float, device: torch.device | str = "cuda"
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
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


def _entry_config() -> SimpleNamespace:
    # The attributes production reads in _init_pre_fc_norms/_init_linear_projections.
    return SimpleNamespace(
        hidden_size=HIDDEN_SIZE, rms_norm_eps=1e-6, hc_count=HC_COUNT
    )


def build_entry_fusion(
    device: torch.device | str = "cuda", dtype: torch.dtype = torch.bfloat16
) -> SimpleNamespace:
    """Entry fusion built through the production constructor API.

    Uses ``Qwen4ExpForCausalLMMTP._init_pre_fc_norms``/
    ``_init_linear_projections`` verbatim (hidden norm is hc*hidden=10240
    wide), then moves every module -- GemmaRMSNorm's ``gemma_weight``
    buffer included -- to the requested device/dtype, and finally loads
    deterministic nonuniform norm/linear weights through the production
    ``weight.weight_loader`` hook so ``gemma_weight`` stays ``weight + 1``.
    """
    fusion = SimpleNamespace(hc_count=HC_COUNT, hidden_size=HIDDEN_SIZE)
    config = _entry_config()
    Qwen4ExpForCausalLMMTP._init_pre_fc_norms(fusion, config)
    Qwen4ExpForCausalLMMTP._init_linear_projections(fusion, config)
    for module in (
        fusion.pre_fc_norm_embedding,
        fusion.pre_fc_norm_hidden,
        fusion.fc_embedding,
        fusion.fc_hidden,
    ):
        module.to(device=device, dtype=dtype)
    # Deterministic nonuniform norm weights (not all-zero), via the loader
    # that keeps the Gemma buffer contract intact.
    for name, width, seed in (
        ("pre_fc_norm_embedding", HIDDEN_SIZE, 901),
        ("pre_fc_norm_hidden", HC_COUNT * HIDDEN_SIZE, 902),
    ):
        norm = getattr(fusion, name)
        loaded = _randn((width,), seed=seed, scale=0.01, device=device)
        norm.weight.weight_loader(norm.weight, loaded.to(dtype))
    for name, seed in (("fc_embedding", 101), ("fc_hidden", 102)):
        linear = getattr(fusion, name)
        linear.weight.data.copy_(
            _randn(
                (HIDDEN_SIZE, HIDDEN_SIZE),
                seed=seed,
                scale=1.0 / math.sqrt(HIDDEN_SIZE),
                device=device,
            )
        )
    return fusion


@pytest.fixture(scope="module")
def entry_fusion() -> SimpleNamespace:
    return build_entry_fusion(device="cuda", dtype=torch.bfloat16)


def _inputs(tokens: int, seed: int):
    embeds = _randn((tokens, HIDDEN_SIZE), seed=seed, scale=0.25)
    hidden = _randn((tokens, HC_COUNT * HIDDEN_SIZE), seed=seed + 1, scale=0.25)
    return embeds, hidden


def _fused(entry_fusion, embeds: torch.Tensor, hidden: torch.Tensor):
    return Qwen4ExpForCausalLMMTP._fuse_residual_linear_shared(
        entry_fusion, embeds, hidden
    )


@torch.no_grad()
def _reference(entry_fusion: SimpleNamespace, embeds, hidden):
    """The original two-Linear (cuBLAS) path, computed with the flag off."""
    backup = os.environ.pop(ENV, None)
    try:
        return _fused(entry_fusion, embeds, hidden)
    finally:
        if backup is not None:
            os.environ[ENV] = backup


def _expected_calls(tokens: int, candidate: bool) -> list:
    if not candidate or tokens * HC_COUNT > DONOR_MAX_M:
        return []
    return [
        ((tokens, HIDDEN_SIZE), (HIDDEN_SIZE, HIDDEN_SIZE)),
        ((tokens * HC_COUNT, HIDDEN_SIZE), (HIDDEN_SIZE, HIDDEN_SIZE)),
    ]


@pytest.mark.parametrize("tokens", TOKEN_WIDTHS)
def test_entry_fusion_matches_linear_reference(
    entry_fusion: SimpleNamespace,
    tokens: int,
    fc_gemv_candidate: bool,
    gemv_calls: list,
) -> None:
    embeds, hidden = _inputs(tokens, seed=200 + tokens)
    normed = entry_fusion.pre_fc_norm_embedding(embeds)
    rows = (
        entry_fusion.pre_fc_norm_hidden(hidden)
        .view(tokens, HC_COUNT, HIDDEN_SIZE)
        .reshape(-1, HIDDEN_SIZE)
    )
    routed = _mtp_fc_gemv_supported(
        normed,
        entry_fusion.fc_embedding.weight,
        rows,
        entry_fusion.fc_hidden.weight,
    )
    assert routed is (fc_gemv_candidate and 1 <= tokens * HC_COUNT <= DONOR_MAX_M)

    actual = _fused(entry_fusion, embeds, hidden)
    # The spy delegates to the real kernel: widths 1/4 under the flag must
    # actually reach bf16_gemv twice, and 6/16 (or flag off) must not at all.
    assert gemv_calls == _expected_calls(tokens, fc_gemv_candidate)
    # Candidate-vs-reference agreement within the donor's documented
    # low-order-bit spread, not bit-for-bit; no CPU result stands in here.
    _assert_normalized_error(
        actual,
        _reference(entry_fusion, embeds, hidden),
        max_nrmse=0.025,
        min_cosine=0.999,
    )


@pytest.mark.parametrize("tokens", TOKEN_WIDTHS)
def test_entry_fusion_cuda_graph_replays_mutated_input(
    entry_fusion: SimpleNamespace,
    tokens: int,
    fc_gemv_candidate: bool,
    gemv_calls: list,
) -> None:
    embeds, hidden = _inputs(tokens, seed=300 + tokens)
    # Two eager warmups before capture, matching Eagle graph initialization,
    # so scratch/scale-of-one never come from the graph's private pool.
    for _ in range(2):
        _fused(entry_fusion, embeds, hidden)
    torch.cuda.synchronize()
    warmup_calls = list(gemv_calls)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = _fused(entry_fusion, embeds, hidden)
    assert gemv_calls[len(warmup_calls) :] == _expected_calls(
        tokens, fc_gemv_candidate
    ), "capture routing diverged from the eager warm-up routing"

    # Nonzero changed-input replay: mutate the static inputs in place.
    with torch.no_grad():
        embeds.copy_(embeds + _randn(embeds.shape, seed=400 + tokens, scale=0.05))
        hidden.copy_(hidden + _randn(hidden.shape, seed=401 + tokens, scale=0.05))
    graph.replay()
    torch.cuda.synchronize()
    actual = graph_output.clone()

    expected = _reference(entry_fusion, embeds, hidden)
    assert actual.abs().sum() > 0, "graph replay produced an all-zero output"
    _assert_normalized_error(actual, expected, max_nrmse=0.025, min_cosine=0.999)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
