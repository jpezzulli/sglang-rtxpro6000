"""Actual-SM120 numerics and CUDA-graph coverage for the donor HC MIX2 port.

UNRUN ON room101: this host has no inference GPU, so nothing here is a measured
claim. It is the focused check set for the box that has one, written against
this base's own resident rowwise-FP8 weights and the current norm formula.
"""

from __future__ import annotations

import math
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    configure_online_fp8,
    rowwise_scale_of,
)
from sglang.srt.layers.hc_mix2_triton import (
    _HC_MIX2_MAX_ROWS,
    hc_norm_mix2,
    hc_norm_mix2_supported,
)
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")

HC = 4
HS = 2560
K = HC * HS
LOWRANK = 320
EPS = 1e-6
# 1 decode, 4/6/16 inside the kernel's row budget (6 is the real C6 draft
# width, the verify batch the donor tuned this path for), 24 = the C6 W4
# target's verification batch and 33 = prefill, both of which must stay on the
# existing path.
ROWS = [1, 4, 6, 16]
FALLBACK_ROWS = [24, 33]


def _is_exact_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


pytestmark = pytest.mark.skipif(
    not _is_exact_sm120(), reason="HC MIX2 requires exactly SM120"
)


def _randn(
    shape, *, seed: int, scale: float, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(shape, generator=generator, device="cuda", dtype=dtype) * scale


def _normalized_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple:
    actual, expected = actual.float(), expected.float()
    nrmse = (
        (actual - expected).square().mean().sqrt()
        / max(expected.square().mean().sqrt().item(), 1e-8)
    ).item()
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    return nrmse, cosine


def _assert_normed(normed: torch.Tensor, expected: torch.Tensor) -> None:
    # K0 and this reference both compute in fp32 and round once to bf16, so they
    # can land one ulp apart (0.03 at |x|=4) wherever the fp32 value sits near a
    # rounding boundary: judge it relatively, not bitwise, and not against a
    # fixed absolute bound that one bf16 step could legitimately exceed.
    nrmse, cosine = _normalized_error(normed, expected)
    assert nrmse <= 0.003 and cosine >= 0.9999, (nrmse, cosine)
    assert (normed.float() - expected.float()).abs().max().item() <= 0.06


# The reference is the current formula on the current bytes: fp32 per-branch
# Gemma RMSNorm, then the mix in fp64 over the dequantized resident FP8 weights.
# No re-quantization anywhere, so a mismatch is the port, not the weights.
def _reference_norm(hyper_input: torch.Tensor, norm_w: torch.Tensor) -> torch.Tensor:
    x = hyper_input.float().unflatten(-1, (HC, HS))
    weight = (1.0 + norm_w.float()).unflatten(-1, (HC, HS))
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (x * inv_rms * weight).to(hyper_input.dtype).flatten(-2)


def _reference_mixed(normed: torch.Tensor, w_down, s_down, w_up, s_up) -> torch.Tensor:
    down = (w_down.float() * s_down[:, None]).double()
    up = (w_up.float() * s_up[:, None]).double()
    x = normed.double()
    t = F.silu(F.linear(x, down) / HC)
    gate = torch.sigmoid(F.linear(t, up)).unflatten(-1, (HC, HS))
    return (gate * x.unflatten(-1, (HC, HS))).mean(dim=-2).to(torch.bfloat16)


@pytest.fixture(scope="module")
def layer() -> GatedResidual:
    configure_online_fp8(
        True, cuda_available=True, capability=torch.cuda.get_device_capability()
    )
    try:
        config = HyperConnectionConfig(
            hc_count=HC,
            hidden_size=HS,
            params_dtype=torch.bfloat16,
            hc_lowrank=LOWRANK,
            rms_norm_eps=EPS,
            hc_per_branch_norm=True,
        )
        with torch.device("cuda"):
            built = GatedResidual(
                config, use_mix=True, use_combine=True, online_fp8=True
            )
        # Model construction runs under the configured BF16 default dtype.
        built.hc_norm.to(dtype=torch.bfloat16)
        # An all-zero norm weight would make (1 + w) the identity and hide a
        # wrong c*HS + s offset in K0, so every branch gets its own bias and
        # every column its own perturbation.
        branch_bias = torch.linspace(-0.35, 0.35, HC, device="cuda").repeat_interleave(
            HS
        )
        built.hc_norm.weight.data.copy_(
            (_randn((K,), seed=13, scale=0.05).float() + branch_bias).to(torch.bfloat16)
        )
        built.block_inject_weight.weight.data.copy_(
            _randn((HC, K), seed=11, scale=1.0 / math.sqrt(K))
        )
        for linear, seed, scale in (
            (built.input_mix_weight_down, 7, 1.0 / math.sqrt(K)),
            (built.input_mix_weight_up, 8, 1.0 / math.sqrt(LOWRANK)),
        ):
            loaded = _randn(tuple(linear.weight.shape), seed=seed, scale=scale)
            linear.weight.weight_loader(linear.weight, loaded)
            assert linear.weight.dtype == torch.float8_e4m3fn
            assert rowwise_scale_of(linear.weight) is not None
        yield built
    finally:
        configure_online_fp8(False, cuda_available=True, capability=(12, 0))


def _weights(layer: GatedResidual):
    down = layer.input_mix_weight_down.weight
    up = layer.input_mix_weight_up.weight
    return down, rowwise_scale_of(down), up, rowwise_scale_of(up)


def _assert_mix2(layer: GatedResidual, hyper_input: torch.Tensor) -> None:
    down, _, up, _ = _weights(layer)
    assert (
        hc_norm_mix2_supported(hyper_input, layer.hc_norm.weight, down, up, HC, HS)
        is True
    )


@pytest.fixture
def mix2_spy(monkeypatch):
    """Calls that reach the real launcher through hyperconnection, in order.

    Wraps (never replaces) `hc_norm_mix2` at the binding `GatedResidual.mix`
    resolves, so the donor kernels still run: a fast-path test cannot pass by
    the dispatch silently never firing, and a fallback test cannot pass because
    the whole path was stubbed out.
    """
    from sglang.srt.layers import hyperconnection

    calls = []
    real = hyperconnection.hc_norm_mix2

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(hyperconnection, "hc_norm_mix2", spy)
    return calls


def test_norm_weight_fixture_acts_on_every_branch(layer):
    """Guard for the checks below: they only exercise (1 + w) and K0's c*HS
    indexing if the norm weight is nonzero and varies within and across all 4
    branches."""
    weight = layer.hc_norm.weight.float()
    branches = weight.unflatten(-1, (HC, HS))
    assert weight.abs().max().item() >= 0.05
    assert bool((branches.std(dim=-1) > 0.01).all())
    means = branches.mean(dim=-1)
    assert bool(((means[1:] - means[:-1]).abs() > 0.05).all())


@pytest.mark.parametrize("rows", ROWS)
def test_hc_norm_mix2_matches_independent_reference(layer, rows):
    hyper_input = _randn((rows, K), seed=100 + rows, scale=0.25)
    _assert_mix2(layer, hyper_input)
    down, s_down, up, s_up = _weights(layer)

    mixed, normed = hc_norm_mix2(
        hyper_input, layer.hc_norm.weight, EPS, down, up, HC, HS
    )
    assert tuple(mixed.shape) == (rows, HS) and mixed.dtype == torch.bfloat16
    assert tuple(normed.shape) == (rows, K) and normed.dtype == torch.bfloat16

    ref_normed = _reference_norm(hyper_input, layer.hc_norm.weight)
    _assert_normed(normed, ref_normed)
    ref_mixed = _reference_mixed(ref_normed, down, s_down, up, s_up)
    # Budget for this port, measured against the SAME resident FP8 bytes and row
    # scales, so the quantization itself cancels and what is left is the kernel's
    # fp32 accumulation, its bf16 rounding of the silu intermediate and K1's
    # atomic ordering. Not the donor's claimed 2 ulp of output RMS -- that was
    # measured with the donor's own 1e-12-clamped quantization.
    nrmse, cosine = _normalized_error(mixed, ref_mixed)
    assert nrmse <= 0.02, nrmse
    assert cosine >= 0.999, cosine


@pytest.mark.parametrize("rows", ROWS)
def test_hc_norm_mix2_atomic_order_is_not_promiseable(layer, rows):
    """K1 accumulates with device-scope atomics: two identical calls must agree
    to tolerance, and the test must not be quietly tightened to bitwise."""
    hyper_input = _randn((rows, K), seed=200 + rows, scale=0.25)
    down, s_down, up, s_up = _weights(layer)
    first, _ = hc_norm_mix2(hyper_input, layer.hc_norm.weight, EPS, down, up, HC, HS)
    second, _ = hc_norm_mix2(hyper_input, layer.hc_norm.weight, EPS, down, up, HC, HS)
    ref = _reference_mixed(
        _reference_norm(hyper_input, layer.hc_norm.weight), down, s_down, up, s_up
    )
    for actual in (first, second):
        nrmse, cosine = _normalized_error(actual, ref)
        assert nrmse <= 0.02 and cosine >= 0.999
    delta = (first.float() - second.float()).abs().max().item()
    assert delta <= (ref.float().abs().max().item() * 0.02 + 1e-2)


@pytest.mark.parametrize("rows", ROWS)
def test_hc_norm_mix2_cuda_graph_replays_changed_input(layer, rows):
    hyper_input = _randn((rows, K), seed=300 + rows, scale=0.25)
    down, s_down, up, s_up = _weights(layer)
    _assert_mix2(layer, hyper_input)

    # Warm up so no JIT/compile work lands inside the capture.
    for _ in range(2):
        hc_norm_mix2(hyper_input, layer.hc_norm.weight, EPS, down, up, HC, HS)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mixed, normed = hc_norm_mix2(
            hyper_input, layer.hc_norm.weight, EPS, down, up, HC, HS
        )

    changed = _randn((rows, K), seed=400 + rows, scale=0.25)
    assert not torch.equal(hyper_input, changed)
    hyper_input.copy_(changed)
    ref_normed = _reference_norm(changed, layer.hc_norm.weight)
    ref_mixed = _reference_mixed(ref_normed, down, s_down, up, s_up)

    graph.replay()
    torch.cuda.synchronize()
    nrmse, cosine = _normalized_error(mixed, ref_mixed)
    assert nrmse <= 0.02 and cosine >= 0.999
    _assert_normed(normed, ref_normed)

    # Replaying twice proves K0 re-zeroes t_raw itself instead of relying on a
    # memset node (there is none in the graph) or leftover values.
    graph.replay()
    torch.cuda.synchronize()
    nrmse, cosine = _normalized_error(mixed, ref_mixed)
    assert nrmse <= 0.02 and cosine >= 0.999


@pytest.mark.parametrize("rows", ROWS)
def test_gated_residual_mix_uses_mix2_and_feeds_combine(layer, rows, mix2_spy):
    hyper_input = _randn((rows, K), seed=500 + rows, scale=0.25)
    down, s_down, up, s_up = _weights(layer)
    mixed, residuals = layer.mix(hyper_input)
    # The optimized path really ran, on the resident weights and their scales --
    # not a stubbed kernel, and not the fallback chain answering for it.
    assert len(mix2_spy) == 1
    _, _, _, spy_down, spy_up = mix2_spy[0][:5]
    assert spy_down is down and spy_up is up
    hyper_input_out, normed = residuals
    # The combine contract is untouched: the raw residual and the normalized
    # row both come back, and the normed row is still alive for the combine.
    assert hyper_input_out is hyper_input
    assert tuple(normed.shape) == (rows, K)
    ref_normed = _reference_norm(hyper_input, layer.hc_norm.weight)
    _assert_normed(normed, ref_normed)
    ref_mixed = _reference_mixed(ref_normed, down, s_down, up, s_up)
    nrmse, cosine = _normalized_error(mixed, ref_mixed)
    assert nrmse <= 0.02 and cosine >= 0.999

    block_output = _randn((rows, HS), seed=600 + rows, scale=0.25).to(torch.bfloat16)
    combined = layer.combine(block_output, residuals)
    inject = 2 * torch.sigmoid(
        F.linear(normed.float(), layer.block_inject_weight.weight.float()) / HC
    )
    expected = (
        (
            hyper_input.float().unflatten(-1, (HC, HS))
            + block_output.float().unsqueeze(-2) * inject.float().unsqueeze(-1)
        )
        .flatten(-2)
        .to(torch.bfloat16)
    )
    nrmse, cosine = _normalized_error(combined, expected)
    assert nrmse <= 0.03 and cosine >= 0.999


@pytest.mark.parametrize("rows", FALLBACK_ROWS)
def test_wider_rows_keep_the_existing_path(layer, rows, mix2_spy):
    hyper_input = _randn((rows, K), seed=700 + rows, scale=0.25)
    down, s_down, up, s_up = _weights(layer)
    assert (
        hc_norm_mix2_supported(hyper_input, layer.hc_norm.weight, down, up, HC, HS)
        is False
    )
    mixed, (residual_raw, normed) = layer.mix(hyper_input)
    assert rows > _HC_MIX2_MAX_ROWS
    ref_normed = _reference_norm(hyper_input, layer.hc_norm.weight)
    _assert_normed(normed, ref_normed)
    ref_mixed = _reference_mixed(ref_normed, down, s_down, up, s_up)
    # The existing path (compiled BF16 chain on dequantized FP8 operands):
    # this base's own budget for it, unchanged by the port.
    nrmse, cosine = _normalized_error(mixed, ref_mixed)
    assert nrmse <= 0.06 and cosine >= 0.995
    assert residual_raw is hyper_input
    assert len(mix2_spy) == 0


def test_tp2_and_deterministic_keep_the_existing_path(layer, monkeypatch, mix2_spy):
    from sglang.srt.layers import hc_mix2_triton, hc_mix_triton

    hyper_input = _randn((4, K), seed=800, scale=0.25)
    down, s_down, up, s_up = _weights(layer)
    assert (
        hc_norm_mix2_supported(hyper_input, layer.hc_norm.weight, down, up, HC, HS)
        is True
    )
    ref_mixed = _reference_mixed(
        _reference_norm(hyper_input, layer.hc_norm.weight), down, s_down, up, s_up
    )
    # Prove the spy is actually installed for the whole test: an empty count is
    # only evidence if something could have filled it.
    from sglang.srt.layers import hyperconnection

    assert hyperconnection.hc_norm_mix2 is not hc_mix2_triton.hc_norm_mix2

    for module, name, value in (
        (hc_mix2_triton, "_tp_size", lambda: 2),
        (hc_mix_triton, "_deterministic_inference", lambda: True),
    ):
        # A nested context per case: monkeypatch.undo() here would roll back
        # the shared mix2_spy fixture too, and the second case would then be
        # asserting on an empty spy that nothing could have filled.
        with monkeypatch.context() as gate_patch:
            gate_patch.setattr(module, name, value)
            assert (
                hc_norm_mix2_supported(
                    hyper_input, layer.hc_norm.weight, down, up, HC, HS
                )
                is False
            )
            mixed, _ = layer.mix(hyper_input)
            nrmse, cosine = _normalized_error(mixed, ref_mixed)
            assert nrmse <= 0.06 and cosine >= 0.995, (name, nrmse, cosine)
            assert len(mix2_spy) == 0, f"{name} reached the MIX2 launcher"
    assert len(mix2_spy) == 0


def test_bf16_mix_weights_are_not_mix2(layer):
    """Online FP8 off => the resident weights are BF16 and stay on the old path."""
    from torch import nn

    hyper_input = _randn((4, K), seed=900, scale=0.25)
    down = nn.Linear(K, LOWRANK, bias=False, device="cuda", dtype=torch.bfloat16)
    up = nn.Linear(LOWRANK, K, bias=False, device="cuda", dtype=torch.bfloat16)
    assert (
        hc_norm_mix2_supported(
            hyper_input, layer.hc_norm.weight, down.weight, up.weight, HC, HS
        )
        is False
    )


if __name__ == "__main__":
    # CI's run_unittest_files launches `python3 <file> -f` (legacy unittest
    # failfast), which pytest would reject; translate it to -x, the same way
    # sglang.test.kernels.utils.multigpu_pytest_main does. Without this block a
    # registered file is merely imported and exits 0 with zero tests run.
    _args = ["-x" if _arg == "-f" else _arg for _arg in sys.argv[1:]]
    sys.exit(pytest.main([__file__, "-v", *_args]))
