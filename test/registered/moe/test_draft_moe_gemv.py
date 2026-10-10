"""Representative checks for the DG1 one-token draft MoE GEMV (SGLANG_OPT_DRAFT_MOE_GEMV).

Adapted from the donor's own checks (flash-next-bench bench/dg1/selftest_cpu.py and
dg1_bench.py, donor snapshot 5105985116eb00dea8e6138aabeb5363387cb9de): the same
mtpft5 MTP shapes (H=2560, I=640, top-10, a 12-expert slice), random weights in the
server's CUTLASS NVFP4 format ([up; gate] rows, 128x4-swizzled e4m3 block scales),
and an independent torch fp32 dequant reference (the donor's "dq" reference, bf16
activations untouched by FP4 requantization). The donor's gates carry over: rel L2
vs the fp32 reference < 2e-2, bitwise determinism across reruns and across CUDA
graph replay, and counters back at zero after every launch. CUTLASS itself is not
imported here; the multirow/scale-present fallbacks are pinned by returning None.
"""

from __future__ import annotations

import types

import pytest
import torch

from sglang.srt.layers.moe import draft_moe_gemv as dmg
from sglang.srt.layers.moe.draft_moe_gemv import (
    DEFAULT_CONFIG,
    DraftMoeGemvConfig,
    DraftMoeGemvRunner,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# mtpft5 MTP MoE layer shapes (Qwen3.8-Flash-Next config.json), as in dg1/grid.py.
H, I, TOPK = 2560, 640, 10
N_EXP = 12  # the donor selftest's expert slice; E=512 only costs memory.
REL_L2_MAX = 2e-2  # the donor selftest_cpu.py gate vs the fp32 dequant reference.

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="draft MoE GEMV requires CUDA"
)

_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def _dequant_ref(q, bs, g):
    """[N, K/2] uint8 + [N, K/16] e4m3 + fp32 scale -> [N, K] fp32 (low nibble = even k)."""
    lut = _E2M1.to(q.device)
    w = torch.empty((q.shape[0], q.shape[1] * 2), dtype=torch.float32, device=q.device)
    w[:, 0::2] = lut[(q & 0xF).long()]
    w[:, 1::2] = lut[(q >> 4).long()]
    return w * bs.to(torch.float32).repeat_interleave(16, dim=1) * g


def _rand_pack(n, k, device, gen):
    q = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device=device, generator=gen
    )
    bs = (
        torch.rand((n, k // 16), dtype=torch.float32, device=device, generator=gen)
        * 3.75
        + 0.25
    ).to(torch.float8_e4m3fn)
    return q, bs


@pytest.fixture(scope="module")
def layer():
    """A duck of the FusedMoE state after process_weights_after_loading (dg1/layer.py)."""
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(0)
    w13q, w13s, w2q, w2s = [], [], [], []
    for _ in range(N_EXP):
        q, bs = _rand_pack(2 * I, H, device, gen)
        w13q.append(q)
        w13s.append(bs)
        q, bs = _rand_pack(H, I, device, gen)
        w2q.append(q)
        w2s.append(bs)
    layer = types.SimpleNamespace(
        layer_id=0,
        w13_weight=torch.stack(w13q),
        w2_weight=torch.stack(w2q),
        # modelopt_quant.py keeps the raw per-expert weight scales on the CUTLASS path;
        # the GEMV reads column 0 for both halves, as CUTLASS's single g1_alphas does.
        w13_weight_scale_2=(
            torch.rand((N_EXP, 2), dtype=torch.float32, device=device, generator=gen)
            * 0.01
            + 0.001
        ),
        w2_weight_scale_2=(
            torch.rand((N_EXP,), dtype=torch.float32, device=device, generator=gen)
            * 0.01
            + 0.001
        ),
    )
    layer.w13_blockscale_swizzled = swizzle_blockscale(torch.stack(w13s))
    layer.w2_blockscale_swizzled = swizzle_blockscale(torch.stack(w2s))
    layer.refs = (torch.stack(w13s), torch.stack(w2s))  # unswizzled, for the reference
    return layer


def _reference_out(layer, x, ids, wts):
    """fp32 expert chain on dequantized weights, [up; gate] row order, bf16 activations."""
    w13s, w2s = layer.refs
    xf = x[0].float()
    y = torch.zeros(H, dtype=torch.float32, device=x.device)
    for r in range(ids.shape[1]):
        e = int(ids[0, r])
        if e < 0:
            continue  # routes with id < 0 are skipped, as in the kernels
        g13 = layer.w13_weight_scale_2[e, 0].float()
        w13 = _dequant_ref(layer.w13_weight[e], w13s[e], g13)
        up = xf @ w13[:I].t()
        gate = xf @ w13[I:].t()
        act = torch.nn.functional.silu(gate) * up
        w2 = _dequant_ref(
            layer.w2_weight[e], w2s[e], layer.w2_weight_scale_2[e].float()
        )
        y += float(wts[0, r]) * (act @ w2.t())
    return y


def _run(layer, cfg, x, ids, wts, ws=None):
    ws = ws or dmg.make_workspace(
        num_routes=ids.shape[1],
        hidden_size=H,
        intermediate_size=I,
        device=x.device,
        cfg=cfg,
    )
    return ws, dmg.draft_moe_gemv(
        hidden_states=x,
        topk_ids=ids,
        topk_weights=wts,
        w13_weight=layer.w13_weight,
        w13_blockscale_swizzled=layer.w13_blockscale_swizzled,
        w13_weight_scale_2=layer.w13_weight_scale_2,
        w2_weight=layer.w2_weight,
        w2_blockscale_swizzled=layer.w2_blockscale_swizzled,
        w2_weight_scale_2=layer.w2_weight_scale_2,
        workspace=ws,
        cfg=cfg,
    )


def _routes(seed):
    ids = torch.randint(0, N_EXP, (TOPK,), dtype=torch.int32).cuda()
    wts = torch.rand((TOPK,), dtype=torch.float32).cuda()
    if seed == 1:
        ids[3] = -1
        ids[7] = -1  # skipped routes, as in the donor selftest
    return ids.unsqueeze(0), torch.softmax(wts, -1).unsqueeze(0)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    "cfg",
    [
        DEFAULT_CONFIG,
        DraftMoeGemvConfig(  # splits2=1 (no split-K path) + different tiles
            block_n1=16,
            block_k1=256,
            warps1=4,
            stages1=2,
            block_n2=8,
            block_k2=64,
            splits2=1,
            warps2=2,
            stages2=3,
        ),
    ],
)
def test_matches_fp32_reference_and_is_deterministic(layer, cfg, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        (1, H), dtype=torch.float32, device="cuda", generator=gen
    ).bfloat16()
    ids, wts = _routes(seed)
    ws, y = _run(layer, cfg, x, ids, wts)
    ref = _reference_out(layer, x, ids, wts)
    rel = ((y[0].float() - ref).norm() / ref.norm()).item()
    assert rel < REL_L2_MAX, f"rel L2 {rel:.2e} vs the fp32 reference (cfg {cfg})"
    assert torch.isfinite(y.float()).all()
    _, y2 = _run(layer, cfg, x, ids, wts, ws=ws)  # rerun: bitwise, incl. counters reuse
    assert torch.equal(y, y2)
    assert int(ws.counters.abs().sum()) == 0
    ids_before, wts_before = ids.clone(), wts.clone()
    _run(layer, cfg, x, ids, wts)
    assert torch.equal(ids, ids_before) and torch.equal(wts, wts_before)


def test_cuda_graph_replay_is_bitwise(layer):
    gen = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(
        (1, H), dtype=torch.float32, device="cuda", generator=gen
    ).bfloat16()
    ids, wts = _routes(0)
    ws, y_eager = _run(layer, DEFAULT_CONFIG, x, ids, wts)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_graph = dmg.draft_moe_gemv(
            hidden_states=x,
            topk_ids=ids,
            topk_weights=wts,
            w13_weight=layer.w13_weight,
            w13_blockscale_swizzled=layer.w13_blockscale_swizzled,
            w13_weight_scale_2=layer.w13_weight_scale_2,
            w2_weight=layer.w2_weight,
            w2_blockscale_swizzled=layer.w2_blockscale_swizzled,
            w2_weight_scale_2=layer.w2_weight_scale_2,
            workspace=ws,
            cfg=DEFAULT_CONFIG,
        )
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y_graph, y_eager)
    assert int(ws.counters.abs().sum()) == 0  # the last arriver reset every slot


def _dispatch(x, ids, wts, scale=None):
    return StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=scale,
        topk_output=StandardTopKOutput(
            topk_weights=wts, topk_ids=ids, router_logits=None
        ),
    )


def test_runner_selects_one_token_and_falls_back(layer):
    """maybe_apply: T=1 bf16 takes the GEMV; multirow / scaled / non-standard fall back."""
    runner = DraftMoeGemvRunner()
    gen = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(
        (1, H), dtype=torch.float32, device="cuda", generator=gen
    ).bfloat16()
    ids, wts = _routes(0)
    out = runner.maybe_apply(layer=layer, dispatch_output=_dispatch(x, ids, wts))
    assert out is not None and out.hidden_states.shape == (1, H)
    _, y_direct = _run(layer, DEFAULT_CONFIG, x, ids, wts)
    assert torch.equal(out.hidden_states, y_direct)
    # Multirow / concurrent calls keep the donor's CUTLASS fallback (no cap, no skip).
    x4 = x.expand(4, H).contiguous()
    ids4, wts4 = ids.expand(4, TOPK).contiguous(), wts.expand(4, TOPK).contiguous()
    assert (
        runner.maybe_apply(layer=layer, dispatch_output=_dispatch(x4, ids4, wts4))
        is None
    )
    one_fp32 = x.float()
    assert (
        runner.maybe_apply(layer=layer, dispatch_output=_dispatch(one_fp32, ids, wts))
        is None
    )
    assert (
        runner.maybe_apply(layer=layer, dispatch_output=_dispatch(x, ids, wts, scale=x))
        is None
    )
    assert runner.maybe_apply(layer=layer, dispatch_output=object()) is None


def test_enablement_rules():
    """The donor's selection rules: plain gated silu, tp/ep 1, shapes on the swizzle grid."""

    def cfg_for(activation="silu", **kw):
        base = dict(
            activation=activation,
            is_gated=True,
            apply_router_weight_on_input=False,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            swiglu_limit=None,
        )
        base.update(kw)
        return types.SimpleNamespace(**base)

    experts = types.SimpleNamespace(
        moe_runner_config=cfg_for(),
        moe_ep_size=1,
        moe_tp_size=1,
        w13_weight=torch.empty((2, 2 * I, H // 2), dtype=torch.uint8),
    )
    assert dmg._layer_unsupported_reason(experts) is None
    experts.moe_runner_config = cfg_for(activation="gelu")
    assert dmg._layer_unsupported_reason(experts) == "needs gated silu"
    experts.moe_runner_config = cfg_for(swiglu_limit=7.0)
    assert "plain silu(gate) * up" in dmg._layer_unsupported_reason(experts)
    experts.moe_runner_config = cfg_for()
    experts.moe_tp_size = 2
    assert "moe_ep_size == moe_tp_size == 1" in dmg._layer_unsupported_reason(experts)
    assert dmg.enable_draft_moe_gemv(torch.nn.Linear(2, 2)) is False
