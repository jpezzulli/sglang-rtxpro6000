"""Route-parallel W4A16 NVFP4 MoE GEMV for the draft model's one-token MoE call.

At one token FlashInfer's CUTLASS NVFP4 MoE runs a routing prologue, an FP4 GEMM1, an
activation + FP4 requantization kernel and an FP4 GEMM2. Two Triton kernels read the
same server-format weights directly and keep the activations in bf16:

  k1  grid (I / BN1, R):  act[r, j] = silu(g_r * <W13[e_r, I + j], x>) * g_r * <W13[e_r, j], x>
  k2  grid (H / BN2, S2): out[n] = sum_r w_r * g2_r * <W2[e_r, n], act[r]>
                          (routes split S2 ways, fp32 partials, deterministic last-arriver sum)

Weight contract, i.e. what ``FlashInferCutlassMoeQuantInfo(quant_type="fp4")`` receives:

* ``w13_weight`` [E, 2I, H/2] uint8, rows [0, I) up and [I, 2I) gate
  (``load_up_proj_weight_first``; CUTLASS's doActivation gates with the second half);
* ``w13/w2_blockscale_swizzled``: e4m3 in the 128x4 layout of ``swizzle_blockscale``;
* ``g_r = w13_weight_scale_2[e_r, 0]`` for both halves, as CUTLASS's single ``g1_alphas``,
  and ``g2_r = w2_weight_scale_2[e_r]``. No input scale: the activations are never
  quantized, so only the weight decode scale applies.

Routes with expert id < 0 are skipped. Only the draft numerics change: CUTLASS
quantizes x and act to FP4, this path keeps both in bf16.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.triton_pdl import PDL, pdl_trigger, pdl_wait
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import (
    _FP4_TRICK,
    _unpack_dequant,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutput

logger = logging.getLogger(__name__)

# swizzle_blockscale (quantization/utils.py) tiles rows by 128 and scale columns by 4.
_SF_TILE_ROWS = 128
_SF_COL_GROUP = 4
_FP4_BLOCK = 16


class DraftMoeGemvConfig(msgspec.Struct, frozen=True):
    """Tile sizes, warps and pipeline stages of k1 (gate/up) and k2 (down)."""

    block_n1: int
    block_k1: int
    warps1: int
    stages1: int
    block_n2: int
    block_k2: int
    splits2: int
    warps2: int
    stages2: int


# Measured with flash-next-bench bench/dg1/dg1_bench.py --sweep (E=512, H=2560, I=640,
# top-10, RTX PRO 6000, 2026-10-01); re-tune when the kernels change.
DEFAULT_CONFIG = DraftMoeGemvConfig(
    block_n1=8,
    block_k1=512,
    warps1=4,
    stages1=2,
    block_n2=16,
    block_k2=128,
    splits2=10,
    warps2=2,
    stages2=3,
)


class DraftMoeWorkspace(msgspec.Struct, frozen=True):
    # [R, I] bf16 activations, written by k1 and read by k2.
    act: torch.Tensor
    # [H * splits2] fp32 split partials of k2.
    partials: torch.Tensor
    # [H / block_n2] int32, zero between launches: the last arriver resets its slot.
    counters: torch.Tensor


@triton.jit
def _swizzled_rows(pid, KP: tl.constexpr, BN: tl.constexpr):
    """Rows of CTA ``pid`` and each row's offset into one expert's swizzled scales.

    A CTA takes BN // 4 adjacent rows from each 32-row group of one 128-row tile, so the
    16 B scale slot that four such rows share is read whole by one CTA.
    """
    Q: tl.constexpr = BN // 4
    PER_TILE: tl.constexpr = 128 // BN
    tile = pid // PER_TILE
    i = tl.arange(0, BN)
    p = (pid % PER_TILE) * Q + i % Q
    m1 = i // Q
    rows = tile * 128 + m1 * 32 + p
    # (m // 128) * 128 * KP + (m % 32) * 16 + ((m % 128) // 32) * 4; k adds (k//4)*512 + k%4.
    sf_rows = tile * (128 * KP) + p * 16 + m1 * 4
    return rows, sf_rows


@triton.jit
def _swizzled_scales(sf_ptr, sf_rows, kb0, BN: tl.constexpr, BK: tl.constexpr):
    """[BN, BK] bf16 block scales for scale columns kb0 .. kb0 + BK // 16."""
    NB: tl.constexpr = BK // 16
    kb = kb0 + tl.arange(0, NB)
    off = sf_rows[:, None] + ((kb // 4) * 512 + kb % 4)[None, :]
    s = tl.load(sf_ptr + off).to(tl.bfloat16)
    s = tl.broadcast_to(s[:, :, None], (BN, NB, 16))
    return tl.reshape(s, (BN, BK))


@triton.jit
def _draft_moe_up_gate_kernel(
    x_ptr,
    ids_ptr,
    q_ptr,
    sf_ptr,
    g_ptr,
    g_stride,
    act_ptr,
    H: tl.constexpr,
    I: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    r = tl.program_id(1)
    KB: tl.constexpr = BK // 2
    KP: tl.constexpr = H // 16
    rows, sf_rows = _swizzled_rows(pid_n, KP, BN)
    offs_kb = tl.arange(0, KB)

    pdl_wait(USE_PDL)
    e = tl.load(ids_ptr + r).to(tl.int64)
    if e >= 0:
        q_up = q_ptr + e * (2 * I * (H // 2)) + rows[:, None] * (H // 2)
        q_gate = q_up + I * (H // 2)
        sf_up = sf_ptr + e * (2 * I * KP)
        sf_gate = sf_up + I * KP
        acc_up = tl.zeros((BN,), tl.float32)
        acc_gate = tl.zeros((BN,), tl.float32)
        for k0 in range(0, H, BK):
            xv = tl.load(x_ptr + k0 + tl.arange(0, BK)).to(tl.float32)
            w_up = _unpack_dequant(tl.load(q_up + (k0 // 2 + offs_kb)[None, :]), BN, KB)
            w_gate = _unpack_dequant(
                tl.load(q_gate + (k0 // 2 + offs_kb)[None, :]), BN, KB
            )
            w_up = w_up * _swizzled_scales(sf_up, sf_rows, k0 // 16, BN, BK)
            w_gate = w_gate * _swizzled_scales(sf_gate, sf_rows, k0 // 16, BN, BK)
            acc_up += tl.sum(w_up.to(tl.float32) * xv[None, :], axis=1)
            acc_gate += tl.sum(w_gate.to(tl.float32) * xv[None, :], axis=1)
        g = tl.load(g_ptr + e * g_stride) * _FP4_TRICK
        up = acc_up * g
        gate = acc_gate * g
        act = gate / (1.0 + tl.exp(-gate)) * up
        tl.store(act_ptr + r * I + rows, act.to(tl.bfloat16))
    pdl_trigger(USE_PDL)


@triton.jit
def _draft_moe_down_kernel(
    act_ptr,
    ids_ptr,
    w_ptr,
    q_ptr,
    sf_ptr,
    g_ptr,
    g_stride,
    out_ptr,
    part_ptr,
    cnt_ptr,
    H: tl.constexpr,
    I: tl.constexpr,
    R: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    S2: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    KB: tl.constexpr = BK // 2
    KP: tl.constexpr = I // 16
    rows, sf_rows = _swizzled_rows(pid_n, KP, BN)
    offs_kb = tl.arange(0, KB)

    pdl_wait(USE_PDL)
    acc = tl.zeros((BN,), tl.float32)
    for r in range(pid_s, R, S2):
        e = tl.load(ids_ptr + r).to(tl.int64)
        if e >= 0:
            q = q_ptr + e * (H * (I // 2)) + rows[:, None] * (I // 2)
            sf = sf_ptr + e * (H * KP)
            part = tl.zeros((BN,), tl.float32)
            for k0 in range(0, I, BK):
                av = tl.load(act_ptr + r * I + k0 + tl.arange(0, BK)).to(tl.float32)
                w = _unpack_dequant(tl.load(q + (k0 // 2 + offs_kb)[None, :]), BN, KB)
                w = w * _swizzled_scales(sf, sf_rows, k0 // 16, BN, BK)
                part += tl.sum(w.to(tl.float32) * av[None, :], axis=1)
            scale = tl.load(g_ptr + e * g_stride) * _FP4_TRICK
            acc += part * (scale * tl.load(w_ptr + r).to(tl.float32))

    if S2 == 1:
        tl.store(out_ptr + rows, acc.to(tl.bfloat16))
    else:
        offs = tl.arange(0, BN)
        base = part_ptr + pid_n * (S2 * BN)
        tl.store(base + pid_s * BN + offs, acc, cache_modifier=".cg")
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if done == S2 - 1:
            tot = tl.zeros((BN,), tl.float32)
            for s_i in tl.static_range(S2):
                tot += tl.load(base + s_i * BN + offs, cache_modifier=".cg")
            tl.store(out_ptr + rows, tot.to(tl.bfloat16))
            # Leave the counter at 0 for the next launch or graph replay.
            tl.store(cnt_ptr + pid_n, 0)
    pdl_trigger(USE_PDL)


def unsupported_reason(
    *, hidden_size: int, intermediate_size: int, cfg: DraftMoeGemvConfig
) -> Optional[str]:
    """Why these shapes cannot take the GEMV, or None when they can."""
    if hidden_size % _SF_TILE_ROWS or intermediate_size % _SF_TILE_ROWS:
        return "H and I must be multiples of 128 (no swizzle padding)"
    if (hidden_size // _FP4_BLOCK) % _SF_COL_GROUP or (
        intermediate_size // _FP4_BLOCK
    ) % _SF_COL_GROUP:
        return "H/16 and I/16 must be multiples of 4 (no swizzle padding)"
    for bn in (cfg.block_n1, cfg.block_n2):
        if bn < 4 or bn > _SF_TILE_ROWS or bn & (bn - 1):
            return f"block_n {bn} must be a power of two in [4, 128]"
    if hidden_size % cfg.block_k1 or intermediate_size % cfg.block_k2:
        return "block_k1 must divide H and block_k2 must divide I"
    return None


def make_workspace(
    *,
    num_routes: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
    cfg: DraftMoeGemvConfig,
) -> DraftMoeWorkspace:
    return DraftMoeWorkspace(
        act=torch.empty(
            (num_routes, intermediate_size), dtype=torch.bfloat16, device=device
        ),
        partials=torch.empty(
            hidden_size * cfg.splits2, dtype=torch.float32, device=device
        ),
        counters=torch.zeros(
            hidden_size // cfg.block_n2, dtype=torch.int32, device=device
        ),
    )


def draft_moe_gemv(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_blockscale_swizzled: torch.Tensor,
    w13_weight_scale_2: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_blockscale_swizzled: torch.Tensor,
    w2_weight_scale_2: torch.Tensor,
    workspace: DraftMoeWorkspace,
    cfg: DraftMoeGemvConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """One-token MoE: hidden_states [1, H] bf16, topk_* [1, R] -> [1, H] bf16 (new tensor)."""
    inter = w13_weight.shape[1] // 2
    hidden = w13_weight.shape[2] * 2
    routes = topk_ids.shape[1]
    out = torch.empty((1, hidden), dtype=torch.bfloat16, device=hidden_states.device)
    _draft_moe_up_gate_kernel[(inter // cfg.block_n1, routes)](
        hidden_states,
        topk_ids,
        w13_weight.view(torch.uint8),
        w13_blockscale_swizzled.view(torch.float8_e4m3fn),
        w13_weight_scale_2,
        w13_weight_scale_2.stride(0),
        workspace.act,
        H=hidden,
        I=inter,
        BN=cfg.block_n1,
        BK=cfg.block_k1,
        USE_PDL=PDL,
        launch_pdl=PDL,
        num_warps=cfg.warps1,
        num_stages=cfg.stages1,
    )
    _draft_moe_down_kernel[(hidden // cfg.block_n2, cfg.splits2)](
        workspace.act,
        topk_ids,
        topk_weights,
        w2_weight.view(torch.uint8),
        w2_blockscale_swizzled.view(torch.float8_e4m3fn),
        w2_weight_scale_2,
        w2_weight_scale_2.stride(0),
        out,
        workspace.partials,
        workspace.counters,
        H=hidden,
        I=inter,
        R=routes,
        BN=cfg.block_n2,
        BK=cfg.block_k2,
        S2=cfg.splits2,
        USE_PDL=PDL,
        launch_pdl=PDL,
        num_warps=cfg.warps2,
        num_stages=cfg.stages2,
    )
    return out


def _is_one_token_call(dispatch_output: StandardDispatchOutput) -> bool:
    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    return (
        x.shape[0] == 1
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and dispatch_output.hidden_states_scale is None
        and topk_output.topk_ids.is_contiguous()
        and topk_output.topk_weights.is_contiguous()
    )


class DraftMoeGemvRunner:
    """The one-token path of one draft MoE layer; owns that layer's scratch."""

    def __init__(self, cfg: DraftMoeGemvConfig = DEFAULT_CONFIG) -> None:
        self._cfg = cfg
        self._workspace: Optional[DraftMoeWorkspace] = None
        self._warned_capture = False

    def maybe_apply(
        self, *, layer: FusedMoE, dispatch_output: DispatchOutput
    ) -> Optional[StandardCombineInput]:
        """The combine input of an eligible one-token call; None means take CUTLASS."""
        if not isinstance(dispatch_output, StandardDispatchOutput):
            return None
        if not _is_one_token_call(dispatch_output):
            return None
        workspace = self._get_workspace(layer=layer, dispatch_output=dispatch_output)
        if workspace is None:
            return None
        topk_output = dispatch_output.topk_output
        out = draft_moe_gemv(
            hidden_states=dispatch_output.hidden_states,
            topk_ids=topk_output.topk_ids,
            topk_weights=topk_output.topk_weights,
            w13_weight=layer.w13_weight,
            w13_blockscale_swizzled=layer.w13_blockscale_swizzled,
            w13_weight_scale_2=layer.w13_weight_scale_2,
            w2_weight=layer.w2_weight,
            w2_blockscale_swizzled=layer.w2_blockscale_swizzled,
            w2_weight_scale_2=layer.w2_weight_scale_2,
            workspace=workspace,
            cfg=self._cfg,
        )
        return StandardCombineInput(hidden_states=out)

    def _get_workspace(
        self, *, layer: FusedMoE, dispatch_output: StandardDispatchOutput
    ) -> Optional[DraftMoeWorkspace]:
        if self._workspace is not None:
            return self._workspace
        # Allocating inside a capture would put the counters in the graph's pool;
        # the draft graph runners warm up eagerly first, so this is only a guard.
        if torch.cuda.is_current_stream_capturing():
            if not self._warned_capture:
                logger.warning(
                    "draft MoE GEMV: first one-token call is inside a CUDA graph "
                    "capture; layer %d keeps CUTLASS for this graph",
                    layer.layer_id,
                )
                self._warned_capture = True
            return None
        self._workspace = make_workspace(
            num_routes=dispatch_output.topk_output.topk_ids.shape[1],
            hidden_size=layer.w13_weight.shape[2] * 2,
            intermediate_size=layer.w13_weight.shape[1] // 2,
            device=dispatch_output.hidden_states.device,
            cfg=self._cfg,
        )
        return self._workspace


def _layer_unsupported_reason(experts: FusedMoE) -> Optional[str]:
    runner_config = experts.moe_runner_config
    if experts.moe_ep_size != 1 or experts.moe_tp_size != 1:
        return "needs moe_ep_size == moe_tp_size == 1"
    if not (runner_config.activation == "silu" and runner_config.is_gated):
        return "needs gated silu"
    if (
        runner_config.apply_router_weight_on_input
        or runner_config.gemm1_alpha is not None
        or runner_config.gemm1_beta is not None
        or runner_config.gemm1_clamp_limit is not None
        or runner_config.swiglu_limit is not None
    ):
        return "needs plain silu(gate) * up with router weights on the output"
    return unsupported_reason(
        hidden_size=experts.w13_weight.shape[2] * 2,
        intermediate_size=experts.w13_weight.shape[1] // 2,
        cfg=DEFAULT_CONFIG,
    )


def enable_draft_moe_gemv(experts: torch.nn.Module) -> bool:
    """Opt one draft-model FusedMoE layer into the one-token GEMV; True if it applied."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )

    if not isinstance(experts, FusedMoE) or not isinstance(
        experts.quant_method, ModelOptNvFp4FusedMoEMethod
    ):
        return False
    reason = _layer_unsupported_reason(experts)
    if reason is not None:
        logger.warning(
            "draft MoE GEMV ignored for layer %d: %s",
            experts.layer_id,
            reason,
        )
        return False
    experts.quant_method.draft_moe_gemv = DraftMoeGemvRunner()
    logger.info("draft MoE GEMV enabled for layer %d", experts.layer_id)
    return True
