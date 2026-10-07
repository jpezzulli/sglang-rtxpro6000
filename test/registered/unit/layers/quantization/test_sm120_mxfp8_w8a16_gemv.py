# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for the opt-in dense block-MXFP8 low-row GEMV candidate.

Everything here runs without a GPU and without touching the stored representation:
the gates that decide whether a dense MXFP8 call may take the candidate, the
UE8M0 / [1, 32] weight-scale indexing contract the kernel implements, the launch
plan/scratch bounds, and the routing inside ``Fp8LinearMethod.apply`` including
the fallback to the untouched dispatch.  No SM120 numerical or throughput claim is
made or implied here — see the kernel file for the device coverage.
"""

from __future__ import annotations

import os

import pytest
import torch

from sglang.kernels.ops.gemm import sm120_w8a16_gemv
from sglang.kernels.ops.quantization.mxfp8_quant import MXFP8Tensor, from_mxfp8
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

ENV = sm120_w8a16_gemv.MX_GEMV_ENV
HEAD_ENV = sm120_w8a16_gemv.GEMV_ENV
SMS = 188  # RTX PRO 6000 Blackwell Max-Q, the part EXACT_SM120 names
SM120 = sm120_w8a16_gemv.EXACT_SM120
SF = sm120_w8a16_gemv.MXFP8_SF

# Representative dense-projection (N, K) shapes at Flash-Next widths: shared_expert
# gate_up/down, attention qkv/o, linear-attention in_proj_qkvz/in_proj_ba/out_proj
# and the MTP dense projections.  Routers, the indexer, PLE, embeddings and the
# routed experts are not dense MXFP8 linears, so they are never offered to this
# gate; the small-N row is the underfilled case the [1, 32] conversion itself
# skips (it requires N >= 128) and that the split-K plan still has to serve.  The
# two wide rows are the range where the planner takes the wide multirow tile
# (cdiv(N, 128) * 2 >= 188 SMs, i.e. N >= 12032) and where the shared-memory
# budget, not the grid, is the binding constraint.
DENSE_SHAPES = [
    (1024, 2560),  # shared_expert gate_up_proj
    (2560, 512),  # shared_expert down_proj
    (2048, 2560),  # attention qkv_proj (TP1 shard)
    (1024, 1024),  # attention o_proj / GDN out_proj
    (8192, 2560),  # GDN in_proj_qkvz
    (64, 2560),  # GDN in_proj_ba (2 * num_v_heads)
    (1024, 1280),  # NEXTN / MTP dense projection
    (12288, 2560),  # wide gate_up_proj shard: multirow tile, over smem at BK 256
    (24576, 2560),  # wider still: same tile, more N blocks
]

# Every M bucket the gate admits: M == 1, the 2..4 bucket, the 5..16 bucket.
ROW_BUCKETS = (1, 2, 4, 5, 8, 12, 16)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _m_pad(rows: int, plan) -> int:
    """The M_PAD ``_launch`` will pass for this plan, same expression."""
    return 16 if (plan[3] or rows > 1) else 1


def _planned_smem(rows: int, plan, sf_group: int = SF) -> int:
    return sm120_w8a16_gemv._smem_bytes(plan[0], plan[1], _m_pad(rows, plan), sf_group)


def _gate_env(monkeypatch, *, dense: bool, head: bool = False):
    for name, on in ((ENV, dense), (HEAD_ENV, head)):
        if on:
            monkeypatch.setenv(name, "1")
        else:
            monkeypatch.delenv(name, raising=False)


def _mxfp8_weight(n: int, k: int, *, seed: int = 0, ue8m0: bool = True, lo=112, hi=143):
    """A stored MXFP8 pair: fp8 e4m3 weight plus UE8M0 [N, K/32] scale bytes."""
    generator = torch.Generator().manual_seed(seed)
    weight = (
        torch.randn((n, k), generator=generator)
        .clamp_(-2.0, 2.0)
        .to(torch.float8_e4m3fn)
    )
    scale = torch.randint(lo, hi, (n, k // SF), generator=generator, dtype=torch.uint8)
    if ue8m0:
        # The MXFP8 loaders mark the stored bytes as UE8M0 (fp8.py sets it for
        # every use_mxfp8 layer); the candidate refuses to guess otherwise.
        scale.format_ue8m0 = True
    return weight, scale


def _supported(x, weight, scale, capability=SM120):
    return sm120_w8a16_gemv.lowrow_mxfp8_gemv_supported(
        x, weight, scale, capability=capability
    )


def _bf16(rows: int, columns: int, dtype: torch.dtype = torch.bfloat16):
    return torch.zeros(rows, columns, dtype=dtype)


def test_dense_mxfp8_gemv_gate_is_independent_and_default_off(monkeypatch):
    weight, scale = _mxfp8_weight(64, 128)
    x = torch.zeros(1, 128, dtype=torch.bfloat16)

    _gate_env(monkeypatch, dense=False)
    assert sm120_w8a16_gemv.mxfp8_gemv_enabled() is False
    assert _supported(x, weight, scale) is False
    assert sm120_w8a16_gemv.lowrow_mxfp8_gemv(x, weight, scale) is None

    # The output-head gate does not enable the dense candidate, and the dense gate
    # does not enable the output-head kernel: two independently reviewable
    # switches, both default off.
    _gate_env(monkeypatch, dense=False, head=True)
    assert sm120_w8a16_gemv.mxfp8_gemv_enabled() is False
    assert sm120_w8a16_gemv.lowrow_gemv_enabled() is True
    assert _supported(x, weight, scale) is False

    _gate_env(monkeypatch, dense=True)
    assert sm120_w8a16_gemv.mxfp8_gemv_enabled() is True
    assert sm120_w8a16_gemv.lowrow_gemv_enabled() is False
    assert _supported(x, weight, scale) is True


def test_dense_gate_covers_rows_dtype_layout_scale_and_part(monkeypatch):
    _gate_env(monkeypatch, dense=True)
    assert sm120_w8a16_gemv.MAX_ROWS == 16
    n, k = 64, 160
    weight, scale = _mxfp8_weight(n, k)

    # 1 decode, 4/12/16 the W4 verify widths.  24 is C6 verification and 48 a
    # mixed draft/verify batch: both keep their ordinary kernels, and 17 shows
    # the donor limit is the limit — it was not raised to make a shape fit.
    for rows in (1, 4, 12, 16):
        assert _supported(_bf16(rows, k), weight, scale)
    for rows in (0, 17, 24, 48, 64):
        assert _supported(_bf16(rows, k), weight, scale) is False, rows

    for dtype in (torch.float32, torch.float16):
        assert _supported(_bf16(4, k, dtype), weight, scale) is False, dtype
    assert _supported(_bf16(4, k + SF), weight, scale) is False  # K != weight K
    hidden_3d = torch.zeros(1, 4, k, dtype=torch.bfloat16)
    assert _supported(hidden_3d, weight, scale) is False  # not flattened to 2-D
    assert _supported(_bf16(k, 4).t(), weight, scale) is False  # non-unit K stride

    # Formats the candidate will not read.
    assert _supported(_bf16(4, k), weight.to(torch.bfloat16), scale) is False
    assert _supported(_bf16(4, k), weight, scale.to(torch.float32)) is False
    assert _supported(_bf16(4, k), weight, scale[:, : k // 64]) is False  # [N, K/32]
    assert _supported(_bf16(4, k), weight, scale.reshape(-1)) is False
    assert _supported(_bf16(4, k), weight, None) is False
    _, unmarked = _mxfp8_weight(n, k, ue8m0=False)
    # uint8 alone does not promise UE8M0.
    assert _supported(_bf16(4, k), weight, unmarked) is False
    # A K that is not a whole number of scale groups cannot carry [N, K/32] bytes.
    ragged_k = 144
    ragged_weight = torch.zeros(n, ragged_k).to(torch.float8_e4m3fn)
    ragged_scale = torch.full((n, _cdiv(ragged_k, SF)), 127, dtype=torch.uint8)
    ragged_scale.format_ue8m0 = True
    assert _supported(_bf16(4, ragged_k), ragged_weight, ragged_scale) is False

    # Strides: a K-strided weight, or a column-major scale (the transposed/swizzled
    # copies the FlashInfer and DeepGEMM backends keep separately), is not the
    # natural stored layout the kernel indexes.
    strided_weight = weight.t().contiguous().t()
    assert strided_weight.stride(1) != 1
    assert _supported(_bf16(4, k), strided_weight, scale) is False
    strided_scale = scale.t().contiguous().t()
    assert strided_scale.stride(1) != 1
    assert _supported(_bf16(4, k), weight, strided_scale) is False

    # Hardware bound, stated rather than inferred from a model name.
    for capability in ((9, 0), (10, 0), (12, 1), (12, 0, 1), ()):
        assert (
            _supported(_bf16(4, k), weight, scale, capability=capability) is False
        ), capability
    assert SM120 == (12, 0)
    # On this CPU-only checkout the device capability is empty, so the public
    # entry refuses even for an otherwise-eligible call.
    assert sm120_w8a16_gemv.lowrow_mxfp8_gemv(_bf16(4, k), weight, scale) is None


def test_ue8m0_scale_contract_matches_the_in_tree_mxfp8_dequant():
    weight, scale = _mxfp8_weight(32, 256, seed=7, lo=1, hi=255)
    in_tree = from_mxfp8(MXFP8Tensor(weight, scale), out_dtype=torch.float32)
    candidate = sm120_w8a16_gemv.dequantize_mxfp8_weight(
        weight, scale, dtype=torch.float32
    )
    torch.testing.assert_close(candidate, in_tree, rtol=0, atol=0)

    decoded = sm120_w8a16_gemv.ue8m0_to_float32(scale)
    assert torch.equal(decoded, torch.pow(2.0, scale.to(torch.int32) - 127))
    groups = candidate.reshape(32, 256 // SF, SF)
    expected = weight.float().reshape(32, 256 // SF, SF) * decoded[:, :, None]
    torch.testing.assert_close(groups, expected, rtol=0, atol=0)
    # The stored values pass through the dequant untouched: nothing requantized,
    # and the fp8 -> bf16 step the kernel takes is exact for these products.
    assert torch.equal(
        candidate.to(torch.bfloat16).float(),
        (weight.float() * decoded.repeat_interleave(SF, dim=1))
        .to(torch.bfloat16)
        .float(),
    )
    # A zero exponent byte only ever rides an all-zero block, where the two
    # in-tree conventions (0.0 and 2**-127) agree.
    zeros = torch.zeros(4, 32, dtype=torch.float8_e4m3fn)
    zero_scale = torch.zeros(4, 1, dtype=torch.uint8)
    assert torch.equal(
        sm120_w8a16_gemv.dequantize_mxfp8_weight(
            zeros, zero_scale, dtype=torch.float32
        ),
        from_mxfp8(MXFP8Tensor(zeros, zero_scale), out_dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="does not match weight"):
        sm120_w8a16_gemv.dequantize_mxfp8_weight(weight, scale[:, :4])
    with pytest.raises(TypeError, match="2-D weight and scale"):
        sm120_w8a16_gemv.dequantize_mxfp8_weight(weight.reshape(-1), scale)


class _Recorder:
    """Stand-in for the Triton kernel: records the grid, args and constexprs."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **constexprs):
            self.launches.append((grid, args, constexprs))

        return launch


_NO_OWNER = object()


def _patch_launcher(monkeypatch, *, slots=None, site_slot=_NO_OWNER):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(sm120_w8a16_gemv, "_num_sms", lambda device: SMS)
    monkeypatch.setattr(sm120_w8a16_gemv, "_device_capability", lambda device: SM120)
    monkeypatch.setattr(sm120_w8a16_gemv, "_LAYER_SLOTS", {})
    monkeypatch.setattr(sm120_w8a16_gemv, "_LAYER_ROTATION", 0)
    monkeypatch.setattr(sm120_w8a16_gemv, "_WS", {})
    if slots is not None:
        monkeypatch.setattr(sm120_w8a16_gemv, "_N_SLOTS", slots)
    if site_slot is not _NO_OWNER:
        monkeypatch.setattr(
            sm120_w8a16_gemv, "_slot_for_launch", lambda device, owner: site_slot
        )
    recorder = _Recorder()
    monkeypatch.setattr(sm120_w8a16_gemv, "_w8a16_gemv_kernel", recorder)
    return recorder


def _dense_launch(monkeypatch, rows: int, n: int, k: int, *, owner=None, **kwargs):
    recorder = _patch_launcher(monkeypatch, **kwargs)
    weight, scale = _mxfp8_weight(n, k, seed=rows + n)
    x = torch.zeros(rows, k, dtype=torch.bfloat16)
    out = sm120_w8a16_gemv.lowrow_mxfp8_gemv(x, weight, scale, owner)
    assert len(recorder.launches) == 1
    ((grid, args, constexprs),) = recorder.launches
    return out, x, grid, args, constexprs, weight, scale


def test_dense_launch_consumes_the_stored_tensors_in_one_launch(monkeypatch):
    rows, n, k = 4, 1024, 1280
    out, x, grid, args, constexprs, weight, scale = _dense_launch(
        monkeypatch, rows, n, k
    )

    assert out.shape == (rows, n) and out.dtype == torch.bfloat16
    # The kernel is handed the stored MXFP8 tensors themselves, not a converted
    # copy: identity, not equality, is the point.
    assert args[0] is x and args[1] is weight and args[2] is scale
    assert (args[6], args[7], args[8]) == (rows, n, k)
    assert (args[11], args[12]) == (weight.stride(0), weight.stride(1))
    # The scale strides come from the stored [N, K/32] tensor, row-major.
    assert (args[15], args[16]) == (scale.stride(0), scale.stride(1))
    assert (args[15], args[16]) == (k // SF, 1)
    assert grid == (_cdiv(n, constexprs["BLOCK_N"]), constexprs["SPLITS"])
    assert constexprs["SF_GROUP"] == SF  # 0 would mean the rowwise head contract
    assert constexprs["USE_DOT"] is True and constexprs["M_PAD"] == 16
    assert constexprs["EVEN_K"] is (k % constexprs["BLOCK_K"] == 0)
    # The rowwise entry keeps its own contract on the shared kernel.
    recorder = _patch_launcher(monkeypatch)
    rowwise_scale = torch.rand(n, dtype=torch.float32)
    sm120_w8a16_gemv.lowrow_fp8_gemv(x, weight.to(torch.float8_e4m3fn), rowwise_scale)
    ((_grid, _args, rowwise_cfg),) = recorder.launches
    assert rowwise_cfg["SF_GROUP"] == 0


def test_dense_plan_bounds_the_scratch_and_skips_the_head_tile_table(monkeypatch):
    for rows in (1, 4, 16):
        for n, k in DENSE_SHAPES:
            block_n, block_k, splits, use_dot, _w, _s = sm120_w8a16_gemv._plan(
                rows, n, k, SMS, False, SF
            )
            assert 0 < block_k and 1 <= splits <= sm120_w8a16_gemv._MAX_SPLITS
            n_blocks = _cdiv(n, block_n)
            m_pad = 16 if (use_dot or rows > 1) else 1
            assert n_blocks <= sm120_w8a16_gemv._WS_COUNTERS
            assert n_blocks * splits * m_pad * block_n <= sm120_w8a16_gemv._WS_FLOATS

    # An underfilled grid must split K — that is the whole point for the small-N
    # projections — and the launcher must size the grid's y with that split count.
    _out, _x, (grid_n, splits), _args, constexprs, _w, _s = _dense_launch(
        monkeypatch, 16, 64, 2560
    )
    assert grid_n == 2 and splits > 1 and splits == constexprs["SPLITS"]

    # The donor's tiles were measured on the rowwise output heads, so the dense
    # path has to bypass that table instead of inheriting an unknown measurement.
    tuned = sm120_w8a16_gemv._plan(4, 32768, 2560, SMS)
    generic = sm120_w8a16_gemv._plan(4, 32768, 2560, SMS, False, SF)
    assert tuned == (128, 256, 1, True, 8, 3)
    assert generic != tuned


def test_dense_plan_fits_the_sm120_shared_memory_budget():
    """The launch plan for every admitted shape has to fit the part's smem.

    Red on the base revision for the wide multirow shape: that plan was
    ``[128, 256]`` at M_PAD 16, whose gathered [1, 32] scale tile makes it
    147456 B -- the exact figure Triton's OutOfResources named against the
    101376 B sm_120 gives a launch.  The plan the fix selects is the same tile
    with a shorter K loop, and the rowwise heads' measured tile is untouched.
    """
    assert sm120_w8a16_gemv.SMEM_LIMIT_SM120 == 101376
    # The old dense tile spelled out, so this check names the failure it guards.
    assert sm120_w8a16_gemv._smem_bytes(128, 256, 16, SF) == 147456
    assert (
        sm120_w8a16_gemv._smem_bytes(128, 256, 16, SF)
        > sm120_w8a16_gemv.SMEM_LIMIT_SM120
    )
    # Without the scale tile -- the rowwise head contract -- the same tile fits,
    # which is why the heads keep their donor-measured plan and its 8 warps.
    assert sm120_w8a16_gemv._smem_bytes(128, 256, 16, 0) == 81920
    assert (
        sm120_w8a16_gemv._smem_bytes(128, 256, 16, 0)
        < sm120_w8a16_gemv.SMEM_LIMIT_SM120
    )

    for rows in ROW_BUCKETS:
        for n, k in DENSE_SHAPES:
            plan = sm120_w8a16_gemv._plan(rows, n, k, SMS, False, SF)
            assert _planned_smem(rows, plan) <= sm120_w8a16_gemv.SMEM_LIMIT_SM120, (
                rows,
                n,
                k,
                plan,
            )

    # The regression itself: BLOCK_N, the grid and the N-block ownership stay as
    # the planner chose them, only the K tile shortens, for both multirow buckets.
    for rows in (5, 8, 12, 16):
        assert sm120_w8a16_gemv._plan(rows, 12288, 2560, SMS, False, SF) == (
            128,
            128,
            1,
            True,
            8,
            3,
        )
        assert sm120_w8a16_gemv._plan(rows, 24576, 2560, SMS, False, SF) == (
            128,
            128,
            1,
            True,
            8,
            3,
        )
    # M == 1 takes the broadcast-reduce tile at any N and was never over budget.
    assert sm120_w8a16_gemv._plan(1, 12288, 2560, SMS, False, SF) == (
        32,
        256,
        1,
        False,
        4,
        3,
    )
    # The head path is not retuned as a side effect of the dense fix.
    assert sm120_w8a16_gemv._plan(8, 24576, 2560, SMS) == (128, 256, 1, True, 8, 3)
    assert sm120_w8a16_gemv._plan(16, 248320, 2560, SMS) == (128, 256, 1, True, 8, 3)


def test_dense_wide_launch_keeps_the_candidate_on_the_safe_tile(monkeypatch):
    """The wide multirow call still launches the candidate, just on a shorter K tile.

    A resource failure has to be planned around, not caught: an OutOfResources
    swallowed here would silently put every wide projection back on W8A8.
    """
    _out, _x, (grid_n, splits), _args, cfg, _w, _s = _dense_launch(
        monkeypatch, 16, 12288, 2560
    )
    assert (cfg["BLOCK_N"], cfg["BLOCK_K"]) == (128, 128)
    assert cfg["SF_GROUP"] == SF and cfg["USE_DOT"] is True and cfg["M_PAD"] == 16
    assert cfg["EVEN_K"] is True and splits == 1 and grid_n == _cdiv(12288, 128)
    assert _m_pad(16, (128, 128, 1, True, 8, 3)) == cfg["M_PAD"]


def test_dense_split_k_scratch_is_per_call_site(monkeypatch):
    recorder = _patch_launcher(monkeypatch)
    device = torch.device("cpu")
    # The two linears that are captured on the main and the alt stream at once,
    # plus a third site to show the rotation is bounded by _N_SLOTS.
    sites = []
    for index, n in enumerate((1024, 64, 1024)):
        layer, weight, scale = object(), *_mxfp8_weight(n, 2560, seed=index)
        sites.append((layer, weight, scale))
    x = torch.zeros(16, 2560, dtype=torch.bfloat16)
    for layer, weight, scale in sites:
        sm120_w8a16_gemv.lowrow_mxfp8_gemv(x, weight, scale, layer)

    slots = [args[4] for _grid, args, _cfg in recorder.launches]
    assert slots[0] is sm120_w8a16_gemv._workspace_slot(device, 0)[0]
    assert slots[1] is sm120_w8a16_gemv._workspace_slot(device, 1)[0]
    # The pair that can be resident together never shares a counter array; the
    # third site wraps around to the first slot, which prealloc materialized.
    assert slots[0] is not slots[1]
    assert slots[2] is slots[0]

    # Stable for the same call site, so a captured graph keeps its own scratch.
    qkvz, qkvz_weight, qkvz_scale = sites[0]
    sm120_w8a16_gemv.lowrow_mxfp8_gemv(x, qkvz_weight, qkvz_scale, qkvz)
    assert recorder.launches[-1][1][4] is slots[0]

    # One slot in the pool: every site shares it, which is what the rotation
    # promises rather than a correctness surprise.
    _out, _x, _grid, args, _cfg, _w, _s = _dense_launch(
        monkeypatch, 16, 64, 2560, slots=1, owner=object()
    )
    assert args[4] is sm120_w8a16_gemv._workspace_slot(device, 0)[0]


class _SpyLinear:
    def __init__(self):
        self.calls = []
        self.result = torch.zeros(1, 1)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _fp8_method(backend=Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS):
    method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    method.use_marlin = False
    method.use_mxfp8 = True
    method.block_quant = True
    method.mxfp8_dense_backend = backend
    method.w8a8_mxfp8_linear = _SpyLinear()
    return method


class _Layer:
    def __init__(self, weight, scale):
        self.weight = weight
        self.weight_scale_inv = scale
        self.weight_scale_inv_swizzled = scale
        self.weight_scale_inv_shuffled = scale


def test_fp8_mxfp8_apply_routes_eligible_calls_and_falls_back(monkeypatch):
    recorder = _patch_launcher(monkeypatch)
    n, k = 128, 128
    weight, scale = _mxfp8_weight(n, k)
    layer = _Layer(weight, scale)
    method = _fp8_method()
    sentinel = method.w8a8_mxfp8_linear.result
    fallbacks = lambda: len(method.w8a8_mxfp8_linear.calls)

    # Eligible: 16 bf16 rows, no bias -> the candidate, and the W8A8 dispatch is
    # not reached at all.
    out = Fp8LinearMethod.apply(method, layer, torch.zeros(16, k, dtype=torch.bfloat16))
    assert out is not sentinel and out.shape == (16, n)
    assert fallbacks() == 0 and len(recorder.launches) == 1

    for x in (
        torch.zeros(24, k, dtype=torch.bfloat16),  # C6 verification
        torch.zeros(48, k, dtype=torch.bfloat16),  # mixed draft/verify batch
        torch.zeros(16, k, dtype=torch.float32),  # not the bf16 activation
    ):
        before = fallbacks()
        assert Fp8LinearMethod.apply(method, layer, x) is sentinel
        assert fallbacks() == before + 1

    # A bias belongs to the dispatch, not to the candidate.
    before = fallbacks()
    assert (
        Fp8LinearMethod.apply(
            method,
            layer,
            torch.zeros(4, k, dtype=torch.bfloat16),
            bias=torch.zeros(n, dtype=torch.bfloat16),
        )
        is sentinel
    )
    assert fallbacks() == before + 1

    # A pre-quantized (fp8 input, input scale) tuple is activation-quantized
    # already; the W8A16 candidate leaves it to the qualified path.
    before = fallbacks()
    quantized = (torch.zeros(4, k, dtype=torch.float8_e4m3fn), torch.zeros(4, 4))
    assert Fp8LinearMethod.apply(method, layer, quantized) is sentinel
    assert fallbacks() == before + 1

    # TRT-LLM permutes the weight rows at load time, so the natural [N, K]
    # ordering the kernel indexes no longer holds; that backend keeps its path.
    shuffled = _fp8_method(Mxfp8DenseGemmBackend.FLASHINFER_TRTLLM)
    shuffled_sentinel = shuffled.w8a8_mxfp8_linear.result
    assert (
        Fp8LinearMethod.apply(shuffled, layer, torch.zeros(4, k, dtype=torch.bfloat16))
        is shuffled_sentinel
    )
    assert len(shuffled.w8a8_mxfp8_linear.calls) == 1

    # Gate off: the dispatch behaves exactly as it did before the candidate.
    monkeypatch.delenv(ENV, raising=False)
    before = fallbacks()
    assert (
        Fp8LinearMethod.apply(method, layer, torch.zeros(4, k, dtype=torch.bfloat16))
        is sentinel
    )
    assert fallbacks() == before + 1


@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="run with TRITON_INTERPRET=1 to execute the kernel body on CPU",
)
def test_dense_kernel_indexing_matches_the_reference_dequant(monkeypatch):
    """Guard on the k // 32 scale gather, no GPU required.

    Triton's interpreter runs the same kernel body the device build compiles, so
    this exercises the real index arithmetic for the M == 1 (no tl.dot) tile.  A
    one-group scale error moves the result by ~50x, so it is a discriminating
    check of the indexing contract — and of nothing else: it says nothing about
    tl.dot, CUDA-graph replay or throughput on the SM120 part.
    """
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(sm120_w8a16_gemv, "_num_sms", lambda device: SMS)
    monkeypatch.setattr(sm120_w8a16_gemv, "_device_capability", lambda device: SM120)
    monkeypatch.setattr(sm120_w8a16_gemv, "_WS", {})
    monkeypatch.setattr(sm120_w8a16_gemv, "_LAYER_SLOTS", {})
    monkeypatch.setattr(sm120_w8a16_gemv, "_LAYER_ROTATION", 0)

    for n, k in ((96, 160), (320, 2560)):
        weight, scale = _mxfp8_weight(n, k, seed=n)
        x = torch.randn(1, k, dtype=torch.bfloat16)
        out = sm120_w8a16_gemv.lowrow_mxfp8_gemv(x, weight, scale)
        assert out is not None
        reference = (
            x.float()
            @ sm120_w8a16_gemv.dequantize_mxfp8_weight(
                weight, scale, dtype=torch.float32
            ).T
        )
        error = (out.float() - reference).abs() / (reference.abs() + 1e-6)
        assert error.max().item() < 2e-2, (n, k, error.max().item())

        shifted = torch.roll(scale, 1, dims=1)
        wrong = (
            x.float()
            @ sm120_w8a16_gemv.dequantize_mxfp8_weight(
                weight, shifted, dtype=torch.float32
            ).T
        )
        mismatch = (out.float() - wrong).abs() / (wrong.abs() + 1e-6)
        assert mismatch.max().item() > 1.0, (n, k, mismatch.max().item())
