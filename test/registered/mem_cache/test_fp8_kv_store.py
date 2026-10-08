"""Focused checks for the ported fused BF16 scale / E4M3 cast / NHD KV store.

``python/sglang/srt/mem_cache/fp8_kv_store.py`` and its ``set_kv_buffer`` hook in
``memory_pool.py`` are the donor's final implementation, so the donor's own check
is carried over as the numerical/byte-parity evidence (``test/srt/layers/
test_glue_d.py::test_fp8_kv_store_matches_reference`` in the donor checkout,
aiueo52): fused output compared as BYTES against torch's own
``clone() -> div_(scale) -> to(float8_e4m3fn)`` reference, over token-strided
(fused-QKV view) and contiguous sources with no / host-float / device-tensor
scales. A tolerance would accept exactly the rounding errors this kernel exists
to catch.

The CPU half checks the two things the port can get wrong without a device: the
dispatch gate in ``MHATokenToKVPool.set_kv_buffer`` (the serving FP8 pools'
eligible shapes, page64 included, reach the fused writer; every non-eligible
shape keeps the resident cast+scatter path), and the ``store_dtype`` re-typing in
``MHATokenToKVPool.__init__`` (native FP8 storage must not move a byte, a shape or
a descriptor that HiCache, PD/NIXL registration and CPU offload read).

The GPU half also replays the write through a CUDA graph with the runner's
repeated-zero graph-padding locations, and compares gate-on against gate-off
bytes through the real pool, so the reserved slot 0 that the resident writer
skips (``store_cache``'s ``reserved_skip_index=0``) stays untouched for the
fused writer too.
"""

from __future__ import annotations

import contextlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache import fp8_kv_store as fp8_kv_store_module
from sglang.srt.mem_cache import memory_pool as M
from sglang.srt.mem_cache.fp8_kv_store import fp8_kv_store
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")
register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEADS, HEAD_DIM, SLOTS, LAYERS = 4, 128, 64, 2
LOCAL_LAYER = 0  # layer_id_override 0 with start_layer 0 -> k_buffer[0]
FP8 = torch.float8_e4m3fn
GATE = "SGLANG_KV_FP8_FUSED_STORE"
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _pool(
    monkeypatch,
    *,
    gate: bool,
    device: str = "cpu",
    dtype: torch.dtype = FP8,
    page_size: int = 1,
    size: int = SLOTS,
    hnd: bool = False,
    kv_cache_layout: str | None = None,
) -> MHATokenToKVPool:
    """A real pool with (gate) or without (no gate) the donor's FP8 hooks."""
    if gate:
        monkeypatch.setenv(GATE, "1")
    else:
        monkeypatch.delenv(GATE, raising=False)
    if hnd:
        monkeypatch.setenv("SGLANG_USE_HND_KVCACHE", "1")
    else:
        monkeypatch.delenv("SGLANG_USE_HND_KVCACHE", raising=False)
    if device == "cpu":
        # The __init__ store_dtype gate is CUDA-only and the CPU suite runs with
        # no device; stand in for the serving platform so the statement under
        # test is the donor's own, not a copy of it.
        monkeypatch.setattr(M, "_is_cuda", gate)
    return MHATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=dtype,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        layer_num=LAYERS,
        device=device,
        enable_memory_saver=False,
        enable_alt_stream=False,
        kv_cache_layout=kv_cache_layout,
    )


def _kv(rows: int, *, strided: bool = False, dtype=torch.bfloat16, device="cpu"):
    if strided:
        fused = torch.randn((rows, 3 * HEADS * HEAD_DIM), dtype=dtype, device=device)
        return (
            fused[:, : HEADS * HEAD_DIM].view(rows, HEADS, HEAD_DIM),
            fused[:, HEADS * HEAD_DIM : 2 * HEADS * HEAD_DIM].view(
                rows, HEADS, HEAD_DIM
            ),
        )
    shape = (rows, HEADS, HEAD_DIM)
    return (
        torch.randn(shape, dtype=dtype, device=device),
        torch.randn(shape, dtype=dtype, device=device),
    )


def _loc(rows: int, *, dtype=torch.int64, device="cpu", offset: int = 1):
    # Slot 0 is the reserved padding slot; every index stays in [0, size + page_size).
    if offset + rows > SLOTS + 1:
        offset = 0
    return torch.arange(offset, offset + rows, dtype=dtype, device=device)


def _write(pool, cache_k, cache_v, loc, *, k_scale=None, v_scale=None, dcp=None):
    pool.set_kv_buffer(
        SimpleNamespace(layer_id=LOCAL_LAYER),
        loc,
        cache_k,
        cache_v,
        k_scale,
        v_scale,
        LOCAL_LAYER,
        dcp_kv_mask=dcp,
    )


class _Recorder:
    """Stands in for the kernel launch, which needs a device to compile."""

    def __init__(self, monkeypatch, pool: MHATokenToKVPool | None = None):
        self.calls: list[tuple[tuple, dict]] = []
        self.stored: list[int] = []
        monkeypatch.setattr(fp8_kv_store_module, "fp8_kv_store", self)
        if pool is not None:
            base = pool._store_kv_layer

            def spy(layer_idx, loc, cache_k, cache_v):
                self.stored.append(layer_idx)
                return base(layer_idx, loc, cache_k, cache_v)

            pool._store_kv_layer = spy

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


# ------------------------------------------------------------------------ CPU


def test_gate_defaults_off_and_retypes_only_native_fp8_nhd(monkeypatch):
    monkeypatch.delenv(GATE, raising=False)
    assert M.envs.SGLANG_KV_FP8_FUSED_STORE.get() is False

    off = _pool(monkeypatch, gate=False)
    assert off.dtype is FP8 and off.store_dtype is torch.uint8
    assert _pool(monkeypatch, gate=True).store_dtype is FP8

    # A non-FP8 cache dtype or a layout whose rows are not token slots keeps the
    # storage it had, so the fused writer never sees a buffer it cannot index.
    bf16 = _pool(monkeypatch, gate=True, dtype=torch.bfloat16)
    assert bf16.dtype is torch.bfloat16 and bf16.store_dtype is torch.bfloat16
    hnd = _pool(monkeypatch, gate=True, hnd=True)
    assert hnd.kv_cache_layout == "hnd" and hnd.store_dtype is torch.uint8
    # PageMajorMHATokenToKVPool hands this label to the base constructor, so the
    # gate sees a layout whose rows are pages, not tokens.
    page_major = _pool(monkeypatch, gate=True, kv_cache_layout="page_major_layer_major")
    assert page_major.kv_cache_layout == "page_major_layer_major"
    assert page_major.store_dtype is torch.uint8


@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("rows", [1, 16, 24, 64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
def test_eligible_shapes_dispatch(monkeypatch, page_size, rows, strided, loc_dtype):
    # page_size=64 is the serving profile's page granularity: it must not change
    # the per-token NHD row the writer indexes as loc * row_size.
    pool = _pool(monkeypatch, gate=True, page_size=page_size)
    rec = _Recorder(monkeypatch)
    cache_k, cache_v = _kv(rows, strided=strided)
    loc = _loc(rows, dtype=loc_dtype)
    _write(pool, cache_k, cache_v, loc)
    assert len(rec.calls) == 1
    args, kwargs = rec.calls[0]
    # (cache_k, cache_v, k_buffer[local], v_buffer[local], loc, k_scale, v_scale)
    assert args[0] is cache_k and args[1] is cache_v
    assert args[2] is pool.k_buffer[LOCAL_LAYER]
    assert args[3] is pool.v_buffer[LOCAL_LAYER]
    assert args[4] is loc
    assert args[5] is None and args[6] is None
    assert kwargs == {}


@pytest.mark.parametrize("scale", [None, 1.0, 0.7])
def test_scales_are_forwarded_untouched(monkeypatch, scale):
    # HybridLinearKVPool injects host 1.0 defaults; the donor applies the scale
    # inside the kernel instead of mutating the caller's tensors.
    pool = _pool(monkeypatch, gate=True)
    rec = _Recorder(monkeypatch)
    cache_k, cache_v = _kv(8)
    _write(pool, cache_k, cache_v, _loc(8), k_scale=scale, v_scale=scale)
    assert len(rec.calls) == 1
    assert rec.calls[0][0][5:] == (scale, scale)
    assert cache_k.stride(0) == HEADS * HEAD_DIM or cache_k.is_contiguous()


# HND and the DCP-masked write own their physical write before _store_kv_layer.
_SELF_WRITES = {"hnd", "dcp_mask"}


@pytest.mark.parametrize(
    "case",
    [
        "gate_off",
        "rows_0",
        "rows_65",
        "bf16_pool",
        "hnd",
        "vectorized_5d",
        "page_major_label",
        "already_fp8_source",
        "kv_shape_mismatch",
        "row_not_contiguous",
        "loc_not_1d",
        "loc_count_mismatch",
        "loc_float",
        "loc_not_contiguous",
        "dcp_mask",
    ],
)
def test_non_eligible_writes_keep_the_resident_path(monkeypatch, case):
    gate = case != "gate_off"
    dtype = torch.bfloat16 if case == "bf16_pool" else FP8
    pool = _pool(
        monkeypatch,
        gate=gate,
        dtype=dtype,
        hnd=case == "hnd",
        kv_cache_layout=(
            "page_major_layer_major" if case == "page_major_label" else None
        ),
    )
    if case == "vectorized_5d":
        # The AITER-only layout is chosen at construction from a ROCm-only env
        # and rebuilds its buffers as 5-D; the gate reads the same attribute the
        # donor's predicate does, so the label alone is what is under test here.
        pool.kv_cache_layout = "vectorized_5d"
    rec = _Recorder(monkeypatch, pool)
    rows = {"rows_0": 0, "rows_65": 65}.get(case, 8)
    cache_k, cache_v = _kv(rows, strided=case == "row_not_contiguous")
    if case == "row_not_contiguous":
        # Rows gathered out of a wider buffer: stride(1) != head_dim.
        cache_k = cache_k.transpose(1, 2)[..., ::2].transpose(1, 2)
    if case == "already_fp8_source":
        cache_k, cache_v = cache_k.to(FP8), cache_v.to(FP8)
    if case == "kv_shape_mismatch":
        cache_v = torch.randn((rows, HEADS, HEAD_DIM), dtype=torch.bfloat16) + 1.0
        cache_v = cache_v[:, : HEADS - 1]
    loc = _loc(rows)
    if case == "loc_not_1d":
        loc = loc.view(1, rows)
    elif case == "loc_count_mismatch":
        loc = _loc(rows + 1)
    elif case == "loc_float":
        loc = loc.to(torch.float32)
    elif case == "loc_not_contiguous":
        loc = _loc(rows * 2 + 1)[::2]
    dcp = torch.ones(rows, dtype=torch.int32) if case == "dcp_mask" else None

    with contextlib.suppress(Exception):
        # Deliberately malformed inputs (a 3-head V, a float loc, a gathered row)
        # are rejected by the resident path, which owns that contract unchanged;
        # what is pinned here is that the write reached the resident path at all.
        _write(pool, cache_k, cache_v, loc, dcp=dcp)
    assert rec.calls == [], f"{case} must not reach the fused writer"
    if case not in _SELF_WRITES:
        assert rec.stored == [LOCAL_LAYER], f"{case} must use the resident writer"


def test_resident_path_still_casts_and_scatters(monkeypatch):
    """The fallback the gate bypasses is the resident one, byte for byte."""
    pool = _pool(monkeypatch, gate=False)
    _Recorder(monkeypatch)  # never called; the fused writer must stay out
    rows = 8
    cache_k, cache_v = _kv(rows)
    loc = _loc(rows)
    _write(pool, cache_k, cache_v, loc, k_scale=1.0, v_scale=1.0)
    for buf, src in (
        (pool.k_buffer[LOCAL_LAYER], cache_k),
        (pool.v_buffer[LOCAL_LAYER], cache_v),
    ):
        expected = torch.zeros(buf.shape, dtype=torch.uint8, device=buf.device)
        expected[loc] = src.to(FP8).view(torch.uint8)
        assert torch.equal(buf.view(torch.uint8), expected)
    untouched = pool.k_buffer[LOCAL_LAYER + 1].view(torch.uint8)
    assert torch.equal(untouched, torch.zeros_like(untouched))


def test_store_dtype_retyping_leaves_the_cache_contracts_alone(monkeypatch):
    off, on = _pool(monkeypatch, gate=False), _pool(monkeypatch, gate=True)
    # HiCache / PD-NIXL / CPU-offload consumers read bytes, shapes and
    # descriptors, never the storage dtype label.
    assert [d.shape for d in off._kv_buffer_descs] == [
        d.shape for d in on._kv_buffer_descs
    ]
    assert [d.row_bytes for d in off._kv_buffer_descs] == [
        d.row_bytes for d in on._kv_buffer_descs
    ]
    assert [d.tokens_per_row for d in off._kv_buffer_descs] == [
        d.tokens_per_row for d in on._kv_buffer_descs
    ]
    assert off.get_contiguous_buf_infos()[1:] == on.get_contiguous_buf_infos()[1:]
    assert off.get_kv_buffer_shape() == on.get_kv_buffer_shape()
    assert off.get_kv_size_bytes() == on.get_kv_size_bytes()
    for buf_a, buf_b in zip(off.k_buffer + off.v_buffer, on.k_buffer + on.v_buffer):
        assert buf_a.shape == buf_b.shape
        assert buf_a.nbytes == buf_b.nbytes
        assert buf_a.element_size() == buf_b.element_size() == 1
    for layer in range(LAYERS):
        for get_buf in ("get_key_buffer", "get_value_buffer"):
            read_a, read_b = getattr(off, get_buf)(layer), getattr(on, get_buf)(layer)
            assert read_a.dtype is FP8 and read_b.dtype is FP8
            assert read_a.shape == read_b.shape


def test_kernel_input_validation_is_unchanged_from_the_donor():
    # Only the checks the donor makes before it touches a device are CPU-checkable.
    cache_k, cache_v = _kv(4)
    k_buffer = torch.zeros((SLOTS, HEADS, HEAD_DIM), dtype=FP8)
    v_buffer = torch.zeros_like(k_buffer)
    with pytest.raises(ValueError, match="3-D"):
        fp8_kv_store(cache_k[0], cache_v, k_buffer, v_buffer, _loc(4))
    with pytest.raises(ValueError, match="1 <= N <= 64"):
        fp8_kv_store(*_kv(65), k_buffer, v_buffer, _loc(65))
    with pytest.raises(ValueError, match="int32/int64"):
        fp8_kv_store(cache_k, cache_v, k_buffer, v_buffer, _loc(4, dtype=torch.uint8))
    with pytest.raises(ValueError, match="CUDA"):
        fp8_kv_store(cache_k, cache_v, k_buffer, v_buffer, _loc(4))


def test_the_reserved_slot_delta_is_the_only_kernel_change():
    """CPU pin for the one local delta, which otherwise only runs on a device."""
    src = inspect.getsource(fp8_kv_store_module)
    assert "store_mask = mask & (slot != 0)" in src
    assert "tl.store(k_cache_ptr + dst, k_fp8, mask=store_mask)" in src
    assert "tl.store(v_cache_ptr + dst, v_fp8, mask=store_mask)" in src
    assert src.count("mask=store_mask") == 2


# ---------------------------------------------------------------------- GPU


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 16])
@pytest.mark.parametrize("scale_kind", ["none", "float", "tensor"])
@pytest.mark.parametrize("strided", [False, True])
def test_fp8_kv_store_matches_reference(tokens: int, scale_kind: str, strided: bool):
    """The donor's parity check, carried over unchanged."""
    torch.manual_seed(2000 + tokens)
    shape = (tokens, 4, 128)
    if strided:
        # k/v as token-strided views of one fused [N, 3*H*D] buffer
        fused = torch.randn((tokens, 3 * 4 * 128), device="cuda", dtype=torch.bfloat16)
        cache_k = fused[:, : 4 * 128].view(tokens, 4, 128)
        cache_v = fused[:, 4 * 128 : 2 * 4 * 128].view(tokens, 4, 128)
    else:
        cache_k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        cache_v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    loc = torch.randperm(63, device="cuda", dtype=torch.int64)[:tokens] + 1
    # Slot 0 is the pool's reserved CUDA-graph padding slot (see the kernel's
    # store_mask note), so the fixture never treats it as data.
    k_buffer = torch.zeros((64, 4, 128), device="cuda", dtype=FP8)
    v_buffer = torch.zeros_like(k_buffer)
    ref_k = torch.zeros_like(k_buffer)
    ref_v = torch.zeros_like(v_buffer)

    if scale_kind == "none":
        k_scale = v_scale = None
    elif scale_kind == "float":
        k_scale = v_scale = 0.7
    else:
        k_scale = torch.tensor(0.7, device="cuda")
        v_scale = torch.tensor(0.7, device="cuda")

    ref_k_values = cache_k.clone()
    ref_v_values = cache_v.clone()
    if k_scale is not None:
        ref_k_values.div_(k_scale)
        ref_v_values.div_(v_scale)
    ref_k[loc] = ref_k_values.to(FP8)
    ref_v[loc] = ref_v_values.to(FP8)

    fp8_kv_store(cache_k, cache_v, k_buffer, v_buffer, loc, k_scale, v_scale)
    assert torch.equal(k_buffer.view(torch.uint8), ref_k.view(torch.uint8))
    assert torch.equal(v_buffer.view(torch.uint8), ref_v.view(torch.uint8))
    # The reserved slot stays reserved: no addressable row can reach it.
    for buf in (k_buffer, v_buffer):
        assert torch.equal(
            buf[0].view(torch.uint8), torch.zeros_like(buf[0].view(torch.uint8))
        )


@requires_cuda
@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("rows", [1, 24, 64, 65])
@pytest.mark.parametrize("strided", [False, True])
def test_pool_write_matches_the_resident_path_byte_for_byte(
    monkeypatch, page_size, rows, strided
):
    """Gate-on vs gate-off through the real pool, on the serving shapes.

    rows=65 is deliberately past the fused window (chunked prefill is), so the
    comparison covers the fused writer and the fallback that must stay identical.
    """
    if rows > 64 and strided:
        pytest.skip("the resident writer cannot view(-1, row_dim) a strided source")
    torch.manual_seed(7 + rows)
    # Each row needs its own destination: the valid slots are [0, size +
    # page_size) and slot 0 is reserved for graph padding, so the source row
    # count has to fit the addressable ones (65 rows need a 66-slot pool).
    size = max(SLOTS, rows + 1 - page_size)
    pools = [
        _pool(
            monkeypatch,
            gate=g,
            device="cuda",
            page_size=page_size,
            size=size,
        )
        for g in (False, True)
    ]
    cache_k, cache_v = (t.cuda() for t in _kv(rows, strided=strided))
    loc = (
        torch.randperm(size + page_size - 1, device="cuda", dtype=torch.int64)[:rows]
        + 1
    )
    if rows >= 3:
        # Padded batches repeat the reserved slot instead of dropping the row;
        # store_cache (reserved_skip_index=0) and the fused kernel must skip it
        # the same way.
        loc[-1] = 0
        loc[-2] = 0
    k_scale = v_scale = 1.0  # what HybridLinearKVPool injects
    zeros = torch.zeros_like(pools[0].k_buffer[LOCAL_LAYER].view(torch.uint8))
    for pool in pools:
        assert torch.equal(pool.k_buffer[LOCAL_LAYER].view(torch.uint8), zeros)
        _write(pool, cache_k, cache_v, loc, k_scale=k_scale, v_scale=v_scale)
    off, on = pools
    for buf_a, buf_b in zip(off.k_buffer + off.v_buffer, on.k_buffer + on.v_buffer):
        assert torch.equal(buf_a.view(torch.uint8), buf_b.view(torch.uint8))
    # Rows that were never addressed stay untouched, on both sides, and the
    # repeated padding location never wrote the reserved row.
    for pool in pools:
        assert torch.equal(pool.k_buffer[LOCAL_LAYER + 1].view(torch.uint8), zeros)
        assert torch.equal(pool.k_buffer[LOCAL_LAYER][0].view(torch.uint8), zeros[0])
        assert torch.equal(pool.v_buffer[LOCAL_LAYER][0].view(torch.uint8), zeros[0])


@requires_cuda
def test_graph_capture_and_replay_reproduce_the_bytes(monkeypatch):
    """The decode write is captured; replay must rewrite from the same buffers.

    The captured batch carries the runner's graph padding: two extra rows whose
    location repeats the reserved slot 0, which no replay may write into.
    """
    torch.manual_seed(11)
    real, pad = 24, 2
    pool = _pool(monkeypatch, gate=True, device="cuda")
    cache_k, cache_v = (t.cuda() for t in _kv(real + pad))
    loc = torch.cat(
        [
            _loc(real, device="cuda"),
            torch.zeros(pad, dtype=torch.int64, device="cuda"),
        ]
    )
    buffer_row = pool.k_buffer[LOCAL_LAYER]
    row0 = buffer_row[0].view(torch.uint8)

    # Warm the kernel outside capture, then capture the pool's own write path.
    _write(pool, cache_k, cache_v, loc, k_scale=1.0, v_scale=1.0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _write(pool, cache_k, cache_v, loc, k_scale=1.0, v_scale=1.0)
    torch.cuda.synchronize()
    buffer_row.zero_()
    row0.fill_(77)  # poison: the padding rows must leave the reserved row alone

    graph.replay()
    torch.cuda.synchronize()
    first = buffer_row.view(torch.uint8).clone()
    assert torch.equal(row0, torch.full_like(row0, 77))
    _check_rows(first, cache_k[:real], loc[:real])

    # Same graph, same locations, new activations -> bytes follow the source.
    cache_k.copy_(torch.randn_like(cache_k))
    graph.replay()
    torch.cuda.synchronize()
    second = buffer_row.view(torch.uint8).clone()
    assert not torch.equal(first[1:], second[1:])
    assert torch.equal(row0, torch.full_like(row0, 77))
    _check_rows(second, cache_k[:real], loc[:real])


def _check_rows(stored: torch.Tensor, cache_k: torch.Tensor, loc: torch.Tensor) -> None:
    """Every addressable row matches torch's own cast at its location."""
    ref = _reference_rows(cache_k, loc, tuple(stored.shape)).view(torch.uint8)
    assert torch.equal(stored[1:], ref[1:])
    assert not torch.equal(stored[1:], torch.zeros_like(stored[1:]))


def _reference_rows(cache_k, loc, shape) -> torch.Tensor:
    ref = torch.zeros(shape, device=cache_k.device, dtype=FP8)
    ref[loc] = cache_k.to(FP8)
    return ref
