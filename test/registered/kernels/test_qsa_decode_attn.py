"""Contract checks for the optional QSA split-KV decode attention.

CPU: shape gating, launch/workspace bounds, workspace ownership and the
backend plumbing (which args reach the kernel, and the prefix-scan decision).
The numerical, CUDA-graph replay and TP-shape checks of the Triton kernel
itself need SM120 hardware: they are prepared here and skipped without it.
"""

import math
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.decode_attn import (
    _BLOCK_H,
    _HEAD_DIM,
    _MAX_ROW_SPLITS,
    QSADecodeAttnWorkspace,
    _launch_config,
    qsa_decode_attention,
    qsa_decode_attention_supported,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QSAMTPSharedSparseIndices,
    QwenSparseAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_KV_HEADS = 2
_Q_HEADS = 12  # TP2-local C6 heads; TP1 is (4, 24), same GQA of 6
_TOPK = 32
_BACKEND = "sglang.srt.layers.attention.qwen_sparse_attn_backend"


def _measured_shape(rows=4, kv_heads=_KV_HEADS, q_heads=_Q_HEADS, head_dim=_HEAD_DIM):
    """The gate's accept-case tensors (dtype/layout/widths are what is checked)."""
    return dict(
        q=torch.zeros(rows, q_heads, head_dim, dtype=torch.bfloat16),
        k_buffer=torch.zeros(64, kv_heads, head_dim, dtype=torch.float8_e4m3fn),
        v_buffer=torch.zeros(64, kv_heads, head_dim, dtype=torch.float8_e4m3fn),
        topk_indices=torch.arange(rows * _TOPK, dtype=torch.int32).view(rows, _TOPK)
        % 32,
    )


def _make_backend(triton_decode_attn=None, state=None):
    runner = SimpleNamespace(
        device="cpu",
        token_to_kv_pool=None,
        req_to_token_pool=None,
        model_config=SimpleNamespace(
            context_len=64, hf_config=SimpleNamespace(indexer_compress_ratio=4)
        ),
    )
    backend = QwenSparseAttnBackend(runner)
    if triton_decode_attn is not None:
        backend._triton_decode_attn = triton_decode_attn
    if state is not None:
        backend.set_mtp_shared_sparse_indices(state)
    return backend


def _make_state(
    triton_decode_attn,
    num_requests=4,
    tail_width=4,
    token_topk=_TOPK,
    device="cpu",
):
    with envs.SGLANG_OPT_TRITON_DECODE_ATTN.override(triton_decode_attn):
        return QSAMTPSharedSparseIndices(
            layer_ids=[48],
            num_requests=num_requests,
            token_topk=token_topk,
            tail_width=tail_width,
            device=device,
        )


def _run_triton_decode(
    backend, rows, device="cpu", mode=ForwardMode.DECODE, capturing=False
):
    """Call the real _forward_triton_decode, returning (out, kernel kwargs).

    The capture probe is pinned: a host with a CUDA build of torch and no
    driver cannot answer it at all.
    """
    tensors = _measured_shape(rows=rows)
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return torch.zeros(rows, _Q_HEADS, _HEAD_DIM, dtype=torch.bfloat16)

    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros(8, 64, dtype=torch.int32, device=device)
    )
    forward_batch = SimpleNamespace(
        forward_mode=mode,
        req_pool_indices=torch.arange(rows, dtype=torch.int32, device=device),
    )
    metadata = SimpleNamespace(
        sequence_lengths=torch.full((rows,), 32, dtype=torch.int32, device=device),
        row_req_pool_indices=torch.arange(rows, dtype=torch.int32, device=device),
    )
    with mock.patch(
        "torch.cuda.is_current_stream_capturing", return_value=capturing
    ), mock.patch(f"{_BACKEND}.qsa_decode_attention", fake):
        out = QwenSparseAttnBackend._forward_triton_decode(
            backend,
            q=tensors["q"],
            k_buffer=tensors["k_buffer"],
            v_buffer=tensors["v_buffer"],
            layer=SimpleNamespace(layer_id=48, scaling=0.0625),
            forward_batch=forward_batch,
            metadata=metadata,
            topk_indices=tensors["topk_indices"],
        )
    return out, seen


# --- CPU: shape gate ------------------------------------------------------


def test_supported_gate_accepts_the_measured_fp8_shape():
    assert qsa_decode_attention_supported(**_measured_shape())


@pytest.mark.parametrize(
    "mutate,why",
    [
        (lambda t: t.update(q=t["q"].to(torch.float16)), "q is not BF16"),
        (
            lambda t: t.update(k_buffer=t["k_buffer"].to(torch.bfloat16)),
            "K is not E4M3",
        ),
        (
            lambda t: t.update(v_buffer=t["v_buffer"].to(torch.bfloat16)),
            "V/K dtype mismatch",
        ),
        (
            lambda t: t.update(
                q=torch.zeros(4, 20, _HEAD_DIM, dtype=torch.bfloat16),
                k_buffer=torch.zeros(64, 1, _HEAD_DIM, dtype=torch.float8_e4m3fn),
                v_buffer=torch.zeros(64, 1, _HEAD_DIM, dtype=torch.float8_e4m3fn),
            ),
            "GQA is wider than the 16 MMA rows",
        ),
        (
            lambda t: t.update(
                q=torch.zeros(4, _Q_HEADS, 128, dtype=torch.bfloat16),
                k_buffer=torch.zeros(64, _KV_HEADS, 128, dtype=torch.float8_e4m3fn),
                v_buffer=torch.zeros(64, _KV_HEADS, 128, dtype=torch.float8_e4m3fn),
            ),
            "head_dim is not 256",
        ),
        (
            lambda t: t.update(
                q=torch.zeros(4, 11, _HEAD_DIM, dtype=torch.bfloat16),
                k_buffer=torch.zeros(64, 2, _HEAD_DIM, dtype=torch.float8_e4m3fn),
                v_buffer=torch.zeros(64, 2, _HEAD_DIM, dtype=torch.float8_e4m3fn),
            ),
            "query heads do not divide into kv heads",
        ),
        (
            lambda t: t.update(
                q=torch.zeros(3, _Q_HEADS, _HEAD_DIM, dtype=torch.bfloat16)
            ),
            "top-k rows do not match q rows",
        ),
        (
            lambda t: t.update(
                k_buffer=torch.zeros(64, 2, _HEAD_DIM * 2, dtype=torch.float8_e4m3fn)[
                    :, :, :_HEAD_DIM
                ],
                v_buffer=torch.zeros(
                    64, _KV_HEADS, _HEAD_DIM, dtype=torch.float8_e4m3fn
                ),
            ),
            "the KV pool is not contiguous",
        ),
    ],
)
def test_supported_gate_rejects_unmeasured_shapes(mutate, why):
    tensors = _measured_shape()
    mutate(tensors)
    assert not qsa_decode_attention_supported(**tensors), why


# --- CPU: launch configuration / workspace ownership ----------------------


def test_launch_config_splits_always_fit_the_workspace():
    """A fixed-capacity workspace is only sound if no *split* launch can ask
    for more partials than it holds, the past-the-buckets shrink included.  One
    split stores straight to the output and touches no partial at all."""
    for rows in range(1, 1025):
        num_splits = _launch_config(rows)[0]
        assert num_splits >= 1
        if num_splits > 1:
            assert rows * num_splits <= _MAX_ROW_SPLITS, rows
        if rows > 1:
            assert _launch_config(rows - 1)[0] >= num_splits, rows


def test_workspace_holds_every_partial_and_starts_zeroed():
    workspace = QSADecodeAttnWorkspace(
        num_kv_heads=_KV_HEADS, head_dim=_HEAD_DIM, device="cpu"
    )
    assert workspace.max_parts == _MAX_ROW_SPLITS * _KV_HEADS
    assert workspace.partial_out.shape == (workspace.max_parts * _BLOCK_H * _HEAD_DIM,)
    assert workspace.partial_out.dtype == torch.float16
    # The combine reads the normalization sum and the maximum of every split, so
    # they are separate fields -- a folded lse cannot express per-split weight.
    for name in ("partial_l", "partial_max"):
        field = getattr(workspace, name)
        assert field.shape == (workspace.max_parts * _BLOCK_H,), name
        assert field.dtype == torch.float32, name
    # The kernel re-zeroes its own counters, so a non-zero counter at entry can
    # only mean a previous launch died mid-flight.
    assert workspace.arrivals.shape == (workspace.max_parts,)
    assert workspace.arrivals.dtype == torch.int32
    assert torch.count_nonzero(workspace.arrivals) == 0


def test_split_launch_refuses_an_undersized_workspace_before_dispatch():
    """One too-small workspace must be a loud error, not an out-of-bounds write
    into a neighbouring allocation."""
    tensors = _measured_shape(rows=4)
    workspace = QSADecodeAttnWorkspace(
        num_kv_heads=_KV_HEADS, head_dim=_HEAD_DIM, device="cpu"
    )
    workspace.max_parts = 1
    with pytest.raises(ValueError, match="workspace too small"):
        qsa_decode_attention(
            q=tensors["q"],
            k_buffer=tensors["k_buffer"],
            v_buffer=tensors["v_buffer"],
            req_to_token=torch.zeros(8, 64, dtype=torch.int32),
            row_req_pool_indices=torch.arange(4, dtype=torch.int32),
            topk_indices=tensors["topk_indices"],
            seq_lens=torch.full((4,), 8, dtype=torch.int32),
            sm_scale=0.0625,
            workspace=workspace,
            prefix_valid=True,
        )


def test_gate_is_off_by_default_and_allocates_nothing():
    """Default-off: no workspace, no behaviour change for the running C6/27B
    configurations, and the shared state keeps its existing row layout."""
    assert not envs.SGLANG_OPT_TRITON_DECODE_ATTN.get()
    backend = _make_backend()
    assert backend._triton_decode_attn is False
    assert backend._decode_attn_workspace is None
    assert _make_state(False).shared_tail_prefix is False


def test_kernel_receives_the_row_lengths_and_row_request_rows():
    """Rows are addressed by their own request row and their own length (the
    speculative layout can repeat or pad requests), never by the batch size."""
    backend = _make_backend(triton_decode_attn=True)
    out, seen = _run_triton_decode(backend, rows=4)
    assert out.shape == (4, _Q_HEADS * _HEAD_DIM)
    assert seen["seq_lens"].shape == (4,)
    assert seen["row_req_pool_indices"].shape == (4,)
    assert seen["sm_scale"] == 0.0625
    assert seen["workspace"] is backend._decode_attn_workspace


def test_workspace_is_allocated_once_and_never_inside_a_capture():
    backend = _make_backend(triton_decode_attn=True)
    _out, _seen = _run_triton_decode(backend, rows=1)
    workspace = backend._decode_attn_workspace
    assert workspace is not None
    # The capture proper must reuse the warmup's storage: the graph bakes the
    # partial/counter addresses in.
    _run_triton_decode(backend, rows=1)
    assert backend._decode_attn_workspace is workspace
    backend._decode_attn_workspace = None
    with pytest.raises(RuntimeError, match="workspace not allocated"):
        _run_triton_decode(backend, rows=1, capturing=True)


def test_prefix_valid_follows_the_shared_state_layout():
    """The kernel may bound its scan at seq_lens only when every valid column
    sits in front: index-shared rows say so through the same gate that packs
    them, so a hole-ridden row is scanned in full instead of silently losing
    the drafted tail.  Users cannot pair the two differently."""
    for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
        for gate in (True, False):
            state = _make_state(gate)
            assert state.shared_tail_prefix is gate
            backend = _make_backend(triton_decode_attn=True, state=state)
            _out, seen = _run_triton_decode(backend, rows=1, mode=mode)
            shares_draft_selection = mode is ForwardMode.DECODE
            assert seen["prefix_valid"] is (gate or not shares_draft_selection)


def test_reference_attention_follows_its_host_side_case_spec():
    """The GPU numerical reference builds its gather from a HOST case spec
    (indices, lengths, request rows, req_to_token) and only moves the resolved
    slot ids to the buffers' device.  Indexing the host map with device indices
    raised on the first real run, and an identity map could never show a wrong
    request row -- both are checked here where the code actually executes."""
    case = _long_case(rows=3, cols=64, holes=True)
    for name in ("req_to_token", "topk_indices", "seq_lens", "row_req_pool_indices"):
        assert not case[name].is_cuda and case[name].device.type == "cpu", name
    q = torch.randn(3, _Q_HEADS, _HEAD_DIM, dtype=torch.bfloat16)
    k_buffer = torch.randn(case["n_slots"], _KV_HEADS, _HEAD_DIM).to(
        torch.float8_e4m3fn
    )
    v_buffer = torch.randn(case["n_slots"], _KV_HEADS, _HEAD_DIM).to(
        torch.float8_e4m3fn
    )
    out = _reference_attention(q, k_buffer, v_buffer, case)
    assert out.shape == q.shape
    # The last case row selects nothing: a plain zero output, not a stale one.
    assert torch.count_nonzero(out[-1]) == 0
    # Two rows with the SAME selection but different request rows must differ,
    # because req_to_token is deliberately not the identity map.
    twin = _long_case(rows=2, cols=64, holes=False)
    twin["topk_indices"][1] = twin["topk_indices"][0]
    twin["seq_lens"][1] = twin["seq_lens"][0]
    assert int(twin["row_req_pool_indices"][0]) != int(twin["row_req_pool_indices"][1])
    twin_q = torch.randn(2, _Q_HEADS, _HEAD_DIM, dtype=torch.bfloat16)
    twin_q[1] = twin_q[0]
    twin_out = _reference_attention(twin_q, k_buffer, v_buffer, twin)
    assert not torch.equal(twin_out[0], twin_out[1])
    # And the resolved slots really are the map's own, not the logical index.
    req0 = int(twin["row_req_pool_indices"][0])
    length0 = int(twin["seq_lens"][0])
    cols = [int(c) for c in twin["topk_indices"][0] if 0 <= int(c) < length0]
    slots = [int(twin["req_to_token"][req0][c]) for c in cols]
    assert slots != cols
    scores = (twin_q[0, 0].float() @ k_buffer[slots].float()[:, 0].t()) / (
        _HEAD_DIM**0.5
    )
    manual = torch.softmax(scores, -1) @ v_buffer[slots].float()[:, 0]
    torch.testing.assert_close(twin_out[0, 0].float(), manual, rtol=2e-2, atol=2e-3)


def test_prepared_numerical_cases_really_launch_several_splits():
    """The GPU numerical cases are the only place the split-KV combine is
    compared against a reference, so they must not silently collapse to a
    single non-empty split (where the combine is dead code and the test would
    pass for the wrong reason).  Checked on CPU, where it cannot be skipped."""
    for rows, _kv, _q, cols, holes in _CASES:
        num_splits, block_n = _launch_config(rows)[0], _launch_config(rows)[1]
        case = _long_case(rows=rows, cols=cols, holes=holes)
        ncols = cols if holes else min(int(case["seq_lens"].max()), cols)
        assert num_splits > 1, (rows, cols, num_splits)
        assert _nonempty_splits(ncols, num_splits, block_n) >= 4, (
            rows,
            cols,
            num_splits,
            block_n,
        )
        assert torch.count_nonzero(case["topk_indices"] >= 0) > 0


# --- GPU (prepared: skipped without CUDA hardware) ------------------------

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# indexer_budget 2048 + compress_ratio - 1: the column width the split-KV launch
# buckets were measured at, and the width at which _launch_config actually hands
# out more than one split.  The short variant keeps the TP-local head coverage
# cheap.
_TOPK_COLS = 2051
_SHORT_COLS = 512


def _row_spec(row, rows, cols, holes):
    """``(seq_len, selected logical indices)`` for one case row: the first row
    sees a single token, the last selects nothing (the zero-output row must not
    poison the combine), the rest are ragged and long enough to span several
    split chunks.  ``holes`` reproduces an index-shared row: a -1 mid-row with a
    valid drafted-tail column behind it, which the scan-bound path may not skip.
    """
    if rows > 1 and row == 0:
        length, width = 1, 1
    elif rows > 2 and row == rows - 1:
        length, width = 8, 0
    else:
        length = 2048 + 3 * (row % 4)
        width = min(cols, length, 2048 - 7 * (row % 9))
    columns = list(range(width))
    if holes and width > 8:
        columns[3] = -1
        columns[-1] = length - 1  # the tail column hides behind the hole
    return length, columns


def _long_case(rows, cols=_TOPK_COLS, holes=False):
    """Host-side case spec: ragged rows, and per-request ``req_to_token`` maps
    with no fixed point (``2c+1`` / ``2c+2``), so a row that resolved the wrong
    request -- or used the logical index as a slot -- reads different memory.
    """
    topk_indices = torch.full((rows, cols), -1, dtype=torch.int32)
    seq_lens = torch.empty(rows, dtype=torch.int32)
    row_req_pool_indices = torch.tensor(
        [row % 2 for row in range(rows)], dtype=torch.int32
    )
    width = 1
    for row in range(rows):
        length, columns = _row_spec(row, rows, cols, holes)
        seq_lens[row] = length
        width = max(width, length)
        assert len(columns) <= cols, "case columns must fit the row width"
        if columns:
            topk_indices[row, : len(columns)] = torch.tensor(columns, dtype=torch.int32)
    logical = torch.arange(width, dtype=torch.int32)
    req_to_token = torch.stack([2 * logical + 1, 2 * (logical + 1)])
    return dict(
        req_to_token=req_to_token,
        topk_indices=topk_indices,
        seq_lens=seq_lens,
        row_req_pool_indices=row_req_pool_indices,
        n_slots=2 * width + 2,
    )


def _reference_attention(q, k_buffer, v_buffer, case):
    """Plain fp32 softmax over exactly the selected columns, no split-KV.

    ``case`` is a HOST spec (the indices, lengths, request rows and the
    req_to_token map are read as integers here); only the resolved slot ids move
    to the buffers' device, so this runs on CPU as well as next to CUDA tensors.
    """
    device = k_buffer.device
    scale = 1.0 / (k_buffer.shape[-1] ** 0.5)
    heads_per_kv = q.shape[1] // k_buffer.shape[1]
    out = torch.zeros(q.shape, dtype=torch.float32, device=device)
    for row in range(q.shape[0]):
        req = int(case["row_req_pool_indices"][row])
        length = int(case["seq_lens"][row])
        cols = [int(c) for c in case["topk_indices"][row] if 0 <= int(c) < length]
        if not cols:
            continue
        index = torch.tensor(cols, dtype=torch.long)
        slots = case["req_to_token"][req].long()[index].to(device)
        k = k_buffer[slots].float()  # [n, kv heads, D]
        v = v_buffer[slots].float()
        for kv_head in range(k_buffer.shape[1]):
            qs = q[row, kv_head * heads_per_kv : (kv_head + 1) * heads_per_kv].float()
            scores = (qs @ k[:, kv_head].t()) * scale
            out[row, kv_head * heads_per_kv : (kv_head + 1) * heads_per_kv] = (
                torch.softmax(scores, -1) @ v[:, kv_head]
            )
    return out.to(q.dtype)


def _nonempty_splits(ncols, num_splits, block_n):
    """How many programs of one row's launch actually get columns, mirroring the
    kernel's ``chunk = cdiv(cdiv(ncols, NUM_SPLITS), BLOCK_N) * BLOCK_N``.  The
    numerical checks are worthless if this is 1: the combine would be dead code.
    """
    chunk = math.ceil(math.ceil(ncols / num_splits) / block_n) * block_n
    return sum(
        1 for s in range(num_splits) if min(s * chunk + chunk, ncols) > s * chunk
    )


# (rows, kv heads, q heads, columns, holes).  24 rows is the C6/W4 target-verify
# batch, 1 row the single-request decode; the last two keep TP-local (TP4) and
# model-level (TP1) head splits covered without a Cartesian suite.
_CASES = [
    (1, _KV_HEADS, _Q_HEADS, _TOPK_COLS, False),
    (4, _KV_HEADS, _Q_HEADS, _TOPK_COLS, False),
    (8, _KV_HEADS, _Q_HEADS, _TOPK_COLS, True),
    (24, _KV_HEADS, _Q_HEADS, _TOPK_COLS, True),
    (8, 1, 6, _SHORT_COLS, False),
    (8, 4, 24, _SHORT_COLS, False),
]


@_CUDA
@pytest.mark.parametrize("rows,kv_heads,q_heads,cols,holes", _CASES)
def test_kernel_matches_reference_at_the_measured_width(
    rows, kv_heads, q_heads, cols, holes
):
    """Every selected contribution survives the split-KV combine -- ragged
    lengths, a one-token row, an empty row, the drafted tail behind a -1 hole and
    the TP-local head splits -- and a relaunch off the same partials is stable.
    """
    torch.manual_seed(0)
    device = "cuda"
    case = _long_case(rows=rows, cols=cols, holes=holes)
    num_splits, block_n = _launch_config(rows)[0], _launch_config(rows)[1]
    # Holey rows are scanned in full (NCOLS), prefix rows stop at seq_lens.
    ncols = cols if holes else min(int(case["seq_lens"].max()), cols)
    assert (
        num_splits > 1 and _nonempty_splits(ncols, num_splits, block_n) >= 4
    ), f"rows={rows} cols={cols}: {num_splits} splits x {block_n} merges nothing"
    q = torch.randn(rows, q_heads, _HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_buffer = torch.randn(case["n_slots"], kv_heads, _HEAD_DIM, device=device).to(
        torch.float8_e4m3fn
    )
    v_buffer = torch.randn(case["n_slots"], kv_heads, _HEAD_DIM, device=device).to(
        torch.float8_e4m3fn
    )
    args = dict(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        req_to_token=case["req_to_token"].to(device),
        row_req_pool_indices=case["row_req_pool_indices"].to(device),
        topk_indices=case["topk_indices"].to(device),
        seq_lens=case["seq_lens"].to(device),
        sm_scale=1.0 / (_HEAD_DIM**0.5),
        prefix_valid=not holes,
    )
    expected = _reference_attention(q, k_buffer, v_buffer, case)
    assert torch.count_nonzero(expected) > 0, "case is vacuously zero"
    workspace = QSADecodeAttnWorkspace(
        num_kv_heads=kv_heads, head_dim=_HEAD_DIM, device=device
    )
    for _ in range(2):
        out = qsa_decode_attention(workspace=workspace, **args)
        assert out.shape == q.shape and out.dtype == q.dtype
        torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-3)
        # The last arriver must hand its counters back empty for the next call.
        assert torch.count_nonzero(workspace.arrivals) == 0


# --- split-normalization contract (device-parameterised) ------------------

# Triton's interpreter runs the real kernel body on CPU; the mode is read when
# triton is imported, so the checks below run in a child process.
_INTERPRET_MODE = os.environ.get("TRITON_INTERPRET") == "1"


def _check_equal_population_split_combine(device):
    """Identical keys tie every selected score, so the softmax is exactly uniform
    and the answer is the fraction of selected columns carrying V=1. A combine
    that weights SPLITS instead of columns -- what a stored lse becomes once
    log2(l) rounds away inside m -- returns 1/22 here instead of 35/2051.
    """
    rows, cols, tail = 1, 2051, 35
    kv_heads, q_heads = 2, 4
    num_splits, block_n = _launch_config(rows)[0], _launch_config(rows)[1]
    chunk = math.ceil(math.ceil(cols / num_splits) / block_n) * block_n
    assert _nonempty_splits(cols, num_splits, block_n) == num_splits, num_splits
    # 21 full splits and one 35-column split: the populations really differ.
    assert (num_splits - 1) * chunk == cols - tail, (num_splits, chunk)
    logical = torch.arange(cols, dtype=torch.int32)
    req_to_token = logical.unsqueeze(0).repeat(2, 1).contiguous()
    topk_indices = logical.unsqueeze(0).clone()
    seq_lens = torch.full((rows,), cols, dtype=torch.int32)
    row_reqs = torch.zeros(rows, dtype=torch.int32)
    k_buffer = torch.ones(cols, kv_heads, _HEAD_DIM).to(torch.float8_e4m3fn)
    v_dense = torch.zeros(cols, kv_heads, _HEAD_DIM)
    v_dense[cols - tail :] = 1.0
    v_buffer = v_dense.to(torch.float8_e4m3fn)
    q = torch.full(
        (rows, q_heads, _HEAD_DIM), 1.0e30, dtype=torch.bfloat16, device=device
    )
    workspace = QSADecodeAttnWorkspace(
        num_kv_heads=kv_heads, head_dim=_HEAD_DIM, device=device
    )
    out = qsa_decode_attention(
        q=q,
        k_buffer=k_buffer.to(device),
        v_buffer=v_buffer.to(device),
        req_to_token=req_to_token.to(device),
        row_req_pool_indices=row_reqs.to(device),
        topk_indices=topk_indices.to(device),
        seq_lens=seq_lens.to(device),
        sm_scale=1.0 / (_HEAD_DIM**0.5),
        workspace=workspace,
        prefix_valid=True,
    )
    assert torch.isfinite(out.float()).all()
    expected = torch.full(out.shape, tail / cols, dtype=torch.float32, device=device)
    # Tight enough that the split-weighted 1/22 answer cannot pass.
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(
        out.float(),
        _reference_attention(
            q,
            k_buffer.to(device),
            v_buffer.to(device),
            {
                "req_to_token": req_to_token,
                "topk_indices": topk_indices,
                "seq_lens": seq_lens,
                "row_req_pool_indices": row_reqs,
            },
        ).float(),
        rtol=1e-2,
        atol=2e-3,
    )
    assert torch.count_nonzero(workspace.arrivals) == 0


def _check_negative_finite_scores_survive(device):
    """Q=1e30 against K=-1 at scale 1/16 gives log2 scores near -2.3e31, i.e.
    below the -1e30 the masking used to wear: the row must still average to V
    (= 1.0), while a row with nothing selected stays exactly 0, never NaN.
    """
    rows, cols = 2, 2051
    kv_heads, q_heads = 2, 4
    selected = 128
    logical = torch.arange(cols, dtype=torch.int32)
    req_to_token = logical.unsqueeze(0).repeat(2, 1).contiguous()
    topk_indices = torch.full((rows, cols), -1, dtype=torch.int32)
    topk_indices[0, :selected] = logical[:selected]
    seq_lens = torch.tensor([selected, 8], dtype=torch.int32)
    row_reqs = torch.zeros(rows, dtype=torch.int32)
    k_buffer = torch.full((cols, kv_heads, _HEAD_DIM), -1.0).to(torch.float8_e4m3fn)
    v_buffer = torch.ones(cols, kv_heads, _HEAD_DIM).to(torch.float8_e4m3fn)
    q = torch.full(
        (rows, q_heads, _HEAD_DIM), 1.0e30, dtype=torch.bfloat16, device=device
    )
    workspace = QSADecodeAttnWorkspace(
        num_kv_heads=kv_heads, head_dim=_HEAD_DIM, device=device
    )
    out = qsa_decode_attention(
        q=q,
        k_buffer=k_buffer.to(device),
        v_buffer=v_buffer.to(device),
        req_to_token=req_to_token.to(device),
        row_req_pool_indices=row_reqs.to(device),
        topk_indices=topk_indices.to(device),
        seq_lens=seq_lens.to(device),
        sm_scale=0.0625,
        workspace=workspace,
        prefix_valid=True,
    )
    assert torch.isfinite(out.float()).all(), "negative scores vanished"
    torch.testing.assert_close(
        out[0].float(), torch.ones_like(out[0].float()), rtol=1e-2, atol=2e-3
    )
    assert torch.equal(out[1], torch.zeros_like(out[1])), "empty selection not zeroed"
    assert torch.count_nonzero(workspace.arrivals) == 0


@_CUDA
def test_equal_population_split_combine_on_gpu():
    _check_equal_population_split_combine("cuda")


@_CUDA
def test_negative_finite_scores_on_gpu():
    _check_negative_finite_scores_survive("cuda")


@pytest.mark.skipif(
    not _INTERPRET_MODE, reason="runs inside the TRITON_INTERPRET child"
)
def test_interpret_contract_equal_population_split_combine():
    _check_equal_population_split_combine("cpu")


@pytest.mark.skipif(
    not _INTERPRET_MODE, reason="runs inside the TRITON_INTERPRET child"
)
def test_interpret_contract_negative_finite_scores():
    _check_negative_finite_scores_survive("cpu")


@pytest.mark.skipif(_INTERPRET_MODE, reason="this is the interpreter child")
def test_split_normalization_contract_runs_in_triton_interpreter():
    """The split combine and the masked-score handling are pure kernel-body
    arithmetic, so Triton's interpreter can check them without a GPU.  The mode
    must be set before triton is imported, hence the child process; the kernel
    still needs SM120 hardware for the MMA dtypes and the timing geometry."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        __file__,
        "-k",
        "interpret_contract",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
    ]
    # Hand the child this process' import path: under CI sglang is installed and
    # sys.executable finds it on its own, but a source checkout run (or a venv
    # that is not the one pytest was started from) would otherwise lose it.
    child_env = {**os.environ, "TRITON_INTERPRET": "1"}
    inherited = [p for p in sys.path if p and os.path.isdir(p)]
    if inherited:
        existing = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = os.pathsep.join(
            [p for p in [*inherited, existing] if p]
        )
    try:
        child = subprocess.run(
            command,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - hung build
        raise AssertionError(f"triton interpreter child timed out: {exc}") from exc
    assert child.returncode == 0, (
        f"interpreter child failed ({child.returncode}):\n"
        f"{child.stdout[-4000:]}\n{child.stderr[-2000:]}"
    )


@_CUDA
def test_finite_bf16_queries_above_the_f16_max_stay_finite():
    """The QK MMA consumes Q as BF16.  Casting it (and E4M3 K) to F16 overflowed
    a finite query above 65504 to Inf, and Inf - Inf then NaN'ed the softmax
    where the resident BF16 path stays finite; the fp32 accumulation of such
    huge scores still has to agree with the reference.
    """
    torch.manual_seed(0)
    device = "cuda"
    rows = 4
    case = _long_case(rows=rows, cols=_TOPK_COLS, holes=False)
    q = torch.full(
        (rows, _Q_HEADS, _HEAD_DIM), 65536.0, dtype=torch.bfloat16, device=device
    )
    q[1] = -65536.0
    q[2] = 1.0e30  # finite in BF16, Inf in F16; scores stay inside fp32 range
    q[3] = 131072.0
    assert float(q.abs().max()) > torch.finfo(torch.float16).max
    assert torch.isfinite(q.float()).all()
    k_buffer = torch.randn(case["n_slots"], _KV_HEADS, _HEAD_DIM, device=device).to(
        torch.float8_e4m3fn
    )
    v_buffer = torch.randn(case["n_slots"], _KV_HEADS, _HEAD_DIM, device=device).to(
        torch.float8_e4m3fn
    )
    workspace = QSADecodeAttnWorkspace(
        num_kv_heads=_KV_HEADS, head_dim=_HEAD_DIM, device=device
    )
    out = qsa_decode_attention(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        req_to_token=case["req_to_token"].to(device),
        row_req_pool_indices=case["row_req_pool_indices"].to(device),
        topk_indices=case["topk_indices"].to(device),
        seq_lens=case["seq_lens"].to(device),
        sm_scale=1.0 / (_HEAD_DIM**0.5),
        workspace=workspace,
        prefix_valid=True,
    )
    assert torch.isfinite(out.float()).all(), "finite BF16 Q produced Inf/NaN"
    expected = _reference_attention(q, k_buffer, v_buffer, case)
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-3)


@_CUDA
def test_launches_are_deterministic_and_leave_the_arrival_counters_zeroed():
    """Replay safety with real traffic: non-zero results (an all-zero output
    would pass any equality check), bit-stable across launches, and the arrival
    counters back at zero after every one."""
    torch.manual_seed(0)
    device = "cuda"
    rows = 8
    case = _long_case(rows=rows, cols=_TOPK_COLS, holes=False)
    q = torch.randn(rows, _Q_HEADS, _HEAD_DIM, dtype=torch.bfloat16, device=device)
    args = dict(
        q=q,
        k_buffer=torch.randn(case["n_slots"], _KV_HEADS, _HEAD_DIM, device=device).to(
            torch.float8_e4m3fn
        ),
        v_buffer=torch.randn(case["n_slots"], _KV_HEADS, _HEAD_DIM, device=device).to(
            torch.float8_e4m3fn
        ),
        req_to_token=case["req_to_token"].to(device),
        row_req_pool_indices=case["row_req_pool_indices"].to(device),
        topk_indices=case["topk_indices"].to(device),
        seq_lens=case["seq_lens"].to(device),
        sm_scale=1.0 / (_HEAD_DIM**0.5),
        prefix_valid=True,
    )
    workspace = QSADecodeAttnWorkspace(
        num_kv_heads=_KV_HEADS, head_dim=_HEAD_DIM, device=device
    )
    first = qsa_decode_attention(workspace=workspace, **args)
    assert torch.count_nonzero(first) > 0
    assert torch.count_nonzero(workspace.arrivals) == 0
    assert torch.equal(first, qsa_decode_attention(workspace=workspace, **args))
    assert torch.count_nonzero(workspace.arrivals) == 0


@_CUDA
def test_graph_capture_replays_shared_tail_rows_workspace_and_counters():
    """The production gate-on decode step end to end in one graph: the shared
    index lookup (the valid-prefix compaction) and the split-KV kernel are both
    recorded, so a replay re-derives the rows from the static buffers, keeps the
    workspace addresses, and a CHANGED input must produce the new reference --
    neither a stale write nor a skipped one can hide behind all-zero output.
    """
    torch.manual_seed(0)
    device = "cuda"
    rows, token_topk, tail_width = 4, 2048, 3
    cols = token_topk + tail_width
    state = _make_state(
        True,
        num_requests=8,
        tail_width=tail_width,
        token_topk=token_topk,
        device=device,
    )
    reqs = torch.tensor([1, 2, 3, 5], dtype=torch.int32, device=device)
    widths = [token_topk - 7 * row for row in range(rows)]
    frozen = torch.full((rows, cols), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        frozen[row, : widths[row]] = torch.arange(widths[row], dtype=torch.int32)
    # A -1 padded frozen row plus a drafted tail: the legacy layout would hide the
    # tail behind up to tail_width holes, the prefix layout puts it at widths[row].
    state.capture(
        frozen, reqs, torch.tensor(widths, dtype=torch.int32, device=device), 48
    )
    positions = torch.tensor(
        [w + tail_width - 1 for w in widths], dtype=torch.int32, device=device
    )
    seq_lens = positions.clone() + 1
    logical = torch.arange(cols, dtype=torch.int32, device=device)
    req_to_token = (
        torch.arange(9, dtype=torch.int32, device=device)[:, None] * 3
        + 2 * logical[None, :]
        + 1
    ).contiguous()
    k_buffer = torch.randn(
        int(req_to_token.max()) + 1, _KV_HEADS, _HEAD_DIM, device=device
    ).to(torch.float8_e4m3fn)
    v_buffer = torch.randn(
        int(req_to_token.max()) + 1, _KV_HEADS, _HEAD_DIM, device=device
    ).to(torch.float8_e4m3fn)
    q = torch.randn(rows, _Q_HEADS, _HEAD_DIM, dtype=torch.bfloat16, device=device)
    backend = _make_backend(triton_decode_attn=True, state=state)
    backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    kwargs = dict(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        layer=SimpleNamespace(layer_id=48, scaling=1.0 / (_HEAD_DIM**0.5)),
        forward_batch=SimpleNamespace(
            forward_mode=ForwardMode.DECODE, req_pool_indices=reqs
        ),
        metadata=SimpleNamespace(sequence_lengths=seq_lens, row_req_pool_indices=reqs),
    )

    def call():
        # Lookup inside the capture: the compaction itself must be recordable.
        return QwenSparseAttnBackend._forward_triton_decode(
            backend, topk_indices=state.lookup(reqs, positions, 48), **kwargs
        )

    def expect():
        # _forward_triton_decode hands back the flattened [rows, heads * dim]
        # contract (its own ``.reshape(q.shape[0], -1)``), so the reference is
        # compared in that layout, not as [rows, heads, dim].
        return _reference_attention(
            q,
            k_buffer,
            v_buffer,
            {
                "req_to_token": req_to_token.cpu(),
                "topk_indices": state.lookup(reqs, positions, 48).cpu(),
                "seq_lens": seq_lens.cpu(),
                "row_req_pool_indices": reqs.cpu(),
            },
        ).reshape(rows, -1)

    eager = call()  # the eager warmup owns the workspace the graph then bakes in
    workspace = backend._decode_attn_workspace
    assert workspace is not None and torch.count_nonzero(eager) > 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    assert backend._decode_attn_workspace is workspace
    expected = expect()
    assert captured.shape == eager.shape == expected.shape
    assert captured.shape == (rows, _Q_HEADS * _HEAD_DIM)
    torch.testing.assert_close(captured.float(), expected.float(), rtol=2e-2, atol=2e-3)
    assert torch.equal(captured, eager)

    # Change the rows behind the replay: one fewer drafted token per row and a
    # different frozen set.  The replay must follow, not repeat the old output.
    positions.copy_(positions - 1)
    seq_lens.copy_(seq_lens - 1)
    stale = captured.clone()
    frozen.fill_(-1)
    for row in range(rows):
        # A different (half-size, even-position) frozen set: the compaction
        # boundary moves, so the tail must move with it.
        frozen[row, : widths[row] // 2] = 2 * torch.arange(
            widths[row] // 2, dtype=torch.int32, device=device
        )
    state.capture(
        frozen, reqs, torch.tensor(widths, dtype=torch.int32, device=device), 48
    )
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(captured, stale), "replay kept the previous output"
    expected = expect()
    assert captured.shape == expected.shape
    torch.testing.assert_close(captured.float(), expected.float(), rtol=2e-2, atol=2e-3)
    assert torch.count_nonzero(captured) > 0
    assert torch.count_nonzero(workspace.arrivals) == 0


@_CUDA
def test_gate_on_falls_back_to_the_packed_path_for_other_shapes(monkeypatch):
    """An eligible model running an ineligible layer (or a BF16 KV cache) must
    keep the existing path instead of a wrong numeric answer."""
    calls = []
    monkeypatch.setattr(
        f"{_BACKEND}.qsa_decode_attention", lambda **kwargs: calls.append("triton")
    )
    monkeypatch.setattr(f"{_BACKEND}._resolve_trtllm_sparse_decode", lambda: object())
    monkeypatch.setattr(
        QwenSparseAttnBackend,
        "_forward_trtllm_sparse",
        lambda *args, **kwargs: calls.append("packed"),
    )
    backend = _make_backend(triton_decode_attn=True)
    tensors = _measured_shape(rows=2)
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda layer_id: tensors["k_buffer"],
        get_value_buffer=lambda layer_id: tensors["v_buffer"],
    )
    backend.forward_metadata = SimpleNamespace(
        sequence_lengths=torch.full((2,), 32, dtype=torch.int32, device="cuda"),
        row_req_pool_indices=torch.arange(2, dtype=torch.int32, device="cuda"),
        is_cuda_graph=False,
    )
    layer = SimpleNamespace(layer_id=48, scaling=0.0625, head_dim=_HEAD_DIM)
    forward_batch = SimpleNamespace(
        req_pool_indices=torch.arange(2, dtype=torch.int32, device="cuda")
    )
    QwenSparseAttnBackend._forward_paged_attention(
        backend,
        # The measured kernel is BF16; a BF16 KV pool is the other rejection.
        tensors["q"].to(torch.float16).cuda(),
        layer,
        forward_batch,
        tensors["topk_indices"].cuda(),
    )
    assert calls == ["packed"]
    assert backend._decode_attn_workspace is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
