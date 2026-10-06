"""Contract checks for the optional QSA split-KV decode attention.

CPU: shape gating, launch/workspace bounds, workspace ownership and the
backend plumbing (which args reach the kernel, and the prefix-scan decision).
The numerical, CUDA-graph replay and TP-shape checks of the Triton kernel
itself need SM120 hardware: they are prepared here and skipped without it.
"""

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


def _make_state(triton_decode_attn, num_requests=4, tail_width=4):
    with envs.SGLANG_OPT_TRITON_DECODE_ATTN.override(triton_decode_attn):
        return QSAMTPSharedSparseIndices(
            layer_ids=[48],
            num_requests=num_requests,
            token_topk=_TOPK,
            tail_width=tail_width,
            device="cpu",
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
    assert workspace.partial_lse.shape == (workspace.max_parts * _BLOCK_H,)
    assert workspace.partial_lse.dtype == torch.float32
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


# --- GPU (prepared: skipped without CUDA hardware) ------------------------

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
_ROWS = 7
_ROW_REQS = [0, 1, 0, 1, 0, 1, 0]


def _dense_case(prefix_valid):
    """Multi-row, ragged, holes/tails and a two-request pool."""
    n_slots = 128
    topk_indices = torch.full((_ROWS, _TOPK), -1, dtype=torch.int32)
    for row in range(_ROWS):
        length = min(int(_LENS[row]), 6)
        topk_indices[row, :length] = torch.arange(length, dtype=torch.int32)
        if not prefix_valid:  # a -1 hole mid-row, the tail behind it
            topk_indices[row, 2] = -1
            tail = torch.arange(length, length + 2, dtype=torch.int32)
            topk_indices[row, 3 : 3 + tail.numel()] = tail
    return dict(
        req_to_token=torch.arange(n_slots, dtype=torch.int32).view(2, n_slots // 2),
        topk_indices=topk_indices,
        seq_lens=_LENS.clone(),
        row_req_pool_indices=torch.tensor(_ROW_REQS, dtype=torch.int32),
        n_slots=n_slots,
    )


_LENS = torch.tensor([1, 5, 5, 17, 17, 33, 33], dtype=torch.int32)


def _reference_attention(q, k_buffer, v_buffer, case):
    """Plain fp32 softmax over exactly the selected columns, no split-KV."""
    scale = 1.0 / (k_buffer.shape[-1] ** 0.5)
    heads_per_kv = q.shape[1] // k_buffer.shape[1]
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    for row in range(q.shape[0]):
        req = int(case["row_req_pool_indices"][row])
        length = int(case["seq_lens"][row])
        cols = [int(c) for c in case["topk_indices"][row] if 0 <= int(c) < length]
        if not cols:
            continue
        slots = (
            case["req_to_token"][req]
            .long()[torch.tensor(cols, device=q.device, dtype=torch.long)]
            .to(q.device)
        )
        k = k_buffer[slots].float()  # [n, kv heads, D]
        v = v_buffer[slots].float()
        for h in range(q.shape[1]):
            kv_head = h // heads_per_kv
            scores = (q[row, h].float() @ k[:, kv_head].t()) * scale
            out[row, h] = torch.softmax(scores, -1) @ v[:, kv_head]
    return out.to(q.dtype)


@_CUDA
@pytest.mark.parametrize("prefix_valid", [True, False])
@pytest.mark.parametrize("kv_heads,q_heads", [(_KV_HEADS, _Q_HEADS), (1, 6), (4, 24)])
def test_kernel_matches_reference_over_fp8_kv(prefix_valid, kv_heads, q_heads):
    """Every selected contribution survives the split-KV combine -- -1 holes,
    the drafted tail, ragged lengths, multi-row and both TP-local head
    splits -- and a second launch reuses the partials without leaking."""
    torch.manual_seed(0)
    device = "cuda"
    case = _dense_case(prefix_valid)
    q = torch.randn(
        _ROWS, q_heads, _HEAD_DIM, dtype=torch.bfloat16, device=device
    ).contiguous()
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
        prefix_valid=prefix_valid,
    )
    workspace = QSADecodeAttnWorkspace(
        num_kv_heads=kv_heads, head_dim=_HEAD_DIM, device=device
    )
    expected = _reference_attention(
        q,
        k_buffer,
        v_buffer,
        {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in case.items()},
    )
    for _ in range(2):
        out = qsa_decode_attention(workspace=workspace, **args)
        assert out.shape == q.shape
        torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-3)


@_CUDA
def test_launches_leave_the_arrival_counters_zeroed():
    """Replay safety: each launch hands its counters back at zero and the
    combine is deterministic across launches."""
    device = "cuda"
    tensors = _measured_shape(rows=5)
    args = dict(
        q=tensors["q"].to(device),
        k_buffer=tensors["k_buffer"].to(device),
        v_buffer=tensors["v_buffer"].to(device),
        req_to_token=torch.zeros(8, 64, dtype=torch.int32, device=device),
        row_req_pool_indices=torch.zeros(5, dtype=torch.int32, device=device),
        topk_indices=tensors["topk_indices"].to(device),
        seq_lens=torch.full((5,), 32, dtype=torch.int32, device=device),
        sm_scale=0.0625,
        prefix_valid=True,
    )
    workspace = QSADecodeAttnWorkspace(
        num_kv_heads=_KV_HEADS, head_dim=_HEAD_DIM, device=device
    )
    first = qsa_decode_attention(workspace=workspace, **args)
    assert torch.count_nonzero(workspace.arrivals) == 0
    assert torch.equal(first, qsa_decode_attention(workspace=workspace, **args))
    assert torch.count_nonzero(workspace.arrivals) == 0


@_CUDA
def test_workspace_survives_capture_and_replay_of_the_decode_graph():
    """The graph bakes the workspace addresses: capture after the eager warmup,
    replay many times, still get the eager result, counters back at zero."""
    backend = _make_backend(triton_decode_attn=True)
    device = "cuda"
    rows = 4
    tensors = _measured_shape(rows=rows)
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros(8, 64, dtype=torch.int32, device=device)
    )
    kwargs = dict(
        q=tensors["q"].to(device),
        k_buffer=tensors["k_buffer"].to(device),
        v_buffer=tensors["v_buffer"].to(device),
        layer=SimpleNamespace(layer_id=48, scaling=0.0625),
        forward_batch=SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(rows, dtype=torch.int32, device=device),
        ),
        metadata=SimpleNamespace(
            sequence_lengths=torch.full((rows,), 32, dtype=torch.int32, device=device),
            row_req_pool_indices=torch.arange(rows, dtype=torch.int32, device=device),
        ),
        topk_indices=tensors["topk_indices"].to(device),
    )
    eager = QwenSparseAttnBackend._forward_triton_decode(backend, **kwargs)
    workspace = backend._decode_attn_workspace
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = QwenSparseAttnBackend._forward_triton_decode(backend, **kwargs)
    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize()
    assert backend._decode_attn_workspace is workspace
    assert torch.equal(eager, captured)
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
