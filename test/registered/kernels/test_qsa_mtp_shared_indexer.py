import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QSAMTPSharedSparseIndices,
    QwenSparseAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

TOPK = 8


def _make_backend(state=None):
    runner = SimpleNamespace(
        device="cpu",
        token_to_kv_pool=None,
        req_to_token_pool=None,
        model_config=SimpleNamespace(
            context_len=64, hf_config=SimpleNamespace(indexer_compress_ratio=4)
        ),
    )
    backend = QwenSparseAttnBackend(runner)
    if state is not None:
        backend.set_mtp_shared_sparse_indices(state)
    return backend


def _make_state(num_requests=8, tail_width=4, triton_decode_attn=False):
    with envs.SGLANG_OPT_TRITON_DECODE_ATTN.override(triton_decode_attn):
        return QSAMTPSharedSparseIndices(
            layer_ids=[48],
            num_requests=num_requests,
            token_topk=TOPK,
            tail_width=tail_width,
            device="cpu",
        )


def _capture(state, req_rows, frozen_rows, captured_lens):
    state.capture(
        torch.tensor(frozen_rows, dtype=torch.int32),
        torch.tensor(req_rows, dtype=torch.int32),
        torch.tensor(captured_lens, dtype=torch.int32),
        layer_id=48,
    )


def _valid_prefix_property(row):
    """True when no valid column sits behind a -1 hole, which is exactly what
    the split-KV decode kernel needs in order to stop scanning at ``seq_lens``."""
    holes = [i for i, value in enumerate(row) if value < 0]
    return not holes or all(row[i] < 0 for i in range(holes[0], len(row)))


def _forward_batch(mode, **kwargs):
    return SimpleNamespace(forward_mode=mode, _original_forward_mode=None, **kwargs)


def test_prefill_extend_capture_anchors_last_row_per_request():
    """The post-prefill draft extend (plain EXTEND, batch-major row mapping)
    seeds the first decode loop; only each request's FINAL row is the seed,
    and that exact row must be what the following MTP decode steps read."""
    state = _make_state()
    backend = _make_backend(state)
    token_to_batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
    req_pool_indices = torch.tensor([5, 2], dtype=torch.int32)
    seqlens = torch.tensor([30, 31, 32, 40, 41], dtype=torch.int32)
    backend.forward_metadata = SimpleNamespace(
        indexer_metadata=SimpleNamespace(
            is_cuda_graph=False,
            get_token_to_batch_idx=lambda: token_to_batch,
            get_seqlens_expanded=lambda: seqlens,
            req_pool_indices=req_pool_indices,
        )
    )
    topk = torch.arange(5 * TOPK, dtype=torch.int32).reshape(5, TOPK)
    forward_batch = _forward_batch(
        ForwardMode.EXTEND, req_pool_indices=req_pool_indices
    )
    assert backend.should_capture_mtp_sparse_indices(forward_batch)
    backend.capture_mtp_sparse_indices(topk, forward_batch, layer_id=48)

    backend.forward_metadata = SimpleNamespace(
        indexer_metadata=SimpleNamespace(
            decode_logical_positions=None,
            get_seqlens_expanded=lambda: torch.tensor([33, 42], dtype=torch.int32),
            token_to_batch_idx_is_identity=True,
        )
    )
    got = backend.lookup_mtp_sparse_indices(
        SimpleNamespace(req_pool_indices=req_pool_indices), layer_id=48
    )
    assert torch.equal(got[0, :TOPK], topk[2])
    assert torch.equal(got[1, :TOPK], topk[4])
    # The tail appends exactly [captured_len, current_position]: request A
    # captured at 32, now at 32 -> tail {32}; request B captured at 41, now
    # at 41 -> tail {41}; the rest of the tail is -1 (dropped downstream).
    assert got[0, TOPK:].tolist() == [32, -1, -1, -1]
    assert got[1, TOPK:].tolist() == [41, -1, -1, -1]


def test_draft_extend_v2_capture_anchors_last_accepted_row():
    """Production DRAFT_EXTEND_V2 packs uniform num_window_tokens blocks per
    request (extend lens are the WINDOW width, not the accept count), and DP
    token padding aliases tail rows to request 0.  The seed must be each
    request's last ACCEPTED row (select_index semantics): the window-end row
    is conditioned on rejected drafts and its captured_len overshoots, which
    silently kills the in-flight tail; a run-boundary anchor would let an
    alias-padding row overwrite request 0's seed nondeterministically."""
    state = _make_state()
    backend = _make_backend(state)
    row_lens = torch.tensor([30, 31, 32, 33, 40, 41, 42, 43, 1, 1], dtype=torch.int32)
    backend.forward_metadata = SimpleNamespace(
        indexer_metadata=SimpleNamespace(
            is_cuda_graph=False,
            get_seqlens_expanded=lambda: row_lens,
        )
    )
    topk = torch.arange(10 * TOPK, dtype=torch.int32).reshape(10, TOPK)
    forward_batch = _forward_batch(
        ForwardMode.DRAFT_EXTEND_V2,
        batch_size=2,
        spec_info=SimpleNamespace(
            extend_seq_lens_tensor=None,
            num_accept_tokens=torch.tensor([3, 2], dtype=torch.int32),
            num_front_tokens=0,
        ),
        extend_seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        req_pool_indices=torch.tensor([5, 2], dtype=torch.int32),
    )
    assert backend.should_capture_mtp_sparse_indices(forward_batch)
    backend.capture_mtp_sparse_indices(topk, forward_batch, layer_id=48)
    assert torch.equal(state.indices[0, 5, :TOPK], topk[2])
    assert torch.equal(state.indices[0, 2, :TOPK], topk[5])
    assert state.captured_len[0, 5].item() == 32
    assert state.captured_len[0, 2].item() == 41


def test_graph_capture_routes_bucket_pad_slot0_to_trash():
    """The draft-extend CUDA graph replays capture ops on static buffers.
    Bucket-padding requests carry extend len == captured width with
    req_pool_indices zeroed (the pool's reserved slot 0): their capture must
    land in the trash row -- state row 0 stays untouched -- while real
    requests anchor their last accepted row and read it back with the tail
    starting exactly at captured_len."""
    state = _make_state()
    backend = _make_backend(state)
    row_lens = torch.tensor(
        [30, 31, 32, 33, 40, 41, 42, 43, 1, 1, 1, 1], dtype=torch.int32
    )
    backend.forward_metadata = SimpleNamespace(
        indexer_metadata=SimpleNamespace(
            is_cuda_graph=True,
            get_seqlens_expanded=lambda: row_lens,
        )
    )
    row0_before = state.indices[0, 0].clone()
    topk = torch.arange(12 * TOPK, dtype=torch.int32).reshape(12, TOPK)
    forward_batch = _forward_batch(
        ForwardMode.DRAFT_EXTEND_V2,
        batch_size=3,
        spec_info=SimpleNamespace(
            extend_seq_lens_tensor=None,
            num_accept_tokens=torch.tensor([3, 2, 4], dtype=torch.int32),
            num_front_tokens=0,
        ),
        extend_seq_lens=torch.tensor([4, 4, 4], dtype=torch.int32),
        req_pool_indices=torch.tensor([5, 2, 0], dtype=torch.int32),
    )
    backend.capture_mtp_sparse_indices(topk, forward_batch, layer_id=48)
    got_a = state.lookup(
        torch.tensor([5]), torch.tensor([32], dtype=torch.int32), layer_id=48
    )
    got_b = state.lookup(
        torch.tensor([2]), torch.tensor([41], dtype=torch.int32), layer_id=48
    )
    assert torch.equal(got_a[0, :TOPK], topk[2])
    assert torch.equal(got_b[0, :TOPK], topk[5])
    assert state.captured_len[0, 5].item() == 32
    assert state.captured_len[0, 2].item() == 41
    assert got_a[0, TOPK] == 32
    assert torch.equal(state.indices[0, 0], row0_before)
    assert state.captured_len[0, 0].item() == 1


def test_capture_routes_zero_extend_to_trash_row():
    """A request with extend length 0 (DP batch padding) has no real row; its
    capture must land in the state's trash row, never a live request's row."""
    state = _make_state()
    backend = _make_backend(state)
    backend.forward_metadata = SimpleNamespace(
        indexer_metadata=SimpleNamespace(
            is_cuda_graph=True,
            get_seqlens_expanded=lambda: torch.tensor([30, 1], dtype=torch.int32),
        )
    )
    before = state.indices[0, 3].clone()
    topk = torch.arange(2 * TOPK, dtype=torch.int32).reshape(2, TOPK)
    forward_batch = _forward_batch(
        ForwardMode.DRAFT_EXTEND_V2,
        batch_size=2,
        spec_info=SimpleNamespace(extend_seq_lens_tensor=None, num_accept_tokens=None),
        extend_seq_lens=torch.tensor([1, 0], dtype=torch.int32),
        req_pool_indices=torch.tensor([6, 3], dtype=torch.int32),
    )
    backend.capture_mtp_sparse_indices(topk, forward_batch, layer_id=48)
    assert torch.equal(state.indices[0, 3], before)
    assert state.captured_len[0, 6].item() == 30


def test_uncaptured_rows_stay_warmup_safe():
    """CUDA-graph warmup replays decode with dummy request rows before any
    draft-extend ran; a never-captured row must resolve to logical index 0
    (attend the first token), never an empty or invalid selection."""
    state = _make_state(num_requests=4)
    got = state.lookup(
        torch.tensor([0, 3]), torch.zeros(2, dtype=torch.int32), layer_id=48
    )
    assert torch.equal(got[:, :TOPK], torch.zeros((2, TOPK), dtype=torch.int32))
    # captured_len defaults to 1 > position 0: the whole tail is invalid.
    assert (got[:, TOPK:] == -1).all()


def test_reuse_and_capture_gating_by_mode_state_and_rewrite():
    """Only decode steps of a backend that carries the shared state may skip
    the indexer, and only genuine extend flavors may capture: the target
    backend (no state), target-verify, and DP MAX_LEN mode rewrites
    (_original_forward_mode set) keep the normal indexer path -- a wrong
    True here either corrupts target selection or captures from a
    fabricated batch with no requests."""
    backend = _make_backend()
    decode_batch = _forward_batch(ForwardMode.DECODE)
    assert not backend.should_reuse_mtp_sparse_indices(decode_batch)
    state = _make_state(num_requests=2)
    backend.set_mtp_shared_sparse_indices(state)
    assert backend.should_reuse_mtp_sparse_indices(decode_batch)

    extend_v2 = _forward_batch(ForwardMode.DRAFT_EXTEND_V2)
    prefill_extend = _forward_batch(ForwardMode.EXTEND)
    target_verify = _forward_batch(ForwardMode.TARGET_VERIFY)
    rewritten = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        _original_forward_mode=ForwardMode.IDLE,
    )
    assert not backend.should_reuse_mtp_sparse_indices(extend_v2)
    assert backend.should_capture_mtp_sparse_indices(extend_v2)
    assert backend.should_capture_mtp_sparse_indices(prefill_extend)
    assert not backend.should_capture_mtp_sparse_indices(target_verify)
    assert not backend.should_capture_mtp_sparse_indices(rewritten)


def test_index_share_flag_reads_override_and_checkpoint_locations():
    """--json-model-override-args lands the flag on the TOP-LEVEL hf_config
    while checkpoints carry it on the nested text_config; reading only
    hf_text_config silently disabled the feature for override-launched
    servers (caught as a boot with no 'index sharing enabled' log and
    OFF-band accept)."""
    from sglang.srt.speculative.eagle_worker_v2 import _qsa_index_share_requested

    override_style = SimpleNamespace(
        index_share_for_mtp_iteration=True, text_config=SimpleNamespace()
    )
    checkpoint_style = SimpleNamespace(
        text_config=SimpleNamespace(index_share_for_mtp_iteration=True)
    )
    flat_style = SimpleNamespace(index_share_for_mtp_iteration=True)
    off = SimpleNamespace(text_config=SimpleNamespace())
    assert _qsa_index_share_requested(override_style)
    assert _qsa_index_share_requested(checkpoint_style)
    assert _qsa_index_share_requested(flat_style)
    assert not _qsa_index_share_requested(off)


# --- shared-tail valid-prefix layout (SGLANG_OPT_TRITON_DECODE_ATTN) ------


def test_gate_off_keeps_the_captured_row_layout_unchanged():
    """Default-off: the frozen rows stay exactly as the indexer wrote them and
    the tail keeps the trailing columns, byte for byte as before."""
    state = _make_state()
    assert not state.shared_tail_prefix
    frozen = [3, 7, 11, 15, 19, -1, -1, -1]
    _capture(state, [5], [frozen], [32])
    got = state.lookup(torch.tensor([5]), torch.tensor([34], dtype=torch.int32), 48)
    assert got[0, :TOPK].tolist() == frozen
    assert got[0, TOPK:].tolist() == [32, 33, 34, -1]


def test_shared_tail_prefix_compacts_holes_without_changing_the_set():
    """A short (``-1``-padded) frozen row plus a drafted tail is the hole case the
    packed prefix readers mis-handle.  The prefix layout reorders the same
    columns -- frozen order first, then the contiguous ``[captured_len,
    position]`` tail -- and nothing else: no missing token, no duplicate, no
    extra token, and the causal bound (every column < seq_len) is untouched."""
    frozen = [3, 7, 11, 15, 19, -1, -1, -1]
    rows = {}
    for gate in (False, True):
        state = _make_state(triton_decode_attn=gate)
        assert state.shared_tail_prefix is gate
        _capture(state, [5], [frozen], [32])
        rows[gate] = state.lookup(
            torch.tensor([5]), torch.tensor([34], dtype=torch.int32), 48
        )[0].tolist()
    assert not _valid_prefix_property(rows[False])  # the bug the layout fixes
    assert _valid_prefix_property(rows[True])
    assert (
        sorted(v for v in rows[True] if v >= 0)
        == sorted(v for v in rows[False] if v >= 0)
        == [3, 7, 11, 15, 19, 32, 33, 34]
    )
    assert rows[True] == [3, 7, 11, 15, 19, 32, 33, 34, -1, -1, -1, -1]


def test_shared_tail_prefix_is_per_row_for_ragged_and_empty_selections():
    """Rows share one launch, so the compaction boundary is per row: a fully
    padded frozen row (``num_frozen == 0``, only the drafted tail) and a full
    one must both end up prefix-packed with their own tail offset."""
    state = _make_state(triton_decode_attn=True)
    _capture(
        state,
        [2, 3, 4],
        [
            [1, 2, -1, -1, -1, -1, -1, -1],
            [0, 5, 9, 13, 17, 21, 25, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        [8, 26, 5],
    )
    got = state.lookup(
        torch.tensor([2, 3, 4], dtype=torch.int32),
        torch.tensor([10, 26, 6], dtype=torch.int32),
        48,
    )
    assert got[0].tolist() == [1, 2, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1]
    assert got[1].tolist() == [0, 5, 9, 13, 17, 21, 25, 26, -1, -1, -1, -1]
    assert got[2].tolist() == [5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    assert all(_valid_prefix_property(row) for row in got.tolist())


def test_shared_tail_prefix_leaves_no_stale_row_and_stays_warmup_safe():
    """A reused request row must show only the new capture (a wider frozen set
    first, then a short one), and a never-captured row must keep attending
    logical index 0 instead of an empty set."""
    state = _make_state(num_requests=4, triton_decode_attn=True)
    _capture(state, [3], [[4, 8, 12, 16, 20, 24, 28, 32]], [33])
    _capture(state, [3], [[1, 4, -1, -1, -1, -1, -1, -1]], [5])
    got = state.lookup(
        torch.tensor([3, 1], dtype=torch.int32),
        torch.tensor([6, 0], dtype=torch.int32),
        48,
    )
    assert got[0].tolist() == [1, 4, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1]
    # Row 1 was never captured: zeros are valid columns, captured_len 1 > 0
    # invalidates the whole tail.
    assert got[1].tolist() == [0] * TOPK + [-1] * 4
    assert got.dtype == torch.int32
    assert got.shape == (2, TOPK + 4)


def test_shared_tail_prefix_shapes_are_static_for_graph_capture():
    """The compaction boundary is a device value, so every row keeps the same
    static width and the lookup stays recordable into the decode graphs."""
    state = _make_state(triton_decode_attn=True)
    _capture(
        state,
        [1, 2],
        [[-1] * TOPK, [0, 1, 2, 3, 4, 5, 6, 7]],
        [9, 9],
    )
    got = state.lookup(
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([12, 12], dtype=torch.int32),
        48,
    )
    assert got.shape == (2, TOPK + 4)
    assert got[0].tolist() == [9, 10, 11, 12] + [-1] * (TOPK)
    assert got[1].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
