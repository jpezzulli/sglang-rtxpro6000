"""FR-Spec hot-vocabulary shard-boundary mapping/scale tests (TP1 + simulated TP2).

The base bug (reporter: RTX PRO 6000 Blackwell TP2 FR-Spec startup):
``EagleDraftWorker.init_lm_head`` selected the hot-vocabulary rows of the
target lm_head using *global* token ids against the *resident shard* tensor.
On TP1 the tensor is full-width so that is exact (the qualified default).
At TP>1 the tensor is this rank's vocab-parallel shard, so global ids
either index the wrong rows inside the shard width -- silently mixing each
row's FP8 data away from its rowwise scale -- or raise index-out-of-bounds
when an id exceeds the shard width (both exercised below against the base
formula).

These tests are CPU-only and simulate the second rank with an injected
``all_gather_object``-compatible exchange; live TP2 on hardware remains
pending qualification, as does anything CUDA-graph.
"""

import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    dequantize_rowwise_weight,
    replace_linear_weight_rowwise_fp8,
    rowwise_scale_of,
    select_rowwise_weight_rows,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.speculative import hot_vocab
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.hot_vocab import (
    hot_vocab_local_rows,
    shared_hot_lm_head,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# org 10, +2 added, padding 4 -> padded org 12, total padded 16; TP2 shards.
ORG_VOCAB = 10
NUM_ADDED = 2
PADDING = 4
TOTAL_PADDED = 16
TP_SIZE = 2
ROWS_PER_RANK = TOTAL_PADDED // TP_SIZE  # 8
K = 3

HOT_GLOBAL_IDS = [0, 5, 6, 9, 11]  # spans both shards and includes an added id

# Canonical ownership for the ids exercised here, derived from the real
# ``_get_indices`` layout below (checked in test_ownership_matches_layer_layout).
OWNERS = {0: (0, 0), 5: (0, 5), 6: (1, 0), 9: (1, 3), 11: (0, 7)}


def _full_weight():
    """Deterministic (TOTAL_PADDED, K) head; every row holds distinct values."""
    return torch.arange(TOTAL_PADDED * K, dtype=torch.float32).reshape(
        TOTAL_PADDED, K
    )


def _shard_indices(rank: int):
    org_vocab_size_padded = ((ORG_VOCAB + PADDING - 1) // PADDING) * PADDING
    return VocabParallelEmbedding._get_indices(
        TOTAL_PADDED,
        org_vocab_size_padded,
        ORG_VOCAB + NUM_ADDED,
        ORG_VOCAB,
        rank,
        TP_SIZE,
    )


SHARD_INDICES = {rank: _shard_indices(rank) for rank in range(TP_SIZE)}


def _owned_map_for(rank: int):
    """Every global id whose ownership resolves to this rank, -> tensor row."""
    indices = SHARD_INDICES[rank]
    owned = {}
    for gid in range(0, ORG_VOCAB + NUM_ADDED):
        row = hot_vocab._owned_row(gid, indices)
        if row is not None:
            owned[gid] = row
    return owned


# Every global id owned by each rank, and the tensor row that holds it.
OWNED_BY_RANK = {rank: _owned_map_for(rank) for rank in range(TP_SIZE)}


def _shard_layer(rank: int, *, rowwise: bool):
    """Rank's resident (ROWS_PER_RANK, K) head, built from the shard layout.

    Row ``row`` holds the full-head row of the global id whose ownership
    resolves to ``(rank, row)`` (padding rows keep zeros / -1 semantics and
    are never addressed by an id). For the rowwise case each row carries a
    distinct scale (a ramp over this shard's rows), so selecting row i with
    the wrong id is detectable in *both* data and scale.
    """
    owned = _owned_map_for(rank)
    full = _full_weight()
    rows = torch.zeros(ROWS_PER_RANK, K)
    for gid, row in owned.items():
        rows[row] = full[gid]
    if rowwise:
        linear = nn.Linear(K, ROWS_PER_RANK, bias=False, dtype=torch.bfloat16)
        linear.weight.data.copy_(rows.to(torch.bfloat16))
        replace_linear_weight_rowwise_fp8(linear)
        scale = 0.25 + 0.1 * torch.arange(ROWS_PER_RANK, dtype=torch.float32)
        setattr(linear.weight, "_sm120_rowwise_scale", scale)
        weight = linear.weight
    else:
        weight = nn.Parameter(rows.clone(), requires_grad=False)
    layer = SimpleNamespace(
        shard_indices=SHARD_INDICES[rank],
        tp_size=TP_SIZE,
        num_embeddings_padded=TOTAL_PADDED,
        weight=weight,
    )
    return layer, weight


# --------------------------------------------------------------------------
# Mapping: exact ranges, fail closed, no filtering
# --------------------------------------------------------------------------


def test_ownership_matches_layer_layout():
    """Ownership mirrors weight_loader + get_sharded_to_full_mapping exactly."""
    assert OWNED_BY_RANK[0][0] == 0 and OWNED_BY_RANK[0][5] == 5
    # Rank 1's org block loads at tensor row 0 (weight_loader narrows the
    # checkpoint to its org range and copies from row 0)...
    assert OWNED_BY_RANK[1][6] == 0 and OWNED_BY_RANK[1][9] == 3
    # ...and rank 0's LoRA-added ids sit behind the *padded* org block.
    assert OWNED_BY_RANK[0][10] == 6 and OWNED_BY_RANK[0][11] == 7
    assert SHARD_INDICES[1].added_vocab_end_index == SHARD_INDICES[
        1
    ].added_vocab_start_index  # empty on rank 1
    assert 15 not in OWNED_BY_RANK[0] and 15 not in OWNED_BY_RANK[1]


def test_mapping_ownership_partitions_the_hot_ids_exactly_once():
    covered = []
    for rank in range(TP_SIZE):
        rows, positions, unowned = hot_vocab_local_rows(
            torch.tensor(HOT_GLOBAL_IDS),
            shard_indices=SHARD_INDICES[rank],
            local_rows=ROWS_PER_RANK,
            allow_unowned=True,
        )
        expected_owned = [
            i
            for i, gid in enumerate(HOT_GLOBAL_IDS)
            if OWNERS.get(gid, (None, None))[0] == rank
        ]
        assert positions.tolist() == expected_owned
        assert rows.tolist() == [OWNERS[HOT_GLOBAL_IDS[i]][1] for i in expected_owned]
        assert unowned == [
            gid for gid in HOT_GLOBAL_IDS if OWNERS.get(gid, (None, None))[0] != rank
        ]
        covered.extend(expected_owned)
    assert sorted(covered) == list(range(len(HOT_GLOBAL_IDS)))


def test_mapping_out_of_shard_id_raises_with_ids():
    with pytest.raises(ValueError, match="FR-Spec hot token ids"):
        hot_vocab_local_rows(
            [6, 15], shard_indices=SHARD_INDICES[1], local_rows=ROWS_PER_RANK
        )


def test_mapping_full_width_is_identity_and_out_of_range_fails_closed():
    rows = hot_vocab_local_rows([3, 0, 7], shard_indices=None, local_rows=8)
    assert rows.tolist() == [3, 0, 7]
    with pytest.raises(ValueError, match="full-width head of 8 rows"):
        hot_vocab_local_rows([3, 9], shard_indices=None, local_rows=8)


def test_mapping_full_width_rejects_negative_ids_including_the_boundary():
    """A negative row is never the end of the head wrapping around.

    ``row >= local_rows`` alone let ``-1`` (the padding-token sentinel and a
    plausible token-map typo) through as a valid full-width row, so the draft
    would silently share the last head row (or the added-block boundary) for
    an id no token owns. The upper boundary stays inclusive: row local_rows-1
    is a resident row, local_rows itself is not.
    """
    with pytest.raises(ValueError, match="full-width head of 12 rows"):
        hot_vocab_local_rows([-1], shard_indices=None, local_rows=12)
    with pytest.raises(ValueError, match="full-width head of 12 rows"):
        hot_vocab_local_rows([5, -2], shard_indices=None, local_rows=12)
    # allow_unowned reports the negative id as unowned instead of selecting it.
    rows, positions, unowned = hot_vocab_local_rows(
        [-1, 11, 0], shard_indices=None, local_rows=12, allow_unowned=True
    )
    assert rows.tolist() == [11, 0] and positions.tolist() == [1, 2]
    assert unowned == [-1]
    # Valid boundaries on the TP1-shaped head: last row in, first row out.
    assert hot_vocab_local_rows([11], shard_indices=None, local_rows=12).tolist() == [11]
    with pytest.raises(ValueError, match="full-width head of 12 rows"):
        hot_vocab_local_rows([12], shard_indices=None, local_rows=12)


# --------------------------------------------------------------------------
# The base formula, pinned behaviorally (documents the red baseline)
# --------------------------------------------------------------------------


def test_base_global_id_selection_is_wrong_on_one_shard_and_raises_on_the_other():
    """The reporter's failure class, reproduced with the base selection call.

    Base did ``select_rowwise_weight_rows(head.weight, hot_token_id)`` (or
    ``head.data[hot_token_id]`` for BF16) with *global* ids against this
    rank's shard. On rank 1, ids owned by rank 0 silently select different
    rows -- carrying the wrong rowwise scale with them -- and an id at/beyond
    the shard width raises index-out-of-bounds at startup.
    """
    layer, weight = _shard_layer(rank=1, rowwise=True)
    assert weight.shape == (ROWS_PER_RANK, K)

    # Ids 0 and 5 are owned by rank 0; base indexes this shard's rows 0/5
    # with them -- which hold the rows for globals 6 and (padding), not 0/5.
    base_selected = select_rowwise_weight_rows(weight, torch.tensor([0, 5]))
    intended = _full_weight()[[0, 5]]
    got = dequantize_rowwise_weight(base_selected)
    assert not torch.allclose(got, intended.to(torch.bfloat16), atol=1e-2), (
        "base selection suddenly picks the right rows on a shard; the "
        "regression this guards changed shape"
    )
    # The scales travelled with the shard rows, not the intended globals.
    torch.testing.assert_close(
        rowwise_scale_of(base_selected), rowwise_scale_of(weight)[[0, 5]]
    )
    # A rank-0-added id exceeds this rank's resident width: startup-time OOB.
    with pytest.raises(IndexError):
        weight.index_select(0, torch.tensor([11]))


# --------------------------------------------------------------------------
# TP1: the qualified path stays pure-local, identical to base
# --------------------------------------------------------------------------


class _DraftModel:
    hot_token_id = None

    def __init__(self):
        self.installed = None

    def set_embed_and_head(self, embed, head):
        self.installed = (embed, head)


class _NonEagle3:
    @staticmethod
    def is_eagle3():
        return False


def _tp1_head():
    linear = nn.Linear(K, 12, bias=False, dtype=torch.bfloat16)
    linear.weight.data.copy_(_full_weight()[:12].to(torch.bfloat16))
    replace_linear_weight_rowwise_fp8(linear)
    return linear.weight


def _tp1_worker(weight, draft_model):
    return SimpleNamespace(
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    lm_head=SimpleNamespace(weight=weight),
                    get_embed_and_head=lambda: (
                        nn.Parameter(torch.ones(12, K)),
                        weight,
                    ),
                )
            )
        ),
        draft_runner=SimpleNamespace(
            model=draft_model, logits_processor=None, spec_algorithm=None
        ),
        hot_token_id=torch.tensor([11, 2, 0]),
        speculative_algorithm=_NonEagle3(),
    )


def test_tp1_shared_head_selection_is_local_and_identical_to_base_formula():
    weight = _tp1_head()
    hot = torch.tensor([11, 2, 0])

    def forbidden_exchange(_payload):
        raise AssertionError("a full-width head must not perform a collective")

    selection = shared_hot_lm_head(
        weight,
        hot_token_id=hot,
        target_lm_head=SimpleNamespace(weight=weight),
        exchange=forbidden_exchange,
    )
    assert not selection.replicated_full_width
    assert selection.draft_vocab_size == 3
    installed = selection.head
    # Byte-for-byte what the base in-place selection produced on TP1.
    base = select_rowwise_weight_rows(weight, hot)
    torch.testing.assert_close(installed.float(), base.float())
    torch.testing.assert_close(
        rowwise_scale_of(installed), rowwise_scale_of(weight)[[11, 2, 0]]
    )
    torch.testing.assert_close(
        dequantize_rowwise_weight(installed),
        dequantize_rowwise_weight(weight)[[11, 2, 0]],
    )


def test_tp1_worker_hook_stays_collective_free_and_silent(caplog):
    weight = _tp1_head()
    draft_model = _DraftModel()
    worker = _tp1_worker(weight, draft_model)
    with caplog.at_level(logging.INFO):
        EagleDraftWorker.init_lm_head(worker)
    installed = draft_model.installed[1]
    torch.testing.assert_close(
        rowwise_scale_of(installed), rowwise_scale_of(weight)[[11, 2, 0]]
    )
    assert "hot-vocabulary" not in caplog.text.lower(), (
        "TP1 must not receive the FR-Spec TP assembly message it could "
        "mistake for a warning"
    )


def test_tp1_out_of_vocab_hot_id_raises_rather_than_filtering():
    weight = _tp1_head()
    with pytest.raises(ValueError, match="full-width head of 12 rows"):
        shared_hot_lm_head(
            weight,
            hot_token_id=torch.tensor([11, 99]),
            target_lm_head=SimpleNamespace(weight=weight),
        )


# --------------------------------------------------------------------------
# Simulated TP2: assembled replicated head keeps values and scales together
# --------------------------------------------------------------------------


def _contributions(hot_ids):
    """Real contributions for both ranks over the shared hot id list."""
    result = {}
    for rank in range(TP_SIZE):
        layer, weight = _shard_layer(rank, rowwise=True)
        result[rank] = (
            layer,
            weight,
            hot_vocab._rank_contributions(
                head_weight=weight,
                hot_token_id=torch.tensor(hot_ids),
                target_lm_head=layer,
                local_rows=weight.shape[0],
                shard_indices=layer.shard_indices,
            ),
        )
    return result


def test_tp2_assembled_head_rows_and_scales_follow_the_hot_order():
    parts = _contributions(HOT_GLOBAL_IDS)
    for rank in range(TP_SIZE):
        layer, weight, _ = parts[rank]
        peer = parts[1 - rank][2]
        selection = shared_hot_lm_head(
            weight,
            hot_token_id=torch.tensor(HOT_GLOBAL_IDS),
            target_lm_head=layer,
            exchange=lambda payload, peer=peer: [payload, peer],
        )
        assert selection.replicated_full_width
        assert selection.draft_vocab_size == len(HOT_GLOBAL_IDS)
        installed = selection.head
        assert isinstance(installed, nn.Parameter)
        assert installed.shape == (len(HOT_GLOBAL_IDS), K)
        scales = rowwise_scale_of(installed)
        assert scales is not None and scales.shape == (len(HOT_GLOBAL_IDS),)
        owner_weight = parts[OWNERS[HOT_GLOBAL_IDS[0]][0]][1]
        del owner_weight
        for position, global_id in enumerate(HOT_GLOBAL_IDS):
            owner_rank, owner_row = OWNERS[global_id]
            _, owner_weight = _shard_layer(owner_rank, rowwise=True)
            # Value bytes come from the owning shard's row...
            torch.testing.assert_close(
                installed[position].float(),
                owner_weight.data[owner_row].float(),
                atol=0,
                rtol=0,
            )
            # ...and the scale is that same row's scale (never a neighbor's).
            assert float(scales[position]) == pytest.approx(
                float(rowwise_scale_of(owner_weight)[owner_row])
            )


def _meta_like(weight):
    """Same shape/dtype/scale-metadata as ``weight`` but on another device.

    ``meta`` is the CPU-only stand-in for "a device this process cannot
    touch": shape/dtype/index arithmetic resolves, and any data-dependent
    CPU read (``bool()``, ``nonzero()``, ``.item()``) raises, so the
    assembly path must never smuggle tensors back to the host.
    """
    from sglang.kernels.ops.gemm.sm120_online_fp8 import _SCALE_ATTR

    parameter = nn.Parameter(weight.data.to("meta"), requires_grad=False)
    scale = rowwise_scale_of(weight)
    if scale is not None:
        setattr(parameter, _SCALE_ATTR, scale.to("meta"))
    return parameter


def _assert_head_device_contract(installed, expected_device):
    assert installed.device == expected_device, (
        "the draft installs the assembled head without its own device move "
        "(Qwen3.5MTP's setter only assigns; the CUDA logits path rejects a "
        "weight on a different device), so hot_vocab must assemble on the "
        "source head's device"
    )
    scale = rowwise_scale_of(installed)
    assert scale is not None, "the rowwise scale attribute must travel"
    assert scale.device == installed.device, (
        "Tensor.to does not carry custom attributes; head and scale must be "
        "placed together at assembly time"
    )
    assert isinstance(installed, nn.Parameter)
    assert installed.shape == (len(HOT_GLOBAL_IDS), K)
    assert scale.shape == (len(HOT_GLOBAL_IDS),)


def test_tp2_assembled_head_and_scale_stay_on_the_head_device(monkeypatch):
    """P1: the replicated head (and its scale) come back on the head device.

    The peer contribution was computed on CPU (object comm), which is
    exactly how the earlier regression assembled the final head on CPU
    despite a CUDA target. Assembling on a ``meta`` clone of the real shard
    catches that on this CPU-only host: meta resolves shape/dtype/index
    arithmetic, and any data-dependent CPU read of the target-device
    tensors (bool/nonzero/item) would raise, so the assembly path cannot
    be smuggling the result back to the host. The local rank's payload
    stays the real CPU bytes (a live CUDA rank D2H-copies it; meta cannot
    carry data), so _rank_contributions is pinned to it.
    """
    parts = _contributions(HOT_GLOBAL_IDS)
    layer, weight, mine = parts[0]
    peer = parts[1][2]
    meta_weight = _meta_like(weight)
    monkeypatch.setattr(
        hot_vocab,
        "_rank_contributions",
        lambda **_kwargs: mine,
    )
    selection = shared_hot_lm_head(
        meta_weight,
        hot_token_id=torch.tensor(HOT_GLOBAL_IDS),
        target_lm_head=layer,
        exchange=lambda payload, peer=peer: [payload, peer],
    )
    assert selection.replicated_full_width
    _assert_head_device_contract(selection.head, torch.device("meta"))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="live CUDA placement pending on a GPU host; the meta contract "
    "above runs here and the parent stays CPU-safe",
)
def test_tp2_assembled_head_and_scale_stay_on_cuda():
    parts = _contributions(HOT_GLOBAL_IDS)
    layer, weight, _ = parts[0]
    peer = parts[1][2]
    device = torch.device("cuda:0")
    from sglang.kernels.ops.gemm.sm120_online_fp8 import _SCALE_ATTR

    cuda_weight = nn.Parameter(weight.data.to(device), requires_grad=False)
    scale = rowwise_scale_of(weight)
    assert scale is not None
    setattr(cuda_weight, _SCALE_ATTR, scale.to(device))
    selection = shared_hot_lm_head(
        cuda_weight,
        hot_token_id=torch.tensor(HOT_GLOBAL_IDS, device=device),
        target_lm_head=layer,
        exchange=lambda payload, peer=peer: [payload, peer],
    )
    assert selection.replicated_full_width
    _assert_head_device_contract(selection.head, device)


def test_tp2_worker_installs_replicated_head_and_stops_the_draft_allgather():
    import sglang.srt.distributed.parallel_state as parallel_state

    parts = _contributions(HOT_GLOBAL_IDS)
    layer, weight, _ = parts[0]
    peer = parts[1][2]

    class _Group:
        world_size = TP_SIZE

        def all_gather_object(self, obj):
            return [obj, peer]

    draft_logits = SimpleNamespace(
        do_tensor_parallel_all_gather=True,
        do_tensor_parallel_all_gather_dp_attn=True,
    )
    draft_model = _DraftModel()
    # The real draft model owns its logits_processor as a submodule; the
    # worker hook reaches it through model.logits_processor.
    draft_model.logits_processor = draft_logits
    runner = SimpleNamespace(model=draft_model, spec_algorithm=None)
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    lm_head=layer,
                    get_embed_and_head=lambda: (
                        nn.Parameter(torch.ones(TOTAL_PADDED, K)),
                        weight,
                    ),
                )
            )
        ),
        draft_runner=runner,
        hot_token_id=torch.tensor(HOT_GLOBAL_IDS),
        speculative_algorithm=_NonEagle3(),
    )

    original = parallel_state.get_tp_group
    parallel_state.get_tp_group = lambda: _Group()
    try:
        EagleDraftWorker.init_lm_head(worker)
    finally:
        parallel_state.get_tp_group = original

    installed = draft_model.installed[1]
    assert installed.shape == (len(HOT_GLOBAL_IDS), K)
    # No TP all-gather may stack on an already full-width hot head (the TP1
    # FR-Spec contract), and graph buffers learn the hot width the way the
    # draft-extend runner's len(hot_token_id) rule already does.
    assert draft_logits.do_tensor_parallel_all_gather is False
    assert draft_logits.do_tensor_parallel_all_gather_dp_attn is False
    assert runner.hot_vocab_width == len(HOT_GLOBAL_IDS)


def test_tp2_unowned_hot_id_raises_fail_closed():
    """A hot id no shard owns aborts startup; nothing is filtered away."""
    ids = [6, 15]  # 6 lives on rank 1; 15 is padding on both ranks
    parts = _contributions(ids)
    layer, weight, mine = parts[0]
    assert mine["positions"] == []  # rank 0 owns neither id

    with pytest.raises(RuntimeError, match="claimed by no TP rank"):
        shared_hot_lm_head(
            weight,
            hot_token_id=torch.tensor(ids),
            target_lm_head=layer,
            exchange=lambda payload: [payload, parts[1][2]],
        )
    # The same hole is visible from the owning rank too (identical gathered
    # list on every rank -> identical error, no rank divergence).
    other_layer, other_weight, _ = parts[1]
    with pytest.raises(RuntimeError, match="claimed by no TP rank"):
        shared_hot_lm_head(
            other_weight,
            hot_token_id=torch.tensor(ids),
            target_lm_head=other_layer,
            exchange=lambda payload: [payload, parts[0][2]],
        )


def test_tp2_id_claimed_by_nobody_raises_after_the_gather():
    """Agreement says covered, but if the owning contribution is emptied, fail."""
    ids = [6, 11]  # 6 on rank 1, 11 (added) on rank 0
    parts = _contributions(ids)
    layer, weight, mine = parts[0]
    peer = dict(parts[1][2])
    assert mine["positions"] == [1] and parts[1][2]["positions"] == [0]
    peer["positions"] = []
    peer["data"] = peer["data"][:0]
    peer["scale"] = peer["scale"][:0] if peer["scale"] is not None else None
    with pytest.raises(RuntimeError, match="claimed by no TP rank"):
        shared_hot_lm_head(
            weight,
            hot_token_id=torch.tensor(ids),
            target_lm_head=layer,
            exchange=lambda payload: [payload, peer],
        )


def test_tp2_duplicate_claim_raises():
    ids = [6, 11]
    parts = _contributions(ids)
    layer, weight, mine = parts[0]
    peer = dict(parts[1][2])
    # Both ranks claim position 1 (rank 1 lying about an owned row).
    peer["positions"] = [0, 1]
    other_layer, other_weight = _shard_layer(1, rowwise=True)
    peer["data"] = other_weight.data[[0, 3]][: len(peer["positions"])]
    peer["scale"] = rowwise_scale_of(other_weight)[[0, 3]]
    with pytest.raises(RuntimeError, match="more than one"):
        shared_hot_lm_head(
            weight,
            hot_token_id=torch.tensor(ids),
            target_lm_head=layer,
            exchange=lambda payload: [payload, peer],
        )
