"""CPU regressions for fixed-capacity QSA extend compression plans."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")

COMPRESS_RATIO = 4


@pytest.mark.parametrize("prefix_len", [1, 2, 3])
def test_qsa_extend_write_plan_accepts_unaligned_prefix(prefix_len):
    """A continuing chunk may start after a private, incomplete QSA group."""

    extend_len = COMPRESS_RATIO - prefix_len
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens=torch.tensor([extend_len], dtype=torch.int32),
        input_ids=torch.zeros(extend_len, dtype=torch.int32),
    )
    backend = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(qsa_compress_ratio=COMPRESS_RATIO),
        _qsa_write_plan=QwenSparseAttnBackend._qsa_write_plan,
    )
    plan = QwenSparseAttnBackend._qsa_build_write_plan(
        backend,
        forward_batch=forward_batch,
        speculative_paged=False,
        token_slot_table=torch.arange(64, 72, dtype=torch.int32).view(1, -1),
        sequence_lengths=torch.tensor([COMPRESS_RATIO], dtype=torch.int32),
    )

    _, _, rows, member_rows, valid, prefix_members = plan
    assert rows.tolist() == [0]
    assert member_rows.tolist() == [-prefix_len]
    assert valid.tolist() == [True]
    assert prefix_members.tolist() == [prefix_len]


def test_qsa_extend_write_plan_preserves_aligned_prefix_control():
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens=torch.tensor([COMPRESS_RATIO], dtype=torch.int32),
        input_ids=torch.zeros(COMPRESS_RATIO, dtype=torch.int32),
    )
    backend = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(qsa_compress_ratio=COMPRESS_RATIO),
        _qsa_write_plan=QwenSparseAttnBackend._qsa_write_plan,
    )
    plan = QwenSparseAttnBackend._qsa_build_write_plan(
        backend,
        forward_batch=forward_batch,
        speculative_paged=False,
        token_slot_table=torch.arange(64, 76, dtype=torch.int32).view(1, -1),
        sequence_lengths=torch.tensor([12], dtype=torch.int32),
    )

    _, positions, _, member_rows, valid, prefix_members = plan
    assert positions.tolist() == [11, 3]
    assert member_rows.tolist() == [0, 0]
    assert valid.tolist() == [True, False]
    assert prefix_members.tolist() == [0, 0]


class _RecordingCompressedPool:
    def __init__(self):
        self.write_locs = None
        self.compressed = None

    def set_qsa_compressed_k_buffer(self, layer_id, locs, compressed):
        self.write_locs = locs.clone()
        self.compressed = compressed.clone()


class _RecordingRingPool:
    index_state_dtype = torch.float32

    def __init__(self, ring_slots=32, compressed_slots=64, device="cpu"):
        self.key_state = torch.zeros(
            ring_slots, 1, 1, dtype=torch.float32, device=device
        )
        self.qsa_rope_position_buffer = torch.zeros(
            ring_slots, 3, dtype=torch.long, device=device
        )
        self.compressed = torch.full(
            (compressed_slots, 1, 1),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )

    def get_qsa_key_state_buffer(self, layer_id):
        return self.key_state

    def set_qsa_key_state_buffer(self, layer_id, locs, keys):
        self.key_state[locs.long()] = keys

    def set_qsa_rope_position_buffer(self, locs, positions):
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        self.qsa_rope_position_buffer[locs.long()] = positions.transpose(0, 1)

    def set_qsa_compressed_k_buffer(self, layer_id, locs, compressed):
        self.compressed[locs.long()] = compressed


class _RecordingMultiLayerRingPool(_RecordingRingPool):
    def __init__(self, num_layers=2):
        super().__init__()
        self.key_states = [self.key_state.clone() for _ in range(num_layers)]
        self.compressed_layers = [self.compressed.clone() for _ in range(num_layers)]

    def get_qsa_key_state_buffer(self, layer_id):
        return self.key_states[layer_id]

    def set_qsa_key_state_buffer(self, layer_id, locs, keys):
        self.key_states[layer_id][locs.long()] = keys

    def set_qsa_compressed_k_buffer(self, layer_id, locs, compressed):
        self.compressed_layers[layer_id][locs.long()] = compressed


def test_qsa_cross_prefix_snapshots_shared_rope_before_any_layer_updates_it():
    pool = _RecordingMultiLayerRingPool()
    pool.key_states[0][8:10, 0, 0] = torch.tensor([10.0, 20.0])
    pool.key_states[1][8:10, 0, 0] = torch.tensor([100.0, 200.0])
    pool.qsa_rope_position_buffer[8] = torch.tensor([1000, 1001, 1002])
    current_positions = torch.arange(2, 9)
    current_rope = torch.stack(
        (current_positions, current_positions + 100, current_positions + 200)
    )
    metadata = SimpleNamespace(
        token_to_kv_pool=pool,
        token_to_batch_idx=torch.zeros(7, dtype=torch.int32),
        req_pool_indices=torch.tensor([2], dtype=torch.int32),
        sequence_lengths=torch.tensor([9]),
        compress_member_rows=torch.tensor([-2, 2]),
        compress_prefix_members=torch.tensor([2, 0]),
        compress_plan_valid=torch.tensor([True, True]),
        has_cross_prefix_group=True,
        compress_group_ring_locs=torch.tensor([[8, 9, 10, 11], [8, 9, 10, 11]]),
        cross_prefix_rope_positions=pool.qsa_rope_position_buffer.index_select(
            0, torch.tensor([8, 8])
        ),
        compress_group_positions=torch.tensor([3, 7]),
        compress_sequence_ids=torch.tensor([0, 0]),
        write_locs=torch.tensor([16, 17], dtype=torch.int32),
        extend_rope_matrix=current_rope.transpose(0, 1),
        is_cuda_graph=False,
    )
    state_slots = build_pending_ring_slots(
        token_to_batch_idx=metadata.token_to_batch_idx,
        req_pool_indices=metadata.req_pool_indices,
        sequence_lengths=metadata.sequence_lengths,
        logical_positions=current_positions,
        compress_ratio=COMPRESS_RATIO,
        is_extend=True,
    )
    current_keys = torch.tensor(
        [
            [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 900.0],
            [300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 9000.0],
        ]
    ).reshape(2, -1, 1, 1)
    normalized_positions = []
    for layer_id in range(2):
        layer_positions = []

        def normalize(keys, positions):
            layer_positions.append(positions.clone())
            return keys

        indexer = SimpleNamespace(
            layer_id=layer_id,
            compress_ratio=COMPRESS_RATIO,
            _use_fused_compress=lambda pool: False,
            _rope_from_matrix=lambda positions: positions.transpose(0, 1),
            normalize_compressed_keys=normalize,
        )
        indexer._compress_cross_prefix_groups = MethodType(
            QSAIndexer._compress_cross_prefix_groups, indexer
        )
        QSAIndexer.update_key_state_and_compress(
            indexer,
            current_keys[layer_id],
            current_positions,
            current_rope,
            metadata,
            state_slots=state_slots,
        )
        normalized_positions.append(layer_positions[0])

    assert pool.compressed_layers[0][16, 0, 0].item() == 25.0
    assert pool.compressed_layers[1][16, 0, 0].item() == 250.0
    expected_rope = torch.tensor([[1000], [1001], [1002]])
    assert torch.equal(normalized_positions[0][:, :1], expected_rope)
    assert torch.equal(normalized_positions[1][:, :1], expected_rope)


def _run_cross_prefix_compression(*, fused, device="cpu"):
    prefix_lens = torch.tensor([2, 3], dtype=torch.long, device=device)
    extend_lens = torch.tensor([7, 1], dtype=torch.long, device=device)
    sequence_lengths = prefix_lens + extend_lens
    row_token_starts = torch.tensor([0, 7], dtype=torch.long, device=device)
    token_slot_table = torch.stack(
        (
            torch.arange(64, 80, dtype=torch.int32, device=device),
            torch.arange(128, 144, dtype=torch.int32, device=device),
        )
    )
    plan = QwenSparseAttnBackend._qsa_write_plan(
        token_slot_table=token_slot_table,
        start_blocks=prefix_lens // COMPRESS_RATIO,
        end_blocks=sequence_lengths // COMPRESS_RATIO,
        capacity=4,
        compress_ratio=COMPRESS_RATIO,
        row_token_starts=row_token_starts,
        prefix_lens=prefix_lens,
    )
    write_locs, positions, sequence_ids, members, valid, prefix_members = plan
    req_pool_indices = torch.tensor([2, 3], dtype=torch.int32, device=device)
    ring_group_locs = build_group_ring_slots(
        req_pool_indices=req_pool_indices,
        group_end_positions=positions,
        sequence_ids=sequence_ids,
        compress_ratio=COMPRESS_RATIO,
    )
    logical_positions = torch.tensor(
        [2, 3, 4, 5, 6, 7, 8, 3], dtype=torch.long, device=device
    )
    token_to_batch = torch.tensor([0] * 7 + [1], dtype=torch.int32, device=device)
    state_slots = build_pending_ring_slots(
        token_to_batch_idx=token_to_batch,
        req_pool_indices=req_pool_indices,
        sequence_lengths=sequence_lengths,
        logical_positions=logical_positions,
        compress_ratio=COMPRESS_RATIO,
        is_extend=True,
    )

    pool = _RecordingRingPool(device=device)
    # Retained private tails: request 2 has positions 0/1; request 3 has 0/1/2.
    pool.key_state[8:10, 0, 0] = torch.tensor([10.0, 20.0], device=device)
    pool.key_state[12:15, 0, 0] = torch.tensor([100.0, 200.0, 300.0], device=device)
    pool.qsa_rope_position_buffer[8] = torch.tensor([1000, 1001, 1002], device=device)
    pool.qsa_rope_position_buffer[12] = torch.tensor([2000, 2001, 2002], device=device)
    token_k = torch.tensor(
        [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 900.0, 400.0],
        device=device,
    ).reshape(-1, 1, 1)
    rope_positions = torch.stack(
        (logical_positions, logical_positions + 100, logical_positions + 200)
    )
    metadata = SimpleNamespace(
        token_to_kv_pool=pool,
        token_to_batch_idx=token_to_batch,
        req_pool_indices=req_pool_indices,
        sequence_lengths=sequence_lengths,
        compress_member_rows=members,
        compress_prefix_members=prefix_members,
        compress_plan_valid=valid,
        has_cross_prefix_group=True,
        compress_group_ring_locs=ring_group_locs,
        cross_prefix_rope_positions=pool.qsa_rope_position_buffer.index_select(
            0, ring_group_locs[:, 0].long()
        ),
        compress_group_positions=positions,
        compress_sequence_ids=sequence_ids,
        write_locs=write_locs,
        extend_rope_matrix=rope_positions.transpose(0, 1),
        is_cuda_graph=False,
    )
    normalized_positions = []

    def normalize(keys, positions):
        normalized_positions.append(positions.clone())
        return keys

    def record_fused(pool, locs, writes, *, source_keys, source_rope):
        pool.set_qsa_compressed_k_buffer(0, writes, source_keys[locs].mean(dim=1))

    indexer = SimpleNamespace(
        layer_id=0,
        compress_ratio=COMPRESS_RATIO,
        _use_fused_compress=lambda pool: fused,
        _fused_compress_store=record_fused,
        _rope_from_matrix=lambda positions: positions.transpose(0, 1),
        normalize_compressed_keys=normalize,
    )
    indexer._compress_cross_prefix_groups = MethodType(
        QSAIndexer._compress_cross_prefix_groups, indexer
    )
    QSAIndexer.update_key_state_and_compress(
        indexer,
        token_k,
        logical_positions,
        rope_positions,
        metadata,
        state_slots=state_slots,
        state_stored=False,
    )
    return pool, write_locs, normalized_positions


@pytest.mark.parametrize("fused", [False, True])
def test_qsa_cross_prefix_uses_old_ring_before_wrapping_tail(fused):
    pool, write_locs, normalized_positions = _run_cross_prefix_compression(fused=fused)

    # The long first chunk leaves position 8 pending, which wraps ring slot 8
    # from old value 10 to new value 900. Compression must already have used
    # the old value for group [0, 4), while group [4, 8) stays chunk-local.
    assert pool.key_state[8, 0, 0].item() == 900.0
    assert pool.compressed[write_locs[0], 0, 0].item() == 25.0
    assert pool.compressed[write_locs[1], 0, 0].item() == 65.0
    assert pool.compressed[write_locs[2], 0, 0].item() == 250.0
    # Crossed groups rotate from the first prefix member's retained MRoPE row.
    assert normalized_positions[0][:, [0, 2]].tolist() == [
        [1000, 2000],
        [1001, 2001],
        [1002, 2002],
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fused", [False, True])
def test_qsa_cross_prefix_cuda_preserves_old_ring_generation(fused):
    pool, write_locs, normalized_positions = _run_cross_prefix_compression(
        fused=fused, device="cuda"
    )

    torch.testing.assert_close(
        pool.compressed[write_locs[:3].long(), 0, 0],
        torch.tensor([25.0, 65.0, 250.0], device="cuda"),
    )
    torch.testing.assert_close(
        normalized_positions[0][:, [0, 2]],
        torch.tensor([[1000, 2000], [1001, 2001], [1002, 2002]], device="cuda"),
    )
    assert pool.key_state[8, 0, 0].item() == 900.0
    torch.cuda.synchronize()


def test_qsa_short_unaligned_extend_retains_incomplete_group():
    pool = _RecordingRingPool()
    pool.key_state[8, 0, 0] = 10.0
    pool.qsa_rope_position_buffer[8] = torch.tensor([10, 11, 12])
    token_k = torch.tensor([20.0]).reshape(-1, 1, 1)
    metadata = SimpleNamespace(
        token_to_kv_pool=pool,
        token_to_batch_idx=torch.tensor([0], dtype=torch.int32),
        req_pool_indices=torch.tensor([2], dtype=torch.int32),
        sequence_lengths=torch.tensor([2], dtype=torch.int32),
        compress_member_rows=torch.tensor([0], dtype=torch.long),
        compress_prefix_members=torch.tensor([0], dtype=torch.long),
        compress_plan_valid=torch.tensor([False]),
        has_cross_prefix_group=True,
        compress_group_ring_locs=torch.tensor([[8, 9, 10, 11]]),
        cross_prefix_rope_positions=pool.qsa_rope_position_buffer.index_select(
            0, torch.tensor([8])
        ),
        compress_group_positions=torch.tensor([3]),
        compress_sequence_ids=torch.tensor([0]),
        write_locs=torch.tensor([0], dtype=torch.int32),
        extend_rope_matrix=torch.tensor([[1, 2, 3]], dtype=torch.long),
        is_cuda_graph=False,
    )
    indexer = SimpleNamespace(
        layer_id=0,
        compress_ratio=COMPRESS_RATIO,
        _use_fused_compress=lambda pool: False,
        _rope_from_matrix=lambda positions: positions.transpose(0, 1),
        normalize_compressed_keys=lambda keys, positions: keys,
    )
    indexer._compress_cross_prefix_groups = MethodType(
        QSAIndexer._compress_cross_prefix_groups, indexer
    )
    QSAIndexer.update_key_state_and_compress(
        indexer,
        token_k,
        torch.tensor([1]),
        torch.tensor([1]),
        metadata,
        state_slots=torch.tensor([9]),
    )

    assert pool.key_state[8:10, 0, 0].tolist() == [10.0, 20.0]


def _run_extend_compression_plan(token_values, end_blocks, capacity, *, fused=False):
    token_k = torch.tensor(token_values, dtype=torch.float32).reshape(-1, 1, 1)
    token_slot_table = torch.arange(4, 20, dtype=torch.int32).reshape(1, -1)
    plan = QwenSparseAttnBackend._qsa_write_plan(
        token_slot_table=token_slot_table,
        start_blocks=torch.zeros(1, dtype=torch.long),
        end_blocks=torch.tensor([end_blocks], dtype=torch.long),
        capacity=capacity,
        compress_ratio=COMPRESS_RATIO,
        row_token_starts=torch.zeros(1, dtype=torch.long),
        prefix_lens=torch.zeros(1, dtype=torch.long),
    )
    (
        write_locs,
        group_positions,
        group_sequence_ids,
        member_rows,
        plan_valid,
        _prefix_members,
    ) = plan
    pool = _RecordingCompressedPool()
    metadata = SimpleNamespace(
        token_to_kv_pool=pool,
        compress_member_rows=member_rows,
        compress_plan_valid=plan_valid,
        is_cuda_graph=False,
        write_locs=write_locs,
        compress_group_positions=group_positions,
        extend_rope_matrix=torch.zeros(token_k.shape[0], 3, dtype=torch.long),
    )

    def record_fused(pool, locs, writes, *, source_keys, source_rope):
        # Exercise the actual dispatch's arguments on CPU. The GPU counterpart
        # runs the real fused kernel separately, including under memcheck.
        pool.set_qsa_compressed_k_buffer(0, writes, source_keys[locs].mean(1))

    indexer = SimpleNamespace(
        layer_id=0,
        compress_ratio=COMPRESS_RATIO,
        _use_fused_compress=lambda pool: fused,
        _fused_compress_store=record_fused,
        _rope_from_matrix=lambda positions: positions[:, 0],
        normalize_compressed_keys=lambda keys, positions: keys,
    )
    QSAIndexer.update_key_state_and_compress(
        indexer,
        token_k,
        torch.arange(token_k.shape[0]),
        torch.arange(token_k.shape[0]),
        metadata,
        state_stored=True,
    )
    return pool, plan_valid, group_sequence_ids


@pytest.mark.parametrize("extend_len", [1, 2, 3])
@pytest.mark.parametrize("fused", [False, True])
def test_qsa_short_extend_padding_uses_in_bounds_dummy_reads(extend_len, fused):
    pool, plan_valid, group_sequence_ids = _run_extend_compression_plan(
        list(range(1, extend_len + 1)), end_blocks=0, capacity=1, fused=fused
    )

    assert plan_valid.tolist() == [False]
    assert group_sequence_ids.tolist() == [0]
    assert pool.write_locs.tolist() == [0]
    assert pool.compressed.flatten().tolist() == [1.0]


@pytest.mark.parametrize("fused", [False, True])
def test_qsa_extend_padding_does_not_change_real_group_compression(fused):
    pool, plan_valid, _ = _run_extend_compression_plan(
        [0.0, 1.0, 2.0, 3.0], end_blocks=1, capacity=2, fused=fused
    )

    assert plan_valid.tolist() == [True, False]
    assert pool.write_locs.tolist() == [1, 0]
    assert pool.compressed.flatten().tolist() == [1.5, 0.0]


@pytest.mark.parametrize("fused", [False, True])
def test_qsa_malformed_real_group_is_not_silently_clamped(fused):
    # Deliberately inconsistent key extent: the valid group requires four
    # rows. Unlike padding, it must not silently replicate the last row.
    with pytest.raises(IndexError):
        _run_extend_compression_plan([1.0, 2.0, 3.0], 1, 2, fused=fused)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("extend_len", [1, 2, 3, 4, 5, 7, 8])
@pytest.mark.parametrize("physical_padding", [0, 3])
def test_qsa_extend_plan_real_fused_kernel(extend_len, physical_padding):
    from sglang.kernels.ops.attention.qsa_indexer import qsa_index_k_compress_store

    device, dim = torch.device("cuda"), 128
    torch.manual_seed(extend_len)
    keys = torch.randn(
        extend_len + physical_padding, 1, dim, dtype=torch.bfloat16, device=device
    )
    table = torch.arange(64, 128, dtype=torch.int32, device=device)[None, :]
    writes, positions, _, members, valid, _ = QwenSparseAttnBackend._qsa_write_plan(
        token_slot_table=table,
        start_blocks=torch.zeros(1, dtype=torch.long, device=device),
        end_blocks=torch.tensor([extend_len // 4], device=device),
        capacity=extend_len // 4 + 1,
        compress_ratio=4,
        row_token_starts=torch.zeros(1, dtype=torch.long, device=device),
        prefix_lens=torch.zeros(1, dtype=torch.long, device=device),
    )
    rope = torch.zeros(keys.shape[0], 3, dtype=torch.long, device=device)
    cos_sin = torch.cat(
        (
            torch.ones(1, dim // 2, device=device),
            torch.zeros(1, dim // 2, device=device),
        ),
        dim=1,
    )
    axis_map = torch.zeros(dim // 2, dtype=torch.int32, device=device)
    weight = torch.zeros(dim, dtype=keys.dtype, device=device)
    output = torch.full((32, dim), float("nan"), dtype=keys.dtype, device=device)

    def fused_store(pool, locs, write_locs, *, source_keys, source_rope):
        qsa_index_k_compress_store(
            source_keys.view(-1, dim),
            locs.int(),
            source_rope,
            cos_sin,
            axis_map,
            weight,
            write_locs,
            output,
            4,
            dim,
            1e-6,
            True,
        )

    metadata = SimpleNamespace(
        token_to_kv_pool=None,
        compress_member_rows=members,
        compress_plan_valid=valid,
        is_cuda_graph=False,
        write_locs=writes,
        compress_group_positions=positions,
        extend_rope_matrix=rope,
    )
    indexer = SimpleNamespace(
        compress_ratio=4,
        _use_fused_compress=lambda pool: True,
        _fused_compress_store=fused_store,
    )
    QSAIndexer.update_key_state_and_compress(
        indexer,
        keys,
        torch.arange(extend_len, device=device),
        torch.arange(extend_len, device=device),
        metadata,
        state_stored=True,
    )
    # Independent reference: complete groups plus one dummy read of row zero.
    real_groups = extend_len // 4
    means = keys[: real_groups * 4].view(real_groups, 4, dim).float().mean(1)
    means = torch.cat((means.to(keys.dtype), keys[0])).float()
    expected = (means * torch.rsqrt(means.square().mean(-1, keepdim=True) + 1e-6)).to(
        keys.dtype
    )
    torch.testing.assert_close(output[writes.long()], expected, rtol=0, atol=0.015625)
    torch.cuda.synchronize()
