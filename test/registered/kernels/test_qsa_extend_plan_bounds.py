"""CPU regressions for fixed-capacity QSA extend compression plans."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")

COMPRESS_RATIO = 4


class _RecordingCompressedPool:
    def __init__(self):
        self.write_locs = None
        self.compressed = None

    def set_qsa_compressed_k_buffer(self, layer_id, locs, compressed):
        self.write_locs = locs.clone()
        self.compressed = compressed.clone()


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
    write_locs, group_positions, group_sequence_ids, member_rows, plan_valid = plan
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
    writes, positions, _, members, valid = QwenSparseAttnBackend._qsa_write_plan(
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
