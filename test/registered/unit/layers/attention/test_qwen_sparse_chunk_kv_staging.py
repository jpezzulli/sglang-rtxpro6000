"""CPU regressions for QSA chunk-prefill transient K/V staging.

QwenSparseAttnBackend.forward_extend's chunk-prefill fallback must join the
per-request gather INDICES and issue one index_select per K/V buffer.
Gathering per request and torch.cat-ing the gathered rows materialises a
second full-context copy of K and of V purely to concatenate it (4x the
packed peak instead of 2x) -- including a pointless copy for the single
request case, because torch.cat allocates even for a one-element list.  The
kernel-visible packed tensors must stay identical.
"""

from types import SimpleNamespace

import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_backend
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_REAL_CAT = torch.cat
_HEAD_DIM = 4
_HEADS = 2
_BUFFER_ROWS = 512
_ROW_ELEMS = _HEADS * _HEAD_DIM


class _CudaLikeTensor(torch.Tensor):
    """CPU-backed tensor that reports is_cuda so the chunk path is taken."""

    @property
    def is_cuda(self):
        return True


def _cuda_like(tensor: torch.Tensor) -> _CudaLikeTensor:
    return torch.Tensor._make_subclass(_CudaLikeTensor, tensor)


def _seed_buffer(offset: int) -> torch.Tensor:
    return (
        torch.arange(_BUFFER_ROWS * _ROW_ELEMS, dtype=torch.float32).reshape(
            _BUFFER_ROWS, _HEADS, _HEAD_DIM
        )
        + offset
    )


def _run_chunk_forward(monkeypatch, sequence_lens):
    """Drive forward_extend's chunk-prefill fallback with recording spies.

    extend < sequence per request so at least one prefix row exists (the
    chunk path requires committed prefixes).  Returns the recorded kernel
    call and a record of every torch.cat use made on the staging path.
    """
    k_buffer = _seed_buffer(0)
    v_buffer = _seed_buffer(1_000_000)
    table_rows = [
        torch.arange(10, 10 + sequence_lens[i], dtype=torch.int64)
        for i in range(len(sequence_lens))
    ]
    width = max(sequence_lens)
    req_to_token = _cuda_like(
        torch.stack(
            [
                torch.cat(
                    [
                        row,
                        torch.full((width - len(row),), -1, dtype=torch.int64),
                    ]
                )
                for row in table_rows
            ]
        )
    )

    calls = []
    cat_sizes = []
    real_cat = torch.cat

    def record_cat(tensors, *args, **kwargs):
        pieces = list(tensors)
        cat_sizes.append([tuple(t.shape) for t in pieces])
        return real_cat(pieces, *args, **kwargs)

    def fake_ck(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
        calls.append(
            {
                "k": k,
                "v": v,
                "cu_k": cu_k.clone(),
                "kv_lens": kv_lens.clone(),
            }
        )
        return torch.zeros(q.shape[0], q.shape[1], q.shape[2])

    monkeypatch.setattr(qsa_backend, "sparse_gqa_fwd_interface_triton_ck", fake_ck)
    monkeypatch.setattr(torch, "cat", record_cat)

    total_tokens = sum(sequence_lens)
    extend_lens = [length - 2 for length in sequence_lens]
    q = _cuda_like(
        torch.arange(total_tokens * _ROW_ELEMS, dtype=torch.float32).reshape(
            total_tokens, _HEADS, _HEAD_DIM
        )
    )
    layer = SimpleNamespace(
        layer_id=0, tp_q_head_num=_HEADS, head_dim=_HEAD_DIM, scaling=0.25
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens_cpu=extend_lens,
        seq_lens_cpu=sequence_lens,
        extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32),
        req_pool_indices=torch.arange(len(sequence_lens), dtype=torch.int64),
        out_cache_loc=torch.arange(total_tokens, dtype=torch.int64),
    )
    backend = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(
            get_key_buffer=lambda layer_id: _cuda_like(k_buffer),
            get_value_buffer=lambda layer_id: _cuda_like(v_buffer),
            set_kv_buffer=lambda *args, **kwargs: None,
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        _is_speculative_paged_mode=staticmethod(
            QwenSparseAttnBackend._is_speculative_paged_mode
        ),
        _pad_extend_output=staticmethod(QwenSparseAttnBackend._pad_extend_output),
    )
    topk_indices = _cuda_like(
        torch.zeros(total_tokens, 1, _HEADS, dtype=torch.int32)
    )

    QwenSparseAttnBackend.forward_extend(
        backend, q, q, q, layer, forward_batch, topk_indices=topk_indices
    )

    assert len(calls) == 1, f"chunk kernel launched {len(calls)} times"
    # cu_seqlens_q/cu_seqlens_k are F.pad(cumsum), never torch.cat; anything
    # recorded here came from the K/V staging path itself.  Snapshot before
    # returning so later patched-cat uses by the test body are not recorded.
    return calls[0], list(cat_sizes), k_buffer, v_buffer, table_rows


def _expected_packed(buffer, table_rows, sequence_lens):
    # Baseline semantics: per-request gather of the table rows, concatenated.
    gathered = [
        buffer.index_select(0, row[: length])
        for row, length in zip(table_rows, sequence_lens)
    ]
    return _REAL_CAT(gathered)


def test_multi_request_packed_kv_identical_to_baseline(monkeypatch):
    sequence_lens = [7, 5, 9]
    call, _cat_sizes, k_buffer, v_buffer, table_rows = _run_chunk_forward(
        monkeypatch, sequence_lens
    )
    assert torch.equal(
        call["k"], _expected_packed(k_buffer, table_rows, sequence_lens)
    )
    assert torch.equal(
        call["v"], _expected_packed(v_buffer, table_rows, sequence_lens)
    )
    assert call["cu_k"].tolist() == [0, 7, 12, 21]
    assert call["kv_lens"].tolist() == sequence_lens


def test_staging_joins_indices_not_gathered_kv(monkeypatch):
    sequence_lens = [64, 32]
    _call, cat_sizes, _k, _v, _rows = _run_chunk_forward(monkeypatch, sequence_lens)
    # Any torch.cat on the staging path may only join 1-D gather-index
    # vectors; concatenating 3-D (rows, heads, head_dim) tensors means a full
    # duplicate K/V copy was materialised purely to cat it.
    joined_index = False
    for pieces in cat_sizes:
        for shape in pieces:
            assert not (
                len(shape) == 3 and shape[1:] == (_HEADS, _HEAD_DIM)
            ), f"full-context K/V concat copy materialised: shape {shape}"
        if len(pieces) > 1 and all(len(shape) == 1 for shape in pieces):
            joined_index = True
    assert joined_index, "multi-request gather indices were never joined"


def test_single_request_packed_kv_without_cat_copy(monkeypatch):
    sequence_lens = [200]
    call, cat_sizes, k_buffer, v_buffer, table_rows = _run_chunk_forward(
        monkeypatch, sequence_lens
    )
    assert torch.equal(
        call["k"], _expected_packed(k_buffer, table_rows, sequence_lens)
    )
    assert torch.equal(
        call["v"], _expected_packed(v_buffer, table_rows, sequence_lens)
    )
    # torch.cat allocates even for a one-element list, so the length-1 case
    # must not stage a gathered-row cat at all.
    for pieces in cat_sizes:
        assert all(
            not (len(shape) == 3 and shape[1:] == (_HEADS, _HEAD_DIM))
            for shape in pieces
        ), f"single request still cats gathered rows: {pieces}"
