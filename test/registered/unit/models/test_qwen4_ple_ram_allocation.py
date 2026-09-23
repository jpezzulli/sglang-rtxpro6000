"""CPU-only contracts for the Flash-Next RAM-PLE allocation path.

Covers the two adopted upstream fixes on the Penny base:

* sgl-project/sglang#39928: ``Qwen4ExpPLELayer`` builds its offloaded n-gram
  template on the meta device and wraps it inside ``Qwen4ExpNGramEmbedding``,
  so the pinned host table is the table's first and only allocation (the
  per-rank shard is never materialised on the accelerator first).
* sgl-project/sglang#40626: the pinned host table is an exact-sized anonymous
  mapping registered with ``cudaHostRegister``, not
  ``torch.empty(..., pin_memory=True)``, so a GiB-scale table locks its exact
  bytes instead of the next power of two.

Registration and ownership run against a recording ``cudaHostRegister`` so the
contract is checkable without a GPU; the supervisor SM120 box re-runs this file
against the real runtime. Non-offload construction (the NVMe PLE profile, whose
plugin forces ``ple_offload_embedding=False``, and the 27B/DFlash2 recipe,
which never enables it) is pinned to the unchanged device-resident path.
"""

import gc
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.models import qwen4_exp as qwen4_exp_module
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpPinnedHostEmbedding,
    Qwen4ExpPLELayer,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_PORTABLE_MAPPED = 0x01 | 0x02


class RecordingCudart:
    def __init__(self, result=0, error=None):
        self.calls = []
        self._result = result
        self._error = error

    def cudaHostRegister(self, ptr, nbytes, flags):
        self.calls.append((int(ptr), int(nbytes), int(flags)))
        if self._error is not None:
            raise self._error
        return self._result


@pytest.fixture
def cudart(monkeypatch):
    recorder = RecordingCudart()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: recorder)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: SimpleNamespace())
    del qwen4_exp_module._PINNED_TABLE_BUFFERS[:]
    yield recorder
    del qwen4_exp_module._PINNED_TABLE_BUFFERS[:]


@pytest.fixture
def tracking_mmap(monkeypatch):
    created = []
    real_mmap = qwen4_exp_module.mmap.mmap

    def recording_mmap(*args, **kwargs):
        buf = real_mmap(*args, **kwargs)
        created.append(buf)
        return buf

    monkeypatch.setattr(qwen4_exp_module.mmap, "mmap", recording_mmap)
    return created


@pytest.fixture
def single_rank():
    """``VocabParallelEmbedding`` reads the TP topology; pin one rank."""
    override = get_context().override_server_args(tp_size=1)
    override.install()
    try:
        with get_parallel().override(
            tp_rank=0, tp_size=1, attn_tp_rank=0, attn_tp_size=1
        ):
            yield
    finally:
        override.restore()


def _config(**overrides):
    values = dict(
        vocab_size=64,
        hidden_size=16,
        hc_count=2,
        ple_embed_dim=64,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000,
        eos_token_id=1,
        ple_offload_embedding=True,
    )
    values.update(overrides)
    return Qwen4ExpTextConfig(**values)


def _build_layer(config):
    with torch.device("cpu"):  # the model loader builds every layer this way
        return Qwen4ExpPLELayer(
            config, prefix="model.layers.0.ple", layer_id=0, ple_layer_index=0
        )


def _table_bytes(emb):
    return emb.weight.numel() * emb.weight.element_size()


def test_offload_registers_exact_size_pinned_table(cudart, single_rank):
    layer = _build_layer(_config())
    emb = layer.ple_embedding.ngram_embedding

    assert isinstance(emb, Qwen4ExpPinnedHostEmbedding)
    assert emb.weight.device.type == "cpu"
    assert emb.weight.dtype == torch.bfloat16
    assert emb.weight.is_contiguous()
    assert tuple(emb.weight.shape) == (
        emb.num_embeddings_per_partition,
        emb.embedding_dim,
    )
    # Exactly one registration of exactly the table bytes, portable|mapped:
    # torch.empty(pin_memory=True) would have rounded this request up to the
    # next power of two (sgl-project/sglang#40626).
    nbytes = _table_bytes(emb)
    assert nbytes != 1 << (nbytes - 1).bit_length()  # not itself a power of two
    assert cudart.calls == [(emb.weight.data_ptr(), nbytes, _PORTABLE_MAPPED)]
    assert len(qwen4_exp_module._PINNED_TABLE_BUFFERS) == 1
    assert layer._prefetch_stream is not None


@pytest.mark.parametrize("ple_embedding_dtype", [None, "float8_e4m3fn"])
def test_offload_preserves_storage_dtype_and_shard_metadata(
    cudart, single_rank, ple_embedding_dtype
):
    baseline = _build_layer(_config(ple_offload_embedding=False))
    template = baseline.ple_embedding.ngram_embedding
    layer = _build_layer(_config(ple_embedding_dtype=ple_embedding_dtype))
    emb = layer.ple_embedding.ngram_embedding

    # Values/dtype, shard geometry and the embedding width survive the
    # meta-template -> host-table substitution unchanged.
    assert emb.weight.dtype == (
        torch.float8_e4m3fn
        if ple_embedding_dtype == "float8_e4m3fn"
        else torch.bfloat16
    )
    assert tuple(emb.weight.shape) == tuple(template.weight.shape)
    assert emb.shard_indices == template.shard_indices
    assert emb.embedding_dim == template.embedding_dim
    assert emb.num_embeddings_per_partition == template.num_embeddings_per_partition
    assert cudart.calls == [
        (emb.weight.data_ptr(), _table_bytes(emb), _PORTABLE_MAPPED)
    ]


def test_registration_failure_closes_mapping_and_fails_loud(
    cudart, tracking_mmap, single_rank, monkeypatch
):
    failing = RecordingCudart(error=RuntimeError("out of locked memory"))
    monkeypatch.setattr(torch.cuda, "cudart", lambda: failing)

    with pytest.raises(RuntimeError, match="unable to page-lock"):
        _build_layer(_config())
    assert len(tracking_mmap) == 1
    assert tracking_mmap[0].closed, "failed registration left the mapping open"
    assert qwen4_exp_module._PINNED_TABLE_BUFFERS == [], "failed table was registered"


def test_nonzero_cuda_error_is_cleaned_up(
    cudart, tracking_mmap, single_rank, monkeypatch
):
    failing = RecordingCudart(result=17)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: failing)

    with pytest.raises(RuntimeError, match="unable to page-lock"):
        _build_layer(_config())
    assert len(tracking_mmap) == 1
    assert tracking_mmap[0].closed
    assert qwen4_exp_module._PINNED_TABLE_BUFFERS == []


def test_offload_template_is_meta_and_host_table_is_the_only_allocation(
    cudart, single_rank, monkeypatch
):
    # sgl-project/sglang#39841: the wrapper previously received a template
    # whose table already lived on the build device, so enabling the offload
    # needed a full per-rank shard of VRAM before the host allocation existed
    # (#39928 moved the construction order).
    sources = []
    real_wrapper = Qwen4ExpPinnedHostEmbedding

    def tracking_wrapper(embedding):
        sources.append(embedding.weight)
        return real_wrapper(embedding)

    monkeypatch.setattr(
        qwen4_exp_module, "Qwen4ExpPinnedHostEmbedding", tracking_wrapper
    )
    layer = _build_layer(_config())

    assert sources and all(weight.is_meta for weight in sources)
    emb = layer.ple_embedding.ngram_embedding
    assert emb.weight.device.type == "cpu"
    assert not any(
        tensor.is_meta for tensor in (*layer.parameters(), *layer.buffers())
    ), "meta tensors remain after load"
    assert not emb.weight_scale.is_meta
    assert len(cudart.calls) == 1  # the table is allocated once, directly on the host


def test_wrapper_consumes_only_meta_template_metadata(cudart, single_rank):
    # The constructor-order fix hands the wrapper a meta table; only its
    # metadata may be read, and the table's values must start zero-filled
    # until the loader writes them (torch.empty on an anonymous mapping).
    with torch.device("meta"):
        template = VocabParallelEmbedding(
            128, 8, params_dtype=torch.float8_e4m3fn, output_dtype=torch.bfloat16
        )
    # The layer keeps the scale off the meta context, on the real device.
    template.register_buffer(
        "weight_scale", torch.ones(1, dtype=torch.bfloat16), persistent=True
    )
    assert template.weight.is_meta

    wrapper = Qwen4ExpPinnedHostEmbedding(template)
    assert wrapper.weight.shape == (128, 8)
    assert wrapper.weight.dtype == torch.float8_e4m3fn
    assert wrapper.weight.device.type == "cpu"
    assert not wrapper.weight.any()
    assert not wrapper.weight_scale.is_meta
    assert wrapper.quant_method is None  # loader must not stage the table to GPU
    # The meta source table is deleted after wrapping; nothing meta remains.
    assert all(not parameter.is_meta for parameter in template.parameters())
    assert cudart.calls == [
        (
            wrapper.weight.data_ptr(),
            128 * 8,
            _PORTABLE_MAPPED,
        )
    ]


def test_loaded_rows_keep_gather_pointer_semantics(cudart, single_rank):
    layer = _build_layer(_config())
    emb = layer.ple_embedding.ngram_embedding
    weight = emb.weight
    pointer = weight.data_ptr()
    rows = (
        torch.arange(weight.numel(), dtype=torch.float32)
        .reshape(weight.shape)
        .to(weight.dtype)
    )

    emb.weight_loader(weight, rows)

    assert weight.data_ptr() == pointer == cudart.calls[0][0]
    assert torch.equal(weight, rows)
    # Process-lifetime ownership: nothing may close the registered mapping out
    # from under the surviving weight tensor or Triton graph replays.
    del layer, emb, weight, rows
    gc.collect()
    assert not qwen4_exp_module._PINNED_TABLE_BUFFERS[0].closed


def test_non_offload_profile_keeps_device_template_untouched(
    cudart, tracking_mmap, single_rank
):
    # The NVMe PLE profile (plugin replaces the table with SSDStreamEmbedding
    # under ple_offload_embedding=False) and the 27B/DFlash2 recipe (no PLE
    # offload at all) must see the constructor behave exactly as before.
    layer = _build_layer(_config(ple_offload_embedding=False))
    emb = layer.ple_embedding.ngram_embedding

    assert type(emb) is VocabParallelEmbedding
    assert not emb.weight.is_meta
    assert emb.weight_scale.device.type == "cpu"
    assert cudart.calls == []
    assert tracking_mmap == []
    assert qwen4_exp_module._PINNED_TABLE_BUFFERS == []
    assert layer._prefetch_stream is None
