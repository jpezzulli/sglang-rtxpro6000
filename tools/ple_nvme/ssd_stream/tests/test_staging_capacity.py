"""CPU-only coverage for the NVMe staging capacity of a PLE lookup.

The capacity comes from ``_STAGING_MIB`` scaled by the storage element size, so
a BF16 table must keep the row budget that the FP8 budget already provides.
This check imports the runtime backend with the uninstalled GPU seams stubbed
(Triton, the native reader, CUDA events/streams/pinned memory); the staging
arithmetic, capacity check and gather call shapes are the real ones. The stubbed
module is kept private to this file: the module key and the parent package
attribute are saved and restored, so neither an earlier real import of
``sglang_ssd_stream.backend`` nor the stubs leak between test modules.
"""

import contextlib
import hashlib
import importlib
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

EMBEDDING_DIM = 160
NGRAM_HEADS = 16
LOOKUP_TOKENS = 4096
LOOKUP_ROWS = LOOKUP_TOKENS * NGRAM_HEADS
TABLE_ROWS = 4096
ROW_BUDGET_ROWS = 16 * 1024**2 // EMBEDDING_DIM  # _STAGING_MIB bytes at 1 B/row
FP8_BUDGET_NBYTES = 16 * 1024**2
BF16_BUDGET_NBYTES = 2 * FP8_BUDGET_NBYTES
_STUBBED = object()
# Saved and restored around the stubbed import: the rest of the suite (and the
# GPU stack) imports the same file for real, so the stub must never be visible
# to them and a real import must never be reused as a stub.
_PACKAGE_MODULES = (
    "sglang_ssd_stream",
    "sglang_ssd_stream.config",
    "sglang_ssd_stream._io",
    "sglang_ssd_stream.backend",
)


def _stub_sglang():
    return {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.distributed": SimpleNamespace(
            get_tp_group=lambda: None,
            tensor_model_parallel_all_reduce=lambda value: value,
        ),
        "sglang.srt.distributed.device_communicators": SimpleNamespace(__path__=[]),
        "sglang.srt.distributed.device_communicators.pynccl_allocator": (
            SimpleNamespace(use_symmetric_memory=lambda *args, **kwargs: None)
        ),
        "sglang.srt.layers": SimpleNamespace(__path__=[]),
        "sglang.srt.layers.communicator": SimpleNamespace(
            get_attn_tp_context=lambda: None
        ),
        "sglang.srt.layers.dp_attention": SimpleNamespace(
            attn_tp_all_reduce=lambda value: value,
            is_allocation_symmetric=lambda: False,
        ),
        "sglang.srt.layers.quantization": SimpleNamespace(__path__=[]),
        "sglang.srt.layers.quantization.unquant": SimpleNamespace(
            UnquantizedEmbeddingMethod=type(
                "UnquantizedEmbeddingMethod", (), {"__module__": "unquant_stub"}
            )
        ),
        "sglang.srt.layers.vocab_parallel_embedding": SimpleNamespace(
            VocabParallelEmbedding=torch.nn.Module
        ),
        "sglang.srt.model_executor": SimpleNamespace(__path__=[]),
        "sglang.srt.model_executor.runner": SimpleNamespace(
            get_is_capture_mode=lambda: False
        ),
        "sglang.srt.utils": SimpleNamespace(
            logger=SimpleNamespace(info=lambda *args, **kwargs: None)
        ),
    }


def _recording_reader(recorder):
    class RecordingReader:
        registered_buffers = True

        def __init__(self, *args, **kwargs):
            self.args = args

        def gather(self, ids, rows):
            recorder.reads.append((ids.shape, rows.shape, rows.dtype))

    return RecordingReader


def _dependency_stubs(reader):
    triton_stub = SimpleNamespace(
        jit=lambda fn=None, **kwargs: fn if fn is not None else (lambda f: f),
        next_power_of_2=lambda value: int(value),
        language=SimpleNamespace(float8e4nv="fp8", bfloat16="bf16"),
    )
    stubs = _stub_sglang()
    stubs.update(
        {
            "triton": triton_stub,
            "triton.language": triton_stub.language,
            "sglang_ssd_stream._io": SimpleNamespace(PageReader=reader),
        }
    )
    return stubs


@contextlib.contextmanager
def _preimported_backend(module):
    """Make ``module`` the already-imported backend, then put things back."""
    package = importlib.import_module("sglang_ssd_stream")
    saved_module = sys.modules.get("sglang_ssd_stream.backend", _STUBBED)
    saved_attribute = getattr(package, "backend", _STUBBED)
    sys.modules["sglang_ssd_stream.backend"] = module
    package.backend = module
    try:
        yield
    finally:
        if saved_module is _STUBBED:
            sys.modules.pop("sglang_ssd_stream.backend", None)
        else:
            sys.modules["sglang_ssd_stream.backend"] = saved_module
        if saved_attribute is _STUBBED:
            with contextlib.suppress(AttributeError):
                del package.backend
        else:
            package.backend = saved_attribute


@contextlib.contextmanager
def _isolated_backend(stubs):
    """Yield a freshly imported backend bound only to ``stubs``.

    The ``sys.modules`` keys and the parent package's ``backend`` attribute are
    saved and restored, so a backend imported before this call comes back
    unchanged and the stubbed one leaves no trace for the tests that follow.
    """
    package = importlib.import_module("sglang_ssd_stream")
    saved = {
        name: sys.modules.get(name, _STUBBED) for name in (*stubs, *_PACKAGE_MODULES)
    }
    saved_attribute = getattr(package, "backend", _STUBBED)
    sys.modules.pop("sglang_ssd_stream.backend", None)
    sys.modules.update(stubs)
    try:
        yield importlib.import_module("sglang_ssd_stream.backend")
    finally:
        for name, value in saved.items():
            if value is _STUBBED:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        if saved_attribute is _STUBBED:
            with contextlib.suppress(AttributeError):
                del package.backend
        else:
            package.backend = saved_attribute


@pytest.fixture(scope="module")
def backend():
    recorder = SimpleNamespace(reads=[])
    reader = _recording_reader(recorder)
    stubs = _dependency_stubs(reader)
    with pytest.MonkeyPatch.context() as monkeypatch:
        # No GPU here: keep the pinned-memory, CUDA-event/stream and kernel
        # launches inert so begin_gather reaches the real capacity check.
        real_empty = torch.empty

        def empty(*args, **kwargs):
            if not torch.cuda.is_available():
                kwargs.pop("pin_memory", None)
            return real_empty(*args, **kwargs)

        monkeypatch.setattr(torch, "empty", empty)
        monkeypatch.setattr(
            torch.Tensor, "record_stream", lambda self, stream: None, raising=False
        )
        monkeypatch.setattr(
            torch.cuda,
            "Event",
            lambda: SimpleNamespace(record=lambda *a: None, synchronize=lambda: None),
            raising=False,
        )
        monkeypatch.setattr(
            torch.cuda,
            "Stream",
            lambda *a, **k: SimpleNamespace(wait_stream=lambda *a2: None),
            raising=False,
        )
        monkeypatch.setattr(
            torch.cuda,
            "stream",
            lambda *a, **k: contextlib.nullcontext(),
            raising=False,
        )
        monkeypatch.setattr(
            torch.cuda, "set_device", lambda device: None, raising=False
        )
        with _isolated_backend(stubs) as backend_module:

            class Kernel:
                def __getitem__(self, grid):
                    def launch(*args, **kwargs):
                        return None

                    return launch

            monkeypatch.setattr(
                backend_module, "_copy_ple_staged_rows_kernel", Kernel()
            )
            # The fresh import proves the stubs, not an earlier real import,
            # are what this module is bound to.
            assert (
                backend_module.UnquantizedEmbeddingMethod.__module__ == "unquant_stub"
            )
            assert backend_module.PageReader is reader
            backend_module.recorder = recorder
            yield backend_module


def _element_size(dtype):
    return torch.empty(0, dtype=dtype).element_size()


def _make_staged(backend, root, dtype):
    element_size = _element_size(dtype)
    raw = (
        (np.arange(TABLE_ROWS * EMBEDDING_DIM * element_size, dtype=np.uint64) * 31 + 7)
        % 251
    ).astype(np.uint8)
    raw = raw.tobytes()
    table = root / "ple" / "table.bin"
    table.parent.mkdir(parents=True)
    table.write_bytes(raw)
    (root / "ssd-stream.json").write_text(
        json.dumps(
            {
                "format": "sglang-ssd-stream",
                "version": 1,
                "tables": [
                    {
                        "layer": 0,
                        "path": "ple/table.bin",
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "dtype": str(dtype).removeprefix("torch."),
                        "rows": TABLE_ROWS,
                        "columns": EMBEDDING_DIM,
                        "row_start": 0,
                        "bytes": len(raw),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    from sglang_ssd_stream.config import load_manifest

    embedding = torch.nn.Module()
    embedding.quant_method = backend.UnquantizedEmbeddingMethod()
    embedding.weight = torch.nn.Parameter(
        torch.zeros(TABLE_ROWS, EMBEDDING_DIM * element_size, dtype=torch.uint8).view(
            dtype
        ),
        requires_grad=False,
    )
    embedding.weight_scale = torch.ones((), dtype=torch.float32)
    embedding.quant_config = None
    embedding.enable_tp = False
    embedding.use_attn_tp_group = False
    embedding.tp_size = 1
    embedding.num_embeddings = TABLE_ROWS
    embedding.org_vocab_size = TABLE_ROWS
    embedding.padding_size = 0
    embedding.num_added_embeddings = 0
    embedding.use_presharded_weights = False
    embedding.org_vocab_size_padded = TABLE_ROWS
    embedding.num_embeddings_padded = TABLE_ROWS
    embedding.num_embeddings_per_partition = TABLE_ROWS
    embedding.num_org_embeddings_per_partition = TABLE_ROWS
    embedding.num_added_embeddings_per_partition = 0
    embedding.embedding_dim = EMBEDDING_DIM
    embedding.shard_indices = SimpleNamespace(
        org_vocab_start_index=0, org_vocab_end_index=TABLE_ROWS
    )
    staged = backend.SSDStreamEmbedding(
        embedding, config=load_manifest(root / "ssd-stream.json"), ple_layer_index=0
    )
    assert staged._element_size == element_size
    return staged


@pytest.fixture(scope="module")
def fp8_staged(backend, tmp_path_factory):
    return _make_staged(backend, tmp_path_factory.mktemp("fp8"), torch.float8_e4m3fn)


@pytest.fixture(scope="module")
def bf16_staged(backend, tmp_path_factory):
    return _make_staged(backend, tmp_path_factory.mktemp("bf16"), torch.bfloat16)


def _lookup(backend, staged, row_count, out=None):
    recorder = backend.recorder
    recorder.reads = []
    ids = torch.arange(row_count, dtype=torch.long)
    if out is None:
        out = torch.zeros((row_count, EMBEDDING_DIM), dtype=torch.bfloat16)
    stream = backend.torch.cuda.Stream()
    ticket = staged.begin_gather(ids, out=out, stream=stream, producer_stream=stream)
    ticket.wait_for_launch()
    return recorder.reads


def test_stubbed_import_is_fresh_and_restored():
    # Order-independent isolation check: a backend that was imported earlier for
    # other dependencies is neither reused by the fixture nor left behind by it,
    # in sys.modules or on the parent package.
    class OtherReader:
        registered_buffers = True

    preimported = types.ModuleType("sglang_ssd_stream.backend")
    preimported.PageReader = OtherReader
    with _preimported_backend(preimported):
        stubs = _dependency_stubs(_recording_reader(SimpleNamespace(reads=[])))
        with _isolated_backend(stubs) as module:
            assert module is not preimported
            assert module.UnquantizedEmbeddingMethod.__module__ == "unquant_stub"
            assert module.PageReader is not OtherReader
        assert sys.modules["sglang_ssd_stream.backend"] is preimported
        assert importlib.import_module("sglang_ssd_stream").backend is preimported


def test_slots_stay_bounded_and_match_pinned_buffers(backend, fp8_staged, bf16_staged):
    for staged in (fp8_staged, bf16_staged):
        assert len(staged._slots) == backend._STAGING_SLOTS == 2
        assert all(
            slot.rows_raw.shape[0] == staged._staging_capacity_rows
            for slot in staged._slots
        )


def test_fp8_staging_budget_is_unchanged(backend, fp8_staged):
    assert fp8_staged._row_nbytes == EMBEDDING_DIM * 1
    assert fp8_staged._staging_capacity_rows == ROW_BUDGET_ROWS
    assert (
        fp8_staged._staging_capacity_rows * fp8_staged._row_nbytes <= FP8_BUDGET_NBYTES
    )


def test_bf16_keeps_the_fp8_row_capacity(backend, bf16_staged, fp8_staged):
    assert bf16_staged._row_nbytes == EMBEDDING_DIM * 2
    assert bf16_staged._staging_capacity_rows == ROW_BUDGET_ROWS
    assert bf16_staged._staging_capacity_rows == fp8_staged._staging_capacity_rows, (
        "BF16 must not halve the FP8 row capacity"
    )
    assert (
        bf16_staged._staging_capacity_rows * bf16_staged._row_nbytes
        <= BF16_BUDGET_NBYTES
    ), "BF16 doubles the byte budget, never the row budget"


def test_default_4096_token_prefill_fits_both_dtypes(backend, fp8_staged, bf16_staged):
    # 4096 prefill tokens x 16 ngram heads, 160 columns per row.
    assert LOOKUP_ROWS <= fp8_staged._staging_capacity_rows
    assert LOOKUP_ROWS <= bf16_staged._staging_capacity_rows

    # The native reader is fed the pinned buffers' numpy views.
    reads = _lookup(backend, fp8_staged, LOOKUP_ROWS)
    assert reads == [((LOOKUP_ROWS,), (LOOKUP_ROWS, fp8_staged._row_nbytes), np.uint8)]
    reads = _lookup(backend, bf16_staged, LOOKUP_ROWS)
    assert reads == [((LOOKUP_ROWS,), (LOOKUP_ROWS, bf16_staged._row_nbytes), np.uint8)]


def test_oversized_lookup_fails_loudly_before_io(backend, fp8_staged, bf16_staged):
    for staged in (fp8_staged, bf16_staged):
        too_many = staged._staging_capacity_rows + 1
        with pytest.raises(
            RuntimeError,
            match=(
                f"needs {too_many} staging rows but the configured "
                f"capacity is {staged._staging_capacity_rows}"
            ),
        ):
            # A cheap placeholder output: the capacity check must fire first.
            _lookup(
                backend,
                staged,
                too_many,
                out=torch.zeros((1, 1), dtype=torch.bfloat16),
            )
        assert backend.recorder.reads == [], "capacity check must precede the gather"
