"""Model-neutral asynchronous streaming for deterministic lookup tables."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext

import torch
import triton
import triton.language as tl
from sglang.srt.distributed import get_tp_group, tensor_model_parallel_all_reduce
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import attn_tp_all_reduce, is_allocation_symmetric
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.utils import logger
from torch import nn

from ._io import PageReader
from .config import SSDStreamConfig

_STAGING_MIB = 16
_STAGING_SLOTS = 2


@triton.jit
def _copy_ple_staged_rows_kernel(
    staged_ptr,
    output_ptr,
    embedding_dim,
    is_fp8: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    if is_fp8:
        staged_ptr = staged_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    else:
        staged_ptr = staged_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
    values = tl.load(
        staged_ptr + row_id * embedding_dim + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        values,
        mask=mask,
    )


class _GatherTicket:
    def __init__(
        self,
        future: Future,
        stream: torch.cuda.Stream,
    ) -> None:
        self.future = future
        self.stream = stream

    def wait_for_launch(self) -> None:
        self.future.result()

    def enqueue_consumer_wait(self, consumer_stream: torch.cuda.Stream) -> None:
        consumer_stream.wait_stream(self.stream)


class _StagingSlot:
    def __init__(
        self,
        capacity_rows: int,
        row_nbytes: int,
        storage_dtype: torch.dtype,
        embedding_dim: int,
    ) -> None:
        self.ids_cpu = torch.empty(
            capacity_rows, dtype=torch.long, device="cpu", pin_memory=True
        )
        self.ids_numpy = self.ids_cpu.numpy()
        self.rows_raw = torch.empty(
            (capacity_rows, row_nbytes),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        self.rows_numpy = self.rows_raw.numpy()
        self.rows_weight = self.rows_raw.view(storage_dtype).reshape(
            capacity_rows, embedding_dim
        )
        self.ids_ready = torch.cuda.Event()
        self.gpu_done = torch.cuda.Event()
        self.future: Future | None = None


class SSDStreamEmbedding(VocabParallelEmbedding):
    """Stream exact PLE rows from SSD into bounded pinned staging buffers."""

    _COPIED_ATTRIBUTES = (
        "quant_config",
        "enable_tp",
        "use_attn_tp_group",
        "tp_size",
        "num_embeddings",
        "org_vocab_size",
        "padding_size",
        "num_added_embeddings",
        "use_presharded_weights",
        "org_vocab_size_padded",
        "num_embeddings_padded",
        "shard_indices",
        "embedding_dim",
        "num_embeddings_per_partition",
        "num_org_embeddings_per_partition",
        "num_added_embeddings_per_partition",
    )

    def __init__(
        self,
        embedding: VocabParallelEmbedding,
        config: SSDStreamConfig,
        ple_layer_index: int,
    ) -> None:
        nn.Module.__init__(self)
        if not isinstance(embedding.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                "SSD Stream requires an unquantized PLE embedding table"
            )
        if embedding.weight.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise TypeError(
                "SSD Stream requires bfloat16 or fp8 PLE weights, got "
                f"{embedding.weight.dtype}"
            )
        if embedding.num_added_embeddings:
            raise NotImplementedError("SSD Stream does not support added PLE rows")
        for name in self._COPIED_ATTRIBUTES:
            setattr(self, name, getattr(embedding, name))
        self.quant_method = None

        source_weight = embedding.weight
        shape = tuple(source_weight.shape)
        self._storage_dtype = source_weight.dtype
        self._element_size = source_weight.element_size()
        self._row_nbytes = self.embedding_dim * self._element_size
        dtype_name = str(self._storage_dtype).removeprefix("torch.")
        tp_start = self.shard_indices.org_vocab_start_index
        tp_end = self.shard_indices.org_vocab_end_index
        table = config.table_for_layer(ple_layer_index)
        if table.dtype != dtype_name:
            raise TypeError(
                f"SSD Stream layer {ple_layer_index} is {table.dtype}, but SGLang "
                f"constructed a {dtype_name} embedding"
            )
        if table.columns != self.embedding_dim:
            raise ValueError(
                f"SSD Stream layer {ple_layer_index} has width {table.columns}, "
                f"expected {self.embedding_dim}"
            )
        table_end = table.row_start + table.rows
        if tp_start < table.row_start or tp_end > table_end:
            raise ValueError(
                f"SSD Stream layer {ple_layer_index} covers rows "
                f"[{table.row_start}, {table_end}), not TP rows [{tp_start}, {tp_end})"
            )
        if shape != (tp_end - tp_start, self.embedding_dim):
            raise ValueError(
                f"SSD Stream local embedding shape {shape} does not match TP range "
                f"[{tp_start}, {tp_end})"
            )

        self.backing_path = str(table.path)
        self._table_sha256 = table.sha256
        self._table_nbytes = table.nbytes
        self.register_buffer("weight_scale", embedding.weight_scale, persistent=True)
        del embedding.weight

        staging_nbytes = max(self._row_nbytes, _STAGING_MIB * 1024**2)
        self._staging_capacity_rows = staging_nbytes // self._row_nbytes
        self._slots = [
            _StagingSlot(
                self._staging_capacity_rows,
                self._row_nbytes,
                self._storage_dtype,
                self.embedding_dim,
            )
            for _ in range(_STAGING_SLOTS)
        ]
        self._next_slot = 0
        self._slot_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"ssd-stream-{ple_layer_index}",
        )
        self._reader = PageReader(
            self.backing_path,
            self._row_nbytes,
            table.row_start,
            tp_start,
            tp_end,
        )
        self._block_d = triton.next_power_of_2(self.embedding_dim)
        logger.info(
            "SSD Stream enabled: backing=%s sha256=%s size=%.2f GiB "
            "staging_slots=%d staging_per_slot=%.2f MiB capacity=%d rows "
            "page_pool=32.00 MiB registered_buffers=%s",
            self.backing_path,
            self._table_sha256,
            self._table_nbytes / 1024**3,
            _STAGING_SLOTS,
            self._staging_capacity_rows * self._row_nbytes / 1024**2,
            self._staging_capacity_rows,
            self._reader.registered_buffers,
        )

    def allocate_output(
        self, shape: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        allocation_context = nullcontext()
        if self.tp_size > 1:
            allocation_context = use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            )
        with allocation_context, torch.inference_mode(False):
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def _acquire_slot(self) -> _StagingSlot:
        with self._slot_lock:
            slot = self._slots[self._next_slot]
            self._next_slot = (self._next_slot + 1) % len(self._slots)
            if slot.future is not None:
                slot.future.result()
                slot.gpu_done.synchronize()
                slot.future = None
            return slot

    def _stage_rows_and_launch(
        self,
        slot: _StagingSlot,
        row_count: int,
        output: torch.Tensor,
        stream: torch.cuda.Stream,
        device: torch.device,
        completion_event: torch.cuda.Event | None,
    ) -> torch.cuda.Event:
        torch.cuda.set_device(device)
        slot.ids_ready.synchronize()

        if self._reader is None:
            raise RuntimeError("SSD Stream page reader is unavailable")
        self._reader.gather(
            slot.ids_numpy[:row_count],
            slot.rows_numpy[:row_count],
        )

        with torch.cuda.stream(stream):
            _copy_ple_staged_rows_kernel[(row_count,)](
                slot.rows_weight.data_ptr(),
                output,
                embedding_dim=self.embedding_dim,
                is_fp8=self._storage_dtype == torch.float8_e4m3fn,
                BLOCK_D=self._block_d,
            )
            slot.gpu_done.record(stream)
            if completion_event is not None:
                completion_event.record(stream)
        return slot.gpu_done

    def begin_gather(
        self,
        input_ids: torch.Tensor,
        *,
        out: torch.Tensor,
        stream: torch.cuda.Stream,
        producer_stream: torch.cuda.Stream,
        completion_event: torch.cuda.Event | None = None,
    ) -> _GatherTicket:
        if get_is_capture_mode():
            raise RuntimeError(
                "SSD Stream I/O must be prepared outside CUDA graph capture"
            )
        row_count = input_ids.numel()
        if row_count > self._staging_capacity_rows:
            raise RuntimeError(
                f"PLE lookup needs {row_count} staging rows but the configured "
                f"capacity is {self._staging_capacity_rows}"
            )
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"invalid PLE prefetch output shape: {tuple(out.shape)} != "
                f"{expected_shape}"
            )
        if out.dtype != torch.bfloat16 or out.device != input_ids.device:
            raise ValueError("PLE prefetch output must be bfloat16 on the id device")
        if not out.is_contiguous():
            raise ValueError("PLE prefetch output must be contiguous")

        slot = self._acquire_slot()
        stream.wait_stream(producer_stream)
        with torch.cuda.stream(stream):
            flat_ids = input_ids.reshape(-1)
            if flat_ids.dtype != torch.long:
                flat_ids = flat_ids.long()
            if not flat_ids.is_contiguous():
                flat_ids = flat_ids.contiguous()
            slot.ids_cpu[:row_count].copy_(flat_ids, non_blocking=True)
            slot.ids_ready.record(stream)
        flat_ids.record_stream(stream)
        future = self._executor.submit(
            self._stage_rows_and_launch,
            slot,
            row_count,
            out.reshape(row_count, self.embedding_dim),
            stream,
            input_ids.device,
            completion_event,
        )
        slot.future = future
        return _GatherTicket(future, stream)

    def gather(
        self, input_ids: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        expected_shape = (*input_ids.shape, self.embedding_dim)
        output = (
            self.allocate_output(expected_shape, input_ids.device)
            if out is None
            else out
        )
        current_stream = torch.cuda.current_stream(input_ids.device)
        ticket = self.begin_gather(
            input_ids,
            out=output,
            stream=current_stream,
            producer_stream=current_stream,
        )
        ticket.wait_for_launch()
        return output

    def reduce(self, output: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1 and not get_attn_tp_context().input_scattered:
            if self.use_attn_tp_group:
                return attn_tp_all_reduce(output)
            return tensor_model_parallel_all_reduce(output)
        return output

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.reduce(self.gather(input_ids))
