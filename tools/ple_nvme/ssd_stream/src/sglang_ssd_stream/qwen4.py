"""SSD Stream adapter for SGLang's Qwen4/Flash-Next implementation."""

from __future__ import annotations

import re
from collections.abc import Iterable

import sglang.srt.models.qwen4_exp as qwen4
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.utils import logger

from .backend import SSDStreamEmbedding
from .config import get_config

_PLE_SHARD = re.compile(r"\.ngram_embedding\.shard_(\d+)\.weight$")


class _GraphCaptureTicket:
    def __init__(self, ready_event: torch.cuda.Event) -> None:
        self.ready_event = ready_event

    def wait_for_launch(self) -> None:
        pass

    def enqueue_consumer_wait(self, consumer_stream: torch.cuda.Stream) -> None:
        consumer_stream.wait_event(self.ready_event)


class Qwen4PLELayer(qwen4.Qwen4ExpPLELayer):
    """Add SSD staging at SGLang's existing decoder/PLE prefetch boundary."""

    def __init__(self, *args, ple_layer_index: int = 0, **kwargs) -> None:
        config = args[0] if args else kwargs["config"]
        original_embedding = qwen4.VocabParallelEmbedding
        original_offload = config.ple_offload_embedding

        def build_meta_embedding(*embedding_args, **embedding_kwargs):
            with torch.device("meta"):
                return original_embedding(*embedding_args, **embedding_kwargs)

        qwen4.VocabParallelEmbedding = build_meta_embedding
        config.ple_offload_embedding = False
        try:
            super().__init__(*args, ple_layer_index=ple_layer_index, **kwargs)
        finally:
            qwen4.VocabParallelEmbedding = original_embedding
            config.ple_offload_embedding = original_offload

        self.ple_embedding.ngram_embedding = SSDStreamEmbedding(
            self.ple_embedding.ngram_embedding,
            config=get_config(),
            ple_layer_index=ple_layer_index,
        )
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream()

        if not isinstance(self.ple_embedding.ngram_embedding, SSDStreamEmbedding):
            raise TypeError(
                "the Qwen4 PLE layer did not create an SSD Stream embedding"
            )
        self._ssd_graph_ready_event = torch.cuda.Event(external=True)
        self._ssd_graph_ready_event.record(torch.cuda.current_stream())

    def _get_prefetch_buffer(
        self, lookup_tokens: int, lookup_ids: torch.Tensor
    ) -> torch.Tensor:
        if get_is_capture_mode():
            buffer = self._graph_prefetch_buffers.get(lookup_tokens)
            if buffer is None:
                buffer = self._allocate_prefetch_buffer(lookup_tokens, lookup_ids)
                buffer.zero_()
                self._graph_prefetch_buffers[lookup_tokens] = buffer
            return buffer
        return super()._get_prefetch_buffer(lookup_tokens, lookup_ids)

    def start_prefetch(
        self,
        batch: qwen4._PLEBatch | None,
        forward_batch: ForwardBatch,
    ) -> None:
        if self._prefetch_state is not None:
            raise RuntimeError("PLE prefetch state was not consumed before reuse")
        if batch is None:
            if not self.ple_embedding.gather_dp_tokens:
                return
            physical_tokens = forward_batch.input_ids.numel()
            ngram_ids = forward_batch.input_ids.new_zeros(
                (physical_tokens, self.ple_embedding.ngram_heads)
            )
        else:
            physical_tokens = batch.physical_tokens
            ngram_ids = self.ple_embedding.compute_ngram_ids(batch)

        lookup_ids, semantic_tokens = self.ple_embedding._prepare_embedding_lookup(
            ngram_ids, forward_batch, physical_tokens
        )
        lookup_tokens = lookup_ids.shape[0]
        if lookup_tokens == 0:
            return
        prefetched = self._get_prefetch_buffer(lookup_tokens, lookup_ids)
        output_view = prefetched.view(lookup_tokens, self.ple_embedding.ngram_heads, -1)
        embedding = self.ple_embedding.ngram_embedding
        stream = self._prefetch_stream
        current_stream = torch.cuda.current_stream()
        if get_is_capture_mode():
            ticket = _GraphCaptureTicket(self._ssd_graph_ready_event)
        else:
            ticket = embedding.begin_gather(
                lookup_ids,
                out=output_view,
                stream=stream,
                producer_stream=current_stream,
            )
        self._prefetch_state = (
            prefetched,
            semantic_tokens,
            physical_tokens,
            ticket,
        )

    def _consume_prefetched_embeddings(
        self, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if self._prefetch_state is None:
            raise RuntimeError("PLE prefetch state is missing")
        embeddings, semantic_tokens, physical_tokens, ticket = self._prefetch_state
        ticket.wait_for_launch()
        ticket.enqueue_consumer_wait(torch.cuda.current_stream())
        embedding = self.ple_embedding.ngram_embedding
        embeddings = embedding.reduce(embeddings) * embedding.weight_scale
        embeddings = self.ple_embedding._finish_embedding_lookup(
            embeddings,
            semantic_tokens,
            forward_batch,
            physical_tokens,
        )
        self._prefetch_state = None
        return embeddings

    def prepare_ssd_stream_graph_replay(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        graph_physical_tokens: int,
    ) -> None:
        embedding = self.ple_embedding.ngram_embedding
        if self._prefetch_state is not None:
            raise RuntimeError("eager PLE prefetch is active during graph replay")
        graph_buffer = self._graph_prefetch_buffers.get(graph_physical_tokens)
        if graph_buffer is None:
            raise RuntimeError(
                "missing captured SSD PLE buffer for "
                f"{graph_physical_tokens} physical tokens"
            )

        stream = self._prefetch_stream
        current_stream = torch.cuda.current_stream(input_ids.device)
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            batch = qwen4._prepare_ple_batch(
                input_ids,
                forward_batch,
                ngram_size=self.ple_embedding.ngram_size,
                ngram_eos_token_id=self.ple_embedding.eos_token_id,
            )
            if batch is None:
                self._ssd_graph_ready_event.record(stream)
                return
            ngram_ids = self.ple_embedding.compute_ngram_ids(batch)
            lookup_ids, _ = self.ple_embedding._prepare_embedding_lookup(
                ngram_ids, forward_batch, batch.physical_tokens
            )

        lookup_tokens = lookup_ids.shape[0]
        if lookup_tokens == 0:
            self._ssd_graph_ready_event.record(stream)
            return
        if lookup_tokens > graph_physical_tokens:
            raise RuntimeError(
                f"SSD PLE replay has {lookup_tokens} real tokens but graph "
                f"buffer holds {graph_physical_tokens}"
            )
        output_view = graph_buffer[:lookup_tokens].view(
            lookup_tokens, self.ple_embedding.ngram_heads, -1
        )
        ticket = embedding.begin_gather(
            lookup_ids,
            out=output_view,
            stream=stream,
            producer_stream=stream,
            completion_event=self._ssd_graph_ready_event,
        )
        ticket.wait_for_launch()


def around_load_weights(
    original_fn, model, weights: Iterable[tuple[str, torch.Tensor]]
):
    """Reject embedded PLE shards and leave ordinary model loading upstream."""
    def filtered_weights():
        for original_name, loaded_weight in weights:
            name = original_name.replace("model.language_model.", "model.")
            if _PLE_SHARD.search(name) is not None:
                raise RuntimeError(
                    "prepared SSD Stream artifacts must not contain PLE shard tensors"
                )
            yield original_name, loaded_weight

    result = set(original_fn(model, filtered_weights()))
    expected_embeddings = {
        module for module in model.modules() if isinstance(module, SSDStreamEmbedding)
    }
    if not expected_embeddings:
        raise RuntimeError(
            "SSD Stream is enabled but the model has no streamed PLE embedding"
        )
    logger.info(
        "Loaded %d immutable SSD Stream table(s)",
        len(expected_embeddings),
    )
    return result
