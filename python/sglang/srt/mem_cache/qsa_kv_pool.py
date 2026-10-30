"""KV pools carrying the QSA sparse-attention indexer caches.

``QSATokenToKVPool`` (compressed, Qwen4-Exp) adds the per-request pending
index-key/RoPE ring and the paged compressed-K cache on top of the hybrid
full/linear KV pool. ``QwenDSATokenToKVPool`` (tokenwise,
Qwen3Next-DSA) adds only the flat per-token index-K cache.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import GB, HybridLinearKVPool, MambaPool

logger = logging.getLogger(__name__)

# State layer IDs are serialized as uint32 by the disaggregation protocols.
# Reserve the value below PLE's request-wide sentinel for QSA's request-wide
# RoPE ring, which is shared by all full-attention layers.
QSA_ROPE_STATE_LAYER_ID = (1 << 32) - 2


def _index_k_bytes(*, kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
    return kv_heads * head_dim * dtype.itemsize


# ``--qsa-indexer-dtype`` choices. The pending key ring stays bf16 regardless.
QSA_INDEXER_DTYPE_CHOICES = ("auto", "bfloat16", "fp8_e4m3")


def resolve_qsa_indexer_dtype(name: str) -> torch.dtype:
    """Storage dtype of the compressed QSA indexer cache for a CLI value."""
    if name in ("auto", "bfloat16"):
        return torch.bfloat16
    if name == "fp8_e4m3":
        return torch.float8_e4m3fn
    raise ValueError(
        f"Unsupported --qsa-indexer-dtype {name!r}; expected one of "
        f"{QSA_INDEXER_DTYPE_CHOICES}"
    )


class QSATokenToKVPool(HybridLinearKVPool):
    """Hybrid KV pool with the minimal BF16 state required by simple QSA."""

    # Full-KV pages are a multiple of the compress ratio, so no group straddles pages;
    # ``compressed_slot = full_slot // ratio`` needs no ownership bookkeeping;
    # lifecycle rides the full-KV allocator and radix tree.
    # Full slot 0 is the reserved padding slot; compressed slot 0 is the inert dump.
    # Pending-ring dtype: the raw keys are averaged from here, so it stays bf16.
    index_state_dtype = torch.bfloat16

    @classmethod
    def qsa_bytes_per_token(
        cls,
        *,
        kv_heads: int,
        head_dim: int,
        compress_ratio: int,
        num_layers: int,
        compressed_dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        """Per-token QSA index-cache cost: compressed keys only;
        the per-request pending ring is budgeted with the other per-request buffers.

        The ring holds ``compress_ratio * qsa_num_groups`` slots per request
        (the groups a verify window can still touch), so its total stays
        bounded by the request-slot count and outside this per-token budget.
        """
        index_k_bytes = _index_k_bytes(
            kv_heads=kv_heads, head_dim=head_dim, dtype=compressed_dtype
        )
        return index_k_bytes // compress_ratio * num_layers

    @classmethod
    def pending_ring_num_groups(
        cls, *, max_num_draft_tokens, compress_ratio: int
    ) -> int:
        """Groups the pending ring must hold so the widest window it is sized
        for fits without a within-forward collision.

        Delegates to ``qsa.metadata.pending_ring_groups_required`` (see that
        docstring for the window-plus-retained-prefix span and for why only
        the one-row window keeps the shipped single-group layout). The count
        is derived from the LAUNCH-MAXIMUM draft-token window, never the
        active one: a narrower active width (an adaptive-MTP step-down) must
        reuse this allocation, because CUDA graphs bind the ring buffers for
        the graph's lifetime and geometry cannot be resized mid-serve.
        """
        from sglang.srt.layers.attention.qsa.metadata import (
            pending_ring_groups_required,
        )

        return pending_ring_groups_required(
            draft_tokens=max_num_draft_tokens, compress_ratio=compress_ratio
        )

    def __init__(
        self,
        *,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        device: str,
        mamba_pool: MambaPool,
        qsa_index_kv_heads: int,
        qsa_index_head_dim: int,
        qsa_compress_ratio: int,
        qsa_token_topk: int,
        num_request_slots: int,
        qsa_num_groups: int = 1,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        start_layer: Optional[int] = None,
        full_kv_pool_class: Optional[type] = None,
        quant_method=None,
        post_capture_active: bool = False,
        qsa_indexer_dtype: torch.dtype = torch.bfloat16,
    ):
        if page_size <= 1 or page_size % qsa_compress_ratio != 0:
            raise ValueError(
                "compressed QSA requires a paged full-KV cache with the page "
                "a multiple of the compress ratio (compressed slots are "
                f"full_slot // ratio): page_size={page_size}, "
                f"ratio={qsa_compress_ratio}. This needs the mamba "
                "extra-buffer strategy or "
                "--disable-radix-cache (see the Qwen4-Exp arg overrides)."
            )
        # super().__init__ computes mem_usage via the overridden get_kv_size_bytes,
        # so the QSA buffers get placeholders first; mem_usage is recomputed last.
        self.qsa_key_state_buffer_pool = []
        self.qsa_compressed_k_buffer_pool = []
        self.qsa_rope_position_buffer = torch.empty(0)
        super().__init__(
            size=size,
            dtype=dtype,
            page_size=page_size,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            mamba_pool=mamba_pool,
            enable_memory_saver=enable_memory_saver,
            enable_kv_cache_copy=enable_kv_cache_copy,
            use_mla=False,
            start_layer=start_layer,
            full_kv_pool_class=full_kv_pool_class,
            quant_method=quant_method,
            post_capture_active=post_capture_active,
        )
        if (
            min(
                qsa_index_kv_heads,
                qsa_index_head_dim,
                qsa_compress_ratio,
                qsa_token_topk,
            )
            <= 0
        ):
            raise ValueError("QSA cache configuration values must be positive")
        if qsa_token_topk % qsa_compress_ratio != 0:
            raise ValueError("qsa_token_topk must be divisible by qsa_compress_ratio")
        self.qsa_compress_ratio = int(qsa_compress_ratio)
        self.qsa_index_head_dim = int(qsa_index_head_dim)
        self.qsa_index_kv_heads = int(qsa_index_kv_heads)
        self.qsa_token_topk = int(qsa_token_topk)
        self.qsa_block_topk = self.qsa_token_topk // self.qsa_compress_ratio
        if qsa_indexer_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError(
                "QSA compressed indexer cache dtype must be bfloat16 or "
                f"float8_e4m3fn, got {qsa_indexer_dtype}"
            )
        # Storage dtype of the compressed keys and the index Q (the GEMM operands).
        self.qsa_compressed_dtype = qsa_indexer_dtype
        logger.info(
            "QSA compressed indexer cache dtype: %s (pending ring %s)",
            self.qsa_compressed_dtype,
            self.index_state_dtype,
        )
        state_size = size + page_size
        # Compressed slots mirror the full-KV slot space 1:ratio; the "page"
        # seen by the scoring kernels is one full-KV page's worth of groups.
        self.qsa_compressed_page_size = page_size // self.qsa_compress_ratio
        self.qsa_compressed_capacity = -(state_size // -self.qsa_compress_ratio)
        # Pre-compression index-K state is a per-request ring, not a per-token
        # cache: the groups a verify window can still touch must survive a
        # forward, addressed as
        # ``req_pool_idx * ratio * qsa_num_groups
        #   + (position // ratio) % qsa_num_groups * ratio + position % ratio``
        # (see qsa.metadata.pending_ring_slot; every producer derives slots
        # from the same count). A verify forward publishes its whole window
        # before compressing the groups it completes, and the first completed
        # group may still read retained prefix members, so one group per
        # request only fits a one-row window; ``qsa_num_groups`` sizes the
        # ring for the launch-maximum window (pending_ring_num_groups).
        # Request slot 0 is never allocated, so its whole span is inert:
        # ring rows [0, ratio) double as the dump target for tokens whose
        # group already compressed in the same forward, and that region stays
        # disjoint from every allocated request at any group count.
        if num_request_slots <= 0:
            raise ValueError(
                f"QSA pending ring needs request slots, got {num_request_slots}"
            )
        if qsa_num_groups < 1:
            raise ValueError(f"QSA pending ring needs groups, got {qsa_num_groups}")
        self.qsa_num_request_slots = int(num_request_slots)
        self.qsa_num_groups = int(qsa_num_groups)
        self.qsa_ring_span = self.qsa_compress_ratio * self.qsa_num_groups
        ring_slots = self.qsa_num_request_slots * self.qsa_ring_span
        # These buffers participate in Mooncake PD transfer just like the base
        # KV and Mamba pools.  Keep their allocation in the same memory-saver
        # and Mooncake custom-pool regions; otherwise MNNVL cannot resolve the
        # ordinary CUDA allocation when the first QSA state page is sent.
        allocation_pool = self.full_kv_pool
        with (
            allocation_pool.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                torch.cuda.use_mem_pool(allocation_pool.custom_mem_pool)
                if allocation_pool.enable_custom_mem_pool
                else nullcontext()
            ),
        ):
            self.qsa_key_state_buffer_pool = [
                torch.zeros(
                    (
                        ring_slots,
                        self.qsa_index_kv_heads,
                        self.qsa_index_head_dim,
                    ),
                    dtype=self.index_state_dtype,
                    device=device,
                )
                for _ in full_attention_layer_ids
            ]
            # RoPE coordinates are layer-independent. Keep the exact Qwen4-Exp
            # MRoPE position of every incomplete key so compression can rotate
            # the pooled key with the group's real starting coordinate.
            self.qsa_rope_position_buffer = torch.zeros(
                (ring_slots, 3), dtype=torch.int64, device=device
            )
            # One contiguous allocation behind per-layer views: every layer's
            # compressed pages are addressable from a single base pointer.
            self.qsa_compressed_flat = torch.zeros(
                (
                    len(full_attention_layer_ids),
                    self.qsa_compressed_capacity
                    * self.qsa_index_kv_heads
                    * self.qsa_index_head_dim,
                ),
                dtype=self.qsa_compressed_dtype,
                device=device,
            )
        self.qsa_compressed_k_buffer_pool = [
            self.qsa_compressed_flat[layer_offset].view(
                self.qsa_compressed_capacity,
                self.qsa_index_kv_heads,
                self.qsa_index_head_dim,
            )
            for layer_offset in range(len(full_attention_layer_ids))
        ]
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def get_qsa_key_state_buffer(self, layer_id: int) -> torch.Tensor:
        return self.qsa_key_state_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def set_qsa_key_state_buffer(
        self, layer_id: int, loc: torch.Tensor, token_k: torch.Tensor
    ) -> None:
        buffer = self.get_qsa_key_state_buffer(layer_id)
        buffer[loc.long()] = token_k.to(buffer.dtype)

    def set_qsa_rope_position_buffer(
        self, loc: torch.Tensor, positions: torch.Tensor
    ) -> None:
        positions = positions.long()
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError(
                f"QSA RoPE positions must be [tokens] or [3, tokens], got {positions.shape}"
            )
        self.qsa_rope_position_buffer[loc.long()] = positions.transpose(0, 1)

    def get_qsa_rope_position_buffer(self, loc: torch.Tensor) -> torch.Tensor:
        return self.qsa_rope_position_buffer[loc.long()]

    def get_qsa_compressed_k_buffer(self, layer_id: int) -> torch.Tensor:
        # The indexer reads compressed keys before attention reads the full KV.
        self._wait_for_layer(layer_id)
        return self.qsa_compressed_k_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def set_qsa_compressed_k_buffer(
        self, layer_id: int, loc: torch.Tensor, compressed_k: torch.Tensor
    ) -> None:
        buffer = self.get_qsa_compressed_k_buffer(layer_id)
        buffer[loc.long()] = compressed_k.to(buffer.dtype)

    @staticmethod
    def _get_paged_state_buf_infos(tensors, page_size: int):
        return (
            [tensor.data_ptr() for tensor in tensors],
            [tensor.nbytes for tensor in tensors],
            [tensor[0].nbytes * page_size for tensor in tensors],
        )

    def get_qsa_pending_state_buf_infos(self):
        """Per-request pending key-state and RoPE ring transfer buffers.

        A pending-state "page" is one request's whole ring: the PD payload
        addresses it by ``req_pool_idx``, so the per-page item length covers
        ``qsa_ring_span`` rows (equal to the historical ratio rows whenever
        the ring holds one group).
        """
        # A PP stage without a local QSA layer never writes the shared RoPE
        # ring.  Do not register it as a transfer source: otherwise that stage
        # can race with a QSA-owning stage and overwrite valid positions with
        # its zero-initialized or stale contents.
        if not self.full_attention_layer_id_mapping:
            return [], [], []
        tensors = [*self.qsa_key_state_buffer_pool, self.qsa_rope_position_buffer]
        return self._get_paged_state_buf_infos(
            tensors,
            self.qsa_ring_span,
        )

    def get_qsa_pending_state_layer_ids(self):
        """Global layer metadata for the compact QSA pending-state list."""
        if not self.full_attention_layer_id_mapping:
            return []
        return [
            *self.full_attention_layer_id_mapping.keys(),
            QSA_ROPE_STATE_LAYER_ID,
        ]

    def get_qsa_compressed_state_layer_ids(self):
        """Global layer metadata for the compact compressed-K list."""
        return list(self.full_attention_layer_id_mapping.keys())

    def get_qsa_compressed_state_buf_infos(self):
        """Per-full-page compressed-K transfer buffers.

        One full KV page maps to one compressed page because the full page size
        is an integer multiple of the compression ratio.
        """
        return self._get_paged_state_buf_infos(
            self.qsa_compressed_k_buffer_pool,
            self.qsa_compressed_page_size,
        )

    def get_kv_size_bytes(self):
        k_size, v_size = super().get_kv_size_bytes()
        qsa_k_size = (
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.qsa_key_state_buffer_pool
            )
            + sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.qsa_compressed_k_buffer_pool
            )
            + self.qsa_rope_position_buffer.numel() * 8
        )
        return k_size + qsa_k_size, v_size
