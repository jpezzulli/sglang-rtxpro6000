"""KV pools carrying the QSA sparse-attention indexer caches.

``QSATokenToKVPool`` (compressed, Qwen4-Exp) adds the per-token BF16 index-key
state, its RoPE coordinates, and the paged compressed-K cache on top of the
hybrid full/linear KV pool. ``QwenDSATokenToKVPool`` (tokenwise,
Qwen3Next-DSA) adds only the flat per-token index-K cache.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from sglang.srt.mem_cache.memory_pool import GB, HybridLinearKVPool, MambaPool


def _index_k_bytes(*, kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
    return kv_heads * head_dim * dtype.itemsize


class QSATokenToKVPool(HybridLinearKVPool):
    """Hybrid KV pool with the minimal BF16 state required by simple QSA."""

    # DSV4-style compressed addressing: the full-KV allocator is paged with
    # the page a multiple of the compress ratio, so every compression
    # group's raw tokens are contiguous in one page and
    # ``compressed_slot = full_slot // compress_ratio`` is a stable,
    # collision-free mapping (any of the group's slots floor-divides to the
    # same value). The compressed cache therefore has exactly
    # ``full_slots // ratio`` slots, its lifecycle rides the full-KV
    # allocator and radix tree (page-granular sharing shares compressed
    # slots by arithmetic), and no ownership bookkeeping exists. Full slot 0
    # is the pools' reserved padding slot, so compressed slot 0 stays the
    # inert dump target for non-boundary rows.
    index_state_dtype = torch.bfloat16

    @classmethod
    def qsa_bytes_per_token(
        cls, *, kv_heads: int, head_dim: int, compress_ratio: int, num_layers: int
    ) -> int:
        """Per-token cost of the QSA index caches: the compressed keys only.

        Pre-compression state is a per-request ring of ``compress_ratio *
        qsa_num_groups`` slots (the groups a verify window can still touch),
        not a per-token cache, so it does not price per token; its total is
        bounded by the request-slot count and stays outside this budget like
        the other per-request buffers.
        """
        index_k_bytes = _index_k_bytes(
            kv_heads=kv_heads, head_dim=head_dim, dtype=cls.index_state_dtype
        )
        return index_k_bytes // compress_ratio * num_layers

    @staticmethod
    def pending_ring_num_groups(
        *, max_num_draft_tokens: Optional[int], compress_ratio: int
    ) -> int:
        """Groups the pending ring must hold so the widest window it is sized
        for fits without a within-forward collision.

        Thin delegation to ``qsa.metadata.pending_ring_groups_required``, the
        single span model every producer shares (see that docstring for the
        published-window-plus-retained-prefix span and for why a window no
        wider than one group keeps the shipped single-group layout). The count
        is derived from the LAUNCH-MAXIMUM draft-token window, never the
        active one: a narrower active width (W4/W8 under a W16 launch, or any
        adaptive-MTP step-down) reuses this allocation, so geometry never has
        to be reallocated while CUDA graphs hold the ring buffers.
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
    ):
        if page_size <= 1 or page_size % qsa_compress_ratio != 0:
            raise ValueError(
                "compressed QSA requires a paged full-KV cache with the page "
                "a multiple of the compress ratio (compressed slots are "
                f"full_slot // ratio): page_size={page_size}, "
                f"ratio={qsa_compress_ratio}. With MambaRadixCache this "
                "needs the mamba extra-buffer strategy or "
                "--disable-radix-cache (see the Qwen4-Exp arg overrides)."
            )
        # The base __init__ computes mem_usage through the overridden
        # get_kv_size_bytes before the QSA buffers exist; give them empty
        # placeholders first and recompute mem_usage at the end.
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
        state_size = size + page_size
        # Compressed slots mirror the full-KV slot space 1:ratio; the "page"
        # seen by the scoring kernels is one full-KV page's worth of groups.
        self.qsa_compressed_page_size = page_size // self.qsa_compress_ratio
        self.qsa_compressed_capacity = -(state_size // -self.qsa_compress_ratio)
        # Pre-compression index-K state is a per-request RING, not a
        # per-token cache: once a group's compressed key is written, its raw
        # members are never read again. Radix-shared prefixes are page-aligned,
        # but a request's private chunk tail can end inside a group. The state
        # that must survive a forward is the groups a forward may still
        # touch -- ``qsa_ring_span = ratio * qsa_num_groups`` slots per request,
        # addressed as
        # ``req_pool_idx * qsa_ring_span + group * ratio + position % ratio``
        # (see qsa.metadata.pending_ring_slot; the in-graph row-metadata
        # kernel must derive the same slot from the same count). Request slot 0
        # is never allocated, so its whole span is inert: ring rows [0, ratio)
        # double as the dump target for tokens whose group already compressed
        # in the same forward, and that region stays disjoint from every
        # allocated request at any group count.
        if num_request_slots <= 0:
            raise ValueError(
                f"QSA pending ring needs request slots, got {num_request_slots}"
            )
        if qsa_num_groups < 1:
            raise ValueError(f"QSA pending ring needs groups, got {qsa_num_groups}")
        self.qsa_num_request_slots = int(num_request_slots)
        # A verify window of ``num_draft_tokens`` positions is written whole
        # before any of it is compressed, and the first completed group may
        # still read retained prefix members, so the ring must not wrap a
        # live member onto itself within one forward -- see
        # pending_ring_num_groups for the exact span and for why the
        # qualified single-group window (W4, num_groups 1) keeps its layout.
        # The count is the LAUNCH capacity (sized off the resolved maximum
        # draft-token window, so W4/W8/W16 candidates all fit) and it is fixed
        # at pool construction because CUDA graphs bind these buffers for the
        # graph's lifetime: an active width step-down reuses this allocation
        # instead of resizing buffers the captured graphs still point at.
        self.qsa_num_groups = int(qsa_num_groups)
        self.qsa_ring_span = self.qsa_compress_ratio * self.qsa_num_groups
        ring_slots = self.qsa_num_request_slots * self.qsa_ring_span
        self.qsa_key_state_buffer_pool = [
            torch.zeros(
                (ring_slots, self.qsa_index_kv_heads, self.qsa_index_head_dim),
                dtype=self.index_state_dtype,
                device=device,
            )
            for _ in full_attention_layer_ids
        ]
        # RoPE coordinates are layer-independent.  Keep the exact Qwen4-Exp MRoPE
        # position of every pending key so compression can rotate the pooled
        # key with the group's real starting coordinate.
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
            dtype=self.index_state_dtype,
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
        return self.qsa_compressed_k_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def set_qsa_compressed_k_buffer(
        self, layer_id: int, loc: torch.Tensor, compressed_k: torch.Tensor
    ) -> None:
        buffer = self.get_qsa_compressed_k_buffer(layer_id)
        buffer[loc.long()] = compressed_k.to(buffer.dtype)

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


class QwenDSATokenToKVPool(HybridLinearKVPool):
    """Hybrid KV pool carrying the per-token index-K cache of tokenwise QSA.

    Only the BF16 reference layout is ported: one flat
    ``[size + page_size, index_kv_heads, index_head_dim]`` buffer per
    full-attention (DSA) layer, addressed by raw KV slots.  The FP8
    deep_gemm layout is intentionally not ported yet and the caller-side
    fast paths fail loudly instead of silently degrading.
    """

    index_state_dtype = torch.bfloat16

    @classmethod
    def qsa_bytes_per_token(
        cls, *, kv_heads: int, head_dim: int, num_layers: int
    ) -> int:
        return (
            _index_k_bytes(
                kv_heads=kv_heads, head_dim=head_dim, dtype=cls.index_state_dtype
            )
            * num_layers
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
        qsa_token_budget: int,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        start_layer: Optional[int] = None,
        full_kv_pool_class: Optional[type] = None,
        quant_method=None,
        post_capture_active: bool = False,
    ):
        if page_size != 64:
            raise ValueError(
                "tokenwise QSA requires KV-cache page_size 64 for its paged "
                f"indexer buffer, got {page_size}"
            )
        self.dsa_index_k_buffer_pool = []
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
        if qsa_index_kv_heads != 1:
            raise ValueError(
                f"tokenwise QSA requires index_kv_heads = 1 (MQA), got "
                f"{qsa_index_kv_heads}"
            )
        if min(qsa_index_kv_heads, qsa_index_head_dim, qsa_token_budget) <= 0:
            raise ValueError("QSA cache configuration values must be positive")
        self.qsa_compress_ratio = 1
        self.qsa_index_kv_heads = int(qsa_index_kv_heads)
        self.qsa_index_head_dim = int(qsa_index_head_dim)
        self.qsa_token_topk = int(qsa_token_budget)
        self.qsa_block_topk = int(qsa_token_budget)
        state_size = size + page_size
        self.dsa_index_k_buffer_pool = [
            torch.zeros(
                (state_size, self.qsa_index_kv_heads, self.qsa_index_head_dim),
                dtype=self.index_state_dtype,
                device=device,
            )
            for _ in full_attention_layer_ids
        ]
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def set_dsa_index_k_buffer(
        self, layer_id: int, loc: torch.Tensor, index_k: torch.Tensor
    ) -> None:
        buffer = self.get_dsa_index_k_buffer(layer_id)
        buffer[loc.long()] = index_k.to(buffer.dtype)

    def get_dsa_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        return self.dsa_index_k_buffer_pool[self._transfer_full_attention_id(layer_id)]

    def get_kv_size_bytes(self):
        k_size, v_size = super().get_kv_size_bytes()
        dsa_k_size = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.dsa_index_k_buffer_pool
        )
        return k_size + dsa_k_size, v_size
