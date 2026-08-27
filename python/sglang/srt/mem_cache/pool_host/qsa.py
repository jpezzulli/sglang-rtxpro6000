from __future__ import annotations

import logging
import threading
from typing import Sequence

import torch

from sglang.srt.mem_cache.pool_host.base import HostKVCache, host_memory_budget_bytes
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    get_allocator_from_storage,
)
from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
_is_hip = is_hip()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_mla,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
    )

logger = logging.getLogger(__name__)


class QSACompressedPoolHost(HostKVCache):
    """Host mirror for QSA's persistent compressed index-key cache.

    QSA addresses one compressed slot per ``compress_ratio`` full-KV slots. A
    full-KV HiCache page is therefore mirrored as one dense compressed page and
    rides the anchor KV page's indices and lifetime.
    """

    def __init__(
        self,
        device_pool,
        anchor_host,
        layout: str,
        *,
        draft_device_pools: Sequence = (),
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        self.device_pool = device_pool
        self.mtp_draft_device_pools = tuple(draft_device_pools)
        self.page_size = anchor_host.page_size
        self.size = anchor_host.size
        self.page_num = anchor_host.page_num
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.dtype = device_pool.index_state_dtype
        self.target_layer_num = len(device_pool.qsa_compressed_k_buffer_pool)
        self.layer_num = self.target_layer_num + sum(
            len(pool.qsa_compressed_k_buffer_pool)
            for pool in self.mtp_draft_device_pools
        )
        self.compress_ratio = int(device_pool.qsa_compress_ratio)
        if self.page_size % self.compress_ratio != 0:
            raise ValueError(
                "QSA HiCache page size must be divisible by the compression "
                f"ratio: page_size={self.page_size}, ratio={self.compress_ratio}"
            )
        self.compressed_page_size = self.page_size // self.compress_ratio
        self.kv_heads = int(device_pool.qsa_index_kv_heads)
        self.head_dim = int(device_pool.qsa_index_head_dim)
        self.page_stride_elements = (
            self.compressed_page_size * self.kv_heads * self.head_dim
        )
        self.page_stride_bytes = self.page_stride_elements * self.dtype.itemsize
        self.page_layout_bytes = self.layer_num * self.page_stride_bytes
        self.size_per_token = self.page_layout_bytes // self.page_size
        self.can_use_jit = False
        self.can_use_write_back_jit = False

        requested_bytes = self.page_num * self.page_layout_bytes
        available_bytes = host_memory_budget_bytes()
        if requested_bytes > available_bytes:
            raise ValueError(
                "Not enough host memory for QSA compressed HiCache. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )
        logger.info(
            "Allocating %.2f GB host memory for QSA compressed index keys "
            "(tokens=%d, layers=%d, ratio=%d, layout=%s).",
            requested_bytes / 1e9,
            self.size,
            self.layer_num,
            self.compress_ratio,
            self.layout,
        )

        self.kv_buffer = self.init_kv_buffer()
        self.fd = getattr(self.allocator, "fd", None)
        self.lock = threading.RLock()
        self.clear()

    @classmethod
    def bytes_per_full_token(cls, pools: Sequence) -> int:
        total = 0
        for pool in pools:
            total += (
                len(pool.qsa_compressed_k_buffer_pool)
                * int(pool.qsa_index_kv_heads)
                * int(pool.qsa_index_head_dim)
                * pool.index_state_dtype.itemsize
                // int(pool.qsa_compress_ratio)
            )
        return total

    def get_size_per_token(self):
        return self.size_per_token

    def get_ksize_per_token(self):
        return self.size_per_token

    def init_kv_buffer(self):
        device_pools = (self.device_pool, *self.mtp_draft_device_pools)
        self.packed_device_buffers = [
            buffer
            for pool in device_pools
            for buffer in pool.qsa_compressed_k_buffer_pool
        ]
        self.device_page_refs = []
        for buffer in self.packed_device_buffers:
            page_count = buffer.shape[0] // self.compressed_page_size
            self.device_page_refs.append(
                buffer[: page_count * self.compressed_page_size].view(
                    page_count, self.page_stride_elements
                )
            )
        self.device_ptrs = torch.tensor(
            [buffer.data_ptr() for buffer in self.device_page_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

        alloc = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        if self.layout == "layer_first":
            self.compressed_k_buffer = alloc(
                (self.layer_num, self.page_num, self.page_stride_elements),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.host_refs = [
                self.compressed_k_buffer[layer] for layer in range(self.layer_num)
            ]
            self.host_ptrs = torch.tensor(
                [buffer.data_ptr() for buffer in self.host_refs],
                dtype=torch.uint64,
                device=self.device_pool.device,
            )
        elif self.layout in ("page_first", "page_first_direct"):
            self.compressed_k_buffer = alloc(
                (self.page_num, self.layer_num, 1, self.page_stride_elements),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.host_refs = []
            self.host_ptrs = None
        else:
            raise ValueError(f"Unsupported QSA HiCache layout: {self.layout}")
        return [self.compressed_k_buffer]

    def get_hybrid_pool_buffer(self):
        return [self.compressed_k_buffer]

    def _page_indices(self, host_indices, device_indices):
        if host_indices.numel() != device_indices.numel():
            raise ValueError(
                "QSA HiCache transfer index size mismatch: "
                f"host={host_indices.numel()}, device={device_indices.numel()}"
            )
        if host_indices.numel() % self.page_size != 0:
            raise ValueError("QSA HiCache transfers require complete KV pages.")
        host_pages = host_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        device_pages = (
            device_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        return host_pages, device_pages

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        # The pool entry's layer mapper has already converted the model layer
        # id to the dense QSA buffer index.  QSATokenToKVPool deliberately does
        # not expose the MHA pool's layer-sharding attributes; its compressed
        # buffers are already local and densely indexed on each rank.
        host_layer = layer_id
        device_layer = 0 if is_draft else layer_id
        host_pages, device_pages = self._page_indices(host_indices, device_indices)
        dst = device_pool.qsa_compressed_k_buffer_pool[device_layer]
        dst_slots = (
            dst.shape[0] // self.compressed_page_size
        ) * self.compressed_page_size
        dst = dst[:dst_slots]
        dst = dst.view(-1, self.page_stride_elements)

        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=self.compressed_k_buffer[host_layer],
                    dst=dst,
                    src_indices=host_pages,
                    dst_indices=device_pages,
                    item_size=self.page_stride_bytes,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_mla_pf_lf(
                    src=self.compressed_k_buffer,
                    dst=dst,
                    src_indices=host_pages,
                    dst_indices=device_pages,
                    layer_id=host_layer,
                    item_size=self.page_stride_bytes,
                    src_layout_dim=self.page_layout_bytes,
                )
            else:
                raise ValueError(f"Unsupported QSA HiCache layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.compressed_k_buffer[host_layer]],
                    dst_layers=[dst],
                    src_indices=host_pages,
                    dst_indices=device_pages,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.compressed_k_buffer],
                    dst_ptrs=[dst],
                    src_indices=host_pages,
                    dst_indices=device_pages,
                    layer_id=host_layer,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported QSA HiCache layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported QSA HiCache I/O backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        host_pages, device_pages = self._page_indices(host_indices, device_indices)
        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_all_layer_mla(
                    src_layers=self.device_ptrs,
                    dst_layers=self.host_ptrs,
                    src_indices=device_pages,
                    dst_indices=host_pages,
                    item_size=self.page_stride_bytes,
                    num_layers=self.layer_num,
                )
            elif self.layout == "page_first":
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=self.device_ptrs,
                    dst=self.compressed_k_buffer,
                    src_indices=device_pages,
                    dst_indices=host_pages,
                    item_size=self.page_stride_bytes,
                    dst_layout_dim=self.page_layout_bytes,
                    num_layers=self.layer_num,
                )
            else:
                raise ValueError(f"Unsupported QSA HiCache layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=self.device_page_refs,
                    dst_layers=self.host_refs,
                    src_indices=device_pages,
                    dst_indices=host_pages,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=self.device_page_refs,
                    dst_ptrs=[self.compressed_k_buffer],
                    src_indices=device_pages,
                    dst_indices=host_pages,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported QSA HiCache layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported QSA HiCache I/O backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        page = int(index) // self.page_size
        if self.layout == "layer_first":
            data = self.compressed_k_buffer[:, page : page + 1]
        else:
            data = self.compressed_k_buffer[page : page + 1]
        return data.flatten() if flat else data

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            self.layer_num * self.page_stride_elements,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        page = int(index) // self.page_size
        if self.layout == "layer_first":
            self.compressed_k_buffer[:, page : page + 1].copy_(
                data_page.reshape(self.layer_num, 1, self.page_stride_elements)
            )
        else:
            self.compressed_k_buffer[page : page + 1].copy_(
                data_page.reshape(1, self.layer_num, 1, self.page_stride_elements)
            )

    def get_page_buffer_meta(self, indices):
        if self.layout not in ("page_first", "page_first_direct"):
            raise ValueError(f"Unsupported QSA zero-copy layout: {self.layout}")
        if len(indices) % self.page_size != 0:
            raise ValueError("QSA storage transfer requires complete KV pages.")
        base = self.compressed_k_buffer.data_ptr()
        pages = indices.tolist()[:: self.page_size]
        ptrs = [
            base + (int(index) // self.page_size) * self.page_layout_bytes
            for index in pages
        ]
        return ptrs, [self.page_layout_bytes] * len(ptrs)

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        return (
            self.layout in ("page_first", "page_first_direct")
            and self.compressed_k_buffer.data_ptr() % page_size_bytes == 0
            and self.page_layout_bytes % page_size_bytes == 0
        )
