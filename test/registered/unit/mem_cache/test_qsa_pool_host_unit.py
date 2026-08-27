import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.pool_host.qsa import QSACompressedPoolHost


class _FakeQSAPool:
    def __init__(self, *, layers: int, slots: int, ratio: int, device: str):
        self.device = device
        self.index_state_dtype = torch.bfloat16
        self.qsa_compress_ratio = ratio
        self.qsa_index_kv_heads = 2
        self.qsa_index_head_dim = 4
        compressed_slots = slots // ratio
        self.qsa_compressed_k_buffer_pool = [
            torch.arange(
                compressed_slots * self.qsa_index_kv_heads * self.qsa_index_head_dim,
                dtype=self.index_state_dtype,
                device=device,
            ).reshape(
                compressed_slots,
                self.qsa_index_kv_heads,
                self.qsa_index_head_dim,
            )
            + layer * 1000
            for layer in range(layers)
        ]


class TestQSACompressedPoolHost(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for QSA HiCache transfer tests.")

    @staticmethod
    def _indices(page_ids, page_size, device):
        return torch.cat(
            [
                torch.arange(
                    page * page_size,
                    (page + 1) * page_size,
                    dtype=torch.int64,
                    device=device,
                )
                for page in page_ids
            ]
        )

    def test_page_first_round_trip_includes_target_and_mtp(self):
        page_size = 8
        slots = 32
        target = _FakeQSAPool(layers=2, slots=slots, ratio=4, device="cuda")
        draft = _FakeQSAPool(layers=1, slots=slots, ratio=4, device="cuda")
        anchor = SimpleNamespace(page_size=page_size, size=slots, page_num=4)
        host = QSACompressedPoolHost(
            target,
            anchor,
            "page_first",
            draft_device_pools=(draft,),
            pin_memory=False,
        )

        device_indices = self._indices([1, 2], page_size, "cuda")
        host_indices = self._indices([0, 1], page_size, "cuda")
        expected_target = [
            buffer.clone() for buffer in target.qsa_compressed_k_buffer_pool
        ]
        expected_draft = draft.qsa_compressed_k_buffer_pool[0].clone()

        host.backup_from_device_all_layer(
            target, host_indices, device_indices, io_backend="kernel"
        )
        torch.cuda.synchronize()

        compressed_page = page_size // target.qsa_compress_ratio
        target_slice = slice(compressed_page, 3 * compressed_page)
        for buffer in target.qsa_compressed_k_buffer_pool:
            buffer[target_slice].zero_()
        draft.qsa_compressed_k_buffer_pool[0][target_slice].zero_()

        for layer in range(len(target.qsa_compressed_k_buffer_pool)):
            host.load_to_device_per_layer(
                target,
                host_indices,
                device_indices,
                layer,
                "kernel",
            )
        host.load_to_device_per_layer(
            draft,
            host_indices,
            device_indices,
            len(target.qsa_compressed_k_buffer_pool),
            "kernel",
            is_draft=True,
        )
        torch.cuda.synchronize()

        for actual, expected in zip(
            target.qsa_compressed_k_buffer_pool, expected_target
        ):
            torch.testing.assert_close(actual[target_slice], expected[target_slice])
        torch.testing.assert_close(
            draft.qsa_compressed_k_buffer_pool[0][target_slice],
            expected_draft[target_slice],
        )

    def test_storage_page_round_trip_and_geometry(self):
        target = _FakeQSAPool(layers=2, slots=32, ratio=4, device="cuda")
        anchor = SimpleNamespace(page_size=8, size=32, page_num=4)
        host = QSACompressedPoolHost(
            target, anchor, "page_first", pin_memory=False
        )
        host.compressed_k_buffer.copy_(
            torch.arange(
                host.compressed_k_buffer.numel(), dtype=host.dtype
            ).reshape_as(host.compressed_k_buffer)
        )
        page = host.get_data_page(8).clone()
        host.compressed_k_buffer[1].zero_()
        host.set_from_flat_data_page(8, page)
        torch.testing.assert_close(host.get_data_page(8), page)
        self.assertEqual(page.numel() * page.element_size(), host.page_layout_bytes)


if __name__ == "__main__":
    unittest.main()
