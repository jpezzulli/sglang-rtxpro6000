"""Unfinished Mamba checkpoints degrade instead of killing the scheduler.

The checkpoint donated at a chunk boundary is an optimization.  If every
Mamba slot is protected (including slots pinned by host backup), the request
must retain its live state and continue without inserting that checkpoint.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _component(*, slot):
    component = object.__new__(MambaComponent)
    allocator = MagicMock()
    allocator.alloc.return_value = slot
    allocator.available_size.return_value = 0
    allocator.size = 28
    pool = SimpleNamespace(
        mamba_allocator=allocator,
        mamba_ckpt_pool=None,
        donate_mamba_ping_pong_slot=MagicMock(return_value=torch.tensor([3])),
    )
    component.cache = SimpleNamespace(
        req_to_token_pool=pool,
        enable_mamba_extra_buffer=True,
        evict_for_alloc=MagicMock(),
    )
    component.tree_core = SimpleNamespace(mamba_evictable_size=lambda: 0)
    return component, allocator


def _req():
    return SimpleNamespace(
        rid="r1",
        mamba_last_track_seqlen=4096,
        mamba_pool_idx=None,
    )


class TestMambaPoolExhaustion(unittest.TestCase):
    def test_no_extra_buffer_exhaustion_keeps_live_request_state(self):
        component, allocator = _component(slot=None)
        component.cache.enable_mamba_extra_buffer = False
        req = _req()
        req.mamba_pool_idx = torch.tensor([11])
        original = req.mamba_pool_idx
        params = InsertParams(prev_prefix_len=128, chunked=True, priority=0)
        self.assertEqual(
            component.prepare_for_caching_req(
                req=req, insert_params=params, token_ids_len=4096, is_finished=False
            ),
            0,
        )
        self.assertIs(req.mamba_pool_idx, original)
        self.assertIsNone(params.mamba_value)
        component.cache.req_to_token_pool.donate_mamba_ping_pong_slot.assert_not_called()
        self.assertEqual(allocator.alloc.call_count, 2)

    def test_required_allocation_remains_asserting(self):
        component, _ = _component(slot=None)
        with self.assertRaisesRegex(AssertionError, "Can not alloc mamba cache"):
            component._alloc_mamba_slot()

    def test_later_checkpoint_retries_after_skipped_donation(self):
        component, allocator = _component(slot=None)
        allocator.alloc.side_effect = [None, None, torch.tensor([7])]
        req = _req()
        params = InsertParams(prev_prefix_len=0, chunked=True, priority=0)
        self.assertEqual(
            component.prepare_for_caching_req(
                req=req, insert_params=params, token_ids_len=4096, is_finished=False
            ),
            0,
        )
        component.cache.req_to_token_pool.donate_mamba_ping_pong_slot.assert_not_called()
        component.cleanup_after_caching_req(
            req, is_finished=False, insert_params=params
        )
        self.assertIsNone(req.mamba_last_track_seqlen)
        # A subsequent model chunk publishes a fresh tracked boundary.
        req.mamba_last_track_seqlen = 8192
        self.assertEqual(
            component.prepare_for_caching_req(
                req=req, insert_params=params, token_ids_len=8192, is_finished=False
            ),
            8192,
        )
        component.cache.req_to_token_pool.donate_mamba_ping_pong_slot.assert_called_once()
        self.assertTrue(torch.equal(params.mamba_value, torch.tensor([3])))

    def test_unfinished_checkpoint_skipped_when_pool_exhausted(self):
        component, allocator = _component(slot=None)
        params = InsertParams(prev_prefix_len=0, chunked=True, priority=0)

        cache_len = component.prepare_for_caching_req(
            req=_req(), insert_params=params, token_ids_len=4096, is_finished=False
        )

        self.assertEqual(cache_len, 0)
        self.assertIsNone(params.mamba_value)
        component.cache.evict_for_alloc.assert_called_once()
        self.assertEqual(allocator.alloc.call_count, 2)

    def test_unfinished_checkpoint_donated_when_replacement_slot_available(self):
        component, allocator = _component(slot=torch.tensor([7]))
        params = InsertParams(prev_prefix_len=0, chunked=True, priority=0)

        cache_len = component.prepare_for_caching_req(
            req=_req(), insert_params=params, token_ids_len=4096, is_finished=False
        )

        self.assertEqual(cache_len, 4096)
        self.assertTrue(torch.equal(params.mamba_value, torch.tensor([3])))
        component.cache.evict_for_alloc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
