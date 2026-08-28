from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("last_track_seqlen", [None, 0])
def test_finished_request_without_checkpoint_does_not_insert_ghost(
    last_track_seqlen,
):
    cache = MambaRadixCache.__new__(MambaRadixCache)
    cache.disable = False
    cache.enable_mamba_extra_buffer = True
    cache.token_to_kv_pool_allocator = MagicMock()
    cache.req_to_token_pool = MagicMock()
    cache.req_to_token_pool.req_to_token = torch.tensor(
        [[10, 11, 12, 13]], dtype=torch.int64
    )
    cache.insert = MagicMock()
    cache.dec_lock_ref = MagicMock()

    req = SimpleNamespace(
        origin_input_ids=array("q", [1, 2, 3]),
        output_ids=array("q", [4]),
        req_pool_idx=0,
        cache_protected_len=2,
        mamba_last_track_seqlen=last_track_seqlen,
        last_node=object(),
    )

    cache.cache_finished_req(req, kv_len_to_handle=4)

    (freed,) = cache.token_to_kv_pool_allocator.free_segment.call_args.args
    assert torch.equal(freed, torch.tensor([12, 13], dtype=torch.int64))
    assert cache.token_to_kv_pool_allocator.free_segment.call_args.kwargs == {
        "start_pos": 2
    }
    cache.req_to_token_pool.free_mamba_cache.assert_called_once_with(req)
    cache.dec_lock_ref.assert_called_once_with(req.last_node)
    cache.insert.assert_not_called()
