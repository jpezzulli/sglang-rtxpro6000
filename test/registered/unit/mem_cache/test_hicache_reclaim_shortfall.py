"""CPU regressions for HiCache reclaim sizing on allocation failure.

Host-pool available_size() counts pending releases that alloc() consumes
lazily via its release-slot merge, so a failed allocation never means "the
pool lacks need_size": it means the pool lacks need_size - available_size.
Every reclaim (host eviction) triggered by a failed allocation must be sized
to that shortfall only; evicting the full need size drops unrelated cached
capacity the allocation did not require, and an allocation already covered by
ready-plus-pending capacity must not evict at all.  Units (slots of the
named pool), page alignment, the full-size retry, and resolve_host_transfers
rollback must be preserved.

The pool double replicates HostKVCache alloc/free/available_size/
_merge_release_slots (pool_host/base.py) slot-for-slot; it exists only to
skip the pinned host-buffer allocation of HostKVCache.__init__.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _HostPoolDouble:
    """Replica of HostKVCache slot bookkeeping with injectable state."""

    def __init__(self, logical_size: int, page_size: int = 1):
        self.logical_size_value = logical_size
        self.page_size = page_size
        self.can_use_write_back_jit = False
        self.layout = "layer_first"
        self.device = "cpu"
        self.size = logical_size
        self.clear()

    @property
    def logical_size(self) -> int:
        return self.logical_size_value

    @property
    def logical_page_size(self) -> int:
        return self.page_size

    def clear(self):
        self.free_slots = torch.arange(self.logical_size_value, dtype=torch.int64)
        self.release_chunks = []
        self.num_release_slots = 0
        self.slot_used = torch.zeros(self.logical_size_value, dtype=torch.bool)

    # -- test-side state injection (exact available_size accounting) --------

    def stage(self, free: int, pending: int):
        """Leave exactly `free` ready slots and `pending` pending releases."""
        self.free_slots = torch.arange(0, free, dtype=torch.int64)
        self.release_chunks = [torch.arange(free, free + pending, dtype=torch.int64)]
        self.num_release_slots = pending
        self.slot_used[:] = False

    def evict(self, start: int, count: int):
        """Model a host eviction returning slots [start, start+count)."""
        self.release_chunks.append(
            torch.arange(start, start + count, dtype=torch.int64)
        )
        self.num_release_slots += count

    # -- HostKVCache allocation protocol -------------------------------------

    def available_size(self):
        return len(self.free_slots) + self.num_release_slots

    def _merge_release_slots(self):
        if self.num_release_slots == 0:
            return
        if len(self.free_slots) == 0 and len(self.release_chunks) == 1:
            self.free_slots = self.release_chunks[0]
        else:
            self.free_slots = torch.cat([self.free_slots, *self.release_chunks])
        self.release_chunks = []
        self.num_release_slots = 0

    def alloc(self, need_size: int):
        assert need_size % self.logical_page_size == 0
        if need_size > self.available_size():
            return None
        if need_size > len(self.free_slots):
            self._merge_release_slots()
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        assert not self.slot_used[select_index].any()
        self.slot_used[select_index] = True
        return select_index

    def free(self, indices: torch.Tensor) -> int:
        indices_cpu = indices.cpu()
        if indices_cpu.numel() == 0:
            return 0
        assert self.slot_used[indices_cpu].all()
        self.slot_used[indices_cpu] = False
        self.release_chunks.append(indices_cpu)
        self.num_release_slots += len(indices_cpu)
        return len(indices)

    def destroy(self):
        pass


def _entry(name, pool, host_evict_fn=None, anchor=False):
    return PoolEntry(
        name=name,
        host_pool=pool,
        device_pool=MagicMock(),
        layer_mapper=lambda layer_id: layer_id,
        is_primary_index_anchor=anchor,
        host_evict_fn=host_evict_fn,
    )


class TestHostPoolGroupAllocShortfall(unittest.TestCase):
    def _group(self, pool, host_evict_fn=None):
        return HostPoolGroup([_entry(PoolName.KV, pool, host_evict_fn, anchor=True)])

    def test_reclaim_receives_shortfall_including_pending_releases(self):
        # 4 ready + 6 pending-release = 10 available; a 16-slot request lacks
        # 6, not 16.  The pending releases already count toward the need.
        pool = _HostPoolDouble(64)
        pool.stage(free=4, pending=6)
        reclaim = MagicMock()
        group = self._group(pool, reclaim)

        indices = group.alloc(16, reclaim=reclaim)

        self.assertIsNone(indices)  # the mocked reclaim frees nothing
        reclaim.assert_called_once_with(6)

    def test_pending_release_covers_need_without_any_reclaim(self):
        pool = _HostPoolDouble(16)
        pool.stage(free=0, pending=16)
        reclaim = MagicMock()
        group = self._group(pool, reclaim)

        indices = group.alloc(16, reclaim=reclaim)

        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), 16)
        reclaim.assert_not_called()
        self.assertEqual(pool.available_size(), 0)

    def test_retry_succeeds_after_shortfall_reclaim_frees_the_gap(self):
        pool = _HostPoolDouble(64)
        pool.stage(free=2, pending=1)
        seen = []

        def reclaim(n):
            seen.append(n)
            if n >= 5:
                pool.evict(20, 5)

        indices = self._group(pool, reclaim).alloc(8, reclaim=reclaim)

        self.assertEqual(seen, [5])  # 8 - (2 + 1) available
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), 8)

    def test_reclaim_sized_to_full_need_only_when_pool_truly_empty(self):
        pool = _HostPoolDouble(64)
        pool.stage(free=0, pending=0)
        reclaim = MagicMock()
        group = self._group(pool, reclaim)

        self.assertIsNone(group.alloc(16, reclaim=reclaim))
        reclaim.assert_called_once_with(16)

    def test_no_reclaim_call_when_no_callback(self):
        pool = _HostPoolDouble(16)
        pool.stage(free=0, pending=2)
        self.assertIsNone(self._group(pool).alloc(8))

    def test_side_pool_units_and_alignment_preserved(self):
        anchor = _HostPoolDouble(16, page_size=4)
        side = _HostPoolDouble(32, page_size=4)
        evicts = []
        group = HostPoolGroup(
            [
                _entry(PoolName.KV, anchor, anchor=True),
                _entry(PoolName.SWA, side, host_evict_fn=lambda n: evicts.append(n)),
            ]
        )
        # Page-aligned requests succeed; unaligned ones assert in the pool.
        self.assertIsNotNone(group.alloc(4, pool=PoolName.SWA))
        with self.assertRaises(AssertionError):
            group.alloc(3, pool=PoolName.SWA)
        # Reclaim is sized per named pool (SWA), not per the anchor pool.
        side.clear()
        side.stage(free=0, pending=4)
        self.assertIsNone(
            group.alloc(12, pool=PoolName.SWA, reclaim=lambda n: evicts.append(n))
        )
        self.assertEqual(evicts, [8])  # 12 - 4 available in the SWA pool


class TestResolveHostTransfersReclaim(unittest.TestCase):
    def test_side_pool_reclaim_gets_shortfall_not_request_size(self):
        side = _HostPoolDouble(32)
        side.stage(free=1, pending=5)
        side_evicts = []

        def reclaim(n):
            side_evicts.append(n)
            if n >= 6:
                side.evict(20, 6)

        anchor = _HostPoolDouble(32)
        group = HostPoolGroup(
            [
                _entry(PoolName.KV, anchor, anchor=True),
                _entry(PoolName.SWA, side, host_evict_fn=reclaim),
            ]
        )
        transfer = PoolTransfer(
            name=PoolName.SWA, device_indices=torch.arange(12, dtype=torch.int64)
        )

        resolved = group.resolve_host_transfers([transfer])

        self.assertEqual(side_evicts, [6])  # 12 - (1 + 5) available
        self.assertIsNotNone(resolved)
        self.assertEqual(len(resolved[0].host_indices), 12)

    def test_rollback_releases_every_allocation_on_failure(self):
        side = _HostPoolDouble(8)
        mamba = _HostPoolDouble(2)
        anchor = _HostPoolDouble(32)
        group = HostPoolGroup(
            [
                _entry(PoolName.KV, anchor, anchor=True),
                _entry(PoolName.SWA, side),
                _entry(PoolName.MAMBA, mamba),
            ]
        )
        side_before = side.available_size()
        transfers = [
            PoolTransfer(
                name=PoolName.SWA, device_indices=torch.arange(4, dtype=torch.int64)
            ),
            PoolTransfer(  # 4 indices do not fit the 2-slot MAMBA host pool
                name=PoolName.MAMBA,
                device_indices=torch.arange(4, dtype=torch.int64),
            ),
        ]

        self.assertIsNone(group.resolve_host_transfers(transfers))
        self.assertIsNone(transfers[0].host_indices)
        self.assertEqual(side.available_size(), side_before)

    def test_derived_transfer_shares_source_without_allocating(self):
        anchor = _HostPoolDouble(32)
        side = _HostPoolDouble(32)
        group = HostPoolGroup(
            [
                _entry(PoolName.KV, anchor, anchor=True),
                _entry(PoolName.SWA, side),
                _entry(PoolName.DRAFT, side),
            ]
        )
        primary_host = anchor.alloc(4)
        transfers = [
            PoolTransfer(
                name=PoolName.SWA, device_indices=torch.arange(4, dtype=torch.int64)
            ),
            PoolTransfer(name=PoolName.DRAFT, indices_from_pool=PoolName.SWA),
        ]
        resolved = group.resolve_host_transfers(
            transfers,
            primary_device_indices=torch.arange(4, dtype=torch.int64),
            primary_host_indices=primary_host,
        )
        self.assertIsNotNone(resolved)
        self.assertTrue(torch.equal(resolved[1].host_indices, resolved[0].host_indices))
        # Each allocation is released exactly once (derived pool skipped).
        self.assertEqual(side.available_size(), 28)  # 32 - 4 staged
        self.assertEqual(group.release_transfers(resolved), 4)
        self.assertEqual(side.available_size(), 32)  # all back as pending


class TestRetractionBackupShortfall(unittest.TestCase):
    def _cache(self, free: int, pending: int, need: int = 16, evict_frees=True):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        host = _HostPoolDouble(64)
        host.stage(free=free, pending=pending)
        cache = object.__new__(UnifiedRadixCache)
        cache.disable = False
        cache.tree_core = SimpleNamespace(page_size=1)
        cache.host_pool_group = HostPoolGroup(
            [_entry(PoolName.KV, host, host_evict_fn=None, anchor=True)]
        )
        evicted = []

        def evict_host(n):
            evicted.append(n)
            if evict_frees and n >= need - (free + pending):
                host.evict(40, n)

        cache.evict_host = evict_host
        cache.cache_controller = MagicMock()
        cache.cache_controller._move_write_operation.return_value = (None, None, None)
        req = SimpleNamespace(seqlen=need + 1, req_pool_idx=0, rid="r")
        device_indices = torch.arange(need, dtype=torch.int64)
        cache._retraction_device_transfers = lambda req: (device_indices, [])
        return cache, host, evicted, req

    def test_allocation_covered_by_pending_releases_evicts_nothing(self):
        # 8 ready + 8 pending = 16 available for a 16-slot retraction: the
        # lazy merge inside alloc() covers the need; no unrelated eviction.
        cache, _host, evicted, req = self._cache(8, 8)
        backup = cache.retraction_backup(req)
        self.assertIsNotNone(backup)
        self.assertEqual(evicted, [])

    def test_reclaim_sized_to_gap_when_pool_genuinely_short(self):
        # 4 available, 12 short: reclaim must be asked for 12, not 16.
        cache, _host, evicted, req = self._cache(4, 0, evict_frees=False)
        backup = cache.retraction_backup(req)
        self.assertIsNone(backup)  # the mocked reclaim frees nothing
        self.assertEqual(evicted, [12])

    def test_full_shortfall_still_retries_the_full_alloc_size(self):
        cache, host, evicted, req = self._cache(0, 0)
        backup = cache.retraction_backup(req)
        self.assertEqual(evicted, [16])
        self.assertIsNotNone(backup)
        self.assertEqual(len(backup.host_indices), 16)


class TestHiRadixWriteBackupShortfall(unittest.TestCase):
    """HiRadixCache.write_backup host reclaim sizing."""

    def _cache(self, free, pending, node_len=10, evict_frees=None):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        cache = object.__new__(HiRadixCache)
        host = _HostPoolDouble(64)
        host.stage(free=free, pending=pending)
        cc = MagicMock()
        cache.cache_controller = cc
        cc.mem_pool_host = host
        cc.write_policy = "write_through"
        write_state = {"calls": 0}

        def write(device_indices, node_id=None, **kwargs):
            # The controller allocates from the host pool on write().
            write_state["calls"] += 1
            return host.alloc(len(device_indices))

        cc.write.side_effect = write
        evicted = []

        def evict_host(n):
            evicted.append(n)
            if evict_frees is not None and n >= evict_frees:
                host.evict(40, n)

        cache.evict_host = evict_host
        cache._get_extra_pools = lambda: {}
        cache._track_write_through_node = lambda node, backup_len: None
        cache.inc_lock_ref = lambda node: None
        node = SimpleNamespace(
            value=torch.arange(node_len, dtype=torch.int64),
            key=list(range(node_len)),
            id=1,
            parent=SimpleNamespace(backuped=True),
            host_value=None,
        )
        cache.root_node = object()
        node.parent.__eq__ = lambda other: other is cache.root_node
        return cache, host, evicted, node, write_state

    def test_backup_reclaim_sized_to_shortfall(self):
        # 4 ready + 2 pending = 6 available for a 10-token node: evict 4.
        cache, _host, evicted, node, _ws = self._cache(4, 2)
        cache.write_backup(node, write_back=True)
        self.assertEqual(evicted, [4])

    def test_backup_covered_by_pending_releases_evicts_nothing(self):
        cache, _host, evicted, node, _ws = self._cache(0, 10)
        written = cache.write_backup(node, write_back=True)
        self.assertEqual(evicted, [])
        self.assertEqual(written, 10)

    def test_backup_zero_available_evicts_full_node_size(self):
        cache, _host, evicted, node, _ws = self._cache(0, 0)
        written = cache.write_backup(node, write_back=True)
        self.assertEqual(evicted, [10])
        self.assertEqual(written, 0)  # mocked reclaim frees nothing


class TestHiRadixLoadBackShortfall(unittest.TestCase):
    """HiRadixCache.load_back device-side reclaim sizing."""

    def _cache(self, device_avail, host_len=12, evict_frees=None):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        cache = object.__new__(HiRadixCache)
        cache.page_size = 1
        cache.load_back_threshold = 10
        cache.kv_events = MagicMock()
        allocator = MagicMock()
        allocator.available_size.return_value = device_avail
        cache.token_to_kv_pool_allocator = allocator
        cc = MagicMock()
        cache.cache_controller = cc
        load_state = {"calls": 0}

        def load(host_indices=None, node_id=None, **kwargs):
            load_state["calls"] += 1
            # First call fails (no device room); succeeds once the allocator
            # reports the shortfall as evicted.
            if load_state["calls"] > 1:
                return torch.arange(host_indices.numel(), dtype=torch.int64)
            return None

        cc.load.side_effect = load
        evicted = []

        def evict(params):
            evicted.append(params.num_tokens)
            if evict_frees is not None and params.num_tokens >= evict_frees:
                pass  # eviction is the controller's problem for this probe

        cache.evict = evict
        cache.inc_lock_ref = lambda node: SimpleNamespace(delta=0)
        cache.dec_lock_ref = lambda node: None
        cache.ongoing_load_back = {}
        cache.evictable_size_ = 0
        parent = SimpleNamespace(evicted=False, backuped=True)
        node = SimpleNamespace(
            evicted=True,
            backuped=True,
            host_value=torch.arange(host_len, dtype=torch.int64),
            id=7,
            parent=parent,
            value=None,
            protect_host=lambda: None,
            release_host=lambda: None,
        )
        parent.__eq__ = lambda other: other is cache.root_node
        cache.root_node = object()
        cache.ancester = parent
        return cache, node, evicted, load_state

    def test_load_back_evicts_only_the_device_shortfall(self):
        # 12 tokens to load, device allocator already holds 5 free: ask the
        # eviction drive for 7, not 12.
        cache, node, evicted, _ls = self._cache(5, host_len=12)
        result = cache.load_back(node)
        self.assertEqual(evicted, [7])
        self.assertIsNotNone(result)

    def test_load_back_full_evict_when_nothing_available(self):
        cache, node, evicted, _ls = self._cache(0, host_len=12)
        cache.load_back(node)
        self.assertEqual(evicted, [12])


class TestDeviceReclaimKeepsFullSize(unittest.TestCase):
    """Device reclaim keeps the absolute-size contract (upstream #40748)."""

    def test_device_evict_fn_receives_full_alloc_size(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        controller = object.__new__(HybridCacheController)
        side = _HostPoolDouble(16)
        side.stage(free=1, pending=5)  # available 6
        device_pool = MagicMock()
        device_pool.alloc.return_value = None
        evicted = []
        entry = PoolEntry(
            name=PoolName.SWA,
            host_pool=side,
            device_pool=device_pool,
            layer_mapper=lambda layer_id: layer_id,
            host_evict_fn=lambda n: evicted.append(("host", n)),
            device_evict_fn=lambda n: evicted.append(("device", n)),
        )
        controller.mem_pool_host = SimpleNamespace(entry_map={PoolName.SWA: entry})
        transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.arange(12, dtype=torch.int64),
        )

        resolved = controller._resolve_device_transfers([transfer])

        self.assertIsNone(resolved)
        self.assertEqual(evicted, [("device", 12)])  # full size, not shortfall


class TestHiRadixStorageHitShortfall(unittest.TestCase):
    """HiRadixCache storage-hit prefetch staging drain."""

    def _run_drain(self, free, pending, alloc_len=16, threshold=4, evict_frees=None):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        cache = object.__new__(HiRadixCache)
        cache.page_size = 1
        cache.prefetch_threshold = threshold
        host = _HostPoolDouble(64)
        host.stage(free=free, pending=pending)
        cc = MagicMock()
        cache.cache_controller = cc
        cc.mem_pool_host = host
        evicted = []

        def evict_host(n):
            evicted.append(n)
            if evict_frees is not None and n >= evict_frees:
                host.evict(40, evict_frees)

        cache.evict_host = evict_host
        operation = SimpleNamespace(
            request_id="r1",
            storage_hit_count=alloc_len,
            host_indices=None,
            hash_value=["h"] * alloc_len,
            is_terminated=lambda: False,
        )
        hit = MagicMock()
        hit.get.return_value = operation
        cc.prefetch_hit_queue = hit
        cache.ongoing_prefetch = {"r1": (MagicMock(), [0] * alloc_len, operation)}
        revoked = []
        cache._revoke_pending_prefetch = lambda rid: revoked.append(rid)
        cache._drain_storage_control_queues_impl(
            n_storage_hit=1,
            n_ack_prefetch=0,
            n_backup=0,
            n_release=0,
            log_metrics=False,
        )
        return cache, host, operation, evicted, revoked

    def test_hit_alloc_reclaims_only_the_shortfall(self):
        # 4 + 6 pending = 10 available for a 16-token hit: reclaim 6, then
        # the page-aligned memory-pressure fallback stages what fits.
        _cache, _host, operation, evicted, _revoked = self._run_drain(4, 6)
        self.assertEqual(evicted, [6])
        self.assertIsNotNone(operation.host_indices)
        self.assertLessEqual(operation.storage_hit_count, 16)

    def test_hit_alloc_stages_full_hit_after_gap_filling_reclaim(self):
        # 0 free + 10 pending for a 16-token hit: shortfall 6; a reclaim that
        # frees exactly the gap must let the FULL hit stage.
        _cache, _host, operation, evicted, revoked = self._run_drain(
            0, 10, evict_frees=6
        )
        self.assertEqual(evicted, [6])
        self.assertEqual(revoked, [])
        self.assertEqual(operation.storage_hit_count, 16)
        self.assertEqual(len(operation.host_indices), 16)

    def test_zero_available_reclaims_full_hit_size_then_revokes(self):
        _cache, _host, operation, evicted, revoked = self._run_drain(0, 0)
        self.assertEqual(evicted, [16])
        self.assertEqual(revoked, ["r1"])
        self.assertIsNone(operation.host_indices)


class TestUnifiedStorageHitShortfall(unittest.TestCase):
    """UnifiedRadixCache storage-hit prefetch staging drain."""

    def _run(self, free, pending, alloc_len=16, threshold=4, evict_frees=None):
        from sglang.srt.mem_cache.unified_radix_cache import (
            _OngoingPrefetch,
            UnifiedRadixCache,
        )

        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(page_size=1)
        cache.prefetch_threshold = threshold
        cache.host_memory_mode = "cache"
        host = _HostPoolDouble(64)
        host.stage(free=free, pending=pending)
        cc = MagicMock()
        cc.prefetch_rate_limited.return_value = False
        cc.mem_pool_host = host
        cache.cache_controller = cc
        evicted = []

        def evict_host(n):
            evicted.append(n)
            if evict_frees is not None and n >= evict_frees:
                host.evict(40, evict_frees)

        cache.evict_host = evict_host
        operation = SimpleNamespace(
            request_id="r1",
            storage_hit_count=alloc_len,
            host_indices=None,
            hash_value=["h"] * alloc_len,
            is_terminated=lambda: False,
        )
        hit = MagicMock()
        hit.get.return_value = operation
        cc.prefetch_hit_queue = hit
        info = _OngoingPrefetch(
            anchor_node_id=1,
            prefetch_key=[0] * alloc_len,
            host_indices=None,
            operation=operation,
            anchor_lock_params=None,
            comp_xfers={},
        )
        cache.ongoing_prefetch = {"r1": info}
        cache._invalidate_absent_from_hit_query = lambda operation: None
        cache._account_prefetch_outcome = lambda operation, revoked: None
        revoked = []
        cache.revoke_pending_prefetch = lambda rid: revoked.append(rid)
        cache._drain_storage_control_queues_impl(
            n_storage_hit=1,
            n_ack_prefetch=0,
            n_backup=0,
            n_release=0,
            extra_release_counts=None,
            log_metrics=False,
        )
        return cache, host, operation, evicted, revoked

    def test_hit_alloc_stages_full_hit_after_gap_filling_reclaim(self):
        _cache, _host, operation, evicted, revoked = self._run(
            0, 10, evict_frees=6
        )
        self.assertEqual(evicted, [6])  # 16 - (0 + 10) available
        self.assertEqual(revoked, [])
        self.assertEqual(operation.storage_hit_count, 16)
        self.assertEqual(len(operation.host_indices), 16)

    def test_zero_available_reclaims_full_hit_size_then_revokes(self):
        _cache, _host, operation, evicted, revoked = self._run(0, 0)
        self.assertEqual(evicted, [16])
        self.assertEqual(revoked, ["r1"])
        self.assertIsNone(operation.host_indices)


class TestSWAComponentPrefetchStaging(unittest.TestCase):
    def _component(self, pool, evict_calls, evict_frees=None):
        from sglang.srt.mem_cache.unified_cache.components.swa_component import (
            SWAComponent,
        )

        component = object.__new__(SWAComponent)
        component._swa_kv_pool_host = pool
        component.full_window_pages = 2
        component.tree_core = SimpleNamespace(is_root=lambda node_id: True)

        def evict_host(size, ct):
            evict_calls.append(size)
            if evict_frees is not None and size >= evict_frees:
                pool.evict(20, size)
            return 0

        component.cache = SimpleNamespace(
            page_size=4,
            host_memory_mode="cache",
            host_pool_group=HostPoolGroup(
                [_entry(PoolName.SWA, pool, host_evict_fn=evict_host, anchor=True)]
            ),
            evict_host=evict_host,
        )
        return component

    def test_staging_covered_by_pending_releases_needs_no_reclaim(self):
        pool = _HostPoolDouble(32)
        pool.stage(free=0, pending=8)
        evict_calls = []
        component = self._component(pool, evict_calls)
        result = component.prepare_prefetch(None, prefetch_tokens=32)
        self.assertFalse(result.alloc_failed)
        self.assertEqual(len(result.host_indices), 8)
        self.assertEqual(evict_calls, [])

    def test_staging_reclaim_sized_to_shortfall_then_retries_full_size(self):
        pool = _HostPoolDouble(32)
        pool.stage(free=1, pending=1)  # need 8 -> shortfall 6
        evict_calls = []
        component = self._component(pool, evict_calls, evict_frees=6)
        result = component.prepare_prefetch(None, prefetch_tokens=32)
        self.assertEqual(evict_calls, [6])
        self.assertFalse(result.alloc_failed)
        self.assertEqual(len(result.host_indices), 8)


class TestMambaComponentPrefetchStaging(unittest.TestCase):
    def _component(self, pool, evict_calls):
        from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
            MambaComponent,
        )

        component = object.__new__(MambaComponent)
        component.cache = SimpleNamespace(
            host_pool_group=HostPoolGroup(
                [
                    _entry(
                        PoolName.MAMBA,
                        pool,
                        host_evict_fn=lambda n: evict_calls.append(n) or 0,
                        anchor=True,
                    )
                ]
            ),
            evict_host=lambda size, ct: evict_calls.append(size) or 0,
        )
        return component

    def test_single_slot_staging_success_skips_reclaim(self):
        pool = _HostPoolDouble(16)
        pool.stage(free=0, pending=1)
        evict_calls = []
        component = self._component(pool, evict_calls)
        result = component.prepare_prefetch(None, prefetch_tokens=8)
        self.assertFalse(result.alloc_failed)
        self.assertEqual(len(result.host_indices), 1)
        self.assertEqual(evict_calls, [])

    def test_single_slot_staging_failure_reclaims_exact_gap(self):
        pool = _HostPoolDouble(16)
        pool.stage(free=0, pending=0)
        evict_calls = []
        component = self._component(pool, evict_calls)
        result = component.prepare_prefetch(None, prefetch_tokens=8)
        self.assertTrue(result.alloc_failed)
        self.assertEqual(evict_calls, [1])  # need 1, available 0


if __name__ == "__main__":
    unittest.main()
