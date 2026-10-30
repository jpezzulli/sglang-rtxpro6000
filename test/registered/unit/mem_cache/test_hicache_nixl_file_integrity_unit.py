"""FILE-backend integrity and bounce-capacity contract for HiCacheNixl.

CPU-only (no nixl install needed): the module-level nixl import is satisfied
with a minimal stub so the storage class itself can be exercised against a
fake agent. Two real-backend holes are pinned here, both on the selected
HiCache + NIXL FILE (POSIX) path:

1. Existence and restore must validate payload bytes, not just metadata
   presence. NIXL <= 1.4 can report a short READ as successful and a failed
   WRITE can leave a zero-byte file registered, so an empty or truncated
   object reads back as a hit and injects garbage into the host pages.
   A size-query failure degrades to the existing failed-read/recompute
   channel instead of escaping into the prefetch worker thread.
   (The controller's ``assume_stored`` hit path skips ``batch_exists``
   entirely, so the read entrypoint is where the byte check must live.)

2. Bounce-backed hybrid (v2) transfers share one ``STORAGE_BATCH_SIZE``
   staging buffer, but the HiCache hybrid controller hands a sidecar
   transfer the FULL operation span (only the KV path is chunked), so
   any span over 128 pages used to fail the whole sidecar write/read and
   degrade every later lookup to recompute. v2 transfers must be split
   into bounded windows instead.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import contextlib
import sys
import types
import unittest
from dataclasses import replace  # noqa: F401  (parity with the CUDA suite)
from types import SimpleNamespace

import torch


def _install_nixl_stub():
    if "nixl._api" in sys.modules:
        return
    nixl = types.ModuleType("nixl")
    api = types.ModuleType("nixl._api")

    class _Agent:
        def __init__(self, *args, **kwargs):
            pass

        def getAvailPlugins(self):
            return []

    class _Bind:
        NIXL_THREAD_SYNC_STRICT = 0
        NIXL_THREAD_SYNC_RW = 1

        nixlAgentConfig = SimpleNamespace(
            useProgThread=False,
            useListenThread=False,
            listenPort=0,
            syncMode=0,
            pthrDelay=0,
            lthrDelay=0,
            captureTelemetry=False,
        )
        nixlAgent = _Agent

    api.nixl_agent = _Agent
    api.nixl_agent_config = lambda *args, **kwargs: SimpleNamespace(
        enable_pthread=False, enable_listen=False, port=0, capture_telemetry=False
    )
    api.nixlBind = _Bind
    nixl._api = api
    sys.modules["nixl"] = nixl
    sys.modules["nixl._api"] = api


_install_nixl_stage = None
try:
    _install_nixl_stub()
except Exception:  # pragma: no cover - defensive
    pass

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from sglang.srt.mem_cache.storage.nixl.hicache_nixl import (  # noqa: E402
    HiCacheNixl,
    _HybridPoolContext,
)

FILE_SIZES = {  # object name -> reported payload size (missing key -> None)
    "full": {"size": 64},
    "short": {"size": 32},
    "empty": {"size": 0},
}


class _FakeAgent:
    """Answers query_memory per path, like the posix backend's size dicts."""

    def __init__(self, entries=None, query_raises=False):
        self._entries = entries or FILE_SIZES
        self._query_raises = query_raises
        self.queries = 0
        self.transfers_started = 0

    def query_memory(self, descs, backend, mem_type=None):
        self.queries += 1
        if self._query_raises:
            raise RuntimeError("metadata query failed")
        out = []
        for (_, _, _, path) in descs:
            name = path[len("/base/") :] if path.startswith("/base/") else path
            out.append(self._entries.get(name))
        return out

    def get_xfer_descs(self, descs, mem_type):
        return ("host", descs)

    def initialize_xfer(self, direction, host_descs, storage_descs, agent):
        self.transfers_started += 1
        return "xfer"

    def transfer(self, handle):
        return "DONE"

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        pass


@contextlib.contextmanager
def _fake_registry_storage():
    yield ("storage",)


def _bare_nixl(
    tmp_self,
    *,
    mem_type="FILE",
    entries=None,
    query_raises=False,
):
    h = HiCacheNixl.__new__(HiCacheNixl)
    h.backend_selector = SimpleNamespace(mem_type=mem_type, backend_name="POSIX")
    h.file_manager = SimpleNamespace(get_file_path=lambda key: f"/base/{key}")
    h.agent = _FakeAgent(entries=entries, query_raises=query_raises)
    h.registry = SimpleNamespace(storage=lambda *a: _fake_registry_storage())
    h.agent_name = "fake-agent"
    h._xfer_and_wait_real = h._xfer_and_wait
    h.is_zero_copy = False
    h.is_mla_model = False
    h.config_suffix = "_t"
    h.tp_config_suffix = "_t"
    return h


class TestFilePayloadValidation(unittest.TestCase):
    """A7: empty/short stored payloads must not read back as hits."""

    def test_exists_requires_a_nonempty_file(self):
        h = _bare_nixl(self)
        self.assertEqual(
            h._query_keys_exist(["empty", "short", "missing", "full"]),
            [False, True, False, True],
            "a zero-byte FILE object is a failed write, not a hit",
        )

    def test_obj_existence_semantics_unchanged(self):
        h = _bare_nixl(self, mem_type="OBJ", entries={"a": {"size": 0}})
        self.assertEqual(h._query_keys_exist(["a", "b"]), [True, False])

    def test_read_validates_size_before_starting_the_transfer(self):
        h = _bare_nixl(self)
        ok = h._xfer_pre_registered([(0x1000, 64)], ["/base/short"], "READ")
        self.assertFalse(ok, "a 32-byte object cannot fill a 64-byte slot")
        self.assertEqual(
            h.agent.transfers_started,
            0,
            "the short read must degrade before any NIXL transfer starts",
        )

    def test_read_rejects_empty_and_missing(self):
        h = _bare_nixl(self)
        self.assertFalse(h._xfer_pre_registered([(0x1000, 64)], ["/base/empty"], "READ"))
        h = _bare_nixl(self)
        self.assertFalse(h._xfer_pre_registered([(0x1000, 64)], ["/base/gone"], "READ"))

    def test_metadata_query_failure_degrades_to_failed_read(self):
        h = _bare_nixl(self, query_raises=True)
        self.assertFalse(
            h._xfer_pre_registered([(0x1000, 64)], ["/base/any"], "READ"),
            "a metadata query error must take the failed-read channel, not raise "
            "into the prefetch worker thread",
        )

    def test_full_size_read_proceeds(self):
        h = _bare_nixl(self)
        self.assertTrue(h._xfer_pre_registered([(0x1000, 64)], ["/base/full"], "READ"))
        self.assertEqual(h.agent.transfers_started, 1)
        # An exact-size buffer against a larger stored object also proceeds.
        self.assertTrue(h._xfer_pre_registered([(0x1000, 32)], ["/base/full"], "READ"))

    def test_writes_are_not_size_checked(self):
        h = _bare_nixl(self, query_raises=True)
        self.assertTrue(
            h._xfer_pre_registered([(0x1000, 64)], ["/base/w"], "WRITE"),
            "the pre-read query belongs to READ only; WRITE must not query",
        )


class _SidecarHostPool:
    page_size = 64

    def __init__(self, numel=8):
        self._numel = numel
        self.copied_out = 0
        self.restored = []

    def get_data_page(self, offset, flat=False):
        self.copied_out += 1
        return torch.zeros(self._numel, dtype=torch.uint8)

    def set_from_flat_data_page(self, offset, page):
        self.restored.append(int(offset))


class TestBoundedSidecarBatches(unittest.TestCase):
    """A8a: v2 bounce transfers must be chunked to the staging capacity."""

    PAGES = 300  # > two STORAGE_BATCH_SIZE windows

    def _setup(self, h):
        host_pool = _SidecarHostPool()
        page_bytes = 8
        bounce = torch.zeros(128, page_bytes, dtype=torch.uint8)
        h._hybrid_pool_ctx = {
            PoolName.MAMBA: _HybridPoolContext(
                host_pool=host_pool,
                is_zero_copy=False,
                bounce_set=bounce,
                bounce_get=bounce.clone(),
                bounce_page_bytes=page_bytes,
            )
        }
        h.registered_pools = {PoolName.MAMBA: host_pool}
        chunks = []

        def spy_xfer(keys, key_strs, host_buffers, direction):
            chunks.append(len(key_strs))
            return [True] * len(key_strs)

        h._batch_xfer = spy_xfer
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.arange(self.PAGES * host_pool.page_size),
            keys=[f"k{i}" for i in range(self.PAGES)],
        )
        return transfer, host_pool, chunks

    def test_set_is_chunked_to_capacity(self):
        h = _bare_nixl(self)
        transfer, _, chunks = self._setup(h)
        results = h.batch_set_v2([transfer])
        self.assertEqual(chunks, [128, 128, 44], chunks)
        self.assertEqual(results[PoolName.MAMBA], [True] * self.PAGES)

    def test_get_is_chunked_and_restores_every_page(self):
        h = _bare_nixl(self)
        transfer, host_pool, chunks = self._setup(h)
        results = h.batch_get_v2([transfer])
        self.assertEqual(chunks, [128, 128, 44], chunks)
        self.assertEqual(results[PoolName.MAMBA], [True] * self.PAGES)
        self.assertEqual(len(host_pool.restored), self.PAGES)

    def test_small_batch_untouched(self):
        h = _bare_nixl(self)
        transfer, _, chunks = self._setup(h)
        transfer = replace(transfer, keys=transfer.keys[:8], host_indices=torch.arange(8 * 64))
        results = h.batch_set_v2([transfer])
        self.assertEqual(chunks, [8])
        self.assertEqual(results[PoolName.MAMBA], [True] * 8)

    def test_bounded_iterator_exists(self):
        h = _bare_nixl(self)
        self.assertTrue(
            hasattr(h, "_iter_bounded_pool_transfers"),
            "bounce-backed v2 transfers need a capacity-bounded iterator",
        )


if __name__ == "__main__":
    unittest.main()
