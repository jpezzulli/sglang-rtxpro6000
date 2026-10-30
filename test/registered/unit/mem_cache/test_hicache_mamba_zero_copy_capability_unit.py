"""Mamba side-state vs NIXL zero-copy capability selection (CPU).

Full-model runtime on buffered POSIX (`use_direct_io=false`) reached
`MambaPoolHost.get_page_buffer_meta` and died on its fail-loud guard:
"Mamba storage zero-copy does not carry registered slot side states; use a
whole-page storage backend." The host pool declares that inability in two
places -- that guard and `is_stride_page_aligned` ("no zero-copy claim is
made") -- but the NIXL selector only consulted the alignment hook UNDER
O_DIRECT (`needs_page_alignment`), so buffered I/O picked the zero-copy
representation the host explicitly cannot export, and L3 backup crashed.

Pins, against the REAL host pool (temporal + conv + heterogeneous registered
slot side states, per test_hicache_mamba_slot_side_states.py) and the REAL
plugin selection/bounce/restore path (fake agent only at the NIXL boundary):

* a side-state mamba host pool is never zero-copy-selected (the crash site
  is the red), and round-trips WHOLE pages -- temporal, conv and EVERY side
  state byte -- through the bounce path;
* compatible plain pools keep zero-copy selection (no blanket disabling);
* the O_DIRECT alignment fallback stays a separate axis (aligned-capable
  pools can still be refused on alignment, and side-state pools are
  refused on capability regardless of alignment mode);
* the host's fail-loud version guard on truncated pages is preserved.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import contextlib
import sys
import types
import unittest
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
            useProgThread=False, useListenThread=False, listenPort=0, syncMode=0,
            pthrDelay=0, lthrDelay=0, captureTelemetry=False,
        )
        nixlAgent = _Agent

    api.nixl_agent = _Agent
    api.nixl_agent_config = lambda *a, **k: SimpleNamespace(
        enable_pthread=False, enable_listen=False, port=0, capture_telemetry=False
    )
    api.nixlBind = _Bind
    nixl._api = api
    sys.modules["nixl"] = nixl
    sys.modules["nixl._api"] = api


_install_nixl_stub()

from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost  # noqa: E402
from sglang.srt.mem_cache.storage.nixl.hicache_nixl import HiCacheNixl  # noqa: E402

NUM_LAYERS = 2
NUM_HOST_SLOTS = 4
CONV_SHAPE = (3, 2)
TEMPORAL_SHAPE = (2, 2)
SIDE_CONV_SHAPE = (5, 3)
NGRAM_WIDTH = 2


class _RegisteredSideState:
    """Ple ShortConvPool / NGramPool stand-in (transfer-entry protocol only)."""

    def __init__(self, name, tensor):
        self.name = name
        self.tensor = tensor

    def iter_transfer_state_entries(self):
        yield self.name, self.tensor, None, 0


def _make_host_pool(with_side_state: bool) -> MambaPoolHost:
    conv = [
        torch.zeros((NUM_LAYERS, NUM_HOST_SLOTS) + CONV_SHAPE, dtype=torch.bfloat16)
    ]
    temporal = torch.zeros(
        (NUM_LAYERS, NUM_HOST_SLOTS) + TEMPORAL_SHAPE, dtype=torch.bfloat16
    )
    siblings = ()
    if with_side_state:
        siblings = (
            _RegisteredSideState(
                "ple_short_conv",
                torch.zeros(
                    (NUM_HOST_SLOTS,) + SIDE_CONV_SHAPE, dtype=torch.bfloat16
                ),
            ),
            _RegisteredSideState(
                "ple_ngram",
                torch.zeros((NUM_HOST_SLOTS, NGRAM_WIDTH), dtype=torch.int64),
            ),
        )
    device_pool = SimpleNamespace(
        mamba_cache=SimpleNamespace(conv=conv, temporal=temporal),
        num_mamba_layers=NUM_LAYERS,
        size=NUM_HOST_SLOTS,
        device="cpu",
        _slot_siblings=siblings,
    )
    host = MambaPoolHost.__new__(MambaPoolHost)
    host.device_pool = device_pool
    host.buffer_info = MambaStateBufferInfo(
        temporal=device_pool.mamba_cache.temporal,
        conv=tuple(device_pool.mamba_cache.conv),
    )
    host.page_size = 1
    host.layout = "page_first"
    host.device = "cpu"
    host.pin_memory = False
    host.num_mamba_layers = NUM_LAYERS
    host.size = NUM_HOST_SLOTS
    host.conv_state_shapes = [c.shape[2:] for c in conv]
    host.temporal_state_shape = temporal.shape[2:]
    host.temporal_state_elem_size = int(temporal[0, 0].numel())
    host.conv_state_elem_sizes = [int(c[0, 0].numel()) for c in conv]
    host.conv_dtype = conv[0].dtype
    host.temporal_dtype = temporal.dtype
    host.dtype = host.conv_dtype
    host.slot_state_device_tensors = [
        state
        for sibling in device_pool._slot_siblings
        for _, state, _, _ in sibling.iter_transfer_state_entries()
    ]
    host.slot_state_buffers = [
        torch.zeros((NUM_HOST_SLOTS,) + tuple(state.shape[1:]), dtype=state.dtype)
        for state in host.slot_state_device_tensors
    ]
    host.temporal_buffer = torch.zeros(
        (NUM_HOST_SLOTS, NUM_LAYERS, 1) + TEMPORAL_SHAPE, dtype=temporal.dtype
    )
    host.conv_buffer = [
        torch.zeros((NUM_HOST_SLOTS, NUM_LAYERS, 1) + CONV_SHAPE, dtype=conv[0].dtype)
    ]
    host.temporal_device_ptrs = torch.zeros(NUM_LAYERS, dtype=torch.uint64)
    host.conv_device_ptrs = [torch.zeros(NUM_LAYERS, dtype=torch.uint64) for _ in conv]
    host.size_per_token = host.get_size_per_token()
    return host


class _FakeAgent:
    def __init__(self):
        self.registrations = 0

    def get_reg_descs(self, descs, mem_type):
        return ("reg", tuple(descs))

    def register_memory(self, descs):
        self.registrations += 1
        return f"reg-{self.registrations}"

    def deregister_memory(self, reg):
        pass

    def query_memory(self, descs, backend, mem_type=None):
        return [{"size": 1 << 20} for _ in descs]

    def get_xfer_descs(self, descs, mem_type):
        return ("host", tuple(descs))

    def initialize_xfer(self, direction, host_descs, storage_descs, agent):
        return (direction, tuple(tuple(h) for h in host_descs))

    def transfer(self, handle):
        return "DONE"

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        pass


@contextlib.contextmanager
def _registry_storage():
    yield ("storage",)


def _bare_plugin(mem_type="FILE", needs_page_alignment=False):
    h = HiCacheNixl.__new__(HiCacheNixl)
    h.backend_selector = SimpleNamespace(mem_type=mem_type, backend_name="POSIX")
    h.file_manager = SimpleNamespace(get_file_path=lambda key: f"/base/{key}")
    h.agent = _FakeAgent()
    h.registry = SimpleNamespace(storage=lambda *a: _registry_storage())
    h.agent_name = "fake-agent"
    h.needs_page_alignment = needs_page_alignment
    h.is_zero_copy = False
    h.is_mla_model = False
    h.backup_skip = False
    h.config_suffix = "_model_0_1"
    h.tp_config_suffix = "_model_0_1"
    h._host_regs = []
    h._hybrid_pool_ctx = {}
    h.registered_pools = {}
    h._l3_cleaner = None
    return h


class TestMambaSideStateCapability(unittest.TestCase):
    def test_plain_mamba_stays_zero_copy_capable(self):
        host = _make_host_pool(False)
        self.assertTrue(host.supports_storage_zero_copy())
        # The tiny fixture tensors genuinely have sub-4KiB strides, so the
        # alignment hook honestly answers False here; capability is the
        # independent axis this item introduces (both axes are pinned
        # separately in test_alignment_axis_is_independent_of_capability).

    def test_side_state_mamba_declares_no_zero_copy(self):
        """Red on v3: the host refuses in get_page_buffer_meta / reports
        is_stride_page_aligned False under O_DIRECT, but the buffered-mode
        selector asked neither question and crashed at backup time."""
        host = _make_host_pool(True)
        self.assertFalse(host.supports_storage_zero_copy())
        self.assertFalse(host.is_stride_page_aligned())  # its own declaration
        h = _bare_plugin()
        with self.assertRaises(NotImplementedError):
            host.get_page_buffer_meta(torch.tensor([2]))  # fail-loud guard kept
        self.assertFalse(
            h._hybrid_pool_supports_zero_copy(host, PoolName.MAMBA),
            "buffered NIXL must not zero-copy-select a host pool that cannot "
            "export every side-state byte through direct descriptors",
        )

    def test_registration_selects_bounce_and_round_trips_the_whole_page(self):
        host = _make_host_pool(True)
        h = _bare_plugin()
        h.register_mem_host_pool_v2(host, PoolName.MAMBA)
        ctx = h._hybrid_pool_ctx[PoolName.MAMBA]
        self.assertFalse(ctx.is_zero_copy)
        expected_page_bytes = host.page_size * host.size_per_token
        self.assertEqual(ctx.bounce_page_bytes, expected_page_bytes)

        # Sentinel every component of slot 2: temporal, conv, and BOTH side
        # states (heterogeneous dtypes; int64 lands at a bf16-sized offset).
        host.temporal_buffer[2].fill_(0.25)
        host.conv_buffer[0][2].fill_(-0.5)
        host.slot_state_buffers[0][2].fill_(1.5)
        host.slot_state_buffers[1][2] = torch.arange(
            NGRAM_WIDTH, dtype=torch.int64
        ).expand(NGRAM_WIDTH)
        want_page = host.get_data_page(2, flat=True).clone()

        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([2], dtype=torch.int64),
            keys=["page0"],
        )
        disk = {}

        def spy_xfer(keys, key_strs, host_buffers, direction):
            self.assertEqual(len(keys), 1, "one WHOLE-page object per slot")
            for key, (addr, size) in zip(key_strs, host_buffers):
                self.assertEqual(size, expected_page_bytes)
                if direction == "WRITE":
                    disk[key] = (
                        ctx.bounce_set[keys.index(key)]
                        .view(torch.uint8)[:size]
                        .clone()
                    )
                else:
                    row = ctx.bounce_get[keys.index(key)]
                    row.view(torch.uint8)[:size].copy_(disk[key])
            return [True] * len(keys)

        h._batch_xfer = spy_xfer
        set_results = h.batch_set_v2([transfer])
        self.assertTrue(all(set_results[PoolName.MAMBA]))
        self.assertEqual(len(disk), 1)
        self.assertTrue(
            torch.equal(next(iter(disk.values())), want_page),
            "the stored page must carry temporal, conv and EVERY side-state "
            "byte (get_data_page is the whole-page contract)",
        )

        # Poison the slot, restore via batch_get_v2, expect byte-exact.
        host.temporal_buffer[2].fill_(9.0)
        host.conv_buffer[0][2].fill_(9.0)
        host.slot_state_buffers[0][2].fill_(9.0)
        host.slot_state_buffers[1][2].fill_(9)
        get_results = h.batch_get_v2([transfer])
        self.assertTrue(all(get_results[PoolName.MAMBA]))
        self.assertTrue(torch.equal(host.get_data_page(2, flat=True), want_page))
        self.assertEqual(host.slot_state_buffers[1][2].tolist(), [0, 1])

    def test_truncated_side_state_page_still_fails_loud(self):
        host = _make_host_pool(True)
        short = torch.zeros(
            host.page_size * host.size_per_token - 1, dtype=torch.uint8
        )
        with self.assertRaises(ValueError):
            host.set_from_flat_data_page(2, short)

    def test_alignment_axis_is_independent_of_capability(self):
        """O_DIRECT refusal still works for a complete-state pool, and
        buffered mode does not refuse on alignment grounds."""
        host = _make_host_pool(False)
        direct = _bare_plugin(needs_page_alignment=True)
        host.is_stride_page_aligned = lambda page_size_bytes=4096: False
        self.assertFalse(direct._hybrid_pool_supports_zero_copy(host, PoolName.MAMBA))
        host.is_stride_page_aligned = lambda page_size_bytes=4096: True
        self.assertTrue(direct._hybrid_pool_supports_zero_copy(host, PoolName.MAMBA))
        buffered = _bare_plugin(needs_page_alignment=False)
        self.assertTrue(buffered._hybrid_pool_supports_zero_copy(host, PoolName.MAMBA))

    def test_plain_mamba_registration_keeps_zero_copy(self):
        host = _make_host_pool(False)
        h = _bare_plugin()
        h.register_mem_host_pool_v2(host, PoolName.MAMBA)
        ctx = h._hybrid_pool_ctx[PoolName.MAMBA]
        self.assertTrue(
            ctx.is_zero_copy,
            "compatible plain pools must keep zero-copy (no blanket disabling)",
        )
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.arange(2, dtype=torch.int64),
            keys=["a", "b"],
        )
        _, key_strs, host_buffers, _, multiplier = h._prepare_pool_transfer(
            transfer, for_write=True
        )
        # temporal + 1 conv per page, TP-suffixed component keys.
        self.assertEqual(multiplier, 2)
        self.assertEqual(len(key_strs), 4)
        self.assertEqual(len(host_buffers), 4)


if __name__ == "__main__":
    unittest.main()
