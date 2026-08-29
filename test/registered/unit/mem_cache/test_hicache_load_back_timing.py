"""Unit tests for the HiCache load-back duration metric."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestLoadBackDurationMetric(CustomTestCase):
    def setUp(self):
        from sglang.srt.managers import cache_controller as cc
        from sglang.srt.mem_cache import l2_transfer as transfer

        transfer._timing_events_supported.cache_clear()
        self.cc = cc
        self.transfer = transfer

    def _completed_pair(self, payload_floats=1024 * 1024):
        start, finish, timing_enabled = self.transfer.make_timing_event_pair()
        self.assertTrue(timing_enabled)
        stream = torch.cuda.Stream()
        start.record()
        with torch.cuda.stream(stream):
            start.wait(stream)
            torch.empty(payload_floats, device="cuda").fill_(0)
            finish.record()
        torch.cuda.synchronize()
        return start, finish

    def test_elapsed_time_works(self):
        start, finish = self._completed_pair()
        self.assertGreater(start.elapsed_time(finish), 0.0)

    def test_timing_fallback_uses_dedicated_events(self):
        events = []

        def create_event(*, enable_timing=False):
            if enable_timing:
                raise TypeError
            event = MagicMock()
            events.append(event)
            return event

        with patch.object(
            self.transfer.device_module, "Event", side_effect=create_event
        ):
            self.transfer._timing_events_supported.cache_clear()
            start, finish, timing_enabled = self.transfer.make_timing_event_pair()

        self.assertFalse(timing_enabled)
        self.assertIs(start, events[0])
        self.assertIs(finish, events[1])
        self.assertIsNot(start, finish)

    def test_loading_check_observes_duration_and_tokens(self):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        start, finish = self._completed_pair()
        ack = self.cc.HiCacheAck(
            start,
            finish,
            node_ids=[1, 2],
            num_tokens=1024,
            timing_enabled=True,
            num_tokens_by_pool={"kv": 1024},
        )
        stub = object.__new__(HiRadixCache)
        stub.cache_controller = SimpleNamespace(ack_load_queue=[ack])
        stub.ongoing_load_back = {1: object(), 2: object()}
        stub.dec_lock_ref = MagicMock()
        stub.metrics_collector = MagicMock()
        stub.pp_rank = 0
        stub._all_reduce = MagicMock()

        stub.loading_check()

        stub.metrics_collector.increment_load_back_num_tokens.assert_called_once_with(
            num_tokens=1024, pool="kv"
        )
        stub.metrics_collector.observe_load_back_duration.assert_called_once()
        (observed,), _ = stub.metrics_collector.observe_load_back_duration.call_args
        self.assertGreater(observed, 0.0)
        self.assertEqual(stub.cache_controller.ack_load_queue, [])

    def test_loading_check_fallback_when_timing_unsupported(self):
        """On backends without enable_timing, count tokens but skip duration."""
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        start = torch.cuda.Event()
        finish = torch.cuda.Event()
        start.record()
        finish.record()
        torch.cuda.synchronize()

        ack = self.cc.HiCacheAck(
            start_event=start,
            finish_event=finish,
            node_ids=[7],
            num_tokens=512,
            timing_enabled=False,
            num_tokens_by_pool={"kv": 512},
        )
        stub = object.__new__(HiRadixCache)
        stub.cache_controller = SimpleNamespace(ack_load_queue=[ack])
        stub.ongoing_load_back = {7: object()}
        stub.dec_lock_ref = MagicMock()
        stub.metrics_collector = MagicMock()
        stub.pp_rank = 0
        stub._all_reduce = MagicMock()

        stub.loading_check()

        stub.metrics_collector.increment_load_back_num_tokens.assert_called_once_with(
            num_tokens=512, pool="kv"
        )
        stub.metrics_collector.observe_load_back_duration.assert_not_called()
        self.assertEqual(stub.cache_controller.ack_load_queue, [])

    def test_slot_siblings_restore_before_layer_release(self):
        calls = []

        class HostPool:
            layer_num = 2

            def load_slot_siblings_to_device(self, *args):
                calls.append("siblings")

            def load_to_device_per_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                *args,
                **kwargs,
            ):
                calls.append(f"layer-{layer_id}")

        transfer = self.transfer.L2Transfer(
            host_pool=HostPool(),
            device_pool=object(),
            host_indices=torch.tensor([0], dtype=torch.int64),
            device_indices=torch.tensor([1], dtype=torch.int64),
        )
        engine = self.transfer.L2TransferEngine("kernel")
        completion = engine.submit_host_to_device([transfer], layer_num=2)
        completion.finish_event.synchronize()

        self.assertEqual(calls, ["siblings", "layer-0", "layer-1"])

    def test_kernel_h2d_binds_tvm_ffi_to_transfer_stream(self):
        from tvm_ffi.core import _env_get_current_stream, _env_set_current_stream

        device_index = torch.cuda.current_device()
        default_stream = torch.cuda.default_stream()
        observed_streams = []

        class HostPool:
            layer_num = 1

            def load_to_device_per_layer(self, *args, **kwargs):
                observed_streams.append(int(_env_get_current_stream(2, device_index)))

        transfer = self.transfer.L2Transfer(
            host_pool=HostPool(),
            device_pool=object(),
            host_indices=torch.tensor([0], dtype=torch.int64),
            device_indices=torch.tensor([1], dtype=torch.int64),
        )
        engine = self.transfer.L2TransferEngine("kernel")
        try:
            _env_set_current_stream(2, device_index, default_stream.cuda_stream)
            completion = engine.submit_host_to_device([transfer], layer_num=1)
            completion.finish_event.synchronize()
            self.assertEqual(
                int(_env_get_current_stream(2, device_index)),
                int(default_stream.cuda_stream),
            )
        finally:
            _env_set_current_stream(2, device_index, default_stream.cuda_stream)

        self.assertEqual(
            observed_streams, [int(engine.host_to_device_stream.cuda_stream)]
        )

    def test_kernel_d2h_binds_tvm_ffi_to_transfer_stream(self):
        from tvm_ffi.core import _env_get_current_stream, _env_set_current_stream

        device_index = torch.cuda.current_device()
        default_stream = torch.cuda.default_stream()
        observed_streams = []

        class HostPool:
            def backup_from_device_all_layer(self, *args, **kwargs):
                observed_streams.append(int(_env_get_current_stream(2, device_index)))

        transfer = self.transfer.L2Transfer(
            host_pool=HostPool(),
            device_pool=object(),
            host_indices=torch.tensor([0], dtype=torch.int64),
            device_indices=torch.tensor([1], dtype=torch.int64),
        )
        engine = self.transfer.L2TransferEngine("kernel")
        try:
            _env_set_current_stream(2, device_index, default_stream.cuda_stream)
            completion = engine.submit_device_to_host([transfer])
            completion.finish_event.synchronize()
            self.assertEqual(
                int(_env_get_current_stream(2, device_index)),
                int(default_stream.cuda_stream),
            )
        finally:
            _env_set_current_stream(2, device_index, default_stream.cuda_stream)

        self.assertEqual(
            observed_streams, [int(engine.device_to_host_stream.cuda_stream)]
        )

    def test_load_fence_orders_restore_after_inflight_forward(self):
        from sglang.srt.managers.cache_controller import (
            CacheOperation,
            HiCacheController,
        )

        restored_page = torch.zeros(1, device="cuda")
        forward_stream = torch.cuda.Stream()
        with torch.cuda.stream(forward_stream):
            torch.cuda._sleep(50_000_000)
            restored_page.fill_(1)

        class HostPool:
            layer_num = 1

            def load_to_device_per_layer(self, *args, **kwargs):
                restored_page.fill_(2)

        op = CacheOperation(
            host_indices=torch.tensor([0], dtype=torch.int64),
            device_indices=torch.tensor([1], dtype=torch.int64),
            node_id=11,
        )
        transfer = self.transfer.L2Transfer(
            host_pool=HostPool(),
            device_pool=object(),
            host_indices=op.host_indices,
            device_indices=op.device_indices,
        )
        controller = object.__new__(HiCacheController)
        controller.load_queue = [op]
        controller.ack_load_queue = []
        controller.layer_num = 1
        controller.layer_done_counter = SimpleNamespace(
            update_producer=MagicMock(return_value=0),
            events=[
                SimpleNamespace(
                    start_event=torch.cuda.Event(), complete=lambda layer_id: None
                )
            ],
        )
        controller._move_op_indices = MagicMock(
            return_value=(op.host_indices, op.device_indices, None)
        )
        controller._l2_load_transfers = MagicMock(return_value=[transfer])
        controller._num_tokens_by_pool = MagicMock(return_value={"kv": 1})
        controller._transfer_num_bytes = MagicMock(return_value=restored_page.nbytes)
        controller.l2_transfer_engine = self.transfer.L2TransferEngine("direct")
        controller.load_fence_stream = forward_stream

        controller.start_loading()
        controller.ack_load_queue[0].finish_event.synchronize()

        self.assertEqual(restored_page.item(), 2)


if __name__ == "__main__":
    unittest.main()
