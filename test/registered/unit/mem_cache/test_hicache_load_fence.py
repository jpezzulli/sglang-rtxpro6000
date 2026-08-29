"""CPU contract test for ordering HiCache load-back after model forwards."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHiCacheLoadFence(CustomTestCase):
    def test_start_loading_fences_transfer_stream_before_submission(self):
        controller = object.__new__(HiCacheController)
        op = CacheOperation(
            host_indices=torch.tensor([0], dtype=torch.int64),
            device_indices=torch.tensor([1], dtype=torch.int64),
            node_id=7,
        )
        controller.load_queue = [op]
        controller.ack_load_queue = []
        controller.layer_num = 1
        start_event = MagicMock()
        controller.layer_done_counter = SimpleNamespace(
            update_producer=MagicMock(return_value=0),
            events=[SimpleNamespace(start_event=start_event, complete=MagicMock())],
        )
        controller._move_op_indices = MagicMock(
            return_value=(op.host_indices, op.device_indices, None)
        )
        controller._l2_load_transfers = MagicMock(return_value=[])
        controller._num_tokens_by_pool = MagicMock(return_value={"kv": 1})
        controller._transfer_num_bytes = MagicMock(return_value=1)

        call_order = []
        transfer_stream = MagicMock()
        transfer_stream.wait_stream.side_effect = lambda stream: call_order.append(
            ("fence", stream)
        )
        completion = SimpleNamespace(
            start_event=MagicMock(),
            finish_event=MagicMock(),
            timing_enabled=False,
        )
        submit = MagicMock(
            side_effect=lambda *args, **kwargs: (
                call_order.append(("submit", None)) or completion
            )
        )
        controller.l2_transfer_engine = SimpleNamespace(
            host_to_device_stream=transfer_stream,
            submit_host_to_device=submit,
        )
        controller.load_fence_stream = object()

        producer_id = controller.start_loading()

        self.assertEqual(producer_id, 0)
        self.assertEqual(
            call_order,
            [("fence", controller.load_fence_stream), ("submit", None)],
        )
        transfer_stream.wait_stream.assert_called_once_with(
            controller.load_fence_stream
        )
        submit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
