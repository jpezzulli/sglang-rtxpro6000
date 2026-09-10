"""Focused coverage for pinned host staging in ``ForwardBatch.init_new``."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import call, patch

import torch

import sglang.srt.model_executor.forward_batch_info as forward_batch_info
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _batch(*, dllm: bool = True):
    reqs = [
        SimpleNamespace(
            rid="r0",
            lora_id=None,
            token_type_ids=None,
            dllm_block_offset=10,
        ),
        SimpleNamespace(
            rid="r1",
            lora_id=None,
            token_type_ids=None,
            dllm_block_offset=20,
        ),
    ]
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.tensor([11, 12, 13, 14, 15, 16]),
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([13, 23]),
        out_cache_loc=torch.arange(6),
        seq_lens_sum=36,
        seq_lens_cpu=torch.tensor([13, 23]),
        orig_seq_lens=None,
        out_cache_loc_dsv4=None,
        mamba_track_indices=None,
        mamba_track_mask=None,
        mamba_track_seqlens=None,
        mamba_cow_src_indices=None,
        mamba_cow_dst_indices=None,
        mamba_clear_indices=None,
        encoder_lens=None,
        encoder_out_cache_loc=None,
        input_embeds=None,
        replace_embeds=None,
        replace_positions=None,
        return_logprob=False,
        is_extend_in_batch=True,
        can_run_decode_cuda_graph=False,
        can_run_dp_prefill_cuda_graph=False,
        global_forward_mode=ForwardMode.EXTEND,
        is_prefill_only=False,
        spec_algorithm=None,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        tbo_split_seq_index=None,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        multimodal_inputs=None,
        encoder_cached=None,
        encoder_lens_cpu=None,
        sampling_info=None,
        spec_info=None,
        reqs=reqs,
        has_grammar=False,
        extend_input_logprob_token_ids=None,
        extend_lens=[3, 3],
        prefix_lens=[10, 20],
        extend_logprob_start_lens=[0, 0],
        extend_num_tokens=6,
        global_num_tokens=[2, 3],
        global_num_tokens_for_logprob=[1, 2],
        dllm_config=SimpleNamespace(block_size=2) if dllm else None,
    )


def _model_runner(device):
    return SimpleNamespace(
        device=device,
        is_draft_worker=False,
        prefill_attention_backend_str="torch_native",
        ngram_embedding_manager=SimpleNamespace(enabled=False),
        model_config=SimpleNamespace(model_is_mrope=False),
        lora_manager=None,
        ps=SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0),
    )


def _init_forward_batch(batch, device):
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                forward_batch_info.envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE,
                "get",
                return_value=False,
            )
        )
        stack.enter_context(
            patch.object(
                forward_batch_info.envs.SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT,
                "get",
                return_value=False,
            )
        )
        stack.enter_context(
            patch.object(
                forward_batch_info, "enable_num_token_non_padded", return_value=True
            )
        )
        stack.enter_context(
            patch.object(
                forward_batch_info,
                "compute_position",
                return_value=(None, None),
            )
        )
        return ForwardBatch.init_new(
            batch,
            _model_runner(device),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )


class _StagedTensor:
    def __init__(self, value, kwargs):
        self.value = value
        self.kwargs = kwargs
        self.to_calls = []

    def to(self, device, *, non_blocking=False):
        self.to_calls.append((device, non_blocking))
        return self


class TestForwardBatchPinnedMetadata(CustomTestCase):
    def test_all_host_metadata_uses_platform_guarded_pinned_staging(self):
        batch = _batch()
        staged = []

        def fake_tensor(value, **kwargs):
            tensor = _StagedTensor(value, kwargs)
            staged.append(tensor)
            return tensor

        with (
            patch.object(forward_batch_info.torch, "tensor", side_effect=fake_tensor),
            patch.object(
                forward_batch_info, "is_pin_memory_available", return_value=True
            ) as pin_available,
        ):
            result = _init_forward_batch(batch, "cuda")

        expected = [
            (6, torch.int32),
            ([2, 3], torch.int64),
            ([1, 2], torch.int64),
            ([10, 11, 20, 21], torch.int32),
            ([3, 3], torch.int32),
            ([10, 20], torch.int32),
        ]
        self.assertEqual(len(staged), len(expected))
        for tensor, (value, dtype) in zip(staged, expected):
            self.assertEqual(tensor.value, value)
            self.assertIs(tensor.kwargs["dtype"], dtype)
            self.assertIs(tensor.kwargs["pin_memory"], True)
            self.assertEqual(tensor.to_calls, [("cuda", True)])
        self.assertEqual(pin_available.call_count, len(expected))
        pin_available.assert_has_calls([call("cuda")] * len(expected))
        self.assertIs(result.num_token_non_padded, staged[0])
        self.assertIs(result.global_num_tokens_gpu, staged[1])
        self.assertIs(result.global_num_tokens_for_logprob_gpu, staged[2])
        self.assertIs(result.positions, staged[3])
        self.assertIs(result.extend_seq_lens, staged[4])
        self.assertIs(result.extend_prefix_lens, staged[5])

    def test_unavailable_pinned_memory_falls_back_without_changing_metadata(self):
        batch = SimpleNamespace(
            global_num_tokens=[4, 7],
            global_num_tokens_for_logprob=[3, 5],
            can_run_decode_cuda_graph=True,
        )
        result = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            input_ids=torch.arange(11),
            req_pool_indices=torch.tensor([0, 1]),
            seq_lens=torch.tensor([4, 7]),
            out_cache_loc=torch.arange(11),
            seq_lens_sum=11,
        )
        real_tensor = torch.tensor
        with (
            patch.object(
                forward_batch_info, "is_pin_memory_available", return_value=False
            ) as pin_available,
            patch.object(
                forward_batch_info.torch, "tensor", wraps=real_tensor
            ) as tensor_constructor,
        ):
            result.init_mlp_sync_metadata(batch, torch.device("cpu"))

        self.assertEqual(pin_available.call_count, 2)
        pin_available.assert_has_calls([call(torch.device("cpu"))] * 2)
        for constructor_call in tensor_constructor.call_args_list:
            self.assertIs(constructor_call.kwargs["pin_memory"], False)
            self.assertIs(constructor_call.kwargs["dtype"], torch.int64)
        torch.testing.assert_close(
            result.global_num_tokens_gpu, real_tensor([4, 7], dtype=torch.int64)
        )
        torch.testing.assert_close(
            result.global_num_tokens_for_logprob_gpu,
            real_tensor([3, 5], dtype=torch.int64),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA pinned memory")
    def test_async_staging_survives_temporary_source_lifetime(self):
        if not forward_batch_info.is_pin_memory_available("cuda"):
            self.skipTest("platform does not support pinned host staging")

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # Delay the H2D copies so the temporary source tensors have left
            # ForwardBatch.init_new before the device consumes their values.
            torch.cuda._sleep(5_000_000)
            result = _init_forward_batch(_batch(), torch.device("cuda"))

        # Exercise the pinned allocator while the copies remain queued. Correct
        # allocator lifetime tracking must not recycle their pending storage.
        for _ in range(256):
            churn = torch.empty(16, dtype=torch.int64, pin_memory=True)
            churn.fill_(-1)
            del churn
        stream.synchronize()

        self.assertEqual(result.num_token_non_padded.dtype, torch.int32)
        self.assertEqual(result.num_token_non_padded.item(), 6)
        torch.testing.assert_close(
            result.global_num_tokens_gpu,
            torch.tensor([2, 3], dtype=torch.int64, device="cuda"),
        )
        torch.testing.assert_close(
            result.global_num_tokens_for_logprob_gpu,
            torch.tensor([1, 2], dtype=torch.int64, device="cuda"),
        )
        torch.testing.assert_close(
            result.positions,
            torch.tensor([10, 11, 20, 21], dtype=torch.int32, device="cuda"),
        )
        torch.testing.assert_close(
            result.extend_seq_lens,
            torch.tensor([3, 3], dtype=torch.int32, device="cuda"),
        )
        torch.testing.assert_close(
            result.extend_prefix_lens,
            torch.tensor([10, 20], dtype=torch.int32, device="cuda"),
        )


if __name__ == "__main__":
    unittest.main()
