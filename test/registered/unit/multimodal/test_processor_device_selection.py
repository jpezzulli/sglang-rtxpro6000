"""The fast-image-processor device comes from the processor's own ServerArgs.

Regression: the device decision read the published global ServerArgs, so every
processor answered with one process-wide device. The encode-server DP workers
each drive their own GPU, which no process-global value can express — the
device has to come from what the worker was handed.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BASE = "sglang.srt.multimodal.processors.base_processor"


class _Processor:
    pass


class _StubProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


def _make(**fields):
    processor = _StubProcessor.__new__(_StubProcessor)
    processor.server_args = ServerArgs(model_path="dummy", **fields)
    return processor


class TestFastImageProcessorDevice(CustomTestCase):
    def _construct(
        self,
        *,
        preprocess_device,
        configured_backend="torchvision",
        actual_backend="torchvision",
        transport="cpu",
        base_gpu_id=0,
        rl_on_policy_target=None,
    ):
        mm = SimpleNamespace(
            allowed_media_domains=None,
            media_url_max_file_size_mb=10,
            mm_feature_transport=transport,
            image_processor_backend=configured_backend,
            disable_fast_image_processor=False,
            mm_process_config={},
            mm_preprocess_cache_size_mb=0,
            trust_mm_content_hashes=False,
            mm_io_worker_num=1,
            mm_processor_worker_num=1,
        )
        serving = SimpleNamespace(skip_tokenizer_init=False, tokenizer_worker_num=1)
        wrapped = SimpleNamespace(
            image_processor=SimpleNamespace(backend=actual_backend),
            tokenizer=SimpleNamespace(encode=lambda _: []),
        )
        server_args = SimpleNamespace(
            base_gpu_id=base_gpu_id,
            tp_size=1,
            rl_on_policy_target=rl_on_policy_target,
        )
        thread_executor = Mock()
        process_executor = Mock()
        with (
            patch(f"{BASE}.get_mm", return_value=mm),
            patch(f"{BASE}.get_serving", return_value=serving),
            patch(
                f"{BASE}.resolve_mm_preprocess_device",
                return_value=preprocess_device,
            ),
            patch(f"{BASE}.concurrent.futures.ThreadPoolExecutor", thread_executor),
            patch(f"{BASE}.concurrent.futures.ProcessPoolExecutor", process_executor),
            patch(f"{BASE}.MmItemMemoryPool") as mm_pool,
        ):
            try:
                processor = _StubProcessor(
                    SimpleNamespace(), server_args, wrapped, transport_mode=None
                )
            except Exception:
                self.assertFalse(thread_executor.called)
                self.assertFalse(process_executor.called)
                self.assertFalse(mm_pool.called)
                raise
        return processor

    def _device(self, processor, **platform):
        flags = {"_is_cpu": False, "_is_xpu": False, "_is_npu": False}
        flags.update(platform)
        with patch.multiple(BASE, **flags):
            return processor._fast_image_processor_device(_Processor())

    def test_device_follows_the_instance_base_gpu_id(self):
        self.assertEqual(self._device(_make(base_gpu_id=3)), "cuda:3")

    def test_engines_in_one_process_keep_their_own_device(self):
        first, second = _make(base_gpu_id=0), _make(base_gpu_id=5)
        self.assertEqual(self._device(first), "cuda:0")
        self.assertEqual(self._device(second), "cuda:5")

    def test_publishing_another_config_does_not_move_the_device(self):
        from sglang.srt.runtime_context import get_context

        processor = _make(base_gpu_id=2)
        override = get_context().override_server_args(base_gpu_id=7)
        override.install()
        self.addCleanup(override.restore)
        self.assertEqual(self._device(processor), "cuda:2")

    def test_rl_on_policy_target_forces_cpu(self):
        processor = _make(base_gpu_id=3, rl_on_policy_target="fsdp")
        self.assertEqual(self._device(processor), "cpu")

    def test_explicit_cuda_conflicting_with_rl_cpu_policy_fails(self):
        with (
            patch.dict(
                "os.environ", {"SGLANG_MM_PREPROCESS_DEVICE": "cuda:1"}, clear=False
            ),
            patch(f"{BASE}.torch.cuda.is_available", return_value=True),
            patch(f"{BASE}.torch.cuda.device_count", return_value=2),
            self.assertRaisesRegex(ValueError, "rl_on_policy_target"),
        ):
            self._device(_make(base_gpu_id=3, rl_on_policy_target="fsdp"))

    def test_constructor_rejects_explicit_cuda_with_configured_pil(self):
        with self.assertRaisesRegex(ValueError, "PIL image processor backend"):
            self._construct(preprocess_device="cuda:0", configured_backend="pil")

    def test_constructor_rejects_explicit_cuda_with_auto_resolved_pil(self):
        with self.assertRaisesRegex(ValueError, "PIL image processor backend"):
            self._construct(
                preprocess_device="cuda:0",
                configured_backend="auto",
                actual_backend="pil",
            )

    def test_constructor_rejects_cross_gpu_device_transport(self):
        for transport in ("cuda_ipc", "cuda_vmm"):
            with (
                self.subTest(transport=transport),
                self.assertRaisesRegex(ValueError, "different from base_gpu_id"),
            ):
                self._construct(
                    preprocess_device="cuda:1",
                    transport=transport,
                    base_gpu_id=0,
                )

    def test_constructor_rejects_explicit_cuda_with_rl_cpu_policy(self):
        with self.assertRaisesRegex(ValueError, "rl_on_policy_target"):
            self._construct(preprocess_device="cuda:0", rl_on_policy_target="fsdp")

    def test_unset_auto_backend_preserves_baseline_for_known_backends(self):
        for actual_backend in ("pil", "torchvision"):
            with self.subTest(actual_backend=actual_backend):
                processor = self._construct(
                    preprocess_device=None,
                    configured_backend="auto",
                    actual_backend=actual_backend,
                )
                self.assertEqual(processor.image_processor_backend, "auto")
                self.assertFalse(processor.disable_fast_image_processor)
                self.assertNotIn(
                    "mm_preprocess_device",
                    processor.preprocess_fingerprint_payload(),
                )
                self.assertEqual(self._device(processor), "cuda:0")

    def test_explicit_preprocess_device_overrides_model_gpu(self):
        with (
            patch.dict(
                "os.environ", {"SGLANG_MM_PREPROCESS_DEVICE": "cuda:1"}, clear=False
            ),
            patch(f"{BASE}.torch.cuda.is_available", return_value=True),
            patch(f"{BASE}.torch.cuda.device_count", return_value=2),
        ):
            self.assertEqual(self._device(_make(base_gpu_id=3)), "cuda:1")

    def test_explicit_cpu_preprocess_device(self):
        with patch.dict(
            "os.environ", {"SGLANG_MM_PREPROCESS_DEVICE": "cpu"}, clear=False
        ):
            self.assertEqual(self._device(_make(base_gpu_id=3)), "cpu")

    def test_cpu_and_xpu_platforms_win_over_base_gpu_id(self):
        processor = _make(base_gpu_id=3)
        self.assertEqual(self._device(processor, _is_cpu=True), "cpu")
        self.assertEqual(self._device(processor, _is_xpu=True), "xpu")

    def test_npu_glm4v_leaves_the_device_unset(self):
        class Glm4vProcessor:
            pass

        processor = _make(base_gpu_id=3)
        with patch.multiple(BASE, _is_cpu=False, _is_xpu=False, _is_npu=True):
            device = processor._fast_image_processor_device(Glm4vProcessor())
        self.assertIsNone(device)

    def test_preprocess_fingerprint_tracks_effective_device_and_decode_policy(self):
        class WrappedProcessor:
            pass

        def payload(device, decode_mode):
            processor = _make(base_gpu_id=0)
            processor._processor = WrappedProcessor()
            processor.mm_preprocess_device = device
            processor.gpu_image_decode = decode_mode
            processor.image_processor_backend = "torchvision"
            processor.mm_feature_transport = "cpu"
            processor.image_config = {}
            processor.video_config = {}
            processor.audio_config = {}
            return processor.preprocess_fingerprint_payload()

        for decode_mode in (True, "nvjpeg_fancy"):
            with self.subTest(decode_mode=decode_mode):
                automatic = payload(None, decode_mode)
                cpu = payload("cpu", decode_mode)
                cuda = payload("cuda:1", decode_mode)
                self.assertNotIn("mm_preprocess_device", automatic)
                self.assertEqual(cpu["gpu_image_decode"], False)
                self.assertEqual(cuda["gpu_image_decode"], decode_mode)
                self.assertNotEqual(automatic, cpu)
                self.assertNotEqual(cpu, cuda)


class TestFastImageProcessorMemoryPool(CustomTestCase):
    def _processor(self, *, transport="cpu", precompute_hash=False):
        processor = _make(base_gpu_id=0)
        processor.mm_feature_transport = transport
        processor.precompute_hash_before_cpu_transfer = precompute_hash
        return processor

    def test_pool_is_limited_to_immediate_cpu_transport(self):
        cases = (
            (self._processor(), "cuda:0", True),
            (self._processor(transport="cuda_ipc"), "cuda:0", False),
            (self._processor(transport="cuda_vmm"), "cuda:0", False),
            (self._processor(precompute_hash=True), "cuda:0", False),
            (self._processor(), "cpu", False),
            (self._processor(), None, False),
        )
        for processor, device, expected in cases:
            with (
                self.subTest(device=device, transport=processor.mm_feature_transport),
                patch(f"{BASE}.torch.cuda.device", return_value=nullcontext()),
                patch(f"{BASE}.torch.cuda.MemPool", return_value="pool") as mem_pool,
                patch(f"{BASE}.torch.cuda.use_mem_pool", return_value=nullcontext()),
            ):
                with processor._temporary_fast_processor_cuda_pool(device):
                    pass
                self.assertEqual(mem_pool.called, expected)

    def test_processor_call_uses_private_pool_until_cpu_copy_finishes(self):
        class ImageProcessor:
            pass

        class Feature:
            def to(self, device):
                events.append(("copy", device))

        feature = Feature()

        class Processor:
            image_processor = ImageProcessor()
            tokenizer = SimpleNamespace(bos_token=None)

            def __call__(self, **kwargs):
                events.append(("call", kwargs["device"]))
                return {"pixel_values": feature}

        events = []
        processor = self._processor()
        processor._processor = Processor()
        processor._tokenizer = processor._processor.tokenizer
        processor._tokenizer_auto_adds_specials = False
        processor.disable_fast_image_processor = False
        processor.image_config = {}
        processor.video_config = {}
        processor.audio_config = {}
        processor.FEATURE_NAMES = ["pixel_values"]

        class PoolContext:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *args):
                events.append("exit")

        with (
            patch(f"{BASE}.BaseImageProcessor", ImageProcessor),
            patch(f"{BASE}.torch.cuda.device", return_value=nullcontext()),
            patch(f"{BASE}.torch.cuda.MemPool", return_value="pool"),
            patch(f"{BASE}.torch.cuda.use_mem_pool", return_value=PoolContext()),
            patch(f"{BASE}.torch.Tensor", Feature),
        ):
            processor.process_mm_data("test", images=["image"])

        self.assertEqual(
            events,
            ["enter", ("call", "cuda:0"), ("copy", "cpu"), "exit"],
        )


if __name__ == "__main__":
    unittest.main()
