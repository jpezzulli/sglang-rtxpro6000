"""Image artifact reuse must not retain completed requests' prompt tensors."""

import gc
import re
import unittest
import weakref
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import (
    MultimodalProcessorOutput,
    _compute_pad_value,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.processors.qwen_vl import (
    QwenVLImagePreprocessArtifact,
    QwenVLImageProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _Processor(QwenVLImageProcessor):
    def __init__(self):
        self.mm_tokens = SimpleNamespace(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_regex=re.compile(
                r"<\|vision_start\|><\|image_pad\|><\|vision_end\|>"
            ),
            image_token_id=2,
            video_token_id=3,
            audio_token_id=4,
        )
        self.vision_start_token_id = 5
        self.vision_end_token_id = 6
        self.position_refs = []

    def _build_image_prompt_template(self, input_text, grid_key):
        # Isolate ownership from tokenizer/model dependencies. The real compose
        # path must release the builder's tensors after the output is released.
        positions = torch.arange(128).repeat(3, 1)
        self.position_refs.append(weakref.ref(positions))
        return (1, 2, 7), ((1, 1),), positions, torch.tensor([0])

    def _prepare_mm_items_for_transport(self, items):
        return items

    def _mark_dp_encoder_features_for_deferred_reconstruction(self, items):
        pass


def _artifact():
    return QwenVLImagePreprocessArtifact(
        content_digest="image",
        artifact_key="image",
        feature_hash=17,
        feature=torch.ones(4, 8),
        model_specific_data={"image_grid_thw": torch.tensor([[1, 2, 2]])},
    )


class TestQwenCallerHashes(CustomTestCase):
    """The composed fast-path prompt must follow a caller-supplied image hash."""

    def test_a_caller_hash_repads_the_composed_prompt(self):
        processor = _Processor()
        output = processor.compose_image_artifacts(
            processor.mm_tokens.image_token + " question", [_artifact()]
        )
        internal_pad = output.mm_items[0].pad_value
        # A KV router replaces the item hash before the request reaches the
        # scheduler, which would otherwise install the pad tokens composed here.
        TokenizerManager._apply_caller_mm_hashes(
            SimpleNamespace(mm_hashes=["12345"]), output
        )
        self.assertEqual(
            output.padded_input_ids,
            MultimodalProcessorOutput.build_padded_input_ids(
                output.input_ids, output.mm_items
            ),
        )
        self.assertEqual(output.mm_items[0].pad_value, _compute_pad_value(0x12345))
        self.assertNotIn(internal_pad, output.padded_input_ids)


class TestQwenPromptLifetime(CustomTestCase):
    def test_completed_prompts_release_position_tensors(self):
        processor = _Processor()
        artifact = _artifact()
        for index in range(8):
            output = processor.compose_image_artifacts(
                f"<|vision_start|><|image_pad|><|vision_end|> conversation turn {index}",
                [artifact],
            )
            self.assertIsNotNone(output)
            del output
        gc.collect()
        self.assertTrue(all(ref() is None for ref in processor.position_refs))

    def test_completed_prompt_does_not_keep_processor_alive(self):
        processor = _Processor()
        output = processor.compose_image_artifacts(
            "<|vision_start|><|image_pad|><|vision_end|> question", [_artifact()]
        )
        ref = weakref.ref(processor)
        del output, processor
        gc.collect()
        self.assertIsNone(ref())

    def test_repeated_prompt_outputs_are_independent(self):
        processor = _Processor()
        artifact = _artifact()
        first = processor.compose_image_artifacts(
            "<|vision_start|><|image_pad|><|vision_end|> question", [artifact]
        )
        expected = first.mrope_positions.clone()
        first.mrope_positions.fill_(-1)
        second = processor.compose_image_artifacts(
            "<|vision_start|><|image_pad|><|vision_end|> question", [artifact]
        )
        self.assertTrue(torch.equal(second.mrope_positions, expected))
        self.assertEqual(first.input_ids, second.input_ids)


if __name__ == "__main__":
    unittest.main()
