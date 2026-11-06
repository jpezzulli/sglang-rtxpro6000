"""Tests for caller-supplied mm_hashes plumbing.

Verifies the contract that:
  1. GenerateReqInput.mm_hashes is an optional list of hex strings.
  2. MultimodalDataItem.set_pad_value() honors a pre-set hash and does NOT
     overwrite it via hash_feature().
  3. The derived pad_value is deterministic across requests with identical
     mm_hashes — the property external KV routers depend on.

The wiring step that copies GenerateReqInput.mm_hashes into per-item
MultimodalDataItem.hash lives in tokenizer_manager.py; TestCallerHashWiring
pins it here, including the padded-prompt republish it owes the scheduler.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
    _compute_pad_value,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _composed_prompt(internal_hashes):
    """A processor output whose prompt was already padded with its own hashes."""
    input_ids = [1, 0, 2, 0, 2][: 2 * len(internal_hashes) + 1]
    offsets = [(1, 1), (3, 3)][: len(internal_hashes)]
    items = [
        MultimodalDataItem(modality=Modality.IMAGE, offsets=[offset])
        for offset in offsets
    ]
    for item, internal_hash in zip(items, internal_hashes):
        item.set_hash(internal_hash)
    return MultimodalProcessorOutput(
        input_ids=input_ids,
        mm_items=items,
        padded_input_ids=MultimodalProcessorOutput.build_padded_input_ids(
            input_ids, items
        ),
    )


def _install_on_request(output):
    """Run the real scheduler step that installs the padded prompt."""
    request = SimpleNamespace(origin_input_ids=array("q", output.input_ids))
    applied = Scheduler._try_apply_padded_mm_input_ids(
        SimpleNamespace(input_ids=output.input_ids),
        request,
        MultimodalInputs(
            mm_items=output.mm_items, padded_input_ids=output.padded_input_ids
        ),
    )
    return applied, list(request.origin_input_ids)


def _adopt(output, hashes):
    TokenizerManager._apply_caller_mm_hashes(SimpleNamespace(mm_hashes=hashes), output)
    return output


def _installed_prompt(output):
    """The prompt the scheduler installs, or None when it has to re-pad locally."""
    applied, origin = _install_on_request(output)
    if output.padded_input_ids is None:
        return None
    if not applied:
        raise AssertionError("the scheduler refused a padded prompt of matching length")
    for item in output.mm_items:
        for start, end in item.offsets:
            if origin[start : end + 1] != [item.pad_value] * (end - start + 1):
                raise AssertionError(f"item pad {item.pad_value} lost its span")
    return origin


class TestCallerHashWiring(CustomTestCase):
    """A caller hash moves pad_value, so the padded prompt has to move with it."""

    def test_adopted_hashes_republish_the_padded_prompt(self):
        output = _adopt(_composed_prompt([0xAA0, 0xBB0]), ["12345", "54321"])
        self.assertEqual([item.hash for item in output.mm_items], [0x12345, 0x54321])
        origin = _installed_prompt(output)
        # Neither stale internal pad survives into the request the scheduler sees.
        self.assertNotIn(_compute_pad_value(0xAA0), origin)
        self.assertNotIn(_compute_pad_value(0xBB0), origin)
        self.assertEqual(
            origin,
            [
                1,
                _compute_pad_value(0x12345),
                2,
                _compute_pad_value(0x54321),
                2,
            ],
        )

    def test_a_malformed_entry_falls_back_without_orphaning_the_other_image(self):
        output = _composed_prompt([0xAA0, 0xBB0])
        expected_pads = [item.pad_value for item in output.mm_items]
        with self.assertLogs("sglang.srt.managers.tokenizer_manager", "WARNING"):
            _adopt(output, ["12345", "not-a-hash"])
        self.assertEqual(output.mm_items[1].pad_value, expected_pads[1])
        origin = _installed_prompt(output)
        self.assertEqual(origin[1], _compute_pad_value(0x12345))
        self.assertEqual(origin[3], expected_pads[1])

    def test_absent_or_mismatched_hashes_leave_the_prompt_alone(self):
        for hashes in (None, [], ["12345"], ["12345", "54321", "9"]):
            with self.subTest(hashes=hashes):
                output = _adopt(_composed_prompt([0xAA0, 0xBB0]), hashes)
                self.assertEqual(
                    [item.hash for item in output.mm_items], [0xAA0, 0xBB0]
                )
                self.assertEqual(
                    output.padded_input_ids,
                    MultimodalProcessorOutput.build_padded_input_ids(
                        output.input_ids, output.mm_items
                    ),
                )

    def test_an_unpadded_prompt_stays_with_the_model_specific_padding(self):
        # A Transformers-backed processor can hand back native placeholders on
        # purpose; adopting a hash must not quietly hand the scheduler a padded
        # sequence that bypasses its model-specific pad_input_ids_func.
        output = _composed_prompt([0xAA0, 0xBB0])
        output.padded_input_ids = None
        _adopt(output, ["12345", "54321"])
        self.assertEqual([item.hash for item in output.mm_items], [0x12345, 0x54321])
        self.assertIsNone(output.padded_input_ids)
        applied, _ = _install_on_request(output)
        self.assertFalse(applied)

    def test_an_unknown_pad_value_is_left_to_the_scheduler(self):
        output = _composed_prompt([0xAA0, 0xBB0])
        output.mm_items[1].hash = None
        output.mm_items[1].pad_value = None
        with self.assertLogs("sglang.srt.managers.tokenizer_manager", "WARNING"):
            _adopt(output, ["12345", "not-a-hash"])
        self.assertEqual(output.mm_items[0].pad_value, _compute_pad_value(0x12345))
        # Padding the known item alone would claim a sequence that does not
        # describe the whole prompt, so the scheduler re-pads it locally.
        self.assertIsNone(output.padded_input_ids)
        self.assertFalse(_install_on_request(output)[0])


class TestMmHashesContract(CustomTestCase):
    def test_generate_req_input_accepts_mm_hashes(self):
        """GenerateReqInput exposes mm_hashes as an optional field."""
        req = GenerateReqInput(
            text="hi",
            image_data=["http://example.com/img.png"],
            mm_hashes=["deadbeefcafe1234"],
        )
        self.assertEqual(req.mm_hashes, ["deadbeefcafe1234"])

    def test_generate_req_input_defaults_mm_hashes_to_none(self):
        """Absent mm_hashes preserves existing (None) behavior."""
        req = GenerateReqInput(text="hi")
        self.assertIsNone(req.mm_hashes)

    def test_content_hashes_are_distinct_from_feature_hashes(self):
        content_hash = "sha256:" + "ab" * 32
        req = GenerateReqInput(
            text="hi",
            image_data=["http://example.com/img.png"],
            mm_hashes=["deadbeef"],
            mm_content_hashes=[content_hash],
        )
        self.assertEqual(req.mm_hashes, ["deadbeef"])
        self.assertEqual(req.mm_content_hashes, [content_hash])

    def test_batched_hashes_follow_each_request(self):
        req = GenerateReqInput(
            text=["one", "two"],
            image_data=[["a"], ["b", "c"]],
            mm_hashes=["01", ["02", "03"]],
            mm_content_hashes=[
                ["sha256:" + "11" * 32],
                ["sha256:" + "22" * 32, "sha256:" + "33" * 32],
            ],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req[0].mm_hashes, ["01"])
        self.assertEqual(req[1].mm_hashes, ["02", "03"])
        self.assertEqual(len(req[1].mm_content_hashes), 2)

    def test_set_pad_value_honors_preset_hash(self):
        """set_pad_value() must use a pre-set hash without recomputing."""
        item = MultimodalDataItem(modality=Modality.IMAGE, hash=0xDEADBEEF)
        # If hash_feature is invoked, the test fails — we patch it to
        # raise so any accidental recompute is loud.
        with patch(
            "sglang.srt.managers.mm_utils.hash_feature",
            side_effect=AssertionError(
                "hash_feature must NOT be called when hash is preset"
            ),
        ):
            item.set_pad_value()
        self.assertEqual(item.hash, 0xDEADBEEF)
        self.assertEqual(item.pad_value, _compute_pad_value(0xDEADBEEF))

    def test_set_pad_value_is_deterministic_across_items(self):
        """Two items with the same preset hash must derive the same pad_value."""
        a = MultimodalDataItem(modality=Modality.IMAGE, hash=0x123456789ABCDEF0)
        b = MultimodalDataItem(modality=Modality.IMAGE, hash=0x123456789ABCDEF0)
        # No feature payload — set_pad_value uses the preset hash.
        a.set_pad_value()
        b.set_pad_value()
        self.assertEqual(a.pad_value, b.pad_value)
        self.assertEqual(a.hash, b.hash)

    def test_set_pad_value_distinguishes_different_preset_hashes(self):
        """Distinct preset hashes must produce distinct pad_values."""
        a = MultimodalDataItem(modality=Modality.IMAGE, hash=0xAAAA)
        b = MultimodalDataItem(modality=Modality.IMAGE, hash=0xBBBB)
        a.set_pad_value()
        b.set_pad_value()
        self.assertNotEqual(a.pad_value, b.pad_value)

    def test_set_hash_updates_an_existing_pad_value(self):
        item = MultimodalDataItem(modality=Modality.IMAGE, hash=0xAAAA)
        item.set_pad_value()

        item.set_hash(0xBBBB)

        self.assertEqual(item.hash, 0xBBBB)
        self.assertEqual(item.pad_value, _compute_pad_value(0xBBBB))


if __name__ == "__main__":
    unittest.main()
