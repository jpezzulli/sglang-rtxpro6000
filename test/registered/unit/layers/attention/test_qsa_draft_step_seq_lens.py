"""Per-draft-step sequence lengths are built as one [steps, bs] table.

Before this, ``QwenSparseMultiStepDraftBackend`` rebuilt
``(seq_lens + step + 1).to(int32)`` for EVERY draft step in front of the draft
CUDA graph -- three eager launches per step, 42 at ``num_steps = 15``, all on
the GPU-idle critical path (donor measurement: ~250 us of a W16 decode step,
draft phase 1,038 -> 998 kernels).  One int32 ``[steps, bs]`` table now feeds
every step, and the host lengths are derived once and offset per step.

The tests are CPU-only and pin the contract the optimization has to keep:
* every step's GPU and host lengths are element-wise what the old per-step
  expression produced (the old expression is the oracle here),
* the per-step ``out_cache_loc`` slice is untouched,
* host and device lengths of one step stay consistent (same base, same
  ``step + 1`` offset) and no host-less batch ever forces a D2H,
* the shared table is not writable through a padded step (no stale value
  leaks into the other steps' rows),
* the rows really do come from ONE build, and the offsets buffer is reused.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseMultiStepDraftBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

STEPS = 4
BATCH = 3
TOPK = 1
BASE_LENS = [37, 129, 1024]


class _RecordingBackend:
    """Stands in for one per-step QwenSparseAttnBackend."""

    def __init__(self):
        self.batches = []
        self.replays = []

    def init_forward_metadata(self, forward_batch):
        self.batches.append(forward_batch)

    def _replay_cuda_graph_metadata(self, **kwargs):
        self.replays.append(kwargs)


def _backend(attn_backends=None):
    backend = QwenSparseMultiStepDraftBackend.__new__(QwenSparseMultiStepDraftBackend)
    backend.model_runner = None
    backend.topk = TOPK
    backend.speculative_num_steps = STEPS
    backend.attn_backends = (
        [_RecordingBackend() for _ in range(STEPS - 1)]
        if attn_backends is None
        else attn_backends
    )
    backend._step_offsets = None
    return backend


def _forward_batch(seq_lens_cpu=True, num_padding=None, dtype=torch.int32):
    seq_lens = torch.tensor(BASE_LENS, dtype=dtype)
    batch = SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cpu=(
            None if not seq_lens_cpu else torch.tensor(BASE_LENS, dtype=torch.int64)
        ),
        batch_size=len(BASE_LENS),
        req_pool_indices=torch.arange(len(BASE_LENS), dtype=torch.int32),
        spec_info=object(),
        out_cache_loc=torch.arange(len(BASE_LENS) * TOPK * STEPS, dtype=torch.int32),
        num_padding=num_padding,
    )
    return batch


def _oracle_lens(step, dtype=torch.int32):
    """The per-step expression this replaces."""
    return (torch.tensor(BASE_LENS, dtype=torch.int64) + step + 1).to(dtype)


class TestStepLengthsMatchTheReplacedExpression(unittest.TestCase):
    def test_gpu_and_host_lengths_and_slots_are_unchanged(self):
        backend = _backend()
        batch = _forward_batch()
        step_lens = backend._all_step_seq_lens(batch)
        for step in range(len(backend.attn_backends)):
            with self.subTest(step=step):
                made = backend._make_step_forward_batch(
                    batch, step, step_lens=step_lens
                )
                self.assertEqual(
                    made.seq_lens.dtype, torch.int32, "the graph buffer stays int32"
                )
                self.assertTrue(
                    torch.equal(made.seq_lens, _oracle_lens(step)),
                    "device lengths must equal the replaced expression",
                )
                self.assertTrue(
                    torch.equal(made.seq_lens_cpu, _oracle_lens(step, torch.int32)),
                    "host lengths must stay consistent with the device side",
                )
                self.assertEqual(made.forward_mode, ForwardMode.DECODE)
                self.assertEqual(made.batch_size, BATCH)
                self.assertIs(made.spec_info, batch.spec_info)
                # The per-step out_cache_loc mapping is the backend's own
                # (draft_forward interleaving) and must not have moved.
                self.assertTrue(
                    torch.equal(
                        made.out_cache_loc,
                        batch.out_cache_loc.reshape(BATCH, TOPK, STEPS)
                        .permute(2, 0, 1)
                        .reshape(STEPS, -1)[step],
                    )
                )

    def test_int64_input_still_produces_int32(self):
        backend = _backend()
        batch = _forward_batch(dtype=torch.int64)
        made = backend._make_step_forward_batch(
            batch, 0, step_lens=backend._all_step_seq_lens(batch)
        )
        self.assertEqual(made.seq_lens.dtype, torch.int32)
        self.assertTrue(torch.equal(made.seq_lens, _oracle_lens(0)))

    def test_fallback_path_is_unchanged_without_a_table(self):
        backend = _backend()
        batch = _forward_batch()
        made = backend._make_step_forward_batch(batch, 1)
        self.assertTrue(torch.equal(made.seq_lens, _oracle_lens(1)))
        self.assertTrue(torch.equal(made.seq_lens_cpu, _oracle_lens(1)))


class TestSingleBuildPerForward(unittest.TestCase):
    def test_all_steps_share_one_table_and_the_host_side_is_derived_once(self):
        backend = _backend()
        calls = []
        original = QwenSparseMultiStepDraftBackend.__dict__["_as_cpu_lengths"].__func__

        def counting(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        backend._as_cpu_lengths = counting
        batch = _forward_batch()
        backend.init_forward_metadata(batch)

        backends = backend.attn_backends
        made = []
        for step, recorded in enumerate(backends):
            self.assertEqual(len(recorded.batches), 1, "one metadata call per step")
            made.append(recorded.batches[-1])
        storages = {b.seq_lens.untyped_storage().data_ptr() for b in made}
        self.assertEqual(len(storages), 1, "one table, not one per step")
        self.assertEqual(
            len(calls), 1, "host lengths derived once for the whole draft phase"
        )
        for step, batch in enumerate(made):
            self.assertEqual(batch.seq_lens.shape, (BATCH,), "each row is [bs]")
            self.assertEqual(batch.seq_lens.dtype, torch.int32)
            self.assertTrue(torch.equal(batch.seq_lens, _oracle_lens(step)))
            self.assertTrue(
                torch.equal(batch.seq_lens_cpu, _oracle_lens(step, torch.int32))
            )

    def test_offsets_buffer_is_reused_and_sized_to_the_steps(self):
        backend = _backend()
        first = backend._all_step_seq_lens(_forward_batch())
        offsets = backend._step_offsets
        second = backend._all_step_seq_lens(_forward_batch())
        self.assertIs(offsets, backend._step_offsets, "no per-forward re-create")
        self.assertEqual(offsets.shape, (len(backend.attn_backends), 1))
        self.assertEqual(offsets.dtype, torch.int32)
        self.assertTrue(
            torch.equal(offsets.reshape(-1), torch.arange(1, STEPS, dtype=torch.int32))
        )
        self.assertNotEqual(first[0].data_ptr(), second[0].data_ptr())

    def test_offsets_are_rebuilt_for_another_device(self):
        backend = _backend()
        batch = _forward_batch()
        backend._all_step_seq_lens(batch)
        offsets = backend._step_offsets
        meta_batch = _forward_batch()
        meta_batch.seq_lens = meta_batch.seq_lens.to("meta")
        backend._all_step_seq_lens(meta_batch)
        self.assertIsNot(offsets, backend._step_offsets)
        self.assertEqual(backend._step_offsets.device, torch.device("meta"))


class TestPaddingIsolation(unittest.TestCase):
    """Padded rows must not be written through the table the other steps share."""

    def test_padded_step_does_not_clobber_the_shared_table(self):
        num_padding = 1
        backend = _backend()
        batch = _forward_batch(num_padding=num_padding)
        backend.init_forward_metadata_out_graph(batch, in_capture=False)

        for step, recorded in enumerate(backend.attn_backends):
            with self.subTest(step=step):
                (replay,) = recorded.replays
                seq_lens = replay["seq_lens"]
                self.assertEqual(
                    int(seq_lens[-num_padding:]),
                    1,
                    "padding rows still collapse to length 1",
                )
                self.assertTrue(
                    torch.equal(
                        seq_lens[:-num_padding],
                        _oracle_lens(step)[:-num_padding],
                    )
                )
                host = replay["seq_lens_cpu"]
                self.assertEqual(
                    int(host[-num_padding:]),
                    1,
                    "host and device padding agree",
                )
                self.assertTrue(
                    torch.equal(host[:-num_padding], _oracle_lens(step)[:-num_padding])
                )
                self.assertEqual(replay["num_padding"], num_padding)

        # A second draft phase off the same backend must see pristine lengths:
        # writing padding into the shared table would have leaked here.
        second = _forward_batch(num_padding=num_padding)
        backend.init_forward_metadata_out_graph(second, in_capture=False)
        for step, recorded in enumerate(backend.attn_backends):
            self.assertEqual(
                int(recorded.replays[-1]["seq_lens"][-num_padding:]),
                1,
            )
            self.assertTrue(
                torch.equal(
                    recorded.replays[-1]["seq_lens"][:-num_padding],
                    _oracle_lens(step)[:-num_padding],
                )
            )

    def test_padding_clamps_to_the_row_count(self):
        backend = _backend()
        batch = _forward_batch(num_padding=BATCH + 5)
        backend.init_forward_metadata_out_graph(batch, in_capture=False)
        for step, recorded in enumerate(backend.attn_backends):
            (replay,) = recorded.replays
            self.assertTrue(torch.equal(replay["seq_lens"], torch.ones(BATCH)))
            self.assertTrue(torch.equal(replay["seq_lens_cpu"], torch.ones(BATCH)))


class TestHostLessBatchNeverSyncs(unittest.TestCase):
    def test_no_host_lengths_no_d2h_and_no_host_side(self):
        backend = _backend()
        original = QwenSparseMultiStepDraftBackend.__dict__["_as_cpu_lengths"].__func__

        def explode(*args, **kwargs):
            raise AssertionError("GPU-only serving must not derive host lengths")

        backend._as_cpu_lengths = explode
        batch = _forward_batch(seq_lens_cpu=False)
        backend.init_forward_metadata(batch)
        for recorded in backend.attn_backends:
            self.assertIsNone(recorded.batches[-1].seq_lens_cpu)
        self.assertIsNone(backend._all_step_seq_lens(batch)[1])
        self.assertTrue(torch.is_tensor(original(torch.tensor([1]), None)))


class TestNoPerStepBackends(unittest.TestCase):
    def test_single_step_draft_phase_builds_nothing_extra(self):
        backend = _backend(attn_backends=[])
        self.assertIsNone(backend._all_step_seq_lens(_forward_batch()))
        backend.init_forward_metadata(_forward_batch())
        backend.init_forward_metadata_out_graph(_forward_batch(), in_capture=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
