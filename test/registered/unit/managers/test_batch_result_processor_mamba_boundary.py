import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


# The decode checkpoint grid is lcm(mamba_cache_chunk_size, tree page,
# interval); keeping all three equal leaves it at the interval under test.
TRACK_INTERVAL = 4


def _make_batch() -> tuple[Req, ScheduleBatch]:
    sampling_params = SamplingParams(max_new_tokens=32)
    sampling_params.normalize(None)
    req = Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2]),
        sampling_params=sampling_params,
        vocab_size=128,
    )
    req.output_ids.append(3)
    req.kv_committed_len = 2

    batch = ScheduleBatch(reqs=[req])
    batch.tree_cache = SimpleNamespace(page_size=TRACK_INTERVAL)
    batch.device = "cpu"
    batch.model_config = SimpleNamespace(is_encoder_decoder=False)
    batch.enable_overlap = True
    batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    batch.sampling_info = SimpleNamespace(
        penalizer_orchestrator=SimpleNamespace(is_required=False)
    )
    batch.hisparse_coordinator = None
    batch.seq_lens = torch.tensor([2], dtype=torch.int64)
    batch.seq_lens_cpu = torch.tensor([2], dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor([2], dtype=torch.int32)
    return req, batch


def _make_processor() -> SchedulerBatchResultProcessor:
    metrics_reporter = MagicMock()
    metrics_reporter.num_generated_tokens = 0
    metrics_reporter.forward_ct_decode = 0
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=True,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=MagicMock(),
        tree_cache=SimpleNamespace(page_size=TRACK_INTERVAL),
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=metrics_reporter,
        draft_worker=None,
        model_worker=MagicMock(),
        logprob_result_processor=None,
        output_streamer=MagicMock(),
        abort_request=lambda *args, **kwargs: None,
    )


def _make_result():
    return SimpleNamespace(
        copy_done=None,
        auxiliary_host_output=None,
        routed_experts_output=None,
        indexer_topk_output=None,
        logits_output=SimpleNamespace(hidden_states=None, customized_info=None),
        next_token_ids=[4],
        can_run_cuda_graph=False,
        num_correct_drafts=0,
        num_block_accept_tokens=0,
        num_cap_tokens=0,
        speculative_num_draft_tokens=0,
    )


class TestMambaBoundaryMaskReuse(unittest.TestCase):
    def test_overlap_scheduler_handles_zero_and_one_batch_lookahead(self):
        for schedule_next_decode, expected_lookahead in ((False, 0), (True, 1)):
            with self.subTest(schedule_next_decode=schedule_next_decode):
                req, batch = _make_batch()
                processor = _make_processor()
                result = _make_result()

                scheduler = Scheduler.__new__(Scheduler)
                scheduler.gracefully_exit = False
                scheduler.request_receiver = MagicMock()
                scheduler.request_receiver.recv_requests.side_effect = [
                    [],
                    [],
                    StopIteration,
                ]
                scheduler.process_input_requests = MagicMock()
                scheduler._engine_paused = False
                scheduler.running_batch = batch
                scheduler.is_disable_overlap_for_batch = MagicMock(return_value=False)
                scheduler.run_batch = MagicMock(return_value=result)
                scheduler._apply_war_barrier = MagicMock()
                scheduler.is_generation = False
                scheduler.last_batch = None

                plan_count = 0

                def get_next_batch_to_run(*, running_batch, last_batch):
                    nonlocal plan_count
                    del running_batch, last_batch
                    plan_count += 1
                    if plan_count == 1:
                        batch.prepare_for_decode()
                        return SimpleNamespace(
                            running_batch=batch,
                            batch_to_run=batch,
                        )
                    if plan_count == 2 and schedule_next_decode:
                        batch.prepare_for_decode()
                        return SimpleNamespace(
                            running_batch=batch,
                            batch_to_run=batch,
                        )
                    return SimpleNamespace(
                        running_batch=batch,
                        batch_to_run=None,
                    )

                scheduler.get_next_batch_to_run = get_next_batch_to_run
                observed_lookahead = []

                def process_batch_result(result_batch, batch_result):
                    observed_lookahead.append(
                        req.decode_batch_idx
                        - result_batch.mamba_decode_batch_idx_cpu[0]
                    )
                    processor.process_batch_result_decode(result_batch, batch_result)

                scheduler.process_batch_result = process_batch_result

                with (
                    # The mamba predicates and the track interval read the
                    # published bags, so publish the configuration under test
                    # (non-lazy extra buffer, interval 4); observability and
                    # disagg reads are served by the same publish at their
                    # defaults.
                    get_context().override_server_args(
                        mamba_radix_cache_strategy="extra_buffer",
                        mamba_track_interval=TRACK_INTERVAL,
                        _mamba_cache_chunk_size=TRACK_INTERVAL,
                    ),
                    patch(
                        "sglang.srt.managers.schedule_batch.alloc_for_decode",
                        return_value=torch.tensor([3], dtype=torch.int64),
                    ),
                    patch(
                        "sglang.srt.managers.schedule_batch.set_mamba_track_indices_from_reqs"
                    ),
                    patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor),
                    patch.object(
                        SchedulerBatchResultProcessor,
                        "_mamba_prefix_cache_update",
                    ) as cache_update,
                ):
                    with self.assertRaises(StopIteration):
                        scheduler.event_loop_overlap()

                self.assertEqual(observed_lookahead, [expected_lookahead])
                if expected_lookahead == 0:
                    cache_update.assert_not_called()
                else:
                    self.assertTrue(cache_update.call_args.kwargs["known_boundary"])


class _SpecAlgorithm:
    def is_none(self) -> bool:
        return False


class _DecodeMode:
    def is_decode(self) -> bool:
        return True

    def is_extend(self) -> bool:
        return False


class _SpecBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.has_grammar = any(req.grammar is not None for req in reqs)
        self.forward_mode = _DecodeMode()
        self.spec_algorithm = _SpecAlgorithm()


class _Grammar:
    """Grammar stub that terminates past `keep` accepted tokens; keep=0 rejects
    the very first one, so nothing is retained."""

    def __init__(self, keep: int):
        self.finished = False
        self._keep = keep
        self._accepted = 0

    def accept_token(self, token_id: int):
        if self._keep == 0:
            raise ValueError("token rejected by grammar")
        self._accepted += 1

    def is_terminated(self) -> bool:
        return self._accepted >= self._keep


class TestSpecGrammarTrackBoundary(unittest.TestCase):
    """A grammar-truncated speculative run must only checkpoint on the tokens
    it actually committed (upstream sgl-project/sglang#43029)."""

    STRATEGIES = ["extra_buffer", "extra_buffer_lazy"]

    def _step(self, *, prompt_len, output_len, grammar, accepted, stride, strategy):
        """Run one spec-v2 decode step through the production post-processing
        and return the boundary the scheduler would record for it."""
        sampling_params = SamplingParams(max_new_tokens=256, temperature=0)
        sampling_params.normalize(None)
        req = Req(
            rid="spec-grammar",
            origin_input_text="",
            origin_input_ids=array("q", [1] * prompt_len),
            sampling_params=sampling_params,
            vocab_size=128,
        )
        req.kv_committed_len = prompt_len
        req.output_ids.extend([7] * output_len)
        req.grammar = grammar

        processor = _make_processor()
        result = SimpleNamespace(
            next_token_ids=torch.tensor(
                accepted + [0] * (stride - len(accepted)), dtype=torch.long
            ),
            accept_lens=torch.tensor([len(accepted)], dtype=torch.long),
            speculative_num_draft_tokens=stride,
            num_correct_drafts=None,
            num_correct_drafts_per_req_cpu=None,
            block_accept_lens=None,
            cap_lens=None,
            copy_done=None,
            grammar_advanced=False,
            grammar_retained_tokens=None,
        )
        batch = _SpecBatch([req])

        with get_context().override_server_args(
            mamba_radix_cache_strategy=strategy,
            mamba_track_interval=TRACK_INTERVAL,
            _mamba_cache_chunk_size=TRACK_INTERVAL,
        ):
            predicted = processor._resolve_spec_v2_tokens(result, batch)[0]
            # process_batch_result_decode appends the committed run before the
            # track-boundary check, so seqlen reflects the retained run only.
            req.output_ids.extend(predicted)
            return processor._mamba_check_track_boundary(req, batch, result, 0)

    def test_truncated_run_below_boundary_records_no_checkpoint(self):
        # seq_len 9 before the step, 4 drafts accepted, the grammar keeps 1: the
        # retained run covers position 9 alone, so no grid line is crossed. The
        # accepted-run step-back still reports line 8 -- a checkpoint this step
        # never wrote -- and moves the ping-pong pointer onto it.
        for strategy in self.STRATEGIES:
            with self.subTest(strategy=strategy):
                boundary = self._step(
                    prompt_len=3,
                    output_len=6,
                    grammar=_Grammar(keep=1),
                    accepted=[101, 102, 103, 104],
                    stride=4,
                    strategy=strategy,
                )
                self.assertEqual(boundary, (False, 0))

    def test_truncated_run_that_reaches_the_boundary_still_checkpoints(self):
        # Truncation must not hide a real crossing: seq_len 8 -> 10 covers line 8.
        for strategy in self.STRATEGIES:
            with self.subTest(strategy=strategy):
                boundary = self._step(
                    prompt_len=4,
                    output_len=4,
                    grammar=_Grammar(keep=2),
                    accepted=[101, 102, 103, 104],
                    stride=4,
                    strategy=strategy,
                )
                self.assertEqual(boundary, (True, 8))

    def test_fully_rejected_run_records_no_checkpoint(self):
        for strategy in self.STRATEGIES:
            with self.subTest(strategy=strategy):
                boundary = self._step(
                    prompt_len=3,
                    output_len=6,
                    grammar=_Grammar(keep=0),
                    accepted=[101, 102, 103, 104],
                    stride=4,
                    strategy=strategy,
                )
                self.assertEqual(boundary, (False, 0))

    def test_untruncated_spec_run_is_unchanged(self):
        # No grammar: the accepted run (drafts + bonus) is the committed run, so
        # the boundary step-back keeps its pre-#43029 shape.
        for strategy, prompt_len, expected in [
            ("extra_buffer", 3, (False, 0)),  # seq_len 5 -> 7, grid line 8 unreached
            ("extra_buffer", 4, (True, 8)),  # seq_len 6 -> 8, crosses on the last token
            ("extra_buffer_lazy", 3, (False, 0)),
            ("extra_buffer_lazy", 4, (True, 8)),
        ]:
            with self.subTest(strategy=strategy, prompt_len=prompt_len):
                boundary = self._step(
                    prompt_len=prompt_len,
                    output_len=2,
                    grammar=None,
                    accepted=[101, 102, 103],
                    stride=4,
                    strategy=strategy,
                )
                self.assertEqual(boundary, expected)


if __name__ == "__main__":
    unittest.main()
