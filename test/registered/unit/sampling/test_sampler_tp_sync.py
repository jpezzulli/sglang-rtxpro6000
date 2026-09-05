"""Token-ID synchronization must respect the selected group's cardinality."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import sampler as sampler_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSamplerTPSync(unittest.TestCase):
    def make_sampler(self, tp_size, attn_tp_size=None):
        tp = SimpleNamespace(world_size=tp_size, device_group=object())
        attn = SimpleNamespace(world_size=attn_tp_size, device_group=object())
        execution = SimpleNamespace(
            deterministic=SimpleNamespace(
                rl_on_policy_target=None, enable_deterministic_inference=False
            ),
            kernel=SimpleNamespace(sampling_backend="pytorch"),
        )
        with (
            patch.object(sampler_mod, "get_tp_group", return_value=tp),
            patch.object(
                sampler_mod,
                "is_dp_attention_enabled",
                return_value=attn_tp_size is not None,
            ),
            patch.object(
                sampler_mod,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_group=attn),
            ),
            patch.object(sampler_mod, "get_exec", return_value=execution),
            patch.object(
                sampler_mod.dist,
                "get_world_size",
                side_effect=AssertionError("construction must not query torch groups"),
            ),
        ):
            sampler = sampler_mod.Sampler()
        return sampler, attn if attn_tp_size is not None else tp

    def test_one_rank_never_runs_a_collective(self):
        for tp_size, attn_size in ((1, None), (8, 1)):
            for forced, grammars in (
                (False, None),
                (False, [object()]),
                (True, None),
                (True, [object()]),
            ):
                with self.subTest(tp=tp_size, attn=attn_size, forced=forced):
                    sampler, group = self.make_sampler(tp_size, attn_size)
                    tokens = torch.tensor([9, 3, 12], dtype=torch.int64)
                    original = tokens.clone()
                    with (
                        patch.object(sampler_mod, "SYNC_TOKEN_IDS_ACROSS_TP", forced),
                        patch.object(sampler_mod.dist, "all_reduce") as reduce,
                    ):
                        sampler._sync_token_ids_across_tp(
                            tokens, SimpleNamespace(grammars=grammars)
                        )
                    reduce.assert_not_called()
                    self.assertIs(sampler.tp_sync_group, group.device_group)
                    torch.testing.assert_close(tokens, original)

    def test_multi_rank_keeps_minimum_on_the_selected_group(self):
        for tp_size, attn_size in ((2, None), (8, 2)):
            for forced, grammars in (
                (False, None),
                (False, [object()]),
                (True, None),
                (True, [object()]),
            ):
                with self.subTest(tp=tp_size, attn=attn_size, forced=forced):
                    sampler, group = self.make_sampler(tp_size, attn_size)
                    tokens = torch.tensor([9, 3, 12], dtype=torch.int64)
                    with (
                        patch.object(sampler_mod, "SYNC_TOKEN_IDS_ACROSS_TP", forced),
                        patch.object(sampler_mod.dist, "all_reduce") as reduce,
                    ):
                        sampler._sync_token_ids_across_tp(
                            tokens, SimpleNamespace(grammars=grammars)
                        )
                    if forced or grammars:
                        reduce.assert_called_once()
                        args, kwargs = reduce.call_args
                        self.assertIs(args[0], tokens)
                        self.assertIs(kwargs["op"], sampler_mod.dist.ReduceOp.MIN)
                        self.assertIs(kwargs["group"], group.device_group)
                    else:
                        reduce.assert_not_called()

    def test_grammar_greedy_forward_preserves_tokens_without_collective(self):
        sampler, _ = self.make_sampler(1)
        logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 0.0]])
        info = SimpleNamespace(
            grammars=[object(), object()],
            is_all_greedy=True,
            return_sampling_masks=None,
        )
        with (
            patch.object(sampler_mod, "SYNC_TOKEN_IDS_ACROSS_TP", False),
            patch.object(sampler_mod, "_use_aiter", False),
            patch.object(sampler, "_preprocess_logits", return_value=logits),
            patch.object(sampler_mod.dist, "all_reduce") as reduce,
        ):
            tokens = sampler(
                SimpleNamespace(next_token_logits=logits),
                info,
                return_logprob=False,
                top_logprobs_nums=[],
                token_ids_logprobs=[],
                positions=torch.tensor([3, 5]),
            )
        reduce.assert_not_called()
        torch.testing.assert_close(tokens, torch.tensor([1, 0]))


if __name__ == "__main__":
    unittest.main()
