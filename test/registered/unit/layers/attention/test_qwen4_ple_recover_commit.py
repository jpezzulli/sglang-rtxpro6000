import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestQwen4PleRecoverCommit(CustomTestCase):
    def test_accepted_and_boundary_states_commit_after_recovery(self):
        backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)

        short_state = torch.full((1, 8, 2), -1.0)
        short_intermediate = torch.arange(1 * 2 * 4 * 2, dtype=torch.float32).view(
            1, 2, 4, 2
        )
        ngram_state = torch.full((8, 3), -1, dtype=torch.int64)
        ngram_intermediate = torch.arange(2 * 4 * 3, dtype=torch.int64).view(2, 4, 3)
        req_pool = SimpleNamespace(
            short_conv_pool=SimpleNamespace(
                conv_state=short_state,
                intermediate_conv_state=short_intermediate,
            ),
            ngram_pool=SimpleNamespace(
                context=ngram_state,
                intermediate_context=ngram_intermediate,
            ),
        )
        backend.linear_attn_backend = SimpleNamespace(req_to_token_pool=req_pool)

        state_indices = torch.tensor([2, 3])
        accepted_steps = torch.tensor([0, 3])
        track_indices = torch.tensor([4, 5])
        track_steps = torch.tensor([-1, 1])

        backend._update_ple_state_after_mtp_verify(
            state_indices,
            accepted_steps,
            track_indices,
            track_steps,
        )

        torch.testing.assert_close(short_state[:, 2], short_intermediate[:, 0, 0])
        torch.testing.assert_close(short_state[:, 3], short_intermediate[:, 1, 3])
        torch.testing.assert_close(short_state[:, 5], short_intermediate[:, 1, 1])
        self.assertTrue(torch.equal(short_state[:, 4], torch.full((1, 2), -1.0)))

        self.assertTrue(torch.equal(ngram_state[2], ngram_intermediate[0, 0]))
        self.assertTrue(torch.equal(ngram_state[3], ngram_intermediate[1, 3]))
        self.assertTrue(torch.equal(ngram_state[5], ngram_intermediate[1, 1]))
        self.assertTrue(torch.equal(ngram_state[4], torch.full((3,), -1)))


if __name__ == "__main__":
    unittest.main()
