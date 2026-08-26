import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen4ExpMTPRuntimeContext(CustomTestCase):
    def test_lm_head_reads_configured_dp_flag(self):
        config = SimpleNamespace(
            hc_count=4,
            hidden_size=8,
            rms_norm_eps=1e-6,
            vocab_size=16,
        )
        parallel = SimpleNamespace(
            tp_size=1,
            config=SimpleNamespace(enable_dp_lm_head=True),
        )

        with (
            patch(
                "sglang.srt.models.qwen4_exp_mtp.get_parallel",
                return_value=parallel,
            ),
            patch("sglang.srt.models.qwen4_exp_mtp.get_pp_group"),
            patch("sglang.srt.models.qwen4_exp_mtp.Qwen4ExpModel"),
            patch("sglang.srt.models.qwen4_exp_mtp.ParallelLMHead") as lm_head,
            patch("sglang.srt.models.qwen4_exp_mtp.LogitsProcessor"),
            patch.object(
                Qwen4ExpForCausalLMMTP,
                "_init_mtp_input_fusion",
                return_value=None,
            ),
        ):
            Qwen4ExpForCausalLMMTP(config)

        self.assertTrue(lm_head.call_args.kwargs["use_attn_tp_group"])


if __name__ == "__main__":
    unittest.main()
