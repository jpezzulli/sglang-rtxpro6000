"""End-to-end RecoverSSM (--gdn-mtp-cache-mode=none) on a 4-GPU hybrid-linear target.

Server-level parity check for the none-mode port: the same launch, with and without
the mode, must reach the same GSM8K accuracy and accept length, and the extra_buffer
variant is the one that exercises the interval-crossing boundary pass (the second
recovery pass none-mode needs, because it caches no per-draft state to scatter).

Model/quantization args mirror the qualified SM120/GB200 recipe; run the pair together
so the control is the same checkpoint, weights and seed.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=1800, stage="base-c", runner_config="4-gpu-b200")

RECOVER_SSM_MODEL = "nvidia/Qwen3.5-397B-A17B-NVFP4"

_BASE_ARGS = [
    "--tensor-parallel-size",
    "4",
    "--expert-parallel-size",
    "1",
    "--quantization",
    "modelopt_fp4",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--fp4-gemm-backend",
    "flashinfer_cutlass",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--attention-backend",
    "trtllm_mha",
    "--moe-runner-backend",
    "flashinfer_trtllm",
    "--disable-radix-cache",
    "--mem-fraction-static",
    "0.85",
    "--chunked-prefill-size",
    "32768",
    "--max-prefill-tokens",
    "32768",
    "--cuda-graph-max-bs",
    "8",
    "--max-running-requests",
    "32",
    "--scheduler-recv-interval",
    "30",
    "--stream-interval",
    "30",
    "--watchdog-timeout",
    "600",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--linear-attn-backend",
    "flashinfer",
]


def _args(*extra):
    return _BASE_ARGS + list(extra)


class _RecoverSSMGSM8KBase(GSM8KMixin, DefaultServerBase):
    model = RECOVER_SSM_MODEL
    timeout = 2400
    gsm8k_accuracy_thres = 0.92
    gsm8k_accept_length_thres = 3.0
    gsm8k_num_threads = 32


class TestGdnMtpRecoverSSM(_RecoverSSMGSM8KBase):
    """none-mode: no per-draft SSM snapshots, accepted state rebuilt after verify."""

    other_args = _args(
        "--mamba-radix-cache-strategy",
        "no_buffer",
        "--gdn-mtp-cache-mode",
        "none",
    )


class TestGdnMtpFullModeControl(_RecoverSSMGSM8KBase):
    """The control: identical launch, full-mode snapshots (current upstream default).

    Comparing the two in one file is the point -- a threshold that one launch misses
    is a recipe problem, a pair that diverges is the port.
    """

    other_args = _args(
        "--mamba-radix-cache-strategy",
        "no_buffer",
        "--gdn-mtp-cache-mode",
        "full",
    )


class TestGdnMtpRecoverSSMExtraBuffer(_RecoverSSMGSM8KBase):
    """none-mode with mamba radix tracking: covers the boundary recovery pass.

    extra_buffer is what makes a verify's accepted window cross a track interval, and
    none-mode has no cached intermediate state to scatter from at that point, so this
    is the only configuration that runs the second (interval-checkpoint) recovery pass
    end to end. Requires the radix cache, unlike the two above.
    """

    other_args = _args(
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--gdn-mtp-cache-mode",
        "none",
    )


if __name__ == "__main__":
    unittest.main()
