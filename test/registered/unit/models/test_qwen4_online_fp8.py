from types import SimpleNamespace

import torch
from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    configure_online_fp8,
    rowwise_scale_of,
)
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from torch import nn

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _fake_model():
    return SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=False),
        pp_group=SimpleNamespace(is_last_rank=True),
        lm_head=nn.Linear(4, 8, bias=False, dtype=torch.bfloat16),
        modules=lambda: [],
    )


def test_qwen4_post_load_is_option_off_noop():
    configure_online_fp8(False, cuda_available=False, capability=None)
    model = _fake_model()
    original = model.lm_head.weight

    Qwen4ExpForConditionalGeneration.post_load_weights(model)

    assert model.lm_head.weight is original
    assert model.lm_head.weight.dtype == torch.bfloat16


def test_qwen4_post_load_replaces_head_once_and_rejects_tied_embeddings():
    configure_online_fp8(True, cuda_available=True, capability=(12, 0))
    try:
        model = _fake_model()
        Qwen4ExpForConditionalGeneration.post_load_weights(model)
        first = model.lm_head.weight
        assert first.dtype == torch.float8_e4m3fn
        assert rowwise_scale_of(first) is not None

        Qwen4ExpForConditionalGeneration.post_load_weights(model)
        assert model.lm_head.weight is first

        model.config.tie_word_embeddings = True
        try:
            Qwen4ExpForConditionalGeneration.post_load_weights(model)
        except RuntimeError as exc:
            assert "tied" in str(exc)
        else:
            raise AssertionError("tied head should fail loudly")
    finally:
        configure_online_fp8(False, cuda_available=False, capability=None)
