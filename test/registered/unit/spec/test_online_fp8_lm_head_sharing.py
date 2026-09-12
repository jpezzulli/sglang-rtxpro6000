from types import SimpleNamespace

import torch
from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    dequantize_rowwise_weight,
    replace_linear_weight_rowwise_fp8,
    rowwise_scale_of,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci
from torch import nn

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _DraftModel:
    hot_token_id = None

    def __init__(self):
        self.installed = None

    def set_embed_and_head(self, embed, head):
        self.installed = (embed, head)


class _NonEagle3:
    @staticmethod
    def is_eagle3():
        return False


def _rowwise_head():
    head = nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
    head.weight.data.copy_(torch.arange(24).reshape(6, 4))
    replace_linear_weight_rowwise_fp8(head)
    return head


def test_eagle_hot_vocab_selects_weight_and_scale_rows_together():
    target_head = _rowwise_head()
    embed = nn.Parameter(torch.ones(6, 4))
    draft_model = _DraftModel()
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    lm_head=target_head,
                    get_embed_and_head=lambda: (embed, target_head.weight),
                )
            )
        ),
        draft_runner=SimpleNamespace(model=draft_model),
        hot_token_id=torch.tensor([5, 2, 0]),
        speculative_algorithm=_NonEagle3(),
    )

    EagleDraftWorker.init_lm_head(worker)

    installed = draft_model.installed[1]
    assert not hasattr(installed, "weight_loader")
    torch.testing.assert_close(
        rowwise_scale_of(installed), rowwise_scale_of(target_head.weight)[[5, 2, 0]]
    )
    torch.testing.assert_close(
        dequantize_rowwise_weight(installed),
        dequantize_rowwise_weight(target_head.weight)[[5, 2, 0]],
    )


def test_qwen_mtp_target_module_sharing_retains_rowwise_metadata():
    target_head = _rowwise_head()
    draft = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=False),
        lm_head=nn.Linear(4, 6, bias=False),
    )

    Qwen3_5ForCausalLMMTP.set_lm_head_from_target(draft, target_head)

    assert draft.lm_head is target_head
    assert rowwise_scale_of(draft.lm_head.weight) is rowwise_scale_of(
        target_head.weight
    )
