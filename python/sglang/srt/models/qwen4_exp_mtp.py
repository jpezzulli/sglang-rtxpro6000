"""Inference-only Qwen4-Exp MTP speculative decoding."""

import copy
import functools
import logging
from contextlib import ExitStack
from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP, _mtp_quant_config
from sglang.srt.models.qwen4_exp import Qwen4ExpModel
from sglang.srt.runtime_context import get_model, get_parallel
from sglang.srt.utils import add_prefix, is_npu

logger = logging.getLogger(__name__)

#: Hard row limit of the donor dense GEMV (``assert M <= 16`` in w8a16_gemv).
_GEMV_DONOR_MAX_M = 16


@functools.lru_cache(maxsize=None)
def _sm120_entry_gemm_target(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (12, 0)


def _bf16_entry_gemm_target_ok(*tensors: torch.Tensor) -> bool:
    """Where this fork accepts the donor dense GEMV at all.

    One compatible colocated CUDA device, exact SM120 only: that is the part
    of the donor hardware contract the resident W8A16/dense-GEMV port was
    measured and accepted for (SGLang #40362/FlashInfer #6227 campaigns stay
    untouched). Anything else -- CPU tensors, split devices, other GPUs --
    keeps the original two-Linear path; the ``bf16_gemv`` wrapper itself has
    no CPU or mixed-dtype fallback (it reaches ``_num_sms`` and its ``tl.dot``
    contract is BF16).
    """
    device = tensors[0].device
    if device.type != "cuda":
        return False
    if any(tensor.device != device for tensor in tensors[1:]):
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _sm120_entry_gemm_target(index)


def _mtp_fc_gemv_supported(
    normed_embeds: torch.Tensor,
    embed_weight: torch.Tensor,
    hidden_rows: torch.Tensor,
    hidden_weight: torch.Tensor,
) -> bool:
    """Eligibility gate for the donor-gated MTP entry fusion GEMVs.

    Validates BOTH normalized activations, BOTH weights (a BF16 embedding
    side with an FP32 hidden side must keep the old path: the donor kernel's
    contract is bf16 activations against bf16 weights), the row budget and
    the K-contiguous weight layout every donor caller provides, plus the
    accepted-hardware contract above.  The donor comment about ``w.stride(0)
    == 1`` lives in ``_plan``'s tile choice; the caller contract this gate
    pins is ``stride(1) == 1`` on every weight and activation.
    """
    from sglang.kernels.ops.gemm.sm120_online_fp8 import gated_by_fast_paths

    if not gated_by_fast_paths(envs.SGLANG_MTP_FC_GEMV.get()):
        # Unset follows the Flash-Next/SM120 default selection (this file's
        # entry fusion is Flash-Next-draft-only by construction); a saved
        # explicit true/false remains the private hatch.
        return False
    for tensor in (normed_embeds, hidden_rows):
        if (
            tensor.dim() != 2
            or tensor.dtype is not torch.bfloat16
            or tensor.stride(1) != 1
        ):
            return False
    for weight in (embed_weight, hidden_weight):
        if (
            weight.dim() != 2
            or weight.dtype is not torch.bfloat16
            or weight.stride(1) != 1
        ):
            return False
    if hidden_rows.shape[0] > _GEMV_DONOR_MAX_M:
        # hidden_rows is tokens * hc with hc >= 1, so this bounds both GEMVs.
        return False
    return _bf16_entry_gemm_target_ok(
        normed_embeds, embed_weight, hidden_rows, hidden_weight
    )


class Qwen4ExpForCausalLMMTP(Qwen3_5ForCausalLMMTP):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.is_multimodal = hasattr(config, "text_config")
        if self.is_multimodal:
            config = config.text_config

        # Deepcopy so MTP-only mutations below don't leak into the main model.
        config = copy.deepcopy(config)
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]
        config.full_attention_interval = 1
        config.ple_layer_ids = []

        quant_config = _mtp_quant_config(quant_config)

        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self._mtp_input_fusion = self._init_mtp_input_fusion(config)

        self.model = Qwen4ExpModel(
            config,
            quant_config,
            prefix=add_prefix("mtp", prefix),
            is_nextn=True,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_parallel().config.enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    def _init_pre_fc_norms(self, config: PretrainedConfig) -> None:
        self.pre_fc_norm_embedding = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        hidden_norm_size = (
            self.hc_count * config.hidden_size
            if self.hc_count > 1
            else config.hidden_size
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            hidden_norm_size, eps=config.rms_norm_eps
        )

    def _init_linear_projections(self, config: PretrainedConfig) -> None:
        self.fc_embedding = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.fc_hidden = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _init_standard_fusion(self, config: PretrainedConfig):
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self._init_pre_fc_norms(config)
        return self._fuse_standard

    def _init_mtp_input_fusion(self, config: PretrainedConfig):
        if self.hc_count <= 1:
            return self._init_standard_fusion(config)

        self._init_linear_projections(config)
        self._init_pre_fc_norms(config)
        return self._fuse_residual_linear_shared

    def _fuse_residual_linear_shared(
        self, input_embeds: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # Donor parity (aiueo52/sglang-rtxpro6000 @5105985, qwen4_exp_mtp.py
        # _fuse_residual_linear_shared; companion W8A16 dense-GEMV series):
        # under SGLANG_MTP_FC_GEMV the decode-size entry GEMMs run through the
        # resident donor bf16 GEMV instead of cuBLAS, which picks poor kernels
        # for these skinny [<=16, hidden] x [hidden, hidden] shapes.  Weights
        # stay BF16; _mtp_fc_gemv_supported guards every case the wrapper
        # cannot execute, and those keep the original path unchanged.
        normed_embeds = self.pre_fc_norm_embedding(input_embeds)
        orig_shape = hidden_states.shape
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        decoder_view = hidden_states.view(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        rows = decoder_view.reshape(-1, self.hidden_size)
        if (
            self.fc_embedding.bias is None
            and self.fc_hidden.bias is None
            and _mtp_fc_gemv_supported(
                normed_embeds,
                self.fc_embedding.weight,
                rows,
                self.fc_hidden.weight,
            )
        ):
            from sglang.srt.layers.quantization.w8a16_gemv import bf16_gemv

            embed_proj = bf16_gemv(normed_embeds, self.fc_embedding.weight)
            encoder_inputs = bf16_gemv(rows.contiguous(), self.fc_hidden.weight).view(
                decoder_view.shape
            )
        else:
            embed_proj = self.fc_embedding(normed_embeds)
            encoder_inputs = self.fc_hidden(decoder_view)
        return (embed_proj.unsqueeze(-2) + encoder_inputs).view(orig_shape)

    def _fuse_standard(
        self, input_embeds: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        input_embeds = self.pre_fc_norm_embedding(input_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        return self.fc(torch.cat((input_embeds, hidden_states), dim=-1))

    def _npu_quant_context(self):
        exit_stack = ExitStack()
        if (
            is_npu()
            and self.quant_config is None
            and get_model().quantization is not None
        ):
            exit_stack.enter_context(envs.SGLANG_DEEPEP_BF16_DISPATCH.override(True))
            exit_stack.enter_context(
                envs.DEEP_NORMAL_MODE_USE_INT8_QUANT.override(False)
            )
        return exit_stack

    def _prepare_input_embeds(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert input_embeds is None
        input_embeds = forward_batch.mm_input_embeds
        if (
            forward_batch.forward_mode.is_extend()
            and forward_batch.contains_mm_inputs()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            assert input_embeds is not None
            last_indices = (
                forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
            ).long()
            input_embeds[last_indices] = self.model.embed_tokens(
                input_ids[last_indices]
            )
        if input_embeds is None:
            input_embeds = self.model.embed_tokens(input_ids)
        return input_embeds

    def _set_hc_logits_hidden_states(
        self,
        logits_output,
        hc_hidden_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> None:
        if hc_hidden_states is None:
            return

        # EAGLE v2 stores one hidden state per request in the future map.
        # When draft extend emits a token-shaped HC tensor, keep only the
        # last token per request so the overlap cache sees [bs, hidden].
        if (
            not forward_batch.forward_mode.is_draft_extend_v2()
            and forward_batch.extend_seq_lens is not None
            and hc_hidden_states.shape[0] != forward_batch.extend_seq_lens.shape[0]
        ):
            last_index = (
                torch.cumsum(forward_batch.extend_seq_lens.to(torch.int64), dim=0) - 1
            )
            hc_hidden_states = hc_hidden_states[last_index]

        assert hc_hidden_states.shape[-1] == self.hc_count * self.hidden_size
        logits_output.hidden_states = hc_hidden_states

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        with self._npu_quant_context():
            input_embeds = self._prepare_input_embeds(
                input_ids, forward_batch, input_embeds
            )
            hidden_states = forward_batch.spec_info.hidden_states
            if not forward_batch.forward_mode.is_idle():
                hidden_states = self._mtp_input_fusion(input_embeds, hidden_states)

            with get_global_expert_distribution_recorder().disable_this_region():
                model_output = self.model(
                    input_ids,
                    positions,
                    forward_batch,
                    hidden_states,
                )

            hc_hidden_states = None
            if isinstance(model_output, tuple):
                hidden_states, hc_hidden_states = model_output
            else:
                hidden_states = model_output

        logits_output = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
        self._set_hc_logits_hidden_states(
            logits_output, hc_hidden_states, forward_batch
        )
        return logits_output


EntryClass = [Qwen4ExpForCausalLMMTP]
