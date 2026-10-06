"""Inference-only Qwen4-Exp MTP speculative decoding."""

# The draft-vocab placeholder path below (`_draft_vocab_weights_are_shared`,
# `_build_with_placeholder_vocab_weight`, `_is_draft_vocab_weight` and the
# `_Qwen4ExpDraftModel` embedding hook) is adapted from
# aiueo52/sglang-rtxpro6000 (flash-next-fast), file
# python/sglang/srt/models/qwen4_exp_mtp.py, donor commit
# 8bc49eff197d92930ddb7e6d00788761ea78021f (blob
# f55c4d61cc08b7e29c15a553a02b154d505660e0; the same file at the donor full
# snapshot 5105985116eb00dea8e6138aabeb5363387cb9de is blob
# bb1cc38f78701f0077bbd6ea34a743583ada0e82), Apache-2.0. Selective reuse: the
# donor's gate let any speculative algorithm through, this one requires the
# workers that really hand in the target's tensors, and the donor's
# `_fc_embed_table` entry projection / MTP-entry GEMV -- and the ~2.4 GB once
# reported for it -- are its own work, not taken here (and not established for
# this checkpoint, TP layout or quantization).

import copy
import logging
from contextlib import ExitStack
from typing import Callable, Iterable, Optional, Tuple

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
from sglang.srt.runtime_context import get_model, get_parallel, get_spec
from sglang.srt.utils import add_prefix, is_npu, set_weight_attrs

logger = logging.getLogger(__name__)


def _draft_vocab_weights_are_shared(quant_config) -> bool:
    """Whether the worker will hand this draft the target's vocab tensors.

    The EAGLE-family (NEXTN resolves to EAGLE) and FROZEN_KV_MTP workers both
    overwrite the draft's embedding and head before its first forward
    (`eagle_worker_v2.init_lm_head` / `frozen_kv_mtp_worker_v2` ->
    `set_embed_and_head`), so building the two [vocab, hidden] tables here only
    for that handoff to `del` and replace them is start-up peak for nothing.
    EAGLE3 keeps a draft-owned head unless it loads the target's
    (`load_lm_head_from_target`), STANDALONE and DFLASH drafts own their vocab
    outright and a quantized head carries packed/scaled parameters rather than
    one swappable `weight`: all of those stay on the full-size path.
    """
    if not envs.SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS.get():
        return False
    if quant_config is not None:
        return False

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
    return algo in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.FROZEN_KV_MTP)


def _build_with_placeholder_vocab_weight(build: Callable[[], nn.Module]) -> nn.Module:
    """Build a vocab-sized module without materialising its [vocab, hidden] table.

    Construction runs on the meta device, so the layout metadata (shard indices,
    padded vocab, num_embeddings, weight loader) is the real one -- the target
    lays its own tensors out identically and the vocab-parallel masking has to
    agree -- while the table costs nothing. The module then gets a 1-row
    parameter on the ambient device for `set_embed_and_head` to delete and
    replace.
    """
    with torch.device("meta"):
        module = build()
    weight = getattr(module, "weight", None)
    assert isinstance(weight, nn.Parameter) and weight.is_meta, (
        f"{type(module).__name__} does not own a single meta weight parameter; "
        "the draft vocab placeholder path only covers the unquantized layout"
    )
    placeholder = nn.Parameter(
        torch.empty(1, weight.shape[1], dtype=weight.dtype),
        requires_grad=False,
    )
    set_weight_attrs(
        placeholder,
        {"input_dim": 1, "output_dim": 0, "weight_loader": module.weight_loader},
    )
    module.register_parameter("weight", placeholder)
    return module


def _is_draft_vocab_weight(name: str) -> bool:
    """Checkpoint tensors that would target a placeholder (see above)."""
    if "mtp" not in name:
        return False
    return (
        name.endswith("embed_tokens.weight")
        or name.endswith("lm_head.weight")
        or "shared_head.head" in name
    )


class _Qwen4ExpDraftModel(Qwen4ExpModel):
    """Draft backbone whose input embedding is a placeholder.

    Only used when the target's table gets shared in; see
    `_draft_vocab_weights_are_shared`.
    """

    def _build_embed_tokens(self, config) -> nn.Module:
        return _build_with_placeholder_vocab_weight(
            lambda: super(_Qwen4ExpDraftModel, self)._build_embed_tokens(config)
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

        self.skip_vocab_weights = _draft_vocab_weights_are_shared(quant_config)
        model_cls = _Qwen4ExpDraftModel if self.skip_vocab_weights else Qwen4ExpModel
        self.model = model_cls(
            config,
            quant_config,
            prefix=add_prefix("mtp", prefix),
            is_nextn=True,
        )

        def build_lm_head() -> nn.Module:
            return ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("model.shared_head.head", prefix),
                use_attn_tp_group=get_parallel().config.enable_dp_lm_head,
            )

        self.lm_head = (
            _build_with_placeholder_vocab_weight(build_lm_head)
            if self.skip_vocab_weights
            else build_lm_head()
        )
        if self.skip_vocab_weights:
            # Nominal (unsharded, unquantized) size of the two tables left
            # unallocated; the per-rank figure depends on the vocab shard.
            logger.info(
                "MTP draft embed_tokens / lm_head are 1-row placeholders "
                "(nominally %.2f GiB of [vocab, hidden] tables); the target's "
                "tensors are shared in before the first forward",
                2 * config.vocab_size * config.hidden_size * 2 / (1 << 30),
            )
        self.logits_processor = LogitsProcessor(config)

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        if self.skip_vocab_weights:
            weights = self._drop_draft_vocab_weights(weights)
        return super().load_weights(weights, is_mtp)

    @staticmethod
    def _drop_draft_vocab_weights(
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """Drop the rows that would target a placeholder vocab table.

        The worker hands in the target's embedding and head either way, so the
        checkpoint's own copy would be overwritten -- but say so, a draft that
        really needs an independent table belongs on the full-size path.
        """
        for name, weight in weights:
            if _is_draft_vocab_weight(name):
                logger.warning_once(
                    "MTP draft checkpoint tensor %r targets a placeholder "
                    "(SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS): the worker shares the "
                    "target's embedding / head in, so this row is discarded as "
                    "before; turn the flag off to keep a draft-owned table.",
                    name,
                )
                continue
            yield name, weight

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
        input_embeds = self.fc_embedding(self.pre_fc_norm_embedding(input_embeds))
        orig_shape = hidden_states.shape
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        decoder_view = hidden_states.view(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        encoder_inputs = self.fc_hidden(decoder_view)
        return (input_embeds.unsqueeze(-2) + encoder_inputs).view(orig_shape)

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
