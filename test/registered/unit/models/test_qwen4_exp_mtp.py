import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers import vocab_parallel_embedding as vpe
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.models import qwen4_exp, qwen4_exp_mtp
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.srt.models.qwen4_exp_mtp import (
    Qwen4ExpForCausalLMMTP,
    _Qwen4ExpDraftModel,
    _build_with_placeholder_vocab_weight,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB = 2048
HIDDEN = 512
TABLE_BYTES = VOCAB * HIDDEN * torch.finfo(torch.get_default_dtype()).bits // 8


def _flag_env(enabled: bool):
    return {"SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS": "1" if enabled else "0"}


def _parallel(tp_size=1, tp_rank=0):
    return SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        attn_tp_size=tp_size,
        attn_tp_rank=tp_rank,
        config=SimpleNamespace(enable_dp_lm_head=False),
    )


def _config():
    return SimpleNamespace(
        hc_count=4,
        hidden_size=HIDDEN,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        tie_word_embeddings=False,
    )


class _FakeBackbone(nn.Module):
    """Stand-in for Qwen4ExpModel: only the vocab table it owns matters here."""

    placeholder_embed = False

    def __init__(self, config, quant_config, prefix="", is_nextn=False):
        super().__init__()
        self.is_nextn = is_nextn
        args = (config.vocab_size, config.hidden_size)
        kwargs = {"org_num_embeddings": config.vocab_size}
        if type(self).placeholder_embed:
            self.embed_tokens = _build_with_placeholder_vocab_weight(
                lambda: VocabParallelEmbedding(*args, **kwargs)
            )
        else:
            self.embed_tokens = VocabParallelEmbedding(*args, **kwargs)


class _FakeDraftBackbone(_FakeBackbone):
    placeholder_embed = True


def _shard_indices(module):
    return vars(module.shard_indices)


def _owned_bytes(module):
    """Bytes of materialised tensor storage this module owns (meta costs nothing)."""
    return sum(
        t.numel() * t.element_size()
        for t in list(module.parameters()) + list(module.buffers())
        if t.device.type != "meta"
    )


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


class TestQwen4ExpMTPDraftVocabWeights(CustomTestCase):
    """The draft's throw-away [vocab, hidden] pair (SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS)."""

    def _gate(self, *, flag, algorithm, quant_config=None):
        with (
            patch.dict(os.environ, _flag_env(flag)),
            patch.object(
                qwen4_exp_mtp,
                "get_spec",
                return_value=SimpleNamespace(speculative_algorithm=algorithm),
            ),
        ):
            return qwen4_exp_mtp._draft_vocab_weights_are_shared(quant_config)

    def _build(self, *, flag, algorithm="EAGLE", tp_size=1, tp_rank=0):
        with (
            patch.dict(os.environ, _flag_env(flag)),
            patch.object(
                qwen4_exp_mtp,
                "get_parallel",
                return_value=_parallel(tp_size, tp_rank),
            ),
            patch.object(
                qwen4_exp_mtp,
                "get_spec",
                return_value=SimpleNamespace(speculative_algorithm=algorithm),
            ),
            patch.object(
                qwen4_exp_mtp,
                "get_pp_group",
                return_value=SimpleNamespace(is_last_rank=True),
            ),
            patch.object(qwen4_exp_mtp, "Qwen4ExpModel", _FakeBackbone),
            patch.object(qwen4_exp_mtp, "_Qwen4ExpDraftModel", _FakeDraftBackbone),
            patch.object(qwen4_exp_mtp, "LogitsProcessor"),
            patch.object(vpe, "get_parallel", return_value=_parallel(tp_size, tp_rank)),
            patch.object(
                Qwen4ExpForCausalLMMTP,
                "_init_mtp_input_fusion",
                return_value=None,
            ),
        ):
            return Qwen4ExpForCausalLMMTP(_config())

    def test_gate_needs_the_flag_and_a_worker_that_hands_the_tensors_in(self):
        self.assertFalse(self._gate(flag=False, algorithm="EAGLE"))
        # NEXTN resolves to EAGLE by the time the draft is built.
        self.assertTrue(self._gate(flag=True, algorithm="EAGLE"))
        self.assertTrue(self._gate(flag=True, algorithm="FROZEN_KV_MTP"))
        for algorithm in ("EAGLE3", "DFLASH", "STANDALONE", "NGRAM", None):
            with self.subTest(algorithm=algorithm):
                self.assertFalse(self._gate(flag=True, algorithm=algorithm))
        # A quantized head carries packed/scaled parameters, not one `weight`.
        self.assertFalse(
            self._gate(flag=True, algorithm="EAGLE", quant_config=object())
        )

    def test_placeholders_replace_the_allocated_vocab_tables(self):
        draft = self._build(flag=True)
        placeholder_bytes = _owned_bytes(draft)
        full = self._build(flag=False)
        full_bytes = _owned_bytes(full)

        self.assertIs(type(draft.model), _FakeDraftBackbone)
        self.assertTrue(draft.model.is_nextn)
        self.assertIs(type(full.model), _FakeBackbone)
        self.assertEqual(tuple(draft.lm_head.weight.shape), (1, HIDDEN))
        self.assertEqual(tuple(draft.model.embed_tokens.weight.shape), (1, HIDDEN))
        self.assertEqual(tuple(full.lm_head.weight.shape), (VOCAB, HIDDEN))
        self.assertEqual(tuple(full.model.embed_tokens.weight.shape), (VOCAB, HIDDEN))
        # The pair of tables really was allocated before, really is not now.
        self.assertGreater(full_bytes, 3 * TABLE_BYTES // 2)
        self.assertLess(placeholder_bytes, TABLE_BYTES // 64)
        # Vocab-parallel layout (masking, sharding, the loaders keyed off it) is
        # the one the target's tensors are laid out for.
        for module, other in (
            (draft.lm_head, full.lm_head),
            (draft.model.embed_tokens, full.model.embed_tokens),
        ):
            for attr in (
                "num_embeddings",
                "org_vocab_size",
                "embedding_dim",
                "num_embeddings_per_partition",
                "tp_size",
            ):
                self.assertEqual(getattr(module, attr), getattr(other, attr))
            self.assertEqual(_shard_indices(module), _shard_indices(other))
            self.assertIsNotNone(module.weight.weight_loader)

    def test_27b_style_dflash_draft_keeps_its_real_tables(self):
        for algorithm in ("DFLASH", "STANDALONE"):
            with self.subTest(algorithm=algorithm):
                draft = self._build(flag=True, algorithm=algorithm)
                self.assertFalse(draft.skip_vocab_weights)
                self.assertIs(type(draft.model), _FakeBackbone)
                self.assertEqual(tuple(draft.lm_head.weight.shape), (VOCAB, HIDDEN))

    def test_placeholder_layout_matches_the_full_layout_at_tp2(self):
        for tp_rank in (0, 1):
            with self.subTest(tp_rank=tp_rank):
                draft = self._build(flag=True, tp_size=2, tp_rank=tp_rank)
                full = self._build(flag=False, tp_size=2, tp_rank=tp_rank)
                self.assertEqual(tuple(draft.lm_head.weight.shape), (1, HIDDEN))
                self.assertEqual(
                    _shard_indices(draft.lm_head), _shard_indices(full.lm_head)
                )
                self.assertEqual(
                    draft.lm_head.num_embeddings_per_partition, VOCAB // 2
                )
                self.assertEqual(
                    draft.model.embed_tokens.num_embeddings_per_partition, VOCAB // 2
                )

    def test_draft_backbone_hook_placeholders_the_real_embedding(self):
        stub = object.__new__(_Qwen4ExpDraftModel)
        config = SimpleNamespace(vocab_size=VOCAB, hidden_size=HIDDEN)
        with (
            patch.object(vpe, "get_parallel", return_value=_parallel()),
            patch.object(qwen4_exp, "is_dp_attention_enabled", return_value=False),
        ):
            module = _Qwen4ExpDraftModel._build_embed_tokens(stub, config)
        self.assertIsInstance(module, VocabParallelEmbedding)
        self.assertEqual(tuple(module.weight.shape), (1, HIDDEN))
        self.assertEqual(module.num_embeddings, VOCAB)

    def test_handoff_leaves_one_owner_of_each_target_tensor(self):
        rows = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(
            VOCAB, HIDDEN
        )
        embed = nn.Parameter(rows)
        head = nn.Parameter(-rows.clone())
        for flag in (True, False):
            with self.subTest(placeholder=flag):
                draft = self._build(flag=flag)
                with patch("torch.cuda.empty_cache"), patch("torch.cuda.synchronize"):
                    draft.set_embed_and_head(embed, head)
                self.assertIs(draft.model.embed_tokens.weight, embed)
                self.assertIs(draft.lm_head.weight, head)
                got_embed, got_head = draft.get_embed_and_head()
                self.assertIs(got_embed, embed)
                self.assertIs(got_head, head)
                self.assertEqual(tuple(draft.lm_head.weight.shape), (VOCAB, HIDDEN))

    def test_checkpoint_rows_for_a_placeholder_never_reach_its_loader(self):
        rows = [
            ("mtp.embed_tokens.weight", torch.zeros(VOCAB, HIDDEN)),
            ("mtp.model.shared_head.head.weight", torch.zeros(VOCAB, HIDDEN)),
            ("model.embed_tokens.weight", torch.zeros(VOCAB, HIDDEN)),
            ("mtp.model.layers.0.gate_proj.weight", torch.zeros(4, HIDDEN)),
        ]
        seen = {}

        def capture(self, weights, is_mtp=False):
            seen["names"] = [name for name, _ in weights]
            return set(seen["names"])

        with patch.object(Qwen3_5ForCausalLMMTP, "load_weights", capture):
            self._build(flag=True).load_weights(iter(rows))
        self.assertEqual(
            seen["names"],
            ["model.embed_tokens.weight", "mtp.model.layers.0.gate_proj.weight"],
        )
        with patch.object(Qwen3_5ForCausalLMMTP, "load_weights", capture):
            self._build(flag=False).load_weights(iter(rows))
        self.assertEqual(len(seen["names"]), len(rows))

    def test_placeholder_builder_keeps_a_real_head_metadata_only(self):
        def build():
            return ParallelLMHead(VOCAB, HIDDEN, prefix="model.shared_head.head")

        with patch.object(vpe, "get_parallel", return_value=_parallel()):
            head = _build_with_placeholder_vocab_weight(build)
            allocated = _owned_bytes(head)
        self.assertEqual(tuple(head.weight.shape), (1, HIDDEN))
        self.assertLess(allocated, TABLE_BYTES // 64)
        self.assertEqual(head.weight.shape[1], HIDDEN)
        self.assertFalse(head.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
