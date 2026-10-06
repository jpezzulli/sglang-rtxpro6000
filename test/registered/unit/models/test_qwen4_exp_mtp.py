import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers import vocab_parallel_embedding as vpe
from sglang.srt.layers.quantization.nvfp4_online import NvFp4OnlineConfig
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import _resolve_explicit_draft_quant_config
from sglang.srt.models import qwen4_exp, qwen4_exp_mtp
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP, _mtp_quant_config
from sglang.srt.models.qwen4_exp_mtp import (
    Qwen4ExpForCausalLMMTP,
    _placeholder_vocab_weight,
    _Qwen4ExpDraftModel,
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
        module = None
        if type(self).placeholder_embed:
            module = _placeholder_vocab_weight(
                lambda: VocabParallelEmbedding(*args, **kwargs)
            )
        if module is None:
            module = VocabParallelEmbedding(*args, **kwargs)
        self.embed_tokens = module


class _FakeDraftBackbone(_FakeBackbone):
    placeholder_embed = True


def _shard_indices(module):
    return vars(module.shard_indices)


def _is_placeholder(param):
    return bool(getattr(param, "is_draft_vocab_placeholder", False))


def _owned_bytes(module):
    """Bytes of materialised tensor storage this module owns (meta costs nothing)."""
    return sum(
        t.numel() * t.element_size()
        for t in list(module.parameters()) + list(module.buffers())
        if t.device.type != "meta"
    )


def _packed_head_config():
    """A quant handle that really hands the head packed / scaled parameters."""

    class _PackedMethod:
        def create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        ):
            for name in ("weight_packed", "weight_scale"):
                layer.register_parameter(
                    name, nn.Parameter(torch.empty(()), requires_grad=False)
                )

    class _PackedConfig:
        def get_name(self):
            return "packed_test"

        def get_quant_method(self, layer, prefix):
            return _PackedMethod()

    return _PackedConfig()


def _scaled_weight_head_config():
    """A quant handle that keeps one `weight` but answers through scales."""

    class _ScaledWeightMethod:
        def create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        ):
            layer.register_parameter(
                "weight",
                nn.Parameter(
                    torch.empty(sum(output_partition_sizes), input_size_per_partition),
                    requires_grad=False,
                ),
            )
            layer.register_buffer(
                "weight_scale_inv", torch.empty(output_size, input_size)
            )

    class _ScaledWeightConfig:
        def get_name(self):
            return "scaled_weight_test"

        def get_quant_method(self, layer, prefix):
            return _ScaledWeightMethod()

    return _ScaledWeightConfig()


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

    def _gate(self, *, flag, algorithm):
        with (
            patch.dict(os.environ, _flag_env(flag)),
            patch.object(
                qwen4_exp_mtp,
                "get_spec",
                return_value=SimpleNamespace(speculative_algorithm=algorithm),
            ),
        ):
            return qwen4_exp_mtp._draft_vocab_weights_are_shared()

    def _build(
        self,
        *,
        flag,
        algorithm="EAGLE",
        tp_size=1,
        tp_rank=0,
        quant_config=None,
        config=None,
    ):
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
            return Qwen4ExpForCausalLMMTP(config or _config(), quant_config)

    def test_gate_needs_the_flag_and_a_worker_that_hands_the_tensors_in(self):
        self.assertFalse(self._gate(flag=False, algorithm="EAGLE"))
        # NEXTN resolves to EAGLE by the time the draft is built.
        self.assertTrue(self._gate(flag=True, algorithm="EAGLE"))
        self.assertTrue(self._gate(flag=True, algorithm="FROZEN_KV_MTP"))
        for algorithm in ("EAGLE3", "DFLASH", "STANDALONE", "NGRAM", None):
            with self.subTest(algorithm=algorithm):
                self.assertFalse(self._gate(flag=True, algorithm=algorithm))

    def test_served_draft_quantization_still_gets_the_placeholders(self):
        """The real Orca/Flash-Next handle: NVFP4 experts, unquantized vocab.

        `--speculative-draft-model-quantization unquant` on a serialized NVFP4
        checkpoint resolves to the online-expert ModelOpt config, which answers
        None for the head: the vocab tensors stay plain parameters, so the
        savings have to apply on the served path (the quantized *experts* are a
        different matter and stay untouched either way).
        """
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config

        checkpoint = ModelOptFp4Config.from_config(
            {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "ignore": ["lm_head", "mtp.*", "model.mtp.*"],
                "packed_modules_mapping": {"gate_up_proj": ["gate_proj", "up_proj"]},
            }
        )
        self.assertTrue(checkpoint.is_checkpoint_nvfp4_serialized)
        self.assertTrue(checkpoint.is_layer_excluded("mtp.layers.0.mlp.experts"))
        draft_quant = _resolve_explicit_draft_quant_config(
            SimpleNamespace(
                is_draft_model=True,
                is_draft_quantization_explicit=True,
                quantization="modelopt_fp4",
            ),
            checkpoint,
        )
        self.assertIsInstance(draft_quant, NvFp4OnlineConfig)
        self.assertEqual(draft_quant.get_name(), "modelopt_fp4")
        self.assertFalse(draft_quant.is_checkpoint_nvfp4_serialized)
        # The MTP constructor keeps that handle -- it only drops out quantization
        # for serialized / modelopt_mixed checkpoints.
        self.assertIs(_mtp_quant_config(draft_quant), draft_quant)

        draft = self._build(flag=True, quant_config=draft_quant)
        self.assertTrue(draft.skip_vocab_weights)
        self.assertIs(draft.quant_config, draft_quant)
        self.assertIsInstance(draft.lm_head.quant_method, UnquantizedEmbeddingMethod)
        self.assertEqual(tuple(draft.lm_head.weight.shape), (1, HIDDEN))
        self.assertEqual(tuple(draft.model.embed_tokens.weight.shape), (1, HIDDEN))
        full = self._build(flag=False, quant_config=draft_quant)
        self.assertEqual(_shard_indices(draft.lm_head), _shard_indices(full.lm_head))
        freed = _owned_bytes(full) - _owned_bytes(draft)
        self.assertGreater(freed, 3 * TABLE_BYTES // 2)

    def test_packed_or_scaled_head_keeps_its_real_table(self):
        for quant_config in (_packed_head_config(), _scaled_weight_head_config()):
            with self.subTest(config=type(quant_config).__name__):
                draft = self._build(flag=True, quant_config=quant_config)
                self.assertFalse(draft.skip_vocab_weights)
                self.assertIs(type(draft.model), _FakeBackbone)
                self.assertEqual(
                    tuple(draft.model.embed_tokens.weight.shape), (VOCAB, HIDDEN)
                )
                # Nothing was placeheld: no parameter carries the marker, so the
                # loaders keep their normal contracts.
                self.assertFalse(any(_is_placeholder(p) for p in draft.parameters()))

    def test_quantized_head_parameters_are_not_left_behind_by_a_placeholder(self):
        packed = self._build(flag=True, quant_config=_packed_head_config())
        self.assertFalse(hasattr(packed.lm_head, "weight"))
        self.assertEqual(
            sorted(n for n, _ in packed.lm_head.named_parameters(recurse=False)),
            ["weight_packed", "weight_scale"],
        )
        scaled = self._build(flag=True, quant_config=_scaled_weight_head_config())
        self.assertEqual(tuple(scaled.lm_head.weight.shape), (VOCAB, HIDDEN))
        self.assertIn("weight_scale_inv", dict(scaled.lm_head.named_buffers()))

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
                self.assertEqual(draft.lm_head.num_embeddings_per_partition, VOCAB // 2)
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
        rows = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
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
        untouched = ["model.embed_tokens.weight", "mtp.model.layers.0.gate_proj.weight"]
        seen = {}

        def capture(self, weights, is_mtp=False):
            seen["names"] = [name for name, _ in weights]
            return set(seen["names"])

        with patch.object(Qwen3_5ForCausalLMMTP, "load_weights", capture):
            self._build(flag=True).load_weights(iter(rows))
        self.assertEqual(seen["names"], untouched)
        with patch.object(Qwen3_5ForCausalLMMTP, "load_weights", capture):
            self._build(flag=False).load_weights(iter(rows))
        self.assertEqual(seen["names"], [name for name, _ in rows])

    def test_loader_contract_returns_to_the_base_after_the_handoff(self):
        """The row filter lives exactly as long as the placeholder does.

        Before the handoff a draft-vocab row must not be written into the 1-row
        parameter; after `set_embed_and_head` replaces it, the same row has to
        reach the base loader and the target-owned tensor it belongs to, tied or
        untied head and whichever path built the draft.
        """
        payload = torch.full((VOCAB, HIDDEN), 3.0)
        rows = [
            ("mtp.embed_tokens.weight", payload),
            ("mtp.model.shared_head.head.weight", payload),
        ]
        seen = {}

        def capture(self, weights, is_mtp=False):
            seen["names"] = [name for name, _ in weights]
            return set(seen["names"])

        for flag, tied in ((True, False), (True, True), (False, False)):
            with self.subTest(placeholder=flag, tied=tied):
                config = _config()
                config.tie_word_embeddings = tied
                draft = self._build(flag=flag, config=config)
                self.assertEqual(_is_placeholder(draft.lm_head.weight), flag)
                with patch.object(Qwen3_5ForCausalLMMTP, "load_weights", capture):
                    draft.load_weights(iter(rows))
                    during = list(seen["names"])
                    embed = nn.Parameter(torch.zeros(VOCAB, HIDDEN))
                    head = nn.Parameter(torch.zeros(VOCAB, HIDDEN))
                    with (
                        patch("torch.cuda.empty_cache"),
                        patch("torch.cuda.synchronize"),
                    ):
                        draft.set_embed_and_head(embed, head)
                    draft.load_weights(iter(rows))
                    after = list(seen["names"])
                # Nothing is routed into a live placeholder, and no valid row is
                # dropped once the target owns the table.
                self.assertEqual(during, [] if flag else [name for name, _ in rows])
                self.assertEqual(after, [name for name, _ in rows])
                self.assertFalse(_is_placeholder(embed))
                self.assertFalse(_is_placeholder(head))
                # The real base loader writes through to the target's tensor.
                draft.load_weights(iter([("mtp.embed_tokens.weight", payload)]))
                self.assertTrue(torch.equal(embed.data, payload))
                self.assertIs(draft.model.embed_tokens.weight, embed)

    def test_placeholder_builder_keeps_a_real_head_metadata_only(self):
        def build():
            return ParallelLMHead(VOCAB, HIDDEN, prefix="model.shared_head.head")

        with patch.object(vpe, "get_parallel", return_value=_parallel()):
            head = _placeholder_vocab_weight(build)
            allocated = _owned_bytes(head)
        self.assertIsNotNone(head)
        self.assertEqual(tuple(head.weight.shape), (1, HIDDEN))
        self.assertLess(allocated, TABLE_BYTES // 64)
        self.assertEqual(head.weight.shape[1], HIDDEN)
        self.assertFalse(head.weight.requires_grad)
        self.assertTrue(_is_placeholder(head.weight))
        self.assertIs(head.weight.weight_loader.__self__, head)


if __name__ == "__main__":
    unittest.main()
