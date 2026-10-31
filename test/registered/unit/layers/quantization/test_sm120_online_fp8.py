# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for the Flash-Next SM120 online row-wise FP8 conversion.

Everything here is the part that decides *what* becomes FP8 and how the values
and scales stay paired. The GEMMs those decisions feed (per-token/per-channel
FP8 for the projections, the low-row head kernel) are GPU-only and covered by
the launch checks; this file needs only torch, so it runs on a CPU worker.

The module under test is imported by file when the serving runtime's optional
dependencies are absent: the conversion helpers are pure torch on purpose, so a
CPU-only host can still exercise the selection and scale-pairing invariants.
"""

import importlib.util
import os
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _load_under_test():
    try:
        from sglang.kernels.ops.gemm import sm120_online_fp8 as module

        return module
    except ModuleNotFoundError:
        path = (
            Path(__file__)
            .resolve()
            .parents[5]
            .joinpath("python/sglang/kernels/ops/gemm/sm120_online_fp8.py")
        )
        spec = importlib.util.spec_from_file_location("sm120_online_fp8", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


fp8 = _load_under_test()


def _convert_head(self):
    """The conversion on its own, so a missing call site reads as a failure
    rather than as an absent helper."""
    fp8.replace_linear_weight_rowwise_fp8(self.lm_head)


class _Unquantized:
    """Stand-in for UnquantizedLinearMethod (the walker takes it as a parameter)."""


class _AlreadyQuantized:
    """Stand-in for a checkpoint-declared method (NVFP4 / FP8 / W4A16)."""


class _MoeBlock(nn.Module):
    """Stand-in for FusedMoE: an excluded subtree that must not be descended."""


def _linear(rows, columns, dtype=torch.bfloat16, method=_Unquantized()):
    layer = nn.Linear(columns, rows, bias=False, dtype=dtype)
    layer.quant_method = method
    return layer


def _candidate_tree():
    """A miniature Flash-Next layer: eligible, protected and already-quantized."""
    root = nn.Module()
    root.self_attn = nn.Module()
    root.self_attn.qkv_proj = _linear(512, 256)
    root.self_attn.o_proj = _linear(256, 256)
    # Router: BF16 and unquantized, but never eligible by name.
    root.gate = _linear(128, 256)
    root.mlp = nn.Module()
    # Too narrow in K to quantize under the accepted shape floor.
    root.mlp.down_proj = _linear(256, 31)
    # Below the row/column floor.
    root.mlp.small_proj = _linear(64, 128)
    # The checkpoint already picked a representation for this one.
    root.mlp.expert_proj = _linear(512, 256, method=_AlreadyQuantized())
    root.mlp.experts = _MoeBlock()
    root.mlp.experts.gate_up_proj = _linear(512, 256)
    return root


class TestSwitchResolution(unittest.TestCase):
    def test_automatic_selection_needs_exact_sm120_and_flash_next_metadata(self):
        cases = [
            # (requested, cuda, capability, metadata, expected)
            (None, True, (12, 0), True, True),
            (None, True, (12, 0), False, False),
            (None, True, (12, 1), True, False),
            (None, True, (9, 0), True, False),
            (None, False, None, True, False),
            (False, True, (12, 0), True, False),
            (True, True, (12, 0), False, True),
        ]
        for requested, cuda, capability, eligible, expected in cases:
            with self.subTest(requested=requested, capability=capability):
                self.assertEqual(
                    fp8._resolve_switch(
                        requested,
                        cuda_available=cuda,
                        capability=capability,
                        model_eligible=eligible,
                    ),
                    expected,
                )

    def test_explicit_request_on_unsupported_hardware_fails_loud(self):
        with self.assertRaises(RuntimeError):
            fp8._resolve_switch(
                True, cuda_available=False, capability=None, model_eligible=True
            )
        with self.assertRaises(RuntimeError):
            fp8._resolve_switch(
                True, cuda_available=True, capability=(9, 0), model_eligible=True
            )

    def test_blank_environment_reads_as_automatic(self):
        try:
            import sglang.srt.environ  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("needs the serving runtime's env registry")
        os.environ["SGLANG_SM120_ONLINE_MXFP8"] = "  "
        try:
            self.assertIsNone(fp8._env_request())
        finally:
            del os.environ["SGLANG_SM120_ONLINE_MXFP8"]

    def test_cache_identity_follows_the_effective_representation(self):
        on = fp8.resolve_precision(
            None, cuda_available=True, capability=(12, 0), model_eligible=True
        )
        off = fp8.resolve_precision(
            None, cuda_available=True, capability=(9, 0), model_eligible=True
        )
        self.assertEqual(on, fp8.ROWWISE_FP8_PRECISION)
        self.assertEqual(off, fp8.OFF_PRECISION)
        self.assertNotEqual(on, "true")
        self.assertNotEqual(on, "false")
        # Automatic-off and explicit-off name the same untouched layout.
        self.assertEqual(
            off,
            fp8.resolve_precision(
                False, cuda_available=True, capability=(12, 0), model_eligible=True
            ),
        )


class TestCheckpointEligibility(unittest.TestCase):
    def test_recognizes_the_family_from_metadata_not_a_path(self):
        self.assertTrue(fp8.flash_next_metadata({"model_type": "qwen4_exp_text"}))
        self.assertTrue(
            fp8.flash_next_metadata(
                {"architectures": ["Qwen4ExpForConditionalGeneration"]}
            )
        )
        self.assertTrue(
            fp8.flash_next_metadata(
                {"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 48}}
            )
        )
        self.assertFalse(fp8.flash_next_metadata({"model_type": "qwen3_next"}))
        self.assertFalse(fp8.flash_next_metadata(None))

    def test_non_bf16_compute_dtype_and_tied_head_stay_off(self):
        meta = {"model_type": "qwen4_exp_text"}
        self.assertTrue(fp8.checkpoint_eligible(meta, torch.bfloat16))
        self.assertFalse(fp8.checkpoint_eligible(meta, torch.float16))
        self.assertFalse(fp8.checkpoint_eligible(meta, torch.float32))
        tied = {"model_type": "qwen4_exp_text", "tie_word_embeddings": True}
        self.assertFalse(fp8.checkpoint_eligible(tied, torch.bfloat16))
        nested = {
            "text_config": {"model_type": "qwen4_exp_text", "tie_word_embeddings": True}
        }
        self.assertFalse(fp8.checkpoint_eligible(nested, torch.bfloat16))

    def test_unknown_checkpoint_keeps_its_own_layout(self):
        # An ineligible checkpoint selects nothing and converts nothing, rather
        # than booting into a representation nobody asked for.
        self.assertFalse(
            fp8.configure_online_fp8(
                None,
                cuda_available=True,
                capability=(12, 0),
                model_eligible=False,
            )
        )
        self.assertFalse(fp8.online_fp8_enabled())
        self.assertEqual(
            fp8.install_rowwise_fp8_on_eligible_linears(
                _candidate_tree(), enabled=False, unquantized_method_type=_Unquantized
            ),
            [],
        )


class TestRowwiseQuantization(unittest.TestCase):
    def test_rejects_anything_but_a_2d_bf16_weight(self):
        with self.assertRaises(TypeError):
            fp8.quantize_rowwise_fp8(torch.zeros(64, 64, dtype=torch.float16))
        with self.assertRaises(TypeError):
            fp8.quantize_rowwise_fp8(torch.zeros(64, dtype=torch.bfloat16))

    def test_codes_and_row_scales_round_trip(self):
        torch.manual_seed(0)
        weight = torch.randn(64, 256, dtype=torch.bfloat16)
        quantized, scale = fp8.quantize_rowwise_fp8(weight)
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(tuple(scale.shape), (64,))
        restored = (quantized.float() * scale[:, None]).to(torch.bfloat16)
        self.assertEqual(restored.dtype, torch.bfloat16)
        # E4M3 carries 3 mantissa bits, so a row's worst relative error is on
        # the order of 2^-4; 2^-3 leaves room for the BF16 round trip.
        self.assertLess(
            (restored.float() - weight.float()).abs().max().item(),
            (weight.float().abs().max() * 2**-3).item(),
        )
        self.assertGreater(scale.min().item(), 0.0)

    def test_missing_or_mismatched_scale_fails_loud(self):
        quantized, scale = fp8.quantize_rowwise_fp8(
            torch.randn(16, 64, dtype=torch.bfloat16)
        )
        with self.assertRaises(RuntimeError):
            fp8.require_rowwise_scale(quantized)
        setattr(quantized, fp8._SCALE_ATTR, scale[:-1])
        with self.assertRaises(RuntimeError):
            fp8.require_rowwise_scale(quantized)
        with self.assertRaises(RuntimeError):
            fp8.require_rowwise_scale(torch.zeros(16, 64, dtype=torch.bfloat16))


class TestHeadWeightLifecycle(unittest.TestCase):
    def setUp(self):
        self.linear = nn.Linear(256, 512, bias=False, dtype=torch.bfloat16)
        self.original = self.linear.weight
        self.reference = self.original.detach().clone().float()
        # sglang layers bind a weight_loader onto every parameter they load.
        setattr(
            self.original,
            "weight_loader",
            lambda param, loaded, *a, **kw: param.data.copy_(loaded),
        )

    def test_converts_once_and_preserves_loader_metadata(self):
        setattr(self.original, "shard_marker", "vocab-shard-0")
        freed = fp8.replace_linear_weight_rowwise_fp8(self.linear)
        weight = self.linear.weight
        self.assertEqual(freed, self.reference.numel() * 2)
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(weight.shape), (512, 256))
        self.assertEqual(getattr(weight, "shard_marker"), "vocab-shard-0")
        self.assertTrue(fp8.is_rowwise_fp8_weight(weight))
        scale = fp8.require_rowwise_scale(weight)
        self.assertEqual(tuple(scale.shape), (512,))
        # One FP32 scale per output row, not one per tensor.
        self.assertFalse(torch.allclose(scale, scale[0].expand_as(scale)))
        # Repeated post-load hooks are a no-op, never a requantization.
        self.assertEqual(fp8.replace_linear_weight_rowwise_fp8(self.linear), 0)
        self.assertIs(self.linear.weight, weight)

    def test_reload_quantizes_values_and_scales_together(self):
        fp8.replace_linear_weight_rowwise_fp8(self.linear)
        weight = self.linear.weight
        stale_scale = fp8.require_rowwise_scale(weight).clone()
        updated = torch.randn(512, 256, dtype=torch.bfloat16)
        self.linear.weight.weight_loader(weight, updated)
        self.assertEqual(self.linear.weight.dtype, torch.float8_e4m3fn)
        restored = fp8.dequantize_rowwise_weight(self.linear.weight).float()
        self.assertLess(
            (restored - updated.float()).abs().max().item(),
            (updated.float().abs().max() * 2**-3).item(),
        )
        # The scales describe the new row maxima, not the ones that were replaced.
        fresh = fp8.require_rowwise_scale(self.linear.weight)
        self.assertFalse(torch.allclose(fresh, stale_scale))
        expected = updated.float().abs().amax(dim=1).clamp_min(1e-8) / 448.0
        self.assertTrue(torch.allclose(fresh, expected, rtol=1e-3))

    def test_non_bf16_resident_weight_is_refused(self):
        self.linear.weight = nn.Parameter(
            torch.zeros(512, 256, dtype=torch.float16), requires_grad=False
        )
        with self.assertRaises(RuntimeError):
            fp8.replace_linear_weight_rowwise_fp8(self.linear)

    def test_row_selection_carries_matching_scales(self):
        fp8.replace_linear_weight_rowwise_fp8(self.linear)
        weight = self.linear.weight
        scales_before = fp8.require_rowwise_scale(weight).clone()
        rows = torch.tensor([7, 0, 511], dtype=torch.long)
        selected = fp8.select_rowwise_weight_rows(weight, rows)
        self.assertEqual(tuple(selected.shape), (3, 256))
        self.assertTrue(
            torch.equal(fp8.require_rowwise_scale(selected), scales_before[rows])
        )
        # The target head itself is untouched by a draft's row selection.
        self.assertIs(self.linear.weight, weight)
        self.assertEqual(tuple(fp8.require_rowwise_scale(weight).shape), (512,))


class _FakePostLoadLinear(nn.Module):
    """Stands in for a LinearBase whose FP8 layout the method must verify."""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight


class TestEligibleLinearSelection(unittest.TestCase):
    def _install(self, root):
        return fp8.install_rowwise_fp8_on_eligible_linears(
            root,
            enabled=True,
            method_factory=lambda: "fp8-method",
            unquantized_method_type=_Unquantized,
            excluded_module_type=(_MoeBlock,),
        )

    def test_selects_only_eligible_unquantized_projections(self):
        root = _candidate_tree()
        converted = self._install(root)
        self.assertEqual(sorted(converted), ["self_attn.o_proj", "self_attn.qkv_proj"])
        self.assertEqual(root.self_attn.qkv_proj.quant_method, "fp8-method")
        # Protected and ineligible modules keep whatever method they had.
        self.assertIsInstance(root.gate.quant_method, _Unquantized)
        self.assertIsInstance(root.mlp.down_proj.quant_method, _Unquantized)
        self.assertIsInstance(root.mlp.small_proj.quant_method, _Unquantized)
        self.assertIsInstance(root.mlp.expert_proj.quant_method, _AlreadyQuantized)
        # The excluded subtree is not visited at all.
        self.assertIsInstance(root.mlp.experts.gate_up_proj.quant_method, _Unquantized)

    def test_excluded_subtree_is_not_converted(self):
        root = _candidate_tree()
        self.assertNotIn("mlp.experts.gate_up_proj", self._install(root))

    def test_a_bf16_weight_is_required_at_selection_time(self):
        root = _candidate_tree()
        root.self_attn.qkv_proj = _linear(512, 256, dtype=torch.float16)
        self.assertNotIn("self_attn.qkv_proj", self._install(root))


class TestReusedUpstreamFp8Method(unittest.TestCase):
    """Invariants of the existing FP8 linear this conversion reuses.

    Skipped where the serving runtime's FP8 stack cannot be imported (a
    torch-only worker); on a CUDA runner it proves the reused method is the
    non-block one whose postprocess quantizes a resident BF16 weight once.
    """

    @classmethod
    def setUpClass(cls):
        try:
            cls.method = fp8.rowwise_fp8_method()
        except Exception as err:  # missing runtime dep, or no CUDA capability
            raise unittest.SkipTest(f"needs the FP8 serving stack ({err!r})")

    def test_config_is_the_online_non_block_fp8_path(self):
        self.assertFalse(self.method.is_checkpoint_fp8_serialized)
        self.assertFalse(self.method.block_quant)
        self.assertFalse(self.method.use_mxfp8)
        self.assertIsNone(self.method.weight_block_size)

    def test_a_second_postprocess_refuses_to_requantize(self):
        # A weight that is already E4M3 (converted, or an FP8 checkpoint) must
        # not be fed back through the quantizer: the second pass would
        # re-interpret the [K, N] storage as [N, K] and call that quantization.
        layer = nn.Linear(256, 512, bias=False, dtype=torch.bfloat16)
        layer.weight = nn.Parameter(
            torch.zeros(512, 256, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        with self.assertRaises(RuntimeError):
            self.method.process_weights_after_loading(layer)

    def test_the_walker_selects_the_runtime_unquantized_method_type(self):
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        root = nn.Module()
        root.attn = nn.Module()
        root.attn.qkv_proj = _linear(512, 256, method=UnquantizedLinearMethod())
        root.attn.o_proj = _linear(256, 256, method=_AlreadyQuantized())
        converted = fp8.install_rowwise_fp8_on_eligible_linears(
            root, enabled=True, excluded_module_type=()
        )
        self.assertEqual(converted, ["attn.qkv_proj"])
        self.assertIs(root.attn.qkv_proj.quant_method, self.method)


class TestHeadProjection(unittest.TestCase):
    """The output-head dispatch, end to end on the chunked projection path.

    The low-row Triton GEMV is GPU-only; everything around it -- scale pairing,
    shape/dtype/width checks, the reshape back to [batch, vocab] -- runs here.
    The same chunked projection serves a wide verification batch or prefill on
    the GPU, so this is arithmetic that matters, not a stub.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from sglang.kernels.ops.gemm import sm120_rowwise_fp8 as proj
        except ModuleNotFoundError as err:
            raise unittest.SkipTest(f"needs the kernel module ({err.name})")
        cls.proj = proj

    def setUp(self):
        torch.manual_seed(1)
        self.dense = torch.randn(1024, 128, dtype=torch.bfloat16)
        self.linear = nn.Linear(128, 1024, bias=False, dtype=torch.bfloat16)
        self.linear.weight = nn.Parameter(self.dense.clone(), requires_grad=False)
        fp8.replace_linear_weight_rowwise_fp8(self.linear)

    def _check(self, hidden):
        logits = fp8.rowwise_fp8_lm_head_logits(hidden, self.linear.weight)
        reference = torch.matmul(hidden, self.dense.float().to(hidden.dtype).T)
        self.assertEqual(logits.shape, (*hidden.shape[:-1], 1024))
        self.assertEqual(logits.dtype, torch.bfloat16)
        self.assertLess(
            (logits.float() - reference.float()).abs().max().item(),
            reference.float().abs().max().item() * 2**-3,
        )
        return logits

    def test_projection_uses_the_bound_row_scales(self):
        self._check(torch.randn(1, 128, dtype=torch.bfloat16))
        self._check(torch.randn(5, 128, dtype=torch.bfloat16))
        # A wider batch takes the chunked path; >32 rows is decode-verify and
        # prefill territory, where cuBLAS is the better kernel anyway.
        self._check(torch.randn(40, 128, dtype=torch.bfloat16))
        # Ranked/token-shaped input comes back ranked.
        self._check(torch.randn(2, 3, 128, dtype=torch.bfloat16))

    def test_wrong_activation_width_and_dtype_are_refused(self):
        with self.assertRaises(RuntimeError):
            fp8.rowwise_fp8_lm_head_logits(
                torch.randn(1, 64, dtype=torch.bfloat16), self.linear.weight
            )
        # A non-BF16 activation is cast, not silently mixed with FP8 codes.
        logits = fp8.rowwise_fp8_lm_head_logits(
            torch.randn(1, 128, dtype=torch.float32), self.linear.weight
        )
        self.assertEqual(logits.dtype, torch.bfloat16)

    def test_a_bf16_weight_is_not_mistaken_for_the_fp8_head(self):
        self.assertFalse(fp8.is_rowwise_fp8_weight(self.dense))
        with self.assertRaises(RuntimeError):
            fp8.rowwise_fp8_lm_head_logits(
                torch.randn(1, 128, dtype=torch.bfloat16), self.dense
            )


class TestDefaultLoaderHeadLifecycle(unittest.TestCase):
    """The conversion through the REAL default-loader lifecycle.

    ``DefaultModelLoader.load_weights_and_postprocess`` runs
    ``model.load_weights`` and then every module's
    ``process_weights_after_loading``; only the loaders that bypass
    ``load_weights`` call the model's post-load helper. A test that calls the
    head helper directly cannot see that seam, and an unconverted head is not a
    crash -- the logits path quietly keeps a BF16 head while the projections
    converted, so the launch reports a representation it does not have.

    The model's real ``load_weights`` is driven over a two-parameter tree, so
    the checkpoint-matching logic under test is the production one.
    """

    VOCAB = 256
    HIDDEN = 128

    @classmethod
    def setUpClass(cls):
        try:
            import sglang.srt.model_loader.loader as loader_module
            from sglang.srt.model_loader.loader import DefaultModelLoader
            from sglang.srt.models.qwen4_exp import (
                Qwen4ExpForConditionalGeneration,
            )
        except ModuleNotFoundError as err:
            raise unittest.SkipTest(f"needs the serving runtime ({err.name})")
        cls.loader_module = loader_module
        cls.DefaultModelLoader = DefaultModelLoader
        cls.model_class = Qwen4ExpForConditionalGeneration

    def setUp(self):
        self._switch = fp8.online_fp8_enabled()

    def tearDown(self):
        fp8.configure_online_fp8(
            self._switch or False,
            cuda_available=True,
            capability=(12, 0),
            model_eligible=True,
        )

    def _model(self, head_method=None):
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedEmbeddingMethod,
            UnquantizedLinearMethod,
        )

        model = nn.Module()
        model.quant_config = None
        model.language_model_only = False
        model.start_layer, model.end_layer = 0, 1
        model.config = SimpleNamespace(
            tie_word_embeddings=False, encoder_only=False, num_experts=None
        )
        model.pp_group = SimpleNamespace(is_last_rank=True)
        model.lm_head = nn.Linear(
            self.HIDDEN, self.VOCAB, bias=False, dtype=torch.bfloat16
        )
        model.lm_head.quant_method = (
            UnquantizedEmbeddingMethod() if head_method is None else head_method
        )
        # A served lm_head parameter always carries its loader; the conversion
        # has to keep that lifetime contract, not just the tensors.
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        model.lm_head.weight.weight_loader = default_weight_loader
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.down_proj = nn.Linear(
            self.HIDDEN, self.VOCAB, bias=False, dtype=torch.bfloat16
        )
        layer.mlp.down_proj.quant_method = UnquantizedLinearMethod()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([layer])
        # The production construction step: eligible unquantized linears get the
        # online-FP8 method. A recording method stands in for it here because the
        # channelwise FP8 GEMM that the real method dispatches to is CUDA-only;
        # what this test pins is the order the two conversions land in.
        self.seen = []
        # Rooted at the layer container, as production roots it at each decoder
        # layer: the output head is not in a layer's subtree, so its own method
        # is left to the head hook below rather than swapped here.
        fp8.install_rowwise_fp8_on_eligible_linears(
            model.model,
            enabled=True,
            method_factory=lambda: self._SpyMethod(self.seen, model),
            unquantized_method_type=UnquantizedLinearMethod,
            excluded_module_type=(),
        )
        return model

    class _SpyMethod:
        """Stands in for the online-FP8 linear method, recording the head state."""

        def __init__(self, seen, model):
            self.seen = seen
            self.model = model

        def process_weights_after_loading(self, module):
            self.seen.append(
                (
                    "postprocess",
                    module.weight.dtype,
                    self.model.lm_head.weight.dtype,
                )
            )

    def _weights(self, value=1.0):
        for name in ("lm_head.weight", "model.layers.0.mlp.down_proj.weight"):
            yield (
                name,
                torch.full((self.VOCAB, self.HIDDEN), value, dtype=torch.bfloat16),
            )

    def _load(self, model, weights):
        with (
            patch.object(self.loader_module, "is_cuda_alike", return_value=False),
            patch.object(
                self.loader_module,
                "device_loading_context",
                side_effect=lambda *_: nullcontext(),
            ),
        ):
            return self.DefaultModelLoader.load_weights_and_postprocess(
                model, iter(list(weights)), torch.device("cpu")
            )

    def _bind_model_side_hooks(self, model):
        """Use the model class's own load/post-load code on the small tree."""
        cls = self.model_class
        model.load_weights = types.MethodType(cls.load_weights, model)
        model._load_qwen4_exp_ple_buffer = types.MethodType(
            cls._load_qwen4_exp_ple_buffer, model
        )
        model.post_load_weights = types.MethodType(
            getattr(cls, "post_load_weights", None) or _convert_head, model
        )
        return model

    class _CheckpointMethod:
        """A method handed out by the checkpoint's own quantization config (e.g.
        ModelOpt's FP8 / NVFP4 linear method for the head prefix): it intends to
        quantize this BF16 weight itself, in its own post-load hook."""

        def __init__(self):
            self.calls = []

        def process_weights_after_loading(self, module):
            self.calls.append(module.weight.dtype)

    def _enable(self, requested=None):
        fp8.configure_online_fp8(
            requested, cuda_available=True, capability=(12, 0), model_eligible=True
        )

    def test_head_is_fp8_before_any_postprocess_or_draft_adoption(self):
        self._enable()
        model = self._bind_model_side_hooks(self._model())
        self._load(model, self._weights())

        # The projection's postprocess ran, and by then the head was already
        # E4M3: nothing downstream (draft head sharing, graph capture) can see a
        # half-converted model.
        self.assertEqual(
            self.seen,
            [
                (
                    "postprocess",
                    torch.bfloat16,
                    torch.float8_e4m3fn,
                )
            ],
        )
        self.assertTrue(fp8.is_rowwise_fp8_weight(model.lm_head.weight))
        scale = fp8.require_rowwise_scale(model.lm_head.weight)
        self.assertEqual(model.lm_head.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(scale.shape), (self.VOCAB,))
        self.assertEqual(scale.dtype, torch.float32)

    def test_generic_postprocess_does_not_drop_the_scales_or_requantize(self):
        self._enable()
        model = self._bind_model_side_hooks(self._model())
        self._load(model, self._weights())
        head = model.lm_head.weight
        self.assertTrue(callable(getattr(head, "weight_loader", None)))
        # A copy: the reload rewrites the resident scale tensor in place, so the
        # stale maximum has to be recorded before it moves.
        scale = fp8.require_rowwise_scale(head).clone()
        self._load(model, self._weights(value=0.5))
        weight = model.lm_head.weight
        self.assertIs(weight, head)
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)
        fresh = fp8.require_rowwise_scale(weight)
        self.assertEqual(tuple(fresh.shape), (self.VOCAB,))
        self.assertLess(fresh.max().item(), scale.max().item())
        # Values and scales were rewritten as one unit: the reload came back at
        # the new magnitude, not at the new value times the old scale (which is
        # what copying BF16 into FP8 storage would leave behind).
        restored = fp8.dequantize_rowwise_weight(weight).float()
        self.assertAlmostEqual(restored.abs().max().item(), 0.5, delta=0.05)

    def test_unselected_launch_keeps_the_bf16_head(self):
        self._enable(False)
        model = self._bind_model_side_hooks(self._model())
        self._load(model, self._weights())
        self.assertEqual(model.lm_head.weight.dtype, torch.bfloat16)
        self.assertIsNone(fp8.rowwise_scale_of(model.lm_head.weight))

    def test_both_unquantized_head_methods_convert(self):
        """An lm_head carries either unquantized method, depending on whether the
        checkpoint config named its prefix; both are ours to convert."""
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedEmbeddingMethod,
            UnquantizedLinearMethod,
        )

        for method in (UnquantizedLinearMethod(), UnquantizedEmbeddingMethod()):
            with self.subTest(method=type(method).__name__):
                self._enable()
                model = self._bind_model_side_hooks(self._model(head_method=method))
                self._load(model, self._weights())
                self.assertEqual(
                    model.lm_head.weight.dtype, torch.float8_e4m3fn, method
                )
                self.assertTrue(fp8.is_rowwise_fp8_weight(model.lm_head.weight))

    def test_head_owned_by_a_checkpoint_method_keeps_its_bf16_weight(self):
        """A BF16 head that another method intends to quantize is not ours."""
        self._enable()
        method = self._CheckpointMethod()
        model = self._bind_model_side_hooks(self._model(head_method=method))
        self.assertFalse(fp8.otherwise_unquantized(model.lm_head))
        loaded = torch.full((self.VOCAB, self.HIDDEN), 1.0, dtype=torch.bfloat16)
        self._load(
            model,
            [
                ("lm_head.weight", loaded.clone()),
                ("model.layers.0.mlp.down_proj.weight", loaded.clone()),
            ],
        )

        # Ownership intact: the head's own post-load hook ran, exactly once, on
        # the BF16 weight it was given -- not on E4M3 codes written underneath it.
        self.assertEqual(method.calls, [torch.bfloat16])
        self.assertEqual(model.lm_head.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(model.lm_head.weight, loaded))
        # No half-applied representation: no scale bound to a BF16 tensor.
        self.assertIsNone(fp8.rowwise_scale_of(model.lm_head.weight))
        self.assertFalse(fp8.is_rowwise_fp8_weight(model.lm_head.weight))
        # The projections are unaffected by the head's ownership.
        self.assertEqual([entry[0] for entry in self.seen], ["postprocess"])


class TestRealParallelLMHeadLoader(unittest.TestCase):
    """The head conversion under a REAL ``ParallelLMHead``.

    A served head is not an equal-shaped ``nn.Linear``: ``ParallelLMHead.weight``
    is the padded, per-rank *shard* of the checkpoint head, and the head's own
    ``weight_loader`` is what turns the full unpadded ``[org_vocab, hidden]``
    checkpoint tensor into that layout (slice to this rank's org-vocab range,
    copy, then zero the padding rows). Replacing that loader with a quantizing
    copy that demands equal shapes is fine for a fake and wrong for the served
    head: the reload arrives at the checkpoint layout on every rank, so either
    it fails outright (TP > 1, or any vocab that needed padding) or, at
    ``padding_size``-aligned vocabularies, it quietly misplaces rows.

    Every check here is against the native BF16 loader on a sibling head, so a
    value that lands on the wrong token id cannot pass by being self-consistent.
    """

    # An org vocab that is NOT a multiple of the 64-row padding unit, plus
    # reserved rows after it: both halves of the native layout in one fixture.
    ORG_VOCAB = 250
    ADDED_VOCAB = 6
    HIDDEN = 128

    @classmethod
    def setUpClass(cls):
        try:
            from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
            from sglang.srt.runtime_context import get_parallel
        except ModuleNotFoundError as err:
            raise unittest.SkipTest(f"needs the serving runtime ({err.name})")
        cls.ParallelLMHead = ParallelLMHead
        cls.get_parallel = staticmethod(get_parallel)

    def setUp(self):
        self._switch = fp8.online_fp8_enabled()
        fp8.configure_online_fp8(
            True, cuda_available=True, capability=(12, 0), model_eligible=True
        )

    def tearDown(self):
        fp8.configure_online_fp8(
            self._switch or False,
            cuda_available=True,
            capability=(12, 0),
            model_eligible=True,
        )

    def _scope(self, tp_size, tp_rank):
        # A logical group is enough to construct and load a head: the loader
        # reads the rank/width recorded in shard_indices, not collectives.
        return self.get_parallel().override(
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=None,
            attn_tp_size=tp_size,
            attn_tp_rank=tp_rank,
            attn_tp_group=None,
            attn_dp_size=1,
            attn_dp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            moe_tp_size=tp_size,
            moe_tp_rank=tp_rank,
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_ep_group=None,
            moe_dp_size=1,
            moe_dp_rank=0,
        )

    def _head(self, tp_size=1, tp_rank=0):
        with self._scope(tp_size, tp_rank):
            return self.ParallelLMHead(
                self.ORG_VOCAB + self.ADDED_VOCAB,
                self.HIDDEN,
                params_dtype=torch.bfloat16,
                org_num_embeddings=self.ORG_VOCAB,
            )

    def _checkpoint(self, seed, rows=None):
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(
            self.ORG_VOCAB if rows is None else rows,
            self.HIDDEN,
            dtype=torch.bfloat16,
            generator=generator,
        )

    def _load_native(self, head, checkpoint):
        """Ground truth: the head's own BF16 loader, with nothing of ours involved."""
        head.weight.weight_loader(head.weight, checkpoint)
        return head.weight.detach().float().clone()

    def _assert_native_placement(self, head, reference):
        weight = head.weight
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(weight.shape), tuple(reference.shape))
        restored = fp8.dequantize_rowwise_weight(weight).float()
        self.assertLess(
            (restored - reference).abs().max().item(),
            reference.abs().max().item() * 2**-3,
        )
        # Padding rows are zeros in both representations, never stale magnitudes.
        self.assertTrue(
            torch.equal(
                restored.abs().sum(1) == 0,
                reference.abs().sum(1) == 0,
            )
        )
        # One scale per resident row, from that row's own maximum.
        scale = fp8.require_rowwise_scale(weight)
        self.assertFalse(bool(torch.isnan(scale).any()))
        self.assertTrue(
            torch.allclose(
                scale,
                reference.abs().amax(1).clamp_min(1e-8) / 448.0,
                rtol=1e-3,
                atol=1e-9,
            )
        )
        return restored

    def _converted_head(self, seed, tp_size=1, tp_rank=0):
        """A head that went through the real load, then the conversion."""
        head = self._head(tp_size, tp_rank)
        reference = self._load_native(head, self._checkpoint(seed))
        fp8.replace_linear_weight_rowwise_fp8(head)
        return head, reference

    def test_padded_tp1_head_reloads_through_its_native_loader(self):
        head, reference = self._converted_head(1)
        # The native layout, so a passing reload means something: 320 resident
        # rows for a 250-row checkpoint tensor.
        self.assertEqual(tuple(head.weight.shape), (320, self.HIDDEN))
        self.assertEqual(head.shard_indices.num_elements_padded, 320)
        self._assert_native_placement(head, reference)

        # A weight update arrives in checkpoint layout, as the first load did.
        updated = self._checkpoint(2)
        head.weight.weight_loader(head.weight, updated)
        self._assert_native_placement(head, self._load_native(self._head(), updated))

    def test_tp2_shards_keep_their_offsets_across_a_reload(self):
        checkpoint = self._checkpoint(3)
        heads = {rank: self._head(2, rank) for rank in (0, 1)}
        references = {
            rank: self._load_native(heads[rank], checkpoint) for rank in heads
        }
        # Rank 0 owns tokens 0..127; rank 1 owns 128..249 and has no added rows.
        self.assertEqual(tuple(references[0].shape), (160, self.HIDDEN))
        self.assertEqual(int((references[1].abs().sum(1) > 0).sum()), 122)

        for head in heads.values():
            fp8.replace_linear_weight_rowwise_fp8(head)
        updated = self._checkpoint(4)
        for rank, head in heads.items():
            head.weight.weight_loader(head.weight, updated)

        placed = {
            rank: self._assert_native_placement(
                heads[rank], self._load_native(self._head(2, rank), updated)
            )
            for rank in heads
        }
        # Row placement, stated against the checkpoint rather than against the
        # other rank: a loader that ignored shard offsets would be internally
        # consistent and wrong for both ranks.
        reference_scale = updated.float().abs().max().item() * 2**-3
        for rank, head in heads.items():
            shard = head.shard_indices
            start, stop = shard.org_vocab_start_index, shard.org_vocab_end_index
            self.assertLess(
                (placed[rank][: stop - start] - updated[start:stop].float())
                .abs()
                .max()
                .item(),
                reference_scale,
            )
        self.assertEqual(int((placed[1][122:].abs().sum(1) == 0).sum()), 38)

    def test_reload_updates_the_head_the_draft_alias_still_points_at(self):
        """``set_embed_and_head`` shares the Parameter object, not a snapshot.

        ``Qwen35MTP.set_embed_and_head`` does ``draft.lm_head.weight = head``, so
        the target and the NEXTN draft read one tensor. A reload that installed a
        new Parameter would leave the draft head at the previous weights while the
        target moved -- and a captured CUDA graph keeps reading the old storage
        too. The reload has to rewrite the resident tensor.
        """
        head, _ = self._converted_head(1)
        adopted = head.weight  # exactly what the draft model stores
        adopted_scale = fp8.rowwise_scale_of(adopted)

        updated = self._checkpoint(5)
        head.weight.weight_loader(head.weight, updated)

        self.assertIs(head.weight, adopted)
        self.assertIs(fp8.rowwise_scale_of(head.weight), adopted_scale)
        self._assert_native_placement(head, self._load_native(self._head(), updated))

    def test_reduced_vocab_head_selection_follows_a_reloaded_target(self):
        """FR-Spec: a hot-token draft head is a row-selection of the target's.

        Values and per-row scales move together, and the selection is a snapshot
        (the dense path's ``head.clone()`` is one too), so a later target reload
        must not leave it holding codes paired with somebody else's scales.
        """
        head, _ = self._converted_head(1)
        hot = torch.tensor([0, 7, self.ORG_VOCAB - 1], dtype=torch.long)
        selected = fp8.select_rowwise_weight_rows(head.weight, hot)

        self.assertEqual(tuple(selected.shape), (3, self.HIDDEN))
        self.assertTrue(
            torch.equal(
                fp8.require_rowwise_scale(selected),
                fp8.require_rowwise_scale(head.weight)[hot],
            )
        )
        # The selection is the draft's own: it carries no loader bound to the
        # target, which would re-quantize the full head against selected rows.
        self.assertFalse(callable(getattr(selected, "weight_loader", None)))

        head.weight.weight_loader(head.weight, self._checkpoint(6))
        refreshed = fp8.select_rowwise_weight_rows(head.weight, hot)
        self.assertFalse(torch.equal(selected.float(), refreshed.float()))
        self._assert_native_placement(
            SimpleNamespace(weight=refreshed),
            self._load_native(self._head(), self._checkpoint(6))[hot],
        )

    def test_a_reload_in_another_dtype_is_refused(self):
        head, _ = self._converted_head(1)
        before = fp8.require_rowwise_scale(head.weight).clone()
        with self.assertRaises(RuntimeError):
            head.weight.weight_loader(
                head.weight, self._checkpoint(7).to(torch.float16)
            )
        # Refused means nothing was half-written: same codes, same scales.
        self.assertTrue(torch.equal(fp8.require_rowwise_scale(head.weight), before))

    def test_the_draft_side_consumers_read_the_converted_head(self):
        """DFlash's gate and the FR-Spec hot-token map, calling their own code.

        Both take the target's resident weight: the gate decides whether a
        reduced-vocabulary draft may project through it at all, and the map turns
        it into the draft's smaller head. Neither may see codes without scales,
        and the rows they pick must be the rows the BF16 head would have picked.
        """
        try:
            from sglang.srt.speculative.dflash_utils import is_rowwise_fp8_head_weight
            from sglang.srt.speculative.eagle_worker_v2 import (
                _restrict_head_to_hot_tokens,
            )
        except ImportError as err:  # pragma: no cover - torch-only worker
            self.skipTest(f"needs the speculative runtime ({err})")

        head, _ = self._converted_head(1)
        hot = torch.tensor([0, 7, self.ORG_VOCAB - 1], dtype=torch.long)

        self.assertTrue(is_rowwise_fp8_head_weight(head.weight))
        reduced = _restrict_head_to_hot_tokens(head.weight, hot)
        self.assertEqual(tuple(reduced.shape), (3, self.HIDDEN))
        self.assertTrue(is_rowwise_fp8_head_weight(reduced))
        self.assertTrue(
            torch.equal(
                fp8.require_rowwise_scale(reduced),
                fp8.require_rowwise_scale(head.weight)[hot],
            )
        )
        # Same rows the dense head hands the same draft, to quantization error.
        dense = self._head()
        self._load_native(dense, self._checkpoint(1))
        dense_reduced = _restrict_head_to_hot_tokens(dense.weight, hot)
        self.assertLess(
            (fp8.dequantize_rowwise_weight(reduced).float() - dense_reduced.float())
            .abs()
            .max()
            .item(),
            dense_reduced.float().abs().max().item() * 2**-3,
        )

    def test_a_loader_that_writes_part_of_the_head_leaves_the_rest_alone(self):
        """Merged/stacked reloads call the loader per shard.

        Staging a zero-filled buffer would let a per-shard loader silently wipe
        the rows it was not asked to write, so the stage starts from what is
        resident and the loader overwrites only its own rows.
        """

        class _HalfShardHead(nn.Module):
            """A head-shaped parameter whose loader takes one shard per call."""

            def __init__(self, weight):
                super().__init__()
                self.weight = weight
                self.quant_method = _Unquantized()
                # What ParallelLMHead does to its parameter, minus the import.
                weight.output_dim = 0
                weight.weight_loader = self.weight_loader

            def weight_loader(self, param, loaded_weight, shard_id):
                rows = loaded_weight.shape[0]
                start = {"upper": 0, "lower": rows}[shard_id]
                param.data[start : start + rows].copy_(loaded_weight)

        module = _HalfShardHead(
            nn.Parameter(
                torch.zeros(512, 256, dtype=torch.bfloat16), requires_grad=False
            )
        )
        upper = torch.randn(256, 256, dtype=torch.bfloat16)
        lower = torch.randn(256, 256, dtype=torch.bfloat16)
        module.weight.weight_loader(module.weight, upper, "upper")
        module.weight.weight_loader(module.weight, lower, "lower")
        fp8.replace_linear_weight_rowwise_fp8(module)

        fresh_upper = torch.randn(256, 256, dtype=torch.bfloat16)
        module.weight.weight_loader(module.weight, fresh_upper, "upper")
        restored = fp8.dequantize_rowwise_weight(module.weight).float()
        self.assertLess(
            (restored[:256] - fresh_upper.float()).abs().max().item(),
            fresh_upper.float().abs().max().item() * 2**-3,
        )
        self.assertLess(
            (restored[256:] - lower.float()).abs().max().item(),
            lower.float().abs().max().item() * 2**-3,
        )


if __name__ == "__main__":
    unittest.main()
