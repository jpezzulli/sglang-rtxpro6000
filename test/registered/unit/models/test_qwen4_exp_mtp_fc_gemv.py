"""Hermetic CPU coverage for the draft MTP entry fusion dispatch.

Review-1 defects pinned here: the gate must DECLINE the donor dense GEMV on
CPU (the real ``bf16_gemv`` wrapper reaches ``_num_sms`` and has no CPU
path), on a BF16 embedding side paired with an FP32 hidden-side activation
or weight (its ``tl.dot`` contract is BF16 on both sides), on the row budget
and layout, and when the opt-in flag is off; every decline keeps the
original two-Linear path.  ``bf16_gemv`` is Triton, so the wiring check
patches it with an exact ``F.linear`` probe AND patches the eligibility
helpers to pass, while the real eligibility checks are exercised with the
helpers in place.  Pure Python; plain ``unittest.TestCase`` (the shared
``test_utils.CustomTestCase`` parses ``CUDA_VISIBLE_DEVICES`` at collection).
"""

import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import w8a16_gemv
from sglang.srt.models import qwen4_exp_mtp
from sglang.srt.models.qwen4_exp_mtp import (
    Qwen4ExpForCausalLMMTP,
    _mtp_fc_gemv_supported,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN = 32
HC = 2


def _load_gpu_test_module():
    """Load the registered SM120 file CPU-side (its pytestmark skips there).

    Pins the fixture-builder CONTRACT -- production init API, norm widths,
    dtype/device/buffer handling and the width table -- without pretending
    any GPU kernel ran.
    """
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "kernels",
        "test_qwen4_exp_mtp_entry_gemv.py",
    )
    spec = importlib.util.spec_from_file_location("mtp_entry_gemv_gpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _IdentityNorm(nn.Module):
    def forward(self, x):
        return x * 1.5


def _fake_self(embed_dtype=torch.bfloat16, hidden_dtype=torch.bfloat16, bias=False):
    return SimpleNamespace(
        hc_count=HC,
        hidden_size=HIDDEN,
        pre_fc_norm_embedding=_IdentityNorm(),
        pre_fc_norm_hidden=_IdentityNorm(),
        fc_embedding=nn.Linear(HIDDEN, HIDDEN, bias=bias).to(embed_dtype),
        fc_hidden=nn.Linear(HIDDEN, HIDDEN, bias=bias).to(hidden_dtype),
    )


def _inputs(tokens: int, embed_dtype=torch.bfloat16, hidden_dtype=torch.bfloat16):
    gen = torch.Generator().manual_seed(7)
    embeds = torch.randn(tokens, HIDDEN, dtype=embed_dtype, generator=gen)
    hidden = torch.randn(tokens, HC * HIDDEN, dtype=hidden_dtype, generator=gen)
    return embeds, hidden


def _fuse(fake_self, embeds, hidden):
    return Qwen4ExpForCausalLMMTP._fuse_residual_linear_shared(
        fake_self, embeds, hidden
    )


def _supported(*args, flag=True, **kwargs):
    with envs.SGLANG_MTP_FC_GEMV.override(flag):
        return _mtp_fc_gemv_supported(*args, **kwargs)


class TestMtpFcGemvEligibility(unittest.TestCase):
    """The real gate, no helper mocks: nothing CPU-side may select the GEMV."""

    def test_cpu_tensors_never_select_the_gemv(self):
        # Review defect (a): these exact shapes used to route to bf16_gemv on
        # CPU, where the wrapper dies in _num_sms.
        embeds, hidden = _inputs(4)
        fake = _fake_self()
        rows = hidden.view(4, HC, HIDDEN).reshape(-1, HIDDEN)
        self.assertFalse(
            _supported(embeds, fake.fc_embedding.weight, rows, fake.fc_hidden.weight)
        )
        with envs.SGLANG_MTP_FC_GEMV.override(True):
            with mock.patch.object(w8a16_gemv, "bf16_gemv") as probe:
                out = _fuse(fake, embeds, hidden)
        probe.assert_not_called()
        self.assertEqual(out.shape, (4, HC * HIDDEN))

    def test_flag_off_declines(self):
        embeds, hidden = _inputs(4)
        fake = _fake_self()
        rows = hidden.view(4, HC, HIDDEN).reshape(-1, HIDDEN)
        self.assertFalse(
            _supported(
                embeds,
                fake.fc_embedding.weight,
                rows,
                fake.fc_hidden.weight,
                flag=False,
            )
        )

    def test_hidden_side_fp32_declines(self):
        # Review defect (b): BF16 embedding side with FP32 hidden-side input
        # AND weight declined to select the kernel twice; the old two-Linear
        # path handles the mixed case.
        embeds, hidden = _inputs(4, hidden_dtype=torch.float32)
        fake = _fake_self(hidden_dtype=torch.float32)
        rows = hidden.view(4, HC, HIDDEN).reshape(-1, HIDDEN)
        with mock.patch.object(
            qwen4_exp_mtp, "_bf16_entry_gemm_target_ok", return_value=True
        ):
            self.assertFalse(
                _supported(
                    embeds, fake.fc_embedding.weight, rows, fake.fc_hidden.weight
                )
            )

    def test_weight_dtype_and_layout_decline(self):
        embeds, hidden = _inputs(4)
        rows = hidden.view(4, HC, HIDDEN).reshape(-1, HIDDEN)
        ok_w = torch.ones(HIDDEN, HIDDEN, dtype=torch.bfloat16)
        bad_w_fp32 = ok_w.float()
        bad_w_nc = ok_w.t().contiguous().t()  # K axis no longer contiguous
        with mock.patch.object(
            qwen4_exp_mtp, "_bf16_entry_gemm_target_ok", return_value=True
        ):
            self.assertFalse(_supported(embeds, bad_w_fp32, rows, ok_w))
            self.assertFalse(_supported(embeds, ok_w, rows, bad_w_fp32))
            self.assertFalse(_supported(embeds, bad_w_nc, rows, ok_w))
            self.assertFalse(_supported(embeds, ok_w, rows, bad_w_nc))

    def test_row_budget_declines(self):
        embeds, hidden = _inputs(16)  # 16 * hc 2 == 32 rows
        fake = _fake_self()
        rows = hidden.view(16, HC, HIDDEN).reshape(-1, HIDDEN)
        with mock.patch.object(
            qwen4_exp_mtp, "_bf16_entry_gemm_target_ok", return_value=True
        ):
            self.assertFalse(
                _supported(
                    embeds, fake.fc_embedding.weight, rows, fake.fc_hidden.weight
                )
            )

    def test_split_devices_decline(self):
        embeds, hidden = _inputs(4)
        fake = _fake_self()
        rows = hidden.view(4, HC, HIDDEN).reshape(-1, HIDDEN)
        other = torch.ones(1)
        with mock.patch.object(
            qwen4_exp_mtp,
            "_bf16_entry_gemm_target_ok",
            side_effect=qwen4_exp_mtp._bf16_entry_gemm_target_ok,
        ) as spy:
            self.assertFalse(
                _supported(
                    embeds, fake.fc_embedding.weight, rows, fake.fc_hidden.weight
                )
            )
            spy.assert_called_once()
        # The device helper itself rejects CPU tensors outright.
        self.assertFalse(qwen4_exp_mtp._bf16_entry_gemm_target_ok(embeds, rows))
        self.assertFalse(qwen4_exp_mtp._bf16_entry_gemm_target_ok(embeds, other))

    def test_eligible_shapes_pass_with_the_target_helper_mocked(self):
        embeds, hidden = _inputs(4)
        fake = _fake_self()
        rows = hidden.view(4, HC, HIDDEN).reshape(-1, HIDDEN)
        with mock.patch.object(
            qwen4_exp_mtp, "_bf16_entry_gemm_target_ok", return_value=True
        ):
            self.assertTrue(
                _supported(
                    embeds, fake.fc_embedding.weight, rows, fake.fc_hidden.weight
                )
            )


class TestMtpFcGemvWiring(unittest.TestCase):
    """Dispatch wiring: eligibility True must call the GEMV for both sides."""

    def test_gemv_dispatch_and_bit_parity(self):
        calls = []

        def probe(x, w, *args, **kwargs):
            calls.append((tuple(x.shape), tuple(w.shape)))
            return nn.functional.linear(x, w)

        fake = _fake_self()
        embeds, hidden = _inputs(4)
        with (
            envs.SGLANG_MTP_FC_GEMV.override(True),
            mock.patch.object(
                qwen4_exp_mtp, "_mtp_fc_gemv_supported", return_value=True
            ),
            mock.patch.object(w8a16_gemv, "bf16_gemv", probe),
            torch.no_grad(),
        ):
            out_gemv = _fuse(fake, embeds, hidden)
        self.assertEqual(
            calls,
            [
                ((4, HIDDEN), (HIDDEN, HIDDEN)),
                ((4 * HC, HIDDEN), (HIDDEN, HIDDEN)),
            ],
        )
        with envs.SGLANG_MTP_FC_GEMV.override(False), torch.no_grad():
            out_cublas = _fuse(fake, *_inputs(4))
        self.assertTrue(torch.equal(out_gemv, out_cublas))

    def test_biased_linears_keep_the_original_path(self):
        fake = _fake_self(bias=True)
        embeds, hidden = _inputs(4)
        with (
            envs.SGLANG_MTP_FC_GEMV.override(True),
            mock.patch.object(
                qwen4_exp_mtp, "_mtp_fc_gemv_supported", return_value=True
            ),
            mock.patch.object(w8a16_gemv, "bf16_gemv") as probe,
        ):
            _fuse(fake, embeds, hidden)
        probe.assert_not_called()


class TestMtpEntryFusionFixtureContract(unittest.TestCase):
    """CPU contract check of the GPU fixture builder (no kernel execution).

    Review2 pinned a fixture that passed ``rms_norm_eps`` to a constructor
    taking ``eps``, left norms on CPU/FP32 and sized the hidden norm at
    2560 instead of the production ``hc_count * hidden_size``. Build the
    exact modules the SM120 file uses, on CPU, and pin those properties.
    """

    @classmethod
    def setUpClass(cls):
        cls.gpu_mod = _load_gpu_test_module()

    def test_fixture_matches_production_shape_and_dtype(self):
        fusion = self.gpu_mod.build_entry_fusion(device="cpu", dtype=torch.bfloat16)
        config = self.gpu_mod._entry_config()
        hc, hidden = config.hc_count, config.hidden_size
        # Production widths: embedding norm hidden, hidden norm hc * hidden.
        self.assertEqual(fusion.pre_fc_norm_embedding.weight.shape, (hidden,))
        self.assertEqual(fusion.pre_fc_norm_hidden.weight.shape, (hc * hidden,))
        self.assertEqual(fusion.fc_embedding.weight.shape, (hidden, hidden))
        self.assertEqual(fusion.fc_hidden.weight.shape, (hidden, hidden))
        self.assertIsNone(fusion.fc_embedding.bias)
        self.assertIsNone(fusion.fc_hidden.bias)
        # Everything -- including GemmaRMSNorm's non-persistent buffer --
        # landed on the requested device with the requested dtype.
        for name in (
            "pre_fc_norm_embedding",
            "pre_fc_norm_hidden",
            "fc_embedding",
            "fc_hidden",
        ):
            module = getattr(fusion, name)
            for tensor in list(module.parameters()) + list(module.buffers()):
                self.assertEqual(tensor.dtype, torch.bfloat16, name)
                self.assertEqual(tensor.device.type, "cpu", name)
        for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden"):
            norm = getattr(fusion, name)
            self.assertTrue(
                torch.equal(norm.gemma_weight, norm.weight + 1.0),
                f"{name}: production loader keeps gemma_weight == weight + 1",
            )
            # Deterministic NONUNIFORM norm weights exercise both sides.
            self.assertGreater(norm.weight.float().std().item(), 0.0)

    def test_width_table_matches_the_kernel_contract(self):
        mod = self.gpu_mod
        self.assertEqual(mod.TOKEN_WIDTHS, [1, 4, 6, 16])
        self.assertEqual(mod.HC_COUNT, 4)
        self.assertEqual(mod.DONOR_MAX_M, 16)
        for tokens in mod.TOKEN_WIDTHS:
            routed = 1 <= tokens * mod.HC_COUNT <= mod.DONOR_MAX_M
            self.assertEqual(len(mod._expected_calls(tokens, True)) == 2, routed)
            self.assertEqual(mod._expected_calls(tokens, False), [])


if __name__ == "__main__":
    unittest.main(verbosity=3)
