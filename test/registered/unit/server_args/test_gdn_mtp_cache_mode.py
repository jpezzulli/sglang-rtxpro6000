"""Unit tests for the RecoverSSM (``--gdn-mtp-cache-mode=none``) contract.

CPU-only: the mode's argument surface, the guards that decide which launches may
ask for it, the verify-kernel capability gate that keeps SM120's full mode on its
established Triton verifier, and the single predicate that decides which recovery
path runs. None of it needs a GPU; none of it proves GPU numerics (see
``test/registered/attention/unittests/gdn/test_recover_ssm_sm120.py``).
"""

import argparse
import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.mamba_hook import handle_gdn_mtp_cache_mode
from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
    fi_recovery_kernel,
    is_flashinfer_gdn_wy_output_only_available,
)
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def none_mode_args(**fields):
    base = dict(model_path="dummy", gdn_mtp_cache_mode="none")
    base.update(fields)
    return ServerArgs(**base)


class TestGdnMtpCacheModeArg(unittest.TestCase):
    def test_default_is_full_and_reaches_the_exec_bag(self):
        """The mode is opt-in: an untouched launch keeps per-draft snapshots."""
        args = prepare_server_args(["--model-path", "dummy"])
        self.assertEqual(args.gdn_mtp_cache_mode, "full")
        from sglang.srt.runtime_context import get_context

        with get_context().override_server_args(gdn_mtp_cache_mode="none"):
            from sglang.srt.runtime_context import get_exec

            self.assertEqual(get_exec().mamba.gdn_mtp_cache_mode, "none")

    def test_cli_and_keyword_forms_agree(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = ServerArgs.from_cli_args(
            parser.parse_args(["--model-path", "dummy", "--gdn-mtp-cache-mode", "none"])
        )
        self.assertEqual(parsed.gdn_mtp_cache_mode, none_mode_args().gdn_mtp_cache_mode)

    def test_choices_reject_an_unknown_mode(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--model-path", "dummy", "--gdn-mtp-cache-mode", "auto"]
            )


class TestValidateGdnMtpCacheMode(unittest.TestCase):
    def test_non_none_mode_bypasses_all_checks(self):
        # Both ReplaySSM flags and a tree draft are legal while the snapshots
        # exist: the guards below are none-mode's contract, not global ones.
        args = ServerArgs(
            model_path="dummy",
            gdn_mtp_cache_mode="full",
            enable_linear_replayssm_spec=True,
            enable_linear_replayssm=True,
            speculative_eagle_topk=8,
        )
        self.assertIsNone(handle_gdn_mtp_cache_mode(args))

    def test_none_mode_rejects_both_replayssm_flags(self):
        # RecoverSSM's runtime gate (GDNAttnBackend._recover_ssm) reads the cache
        # mode alone, so this validator is the only thing keeping the two verify
        # protocols from both claiming the verify-commit path. --enable-linear-
        # replayssm is separately reachable: it requires no_buffer, which
        # none-mode also permits, so no other guard excludes the combination.
        for flag in ("enable_linear_replayssm_spec", "enable_linear_replayssm"):
            with self.subTest(flag=flag):
                with self.assertRaises(ValueError) as cm:
                    handle_gdn_mtp_cache_mode(none_mode_args(**{flag: True}))
                self.assertIn(
                    "--gdn-mtp-cache-mode=none is mutually exclusive with ReplaySSM",
                    str(cm.exception),
                )

    def test_replayssm_guard_wins_over_topk_guard(self):
        with self.assertRaises(ValueError) as cm:
            handle_gdn_mtp_cache_mode(
                none_mode_args(
                    enable_linear_replayssm_spec=True, speculative_eagle_topk=8
                )
            )
        msg = str(cm.exception)
        self.assertIn("mutually exclusive with ReplaySSM", msg)
        self.assertNotIn("linear draft chain", msg)

    def test_none_mode_rejects_topk_gt_one(self):
        with self.assertRaises(ValueError) as cm:
            handle_gdn_mtp_cache_mode(none_mode_args(speculative_eagle_topk=2))
        msg = str(cm.exception)
        self.assertIn("--gdn-mtp-cache-mode=none requires a linear draft chain", msg)
        self.assertIn("--speculative-eagle-topk=2", msg)

    def test_none_mode_allows_linear_chain_and_any_radix_strategy(self):
        # none-mode + extra_buffer is supported on both recover paths (FlashInfer
        # via native output_state_indices, Triton via the output-index arg): a
        # boundary pass recomputes the tracked prefix state at runtime.
        for topk in (None, 1):
            for disable_radix in (True, False):
                for strategy in ("no_buffer", "extra_buffer", "extra_buffer_lazy"):
                    with self.subTest(topk=topk, strategy=strategy):
                        args = ServerArgs(
                            model_path="dummy",
                            gdn_mtp_cache_mode="none",
                            speculative_eagle_topk=topk,
                            disable_radix_cache=disable_radix,
                            mamba_radix_cache_strategy=strategy,
                        )
                        self.assertIsNone(handle_gdn_mtp_cache_mode(args))


class TestFiRecoveryKernel(unittest.TestCase):
    @staticmethod
    def _backend(**dispatcher_fields):
        return SimpleNamespace(kernel_dispatcher=SimpleNamespace(**dispatcher_fields))

    @staticmethod
    def _fake_fi_kernel(use_state_pool):
        kernel = FlashInferGDNKernel.__new__(FlashInferGDNKernel)
        kernel.use_state_pool = use_state_pool
        return kernel

    def test_missing_kernel_dispatcher(self):
        self.assertIsNone(fi_recovery_kernel(SimpleNamespace()))

    def test_dispatcher_without_decode_kernel(self):
        self.assertIsNone(fi_recovery_kernel(self._backend(decode_kernel=None)))

    def test_non_flashinfer_decode_kernel(self):
        self.assertIsNone(fi_recovery_kernel(self._backend(decode_kernel=object())))

    def test_flashinfer_kernel_needs_the_state_pool(self):
        # The state-pool requirement is what selects FlashInfer recovery (k/v read
        # as strided views of the persistent conv-out buffer) over the Triton
        # recover kernel with a flat k/v stash.
        with_pool = self._backend(decode_kernel=self._fake_fi_kernel(True))
        self.assertIs(fi_recovery_kernel(with_pool), with_pool.kernel_dispatcher.decode_kernel)
        without_pool = self._backend(decode_kernel=self._fake_fi_kernel(False))
        self.assertIsNone(fi_recovery_kernel(without_pool))


class TestFlashInferTargetVerifyPolicy(unittest.TestCase):
    @staticmethod
    def _kernel(*, regular, none_only):
        kernel = FlashInferGDNKernel.__new__(FlashInferGDNKernel)
        kernel.supports_target_verify = regular
        kernel.supports_none_mode_target_verify = none_only
        return kernel

    def test_sm120_none_mode_is_narrowly_enabled(self):
        kernel = self._kernel(regular=False, none_only=True)
        self.assertTrue(kernel.can_target_verify("none"))
        self.assertFalse(kernel.can_target_verify("full"))

    def test_existing_architecture_support_is_unchanged(self):
        kernel = self._kernel(regular=True, none_only=False)
        self.assertTrue(kernel.can_target_verify("full"))
        self.assertTrue(kernel.can_target_verify("none"))

    def test_unsupported_kernel_stays_disabled(self):
        kernel = self._kernel(regular=False, none_only=False)
        self.assertFalse(kernel.can_target_verify("none"))
        self.assertFalse(kernel.can_target_verify("full"))

    def test_wy_output_only_gate_is_a_quiet_bool(self):
        # The dispatcher consults this on every launch, on every platform, so the
        # probe must answer with a bool (and the same answer twice) rather than
        # raise or re-import when FlashInfer or its WY kernel is missing.
        first = is_flashinfer_gdn_wy_output_only_available()
        self.assertIsInstance(first, bool)
        self.assertIs(first, is_flashinfer_gdn_wy_output_only_available())


if __name__ == "__main__":
    unittest.main()
