"""Unit tests for the RecoverSSM (``--gdn-mtp-cache-mode=none``) contract.

CPU-only: the mode's argument surface, the guards that decide which launches may
ask for it, the verify-kernel capability gate that keeps SM120's full mode on its
established Triton verifier, and the single predicate that decides which recovery
path runs. None of it needs a GPU; none of it proves GPU numerics (see
``test/registered/attention/unittests/gdn/test_recover_ssm_sm120.py``).
"""

import argparse
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.mamba_hook import handle_gdn_mtp_cache_mode
from sglang.srt.layers.attention.linear import gdn_backend as gdn_backend_module
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    GDNKernelDispatcher,
)
from sglang.srt.layers.attention.linear.kernels import gdn_flashinfer as fi_module
from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
    fi_recovery_kernel,
    is_flashinfer_gdn_wy_output_only_available,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
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
            parser.parse_args(["--model-path", "dummy", "--gdn-mtp-cache-mode", "auto"])


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
        self.assertIs(
            fi_recovery_kernel(with_pool), with_pool.kernel_dispatcher.decode_kernel
        )
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


def _fake_flashinfer_kernel(*, regular, none_only):
    """A FlashInfer kernel whose can_target_verify is the real one.

    Built with __new__ because __init__ needs a device and a FlashInfer import;
    the two capability flags below are everything the selection logic reads.
    """
    kernel = FlashInferGDNKernel.__new__(FlashInferGDNKernel)
    kernel.supports_target_verify = regular
    kernel.supports_none_mode_target_verify = none_only
    kernel.use_state_pool = True
    return kernel


@contextmanager
def _flashinfer_factory(kernel):
    """Make the dispatcher's lazy `FlashInferGDNKernel()` return `kernel`.

    is_cuda is faked too: the dispatcher refuses a FlashInfer backend off-CUDA, and
    the selection logic under test is architecture-independent by design (the
    capability flags live on the kernel, not on this process's device).
    """
    with (
        patch.object(fi_module, "FlashInferGDNKernel", lambda: kernel),
        patch.object(gdn_backend_module, "is_cuda", return_value=True),
    ):
        yield


class TestDispatcherReceivesTheMode(unittest.TestCase):
    """The mode has to reach the kernel selection, and reach it as a decision.

    A dispatcher that ignores the fourth argument (or takes none) either breaks
    every GDN launch at construction or leaves SM120 none-mode unable to select
    the WY verifier it is gated on -- so both the arity and the routing are
    pinned here, on a platform with no CUDA and no FlashInfer.
    """

    def test_the_constructor_takes_the_mode_positionally(self):
        with _flashinfer_factory(
            _fake_flashinfer_kernel(regular=True, none_only=False)
        ):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.FLASHINFER,
                LinearAttnKernelBackend.FLASHINFER,
                None,
                "none",
            )
        self.assertTrue(dispatcher.verify_kernel_is_flashinfer)

    def test_three_argument_call_keeps_the_full_mode_behaviour(self):
        # An omitted mode is "full": nothing may newly opt in to the WY path.
        kernel = _fake_flashinfer_kernel(regular=False, none_only=True)
        with _flashinfer_factory(kernel):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.FLASHINFER,
                LinearAttnKernelBackend.FLASHINFER,
            )
        self.assertIsInstance(dispatcher.verify_kernel, TritonGDNKernel)
        self.assertFalse(dispatcher.verify_kernel_is_flashinfer)

    def _select(self, *, kernel, mode):
        with _flashinfer_factory(kernel):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.FLASHINFER,
                LinearAttnKernelBackend.FLASHINFER,
                None,
                mode,
            )
        return dispatcher

    def test_sm120_admits_none_mode_and_keeps_full_mode_on_triton(self):
        # SM120 today: Triton verifier in full mode, the none-mode WY kernel in
        # none mode -- and only there.
        kernel = _fake_flashinfer_kernel(regular=False, none_only=True)
        full = self._select(kernel=kernel, mode="full")
        self.assertIsInstance(full.verify_kernel, TritonGDNKernel)
        self.assertFalse(full.verify_kernel_is_flashinfer)
        none = self._select(kernel=kernel, mode="none")
        self.assertIs(none.verify_kernel, kernel)
        self.assertTrue(none.verify_kernel_is_flashinfer)

    def test_architectures_that_already_verify_keep_doing_so_in_both_modes(self):
        # SM90/SM100: unchanged by the mode, including when their FlashInfer build
        # has no WY kernel at all.
        for none_only in (True, False):
            for mode in ("full", "none"):
                with self.subTest(none_only=none_only, mode=mode):
                    dispatcher = self._select(
                        kernel=_fake_flashinfer_kernel(
                            regular=True, none_only=none_only
                        ),
                        mode=mode,
                    )
                    self.assertTrue(dispatcher.verify_kernel_is_flashinfer)

    def test_an_explicit_triton_verify_backend_still_wins(self):
        with _flashinfer_factory(_fake_flashinfer_kernel(regular=True, none_only=True)):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.FLASHINFER,
                LinearAttnKernelBackend.FLASHINFER,
                LinearAttnKernelBackend.TRITON,
                "none",
            )
        self.assertIsInstance(dispatcher.verify_kernel, TritonGDNKernel)
        self.assertFalse(dispatcher.verify_kernel_is_flashinfer)

    def test_an_unsupported_kernel_stays_on_triton_in_both_modes(self):
        for mode in ("full", "none"):
            with self.subTest(mode=mode):
                dispatcher = self._select(
                    kernel=_fake_flashinfer_kernel(regular=False, none_only=False),
                    mode=mode,
                )
                self.assertIsInstance(dispatcher.verify_kernel, TritonGDNKernel)


class TestBackendPassesTheModeToTheDispatcher(unittest.TestCase):
    """The real GDNAttnBackend.__init__ hands its resolved mode to the dispatcher."""

    class _Recorder:
        def __init__(self, *args):
            self.args = args
            self.tree_verify_kernel = TritonGDNKernel.__new__(TritonGDNKernel)
            self.decode_kernel = self.extend_kernel = self.verify_kernel = (
                self.tree_verify_kernel
            )
            self.supports_packed_decode = False

        def target_verify_supports_strided_qkv(self, retrieve_parent_token):
            return False

    def _build(self, *, mode):
        backend = GDNAttnBackend.__new__(GDNAttnBackend)
        pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(conv=[SimpleNamespace(shape=(1, 3))])
        )
        reqs = SimpleNamespace(size=8)

        def _base_init(self, model_runner):
            self.state_pool = pool
            self.req_to_token_pool = reqs
            self.is_draft_worker = False

        published = SimpleNamespace(
            features=SimpleNamespace(enable_mis=False),
            mamba=SimpleNamespace(gdn_mtp_cache_mode=mode),
            deterministic=SimpleNamespace(enable_deterministic_inference=False),
        )
        model_runner = SimpleNamespace(
            linear_attn_backends=SimpleNamespace(
                decode=LinearAttnKernelBackend.FLASHINFER,
                prefill=LinearAttnKernelBackend.FLASHINFER,
                verify=None,
            ),
            device="cpu",
        )
        with (
            patch.object(gdn_backend_module, "get_exec", return_value=published),
            patch.object(gdn_backend_module, "_validate_gdn_linear_attn_backends"),
            patch.object(gdn_backend_module, "validate_gdn_mis_backend"),
            patch.object(
                gdn_backend_module,
                "build_verify_intermediate_state_indices",
                return_value=SimpleNamespace(),
            ),
            patch.object(gdn_backend_module, "GDNKernelDispatcher", self._Recorder),
            patch.object(
                gdn_backend_module.MambaAttnBackendBase, "__init__", _base_init
            ),
        ):
            GDNAttnBackend.__init__(backend, model_runner)
        return backend

    def test_none_mode_reaches_the_dispatcher_and_the_runtime_gate(self):
        backend = self._build(mode="none")
        self.assertEqual(backend.kernel_dispatcher.args[3], "none")
        self.assertTrue(backend._recover_ssm)
        # The recovery buffers exist (empty) from construction: forward_extend fills
        # them per layer, and a missing attribute here is an AttributeError at the
        # first verify instead of a mode that silently does nothing.
        self.assertEqual(backend._no_cache_stash, {})
        self.assertEqual(backend._conv_out_persist, {})
        self.assertIsNone(backend._no_cache_draft_token_num)

    def test_full_mode_is_unchanged(self):
        backend = self._build(mode="full")
        self.assertEqual(backend.kernel_dispatcher.args[3], "full")
        self.assertFalse(backend._recover_ssm)


if __name__ == "__main__":
    unittest.main()
