"""TVM-FFI stream binding for HiCache kernel-io transfers (CPU contract).

The JIT HiCache copy kernels are TVM-FFI modules: their C++ launchers resolve
the stream from TVM-FFI's own thread-local stream table
(``TVMFFIEnvGetStream``), and in tvm-ffi 0.1.x nothing refreshes that table
when Python enters a ``torch.cuda.stream(...)`` context -- an unbound handle
of a non-default stream must be installed via the public ``use_torch_stream``
/ ``use_raw_stream`` contexts. ``L2TransferEngine`` submits the ``io_backend
== "kernel"`` copies inside a dedicated transfer-stream context and records
the completion ACK event on that stream, so an unbound FFI launch escapes the
event's span and the ACK can publish before the copy has run (host regions
reused under an in-flight write).

Pinned here on CPU, against an emulated tvm-ffi env-stream slot:

* the kernel backend enters the dependency's PUBLIC stream context and the
  prior FFI stream is restored afterwards (balanced enter/exit, slot reset);
* an exception from the transfer body propagates UNCHANGED (the bridge must
  not mistake it for a binding error and yield twice -- contextlib would
  mask the real failure with "generator didn't stop after throw"), and the
  prior FFI stream is still restored;
* a binding-entry failure is fatal BEFORE the transfer body runs -- warn and
  continue would launch unbound on the FFI default stream, the exact
  ACK-before-copy corruption this bridge fixes;
* no-ops happen only where no JIT kernel can be selected: non-kernel io
  backends, no accelerator in the process, or tvm-ffi absent (the JIT
  HiCache selectors build their modules through the same tvm-ffi load path,
  so their pool-init flags are False and kernel-io consists of prebuilt
  torch-stream-honoring launches).

The GPU end-to-end check (the ACK really covers the JIT copy on a real
transfer stream, and the short/empty FILE-read degradations under the real
posix plugin) runs with the HiCache regression on thegrid.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.l2_transfer import L2TransferEngine

SENTINEL = RuntimeError("transfer body failure")
BIND_FAILURE = RuntimeError("FFI stream binding failed")


class _FakeFFI:
    """Emulates tvm-ffi's per-device env-stream slot and public context."""

    def __init__(self, *, enter_raises=False):
        self.slot = "default"  # the env stream before anything binds
        self.entered = 0
        self.exited = 0
        self.calls = 0
        self._enter_raises = enter_raises

    def use_torch_stream(self, context=None):
        self.calls += 1
        if self._enter_raises:
            raise BIND_FAILURE
        fake = self

        class _Ctx:
            def __enter__(inner):
                inner.prev = fake.slot
                fake.slot = "transfer"
                fake.entered += 1

            def __exit__(inner, *exc):
                fake.slot = inner.prev
                fake.exited += 1
                return False

        return _Ctx()


class TestFfiStreamBridge(unittest.TestCase):
    def _bridge(self):
        module = sys.modules["sglang.srt.mem_cache.l2_transfer"]
        bridge = getattr(L2TransferEngine, "_ffi_kernel_stream_bridge", None) or getattr(
            module, "_ffi_kernel_stream_bridge", None
        )
        self.assertIsNotNone(
            bridge, "kernel-io submissions must bind the FFI launch stream"
        )
        return bridge

    def _install(self, fake):
        patch = mock.patch.dict(sys.modules, {"tvm_ffi": fake})
        patch.start()
        self.addCleanup(patch.stop)
        return patch

    def _accelerator(self):
        patch = mock.patch.object(torch.cuda, "is_available", return_value=True)
        patch.start()
        self.addCleanup(patch.stop)

    # ---- required path: kernel backend + accelerator + installed API ----

    def test_kernel_io_enters_public_context_and_restores_prior_stream(self):
        fake = _FakeFFI()
        self._install(fake)
        self._accelerator()
        body_ran = 0
        with self._bridge()(True):
            body_ran += 1
            self.assertEqual(fake.slot, "transfer")
            self.assertEqual((fake.entered, fake.exited), (1, 0))
        self.assertEqual(body_ran, 1)
        self.assertEqual((fake.entered, fake.exited), (1, 1))
        self.assertEqual(fake.slot, "default")  # prior stream restored

    def test_body_exception_propagates_unchanged_and_stream_restored(self):
        """The review pin: the body's own failure must not be mistaken for a
        binding failure (a second yield raises 'generator didn't stop after
        throw' and masks the real error)."""
        fake = _FakeFFI()
        self._install(fake)
        self._accelerator()
        with self.assertRaises(RuntimeError) as caught:
            with self._bridge()(True):
                raise SENTINEL
        self.assertIs(caught.exception, SENTINEL)
        self.assertEqual(fake.calls, 1)
        self.assertEqual((fake.entered, fake.exited), (1, 1))
        self.assertEqual(fake.slot, "default")

    def test_binding_failure_is_fatal_before_the_body(self):
        """Warn-and-continue would launch unbound on the FFI default stream,
        recreating the ACK-before-copy hazard the bridge exists to fix."""
        fake = _FakeFFI(enter_raises=True)
        self._install(fake)
        self._accelerator()
        body_ran = 0
        with self.assertRaises(RuntimeError) as caught:
            with self._bridge()(True):
                body_ran += 1
        self.assertIs(caught.exception, BIND_FAILURE)
        self.assertEqual(body_ran, 0)
        self.assertEqual((fake.entered, fake.exited), (0, 0))

    # ---- no-op gates: places where no JIT kernel can be selected ----

    def test_direct_io_is_a_noop_without_binding(self):
        fake = _FakeFFI()
        self._install(fake)
        self._accelerator()
        with self._bridge()(False):
            pass
        self.assertEqual(fake.calls, 0)

    def test_no_accelerator_is_a_noop(self):
        fake = _FakeFFI(enter_raises=True)  # would blow up if bound
        self._install(fake)
        ran = 0
        with self._bridge()(True):  # torch.cuda.is_available() False here
            ran += 1
        self.assertEqual(ran, 1)
        self.assertEqual(fake.calls, 0)

    def test_tvm_ffi_absent_runs_unbound_because_jit_cannot_be_selected(self):
        """On an accelerator host without tvm-ffi the bridge no-ops: the JIT
        HiCache selectors load through the same dependency, so their pool-init
        flags are False and no FFI launch exists to bind (see the bridge
        docstring). What must NOT happen is a silent failure on the required
        installed path -- covered by test_binding_failure_is_fatal_before_the_body."""
        self._install(None)  # sys.modules entry None -> import raises ImportError
        self._accelerator()
        ran = 0
        with self._bridge()(True):  # must not raise
            ran += 1
        self.assertEqual(ran, 1)

    def test_submission_wires_the_bridge_to_the_kernel_backend(self):
        source = inspect.getsource(L2TransferEngine._submission)
        self.assertIn(
            "_ffi_kernel_stream_bridge",
            source,
            "_submission must gate the FFI binding on the kernel io backend",
        )
        self.assertIn("self.io_backend", source)


if __name__ == "__main__":
    unittest.main()
