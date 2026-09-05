"""CPU ownership tests for breakable-graph replay closures.

These exercise the real eager wrapper without a CUDA capture. The CUDA-only
non-owning tensor helper is modeled by weakref.proxy; allocator reuse itself
is covered by the shared-pool CUDA regression in the breakable graph suite.
"""

import gc
import sys
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBreakableCudaGraphLifetime(unittest.TestCase):
    def setUp(self):
        self.graph = bcg.BreakableCUDAGraph()
        self.events = []
        self.capture = SimpleNamespace(
            cuda_graph=self.graph,
            _end_current_segment=lambda: self.events.append("end"),
            _begin_new_segment=lambda: self.events.append("begin"),
            _barrier_fn=lambda: self.events.append("barrier"),
        )
        self.capture_token = bcg._current_capture_var.set(self.capture)
        self.addCleanup(bcg._current_capture_var.reset, self.capture_token)
        # The pre-fix implementation calls a CUDA-only non-owning helper.
        # Model that ownership boundary without initializing CUDA on CPU CI.
        self.weak_module_patch = patch.dict(
            sys.modules,
            {
                "sglang.srt.compilation.weak_ref_tensor": SimpleNamespace(
                    weak_ref_tensors=weakref.proxy
                )
            },
        )
        self.weak_module_patch.start()
        self.addCleanup(self.weak_module_patch.stop)

    def test_inputs_are_owned_until_the_replay_closure_is_released(self):
        for placement in ("positional", "tuple", "list", "keyword", "dict"):
            with self.subTest(placement=placement):

                @bcg.eager_on_graph(enable=True)
                def bridge(*args, **kwargs):
                    value = kwargs["value"] if kwargs else args[0]
                    if isinstance(value, (tuple, list)):
                        value = value[0]
                    if isinstance(value, dict):
                        value = value["tensor"]
                    return value * 2

                value = torch.tensor([1.0, 2.0])
                value_ref = weakref.ref(value)
                if placement == "tuple":
                    output = bridge((value,))
                elif placement == "list":
                    output = bridge([value])
                elif placement == "keyword":
                    output = bridge(value=value)
                elif placement == "dict":
                    output = bridge({"tensor": value})
                else:
                    output = bridge(value)
                del value
                gc.collect()

                self.assertIsNotNone(
                    value_ref(), "replay must own its input after capture locals die"
                )
                value_ref().add_(3)
                self.graph._break_fns[-1]()
                torch.testing.assert_close(output, torch.tensor([8.0, 10.0]))
                self.graph._break_fns.clear()
                gc.collect()
                self.assertIsNone(value_ref(), "released graphs must release inputs")

    def test_replay_snapshots_sequence_inputs_before_caller_mutation(self):
        for placement in ("positional", "keyword", "nested"):
            for mutation in ("replace", "clear"):
                with self.subTest(placement=placement, mutation=mutation):

                    @bcg.eager_on_graph(enable=True)
                    def bridge(values):
                        if isinstance(values, tuple):
                            values = values[0]
                        return values[0] * 3

                    values = [torch.tensor([2.0])]
                    original_ref = weakref.ref(values[0])
                    if placement == "keyword":
                        output = bridge(values=values)
                    elif placement == "nested":
                        output = bridge((values,))
                    else:
                        output = bridge(values)
                    if mutation == "replace":
                        values[0] = torch.tensor([99.0])
                    else:
                        values.clear()
                    gc.collect()

                    self.assertIsNotNone(
                        original_ref(), "caller mutation must not release replay inputs"
                    )
                    self.graph._break_fns[-1]()
                    torch.testing.assert_close(output, torch.tensor([6.0]))
                    self.graph._break_fns.clear()
                    gc.collect()
                    self.assertIsNone(original_ref())

    def test_capture_stub_replay_owns_inputs_and_output_bridge(self):
        calls = []

        def stub(value):
            calls.append("stub")
            return torch.zeros_like(value)

        @bcg.eager_on_graph(enable=True, capture_stub=stub)
        def bridge(value):
            calls.append("inner")
            return value * 3

        value = torch.tensor([2.0])
        value_ref = weakref.ref(value)
        output = bridge(value)
        output_ref = weakref.ref(output)
        del value, output
        gc.collect()
        self.assertEqual(calls, ["stub"])
        self.assertEqual(self.events, ["end", "barrier", "begin"])
        self.assertIsNotNone(value_ref())
        self.assertIsNotNone(output_ref())
        self.graph._break_fns[0]()
        self.assertEqual(calls, ["stub", "inner"])
        torch.testing.assert_close(output_ref(), torch.tensor([6.0]))
        self.graph._break_fns.clear()
        gc.collect()
        self.assertIsNone(value_ref())
        self.assertIsNone(output_ref())

    def test_outside_capture_does_not_retain_arguments(self):
        token = bcg._current_capture_var.set(None)
        try:

            @bcg.eager_on_graph(enable=True)
            def bridge(value):
                return value + 1

            value = torch.tensor([2.0])
            value_ref = weakref.ref(value)
            output = bridge(value)
            del value
            gc.collect()
            self.assertIsNone(value_ref())
            self.assertFalse(self.graph._break_fns)
            self.assertFalse(self.events)
            torch.testing.assert_close(output, torch.tensor([3.0]))
        finally:
            bcg._current_capture_var.reset(token)


if __name__ == "__main__":
    unittest.main()
