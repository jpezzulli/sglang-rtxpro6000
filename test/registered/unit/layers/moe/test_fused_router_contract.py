"""CPU checks of router dispatch and source-level dependency ordering.

Load the production host functions without importing GPU backends; only their
kernel boundaries are mocked. Numerical and compiled PDL ordering checks live
in the CUDA router tests, since source order alone cannot prove device order.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import triton

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[5]
GATE = ROOT / "python/sglang/kernels/ops/moe/moe_fused_gate.py"
TOPK = ROOT / "python/sglang/srt/layers/moe/topk.py"
RADIX = ROOT / "python/sglang/kernels/jit/csrc/moe/route_radix.cuh"


def _function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def gate():
    kernel = MagicMock()
    radix = MagicMock()
    radix.covered.return_value = False
    namespace = {
        "torch": torch,
        "triton": triton,
        "_SCORING_FUNC_MAP": {"sigmoid": 0, "sqrtsoftplus": 1, "softmax": 2},
        "_router_triton_kernel": kernel,
        "moe_route_radix": radix,
        "is_arch_support_pdl": lambda: True,
    }
    return _function(GATE, "moe_fused_gate", namespace), kernel, radix, namespace


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
def test_softmax_none_bias_has_no_buffer_or_bias_load(gate, pdl, renormalize):
    fn, kernel, radix, namespace = gate
    namespace["is_arch_support_pdl"] = lambda: pdl
    scores = torch.randn(4, 512)
    with patch.object(
        torch, "zeros", side_effect=AssertionError("zero bias allocated")
    ):
        weights, indices = fn(scores, None, 10, "softmax", renormalize=renormalize)
    args, kwargs = kernel.__getitem__.return_value.call_args
    assert args[0] is scores and args[1] is scores
    assert args[2] is weights and args[3] is indices
    assert kwargs["HAS_BIAS"] is False
    assert kwargs["USE_PDL"] is pdl
    assert kwargs.get("launch_pdl", False) is pdl
    assert kwargs["RENORMALIZE"] is renormalize
    assert kwargs["SCORING_FUNC"] == 2
    radix.covered.assert_not_called()


@pytest.mark.parametrize("scoring", ["softmax", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_real_bias_keeps_dtype_and_kernel_pointer(gate, scoring, dtype):
    fn, kernel, _, _ = gate
    scores, bias = torch.randn(4, 128), torch.randn(128).to(dtype)
    fn(scores, bias, 8, scoring)
    args, kwargs = kernel.__getitem__.return_value.call_args
    assert args[1] is bias
    assert kwargs["HAS_BIAS"] is True
    assert kwargs["USE_PDL"] is True


@pytest.mark.parametrize("scoring", ["sigmoid", "sqrtsoftplus"])
def test_non_softmax_none_bias_remains_fail_loud(gate, scoring):
    fn, kernel, _, _ = gate
    with pytest.raises(
        AssertionError, match="bias is required for non-softmax routing"
    ):
        fn(torch.randn(4, 128), None, 8, scoring)
    kernel.__getitem__.assert_not_called()


def test_sigmoid_radix_dispatch_keeps_real_bias(gate):
    fn, kernel, radix, _ = gate
    radix.covered.return_value = True
    scores, bias = torch.randn(4, 896), torch.randn(896)
    result = fn(scores, bias, 16, "sigmoid")
    assert result is radix.route_radix.return_value
    args, kwargs = radix.route_radix.call_args
    assert args[0] is scores and args[1] is bias
    assert args[2] == 16 and kwargs == {"sorted": False}
    kernel.__getitem__.assert_not_called()


def test_fused_topk_passes_none_without_allocating_zero_bias():
    router = MagicMock(return_value=(torch.empty(4, 10), torch.empty(4, 10)))
    module_name = "sglang.kernels.ops.moe.moe_fused_gate"
    module = types.ModuleType(module_name)
    module.moe_fused_gate = router
    fn = _function(
        TOPK, "fused_topk", {"torch": torch, "_is_cuda": True, "_use_aiter": False}
    )
    scores = torch.randn(4, 512)
    with (
        patch.dict(sys.modules, {module_name: module}),
        patch.object(torch, "zeros", side_effect=AssertionError("zero bias allocated")),
    ):
        result = fn(torch.randn(4, 64), scores, 10, True)
    assert result == router.return_value
    args, kwargs = router.call_args
    assert args[0] is scores and args[1] is None and args[2] == 10
    assert kwargs == {"scoring_func": "softmax", "renormalize": True}


def test_triton_source_wait_precedes_all_input_loads():
    tree = ast.parse(GATE.read_text())
    kernel = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_router_triton_kernel"
    )
    calls = [n for n in ast.walk(kernel) if isinstance(n, ast.Call)]
    wait = [n for n in calls if ast.unparse(n.func) == "tl.extra.cuda.gdc_wait"]
    loads = [n for n in calls if ast.unparse(n.func) == "tl.load"]
    assert len(wait) == 1 and loads
    assert all(wait[0].lineno < n.lineno for n in loads)


def test_radix_source_wait_precedes_both_input_loads():
    # Scope the check to the common body also used by route_quant_fused.cuh.
    body = RADIX.read_text().split("SGL_DEVICE void route_radix_block(", 1)[1]
    wait = body.index("PDLWaitPrimary<kUsePDL>();")
    assert wait < body.index("bias_vec.load(params.bias, tx);")
    assert wait < body.index("scores_vec.load(scores, tx);")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
