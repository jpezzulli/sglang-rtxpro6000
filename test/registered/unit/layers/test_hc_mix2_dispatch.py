"""CPU dispatch / integration coverage for the direct donor HC MIX2 port.

Gate truth table, scale pairing, launch geometry, workspace lifetime and the
fallbacks, without a GPU: the three kernel objects are replaced by recorders,
so this suite checks what the launcher hands the donor kernels. What SM120 does
with them lives in ``test/registered/kernels/test_hc_mix2.py``.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../python"))


def _is_missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except ValueError:
        # `python3 FILE -f` runs this file as __main__ (stubs installed, spec
        # None) and pytest imports it a second time afterwards: the stubs are
        # already in sys.modules, which is exactly what that run needs.
        return False


_STUBBED = [name for name in ("triton", "msgspec") if _is_missing(name)]


def _package(name, path=None):
    module = types.ModuleType(name)
    module.__path__ = [path] if path else []
    sys.modules[name] = module
    return module


def _install_import_stubs():
    """Stand in for build-time deps this CPU host lacks; nothing under test here.

    ``triton.jit`` decorates the real kernel functions (bodies untouched, never
    traced), the ``sglang`` submodules are the real files resolved through the
    package paths, and the package __init__ chains a wheel-less checkout would
    trip over are bypassed.
    """
    _package("sglang", os.path.join(_PY, "sglang"))
    for name in (
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.gemm",
        "sglang.srt",
        "sglang.srt.layers",
    ):
        _package(name, os.path.join(_PY, *name.split(".")))

    triton = types.ModuleType("triton")

    class _Kernel:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            return lambda *args, **kwargs: None

    triton.jit = lambda fn: _Kernel(fn)
    triton.cdiv = lambda a, b: -(-a // b)
    triton.next_power_of_2 = lambda n: max(1, 2 ** math.ceil(math.log2(max(1, n))))
    language = types.ModuleType("triton.language")
    extra = types.ModuleType("triton.language.extra")
    cuda = types.ModuleType("triton.language.extra.cuda")
    language.constexpr = "constexpr"
    language.extra = extra
    extra.cuda = cuda
    cuda.gdc_wait = lambda: None
    cuda.gdc_launch_dependents = lambda: None
    sys.modules.update(
        {
            "triton": triton,
            "triton.language": language,
            "triton.language.extra": extra,
            "triton.language.extra.cuda": cuda,
        }
    )

    msgspec = types.ModuleType("msgspec")

    class Struct:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    msgspec.Struct = Struct
    sys.modules["msgspec"] = msgspec

    environ = types.ModuleType("sglang.srt.environ")

    class _StubEnvs:
        """Every SGLANG_* flag reads as unset, like a fresh environment."""

        def __getattr__(self, name):
            return SimpleNamespace(get=lambda: False)

    environ.envs = _StubEnvs()
    sys.modules["sglang.srt.environ"] = environ


if _STUBBED:
    _install_import_stubs()

from sglang.kernels.ops.gemm.sm120_online_fp8 import (  # noqa: E402
    _SCALE_ATTR,
    quantize_rowwise_fp8,
)
from sglang.srt.layers import hc_mix2_triton as mix2  # noqa: E402
from sglang.srt.layers import hc_mix_triton  # noqa: E402
from sglang.srt.layers import hyperconnection as hyp  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HC = 4
HS = 2560
K = HC * HS
LOWRANK = 320
EPS = 1e-6


def _attrs_tensor(
    shape,
    dtype=torch.bfloat16,
    *,
    cuda=True,
    contiguous=True,
    device=torch.device("cuda", 0),
    scale=None,
):
    """Stand-in carrying exactly the attributes the gate reads."""
    tensor = SimpleNamespace(
        shape=torch.Size(shape),
        dtype=dtype,
        is_cuda=cuda,
        device=device,
        dim=lambda: len(shape),
        numel=lambda: math.prod(shape),
        is_contiguous=lambda: contiguous,
    )
    if scale is not None:
        setattr(tensor, _SCALE_ATTR, scale)
    return tensor


def _fp8_pair(*, scales=True, shapes=True):
    down = _attrs_tensor(
        (LOWRANK, K) if shapes else (LOWRANK, 2 * K),
        torch.float8_e4m3fn,
        scale=_attrs_tensor([LOWRANK], torch.float32) if scales else None,
    )
    up = _attrs_tensor(
        (K, LOWRANK) if shapes else (K, 2 * LOWRANK),
        torch.float8_e4m3fn,
        scale=_attrs_tensor([K], torch.float32) if scales else None,
    )
    return down, up


def _resident_pair(seed=0, rows=16):
    """Real FP8 bytes + real FP32 row scales from this base's own ingestion.

    CPU tensors, because `hc_norm_mix2` allocates its workspace on
    `hyper_input.device` and the launcher has to be inspectable without SM120.
    """
    generator = torch.Generator().manual_seed(seed)

    def _quantize(out_rows, columns, scale):
        weight = (
            torch.randn(out_rows, columns, generator=generator, dtype=torch.float32)
            * scale
        ).to(torch.bfloat16)
        quantized, row_scale = quantize_rowwise_fp8(weight)
        setattr(quantized, _SCALE_ATTR, row_scale)
        return quantized

    hyper_input = (
        torch.randn(rows, K, generator=generator, dtype=torch.float32) * 0.25
    ).to(torch.bfloat16)
    norm_w = (torch.randn(K, generator=generator, dtype=torch.float32) * 0.02).to(
        torch.bfloat16
    )
    return (
        hyper_input,
        norm_w,
        _quantize(LOWRANK, K, 1.0 / math.sqrt(K)),
        _quantize(K, LOWRANK, 1.0 / math.sqrt(LOWRANK)),
    )


@pytest.fixture
def sm120(monkeypatch):
    """This base's target hardware, as far as the gate is concerned."""
    monkeypatch.setattr(mix2, "_is_exact_sm120", lambda device: True)
    monkeypatch.setattr(mix2, "_tp_size_cached", 1)
    monkeypatch.setattr(hc_mix_triton, "_deterministic_inference", lambda: False)
    return mix2


def test_gate_accepts_the_rowwise_fp8_decode_shapes(sm120):
    down, up = _fp8_pair()
    for rows in (1, 4, 8, 16):
        assert (
            mix2.hc_norm_mix2_supported(
                _attrs_tensor((rows, K)), _attrs_tensor([K]), down, up, HC, HS
            )
            is True
        ), rows


def test_gate_keeps_everything_else_on_the_existing_path(sm120):
    down, up = _fp8_pair()
    activation, norm_w = _attrs_tensor((4, K)), _attrs_tensor([K])
    reject = {
        "empty batch": (_attrs_tensor((0, K)), norm_w, down, up, HC, HS),
        "C6 verify width 24": (_attrs_tensor((24, K)), norm_w, down, up, HC, HS),
        "prefill 33": (_attrs_tensor((33, K)), norm_w, down, up, HC, HS),
        "fp16 activation": (
            _attrs_tensor((4, K), torch.float16),
            norm_w,
            down,
            up,
            HC,
            HS,
        ),
        "non-contiguous activation": (
            _attrs_tensor((4, K), contiguous=False),
            norm_w,
            down,
            up,
            HC,
            HS,
        ),
        "3-D activation": (_attrs_tensor((1, 4, K)), norm_w, down, up, HC, HS),
        "cpu activation": (
            _attrs_tensor((4, K), cuda=False, device=torch.device("cpu")),
            norm_w,
            down,
            up,
            HC,
            HS,
        ),
        "shared-weight norm": (
            _attrs_tensor((4, K)),
            _attrs_tensor([HS]),
            down,
            up,
            HC,
            HS,
        ),
        "bf16 mix weights": (
            activation,
            norm_w,
            _attrs_tensor((LOWRANK, K), torch.bfloat16),
            _attrs_tensor((K, LOWRANK), torch.bfloat16),
            HC,
            HS,
        ),
        "wrong lowrank shape": (activation, norm_w, *_fp8_pair(shapes=False), HC, HS),
        "wrong hc": (activation, norm_w, down, up, 8, HS // 2),
        "wrong hidden": (activation, norm_w, down, up, 4, 1280),
    }
    for name, args in reject.items():
        assert mix2.hc_norm_mix2_supported(*args) is False, name


def test_gate_rejects_other_hardware_tp_and_determinism(sm120, monkeypatch):
    down, up = _fp8_pair()
    args = (_attrs_tensor((4, K)), _attrs_tensor([K]), down, up, HC, HS)
    assert mix2.hc_norm_mix2_supported(*args) is True

    # Larger row counts (the C6 W4 target's 24) and TP2 keep the existing path;
    # so does deterministic inference, because K1 accumulates the down
    # projection with device-scope atomics whose order is not reproducible.
    monkeypatch.setattr(mix2, "_is_exact_sm120", lambda device: False)
    assert mix2.hc_norm_mix2_supported(*args) is False
    monkeypatch.setattr(mix2, "_is_exact_sm120", lambda device: True)
    monkeypatch.setattr(mix2, "_tp_size", lambda: 2)
    assert mix2.hc_norm_mix2_supported(*args) is False
    monkeypatch.setattr(mix2, "_tp_size", lambda: 1)
    monkeypatch.setattr(hc_mix_triton, "_deterministic_inference", lambda: True)
    assert mix2.hc_norm_mix2_supported(*args) is False


def test_tp_size_is_read_from_the_runtime_context(sm120, monkeypatch):
    """The gate follows the fork's accessor; the legacy shim stays ratcheted.

    test_legacy_global_ratchet.py is already red on this base because of
    hc_mix_triton, so the port must not add a call-site of its own -- and
    `sglang.srt.server_args` must stay out of the port's imports entirely.
    """
    import inspect

    source = inspect.getsource(mix2)
    assert "get_global_server_args" not in source
    assert "sglang.srt.server_args" not in source
    assert "runtime_context" in inspect.getsource(mix2._tp_size)

    stub = types.ModuleType("sglang.srt.runtime_context")
    down, up = _fp8_pair()
    args = (_attrs_tensor((4, K)), _attrs_tensor([K]), down, up, HC, HS)
    monkeypatch.setattr(mix2, "_tp_size_cached", None)
    stub.get_server_args = lambda: SimpleNamespace(tp_size=2)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", stub)
    assert mix2._tp_size() == 2
    assert mix2.hc_norm_mix2_supported(*args) is False

    monkeypatch.setattr(mix2, "_tp_size_cached", None)
    stub.get_server_args = lambda: SimpleNamespace(tp_size=1)
    assert mix2._tp_size() == 1
    assert mix2.hc_norm_mix2_supported(*args) is True


def test_unpublished_runtime_context_does_not_latch_tp_size(sm120, monkeypatch):
    """The no-context fallback is per-call, never permanent.

    The gate reads the TP width before activation eligibility, so a probe made
    while no ServerArgs is published has to answer 1 for that call only and
    leave the cache unset. Caching the fallback would strand a later TP2 rank
    on MIX2 -- which is exactly the configuration the guard exists for.
    """
    down, up = _fp8_pair()
    args = (_attrs_tensor((4, K)), _attrs_tensor([K]), down, up, HC, HS)
    monkeypatch.setattr(mix2, "_tp_size_cached", None)

    def unpublished():
        raise AttributeError("runtime context has not published server_args")

    stub = types.ModuleType("sglang.srt.runtime_context")
    stub.get_server_args = unpublished
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", stub)
    assert mix2._tp_size() == 1
    assert mix2._tp_size_cached is None
    assert mix2.hc_norm_mix2_supported(*args) is True

    # Same answer on the import-failure path, still without latching.
    with monkeypatch.context() as gone:
        gone.delitem(sys.modules, "sglang.srt.runtime_context")
        assert mix2._tp_size() == 1
        assert mix2._tp_size_cached is None

    # Publication after the probe is seen, and a successful read does stick.
    stub.get_server_args = lambda: SimpleNamespace(tp_size=2)
    assert mix2._tp_size() == 2
    assert mix2._tp_size_cached == 2
    assert mix2.hc_norm_mix2_supported(*args) is False
    with monkeypatch.context() as gone:
        gone.delitem(sys.modules, "sglang.srt.runtime_context")
        assert mix2._tp_size() == 2
    assert mix2.hc_norm_mix2_supported(*args) is False


def test_half_quantized_pair_fails_loudly(sm120):
    down, _ = _fp8_pair()
    up = _attrs_tensor((K, LOWRANK), torch.float8_e4m3fn)  # FP8 storage, no row scale
    with pytest.raises(RuntimeError, match="rowwise-FP8"):
        mix2.hc_norm_mix2_supported(
            _attrs_tensor((4, K)), _attrs_tensor([K]), down, up, HC, HS
        )


class _Recorder:
    def __init__(self):
        self.launches = []

    def kernel(self, name):
        launches = self.launches

        class _Kernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    launches.append((name, grid, args, kwargs))

                return launch

        return _Kernel()

    def by_name(self):
        return {name: (grid, args) for name, grid, args, _ in self.launches}


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    for name in ("_hc_branch_stats_kernel", "_hc_down_kernel", "_hc_up_kernel"):
        monkeypatch.setattr(mix2, name, rec.kernel(name))
    return rec


def _launch(sm120, recorder, rows=4, seed=0):
    hyper_input_all, norm_w, down, up = _resident_pair(seed=seed)
    hyper_input = hyper_input_all[:rows]
    mixed, normed = mix2.hc_norm_mix2(hyper_input, norm_w, EPS, down, up, HC, HS)
    return mixed, normed, recorder.by_name(), (hyper_input, norm_w, down, up)


def test_launch_geometry_and_pdl_pairing(sm120, recorder):
    mixed, normed, launched, (hyper_input, norm_w, down, up) = _launch(sm120, recorder)
    rows = 4
    assert [name for name, _, _, _ in recorder.launches] == [
        "_hc_branch_stats_kernel",
        "_hc_down_kernel",
        "_hc_up_kernel",
    ]
    k0, k1, k2 = (launched[n][1] for n in launched)
    # K0 owns the normalization AND the clear K1's atomics accumulate into.
    assert k0[0] is hyper_input and k0[1] is norm_w
    assert k0[3] is k1[2] is k2[2]
    assert k0[4] == rows * HC and k0[5] == rows * LOWRANK
    assert k0[6:9] == (K, HS, EPS)
    assert launched["_hc_branch_stats_kernel"][0] == (rows * HC,)
    # K1: split-K over the branches x lowrank tiles, scaled by its own rows.
    assert k1[0] is down and k1[1] is k0[2]
    assert k1[3] is getattr(down, _SCALE_ATTR)
    assert k1[4:7] == (rows, K, LOWRANK)
    assert launched["_hc_down_kernel"][0] == (K // (128 * 4), LOWRANK // 64)
    # K2: silu -> up projection -> gate and branch mean, reading K0's normed.
    assert k2[0] is k0[2] and k2[1] is up and k2[3] is mixed
    assert k2[4] is getattr(up, _SCALE_ATTR)
    assert k2[5:10] == (rows, K, HS, LOWRANK, 1.0 / HC)
    assert launched["_hc_up_kernel"][0] == (HS // 16,)
    for _, _, _, kwargs in recorder.launches:
        # Paired: the body's gdc_wait/gdc_launch_dependents exist exactly when
        # the launch carries the attribute, and never otherwise.
        assert kwargs["USE_PDL"] is kwargs["launch_pdl"] is mix2.PDL
    assert tuple(mixed.shape) == (rows, HS) and mixed.dtype == torch.bfloat16
    assert tuple(normed.shape) == (rows, K) and normed.dtype == torch.bfloat16
    t_raw = k0[3]
    assert tuple(t_raw.shape) == (mix2._HC_MIX2_MAX_ROWS, LOWRANK)
    assert t_raw.dtype == torch.float32


def test_workspace_is_per_call_not_a_shared_scratch(sm120, recorder):
    mixed_a, normed_a, launched_a, _ = _launch(sm120, recorder)
    mixed_b, normed_b, launched_b, _ = _launch(sm120, recorder)
    t_raw_a = launched_a["_hc_branch_stats_kernel"][1][3]
    t_raw_b = launched_b["_hc_branch_stats_kernel"][1][3]
    # Two in-flight calls -- or two calls captured into one decode graph -- must
    # never share the split-K accumulator or write each other's outputs.
    for first, second in ((t_raw_a, t_raw_b), (mixed_a, mixed_b), (normed_a, normed_b)):
        assert first is not second
        assert first.data_ptr() != second.data_ptr()


def test_scales_follow_a_replaced_weight(sm120, recorder):
    hyper_input, norm_w, down_a, up_a = _resident_pair(seed=1)
    _, _, down_b, up_b = _resident_pair(seed=2)
    assert getattr(down_b, _SCALE_ATTR) is not getattr(down_a, _SCALE_ATTR)
    mix2.hc_norm_mix2(hyper_input[:4], norm_w, EPS, down_a, up_a, HC, HS)
    recorder.launches.clear()
    mix2.hc_norm_mix2(hyper_input[:4], norm_w, EPS, down_b, up_b, HC, HS)
    launched = recorder.by_name()
    assert launched["_hc_down_kernel"][1][3] is getattr(down_b, _SCALE_ATTR)
    assert launched["_hc_up_kernel"][1][4] is getattr(up_b, _SCALE_ATTR)
    # The gate validates the very scale tensors this base's ingestion produces
    # (one FP32 scale per FP8 output row), and it reads them off the live
    # weights rather than remembering the first pair it saw.
    gate_down = _attrs_tensor(
        (LOWRANK, K), torch.float8_e4m3fn, scale=getattr(down_b, _SCALE_ATTR)
    )
    gate_up = _attrs_tensor(
        (K, LOWRANK), torch.float8_e4m3fn, scale=getattr(up_b, _SCALE_ATTR)
    )
    assert mix2.hc_norm_mix2_supported(
        _attrs_tensor((4, K)), _attrs_tensor([K]), gate_down, gate_up, HC, HS
    )


def test_scale_override_is_passed_through(sm120, recorder):
    hyper_input, norm_w, down, up = _resident_pair(seed=3)
    s_down = getattr(down, _SCALE_ATTR).clone()
    s_up = getattr(up, _SCALE_ATTR).clone()
    mix2.hc_norm_mix2(hyper_input[:4], norm_w, EPS, down, up, HC, HS, s_down, s_up)
    launched = recorder.by_name()
    assert launched["_hc_down_kernel"][1][3] is s_down
    assert launched["_hc_up_kernel"][1][4] is s_up


class _CountingNorm:
    """The real GroupedGemmaRMSNorm, with a call counter and attribute access."""

    def __init__(self, module):
        self.module = module
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.module(x)

    @property
    def weight(self):
        return self.module.weight

    @property
    def variance_epsilon(self):
        return self.module.variance_epsilon


def _bare_layer():
    """A GatedResidual carrying only what mix() reads (no CUDA construction)."""
    layer = hyp.GatedResidual.__new__(hyp.GatedResidual)
    layer.config = hyp.HyperConnectionConfig(
        hc_count=HC,
        hidden_size=HS,
        params_dtype=torch.bfloat16,
        hc_lowrank=LOWRANK,
        rms_norm_eps=EPS,
        hc_per_branch_norm=True,
    )
    layer.hc_count, layer.hidden_size, layer.params_dtype = HC, HS, torch.bfloat16
    norm = hyp.GroupedGemmaRMSNorm(K, eps=EPS, group_size=HS)
    norm.weight = nn.Parameter(torch.zeros(K, dtype=torch.bfloat16))
    layer.hc_norm = _CountingNorm(norm)
    layer._jit_mix_ok = False
    layer._mix_up_weight_padded = None
    layer.input_mix_weight_down = SimpleNamespace(weight=None)
    layer.input_mix_weight_up = SimpleNamespace(weight=None)
    layer._mix_compute = lambda x, wd, wu, hc, hs: "mixed"
    return layer


def test_mix_routes_to_mix2_before_normalizing(monkeypatch, sm120):
    normed_sentinel = torch.zeros(4, K, dtype=torch.bfloat16)
    mixed_sentinel = torch.zeros(4, HS, dtype=torch.bfloat16)
    monkeypatch.setattr(hyp, "hc_norm_mix2_supported", lambda *args: True)
    monkeypatch.setattr(
        hyp, "hc_norm_mix2", lambda *args: (mixed_sentinel, normed_sentinel)
    )
    layer = _bare_layer()
    hyper_input = torch.zeros(4, K, dtype=torch.bfloat16)
    mixed, residuals = layer.mix(hyper_input)
    assert mixed is mixed_sentinel
    # Unchanged combine contract: (raw residual, normed) -- and the separate
    # normalization did not also run, because K0 is that normalization.
    assert residuals[0] is hyper_input
    assert residuals[1] is normed_sentinel
    assert layer.hc_norm.calls == 0


def test_mix_passes_the_live_weights_to_mix2(monkeypatch, sm120):
    seen = []
    monkeypatch.setattr(hyp, "hc_norm_mix2_supported", lambda *args: True)
    monkeypatch.setattr(
        hyp,
        "hc_norm_mix2",
        lambda hyper_input, norm_w, eps, w_down, w_up, hc, hs: seen.append(
            (w_down, w_up, norm_w, eps, hc, hs)
        )
        or (torch.zeros(4, HS, dtype=torch.bfloat16), hyper_input),
    )
    layer = _bare_layer()
    down, up = _fp8_pair()
    layer.input_mix_weight_down.weight = down
    layer.input_mix_weight_up.weight = up
    hyper_input = torch.zeros(4, K, dtype=torch.bfloat16)
    layer.mix(hyper_input)
    # A weight update replaces Parameter and scale together (see
    # _ingest_rowwise_weight); mix() must not hand the kernels a stale reference.
    new_down, new_up = _fp8_pair()
    layer.input_mix_weight_down.weight = new_down
    layer.input_mix_weight_up.weight = new_up
    layer.mix(hyper_input)
    assert [pair[0] for pair in seen] == [down, new_down]
    assert [pair[1] for pair in seen] == [up, new_up]
    assert seen[0][2] is layer.hc_norm.weight
    assert seen[0][3] == EPS and seen[0][4:] == (HC, HS)


def test_mix_falls_back_to_the_existing_path(monkeypatch, sm120):
    monkeypatch.setattr(hyp, "hc_norm_mix2_supported", lambda *args: False)
    calls = []
    layer = _bare_layer()
    layer._mix_compute = lambda x, wd, wu, hc, hs: calls.append(
        (x, wd, wu)
    ) or torch.zeros(24, HS, dtype=torch.bfloat16)
    down, up = (
        _attrs_tensor((LOWRANK, K), torch.bfloat16),
        _attrs_tensor((K, LOWRANK), torch.bfloat16),
    )
    layer.input_mix_weight_down.weight = down
    layer.input_mix_weight_up.weight = up
    hyper_input = torch.ones(24, K, dtype=torch.bfloat16)
    mixed, residuals = layer.mix(hyper_input)
    assert tuple(mixed.shape) == (24, HS)
    assert residuals[0] is hyper_input
    assert tuple(residuals[1].shape) == (24, K)
    # The untouched path: separate per-branch normalization, then the compiled
    # mix on the resident weights.
    assert layer.hc_norm.calls == 1
    assert len(calls) == 1 and calls[0][1] is down and calls[0][2] is up


def test_registered_files_have_an_executable_entry_point():
    """python/sglang/test/ci/ci_utils.py runs `python3 <file> -f`, not pytest.

    A registered file with no `__main__` block is imported, runs zero tests and
    exits 0, which is how a suite like this one can look green while collecting
    nothing. Both files must therefore carry the entry point, and must survive
    the legacy failfast argument.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    for relpath in (
        "test/registered/unit/layers/test_hc_mix2_dispatch.py",
        "test/registered/kernels/test_hc_mix2.py",
    ):
        with open(os.path.join(root, relpath), encoding="utf-8") as handle:
            source = handle.read()
        assert 'if __name__ == "__main__":' in source, relpath
        assert "sys.exit(pytest.main(" in source, relpath
        assert '"-f"' in source, relpath


if __name__ == "__main__":
    # CI's run_unittest_files launches `python3 <file> -f` (legacy unittest
    # failfast), which pytest would reject; translate it to -x, the same way
    # sglang.test.kernels.utils.multigpu_pytest_main does, so the registered
    # invocation actually collects and runs this file instead of importing it
    # and exiting 0 with zero tests run.
    _args = ["-x" if _arg == "-f" else _arg for _arg in sys.argv[1:]]
    sys.exit(pytest.main([__file__, "-v", *_args]))
