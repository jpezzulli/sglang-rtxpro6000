"""CPU-only dispatch checks; numerical/graph coverage lives in offload tests."""

import ast
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops import qwen4_ple
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _tensor(shape, dtype=torch.long, device="cuda", pinned=False):
    return SimpleNamespace(
        shape=shape,
        ndim=len(shape),
        dim=lambda: len(shape),
        numel=lambda: math.prod(shape),
        dtype=dtype,
        device=torch.device(device),
        is_cuda=device == "cuda",
        is_contiguous=lambda: True,
        is_pinned=lambda: pinned,
        data_ptr=lambda: 1234,
    )


def _inputs(tokens=4, width=160, dtype=torch.float8_e4m3fn):
    return (
        _tensor((tokens, 3)),
        _tensor((3,)),
        _tensor((16,)),
        _tensor((16,)),
        _tensor((850, width), dtype, "cpu", pinned=True),
    )


@pytest.mark.parametrize(
    "tokens,width,dtype,tp_size,capability,expected",
    [
        (4, 160, torch.float8_e4m3fn, 1, (12, 0), True),
        (16, 160, torch.float8_e4m3fn, 1, (12, 0), True),
        *[
            (n, 160, torch.float8_e4m3fn, 1, (12, 0), False)
            for n in (0, 1, 8, 17, 4096)
        ],
        (4, 128, torch.float8_e4m3fn, 1, (12, 0), False),
        (4, 160, torch.bfloat16, 1, (12, 0), False),
        (4, 160, torch.float8_e4m3fn, 2, (12, 0), False),
        (4, 160, torch.float8_e4m3fn, 1, (9, 0), False),
    ],
)
def test_fused_gather_shape_guard(
    monkeypatch, tokens, width, dtype, tp_size, capability, expected
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    assert (
        qwen4_ple.can_fuse_qwen4_ngram_gather(*_inputs(tokens, width, dtype), tp_size)
        is expected
    )


@pytest.mark.parametrize("invalid", ["context", "pinning", "layout", "device"])
def test_fused_gather_rejects_unsupported_storage(monkeypatch, invalid):
    inputs = _inputs()
    if invalid == "context":
        inputs[0].shape = (4, 4)
    elif invalid == "pinning":
        inputs[-1].is_pinned = lambda: False
    elif invalid == "layout":
        inputs[-1].is_contiguous = lambda: False
    else:
        inputs[-1].device = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    assert not qwen4_ple.can_fuse_qwen4_ngram_gather(*inputs, 1)


@pytest.mark.parametrize("num_warps", [None, 1])
def test_fused_gather_launch_warps(monkeypatch, num_warps):
    contexts, multipliers, sizes, offsets, weight = _inputs()
    out = _tensor((4, 16, 160), torch.bfloat16)
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((grid, kwargs))

    monkeypatch.setattr(qwen4_ple, "_qwen4_ngram_gather_kernel", Kernel())
    kwargs = {} if num_warps is None else {"num_warps": num_warps}
    assert (
        qwen4_ple.fused_qwen4_ngram_gather(
            contexts, multipliers, sizes, offsets, 0, weight, 0, 850, out, **kwargs
        )
        is out
    )
    assert launches[0][0] == (64,)
    assert launches[0][1].get("num_warps", 4) == (4 if num_warps is None else 1)


def _start_prefetch():
    # Importing the complete model loads CUDA extensions even in CPU CI. Execute
    # its actual method with stream/tensor doubles instead of importing them.
    path = Path(qwen4_ple.__file__).parents[2] / "srt/models/qwen4_exp.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpPLELayer"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "start_prefetch"
    )
    namespace = {"torch": torch}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["start_prefetch"]


@pytest.mark.parametrize("mode", ["verify", "decode", "extend"])
@pytest.mark.parametrize("eligible", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_prefetch_route_and_fallback_order(monkeypatch, mode, eligible, enabled):
    events = []
    contexts, multipliers, sizes, offsets, weight = _inputs()
    ids = _tensor((4, 16))
    out = _tensor((4, 16, 160), torch.bfloat16)
    out.view = lambda *shape: out
    current = "main"
    stream = SimpleNamespace(wait_stream=lambda source: events.append(("wait", source)))
    contexts.record_stream = ids.record_stream = lambda target: events.append(
        ("record", target)
    )

    @contextmanager
    def use_stream(target):
        nonlocal current
        current = target
        try:
            yield
        finally:
            current = "main"

    def hash_contexts(*args, **kwargs):
        events.append(("hash", current))
        return ids

    def fused(*args, **kwargs):
        events.append(("fused", current))
        assert kwargs["num_warps"] == 1

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current)
    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    monkeypatch.setattr(qwen4_ple, "can_fuse_qwen4_ngram_hash", lambda *args: eligible)
    monkeypatch.setattr(
        qwen4_ple, "can_fuse_qwen4_ngram_gather", lambda *args: eligible, raising=False
    )
    monkeypatch.setattr(qwen4_ple, "fused_qwen4_ngram_gather", fused)
    offloaded = SimpleNamespace(
        weight=weight,
        tp_size=1,
        shard_indices=SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=850),
        gather=lambda *args, **kwargs: events.append(("gather", current)),
    )
    embedding = SimpleNamespace(
        gather_dp_tokens=False,
        enable_ple_fusion=enabled,
        prepare_ngram_contexts=lambda batch: contexts,
        _hash_contexts=hash_contexts,
        ngram_embedding=offloaded,
        layer_multipliers=multipliers,
        ngram_heads_vocab_sizes=sizes,
        ngram_heads_offsets=offsets,
        eos_token_id=0,
        ngram_heads=16,
    )
    layer = SimpleNamespace(
        ple_embedding=embedding,
        _prefetch_stream=stream,
        _prefetch_state=None,
        _get_prefetch_buffer=lambda *args: out,
    )
    batch = SimpleNamespace(
        physical_tokens=4,
        mode=SimpleNamespace(
            is_decode=lambda: mode == "decode",
            is_target_verify=lambda: mode == "verify",
        ),
    )
    _start_prefetch()(layer, batch, SimpleNamespace())
    selected = mode == "verify" and eligible and enabled
    assert [event[0] for event in events] == (
        ["wait", "record", "fused"]
        if selected
        else ["hash", "wait", "record", "gather"]
    )
    if not selected:
        assert events[0] == ("hash", "main")
    assert events[-1][1] is stream
    assert layer._prefetch_state == (out, 4, 4)
