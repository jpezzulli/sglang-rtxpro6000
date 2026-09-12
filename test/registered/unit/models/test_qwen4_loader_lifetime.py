"""Exercise the real loader body with CPU-only lifetime/PLE fixtures.

AST extraction avoids initializing the GPU model; it does not rewrite the
method under test. This establishes reference lifetime, not VRAM savings.
"""

import ast
import gc
import logging
import weakref
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import torch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Parameter:
    pass


class OtherModule:
    pass


class PLE:
    def __init__(self):
        self.ngram_embedding = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.zeros((4, 2), dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
            org_vocab_size=4,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=0, org_vocab_end_index=4
            ),
        )


class Model:
    config = SimpleNamespace(num_experts=None, split_ngram_parts=2)
    language_model_only = False

    def __init__(self, ple=None):
        self.weight = Parameter()
        self.ple = ple

    def named_parameters(self, **kwargs):
        return [("lm_head.weight", self.weight)]

    def named_buffers(self):
        return []

    def named_modules(self):
        return [("model.ple", self.ple)] if self.ple is not None else []

    def modules(self):
        return []

    def _load_qwen4_exp_ple_buffer(self, *args):
        return False

    def post_load_weights(self):
        # Precision conversion is tested independently; this fixture measures
        # whether the real loader releases its captured parameter snapshot.
        pass


def real_loader():
    source = (
        Path(__file__).resolve().parents[4] / "python/sglang/srt/models/qwen4_exp.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Qwen4ExpForConditionalGeneration"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
    )
    namespace = {
        "torch": torch,
        "Iterable": Iterable,
        "Tuple": tuple,
        "Set": set,
        "Qwen4ExpNGramEmbedding": PLE,
        "Qwen4ExpPinnedHostEmbedding": OtherModule,
        "Qwen3_5GatedDeltaNet": OtherModule,
        "logger": logging.getLogger("qwen4-loader-lifetime-test"),
    }
    exec(  # noqa: S102 -- execute only the repository's method under test
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["load_weights"]


def test_replaced_parameter_released_without_cyclic_gc():
    load = real_loader()
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        model = Model()
        original = weakref.ref(model.weight)

        def weights():
            # The real loader already took its named-parameter snapshot when
            # iteration starts. Mimic a later parameter replacement during load.
            model.weight = Parameter()
            yield from ()

        assert load(model, weights()) == set()
        assert original() is None, "loader retains replaced parameter until cyclic GC"
    finally:
        gc.collect()
        if enabled:
            gc.enable()


def test_ple_downcast_warns_once_per_load_and_copies_both_shards(caplog):
    load = real_loader()
    model = Model(PLE())
    weights = [
        (
            f"model.ple.ngram_embedding.shard_{i}.weight",
            torch.full((2, 2), i + 1, dtype=torch.bfloat16),
        )
        for i in range(2)
    ]
    with caplog.at_level(logging.WARNING, logger="qwen4-loader-lifetime-test"):
        for _ in range(2):
            assert load(model, weights) == {"model.ple.ngram_embedding.weight"}
    warnings = [r for r in caplog.records if "downcasting is lossy" in r.message]
    assert len(warnings) == 2
    torch.testing.assert_close(
        model.ple.ngram_embedding.weight.float(),
        torch.tensor([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]),
    )
