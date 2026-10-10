import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.model_executor.model_runner_components import (
    attention_backend_setup,
)
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    ResolvedAttentionBackendStr,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeBackend:
    def __init__(self, name):
        self.name = name
        # Real backends always carry this (AttentionBackend class attribute).
        self.needs_cpu_seq_lens = True
        # HybridLinearAttnBackend.__init__ reads these off the full side.
        self.token_to_kv_pool = object()
        self.req_to_token_pool = object()


def test_split_full_attention_applies_model_wrapper_once():
    # The hybrid backend takes the speculative attention mode from the
    # published configuration.
    from sglang.srt.runtime_context import get_context

    override = get_context().override_server_args(speculative_attention_mode="prefill")
    override.install()
    try:
        runner = SimpleNamespace(
            server_args=SimpleNamespace(speculative_attention_mode="prefill"),
            model_config=SimpleNamespace(context_len=2048),
            kv_cache_dtype=None,
            token_to_kv_pool=object(),
            req_to_token_pool=object(),
            init_new_workspace=None,
        )
        wrapper_inputs = []
        wrapped_backend = object()

        def wrap_once(model_runner, backend):
            assert model_runner is runner
            wrapper_inputs.append(backend)
            return wrapped_backend

        constructors = {
            "decode-test": lambda model_runner: _FakeBackend("decode"),
            "prefill-test": lambda model_runner: _FakeBackend("prefill"),
        }
        resolved = ResolvedAttentionBackendStr(
            decode="decode-test", prefill="prefill-test"
        )

        with (
            patch.dict(attention_backend_setup.ATTENTION_BACKENDS, constructors),
            patch.object(
                attention_backend_setup,
                "attn_backend_wrapper",
                side_effect=wrap_once,
            ),
        ):
            result = attention_backend_setup._build_resolved_backend(
                model_runner=runner,
                resolved=resolved,
                init_new_workspace=True,
            )

        assert result is wrapped_backend
        assert len(wrapper_inputs) == 1
        split_backend = wrapper_inputs[0]
        assert isinstance(split_backend, HybridAttnBackend)
        assert split_backend.decode_backend.name == "decode"
        assert split_backend.prefill_backend.name == "prefill"
        assert runner.init_new_workspace is True
    finally:
        override.restore()


def test_equal_resolved_backends_ignore_stale_global_backend():
    runner = SimpleNamespace(
        server_args=SimpleNamespace(
            attention_backend="global-test",
            speculative_attention_mode="prefill",
        ),
        kv_cache_dtype=None,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        init_new_workspace=None,
    )
    constructors = {
        "global-test": lambda _runner: _FakeBackend("global"),
        "resolved-test": lambda _runner: _FakeBackend("resolved"),
    }
    resolved = ResolvedAttentionBackendStr(
        decode="resolved-test",
        prefill="resolved-test",
    )

    with (
        patch.dict(attention_backend_setup.ATTENTION_BACKENDS, constructors),
        patch.object(
            attention_backend_setup,
            "attn_backend_wrapper",
            side_effect=lambda _runner, backend: backend,
        ),
    ):
        result = attention_backend_setup._build_resolved_backend(
            model_runner=runner,
            resolved=resolved,
            init_new_workspace=False,
        )

    assert result.name == "resolved"


# ---------------------------------------------------------------------------
# Narrow FlashInfer->QSA skip: the discarded generic FlashInfer construction
# (and its 384 MiB process-pinned workspace) must not happen when the
# existing hybrid GDN/QSA selection will replace the full-attention side.
# Everything else on this dispatch keeps the original eager behavior.
# ---------------------------------------------------------------------------


def _hybrid_gdn_qsa_runner(name="flashinfer"):
    """Real Qwen4-Exp text config + recording flashinfer-class entries.

    Only the kernel-side backends the wrapper builds are faked (a CPU box
    has no pools or streams for them); setup dispatch, the wrapper branch
    and the shared replacement selector all run for real.
    """
    from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig

    class _GDN:
        _recover_ssm = False
        needs_cpu_seq_lens = False

        def __init__(self, runner):
            self.runner = runner

    class _QSA:
        needs_cpu_seq_lens = True

        def __init__(self, runner):
            self.runner = runner
            self.token_to_kv_pool = object()
            self.req_to_token_pool = object()

    cfg = Qwen4ExpTextConfig(
        hidden_size=2560,
        num_hidden_layers=4,
        layer_types=[
            "full_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ],
        indexer_n_heads=16,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    model_config = SimpleNamespace(
        hf_config=cfg,
        hf_text_config=cfg,
        is_draft_model=False,
        is_encoder_decoder=False,
        linear_attn_registry_result=None,
    )
    runner = SimpleNamespace(
        model_config=model_config,
        use_mla_backend=False,
        is_draft_worker=False,
        prefill_attention_backend_str=name,
        decode_attention_backend_str=name,
        init_new_workspace=None,
        linear_attn_backends=None,
    )
    return cfg, runner, _GDN, _QSA


def _build_with_real_wrapper(runner, name, entries, gdn_cls, qsa_cls, **args_overrides):
    from sglang.srt import utils as srt_utils
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_module
    from sglang.srt.layers.attention.linear import gdn_backend
    from sglang.srt.runtime_context import get_context

    override = get_context().override_server_args(**args_overrides)
    override.install()
    try:
        with (
            patch.dict(attention_backend_setup.ATTENTION_BACKENDS, entries),
            patch.object(gdn_backend, "GDNAttnBackend", gdn_cls),
            patch.object(
                gdn_backend, "flashinfer_gdn_prefill_default", lambda _r: None
            ),
            patch.object(srt_utils, "is_blackwell", lambda: False),
            patch.object(qsa_module, "QwenSparseAttnBackend", qsa_cls),
        ):
            return attention_backend_setup._build_resolved_backend(
                model_runner=runner,
                resolved=ResolvedAttentionBackendStr(
                    prefill=name, decode=name, is_draft_override=False
                ),
                init_new_workspace=True,
            )
    finally:
        override.restore()


def test_qsa_flashinfer_selection_does_not_construct_discarded_backend():
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )

    cfg, runner, gdn_cls, qsa_cls = _hybrid_gdn_qsa_runner()
    constructed = []
    generic = _FakeBackend("flashinfer")

    def ctor(model_runner):
        assert model_runner is runner
        constructed.append(model_runner)
        return generic

    result = _build_with_real_wrapper(
        runner, "flashinfer", {"flashinfer": ctor}, gdn_cls, qsa_cls
    )

    # The proven replace path never pays for the generic backend or the
    # workspace its constructor pins (red at the base: built, then dropped).
    assert constructed == []
    assert isinstance(result, HybridLinearAttnBackend)
    assert isinstance(result.full_attn_backend, qsa_cls)
    assert isinstance(result.linear_attn_backend, gdn_cls)
    assert result.full_attn_layers == cfg.full_attention_layer_ids == [0, 2]
    assert runner.init_new_workspace is True  # state preserved on the skip


def test_non_qsa_hybrid_flashinfer_selection_still_constructs_eagerly():
    cfg, runner, gdn_cls, qsa_cls = _hybrid_gdn_qsa_runner()
    for field in (
        "indexer_n_heads",
        "indexer_kv_heads",
        "indexer_head_dim",
        "indexer_budget",
        "indexer_compress_ratio",
    ):
        delattr(cfg, field)
    constructed = []
    generic = _FakeBackend("flashinfer")

    result = _build_with_real_wrapper(
        runner,
        "flashinfer",
        {"flashinfer": lambda mr: constructed.append(mr) or generic},
        gdn_cls,
        qsa_cls,
    )

    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )

    assert constructed == [runner]  # outside the QSA domain: untouched path
    assert isinstance(result, HybridLinearAttnBackend)
    assert result.full_attn_backend is generic


@pytest.mark.parametrize(
    ("name", "match", "overrides"),
    [
        ("trtllm_mla", "can only be used with MLA models", {}),
        (
            "hpc_ops",
            "does not support speculative decoding",
            {"speculative_algorithm": "nextn"},
        ),
    ],
)
def test_qsa_still_raises_registered_incompatible_backends(name, match, overrides):
    # Not flashinfer-named: the eager path is taken even when QSA would
    # replace the backend, so the real registered preamble still rejects
    # the selection (no silent QSA fallback for explicit choices).
    cfg, runner, gdn_cls, qsa_cls = _hybrid_gdn_qsa_runner(name=name)
    with pytest.raises(ValueError, match=match):
        _build_with_real_wrapper(runner, name, {}, gdn_cls, qsa_cls, **overrides)


def test_qsa_still_warns_on_nsa_alias_before_dispatch():
    import warnings

    cfg, runner, gdn_cls, qsa_cls = _hybrid_gdn_qsa_runner(name="nsa")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            _build_with_real_wrapper(runner, "nsa", {}, gdn_cls, qsa_cls)
        except Exception:
            pass  # eager path continues into the real DSA constructor
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "'nsa' is deprecated" in str(w.message)
        for w in caught
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
