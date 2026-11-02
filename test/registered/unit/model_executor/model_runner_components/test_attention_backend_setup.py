import contextlib
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
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _FakeBackend:
    def __init__(self, name):
        self.name = name
        # Real backends always carry this (AttentionBackend class attribute).
        self.needs_cpu_seq_lens = True
        # HybridLinearAttnBackend.__init__ reads these off the full side.
        self.token_to_kv_pool = object()
        self.req_to_token_pool = object()
        self.runner_ids = None
        self.use_captured_forward_metadata_for_breakable_cuda_graph = False


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
            runner_ids=None,
            init_new_workspace=None,
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
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
        is_draft_worker=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
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


def _recording_flashinfer_class(built, supports_dcp=False):
    """Stand-in for the class the actual registered ``flashinfer`` factory builds.

    The factory itself is left in place, so its preamble (MHA vs MLA class
    selection, workspace flag forwarding) runs for real and only the kernel
    class -- which a CPU box cannot build -- is recorded.
    """

    class _Recorded:
        # The real FlashInfer MHA/MLA backends inherit AttentionBackend's False
        # default for the guard's flag, so the fake states it directly.
        supports_dcp_target_verify = supports_dcp
        needs_cpu_seq_lens = True
        use_captured_forward_metadata_for_breakable_cuda_graph = False

        def __init__(self, runner, init_new_workspace=False, **kwargs):
            self.name = "flashinfer"
            self.runner = runner
            self.init_new_workspace = init_new_workspace
            self.token_to_kv_pool = object()
            self.req_to_token_pool = object()
            self.runner_ids = None
            built.append(self)

    return _Recorded


def _hybrid_gdn_qsa_runner(name="flashinfer", spec=SpeculativeAlgorithm.NONE):
    """Real Qwen4-Exp text config + a runner the dispatch and wrapper accept.

    Only the kernel-side backends the wrapper builds are faked (a CPU box
    has no pools or streams for them); setup dispatch, the wrapper branch
    and the shared replacement selector all run for real.
    """
    from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig

    class _GDN:
        _recover_ssm = False
        needs_cpu_seq_lens = False
        use_captured_forward_metadata_for_breakable_cuda_graph = False

        def __init__(self, runner):
            self.runner = runner

    class _QSA:
        needs_cpu_seq_lens = True
        use_captured_forward_metadata_for_breakable_cuda_graph = False

        def __init__(self, runner):
            self.runner = runner
            self.token_to_kv_pool = object()
            self.req_to_token_pool = object()
            self.runner_ids = None

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
        context_len=2048,
        is_draft_model=False,
        is_encoder_decoder=False,
        linear_attn_registry_result=None,
        quantization=None,
    )
    runner = SimpleNamespace(
        model_config=model_config,
        server_args=SimpleNamespace(speculative_attention_mode="prefill"),
        use_mla_backend=False,
        is_draft_worker=False,
        draft_attention_backend=None,
        prefill_attention_backend_str=name,
        decode_attention_backend_str=name,
        init_new_workspace=None,
        linear_attn_backends=None,
        spec_algorithm=spec,
        kv_cache_dtype=None,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        runner_ids=None,
    )
    return cfg, runner, _GDN, _QSA


def _build_with_real_wrapper(
    runner,
    name,
    gdn_cls,
    qsa_cls,
    entries=None,
    resolved=None,
    flashinfer_cls=None,
    init_new_workspace=True,
    **args_overrides,
):
    """Run the real dispatch + real wrapper with only kernel classes faked."""
    import sglang.srt.layers.attention.flashinfer_backend as flashinfer_module
    from sglang.srt import utils as srt_utils
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_module
    from sglang.srt.layers.attention.linear import gdn_backend
    from sglang.srt.runtime_context import get_context

    if resolved is None:
        resolved = ResolvedAttentionBackendStr(
            prefill=name, decode=name, is_draft_override=False
        )
    override = get_context().override_server_args(**args_overrides)
    override.install()
    patches = [
        patch.dict(attention_backend_setup.ATTENTION_BACKENDS, entries or {}),
        patch.object(gdn_backend, "GDNAttnBackend", gdn_cls),
        patch.object(gdn_backend, "flashinfer_gdn_prefill_default", lambda _r: None),
        patch.object(srt_utils, "is_blackwell", lambda: False),
        patch.object(qsa_module, "QwenSparseAttnBackend", qsa_cls),
    ]
    if flashinfer_cls is not None:
        patches.append(
            patch.object(flashinfer_module, "FlashInferAttnBackend", flashinfer_cls)
        )
    try:
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            return attention_backend_setup._build_resolved_backend(
                model_runner=runner,
                resolved=resolved,
                init_new_workspace=init_new_workspace,
            )
    finally:
        override.restore()


def _hybrid_linear_cls():
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )

    return HybridLinearAttnBackend


def test_qsa_flashinfer_selection_does_not_construct_discarded_backend():
    cfg, runner, gdn_cls, qsa_cls = _hybrid_gdn_qsa_runner()
    built = []

    result = _build_with_real_wrapper(
        runner,
        "flashinfer",
        gdn_cls,
        qsa_cls,
        flashinfer_cls=_recording_flashinfer_class(built),
    )

    # The proven replace path never pays for the generic backend or the
    # workspace its constructor pins (red at the base: built, then dropped).
    assert built == []
    assert isinstance(result, _hybrid_linear_cls())
    assert isinstance(result.full_attn_backend, qsa_cls)
    assert isinstance(result.linear_attn_backend, gdn_cls)
    assert result.full_attn_layers == cfg.full_attention_layer_ids == [0, 2]
    assert runner.init_new_workspace is True  # state preserved on the skip


def test_hybrid_gdn_mla_refusal_does_not_construct_flashinfer():
    # The wrapper rejects hybrid GDN + MLA before anything is kept; on the
    # base the rejection landed after the discarded construction.
    cfg, runner, gdn_cls, qsa_cls = _hybrid_gdn_qsa_runner()
    runner.use_mla_backend = True
    built = []

    with pytest.raises(AssertionError, match="hybrid_gdn can only be used"):
        _build_with_real_wrapper(
            runner,
            "flashinfer",
            gdn_cls,
            qsa_cls,
            flashinfer_cls=_recording_flashinfer_class(built),
        )

    assert built == []


def test_non_qsa_hybrid_flashinfer_selection_still_constructs_eagerly():
    # The maintained non-QSA hybrid path (same GDN wrapper, no indexer
    # fields) keeps its full-attention backend, so it still builds it.
    cfg, runner, gdn_cls, qsa_cls = _hybrid_gdn_qsa_runner()
    for field in (
        "indexer_n_heads",
        "indexer_kv_heads",
        "indexer_head_dim",
        "indexer_budget",
        "indexer_compress_ratio",
    ):
        delattr(cfg, field)
    built = []

    result = _build_with_real_wrapper(
        runner,
        "flashinfer",
        gdn_cls,
        qsa_cls,
        flashinfer_cls=_recording_flashinfer_class(built),
    )

    assert len(built) == 1 and built[0].runner is runner
    assert isinstance(result, _hybrid_linear_cls())
    assert result.full_attn_backend is built[0]
    assert built[0].init_new_workspace is True  # workspace flag still forwarded


def test_plain_model_flashinfer_selection_still_constructs_eagerly():
    # The generic plain-model control (no hybrid/linear side at all, the
    # maintained 27B/default shape of this dispatch): untouched eager path.
    from transformers import LlamaConfig

    cfg, runner, gdn_cls, qsa_cls = _hybrid_gdn_qsa_runner()
    plain = LlamaConfig(hidden_size=32, num_hidden_layers=2)
    runner.model_config.hf_config = plain
    runner.model_config.hf_text_config = plain
    built = []

    result = _build_with_real_wrapper(
        runner,
        "flashinfer",
        gdn_cls,
        qsa_cls,
        flashinfer_cls=_recording_flashinfer_class(built),
    )

    assert len(built) == 1 and built[0].runner is runner
    assert result is built[0]  # no wrapper for a plain model


@pytest.mark.parametrize(
    ("name", "match", "overrides"),
    [
        ("trtllm_mla", "can only be used with MLA models", {}),
        ("bogus", "Invalid attention backend: bogus", {}),
        (
            "hpc_ops",
            "does not support speculative decoding",
            {"speculative_algorithm": "nextn"},
        ),
    ],
)
def test_qsa_still_raises_registered_incompatible_backends(name, match, overrides):
    # Not deferred: the eager path is taken even when QSA would replace the
    # backend, so the real registered preamble still rejects the selection
    # (no silent QSA fallback for explicit choices).
    cfg, runner, gdn_cls, qsa_cls = _hybrid_gdn_qsa_runner(name=name)
    with pytest.raises(ValueError, match=match):
        _build_with_real_wrapper(runner, name, gdn_cls, qsa_cls, **overrides)


def test_qsa_still_warns_on_nsa_alias_before_dispatch():
    import warnings

    cfg, runner, gdn_cls, qsa_cls = _hybrid_gdn_qsa_runner(name="nsa")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            _build_with_real_wrapper(runner, "nsa", gdn_cls, qsa_cls)
        except Exception:
            pass  # eager path continues into the real DSA constructor
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "'nsa' is deprecated" in str(w.message)
        for w in caught
    )


def test_qsa_split_children_stay_eager():
    # Split prefill/decode keeps eager children: the skip does not leak into
    # that composition, which the wrapper still applies once (as on the base,
    # where the composed hybrid is then replaced by QSA).
    cfg, runner, gdn_cls, qsa_cls = _hybrid_gdn_qsa_runner()
    built = []
    other_children = []

    def other_child(mr):
        other_children.append(_FakeBackend("decode-test"))
        return other_children[-1]

    result = _build_with_real_wrapper(
        runner,
        "flashinfer",
        gdn_cls,
        qsa_cls,
        entries={"decode-test": other_child},
        resolved=ResolvedAttentionBackendStr(
            prefill="flashinfer", decode="decode-test"
        ),
        flashinfer_cls=_recording_flashinfer_class(built),
    )

    assert len(built) == 1 and len(other_children) == 1
    assert isinstance(result, _hybrid_linear_cls())
    assert isinstance(result.full_attn_backend, qsa_cls)
    assert result.full_attn_layers == cfg.full_attention_layer_ids


def test_qsa_draft_override_keeps_its_own_backend_and_skips_the_discard(caplog):
    import logging

    cfg, runner, gdn_cls, qsa_cls = _hybrid_gdn_qsa_runner(name="triton")
    runner.is_draft_worker = True
    runner.draft_attention_backend = "flashinfer"
    built = []
    with caplog.at_level(logging.WARNING):
        resolved = attention_backend_setup.resolve_attention_backend_strs(
            model_runner=runner
        )
        assert resolved == ResolvedAttentionBackendStr(
            prefill="flashinfer", decode="flashinfer", is_draft_override=True
        )
        assert "Overriding draft attention backend to flashinfer" in caplog.text
        result = _build_with_real_wrapper(
            runner,
            "flashinfer",
            gdn_cls,
            qsa_cls,
            resolved=resolved,
            flashinfer_cls=_recording_flashinfer_class(built),
        )

    assert built == []  # the draft's explicit backend is also replaced by QSA
    assert runner.init_new_workspace is True
    assert isinstance(result.full_attn_backend, qsa_cls)
    assert result.full_attn_layers == [0]  # draft assumption preserved


@pytest.mark.parametrize(
    ("overrides", "supports_dcp", "expected_built", "rejects"),
    [
        # Guard governs, backend cannot run DCP target verify: rejected, and
        # the rejection still sees the constructed backend (no evasion).
        ({"dcp_size": 2, "enable_unified_memory": True}, False, 1, True),
        # Guard governs, backend declares support: accepted, still eager.
        ({"dcp_size": 2, "enable_unified_memory": True}, True, 1, False),
        # Guard inert (no unified pool): the skip applies with DCP configured.
        ({"dcp_size": 2, "enable_unified_memory": False}, False, 0, False),
    ],
)
def test_qsa_flashinfer_skip_respects_the_dcp_guard(
    overrides, supports_dcp, expected_built, rejects
):
    cfg, runner, gdn_cls, qsa_cls = _hybrid_gdn_qsa_runner(
        spec=SpeculativeAlgorithm.from_string("nextn")
    )
    built = []

    def build():
        return _build_with_real_wrapper(
            runner,
            "flashinfer",
            gdn_cls,
            qsa_cls,
            flashinfer_cls=_recording_flashinfer_class(built, supports_dcp),
            **overrides,
        )

    if rejects:
        # A backend that cannot verify under DCP is still refused, so the
        # skip cannot accidentally evade the guard.
        with pytest.raises(ValueError, match="does not implement target verify"):
            build()
    else:
        result = build()
        assert isinstance(result.full_attn_backend, qsa_cls)
    assert len(built) == expected_built


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
