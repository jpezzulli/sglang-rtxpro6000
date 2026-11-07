"""The draft binds the target embed/lm_head once, before KV-pool profiling.

MTP draft classes allocate their own embedding + lm_head shell; the
rebinding in ``set_embed_and_head`` (``del`` + empty_cache) is what makes
that memory reclaimable, and it must run before ``init_target_memory_pool()``
sizes the KV pool. With ``--speculative-token-map`` the bind additionally
materializes a hot-vocab selection head, so a SECOND bind of unchanged
inputs is not a no-op: the new head is allocated while the first is still
referenced (peak = two selected heads, after the pools were sized). This
test exercises the real ``init_token_map``/``init_lm_head``, the real
``shared_hot_lm_head`` selection, and the native MTP ``set_embed_and_head``
on small CPU tensors, and pins the Pennyroyal #159 review contract:

* construction binds exactly once, before any pool call exists;
* re-invoking the bind pair (the old ``alloc_memory_pool`` tail) does NOT
  allocate a replacement selected head;
* without a map the draft head shares the target storage;
* a weight reload (in-place mutation) reopens the gate and rebinds.
"""

import inspect
import os
import tempfile
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative import hot_vocab as hot_vocab_mod
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VOCAB = 32
HIDDEN = 8
HOT_IDS = [2, 5, 9, 17, 31]


@contextmanager
def _noop_ctx(*args, **kwargs):
    yield


def _single_gpu_ps():
    return ParallelState(
        **{
            k: (0 if k.endswith("_rank") or k == "gpu_id" else 1)
            for k in (
                "tp_rank tp_size pp_rank pp_size dp_rank dp_size"
                " attn_tp_rank attn_tp_size attn_cp_rank attn_cp_size"
                " attn_dcp_rank attn_dcp_size attn_dp_rank attn_dp_size"
                " moe_ep_rank moe_ep_size moe_dp_rank moe_dp_size gpu_id"
            ).split()
        }
    )


class _TargetModel(nn.Module):
    """Real tensors; no shard metadata, so hot selection takes the
    qualified TP1 local path (pure torch, CPU-clean)."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.lm_head = nn.Module()
        self.lm_head.weight = nn.Parameter(
            torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
        )

    def get_embed_and_head(self):
        return self.embed_tokens.weight, self.lm_head.weight


def _make_draft_shell():
    shell = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=False),
        model=SimpleNamespace(embed_tokens=nn.Embedding(4, HIDDEN)),
        lm_head=SimpleNamespace(weight=nn.Parameter(torch.full((4, HIDDEN), 7.0))),
        hot_token_id=None,
    )
    # Native MTP setter/unsetter, bound to the shell: the real del + rebind
    # semantics under test, not a mock.
    shell.set_embed_and_head = types.MethodType(
        Qwen3_5ForCausalLMMTP.set_embed_and_head, shell
    )
    shell.set_lm_head_from_target = types.MethodType(
        Qwen3_5ForCausalLMMTP.set_lm_head_from_target, shell
    )
    return shell


class _Harness:
    """Constructs a real EagleDraftWorker with CPU tensor models."""

    def __init__(self, token_map_path=None):
        self.target_model = _TargetModel()
        self.draft_shell = _make_draft_shell()
        draft_runner = SimpleNamespace(
            model=self.draft_shell, model_config=SimpleNamespace(context_len=8192)
        )

        class FakeTpModelWorker:
            def __init__(self, **kwargs):
                self.model_runner = draft_runner

        self.target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=self.target_model,
                model_config=SimpleNamespace(context_len=8192),
                spec_aux_config=SimpleNamespace(draft_model_config=None),
            ),
            random_seed=1,
        )
        self._tmp = tempfile.TemporaryDirectory()
        if token_map_path is None:
            self.map_path = None
        else:
            self.map_path = token_map_path
            torch.save(torch.tensor(HOT_IDS, dtype=torch.int64), self.map_path)

        M = "sglang.srt.speculative.eagle_worker_v2."
        self._patches = [
            patch(M + "TpModelWorker", FakeTpModelWorker),
            patch(M + "draft_tp_context", _noop_ctx),
            patch(M + "speculative_moe_backend_context", _noop_ctx),
            patch(M + "speculative_moe_a2a_backend_context", _noop_ctx),
            patch(M + "draft_model_build_scope", _noop_ctx),
            patch(M + "get_plan_stream", lambda device: (object(), _noop_ctx())),
            patch.object(
                EagleDraftWorker, "_init_dsa_index_share_state", lambda self: None
            ),
            patch.object(
                EagleDraftWorker, "_rebuild_topk1_chain_buffers", lambda self: None
            ),
            patch("torch.cuda.empty_cache", lambda: None),
            patch("torch.cuda.synchronize", lambda: None),
        ]
        kw = dict(
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=1,
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            speculative_use_rejection_sampling=False,
        )
        if self.map_path is not None:
            kw["speculative_token_map"] = self.map_path
        self.override = get_context().override_server_args(**kw)

    def __enter__(self):
        for p in self._patches:
            p.start()
        self.override.install()
        self.worker = EagleDraftWorker(
            server_args=SimpleNamespace(),
            gpu_id=0,
            ps=_single_gpu_ps(),
            nccl_port=0,
            target_worker=self.target_worker,
        )
        return self

    def __exit__(self, *exc):
        self.override.restore()
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def bound_head(self):
        return self.worker.draft_runner.model.lm_head.weight

    def rebind_pair(self):
        """Exactly what alloc_memory_pool() used to run at its tail."""
        self.worker.init_token_map()
        self.worker.init_lm_head()


def _selection_spy():
    calls = []
    real = hot_vocab_mod.shared_hot_lm_head

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out.head.data_ptr())
        return out

    return calls, patch.object(hot_vocab_mod, "shared_hot_lm_head", spy)


class TestEmbedBindOnceBeforePool(unittest.TestCase):
    def test_construction_binds_with_map_and_rows_are_selected(self):
        with tempfile.TemporaryDirectory() as td:
            with _Harness(os.path.join(td, "map.pt")) as h:
                head = h.bound_head()
                # input embedding is shared with the target, not copied
                self.assertEqual(
                    h.worker.draft_runner.model.model.embed_tokens.weight.data_ptr(),
                    h.target_model.embed_tokens.weight.data_ptr(),
                )
                # head is the hot selection: exactly the mapped rows of the
                # TARGET head, in map order.
                self.assertEqual(tuple(head.shape), (len(HOT_IDS), HIDDEN))
                tgt = h.target_model.lm_head.weight
                for i, gid in enumerate(HOT_IDS):
                    torch.testing.assert_close(head[i], tgt[gid])

    def test_repeated_bind_allocates_no_replacement_head(self):
        calls, spy_patch = _selection_spy()
        with spy_patch:
            with tempfile.TemporaryDirectory() as td:
                with _Harness(os.path.join(td, "map.pt")) as h:
                    ptr_ctor = h.bound_head().data_ptr()
                    self.assertEqual(len(calls), 1, "construction binds exactly once")
                    h.rebind_pair()
                    h.rebind_pair()
                    self.assertEqual(len(calls), 1, "no replacement selected head")
                    self.assertEqual(h.bound_head().data_ptr(), ptr_ctor)

    def test_no_map_shares_target_storage_and_stays_quiet(self):
        with _Harness() as h:
            tgt = h.target_model.lm_head.weight.data_ptr()
            self.assertEqual(h.bound_head().data_ptr(), tgt)
            h.rebind_pair()
            self.assertEqual(h.bound_head().data_ptr(), tgt)

    def test_weight_reload_reopens_the_bind_gate(self):
        calls, spy_patch = _selection_spy()
        with spy_patch:
            with tempfile.TemporaryDirectory() as td:
                with _Harness(os.path.join(td, "map.pt")) as h:
                    ptr_ctor = h.bound_head().data_ptr()
                    # In-place target head update (weight-load semantics).
                    with torch.no_grad():
                        h.target_model.lm_head.weight.add_(1.0)
                    h.rebind_pair()
                    self.assertEqual(len(calls), 2, "reload rebinds")
                    new = h.bound_head()
                    self.assertNotEqual(new.data_ptr(), ptr_ctor)
                    tgt = h.target_model.lm_head.weight
                    for i, gid in enumerate(HOT_IDS):
                        torch.testing.assert_close(new[i], tgt[gid])

    def test_alloc_memory_pool_does_not_rebind(self):
        src = inspect.getsource(EagleDraftWorker.alloc_memory_pool)
        self.assertNotIn("init_token_map", src)
        self.assertNotIn("init_lm_head", src)


if __name__ == "__main__":
    unittest.main()
