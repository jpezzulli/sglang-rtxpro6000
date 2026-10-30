"""QSA pending-ring sizing through the real KV-pool factory branch (CPU).

The ring group count comes from the LAUNCH-MAXIMUM verify width, resolved by
the runtime-context capacity accessor ``max_speculative_num_draft_tokens``
(the no-arg interface this file's other sizing sites use; a same-named
arg-taking helper in ``arg_groups.overrides`` is re-export-shadowed at module
level, and calling it as if it took ``server_args`` crashed the first GPU
bring-up with ``TypeError: takes 0 positional arguments but 1 was given``).

These tests run the REAL ``_build_hybrid_linear_kv_pool`` QSA branch and the
REAL ``parse_qsa_profile`` / accessor, with fakes only around the surrounding
config records and allocations, and pin:

* no speculation and native widths 4/8 produce the correct pool ring sizing
  through the actual no-arg capacity accessor;
* an adaptive step-down (active width narrower than the launch capacity)
  does not shrink the ring the captured graphs bound;
* the sizing call is made with the accessor's real no-arg signature (the
  regression that escaped arithmetic-only tests and unit fakes).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

import sglang.srt.mem_cache.kv_cache_configurator as kcc
import sglang.srt.runtime_context as runtime_context
from sglang.srt.layers.attention.qsa.config import parse_qsa_profile
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

RATIO = 4
SLOTS = 5  # request slots in the fake req_to_token table


def _spec(algorithm=None, draft=None, max_draft=None):
    return SimpleNamespace(
        speculative_algorithm=algorithm,
        speculative_num_draft_tokens=draft,
        max_speculative_num_draft_tokens=max_draft,
    )


class _FakeModelConfig:
    hf_config = SimpleNamespace(
        indexer_n_heads=8,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=512 * RATIO,  # block_topk 512: a real compressed profile
        indexer_compress_ratio=RATIO,
    )
    head_dim = 32

    @staticmethod
    def get_num_kv_heads(tp, dcp):
        assert tp == 1 and dcp == 1
        return 2


def _fake_configurator():
    cfg = SimpleNamespace()
    cfg.is_draft_worker = False
    cfg.use_mla_backend = False
    cfg.post_capture_kv_active = False
    cfg.pool_page_size = 64  # a multiple of the compress ratio
    cfg.kv_cache_dtype = torch.bfloat16
    cfg.device = "cpu"
    cfg.layer_info = SimpleNamespace(start_layer=0, end_layer=12)
    cfg.mambaish_config = SimpleNamespace(full_attention_layer_ids=[0, 4, 8])
    cfg.model_config = _FakeModelConfig()
    cfg.server_args = SimpleNamespace()  # passed positionally by the v2 defect
    cfg._build_mha_quant_method = lambda *, num_layers: None
    cfg._hybrid_full_attention_pool_class = lambda *, mha_pool_class, mla_pool_class, dsa_pool_class: mha_pool_class
    cfg._build_hybrid_linear_kv_pool = MethodType(
        kcc.KVCacheConfigurator._build_hybrid_linear_kv_pool, cfg
    )
    return cfg


class TestQsaRingFactorySizing(unittest.TestCase):
    def _build(self, spec):
        cfg = _fake_configurator()
        req_pool = SimpleNamespace(
            req_to_token=torch.zeros(SLOTS, 8, dtype=torch.int32)
        )

        def _fake_copy_warmup(self):
            # Allocation-adjacent fake: the tiled byte-copy warmup needs a
            # triton driver (GPU); the ring sizing does not.
            self._kv_copy_config = None

        with (
            mock.patch.object(kcc, "get_spec", lambda: spec),
            mock.patch.object(kcc, "get_exec", lambda: SimpleNamespace(
                features=SimpleNamespace(enable_memory_saver=False))),
            mock.patch.object(kcc, "get_parallel", lambda: SimpleNamespace(
                attn_tp_size=1, attn_dcp_size=1)),
            mock.patch.object(kcc, "get_model", lambda: SimpleNamespace(
                qsa_indexer_dtype="auto")),
            mock.patch.object(MHATokenToKVPool, "_init_kv_copy_and_warmup", _fake_copy_warmup),
            # The accessor under test runs for real: patch only the resolved
            # spec record the runtime context reads.
            mock.patch.object(runtime_context, "get_spec", lambda: spec),
        ):
            pool = cfg._build_hybrid_linear_kv_pool(
                max_total_num_tokens=256,
                req_to_token_pool=req_pool,
                state_pool=None,
                mha_pool_class=MHATokenToKVPool,
                mla_pool_class=MHATokenToKVPool,
                dsa_pool_class=MHATokenToKVPool,
            )
        return pool

    def _expect_groups(self, pool, groups):
        self.assertIsInstance(pool, QSATokenToKVPool)
        self.assertEqual(pool.qsa_num_groups, groups)
        self.assertEqual(pool.qsa_ring_span, RATIO * groups)
        self.assertEqual(
            pool.get_qsa_key_state_buffer(0).shape, (SLOTS * RATIO * groups, 1, 16)
        )
        self.assertEqual(
            pool.qsa_rope_position_buffer.shape, (SLOTS * RATIO * groups, 3)
        )

    def test_profile_parses(self):
        self.assertEqual(
            parse_qsa_profile(_FakeModelConfig.hf_config).compress_ratio, RATIO
        )

    def test_no_speculation_keeps_the_single_group_layout(self):
        self._expect_groups(self._build(_spec()), 1)

    def test_native_width_4_sizes_two_groups(self):
        self._expect_groups(
            self._build(_spec(algorithm="EAGLE", draft=4)), 2
        )

    def test_native_width_8_sizes_three_groups(self):
        self._expect_groups(
            self._build(_spec(algorithm="NEXTN", draft=8)), 3
        )

    def test_adaptive_step_down_keeps_the_launch_capacity(self):
        # Active width 4 under an adaptive launch that may widen to 8: the
        # ring is bound for the widest window the captured graphs can see.
        self._expect_groups(
            self._build(_spec(algorithm="EAGLE", draft=4, max_draft=8)), 3
        )

    def test_sizing_uses_the_real_no_arg_capacity_accessor(self):
        """Red on the v2 factory branch: it passed ``self.server_args`` to
        the no-arg accessor exactly as thegrid's bring-up did. Call-shape is
        pinned here independent of the returned width."""
        calls = []

        def strict_accessor():  # no-arg, like the runtime-context accessor
            calls.append(None)
            return None

        cfg = _fake_configurator()
        req_pool = SimpleNamespace(
            req_to_token=torch.zeros(SLOTS, 8, dtype=torch.int32)
        )
        with (
            mock.patch.object(kcc, "max_speculative_num_draft_tokens", strict_accessor),
            mock.patch.object(kcc, "get_spec", lambda: _spec()),
            mock.patch.object(kcc, "get_exec", lambda: SimpleNamespace(
                features=SimpleNamespace(enable_memory_saver=False))),
            mock.patch.object(kcc, "get_parallel", lambda: SimpleNamespace(
                attn_tp_size=1, attn_dcp_size=1)),
            mock.patch.object(kcc, "get_model", lambda: SimpleNamespace(
                qsa_indexer_dtype="auto")),
        ):
            pool = cfg._build_hybrid_linear_kv_pool(
                max_total_num_tokens=256,
                req_to_token_pool=req_pool,
                state_pool=None,
                mha_pool_class=MHATokenToKVPool,
                mla_pool_class=MHATokenToKVPool,
                dsa_pool_class=MHATokenToKVPool,
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(pool.qsa_num_groups, 1)


if __name__ == "__main__":
    unittest.main()
