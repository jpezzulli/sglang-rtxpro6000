"""CPU-only accounting for the speculative Mamba verify scratch.

``MambaPool`` allocates two per-draft-token buffers beside the committed state
(see ``memory_pool.py``): ``intermediate_ssm_state_cache``
``[num_layers, spec_slots, draft_tokens, *temporal]`` and
``intermediate_conv_window_cache``, in either the deduplicated sliding-window or
the dense layout. The memory solver used to stand the whole ``mamba_cache_per_req``
(conv + SSM) in for the snapshots, so a ``gdn_mtp_cache_mode=none`` rank kept paying
for a draft-token-deep copy of the state its pool never allocates; it never charged
the conv rollback windows that EVERY spec mode allocates, which it would also have
mispriced at ``draft * (K-1)`` columns instead of the ``draft + K - 2`` the dedup
layout physically stores. Where the old solver did drop the scratch (ReplaySSM spec)
it dropped all of it, windows included -- under-reserving instead of over-reserving.

These tests pin the reserve to the buffers the pool really allocates: the SSM term
only for the modes that allocate it, the conv rollback windows (the accepted
conv-state history, which stays in EVERY spec mode) always, sized from the real
dtype / TP-sharded / PP-stage geometry plus the pool's padding slot. No model names:
everything comes from the cache params and the published spec/mamba config, so a
NEXTN-style and a DFlash-style draft (both linear chains, topk 1) or a KDA pool
(dense windows) are just configurations.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt import runtime_context as rc
from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.mem_cache import kv_cache_configurator as kcc_module
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.runtime_context import get_schedule
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

GB = 1 << 30
# A Qwen3-Next-shaped linear layer: conv state (3072, K-1), SSM state
# (32, 128, 128), bf16 conv / fp32 SSM.
INTERMEDIATE_SIZE = 2048
N_GROUPS = 4
NUM_HEADS = 32
HEAD_DIM = 128
STATE_SIZE = 128
CONV_KERNEL = 4
CONV_DIM = INTERMEDIATE_SIZE + 2 * N_GROUPS * STATE_SIZE  # 3072
WIN = CONV_KERNEL - 1  # the conv state's sliding-window axis is (dim, K-1)
SSM_ELEMS = (NUM_HEADS, HEAD_DIM, STATE_SIZE)
MAMBA_LAYERS = [0, 1, 2, 3, 8, 9, 10, 11]
N_LAYERS = len(MAMBA_LAYERS)
DRAFT = 5
REPLAYSSM_RECORD_LEN = 16  # --linear-replayssm-cache-len default
BUDGET_GB = 24.0
MAMBA_MEM_RATIO = 0.5
MAX_RUNNING = 8


# --- per-spec-slot byte expectations, written out from the pool's shapes -------
def ssm_charge(draft=DRAFT, *, layers=N_LAYERS, itemsize=4, elems=SSM_ELEMS):
    return elems[0] * elems[1] * elems[2] * itemsize * draft * layers


def conv_dedup_charge(draft=DRAFT, *, layers=N_LAYERS, itemsize=2, conv_dim=CONV_DIM):
    return conv_dim * (draft + WIN - 1) * itemsize * layers


def conv_dense_charge(draft=DRAFT, *, layers=N_LAYERS, itemsize=2, conv_dim=CONV_DIM):
    return conv_dim * WIN * draft * itemsize * layers


def _dtype(ssm_dtype=torch.float32):
    return Mamba2StateDType(conv=torch.bfloat16, temporal=ssm_dtype)


def _params(layers=MAMBA_LAYERS, *, tp=1, ssm_dtype=torch.float32):
    shape = Mamba2StateShape.create(
        tp_world_size=tp,
        intermediate_size=INTERMEDIATE_SIZE,
        n_groups=N_GROUPS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        state_size=STATE_SIZE,
        conv_kernel=CONV_KERNEL,
    )
    return Mamba2CacheParams(shape=shape, dtype=_dtype(ssm_dtype), layers=list(layers))


def _kda_params(layers=MAMBA_LAYERS):
    shape = KimiLinearStateShape.create(
        tp_world_size=1,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_k_heads=8,
        head_k_dim=STATE_SIZE,
        conv_kernel_size=CONV_KERNEL,
    )
    return KimiLinearCacheParams(shape=shape, dtype=_dtype(), layers=list(layers))


def _kda_conv_dense_charge(params, draft=DRAFT):
    return (
        sum(c[0] * c[1] for c in params.shape.conv)
        * draft
        * params.dtype.conv.itemsize
        * len(params.layers)
    )


def _autofit(
    budget_gb,
    resident,
    intermediate,
    ratio,
    max_running,
    dp_size=1,
    scratch_slots=1,
    fixed=0,
):
    """Mirror of the solver's joint equation:
      (K + 1) * resident + (K / ratio + scratch_slots) * intermediate + fixed
        = mamba budget
    ``scratch_slots`` is the pool's padding slot (1) plus any PD-decode pre-alloc
    slots; ``fixed`` is request-keyed scratch that does not scale with K (the GDN
    ReplaySSM ring, sized like the request pool rather than the state pool).
    returns the solved slot count and the memory left for the KV pool."""
    budget_bytes = budget_gb * MAMBA_MEM_RATIO / (1 + MAMBA_MEM_RATIO) * GB
    slots = int(
        (budget_bytes - fixed - resident - scratch_slots * intermediate)
        // (resident + intermediate / ratio)
    )
    capped = min(max_running // dp_size, slots // ratio)
    rest = (
        budget_gb
        - (slots + 1) * resident / GB
        - (capped + scratch_slots) * intermediate / GB
        - fixed / GB
    )
    return slots, rest


class _Solver:
    """Run ``_handle_max_mamba_cache`` over a published configuration."""

    def __init__(self, params, *, pp_size=1, pp_layers=None):
        self.params = params
        self.pp_size = pp_size
        # Layer ids the PP split is computed over (params.layers by default, i.e.
        # every mamba layer of the model -- the rank then charges its own slice).
        self.pp_layers = pp_layers

    def solve(
        self,
        *,
        spec=None,
        topk=None,
        adaptive=False,
        mode="full",
        replayssm_spec=False,
        disagg="null",
        extra_slots=None,
        max_mamba_cache_size=None,
        max_running_requests=MAX_RUNNING,
        disable_radix=False,
        budget_gb=BUDGET_GB,
        dp_size=1,
        cuda=True,
        publish_mode=True,
    ):
        # A plain namespace: _handle_max_mamba_cache and _calculate_mamba_ratio are
        # called unbound, so only the attributes they read have to exist.
        fake = SimpleNamespace(
            mambaish_config=SimpleNamespace(mamba2_cache_params=self.params),
            spec_algorithm=SimpleNamespace(is_none=lambda: spec is None),
            attn_dp_size=dp_size,
            pp_size=self.pp_size,
            # No fused draft-state block: the solver prices the plain mamba entry.
            fused_entry_bytes=lambda name: None,
            extra_mamba_cache_bytes_per_req=0,
            hybrid_gdn_config=object(),
            hybrid_kda_config=None if not self.params.is_kda else object(),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(),
                num_hidden_layers=max(self.pp_layers or self.params.layers) + 1,
            ),
        )
        fake._calculate_mamba_ratio = lambda: (
            KVCacheConfigurator._calculate_mamba_ratio(fake)
        )
        fields = dict(
            disable_radix_cache=disable_radix,
            max_mamba_cache_size=max_mamba_cache_size,
            max_running_requests=max_running_requests,
            mamba_full_memory_ratio=MAMBA_MEM_RATIO,
            enable_linear_replayssm_spec=replayssm_spec,
            enable_linear_replayssm=False,
            speculative_algorithm="NEXTN" if spec is not None else "NONE",
            speculative_num_draft_tokens=spec,
            speculative_eagle_topk=topk,
            speculative_adaptive=adaptive,
            disaggregation_mode=disagg,
            disaggregation_decode_extra_slots=extra_slots,
            mamba_radix_cache_strategy="extra_buffer",
            disable_overlap_schedule=False,
        )
        if publish_mode:
            fields["gdn_mtp_cache_mode"] = mode
        with rc.get_context().override_server_args(**fields):
            with (
                # create=True so the same solver runs against a checkout that has
                # no platform flags yet (the base the reserve bug came from).
                patch.object(kcc_module, "_is_cpu", not cuda, create=True),
                patch.object(kcc_module, "_is_npu", False, create=True),
            ):
                rest = KVCacheConfigurator._handle_max_mamba_cache(fake, budget_gb)
                ratio = KVCacheConfigurator._calculate_mamba_ratio(fake)
                return rest, get_schedule().max_mamba_cache_size, ratio


class TestIntermediateBufferSizing(unittest.TestCase):
    """The pure sizing helpers mirror the pool's allocation shapes."""

    def setUp(self):
        self.params = _params()

    def test_ssm_snapshots(self):
        self.assertEqual(
            self.params.intermediate_ssm_bytes_per_slot(DRAFT), ssm_charge()
        )
        # no draft window -> no snapshots (nothing to price in the solver)
        self.assertEqual(self.params.intermediate_ssm_bytes_per_slot(0), 0)

    def test_ssm_snapshots_follow_the_real_dtypes(self):
        bf16 = _params(ssm_dtype=torch.bfloat16)
        self.assertEqual(
            bf16.intermediate_ssm_bytes_per_slot(DRAFT), ssm_charge(itemsize=2)
        )

    def test_ssm_snapshots_follow_the_tp_shard(self):
        self.assertEqual(
            _params(tp=2).intermediate_ssm_bytes_per_slot(DRAFT),
            ssm_charge(elems=(NUM_HEADS // 2, HEAD_DIM, STATE_SIZE)),
        )

    def test_conv_windows_dedup_vs_dense(self):
        self.assertEqual(
            self.params.intermediate_conv_window_bytes_per_slot(DRAFT, dedup=True),
            conv_dedup_charge(),
        )
        self.assertEqual(
            self.params.intermediate_conv_window_bytes_per_slot(DRAFT, dedup=False),
            conv_dense_charge(),
        )
        # the whole point of the dedup layout
        self.assertLess(
            self.params.intermediate_conv_window_bytes_per_slot(DRAFT, dedup=True),
            self.params.intermediate_conv_window_bytes_per_slot(DRAFT, dedup=False),
        )
        for dedup in (True, False):
            self.assertEqual(
                self.params.intermediate_conv_window_bytes_per_slot(0, dedup=dedup), 0
            )

    def test_conv_windows_honor_layer_count_and_tp_shard(self):
        self.assertEqual(
            _params(tp=2).intermediate_conv_window_bytes_per_slot(DRAFT, dedup=True),
            conv_dedup_charge(conv_dim=CONV_DIM // 2),
        )
        self.assertEqual(
            _params(layers=[0]).intermediate_conv_window_bytes_per_slot(
                DRAFT, dedup=True
            ),
            conv_dedup_charge(layers=1),
        )

    def test_kda_windows_are_dense(self):
        # KDA stores conv state as (K-1, dim), so the overlapping view stays off.
        params = _kda_params()
        self.assertTrue(params.shape.disable_conv_window_dedup)
        self.assertTrue(params.is_kda)
        self.assertEqual(
            params.intermediate_conv_window_bytes_per_slot(DRAFT, dedup=False),
            _kda_conv_dense_charge(params),
        )


class TestAutoFitSpecReserve(unittest.TestCase):
    """The auto-fit branch: joint solve over resident state + verify scratch."""

    def setUp(self):
        self.params = _params()
        self.solver = _Solver(self.params)
        self.resident = self.params.mamba_cache_per_req

    def charged_intermediate(self, rest, slots):
        """Bytes the solver left for the per-draft-token scratch."""
        return round((BUDGET_GB - rest - (slots + 1) * self.resident / GB) * GB)

    def test_mode_none_charges_conv_windows_only(self):
        rest, slots, ratio = self.solver.solve(spec=DRAFT, mode="none")
        want_slots, want_rest = _autofit(
            BUDGET_GB, self.resident, conv_dedup_charge(), ratio, MAX_RUNNING
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)

    def test_mode_full_charges_snapshots_and_windows(self):
        rest, slots, ratio = self.solver.solve(spec=DRAFT, mode="full")
        want_slots, want_rest = _autofit(
            BUDGET_GB,
            self.resident,
            conv_dedup_charge() + ssm_charge(),
            ratio,
            MAX_RUNNING,
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)

    def test_none_frees_exactly_the_dropped_snapshots(self):
        """The base bug: none-mode kept paying for snapshots its pool never holds.

        The reserve the auto-fit branch actually spends on the verify scratch is
        read back out of the returned budget; in none-mode it must be exactly the
        conv rollback buffers (which stay allocated in every spec mode), and the
        freed bytes must come back as mamba state slots.
        """
        none_rest, none_slots, ratio = self.solver.solve(spec=DRAFT, mode="none")
        full_rest, full_slots, _ = self.solver.solve(spec=DRAFT, mode="full")
        self.assertEqual(
            self.charged_intermediate(none_rest, none_slots),
            (min(MAX_RUNNING, none_slots // ratio) + 1) * conv_dedup_charge(),
        )
        self.assertGreater(none_slots, full_slots)  # the saving goes to the pool

    def test_reserve_is_the_real_scratch_not_the_per_request_stand_in(self):
        # Behaviourally red on the exact base, with no mode field in play: whatever
        # the mode, the reserve has to be the buffers the pool allocates. This launch
        # keeps the snapshots (the default mode), so the charge is the snapshots PLUS
        # the conv rollback windows. The pre-fix solver charged the whole committed
        # state (conv AND ssm) once per draft token and nothing for the windows --
        # the wrong total and the wrong shape, pricing the windows at
        # draft * (K-1) columns instead of the dedup layout`s draft + K - 2.
        rest, slots, ratio = self.solver.solve(spec=DRAFT, publish_mode=False)
        self.assertEqual(
            self.charged_intermediate(rest, slots),
            (min(MAX_RUNNING, slots // ratio) + 1)
            * (conv_dedup_charge() + ssm_charge()),
        )

    def test_base_stand_in_over_reserved(self):
        """What the pre-fix reserve charged: mamba_cache_per_req * slots * draft."""
        legacy = self.resident * (MAX_RUNNING + 1) * DRAFT
        correct = (MAX_RUNNING + 1) * conv_dedup_charge()
        self.assertGreater(legacy, 4 * correct)

    def test_tree_verify_falls_back_to_dense_windows(self):
        rest, slots, ratio = self.solver.solve(spec=DRAFT, topk=2)
        want_slots, want_rest = _autofit(
            BUDGET_GB,
            self.resident,
            conv_dense_charge() + ssm_charge(),
            ratio,
            MAX_RUNNING,
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)

    def test_platform_without_dedup_charges_dense_windows(self):
        cuda_rest, cuda_slots, ratio = self.solver.solve(spec=DRAFT, mode="none")
        self.assertEqual(
            self.charged_intermediate(cuda_rest, cuda_slots),
            (min(MAX_RUNNING, cuda_slots // ratio) + 1) * conv_dedup_charge(),
        )
        cpu_rest, cpu_slots, _ = self.solver.solve(spec=DRAFT, mode="none", cuda=False)
        self.assertEqual(
            self.charged_intermediate(cpu_rest, cpu_slots),
            (min(MAX_RUNNING, cpu_slots // ratio) + 1) * conv_dense_charge(),
        )
        self.assertGreater(
            self.charged_intermediate(cpu_rest, cpu_slots),
            self.charged_intermediate(cuda_rest, cuda_slots),
        )

    def test_adaptive_draft_window_is_priced_at_its_bound(self):
        """Adaptive NEXTN runs W1..W8, so the buffers -- and the reserve -- are
        sized to the widest candidate window, not the flat draft-token count."""
        flat_rest, flat_slots, ratio = self.solver.solve(spec=2)
        adapt_rest, adapt_slots, _ = self.solver.solve(spec=2, adaptive=True)
        bound = 8  # max(DEFAULT_ADAPTIVE_CONFIG candidate_steps) + 1
        want_slots, want_rest = _autofit(
            BUDGET_GB,
            self.resident,
            conv_dedup_charge(bound) + ssm_charge(bound),
            ratio,
            MAX_RUNNING,
        )
        self.assertEqual(adapt_slots, want_slots)
        self.assertAlmostEqual(adapt_rest, want_rest, delta=1e-9)
        self.assertLess(adapt_slots, flat_slots)

    def test_no_spec_allocates_no_scratch(self):
        rest, slots, ratio = self.solver.solve(spec=None)
        want_slots, want_rest = _autofit(
            BUDGET_GB, self.resident, 0, ratio, MAX_RUNNING
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)
        # the cache mode is irrelevant once nothing allocates a draft window
        none_rest, none_slots, _ = self.solver.solve(spec=None, mode="none")
        self.assertAlmostEqual(none_rest, rest, delta=1e-9)
        self.assertEqual(none_slots, slots)

    def test_prefill_only_server_allocates_no_scratch(self):
        """A PD prefill server builds its pool with no draft window at all."""
        rest, slots, ratio = self.solver.solve(spec=DRAFT, disagg="prefill")
        want_slots, want_rest = _autofit(
            BUDGET_GB, self.resident, 0, ratio, MAX_RUNNING
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)

    def test_pd_decode_charges_its_prealloc_scratch_slots(self):
        """HybridMambaDecodeReqToTokenPool keys the scratch by size + pre_alloc_size
        (see disaggregation/decode.py), so the in-transfer slots must be reserved
        too -- capped_reqs plus the padding slot alone underbudgets them."""
        extra = 16
        intermediate = conv_dedup_charge()  # mode=none: windows only
        rest, slots, ratio = self.solver.solve(
            spec=DRAFT, mode="none", disagg="decode", extra_slots=extra
        )
        want_slots, want_rest = _autofit(
            BUDGET_GB,
            self.resident,
            intermediate,
            ratio,
            MAX_RUNNING,
            scratch_slots=1 + extra,
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)
        self.assertEqual(
            self.charged_intermediate(rest, slots),
            (min(MAX_RUNNING, slots // ratio) + 1 + extra) * intermediate,
        )
        # Only the decode role pays: the same extra_slots published for a null or
        # prefill launch describe slots that pool does not have.
        other_rest, other_slots, _ = self.solver.solve(spec=DRAFT, extra_slots=extra)
        base_rest, base_slots, _ = self.solver.solve(spec=DRAFT)
        self.assertEqual((other_rest, other_slots), (base_rest, base_slots))

    def test_replayssm_spec_drops_snapshots_and_charges_the_ring(self):
        params = _params(layers=list(range(16)))
        solver = _Solver(params)
        rest, slots, ratio = solver.solve(spec=DRAFT, replayssm_spec=True)
        # The candidate sizes the ReplaySSM window from --linear-replayssm-cache-len
        # (the published default here), not from the draft width.
        ring = params.replayssm_ring_bytes_per_req(record_len=REPLAYSSM_RECORD_LEN)
        intermediate = conv_dedup_charge(DRAFT, layers=16)
        # GDN sizes the fold window to the draft maximum and keys it by REQUEST
        # (see MambaPool + _handle_max_mamba_cache), so it is fixed bytes, not a
        # per-slot term; the KDA ring stays attached to each mamba slot.
        fixed = ring * (MAX_RUNNING + 1)
        want_slots, want_rest = _autofit(
            BUDGET_GB,
            params.mamba_cache_per_req,
            intermediate,
            ratio,
            MAX_RUNNING,
            fixed=fixed,
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)

    def test_kda_pool_charges_dense_windows_and_snapshots(self):
        params = _kda_params()
        solver = _Solver(params)
        rest, slots, ratio = solver.solve(spec=DRAFT, mode="full")
        intermediate = _kda_conv_dense_charge(params) + ssm_charge()
        want_slots, want_rest = _autofit(
            BUDGET_GB, params.mamba_cache_per_req, intermediate, ratio, MAX_RUNNING
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)


class TestExplicitSizeReserve(unittest.TestCase):
    """The explicit-max_mamba_cache_size and disabled-radix branches."""

    def setUp(self):
        self.params = _params()
        self.solver = _Solver(self.params)
        self.resident = self.params.mamba_cache_per_req

    def _rest(self, *, mode, size=64, spec=DRAFT, **kwargs):
        rest, slots, ratio = self.solver.solve(
            spec=spec, mode=mode, max_mamba_cache_size=size, **kwargs
        )
        return rest, slots, ratio, size

    def test_explicit_size_none_mode_saving_is_the_snapshots(self):
        """With K pinned, the whole difference is the dropped snapshots."""
        rest_none, _, ratio, size = self._rest(mode="none")
        rest_full, _, _, _ = self._rest(mode="full")
        self.assertAlmostEqual(
            rest_none - rest_full,
            (min(MAX_RUNNING, size // ratio) + 1) * ssm_charge() / GB,
            delta=1e-9,
        )

    def test_explicit_size_charges_the_mode_scratch(self):
        for mode, intermediate in (
            ("none", conv_dedup_charge()),
            ("full", conv_dedup_charge() + ssm_charge()),
        ):
            rest, _, ratio, size = self._rest(mode=mode)
            capped = min(MAX_RUNNING, size // ratio)
            self.assertAlmostEqual(
                rest,
                BUDGET_GB
                - (size + 1) * self.resident / GB
                - (capped + 1) * intermediate / GB,
                delta=1e-9,
                msg=mode,
            )

    def test_explicit_size_pd_decode_prealloc_exact_bytes(self):
        extra, size = 16, 64
        per_slot = conv_dedup_charge()  # mode=none: conv rollback windows only
        rest_dec, slots_dec, _, _ = self._rest(
            mode="none", size=size, disagg="decode", extra_slots=extra
        )
        rest_plain, slots_plain, _, _ = self._rest(mode="none", size=size)
        self.assertEqual(slots_dec, slots_plain)  # explicit size is not solved
        self.assertAlmostEqual(rest_plain - rest_dec, extra * per_slot / GB, delta=1e-9)

    def test_disabled_radix_uses_the_request_sized_pool(self):
        rest, slots, ratio = self.solver.solve(
            spec=DRAFT, mode="none", disable_radix=True, max_running_requests=12
        )
        self.assertEqual((slots, ratio), (12, 1))  # ratio is 1 with the radix cache off
        intermediate = conv_dedup_charge()
        self.assertAlmostEqual(
            rest,
            BUDGET_GB
            - (slots + 1) * self.resident / GB
            - (slots + 1) * intermediate / GB,
            delta=1e-9,
        )
        # none-mode still frees the snapshots here too
        full_rest, _, _ = self.solver.solve(
            spec=DRAFT, mode="full", disable_radix=True, max_running_requests=12
        )
        self.assertAlmostEqual(
            rest - full_rest, (slots + 1) * ssm_charge() / GB, delta=1e-9
        )

    def test_dp_shard_divides_the_explicit_pool(self):
        rest, slots, _ = self.solver.solve(
            spec=DRAFT, max_mamba_cache_size=64, dp_size=2, max_running_requests=8
        )
        self.assertEqual(slots, 32)


class TestPPStageScratch(unittest.TestCase):
    """A PP rank holds its own layer slice, so it charges only that slice -- for
    the verify scratch too, not just the committed state."""

    TOTAL_LAYERS = 24
    ALL_MAMBA_LAYERS = [i for i in range(TOTAL_LAYERS) if i % 2 == 0]

    def _solve(self, pp_rank, mode):
        params = _params(layers=self.ALL_MAMBA_LAYERS)
        solver = _Solver(params, pp_size=2, pp_layers=list(range(self.TOTAL_LAYERS)))
        rest, slots, ratio = solver.solve(spec=DRAFT, mode=mode, max_running_requests=4)
        stage = sum(
            1
            for i in self.ALL_MAMBA_LAYERS
            if i in range(*get_pp_indices(self.TOTAL_LAYERS, pp_rank, 2))
        )
        scale = stage / len(self.ALL_MAMBA_LAYERS)
        resident = int(params.mamba_cache_per_req * scale)
        layers = len(self.ALL_MAMBA_LAYERS)
        intermediate = int(
            (
                conv_dedup_charge(layers=layers)
                + (ssm_charge(layers=layers) if mode != "none" else 0)
            )
            * scale
        )
        want_slots, want_rest = _autofit(BUDGET_GB, resident, intermediate, ratio, 4)
        return rest, slots, want_rest, want_slots

    def test_scratch_scales_with_the_stage(self):
        for mode in ("full", "none"):
            rest, slots, want_rest, want_slots = self._solve(0, mode)
            self.assertEqual(slots, want_slots, mode)
            self.assertAlmostEqual(rest, want_rest, delta=1e-9, msg=mode)

    def test_none_mode_grows_the_stage_pool(self):
        _, none_slots, _, _ = self._solve(0, "none")
        _, full_slots, _, _ = self._solve(0, "full")
        self.assertGreater(none_slots, full_slots)

    def test_both_ranks_agree(self):
        _, rank0_slots, _, _ = self._solve(0, "full")
        _, rank1_slots, _, _ = self._solve(1, "full")
        self.assertEqual(rank0_slots, rank1_slots)


class TestPoolAllocationGate(unittest.TestCase):
    """The solver charges what ``MambaPool`` allocates; this pins the allocation.

    Both the mode gate and the conv rollback buffers are pool behaviour, so the
    reserve and the buffers cannot drift apart silently: none-mode drops the SSM
    snapshots and nothing else.
    """

    SPEC_SLOTS = 4  # spec_state_size; the pool adds one padding slot

    def _pool(self, *, mode, spec=DRAFT, replayssm_spec=False, topk=None):
        # The mode is handed to the pool, not read from the ambient context: the
        # allocation decision belongs to the caller (KVCacheConfigurator) and the
        # pool must be constructible with no published config at all.
        return MambaPool(
            size=4,
            spec_state_size=self.SPEC_SLOTS,
            cache_params=_params(),
            mamba_layer_ids=MAMBA_LAYERS,
            device="cpu",
            speculative_num_draft_tokens=spec,
            speculative_eagle_topk=topk,
            enable_linear_replayssm_spec=replayssm_spec,
            gdn_mtp_cache_mode=mode,
        )

    def test_none_mode_drops_only_the_ssm_snapshots(self):
        pool = self._pool(mode="none")
        self.assertIsNone(pool.mamba_cache.intermediate_ssm)
        # The conv rollback windows stay, in the dense layout this platform uses.
        self.assertEqual(
            [get_tensor_bytes(w) for w in pool.mamba_cache.intermediate_conv_window],
            [conv_dense_charge() * (self.SPEC_SLOTS + 1)],
        )
        # No CUDA event to fence without a CUDA state pool.
        self.assertIsNone(pool.state_recovery_event)

    def test_full_mode_keeps_snapshots_and_windows(self):
        pool = self._pool(mode="full")
        self.assertEqual(
            get_tensor_bytes(pool.mamba_cache.intermediate_ssm),
            ssm_charge() * (self.SPEC_SLOTS + 1),
        )
        self.assertEqual(
            [get_tensor_bytes(w) for w in pool.mamba_cache.intermediate_conv_window],
            [conv_dense_charge() * (self.SPEC_SLOTS + 1)],
        )

    def test_replayssm_spec_and_none_mode_agree_on_the_ssm_term(self):
        # Both protocols own the rollback elsewhere, so both leave intermediate_ssm
        # unallocated -- which is why the runtime gate reads the MODE and not
        # whether the pool happens to have left a buffer out.
        self.assertIsNone(self._pool(mode="none").mamba_cache.intermediate_ssm)
        self.assertIsNone(
            self._pool(mode="full", replayssm_spec=True).mamba_cache.intermediate_ssm
        )

    def test_no_draft_window_allocates_no_scratch(self):
        pool = self._pool(mode="full", spec=None)
        self.assertIsInstance(pool.mamba_cache, MambaPool.State)
        self.assertFalse(hasattr(pool.mamba_cache, "intermediate_ssm"))


def get_tensor_bytes(t):
    return int(torch.numel(t) * t.element_size())


if __name__ == "__main__":
    unittest.main()
