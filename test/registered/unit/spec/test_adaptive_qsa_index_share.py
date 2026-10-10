"""CPU-only checks for QSA MTP index sharing under the adaptive widths.

``index_share_for_mtp_iteration`` lets the draft-extend pass run the QSA indexer
once and the following speculative decode steps reuse that target-aligned
selection (``QSAMTPSharedSparseIndices``). Adaptive speculative decoding keeps
one ``SpecRuntimeState`` per candidate width -- W4/W8/W16 are 3/7/15 steps at
topk=1 -- and each of them builds its own backends and captures its own CUDA
graphs, so the selection buffers are part of the state a width owns: a captured
graph keeps the buffer addresses it was installed with, and the tail columns are
sized for that width's own chain (``tail_width = steps + 1``).

These checks pin the four mechanics that make that true instead of trusting that
removing the old "adaptive disables index sharing" guard is enough:

* every adaptive width allocates its OWN state, sized for its own tail, and binds
  both of its backends to it (the donor behaviour this is ported from; the base
  refused sharing outright under adaptive, so these are red there);
* a transition hands the authoritative rows over: the frozen selection and its
  captured lengths belong to the request, the buffers to a width, and the
  controller activates widths at the next actual forward -- so without the
  handoff the first draft of the incoming width reads a cache its own
  draft-extend never wrote (zeros after a foreign prefill, a previous request's
  rows on a return, tail positions dropped);
* the fixed-width path keeps reusing the one state its backend already holds --
  the guard's replacement is per-width ownership, not a fresh allocation on
  every call -- and an adaptive table that can reach steps <= 1 (the generic
  default shape) keeps its prior unshared path at every width, so no transition
  straddles the seeded and unseeded regimes;
* a candidate may not inherit the launch width's state even when the factory
  hands it the draft runner's own (shared) draft-extend backend, which is why
  the build defers the binding in ``init_attention_backend`` and configures the
  share itself once ``_own_draft_extend_backend`` has settled the backends --
  before ``_capture_cuda_graphs`` bakes the addresses;
* the geometry a width's graphs record and replay against: frozen selection
  columns plus exactly ``steps + 1`` tail columns, with the unused tail slots
  dropped (-1) rather than read as token indices.

All evidence is CPU-only (no CUDA on this host): real capture, replay and
numerics remain GPU-window work.
"""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QSAMTPSharedSparseIndices,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.adaptive_spec_params import DEFAULT_ADAPTIVE_CONFIG
from sglang.srt.speculative.eagle_worker_v2 import (
    EagleDraftWorker,
    EAGLEWorkerV2,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The candidate table of configs/pennyroyal/adaptive-next.json.
WIDTHS = (3, 7, 15)
TOKEN_TOPK = 8
COMPRESS_RATIO = 4
# The indexer expansion emits token_topk + ratio - 1 selection columns.
EXPANDED_TOPK = TOKEN_TOPK + COMPRESS_RATIO - 1
NUM_REQUESTS = 6
LAYER_IDS = [1, 3]


def _hf_config():
    """A compressed-QSA draft config that asks for MTP index sharing."""
    return SimpleNamespace(
        model_type="qwen4_exp",
        indexer_n_heads=64,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=COMPRESS_RATIO * 512,
        indexer_compress_ratio=COMPRESS_RATIO,
        index_share_for_mtp_iteration=True,
    )


def _indexer(layer_id):
    """A QSAIndexer instance the worker's module scan can find."""
    stub = QSAIndexer.__new__(QSAIndexer)
    nn.Module.__init__(stub)
    stub.layer_id = layer_id
    return stub


class _Backend:
    """The selection hooks of a QSA sparse backend (no kernels, no CUDA)."""

    def __init__(self, name):
        self.name = name
        self._mtp_shared_sparse_indices = None

    def set_mtp_shared_sparse_indices(self, state) -> None:
        self._mtp_shared_sparse_indices = state


class _DraftWorker:
    """Enough EagleDraftWorker to run the real index-share configuration."""

    _qsa_index_share_unsupported_widths = getattr(
        EagleDraftWorker, "_qsa_index_share_unsupported_widths", ()
    )

    def _rebuild_topk1_chain_buffers(self) -> None:
        pass

    _configure_qsa_mtp_index_share = EagleDraftWorker._configure_qsa_mtp_index_share

    @staticmethod
    def _install_qsa_mtp_index_share(**kwargs) -> None:
        """Resolve the worker's installer at call time, so a tree without it
        fails the checks that need it instead of the whole file at import."""
        install = getattr(EagleDraftWorker, "_install_qsa_mtp_index_share", None)
        if install is None:
            raise AssertionError(
                "EagleDraftWorker._install_qsa_mtp_index_share is missing: no "
                "per-width selection is bound to this state's backends"
            )
        install(**kwargs)

    def __init__(self, steps, extend_backend=None):
        self._qsa_index_share_deferred = False
        self.topk = 1
        self.speculative_num_steps = steps
        self.speculative_num_draft_tokens = steps + 1
        self.device = "cpu"
        self.draft_attn_backend = _Backend(f"draft-decode-{steps}")
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None
        self.draft_extend_attn_backend = extend_backend or _Backend(
            f"draft-extend-{steps}"
        )
        self.draft_runner = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=_hf_config()),
            model=SimpleNamespace(
                modules=lambda: [_indexer(layer) for layer in LAYER_IDS]
            ),
            token_to_kv_pool=SimpleNamespace(
                qsa_token_topk=TOKEN_TOPK, qsa_compress_ratio=COMPRESS_RATIO
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.empty((NUM_REQUESTS, 8), dtype=torch.int32)
            ),
            device="cpu",
            draft_attn_backend=None,
            attn_backend=None,
        )


class _AdaptiveCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.adaptive(False)

    def adaptive(self, enabled):
        get_context().override(
            "test.adaptive_qsa_index_share", speculative_adaptive=bool(enabled)
        )


class TestPerWidthSelection(_AdaptiveCase):
    def test_each_adaptive_width_allocates_and_binds_its_own_state(self):
        self.adaptive(True)
        states = {}
        for steps in WIDTHS:
            worker = _DraftWorker(steps)
            worker._configure_qsa_mtp_index_share()
            decode_state = worker.draft_attn_backend._mtp_shared_sparse_indices
            extend_state = worker.draft_extend_attn_backend._mtp_shared_sparse_indices
            self.assertIsNotNone(
                decode_state,
                "adaptive widths run the indexer on every draft step again",
            )
            self.assertIs(decode_state, extend_state)
            self.assertEqual(decode_state.tail_width, steps + 1)
            self.assertEqual(
                decode_state.indices.shape,
                (len(LAYER_IDS), NUM_REQUESTS + 1, EXPANDED_TOPK + steps + 1),
            )
            states[steps] = decode_state
        self.assertEqual(
            len({id(state) for state in states.values()}),
            len(WIDTHS),
            "two widths share one selection buffer set",
        )

    def test_fixed_width_still_reuses_the_state_it_captured(self):
        self.adaptive(False)
        worker = _DraftWorker(15)
        worker._configure_qsa_mtp_index_share()
        first = worker.draft_extend_attn_backend._mtp_shared_sparse_indices
        self.assertIsNotNone(first)
        worker._configure_qsa_mtp_index_share()
        self.assertIs(
            worker.draft_extend_attn_backend._mtp_shared_sparse_indices,
            first,
            "the fixed-width path re-allocates its captured selection",
        )

    def test_candidate_does_not_inherit_the_launch_width_state(self):
        # A compressed-QSA draft gets the draft runner's OWN backend from the
        # factory, so the launch state and a candidate can reach the same
        # backend object; the candidate still has to bring its own buffers.
        self.adaptive(True)
        shared_extend = _Backend("runner-own-extend")
        launch = _DraftWorker(WIDTHS[-1], extend_backend=shared_extend)
        launch._configure_qsa_mtp_index_share()
        launch_state = shared_extend._mtp_shared_sparse_indices

        candidate = _DraftWorker(3, extend_backend=shared_extend)
        candidate._configure_qsa_mtp_index_share()
        candidate_state = candidate.draft_attn_backend._mtp_shared_sparse_indices
        self.assertIsNot(candidate_state, launch_state)
        self.assertEqual(candidate_state.tail_width, 4)
        self.assertEqual(launch_state.tail_width, WIDTHS[-1] + 1)


class TestCaptureBindingOrder(_AdaptiveCase):
    def test_factory_backend_stays_unbound_while_a_state_is_deferred(self):
        decode, extend = _Backend("decode"), _Backend("extend")

        class _Factory:
            def __init__(self, *args, **kwargs):
                pass

            def create_decode_backend(self):
                return decode

            def create_draft_extend_backend(self):
                return extend

        worker = _DraftWorker(7)
        worker.seed_dsa_topk_from_draft_extend = False
        worker.tree_mask_mode = None
        worker._qsa_index_share_deferred = True
        with patch(
            "sglang.srt.speculative.eagle_worker_v2.DraftBackendFactory", _Factory
        ), patch(
            "sglang.srt.speculative.eagle_worker_v2.default_tree_mask_mode",
            lambda: "default",
        ):
            EagleDraftWorker.init_attention_backend(worker)
            self.assertIsNone(
                decode._mtp_shared_sparse_indices,
                "the factory backend was bound while the build still owns the state",
            )
            self.assertIsNone(extend._mtp_shared_sparse_indices)
            # Whatever the twin decision left behind is what gets bound next.
            worker._qsa_index_share_deferred = False
            worker._configure_qsa_mtp_index_share()
            self.assertEqual(decode._mtp_shared_sparse_indices.tail_width, 8)
            self.assertIs(
                extend._mtp_shared_sparse_indices,
                decode._mtp_shared_sparse_indices,
            )
            self.assertEqual(worker.tree_mask_mode, "default")

    def test_build_binds_the_share_between_the_twin_and_the_capture(self):
        source = inspect.getsource(EAGLEWorkerV2.build_adaptive_runtime_state)
        twin = source.index("self._own_draft_extend_backend()")
        bind = source.index("self._draft_worker._configure_qsa_mtp_index_share()")
        capture = source.index("self._draft_worker._capture_cuda_graphs()")
        self.assertLess(twin, bind, "the share is bound before the twin may replace it")
        self.assertLess(
            bind, capture, "graphs capture a selection nothing bound to them"
        )

    def test_override_window_defers_and_restores_the_draft_worker(self):
        worker = SimpleNamespace(
            _draft_worker=_DraftWorker(7),
            speculative_num_steps=7,
            speculative_num_draft_tokens=8,
            _target_worker=SimpleNamespace(model_runner=SimpleNamespace()),
            _rebuild_topk1_chain_buffers=lambda: None,
        )
        with EAGLEWorkerV2._override_worker_state(worker, 3, 4):
            self.assertTrue(worker._draft_worker._qsa_index_share_deferred)
        self.assertFalse(worker._draft_worker._qsa_index_share_deferred)


class TestSelectionGeometry(_AdaptiveCase):
    def _state(self, steps):
        return QSAMTPSharedSparseIndices(
            layer_ids=LAYER_IDS,
            num_requests=NUM_REQUESTS,
            token_topk=EXPANDED_TOPK,
            tail_width=steps + 1,
            device="cpu",
        )

    def test_tail_columns_follow_the_width_that_captured(self):
        selection = torch.arange(EXPANDED_TOPK, dtype=torch.int32).unsqueeze(0) + 100
        for steps in WIDTHS:
            state = self._state(steps)
            captured_len = 41
            state.capture(
                selection,
                torch.tensor([2], dtype=torch.int64),
                torch.tensor([captured_len], dtype=torch.int32),
                LAYER_IDS[0],
            )
            row = state.lookup(
                torch.tensor([2], dtype=torch.int64),
                torch.tensor([captured_len + steps], dtype=torch.int32),
                LAYER_IDS[0],
            )
            self.assertEqual(row.shape, (1, EXPANDED_TOPK + steps + 1))
            self.assertEqual(row[0, :EXPANDED_TOPK].tolist(), selection[0].tolist())
            tail = row[0, EXPANDED_TOPK:].tolist()
            # The frozen selection plus exactly the drafted positions, the rest
            # of the width dropped instead of pointing at a token.
            self.assertEqual(tail, list(range(captured_len, captured_len + steps + 1)))

            short = state.lookup(
                torch.tensor([2], dtype=torch.int64),
                torch.tensor([captured_len + 1], dtype=torch.int32),
                LAYER_IDS[0],
            )
            self.assertEqual(
                short[0, EXPANDED_TOPK:].tolist(),
                [captured_len, captured_len + 1] + [-1] * (steps - 1),
            )

    def test_layer_slots_are_not_shared_across_widths(self):
        states = {steps: self._state(steps) for steps in WIDTHS}
        rows = {id(state) for state in states.values()}
        self.assertEqual(len(rows), len(WIDTHS))
        for steps, state in states.items():
            self.assertEqual(state.indices.shape[-1], EXPANDED_TOPK + steps + 1)
            self.assertEqual(state.trash_row, NUM_REQUESTS)


class _CacheHost:
    """Enough EAGLEWorkerV2 to run a real width transition on CPU tensors."""

    apply_runtime_state = EAGLEWorkerV2.apply_runtime_state

    def _hand_off_qsa_mtp_index_share(self, **kwargs):
        hand_off = getattr(EAGLEWorkerV2, "_hand_off_qsa_mtp_index_share", None)
        if hand_off is None:
            raise AssertionError(
                "EAGLEWorkerV2._hand_off_qsa_mtp_index_share is missing: a switch "
                "repoints backends without seeding the incoming selection"
            )
        return hand_off(**kwargs)

    def __init__(self, steps, state):
        self.state = state
        self.speculative_num_steps = steps
        self.speculative_num_draft_tokens = steps + 1
        self.device = "cpu"
        self._draft_worker = SimpleNamespace(
            speculative_num_steps=steps,
            speculative_num_draft_tokens=steps + 1,
            draft_attn_backend=state.draft_attn_backend,
            draft_extend_attn_backend=state.draft_extend_attn_backend,
            cuda_graph_runner=None,
            cuda_graph_runner_for_draft_extend=None,
            draft_runner=SimpleNamespace(
                attn_backend=state.draft_extend_attn_backend,
                draft_attn_backend=state.draft_attn_backend,
            ),
            _rebuild_topk1_chain_buffers=lambda: None,
        )
        self._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=state.target_attn_backend,
                decode_cuda_graph_runner=None,
            )
        )

    def switch(self, state):
        self.apply_runtime_state(state)
        self.state = state
        dw = self._draft_worker
        dw.speculative_num_steps = state.speculative_num_steps
        dw.draft_attn_backend = state.draft_attn_backend
        dw.draft_extend_attn_backend = state.draft_extend_attn_backend
        dw.draft_runner.attn_backend = state.draft_extend_attn_backend
        dw.draft_runner.draft_attn_backend = state.draft_attn_backend


def _cache(steps):
    return QSAMTPSharedSparseIndices(
        layer_ids=LAYER_IDS,
        num_requests=NUM_REQUESTS,
        token_topk=EXPANDED_TOPK,
        tail_width=steps + 1,
        device="cpu",
    )


def _width_state(steps, cache):
    from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState

    decode = _Backend(f"draft-decode-{steps}")
    decode.set_mtp_shared_sparse_indices(cache)
    extend = _Backend(f"draft-extend-{steps}")
    extend.set_mtp_shared_sparse_indices(cache)
    return SpecRuntimeState(
        speculative_num_steps=steps,
        speculative_num_draft_tokens=steps + 1,
        draft_attn_backend=decode,
        cuda_graph_runner=SimpleNamespace(name=f"draft-graph-{steps}"),
        target_attn_backend=SimpleNamespace(name=f"target-{steps}"),
        target_graph_runner=None,
        draft_extend_attn_backend=extend,
        cuda_graph_runner_for_draft_extend=SimpleNamespace(
            name=f"extend-graph-{steps}"
        ),
    )


def _capture(cache, row, selection, captured_len):
    cache.capture(
        selection,
        torch.tensor([row], dtype=torch.int64),
        torch.tensor([captured_len], dtype=torch.int32),
        LAYER_IDS[0],
    )


def _lookup(cache, row, position):
    return cache.lookup(
        torch.tensor([row], dtype=torch.int64),
        torch.tensor([position], dtype=torch.int32),
        LAYER_IDS[0],
    )


class TestSeedHandoffAcrossSwitches(_AdaptiveCase):
    """The live frozen selection has to reach the incoming width's cache.

    Each width owns its selection buffers (that is what the captured graphs bake),
    and a transition only repoints the backends. The controller activates a width
    at the NEXT actual forward, so the draft that runs there reads a cache its own
    draft-extend never wrote -- unless the authoritative rows are handed over in
    place first.
    """

    ROW = 2

    def test_prefill_seed_survives_the_switch_to_a_narrower_width(self):
        wide, narrow = _cache(15), _cache(3)
        wide_state, narrow_state = _width_state(15, wide), _width_state(3, narrow)
        worker = _CacheHost(15, wide_state)

        prefill = torch.arange(EXPANDED_TOPK, dtype=torch.int32).unsqueeze(0) + 100
        _capture(wide, self.ROW, prefill, 41)  # draft-extend at the launch width
        pointers = (narrow.indices.data_ptr(), narrow.captured_len.data_ptr())
        worker.switch(narrow_state)  # C2 selects W4 at the next forward
        # Graph-safe: the incoming buffers are still the ones its graphs baked.
        self.assertEqual(
            (narrow.indices.data_ptr(), narrow.captured_len.data_ptr()), pointers
        )

        row = _lookup(narrow, self.ROW, 43)[0]
        self.assertEqual(
            row[:EXPANDED_TOPK].tolist(),
            prefill[0].tolist(),
            "the first draft after a switch looked up a selection it never captured",
        )
        self.assertEqual(row[EXPANDED_TOPK:].tolist(), [41, 42, 43] + [-1])

    def test_return_to_a_wider_width_reads_the_live_rows_not_its_own_history(self):
        wide, narrow = _cache(15), _cache(3)
        wide_state, narrow_state = _width_state(15, wide), _width_state(3, narrow)
        worker = _CacheHost(15, wide_state)
        _capture(wide, self.ROW, torch.zeros((1, EXPANDED_TOPK), dtype=torch.int32), 41)

        worker.switch(narrow_state)
        live = torch.arange(1, EXPANDED_TOPK + 1, dtype=torch.int32).unsqueeze(0) * 3
        _capture(narrow, self.ROW, live, 45)  # the W4 draft-extend is authoritative
        worker.switch(wide_state)

        row = _lookup(wide, self.ROW, 55)[0]
        self.assertEqual(
            row[:EXPANDED_TOPK].tolist(),
            live[0].tolist(),
            "a returning width reused its own stale rows",
        )
        self.assertEqual(
            row[EXPANDED_TOPK:].tolist(),
            list(range(45, 56)) + [-1] * (16 - 11),
        )

    def test_a_reused_request_slot_does_not_keep_the_previous_request(self):
        wide, narrow = _cache(15), _cache(3)
        wide_state, narrow_state = _width_state(15, wide), _width_state(3, narrow)
        worker = _CacheHost(15, wide_state)
        _capture(wide, self.ROW, torch.zeros((1, EXPANDED_TOPK), dtype=torch.int32), 9)
        worker.switch(narrow_state)
        retired = _lookup(narrow, self.ROW, 9)[0][:EXPANDED_TOPK].tolist()
        self.assertEqual(retired, [0] * EXPANDED_TOPK)  # nothing to see yet

        fresh = torch.arange(EXPANDED_TOPK, dtype=torch.int32).unsqueeze(0) + 500
        _capture(narrow, self.ROW, fresh, 501)  # a new request, the same pool slot
        worker.switch(wide_state)
        self.assertEqual(
            _lookup(wide, self.ROW, 503)[0][:EXPANDED_TOPK].tolist(), fresh[0].tolist()
        )

    def test_handoff_copies_the_shared_part_and_never_the_tails(self):
        wide, narrow = _cache(15), _cache(3)
        selection = torch.arange(EXPANDED_TOPK, dtype=torch.int32).unsqueeze(0) + 20
        _capture(narrow, 1, selection, 33)
        junk = wide.indices.clone()
        _hand_off(narrow, wide)
        self.assertEqual(
            wide.indices[0, 1, :EXPANDED_TOPK].tolist(), selection[0].tolist()
        )
        self.assertEqual(int(wide.captured_len[0, 1]), 33)
        # Differently sized tail columns stay as this width left them.
        self.assertEqual(
            wide.indices[0, 1, EXPANDED_TOPK:].tolist(),
            junk[0, 1, EXPANDED_TOPK:].tolist(),
        )
        self.assertEqual(wide.tail_width, 16)
        self.assertEqual(narrow.tail_width, 4)


class TestLowWidthTablesStayUnshared(_AdaptiveCase):
    """A table that can reach steps <= 1 keeps the path it had before.

    Such a width has no draft decode step and its draft-extend never seeds a
    shared cache, so per-width caches would straddle two regimes across a switch
    (an A3 -> 0/1 -> 3 transition reading an old wide cache or another request's
    row). The whole table therefore stays unshared -- eligibility is a property of
    the configured candidate set, not of whichever width happens to be live.
    """

    GENERIC_TABLE = (0, 1, 3)

    def test_eligibility_comes_from_the_whole_table(self):
        unsupported = _unsupported
        self.assertEqual(unsupported(WIDTHS), ())
        self.assertEqual(unsupported(self.GENERIC_TABLE), (0, 1))
        self.assertEqual(unsupported([1]), (1,))
        self.assertEqual(unsupported([2, 7]), ())
        # The shipped default table really does reach those widths, so this is
        # generic behaviour being preserved and not a hypothetical.
        default_union = sorted(
            {
                s
                for entry in DEFAULT_ADAPTIVE_CONFIG.values()
                for s in entry["candidate_steps"]
            }
        )
        self.assertEqual(default_union[:2], [0, 1])
        self.assertEqual(unsupported(default_union), (0, 1))

    def test_low_width_table_shares_nothing_at_any_configured_width(self):
        self.adaptive(True)
        low = _unsupported(self.GENERIC_TABLE)
        for steps in self.GENERIC_TABLE:
            worker = _DraftWorker(steps)
            worker._qsa_index_share_unsupported_widths = low
            worker._configure_qsa_mtp_index_share()
            self.assertIsNone(
                worker.draft_attn_backend._mtp_shared_sparse_indices,
                f"steps={steps} of a 0/1-capable table took the shared path",
            )
            self.assertIsNone(
                worker.draft_extend_attn_backend._mtp_shared_sparse_indices
            )
        # A transition inside such a table hands off nothing and cannot invent a
        # cache for the incoming width.
        out_state = _width_state(3, None)
        in_state = _width_state(1, None)
        EAGLEWorkerV2._hand_off_qsa_mtp_index_share(
            outgoing=(out_state.draft_extend_attn_backend,),
            incoming=(in_state.draft_attn_backend,),
        )
        self.assertIsNone(in_state.draft_attn_backend._mtp_shared_sparse_indices)
        # The very same widths do share once no configured width is at/below 1.
        worker = _DraftWorker(3)
        worker._configure_qsa_mtp_index_share()
        self.assertIsNotNone(worker.draft_attn_backend._mtp_shared_sparse_indices)

    def test_fixed_width_launch_never_consults_the_table(self):
        self.adaptive(False)
        worker = _DraftWorker(3)
        worker._qsa_index_share_unsupported_widths = (0, 1)
        worker._configure_qsa_mtp_index_share()
        self.assertIsNotNone(
            worker.draft_attn_backend._mtp_shared_sparse_indices,
            "the adaptive opt-out leaked into a fixed-width launch",
        )


def _unsupported(candidate_steps):
    fn = getattr(EAGLEWorkerV2, "_qsa_index_share_unsupported", None)
    if fn is None:
        raise AssertionError(
            "EAGLEWorkerV2._qsa_index_share_unsupported is missing: an adaptive "
            "table with a 0/1-step candidate would straddle both regimes"
        )
    return fn(candidate_steps)


def _hand_off(source, target):
    handoff = getattr(target, "handoff_from", None)
    if handoff is None:
        raise AssertionError(
            "QSAMTPSharedSparseIndices.handoff_from is missing: the incoming "
            "width keeps a selection its own draft-extend never captured"
        )
    handoff(source)


if __name__ == "__main__":
    unittest.main()
