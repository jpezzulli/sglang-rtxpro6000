"""Per-width ownership of adaptive-MTP (W4/W8) runtime state, CPU-only.

Adaptive speculative decoding keeps one SpecRuntimeState per candidate step
count -- W4/W8 are 3/7 steps at topk=1 -- and each of those widths
captures its own CUDA graphs. So the mutable resources a width bakes in (its
attention backends and their graph metadata, the draft kernel workspace, its
static input buffers, its GDN recovery graphs) must belong to exactly one
state, while the large allocations (model weights, the KV / Mamba / QSA-ring
pools) stay shared. These checks pin the three mechanics that keep both rules
true across a transition:

* no candidate may be wider than the launch width the start-up buffers were
  sized for, and no selectable width may exceed the fixed launch-maximum
  capacity (the QSA pending index-key ring) -- refuse it, never resize what
  captured graphs point at;
* the state under construction gets private copies of the mutable capture
  resources it would otherwise alias onto the live state, and the draft-extend
  twin it builds is limited to the one backend the factory really shares (a
  compressed-QSA draft): the generic families must keep the backend
  DraftBackendFactory chose for them;
* a switch drains the outgoing backends' in-flight side-stream SSM recovery
  before anything is repointed.

All evidence is CPU-only (no CUDA on this host): real graph capture, replay and
numerics remain GPU-window work.
"""

import contextlib
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.model_executor.input_buffers import (
    set_private_input_buffers,
    share_input_buffer,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.draft_utils import (
    DraftBackendFactory,
    draft_extend_backend_is_runner_own,
)
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# The candidate table under review: C1 = W4/W8 (3/7 speculative steps), the
# bounded launch profile this series is qualified on (no W16 capture).
CANDIDATE_STEPS = [3, 7]
COMPRESS_RATIO = 4


def _hf_config(variant):
    """A model_config.hf_config stand-in qsa.config.parse_qsa_profile understands.

    These are the three shapes DraftBackendFactory branches on: the Qwen4-Exp
    block-compressed indexer, the qsa_0511 tokenwise indexer and no indexer at
    all (an ordinary draft model).
    """
    if variant == "compressed":
        return SimpleNamespace(
            model_type="qwen4_exp",
            indexer_n_heads=64,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=COMPRESS_RATIO * 512,  # 512 blocks: a supported width
            indexer_compress_ratio=COMPRESS_RATIO,
        )
    if variant == "tokenwise":
        return SimpleNamespace(
            model_type="qwen3_5",
            index_topk=2048,
            index_n_heads=64,
            index_kv_heads=1,
            index_head_dim=128,
        )
    return SimpleNamespace(model_type="deepseek_v3")


def _state(steps, **overrides):
    """A runtime state whose every resource is an identifiable sentinel."""
    state = SpecRuntimeState(
        speculative_num_steps=steps,
        speculative_num_draft_tokens=steps + 1,
        draft_attn_backend=SimpleNamespace(name=f"draft-{steps}"),
        cuda_graph_runner=SimpleNamespace(name=f"draft-graph-{steps}"),
        target_attn_backend=SimpleNamespace(name=f"target-{steps}"),
        target_graph_runner=SimpleNamespace(name=f"target-graph-{steps}"),
        draft_extend_attn_backend=SimpleNamespace(name=f"extend-{steps}"),
        cuda_graph_runner_for_draft_extend=SimpleNamespace(
            name=f"extend-graph-{steps}"
        ),
    )
    for name, value in overrides.items():
        setattr(state, name, value)
    return state


class _ModelRunnerStub(SimpleNamespace):
    """Target runner stub that records when its live backend is repointed."""

    def __init__(self, events, **kwargs):
        self._events = events
        self._backend_name = "none"
        self._attn_backend = None
        super().__init__(**kwargs)

    @property
    def attn_backend(self):
        return self._attn_backend

    @attn_backend.setter
    def attn_backend(self, value):
        self._events.append(f"repoint-target:{self._backend_name}")
        self._attn_backend = value
        self._backend_name = getattr(value, "name", "?")


def _ownership_check(name):
    """Bind a worker method that the state-ownership fix is expected to provide.

    Resolved at call time so a tree without it fails THIS check with a named
    reason, instead of failing the whole file at import.
    """

    def _bind(self, *args, **kwargs):
        fn = getattr(EAGLEWorkerV2, name, None)
        if fn is None:
            self.fail(f"EAGLEWorkerV2.{name} is missing: width state is not owned")
        return fn(self, *args, **kwargs)

    return _bind


class _WorkerStub:
    """Enough EAGLEWorkerV2 to drive the width state machine without a GPU."""

    apply_runtime_state = EAGLEWorkerV2.apply_runtime_state
    build_adaptive_runtime_state = EAGLEWorkerV2.build_adaptive_runtime_state
    _override_worker_state = EAGLEWorkerV2._override_worker_state
    _own_draft_extend_backend = _ownership_check("_own_draft_extend_backend")
    _private_capture_scope = _ownership_check("_private_capture_scope")
    _validate_adaptive_widths = _ownership_check("_validate_adaptive_widths")

    def __init__(self, launch_steps=CANDIDATE_STEPS[-1], events=None, qsa_variant=None):
        self.events = [] if events is None else events
        self.speculative_num_steps = launch_steps
        self.speculative_num_draft_tokens = launch_steps + 1
        self._additional_graph_memory_usage = {}
        self._additional_graph_time_usage = {}
        self.device = "cpu"
        self.gpu_id = 0
        self.qsa_variant = qsa_variant
        self.shared_pool = SimpleNamespace(name="shared-pools")
        self.captured_widths = {}  # mirrors _capture_cuda_graphs' binding
        self.captured_aliases = {}  # width -> draft-extend backend IS runner's
        self.factory_backends = {}  # width -> object the fake factory returned
        self.twin_requests = []  # draft-extend twins the worker asked for
        self.draft_runner = SimpleNamespace(
            attn_backend=SimpleNamespace(
                name="runner-own-backend", pool=self.shared_pool
            ),
            draft_attn_backend=None,
            init_new_workspace=False,
            token_to_kv_pool=self.shared_pool,
            model_config=SimpleNamespace(hf_config=_hf_config(qsa_variant)),
            draft_attention_backend=None,
            _get_attention_backend=self._draft_twin,
        )
        self._draft_worker = SimpleNamespace(
            speculative_num_steps=launch_steps,
            speculative_num_draft_tokens=launch_steps + 1,
            draft_attn_backend=None,
            draft_extend_attn_backend=None,
            cuda_graph_runner=None,
            cuda_graph_runner_for_draft_extend=None,
            draft_runner=self.draft_runner,
            init_attention_backend=self._init_attention_backend,
            _capture_cuda_graphs=self._capture_cuda_graphs,
            _rebuild_topk1_chain_buffers=lambda: None,
        )
        # The launch state, as init_attention_backend left it at startup: the
        # runner's own object for a compressed-QSA draft, None for tokenwise QSA
        # (the eager draft-extend path), and the factory's own object otherwise --
        # which is also what the runner's attn_backend now is.
        if qsa_variant == "tokenwise":
            launch_extend = None
        elif qsa_variant == "compressed":
            launch_extend = self.draft_runner.attn_backend
        else:
            launch_extend = SimpleNamespace(
                name="factory-draft-extend", pool=self.shared_pool
            )
            self.draft_runner.attn_backend = launch_extend
        self._draft_worker.draft_extend_attn_backend = launch_extend
        self.launch_extend = launch_extend
        self._target_worker = SimpleNamespace(
            model_runner=_ModelRunnerStub(
                self.events,
                init_new_workspace=False,
                decode_cuda_graph_runner=None,
                token_to_kv_pool=self.shared_pool,
                _get_attention_backend=lambda init_new_workspace=False: (
                    SimpleNamespace(name="target-backend")
                ),
                maybe_capture_gdn_recovery_graphs=lambda **kwargs: None,
            )
        )
        self.adaptive_controller = None

    def _draft_twin(self, init_new_workspace=False):
        """ModelRunner._get_attention_backend on the draft runner (the twin path)."""
        self.twin_requests.append(init_new_workspace)
        return SimpleNamespace(name="draft-extend-twin", pool=self.shared_pool)

    def _init_attention_backend(self):
        """Mirror of EagleDraftWorker.init_attention_backend's assignments.

        DraftBackendFactory returns a fresh backend for the generic families,
        the draft runner's OWN object for a compressed-QSA draft
        (draft_utils.create_draft_extend_backend) and None for tokenwise QSA;
        the runner's attn_backend is then repointed at whatever came back, so
        the two are identical for every non-null backend.
        """
        dw = self._draft_worker
        dw.draft_attn_backend = SimpleNamespace(name="draft-decode")
        dw.draft_extend_attn_backend = None
        if self.qsa_variant == "compressed":
            backend = self.draft_runner.attn_backend
        elif self.qsa_variant == "tokenwise":
            backend = None
        else:
            backend = SimpleNamespace(
                name=f"factory-draft-extend-{self.speculative_num_steps}",
                pool=self.shared_pool,
            )
        dw.draft_extend_attn_backend = backend
        if backend is not None:
            self.draft_runner.attn_backend = backend
        self.factory_backends[self.speculative_num_steps] = backend

    def _capture_cuda_graphs(self):
        # The extend graph runner binds whatever the worker attribute holds at
        # capture time; record it, and whether it IS the runner's own backend.
        dw = self._draft_worker
        width = self.speculative_num_steps
        self.captured_widths[width] = dw.draft_extend_attn_backend
        self.captured_aliases[width] = (
            dw.draft_extend_attn_backend is self.draft_runner.attn_backend
        )


class PublishedConfigCase(CustomTestCase):
    """Base: the adaptive state machine writes its live spec switches through
    ``get_context().override``, which needs a published config."""

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)


class TestLaunchCapacity(PublishedConfigCase):
    def setUp(self):
        super().setUp()
        self._dir = tempfile.mkdtemp(prefix="adaptive-cfg-")

    def tearDown(self):
        import shutil

        shutil.rmtree(self._dir, ignore_errors=True)

    def _controller(self, launch_steps, candidates):
        path = os.path.join(self._dir, "candidates.json")
        with open(path, "w") as f:
            json.dump({"1": {"candidate_steps": list(candidates)}}, f)
        worker = SimpleNamespace(
            speculative_num_steps=launch_steps,
            build_adaptive_runtime_state=lambda **kwargs: _state(
                kwargs["speculative_num_steps"]
            ),
            apply_runtime_state=lambda state: None,
        )
        return AdaptiveController(worker, config_path=path)

    def test_candidate_wider_than_the_launch_width_is_refused(self):
        # Launching at W4 but keeping the W8 candidate would run an 8-token
        # verify window against start-up buffers sized for 4; growing them
        # instead would free what the launch state's graphs already point at.
        controller = self._controller(3, CANDIDATE_STEPS)
        with self.assertRaises(ValueError) as ctx:
            controller.init_states()
        message = str(ctx.exception)
        self.assertIn("7", message)
        self.assertIn("CUDA graphs", message)

    def test_launch_at_the_widest_candidate_accepts_the_table(self):
        controller = self._controller(CANDIDATE_STEPS[-1], CANDIDATE_STEPS)
        controller.init_states()
        self.assertEqual(sorted(controller._states), CANDIDATE_STEPS)

    def _ring(self, *, max_draft_tokens):
        """A QSA pool whose ring was sized off the resolved launch maximum."""
        from sglang.srt.layers.attention.qsa.metadata import (
            pending_ring_groups_required,
        )

        groups = pending_ring_groups_required(
            draft_tokens=max_draft_tokens, compress_ratio=COMPRESS_RATIO
        )
        return SimpleNamespace(
            qsa_num_groups=groups,
            qsa_compress_ratio=COMPRESS_RATIO,
            qsa_ring_span=COMPRESS_RATIO * groups,
        )

    def _worker_with_rings(self, target_ring, draft_ring=None):
        worker = _WorkerStub(launch_steps=CANDIDATE_STEPS[-1])
        worker.adaptive_controller = SimpleNamespace(candidate_steps=CANDIDATE_STEPS)
        worker._target_worker.model_runner.token_to_kv_pool = target_ring
        worker._draft_worker.draft_runner.token_to_kv_pool = (
            target_ring if draft_ring is None else draft_ring
        )
        return worker

    def test_launch_capacity_serves_every_narrower_width(self):
        # A W8 launch allocates 3 groups at ratio 4; W8 and W4 both fit, so an
        # adaptive step-down reuses the allocation its graphs bind.
        worker = self._worker_with_rings(self._ring(max_draft_tokens=8))
        worker._validate_adaptive_widths()

    def test_width_above_the_ring_capacity_is_refused_at_start_up(self):
        # Same table, but the ring was sized for W4: the 8-token window would
        # alias two live verify positions onto one ring slot. Refuse rather
        # than resize a buffer the captured graphs already point at.
        worker = self._worker_with_rings(self._ring(max_draft_tokens=4))
        with self.assertRaises(ValueError) as ctx:
            worker._validate_adaptive_widths()
        message = str(ctx.exception)
        self.assertIn("target", message)
        self.assertIn("verify window", message)
        self.assertIn("holds 1 group", message)

    def test_draft_ring_counts_even_when_the_target_ring_is_wide_enough(self):
        worker = self._worker_with_rings(
            self._ring(max_draft_tokens=8), self._ring(max_draft_tokens=4)
        )
        with self.assertRaises(ValueError) as ctx:
            worker._validate_adaptive_widths()
        self.assertIn("draft", str(ctx.exception))

    def test_pool_without_a_pending_ring_is_not_measured(self):
        # Tokenwise QSA / non-QSA pools carry no pending ring at all; the
        # historical single-group answer is right for them and nothing raises.
        worker = self._worker_with_rings(SimpleNamespace())
        worker._validate_adaptive_widths()

    def test_no_controller_means_no_adaptive_width_to_check(self):
        # Fixed W4, and every non-adaptive path, never reaches this.
        _WorkerStub(launch_steps=3)._validate_adaptive_widths()


class TestWidthTransitions(PublishedConfigCase):
    def test_w8_to_w4_to_w8_swaps_only_the_per_width_resources(self):
        # Launch at W4 so every application below is a real switch; the
        # transition under test is the shipped W8 -> W4 -> W8 sequence
        # (re-applying the active width is a no-op, checked in the suite below).
        worker = _WorkerStub(launch_steps=3)
        wide, narrow = _state(7), _state(3)

        worker.apply_runtime_state(wide)
        self.assertEqual(worker.speculative_num_draft_tokens, 8)
        self.assertIs(
            worker.draft_runner.attn_backend, wide.draft_extend_attn_backend
        )

        worker.apply_runtime_state(narrow)
        self.assertEqual(worker.speculative_num_steps, 3)
        self.assertIs(
            worker.draft_runner.attn_backend, narrow.draft_extend_attn_backend
        )
        # The narrow width inherits none of the wide one's captured resources.
        self.assertIsNot(
            worker._draft_worker.cuda_graph_runner_for_draft_extend,
            wide.cuda_graph_runner_for_draft_extend,
        )
        self.assertIsNot(
            worker._target_worker.model_runner.decode_cuda_graph_runner,
            wide.target_graph_runner,
        )
        # ... and it duplicated no large allocation: every width addresses the
        # one KV / Mamba / QSA-ring pool.
        self.assertIs(
            worker._target_worker.model_runner.token_to_kv_pool, worker.shared_pool
        )
        self.assertIs(
            worker._draft_worker.draft_runner.token_to_kv_pool, worker.shared_pool
        )

        worker.apply_runtime_state(wide)
        self.assertEqual(worker.speculative_num_steps, 7)
        self.assertIs(
            worker._draft_worker.cuda_graph_runner, wide.cuda_graph_runner
        )
        self.assertIs(
            worker._draft_worker.cuda_graph_runner_for_draft_extend,
            wide.cuda_graph_runner_for_draft_extend,
        )
        self.assertIs(
            worker._target_worker.model_runner.decode_cuda_graph_runner,
            wide.target_graph_runner,
        )
        self.assertIs(
            worker.draft_runner.attn_backend, wide.draft_extend_attn_backend
        )

    def test_a_step_with_no_draft_extend_backend_keeps_the_initialized_one(self):
        # Tokenwise QSA runs draft-extend eagerly; a width switch with a None
        # backend must leave the runner's own backend in place rather than
        # leaking another width's onto it.
        worker = _WorkerStub(launch_steps=CANDIDATE_STEPS[-1])
        initialized = SimpleNamespace(name="runner-initialized-backend")
        worker.draft_runner.attn_backend = initialized
        for steps in (7, 3, 7):
            worker.apply_runtime_state(_state(steps, draft_extend_attn_backend=None))
            self.assertIs(worker.draft_runner.attn_backend, initialized)

    def test_the_outgoing_backends_are_drained_before_anything_is_repointed(self):
        def recovery_backend(tag):
            backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
            backend.name = tag
            backend._recovery_event_pending = True
            backend._recovery_event = SimpleNamespace(
                wait=lambda: worker.events.append(f"drain:{tag}")
            )
            return backend

        worker = _WorkerStub(launch_steps=CANDIDATE_STEPS[-1])
        model_runner = worker._target_worker.model_runner
        model_runner.attn_backend = recovery_backend("target")
        shared_extend = recovery_backend("extend")
        worker.draft_runner.attn_backend = shared_extend
        worker._draft_worker.draft_extend_attn_backend = shared_extend
        worker._draft_worker.draft_attn_backend = recovery_backend("draft")
        worker.events.clear()  # drop the setup repoint

        worker.apply_runtime_state(_state(3))
        # Re-selecting the active width must not drain, repoint or double-count.
        worker.apply_runtime_state(_state(3))

        self.assertEqual(worker.events[0], "drain:target")
        self.assertEqual(
            set(worker.events[:3]), {"drain:target", "drain:extend", "drain:draft"}
        )
        self.assertEqual(worker.events.count("drain:extend"), 1, "deduped by identity")
        # Every drain precedes the repoint that retires the backend, so the
        # incoming width cannot race the outgoing one's side-stream recovery.
        self.assertEqual(worker.events[-1], "repoint-target:target")
        self.assertTrue(all(e.startswith("drain") for e in worker.events[:-1]))

    def test_backends_without_recovery_bookkeeping_do_not_disturb_a_switch(self):
        worker = _WorkerStub(launch_steps=CANDIDATE_STEPS[-1])
        worker.apply_runtime_state(_state(3))
        self.assertEqual(worker.speculative_num_steps, 3)
        self.assertEqual(worker.events, ["repoint-target:none"])


class TestPerStateConstruction(PublishedConfigCase):
    """build_adaptive_runtime_state must not alias the live state's resources."""

    def _build(self, worker, steps):
        model_runner = worker._target_worker.model_runner
        recovery_calls = []
        model_runner.maybe_capture_gdn_recovery_graphs = (
            lambda **kwargs: recovery_calls.append(kwargs)
        )
        model_runner._get_attention_backend = lambda init_new_workspace=False: (
            SimpleNamespace(name=f"target-{steps}", workspace_mine=True)
        )
        with (
            patch(
                "sglang.srt.speculative.eagle_worker_v2.check_cuda_graph_backend",
                return_value=False,
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.DecodeCudaGraphRunner",
                lambda *args, **kwargs: SimpleNamespace(capture_bs=[1, 8, 64]),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_available_gpu_memory",
                return_value=0.0,
            ),
        ):
            state = worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
            )
        return state, recovery_calls

    def test_recovery_graphs_are_captured_for_the_states_own_backend(self):
        worker = _WorkerStub(launch_steps=CANDIDATE_STEPS[-1])
        wide, wide_calls = self._build(worker, 7)
        narrow, narrow_calls = self._build(worker, 3)

        for calls, state in ((wide_calls, wide), (narrow_calls, narrow)):
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0]["attn_backend"], state.target_attn_backend)
            self.assertEqual(calls[0]["capture_bs"], [1, 8, 64])
        # Two widths, two target backends: the recovery graph sets cannot alias
        # each other's address-stable index buffers.
        self.assertIsNot(wide.target_attn_backend, narrow.target_attn_backend)

    def test_compressed_qsa_state_gets_its_own_draft_extend_backend(self):
        worker = _WorkerStub(
            launch_steps=CANDIDATE_STEPS[-1], qsa_variant="compressed"
        )
        launch_backend = worker.draft_runner.attn_backend

        wide, _ = self._build(worker, 7)
        narrow, _ = self._build(worker, 3)

        # Each width captured its draft-extend graph against its own backend...
        self.assertEqual(sorted(worker.captured_widths), [3, 7])
        self.assertIsNot(worker.captured_widths[7], worker.captured_widths[3])
        for state in (wide, narrow):
            self.assertIsNot(state.draft_extend_attn_backend, launch_backend)
        # ... because the worker asked the draft runner for a twin, against a
        # kernel workspace of its own, with the runner's flag restored afterwards
        # (backends built through the attention registry read that flag off the
        # runner).
        self.assertEqual(worker.twin_requests, [True, True])
        self.assertFalse(worker.draft_runner.init_new_workspace)
        # ... and the twin still addresses the one shared pool: no large cache
        # allocation is duplicated per width.
        self.assertIs(wide.draft_extend_attn_backend.pool, worker.shared_pool)
        # The launch state's own backend survived every build untouched.
        self.assertIs(worker.draft_runner.attn_backend, launch_backend)
        self.assertIs(
            worker._draft_worker.draft_extend_attn_backend, launch_backend
        )

    def test_a_non_qsa_draft_keeps_the_backend_the_factory_chose(self):
        # The generic families receive a fresh object from DraftBackendFactory per
        # state, and init_attention_backend then repoints the runner's own
        # attn_backend at it -- so at capture time the two identities are EQUAL
        # and identity cannot tell this case from compressed QSA. Rebuilding here
        # would swap a deliberate factory choice (cutedsl_mla's TRTLLM-gen
        # draft-extend fallback, a Blackwell hybrid_linear_attn draft's plain
        # Triton, and so on) for the runner's target-style backend.
        worker = _WorkerStub(launch_steps=CANDIDATE_STEPS[-1], qsa_variant=None)

        wide, _ = self._build(worker, 7)
        narrow, _ = self._build(worker, 3)

        self.assertEqual(worker.twin_requests, [])
        self.assertTrue(worker.captured_aliases[7], "identity was equal here")
        self.assertTrue(worker.captured_aliases[3])
        self.assertIs(wide.draft_extend_attn_backend, worker.factory_backends[7])
        self.assertIs(narrow.draft_extend_attn_backend, worker.factory_backends[3])
        self.assertIsNot(
            wide.draft_extend_attn_backend, narrow.draft_extend_attn_backend
        )

    def test_the_tokenwise_eager_draft_extend_path_builds_no_backend(self):
        # Tokenwise QSA has no graph-stable draft-extend metadata: the factory
        # returns None and that width runs draft-extend eagerly. A switch must
        # neither invent a backend for it nor fall back to a dense one.
        worker = _WorkerStub(
            launch_steps=CANDIDATE_STEPS[-1], qsa_variant="tokenwise"
        )
        runner_backend = worker.draft_runner.attn_backend

        wide, _ = self._build(worker, 7)

        self.assertIsNone(wide.draft_extend_attn_backend)
        self.assertEqual(worker.twin_requests, [])
        self.assertIs(worker.draft_runner.attn_backend, runner_backend)


class TestOwnershipTrigger(PublishedConfigCase):
    """The per-width ownership trigger is the factory's own QSA branch; an
    identity test cannot be it, because init_attention_backend assigns every
    non-null factory-produced draft-extend backend onto draft_runner.attn_backend.
    """

    def _factory(self, variant):
        runner = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=_hf_config(variant)),
            draft_attention_backend=None,
            attn_backend=SimpleNamespace(name="runner-own-backend"),
        )
        return DraftBackendFactory(runner, topk=1, speculative_num_steps=7), runner

    def test_only_the_compressed_qsa_draft_takes_the_runners_backend(self):
        factory, runner = self._factory("compressed")
        self.assertTrue(draft_extend_backend_is_runner_own(runner))
        self.assertIs(factory.create_draft_extend_backend(), runner.attn_backend)

    def test_tokenwise_qsa_returns_no_backend_and_owns_nothing(self):
        factory, runner = self._factory("tokenwise")
        self.assertFalse(draft_extend_backend_is_runner_own(runner))
        self.assertIsNone(factory.create_draft_extend_backend())

    def test_a_draft_without_a_qsa_indexer_is_not_runner_own(self):
        # parse_qsa_profile sees no indexer at all here, so the predicate stays
        # False no matter how the backend the generic map returns is dressed up.
        _, runner = self._factory(None)
        self.assertFalse(draft_extend_backend_is_runner_own(runner))



class TestPrivateInputBuffers(PublishedConfigCase):
    def test_capture_of_a_second_state_does_not_alias_the_firsts_buffers(self):
        worker = _WorkerStub()
        base = torch.zeros((8,), dtype=torch.int64)
        share_input_buffer("adaptive_probe_positions", base)

        with worker._private_capture_scope():
            self.assertTrue(
                worker.draft_runner.init_new_workspace,
                "the draft backends built inside get their own workspace",
            )
            own = torch.ones((8,), dtype=torch.int64)
            handed = share_input_buffer("adaptive_probe_positions", own)
            self.assertIs(handed, own, "private storage, not the pooled buffer")

        self.assertFalse(worker.draft_runner.init_new_workspace)
        # Outside the build window the pool coalesces again, unchanged for every
        # non-adaptive caller.
        later = torch.zeros((8,), dtype=torch.int64)
        self.assertEqual(
            share_input_buffer("adaptive_probe_positions", later).data_ptr(),
            base.data_ptr(),
        )

    def test_the_scope_is_restored_even_when_the_capture_raises(self):
        worker = _WorkerStub()
        with contextlib.suppress(RuntimeError):
            with worker._private_capture_scope():
                self.assertTrue(set_private_input_buffers(False))
                raise RuntimeError("capture failed")
        self.assertFalse(set_private_input_buffers(False))
        self.assertFalse(worker.draft_runner.init_new_workspace)


if __name__ == "__main__":
    unittest.main()
