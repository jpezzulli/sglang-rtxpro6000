import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sglang.srt.speculative.adaptive_spec_params import AdaptiveSpeculativeParams

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
    from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
        EAGLEDraftExtendCudaGraphRunner,
    )


# Ported from https://github.com/aiueo52/sglang-rtxpro6000 branch
# flash-next-fast (snapshot 5105985), where the extra adaptive target graphs
# were captured untuned. Opt in with SGLANG_ADAPTIVE_TARGET_AUTOTUNE=1.
@contextmanager
def adaptive_target_graph_warmup(model_runner, attn_backend):
    """Opt in to target autotuning before each additional state's capture.

    BaseRunner.warmup normally runs once per model. Additional adaptive
    states share that model but introduce a new verify width after the draft
    autotuner has run. Skipping their target warmup can capture untuned MoE
    tactics (including a separate finalize kernel) for the state's lifetime.

    Reuse the normal warmup and its disable/determinism gates. The dummy
    forward reads model_runner.attn_backend, so temporarily install the
    candidate's backend as well; never warm up against the base state's
    graph metadata. This context is only used during state construction, and
    both attributes go back exactly as they were -- including a model runner
    that never had _kernel_warmed_up at all -- so a failed capture cannot
    leave the live state warm against a dead backend.
    """
    if os.environ.get("SGLANG_ADAPTIVE_TARGET_AUTOTUNE", "") != "1":
        yield
        return
    missing = object()
    warmed = getattr(model_runner, "_kernel_warmed_up", missing)
    backend = model_runner.attn_backend
    model_runner._kernel_warmed_up = False
    model_runner.attn_backend = attn_backend
    try:
        yield
    finally:
        model_runner.attn_backend = backend
        if warmed is missing:
            del model_runner._kernel_warmed_up
        else:
            model_runner._kernel_warmed_up = warmed


@dataclass
class SpecRuntimeState:
    """A complete set of runtime resources bound to a specific speculative
    decoding configuration.

    Each decode round runs three stages — draft, verify, extend — and every
    stage has shape-dependent resources (attention backends and CUDA graphs)
    that must match the current configuration.  Switching adaptive steps
    means swapping the entire state atomically.
    """

    # -- Configuration (determines shapes for all stages) --
    speculative_num_steps: int
    speculative_num_draft_tokens: int

    # -- Draft stage: draft model multi-step autoregressive generation --
    draft_attn_backend: "AttentionBackend | None"
    cuda_graph_runner: "EAGLEDraftCudaGraphRunner | None"

    # -- Verify stage: target model one-pass tree verification --
    target_attn_backend: "AttentionBackend"
    target_graph_runner: "DecodeCudaGraphRunner | CPUGraphRunner | None"

    # -- Extend stage: draft model KV cache catch-up after verify --
    draft_extend_attn_backend: "AttentionBackend | None"
    cuda_graph_runner_for_draft_extend: "EAGLEDraftExtendCudaGraphRunner | None"


class AdaptiveSpecWorker(Protocol):
    """Protocol that a worker must implement to use AdaptiveController."""

    speculative_num_steps: int

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ) -> SpecRuntimeState: ...

    def apply_runtime_state(self, state: SpecRuntimeState) -> None: ...


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements AdaptiveSpecWorker protocol:
      - build_adaptive_runtime_state(steps, draft_tokens) → runtime state
      - apply_runtime_state(state) → apply it to the worker

    The worker only needs to:
      1. Call register() for the initial state, then init_states()
         once during startup.
      2. Call on_verify_complete(results, batch_size) after each decode verify,
         to feed the policy of that batch's size.
      3. Call activate_step_by_batch(batch_size) before each decode forward --
         that is the only place the active width changes at run time.
    """

    def __init__(self, worker: AdaptiveSpecWorker, config_path: str | None = None):
        self.worker = worker
        # The width the server launched with. Everything the start-up path sizes
        # off the flat speculative-num-steps / num-draft-tokens leaf -- the
        # draft-extend graph metadata of a backend shared between states, the
        # QSA MTP shared-selection tail width -- is dimensioned for THIS width,
        # so no candidate may be wider; see validate_candidates().
        self.initial_steps = worker.speculative_num_steps
        self.params = AdaptiveSpeculativeParams(
            initial_steps=worker.speculative_num_steps,
            cfg_path=config_path,
        )
        self._states: dict[int, SpecRuntimeState] = {}

    @property
    def candidate_steps(self) -> list[int]:
        return self.params.candidate_steps

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """Register a pre-built runtime state.

        *steps* defaults to state.speculative_num_steps when not given.
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def init_states(self, cuda_graph_bs: list[int] | None = None) -> None:
        """Build and register runtime states for all candidate steps."""
        self.validate_candidates()
        self.params.set_cuda_graph_bs(cuda_graph_bs)

        for steps in self.candidate_steps:
            if steps in self._states:
                continue

            pruned_bs = self.params.cuda_graph_bs_for_step(steps)
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                cuda_graph_bs=pruned_bs,
            )
            self._states[steps] = state

        # Start on the initial step.
        self._activate(self.worker.speculative_num_steps)

    def validate_candidates(self) -> None:
        """Refuse a candidate wider than the launch width before capturing graphs.

        ``--speculative-num-steps`` is what the start-up path sizes against
        (``speculative_num_draft_tokens`` resolves to it + 1), so a candidate
        above it would run a verify window wider than the buffers that were
        sized once at the launch width. Growing those buffers after the fact
        frees what the launch state's captured graphs bake in -- an illegal
        memory access at the next replay, not a clean error -- so raise here
        instead. The narrower side of the candidate table needs no such check:
        the fixed launch-maximum allocations (the QSA pending ring, the request
        reservation) are sized off the widest candidate and every narrower
        width reuses them.

        Ported from https://github.com/aiueo52/sglang-rtxpro6000 branch
        flash-next-fast (snapshot 5105985).
        """
        over = sorted(s for s in self.candidate_steps if s > self.initial_steps)
        if over:
            raise ValueError(
                f"speculative_adaptive_config candidate_steps {over} exceed the "
                f"launch --speculative-num-steps ({self.initial_steps}); launch "
                "with the widest candidate as --speculative-num-steps and list "
                "the narrower ones as candidates, so nothing has to be resized "
                "while CUDA graphs still point at the old buffers"
            )

    def activate_step_by_batch(self, batch_size: int) -> None:
        target = self.params.get_steps_for_batch(batch_size)
        if target != self.worker.speculative_num_steps:
            self._activate(target)

    def observe_confidence(self, confidences: list[float], batch_size: int) -> None:
        """Draft confidence for the chain that is about to be drafted.

        Routed by ``batch_size`` like every other observation, so a C1 slot only
        ever sees C1 chains; the fixed C>=2 tier has one candidate and cannot be
        widened by a sample at all. The confidence itself arrives through the
        async side channel (``ConfidenceChannel.latest_position0``), which never
        synchronises a stream.
        """
        self.params.observe_confidence(confidences, batch_size)

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> None:
        """Feed verify results to the policy of the batch that produced them.

        This deliberately never touches the active runtime state. It runs from
        the batch-result processor as soon as a batch's accept counts reach the
        CPU, which under overlap is after the NEXT batch has already been
        launched; activating here repointed the backends and graph runners that
        in-flight batch was still running against. The donor controller did
        exactly that, and a completion of an old C1 batch left its wider policy
        (steps=15) pending over a C4 batch whose own slot says steps=3. Results
        still land on the slot routed by their own batch size; the width a batch
        runs at is chosen by activate_step_by_batch(), at that batch's own
        forward boundary and from its own size, which is also where the outgoing
        state is drained before anything is repointed.
        """
        self.params.on_verify_complete(num_correct_drafts_per_req, batch_size)

    def _activate(self, speculative_num_steps: int) -> None:
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)
