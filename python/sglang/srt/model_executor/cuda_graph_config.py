# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Phase / backend identifiers, the canonical default for
cuda_graph_config, and the --cuda-graph-config JSON CLI parser.

Module-level imports are pure stdlib — no torch / sglang.srt deps — so
ServerArgs can import everything here without pulling in backend
classes. check_cuda_graph_backend lazy-imports the config accessor
inside the function body to preserve that invariant.
"""

import argparse
import contextlib
import copy
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class Phase:
    """The two phases of model forward."""

    DECODE = "decode"
    PREFILL = "prefill"
    ALL = (DECODE, PREFILL)


class Backend:
    """CUDA graph capture backends a phase can use."""

    FULL = "full"
    BREAKABLE = "breakable"
    TC_PIECEWISE = "tc_piecewise"
    DISABLED = "disabled"
    ALL = (FULL, BREAKABLE, TC_PIECEWISE, DISABLED)


ALLOWED_BACKENDS_PER_PHASE = {
    Phase.DECODE: (
        Backend.FULL,
        Backend.BREAKABLE,
        Backend.TC_PIECEWISE,
        Backend.DISABLED,
    ),
    # full for prefill captures one whole-forward graph per num_tokens
    # bucket with a fixed request-slot count; replay pads num_tokens up to
    # the nearest captured bucket. Opt-in: the padding waste is the
    # operator's call.
    Phase.PREFILL: (
        Backend.FULL,
        Backend.BREAKABLE,
        Backend.TC_PIECEWISE,
        Backend.DISABLED,
    ),
}

# Per-phase settings schema. Keys other than backend are runner-level
# (read by any backend in that phase); tc_compiler is the lone
# backend-specific knob (only meaningful when backend == tc_piecewise).
# For prefill, bs carries aggregate-token capture buckets for every backend;
# full_prefill_max_req separately controls Full's fixed request-slot count.
# full_prefill_max_req and full_prefill_prefix_chunk_tokens are prefill-only and
# only meaningful when backend == full.
ALLOWED_KEYS_PER_PHASE = {
    Phase.DECODE: ("backend", "max_bs", "bs", "tc_compiler"),
    Phase.PREFILL: (
        "backend",
        "max_bs",
        "bs",
        "tc_compiler",
        "full_prefill_max_req",
        "full_prefill_prefix_chunk_tokens",
    ),
}


@dataclass
class PhaseConfig:
    """Per-phase CUDA graph settings."""

    backend: str = Backend.DISABLED
    max_bs: Optional[int] = None
    bs: Optional[List[int]] = None
    # Only meaningful when backend == tc_piecewise; ignored otherwise.
    tc_compiler: str = "eager"
    # Only meaningful for the prefill phase with backend == full: max number of
    # request slots baked into each captured graph. Real bs <= full_prefill_max_req
    # reuses the graph (unused slots become zero-length sentinels); larger
    # batches fall back to eager. Ignored by BCG and TC_PIECEWISE. None
    # auto-derives chunked_prefill_size // 512.
    full_prefill_max_req: Optional[int] = None
    # Only meaningful for Full prefill CUDA graphs that capture a distinct
    # cached-prefix topology: aggregate cached-prefix tokens represented by one
    # fixed-capacity chunk across all request slots. FullCG captures 1/2/4/8/16
    # chunk variants and chooses the smallest one covering a batch. None uses
    # the scheduler's aggregate chunked_prefill_size token budget.
    full_prefill_prefix_chunk_tokens: Optional[int] = None


def default_prefill_backend() -> str:
    """BCG (breakable) is the prefill default on CUDA only; other platforms
    (HIP/NPU/...) keep tc_piecewise until BCG is validated there. Full-graph
    prefill capture is opt-in per model architecture via the declarative
    registry (see _inkling_overrides in arg_groups/overrides.py), not a global
    default. Lazy import keeps this module's stdlib-only import invariant (see
    module docstring)."""
    from sglang.srt.utils import is_cuda

    return Backend.BREAKABLE if is_cuda() else Backend.TC_PIECEWISE


@dataclass
class CudaGraphConfig:
    """Top-level CUDA graph config: one PhaseConfig per phase."""

    decode: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(backend=Backend.FULL)
    )
    prefill: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(backend=default_prefill_backend())
    )

    def __getitem__(self, phase: str) -> PhaseConfig:
        """Phase-string lookup; kept for migration ergonomics."""
        if phase not in Phase.ALL:
            raise KeyError(phase)
        return getattr(self, phase)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        # Diff-only, not asdict: the parser locks every (phase, key) it sees,
        # so emitting defaults would lock fields the caller never set.
        baseline = default_cuda_graph_config()
        return {
            Phase.DECODE: _diff_phase(self.decode, baseline.decode),
            Phase.PREFILL: _diff_phase(self.prefill, baseline.prefill),
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Dict[str, Any]]]) -> "CudaGraphConfig":
        """Build from a (partial) dict of overrides, defaults fill the rest.
        Unknown phases / keys are silently dropped — the JSON-input
        validator (parse_cuda_graph_config_arg) rejects them upstream."""
        cfg = cls()
        if not raw:
            return cfg
        for phase, phase_settings in raw.items():
            if phase not in Phase.ALL or not isinstance(phase_settings, dict):
                continue
            phase_cfg = getattr(cfg, phase)
            allowed = ALLOWED_KEYS_PER_PHASE[phase]
            for key, value in phase_settings.items():
                if key in allowed:
                    setattr(phase_cfg, key, value)
        return cfg


def default_cuda_graph_config() -> CudaGraphConfig:
    """Fresh CudaGraphConfig populated with canonical defaults."""
    return CudaGraphConfig()


def _diff_phase(actual: PhaseConfig, baseline: PhaseConfig) -> Dict[str, Any]:
    """Return only fields whose value differs from the per-phase default."""
    return {
        f.name: getattr(actual, f.name)
        for f in dataclasses.fields(actual)
        if getattr(actual, f.name) != getattr(baseline, f.name)
    }


def check_cuda_graph_backend(phase: str, backend: str) -> bool:
    """True if cuda_graph_config[phase].backend == backend on the
    published config. Returns False if the config has not been published
    yet (e.g. unit tests, early startup)."""
    from sglang.srt.runtime_context import get_exec

    try:
        cfg = get_exec().graph.cuda_graph_config
    except ValueError:
        return False
    if cfg is None or phase not in Phase.ALL:
        return False
    return getattr(cfg, phase).backend == backend


def cuda_graph_fully_disabled() -> bool:
    """True iff cuda_graph_config has Backend.DISABLED on every phase.

    Use at sites that ask the legacy server_args.disable_cuda_graph
    question ("no CG anywhere globally") — e.g., preallocating buffers
    that any captured graph would otherwise reuse, or one-shot init
    that's a no-op when CG is completely off.
    """
    return check_cuda_graph_backend(
        Phase.DECODE, Backend.DISABLED
    ) and check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)


def scoped_capture_cuda_graph_config(cuda_graph_bs: List[int]) -> CudaGraphConfig:
    """The canonical graph config for one adaptive-capture window.

    Adaptive speculative decoding builds one state per candidate width and
    prunes the decode capture buckets to the batch sizes that width can reach
    (possibly to the empty list = no graphs for that width). The capture-list
    consumers -- ``get_batch_sizes_to_capture`` and ``check_cuda_graph_backend``
    -- read ``cuda_graph_config[decode].bs`` / ``.backend``, so that is the leaf
    the scoped override carries; the legacy ``cuda_graph_bs_decode`` /
    ``disable_cuda_graph`` leaves are folded into this config once at startup
    and ``RuntimeContext.override`` does not synchronize them, so writing those
    renames the knob without moving the value.

    Fresh copies at the top level and for the decode phase (never mutate the
    published config in place -- it is the global resolved state every other
    width and every later reader shares); ``prefill`` is carried by reference
    and never touched here. An empty bucket list disables the decode phase the
    way the runner asks about it -- ``backend=disabled`` -- because dropping
    only ``bs`` would leave ``FULL`` advertised and a caller reading
    ``check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)`` would still
    build a graph runner against an empty capture list.
    """
    from sglang.srt.runtime_context import get_exec

    base = get_exec().graph.cuda_graph_config or default_cuda_graph_config()
    scoped = copy.copy(base)
    decode = copy.copy(base.decode)
    decode.bs = list(cuda_graph_bs)
    if not cuda_graph_bs:
        decode.backend = Backend.DISABLED
    scoped.decode = decode
    return scoped


# One-shot startup handoff for the adaptive launch-width capture scope: the
# worker plans it before the scheduler begins the start-up captures, and the
# actual decode-capture boundaries consume it. A module global (not a context
# wrapping whole workers) because the process-shared logits buffer, the
# EagerRunner fixed-max buffers and the prefill capture inside
# cuda_graph_setup.capture_cuda_graphs MUST be provisioned from the FULL
# canonical bucket list: GraphSharedOutput.create_for_model_runner sizes
# max_decode_logits_rows off the decode buckets before EagerRunner allocation,
# and the shared buffer can never be resized after captured graphs point at
# it. Pruning the whole startup window under-sized it for every bucket and
# width the run later serves.
_adaptive_launch_capture_scope: Optional[CudaGraphConfig] = None


def set_adaptive_launch_capture_scope(cfg: CudaGraphConfig) -> None:
    """Queue the scoped config for the start-up decode-capture boundaries."""
    global _adaptive_launch_capture_scope
    if _adaptive_launch_capture_scope is not None:
        raise RuntimeError(
            "adaptive launch capture scope already pending: a previous "
            "startup-capture planning was never cleared (fail loud rather "
            "than leak a pruned config into the next capture)"
        )
    _adaptive_launch_capture_scope = cfg


def clear_adaptive_launch_capture_scope() -> None:
    global _adaptive_launch_capture_scope
    _adaptive_launch_capture_scope = None


@contextlib.contextmanager
def adaptive_launch_capture_scope():
    """Enter the queued launch-width capture scope for one capture boundary.

    Used by the target verify boundary in cuda_graph_setup.capture_cuda_graphs
    and by the initial EagleDraftWorker draft decode / draft-extend capture.
    The published config is restored when the boundary exits, so the next
    provisioning step and AdaptiveController.init_states both see the FULL
    canonical buckets again. No scope queued (fixed width, graphs disabled,
    another speculative algorithm): an ordinary no-op window.
    """
    cfg = _adaptive_launch_capture_scope
    if cfg is None:
        yield
        return
    from sglang.srt.runtime_context import get_context, get_exec

    original = get_exec().graph.cuda_graph_config
    get_context().override("adaptive_spec.launch_capture", cuda_graph_config=cfg)
    try:
        yield
    finally:
        get_context().override(
            "adaptive_spec.launch_capture_restore", cuda_graph_config=original
        )


def parse_cuda_graph_config_arg(raw: str) -> Dict[str, Dict[str, Any]]:
    """argparse type for --cuda-graph-config: parse JSON dict of
    phase → settings dict. Each phase's settings dict is itself validated
    against ALLOWED_KEYS_PER_PHASE. Returns a plain dict — the
    precedence pipeline in ServerArgs converts to CudaGraphConfig
    after merging."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"--cuda-graph-config must be JSON: {e}")
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            f"--cuda-graph-config must be a JSON object, got {type(parsed).__name__}"
        )

    result: Dict[str, Dict[str, Any]] = {}
    for phase, phase_settings in parsed.items():
        phase = str(phase)
        if phase not in Phase.ALL:
            raise argparse.ArgumentTypeError(
                f"--cuda-graph-config: unknown phase '{phase}', expected one of {Phase.ALL}"
            )
        if not isinstance(phase_settings, dict):
            raise argparse.ArgumentTypeError(
                f"--cuda-graph-config['{phase}'] must be a JSON object, got "
                f"{type(phase_settings).__name__}"
            )
        allowed = ALLOWED_KEYS_PER_PHASE[phase]
        result[phase] = {}
        for key, value in phase_settings.items():
            if key not in allowed:
                raise argparse.ArgumentTypeError(
                    f"--cuda-graph-config['{phase}']: unknown key '{key}', expected one of {allowed}"
                )
            result[phase][key] = value
    return result


def explicit_keys_in(
    settings: Optional[Dict[str, Dict[str, Any]]],
) -> set:
    """Return the set of (phase, key) tuples present in settings
    (the raw dict form, as it arrives from CLI/SDK). Used by ServerArgs
    to track keys the user explicitly set so the auto-disable cascade can
    skip them."""
    out: set = set()
    if not settings:
        return out
    for phase, phase_settings in settings.items():
        if not isinstance(phase_settings, dict):
            continue
        for key in phase_settings.keys():
            out.add((phase, key))
    return out
