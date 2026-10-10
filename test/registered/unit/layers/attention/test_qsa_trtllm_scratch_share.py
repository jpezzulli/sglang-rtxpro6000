"""CPU regressions for the QSA trtllm-gen/XQA decode scratch's ownership scope.

``QwenSparseAttnBackend`` used to lazily own a private zeroed 128 MiB workspace
per backend instance -- 4 per launch at the fixed W4 recipe (target verify, draft
extend and the 2 draft-step children its 3 steps build) and 12 under the adaptive
W4/W8 table, because every candidate width builds its own backends: a width's
draft-token count is ``speculative_num_steps`` + 1, and
``QwenSparseMultiStepDraftBackend`` owns one child per step after the first, so
W4 is 3 steps / 2 children and W8 is 7 steps / 6 children. The buffer is not
cache state:
FlashInfer's trtllm/XQA paged decode (the SM120 auto-dispatch; 0.7.0.post1
``decode.py`` -> ``xqa_wrapper.cu`` / ``mha.cu``) uses the first 8 MiB for split-K
semaphores that a completed launch returns to zero, and the rest for scratch that
every launch overwrites. Zero-initialisation is a FIRST-use requirement, so the
region is exclusive per launch and one stable allocation may serve every consumer
whose kernels are stream-ordered -- which is the whole spec-v2 round: draft ->
target verify -> draft extend are all enqueued on the one compute stream, the plan
stream only prepares metadata and is joined with an explicit ``wait_stream``
before any forward, and CUDA-graph capture records an address without executing
while every replay rides that same compute stream.

What is pinned here, in the order the failure modes come in:

* the region is still exactly the 128 MiB zeroed uint8 buffer the paged-decode
  dispatch is qualified for;
* the target-verify backend, the draft-extend backend and every draft-step child
  that ``QwenSparseMultiStepDraftBackend`` really constructs resolve to ONE
  backing storage, initialised once;
* that storage belongs to the process, not to the backend that allocated it, so a
  width transition can drop the retired state's backends without freeing an
  address their captured graphs still bake -- while the per-width static metadata
  those same backends own stays distinct, because sharing it too would make one
  width's replay read another width's tables;
* the sharing refuses the launch modes that put attention on real concurrent
  streams (PD-multiplexing green contexts, TBO children) and any runner whose
  mode it cannot read -- those keep the pre-sharing private workspace per backend;
* scratch first touched INSIDE a CUDA-graph capture stays private to that capture:
  the capture's memory pool, not this registry, decides how long an allocation
  made during capture lives and which other capture may be handed those bytes;
* an index-less ``cuda`` device cannot split one domain into two allocations
  between the graph-state hook (``runner.device``) and the forward
  (``q.device``), and a second device cannot inherit another rank's scratch.

CPU cannot prove the GPU stream ordering, nor that a replay writes the shared
semaphores back to zero; those are the GPU-window checks listed at the bottom.
"""

import gc
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    QwenSparseMultiStepDraftBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The kernel-visible region: 8 MiB of XQA semaphores + 120 MiB of scratch.
WORKSPACE_BYTES = 128 * 1024 * 1024
# The same region at test scale -- identity and lifetime are what the cases
# check, and the size is pinned by its own case.
TEST_BYTES = 4096
PAGE = qsa._TRTLLM_SPARSE_PAGE_SIZE
CPU = torch.device("cpu")
OTHER_DEVICE = torch.device("meta")
CPU_SLOT = (CPU.type, -1)
OTHER_SLOT = (OTHER_DEVICE.type, -1)
STEPS = 3
ADAPTIVE_STEPS = 7

_REAL_ZEROS = torch.zeros


def _runner(device=CPU, *, pdmux=False, tbo=False, record_modes=True):
    """A ModelRunner-shaped owner for one QSA backend: the device it runs on, the
    pools every consumer of the launch shares, and the launch record the scratch's
    execution domain is proven from."""
    launch_record = {"enable_pdmux": pdmux, "enable_two_batch_overlap": tbo}
    return SimpleNamespace(
        device=device,
        server_args=SimpleNamespace(**(launch_record if record_modes else {})),
        token_to_kv_pool=SimpleNamespace(
            qsa_compressed_page_size=PAGE,
            qsa_num_groups=1,
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=None),
        model_config=SimpleNamespace(
            context_len=4096,
            hf_text_config=SimpleNamespace(indexer_compress_ratio=4),
        ),
    )


def _width_backends(steps=STEPS):
    """One width's real consumer set: its target-verify and draft-extend backends
    plus the per-step children the production multi-step draft backend builds."""
    target = QwenSparseAttnBackend(_runner())
    extend = QwenSparseAttnBackend(_runner())
    children = QwenSparseMultiStepDraftBackend(
        _runner(), topk=1, speculative_num_steps=steps
    ).attn_backends
    return [target, extend], children


def _scratch_env(monkeypatch):
    """A fresh registry at test scale, plus the list of devices it allocated for."""
    monkeypatch.setattr(qsa, "_TRTLLM_WORKSPACE_BYTES", TEST_BYTES)
    monkeypatch.setattr(qsa, "_TRTLLM_SHARED_WORKSPACES", {})
    allocations = []
    real_new_workspace = qsa._new_trtllm_workspace

    def counting(device):
        allocations.append(device)
        return real_new_workspace(device)

    monkeypatch.setattr(qsa, "_new_trtllm_workspace", counting)
    return allocations


def _without_pinned_host_memory(monkeypatch):
    """This host has no page-locked allocator (a GPU-window detail, not a scratch
    one), so the graph-state length staging runs unpinned."""

    def zeros(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return _REAL_ZEROS(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros)


def test_workspace_is_the_zeroed_uint8_region_the_kernel_requires(monkeypatch):
    assert qsa._TRTLLM_WORKSPACE_BYTES == WORKSPACE_BYTES, (
        "the QSA trtllm/XQA scratch must stay the 128 MiB region (8 MiB of "
        "semaphores + 120 MiB of scratch) the paged-decode dispatch is "
        "qualified for"
    )
    monkeypatch.setattr(qsa, "_TRTLLM_WORKSPACE_BYTES", TEST_BYTES)
    workspace = qsa._new_trtllm_workspace(CPU)

    assert workspace.dtype is torch.uint8
    assert workspace.numel() == TEST_BYTES
    # The API's first-use contract: zero-initialised, never torch.empty.
    assert not bool(workspace.any()), "trtllm scratch could reach a launch dirty"


def test_stream_ordered_consumers_share_one_backing_storage(monkeypatch):
    allocations = _scratch_env(monkeypatch)
    pair, children = _width_backends()

    workspaces = [b._ensure_trtllm_workspace(CPU) for b in pair + children]

    assert len(children) == STEPS - 1, (
        "the multi-step draft backend stopped owning one child per draft step; "
        "this case no longer covers the consumers a width actually runs"
    )
    shared = workspaces[0]
    for backend, workspace in zip(pair + children, workspaces):
        assert workspace is shared, (
            f"{type(backend).__name__} resolved its own scratch: one width's "
            "consumers are stream-ordered and share one region"
        )
    assert (
        len(allocations) == 1
    ), f"the shared scratch was initialised {len(allocations)} times"
    assert qsa._TRTLLM_SHARED_WORKSPACES[CPU_SLOT] is shared, (
        "the workspace must be owned by the process registry, not by the backend "
        "that happened to ask first"
    )


def test_scratch_outlives_the_width_that_allocated_it(monkeypatch):
    allocations = _scratch_env(monkeypatch)
    old_pair, old_children = _width_backends()
    allocated = old_pair[0]._ensure_trtllm_workspace(CPU)
    del old_pair, old_children
    gc.collect()

    pair, children = _width_backends(steps=ADAPTIVE_STEPS)
    for backend in pair + children:
        assert backend._ensure_trtllm_workspace(CPU) is allocated, (
            "a width transition re-took the scratch: the retired width's graphs "
            "baked this address and the incoming width's graphs must bake the "
            "same one"
        )
    assert len(allocations) == 1

    # Per-width metadata is a separate story: both widths replay against their own
    # static tables, so the shared scratch must not pull those together either.
    target_tables = pair[0]._get_trtllm_sparse_tables(4, 2, PAGE, CPU)
    child_tables = children[0]._get_trtllm_sparse_tables(4, 2, PAGE, CPU)
    assert (
        target_tables[1].data_ptr() != child_tables[1].data_ptr()
    ), "width-specific graph metadata was merged into the shared scratch"


def test_concurrent_and_unprovable_launches_keep_private_scratch(monkeypatch):
    allocations = _scratch_env(monkeypatch)

    for label, runner in (
        ("PD-multiplexing", _runner(pdmux=True)),
        ("two-batch overlap", _runner(tbo=True)),
        ("no launch record", SimpleNamespace()),
        ("launch record without the mode fields", _runner(record_modes=False)),
    ):
        workspaces = [
            QwenSparseAttnBackend(runner)._ensure_trtllm_workspace(CPU)
            for _ in range(2)
        ]
        assert workspaces[0].data_ptr() != workspaces[1].data_ptr(), (
            f"{label} runs attention on concurrent streams, so two of its "
            "backends must not share one XQA semaphore region"
        )

    assert len(allocations) == 8, "an unshareable launch went through the registry"
    assert (
        qsa._TRTLLM_SHARED_WORKSPACES == {}
    ), "an unshareable launch registered a shared scratch"


def test_workspace_first_taken_inside_a_capture_stays_private(monkeypatch):
    allocations = _scratch_env(monkeypatch)
    capturing = []
    # The capture predicate is a driver call; force the branch on CPU and pin its
    # consequence (private and unregistered), not the call itself.
    monkeypatch.setattr(
        qsa, "_trtllm_scratch_is_capturing", lambda device: bool(capturing)
    )

    capturing.append(True)
    inside_capture = QwenSparseAttnBackend(_runner())._ensure_trtllm_workspace(CPU)
    assert len(allocations) == 1
    assert qsa._TRTLLM_SHARED_WORKSPACES == {}, (
        "scratch allocated inside a capture belongs to that capture's memory "
        "pool; registering it lends that pool's lifetime to another graph"
    )

    capturing.clear()
    outside_capture = QwenSparseAttnBackend(_runner())._ensure_trtllm_workspace(CPU)
    assert outside_capture is not inside_capture
    assert list(qsa._TRTLLM_SHARED_WORKSPACES) == [CPU_SLOT]


def test_scratch_is_isolated_per_device_and_one_cuda_slot_cannot_split(monkeypatch):
    allocations = _scratch_env(monkeypatch)
    monkeypatch.setattr(qsa.torch.cuda, "current_device", lambda: 1)

    on_cpu = qsa._acquire_trtllm_workspace(True, CPU)
    on_other_device = qsa._acquire_trtllm_workspace(True, OTHER_DEVICE)
    shared_again = qsa._acquire_trtllm_workspace(True, OTHER_DEVICE)

    assert on_cpu is not on_other_device
    assert on_other_device.device == OTHER_DEVICE
    assert (
        shared_again is on_other_device
    ), "a second device's scratch was not shared either"
    assert len(allocations) == 2
    assert sorted(qsa._TRTLLM_SHARED_WORKSPACES) == [CPU_SLOT, OTHER_SLOT]

    # The graph-state hook reads ``runner.device`` ("cuda", no index) and the
    # forward reads ``q.device`` ("cuda:1"): one domain, one slot, one allocation.
    assert qsa._trtllm_workspace_slot("cuda") == qsa._trtllm_workspace_slot(
        torch.device("cuda:1")
    )
    assert qsa._trtllm_workspace_slot("cuda:0") != qsa._trtllm_workspace_slot(
        torch.device("cuda:1")
    ), "one rank's scratch would answer for another rank's stream"
    assert qsa._trtllm_workspace_slot(CPU) == CPU_SLOT
    assert len(allocations) == 2, "slot arithmetic must not allocate"


def test_the_sparse_decode_launch_is_handed_the_shared_scratch(monkeypatch):
    """Drive the real trtllm call site, not just the ownership helper.

    The two QSA triton packs are the CPU-impassable part of the path (they are
    triton launches); everything between them and the FlashInfer call -- the
    static sparse tables, the packed scratch views and the workspace resolution
    under test -- is the production code.
    """
    allocations = _scratch_env(monkeypatch)
    monkeypatch.setattr(qsa, "qwen_sparse_valid_counts_triton", lambda *a, **k: None)
    monkeypatch.setattr(
        qsa, "qwen_sparse_kv_extraction_compact_triton", lambda *a, **k: None
    )
    launched = []

    def recording_decode(*, query, workspace_buffer, block_tables, **kwargs):
        launched.append((workspace_buffer, block_tables))
        return torch.zeros(query.shape[0], query.shape[1], query.shape[2])

    pair, children = _width_backends()
    metadata = SimpleNamespace(
        is_cuda_graph=False,
        sequence_lengths=torch.tensor([64, 64], dtype=torch.int32),
        fa2_valid_counts=None,
        row_req_pool_indices=None,
    )
    layer = SimpleNamespace(layer_id=0, scaling=0.25)
    k_buffer = torch.zeros(8, 1, 4, dtype=torch.uint8)
    topk_indices = torch.zeros(2, 64, dtype=torch.int32)
    forward_batch = SimpleNamespace(req_pool_indices=torch.arange(2, dtype=torch.int64))
    q = torch.zeros(2, 1, 4, dtype=torch.bfloat16)

    for backend in pair + children:
        output = backend._forward_trtllm_sparse(
            q,
            k_buffer,
            k_buffer,
            layer,
            forward_batch,
            metadata,
            topk_indices,
            recording_decode,
        )
        assert output.shape == (2, 4)

    assert len(launched) == STEPS + 1
    assert (
        len(allocations) == 1
    ), "the launch site materialised its own workspace instead of the domain's"
    shared = qsa._TRTLLM_SHARED_WORKSPACES[CPU_SLOT]
    for workspace, _tables in launched:
        assert workspace is shared, "a consumer launched against a private scratch"
    # ... while the static sparse tables each instance bakes into ITS graphs stay
    # per-instance: sharing those would make one width's replay read another's.
    tables = {per_tables.data_ptr() for _workspace, per_tables in launched}
    assert len(tables) == len(
        launched
    ), "backends share more than the scratch: their graph sparse tables aliased"


@pytest.mark.parametrize("trtllm_dispatch", [object(), None])
def test_graph_state_init_provisions_the_scratch_outside_capture(
    monkeypatch, trtllm_dispatch
):
    allocations = _scratch_env(monkeypatch)
    _without_pinned_host_memory(monkeypatch)
    # The real dispatch gate: the region is provisioned exactly where trtllm is
    # this platform's sparse-decode kernel.
    monkeypatch.setattr(qsa, "_resolve_trtllm_sparse_decode", lambda: trtllm_dispatch)
    backend = QwenSparseAttnBackend(_runner())
    assert backend._trtllm_workspace is None

    backend.init_cuda_graph_state(max_bs=8, max_num_tokens=32)

    if trtllm_dispatch is None:  # parametrised: the FA varlen fallback platform
        assert allocations == [], (
            "a platform behind the FA varlen fallback must not pay for a trtllm "
            "workspace it never launches"
        )
        return
    assert len(allocations) == 1, (
        "the scratch must be provisioned before capture, where it is an ordinary "
        "eager allocation every width's graphs can bake"
    )
    assert backend._trtllm_workspace is qsa._TRTLLM_SHARED_WORKSPACES[CPU_SLOT]


# GPU window -- the smallest checks that close what CPU cannot reach (for Codex):
#  1. one spec-v2 decode step at the fixed W4 recipe, then
#     ``nvidia-smi --query-gpu=memory.used`` / a memory-snapshot diff: one 128 MiB
#     region across that width's four QSA backends (target verify, draft extend,
#     2 draft-step children) -- and one, not twelve, across the adaptive W4+W8
#     pair -- and still one after a W4 -> W8 -> W4
#     adaptive transition (the retired state's backends are gone, the pool is not);
#  2. long-context decode logits at each width, diffed against this base
#     revision's per-backend-workspace build: identical, i.e. a reused semaphore
#     region does not perturb the split-K reduction across consumers;
#  3. target-verify graph captured AFTER an eager warmup at another shape, replayed
#     against the eager-published allocation: no IMA and no second 128 MiB block
#     in ``torch.cuda.memory_snapshot`` for that graph's capture pool.
