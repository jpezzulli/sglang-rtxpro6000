"""CPU coverage for min_p in the speculative target distribution of both
supported profiles.

The ordinary sampler filters in the order top-k -> top-p -> min_p, each stage
renormalized. The two speculative verifies build their own target distribution
and applied only top-k and top-p, so a request with a nonzero min_p was
verified against a wider distribution than the sampler would have drawn from:
excluded tokens could still win the acceptance coin, and the final / bonus draw
came from that same unfiltered row. The three producers covered here are the
EAGLE sampling verify (``eagle_sample``), DFlash's shared builder
(``build_dflash_verify_target_probs``, the caller at dflash_worker_v2's
sampling branch), in both its sparse-topk and its dense fallback branch, and
the DSpark selector route (``_accept_sampling_core``), whose unfiltered
SoftmaxTemp shortcut used to be selected on top-k/top-p alone -- so a
min_p-only request, the LM Studio default, never reached the builder at all.

What is checkable without a device is the contract around the kernels: the one
``target_probs`` tensor handed to ``tree_speculative_sampling_target_only`` /
``chain_speculative_sampling_triton``, which the acceptance test and the
final/bonus sample both consume. The EAGLE renorms are CPU stand-ins (the AOT
kernels are not importable here); DFlash's own torch fallback renorms are the
real product code. Acceptance-rate numerics of a live run stay with the GPU
spec suites. Every test uses logits whose token order is NOT the probability
order, so a filter that kept the first k columns instead of the k largest
values would not pass by accident.
"""

import contextlib
import sys
import types
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.kernels.ops.speculative.dspark import dspark_accept
from sglang.srt.speculative import dflash_utils, eagle_utils
from sglang.srt.speculative.dflash_utils import build_dflash_verify_target_probs
from sglang.srt.speculative.eagle_utils import eagle_sample

VOCAB = 6
DRAFT_TOKEN_NUM = 2
BS = 2
ROWS = BS * DRAFT_TOKEN_NUM

# Rows 0/2 and 1/3 are the two requests' identical logits rows; the only thing
# that differs between the requests is their own min_p (0.0 vs 0.5). Every row
# is deliberately unsorted in token-id order: the largest logit sits at column
# 2 for the rows whose top-3 is (2, 4, 0).
LOGITS = torch.tensor(
    [
        [2.0, -1.0, 3.0, 0.5, 2.5, 1.0],
        [0.5, 2.5, 1.0, -1.0, 3.0, 0.0],
        [2.0, -1.0, 3.0, 0.5, 2.5, 1.0],
        [0.5, 2.5, 1.0, -1.0, 3.0, 0.0],
    ],
    dtype=torch.float32,
)
TEMPERATURES = torch.tensor([[1.0], [0.7]])
TOP_KS = torch.tensor([3, 3], dtype=torch.int32)
TOP_PS = torch.tensor([0.9, 0.9])
MIN_PS = torch.tensor([0.0, 0.5])
# The chain these numbers come from: the k largest logits are kept and
# renormalized, top-p keeps the shortest prefix reaching 0.9, and half of the
# surviving maximum cuts everything below it. Request 0 runs at temperature 1.0,
# request 1 at 0.7; rows 0/2 and rows 1/3 share their logits.
EXPECTED_OPEN_ROW = torch.tensor([0.1863, 0.0, 0.5065, 0.0, 0.3072, 0.0])
EXPECTED_CUT_ROW = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
EXPECTED_OPEN_ROW_2 = torch.tensor([0.0, 0.3775, 0.0, 0.0, 0.6225, 0.0])
EXPECTED_CUT_ROW_2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
# The same two logits rows at request 1's temperature 0.7, unfiltered.
EXPECTED_OPEN_ROW_T07 = torch.tensor([0.1386, 0.0, 0.5783, 0.0, 0.2831, 0.0])
EXPECTED_OPEN_ROW_2_T07 = torch.tensor([0.0, 0.3287, 0.0, 0.0, 0.6713, 0.0])
# Same logits and same temperature, only the request's min_p differs: half the
# surviving maximum (0.5065) falls between the second and the third entry.
EXPECTED_CUT_ROW_SAME_TEMP = torch.tensor([0.0, 0.0, 0.6225, 0.0, 0.3775, 0.0])


def _reference_top_k_renorm(probs, top_ks):
    """Reference for the AOT kernel: keep the k LARGEST entries (not the first
    k columns), renormalize."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    ranks = torch.arange(probs.shape[-1], device=probs.device)
    kept = probs_sort * (ranks < top_ks.view(-1, 1))
    out = torch.zeros_like(probs).scatter_(-1, probs_idx, kept)
    return out / out.sum(dim=-1, keepdim=True)


def _reference_top_p_renorm(probs, top_ps):
    """Reference for the AOT kernel: keep the shortest prefix reaching top_p,
    renormalize."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    keep_sorted = (probs_sort.cumsum(dim=-1) - probs_sort) <= top_ps.view(-1, 1)
    keep = torch.zeros_like(probs, dtype=torch.bool).scatter_(
        -1, probs_idx, keep_sorted
    )
    out = probs * keep
    return out / out.sum(dim=-1, keepdim=True)


def _reference_min_p_renorm(probs, min_ps):
    """Independent min_p stage: drop p < min_p * max(p), renormalize."""
    keep = probs >= probs.amax(dim=-1, keepdim=True) * min_ps.view(-1, 1)
    out = probs * keep
    return out / out.sum(dim=-1, keepdim=True)


def _reference_target_probs(
    logits=LOGITS,
    temperatures=TEMPERATURES,
    top_ks=TOP_KS,
    top_ps=TOP_PS,
    min_ps=MIN_PS,
    apply_min_p=True,
):
    """The ordinary sampler's chain, per request, expanded over draft rows."""
    expand = lambda t: t.repeat_interleave(DRAFT_TOKEN_NUM, dim=0).to(  # noqa: E731
        torch.float32
    )
    probs = torch.softmax(logits / expand(temperatures), dim=-1)
    probs = _reference_top_k_renorm(probs, expand(top_ks).to(torch.int32))
    probs = _reference_top_p_renorm(probs, expand(top_ps))
    if apply_min_p:
        probs = _reference_min_p_renorm(probs, expand(min_ps))
    return probs.view(BS, DRAFT_TOKEN_NUM, -1)


def _sampling_info(
    *,
    temperatures=TEMPERATURES,
    top_ks=TOP_KS,
    top_ps=TOP_PS,
    min_ps=MIN_PS,
    is_all_greedy=False,
    flags=True,
):
    """Mirrors SamplingBatchInfo.create_initialized: the need_* flags come from
    the per-request values, so a batch of defaults must trigger no filter."""
    info = SimpleNamespace(
        is_all_greedy=is_all_greedy,
        temperatures=temperatures,
        top_ks=top_ks,
        top_ps=top_ps,
        min_ps=min_ps,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        sampling_seed=None,
    )
    # `flags=False` stands in for the callers that hand the builder a bare
    # object without the need_* attributes at all.
    if flags:
        info.need_top_k_sampling = bool((top_ks != VOCAB).any())
        info.need_top_p_sampling = bool((top_ps != 1.0).any())
        info.need_min_p_sampling = bool((min_ps > 0).any())
    return info


def _close(actual, expected, atol=2e-4):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-4)


# ---------------------------------------------------------------- EAGLE verify


def _eagle_verify_input():
    return SimpleNamespace(
        draft_token=torch.arange(ROWS, dtype=torch.int64),
        draft_token_num=DRAFT_TOKEN_NUM,
        max_tree_depth=DRAFT_TOKEN_NUM,
        tree_topk=1,
        retrieve_index=torch.arange(ROWS, dtype=torch.int32),
        retrieve_next_token=torch.zeros(ROWS, dtype=torch.int32),
        retrieve_next_sibling=torch.zeros(ROWS, dtype=torch.int32),
        draft_probs=None,
    )


def _eagle_batch(sampling_info):
    return SimpleNamespace(
        device="cpu",
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.arange(BS, dtype=torch.int64),
        sampling_info=sampling_info,
    )


@contextlib.contextmanager
def _fake_eagle_kernels(captured):
    """Route the EAGLE verify through the CPU references and record what the
    tree kernel is handed."""

    def _tree_sampling_target_only(**kwargs):
        captured["target_probs"] = kwargs["target_probs"]
        captured["draft_probs"] = kwargs["draft_probs"]

    kernel = types.ModuleType("sgl_kernel")
    kernel.top_k_renorm_prob = _reference_top_k_renorm
    kernel.top_p_renorm_prob = _reference_top_p_renorm
    kernel.tree_speculative_sampling_target_only = _tree_sampling_target_only
    reject = types.ModuleType("sglang.kernels.ops.speculative.reject_sampling")
    reject.chain_speculative_sampling_triton = mock.MagicMock()

    spec = SimpleNamespace(
        speculative_use_rejection_sampling=False,
        speculative_accept_threshold_single=1.0,
        speculative_accept_threshold_acc=1.0,
    )
    with mock.patch.dict(
        sys.modules, {"sgl_kernel": kernel, reject.__name__: reject}
    ), mock.patch.object(eagle_utils, "get_spec", lambda: spec), mock.patch.object(
        eagle_utils, "borrow_graph_pool", lambda user: contextlib.nullcontext()
    ), mock.patch(
        "sglang.srt.distributed.get_tp_group", lambda: SimpleNamespace(world_size=1)
    ), mock.patch(
        "sglang.srt.layers.dp_attention.is_dp_attention_enabled", lambda: False
    ):
        yield captured


def _run_eagle(sampling_info):
    with _fake_eagle_kernels({}) as captured:
        eagle_sample(
            verify_input=_eagle_verify_input(),
            batch=_eagle_batch(sampling_info),
            logits_output=SimpleNamespace(next_token_logits=LOGITS.clone()),
            grammar_mask=None,
        )
    return captured


def test_top_k_reference_selects_by_value_not_column_order():
    # The stand-ins above must not be a first-k shortcut, or every expectation
    # below would be self-fulfilling.
    probs = torch.tensor([[0.1, 0.4, 0.05, 0.2, 0.25]])
    _close(
        _reference_top_k_renorm(probs, torch.tensor([3])),
        torch.tensor([[0.0, 0.4706, 0.0, 0.2353, 0.2941]]),
    )


def test_eagle_nonzero_min_p_filters_the_target_distribution():
    target_probs = _run_eagle(_sampling_info())["target_probs"]
    assert target_probs.shape == (BS, DRAFT_TOKEN_NUM, VOCAB)
    _close(target_probs, _reference_target_probs())
    # Every row is still a distribution, because the one tensor feeds both the
    # acceptance test and the final sample.
    torch.testing.assert_close(
        target_probs.sum(dim=-1), torch.ones(BS, DRAFT_TOKEN_NUM)
    )


def test_eagle_each_request_is_filtered_with_its_own_min_p():
    rows = _run_eagle(_sampling_info())["target_probs"].reshape(ROWS, VOCAB)
    # Base behavior: every one of these rows came out of the unfiltered
    # top-k/top-p chain. Which columns survive is not the first k in token
    # order, and the two rows of a request are not interchangeable.
    _close(rows[0], EXPECTED_OPEN_ROW)
    _close(rows[2], EXPECTED_CUT_ROW)
    _close(rows[1], EXPECTED_OPEN_ROW_2)
    _close(rows[3], EXPECTED_CUT_ROW_2)
    assert int((rows[0] > 0).sum()) == 3, "min_p=0 keeps the top-p support"
    assert int((rows[2] > 0).sum()) == 1, "min_p=0.5 keeps only the row maximum"
    _close(rows[2].sum(), torch.tensor(1.0))


def test_eagle_min_p_is_the_only_difference_between_two_requests():
    # Same logits, same temperature, two draft rows each; the requests differ
    # only in min_p (0.0 vs 0.5). On the base they were the same row.
    rows = (
        _run_eagle(_sampling_info(temperatures=torch.ones(BS, 1)))["target_probs"]
        .reshape(ROWS, VOCAB)
    )
    _close(rows[0], EXPECTED_OPEN_ROW)
    _close(rows[2], EXPECTED_CUT_ROW_SAME_TEMP)
    assert int((rows[0] > 0).sum()) == 3
    assert int((rows[2] > 0).sum()) == 2
    _close(rows[2].sum(), torch.tensor(1.0))


def test_eagle_default_min_p_leaves_the_existing_distribution_untouched():
    zero_min_ps = torch.zeros(BS)
    rows = _run_eagle(_sampling_info(min_ps=zero_min_ps))["target_probs"].reshape(
        ROWS, VOCAB
    )
    _close(
        rows,
        _reference_target_probs(min_ps=zero_min_ps, apply_min_p=False).reshape(
            ROWS, VOCAB
        ),
    )
    # Rows where the filter would have bitten are untouched when the request
    # did not ask for it, at both temperatures.
    _close(rows[0], EXPECTED_OPEN_ROW)
    _close(rows[2], EXPECTED_OPEN_ROW_T07)
    _close(rows[3], EXPECTED_OPEN_ROW_2_T07)


def test_eagle_greedy_requests_never_reach_the_min_p_filter():
    greedy = _sampling_info(min_ps=torch.ones(BS), is_all_greedy=True)
    greedy_fn = mock.MagicMock(
        side_effect=lambda **kwargs: (
            kwargs["predicts"],
            kwargs["accept_index"],
            kwargs["accept_token_num"],
        )
    )
    with _fake_eagle_kernels({}) as captured, mock.patch.object(
        eagle_utils, "verify_tree_greedy_func", greedy_fn
    ):
        eagle_sample(
            verify_input=_eagle_verify_input(),
            batch=_eagle_batch(greedy),
            logits_output=SimpleNamespace(next_token_logits=LOGITS.clone()),
            grammar_mask=None,
        )
    assert greedy_fn.called
    assert captured == {}  # the sampling kernel, and the filter, were not used


# ------------------------------------------------------------ DFlash verify


def _run_dflash_builder(sampling_info, *, use_sparse_topk, max_top_k=None):
    """Call the shared builder the DFlash worker's sampling branch uses."""
    return build_dflash_verify_target_probs(
        next_token_logits=LOGITS.clone(),
        sampling_info=sampling_info,
        draft_token_num=DRAFT_TOKEN_NUM,
        bs=BS,
        max_top_k=max_top_k,
        use_sparse_topk=use_sparse_topk,
    )


def test_dflash_sparse_topk_branch_honors_min_p():
    probs = _run_dflash_builder(_sampling_info(), use_sparse_topk=True)
    _close(probs, _reference_target_probs())
    _close(probs.reshape(ROWS, VOCAB)[2], EXPECTED_CUT_ROW)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(BS, DRAFT_TOKEN_NUM))


def test_dflash_dense_fallback_branch_honors_min_p():
    probs = _run_dflash_builder(_sampling_info(), use_sparse_topk=False)
    _close(probs, _reference_target_probs())
    _close(probs.reshape(ROWS, VOCAB)[2], EXPECTED_CUT_ROW)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(BS, DRAFT_TOKEN_NUM))


def test_dflash_branches_agree_with_and_without_min_p():
    # The two branches differ only in how they reach the distribution, so they
    # have to agree -- including on the min_p cut, not just without it.
    _close(
        _run_dflash_builder(_sampling_info(), use_sparse_topk=True),
        _run_dflash_builder(_sampling_info(), use_sparse_topk=False),
    )
    no_filter = _sampling_info(
        min_ps=torch.zeros(BS),
    )
    _close(
        _run_dflash_builder(no_filter, use_sparse_topk=True),
        _reference_target_probs(apply_min_p=False),
    )
    _close(
        _run_dflash_builder(no_filter, use_sparse_topk=False),
        _reference_target_probs(apply_min_p=False),
    )


def test_dflash_min_p_only_request_is_filtered_without_top_k_or_top_p():
    all_out = torch.tensor([VOCAB, VOCAB], dtype=torch.int32)
    ones = torch.ones(BS)
    info = _sampling_info(top_ks=all_out, top_ps=ones, min_ps=torch.tensor([0.0, 0.5]))
    assert not info.need_top_k_sampling and not info.need_top_p_sampling
    assert info.need_min_p_sampling
    for sparse in (True, False):
        rows = _run_dflash_builder(info, use_sparse_topk=sparse).reshape(ROWS, VOCAB)
        raw = torch.softmax(
            LOGITS / TEMPERATURES.repeat_interleave(DRAFT_TOKEN_NUM, 0), -1
        )
        _close(rows[0], raw[0])  # min_p=0: nothing is taken away
        _close(rows[2], _reference_min_p_renorm(raw[2:3], torch.tensor([0.5]))[0])
        assert int((rows[2] > 0).sum()) < VOCAB


def test_dflash_builder_tolerates_a_sampling_info_without_the_flags():
    # The builder has always read the need_* flags through getattr; a bare
    # caller object must still get the top-k/top-p chain and no min_p stage.
    info = _sampling_info(flags=False)
    probs = _run_dflash_builder(info, use_sparse_topk=False)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(BS, DRAFT_TOKEN_NUM))
    _close(probs, _reference_target_probs(apply_min_p=False, top_ps=torch.ones(BS)))


# ------------------------------------------------------- DSpark selector route


def _run_accept_sampling_core(sampling_info):
    """Run the selector route's core with the chain kernel faked out, and
    report both what it handed the kernel and whether the shared builder ran."""
    candidates = torch.arange(ROWS, dtype=torch.int64).view(BS, DRAFT_TOKEN_NUM)
    draft_probs = torch.full((BS, DRAFT_TOKEN_NUM, VOCAB), 1.0 / VOCAB)
    captured = {}

    def _chain(**kwargs):
        captured["target_probs"] = kwargs["target_probs"]
        kwargs["predicts"].zero_()
        kwargs["accept_index"].fill_(-1)
        kwargs["accept_token_num"].zero_()

    builder = mock.MagicMock(wraps=build_dflash_verify_target_probs)
    with mock.patch.object(
        dspark_accept, "chain_speculative_sampling_triton", _chain
    ), mock.patch.object(dspark_accept, "build_dflash_verify_target_probs", builder):
        dspark_accept._accept_sampling_core(
            candidates=candidates,
            target_logits=LOGITS.clone(),
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=SimpleNamespace(
                max_top_k=None, uniform_top_k_value=None
            ),
            gamma=DRAFT_TOKEN_NUM,
            verify_num_draft_tokens=DRAFT_TOKEN_NUM,
            cutoff_verify_lens=None,
        )
    captured["builder_calls"] = builder.call_count
    return captured


def test_selector_route_sends_min_p_only_requests_to_the_builder():
    all_out = torch.tensor([VOCAB, VOCAB], dtype=torch.int32)
    ones = torch.ones(BS)
    info = _sampling_info(top_ks=all_out, top_ps=ones, min_ps=torch.tensor([0.0, 0.5]))
    out = _run_accept_sampling_core(info)
    rows = out["target_probs"].reshape(ROWS, VOCAB)
    raw = torch.softmax(
        LOGITS / TEMPERATURES.repeat_interleave(DRAFT_TOKEN_NUM, 0), -1
    )
    # The base route saw no filter needed and took the plain softmax/temperature
    # shortcut, so this row was the unfiltered distribution.
    assert out["builder_calls"] == 1
    _close(rows[0], raw[0])
    _close(rows[2], _reference_min_p_renorm(raw[2:3], torch.tensor([0.5]))[0])
    assert int((rows[2] > 0).sum()) < VOCAB


def test_selector_route_keeps_the_unfiltered_shortcut_when_nothing_is_asked():
    all_out = torch.tensor([VOCAB, VOCAB], dtype=torch.int32)
    ones = torch.ones(BS)
    info = _sampling_info(top_ks=all_out, top_ps=ones, min_ps=ones * 0.0)
    out = _run_accept_sampling_core(info)
    assert out["builder_calls"] == 0  # the cheap path is still the cheap path
    raw = torch.softmax(
        LOGITS / TEMPERATURES.repeat_interleave(DRAFT_TOKEN_NUM, 0), -1
    )
    _close(out["target_probs"].reshape(ROWS, VOCAB), raw)


# ------------------------------------------------------- the shared stage


def test_the_shared_min_p_stage_is_the_samplers_rule():
    from sglang.srt.layers.sampler import min_p_normalize_probs_torch

    probs = torch.tensor([[0.6, 0.2, 0.15, 0.05], [0.9, 0.05, 0.04, 0.01]])
    out = min_p_normalize_probs_torch(probs, torch.tensor([0.4, 0.02]))
    _close(out[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    # 0.01 < 0.9 * 0.02, so the tail is cut and the rest is renormalized.
    _close(out[1], torch.tensor([0.9, 0.05, 0.04, 0.0]) / 0.99)
    # A row that loses nothing is returned unchanged (already a distribution).
    _close(min_p_normalize_probs_torch(probs, torch.zeros(2)), probs)
    # The cut is invariant to renormalizing first, which is why the stage's
    # position after top-k/top-p is not a correctness question.
    scaled = probs * 0.5
    _close(
        min_p_normalize_probs_torch(scaled, torch.tensor([0.4, 0.02])),
        min_p_normalize_probs_torch(probs, torch.tensor([0.4, 0.02])),
    )
