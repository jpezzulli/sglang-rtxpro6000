"""CPU-only TP>1 NCCL P2P startup-guidance checks (item 3).

Guidance fires from ``init_torch_distributed`` before torch's
``init_process_group`` can turn into a silent transport hang; it never sets
or mutates NCCL/BIOS state and stays completely silent at TP1 (the
qualified Pennyroyal default).
"""

from sglang.srt.distributed.p2p_guidance import p2p_startup_guidance
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_tp1_is_silent():
    assert (
        p2p_startup_guidance(tp_size=1, backend="nccl", env={}) is None
    ), "a TP1 launch must not see any P2P guidance it could misread"


def test_tp2_guidance_before_distributed_init_mentions_p2p_workaround():
    message = p2p_startup_guidance(tp_size=2, backend="nccl", env={})
    assert message is not None
    assert "NCCL_P2P_DISABLE=1" in message
    # The performance caveat is part of the contract, not a footnote.
    assert "host memory" in message and ("throughput" in message or "latency" in message)
    # It guides the operator; it does not claim a diagnosis or change state.
    assert "can help isolate" in message
    # No overclaiming: this host's peer access and TP2 itself are unverified.
    assert "not verify GPU peer access" in message
    assert "not yet hardware-qualified" in message
    assert "qualifies" not in message and "qualified default" not in message


def test_operator_workaround_is_recognized_with_its_caveat():
    message = p2p_startup_guidance(
        tp_size=2, backend="nccl", env={"NCCL_P2P_DISABLE": "1"}
    )
    assert message is not None
    assert "NCCL_P2P_DISABLE=1 is exported" in message
    assert "host memory" in message


def test_only_the_exact_nccl_value_1_claims_p2p_disabled():
    # NCCL parses other values its own way; we never claim it is disabled.
    for value in ("0", "2", "true", "1 ", ""):
        message = p2p_startup_guidance(
            tp_size=2, backend="nccl", env={"NCCL_P2P_DISABLE": value}
        )
        assert message is not None
        assert "is exported" not in message, value


def test_only_first_rank_of_first_node_logs():
    kwargs = dict(tp_size=2, backend="nccl", env={})
    assert p2p_startup_guidance(**kwargs, node_rank=0, tp_rank=0) is not None
    assert p2p_startup_guidance(**kwargs, node_rank=0, tp_rank=1) is None
    assert p2p_startup_guidance(**kwargs, node_rank=1, tp_rank=0) is None


def test_non_nccl_backends_keep_their_own_transport_messaging():
    assert (
        p2p_startup_guidance(tp_size=2, backend="mooncake", env={}) is None
    )
    assert p2p_startup_guidance(tp_size=4, backend="gloo", env={}) is None


def test_bootstrap_calls_the_guidance_before_process_group_init():
    """The call site is the one that runs *before* init_process_group."""
    import inspect

    from sglang.srt.distributed import bootstrap

    source = inspect.getsource(bootstrap.init_torch_distributed)
    call = source.index("p2p_startup_guidance(")
    groups = source.index("_init_parallel_groups(")
    assert call < groups, "P2P guidance must precede distributed group init"
    assert "if not is_draft_worker:" in source, (
        "guidance logs once per scheduler role, not once per worker "
        "(draft runners re-enter init_torch_distributed)"
    )
    assert "logger.warning(guidance)" in source, (
        "the hint must surface through the scheduler logger, not stdout"
    )


def test_guidance_never_mutates_environment():
    import os


    before = dict(os.environ)
    assert (
        p2p_startup_guidance(tp_size=2, backend="nccl", node_rank=0, tp_rank=0)
        is not None
    )
    assert os.environ == before, "guidance only: no NCCL/P2P env writes"
