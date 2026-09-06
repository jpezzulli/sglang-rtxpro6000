"""Tests for fused sigmoid gating delta rule MTP kernel (GDN target_verify).

Compares the fused kernel `fused_sigmoid_gating_delta_rule_update` against
the reference two-step implementation:
    1. g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    2. o = fused_recurrent_gated_delta_rule_update(q, k, v, g, beta, ...)
"""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

try:
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
        fused_recurrent_gated_delta_rule_update,
    )
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_recover_final_state,
        fused_sigmoid_gating_delta_rule_update,
    )

    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="nightly-amd-kernel-1-gpu", nightly=True)


def _make_tensors(N, T, H, HV, K, V, device="cuda", seed=2025):
    """Create input tensors for GDN target_verify."""
    torch.manual_seed(seed)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device=device)
    a = torch.randn(1, N * T, HV, dtype=torch.bfloat16, device=device)
    b = torch.randn(1, N * T, HV, dtype=torch.bfloat16, device=device)
    q = torch.randn(1, N * T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, N * T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, N * T, HV, V, dtype=torch.bfloat16, device=device)
    indices = torch.arange(N, dtype=torch.int32, device=device)
    initial_state = torch.randn(N, HV, K, V, dtype=torch.float, device=device)
    cu_seqlens = torch.arange(0, N * T + 1, T, dtype=torch.int32, device=device)
    return A_log, dt_bias, a, b, q, k, v, initial_state, indices, cu_seqlens


def run_reference(
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    initial_state_source,
    initial_state_indices,
    cu_seqlens,
    disable_state_update=True,
    intermediate_states_buffer=None,
    intermediate_state_indices=None,
    cache_steps=None,
    retrieve_parent_token=None,
):
    """Reference: fused_gdn_gating + fused_recurrent_gated_delta_rule_update."""
    # fused_gdn_gating expects 2D [seq_len, HV]
    a_2d = a.view(-1, a.shape[-1])
    b_2d = b.view(-1, b.shape[-1])
    g, beta = fused_gdn_gating(A_log, a_2d, b_2d, dt_bias)
    # fused_recurrent expects 3D [B, T, HV]
    g = g.view(a.shape)
    beta = beta.view(b.shape)

    # fused_recurrent requires intermediate_state_indices when cu_seqlens is used
    if cu_seqlens is not None and intermediate_state_indices is None:
        N = len(cu_seqlens) - 1
        intermediate_state_indices = torch.arange(N, dtype=torch.int32, device=q.device)

    return fused_recurrent_gated_delta_rule_update(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
        retrieve_parent_token=retrieve_parent_token,
    )


def run_fused_mtp(
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    initial_state_source,
    initial_state_indices,
    cu_seqlens,
    disable_state_update=True,
    intermediate_states_buffer=None,
    intermediate_state_indices=None,
    cache_steps=None,
    retrieve_parent_token=None,
    beta_in_activation_dtype=True,
):
    """Fused: fused_sigmoid_gating_delta_rule_update."""
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=False,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
        retrieve_parent_token=retrieve_parent_token,
        beta_in_activation_dtype=beta_in_activation_dtype,
    )


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernel not available")
@pytest.mark.parametrize("N,H,HV", [(1, 1, 1), (4, 16, 48)])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_target_verify_matches_packed_decode_beta_semantics(N, H, HV, state_dtype):
    """A one-token verify must preserve packed decode's beta rounding."""
    K = V = 128
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.ones(1, N, H, K, dtype=dtype, device=device)
    k = torch.ones_like(q)
    v = torch.ones(1, N, HV, V, dtype=dtype, device=device)
    mixed_qkv = torch.cat(
        (q.reshape(N, -1), k.reshape(N, -1), v.reshape(N, -1)), dim=-1
    ).contiguous()
    a = torch.zeros(N, HV, dtype=dtype, device=device)
    # sigmoid(-0.5) needs BF16 rounding in the ordinary packed-decode path.
    b = torch.full_like(a, -0.5)
    A_log = torch.zeros(HV, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(HV, dtype=dtype, device=device)
    indices = torch.arange(N, dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(N + 1, dtype=torch.int32, device=device)
    packed_state = torch.zeros(N, HV, V, K, dtype=state_dtype, device=device)
    packed_out = torch.empty(N, 1, HV, V, dtype=dtype, device=device)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=packed_state,
        out=packed_out,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )
    verify_state = torch.zeros_like(packed_state)
    intermediate = torch.empty(N, 1, HV, V, K, dtype=state_dtype, device=device)
    verify_out = run_fused_mtp(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        verify_state,
        indices,
        cu_seqlens,
        disable_state_update=True,
        intermediate_states_buffer=intermediate,
        intermediate_state_indices=indices,
        cache_steps=1,
    )
    torch.testing.assert_close(
        verify_out.reshape_as(packed_out), packed_out, rtol=0, atol=0
    )
    torch.testing.assert_close(intermediate[:, 0], packed_state, rtol=0, atol=0)
    torch.testing.assert_close(
        verify_state, torch.zeros_like(verify_state), rtol=0, atol=0
    )


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernel not available")
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("boundary_checkpoint", [False, True])
@pytest.mark.parametrize("beta_in_activation_dtype", [False, True])
def test_gdn_recovery_matches_verified_prefix(
    state_dtype, boundary_checkpoint, beta_in_activation_dtype
):
    """Accepted and radix-boundary recovery must commit verify's recurrence."""
    N, T, H, HV, K, V = 4, 4, 16, 48, 128, 128
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.ones(1, N * T, H, K, dtype=dtype, device=device)
    k = torch.ones_like(q)
    v = torch.ones(1, N * T, HV, V, dtype=dtype, device=device)
    a = torch.zeros(N * T, HV, dtype=dtype, device=device)
    b = torch.full_like(a, -0.5)
    A_log = torch.zeros(HV, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(HV, dtype=dtype, device=device)
    source_indices = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=device)
    accepted_steps = torch.arange(N, dtype=torch.int32, device=device)
    output_indices = (
        torch.tensor([6, 4, -1, 5], dtype=torch.int32, device=device)
        if boundary_checkpoint
        else source_indices
    )
    state = torch.zeros(8, HV, V, K, dtype=state_dtype, device=device)
    state[4:] = -7  # Unused slots and skipped checkpoint rows must be untouched.
    original = state.clone()
    intermediate = torch.empty(N, T, HV, V, K, dtype=state_dtype, device=device)
    run_fused_mtp(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state,
        source_indices,
        torch.arange(0, N * T + 1, T, dtype=torch.int32, device=device),
        intermediate_states_buffer=intermediate,
        intermediate_state_indices=torch.arange(N, dtype=torch.int32, device=device),
        cache_steps=T,
        beta_in_activation_dtype=beta_in_activation_dtype,
    )
    torch.testing.assert_close(state, original, rtol=0, atol=0)
    fused_sigmoid_gating_delta_rule_recover_final_state(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        k=k,
        v=v,
        b=b,
        initial_state_source=state,
        initial_state_indices=source_indices,
        accepted_steps=accepted_steps,
        cache_steps=T,
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
        output_state_indices=output_indices if boundary_checkpoint else None,
        beta_in_activation_dtype=beta_in_activation_dtype,
    )
    expected = original.clone()
    for row, out_index in enumerate(output_indices.tolist()):
        if out_index >= 0:
            expected[out_index] = intermediate[row, row]
    torch.testing.assert_close(state, expected, rtol=0, atol=1e-7)


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernel not available")
def test_gdn_verify_preserves_unrounded_beta_contract():
    """Mixed FlashInfer decode/Triton verify retains the pre-fix FP32 beta."""
    N, T, H, HV, K, V = 1, 1, 1, 1, 128, 128
    args = _make_tensors(N, T, H, HV, K, V)
    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = args
    A_log.zero_()
    dt_bias.zero_()
    a.zero_()
    b.fill_(-0.5)
    q.fill_(1)
    k.fill_(1)
    v.fill_(1)
    state.zero_()
    intermediate = torch.empty(N, T, HV, V, K, device="cuda")
    run_fused_mtp(
        *args[:2],
        q,
        k,
        v,
        a,
        b,
        state,
        indices,
        cu_seqlens,
        intermediate_states_buffer=intermediate,
        intermediate_state_indices=indices,
        cache_steps=T,
        beta_in_activation_dtype=False,
    )
    beta_fp32 = torch.sigmoid(b.float()).item()
    expected = beta_fp32 / (K + 1e-6) ** 0.5
    torch.testing.assert_close(
        intermediate, torch.full_like(intermediate, expected), rtol=0, atol=1e-7
    )


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernel not available")
@pytest.mark.parametrize("N", [1, 8, 16])
@pytest.mark.parametrize("T", [1, 4, 8])
def test_fused_gdn_mtp_precision(N: int, T: int):
    """Compare fused MTP output against reference."""
    H, HV, K, V = 16, 32, 128, 128

    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = _make_tensors(
        N, T, H, HV, K, V
    )

    state_ref = state.clone()
    state_fused = state.clone()

    out_ref = run_reference(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state_ref,
        indices,
        cu_seqlens,
        disable_state_update=True,
    )
    out_fused = run_fused_mtp(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state_fused,
        indices,
        cu_seqlens,
        disable_state_update=True,
    )

    torch.testing.assert_close(out_ref, out_fused, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")
@pytest.mark.parametrize("N", [1, 16, 128])
def test_mtp_single_step_decode(N: int):
    """Verify MTP kernel matches reference for T=1 (decode scenario)."""
    T = 1
    H, HV, K, V = 16, 32, 128, 128

    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = _make_tensors(
        N, T, H, HV, K, V
    )

    state_ref = state.clone()
    state_fused = state.clone()

    out_ref = run_reference(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state_ref,
        indices,
        cu_seqlens,
        disable_state_update=False,
    )
    out_fused = run_fused_mtp(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state_fused,
        indices,
        cu_seqlens,
        disable_state_update=False,
    )

    torch.testing.assert_close(out_ref, out_fused, rtol=1e-2, atol=1e-2)

    # Also verify states match after update
    state_diff = (state_ref.float() - state_fused.float()).abs()
    state_max_diff = state_diff.max().item()
    state_fail_rate = (state_diff > 0.1).float().mean().item() * 100
    print(
        f"  single_step state N={N}: max_diff={state_max_diff:.2e}, "
        f"fail_rate={state_fail_rate:.2f}%"
    )
    assert state_fail_rate < 0.01, f"State mismatch: fail_rate={state_fail_rate:.2f}%"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
