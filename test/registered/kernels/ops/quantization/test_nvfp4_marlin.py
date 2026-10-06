import math
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel.scalar_type import scalar_types

from sglang.srt.layers.quantization.marlin_utils import check_marlin_supported
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    nvfp4_marlin_process_global_scale,
    prepare_nvfp4_layer_for_marlin,
)
from sglang.srt.speculative import proposal_head
from sglang.srt.utils.common import (
    is_sm80_supported,
    is_sm90_supported,
    is_sm120_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import make_nvfp4_weight_and_ref

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported() or is_sm120_supported()),
    reason="NVFP4 Marlin fallback tests require CUDA SM8X/SM9X/SM120",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_marlin_support_and_scale_transforms_sm80_sm90_sm120(dtype):
    major, minor = torch.cuda.get_device_capability()
    capability = major * 10 + minor
    assert check_marlin_supported(
        scalar_types.float4_e2m1f,
        group_size=16,
        has_zp=False,
        device_capability=capability,
    )

    global_scale = torch.tensor(1.0, dtype=dtype, device="cuda")
    actual_global_scale = nvfp4_marlin_process_global_scale(global_scale)
    assert actual_global_scale.is_cuda
    assert actual_global_scale.ndim == 1
    assert actual_global_scale.numel() == 1
    if dtype == torch.float16:
        assert actual_global_scale.item() == 128.0
    else:
        assert actual_global_scale.item() == 2.0**119


@pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported() or is_sm120_supported()),
    reason="NVFP4 Marlin dense numeric test requires CUDA SM80, SM86, SM90 or SM120",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_marlin_dense_matches_dequant_reference(dtype):
    torch.manual_seed(0)

    size_m = 17
    size_k = 256
    size_n = 192
    group_size = 16

    a_input = torch.randn((size_m, size_k), dtype=dtype, device="cuda") / 10
    fp4_weight, scales, global_scale, weight_ref = make_nvfp4_weight_and_ref(
        size_n, size_k, dtype, group_size=group_size
    )

    layer = torch.nn.Module()
    layer.quant_config = SimpleNamespace(group_size=group_size)
    layer.output_size_per_partition = size_n
    layer.input_size_per_partition = size_k
    layer.params_dtype = dtype
    layer.weight = torch.nn.Parameter(fp4_weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(
        global_scale.reshape(1), requires_grad=False
    )
    prepare_nvfp4_layer_for_marlin(layer)

    output = apply_fp4_marlin_linear(
        a_input,
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.workspace,
        size_n,
        size_k,
        use_fp32_reduce=True,
    )

    output_ref = torch.matmul(a_input, weight_ref.T)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, output_ref, rtol=0.04, atol=0.04)


# ---------------------------------------------------------------------------
# Optional FR-Spec proposal head (SGLANG_FR_SPEC_PROPOSAL_HEAD_PRECISION=nvfp4)
#
# Prepared, never run on a GPU yet: no speed or accept-rate claim comes out of
# these checks, only the contract that the quantized proposal head is the Marlin
# kernel's own dequantized weight, that it captures and replays, and that the
# target's storage is what the verifier still reads.
# ---------------------------------------------------------------------------

# The pinned 65,536-ID FR-Spec map and the Flash-Next hidden width. Both are
# exactly FP4/Marlin aligned, so the hot head must reach the kernel with no
# padded rows/columns. Batch rows: 1 decode, 4/12/16 the W4 verify widths and
# 24 C6's verification batch.
HOT_ROWS = 65_536
HOT_FEATURES = 2_560
HOT_BATCH_ROWS = [1, 4, 12, 16, 24]
PROPOSAL_HEAD_ENV = "SGLANG_FR_SPEC_PROPOSAL_HEAD_PRECISION"

CUDA_MARLIN = is_sm80_supported() or is_sm90_supported() or is_sm120_supported()
# Gated to the qualified part: the dense Marlin kernel itself also runs on
# SM80/SM90 (covered by the two checks above), but the FlashInfer online-NVFP4
# packer this feature uses is only as available as the fork's SM100/SM120 rule
# allows, and FR-Spec is an SM120 profile.
SM120_MARLIN = CUDA_MARLIN and is_sm120_supported()


def _normalized_error(actual: torch.Tensor, expected: torch.Tensor):
    assert actual.shape == expected.shape
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    error_rms = (actual_fp32 - expected_fp32).square().mean().sqrt().item()
    reference_rms = expected_fp32.square().mean().sqrt().item()
    cosine = F.cosine_similarity(
        actual_fp32.flatten(), expected_fp32.flatten(), dim=0
    ).item()
    return error_rms / max(reference_rms, 1e-8), cosine


@pytest.fixture(scope="module", params=["bf16", "rowwise_fp8"])
def frspec_proposal_head(request):
    """A draft lm_head prepared from a BF16 or an SM120 rowwise-FP8 head."""
    from sglang.kernels.ops.gemm.sm120_online_fp8 import (
        dequantize_rowwise_weight,
        replace_linear_weight_rowwise_fp8,
        rowwise_scale_of,
    )

    generator = torch.Generator(device="cuda").manual_seed(7)
    source = (
        torch.randn(HOT_ROWS, HOT_FEATURES, generator=generator, device="cuda")
        / math.sqrt(HOT_FEATURES)
    ).to(torch.bfloat16)
    # The target's own storage: the verifier reads this, so preparation must
    # never rewrite it in place.
    target_head = source.clone()
    layer = torch.nn.Module()
    if request.param == "rowwise_fp8":
        linear = torch.nn.Linear(
            HOT_FEATURES, HOT_ROWS, bias=False, dtype=torch.bfloat16, device="cuda"
        )
        linear.weight.data.copy_(source)
        replace_linear_weight_rowwise_fp8(linear)
        layer.weight = linear.weight
        reference = dequantize_rowwise_weight(linear.weight)
        source_scale = rowwise_scale_of(linear.weight)
    else:
        layer.weight = source.clone()
        reference = source.clone()
        source_scale = None

    with mock.patch.dict(os.environ, {PROPOSAL_HEAD_ENV: "nvfp4"}):
        prepared = proposal_head.prepare_nvfp4_proposal_head(
            layer, shared_tensors=(target_head,)
        )
    assert prepared
    return {
        "layer": layer,
        "reference": reference,
        "target_head": target_head,
        "source": source,
        "source_dtype": request.param,
        "source_scale": source_scale,
    }


@pytest.mark.skipif(
    not SM120_MARLIN, reason="FR-Spec NVFP4 proposal head is an exact-SM120 path"
)
def test_frspec_proposal_head_is_prepared_without_touching_the_target(
    frspec_proposal_head,
) -> None:
    head = frspec_proposal_head["layer"]
    assert head.weight.dtype == torch.int32
    assert head.quant_method.__class__.__name__ == "ModelOptNvFp4A16LinearMethod"
    assert head.input_size_per_partition == HOT_FEATURES
    assert head.output_size_per_partition == HOT_ROWS
    assert head.params_dtype in (torch.bfloat16, torch.float16)
    assert head.weight_global_scale.shape == (1,)
    # The split-K workspace is fixed by the device's SM count, not the batch, so
    # every captured graph shares one address and size.
    assert head.workspace.numel() == (
        torch.cuda.get_device_properties(head.workspace.device).multi_processor_count
    )
    # Proposal precision is not global precision: the target's resident head --
    # the verifier -- still holds every row it started with.
    torch.testing.assert_close(
        frspec_proposal_head["target_head"], frspec_proposal_head["source"]
    )
    if frspec_proposal_head["source_dtype"] == "rowwise_fp8":
        assert frspec_proposal_head["source_scale"].shape == (HOT_ROWS,)


@pytest.mark.skipif(
    not SM120_MARLIN, reason="FR-Spec NVFP4 proposal head is an exact-SM120 path"
)
@pytest.mark.parametrize("rows", HOT_BATCH_ROWS)
def test_frspec_proposal_head_logits_and_proposals_track_the_reference(
    frspec_proposal_head, rows: int
) -> None:
    head = frspec_proposal_head["layer"]
    generator = torch.Generator(device="cuda").manual_seed(100 + rows)
    hidden = (
        torch.randn(rows, HOT_FEATURES, generator=generator, device="cuda") * 0.25
    ).to(torch.bfloat16)

    logits = head.quant_method.apply(head, hidden)
    expected = hidden @ frspec_proposal_head["reference"].T

    nrmse, cosine = _normalized_error(logits, expected)
    # FP4 weight error shows up as ~the per-weight relative error on every
    # logit; these floors are wide enough for the quantizer and tight enough
    # that a mismatched group scale, global scale or packed row cannot pass.
    assert nrmse <= 0.30, f"NRMSE {nrmse:.6f} exceeds 0.30"
    assert cosine >= 0.95, f"cosine {cosine:.6f} is below 0.95"

    # Proposal-rate proxy on synthetic hidden states (what the draft would hand
    # the verifier): recorded, never asserted. Agreement is an observation about
    # a random tensor, not an invariant -- at rows=1 an exact argmax would be a
    # demand that FP4 losslessy preserve one maximum, which the allowed
    # approximation above does not imply. Real accept rates come from the
    # authorized model run.
    agreement = (logits.argmax(-1) == expected.argmax(-1)).float().mean().item()
    print(
        f"FR-Spec NVFP4 proposal head ({frspec_proposal_head['source_dtype']}, "
        f"rows={rows}): nrmse={nrmse:.4f} cosine={cosine:.4f} "
        f"top1_agreement={agreement:.4f}"
    )


@pytest.mark.skipif(
    not SM120_MARLIN, reason="FR-Spec NVFP4 proposal head is an exact-SM120 path"
)
@pytest.mark.parametrize("rows", [1, 4, 24])
def test_frspec_proposal_head_captures_and_replays(
    frspec_proposal_head, rows: int
) -> None:
    head = frspec_proposal_head["layer"]
    generator = torch.Generator(device="cuda").manual_seed(500 + rows)
    hidden = (
        torch.randn(rows, HOT_FEATURES, generator=generator, device="cuda") * 0.25
    ).to(torch.bfloat16)
    workspace_ptr = head.workspace.data_ptr()
    weight_ptr = head.weight.data_ptr()

    # JIT/compile and warm the allocator before capture, like the FP8 head does.
    for _ in range(2):
        eager = head.quant_method.apply(head, hidden)
    torch.cuda.synchronize()
    expected = eager.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = head.quant_method.apply(head, hidden)
    graph.replay()
    torch.cuda.synchronize()

    nrmse, _ = _normalized_error(graph_output.clone(), expected)
    assert nrmse <= 0.02, f"replayed NRMSE {nrmse:.6f} exceeds 0.02"
    # Replay-safe workspace: allocated during preparation (before capture), so
    # the graph never owns it and the counter buffer survives every replay.
    assert head.workspace.data_ptr() == workspace_ptr
    assert head.weight.data_ptr() == weight_ptr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
