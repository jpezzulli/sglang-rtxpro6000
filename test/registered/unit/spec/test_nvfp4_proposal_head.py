"""CPU contract coverage for the optional NVFP4 FR-Spec proposal head.

GPU numerics live with the existing NVFP4 Marlin suite; what is checkable
without a device is the contract: the gate defaults off and names its exact
values, preparation installs the whole ``ModelOptNvFp4A16LinearMethod`` state
through the real (kernel-patched) ``prepare_nvfp4_layer_for_marlin``, the logits
processor would actually dispatch the prepared head, the target's shared storage
is never quantized, and the identity string the launcher pins is the one the
runtime demands.
"""

import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    _SCALE_ATTR,
    dequantize_rowwise_weight,
    replace_linear_weight_rowwise_fp8,
    rowwise_scale_of,
)
from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.layers.quantization import marlin_utils_fp4
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptNvFp4A16LinearMethod,
)
from sglang.srt.speculative import proposal_head
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ENV = "SGLANG_FR_SPEC_PROPOSAL_HEAD_PRECISION"
ROWS = 128
COLUMNS = 64  # a multiple of the NVFP4 group width, like the real K = 2,560
LAUNCHERS = (
    "configs/pennyroyal/serve-flash-next-frspec.sh",
    "docker/pennyroyal/launch/config/start-flash-next-frspec.sh",
)


def _bf16_head(rows=ROWS, columns=COLUMNS, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, columns, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )


def _rowwise_fp8_head(rows=ROWS, columns=COLUMNS, seed=0):
    linear = nn.Linear(columns, rows, bias=False, dtype=torch.bfloat16)
    linear.weight.data.copy_(_bf16_head(rows, columns, seed))
    replace_linear_weight_rowwise_fp8(linear)
    return linear.weight


def _draft_head_layer(weight):
    """A stand-in for the draft's ParallelLMHead after the FR-Spec share."""
    layer = nn.Module()
    layer.weight = weight
    return layer


@contextlib.contextmanager
def _fake_cuda_kernels():
    """CPU stand-ins for the two CUDA-only helpers preparation touches.

    ``gptq_marlin_repack`` is a JIT CUDA op and ``marlin_make_workspace`` reads
    the SM count; everything else -- the shape assert, the padding decision, the
    scale permutation, the E4M3 and global-scale transforms -- is the real
    ``prepare_nvfp4_layer_for_marlin``.
    """
    calls = []

    def fake_repack(*, b_q_weight, perm, size_k, size_n, num_bits):
        calls.append(
            {
                "input_shape": tuple(b_q_weight.shape),
                "input_dtype": b_q_weight.dtype,
                "size_k": size_k,
                "size_n": size_n,
                "num_bits": num_bits,
            }
        )
        pack_factor = 32 // num_bits
        return torch.zeros(
            (size_k // 16, size_n * 16 // pack_factor),
            dtype=b_q_weight.dtype,
            device=b_q_weight.device,
        )

    def fake_workspace(device, max_blocks_per_sm=1):
        return torch.zeros(8 * max_blocks_per_sm, dtype=torch.int, device=device)

    # gptq_marlin_repack is only imported into that module on a CUDA host.
    with mock.patch.object(
        marlin_utils_fp4, "gptq_marlin_repack", fake_repack, create=True
    ), mock.patch.object(marlin_utils_fp4, "marlin_make_workspace", fake_workspace):
        yield calls


def _fake_quantizer(records):
    """A deterministic stand-in for the FlashInfer NVFP4 packer."""

    def quantize(weight, weight_scale_2=None, backend="cute-dsl"):
        records.append(
            {
                "weight": weight.clone(),
                "shape": tuple(weight.shape),
                "dtype": weight.dtype,
                "backend": backend,
            }
        )
        rows, columns = weight.shape
        float_weight = weight.float()
        amax = float_weight.abs().amax().clamp_min(1e-6)
        # Two E2M1 values per byte: low nibble = even column, high = odd.
        low = (float_weight[:, 0::2].clamp(-1, 1) * 127).to(torch.uint8)
        high = (float_weight[:, 1::2].clamp(-1, 1) * 127).to(torch.uint8)
        return (
            low | (high << 4),
            float_weight.abs()
            .reshape(rows, columns // 16, 16)
            .amax(2)
            .div(448.0)
            .clamp_min(1e-6)
            .to(torch.float8_e4m3fn),
            (amax / (448.0 * 6.0)).reshape(()),
        )

    return quantize


# ---------------------------------------------------------------------------
# The gate and its identity
# ---------------------------------------------------------------------------


def test_precision_gate_defaults_off_and_names_the_exact_values(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert proposal_head.proposal_head_precision() == proposal_head.OFF
    assert not proposal_head.nvfp4_proposal_head_enabled()
    assert proposal_head.namespace_field_value() is None

    monkeypatch.setenv(ENV, " nvfp4 ")
    assert proposal_head.proposal_head_precision() == proposal_head.NVFP4
    assert proposal_head.nvfp4_proposal_head_enabled()
    assert proposal_head.namespace_field_value() == proposal_head.NVFP4_NAMESPACE_LABEL

    monkeypatch.setenv(ENV, "fp4")
    with pytest.raises(RuntimeError, match=ENV):
        proposal_head.proposal_head_precision()


def test_the_launchers_pin_the_identity_the_runtime_demands():
    # Bash cannot import the Python constant, so pin both spellings here: the
    # runtime refuses any root whose pinned value differs, which turns drift
    # into a boot failure instead of a silently shared cache namespace.
    expected = (
        f'--field "{proposal_head.PROPOSAL_HEAD_NAMESPACE_FIELD}'
        f'={proposal_head.NVFP4_NAMESPACE_LABEL}"'
    )
    root = Path(__file__).resolve().parents[4]
    for relative in LAUNCHERS:
        source = (root / relative).read_text(encoding="utf-8")
        assert expected in source, relative
        assert ENV in source, relative


# ---------------------------------------------------------------------------
# The prepared-module contract
# ---------------------------------------------------------------------------


def test_preparation_installs_the_full_marlin_contract(monkeypatch):
    monkeypatch.setenv(ENV, "nvfp4")
    records = []
    monkeypatch.setattr(proposal_head, "_nvfp4_quantize", _fake_quantizer(records))
    layer = _draft_head_layer(_bf16_head())
    target_head = _bf16_head(seed=1)

    with _fake_cuda_kernels() as repacks:
        assert proposal_head.prepare_nvfp4_proposal_head(
            layer, shared_tensors=(target_head,)
        )

    # The packer sees the draft's own rows, at the logical hot shape.
    assert [record["shape"] for record in records] == [(ROWS, COLUMNS)]
    assert records[0]["dtype"] == torch.bfloat16
    assert records[0]["backend"] in ("cute-dsl", "cuda")
    assert layer.input_size_per_partition == COLUMNS
    assert layer.output_size_per_partition == ROWS
    assert layer.params_dtype == torch.bfloat16
    assert layer.quant_config.group_size == proposal_head.NVFP4_GROUP_SIZE
    assert isinstance(layer.quant_method, ModelOptNvFp4A16LinearMethod)
    assert len(repacks) == 1
    assert repacks[0]["size_k"] == COLUMNS and repacks[0]["size_n"] == ROWS
    assert repacks[0]["num_bits"] == 4
    assert repacks[0]["input_dtype"] == torch.int32
    # Values, group scales, the tensor-wide scale and the graph-stable workspace
    # all live on the layer. A ``.weight`` swap alone would leave the target's
    # 248,320-row scales and stale geometry behind.
    assert layer.weight.dtype == torch.int32
    assert layer.weight_scale.dtype == torch.float8_e4m3fn
    # The tensor-wide scale arrives FP32 from the converter and leaves in the
    # activation dtype with Marlin's exponent bias folded in (1-D, one element:
    # the kernel broadcasts it over the whole projection).
    assert layer.weight_global_scale.dtype == torch.bfloat16
    assert layer.weight_global_scale.shape == (1,)
    assert layer.workspace.dtype == torch.int
    assert not hasattr(layer, "weight_scale_2")
    assert should_apply_lm_head_quant_method(layer, layer.quant_method)


def test_the_logits_processor_predicate_is_the_prepared_contract_check(monkeypatch):
    """A half-prepared head must not be dispatchable, which is what boot checks."""
    monkeypatch.setenv(ENV, "nvfp4")
    records = []
    monkeypatch.setattr(proposal_head, "_nvfp4_quantize", _fake_quantizer(records))
    layer = _draft_head_layer(_bf16_head())
    with _fake_cuda_kernels():
        proposal_head.prepare_nvfp4_proposal_head(layer)
    method = ModelOptNvFp4A16LinearMethod(
        ModelOptFp4Config(is_checkpoint_nvfp4_serialized=False, group_size=16)
    )
    assert should_apply_lm_head_quant_method(layer, method)
    layer.weight = torch.nn.Parameter(
        layer.weight.view(torch.uint8), requires_grad=False
    )
    assert not should_apply_lm_head_quant_method(layer, method)


def test_a_rowwise_fp8_head_is_requantized_from_its_own_rows(monkeypatch):
    """Values and row scales travel together; the FP8 source stays untouched."""
    monkeypatch.setenv(ENV, "nvfp4")
    records = []
    monkeypatch.setattr(proposal_head, "_nvfp4_quantize", _fake_quantizer(records))
    weight = _rowwise_fp8_head()
    expected = dequantize_rowwise_weight(weight)
    layer = _draft_head_layer(weight)

    with _fake_cuda_kernels():
        proposal_head.prepare_nvfp4_proposal_head(layer)

    assert len(records) == 1
    torch.testing.assert_close(records[0]["weight"], expected)
    assert layer.weight.dtype == torch.int32


def test_off_leaves_the_shared_head_untouched(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    weight = _rowwise_fp8_head()
    layer = _draft_head_layer(weight)

    with _fake_cuda_kernels() as repacks:
        assert not proposal_head.prepare_nvfp4_proposal_head(layer)

    assert not repacks
    assert layer.weight is weight
    assert rowwise_scale_of(layer.weight) is not None
    assert not hasattr(layer, "weight_scale")
    assert not hasattr(layer, "workspace")


def test_preparation_refuses_storage_shared_with_the_target(monkeypatch):
    monkeypatch.setenv(ENV, "nvfp4")
    target_head = _bf16_head()
    embedding = _bf16_head(seed=3)
    layer = _draft_head_layer(target_head)

    with _fake_cuda_kernels() as repacks:
        with pytest.raises(RuntimeError, match="shared"):
            proposal_head.prepare_nvfp4_proposal_head(
                layer, shared_tensors=(target_head, embedding)
            )

    assert not repacks
    assert layer.weight is target_head
    torch.testing.assert_close(target_head, _bf16_head())


def test_preparation_guards_geometry_and_a_stale_row_scale(monkeypatch):
    monkeypatch.setenv(ENV, "nvfp4")
    records = []
    monkeypatch.setattr(proposal_head, "_nvfp4_quantize", _fake_quantizer(records))
    unaligned = _draft_head_layer(_bf16_head(ROWS, COLUMNS - 1))
    with _fake_cuda_kernels():
        with pytest.raises(RuntimeError, match=str(proposal_head.NVFP4_GROUP_SIZE)):
            proposal_head.prepare_nvfp4_proposal_head(unaligned)
        with pytest.raises(RuntimeError, match="2-D lm_head"):
            proposal_head.prepare_nvfp4_proposal_head(_draft_head_layer(None))
    assert not records


def test_a_half_prepared_head_fails_boot_instead_of_projecting_stale_state():
    """_require_prepared_contract asks the dispatcher's own predicate."""
    method = ModelOptNvFp4A16LinearMethod(
        ModelOptFp4Config(is_checkpoint_nvfp4_serialized=False, group_size=16)
    )
    # BF16 storage under a Marlin method: should_apply_lm_head_quant_method
    # answers False, and boot must refuse instead of falling back silently.
    layer = _draft_head_layer(_bf16_head())
    layer.weight_scale = layer.weight_global_scale = layer.workspace = _bf16_head(1, 1)
    layer.input_size_per_partition = COLUMNS
    layer.output_size_per_partition = ROWS
    with pytest.raises(RuntimeError, match="not fully prepared"):
        proposal_head._require_prepared_contract(layer, method)


def test_a_mismatched_rowwise_scale_fails_boot(monkeypatch):
    monkeypatch.setenv(ENV, "nvfp4")
    weight = _rowwise_fp8_head()
    setattr(weight, _SCALE_ATTR, torch.ones(weight.shape[0] + 1, dtype=torch.float32))
    with _fake_cuda_kernels():
        with pytest.raises(RuntimeError, match="rowwise"):
            proposal_head.prepare_nvfp4_proposal_head(_draft_head_layer(weight))


# ---------------------------------------------------------------------------
# Worker wiring
# ---------------------------------------------------------------------------


class _NonEagle3:
    @staticmethod
    def is_eagle3():
        return False


class _DraftHeadLayer(nn.Module):
    def __init__(self, rows, columns):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(rows, columns, dtype=torch.bfloat16), requires_grad=False
        )


class _DraftModel:
    """The MTP draft: its own ParallelLMHead, target embedding shared in."""

    hot_token_id = None

    def __init__(self, rows, columns):
        self.lm_head = _DraftHeadLayer(rows, columns)
        self.logits_processor = None
        self.embed = None

    def set_embed_and_head(self, embed, head):
        # Mirrors Qwen3_5ForCausalLMMTP.set_embed_and_head: the draft drops its
        # own head Parameter before the selected rows go in.
        del self.lm_head.weight
        self.embed = embed
        self.lm_head.weight = head


def _worker(target_head_weight, embedding, hot_ids, draft_model):
    return SimpleNamespace(
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    lm_head=None,
                    get_embed_and_head=lambda: (embedding, target_head_weight),
                )
            )
        ),
        draft_runner=SimpleNamespace(model=draft_model),
        hot_token_id=hot_ids,
        speculative_algorithm=_NonEagle3(),
    )


def test_init_lm_head_prepares_the_drafts_own_head_in_map_order(monkeypatch):
    columns = COLUMNS
    hot_ids = torch.tensor([7, 3, 1, 0, 2, 6, 5, 4])
    target_head = _bf16_head(ROWS, columns)
    draft_model = _DraftModel(hot_ids.shape[0], columns)
    worker = _worker(
        target_head, _bf16_head(ROWS, columns, seed=5), hot_ids, draft_model
    )
    records = []
    monkeypatch.setattr(proposal_head, "_nvfp4_quantize", _fake_quantizer(records))
    monkeypatch.setenv(ENV, "nvfp4")

    with _fake_cuda_kernels():
        EagleDraftWorker.init_lm_head(worker)

    assert records[0]["shape"] == (hot_ids.shape[0], columns)
    torch.testing.assert_close(records[0]["weight"], target_head[hot_ids])
    assert draft_model.lm_head.weight.dtype == torch.int32
    assert draft_model.lm_head.quant_method.__class__ is ModelOptNvFp4A16LinearMethod
    # The target's resident head -- the verifier -- keeps every row and its
    # original values, and the shared embedding is untouched as well.
    torch.testing.assert_close(target_head, _bf16_head(ROWS, columns))


def test_init_lm_head_default_keeps_the_rowwise_fp8_head(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    hot_ids = torch.tensor([2, 0, 1])
    target_head = _rowwise_fp8_head(ROWS, COLUMNS)
    installed = {}

    class _Model(_DraftModel):
        def set_embed_and_head(self, embed, head):
            installed["head"] = head

    worker = _worker(target_head, _bf16_head(seed=5), hot_ids, _Model(ROWS, COLUMNS))
    with _fake_cuda_kernels() as repacks:
        EagleDraftWorker.init_lm_head(worker)

    assert not repacks
    head = installed["head"]
    torch.testing.assert_close(
        rowwise_scale_of(head), rowwise_scale_of(target_head)[hot_ids]
    )


def test_init_lm_head_without_a_map_never_quantizes_the_shared_head(monkeypatch):
    """No FR-Spec map means the draft shares the target module: leave it alone."""
    monkeypatch.setenv(ENV, "nvfp4")
    target_head = _bf16_head(ROWS, COLUMNS)
    installed = {}

    class _Model(_DraftModel):
        def set_embed_and_head(self, embed, head):
            installed["head"] = head

    draft_model = _Model(ROWS, COLUMNS)
    draft_model.lm_head = target_head  # what set_lm_head_from_target does
    worker = _worker(target_head, _bf16_head(seed=5), None, draft_model)

    with _fake_cuda_kernels() as repacks:
        EagleDraftWorker.init_lm_head(worker)

    assert not repacks
    assert installed["head"] is target_head
    assert not hasattr(target_head, "quant_method")
    assert not hasattr(target_head, "weight_scale")
