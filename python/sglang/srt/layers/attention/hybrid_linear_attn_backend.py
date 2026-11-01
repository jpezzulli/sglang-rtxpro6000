from __future__ import annotations

import bisect
import logging
from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.kernels.ops.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.kernels.ops.mamba.mamba_state_indices_triton import (
    fused_replay_state_indices,
)
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    fused_conv_window_scatter_with_mask,
    fused_mamba_state_scatter_with_mask,
    scatter_mamba_states_after_mtp_verify,
    track_mamba_states_all_layers,
    track_mamba_states_if_needed,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.attention.linear.utils import pp_spec_stable_rows_enabled
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.layers.attention.mamba.prefill_track_metadata import (
    build_prefill_track_plan,
)
from sglang.srt.layers.attention.mamba.replay_state_indices_validator import (
    validate_replay_state_indices_cpu,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv.router import STATE_PART, KVLayer
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.ple_state_pool import NGRAM_PART, SHORT_CONV_PART
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_exec, get_memory, get_spec
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import is_cuda, is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.layers.attention.verify_mask import VerifyMask
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

logger = logging.getLogger(__name__)
_validate_mamba_replay_state_indices = (
    envs.SGLANG_VALIDATE_MAMBA_REPLAY_STATE_INDICES.get()
)


class MambaAttnBackendBase(AttentionBackend):
    supports_mis: bool = False

    @classmethod
    def validate_mis_support(cls, server_args) -> None:
        if server_args.enable_mis and not cls.supports_mis:
            raise ValueError(
                f"{cls.__name__} does not support multi-item scoring. "
                "Hybrid models require a linear-attention backend that explicitly "
                "declares MIS support."
            )

    # Per-slot accept lengths for the KDA fused-accept spec path; allocated only
    # by KDAAttnBackend where `_can_fuse_accept_state` holds. None everywhere
    # else — update_mamba_state_after_mtp_verify keys the fused branch on it.
    accept_lens_pool: Optional[torch.Tensor] = None

    # Most sequences the prefill graph tables hold (larger batches replay no
    # graph); set at init by backends whose supports_prefill_graph_extend holds.
    prefill_graph_max_seqs: int

    def __init__(self, model_runner: ModelRunner):
        self.validate_mis_support(model_runner.server_args)
        super().__init__()
        self.pad_slot_id = PAD_SLOT_ID
        self.device = model_runner.device
        self.topk = get_spec().speculative_eagle_topk or 0
        self.is_draft_worker = model_runner.is_draft_worker
        self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        # State views and local layer indices come from the same router binding.
        router = self.token_to_kv_pool.layer_router
        self.state_pool = router.part_view(STATE_PART)
        self.state_layer_indices = router.part_indices(STATE_PART)
        self.enable_unified_memory = get_memory().enable_unified_memory
        # model_config must not be touched here: backend selection reads the
        # linear_attn_backends stamp first, and that guard test constructs
        # backends on runners without a real model_config.
        self._model_runner = model_runner
        self._mamba_chunk_size: Optional[int] = None
        pool = self.req_to_token_pool
        # The request table's slot ids as the state pool indexes them: the
        # identity or one gather through the id space's table, both of which
        # `fused_replay_state_indices` reproduces in its own launch.
        self._state_ids = model_runner.runner_ids.state_ids(pool.state_sub_pool)
        self._fused_state_indices_ok = (
            torch.device(self.device).type == "cuda" and pool.holds_state_slots
        )
        self.forward_metadata: ForwardMetadata = None
        self.state_indices_list = []
        # Static (max_bs,) track-dest buffer captured by pointer, refreshed in-place
        # each replay; the captured track-save reads this, not the InputBuffer slot.
        self.mamba_track_indices_buf = None
        # Per-bs static write-cursor / force-flush buffers for cuda-graph; None
        # unless --enable-linear-replayssm is set.
        self.replayssm_write_pos_list = None
        self.replayssm_force_flush_list = None
        self.query_start_loc_list = []
        self.retrieve_next_token_list = []
        self.retrieve_next_sibling_list = []
        self.retrieve_parent_token_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self.conv_states_shape: tuple[int, int] = None
        # Constant (== 1) for mamba-like backends; hoisted so the replay path
        # skips the per-cycle method dispatch.
        self._graph_seq_len_fill_value = self.get_cuda_graph_seq_len_fill_value()

    def state_layer(self, layer_id: int) -> KVLayer:
        """The layer's recurrent-state route, once a layer-wise host load-back
        has brought the layer in."""
        return self.token_to_kv_pool.layer_router.read(layer_id, STATE_PART)

    def state_layer_cache(self, layer_id: int):
        """The layer's recurrent state, as the state pool slices it."""
        state = self.state_layer(layer_id)
        return state.view.mamba2_layer_cache(state.view_layer_id)

    def supports_prefill_graph_extend(self) -> bool:
        """Whether breakable prefill CUDA graphs can capture this backend's
        extend on static per-bucket tables instead of breaking at each layer."""
        return False

    def prefill_graph_extend_active(self) -> bool:
        """Whether the extend being captured or replayed runs on those tables."""
        return False

    def init_prefill_graph_metadata(self, forward_batch: ForwardBatch):
        raise NotImplementedError()

    def refresh_prefill_graph_metadata(self, meta, forward_batch: ForwardBatch):
        raise NotImplementedError()

    @property
    def mamba_chunk_size(self) -> int:
        if self._mamba_chunk_size is None:
            self._mamba_chunk_size = getattr(
                self._model_runner.model_config.hf_text_config, "mamba_chunk_size", 64
            )
        return self._mamba_chunk_size

    def _translate_mamba_indices(self, mamba_indices: torch.Tensor) -> torch.Tensor:
        """The request table's slot ids as the state pool indexes them (the
        identity unless its allocator hands out virtual slot ids). Must run
        everywhere mamba ids feed the SSM/conv kernels or mamba-pool state ops,
        incl. the cuda-graph replay-prep copy into ``state_indices_list``."""
        return self._state_ids.ids(mamba_indices)

    @staticmethod
    def _has_cpu_prefill_track_metadata(forward_batch: ForwardBatch) -> bool:
        return (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and all(
                values is not None and len(values) == forward_batch.batch_size
                for values in (
                    forward_batch.mamba_prefill_track_mask_cpu,
                    forward_batch.mamba_track_seqlens_cpu,
                    forward_batch.extend_seq_lens_cpu,
                    forward_batch.extend_prefix_lens_cpu,
                )
            )
        )

    def _forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size

        logical_mode = forward_batch.logical_forward_mode

        retrieve_next_token = None
        retrieve_next_sibling = None
        retrieve_parent_token = None
        track_conv_indices = None
        track_ssm_h_src = None
        track_chunk_idx = None
        track_ssm_h_dst = None
        track_ssm_h_batch_src = None
        track_ssm_final_src = None
        track_ssm_final_dst = None
        track_ssm_seq_idx = None
        track_ssm_end_locs = None
        track_ssm_recompute_dst = None
        logical_num_tokens = None
        track_mask_indices = None

        # Padded rows carry req-pool row 0, which maps to the reserved mamba
        # padding slot 0 in both the static and the unified pool.
        mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        )
        mamba_cache_indices = self._translate_mamba_indices(mamba_cache_indices)
        if forward_batch.mamba_track_indices is not None:
            forward_batch.mamba_track_indices = self._translate_mamba_indices(
                forward_batch.mamba_track_indices
            )
        # Resolve the tracked-row selection once per forward
        cpu_track_metadata = self._has_cpu_prefill_track_metadata(forward_batch)
        if cpu_track_metadata:
            rows = [
                i
                for i, track in enumerate(forward_batch.mamba_prefill_track_mask_cpu)
                if track
            ]
            has_mamba_track_mask = bool(rows)
            track_mask_indices = self._track_indices_to_device(rows) if rows else None
        else:
            has_mamba_track_mask = bool(
                forward_batch.mamba_track_mask is not None
                and forward_batch.mamba_track_mask.any()
            )
        replayssm_write_pos = None
        replayssm_force_flush = None
        if logical_mode.is_decode_or_idle():
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
            # The ring cursor is a per-slot decode counter shared by all GDN layers;
            # manage it once here (snapshot, hand to layers, advance mod L), not per-layer.
            # Gate on the linear_replayssm FLAG, not on cursor-tensor presence: the
            # spec-verify ring (--enable-linear-replayssm-spec) shares the write_pos
            # allocation but owns it exclusively via commit_gdn_replayssm_spec
            # (advance-by-accept-count once per verify step). Advancing it here as
            # well inserts one phantom/stale ring entry per step and cumulatively
            # poisons the reconstruction (degenerate repetition at 10k+ tokens).
            mamba_pool = self.state_pool
            write_pos_buf = (
                mamba_pool.replayssm_write_pos
                if mamba_pool is not None and mamba_pool.enable_linear_replayssm
                else None
            )
            if write_pos_buf is not None:
                slots = mamba_cache_indices.to(torch.long)
                # Padded rows point at the reserved padding slot 0; the clamp keeps
                # a -1 sentinel in bounds (the kernel zeroes rows with state_idx < 0).
                safe_slots = slots.clamp(min=0)
                replayssm_write_pos = write_pos_buf[safe_slots].clone()
                L = mamba_pool.linear_replayssm_cache_len
                # KDA has no radix coordination: flush only on the natural write_pos
                # == L-1 wrap. GDN adds the radix-aligned force-flush below.
                is_kda = mamba_pool.replayssm_is_kda
                # Force-flush on the radix track's seq_lens % mamba_track_interval
                # == 0 boundary so the ring folds into temporal[slot] when read.
                if not is_kda:
                    force_flush_bool = self._replayssm_track_flush_mask(
                        forward_batch.seq_lens_cpu, bs
                    )
                    replayssm_force_flush = force_flush_bool.to(
                        device=self.device, dtype=torch.int32
                    )
                # Advance only valid slots, scattered over unique slots (dup-index
                # race; padded rows share slot 0); a forced flush -> next write_pos 0.
                valid_mask = slots >= 0
                valid_slots = slots[valid_mask]
                if valid_slots.numel() > 0:
                    flushed = replayssm_write_pos == (L - 1)
                    if replayssm_force_flush is not None:
                        flushed = flushed | (replayssm_force_flush != 0)
                    next_pos = torch.where(
                        flushed,
                        torch.zeros_like(replayssm_write_pos),
                        (replayssm_write_pos + 1) % L,
                    )
                    # Dedup: rows sharing a slot share write_pos/flush, so the
                    # scattered value is identical regardless of which row wins.
                    uniq_slots, inv = torch.unique(valid_slots, return_inverse=True)
                    next_for_valid = next_pos[valid_mask]
                    new_vals = torch.empty(
                        uniq_slots.shape[0],
                        dtype=write_pos_buf.dtype,
                        device=write_pos_buf.device,
                    )
                    new_vals[inv] = next_for_valid.to(write_pos_buf.dtype)
                    write_pos_buf[uniq_slots] = new_vals
        elif logical_mode.is_extend(include_draft_extend_v2=True):
            has_extend_meta = (
                forward_batch.extend_start_loc is not None
                and forward_batch.extend_seq_lens is not None
            )
            if forward_batch.forward_mode.is_draft_extend_v2() and not has_extend_meta:
                # Draft-extend-v2 may omit linear metadata when the draft runs only
                # full-attention layers.
                query_start_loc = None
            elif forward_batch.forward_mode.is_target_verify():
                ragged_layout = forward_batch.ragged_verify_layout
                if ragged_layout is not None:
                    # Compact ragged verify: variable per-request verify lens.
                    query_start_loc = ragged_layout.qo_indptr_device
                else:
                    query_start_loc = torch.arange(
                        0,
                        forward_batch.input_ids.shape[0] + 1,
                        step=forward_batch.spec_info.draft_token_num,
                        dtype=torch.int32,
                        device=forward_batch.input_ids.device,
                    )

                if self.topk > 1:
                    retrieve_next_token = forward_batch.spec_info.retrieve_next_token
                    retrieve_next_sibling = (
                        forward_batch.spec_info.retrieve_next_sibling
                    )
                    # None during dummy run
                    if retrieve_next_token is not None:
                        retrieve_parent_token = torch.empty_like(retrieve_next_token)
            else:
                query_start_loc = torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                )
                query_start_loc[:bs] = forward_batch.extend_start_loc
                query_start_loc[bs] = (
                    forward_batch.extend_start_loc[-1]
                    + forward_batch.extend_seq_lens[-1]
                )
                if (
                    forward_batch.extend_seq_lens_cpu is not None
                    and len(forward_batch.extend_seq_lens_cpu) == bs
                    and forward_batch.tbo_parent_token_range is None
                ):
                    logical_num_tokens = sum(forward_batch.extend_seq_lens_cpu)
                else:
                    logical_num_tokens = int(query_start_loc[-1])
                if has_mamba_track_mask:
                    track_conv_indices = self._init_track_conv_indices(
                        query_start_loc, forward_batch, track_mask_indices
                    )

                    (
                        track_chunk_idx,
                        track_ssm_h_src,
                        track_ssm_h_dst,
                        track_ssm_h_batch_src,
                        track_ssm_final_src,
                        track_ssm_final_dst,
                        track_ssm_seq_idx,
                        track_ssm_end_locs,
                        track_ssm_recompute_dst,
                    ) = self._init_track_ssm_indices(mamba_cache_indices, forward_batch)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")

        return ForwardMetadata(
            query_start_loc=query_start_loc,
            logical_num_tokens=logical_num_tokens,
            mamba_track_mask_indices=track_mask_indices,
            mamba_cache_indices=mamba_cache_indices,
            # Physical track destinations (None when tracking off); cuda-graph
            # supplies this via the static backend buffer in _replay_metadata.
            mamba_track_indices=getattr(forward_batch, "mamba_track_indices", None),
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
            track_conv_indices=track_conv_indices,
            track_ssm_h_src=track_ssm_h_src,
            track_ssm_h_dst=track_ssm_h_dst,
            track_ssm_h_batch_src=track_ssm_h_batch_src,
            track_ssm_final_src=track_ssm_final_src,
            track_ssm_final_dst=track_ssm_final_dst,
            track_chunk_idx=track_chunk_idx,
            track_ssm_seq_idx=track_ssm_seq_idx,
            track_ssm_end_locs=track_ssm_end_locs,
            track_ssm_recompute_dst=track_ssm_recompute_dst,
            has_mamba_track_mask=has_mamba_track_mask,
            replayssm_write_pos=replayssm_write_pos,
            replayssm_force_flush=replayssm_force_flush,
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        self.forward_metadata = self._replay_metadata(
            forward_batch.batch_size,
            forward_batch.req_pool_indices,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            forward_batch.seq_lens_cpu if not in_capture else None,
            num_padding=(
                0 if in_capture else getattr(forward_batch, "num_padding", None)
            ),
            in_capture=in_capture,
            mamba_track_indices=getattr(forward_batch, "mamba_track_indices", None),
            ragged_verify_layout=forward_batch.ragged_verify_layout,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.forward_metadata = self._forward_metadata(forward_batch)

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        # Plan-stream fixup: slot indices / static query_start_loc are
        # draft-independent, but tree verify (topk > 1) copies the
        # draft-produced tree links into the captured buffers on the plan
        # stream, racing the draft — re-copy after the stream join. Eager
        # verify reads the spec_info tensors directly; parent links are
        # derived from these buffers at execution time.
        if self.topk <= 1 or cuda_graph_bs is None:
            return
        if (
            not isinstance(spec_info, EagleVerifyInput)
            or spec_info.retrieve_next_token is None  # dummy / capture runs
        ):
            return
        bs_without_pad = spec_info.retrieve_next_token.shape[0]
        self.retrieve_next_token_list[cuda_graph_bs - 1][:bs_without_pad].copy_(
            spec_info.retrieve_next_token
        )
        self.retrieve_next_sibling_list[cuda_graph_bs - 1][:bs_without_pad].copy_(
            spec_info.retrieve_next_sibling
        )

    def _init_track_conv_indices(
        self,
        query_start_loc: torch.Tensor,
        forward_batch: ForwardBatch,
        track_mask_indices: Optional[torch.Tensor] = None,
    ):
        """Flattened input positions of conv states to track during extend (up to
        the last complete chunk boundary, mamba_track_mask rows only)."""
        conv_state_len = self.conv_states_shape[-1]

        # Shared with the Qwen4-Exp PLE side states so the boundary can never
        # drift between them.
        aligned_len = forward_batch.mamba_track_aligned_lens()
        assert aligned_len is not None, (
            "conv-state tracking requires mamba_track_seqlens and extend_prefix_lens; "
            "this path should only run when the track mask is set on an extend batch"
        )
        start_indices = query_start_loc[:-1] + aligned_len - conv_state_len
        start_indices = (
            start_indices.index_select(0, track_mask_indices)
            if track_mask_indices is not None
            else start_indices[forward_batch.mamba_track_mask]
        )

        indices = start_indices.unsqueeze(-1) + torch.arange(
            conv_state_len,
            device=self.device,
            dtype=start_indices.dtype,
        )

        return indices.clamp(0, query_start_loc[-1] - 1)

    def _init_track_ssm_indices(
        self, mamba_cache_indices: torch.Tensor, forward_batch: ForwardBatch
    ):
        """src/dst indices to track SSM states for prefix caching: aligned seqs
        cache last_recurrent_state, unaligned cache intermediate `h` at the last
        chunk boundary. Also returns ``track_ssm_h_batch_src``: the batch rows of
        the unaligned tracked seqs, used to integer-index the fp32 snapshot
        buffer on the KDA path so the copy stays free of GPU syncs."""
        if self._has_cpu_prefill_track_metadata(forward_batch):
            return self._init_track_ssm_indices_from_cpu(
                mamba_cache_indices, forward_batch
            )
        state_chunk_size = self.mamba_chunk_size
        # CPU to avoid kernel launches for the masking ops
        mamba_track_mask = forward_batch.mamba_track_mask.cpu()
        extend_seq_lens = forward_batch.extend_seq_lens.cpu()
        mamba_track_indices = forward_batch.mamba_track_indices.cpu()
        mamba_cache_indices = mamba_cache_indices.cpu()
        mamba_track_seqlens = forward_batch.mamba_track_seqlens.cpu()
        prefix_lens = forward_batch.extend_prefix_lens.cpu()

        lens_to_track = mamba_track_seqlens - prefix_lens
        lens_masked = lens_to_track[mamba_track_mask]
        dst_masked = mamba_track_indices[mamba_track_mask]

        is_aligned = (lens_masked % state_chunk_size) == 0

        # Aligned: last_recurrent_state from ssm_states.
        track_ssm_final_src = mamba_cache_indices[mamba_track_mask][is_aligned]
        track_ssm_final_dst = dst_masked[is_aligned]

        # Unaligned: intermediate state from h.
        not_aligned = ~is_aligned
        track_ssm_h_dst = dst_masked[not_aligned]
        track_ssm_seq_idx = None
        track_ssm_end_locs = None
        track_ssm_recompute_dst = None

        if isinstance(self, Mamba2AttnBackend):
            # `h` is one chunk grid over the whole flattened batch, so the wanted
            # state is in it only when query_start_loc[i] is chunk-aligned.
            query_start_loc = torch.zeros_like(extend_seq_lens)
            query_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
            aligned_len = (
                lens_masked[not_aligned] // state_chunk_size
            ) * state_chunk_size
            seq_idx_unaligned = torch.nonzero(mamba_track_mask, as_tuple=True)[0][
                not_aligned
            ]
            end_locs = query_start_loc[seq_idx_unaligned] + aligned_len
            on_grid = (end_locs % state_chunk_size) == 0

            track_ssm_h_src = end_locs[on_grid] // state_chunk_size
            track_ssm_h_dst = track_ssm_h_dst[on_grid]

            track_ssm_seq_idx = seq_idx_unaligned[~on_grid]
            track_ssm_end_locs = end_locs[~on_grid]
            track_ssm_recompute_dst = dst_masked[not_aligned][~on_grid]
        else:
            num_h_states = (extend_seq_lens - 1) // state_chunk_size + 1
            track_ssm_src_offset = torch.zeros_like(num_h_states)
            track_ssm_src_offset[1:] = torch.cumsum(num_h_states[:-1], dim=0)
            track_ssm_h_src = track_ssm_src_offset[mamba_track_mask][not_aligned] + (
                lens_masked[not_aligned] // state_chunk_size
            )

        def to_device(t):
            return None if t is None else t.to(self.device, non_blocking=True)

        track_chunk_idx = torch.full((lens_to_track.shape[0],), -1, dtype=torch.int32)
        tracked_seqs = mamba_track_mask.nonzero(as_tuple=True)[0][not_aligned]
        track_chunk_idx[tracked_seqs] = (
            lens_masked[not_aligned] // state_chunk_size
        ).to(torch.int32)

        return (
            to_device(track_chunk_idx),
            to_device(track_ssm_h_src),
            to_device(track_ssm_h_dst),
            to_device(tracked_seqs),
            to_device(track_ssm_final_src),
            to_device(track_ssm_final_dst),
            to_device(track_ssm_seq_idx),
            to_device(track_ssm_end_locs),
            to_device(track_ssm_recompute_dst),
        )

    def _track_indices_to_device(self, values, dtype=torch.int64):
        return torch.tensor(
            values, dtype=dtype, pin_memory=is_pin_memory_available(self.device)
        ).to(self.device, non_blocking=True)

    def _init_track_ssm_indices_from_cpu(self, mamba_cache_indices, forward_batch):
        is_mamba2 = isinstance(self, Mamba2AttnBackend)
        plan = build_prefill_track_plan(
            forward_batch.mamba_prefill_track_mask_cpu,
            forward_batch.mamba_track_seqlens_cpu,
            forward_batch.extend_seq_lens_cpu,
            forward_batch.extend_prefix_lens_cpu,
            self.mamba_chunk_size,
            mamba2=is_mamba2,
        )
        to_device = self._track_indices_to_device
        final_rows = to_device(plan.final_rows)
        h_rows = to_device(plan.h_rows)
        recompute_rows = to_device(plan.recompute_rows) if is_mamba2 else None
        destinations = forward_batch.mamba_track_indices
        return (
            to_device(plan.chunk_indices, torch.int32),
            to_device(plan.h_src),
            destinations.index_select(0, h_rows),
            to_device(plan.unaligned_rows),
            mamba_cache_indices.index_select(0, final_rows),
            destinations.index_select(0, final_rows),
            recompute_rows,
            to_device(plan.recompute_end_locs) if is_mamba2 else None,
            destinations.index_select(0, recompute_rows) if is_mamba2 else None,
        )

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        self.forward_metadata = self._capture_metadata(
            bs, req_pool_indices, forward_mode, spec_info
        )

    def _replayssm_enabled(self) -> bool:
        """True iff --enable-linear-replayssm is on for this pool.

        Gate on the FLAG, not on ``replayssm_write_pos is not None``: the
        spec-verify ring (--enable-linear-replayssm-spec) also allocates the
        cursor tensor but owns it exclusively via commit_gdn_replayssm_spec.
        The decode-ring metadata machinery gated here (per-bs static cursor
        buffers, the per-replay snapshot + advance-by-one in _replay_metadata,
        and the decode-kernel ring rerouting downstream) must stay fully
        dormant for the spec ring.
        """
        mamba_pool = self.state_pool
        if mamba_pool is None:
            return False
        return bool(mamba_pool.enable_linear_replayssm)

    def _replayssm_track_flush_mask(
        self, seq_lens_cpu: torch.Tensor, bs: int
    ) -> torch.Tensor:
        """Per-row (length bs) bool flush mask = the radix track's seq_lens_cpu %
        mamba_track_interval == 0, so force-flush and snapshot fire on the same
        steps (no off-by-one)."""
        interval = get_exec().mamba.mamba_track_interval
        if seq_lens_cpu is None:
            # Should not happen for the supported config; stay safe and never flush.
            return torch.zeros((bs,), dtype=torch.bool)
        mask = (seq_lens_cpu[:bs].to(torch.int64) % interval) == 0
        if mask.shape[0] < bs:
            pad = torch.zeros((bs - mask.shape[0],), dtype=torch.bool)
            mask = torch.cat([mask, pad])
        return mask.cpu()

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        assert max_num_tokens % max_bs == 0, (
            f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        )
        draft_token_num = max_num_tokens // max_bs
        # Per-bs static write-cursor / force-flush buffers, captured by pointer +
        # refreshed in-place each replay; sized like state_indices_list. None when off.
        self.replayssm_write_pos_list = [] if self._replayssm_enabled() else None
        self.replayssm_force_flush_list = [] if self._replayssm_enabled() else None
        # int64 to match DecodeInputBuffers.mamba_track_indices + the track-save
        # kernel's int64 index load. Refreshed in-place by _replay_metadata.
        self.mamba_track_indices_buf = torch.zeros(
            (max_bs,), dtype=torch.int64, device=self.device
        )
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            if self.replayssm_write_pos_list is not None:
                self.replayssm_write_pos_list.append(
                    torch.zeros((i + 1,), dtype=torch.int32, device=self.device)
                )
            if self.replayssm_force_flush_list is not None:
                self.replayssm_force_flush_list.append(
                    torch.zeros((i + 1,), dtype=torch.int32, device=self.device)
                )
            self.query_start_loc_list.append(
                torch.zeros((i + 2,), dtype=torch.int32, device=self.device)
            )
            self.retrieve_next_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_next_sibling_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_parent_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cached_cuda_graph_verify_query_start_loc = torch.arange(
            0,
            max_bs * draft_token_num + 1,
            step=draft_token_num,
            dtype=torch.int32,
            device=self.device,
        )

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        assert max_num_tokens % max_bs == 0, (
            f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        )
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device=self.device)
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )

    def _capture_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        # Captured Mamba kernels read state_indices_list as PHYSICAL ids; translate
        # before copying (no-op for non-unified pool).
        mamba_indices = self._translate_mamba_indices(mamba_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)

        # Capture records the pointer to the static per-bs buffers; their zeros are
        # overwritten in-place by _replay_metadata before each replay. None when off.
        replayssm_write_pos = (
            self.replayssm_write_pos_list[bs - 1]
            if self.replayssm_write_pos_list is not None
            else None
        )
        replayssm_force_flush = (
            self.replayssm_force_flush_list[bs - 1]
            if self.replayssm_force_flush_list is not None
            else None
        )

        if forward_mode.is_target_verify() and self.topk > 1:
            # retrieve_* are None during capture, so skip the copy.
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        num_padding: Optional[int] = None,
        in_capture: bool = False,
        mamba_track_indices: Optional[torch.Tensor] = None,
        ragged_verify_layout: Optional[RaggedVerifyLayout] = None,
    ):
        if num_padding is None:
            if seq_lens_cpu is None:
                num_padding = 0
            else:
                num_padding = torch.count_nonzero(
                    seq_lens_cpu == self._graph_seq_len_fill_value
                )
        num_padding = int(num_padding)
        if self._fused_state_indices_ok and self.replayssm_write_pos_list is None:
            # Single-launch fast path: mapping gather + padding sentinel + store
            # into the static buffer, plus zeroing padded req_pool_indices rows —
            # bit-identical to the reference chain below.
            mamba_indices = fused_replay_state_indices(
                req_pool_indices=req_pool_indices,
                mamba_index_mapping=self.req_to_token_pool.req_index_to_mamba_index_mapping,
                out_state_indices=self.state_indices_list[bs - 1],
                valid_bs=bs - int(num_padding),
                total_bs=bs,
                v2p=self._state_ids.read_v2p,
            )
        else:
            # Make sure forward metadata is correctly handled for padding reqs
            req_pool_indices[bs - num_padding :] = 0
            mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
            # Translate using the LIVE v2p table BEFORE the padding sentinel below;
            # captured Mamba kernels read state_indices_list as PHYSICAL ids.
            mamba_indices = self._translate_mamba_indices(mamba_indices)
            mamba_indices[bs - num_padding :] = -1
            self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        if _validate_mamba_replay_state_indices and not in_capture:
            # This pre-replay diagnostic intentionally syncs to reject malformed
            # live or padded indices before a captured state update uses them.
            valid_bs = bs - int(num_padding)
            validate_replay_state_indices_cpu(
                mamba_indices.detach().cpu(),
                valid_bs=valid_bs,
                total_bs=bs,
                num_state_slots=self.state_pool.size + 1,
                pad_slot_id=self.pad_slot_id,
            )
        # Refresh the static track-dest buffer in-place (translated); the captured
        # track-save reads it, leaving the handed-in InputBuffer slot read-only.
        # Hand out only the refreshed [:bs] prefix — Mamba2's track-save slices
        # [-num_decodes:], which on the full max_bs buffer binds the stale tail.
        track_buf = None
        if mamba_track_indices is not None:
            assert len(mamba_track_indices) >= bs, (
                f"{len(mamba_track_indices)=} < {bs=}"
            )
            track_buf = self.mamba_track_indices_buf[:bs]
            track_buf.copy_(self._translate_mamba_indices(mamba_track_indices[:bs]))
        # Refresh the static write cursor in-place (mirrors the eager
        # snapshot-then-advance). Skip the advance during capture: dummy slots
        # would corrupt real ring positions.
        replayssm_write_pos = None
        replayssm_force_flush = None
        if self.replayssm_write_pos_list is not None:
            mamba_pool = self.state_pool
            write_pos_buf = mamba_pool.replayssm_write_pos
            static_wp = self.replayssm_write_pos_list[bs - 1]
            static_ff = self.replayssm_force_flush_list[bs - 1]
            # Hand the full captured per-bs buffers to the kernel; it indexes per row.
            replayssm_write_pos = static_wp
            replayssm_force_flush = static_ff
            if write_pos_buf is not None:
                # this replay's per-row physical slots (padded rows == -1)
                slots = mamba_indices.to(torch.long)
                safe_slots = slots.clamp(min=0)
                # Snapshot this step's cursor into the captured buffer in-place
                # (copy_, never reassign the object).
                static_wp[: len(mamba_indices)].copy_(write_pos_buf[safe_slots])
                # Refresh the force-flush buffer in-place from this step's seq_lens
                # (same condition as the radix track). Zeroed during capture.
                force_flush_dev = None
                # KDA: no radix coordination -> leave zeroed so the advance is a pure wrap.
                is_kda = mamba_pool.replayssm_is_kda
                if (
                    not is_kda
                    and forward_mode.is_decode_or_idle()
                    and seq_lens_cpu is not None
                ):
                    ff_mask = self._replayssm_track_flush_mask(seq_lens_cpu, bs)
                    force_flush_dev = ff_mask.to(device=self.device, dtype=torch.int32)
                    static_ff.copy_(force_flush_dev)
                else:
                    static_ff.zero_()
                # Defense in depth: the decode-ring advance is only meaningful for
                # decode/idle forwards (mirrors the eager path's gating). A
                # TARGET_VERIFY replay must never advance the cursor.
                if not in_capture and forward_mode.is_decode_or_idle():
                    L = mamba_pool.linear_replayssm_cache_len
                    # Advance only valid (non-padded) slots; a forced flush empties
                    # the ring -> next write_pos 0, like the natural L-1 wrap.
                    valid_mask = slots >= 0
                    valid_slots = slots[valid_mask]
                    if valid_slots.numel() > 0:
                        cur_pos = write_pos_buf[safe_slots]
                        flushed = cur_pos == (L - 1)
                        if force_flush_dev is not None:
                            flushed = flushed | (force_flush_dev != 0)
                        next_pos = torch.where(
                            flushed,
                            torch.zeros_like(cur_pos),
                            (cur_pos + 1) % L,
                        )
                        # Dedup; rows sharing a slot share write_pos+flush.
                        uniq_slots, inv = torch.unique(valid_slots, return_inverse=True)
                        next_for_valid = next_pos[valid_mask]
                        new_vals = torch.empty(
                            uniq_slots.shape[0],
                            dtype=write_pos_buf.dtype,
                            device=write_pos_buf.device,
                        )
                        new_vals[inv] = next_for_valid.to(write_pos_buf.dtype)
                        write_pos_buf[uniq_slots] = new_vals
        is_target_verify = forward_mode.is_target_verify()
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    bs - num_padding
                )
        elif forward_mode.is_target_verify():
            if ragged_verify_layout is not None:
                # Mamba kernels index dense [bs, N] scratch, so they need the
                # capped variant (see padded_to_bucket). Padding rows carry
                # mamba slot -1 and are skipped.
                if ragged_verify_layout.bs != bs or ragged_verify_layout.cap is None:
                    ragged_verify_layout = ragged_verify_layout.padded_to_bucket(
                        padded_bs=bs, cap=spec_info.draft_token_num
                    )
                self.query_start_loc_list[bs - 1].copy_(
                    ragged_verify_layout.qo_indptr_device
                )
            elif num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    (bs - num_padding) * spec_info.draft_token_num
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")
        qsl_buf = self.query_start_loc_list[bs - 1]

        if is_target_verify and self.topk > 1:
            if (
                spec_info is not None
                and getattr(spec_info, "retrieve_next_token", None) is not None
            ):
                bs_without_pad = spec_info.retrieve_next_token.shape[0]
                self.retrieve_next_token_list[bs - 1][:bs_without_pad].copy_(
                    spec_info.retrieve_next_token
                )
                self.retrieve_next_sibling_list[bs - 1][:bs_without_pad].copy_(
                    spec_info.retrieve_next_sibling
                )
            return ForwardMetadata(
                query_start_loc=qsl_buf,
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_track_indices=track_buf,
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
        else:
            return ForwardMetadata(
                query_start_loc=qsl_buf,
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_track_indices=track_buf,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1  # Mamba attn does not use seq lens to index kv cache

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def _track_pools(self):
        """Full [num_layers, pool_size, ...] conv/ssm pools plus the pool index
        of the last mamba layer, for the fused all-layers track launch. None if
        the pool shape is not the expected layout."""
        cached = getattr(self, "_track_pools_cache", False)
        if cached is False:
            pools = None
            try:
                mamba_cache = self.state_pool.mamba_cache
                conv_pool = mamba_cache.conv[0]
                ssm_pool = mamba_cache.temporal
                last_pool_idx = max(self.state_layer_indices.values())
                if (
                    conv_pool.dim() >= 3
                    and ssm_pool.dim() >= 3
                    and conv_pool.shape[0] == ssm_pool.shape[0] == last_pool_idx + 1
                ):
                    pools = (conv_pool, ssm_pool, last_pool_idx)
            except (AttributeError, IndexError):
                pools = None
            self._track_pools_cache = pools
            cached = pools
        return cached

    def _track_mamba_state_decode(
        self,
        forward_batch: ForwardBatch,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        layer_id: Optional[int] = None,
    ):
        """Copy decode conv/SSM states to track slots for prefix caching. Track
        dests come from the metadata (under cuda-graph: the static buffer), so the
        InputBuffer registry slot is never mutated.

        With a known layer_id, the per-layer launches collapse into ONE
        all-layers launch fired at the last mamba layer: the mask/src/dst
        indices are shared across layers and every layer's state is final by
        then, so the result is identical."""
        if forward_batch.mamba_track_mask is None:
            return
        if layer_id is not None:
            pools = self._track_pools()
            if pools is not None:
                conv_pool, ssm_pool, last_pool_idx = pools
                if self.state_layer_indices.get(layer_id) != last_pool_idx:
                    return
                track_mamba_states_all_layers(
                    conv_pool,
                    ssm_pool,
                    cache_indices,
                    forward_batch.mamba_track_mask,
                    self.forward_metadata.mamba_track_indices,
                    forward_batch.batch_size,
                    check_freed_slots=self.enable_unified_memory,
                )
                return
        track_mamba_states_if_needed(
            conv_states,
            ssm_states,
            cache_indices,
            forward_batch.mamba_track_mask,
            self.forward_metadata.mamba_track_indices,
            forward_batch.batch_size,
            check_freed_slots=self.enable_unified_memory,
        )

    def _track_mamba_state_extend(
        self,
        forward_batch: ForwardBatch,
        h: Optional[torch.Tensor],
        ssm_states: torch.Tensor,
        forward_metadata: ForwardMetadata,
        track_states: Optional[torch.Tensor] = None,
        *,
        h_track_buf: Optional[torch.Tensor] = None,
    ):
        """Copy extend SSM state at the last chunk boundary to track slots (source
        depends on chunk alignment; see `_init_track_ssm_indices`).

        Unaligned rows read the ``h_track_buf`` snapshot written in-kernel
        when given (in the kernel's native state dtype; its rows follow the batch,
        selected by the integer index
        ``track_ssm_h_batch_src`` — a boolean mask would nonzero() and sync the
        stream once per layer); otherwise they fall back to the per-chunk
        states ``h`` (already rounded to the activation dtype)."""
        if forward_metadata.has_mamba_track_mask:
            # Triton always returns h; FlashInfer returns it only when checkpoints
            # were requested. Aligned-only tracking reads the final state below.
            if forward_metadata.track_ssm_h_src.numel() > 0:
                if h_track_buf is not None:
                    ssm_states[forward_metadata.track_ssm_h_dst] = h_track_buf[
                        forward_metadata.track_ssm_h_batch_src
                    ].to(ssm_states.dtype, copy=False)
                else:
                    assert h is not None
                    h = h.squeeze(0)
                    ssm_states[forward_metadata.track_ssm_h_dst] = h[
                        forward_metadata.track_ssm_h_src
                    ].to(ssm_states.dtype, copy=False)
            if (
                forward_metadata.track_ssm_recompute_dst is not None
                and forward_metadata.track_ssm_recompute_dst.numel() > 0
            ):
                assert track_states is not None
                ssm_states[forward_metadata.track_ssm_recompute_dst] = (
                    track_states.squeeze(0).to(ssm_states.dtype, copy=False)
                )
            if forward_metadata.track_ssm_final_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_final_dst] = ssm_states[
                    forward_metadata.track_ssm_final_src
                ]


class Mamba2AttnBackend(MambaAttnBackendBase):
    """Attention backend wrapper for Mamba2Mixer kernels."""

    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = self.state_pool.mamba_cache.conv[0].shape

        if get_exec().mamba.enable_mamba_extra_buffer:
            assert self.conv_states_shape[-1] < self.mamba_chunk_size, (
                f"{self.conv_states_shape[-1]=} should be less than {self.mamba_chunk_size}"
            )
            assert get_exec().mamba.mamba_track_interval >= self.mamba_chunk_size, (
                f"mamba_track_interval ({get_exec().mamba.mamba_track_interval}) must be >= mamba_chunk_size ({self.mamba_chunk_size})"
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        metadata = self._replay_metadata(
            forward_batch.batch_size,
            forward_batch.req_pool_indices,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            forward_batch.seq_lens_cpu if not in_capture else None,
            num_padding=(
                0 if in_capture else getattr(forward_batch, "num_padding", None)
            ),
            in_capture=in_capture,
            mamba_track_indices=getattr(forward_batch, "mamba_track_indices", None),
            ragged_verify_layout=forward_batch.ragged_verify_layout,
        )
        spec_info = forward_batch.spec_info
        draft_token_num = spec_info.draft_token_num if spec_info is not None else 1
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata,
            forward_batch.seq_lens,
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
        )
        # `forward` slices the track destinations from ([-num_decodes:])
        assert self.forward_metadata.num_decodes == forward_batch.batch_size, (
            f"{self.forward_metadata.num_decodes=} != {forward_batch.batch_size=}"
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        metadata = self._forward_metadata(forward_batch)
        self.forward_metadata = Mamba2Metadata.prepare_mixed(
            metadata,
            self.mamba_chunk_size,
            forward_batch,
        )

    def forward(
        self,
        mixer: MambaMixer2,
        hidden_states: torch.Tensor,
        output: Optional[torch.Tensor],
        layer_id: int,
        forward_batch: ForwardBatch,
        mup_vector: Optional[torch.Tensor] = None,
        use_triton_causal_conv: bool = False,
    ):
        assert isinstance(self.forward_metadata, Mamba2Metadata)
        # Page-major stores state strided; only the stride-aware Triton causal-conv
        # reads it (CUDA causal_conv1d garbles it). A model may also force Triton.
        use_triton_causal_conv = (
            use_triton_causal_conv or get_memory().enable_page_major_kv_layout
        )
        layer_cache = self.state_layer_cache(layer_id)
        mixer_out, intermediate_states, track_states = mixer.forward(
            hidden_states=hidden_states,
            output=output,
            layer_cache=layer_cache,
            metadata=self.forward_metadata,
            mup_vector=mup_vector,
            use_triton_causal_conv=use_triton_causal_conv,
        )

        if forward_batch.mamba_track_mask is not None:
            if intermediate_states is not None:
                self._track_mamba_state_extend(
                    forward_batch,
                    intermediate_states,
                    layer_cache.temporal,
                    self.forward_metadata,
                    track_states=track_states,
                )

            if self.forward_metadata.num_decodes > 0:
                num_decodes = self.forward_metadata.num_decodes
                track_mamba_states_if_needed(
                    layer_cache.conv[0],
                    layer_cache.temporal,
                    self.forward_metadata.mamba_cache_indices[-num_decodes:],
                    forward_batch.mamba_track_mask[-num_decodes:],
                    self.forward_metadata.mamba_track_indices[-num_decodes:],
                    num_decodes,
                    check_freed_slots=self.enable_unified_memory,
                )

        return mixer_out

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )


class HybridLinearAttnBackend(AttentionBackend):
    """Manages a full and linear attention backend"""

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: MambaAttnBackendBase,
        full_attn_layers: list[int],
    ):
        self.full_attn_layers = full_attn_layers
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend
        self.attn_backend_list = [full_attn_backend, linear_attn_backend]
        self.token_to_kv_pool = full_attn_backend.token_to_kv_pool
        self.req_to_token_pool = full_attn_backend.req_to_token_pool
        self.runner_ids = full_attn_backend.runner_ids
        self.max_context_len = getattr(full_attn_backend, "max_context_len", None)
        self.needs_cpu_seq_lens = (
            full_attn_backend.needs_cpu_seq_lens
            or linear_attn_backend.needs_cpu_seq_lens
        )
        self.extend_dummy_seqs_capped_by_req_pool = getattr(
            full_attn_backend, "extend_dummy_seqs_capped_by_req_pool", False
        ) or getattr(linear_attn_backend, "extend_dummy_seqs_capped_by_req_pool", False)
        # Breakable prefill CUDA graphs capture the linear extend when the linear
        # backend can run it on static tables (linear/kda_prefill_graph.py);
        # those tables follow the runner's captured-metadata contract.
        self.use_captured_forward_metadata_for_breakable_cuda_graph = (
            check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE)
            and linear_attn_backend.supports_prefill_graph_extend()
            and not full_attn_backend.use_captured_forward_metadata_for_breakable_cuda_graph
        )
        if self.use_captured_forward_metadata_for_breakable_cuda_graph:
            self.prefill_cuda_graph_max_batch_size = (
                linear_attn_backend.prefill_graph_max_seqs
            )
            logger.info(
                "Breakable prefill CUDA graphs capture the %s extend.",
                type(linear_attn_backend).__name__,
            )
        # Mirrors the linear sidecar's RecoverSSM mode; False for every non-GDN
        # sidecar, so the verify-commit dispatch below stays GDN-only by
        # construction rather than by inspecting pool buffers.
        self._recover_ssm = getattr(linear_attn_backend, "_recover_ssm", False)
        # gdn_mtp_cache_mode=none recovery overlap: run the (eager) SSM-state
        # recovery on a dedicated side stream so its GPU work overlaps the next
        # step's draft-phase compute, then join before anything else reads or
        # overwrites the SSM pool (see _wait_recovery_if_pending and
        # state_producer_events).
        self._recovery_stream: Optional[torch.cuda.Stream] = None
        self._recovery_event: Optional[torch.cuda.Event] = None
        self._recovery_event_pending: bool = False
        # Created with _recovery_stream and reused every step, because
        # _no_cache_mtp_recompute runs on the bs=1 host-latency critical path: a
        # bare Event.record() resolves the current stream in C++, where
        # torch.cuda.current_stream() costs ~40us of interpreter time
        # (_get_device_index -> _cuda_getDevice plus a fresh Stream wrapper), and
        # torch.cuda.stream() allocates a new StreamContext per call.
        self._recovery_join_event: Optional[torch.cuda.Event] = None
        self._recovery_stream_ctx: Optional[torch.cuda.StreamContext] = None
        # CUDA graph for FlashInfer recovery: replace the per-layer kernel
        # launches (their CPU dispatch cost) with a single graph replay. Captured
        # AND replayed on _recovery_stream so it still overlaps draft_extend + next
        # draft. Stable fixed-address buffers hold the per-step state/accepted-step
        # indices; contents are refreshed each step before replay.
        self._rec_state_idx_buf: Optional[torch.Tensor] = None
        self._rec_acc_steps_buf: Optional[torch.Tensor] = None
        # Interval-checkpoint (mamba radix track) recovery buffers, used only when
        # gdn_mtp_cache_mode=none runs with mamba radix tracking (extra_buffer): a
        # second recovery pass reconstructs the state at the exact track boundary
        # and writes it to the ping-pong track slot. Output slots come from
        # mamba_track_indices, accepted_steps from mamba_steps_to_track (both
        # -1-masked to reserved slot 0 / step 0 for non-crossing requests).
        self._rec_track_idx_buf: Optional[torch.Tensor] = None
        self._rec_track_steps_buf: Optional[torch.Tensor] = None
        self._rec_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        # Per-bucket graphs for the boundary pass, captured alongside _rec_graphs
        # only when mamba radix tracking is enabled. Empty when tracking is off or
        # capture failed, in which case the boundary runs eager on the side stream.
        self._rec_boundary_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        # Sorted bucket sizes for which a recovery graph was captured at warmup.
        # None until capture_recovery_graphs() succeeds; while None, recovery runs
        # eagerly on the side stream (the known-good overlap path).
        self._rec_capture_bs: Optional[list[int]] = None
        # Every field above must exist before the helper runs: it assigns the stream,
        # the shared event and the cached join helpers, so calling it earlier would
        # either read an attribute that does not exist yet or have its work erased by
        # the initializations below. It is created here, not lazily, because the
        # HiCache write fence registers THIS event object once, at startup, and joins
        # its latest record on every submit (see state_producer_events).
        if self._recover_ssm and is_cuda():
            self._ensure_recovery_stream()

    def _ensure_recovery_stream(self):
        """Lazily create the recovery side stream and its per-step helpers.

        Both the warmup capture and the serving path enter through here so the
        cached join event / stream context can never be left unset by whichever
        one runs first. The completion event is the STATE POOL's (see
        MambaPool.state_recovery_event): every backend serving that pool records
        the same object, so the HiCache write fence's single registration cannot be
        orphaned by a backend swap. A pool with no event (a backend whose sidecar has
        no cuda state pool) keeps a private one and loses only that fence.
        """
        if self._recovery_stream is not None:
            return
        pool_event = getattr(
            self.linear_attn_backend.state_pool, "state_recovery_event", None
        )
        self._recovery_stream = torch.cuda.Stream()
        self._recovery_event = pool_event or torch.cuda.Event()
        self._recovery_join_event = torch.cuda.Event()
        self._recovery_stream_ctx = torch.cuda.stream(self._recovery_stream)

    def _wait_recovery_if_pending(self):
        """Join the side-stream gdn_mtp_cache_mode=none SSM recovery into the current
        (compute) stream before this forward reads the SSM pool or writes the conv /
        a-b stash. The wait also prevents the upcoming forward from overwriting the
        per-layer recovery stash while the side-stream recovery is still reading it.

        Reached only from ``drain_pending_recovery``, i.e. only from uncaptured host
        code at a forward's dispatch seam or at backend retirement -- never from
        metadata preparation, which can be captured, replayed or planned on another
        stream.
        """
        if self._recovery_event_pending:
            # Event.wait() with no stream defaults to the current stream inside
            # C++ -- same ordering as current_stream().wait_event(event), without
            # building the Python Stream wrapper.
            self._recovery_event.wait()
            self._recovery_event_pending = False

    def state_producer_events(self) -> list:
        """Events that mark this backend as a still-running writer of the state
        pool, for consumers that do NOT run on the forward stream.

        The HiCache L2 write leg joins its transfer streams to
        ``cache_controller.write_fence_stream`` (the model runner's forward
        stream), and the side-stream recovery is not ordered into that stream
        until the NEXT target forward plans its metadata. A backup submitted
        until the next target forward plans its metadata. A backup submitted
        before then would read the rows this recovery is still committing -- and
        publish them under a valid key. The event is created once and reused for
        every recovery, so a waiter registered at startup joins whichever record
        is the latest one at submit time; before the first record it is inert.
        Returns [] when there is nothing to fence (no none-mode, no cuda pool).
        """
        pool_event = getattr(
            self.linear_attn_backend.state_pool, "state_recovery_event", None
        )
        if pool_event is not None:
            return [pool_event]
        if self._recovery_event is None:
            return []
        return [self._recovery_event]

    def drain_pending_recovery(self) -> None:
        """Join a side-stream recovery that no forward has joined yet.

        This is the ONE consuming join point, and it sits on the uncaptured
        dispatch seam of a forward (``ModelRunner._forward_raw``, before the
        cuda-graph branch) rather than in metadata preparation. Preparation is
        the wrong owner twice over: ``SGLANG_ENABLE_METADATA_GLUE_GRAPH``
        captures ``init_forward_metadata_out_graph`` into a graph and then
        REPLAYS it, so the Python join neither runs on the steps that matter nor
        survives recording (a recorded ``cudaStreamWaitEvent`` on an event
        recorded outside the graph freezes the dependency at capture time), and
        planning/capture on a side stream would otherwise clear the pending flag
        before the forward takes ownership of the ordering. See
        ``MetadataGlueGraph.run``: after the first capture it only reinstalls
        metadata and replays device ops.

        A speculative width switch replaces the whole backend set, so after a
        switch the incoming backend's first target forward has no reason to look
        at the outgoing one's ``_recovery_event_pending`` -- and that
        still-running recovery reads the SSM pool (one allocation every state's
        backend addresses) exactly where the new state's forward writes. Whoever
        retires a backend therefore drains it first; the SSM/KV pools stay
        shared, only the join is ordered.
        """
        self._wait_recovery_if_pending()

    @property
    def data_type(self):
        # KV-cache dtype readers (e.g. the trtllm_mla fused-rope check) reach the
        # wrapper since split backends are wrapped once (#31439); the full-attn
        # side owns the KV cache, so its dtype is authoritative.
        return self.full_attn_backend.data_type

    @property
    def supports_ragged_verify_graph(self) -> bool:
        return (
            self.full_attn_backend.supports_ragged_verify_graph
            and self.linear_attn_backend.supports_ragged_verify_graph
        )

    @property
    def use_mha(self) -> bool:
        return getattr(self.full_attn_backend, "use_mha", False)

    @property
    def kv_cache_dtype(self):
        # Expose the full-attention backend's cache dtype because fused DSA/NSA RoPE
        # reads it from this wrapper.
        return getattr(self.full_attn_backend, "kv_cache_dtype", None)

    def _is_full_attn(
        self, layer: Optional[RadixAttention], layer_id: Optional[int] = None
    ) -> bool:
        if layer is not None:
            layer_id = layer.layer_id
        assert layer_id is not None, "either layer or layer_id must be provided"
        return layer_id in self.full_attn_layers

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        # Deliberately no recovery join here: this prep is exactly what the
        # metadata glue graph captures and replays, so a join would be recorded
        # once (and then wait on nothing) or skipped entirely. The consuming wait
        # is at the forward dispatch seam -- see drain_pending_recovery.
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

    def shared_read_ends(self, fm: ForwardMode) -> SharedReadEnds:
        return SharedReadEnds.max_of(
            b.shared_read_ends(fm) for b in self.attn_backend_list
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_in_graph(forward_batch)

    @staticmethod
    def _is_plain_extend(forward_batch: ForwardBatch) -> bool:
        mode = forward_batch.forward_mode
        return (
            mode.is_extend()
            and not mode.is_target_verify()
            and not mode.is_draft_extend_v2()
        )

    def linear_extend_in_graph(self) -> bool:
        """Whether the linear layers of the prefill being captured run inside
        the graph (no eager break)."""
        return self.linear_attn_backend.prefill_graph_extend_active()

    def init_forward_metadata_for_breakable_cuda_graph_capture(
        self, forward_batch: ForwardBatch
    ):
        self.full_attn_backend.init_forward_metadata(forward_batch)
        if not self._is_plain_extend(forward_batch):
            if not forward_batch.forward_mode.is_draft_extend_v2():
                self.linear_attn_backend.init_forward_metadata(forward_batch)
            return None
        return self.linear_attn_backend.init_prefill_graph_metadata(forward_batch)

    def prepare_forward_metadata_for_breakable_cuda_graph_replay(
        self,
        capture_metadata,
        forward_batch: ForwardBatch,
        *,
        static_forward_batch: Optional[ForwardBatch] = None,
    ) -> None:
        self.full_attn_backend.init_forward_metadata(forward_batch)
        if capture_metadata is not None:
            self.linear_attn_backend.refresh_prefill_graph_metadata(
                capture_metadata, forward_batch
            )
        elif not forward_batch.forward_mode.is_draft_extend_v2():
            self.linear_attn_backend.init_forward_metadata(forward_batch)

    def get_indexer_metadata(self, layer_id: int, forward_batch: ForwardBatch):
        if layer_id in self.full_attn_layers:
            return self.full_attn_backend.get_indexer_metadata(layer_id, forward_batch)
        return None

    def on_after_cuda_graph_warmup(self):
        for attn_backend in self.attn_backend_list:
            attn_backend.on_after_cuda_graph_warmup()

    @property
    def verify_mask(self) -> Optional[VerifyMask]:
        # The mask lives on the full-attn child; the linear side reads none.
        return self.full_attn_backend.verify_mask

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        # Plan-stream fixup after draft completes: forward to both children.
        # Sub-backends that cannot run under the plan stream keep the fail-loud
        # NotImplementedError base behavior.
        for attn_backend in self.attn_backend_list:
            attn_backend.update_verify_buffers_to_fill_after_draft(
                spec_info=spec_info, cuda_graph_bs=cuda_graph_bs
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_draft_extend_v2():
            # DRAFT_EXTEND_V2 runs only full-attn layers in the draft model; skip
            # linear/mamba metadata (it requires query_start_loc).
            self.full_attn_backend.init_forward_metadata(forward_batch)
            return
        # Same reason as in init_forward_metadata_out_graph: preparation, in any
        # variant, is not allowed to consume the recovery join. It may be captured
        # (breakable/capturable prep), it may run on the plan stream, and the
        # caller that dispatches the forward has already joined at its seam.
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)

    def init_mha_chunk_metadata(
        self, forward_batch: ForwardBatch, disable_flashinfer_ragged: bool = False
    ):
        # Hybrid MLA models resolve get_attn_backend() to this wrapper; delegate
        # so the full-attn backend plans its chunked-prefill metadata.
        init = getattr(self.full_attn_backend, "init_mha_chunk_metadata", None)
        if init is not None:
            init(forward_batch, disable_flashinfer_ragged)

    def get_indexer_metadata(self, layer_id, forward_batch):
        return self.full_attn_backend.get_indexer_metadata(layer_id, forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cpu_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_capture_cpu_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                out_cache_loc,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.full_attn_backend.get_cuda_graph_seq_len_fill_value()

    def get_cpu_graph_seq_len_fill_value(self):
        return self.full_attn_backend.get_cpu_graph_seq_len_fill_value()

    def forward_decode(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For GDN linear attention
        b: Optional[torch.Tensor] = None,  # For GDN linear attention
        **kwargs,
    ):
        if self._is_full_attn(layer, kwargs.get("layer_id")):
            return self.full_attn_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        return self.linear_attn_backend.forward_decode(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def forward_extend(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For GDN linear attention
        b: Optional[torch.Tensor] = None,  # For GDN linear attention
        **kwargs,
    ):
        if self._is_full_attn(layer, kwargs.get("layer_id")):
            return self.full_attn_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        return self.linear_attn_backend.forward_extend(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def forward(
        self,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        layer: RadixAttention = None,
        forward_batch: ForwardBatch = None,
        save_kv_cache: bool = True,
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For linear attention
        b: Optional[torch.Tensor] = None,  # For linear attention
        **kwargs,
    ):
        is_linear_attn = not self._is_full_attn(layer, kwargs.get("layer_id"))

        # For linear attention, dispatch on the logical mode to match the
        # built metadata.
        dispatch_mode = (
            forward_batch.logical_forward_mode
            if is_linear_attn
            else forward_batch.forward_mode
        )

        if forward_batch.forward_mode.is_idle():
            if is_linear_attn:
                return mixed_qkv.new_empty(
                    mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim
                )
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif dispatch_mode.is_decode():
            return self.forward_decode(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )
        else:
            return self.forward_extend(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )

    def update_mamba_state_after_mtp_verify(
        self,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        model,
        req_pool_indices: Optional[torch.Tensor] = None,
    ):
        """Commit accepted Mamba states; PP rows identify the delayed batch."""
        request_number = last_correct_step_indices.shape[0]
        src_indices_raw = None
        if pp_spec_stable_rows_enabled() and req_pool_indices is not None:
            src_indices_raw = req_pool_indices[:request_number]

        # `mamba_track_indices` is VIRTUAL; the scatter writes physical views.
        if mamba_track_indices is not None:
            mamba_track_indices = self.linear_attn_backend._translate_mamba_indices(
                mamba_track_indices
            )

        if src_indices_raw is not None:
            state_indices_tensor = self.linear_attn_backend._translate_mamba_indices(
                self.linear_attn_backend.req_to_token_pool.get_mamba_indices(
                    req_pool_indices[:request_number]
                )
            )
        else:
            state_indices_tensor = (
                self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                    :request_number
                ]
            )

        mamba_pool = self.linear_attn_backend.state_pool
        mamba_caches = mamba_pool.get_speculative_mamba2_params_all_layers()

        # ReplaySSM-KDA: the accepted drafts live in the per-slot ring (written
        # during verify); no intermediate_ssm is allocated. Replay the accepted
        # prefix into `temporal` instead of scattering an intermediate state.
        # dspark/dflash call this method directly; the generic spec_utils commit
        # handles replayssm before reaching here (returns early), so this branch is
        # only hit by the direct callers. Chain layout only (topk <= 1), so
        # accept_lens == last_correct_step_indices + 1.
        if mamba_pool.replayssm_is_kda:
            from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (
                commit_kda_replayssm_after_verify,
            )

            commit_kda_replayssm_after_verify(
                spec_state=mamba_caches,
                state_batch_indices=state_indices_tensor,
                accept_lens=last_correct_step_indices + 1,
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                src_indices_raw=src_indices_raw,
                null_block_id=-1,
            )
            return

        # KDA fused-accept: the next verify seeds itself in-kernel from the
        # accepted checkpoint slot (recurrent_kda's num_accepted_tokens), so the
        # SSM state never round-trips through `temporal` and only the conv
        # windows still need the accept rollback. Recording this round's accept
        # length is what selects that seed next round; chain layout only (see
        # above), so accept_lens == last_correct_step_indices + 1. The pool
        # exists only where KDAAttnBackend found the contract satisfied, which
        # includes mamba radix tracking being off.
        accept_lens_pool = self.linear_attn_backend.accept_lens_pool
        if accept_lens_pool is not None:
            assert mamba_track_indices is None, "fused-accept runs with radix off"
            for conv_states, intermediate_conv_window in zip(
                mamba_caches.conv, mamba_caches.intermediate_conv_window
            ):
                fused_conv_window_scatter_with_mask(
                    conv_states,
                    intermediate_conv_window,
                    state_indices_tensor,
                    last_correct_step_indices,
                    src_indices_raw,
                )
            accept_lens_pool[state_indices_tensor.to(torch.int64)] = (
                last_correct_step_indices.to(torch.int32) + 1
            )
            return

        # RecoverSSM (gdn_mtp_cache_mode=none) skips the per-draft SSM snapshots, so
        # rerun the recurrence from h_0 over the accepted prefix and roll back the
        # conv state. All other modes (full, ReplaySSM) go through the upstream
        # fused gather-scatter, which handles the ssm scatter, conv rollback, and
        # radix track indices. Keyed on the resolved mode, not on which buffers the
        # pool left unallocated -- ReplaySSM also skips intermediate_ssm, and which
        # of its ring buffers exist depends on its verify protocol. server_args
        # rejects none-mode alongside both ReplaySSM flags, so this is exclusive.
        if self._recover_ssm:
            # Mamba radix tracking (extra_buffer) in none-mode is supported on both
            # recovery paths (FlashInfer via native output_state_indices, Triton via
            # the output-index arg): a second boundary pass recomputes the tracked
            # prefix state, which none-mode does not cache. Write h_K directly
            # without materializing outputs; when tracking is active, also recompute
            # + write the boundary checkpoint state to the ping-pong track slot.
            self._no_cache_mtp_recompute(
                accepted_steps=last_correct_step_indices,
                state_indices_tensor=state_indices_tensor,
                mamba_track_indices=mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
            )
            # Conv-state rollback uses the cached (deduplicated sliding-window)
            # conv windows via the strided-read scatter variant, and the conv
            # boundary checkpoint mirrors the SSM boundary pass (same masked
            # scatter the full-mode path uses; step == -1 rows are skipped).
            for conv_states, intermediate_conv_window_cache in zip(
                mamba_caches.conv, mamba_caches.intermediate_conv_window
            ):
                fused_conv_window_scatter_with_mask(
                    conv_states,
                    intermediate_conv_window_cache,
                    state_indices_tensor,
                    last_correct_step_indices,
                    src_indices_raw,
                )
                if mamba_track_indices is not None:
                    fused_conv_window_scatter_with_mask(
                        conv_states,
                        intermediate_conv_window_cache,
                        mamba_track_indices,
                        mamba_steps_to_track,
                        src_indices_raw,
                    )
        else:
            scatter_mamba_states_after_mtp_verify(
                mamba_caches,
                state_indices_tensor,
                last_correct_step_indices,
                mamba_track_indices,
                mamba_steps_to_track,
                src_indices_raw=src_indices_raw,
            )

        self._update_ple_state_after_mtp_verify(
            state_indices_tensor,
            last_correct_step_indices,
            mamba_track_indices,
            mamba_steps_to_track,
            src_indices_raw,
        )

    def _no_cache_mtp_recompute(
        self,
        accepted_steps: torch.Tensor,
        state_indices_tensor: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor] = None,
        mamba_steps_to_track: Optional[torch.Tensor] = None,
    ):
        """Recover accepted GDN SSM state for gdn_mtp_cache_mode=none.

        Replays the state-update recurrence over stashed post-conv k/v/a/b and
        writes h_{accepted_step} directly to the request's SSM state slot.

        When mamba radix tracking is active (extra_buffer), a request's accepted
        draft window may cross a track boundary. none-mode caches no intermediate
        state, so it also runs a second boundary pass that folds only
        ``mamba_steps_to_track`` steps from h_0 and writes the resulting boundary
        state to the ping-pong track slot ``mamba_track_indices``. That pass reads
        h_0 from the working slot, so it MUST run before the in-place working
        recovery overwrites h_0 with h_K. Requests that cross no boundary carry
        step == -1 and are redirected to reserved slot 0 / step 0 (harmless).
        """
        # Local imports to avoid a circular dependency at module load time.
        from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_recover_final_state,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
            fi_recovery_kernel,
        )

        stash_per_layer: dict = getattr(self.linear_attn_backend, "_no_cache_stash", {})
        if not stash_per_layer:
            # No GDN layer ran in cache_mode=none for this batch.
            return

        pool = self.linear_attn_backend.state_pool
        use_fi_recovery = fi_recovery_kernel(self.linear_attn_backend) is not None

        # Recovery runs outside the captured graph. Derive per-call sizes from
        # current tensors because the stash spans multiple batch-size captures.
        batch_size = accepted_steps.shape[0]
        draft_token_num = self.linear_attn_backend._no_cache_draft_token_num
        assert draft_token_num is not None, (
            "draft_token_num not cached -- forward_extend was never called "
            "in target_verify mode before _mamba_verify_update."
        )
        actual_seq_len = batch_size * draft_token_num
        cache_steps = draft_token_num

        # The Triton recover kernel takes these as kernel arguments, so it needs
        # materialized int32 tensors (and keeps them alive for record_stream
        # below). The FlashInfer path only ever copies them into the stable
        # _rec_* int32 buffers, and Tensor.copy_ narrows on the way in -- so the
        # conversion there is two eager dispatches of pure overhead, which a
        # host-latency-bound bs=1 decode pays in wall clock.
        state_idx_i32 = None
        accepted_steps_i32 = None
        if not use_fi_recovery:
            state_idx_i32 = state_indices_tensor.to(torch.int32).contiguous()
            accepted_steps_i32 = accepted_steps.to(torch.int32).contiguous()

        # Interval-checkpoint boundary pass runs on both recovery paths. FI uses
        # native output_state_indices; the Triton recover kernel takes a separate
        # output-index arg (default in-place). Per-step Triton boundary tensors are
        # built below (FI uses the stable _rec_track_* buffers instead).
        do_boundary = mamba_track_indices is not None
        track_out_i32 = None
        track_steps_i32 = None

        # One launch per GDN layer. Factored into a closure so it can run either
        # inline (CUDA graph capture) or on the side stream (eager overlap).
        B_bucket = None
        if use_fi_recovery:
            from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
                gated_delta_rule_mtp,
            )

            B = batch_size
            # Stable fixed-address index buffers (normally allocated at warmup in
            # capture_recovery_graphs; allocate here for the no-warmup path).
            if self._rec_state_idx_buf is None:
                dev = state_indices_tensor.device
                self._rec_state_idx_buf = torch.empty(
                    pool.size, dtype=torch.int32, device=dev
                )
                self._rec_acc_steps_buf = torch.empty(
                    pool.size, dtype=torch.int32, device=dev
                )
                self._rec_track_idx_buf = torch.empty(
                    pool.size, dtype=torch.int32, device=dev
                )
                self._rec_track_steps_buf = torch.empty(
                    pool.size, dtype=torch.int32, device=dev
                )
            # Smallest warmup-captured bucket >= B (None -> eager, no graph).
            B_bucket = self._rec_pad_to_bucket(B)
            # Refresh stable buffers on the main stream before the side-stream
            # read (the join event recorded below orders after these copies).
            # Batched into one grouped _foreach_copy_: the buffers are int32 and
            # copy_ narrows the int64 sources itself, so no .to()/.contiguous()
            # temporaries are needed and the whole refresh is one dispatch.
            from sglang.srt.model_executor.cuda_graph_buffer_registry import (
                _grouped_foreach_copy_,
            )

            refresh_dsts = [self._rec_state_idx_buf[:B], self._rec_acc_steps_buf[:B]]
            refresh_srcs = [state_indices_tensor, accepted_steps]
            crossed = None
            if do_boundary:
                # Boundary pass output slots come from mamba_track_indices, its
                # fold count from mamba_steps_to_track. Requests crossing no
                # boundary carry step == -1 -> redirect to reserved slot 0 / step 0
                # (folds 0 steps, writing h_0 to the discard slot).
                #
                # clamp(min=0) is exactly where(steps >= 0, steps, 0) for every
                # integer input, so the fold count needs no explicit mask. The
                # output slot still does -- leaving a non-crossing request pointed
                # at its real track slot would clobber a live ping-pong
                # checkpoint with h_0 -- but that mask applies in place on the
                # stable buffer below, which avoids torch.where's output alloc.
                crossed = mamba_steps_to_track >= 0
                refresh_dsts += [
                    self._rec_track_idx_buf[:B],
                    self._rec_track_steps_buf[:B],
                ]
                refresh_srcs += [
                    mamba_track_indices,
                    mamba_steps_to_track.clamp(min=0),
                ]
            _grouped_foreach_copy_(refresh_dsts, refresh_srcs)
            if do_boundary:
                self._rec_track_idx_buf[:B].mul_(crossed)
            if B_bucket is not None and B_bucket > B:
                # Pad rows [B:bucket] -> reserved slot 0 (never a real request, so
                # their recovery output is discarded harmlessly).
                self._rec_state_idx_buf[B:B_bucket].fill_(0)
                self._rec_acc_steps_buf[B:B_bucket].fill_(0)
                if do_boundary:
                    # The boundary graph is captured at bucket size, so its pad
                    # rows need a destination too -> reserved slot 0.
                    self._rec_track_idx_buf[B:B_bucket].fill_(0)
                    self._rec_track_steps_buf[B:B_bucket].fill_(0)
        elif do_boundary:
            # Triton boundary: per-step regular tensors (no stable buffers). The
            # recover kernel already skips rows with a negative output index, so
            # non-crossing requests (step == -1) get output slot -1 (skipped on
            # store) instead of being redirected to a discard slot. h_0 is read
            # from the working slot (state_idx_i32) for all rows.
            track_steps_i32 = mamba_steps_to_track.to(torch.int32).contiguous()
            track_out_i32 = torch.where(
                track_steps_i32 >= 0,
                mamba_track_indices.to(torch.int32),
                torch.full_like(track_steps_i32, -1),
            ).contiguous()

        def _triton_recover_launch(init_indices, out_indices, acc_steps):
            # One recover launch per GDN layer. init_indices supplies h_0; the
            # folded state is written to out_indices (== init_indices for in-place
            # working recovery, the track slot for the boundary pass).
            for layer_id, stash in stash_per_layer.items():
                layer_ssm_states = self.linear_attn_backend.state_layer_cache(
                    layer_id
                ).temporal

                conv_dims = stash.get("conv_dims")
                if conv_dims is not None:
                    # k/v are strided views of the persistent conv-out buffer
                    # ([1, actual_seq_len, H, D]); no stash copy.
                    k_recov, v_recov = self._persist_kv(
                        layer_id, conv_dims, batch_size, cache_steps
                    )
                else:
                    k_recov = stash["k"][:, :actual_seq_len]
                    v_recov = stash["v"][:, :actual_seq_len]

                fused_sigmoid_gating_delta_rule_recover_final_state(
                    A_log=stash["A_log"],
                    a=stash["a"][:actual_seq_len],
                    dt_bias=stash["dt_bias"],
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    k=k_recov,
                    v=v_recov,
                    b=stash["b"][:actual_seq_len],
                    initial_state_source=layer_ssm_states,
                    initial_state_indices=init_indices,
                    accepted_steps=acc_steps,
                    cache_steps=cache_steps,
                    use_qk_l2norm_in_kernel=True,
                    is_kda=False,
                    output_state_indices=out_indices,
                )

        def _run_boundary():
            # Interval-checkpoint recovery: fold mamba_steps_to_track steps from h_0
            # (working slot) and write the boundary state to the ping-pong track
            # slot. Must run before _run_recovery() overwrites h_0 with h_K.
            if use_fi_recovery:
                self._fi_recovery_launch(
                    B,
                    stash_per_layer,
                    pool,
                    gated_delta_rule_mtp,
                    init_idx_buf=self._rec_state_idx_buf,
                    out_idx_buf=self._rec_track_idx_buf,
                    acc_steps_buf=self._rec_track_steps_buf,
                )
                return
            _triton_recover_launch(state_idx_i32, track_out_i32, track_steps_i32)

        def _run_recovery():
            if use_fi_recovery:
                # One launch per GDN layer, reading the stable index buffers.
                self._fi_recovery_launch(B, stash_per_layer, pool, gated_delta_rule_mtp)
                return

            # In-place working recovery: out == init == the working SSM slot.
            _triton_recover_launch(state_idx_i32, state_idx_i32, accepted_steps_i32)

        # During CUDA graph capture, recovery must stay on the capture stream
        # (it is normally eager / outside the graph; this is a safety guard).
        if torch.cuda.is_current_stream_capturing():
            if do_boundary:
                _run_boundary()
            _run_recovery()
            return

        # Eager path: overlap recovery on a dedicated side stream so its GPU work
        # hides behind the next step's draft-phase compute (which does not touch
        # the SSM pool). The next target forward joins on _recovery_event (see
        # _wait_recovery_if_pending); anything that neither runs on the forward
        # stream nor plans metadata -- the HiCache L2 write leg -- joins the same
        # event through state_producer_events().
        self._ensure_recovery_stream()
        # Recovery must observe the stash writes and (FI) stable-buffer copies
        # issued on the current stream during this step's verify forward.
        # record() with no argument resolves the current stream in C++, so this
        # pair is wait_stream(current_stream()) without the Python round trip.
        self._recovery_join_event.record()
        self._recovery_stream.wait_event(self._recovery_join_event)

        if use_fi_recovery and B_bucket is not None:
            # Replay the warmup-captured graph for this bucket on the side stream
            # (pure async launch -- overlaps draft_extend + next draft). No capture
            # ever happens on the live path. The boundary pass replays its own
            # captured graph (falling back to eager launches if it was not
            # captured) BEFORE the working replay, so it reads h_0 before the
            # working graph overwrites it with h_K.
            with self._recovery_stream_ctx:
                if do_boundary:
                    boundary_graph = self._rec_boundary_graphs.get(B_bucket)
                    if boundary_graph is not None:
                        boundary_graph.replay()
                    else:
                        _run_boundary()
                self._rec_graphs[B_bucket].replay()
        else:
            # No warmup graph (Triton fallback, B beyond the largest captured
            # bucket, or capture disabled/failed): eager recovery on the side
            # stream -- the known-good overlap path.
            with self._recovery_stream_ctx:
                if do_boundary:
                    _run_boundary()
                _run_recovery()

        self._recovery_event.record(self._recovery_stream)
        # FlashInfer path reads only the long-lived stable buffers on the side
        # stream, so the per-step state_idx_i32 / accepted_steps_i32 need no
        # record_stream. Triton path reads them directly on the side stream:
        # wait_stream orders but does not extend their lifetime, so pin the
        # blocks until the side-stream recovery is done.
        if not use_fi_recovery:
            state_idx_i32.record_stream(self._recovery_stream)
            accepted_steps_i32.record_stream(self._recovery_stream)
            if do_boundary:
                # Triton boundary reads these per-step tensors on the side stream.
                track_out_i32.record_stream(self._recovery_stream)
                track_steps_i32.record_stream(self._recovery_stream)
        self._recovery_event_pending = True

    def _persist_kv(self, layer_id, conv_dims, b_rows, cache_steps):
        """FlashInfer-recovery post-conv (k, v) as zero-copy strided views of the
        persistent conv-out buffer, shaped [1, b_rows*cache_steps, H, D]. The
        buffer is [pool_size, cache_steps, conv_dim] token-major; k/v are the column
        slices [q_dim:q_dim+k_dim] / [q_dim+k_dim:...] (token stride = conv_dim,
        feature contiguous). The recovery kernel reads k/v via runtime strides, so
        no copy is materialized -- and, because the view is pure host-side metadata
        over an address-stable buffer, nothing extra is captured into the recovery
        graph."""
        persist = self.linear_attn_backend._conv_out_persist[layer_id]
        q_dim, k_dim, v_dim, Hk, Dk, Hv, Dv = conv_dims
        n_tok = b_rows * cache_steps
        mixed = persist[:b_rows].reshape(n_tok, q_dim + k_dim + v_dim)
        k = mixed[:, q_dim : q_dim + k_dim].view(1, n_tok, Hk, Dk)
        v = mixed[:, q_dim + k_dim : q_dim + k_dim + v_dim].view(1, n_tok, Hv, Dv)
        return k, v

    def _fi_recovery_launch(
        self,
        n,
        stash_per_layer,
        pool,
        gated_delta_rule_mtp,
        init_idx_buf=None,
        out_idx_buf=None,
        acc_steps_buf=None,
    ):
        """Issue the per-layer FlashInfer recovery launches for the first ``n``
        rows, reading the stable index buffers and the [:n] stash slices. Shared
        by warmup graph capture, graph-less eager fallback, and the warm-compile
        pass. The stash is pre-shaped [pool_size, T, H] so [:n] is [n, T, H].
        FI recovery reads k/v from strided views of the persistent conv-out
        buffer, a/b from the stash.

        Defaults reconstruct h_K in place at the accepted length (init == output ==
        the request's working SSM slot). The interval-checkpoint boundary pass
        overrides ``out_idx_buf`` (ping-pong track output slots) and
        ``acc_steps_buf`` (the per-request boundary step) while keeping
        ``init_idx_buf`` at the working slot, since h_0 lives there -- so it MUST
        run before the in-place working recovery overwrites h_0 with h_K."""
        init_idx_buf = self._rec_state_idx_buf if init_idx_buf is None else init_idx_buf
        out_idx_buf = self._rec_state_idx_buf if out_idx_buf is None else out_idx_buf
        acc_steps_buf = (
            self._rec_acc_steps_buf if acc_steps_buf is None else acc_steps_buf
        )
        _init_idx = init_idx_buf[:n]
        # Reuse the SAME slice object when read and write indices alias: the FI
        # kernel picks its single-pool fast path via an identity check
        # (`output_state_indices is initial_state_indices`), and a second
        # `buf[:n]` would be a distinct view object that silently forces the
        # slower split-pool codegen (extra index LDG + separate write-side STGs).
        _out_idx = _init_idx if out_idx_buf is init_idx_buf else out_idx_buf[:n]
        _acc_steps = acc_steps_buf[:n]
        _T = self.linear_attn_backend._no_cache_draft_token_num
        for layer_id, stash in stash_per_layer.items():
            layer_ssm_states = self.linear_attn_backend.state_layer_cache(
                layer_id
            ).temporal
            # The FI path always stashes conv_dims (never k/v), so k/v come from
            # the persistent conv-out buffer. Reshape the [1, n*T, H, D] views to
            # the [n, T, H, D] the kernel expects (q == k: the kernel l2-norms
            # both).
            conv_dims = stash["conv_dims"]
            k_pv, v_pv = self._persist_kv(layer_id, conv_dims, n, _T)
            q_k = k_pv.view(n, _T, k_pv.shape[2], k_pv.shape[3])
            v_bat = v_pv.view(n, _T, v_pv.shape[2], v_pv.shape[3])
            gated_delta_rule_mtp(
                A_log=stash["A_log_f32"],
                a=stash["a"][:n],
                dt_bias=stash["dt_bias"],
                q=q_k,
                k=q_k,
                v=v_bat,
                b=stash["b"][:n],
                initial_state_source=layer_ssm_states,
                initial_state_indices=_init_idx,
                output_state_indices=_out_idx,
                accepted_steps=_acc_steps,
                disable_state_update=False,
                disable_output=True,
                use_qk_l2norm_in_kernel=True,
                scale=None,
                output=None,
            )

    def _rec_pad_to_bucket(self, batch_size: int) -> Optional[int]:
        """Smallest captured bucket >= batch_size, or None if no warmup graphs
        exist or batch_size exceeds the largest captured bucket (-> eager
        fallback)."""
        if not self._rec_capture_bs:
            return None
        i = bisect.bisect_left(self._rec_capture_bs, batch_size)
        if i == len(self._rec_capture_bs):
            return None
        return self._rec_capture_bs[i]

    def capture_recovery_graphs(self, capture_bs):
        """Warmup: capture one FlashInfer recovery graph per batch-size bucket, on
        the side stream, into a single shared mempool. Serving then pads the real
        batch up to a bucket and replays (no live capture -> no per-step stall /
        global-mode crash). Must run AFTER a target_verify forward has allocated
        the per-layer stash at its final addresses (else this is a no-op and
        recovery falls back to eager side-stream launches).

        Padded rows (bucket - B) point at reserved SSM slot 0 (never allocated to
        a real request; free_slots starts at 1), so their output is harmless.
        """
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
            fi_recovery_kernel,
        )

        if not self._recover_ssm:
            return
        if fi_recovery_kernel(self.linear_attn_backend) is None:
            # Triton recovery or non-FI backend: no recovery graphs to capture;
            # recovery runs eager on the side stream.
            return
        stash_per_layer = getattr(self.linear_attn_backend, "_no_cache_stash", {})
        if not stash_per_layer:
            logger.warning(
                "[gdn_recovery] stash not allocated at capture time; recovery will "
                "run eagerly on the side stream (no cuda graph)."
            )
            return

        pool = self.linear_attn_backend.state_pool
        from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp

        dev = self.linear_attn_backend.state_layer_cache(
            next(iter(stash_per_layer))
        ).temporal.device
        if self._rec_state_idx_buf is None:
            self._rec_state_idx_buf = torch.empty(
                pool.size, dtype=torch.int32, device=dev
            )
            self._rec_acc_steps_buf = torch.empty(
                pool.size, dtype=torch.int32, device=dev
            )
            # Boundary (interval-checkpoint) pass buffers: ping-pong track output
            # slots and the per-request boundary step. Allocated here so the eager
            # boundary launch reuses address-stable buffers.
            self._rec_track_idx_buf = torch.empty(
                pool.size, dtype=torch.int32, device=dev
            )
            self._rec_track_steps_buf = torch.empty(
                pool.size, dtype=torch.int32, device=dev
            )
        # Dummy indices -> reserved slot 0 during capture (records only; the real
        # per-step indices are copied in before each replay).
        self._rec_state_idx_buf.fill_(0)
        self._rec_acc_steps_buf.fill_(0)
        self._rec_track_idx_buf.fill_(0)
        self._rec_track_steps_buf.fill_(0)

        # Capture the interval-checkpoint boundary pass too when mamba radix
        # tracking is on. Tracking is a server-level setting, so this is decided
        # once here; batches that carry no track indices simply skip the replay.
        capture_boundary = get_exec().mamba.enable_mamba_extra_buffer

        self._ensure_recovery_stream()
        # Producer -> consumer handoff, the same pair the serving path uses in
        # _no_cache_mtp_recompute: the buffer zero-fills above, the stash and the
        # state pool they were written into were all issued on the CALLER stream, and
        # the first warmup launch below runs on the recovery stream. Without this
        # join, the (cached, address-stable) kernel can read uninitialized index/step
        # rows or write state the producer has not finished writing. The synchronize
        # that FOLLOWS the launch is not protection -- by then the launch is already
        # queued -- and the graph's own later sync is far too late.
        self._recovery_join_event.record()
        self._recovery_stream.wait_event(self._recovery_join_event)

        buckets = sorted({int(b) for b in capture_bs if 0 < int(b) <= pool.size})
        if not buckets:
            return
        try:
            shared_pool = torch.cuda.graph_pool_handle()
            # Capture largest first so smaller graphs reuse the shared pool.
            for B in reversed(buckets):
                # Warm-compile this bucket's kernel + populate the kernel's per-B
                # argument defaults OUTSIDE capture (writes to reserved slot 0, so
                # harmless), so the capture itself is JIT-free and alloc-free.
                with torch.cuda.stream(self._recovery_stream):
                    self._fi_recovery_launch(
                        B, stash_per_layer, pool, gated_delta_rule_mtp
                    )
                self._recovery_stream.synchronize()

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    g,
                    pool=shared_pool,
                    stream=self._recovery_stream,
                    capture_error_mode="thread_local",
                ):
                    self._fi_recovery_launch(
                        B, stash_per_layer, pool, gated_delta_rule_mtp
                    )
                self._rec_graphs[B] = g

                if capture_boundary:
                    # Boundary variant: reads h_0 from the working slots, writes to
                    # the ping-pong track slots. Distinct output indices select the
                    # kernel's split-pool codegen, so this is a separate compiled
                    # variant -- warm-compile it outside capture as well (all index
                    # buffers hold 0 here, so it only touches the discard slot).
                    with torch.cuda.stream(self._recovery_stream):
                        self._fi_recovery_launch(
                            B,
                            stash_per_layer,
                            pool,
                            gated_delta_rule_mtp,
                            init_idx_buf=self._rec_state_idx_buf,
                            out_idx_buf=self._rec_track_idx_buf,
                            acc_steps_buf=self._rec_track_steps_buf,
                        )
                    self._recovery_stream.synchronize()

                    gb = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(
                        gb,
                        pool=shared_pool,
                        stream=self._recovery_stream,
                        capture_error_mode="thread_local",
                    ):
                        self._fi_recovery_launch(
                            B,
                            stash_per_layer,
                            pool,
                            gated_delta_rule_mtp,
                            init_idx_buf=self._rec_state_idx_buf,
                            out_idx_buf=self._rec_track_idx_buf,
                            acc_steps_buf=self._rec_track_steps_buf,
                        )
                    self._rec_boundary_graphs[B] = gb
            self._rec_capture_bs = buckets
            logger.info(
                "[gdn_recovery] captured %d recovery cuda graphs (buckets=%s, "
                "boundary=%s)",
                len(buckets),
                buckets,
                capture_boundary,
            )
        except Exception as e:
            logger.warning(
                "[gdn_recovery] recovery cuda graph capture failed (%s); falling "
                "back to eager side-stream recovery.",
                e,
            )
            self._rec_graphs.clear()
            self._rec_boundary_graphs.clear()
            self._rec_capture_bs = None

    @staticmethod
    def _scatter_speculative_state_with_mask(
        dst: torch.Tensor,
        src: torch.Tensor,
        dst_indices_raw: torch.Tensor,
        step_indices_raw: torch.Tensor,
        src_indices_raw: Optional[torch.Tensor] = None,
    ):
        if dst is None or src is None or step_indices_raw.numel() == 0:
            return
        if dst.is_cuda and src.is_cuda:
            fused_mamba_state_scatter_with_mask(
                dst,
                src,
                dst_indices_raw,
                step_indices_raw,
                src_indices_raw,
            )
            return

        device = dst.device
        dst_indices = dst_indices_raw.to(device=device, dtype=torch.long)
        steps = step_indices_raw.to(device=device, dtype=torch.long)
        src_indices = (
            torch.arange(steps.shape[0], device=device, dtype=torch.long)
            if src_indices_raw is None
            else src_indices_raw.to(device=device, dtype=torch.long)
        )
        valid = (
            (steps >= 0)
            & (steps < src.shape[2])
            & (dst_indices >= 0)
            & (dst_indices < dst.shape[1])
            & (src_indices < src.shape[1])
        )
        valid_indices = valid.nonzero(as_tuple=True)[0]
        if valid_indices.numel() == 0:
            return
        dst[:, dst_indices[valid_indices]] = src[
            :, src_indices[valid_indices], steps[valid_indices]
        ]

    def _update_ple_state_after_mtp_verify(
        self,
        state_indices_tensor: torch.Tensor,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        src_indices_raw: Optional[torch.Tensor] = None,
    ):
        """Roll the accepted per-step PLE side states into their main slots."""
        router = self.linear_attn_backend.token_to_kv_pool.layer_router
        if mamba_track_indices is not None:
            assert mamba_steps_to_track is not None

        state_pairs = []
        short_conv_pool = router.part_view(SHORT_CONV_PART)
        if (
            short_conv_pool is not None
            and short_conv_pool.conv_state is not None
            and short_conv_pool.intermediate_conv_state is not None
        ):
            state_pairs.append(
                (
                    short_conv_pool.conv_state,
                    short_conv_pool.intermediate_conv_state,
                )
            )

        ngram_pool = router.part_view(NGRAM_PART)
        if (
            ngram_pool is not None
            and ngram_pool.context is not None
            and ngram_pool.intermediate_context is not None
        ):
            state_pairs.append(
                (
                    ngram_pool.context.unsqueeze(0),
                    ngram_pool.intermediate_context.unsqueeze(0),
                )
            )

        for state, intermediate_state in state_pairs:
            self._scatter_speculative_state_with_mask(
                state,
                intermediate_state,
                state_indices_tensor,
                last_correct_step_indices,
                src_indices_raw,
            )
            if mamba_track_indices is not None:
                self._scatter_speculative_state_with_mask(
                    state,
                    intermediate_state,
                    mamba_track_indices,
                    mamba_steps_to_track,
                    src_indices_raw,
                )


class ShortConvHybridAttnBackend(HybridLinearAttnBackend):
    """HybridLinearAttnBackend variant for short-conv hybrid models (ZAYA1 CCA,
    LFM2 short conv).

    The linear sidecar is a :class:`ShortConvAttnBackend
    <sglang.srt.layers.attention.linear.short_conv_backend.ShortConvAttnBackend>`
    that owns the per-request conv-state plumbing. The model's conv module
    reaches it via :meth:`conv_state_metadata` (``get_attn_backend()`` returns
    this wrapper) and runs its own conv kernel against the returned handle, so
    the model definition holds no pool access. The sidecar is never reached
    through the full-vs-linear ``forward_decode`` / ``forward_extend`` dispatch.
    """

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        short_conv_backend: MambaAttnBackendBase,
        full_attn_layers: list,
    ):
        # Register short_conv_backend as the linear sidecar so it rides in
        # attn_backend_list and inherits the metadata / cuda-graph fan-out.
        super().__init__(full_attn_backend, short_conv_backend, full_attn_layers)
        self.short_conv_backend = short_conv_backend

    def conv_state_metadata(self, layer_id: int, forward_batch: ForwardBatch):
        return self.short_conv_backend.conv_state_metadata(layer_id, forward_batch)
