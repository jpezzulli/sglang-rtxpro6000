# SPDX-License-Identifier: Apache-2.0
"""Hot-vocabulary (FR-Spec) selection for a target lm_head shared with a draft.

``--speculative-token-map`` stores *global* token ids.  The resident target
lm_head Parameter, though, is indexed by *local* rows: under the
``VocabParallelEmbedding``/``ParallelLMHead`` layout the vocab is padded and
sharded across the TP group, so this rank's tensor covers only its
``shard_indices`` ranges, and LoRA-added ids load behind the padded
org-vocab block.  Indexing that tensor with global ids either raises
index-out-of-bounds or -- worse, when the id happens to fit -- mixes rows
across the shard boundary.  With the SM120 online-FP8 head the mix-up also
separates quantized values from their per-row scales, because
``select_rowwise_weight_rows`` reindexes data and scales together and so
trusts the caller's row indices to be local rows of *this* rank.  This is
the FR-Spec NEXTN TP>1 startup failure class; the qualified TP1 default is a
full-width head whose global ids already are resident rows.

``hot_vocab_local_rows`` is the global-id -> local-row translation, built
from the head layer's own shard ranges, and it fails closed (never drops,
clamps, or filters ids) when an id is not a row this tensor owns.

When the shared head really is TP-sharded, no single rank owns the whole
hot set, so ``shared_hot_lm_head`` assembles the selected rows into one
replicated ``(num_hot, K)`` head on every rank -- each value travelling with
its rowwise scale, contributed over the TP group's object comm -- and the
caller marks the draft's logits processor as not gathering, because every
rank's draft logits then already span the full hot width.  That is the same
contract the qualified TP1 FR-Spec path already uses (and the width the
draft-extend graph buffer takes from ``len(hot_token_id)``), so token-id
space and accept-rate semantics are unchanged; the price is a replicated
draft head and one full-width projection per rank (the exact alternative --
local shard selection plus the usual logits all-gather -- needs the
d2t/draft-logits width plumbing the eager draft path lacks, so it is not
reached by silently filtering ids).  A full-width head (TP1, tensor-parallel
disabled, or replicated vocab) keeps the pure local selection: no
collectives, no extra log, and no hot-vocab message that a TP1 operator
could mistake for a warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch

_MAX_REPORTED_IDS = 8


def _format_ids(ids: Sequence[int]) -> str:
    shown = ", ".join(str(int(x)) for x in ids[:_MAX_REPORTED_IDS])
    if len(ids) > _MAX_REPORTED_IDS:
        shown += f", ... ({len(ids)} total)"
    return shown


def _owned_row(global_id: int, shard_indices) -> int | None:
    """Local row of ``global_id`` in this rank's resident head, if it owns it.

    Mirrors ``VocabParallelEmbedding`` exactly: ``weight_loader`` narrows the
    checkpoint to ``[org_vocab_start_index, org_vocab_end_index)`` and writes
    it from parameter row 0, and added (LoRA) ids follow the *padded* org
    block (see ``get_masked_input_and_mask`` /
    ``get_sharded_to_full_mapping``); rows in between are org padding whose
    token id is -1 and are therefore not owned by any id.
    """
    if (
        shard_indices.org_vocab_start_index
        <= global_id
        < shard_indices.org_vocab_end_index
    ):
        return global_id - shard_indices.org_vocab_start_index
    if (
        shard_indices.added_vocab_start_index
        <= global_id
        < shard_indices.added_vocab_end_index
    ):
        return (
            global_id
            - shard_indices.added_vocab_start_index
            + shard_indices.num_org_elements_padded
        )
    return None


def hot_vocab_local_rows(
    global_ids: Sequence[int] | torch.Tensor,
    *,
    shard_indices=None,
    local_rows: int,
    allow_unowned: bool = False,
    device: torch.device | str | None = None,
):
    """Translate global hot token ids into this rank's local head rows.

    ``shard_indices`` is a ``VocabParallelEmbeddingShardIndices`` for *this*
    rank's vocab-parallel head; ``None`` means a full-width unsharded tensor
    (an ordinary ``nn.Parameter`` head, or a replicated vocab-parallel head),
    where global ids already are resident rows.  ``local_rows`` is the
    resident tensor's row count, which the result must stay inside; the row
    must also be non-negative, so a negative id (the ``-1`` padding sentinel
    included) is never treated as the end of the head wrapping around.

    By default an id this head cannot address raises (fail closed), listing
    the offending ids so nobody reads a silently wrong draft vocabulary;
    ``allow_unowned=True`` instead returns ``(rows, positions, unowned_ids)``,
    where ``rows``/``positions`` describe what this rank owns: its local
    tensor rows and the hot-list positions they fill.

    ``device`` places the returned index tensors on the head's device (CUDA
    rejects a device tensor indexed by a CPU index); the default keeps them
    on CPU exactly like the base selection code did.
    """
    index_kwargs = {"dtype": torch.long} if device is None else {
        "dtype": torch.long,
        "device": device,
    }
    ids = (
        global_ids.tolist()
        if isinstance(global_ids, torch.Tensor)
        else [int(x) for x in global_ids]
    )
    positions: List[int] = []
    rows: List[int] = []
    unowned: List[int] = []
    for position, global_id in enumerate(ids):
        if shard_indices is None:
            row: int | None = int(global_id)
        else:
            row = _owned_row(global_id, shard_indices)
        if row is None or not 0 <= row < local_rows:
            unowned.append(global_id)
        else:
            positions.append(position)
            rows.append(int(row))
    if unowned and not allow_unowned:
        if shard_indices is None:
            layout = f"full-width head of {local_rows} rows"
        else:
            layout = (
                f"shard [org {shard_indices.org_vocab_start_index}."
                f"{shard_indices.org_vocab_end_index}, added "
                f"{shard_indices.added_vocab_start_index}."
                f"{shard_indices.added_vocab_end_index}) of {local_rows} rows"
            )
        raise ValueError(
            "FR-Spec hot token ids are not rows of this rank's lm_head "
            f"{layout}: {_format_ids(unowned)}"
        )
    if allow_unowned:
        return (
            torch.tensor(rows, **index_kwargs),
            torch.tensor(positions, **index_kwargs),
            unowned,
        )
    return torch.tensor(rows, **index_kwargs)


def select_head_rows(head_weight: torch.Tensor, local_rows: torch.Tensor):
    """Select rows from a target head without splitting values from scales.

    Mirrors the two resident-weight states: an SM120 online-FP8 head
    reindexes its rowwise scales through ``select_rowwise_weight_rows``, an
    ordinary BF16 head selects data rows in place on a clone.
    """
    from sglang.kernels.ops.gemm.sm120_online_fp8 import (
        rowwise_scale_of,
        select_rowwise_weight_rows,
    )

    if rowwise_scale_of(head_weight) is not None:
        # The target's per-row scales must undergo the identical
        # hot-vocabulary selection before installation in the draft.
        return select_rowwise_weight_rows(head_weight, local_rows)
    head = head_weight.clone()
    head.data = head.data[local_rows]
    return head


def _wire_dtype(tensor: torch.Tensor) -> torch.Tensor:
    # FP8 has no NCCL/Gloo collective support; the bytes travel as int8.
    return tensor.view(torch.int8) if tensor.dtype == torch.float8_e4m3fn else tensor


def _rank_contributions(
    *,
    head_weight: torch.Tensor,
    hot_token_id: torch.Tensor,
    target_lm_head,
    local_rows: int,
    shard_indices,
):
    """This rank's contribution dict (and the local scale source) or None.

    ``None`` means the head is not actually sharded, i.e. the caller takes
    the full-width local selection path.
    """
    from sglang.kernels.ops.gemm.sm120_online_fp8 import rowwise_scale_of

    owned_rows, owned_positions, _unowned = hot_vocab_local_rows(
        hot_token_id,
        shard_indices=shard_indices,
        local_rows=local_rows,
        allow_unowned=True,
        device=head_weight.device,
    )
    selected = select_head_rows(head_weight, owned_rows)
    # Ship the scale from the source by the same row indices: a device move
    # of the selected tensor may drop its custom scale attribute.
    source_scale = rowwise_scale_of(head_weight)
    # No per-rank pass/fail field: an id outside *this* shard is normal at
    # TP>1 (a peer owns it). Coverage is verified after the exchange from
    # the identical gathered list on every rank, so a hole or a double claim
    # raises the same error on all ranks instead of only on one.
    return {
        "positions": owned_positions.tolist(),
        "data": _wire_dtype(selected.cpu()),
        "scale": (
            None
            if source_scale is None
            # owned_rows is already on the head's device, which is the
            # scale's device for an SM120 online-FP8 weight.
            else source_scale.index_select(0, owned_rows).cpu()
        ),
    }


@dataclass(frozen=True)
class HotHeadSelection:
    """The head to install on the draft, plus the logits-gathering contract."""

    head: torch.Tensor
    # True when every rank received the full hot width, so the draft must NOT
    # all-gather its logits across TP (doing so would double the width).
    replicated_full_width: bool
    draft_vocab_size: int


def shared_hot_lm_head(
    head_weight: torch.Tensor,
    *,
    hot_token_id: torch.Tensor,
    target_lm_head=None,
    logger: logging.Logger | None = None,
    exchange: Callable[[object], List[object]] | None = None,
) -> HotHeadSelection:
    """Build the hot-vocabulary head the draft shares, honoring shard layout.

    The assembled replicated head (values *and* rowwise scale) is returned on
    the source head's own device, so the draft can install it without moving
    the custom scale attribute to the wrong device.

    ``head_weight`` is the target's resident lm_head Parameter for this rank
    and ``target_lm_head`` its owning layer, which supplies the shard layout
    metadata (``shard_indices``/``tp_size``/``num_embeddings_padded``) -- the
    same object ``init_lm_head`` already unwraps from LoRA for the sharing
    check, so no process-global parallel state has to be published to decide
    the path.  Full-width heads take the local selection path at any TP size
    (the qualified TP1 behavior, byte for byte).  A sharded head assembles
    the replicated ``(num_hot, K)`` head instead of indexing this rank's
    shard with global ids: every rank contributes the rows it owns (values
    and rowwise scales together) via ``exchange``, an
    ``GroupCoordinator.all_gather_object``-compatible callable, so the
    exchange runs on the group's CPU comm and never depends on NCCL
    supporting FP8 payloads.
    """
    layer = target_lm_head if target_lm_head is not None else head_weight
    local_rows = int(head_weight.shape[0])
    shard_indices = getattr(layer, "shard_indices", None)
    layer_tp_size = int(getattr(layer, "tp_size", 1) or 1)
    total_padded = getattr(layer, "num_embeddings_padded", None)
    sharded = (
        shard_indices is not None
        and total_padded is not None
        and layer_tp_size > 1
        and local_rows < int(total_padded)
    )
    if not sharded:
        # TP1 / unsharded / replicated vocab: the qualified local path. No
        # collective, no log, and ids that are not resident rows are a hard
        # error rather than a filter.
        rows = hot_vocab_local_rows(
            hot_token_id,
            shard_indices=None,
            local_rows=local_rows,
            device=head_weight.device,
        )
        return HotHeadSelection(
            head=select_head_rows(head_weight, rows),
            replicated_full_width=False,
            draft_vocab_size=int(hot_token_id.shape[0]),
        )

    contribution = _rank_contributions(
        head_weight=head_weight,
        hot_token_id=hot_token_id,
        target_lm_head=layer,
        local_rows=local_rows,
        shard_indices=shard_indices,
    )
    from sglang.kernels.ops.gemm.sm120_online_fp8 import rowwise_scale_of

    expects_scales = rowwise_scale_of(head_weight) is not None

    if exchange is None:
        from sglang.srt.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        if group.world_size != layer_tp_size:
            raise RuntimeError(
                "FR-Spec hot-vocabulary draft sharing expects the target "
                f"lm_head shard count ({layer_tp_size}) to match its TP group "
                f"({group.world_size})"
            )
        exchange = group.all_gather_object
    pieces: List[object] = exchange(contribution)
    if len(pieces) != layer_tp_size:
        raise RuntimeError(
            "FR-Spec hot-vocabulary gather expected one contribution per TP "
            f"rank ({layer_tp_size}), got {len(pieces)}"
        )
    for rank, piece in enumerate(pieces):
        if not isinstance(piece, dict) or not isinstance(piece.get("positions"), list):
            raise RuntimeError(
                "FR-Spec hot-vocabulary gather received a malformed "
                f"contribution from rank {rank}"
            )

    num_hot = int(hot_token_id.shape[0])
    feature_width = int(head_weight.shape[1])
    is_fp8 = head_weight.dtype == torch.float8_e4m3fn
    # Assemble directly on the head's own device (and stage indices there
    # too): the draft installs this Parameter -- with its rowwise scale
    # attribute, which ``Tensor.to`` would leave behind -- without any
    # device move of its own, and the CUDA logits path rejects a weight
    # that does not match the model's device. Coverage bookkeeping stays on
    # the CPU (positions already travel as plain lists), so no data-dependent
    # tensor read (bool/nonzero) is ever attempted on the target device.
    device = head_weight.device
    assembled = torch.zeros(
        (num_hot, feature_width),
        dtype=torch.float32 if is_fp8 else head_weight.dtype,
        device=device,
    )
    assembled_scale = (
        torch.zeros(num_hot, dtype=torch.float32, device=device)
        if expects_scales
        else None
    )
    seen = [False] * num_hot
    for rank, piece in enumerate(pieces):
        positions = piece["positions"]
        index = torch.tensor(positions, dtype=torch.long, device=device)
        data = piece["data"]
        if is_fp8 and data.dtype == torch.int8:
            data = data.view(torch.float8_e4m3fn)
        if data.shape[0] != len(positions) or data.shape[1] != feature_width:
            raise RuntimeError(
                f"FR-Spec hot-vocabulary contribution from rank {rank} has "
                f"rows {tuple(data.shape)} for {len(positions)} positions of "
                f"width {feature_width}"
            )
        if any(seen[position] for position in positions):
            raise RuntimeError(
                f"FR-Spec hot token positions were claimed by more than one "
                f"TP rank (rank {rank} re-claims an owned hot id); shard "
                "ownership must be disjoint"
            )
        for position in positions:
            seen[position] = True
        # Gathered payloads travel on CPU (object comm); bring them to the
        # assembly device before the in-place copies.
        assembled.index_copy_(
            0, index, data.float().to(device=device, dtype=assembled.dtype)
        )
        if assembled_scale is not None:
            scale = piece.get("scale")
            if scale is None or scale.shape[0] != len(positions):
                raise RuntimeError(
                    f"FR-Spec hot-vocabulary contribution from rank {rank} "
                    "lost the rowwise scales matching its selected rows"
                )
            assembled_scale.index_copy_(0, index, scale.float().to(device=device))
    if not all(seen):
        hot_ids = hot_token_id.detach().cpu().tolist()
        missing = [hot_ids[position] for position, hit in enumerate(seen) if not hit]
        raise RuntimeError(
            "FR-Spec hot token ids were claimed by no TP rank's lm_head "
            f"shard: {_format_ids(missing)}"
        )

    head = assembled.to(torch.float8_e4m3fn) if is_fp8 else assembled
    assert head.device == device and assembled.device == device
    # The sharing code installs this through ``model.lm_head.weight``; keep
    # the same Parameter type the target head had (select_rowwise_weight_rows
    # and the BF16 clone path both preserve it).
    head = torch.nn.Parameter(head, requires_grad=False)
    if assembled_scale is not None:
        from sglang.kernels.ops.gemm.sm120_online_fp8 import _SCALE_ATTR

        setattr(head, _SCALE_ATTR, assembled_scale)
    if logger is not None:
        logger.info(
            "FR-Spec at TP=%d: assembled the %d-row hot-vocabulary lm_head "
            "from the target's vocab-parallel shards and installed it "
            "replicated on the draft; the draft's logits processor skips the "
            "TP all-gather because each rank already projects the full hot "
            "width (exact for this topology; the cost is a replicated draft "
            "head and one full-width projection per rank).",
            layer_tp_size,
            num_hot,
        )
    return HotHeadSelection(
        head=head, replicated_full_width=True, draft_vocab_size=num_hot
    )
