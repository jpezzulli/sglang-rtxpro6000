"""Slot arithmetic for the QSA pending index-K ring under N compression groups.

Ported from sgl-project/sglang PR 40143 ("[Spec] QSA: let the pending
index-K ring hold more than one compression group", aiueo52, 2026), with
Pennyroyal additions pinning the pool-side group derivation and the
speculative-window guard. The ring holds ``num_groups`` groups of
``compress_ratio`` slots per request, so a request owns
``compress_ratio * num_groups`` slots and a verify window up to that width
maps every position to a distinct slot. ``num_groups == 1`` is the
single-group layout non-speculative serving runs on, and these tests pin the
new arithmetic to it bit-for-bit: a run without speculation must not move.

The width-support layer added on top of that port is pinned here too: the
group count is the LAUNCH capacity derived from the resolved maximum
draft-token window, eager and in-graph metadata must derive slots from that
one capacity, and a narrower active width must reuse the allocation the
captured graphs hold. The sizing bound carries no small-width exemption, per
sgl-project/sglang#46125 ("[Fix] QSA: Stop speculative verify from
overwriting the pending index-key ring", andyl98) -- at ratio 4 the answer is
W1 -> 1, W2-W4 -> 2, W8 -> 3, W16 -> 5 groups, and every window that can
complete a group with retained prefix members is refused on a ring too small
to hold them apart.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
    pending_ring_groups,
    pending_ring_groups_required,
    pending_ring_slot,
)
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

RATIO = 4


def _legacy_slots(requests, positions, ratio):
    """The pre-parameterization formula, kept as the ``num_groups == 1`` oracle."""
    return requests * ratio + positions % ratio


def _reqs_like(positions, req):
    return torch.full_like(positions, req, dtype=torch.long)


class TestPendingRingSlot(unittest.TestCase):
    def test_single_group_matches_legacy_formula(self):
        """The safety net: N=1 must reproduce the old slots exactly."""
        positions = torch.arange(-8, 48, dtype=torch.long)
        for req in (1, 2, 7):
            with self.subTest(req=req):
                requests = _reqs_like(positions, req)
                got = pending_ring_slot(
                    requests, positions, compress_ratio=RATIO, num_groups=1
                )
                want = _legacy_slots(requests, positions, RATIO)
                self.assertTrue(
                    torch.equal(got, want),
                    f"N=1 diverged for req={req}: {got.tolist()} != {want.tolist()}",
                )

    def test_window_within_capacity_is_collision_free(self):
        """A window of exactly ``ratio * num_groups`` positions maps injectively."""
        for num_groups in (1, 2, 4):
            with self.subTest(num_groups=num_groups):
                capacity = RATIO * num_groups
                positions = torch.arange(100, 100 + capacity, dtype=torch.long)
                slots = pending_ring_slot(
                    _reqs_like(positions, 3),
                    positions,
                    compress_ratio=RATIO,
                    num_groups=num_groups,
                )
                self.assertEqual(slots.unique().numel(), capacity)

    def test_window_beyond_capacity_reuses_slots(self):
        """The documented limit: capacity + 1 positions cannot all be distinct."""
        for num_groups in (1, 2, 4):
            with self.subTest(num_groups=num_groups):
                capacity = RATIO * num_groups
                positions = torch.arange(100, 100 + capacity + 1, dtype=torch.long)
                slots = pending_ring_slot(
                    _reqs_like(positions, 3),
                    positions,
                    compress_ratio=RATIO,
                    num_groups=num_groups,
                )
                self.assertLess(slots.unique().numel(), capacity + 1)

    def test_requests_own_disjoint_slot_ranges(self):
        """Request r owns exactly ``[r * ratio * N, (r + 1) * ratio * N)``."""
        for num_groups in (1, 2, 4):
            with self.subTest(num_groups=num_groups):
                capacity = RATIO * num_groups
                positions = torch.arange(0, 64, dtype=torch.long)
                seen = {}
                for req in (1, 2, 3):
                    slots = pending_ring_slot(
                        _reqs_like(positions, req),
                        positions,
                        compress_ratio=RATIO,
                        num_groups=num_groups,
                    )
                    self.assertGreaterEqual(int(slots.min()), req * capacity)
                    self.assertLess(int(slots.max()), (req + 1) * capacity)
                    seen[req] = set(slots.tolist())
                self.assertFalse(seen[1] & seen[2])
                self.assertFalse(seen[2] & seen[3])

    def test_group_boundary_neighbours_land_in_adjacent_groups(self):
        """``k * ratio +/- 1`` straddle a group boundary, not the same slot."""
        for num_groups in (2, 4):
            with self.subTest(num_groups=num_groups):
                boundary = RATIO * num_groups * 3  # a group boundary, far from 0
                positions = torch.tensor(
                    [boundary - 1, boundary, boundary + 1], dtype=torch.long
                )
                slots = pending_ring_slot(
                    _reqs_like(positions, 1),
                    positions,
                    compress_ratio=RATIO,
                    num_groups=num_groups,
                )
                self.assertEqual(slots.unique().numel(), 3)


class TestBuildPendingRingSlots(unittest.TestCase):
    def _build(self, positions, reqs, lengths, num_groups, is_extend):
        positions = torch.tensor(positions, dtype=torch.long)
        return build_pending_ring_slots(
            token_to_batch_idx=torch.arange(positions.numel()),
            req_pool_indices=torch.tensor(reqs, dtype=torch.long),
            sequence_lengths=torch.tensor(lengths, dtype=torch.long),
            logical_positions=positions,
            compress_ratio=RATIO,
            num_groups=num_groups,
            is_extend=is_extend,
        )

    def test_defaults_to_single_group(self):
        """Omitting ``num_groups`` keeps the historical layout."""
        positions = [0, 1, 2, 3]
        got = build_pending_ring_slots(
            token_to_batch_idx=torch.arange(4),
            req_pool_indices=torch.tensor([1, 1, 1, 1]),
            sequence_lengths=torch.tensor([8, 8, 8, 8]),
            logical_positions=torch.tensor(positions),
            compress_ratio=RATIO,
            is_extend=False,
        )
        want = _legacy_slots(torch.tensor([1, 1, 1, 1]), torch.tensor(positions), RATIO)
        self.assertTrue(torch.equal(got, want))

    def test_extend_dump_stays_in_the_inert_region(self):
        """Non-pending extend tokens dump into rows ``[0, ratio)``.

        Request slot 0 is never allocated, so that region is inert. It must stay
        disjoint from every allocated request's range at any N.
        """
        # request 1, sequence length 8 -> pending tail starts at position 8.
        positions = [0, 3, 4, 8, 9]
        lengths = [8] * len(positions)
        reqs = [1] * len(positions)
        for num_groups in (1, 2, 4):
            with self.subTest(num_groups=num_groups):
                slots = self._build(positions, reqs, lengths, num_groups, True)
                capacity = RATIO * num_groups
                dumped = slots[:3]  # positions 0, 3, 4 are before the pending tail
                pending = slots[3:]  # positions 8, 9 are the pending group
                self.assertTrue(bool((dumped < RATIO).all()))
                self.assertTrue(bool((pending >= capacity).all()))

    def test_extend_pending_tail_matches_plain_formula(self):
        """Pending tokens use the same slots whether or not extend is set."""
        positions = [8, 9, 10, 11]
        for num_groups in (1, 2, 4):
            with self.subTest(num_groups=num_groups):
                extended = self._build(positions, [1] * 4, [8] * 4, num_groups, True)
                plain = self._build(positions, [1] * 4, [8] * 4, num_groups, False)
                self.assertTrue(torch.equal(extended, plain))


class TestBuildGroupRingSlots(unittest.TestCase):
    def _build(self, group_ends, reqs, num_groups):
        return build_group_ring_slots(
            req_pool_indices=torch.tensor(reqs, dtype=torch.long),
            group_end_positions=torch.tensor(group_ends, dtype=torch.long),
            sequence_ids=torch.arange(len(group_ends)),
            compress_ratio=RATIO,
            num_groups=num_groups,
        )

    def test_members_are_oldest_first(self):
        """Column k holds the k-th oldest member of the group."""
        # group ending at 7 spans positions 4..7, oldest first.
        slots = self._build([7], [1], num_groups=1)
        self.assertEqual(slots.shape, (1, RATIO))
        positions = [4, 5, 6, 7]
        want = _legacy_slots(torch.tensor([1] * 4), torch.tensor(positions), RATIO)
        self.assertTrue(torch.equal(slots[0], want))

    def test_all_members_share_one_ring_group(self):
        """Members of one group share a group index even when the end is unaligned.

        The extend producer always emits ``group_end = blocks * ratio + ratio - 1``
        (aligned), but the graph producer feeds ``lengths - 1``, which is only
        aligned on the boundary rows. Deriving the group from ``group_end`` rather
        than from each member's own position keeps a group together either way.
        """
        for num_groups in (2, 4):
            with self.subTest(num_groups=num_groups):
                capacity = RATIO * num_groups
                group_ends = [RATIO * 4 - 1, RATIO * 4, RATIO * 4 + 1]
                slots = self._build(group_ends, [1] * 3, num_groups)
                for row, end in enumerate(group_ends):
                    group = (end // RATIO) % num_groups
                    lo = capacity + group * RATIO
                    hi = lo + RATIO
                    self.assertTrue(
                        bool(((slots[row] >= lo) & (slots[row] < hi)).all()),
                        f"row {row} (end={end}) straddled ring groups: "
                        f"{slots[row].tolist()}",
                    )

    def test_single_group_matches_legacy_formula(self):
        """N=1 regression oracle for the group builder."""
        group_ends = [7, 11, 15]
        slots = self._build(group_ends, [1, 2, 3], num_groups=1)
        for row, end in enumerate(group_ends):
            positions = list(range(end - RATIO + 1, end + 1))
            want = _legacy_slots(
                torch.tensor([row + 1] * RATIO), torch.tensor(positions), RATIO
            )
            self.assertTrue(torch.equal(slots[row], want))

    def test_members_clamp_at_zero(self):
        """A group at the very start clamps negative members to position 0."""
        slots = self._build([1], [1], num_groups=1)
        positions = [0, 0, 0, 1]
        want = _legacy_slots(torch.tensor([1] * 4), torch.tensor(positions), RATIO)
        self.assertTrue(torch.equal(slots[0], want))


class TestPendingRingNumGroups(unittest.TestCase):
    def test_rounds_up_over_window_plus_retained_prefix(self):
        """The ring spans the published window plus the retained prefix tail.

        A paged forward publishes every window position before compressing
        any group (qsa_indexer.update_key_state_and_compress), and the first
        completed group may still read up to ratio-1 retained prefix
        members, so capacity covers draft_tokens + ratio - 1 consecutive
        positions from W2 up. Only an absent draft-token count
        (non-speculative run) or a one-row window (plain decode) keeps the
        single-group layout: the window's own later rows are what overwrite
        the members it reads, and one row has no later rows.
        """
        cases = [
            (None, 1),
            (0, 1),
            (1, 1),  # decode / no speculation: unchanged layout
            (2, 2),  # #46125: a 2-row window at an unaligned start, and
            (3, 2),  # every wider window up to the ratio, overwrite a live
            (4, 2),  # member that a one-group ring keeps no room for
            (5, 3),
            (7, 3),
            (8, 3),
            (9, 4),
            (12, 4),
            (16, 5),
        ]
        for max_num_draft_tokens, expected in cases:
            got = QSATokenToKVPool.pending_ring_num_groups(
                max_num_draft_tokens=max_num_draft_tokens, compress_ratio=4
            )
            self.assertEqual(got, expected, (max_num_draft_tokens, got))
        with self.assertRaises(ValueError):
            QSATokenToKVPool.pending_ring_num_groups(
                max_num_draft_tokens=4, compress_ratio=0
            )


class TestPendingRingCapacity(unittest.TestCase):
    """The ring buffer scales with the group count a wider window needs.

    Dropping ``* qsa_num_groups`` from the capacity still allocates a
    valid-looking buffer, so nothing else in the suite notices until a long
    window overwrites itself; pin the shape at boot instead.
    """

    def _pool(self, num_groups):
        from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

        return QSATokenToKVPool(
            size=256,
            dtype=torch.bfloat16,
            page_size=64,
            head_num=2,
            head_dim=64,
            full_attention_layer_ids=[0],
            device="cpu",
            mamba_pool=None,
            qsa_index_kv_heads=1,
            qsa_index_head_dim=8,
            qsa_compress_ratio=RATIO,
            qsa_token_topk=8,
            num_request_slots=3,
            qsa_num_groups=num_groups,
        )

    def test_capacity_tracks_groups(self):
        self.assertEqual(
            self._pool(1).qsa_key_state_buffer_pool[0].shape, (3 * RATIO * 1, 1, 8)
        )
        self.assertEqual(
            self._pool(2).qsa_key_state_buffer_pool[0].shape, (3 * RATIO * 2, 1, 8)
        )
        self.assertEqual(
            self._pool(2).qsa_rope_position_buffer.shape, (3 * RATIO * 2, 3)
        )

    def test_rejects_degenerate_groups(self):
        with self.assertRaises(ValueError):
            self._pool(0)


class TestPublishedWindowRingLifetime(unittest.TestCase):
    """Production-path lifetime of the pending ring across a W8 verify.

    Exercises the real QSAIndexer.update_key_state_and_compress on CPU: a
    retained prefix member p0 (written by an earlier forward), then one
    speculative-paged forward publishing p1..p8 (each token key equals its
    position), where the length-4 row completes group 0 whose members are
    read back FROM the ring after the whole window was published. The naive
    ceil(window / ratio) capacity (2 groups) aliases p8 onto p0 and the
    compressed group silently mixes positions from two different walks of
    the ring (mean 3.5, and p0 stays corrupted after a rejection); the
    window + retained-prefix span (3 groups at W8) keeps them distinct. The
    fused-prep (state_stored) call must not republish and must reach the
    identical result, and the cross-prefix ordering guard stays armed.
    """

    R = 4
    REQ = 1

    def _indexer(self):
        from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

        indexer = QSAIndexer.__new__(QSAIndexer)
        indexer.layer_id = 0
        indexer.compress_ratio = self.R
        indexer.rotary_emb = SimpleNamespace(mrope_section=None)
        # Keep the CPU window off the triton kernels while still running the
        # real publish -> gather -> mean path (the fused-store equivalence
        # is a GPU-window check; here we pin addressing and ordering).
        indexer._use_fused_compress = lambda pool: False
        indexer.normalize_compressed_keys = lambda keys, positions: keys
        return indexer

    def _metadata(self, *, pool, lengths, positions):
        from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            QwenSparseAttnBackend,
        )

        rows = len(lengths)
        width = 128
        cols = torch.arange(width)
        table = (
            ((self.REQ + cols // 64) * 64 + cols % 64).to(torch.int32).repeat(rows, 1)
        )
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.token_to_kv_pool = SimpleNamespace(qsa_compress_ratio=self.R)
        plan = backend._qsa_build_write_plan(
            forward_batch=SimpleNamespace(
                forward_mode=SimpleNamespace(is_decode=lambda: False),
                extend_seq_lens=None,
                input_ids=torch.zeros(rows),
            ),
            speculative_paged=True,
            token_slot_table=table,
            sequence_lengths=torch.tensor(lengths, dtype=torch.int32),
        )
        write_locs, group_positions, sequence_ids, member_rows, _, _ = plan
        return QSAIndexerMetadata(
            sequence_lengths=torch.tensor(lengths, dtype=torch.int32),
            token_to_batch_idx=torch.zeros(rows, dtype=torch.long),
            token_slot_table=table,
            out_cache_loc=torch.arange(rows, dtype=torch.int64),
            token_to_kv_pool=pool,
            compress_ratio=self.R,
            block_topk=2,
            req_pool_indices=torch.full((rows,), self.REQ, dtype=torch.long),
            write_locs=write_locs,
            compress_group_positions=group_positions,
            compress_sequence_ids=sequence_ids,
            compress_member_rows=member_rows,
            is_cuda_graph=False,
        )

    def _pool(self, num_groups):
        return QSATokenToKVPool(
            size=256,
            dtype=torch.bfloat16,
            page_size=64,
            head_num=2,
            head_dim=64,
            full_attention_layer_ids=[0],
            device="cpu",
            mamba_pool=None,
            qsa_index_kv_heads=1,
            qsa_index_head_dim=self.R,
            qsa_compress_ratio=self.R,
            qsa_token_topk=8,
            num_request_slots=4,
            qsa_num_groups=num_groups,
        )

    def _run(self, num_groups: int):
        pool = self._pool(num_groups)
        indexer = self._indexer()

        def keys(positions):
            return (
                torch.tensor(positions, dtype=torch.float32)
                .view(-1, 1, 1)
                .expand(-1, 1, self.R)
                .contiguous()
            )

        # Forward A: the prefix leaves p0 pending in the ring.
        indexer.update_key_state_and_compress(
            keys([0]),
            torch.tensor([0]),
            torch.zeros(3, 1, dtype=torch.int64),
            self._metadata(pool=pool, lengths=[1], positions=[0]),
        )
        # Forward B: the verify window publishes p1..p8 and then compresses
        # group 0 (length 4) from the ring.
        positions = list(range(1, 9))
        lengths = [position + 1 for position in positions]
        meta = self._metadata(pool=pool, lengths=lengths, positions=positions)
        indexer.update_key_state_and_compress(
            keys(positions),
            torch.tensor(positions),
            torch.zeros(3, len(positions), dtype=torch.int64),
            meta,
        )
        # The fused-prep call (state_stored) must not republish and must
        # reach the identical result.
        indexer.update_key_state_and_compress(
            torch.zeros(len(positions), 1, self.R),
            torch.tensor(positions),
            torch.zeros(3, len(positions), dtype=torch.int64),
            meta,
            state_stored=True,
        )
        state = pool.get_qsa_key_state_buffer(0)
        p0_slot = pending_ring_slot(
            torch.tensor([self.REQ]),
            torch.tensor([0]),
            compress_ratio=self.R,
            num_groups=num_groups,
        )
        first_group_loc = int(meta.write_locs[0])
        compressed = pool.get_qsa_compressed_k_buffer(0)[first_group_loc]
        return float(state[p0_slot].float().mean()), float(compressed.float().mean())

    def test_derived_capacity_preserves_the_retained_prefix(self):
        num_groups = QSATokenToKVPool.pending_ring_num_groups(
            max_num_draft_tokens=8, compress_ratio=self.R
        )
        self.assertEqual(num_groups, 3)
        ring_p0, group0_mean = self._run(num_groups)
        self.assertEqual(ring_p0, 0.0)
        self.assertEqual(group0_mean, 1.5)  # mean(0,1,2,3), not mean(8,1,2,3)

    def test_naive_two_group_capacity_reproduces_the_clobber(self):
        """Pin the exact failure the span formula prevents (supervisor repro)."""
        ring_p0, group0_mean = self._run(2)
        self.assertEqual(ring_p0, 8.0)
        self.assertEqual(group0_mean, 3.5)

    def test_cross_prefix_publish_order_guard_stays_armed(self):
        from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata

        pool = self._pool(3)
        indexer = self._indexer()
        meta = QSAIndexerMetadata(
            sequence_lengths=torch.tensor([2], dtype=torch.int32),
            token_to_batch_idx=torch.zeros(1, dtype=torch.long),
            token_slot_table=torch.zeros((1, 8), dtype=torch.int32),
            out_cache_loc=torch.arange(1, dtype=torch.int64),
            token_to_kv_pool=pool,
            compress_ratio=self.R,
            block_topk=2,
            req_pool_indices=torch.tensor([self.REQ], dtype=torch.long),
            compress_member_rows=torch.tensor([0], dtype=torch.long),
            has_cross_prefix_group=True,
            is_cuda_graph=False,
        )
        with self.assertRaises(RuntimeError):
            indexer.update_key_state_and_compress(
                torch.zeros(1, 1, self.R),
                torch.tensor([1]),
                torch.zeros(3, 1, dtype=torch.int64),
                meta,
                state_stored=True,
            )


class TestRequireChainSpeculation(unittest.TestCase):
    """The verify-window guard asks the ring's own span model, not a bare ratio.

    ``_require_chain_speculation`` and the pool's construction-time capacity
    must be the same arithmetic: a window is legal exactly when the groups it
    needs -- ``pending_ring_groups_required``, the function that sized the ring
    -- fit in the groups the pool holds. The pre-change RC2 guard refused every
    window wider than the ratio outright; a guard that merely multiplies the
    ratio by the group count waves through windows the ring cannot actually
    hold (8 draft tokens at 2 groups alias the retained prefix member, which is
    the clobber TestPublishedWindowRingLifetime pins) -- as did the small-width
    exemption #46125 removed (4 draft tokens at 1 group, the same overwrite at
    the width the shipped policy routed).
    """

    class _Mode:
        def is_target_verify(self):
            return True

    class _NonVerifyMode:
        def is_target_verify(self):
            return False

    def _check(self, draft_tokens, num_groups, ratio=RATIO):
        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            QwenSparseAttnBackend,
        )

        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.compress_ratio = ratio
        backend.token_to_kv_pool = (
            None if num_groups is None else SimpleNamespace(qsa_num_groups=num_groups)
        )
        spec_info = SimpleNamespace(topk=1, draft_token_num=draft_tokens)
        backend._require_chain_speculation(
            TestRequireChainSpeculation._Mode(), spec_info
        )

    # (draft tokens, ring groups, accepted) -- legal iff
    # pending_ring_groups_required(draft) <= groups, at ratio 4.
    WINDOW_CASES = (
        (2, 1, False),  # #46125: no small-width exemption any more
        (3, 1, False),
        (4, 1, False),  # the pre-fix fixed-W4 capacity, now refused at capture
        (4, 2, True),  # the W4 launch capacity
        (5, 2, False),  # one position past the two-group span
        (8, 1, False),  # RC2's restriction stays for a pool nobody re-sized
        (8, 2, False),  # two groups alias the retained prefix member
        (8, 3, True),  # the W8 launch capacity
        (9, 3, False),  # one position past the W8 span
        (9, 4, True),
        (12, 4, True),
        (16, 3, False),  # W16 needs its own launch capacity
        (16, 4, False),
        (16, 5, True),  # the W16 launch capacity
        (17, 5, False),
    )

    def test_window_accepted_exactly_when_the_ring_holds_its_groups(self):
        for draft_tokens, num_groups, accepted in self.WINDOW_CASES:
            with self.subTest(draft_tokens=draft_tokens, num_groups=num_groups):
                if accepted:
                    self._check(draft_tokens, num_groups)
                else:
                    with self.assertRaises(NotImplementedError):
                        self._check(draft_tokens, num_groups)

    def test_launch_capacity_serves_every_narrower_active_width(self):
        """W16-to-W4-to-W16 needs no re-allocation: one capacity serves all.

        The ring is sized off the launch maximum, so any active width the run
        can step down to must still pass the guard against that same group
        count (the adaptive-MTP prerequisite).
        """
        for launch_max in (4, 8, 16):
            groups = QSATokenToKVPool.pending_ring_num_groups(
                max_num_draft_tokens=launch_max, compress_ratio=RATIO
            )
            for active in (1, 2, 4, 8, 16):
                if active > launch_max:
                    continue
                with self.subTest(launch_max=launch_max, active=active):
                    self._check(active, groups)

    def test_guard_matches_the_capacity_the_pool_allocates(self):
        """Guard and pool never disagree about what fits (one span model)."""
        for launch_max in (0, 1, 4, 5, 8, 9, 12, 16, 32):
            groups = QSATokenToKVPool.pending_ring_num_groups(
                max_num_draft_tokens=launch_max, compress_ratio=RATIO
            )
            with self.subTest(launch_max=launch_max):
                self._check(launch_max, groups)
                self.assertEqual(
                    groups,
                    pending_ring_groups_required(
                        draft_tokens=launch_max, compress_ratio=RATIO
                    ),
                )

    def test_unresolved_pool_is_unknown_capacity_not_a_tokenwise_pool(self):
        """No geometry to ask means the window is unproven, not free of charge.

        ``pending_ring_groups`` keeps answering 1 for a pool with no ring, so a
        tokenwise-QSA run (ratio 1, no groups at all) holds its existing
        one-token-only bound, and a backend whose pool has not resolved yet
        refuses every multi-row verify window instead of admitting a geometry
        nobody measured. Capture resolves the real pool before this guard runs
        (``_capture_cuda_graph_metadata``), so the refusal is a start-up error,
        never a mid-serve surprise.
        """
        self._check(1, None)  # decode: one row, one group, always legal
        for draft_tokens in (2, 3, 4, 5):
            with self.subTest(draft_tokens=draft_tokens):
                with self.assertRaises(NotImplementedError):
                    self._check(draft_tokens, None)
        # A genuine tokenwise pool (compress ratio 1, no ring) keeps the
        # W1-only bound it had before the small-width correction.
        for draft_tokens in (2, 4):
            with self.subTest(ratio=1, draft_tokens=draft_tokens):
                with self.assertRaises(NotImplementedError):
                    self._check(draft_tokens, None, ratio=1)
        self.assertEqual(pending_ring_groups(None), 1)
        self.assertEqual(pending_ring_groups(SimpleNamespace()), 1)

    def test_non_verify_modes_are_not_guarded(self):
        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            QwenSparseAttnBackend,
        )

        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.compress_ratio = RATIO
        backend.token_to_kv_pool = SimpleNamespace(qsa_num_groups=1)
        # A wide window on a decode/extend forward is not a verify window.
        backend._require_chain_speculation(
            TestRequireChainSpeculation._NonVerifyMode(),
            SimpleNamespace(topk=1, draft_token_num=64),
        )
        backend._require_chain_speculation(None, None)


# ---------------------------------------------------------------------------
# Width support: the ring is sized for the LAUNCH maximum, and every producer
# (eager builders, the indexer fallbacks, the in-graph kernels and the host
# replay refresh) must derive slots from that one capacity.
# ---------------------------------------------------------------------------

# The verify windows the next adaptive-MTP step must be able to select. RATIO
# (=4) is the shipped window; W8/W16 are the wider candidates.
WIDTHS = (4, 8, 16)


def groups_for(width):
    """The ring capacity a launch offering ``width`` must be sized for."""
    return QSATokenToKVPool.pending_ring_num_groups(
        max_num_draft_tokens=width, compress_ratio=RATIO
    )


def _ring_pool(num_groups, *, num_request_slots=4, head_num=2, index_head_dim=RATIO):
    return QSATokenToKVPool(
        size=256,
        dtype=torch.bfloat16,
        page_size=64,
        head_num=head_num,
        head_dim=64,
        full_attention_layer_ids=[0],
        device="cpu",
        mamba_pool=None,
        qsa_index_kv_heads=1,
        qsa_index_head_dim=index_head_dim,
        qsa_compress_ratio=RATIO,
        qsa_token_topk=8,
        num_request_slots=num_request_slots,
        qsa_num_groups=num_groups,
    )


def _cpu_indexer():
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    indexer = QSAIndexer.__new__(QSAIndexer)
    indexer.layer_id = 0
    indexer.compress_ratio = RATIO
    indexer.rotary_emb = SimpleNamespace(mrope_section=None)
    # Keep the CPU window off the triton kernels while still running the real
    # publish -> gather -> mean path (the fused-store equivalence is a
    # GPU-window check; here we pin addressing and ordering).
    indexer._use_fused_compress = lambda pool: False
    indexer.normalize_compressed_keys = lambda keys, positions: keys
    return indexer


def _paged_metadata(pool, lengths, req):
    """Indexer metadata for one speculative-paged forward (row i has length
    ``lengths[i]``, every row of the forward belonging to request ``req``)."""
    from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QwenSparseAttnBackend,
    )

    rows = len(lengths)
    cols = torch.arange(128)
    table = ((req + cols // 64) * 64 + cols % 64).to(torch.int32).repeat(rows, 1)
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool = SimpleNamespace(qsa_compress_ratio=RATIO)
    write_locs, group_positions, sequence_ids, member_rows, _, _ = (
        backend._qsa_build_write_plan(
            forward_batch=SimpleNamespace(
                forward_mode=SimpleNamespace(is_decode=lambda: False),
                extend_seq_lens=None,
                input_ids=torch.zeros(rows),
            ),
            speculative_paged=True,
            token_slot_table=table,
            sequence_lengths=torch.tensor(lengths, dtype=torch.int32),
        )
    )
    return QSAIndexerMetadata(
        sequence_lengths=torch.tensor(lengths, dtype=torch.int32),
        token_to_batch_idx=torch.zeros(rows, dtype=torch.long),
        token_slot_table=table,
        out_cache_loc=torch.arange(rows, dtype=torch.int64),
        token_to_kv_pool=pool,
        compress_ratio=RATIO,
        block_topk=2,
        req_pool_indices=torch.full((rows,), req, dtype=torch.long),
        write_locs=write_locs,
        compress_group_positions=group_positions,
        compress_sequence_ids=sequence_ids,
        compress_member_rows=member_rows,
        is_cuda_graph=False,
    )


def _keys(positions, dim=RATIO):
    """One constant-valued index key per position, so a clobber is readable."""
    return (
        torch.tensor(positions, dtype=torch.float32)
        .view(-1, 1, 1)
        .expand(-1, 1, dim)
        .contiguous()
    )


def _slot_of(req, position, num_groups):
    return int(
        pending_ring_slot(
            torch.tensor([req]),
            torch.tensor([position]),
            compress_ratio=RATIO,
            num_groups=num_groups,
        )[0]
    )


class TestLaunchCapacityGeometry(unittest.TestCase):
    """The pool's group count is the launch capacity, and it is all the ring is.

    These are the numbers the W4/W8/W16 candidates of the next adaptive-MTP
    step run on; the per-request stride (``qsa_ring_span``), the buffer's row
    count and the request-slot count have to agree with it, because the
    allocation is what the captured CUDA graphs bind for their lifetime.
    """

    def test_supported_widths_and_their_capacities(self):
        self.assertEqual({w: groups_for(w) for w in WIDTHS}, {4: 2, 8: 3, 16: 5})

    def test_capacity_is_monotonic_in_the_launch_width(self):
        capacities = [groups_for(w) for w in range(1, 33)]
        self.assertEqual(capacities, sorted(capacities))
        self.assertEqual(capacities[0], 1)

    def test_ring_rows_are_request_slots_times_the_span(self):
        for width in WIDTHS:
            num_groups = groups_for(width)
            for request_slots in (1, 4, 37):
                pool = _ring_pool(num_groups, num_request_slots=request_slots)
                span = RATIO * num_groups
                with self.subTest(width=width, request_slots=request_slots):
                    self.assertEqual(pool.qsa_num_groups, num_groups)
                    self.assertEqual(pool.qsa_ring_span, span)
                    self.assertEqual(pool.qsa_num_request_slots, request_slots)
                    rows = request_slots * span
                    self.assertEqual(
                        pool.get_qsa_key_state_buffer(0).shape, (rows, 1, RATIO)
                    )
                    self.assertEqual(pool.qsa_rope_position_buffer.shape, (rows, 3))
                    # Every slot any producer can address for the last
                    # allocated request lands inside the rows this pool
                    # allocated (the ring wraps within a request, never into a
                    # neighbour).
                    positions = torch.arange(0, 4 * span, dtype=torch.long)
                    slots = pending_ring_slot(
                        torch.full_like(positions, request_slots - 1),
                        positions,
                        compress_ratio=RATIO,
                        num_groups=num_groups,
                    )
                    self.assertTrue(bool((slots < rows).all()))
                    self.assertTrue(bool((slots >= (request_slots - 1) * span).all()))

    def test_ring_geometry_is_tensor_parallel_invariant(self):
        """TP1 and TP2 ranks allocate the same ring rows.

        The ring is indexed by (request slot, group, offset within the group):
        no term comes from the attention head count, the layer split or the TP
        degree, so both ranks of the TP2 source path address the same rows and
        a rank-local shard cannot shrink the ring out from under a window.
        """
        num_groups = groups_for(16)
        tp1 = _ring_pool(num_groups, head_num=2)
        tp2 = _ring_pool(num_groups, head_num=1)
        self.assertEqual(
            tp1.get_qsa_key_state_buffer(0).shape,
            tp2.get_qsa_key_state_buffer(0).shape,
        )
        self.assertEqual(tp1.qsa_ring_span, tp2.qsa_ring_span)
        self.assertEqual(
            tp1.qsa_rope_position_buffer.shape, tp2.qsa_rope_position_buffer.shape
        )

    def test_producers_read_the_capacity_off_the_pool(self):
        """``pending_ring_groups`` is the only read of the pool's capacity."""
        self.assertEqual(pending_ring_groups(_ring_pool(5)), 5)
        # Pools with no pending ring (tokenwise QSA) and an unresolved backend
        # keep the single-group layout instead of raising AttributeError.
        self.assertEqual(pending_ring_groups(SimpleNamespace()), 1)
        self.assertEqual(pending_ring_groups(None), 1)


class TestLiveSpanIsCollisionFree(unittest.TestCase):
    """What a window really occupies in the ring: itself plus the prefix tail.

    The paged forward publishes every window position before compressing any
    group, and the first group it completes still reads up to ratio-1 retained
    members, so the live span is ``width + ratio - 1`` consecutive positions.
    The launch capacity must keep that whole span distinct -- a bare
    ``ceil(width / ratio)`` round-up does not, which is the bug the sizing rule
    exists to prevent, and it is also why the rule has no small-width case:
    the aliasing rows are the window's own, so a window no wider than one
    group is not exempt (#46125).
    """

    # Every width a speculative launch can route to, the shipped W4 included.
    WIDE = (4, 8, 16)
    STARTS = (5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 64, 65, 66)

    def _live_span(self, start, width):
        # Physical positions only: a prefix tail at the very start of a request
        # clamps onto position 0 (and clamped duplicates are the same key).
        return torch.arange(
            start - (RATIO - 1), start + width, dtype=torch.long
        ).clamp_min(0)

    def _slots(self, positions, num_groups):
        return pending_ring_slot(
            _reqs_like(positions, 3),
            positions,
            compress_ratio=RATIO,
            num_groups=num_groups,
        )

    def test_launch_capacity_keeps_the_live_span_distinct(self):
        for width in self.WIDE:
            num_groups = groups_for(width)
            for start in self.STARTS:
                with self.subTest(width=width, start=start):
                    positions = self._live_span(start, width)
                    slots = self._slots(positions, num_groups)
                    self.assertEqual(
                        slots.unique().numel(),
                        positions.unique().numel(),
                        f"W{width} at groups={num_groups} aliased the live span: "
                        f"{positions.tolist()} -> {slots.tolist()}",
                    )

    def test_naive_window_roundup_reproduces_the_clobber(self):
        """The counter-example the sizing rule exists for."""
        for width in self.WIDE:
            naive = max(1, width // RATIO)
            self.assertLess(naive, groups_for(width))
            for start in self.STARTS:
                with self.subTest(width=width, start=start):
                    positions = self._live_span(start, width)
                    slots = self._slots(positions, naive)
                    self.assertLess(
                        slots.unique().numel(),
                        positions.unique().numel(),
                        f"W{width} at the naive groups={naive} happened not to "
                        f"alias: {positions.tolist()} -> {slots.tolist()}",
                    )

    def test_capacity_covers_the_span_it_is_sized_for(self):
        """No slack, no shortage, at the width the ring was sized for."""
        for width in self.WIDE:
            with self.subTest(width=width):
                self.assertGreaterEqual(RATIO * groups_for(width), width + RATIO - 1)
                self.assertLess(RATIO * (groups_for(width) - 1), width + RATIO - 1)

    def test_w4_no_longer_keeps_an_inherited_prefix_tail_alias(self):
        """The window the pre-fix single-group layout could not cover.

        A W4 verify's live span is 7 positions and the pre-fix capacity gave it
        4 rows: PR41's inherited prefix-tail alias, kept "not claimed correct"
        until #46125 measured what it cost. The sizing rule now rounds W2-W4 up
        to two groups, the span is distinct, and every wider launch capacity
        (the W8/W16 rings, the adaptive-MTP candidate set, the DFlash2 W6) still
        covers an active W4 window as it always did.
        """
        self.assertGreaterEqual(RATIO * groups_for(4), 4 + RATIO - 1)
        positions = self._live_span(8, 4)
        self.assertEqual(positions.unique().numel(), 4 + RATIO - 1)
        self.assertEqual(
            self._slots(positions, groups_for(4)).unique().numel(),
            positions.unique().numel(),
        )
        for num_groups in (2, groups_for(8), groups_for(16)):
            with self.subTest(num_groups=num_groups):
                self.assertEqual(
                    self._slots(positions, num_groups).unique().numel(),
                    positions.unique().numel(),
                )
        # The pre-fix capacity, pinned as the counter-example it is.
        self.assertLess(
            self._slots(positions, 1).unique().numel(), positions.unique().numel()
        )


class TestWideWindowRollbackLifetime(unittest.TestCase):
    """A W16 verify across group boundaries, through a partial accept.

    Production path (``QSAIndexer.update_key_state_and_compress`` on CPU):

    A: the retained prefix leaves position 0 pending in the ring.
    B: a 16-wide verify publishes positions 1..16 and compresses the four
       groups it completes -- the first still reads position 0 from the ring.
    C: only 12 tokens are accepted, so the next verify republishes 13..28 with
       the replacement tokens and re-compresses the group ending at 15, whose
       oldest member (position 12) is retained from B: publishing 13..28 must
       not wrap onto it.

    At the naive ``ceil(window / ratio)`` capacity (4 groups) position 16 lands
    on position 0's slot and position 28 on position 12's, so both compressed
    groups silently mix two walks of the ring. The launch capacity (5 groups)
    keeps every live position distinct.
    """

    REQ = 1

    def _run(self, num_groups):
        pool = _ring_pool(num_groups, num_request_slots=4)
        indexer = _cpu_indexer()

        def publish(positions, key_values):
            meta = _paged_metadata(
                pool, [position + 1 for position in positions], self.REQ
            )
            indexer.update_key_state_and_compress(
                _keys(key_values),
                torch.tensor(positions),
                torch.zeros(3, len(positions), dtype=torch.int64),
                meta,
            )
            return meta

        publish([0], [0])
        meta_b = publish(list(range(1, 17)), list(range(1, 17)))
        positions_c = list(range(13, 29))
        meta_c = publish(positions_c, [100 + position for position in positions_c])

        state = pool.get_qsa_key_state_buffer(0)
        compressed = pool.get_qsa_compressed_k_buffer(0)
        return {
            "group_0_3": float(compressed[int(meta_b.write_locs[0])].float().mean()),
            "p12": float(state[_slot_of(self.REQ, 12, num_groups)].float().mean()),
            "group_12_15": float(compressed[int(meta_c.write_locs[0])].float().mean()),
        }

    def test_launch_capacity_survives_the_publish_and_rollback(self):
        self.assertEqual(groups_for(16), 5)
        self.assertEqual(
            self._run(5), {"group_0_3": 1.5, "p12": 12.0, "group_12_15": 88.5}
        )
        # (position 20 legitimately reuses position 0's ring row: the group it
        # belongs to was compressed a whole forward earlier, so only the
        # still-live member 12 must survive.)

    def test_naive_four_group_capacity_clobbers_both_groups(self):
        observed = self._run(4)
        self.assertEqual(observed["p12"], 128.0)  # position 28 landed on position 12
        self.assertEqual(
            observed["group_0_3"], 5.5
        )  # mean(16,1,2,3), not mean(0,1,2,3)
        self.assertEqual(
            observed["group_12_15"], 117.5
        )  # mean(128,113,114,115), not mean(12,113,114,115)


def _verify_layout(bases, reqs, width, num_padding=0):
    """Rows the in-graph layout kernel produces for a target-verify batch.

    Independent re-derivation (request i owns ``width`` rows of growing length,
    the DP-padded tail aliases to request slot 0 with length 1) used as the
    oracle the graph buffers are compared against.
    """
    real = len(bases) - num_padding
    lengths, row_reqs = [], []
    for index, (base, req) in enumerate(zip(bases, reqs)):
        if index < real:
            lengths += [base + offset + 1 for offset in range(width)]
            row_reqs += [req] * width
    lengths += [1] * (num_padding * width)
    row_reqs += [0] * (num_padding * width)
    return torch.tensor(lengths, dtype=torch.int32), torch.tensor(
        row_reqs, dtype=torch.int32
    )


class _TargetVerifyMode:
    """Stand-in for ``ForwardMode.TARGET_VERIFY``.

    The backend caches captured graph metadata keyed on the mode object, so this
    is a hashable singleton rather than a SimpleNamespace.
    """

    def is_target_verify(self):
        return True

    def is_draft_extend_v2(self):
        return False

    def is_decode(self):
        return False


TARGET_VERIFY = _TargetVerifyMode()


class _GraphMetadataHarness:
    """A backend with CUDA-graph state on CPU, for the replay-metadata paths.

    ``init_cuda_graph_state`` is exercised for real; only the pinned draft-extend
    staging (which needs an accelerator allocator this host has none of, and
    which the target-verify path never reads) is filled in by hand.
    """

    CTX = 256
    MAX_BS = 3

    def __init__(self, num_groups, *, bases=(10, 20, 30), reqs=(1, 2, 3)):
        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            QwenSparseAttnBackend,
        )

        self.bases = list(bases)
        self.reqs = list(reqs)
        self.bs = len(self.bases)
        self.num_groups = num_groups
        self.pool = _ring_pool(num_groups, num_request_slots=max(self.reqs) + 1)
        self.req_to_token = torch.arange(
            (max(self.reqs) + 1) * self.CTX, dtype=torch.int32
        ).reshape(max(self.reqs) + 1, self.CTX)
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.device = "cpu"
        backend.token_to_kv_pool = self.pool
        backend.req_to_token = self.req_to_token
        backend.req_to_token_pool = SimpleNamespace(req_to_token=self.req_to_token)
        backend.runner = None
        backend.compress_ratio = RATIO
        backend.max_context_len = self.CTX
        backend.qsa_profile = None
        backend.qsa_stall_diagnostics = None
        backend._cuda_graph_metadata = {}
        backend._graph_seq_lens = None
        try:
            backend.init_cuda_graph_state(self.MAX_BS, self.MAX_BS * max(WIDTHS))
        except RuntimeError:  # no pinned-memory allocator on a CPU-only host
            backend._graph_extend_lens = torch.zeros(
                self.MAX_BS, dtype=torch.int32, device="cpu"
            )
            backend._graph_extend_lens_pin = [
                torch.zeros(self.MAX_BS, dtype=torch.int32) for _ in range(2)
            ]
            backend._extend_lens_pin_idx = 0
        self.backend = backend

    def forward_mode(self, target_verify=True):
        return TARGET_VERIFY

    def spec_info(self, width):
        return SimpleNamespace(topk=1, draft_token_num=width)

    def req_pool_indices(self):
        return torch.tensor(self.reqs[: self.bs], dtype=torch.int32, device="cpu")

    def seq_lens(self):
        return torch.tensor(self.bases[: self.bs], dtype=torch.int32, device="cpu")

    def capture(self, width, num_padding=0):
        self.backend._capture_cuda_graph_metadata(
            bs=self.bs,
            num_tokens=self.bs * width,
            req_pool_indices=self.req_pool_indices(),
            seq_lens=self.seq_lens(),
            forward_mode=self.forward_mode(),
            spec_info=self.spec_info(width),
        )
        return self.backend._cuda_graph_metadata[(self.forward_mode(), self.bs)]

    def capture_and_replay(self, width, num_padding=0):
        """Capture the bucket, then replay the recorded metadata kernels.

        Capture fills the row buffers with the capture-time (all-padding) layout;
        the replay is what the serving path records into the graph, so every
        expectation in these tests is checked against the replayed state.
        """
        metadata = self.capture(width)
        return self.replay_on_device(metadata, width, num_padding)

    def replay_on_device(self, metadata, width, num_padding=0):
        self.backend._replay_cuda_graph_metadata_gpu(
            metadata,
            bs=self.bs,
            req_pool_indices=self.req_pool_indices(),
            seq_lens=self.seq_lens(),
            forward_mode=self.forward_mode(),
            spec_info=self.spec_info(width),
            seq_lens_cpu=self.bases[: self.bs],
            num_padding=num_padding,
        )
        return metadata


class TestGraphMetadataMatchesEager(unittest.TestCase):
    """In-graph row metadata and the eager builders must be the same arithmetic.

    The capture/replay kernels rebuild the pending-ring slots on GPU from
    lengths plus ``req_to_token``; the eager path rebuilds them from the batch.
    If either side grew its own idea of the group count, a wide window silently
    reads another group's keys -- and on a real replay nothing would complain.
    The kernels are executed here through Triton's CPU interpreter, so this is
    the recorded kernel's own output compared with the eager builders, not a
    transcription of the formula.
    """

    @staticmethod
    def _enter_interpret():
        try:
            import triton
        except ImportError:  # CPU-only CI image without triton
            return None

        knobs = getattr(getattr(triton, "knobs", None), "runtime", None)
        if knobs is None or not hasattr(knobs, "interpret"):
            return None
        previous = knobs.interpret
        knobs.interpret = True
        return (knobs, previous)

    def tearDown(self):
        if getattr(self, "_interpret", None) is not None:
            knobs, previous = self._interpret
            knobs.interpret = previous
            self._interpret = None

    def _expect(self, num_groups, lengths, row_reqs):
        positions = lengths.long() - 1
        want_slots = build_pending_ring_slots(
            token_to_batch_idx=torch.arange(lengths.numel()),
            req_pool_indices=row_reqs,
            sequence_lengths=lengths,
            logical_positions=positions,
            compress_ratio=RATIO,
            is_extend=False,
            num_groups=num_groups,
        )
        want_groups = build_group_ring_slots(
            req_pool_indices=row_reqs,
            group_end_positions=positions,
            sequence_ids=torch.arange(lengths.numel()),
            compress_ratio=RATIO,
            num_groups=num_groups,
        ).to(torch.int32)
        return want_slots, want_groups

    def test_recorded_kernels_match_the_eager_builders(self):
        self._interpret = self._enter_interpret()
        if self._interpret is None:
            self.skipTest("this triton build cannot run kernels on the CPU")
        for width in WIDTHS:
            num_groups = groups_for(width)
            for num_padding in (0, 1):
                with self.subTest(width=width, num_padding=num_padding):
                    harness = _GraphMetadataHarness(num_groups)
                    metadata = harness.capture_and_replay(width, num_padding)
                    lengths, row_reqs = _verify_layout(
                        harness.bases, harness.reqs, width, num_padding
                    )
                    indexer_metadata = metadata.indexer_metadata
                    self.assertTrue(
                        torch.equal(indexer_metadata.sequence_lengths, lengths),
                        "the recorded layout kernel disagree with the verify row lengths",
                    )
                    self.assertTrue(
                        torch.equal(
                            metadata.row_req_pool_indices.long(),
                            row_reqs.long(),
                        ),
                        "padding rows must alias the never-allocated request 0",
                    )
                    want_slots, want_groups = self._expect(
                        num_groups, lengths, row_reqs
                    )
                    self.assertTrue(
                        torch.equal(indexer_metadata.pending_ring_slots, want_slots),
                        f"W{width} in-graph state slots != eager slots",
                    )
                    self.assertTrue(
                        torch.equal(
                            indexer_metadata.graph_ring_group_locs, want_groups
                        ),
                        f"W{width} in-graph group member slots != eager slots",
                    )

    def test_host_replay_refresh_matches_the_kernels(self):
        """The non-CUDA fallback refresh writes what the recorded kernels write.

        Both graph paths must agree with each other and with the eager builders
        at the pool's capacity, or a device change silently changes addressing.
        """
        self._interpret = self._enter_interpret()
        if self._interpret is None:
            self.skipTest("this triton build cannot run kernels on the CPU")
        for width in WIDTHS:
            num_groups = groups_for(width)
            with self.subTest(width=width):
                harness = _GraphMetadataHarness(num_groups)
                metadata = harness.capture_and_replay(width)
                kernel_slots = metadata.indexer_metadata.pending_ring_slots.clone()
                kernel_groups = metadata.indexer_metadata.graph_ring_group_locs.clone()
                harness.backend._update_qsa_cuda_graph_metadata(
                    metadata.indexer_metadata, metadata.row_req_pool_indices
                )
                self.assertTrue(
                    torch.equal(
                        kernel_slots, metadata.indexer_metadata.pending_ring_slots
                    )
                )
                self.assertTrue(
                    torch.equal(
                        kernel_groups, metadata.indexer_metadata.graph_ring_group_locs
                    )
                )
                lengths, row_reqs = _verify_layout(harness.bases, harness.reqs, width)
                want_slots, want_groups = self._expect(num_groups, lengths, row_reqs)
                self.assertTrue(
                    torch.equal(
                        metadata.indexer_metadata.pending_ring_slots, want_slots
                    )
                )
                self.assertTrue(
                    torch.equal(
                        metadata.indexer_metadata.graph_ring_group_locs, want_groups
                    )
                )


class TestWidthTransitionUsesStableStorage(unittest.TestCase):
    """Active 16 -> 4 -> 16 must not move the geometry the graphs point at.

    The launch allocates the ring and every graph buffer once; an adaptive-MTP
    step-down narrows the verify window without touching them. Re-capturing and
    replaying at each width has to reuse the same storage (the captured graph
    holds those addresses) and still produce slots derived from the launch
    capacity, and a window the ring was never sized for must be refused rather
    than silently aliased.
    """

    def _pointers(self, harness):
        backend = harness.backend
        return [
            tensor.untyped_storage().data_ptr()
            for tensor in (
                backend._graph_state_slots,
                backend._graph_ring_group_locs,
                backend._graph_logical_positions,
                backend._graph_write_locs,
                backend._graph_seq_lens,
                backend._graph_row_req_pool_indices,
                harness.pool.get_qsa_key_state_buffer(0),
                harness.pool.qsa_rope_position_buffer,
            )
        ]

    def test_active_width_transitions_reuse_the_launch_allocation(self):
        num_groups = groups_for(16)
        harness = _GraphMetadataHarness(num_groups)
        # init_cuda_graph_state sized the ring-group rows off the ratio, never
        # the widest window, so a W16 launch and a W4 launch share the shape.
        self.assertEqual(harness.backend._graph_ring_group_locs.shape[1], RATIO)
        before = self._pointers(harness)
        shapes = [
            harness.backend._graph_state_slots.shape,
            harness.backend._graph_ring_group_locs.shape,
            harness.pool.get_qsa_key_state_buffer(0).shape,
        ]
        for width in (16, 4, 16, 8, 16):
            with self.subTest(width=width):
                metadata = harness.capture_and_replay(width)
                self.assertEqual(self._pointers(harness), before)
                self.assertEqual(
                    [
                        harness.backend._graph_state_slots.shape,
                        harness.backend._graph_ring_group_locs.shape,
                        harness.pool.get_qsa_key_state_buffer(0).shape,
                    ],
                    shapes,
                )
                lengths, row_reqs = _verify_layout(harness.bases, harness.reqs, width)
                positions = lengths.long() - 1
                want = pending_ring_slot(
                    row_reqs.long(),
                    positions,
                    compress_ratio=RATIO,
                    num_groups=num_groups,
                )
                got = metadata.indexer_metadata.pending_ring_slots[: lengths.numel()]
                self.assertTrue(
                    torch.equal(got, want),
                    f"active W{width} slots were not derived from the W16 capacity",
                )

    def test_capture_refuses_a_window_wider_than_the_ring(self):
        for width, groups in ((8, groups_for(4)), (16, groups_for(8))):
            with self.subTest(width=width, groups=groups):
                harness = _GraphMetadataHarness(groups)
                with self.assertRaises(NotImplementedError):
                    harness.capture(width)

    def test_padding_rows_only_ever_touch_the_inert_request_span(self):
        """DP-padded rows alias request slot 0, which no request owns.

        Their stores have to land inside slot 0's ``ratio * num_groups`` rows,
        and no allocated request may be addressed there, at any launch capacity
        -- that is what makes the dummy rows' writes inert.
        """
        for width in WIDTHS:
            num_groups = groups_for(width)
            harness = _GraphMetadataHarness(num_groups)
            metadata = harness.capture_and_replay(width, num_padding=1)
            lengths = metadata.indexer_metadata.sequence_lengths
            row_reqs = metadata.row_req_pool_indices.long()
            inert = RATIO * num_groups
            padding_rows = (lengths == 1) & (row_reqs == 0)
            self.assertTrue(bool(padding_rows.any()))
            slots = metadata.indexer_metadata.pending_ring_slots
            groups = metadata.indexer_metadata.graph_ring_group_locs.long()
            self.assertTrue(bool((slots[padding_rows] < RATIO).all()))
            self.assertTrue(bool((groups[padding_rows] < RATIO).all()))
            live = row_reqs > 0
            self.assertTrue(bool((slots[live] >= inert).all()))
            self.assertTrue(bool((groups[live] >= inert).all()))


# ---------------------------------------------------------------------------
# sgl-project/sglang#46125 (Penny issue 123) as correctness regressions, on
# Penny's OWN ring geometry.
#
# Upstream replaced its flat ``req * ratio + pos % ratio`` ring with a flat
# ``req * ring + pos % ring`` (ring = pow2 >= ratio + rows - 1). Penny took
# PR 40143's group-partitioned form instead -- position ``p`` lands in ring
# group ``(p // ratio) % groups`` at offset ``p % ratio`` -- so what transfers
# from #46125 is the bound, not the layout: the allocation has to keep the
# published window and the retained prefix members that window reads distinct,
# and it has no exemption for a narrow window, because the rows that do the
# overwriting belong to the window itself. Everything below therefore asks the
# sizing rule for the capacity (``pending_ring_num_groups`` of the launch
# width, never a hand-picked group count) and replays the production write plan
# and ``QSAIndexer``, not a transcription of the formula.
# ---------------------------------------------------------------------------

# A committed length in the middle of a request's history, group-aligned.
PREFIX_START = 4 * 16
REQ = 1

# Every verify width a shipped Penny launch line can route to, including the
# optional Flash-Next DFlash2 widths and a W16 adaptive-MTP candidate.
SHIPPED_WIDTHS = (4, 6, 8, 16)


def _launch_groups(width):
    """The capacity the pool derives for a launch whose maximum is ``width``."""
    return QSATokenToKVPool.pending_ring_num_groups(
        max_num_draft_tokens=width, compress_ratio=RATIO
    )


def _ring_replay(pool, num_groups, *, start, width, follow=None):
    """Publish a committed prefix, then one ``width``-row speculative forward.

    Uses the backend's own speculative-paged write plan and
    ``QSAIndexer.update_key_state_and_compress``, so the compressed slot, the
    published slots and the group's member slots are all production output.
    Every key and every RoPE coordinate equals its token's position, so a
    compressed group names the tokens it averaged and a ring row names the
    token that last wrote it.

    ``follow`` replays what a partial acceptance does: the next verify window
    starts there and re-publishes the accepted positions, but the prefix tokens
    are not re-sent, so a group whose members were destroyed can never come out
    right again -- and a correct ring recompresses it to the same key.
    """
    indexer = _cpu_indexer()
    q0 = (start // RATIO) * RATIO

    def publish(positions):
        meta = _paged_metadata(pool, [position + 1 for position in positions], REQ)
        indexer.update_key_state_and_compress(
            _keys(positions),
            torch.tensor(positions),
            torch.tensor(positions),
            meta,
        )
        return meta

    def group_mean(meta):
        # The plan is compacted: entry i is the i-th block this forward
        # completes, and ``compress_group_positions`` names the block by its
        # last position. Padding entries name group 0 (end position ratio-1),
        # never this one, so a match here is a block this forward really wrote.
        ends = meta.compress_group_positions.long().flatten()
        owned = (ends == q0 + RATIO - 1).nonzero().flatten()
        if owned.numel() == 0:
            return None  # this window completes no group
        return float(
            pool.get_qsa_compressed_k_buffer(0)[int(meta.write_locs[int(owned[0])])]
            .float()
            .mean()
        )

    # The previous whole group plus the retained partial group: exactly the
    # state a serving run holds when the verify forward starts.
    publish(list(range(max(0, q0 - RATIO), start)))
    meta = publish(list(range(start, start + width)))
    out = {
        "correct": q0 + (RATIO - 1) / 2,
        "group_mean": group_mean(meta),
    }
    if follow is not None:
        out["follow_mean"] = group_mean(publish(list(range(follow, follow + width))))
    state = pool.get_qsa_key_state_buffer(0)
    retained_slots = [_slot_of(REQ, q, num_groups) for q in range(q0, start)]
    out.update(
        {
            "retained": [float(state[slot].float().mean()) for slot in retained_slots],
            # Compression rotates the group at the RoPE coordinate stored in the
            # FIRST member's slot, so an overwrite is wrong in its value and in
            # its rotation at once -- whichever later row landed there.
            "retained_rope": [
                int(pool.qsa_rope_position_buffer[slot][0]) for slot in retained_slots
            ],
        }
    )
    return out


class TestVerifyWindowRingLifetime(unittest.TestCase):
    """Every routed width keeps its own live span apart, at the sized capacity.

    A speculative-paged forward publishes every row of its window before
    compressing the group that window completes. At an unaligned committed
    length that group still reads members retained from the prefix, and on a
    one-group ring the window's own later rows have already wrapped onto those
    slots (``(p + ratio) % ratio == p % ratio``): the stored compressed key
    averages this forward's tokens and is rotated at the coordinate written into
    the first member's slot. #46125 measured that on the served
    Qwen3.8-Flash-Next model; the sizing rule now refuses it at launch, so no
    shipped width can reach it.
    """

    def _run(self, width, *, start, follow=None):
        num_groups = _launch_groups(width)
        return _ring_replay(
            _ring_pool(num_groups, num_request_slots=4),
            num_groups,
            start=start,
            width=width,
            follow=follow,
        )

    def test_small_windows_keep_their_retained_members_distinct(self):
        for width in (2, 3, 4):
            for residue in (0, 1, 2, 3):
                with self.subTest(width=width, residue=residue):
                    got = self._run(width, start=PREFIX_START + residue)
                    intact = [float(PREFIX_START + index) for index in range(residue)]
                    self.assertEqual(got["retained"], intact)
                    self.assertEqual(
                        got["retained_rope"],
                        [PREFIX_START + index for index in range(residue)],
                    )
                    # A window that starts too late in the group to complete it
                    # writes no compressed key at all; anything it does write is
                    # the prefill-equivalent average.
                    self.assertEqual(
                        got["group_mean"],
                        (
                            got["correct"]
                            if residue + width > RATIO
                            else (got["correct"] if residue + width == RATIO else None)
                        ),
                    )

    def test_partial_acceptance_recompresses_to_the_same_key(self):
        """Rejected rows are not inert, and a correct ring is idempotent.

        W4 at residue 1 with one accepted draft token: the next verify window
        re-publishes the accepted positions and re-compresses the same group.
        Sized capacity (2 groups, and the maintained W8 launch's 3 for its C2+
        W4 tier) returns the prefill-equivalent key both rounds.
        """
        start = PREFIX_START + 1
        for width in (4, 8):
            with self.subTest(launch_max=width):
                got = self._run(width, start=start, follow=start + 1)
                self.assertEqual(got["group_mean"], got["correct"])
                self.assertEqual(got["follow_mean"], got["correct"])

    def test_one_row_window_keeps_the_single_group_layout(self):
        """The layout non-speculative serving and plain decode run on: unchanged.

        A decode forward publishes its one row and reads the rest of the group,
        and its row IS the group's newest member, so nothing aliases. This is
        the whole reason the corrected rule still answers 1 below W2 -- the
        per-request ring of a launch without speculation does not move.
        """
        self.assertEqual(_launch_groups(1), 1)
        self.assertEqual(_launch_groups(0), 1)
        self.assertEqual(_launch_groups(None), 1)
        for residue in (1, 2, 3):
            with self.subTest(residue=residue):
                got = _ring_replay(
                    _ring_pool(1, num_request_slots=4),
                    1,
                    start=PREFIX_START + residue,
                    width=1,
                )
                self.assertEqual(
                    got["retained"],
                    [float(PREFIX_START + index) for index in range(residue)],
                )
                self.assertEqual(
                    got["group_mean"],
                    got["correct"] if residue == RATIO - 1 else None,
                )


class TestPreCorrectionSingleGroupCapacity(unittest.TestCase):
    """The capacity Penny shipped for W4, pinned as the failure it now refuses.

    One group at ratio 4 was ``pending_ring_groups_required``'s small-width
    exemption -- the fixed-W4 launch (steps 3, 4 draft tokens) sized its ring to
    the window and nothing else. The replay is unchanged, the sizing is not, so
    the overwrite stays reproducible here as the counter-example even though no
    shipped launch can allocate that ring any more (the guard refuses it, see
    ``TestGraphMetadataCarriesTheSizedSpan``).
    """

    def _run(self, *, start, width=4, follow=None):
        return _ring_replay(
            _ring_pool(1, num_request_slots=4),
            1,
            start=start,
            width=width,
            follow=follow,
        )

    def test_w4_overwrote_every_retained_member(self):
        for residue in (1, 2, 3):
            with self.subTest(residue=residue):
                got = self._run(start=PREFIX_START + residue)
                # The group averaged the whole window instead of the retained
                # members plus the window's first rows.
                self.assertEqual(got["group_mean"], got["correct"] + residue)
                # Every retained member's slot -- including the first member's,
                # whose stored coordinate compression rotates the group at --
                # now holds the token ratio positions later.
                self.assertEqual(
                    got["retained"],
                    [float(PREFIX_START + RATIO + index) for index in range(residue)],
                )
                self.assertEqual(
                    got["retained_rope"],
                    [PREFIX_START + RATIO + index for index in range(residue)],
                )

    def test_narrow_windows_overwrote_too(self):
        """#46125: every 2..ratio-row window, not only a window == ratio.

        The outcome turns on how many rows follow the boundary row, i.e. on
        ``residue + width`` against the ratio: past it the group is wrong, at it
        the boundary row is last and the group is exact, short of it nothing
        compresses. A W4 window is always past it.
        """
        for width in (2, 3, 4):
            for residue in (1, 2, 3):
                with self.subTest(width=width, residue=residue):
                    got = self._run(start=PREFIX_START + residue, width=width)
                    intact = [float(PREFIX_START + index) for index in range(residue)]
                    if residue + width > RATIO:
                        self.assertGreater(got["group_mean"], got["correct"])
                        self.assertEqual(
                            got["retained"][0], float(PREFIX_START + RATIO)
                        )
                        self.assertEqual(got["retained_rope"][0], PREFIX_START + RATIO)
                    elif residue + width == RATIO:
                        self.assertEqual(got["group_mean"], got["correct"])
                        self.assertEqual(got["retained"], intact)
                    else:
                        self.assertIsNone(got["group_mean"])
                        self.assertEqual(got["retained"], intact)

    def test_partial_acceptance_did_not_undo_the_overwrite(self):
        start = PREFIX_START + 1
        got = self._run(start=start, follow=start + 1)
        self.assertEqual(got["group_mean"], got["correct"] + 1)
        # The destroyed members are never re-sent, so the recompression drifts
        # further instead of converging.
        self.assertEqual(got["follow_mean"], got["correct"] + 2)


class TestGraphMetadataCarriesTheSizedSpan(unittest.TestCase):
    """The recorded in-graph kernel, replayed at each width's sized capacity.

    Verify and draft-extend forwards read their group members from
    ``graph_ring_group_locs``, written by ``_qsa_graph_row_metadata_kernel``
    through Triton's CPU interpreter here. At the capacity the sizing rule
    derives, no member slot of a boundary row can collide with the slots of the
    rows published after it in the same window, for every width a Penny recipe
    can launch; the pre-fix single-group W4 ring does not get that far, because
    capture asks the guard first.
    """

    # One request per unaligned residue.
    BASES = (PREFIX_START + 1, PREFIX_START + 2, PREFIX_START + 3)
    REQS = (1, 2, 3)

    def _collisions(self, num_groups, width):
        import triton

        knobs = triton.knobs.runtime
        previous, knobs.interpret = knobs.interpret, True
        try:
            harness = _GraphMetadataHarness(
                num_groups, bases=self.BASES, reqs=self.REQS
            )
            metadata = harness.capture_and_replay(width)
        finally:
            knobs.interpret = previous
        indexer_metadata = metadata.indexer_metadata
        rows = torch.arange(indexer_metadata.sequence_lengths.numel())
        boundary = (indexer_metadata.sequence_lengths.long() % RATIO) == 0
        self.assertTrue(bool(boundary.any()), "no row completed a group")
        stores = indexer_metadata.pending_ring_slots.long()
        members = indexer_metadata.graph_ring_group_locs.long()
        row_reqs = metadata.row_req_pool_indices.long()
        # Every slot a boundary row's group reads that a LATER row of the same
        # window published into: a live position sharing one slot.
        aliased = set()
        for row in rows[boundary]:
            later = rows[(row_reqs == row_reqs[row]) & (rows > row)]
            aliased.update(
                int(slot)
                for slot in (set(members[row].tolist()) & set(stores[later].tolist()))
            )
        return sorted(aliased)

    def test_sized_capacity_keeps_member_slots_out_of_later_rows(self):
        for width in SHIPPED_WIDTHS:
            with self.subTest(width=width):
                self.assertEqual(self._collisions(_launch_groups(width), width), [])

    def test_capture_refuses_the_pre_correction_single_group_w4_ring(self):
        """The old capacity is now a start-up refusal, not a silent overwrite.

        Deliberately replaying a W4 window on a one-group ring cannot even
        record its metadata: ``_require_chain_speculation`` runs after the pool
        is resolved and before any kernel, so the undersized ring is reported at
        capture instead of mixing keys mid-serve.
        """
        harness = _GraphMetadataHarness(1, bases=self.BASES, reqs=self.REQS)
        with self.assertRaises(NotImplementedError):
            harness.capture(4)


POLICY_DIR = Path(__file__).resolve().parents[5] / "configs" / "pennyroyal"

# policy -> (bundled --speculative-adaptive-config, the draft-token count the
# recipe that ships it launches, the ring groups that capacity must size).
# configs/pennyroyal/mtp-policy.sh pairs each file with its
# --speculative-num-steps / --speculative-num-draft-tokens.
MAINTAINED_POLICIES = {
    "tiered (default)": ("adaptive-next.json", 8, 3),
    "adaptive": ("adaptive-next-v3.0.2.json", 8, 3),
    "fixed-w4": ("adaptive-next-fixed-w4.json", 4, 2),
    "next-dflash2": ("adaptive-dflash2.json", 6, 3),
}


class TestMaintainedPolicyRingCapacity(unittest.TestCase):
    """Applicability by profile/policy, derived from the shipped launch lines.

    The ring group count comes from ``max_speculative_num_draft_tokens()``, i.e.
    the widest routed candidate (adaptive route: ``max(candidate_steps) + 1``,
    which is what every maintained recipe launches), so the policy files decide
    it. After the correction no maintained launch capacity is below the live
    span of its own widest window -- and the ``27b`` profile is not in this
    table for a stronger reason: it is not a compressed-QSA model, so it
    allocates no pending ring at all.
    """

    def _launch_draft_tokens(self, config_path):
        from sglang.srt.speculative.adaptive_spec_params import (
            resolve_candidate_steps_from_config,
        )

        return max(resolve_candidate_steps_from_config(str(config_path))) + 1

    def test_every_maintained_capacity_covers_its_own_live_span(self):
        for policy, (name, launched, expected_groups) in MAINTAINED_POLICIES.items():
            config = POLICY_DIR / name
            with self.subTest(policy=policy):
                # The recipe and the policy file agree, so the capacity below is
                # the one the profile really boots with.
                self.assertEqual(self._launch_draft_tokens(config), launched)
                groups = _launch_groups(launched)
                self.assertEqual(groups, expected_groups, policy)
                # Rows the launch allocates per request, and the span the
                # widest routed window plus its retained prefix tail occupies.
                pool = _ring_pool(groups, num_request_slots=3)
                self.assertEqual(
                    pool.get_qsa_key_state_buffer(0).shape[0], 3 * RATIO * groups
                )
                self.assertGreaterEqual(RATIO * groups, launched + RATIO - 1)

    def test_pools_without_a_compressed_ring_cannot_be_exposed(self):
        """27B / tokenwise QSA isolation: no ring, so nothing to collide.

        The pool class is chosen from the parsed QSA profile, and the tokenwise
        variant (and a model with no QSA profile, which is what the ``27b``
        recipe launches: plain FlashInfer prefill + TRTLLM-MHA decode, and no
        ``qsa_compressed_hicache`` namespace field) carries a flat per-token
        index-K cache with no compression groups and no per-request ring. The
        guard's answer for those pools is ``1`` group, which is why it refuses
        every multi-row window there rather than skipping the check.
        """
        from sglang.srt.layers.attention.qsa.config import parse_qsa_profile
        from sglang.srt.mem_cache.qsa_kv_pool import QwenDSATokenToKVPool

        tokenwise = SimpleNamespace(
            model_type="qwen3_next",
            index_topk=2048,
            index_n_heads=16,
            index_kv_heads=1,
            index_head_dim=128,
        )
        self.assertEqual(parse_qsa_profile(tokenwise).variant, "tokenwise")
        self.assertEqual(parse_qsa_profile(tokenwise).compress_ratio, 1)
        self.assertIsNone(parse_qsa_profile(SimpleNamespace(model_type="qwen3_5")))
        pool = QwenDSATokenToKVPool(
            size=256,
            dtype=torch.bfloat16,
            page_size=64,
            head_num=2,
            head_dim=64,
            full_attention_layer_ids=[0],
            device="cpu",
            mamba_pool=None,
            qsa_index_kv_heads=1,
            qsa_index_head_dim=8,
            qsa_token_budget=2048,
        )
        self.assertFalse(hasattr(pool, "qsa_num_groups"))
        self.assertFalse(hasattr(pool, "qsa_ring_span"))
        self.assertEqual(pending_ring_groups(pool), 1)


if __name__ == "__main__":
    unittest.main()
