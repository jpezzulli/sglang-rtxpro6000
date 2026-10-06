"""Slot arithmetic for the QSA pending index-K ring under N compression groups.

Ported from sgl-project/sglang PR 40143 ("[Spec] QSA: let the pending
index-K ring hold more than one compression group", aiueo52, 2026), with
Pennyroyal additions pinning the pool-side group derivation and the
speculative-window guard. The ring holds ``num_groups`` groups of
``compress_ratio`` slots per request, so a request owns
``compress_ratio * num_groups`` slots and a verify window up to that width
maps every position to a distinct slot. ``num_groups == 1`` is the
historical single-group layout, and these tests pin the new arithmetic to
it bit-for-bit: the shipped W4/default path must not move.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
    pending_ring_slot,
)
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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
        positions. The shipped single-group window (draft <= ratio, W4)
        keeps its qualified layout instead of rounding up. An absent
        draft-token count (non-speculative run) must not produce an empty
        ring.
        """
        cases = [
            (None, 1),
            (0, 1),
            (1, 1),
            (4, 1),
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
    """The verify-window guard reads the ring's capacity, not the bare ratio."""

    class _Mode:
        def is_target_verify(self):
            return True

    def _check(self, draft_tokens, num_groups):
        from types import SimpleNamespace

        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            QwenSparseAttnBackend,
        )

        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.compress_ratio = RATIO
        backend.token_to_kv_pool = SimpleNamespace(qsa_num_groups=num_groups)
        spec_info = SimpleNamespace(topk=1, draft_token_num=draft_tokens)
        backend._require_chain_speculation(
            TestRequireChainSpeculation._Mode(), spec_info
        )

    def test_w4_default_accepted_at_one_group(self):
        self._check(4, 1)

    def test_w8_rejected_at_one_group(self):
        with self.assertRaises(NotImplementedError):
            self._check(8, 1)

    def test_w8_accepted_at_two_groups(self):
        self._check(8, 2)

    def test_beyond_capacity_rejected(self):
        with self.assertRaises(NotImplementedError):
            self._check(9, 2)


if __name__ == "__main__":
    unittest.main()
