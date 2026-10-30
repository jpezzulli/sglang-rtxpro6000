"""CPU contract for the QSA pending index-K ring under multi-group rings.

The pending ring addresses one compression group per request
(``req * ratio + position % ratio``). A speculative-paged forward (target
verify, and draft-extend windows wider than one group) publishes EVERY row
of its window into the ring before compressing the groups that window
completes, and the first completed group may still read retained prefix
members from the ring. From a two-row window at an unaligned committed
length onward, the window's own later rows land on exactly the slots the
straddling group reads (``(p + ratio) % ratio == p % ratio``), so a
single-group ring silently averages the wrong members. Only the one-row
window (plain decode) is exact at one group, which is why the single-group
layout must stay byte-for-byte the default.

These tests pin, on CPU and without a server: the slot arithmetic for N
groups (with the N==1 legacy oracle), the launch sizing bound
``(draft - 1) // ratio + 2`` with no small-width exemption, the pool's
capacity, the verify-window guard, the host-side graph refresh, and the
in-graph metadata kernels deriving every producer from the pool's one
capacity. The end-to-end publish/compress lifetime runs the real
``QSAIndexer.update_key_state_and_compress`` against the real pool.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qsa import graph_metadata
from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
)
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

RATIO = 4
REQ = 1  # request slot 0 is never allocated (its rows are the inert dump)


def _pool(num_groups=None, *, num_request_slots=4, index_head_dim=RATIO):
    kwargs = {} if num_groups is None else {"qsa_num_groups": num_groups}
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
        qsa_index_head_dim=index_head_dim,
        qsa_compress_ratio=RATIO,
        qsa_token_topk=8,
        num_request_slots=num_request_slots,
        **kwargs,
    )


def _cpu_indexer(ratio=RATIO):
    indexer = QSAIndexer.__new__(QSAIndexer)
    indexer.layer_id = 0
    indexer.compress_ratio = ratio
    indexer.index_kv_heads = 1
    indexer.index_head_dim = ratio
    indexer.rotary_emb = SimpleNamespace(mrope_section=None)
    # Stay off the triton fused kernels while still running the real
    # publish -> gather-from-ring -> mean path. The fused-store equivalence
    # is a GPU-window check for thegrid; this pins addressing and ordering.
    indexer._use_fused_compress = lambda pool: False
    indexer.normalize_compressed_keys = lambda keys, positions: keys
    return indexer


def _keys(positions, dim=RATIO):
    """One constant-valued index key per position, so a clobber is readable."""
    return (
        torch.tensor(positions, dtype=torch.float32)
        .view(-1, 1, 1)
        .expand(-1, 1, dim)
        .contiguous()
    )


def _paged_metadata(pool, lengths):
    """Indexer metadata for one speculative-paged forward.

    Row i has committed length ``lengths[i]`` (one row per query/token of the
    window, all rows belonging to request ``REQ``); the write plan lists the
    groups the window completes (each aligned end exactly once) with the ring
    as the member source (``compress_member_rows=None``).
    """
    lengths_t = torch.tensor(lengths, dtype=torch.int32)
    # A group completes at position length - 1 when the length is aligned.
    group_ends = [length - 1 for length in lengths if length % RATIO == 0]
    # One synthetic distinct compressed slot per planned group.
    write_locs = torch.tensor(
        [1 + i for i in range(len(group_ends))], dtype=torch.int64
    )
    # Row whose length completes each group.
    seq_ids = torch.tensor(
        [lengths.index(end + 1) for end in group_ends], dtype=torch.long
    )
    from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata

    return QSAIndexerMetadata(
        sequence_lengths=lengths_t,
        token_to_batch_idx=torch.arange(lengths_t.numel(), dtype=torch.long),
        token_slot_table=torch.zeros((lengths_t.numel(), 8), dtype=torch.int32),
        out_cache_loc=torch.arange(lengths_t.numel(), dtype=torch.int64),
        token_to_kv_pool=pool,
        compress_ratio=RATIO,
        block_topk=2,
        req_pool_indices=torch.full((lengths_t.numel(),), REQ, dtype=torch.long),
        write_locs=write_locs,
        compress_group_positions=torch.tensor(group_ends, dtype=torch.long),
        compress_sequence_ids=seq_ids,
        compress_member_rows=None,
        is_cuda_graph=False,
    )


def _publish(pool, indexer, positions):
    meta = _paged_metadata(pool, [position + 1 for position in positions])
    indexer.update_key_state_and_compress(
        _keys(positions),
        torch.tensor(positions),
        torch.zeros(3, len(positions), dtype=torch.int64),
        meta,
    )
    return meta


class TestPendingRingSlotArithmetic(unittest.TestCase):
    """The slot formula parameterized by the ring's group count."""

    def _slots(self, positions, num_groups):
        positions = torch.tensor(positions, dtype=torch.long)
        return build_pending_ring_slots(
            token_to_batch_idx=torch.arange(positions.numel()),
            req_pool_indices=torch.full((positions.numel(),), REQ, dtype=torch.long),
            sequence_lengths=torch.full((positions.numel(),), 8, dtype=torch.long),
            logical_positions=positions,
            compress_ratio=RATIO,
            is_extend=False,
            num_groups=num_groups,
        )

    def test_single_group_is_the_legacy_layout(self):
        positions = list(range(0, 48))
        want = torch.tensor([REQ * RATIO + p % RATIO for p in positions])
        self.assertTrue(
            torch.equal(self._slots(positions, 1), want),
            "num_groups=1 moved the layout non-speculative serving runs on",
        )

    def test_window_plus_prefix_is_collision_free_at_capacity(self):
        # W=8 live span at an unaligned start: retained prefix of 3 plus 8 rows.
        positions = list(range(5, 16))
        slots = self._slots(positions, 3)
        self.assertEqual(slots.unique().numel(), len(positions))

    def test_single_group_aliases_the_live_span(self):
        positions = list(range(5, 13))
        slots = self._slots(positions, 1)
        self.assertLess(slots.unique().numel(), len(positions))

    def test_extend_dump_stays_inert_at_any_group_count(self):
        positions = torch.tensor([0, 3, 8, 9], dtype=torch.long)
        for num_groups in (1, 2, 4):
            slots = build_pending_ring_slots(
                token_to_batch_idx=torch.arange(4),
                req_pool_indices=torch.full((4,), REQ, dtype=torch.long),
                sequence_lengths=torch.full((4,), 8, dtype=torch.long),
                logical_positions=positions,
                compress_ratio=RATIO,
                is_extend=True,
                num_groups=num_groups,
            )
            self.assertTrue(bool((slots[:2] < RATIO).all()))  # dumped rows
            self.assertTrue(
                bool((slots[2:] >= RATIO * num_groups).all()),  # pending rows
                "the inert dump region overlapped an allocated request",
            )


class TestGroupRingSlots(unittest.TestCase):
    def _build(self, group_ends, reqs, num_groups):
        return build_group_ring_slots(
            req_pool_indices=torch.tensor(reqs, dtype=torch.long),
            group_end_positions=torch.tensor(group_ends, dtype=torch.long),
            sequence_ids=torch.arange(len(group_ends)),
            compress_ratio=RATIO,
            num_groups=num_groups,
        )

    def test_members_oldest_first_single_group_matches_legacy(self):
        slots = self._build([7], [REQ], num_groups=1)
        want = torch.tensor(
            [REQ * RATIO + p % RATIO for p in range(4, 8)], dtype=torch.int64
        )
        self.assertTrue(torch.equal(slots[0], want))

    def test_all_members_share_the_group_of_their_end(self):
        # Unaligned ends (graph producer feeds lengths - 1 on every row): the
        # group index comes from the end so a group never straddles ring
        # groups at N > 1.
        for num_groups in (2, 4):
            capacity = RATIO * num_groups
            slots = self._build([16, 17], [REQ, REQ], num_groups)
            for row, end in enumerate((16, 17)):
                group = (end // RATIO) % num_groups
                lo = REQ * capacity + group * RATIO
                self.assertTrue(
                    bool(((slots[row] >= lo) & (slots[row] < lo + RATIO)).all()),
                    f"group ending at {end} straddled ring groups: {slots[row]}",
                )

    def test_members_clamp_at_zero(self):
        slots = self._build([1], [REQ], num_groups=1)
        want = torch.tensor(
            [REQ * RATIO + p % RATIO for p in (0, 0, 0, 1)], dtype=torch.int64
        )
        self.assertTrue(torch.equal(slots[0], want))


class TestRingSizing(unittest.TestCase):
    """The launch sizing bound has no small-width exemption."""

    def test_required_groups_at_ratio_4(self):
        cases = {
            None: 1,
            0: 1,
            1: 1,  # decode / non-speculative: the shipped layout
            2: 2,
            3: 2,
            4: 2,  # the width a retained prefix still aliases at 1 group
            5: 3,
            8: 3,
            9: 4,
            16: 5,
        }
        for draft_tokens, expected in cases.items():
            got = QSATokenToKVPool.pending_ring_num_groups(
                max_num_draft_tokens=draft_tokens, compress_ratio=4
            )
            self.assertEqual(got, expected, (draft_tokens, got))

    def test_required_groups_at_ratio_2(self):
        for draft_tokens, expected in {1: 1, 2: 2, 3: 3, 4: 3, 5: 4}.items():
            self.assertEqual(
                QSATokenToKVPool.pending_ring_num_groups(
                    max_num_draft_tokens=draft_tokens, compress_ratio=2
                ),
                expected,
                (draft_tokens,),
            )

    def test_pool_capacity_scales_with_groups(self):
        for num_groups in (1, 3):
            pool = _pool(num_groups, num_request_slots=5)
            rows = 5 * RATIO * num_groups
            self.assertEqual(pool.qsa_num_groups, num_groups)
            self.assertEqual(pool.qsa_ring_span, RATIO * num_groups)
            self.assertEqual(
                pool.get_qsa_key_state_buffer(0).shape, (rows, 1, RATIO)
            )
            self.assertEqual(pool.qsa_rope_position_buffer.shape, (rows, 3))

    def test_pending_state_transfer_page_is_the_ring_span(self):
        # The PD state transfer indexes whole request rings by req_pool_idx;
        # the per-page item length must cover one request's full ring.
        pool = _pool(3)
        _, _, item_lens = pool.get_qsa_pending_state_buf_infos()
        state_bytes = (
            pool.get_qsa_key_state_buffer(0)[0].numel()
            * pool.get_qsa_key_state_buffer(0)[0].element_size()
        )
        self.assertEqual(item_lens[0], state_bytes * RATIO * 3)

    def test_degenerate_group_counts_rejected(self):
        with self.assertRaises(ValueError):
            _pool(0)


class TestVerifyWindowGuard(unittest.TestCase):
    """The guard admits exactly the windows the pool's ring can hold."""

    class _VerifyMode:
        @staticmethod
        def is_target_verify():
            return True

    def _check(self, draft_tokens, pool):
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.compress_ratio = RATIO
        backend.token_to_kv_pool = pool
        backend._require_chain_speculation(
            TestVerifyWindowGuard._VerifyMode(), SimpleNamespace(topk=1,
            draft_token_num=draft_tokens)
        )

    def test_decode_window_always_allowed(self):
        self._check(1, _pool(1))
        self._check(1, None)  # unresolved pool: one row needs no room

    def test_w8_verify_allowed_on_a_w8_sized_ring(self):
        self._check(8, _pool(QSATokenToKVPool.pending_ring_num_groups(
            max_num_draft_tokens=8, compress_ratio=RATIO
        )))

    def test_narrow_window_refused_on_a_single_group_ring(self):
        # W4 at a retained-prefix start overwrites a live member when the
        # ring holds one group; the old ratio-bound waved this through.
        with self.assertRaises(NotImplementedError):
            self._check(4, _pool(1))

    def test_window_beyond_the_ring_refused(self):
        with self.assertRaises(NotImplementedError):
            self._check(9, _pool(3))
        with self.assertRaises(NotImplementedError):
            self._check(4, None)  # unknown capacity, not a proven one

    def test_launch_capacity_serves_every_narrower_active_width(self):
        groups = QSATokenToKVPool.pending_ring_num_groups(
            max_num_draft_tokens=16, compress_ratio=RATIO
        )
        pool = _pool(groups)
        for active in (1, 2, 4, 8, 16):
            self._check(active, pool)  # adaptive step-down needs no resize


class TestPublishedWindowRingLifetime(unittest.TestCase):
    """The production publish -> compress ordering against the real pool."""

    def _run_w4(self, num_groups=None):
        pool = _pool(num_groups)
        indexer = _cpu_indexer()
        # Forward A leaves position 0 pending (r0 = 1 retained member).
        _publish(pool, indexer, [0])
        # Forward B: a W4 verify publishes 1..4 and compresses group 0..3,
        # whose oldest member is the retained position 0.
        meta = _publish(pool, indexer, [1, 2, 3, 4])
        compressed = pool.get_qsa_compressed_k_buffer(0)
        return float(compressed[int(meta.write_locs[0])].float().mean())

    def test_single_group_default_reproduces_the_clobber(self):
        """The counter-example, pinned at the default layout.

        Position 4 (the window's last row) aliases position 0's slot before
        the straddling group is read back, so the compressed key averages
        mean(4,1,2,3) instead of mean(0,1,2,3).
        """
        self.assertEqual(self._run_w4(), 2.5)

    def test_two_group_ring_preserves_the_retained_prefix(self):
        self.assertEqual(self._run_w4(num_groups=2), 1.5)

    def test_w8_lifetime_at_launch_capacity(self):
        pool = _pool(
            QSATokenToKVPool.pending_ring_num_groups(
                max_num_draft_tokens=8, compress_ratio=RATIO
            )
        )
        indexer = _cpu_indexer()
        _publish(pool, indexer, [0])
        meta = _publish(pool, indexer, list(range(1, 9)))
        state = pool.get_qsa_key_state_buffer(0)
        compressed = pool.get_qsa_compressed_k_buffer(0)
        first_group = int(meta.write_locs[0])
        self.assertEqual(float(compressed[first_group].float().mean()), 1.5)
        self.assertEqual(float(compressed[first_group + 1].float().mean()), 5.5)
        # p0's ring row still holds p0's key after the whole window.
        slot = REQ * pool.qsa_ring_span + 0
        self.assertEqual(float(state[slot].float().mean()), 0.0)


class TestGraphProducersUsePoolCapacity(unittest.TestCase):
    """The graph metadata kernels and the host refresh take the pool's count."""

    class _Recorder:
        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return launch

    def _graph_metadata(self, pool, rows):
        def buf(dtype, *shape):
            return torch.zeros(*shape, dtype=dtype)

        # Attribute reads only; a plain namespace keeps the frozen struct out
        # of the way of the host-refresh test.
        indexer_metadata = SimpleNamespace(
            sequence_lengths=buf(torch.int32, rows),
            token_to_batch_idx=torch.arange(rows, dtype=torch.int32),
            token_slot_table=buf(torch.int32, rows, 8),
            out_cache_loc=buf(torch.int64, rows),
            token_to_kv_pool=pool,
            compress_ratio=RATIO,
            block_topk=2,
            req_pool_indices=buf(torch.int32, rows),
            is_cuda_graph=True,
            graph_write_locs=buf(torch.int32, rows),
            graph_compressed_page_table=buf(torch.int32, rows, 4),
            graph_compressed_lengths=buf(torch.int32, rows),
            graph_prefix_lengths=buf(torch.int32, rows),
            decode_logical_positions=buf(torch.int32, rows),
            pending_ring_slots=buf(torch.int64, rows),
            graph_ring_group_locs=buf(torch.int32, rows, RATIO),
        )
        return SimpleNamespace(
            indexer_metadata=indexer_metadata,
            sequence_lengths=buf(torch.int32, rows),
            row_req_pool_indices=buf(torch.int32, rows),
        )

    def test_launch_graph_metadata_passes_the_pool_group_count(self):
        pool = _pool(3)
        metadata = self._graph_metadata(pool, 8)
        recorder = TestGraphProducersUsePoolCapacity._Recorder()
        original = graph_metadata._qsa_graph_row_metadata_kernel
        graph_metadata._qsa_graph_row_metadata_kernel = recorder
        try:
            graph_metadata.launch_graph_metadata(
                mode=1,
                bs=2,
                num_rows=8,
                seq_lens=torch.zeros(2, dtype=torch.int32),
                req_pool_indices=torch.zeros(2, dtype=torch.int32),
                extend_lens=torch.full((2,), 4, dtype=torch.int32),
                extend_len=4,
                num_padding=0,
                metadata=metadata,
                req_to_token=torch.zeros(8, 256, dtype=torch.int32),
                pool=pool,
            )
        finally:
            graph_metadata._qsa_graph_row_metadata_kernel = original
        self.assertEqual(len(recorder.calls), 1)
        _, _, kwargs = recorder.calls[0]
        self.assertEqual(kwargs.get("NUM_GROUPS"), pool.qsa_num_groups)
        self.assertEqual(kwargs.get("RATIO"), RATIO)

    def test_draft_graph_metadata_args_carry_the_group_count(self):
        pool = _pool(5)
        metadata = [self._graph_metadata(pool, 4) for _ in range(2)]
        req_to_token = torch.zeros(8, 256, dtype=torch.int32)
        args = graph_metadata.prepare_draft_graph_metadata(metadata, req_to_token, pool)
        # (buffers, req_to_token, stride, max_pages, ratio, num_groups, full_page)
        self.assertEqual(args[5], pool.qsa_num_groups)

    def test_kernel_matches_eager_under_triton_interpreter(self):
        """Run the recorded row-metadata kernel where a triton driver exists.

        Skips on CPU-only hosts without a triton backend driver; thegrid's
        runners execute this and compare the kernel's own output with the
        eager builder at the launch capacity.
        """
        try:
            import triton
        except ImportError:
            self.skipTest("no triton")
        knobs = getattr(getattr(triton, "knobs", None), "runtime", None)
        if knobs is None or not hasattr(knobs, "interpret"):
            self.skipTest("triton without interpret mode")
        previous = knobs.interpret
        knobs.interpret = True
        try:
            self._run_kernel_parity(num_groups=3)
        except RuntimeError as e:  # no driver on this host
            if "active drivers" not in str(e):
                raise
            self.skipTest(f"no triton driver: {e}")
        finally:
            knobs.interpret = previous

    def _run_kernel_parity(self, num_groups):
        pool = _pool(num_groups)
        rows = 4
        seq_lens = torch.tensor([4, 6, 7, 8], dtype=torch.int32)
        reqs = torch.full((rows,), REQ, dtype=torch.int32)
        metadata = self._graph_metadata(pool, rows)
        indexer = metadata.indexer_metadata
        req_to_token = torch.arange(8 * 256, dtype=torch.int32).reshape(8, 256)
        graph_metadata._qsa_graph_row_metadata_kernel[(rows, 1)](
            metadata.sequence_lengths,
            metadata.row_req_pool_indices,
            indexer.graph_compressed_lengths,
            indexer.graph_write_locs,
            indexer.graph_compressed_page_table,
            indexer.decode_logical_positions,
            indexer.pending_ring_slots,
            indexer.graph_ring_group_locs,
            req_to_token,
            req_to_token.stride(0),
            1,
            RATIO=RATIO,
            NUM_GROUPS=num_groups,
            FULL_PAGE=pool.qsa_compressed_page_size * RATIO,
            PAGE_BLOCK=128,
            seq_lens_ptr=seq_lens,
            req_pool_ptr=reqs,
            row_prefix_lens_ptr=indexer.graph_prefix_lengths,
            bs=rows,
            num_padding=0,
            extend_len=0,
            seq_offset=0,
            MODE=0,
            num_warps=1,
        )
        want = build_pending_ring_slots(
            token_to_batch_idx=torch.arange(rows),
            req_pool_indices=reqs.long(),
            sequence_lengths=seq_lens,
            logical_positions=seq_lens.long() - 1,
            compress_ratio=RATIO,
            is_extend=False,
            num_groups=num_groups,
        )
        self.assertTrue(
            torch.equal(indexer.pending_ring_slots, want),
            f"in-graph slots diverged from eager: "
            f"{indexer.pending_ring_slots.tolist()} != {want.tolist()}",
        )
        # Row 0 (length 4, boundary): the group ending at position 3 lives in
        # ring group 0 of request REQ -> contiguous block at REQ * span.
        self.assertEqual(
            indexer.graph_ring_group_locs[0].tolist(),
            [REQ * RATIO * num_groups + i for i in range(RATIO)],
        )

    def test_host_refresh_matches_the_eager_builders(self):
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        pool = _pool(3)
        backend.device = "cpu"
        backend.req_to_token = torch.arange(8 * 256, dtype=torch.int32).reshape(8, 256)
        metadata = self._graph_metadata(pool, 4)
        lengths = torch.tensor([4, 6, 7, 8], dtype=torch.int32)
        reqs = torch.full((4,), REQ, dtype=torch.int32)
        metadata.indexer_metadata.sequence_lengths = lengths
        metadata.indexer_metadata.req_pool_indices = reqs
        backend._update_qsa_cuda_graph_metadata(metadata.indexer_metadata, reqs)
        positions = lengths.long() - 1
        want = build_pending_ring_slots(
            token_to_batch_idx=torch.arange(4),
            req_pool_indices=reqs.long(),
            sequence_lengths=lengths,
            logical_positions=positions,
            compress_ratio=RATIO,
            is_extend=False,
            num_groups=pool.qsa_num_groups,
        )
        self.assertTrue(
            torch.equal(metadata.indexer_metadata.pending_ring_slots, want),
            "the host graph refresh diverged from the eager slot formula",
        )


if __name__ == "__main__":
    unittest.main()
