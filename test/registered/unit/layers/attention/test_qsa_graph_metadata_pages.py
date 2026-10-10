"""Page-parallel QSA graph row metadata (``SGLANG_QSA_META_PAGE_PARALLEL``).

``_qsa_graph_row_metadata_kernel`` is launched one program per row with
``num_warps=1`` and then walks the WHOLE ``max_pages`` page table in a serial
``_PAGE_BLOCK``-strided loop. Every supported recipe serves
``--context-length 524288 --page-size 64``, so the graph page table is 8192
wide and the loop is 64 trips of a strided gather issued by a single warp; the
donor's measurements (and its 4096-page shape, kept covered below) were taken at
the older 262144 context.

With the flag on the page dimension moves to ``program_id(1)``: one program per
``_PAGE_BLOCK`` slice, and the scalar fields (compressed length, boundary write
slot, logical position, ring state slot, trailing-group ring slots) are written
by the row's first page program only.  Every store is the same value at the
same address as the serial launch, so the buffers must come out bit-identical
-- that is what the parity and reference cases check, at the fixed W4 and W8
ring widths the pool derives (and one generic multi-group width).

Run on CPU (no GPU, triton's interpret mode executes the kernel on CPU):
    TRITON_INTERPRET=1 python -m pytest test/registered/unit/layers/attention/test_qsa_graph_metadata_pages.py
The parity/reference cases also run natively when a CUDA device is visible and
skip themselves otherwise; the grid-planner and wiring cases always run.
"""

import unittest

import torch
import triton

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa import graph_metadata as gm
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# configs/pennyroyal/serve-flash-next*.sh and serve-qwen38-27b-dflash2.sh all set
# --context-length 524288 --page-size 64 (page ids are full-KV pages).
FULL_PAGE = 64
SERVED_PAGES = 524288 // FULL_PAGE  # 8192
# The donor's geometry, i.e. the shape its numbers were taken at.
DONOR_PAGES = 262144 // FULL_PAGE  # 4096
MAX_PAGES = SERVED_PAGES
RATIO = 4
SENTINEL = -7
INTERPRET = triton.knobs.runtime.interpret

# The shipped ring widths come from the pool derivation, not from a guess: W4 is
# the default window, W8 the optional fixed build (PENNY_SPEC_WIDTH).
W4_GROUPS = QSATokenToKVPool.pending_ring_num_groups(
    max_num_draft_tokens=4, compress_ratio=RATIO
)
W8_GROUPS = QSATokenToKVPool.pending_ring_num_groups(
    max_num_draft_tokens=8, compress_ratio=RATIO
)
GENERIC_GROUPS = 2
RING_WIDTHS = (W4_GROUPS, W8_GROUPS, GENERIC_GROUPS)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _buffers(rows, max_pages, device):
    """Pre-filled with a sentinel so an unwritten entry cannot pass as equal."""
    return {
        "compressed_lens": torch.full(
            (rows,), SENTINEL, dtype=torch.int32, device=device
        ),
        "write_locs": torch.full((rows,), SENTINEL, dtype=torch.int32, device=device),
        "page_table": torch.full(
            (rows, max_pages), SENTINEL, dtype=torch.int32, device=device
        ),
        "logical_positions": torch.full(
            (rows,), SENTINEL, dtype=torch.int32, device=device
        ),
        "state_slots": torch.full((rows,), SENTINEL, dtype=torch.int64, device=device),
        "ring_locs": torch.full(
            (rows, RATIO), SENTINEL, dtype=torch.int32, device=device
        ),
    }


def _inputs(rows, max_pages, num_requests=8, device="cpu", seed=1234, row_pages=None):
    """Draft-loop-shaped row set: one request per row, long growing contexts.

    The kernel reads ``req_to_token[req, seq_len - 1]`` unmasked, so the fixture
    keeps every length inside the row's own width. ``row_pages`` lets a case make
    the request row narrower than the graph page table, which is what leaves
    table entries unmasked-out (and still sentinel) at the tail.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    stride = (row_pages or max_pages) * FULL_PAGE
    req_to_token = torch.randint(
        0,
        1 << 20,
        (num_requests, stride),
        dtype=torch.int32,
        generator=generator,
    ).to(device)
    base = max(1, min(30000, stride - rows - 2))
    seq = torch.arange(base, base + rows, dtype=torch.int32, device=device)
    req = (torch.arange(rows, device=device) % num_requests).to(torch.int32)
    return seq, req, req_to_token


def _run_kernel(
    page_parallel, seq, req, req_to_token, max_pages, num_groups, out, page_block=None
):
    gm._qsa_graph_row_metadata_kernel[
        gm.qsa_row_metadata_grid(
            seq.shape[0], max_pages, page_block or gm._PAGE_BLOCK, page_parallel
        )
    ](
        seq,
        req,
        out["compressed_lens"],
        out["write_locs"],
        out["page_table"],
        out["logical_positions"],
        out["state_slots"],
        out["ring_locs"],
        req_to_token,
        req_to_token.stride(0),
        max_pages,
        RATIO=RATIO,
        NUM_GROUPS=num_groups,
        FULL_PAGE=FULL_PAGE,
        PAGE_BLOCK=page_block or gm._PAGE_BLOCK,
        PAGE_PARALLEL=page_parallel,
        num_warps=1,
    )
    return out


def _reference(seq, req, req_to_token, max_pages, num_groups, page_block):
    """Torch model of the kernel, so parity is not just new-vs-old kernel.

    Everything is computed on the host: the caller's tensors may live on CUDA and
    the comparison below is device-normalised.
    """
    rows = seq.shape[0]
    seq = seq.cpu()
    req = req.cpu()
    req_to_token = req_to_token.cpu()
    stride = req_to_token.shape[1]
    seq64 = seq.to(torch.int64)
    req64 = req.to(torch.int64)
    current = torch.clamp(seq64 - 1, min=0)
    last_loc = req_to_token[req64, current].to(torch.int64)
    ring_span = RATIO * num_groups
    group = (current // RATIO) % num_groups
    base = req64 * ring_span + group * RATIO
    out = {
        "compressed_lens": (seq64 // RATIO).to(torch.int32),
        "write_locs": torch.where(
            (seq64 > 0) & (seq64 % RATIO == 0),
            last_loc // RATIO,
            torch.zeros_like(last_loc),
        ).to(torch.int32),
        "logical_positions": current.to(torch.int32),
        "state_slots": base + (current % RATIO),
        "ring_locs": torch.stack(
            [
                base
                + torch.clamp(current - (RATIO - 1 - k), min=0).to(torch.int64) % RATIO
                for k in range(RATIO)
            ],
            dim=1,
        ).to(torch.int32),
    }
    limit = min(max_pages, stride // FULL_PAGE)
    page_ids = torch.full((rows, max_pages), SENTINEL, dtype=torch.int64)
    # The kernel reads the FULL_PAGE-strided columns of the request row: the
    # first slot of each page is that page's id.
    gathered = req_to_token[req64][:, ::FULL_PAGE]
    page_ids[:, :limit] = (gathered[:, :limit].to(torch.int64) // FULL_PAGE).clamp(
        min=0
    )
    out["page_table"] = page_ids.to(torch.int32)
    assert page_block > 0
    return out


class TestRowMetadataGrid(unittest.TestCase):
    """Pure python launch planner: runs anywhere, no triton execution."""

    def test_serial_grid_is_unchanged(self):
        for rows in (1, 4, 16):
            self.assertEqual(
                gm.qsa_row_metadata_grid(rows, MAX_PAGES, 128, False), (rows,)
            )

    def test_parallel_grid_covers_every_page(self):
        for rows in (1, 4, 16, 129):
            grid = gm.qsa_row_metadata_grid(rows, MAX_PAGES, gm._PAGE_BLOCK, True)
            self.assertEqual(grid[0], rows)
            programs = grid[1]
            self.assertGreaterEqual(programs * gm._PAGE_BLOCK, MAX_PAGES)
            self.assertLess((programs - 1) * gm._PAGE_BLOCK, MAX_PAGES)

    def test_served_page_count_gets_one_program_per_page_block(self):
        # The number this item is about: 64 serial loop trips at the served
        # 524288 context, and the donor's 32 at its 262144 measurement shape.
        self.assertEqual(
            gm.qsa_row_metadata_grid(1, SERVED_PAGES, gm._PAGE_BLOCK, True), (1, 64)
        )
        self.assertEqual(
            gm.qsa_row_metadata_grid(1, DONOR_PAGES, gm._PAGE_BLOCK, True), (1, 32)
        )

    def test_shipped_ring_widths_are_the_pool_derivation(self):
        """Labels used below must match the pool, not a hard-coded assumption."""
        self.assertEqual(W4_GROUPS, 1, "fixed W4/default stays a single group")
        self.assertEqual(W8_GROUPS, 3, "fixed W8/R4 spans 3 ring groups")
        self.assertEqual(len(set(RING_WIDTHS)), 3, "W8 differs from W4 and generic")

    def test_ragged_page_count_rounds_up(self):
        self.assertEqual(gm.qsa_row_metadata_grid(2, 129, 128, True), (2, 2))
        self.assertEqual(gm.qsa_row_metadata_grid(2, 128, 128, True), (2, 1))
        self.assertEqual(gm.qsa_row_metadata_grid(1, 1, 128, True), (1, 1))

    def test_rejects_degenerate_shapes(self):
        for rows, pages, block in ((0, 4096, 128), (1, 0, 128), (1, 4096, 0)):
            with self.subTest(rows=rows, pages=pages, block=block):
                with self.assertRaises(ValueError):
                    gm.qsa_row_metadata_grid(rows, pages, block, True)

    def test_flag_defaults_off(self):
        self.assertFalse(gm.page_parallel_enabled())
        with envs.SGLANG_QSA_META_PAGE_PARALLEL.override(True):
            self.assertTrue(gm.page_parallel_enabled())
        self.assertFalse(gm.page_parallel_enabled())


class TestLaunchWiring(unittest.TestCase):
    """launch_graph_metadata must pass the grid and the constexpr together."""

    def _launch(self, page_parallel, num_rows=3):
        recorded = []

        class _Kernel:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, grid):
                outer = self

                def run(*args, **kwargs):
                    recorded.append((outer.name, grid, kwargs))
                    return None

                return run

        layout, row = gm._qsa_graph_layout_kernel, gm._qsa_graph_row_metadata_kernel
        table_pages = 257  # ragged on purpose: not a multiple of PAGE_BLOCK
        metadata = type(
            "M",
            (),
            {
                "indexer_metadata": type(
                    "I",
                    (),
                    {
                        "graph_compressed_page_table": torch.empty(
                            num_rows, table_pages, dtype=torch.int32
                        ),
                        "graph_prefix_lengths": torch.zeros(
                            num_rows, dtype=torch.int32
                        ),
                        "graph_compressed_lengths": torch.zeros(
                            num_rows, dtype=torch.int32
                        ),
                        "graph_write_locs": torch.zeros(num_rows, dtype=torch.int32),
                        "decode_logical_positions": torch.zeros(
                            num_rows, dtype=torch.int32
                        ),
                        "pending_ring_slots": torch.zeros(num_rows, dtype=torch.int64),
                        "graph_ring_group_locs": torch.zeros(
                            num_rows, RATIO, dtype=torch.int32
                        ),
                        "compress_ratio": RATIO,
                    },
                )(),
                "sequence_lengths": torch.zeros(num_rows, dtype=torch.int32),
                "row_req_pool_indices": torch.zeros(num_rows, dtype=torch.int32),
            },
        )()
        pool = type(
            "P",
            (),
            {
                "qsa_num_groups": W8_GROUPS,
                "qsa_compressed_page_size": FULL_PAGE // RATIO,
            },
        )()
        with envs.SGLANG_QSA_META_PAGE_PARALLEL.override(page_parallel):
            gm._qsa_graph_layout_kernel = _Kernel("layout")
            gm._qsa_graph_row_metadata_kernel = _Kernel("row")
            try:
                gm.launch_graph_metadata(
                    mode=0,
                    bs=num_rows,
                    num_rows=num_rows,
                    seq_lens=torch.zeros(num_rows, dtype=torch.int32),
                    req_pool_indices=torch.zeros(num_rows, dtype=torch.int32),
                    extend_lens=None,
                    extend_len=0,
                    num_padding=0,
                    metadata=metadata,
                    req_to_token=torch.zeros(4, 4096 * FULL_PAGE, dtype=torch.int32),
                    pool=pool,
                )
            finally:
                gm._qsa_graph_layout_kernel, gm._qsa_graph_row_metadata_kernel = (
                    layout,
                    row,
                )
        names = [name for name, _, _ in recorded]
        self.assertEqual(names, ["layout", "row"], "both kernels, layout first")
        return {name: (grid, kwargs) for name, grid, kwargs in recorded}["row"]

    def test_flag_off_keeps_the_1d_grid_and_serial_kernel(self):
        grid, kwargs = self._launch(False)
        self.assertEqual(grid, (3,))
        self.assertFalse(kwargs["PAGE_PARALLEL"])
        self.assertEqual(kwargs["PAGE_BLOCK"], gm._PAGE_BLOCK)
        # The pool's ring width (here the fixed W8 derivation) and the full-KV
        # page width still reach the kernel unchanged, so the graph buffers the
        # scoring kernels read keep their layout.
        self.assertEqual(kwargs["NUM_GROUPS"], W8_GROUPS)
        self.assertEqual(kwargs["FULL_PAGE"], FULL_PAGE)

    def test_flag_on_sends_ragged_page_count_to_a_second_dimension(self):
        grid, kwargs = self._launch(True)
        self.assertEqual(grid, (3, triton.cdiv(257, gm._PAGE_BLOCK)))
        self.assertTrue(kwargs["PAGE_PARALLEL"])
        self.assertEqual(kwargs["PAGE_BLOCK"], gm._PAGE_BLOCK)
        self.assertEqual(kwargs["NUM_GROUPS"], W8_GROUPS)
        self.assertEqual(kwargs["FULL_PAGE"], FULL_PAGE)


class TestKernelParity(unittest.TestCase):
    """Flag on vs off must be bit-identical on every output buffer."""

    def _compare(
        self, rows, max_pages, num_groups, page_block=None, device=None, row_pages=None
    ):
        device = device or _device()
        seq, req, req_to_token = _inputs(
            rows, max_pages, device=device, row_pages=row_pages
        )
        self.assertLess(
            int(seq.max()) - 1,
            req_to_token.shape[1],
            "fixture lengths must stay inside the request row",
        )
        serial = _run_kernel(
            False,
            seq,
            req,
            req_to_token,
            max_pages,
            num_groups,
            _buffers(rows, max_pages, device),
            page_block=page_block,
        )
        parallel = _run_kernel(
            True,
            seq,
            req,
            req_to_token,
            max_pages,
            num_groups,
            _buffers(rows, max_pages, device),
            page_block=page_block,
        )
        for key, got in parallel.items():
            self.assertTrue(
                torch.equal(serial[key], got),
                f"{key} differs (rows={rows} max_pages={max_pages} "
                f"num_groups={num_groups} page_block={page_block})",
            )
        self.assertFalse(
            torch.all(serial["page_table"] == SENTINEL),
            "the page table must actually have been written",
        )
        return serial

    def setUp(self):
        if not (torch.cuda.is_available() or INTERPRET):
            self.skipTest(
                "kernel execution needs CUDA or TRITON_INTERPRET=1 (CPU interpreter)"
            )

    def test_bit_exact_at_the_served_page_count(self):
        # The table the current recipes actually allocate: 8192 pages, so the
        # serial form walks 64 trips while the parallel form spreads 64 CTAs.
        self._compare(1, SERVED_PAGES, num_groups=W4_GROUPS)

    def test_bit_exact_at_the_donors_measured_page_count(self):
        self._compare(1, DONOR_PAGES, num_groups=W4_GROUPS)

    def test_bit_exact_over_several_row_counts(self):
        for rows in (1, 4, 16):
            with self.subTest(rows=rows):
                self._compare(rows, 2048, num_groups=W4_GROUPS)

    def test_bit_exact_on_a_ragged_page_count(self):
        for pages in (129, 130, 257, 1):
            with self.subTest(pages=pages):
                self._compare(2, pages, num_groups=W4_GROUPS)

    def test_bit_exact_for_every_supported_ring_width(self):
        # Fixed W4/default (1 group), the optional fixed W8 (3 groups at R4) and
        # a generic multi-group width: the page split must not move a single ring
        # or state slot at any of them.
        for num_groups in RING_WIDTHS:
            with self.subTest(num_groups=num_groups):
                self._compare(4, 512, num_groups=num_groups)

    def test_bit_exact_with_small_page_blocks(self):
        for page_block in (1, 2, 8):
            with self.subTest(page_block=page_block):
                self._compare(2, 17, num_groups=W4_GROUPS, page_block=page_block)

    def test_table_wider_than_the_request_row_keeps_the_masked_tail(self):
        # Three page programs for a 257-page table whose rows only reach page
        # 130: the tail must stay untouched (sentinel) identically both ways.
        serial = self._compare(2, 257, num_groups=W4_GROUPS, row_pages=130)
        self.assertTrue(
            torch.all(serial["page_table"][:, 130:] == SENTINEL),
            "entries past the request row are not written by either launch",
        )
        self.assertTrue(torch.all(serial["page_table"][:, :130] != SENTINEL))


class TestKernelMatchesReference(unittest.TestCase):
    """Independent torch model of every stored field, flag on and off."""

    def setUp(self):
        if not (torch.cuda.is_available() or INTERPRET):
            self.skipTest(
                "kernel execution needs CUDA or TRITON_INTERPRET=1 (CPU interpreter)"
            )

    def test_stores_match_the_reference(self):
        rows, max_pages, page_block = 8, 640, 64
        for page_parallel in (False, True):
            for num_groups in RING_WIDTHS:
                with self.subTest(page_parallel=page_parallel, num_groups=num_groups):
                    device = _device()
                    seq, req, req_to_token = _inputs(rows, max_pages, device=device)
                    got = _run_kernel(
                        page_parallel,
                        seq,
                        req,
                        req_to_token,
                        max_pages,
                        num_groups,
                        _buffers(rows, max_pages, device),
                        page_block=page_block,
                    )
                    want = _reference(
                        seq, req, req_to_token, max_pages, num_groups, page_block
                    )
                    for key, expected in want.items():
                        self.assertEqual(expected.device.type, "cpu")
                        self.assertEqual(got[key].dtype, expected.dtype, key)
                        self.assertEqual(got[key].shape, expected.shape, key)
                        self.assertTrue(
                            torch.equal(got[key].cpu(), expected),
                            f"{key} differs from the reference "
                            f"(page_parallel={page_parallel}, "
                            f"num_groups={num_groups})",
                        )

    def test_reference_is_computed_on_the_host_for_a_device_input_set(self):
        """The comparison must not depend on where the kernel ran."""
        device = _device()
        seq, req, req_to_token = _inputs(4, 128, device=device)
        want = _reference(seq, req, req_to_token, 128, W8_GROUPS, 64)
        for key, expected in want.items():
            self.assertEqual(expected.device.type, "cpu", key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
