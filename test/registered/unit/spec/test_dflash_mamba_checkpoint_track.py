"""CPU regression for DFLASH's decode-checkpoint arithmetic (#115).

``DFlashWorkerV2._update_target_mamba_state_after_verify`` decides, per request,
whether the tokens a verify step *commits* cross a mamba checkpoint grid and,
if so, which accepted step holds that checkpoint. The crossing check must be
made against the committed (post-verify) lengths.

The committed device lengths are not published when the helper runs: the caller
clones ``batch.seq_lens`` into ``seq_lens_pre_verify`` before verify and only
derives/publishes ``new_seq_lens`` after this helper returns, and neither
``prepare_for_verify`` nor target verify advances the device tensor. So the
helper has to derive the post lengths itself (``seq_lens_pre_verify +
commit_lens``) rather than read ``batch.seq_lens`` -- otherwise the crossing
mask is always false and the backend is told ``mamba_steps_to_track = -1`` even
when an accepted block clearly crosses the grid (native NEXTN's
``spec_utils._verify_commit_step_indices`` has always derived post lengths; it
is unchanged here and serves as the reference).

Evidence boundary: this file is pure CPU and never imports the heavyweight
``sglang`` runtime (its ``__init__`` pulls torch / FlashInfer / JIT). It
instead reads the *actual* helper source from this checkout, execs that
function with ``mamba_track_grid`` injected, and asserts the exact backend
``mamba_steps_to_track`` and live-commit indices -- so it exercises production
arithmetic, not a reimplementation. It does NOT exercise the backend scatter or
real on-GPU cache-resume path (that needs HybridLinearAttnBackend and is
unvalidated at this checkpoint); explicit expected-step assertions, not a
looseness check, are what make the crossing cases base-red.
"""

import ast
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch


def register_cpu_ci(*_args, **_kwargs):
    """CPU-CI registration marker: ``sglang.test.ci.ci_register`` parses this
    call's literal args out of the AST (a runtime no-op). Defined locally so the
    registration can sit at module scope and this arithmetic test stays
    collectable/runnable on a source-only host without importing the heavy
    runtime (``sglang/__init__`` would load torch / FlashInfer / JIT)."""
    return None


# Module-scope (not inside try): RegistryVisitor.visit_Module only records
# top-level expression calls, so a nested/try-wrapped call would collect zero.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_HELPER = "_update_target_mamba_state_after_verify"
# Repo-root-relative path to the production module, resolved from THIS test
# file, so an editable install pointing at another checkout cannot be tested.
_MODULE_REL = "python/sglang/srt/speculative/dflash_worker_v2.py"


def _helper_source() -> str:
    """Read the production helper source from this checkout (no ``import``).

    The path is anchored to this test file, never to an installed/editable
    ``sglang`` package, so the arithmetic tested is the candidate sitting beside
    the test.
    """
    path = Path(__file__).resolve().parents[4] / _MODULE_REL
    source = path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == _HELPER:
            seg = ast.get_source_segment(source, node)
            if seg is not None:
                return seg
    raise AssertionError(f"{_HELPER} not found in {path}")


def _load_helper(grid: int):
    """Exec the production helper with a fixed ``mamba_track_grid`` injected."""
    namespace = {
        "torch": torch,
        "ScheduleBatch": object,
        "mamba_track_grid": lambda _page: grid,
    }
    exec(textwrap.dedent(_helper_source()), namespace)
    return namespace[_HELPER]


def _make_batch(pre, dtype, track_indices):
    return SimpleNamespace(
        tree_cache=SimpleNamespace(page_size=16),
        seq_lens=torch.tensor(
            pre, dtype=dtype
        ),  # device lens STILL == pre at call time
        mamba_track_indices=track_indices,
        req_pool_indices=torch.arange(len(pre) + 3, dtype=torch.int64),
    )


class TestDflashCheckpointTrack(unittest.TestCase):
    def _run(self, pre, commit, grid, *, dtype=torch.int32, tracking=True):
        helper = _load_helper(grid)
        track_indices = torch.arange(len(pre), dtype=torch.int64) if tracking else None
        batch = _make_batch(pre, dtype, track_indices)
        pre_verify = torch.tensor(pre, dtype=dtype)
        commit_lens = torch.tensor(commit, dtype=torch.int32)
        pre_before = batch.seq_lens.clone()
        pre_verify_before = pre_verify.clone()

        worker = MagicMock()
        worker._need_mamba_verify_commit = True
        helper(
            worker,
            batch=batch,
            seq_lens_pre_verify=pre_verify,
            commit_lens=commit_lens,
        )

        hook = (
            worker.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify
        )
        hook.assert_called_once()
        kwargs = hook.call_args[1]
        # Live-state commit indices (last accepted step) are unchanged by the fix.
        self.assertEqual(
            kwargs["last_correct_step_indices"].tolist(),
            [c - 1 for c in commit],
        )
        self.assertEqual(kwargs["last_correct_step_indices"].dtype, torch.int64)
        # req_pool_indices is sliced to the number of committed requests.
        self.assertEqual(kwargs["req_pool_indices"].shape[0], len(commit))
        # The helper must not mutate the pre-verify clone or the device tensor:
        # publication stays the caller's job (no early seq_lens write).
        self.assertTrue(torch.equal(batch.seq_lens, pre_before))
        self.assertTrue(torch.equal(pre_verify, pre_verify_before))

        steps = kwargs["mamba_steps_to_track"]
        if steps is None:
            self.assertFalse(tracking)
            return None
        self.assertEqual(steps.dtype, torch.int64)
        return steps.tolist()

    # --- both profiles cross the grid: base sent -1, fix sends the boundary step ---
    def test_flash_next_grid64_crossing_tracks_step_zero(self):
        # pre=63, 4 committed -> post=67 crosses 64; checkpoint at step 64-63-1=0.
        self.assertEqual(self._run([63], [4], 64), [0])

    def test_recover_ssm_grid256_crossing_tracks_step_zero(self):
        # 27B extra_buffer_lazy: pre=255, 8 committed -> post=263 crosses 256.
        self.assertEqual(self._run([255], [8], 256), [0])

    # --- boundary/non-boundary and no-crossing cases ---
    def test_exact_boundary_arrival_tracks_last_accepted_step(self):
        # post lands exactly on the grid: pre=60, commit=4 -> post=64, step 3.
        self.assertEqual(self._run([60], [4], 64), [3])

    def test_crossing_late_in_accepted_block_tracks_early_step(self):
        # pre=61, commit=8 -> post=69, checkpoint at step 64-61-1=2.
        self.assertEqual(self._run([61], [8], 64), [2])

    def test_start_on_boundary_does_not_cross(self):
        # pre already on the grid and stays in the same bucket -> no checkpoint.
        self.assertEqual(self._run([64], [4], 64), [-1])

    def test_within_bucket_is_minus_one(self):
        self.assertEqual(self._run([10], [4], 64), [-1])

    # --- mixed committed lengths: same grid, per-request commit changes
    # crossing eligibility (base collapses all three to -1) ---
    def test_mixed_committed_lengths_grid64(self):
        # req0 pre=60 commit=1 -> post=61, no cross; req1 pre=60 commit=4 ->
        # post=64, step 3; req2 pre=63 commit=1 -> post=64, step 0.
        self.assertEqual(
            self._run([60, 60, 63], [1, 4, 1], 64),
            [-1, 3, 0],
        )

    def test_mixed_committed_lengths_grid256(self):
        # req0 pre=250 commit=1 -> post=251, no cross; req1 pre=250 commit=6 ->
        # post=256, step 5; req2 pre=255 commit=1 -> post=256, step 0.
        self.assertEqual(
            self._run([250, 250, 255], [1, 6, 1], 256),
            [-1, 5, 0],
        )

    # --- dtypes / disabled paths ---
    def test_int64_lengths_track_the_same_step(self):
        self.assertEqual(self._run([63], [4], 64, dtype=torch.int64), [0])

    def test_tracking_disabled_returns_none_but_still_commits_live_state(self):
        # mamba_track_indices None -> no checkpoint steps, but the live commit
        # hook still runs (covered inside _run via assert_called_once + indices).
        self.assertIsNone(self._run([63], [4], 64, tracking=False))

    def test_no_mamba_commit_skips_backend_entirely(self):
        helper = _load_helper(64)
        batch = _make_batch([63], torch.int32, torch.arange(1, dtype=torch.int64))
        worker = MagicMock()
        worker._need_mamba_verify_commit = False
        helper(
            worker,
            batch=batch,
            seq_lens_pre_verify=torch.tensor([63], dtype=torch.int32),
            commit_lens=torch.tensor([4], dtype=torch.int32),
        )
        worker.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
