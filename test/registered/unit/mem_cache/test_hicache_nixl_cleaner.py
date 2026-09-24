"""Unit tests for the NIXL FILE L3 cleaner."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import shutil
import tempfile
import unittest

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.nixl.nixl_cleaner import (
    GIBIBYTE,
    HiCacheL3Cleaner,
    _allocated_bytes,
    _parse_group_key,
    _safe_unlink,
)
from sglang.srt.mem_cache.storage.nixl.nixl_utils import (
    NixlBackendConfig,
    NixlFileManager,
)
from sglang.test.test_utils import CustomTestCase


class TestHiCacheL3Cleaner(CustomTestCase):
    """Tests for watermark-driven cleanup over bucketed NIXL FILE layout."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_nixl_l3_cleaner_")
        self.base_dirs = [os.path.join(self.test_dir, f"disk{i}") for i in range(2)]
        self.file_manager = NixlFileManager(self.base_dirs, use_direct_io=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _write_key(self, key: str, *, mtime: float, size: int = 16) -> str:
        path = self.file_manager.get_file_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x" * size)
        os.utime(path, (mtime, mtime))
        return path

    def _run_single_group_cleanup(self) -> None:
        cleaner = HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            high_watermark=80.0,
            low_watermark=70.0,
            recheck_groups=1,
            unlink_workers=1,
        )
        usage_calls: dict[str, int] = {}

        def fake_usage(path: str) -> float:
            usage_calls[path] = usage_calls.get(path, 0) + 1
            return 90.0 if usage_calls[path] == 1 else 60.0

        cleaner._disk_usage_pct = fake_usage
        self.assertTrue(cleaner._tick())

    def test_parse_group_key_strips_rank_and_kv_suffix(self):
        """Keys for TP ranks and zero-copy K/V files share one cleanup group."""
        self.assertEqual(_parse_group_key("page-a_model_0_8"), "page-a_model")
        self.assertEqual(_parse_group_key("page-a_model_7_8_k"), "page-a_model")
        self.assertEqual(_parse_group_key("page-a_model_7_8_v"), "page-a_model")
        self.assertEqual(_parse_group_key("page-a_model_k"), "page-a_model")

    def test_parse_group_key_strips_hybrid_component_suffix(self):
        """All hybrid component shapes share the logical page's cleanup group."""
        names = [
            "page-a_model_7_8_kv_k",
            "page-a_model_7_8_swa_k",
            "page-a_model_7_8_swa_v",
            "page-a_model_7_8_mamba_temporal",
            "page-a_model_7_8_mamba_conv_0",
            "page-a_model_7_8_indexer_2",
            "page-a_model_7_8_draft_swa",
            "page-a_model_deepseek_v4_c4_indexer_state_2",
        ]

        for name in names:
            with self.subTest(name=name):
                self.assertEqual(_parse_group_key(name), "page-a_model")

    def test_tick_deletes_oldest_group_across_bucketed_dirs(self):
        """A cleaner batch deletes all files in the oldest logical key group."""
        old_keys = ["page-old_model_0_2", "page-old_model_1_2"]
        new_keys = ["page-new_model_0_2", "page-new_model_1_2"]
        old_paths = [self._write_key(key, mtime=100.0) for key in old_keys]
        new_paths = [self._write_key(key, mtime=200.0) for key in new_keys]

        self._run_single_group_cleanup()
        self.assertFalse(any(os.path.exists(path) for path in old_paths))
        self.assertTrue(all(os.path.exists(path) for path in new_paths))

    def test_tick_deletes_hybrid_components_atomically(self):
        """Evict every pool component and TP rank for one logical page."""
        physical_suffixes = [
            "",
            "_k",
            "_v",
            "_kv_k",
            "_kv_v",
            "_swa_k",
            "_swa_v",
            "_mamba_temporal",
            "_mamba_conv_0",
        ]
        old_keys = [
            f"page-old_model_{rank}_2{suffix}"
            for rank in range(2)
            for suffix in physical_suffixes
        ]
        new_keys = [
            f"page-new_model_{rank}_2{suffix}"
            for rank in range(2)
            for suffix in physical_suffixes
        ]
        old_paths = [self._write_key(key, mtime=100.0) for key in old_keys]
        new_paths = [self._write_key(key, mtime=200.0) for key in new_keys]

        self._run_single_group_cleanup()
        self.assertFalse(any(os.path.exists(path) for path in old_paths))
        self.assertTrue(all(os.path.exists(path) for path in new_paths))

    def test_tick_ignores_non_bucket_directories(self):
        """Only hash-bucket directories are treated as NIXL FILE cache entries."""
        non_bucket = os.path.join(self.base_dirs[0], "not-a-bucket")
        os.makedirs(non_bucket, exist_ok=True)
        unrelated = os.path.join(non_bucket, "page-old_model_0_2")
        with open(unrelated, "wb") as f:
            f.write(b"x")

        cleaner = HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            high_watermark=80.0,
            low_watermark=70.0,
            unlink_workers=1,
        )
        cleaner._disk_usage_pct = lambda _path: 90.0

        self.assertFalse(cleaner._tick())
        self.assertTrue(os.path.exists(unrelated))

    def test_safe_unlink_tolerates_missing_and_os_errors(self):
        """Cleanup races should not fail the cleaner tick."""
        missing = os.path.join(self.test_dir, "missing")
        existing = os.path.join(self.test_dir, "existing")
        with open(existing, "wb") as f:
            f.write(b"abc")

        allocated = _allocated_bytes(os.stat(existing))
        self.assertGreaterEqual(allocated, 3)
        self.assertEqual(_safe_unlink(missing), (False, 0))
        self.assertEqual(_safe_unlink(self.test_dir), (False, 0))
        self.assertEqual(_safe_unlink(existing), (True, allocated))
        self.assertFalse(os.path.exists(existing))

    def test_start_only_runs_on_tp_rank_zero(self):
        """Only TP rank 0 owns file cleanup for a shared storage directory."""
        cleaner = HiCacheL3Cleaner(self.base_dirs, tp_rank=1, interval_sec=0.01)
        cleaner.start()
        self.assertIsNone(cleaner._thread)

    def test_nixl_config_parses_l3_cleaner_options(self):
        """Cleaner settings are top-level NIXL config, not plugin init params."""
        cfg = NixlBackendConfig(
            {
                "use_uring": "true",
                "l3_cleaner_enabled": False,
                "l3_cleaner_high_watermark": "85",
                "l3_cleaner_low_watermark": 75,
            }
        )

        cleaner_config = cfg.get_l3_cleaner_config()
        self.assertFalse(cleaner_config["enabled"])
        self.assertEqual(cleaner_config["high_watermark"], 85.0)
        self.assertEqual(cleaner_config["low_watermark"], 75.0)
        self.assertEqual(cfg.get_backend_initparams("POSIX"), {"use_uring": "true"})

        default_config = NixlBackendConfig().get_l3_cleaner_config()
        self.assertTrue(default_config["enabled"])

    def test_nixl_config_rejects_non_boolean_l3_cleaner_enabled(self):
        """Cleaner enablement uses native config booleans only."""
        cfg = NixlBackendConfig({"l3_cleaner_enabled": "false"})

        with self.assertRaises(ValueError):
            cfg.get_l3_cleaner_config()


class TestHiCacheL3CleanerQuota(CustomTestCase):
    """Tests for the optional cache-owned byte budget (GiB soft quota)."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_nixl_l3_quota_")
        self.base_dirs = [os.path.join(self.test_dir, f"disk{i}") for i in range(2)]
        self.file_manager = NixlFileManager(self.base_dirs, use_direct_io=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _write_key(self, key: str, *, mtime: float, size: int = 4096) -> str:
        path = self.file_manager.get_file_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x" * size)
        os.utime(path, (mtime, mtime))
        return path

    def _make_cleaner(self, *, max_cache_bytes: float, unlink_workers: int = 1):
        cleaner = HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            high_watermark=80.0,
            low_watermark=70.0,
            recheck_groups=1,
            unlink_workers=unlink_workers,
            max_cache_bytes=max_cache_bytes,
        )
        # Filesystem is nowhere near its watermarks: only the quota can trigger.
        cleaner._disk_usage_pct = lambda _path: 10.0
        return cleaner

    def _make_sparse_key(self, key: str, *, mtime: float, apparent: int) -> str:
        path = self.file_manager.get_file_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x" * 4096)
            f.truncate(apparent)
        os.utime(path, (mtime, mtime))
        return path

    def _allocated(self) -> int:
        return _scan_allocated(self.base_dirs)

    def test_missing_feature_quota_only_pressure_evicts_oldest_group(self):
        """A quota over budget triggers eviction while the filesystem is free."""
        old_keys = ["page-old_model_0_2", "page-old_model_1_2"]
        new_keys = ["page-new_model_0_2", "page-new_model_1_2"]
        old_paths = [self._write_key(key, mtime=100.0) for key in old_keys]
        new_paths = [self._write_key(key, mtime=200.0) for key in new_keys]
        allocated = _allocated_bytes(os.stat(old_paths[0])) + _allocated_bytes(
            os.stat(old_paths[1])
        )
        cap = (
            allocated
            + _allocated_bytes(os.stat(new_paths[0]))
            + _allocated_bytes(os.stat(new_paths[1]))
        )

        cleaner = self._make_cleaner(max_cache_bytes=cap)
        self.assertFalse(cleaner._tick(), "at the cap there is no quota pressure")

        cleaner = self._make_cleaner(max_cache_bytes=cap - 1)
        self.assertTrue(cleaner._tick())
        self.assertFalse(any(os.path.exists(path) for path in old_paths))
        self.assertTrue(all(os.path.exists(path) for path in new_paths))

    def test_quota_eviction_needs_no_filesystem_pressure(self):
        """The budget works alongside, not instead of, disk watermarks."""
        self._write_key("page-a_model_0_2", mtime=100.0)
        cleaner = self._make_cleaner(max_cache_bytes=1)
        cleaner._disk_usage_pct = lambda _path: 0.0
        self.assertTrue(cleaner._tick())
        self.assertEqual(self._allocated(), 0)

    def test_disabled_by_default_and_zero(self):
        """Default/zero budget keeps the old filesystem-only behavior."""
        path = self._write_key("page-old_model_0_2", mtime=100.0)

        for max_cache_bytes in (None, 0):
            kwargs = {} if max_cache_bytes is None else {"max_cache_bytes": 0}
            cleaner = HiCacheL3Cleaner(
                self.base_dirs,
                tp_rank=0,
                high_watermark=80.0,
                low_watermark=70.0,
                unlink_workers=1,
                **kwargs,
            )
            cleaner._disk_usage_pct = lambda _path: 10.0
            with self.subTest(max_cache_bytes=max_cache_bytes):
                self.assertFalse(cleaner._tick())
                self.assertTrue(os.path.exists(path))

    def test_quota_target_hysteresis_keeps_newer_groups(self):
        """Quota eviction stops at 90% of the cap, not just below the cap."""
        keys = [f"page-g{i}_model_0_2" for i in range(4)]
        paths = [self._write_key(key, mtime=100.0 + i) for i, key in enumerate(keys)]
        per_group = _allocated_bytes(os.stat(paths[0]))
        cap = 4 * per_group - 1  # one byte of quota pressure

        cleaner = self._make_cleaner(max_cache_bytes=cap)
        self.assertTrue(cleaner._tick())
        # Evicting the single oldest group leaves 3 groups = 75% < 90% target.
        self.assertFalse(os.path.exists(paths[0]))
        self.assertTrue(all(os.path.exists(path) for path in paths[1:]))

    def test_mixed_pressure_survives_unlinks_and_disappearing_files(self):
        """Quota+filesystem cleanup tolerates failed and racy unlinks."""
        doomed = self._write_key("page-old_model_0_2", mtime=100.0)
        ghost = self._write_key("page-ghost_model_0_2", mtime=90.0)
        keeper = self._write_key("page-new_model_0_2", mtime=200.0)
        real_unlink = os.unlink

        def flaky_unlink(path):
            if path == doomed:
                raise PermissionError(path)
            if path == ghost:
                # Vanished between the scan and the unlink.
                raise FileNotFoundError(path)
            real_unlink(path)

        cleaner = self._make_cleaner(max_cache_bytes=1)
        cleaner._disk_usage_pct = lambda _path: 90.0
        os.unlink = flaky_unlink
        try:
            self.assertTrue(cleaner._tick())
        finally:
            os.unlink = real_unlink

        self.assertTrue(os.path.exists(doomed), "failed unlink keeps the file")
        self.assertTrue(os.path.exists(ghost), "vanished file is not a failure")
        self.assertFalse(os.path.exists(keeper), "filesystem watermark still drives")

    def test_sparse_files_count_allocated_blocks_not_apparent_size(self):
        """Quota measures st_blocks*512, never the apparent sparse length."""
        sparse = self._make_sparse_key(
            "page-sparse_model_0_2", mtime=100.0, apparent=512 * GIBIBYTE
        )
        stat = os.stat(sparse)
        allocated = _allocated_bytes(stat)
        self.assertGreater(stat.st_size, 20 * allocated, "file really is sparse")
        self.assertGreater(allocated, 0, "fixture has allocated blocks")

        # Apparent size would blow through a 1 GiB cap; allocated size does not.
        free_of_apparent = self._make_cleaner(max_cache_bytes=1 * GIBIBYTE)
        self.assertFalse(free_of_apparent._tick())
        self.assertTrue(os.path.exists(sparse))

        below_allocated = self._make_cleaner(max_cache_bytes=allocated - 1)
        self.assertTrue(below_allocated._tick())
        self.assertFalse(os.path.exists(sparse))

    def test_whole_hybrid_group_across_dirs_evicted_for_quota(self):
        """TP ranks and hybrid components leave together when quota drives."""
        suffixes = ["", "_k", "_v", "_mamba_temporal", "_mamba_conv_0"]
        old_paths = [
            self._write_key(f"page-old_model_{rank}_2{sfx}", mtime=100.0)
            for rank in range(2)
            for sfx in suffixes
        ]
        new_paths = [
            self._write_key(f"page-new_model_{rank}_2{sfx}", mtime=200.0)
            for rank in range(2)
            for sfx in suffixes
        ]
        per_group = sum(_allocated_bytes(os.stat(p)) for p in old_paths)

        cleaner = self._make_cleaner(
            max_cache_bytes=2 * per_group - 1, unlink_workers=4
        )
        self.assertTrue(cleaner._tick())
        self.assertFalse(any(os.path.exists(p) for p in old_paths))
        self.assertTrue(all(os.path.exists(p) for p in new_paths))

    def test_quota_does_not_touch_sibling_namespaces(self):
        """Only the cleaner's configured storage dirs are scanned or evicted."""
        inside = self._write_key("page-old_model_0_2", mtime=100.0)
        sibling_dir = os.path.join(self.test_dir, "sibling-namespace")
        # A full bucket directory with a stale file outside configured dirs.
        sibling_bucket = os.path.join(sibling_dir, "00")
        os.makedirs(sibling_bucket, exist_ok=True)
        sibling = os.path.join(sibling_bucket, "page-old_model_0_2")
        with open(sibling, "wb") as f:
            f.write(b"y" * 4096)
        outside_symlink = os.path.join(self.base_dirs[0], "linked")
        os.symlink(sibling, outside_symlink)
        stray_dir = os.path.join(self.base_dirs[0], "stray")
        os.makedirs(stray_dir, exist_ok=True)
        stray = os.path.join(stray_dir, "page-old_model_0_2")
        with open(stray, "wb") as f:
            f.write(b"z" * 4096)

        cleaner = self._make_cleaner(max_cache_bytes=1)
        self.assertTrue(cleaner._tick())
        self.assertFalse(os.path.exists(inside))
        self.assertTrue(os.path.exists(sibling))
        self.assertTrue(os.path.exists(stray))
        self.assertTrue(os.path.islink(outside_symlink))

    def _write_raw(self, base_dir: str, name: str, *, mtime: float, size: int) -> str:
        bucket = os.path.join(base_dir, "00")
        os.makedirs(bucket, exist_ok=True)
        path = os.path.join(bucket, name)
        with open(path, "wb") as f:
            f.write(b"x" * size)
        os.utime(path, (mtime, mtime))
        return path

    def test_mixed_pressure_stops_cold_groups_after_quota_relieved(self):
        """Filesystem pressure must not evict cool-disk-only groups once the
        quota target was reached."""
        hot, cool = self.base_dirs
        per_file = 4096
        cold_old = self._write_raw(cool, "p1_model_0_2", mtime=100.0, size=per_file)
        hot_a = self._write_raw(hot, "p2_model_0_2", mtime=200.0, size=per_file)
        hot_b = self._write_raw(hot, "p3_model_0_2", mtime=300.0, size=per_file)
        cold_new = self._write_raw(cool, "p4_model_0_2", mtime=400.0, size=per_file)
        for path in (cold_old, hot_a, hot_b, cold_new):
            self.assertEqual(_allocated_bytes(os.stat(path)), per_file)

        cleaner = HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            high_watermark=80.0,
            low_watermark=70.0,
            recheck_groups=1,
            unlink_workers=1,
            # Over budget: evicting the oldest (cool-only) group and the next
            # oldest crosses the 90% quota target.
            max_cache_bytes=3 * per_file + 1,
        )
        cleaner._disk_usage_pct = lambda path: (
            90.0 if os.path.samefile(path, hot) else 10.0
        )
        self.assertTrue(cleaner._tick())

        self.assertFalse(os.path.exists(cold_old), "quota evicted the oldest group")
        self.assertFalse(os.path.exists(hot_a))
        self.assertFalse(os.path.exists(hot_b), "hot filesystem still drains")
        self.assertTrue(
            os.path.exists(cold_new),
            "cool-disk-only group survives once quota was relieved",
        )

    def test_default_recheck_batch_stops_near_quota_target(self):
        """Production default recheck_groups=50 must not wipe whole batches.

        With 101 equal groups over a cap of 100, reaching the 90%-of-cap
        target needs ~11 groups; the default batch size must not delete 50.
        """
        unit = 4096
        n_groups = 101
        paths = []
        for i in range(n_groups):
            path = self._write_key(f"page-g{i:03d}_model_0_2", mtime=100.0 + i)
            self.assertEqual(_allocated_bytes(os.stat(path)), unit)
            paths.append(path)
        cap = 100 * unit

        cleaner = HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            high_watermark=80.0,
            low_watermark=70.0,
            unlink_workers=1,
            max_cache_bytes=cap,
        )
        self.assertEqual(cleaner.recheck_groups, 50)  # production default
        cleaner._disk_usage_pct = lambda _path: 10.0
        self.assertTrue(cleaner._tick())

        survivors = [path for path in paths if os.path.exists(path)]
        deleted = n_groups - len(survivors)
        # Enough to reach 90% of cap (11 groups), and nowhere near 50.
        self.assertGreaterEqual(deleted, 11)
        self.assertLessEqual(deleted, 15)
        self.assertFalse(os.path.exists(paths[0]), "oldest groups go first")
        self.assertTrue(os.path.exists(paths[-1]), "newest group survives")

    def test_default_recheck_batch_slight_overcap(self):
        """Cap = total-1 must evict ~10% of groups, not every group."""
        unit = 4096
        n_groups = 41
        paths = [
            self._write_key(f"page-g{i:03d}_model_0_2", mtime=100.0 + i)
            for i in range(n_groups)
        ]
        for path in paths:
            self.assertEqual(_allocated_bytes(os.stat(path)), unit)

        cleaner = HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            high_watermark=80.0,
            low_watermark=70.0,
            unlink_workers=1,
            max_cache_bytes=n_groups * unit - 1,
        )
        cleaner._disk_usage_pct = lambda _path: 10.0
        self.assertTrue(cleaner._tick())

        survivors = [path for path in paths if os.path.exists(path)]
        deleted = n_groups - len(survivors)
        # Reaching 90% of (total-1) needs 5 groups; 41/41 is the old defect.
        self.assertGreaterEqual(deleted, 4)
        self.assertLessEqual(deleted, 8)
        self.assertTrue(survivors, "most groups survive a one-byte overcap")

    def test_stale_scan_credits_vanished_files_and_keeps_newer_groups(self):
        """Files already removed by another actor count against the quota.

        Only successful deletions used to reduce the estimate, so the
        cleaner kept deleting newer groups while actual usage was already
        under the quota target.
        """
        unit = 4096
        paths = [
            self._write_key(f"page-g{i}_model_0_2", mtime=100.0 + i) for i in range(10)
        ]
        for path in paths:
            self.assertEqual(_allocated_bytes(os.stat(path)), unit)
        vanished = {paths[0], paths[1]}
        real_unlink = os.unlink

        def racing_unlink(path):
            if path in vanished:
                # Another actor wins the race after our scan saw the file.
                real_unlink(path)
                raise FileNotFoundError(path)
            real_unlink(path)

        cleaner = HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            high_watermark=80.0,
            low_watermark=70.0,
            recheck_groups=1,
            unlink_workers=1,
            # cap = 5 groups' worth; target = 90% of that.
            max_cache_bytes=5 * unit,
        )
        cleaner._disk_usage_pct = lambda _path: 10.0
        os.unlink = racing_unlink
        try:
            self.assertTrue(cleaner._tick())
        finally:
            os.unlink = real_unlink

        # 10 units on disk, target 4.5 units: 6 groups must go (2 of them
        # vanished). The newer half stays even though the stale estimate
        # alone would have justified deleting further.
        for path in paths[:6]:
            self.assertFalse(os.path.exists(path), f"{path} should be gone")
        for path in paths[6:]:
            self.assertTrue(os.path.exists(path), f"{path} must survive")

    def test_failed_unlink_is_not_credited_as_freed(self):
        """Genuine unlink failures (file remains) never reduce the estimate."""
        unit = 4096
        paths = [
            self._write_key(f"page-g{i}_model_0_2", mtime=100.0 + i) for i in range(10)
        ]
        real_unlink = os.unlink
        blocked = {paths[0]}

        def failing_unlink(path):
            if path in blocked:
                raise PermissionError(path)
            real_unlink(path)

        cleaner = HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            high_watermark=80.0,
            low_watermark=70.0,
            recheck_groups=1,
            unlink_workers=1,
            max_cache_bytes=5 * unit,
        )
        cleaner._disk_usage_pct = lambda _path: 10.0
        os.unlink = failing_unlink
        try:
            self.assertTrue(cleaner._tick())
        finally:
            os.unlink = real_unlink

        self.assertTrue(os.path.exists(paths[0]), "still on disk, not credited")
        # Six successful deletions (paths[1..6]) are needed to reach the
        # target because the stuck file contributes nothing; paths[6] is the
        # last one credited and paths[7] onward must survive.
        self.assertFalse(os.path.exists(paths[6]))
        self.assertTrue(os.path.exists(paths[7]), "newer groups stop at target")

    def test_mixed_pressure_default_batch_preserves_cold_groups(self):
        """Default-size batches still stop cold-group eviction once the
        quota target was reached while filesystem pressure continues."""
        hot, cool = self.base_dirs
        unit = 4096
        cold_old = self._write_raw(cool, "p1_model_0_2", mtime=100.0, size=unit)
        hot_a = self._write_raw(hot, "p2_model_0_2", mtime=200.0, size=unit)
        hot_b = self._write_raw(hot, "p3_model_0_2", mtime=300.0, size=unit)
        cold_new = self._write_raw(cool, "p4_model_0_2", mtime=400.0, size=unit)
        for path in (cold_old, hot_a, hot_b, cold_new):
            self.assertEqual(_allocated_bytes(os.stat(path)), unit)

        cleaner = HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            high_watermark=80.0,
            low_watermark=70.0,
            unlink_workers=1,
            max_cache_bytes=3 * unit + 1,
        )
        cleaner._disk_usage_pct = lambda path: (
            90.0 if os.path.samefile(path, hot) else 10.0
        )
        self.assertTrue(cleaner._tick())

        self.assertFalse(os.path.exists(cold_old))
        self.assertFalse(os.path.exists(hot_a))
        self.assertFalse(os.path.exists(hot_b), "hot filesystem still drains")
        self.assertTrue(
            os.path.exists(cold_new),
            "cool-disk-only group survives once quota was relieved",
        )

    def test_quota_logging_accurate_after_filesystem_only_deletions(self):
        """Filesystem-pressure deletions update the quota estimate too.

        Quota enabled but not initially over cap: the cleanup log must show
        the running estimate dropping by exactly the bytes actually gone.
        """
        from unittest import mock

        from sglang.srt.mem_cache.storage.nixl import nixl_cleaner as nc

        paths = [
            self._write_key(f"page-g{i}_model_0_2", mtime=100.0 + i) for i in (0, 1)
        ]
        for path in paths:
            self.assertEqual(os.stat(path).st_size, 4096)

        cleaner = HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            high_watermark=80.0,
            low_watermark=70.0,
            recheck_groups=1,
            unlink_workers=1,
            # 5 GiB cap: the 2 GiB of scaled fixture cache is not quota
            # pressure, so only the filesystem watermarks drive deletion.
            max_cache_bytes=5 * GIBIBYTE,
        )
        usages = iter([90.0, 90.0, 60.0, 60.0])
        cleaner._disk_usage_pct = lambda _path: next(usages, 60.0)

        # Scale the fixture so GiB log formatting is meaningful without
        # writing GiB of real data (scan and unlink share the same helper).
        with mock.patch.object(
            nc, "_allocated_bytes", lambda stat: (stat.st_size // 4096) * GIBIBYTE
        ), self.assertLogs(nc.__name__, level="INFO") as logs:
            self.assertTrue(cleaner._tick())
        message = "\n".join(logs.output)
        self.assertIn("cache_bytes=2.00->1.00 GiB of 5.00 GiB budget", message)
        self.assertFalse(os.path.exists(paths[0]), "oldest group drained first")
        self.assertTrue(os.path.exists(paths[1]))

    def test_nonfinite_budget_rejected(self):
        """A NaN/inf byte budget is a configuration error."""
        for bad in (float("nan"), float("inf"), -1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    HiCacheL3Cleaner(self.base_dirs, tp_rank=0, max_cache_bytes=bad)

    def test_allocated_bytes_fallback_without_st_blocks(self):
        """Portability fallback uses apparent size only without st_blocks."""
        from types import SimpleNamespace

        self.assertEqual(
            _allocated_bytes(SimpleNamespace(st_blocks=None, st_size=42)), 42
        )
        self.assertEqual(
            _allocated_bytes(SimpleNamespace(st_blocks=3, st_size=1)), 3 * 512
        )

    def test_start_logs_active_budget_and_scope(self):
        """The startup line states the budget and the dirs it covers."""
        cleaner = HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            interval_sec=3600.0,
            max_cache_bytes=200 * GIBIBYTE,
        )
        with self.assertLogs(
            "sglang.srt.mem_cache.storage.nixl.nixl_cleaner", level="INFO"
        ) as logs:
            cleaner.start()
            cleaner.stop()
        message = "\n".join(logs.output)
        self.assertIn("200.00 GiB", message)
        self.assertIn("180.00 GiB", message)  # 90% quota target
        for base_dir in self.base_dirs:
            self.assertIn(base_dir, message)


class TestHiCacheL3CleanerQuotaConfig(CustomTestCase):
    """Config/env plumbing for SGLANG_HICACHE_NIXL_MAX_CACHE_GB."""

    def test_default_zero_and_env_budget(self):
        """Unset disables the quota; the env var enables it in GiB."""
        default = NixlBackendConfig().get_l3_cleaner_config()
        self.assertEqual(default["max_cache_gb"], 0.0)

        with envs.SGLANG_HICACHE_NIXL_MAX_CACHE_GB.override("200"):
            cfg = NixlBackendConfig().get_l3_cleaner_config()
        self.assertEqual(cfg["max_cache_gb"], 200.0)

    def test_zero_env_disables_budget(self):
        with envs.SGLANG_HICACHE_NIXL_MAX_CACHE_GB.override("0"):
            cfg = NixlBackendConfig().get_l3_cleaner_config()
        self.assertEqual(cfg["max_cache_gb"], 0.0)

    def test_top_level_config_overrides_env(self):
        """NIXL config precedence matches use_direct_io: config wins."""
        with envs.SGLANG_HICACHE_NIXL_MAX_CACHE_GB.override("200"):
            cfg = NixlBackendConfig({"l3_cleaner_max_cache_gb": 50})
            self.assertEqual(cfg.get_l3_cleaner_config()["max_cache_gb"], 50.0)
            # Without the key the env value applies.
            self.assertEqual(
                NixlBackendConfig({"use_uring": "true"}).get_l3_cleaner_config()[
                    "max_cache_gb"
                ],
                200.0,
            )

    def test_invalid_settings_fail_clearly(self):
        """Negative/non-numeric/bool budget values raise ValueError."""
        for bad in ("20x", -5, "-1.5", True, False, float("inf"), float("nan"), None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    NixlBackendConfig(
                        {"l3_cleaner_max_cache_gb": bad}
                    ).get_l3_cleaner_config()

        for bad in ("abc", "-5", "20GiB", "true", "nan", "inf"):
            with self.subTest(env=bad):
                with envs.SGLANG_HICACHE_NIXL_MAX_CACHE_GB.override(bad):
                    with self.assertRaises(ValueError):
                        NixlBackendConfig().get_l3_cleaner_config()

    def test_quota_keys_stay_out_of_plugin_initparams(self):
        """SGLang config keys do not leak into NIXL plugin params."""
        cfg = NixlBackendConfig(
            {
                "use_uring": "true",
                "l3_cleaner_enabled": True,
                "l3_cleaner_max_cache_gb": 200,
            }
        )
        self.assertEqual(cfg.get_backend_initparams("POSIX"), {"use_uring": "true"})

        cfg_full = NixlBackendConfig(
            {
                "use_direct_io": True,
                "l3_cleaner_max_cache_gb": 200,
                "plugin": {"posix": {"active": True, "use_uring": "true"}},
            }
        )
        self.assertEqual(
            cfg_full.get_backend_initparams("POSIX"),
            {"active": "True", "use_uring": "true"},
        )
        self.assertEqual(cfg_full.get_l3_cleaner_config()["max_cache_gb"], 200.0)

    def test_gib_converted_to_bytes_for_cleaner(self):
        """HiCacheNixl hands the GiB budget to the cleaner as bytes."""
        import inspect
        import sys
        import types

        if "nixl._api" not in sys.modules:
            # The native nixl package is only needed by CUDA CI; stub it here.
            stub = types.ModuleType("nixl._api")
            stub.nixl_agent = object
            stub.nixl_agent_config = object
            stub.nixlBind = types.SimpleNamespace(
                NIXL_THREAD_SYNC_RW=0, NIXL_THREAD_SYNC_STRICT=1
            )
            nixl_pkg = types.ModuleType("nixl")
            nixl_pkg._api = stub
            sys.modules.setdefault("nixl", nixl_pkg)
            sys.modules["nixl._api"] = stub

        from sglang.srt.mem_cache.storage.nixl import hicache_nixl

        source = inspect.getsource(hicache_nixl.HiCacheNixl.__init__)
        self.assertIn(
            'max_cache_bytes=cleaner_config["max_cache_gb"] * GIBIBYTE', source
        )
        self.assertEqual(GIBIBYTE, 1024**3)


def _scan_allocated(base_dirs) -> int:
    total = 0
    for base in base_dirs:
        for root, _dirs, files in os.walk(base):
            for name in files:
                total += _allocated_bytes(os.stat(os.path.join(root, name)))
    return total


if __name__ == "__main__":
    unittest.main()
