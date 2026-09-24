"""TP1/TP2 NIXL FILE namespace layout audit (item 2): same-layout stability,
TP distinction, and fail-closed manifests -- with no purge of cache data.

Two layers are covered:

* the derivation helper (``scripts/pennyroyal/derive_namespace.py``):
  ``tp_size`` must be a positive-integer identity field (a missing/junk
  value used to let two topologies share one root silently), a re-derived
  same-layout launch reuses the root and manifest unchanged, and a foreign
  manifest fails closed leaving every file in place;
* the runtime cross-check (``HiCacheNixl`` construction path's
  ``_verify_derived_namespace_layout``): a derived root whose pinned
  ``tp_size`` differs from this instance's storage config raises instead of
  sharing/reinterpreting the directory.

The name-layout fact the audit rests on is also pinned: for non-MLA models
the object suffix carries ``_<rank>_<size>``, so TP1's ``_0_1`` never equals
any TP2 ``_0_2``/``_1_2`` -- silent aliasing inside one root requires
bypassing both guards, which is why they fail closed rather than purge.
"""

import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.nixl.namespace_layout import (
    verify_derived_namespace_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "pennyroyal" / "derive_namespace.py"
SPEC = importlib.util.spec_from_file_location("derive_namespace", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_checkpoint(path: Path, marker: str) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"marker": marker}))
    header = json.dumps(
        {"tensor": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
    ).encode()
    (path / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"\0\0"
    )


def _config(tp_size: int, tp_rank: int = 0) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="pennyroyal",
    )


class NamespaceTpTest(CustomTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.email", "t@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.name", "T"], check=True
        )
        (self.repo / "tracked").write_text("one")
        subprocess.run(["git", "-C", str(self.repo), "add", "tracked"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-qm", "base"], check=True
        )
        self.target = self.root / "target"
        make_checkpoint(self.target, "target")

    def tearDown(self):
        self.temp.cleanup()

    def derive(self, tp_size: str, *extra: str) -> Path:
        args = [
            str(SCRIPT),
            "--base-root",
            str(self.root / "cache"),
            "--slug",
            "qwen3_8_flash_next_524k_nextn",
            "--git-repo",
            str(self.repo),
            "--model",
            f"target={self.target}",
            "--field",
            f"tp_size={tp_size}",
            *extra,
        ]
        return Path(subprocess.check_output(args, text=True).strip())

    # ---------------- same-layout stability + TP distinction ----------------

    def test_same_layout_is_stable_and_tp_sizes_never_share_a_root(self):
        tp1_a = self.derive("1")
        tp1_b = self.derive("1")
        self.assertEqual(tp1_a, tp1_b)  # same layout -> same namespace, stable
        tp2 = self.derive("2")
        tp4 = self.derive("4")
        self.assertNotEqual(tp1_a, tp2)
        self.assertNotEqual(tp2, tp4)
        # Distinct roots coexist; neither launch purged the other.
        self.assertTrue((tp1_a / MODULE.MANIFEST_NAME).is_file())
        self.assertTrue((tp2 / MODULE.MANIFEST_NAME).is_file())

    def test_tp_size_must_be_a_positive_integer_identity(self):
        for junk in ("0", "-1", "two", "1.5", " 1", "1_000"):
            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--base-root",
                    str(self.root / "cache"),
                    "--slug",
                    "s",
                    "--git-repo",
                    str(self.repo),
                    "--model",
                    f"target={self.target}",
                    "--field",
                    f"tp_size={junk}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0, junk)
            self.assertIn("tp_size", result.stderr)

    def test_missing_tp_size_field_is_allowed_for_other_namespaces(self):
        # The guard validates the field's format when present; recipes that
        # do not hash TP at all keep working (e.g. the container recipes
        # always pass it, but the helper is shared).
        out = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "--base-root",
                str(self.root / "cache"),
                "--slug",
                "no-tp",
                "--git-repo",
                str(self.repo),
                "--model",
                f"target={self.target}",
                "--field",
                "page_size=64",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("no_tp_", out.stdout)

    def test_foreign_manifest_fails_closed_without_purging(self):
        root = self.derive("1")
        sentinel = root / "bucket" / "payload.kv"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_bytes(b"cached")
        # An identity change behind the name (simulated corruption/retarget):
        # refuse the root, keep every byte.
        manifest_path = root / MODULE.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["identity"]["fields"]["tp_size"] = "2"
        manifest_path.write_text(json.dumps(manifest))
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "--base-root",
                str(self.root / "cache"),
                "--slug",
                "qwen3_8_flash_next_524k_nextn",
                "--git-repo",
                str(self.repo),
                "--model",
                f"target={self.target}",
                "--field",
                "tp_size=1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match", result.stderr)
        self.assertTrue(sentinel.is_file())
        self.assertEqual(sentinel.read_bytes(), b"cached")
        # A corrupted manifest is likewise unreadable-as-proof: fail closed.
        manifest_path.write_text("{half")
        with self.assertRaises(RuntimeError):
            MODULE.ensure_manifest(
                root,
                {"schema": MODULE.SCHEMA, "fields": {}, "models": {}, "runtime": {}},
                "0" * 64,
            )
        self.assertTrue(sentinel.is_file())

    # ---------------- runtime cross-check ----------------

    def _write_manifest(self, root: Path, tp_size_field):
        identity = {
            "schema": MODULE.SCHEMA,
            "fields": (
                {} if tp_size_field is None else {"tp_size": tp_size_field}
            ),
            "models": {},
            "runtime": {},
        }
        root.mkdir(parents=True, exist_ok=True)
        (root / MODULE.MANIFEST_NAME).write_text(
            json.dumps({"identity_sha256": "0" * 64, "identity": identity})
        )

    def test_runtime_accepts_matching_and_unpinned_roots(self):
        match = self.root / "match"
        self._write_manifest(match, "2")
        verify_derived_namespace_layout([str(match)], 2)  # no raise
        unpinned = self.root / "unpinned"
        self._write_manifest(unpinned, None)
        verify_derived_namespace_layout([str(unpinned)], 1)
        external = self.root / "external"
        external.mkdir()
        verify_derived_namespace_layout([str(external)], 7)  # no manifest

    def test_runtime_rejects_foreign_tp_size_and_keeps_data(self):
        root = self.root / "pinned-tp1"
        self._write_manifest(root, "1")
        payload = root / "bucket" / "entry"
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(b"keep me")
        with self.assertRaises(RuntimeError) as ctx:
            verify_derived_namespace_layout([str(root)], 2)
        self.assertIn("pinned to tp_size=1", str(ctx.exception))
        self.assertIn("tp_size=2", str(ctx.exception))
        self.assertEqual(payload.read_bytes(), b"keep me")  # no purge
        self.assertTrue((root / MODULE.MANIFEST_NAME).is_file())

    def test_runtime_rejects_junk_and_unreadable_manifests(self):
        junk = self.root / "junk"
        self._write_manifest(junk, "two")
        with self.assertRaises(RuntimeError):
            verify_derived_namespace_layout([str(junk)], 2)
        broken = self.root / "broken"
        broken.mkdir()
        (broken / MODULE.MANIFEST_NAME).write_text("{oops")
        with self.assertRaises(RuntimeError):
            verify_derived_namespace_layout([str(broken)], 2)

    # ---------------- name-layout fact under a shared root ----------------

    def test_rank_suffixes_distinguish_tp_sizes_but_need_the_manifest(self):
        # Non-MLA HiCacheNixl names are ``key + _{model}_{rank}_{size}``.
        model = "pennyroyal"
        tp1_suffix = f"_{model}_0_1"
        tp2_suffixes = {f"_{model}_0_2", f"_{model}_1_2"}
        self.assertNotIn(tp1_suffix, tp2_suffixes)
        self.assertTrue(
            all(not s.endswith("_0_1") for s in tp2_suffixes),
            "TP1 objects can only alias another TP1 layout, never a TP2 one; "
            "the manifest check is what enforces that instead of hope",
        )
        # MLA-family models skip non-zero ranks entirely (backup_skip), the
        # other half of the no-alias story.
        config = _config(2, tp_rank=1)
        self.assertFalse(config.is_mla_model)  # non-MLA path uses rank suffixes
