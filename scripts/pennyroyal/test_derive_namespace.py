import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("derive_namespace.py")
SPEC = importlib.util.spec_from_file_location("derive_namespace", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_checkpoint(path: Path, marker: str) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"marker": marker}))
    header = json.dumps({"tensor": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (path / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0")


class NamespaceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "Test"], check=True)
        (self.repo / "tracked").write_text("one")
        subprocess.run(["git", "-C", str(self.repo), "add", "tracked"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "base"], check=True)
        self.target = self.root / "target"
        self.draft = self.root / "draft"
        make_checkpoint(self.target, "target")
        make_checkpoint(self.draft, "draft")

    def tearDown(self):
        self.temp.cleanup()

    def derive(self, *extra: str, dry_run: bool = False) -> Path:
        args = [
            str(SCRIPT),
            "--base-root",
            str(self.root / "cache"),
            "--slug",
            "Qwen 3.8 / DFlash2",
            "--git-repo",
            str(self.repo),
            "--model",
            f"target={self.target}",
            "--model",
            f"draft={self.draft}",
            "--field",
            "layout=page_first",
            *extra,
        ]
        if dry_run:
            args.append("--dry-run")
        return Path(subprocess.check_output(args, text=True).strip())

    def test_identical_identity_reuses_namespace_and_manifest(self):
        first = self.derive()
        second = self.derive()
        self.assertEqual(first, second)
        manifest = json.loads((first / MODULE.MANIFEST_NAME).read_text())
        self.assertEqual(manifest["identity"]["fields"]["layout"], "page_first")

    def test_representation_change_selects_independent_namespace(self):
        first = self.derive()
        changed = self.derive("--field", "page_size=128")
        self.assertNotEqual(first, changed)
        self.assertTrue(first.exists())
        self.assertTrue(changed.exists())

    def test_checkpoint_and_runtime_changes_select_new_namespaces(self):
        first = self.derive()
        os.utime(self.draft / "model.safetensors", None)
        changed_model = self.derive()
        self.assertNotEqual(first, changed_model)
        (self.repo / "tracked").write_text("two")
        changed_runtime = self.derive()
        self.assertNotEqual(changed_model, changed_runtime)

    def test_payload_change_with_preserved_stat_selects_new_namespace(self):
        first = self.derive()
        weight = self.draft / "model.safetensors"
        before = weight.stat()
        payload = bytearray(weight.read_bytes())
        payload[-1] ^= 1
        weight.write_bytes(payload)
        os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns))
        changed = self.derive()
        self.assertNotEqual(first, changed)

    def test_dry_run_does_not_create_namespace(self):
        root = self.derive(dry_run=True)
        self.assertFalse(root.exists())

    def test_duplicate_fields_fail(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.derive("--field", "layout=layer_first")

    def test_mismatched_existing_manifest_fails_closed(self):
        root = self.derive()
        manifest_path = root / MODULE.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["identity"]["fields"]["layout"] = "corrupt"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(subprocess.CalledProcessError):
            self.derive()


if __name__ == "__main__":
    unittest.main()
