"""CPU-only tests for the interactive setup front end.

Run from anywhere with:  python3 scripts/pennyroyal/test_penny_setup.py

Fixtures are temporary; nothing is downloaded, installed, started, or deleted.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "pennyroyal"
sys.path.insert(0, str(SCRIPTS))

import penny_config as pc  # noqa: E402
import penny_setup as ps  # noqa: E402

# Same metadata shapes the qualified checkpoints ship (architectures plus
# model_type and text_config.model_type).
NEXT_ARCH = {"architectures": ["Qwen4ExpForConditionalGeneration"],
             "model_type": "qwen4_exp",
             "text_config": {"model_type": "qwen4_exp_text"}}
DENSE_ARCH = {"architectures": ["Qwen3_5ForConditionalGeneration"],
              "model_type": "qwen3_5",
              "text_config": {"model_type": "qwen3_5_text"}}


def make_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)


class FixtureMixin(unittest.TestCase):
    """One reusable layout: repo with stub recipes, venv, model dirs, caches.

    GPU discovery and the home directory are always mocked or redirected; no
    test touches the host GPU query or the real account's cache defaults.
    """

    def setUp(self) -> None:
        patcher = mock.patch.object(pc, "discover_gpus",
                                    return_value=([], "mocked: no GPUs"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.fake_home = self.base0 = None
        self._stack = context = __import__("contextlib").ExitStack()
        self.addCleanup(context.close)
        base = Path(self.mkdtemp())
        self.base = base
        self.repo = base / "repo"
        for recipe in pc.RECIPE_BY_PROFILE.values():
            stub = self.repo / "configs" / "pennyroyal" / recipe
            make_executable(stub, "#!/usr/bin/env bash\nexit 0\n")
        self.venv = base / "venv with space"
        make_executable(self.venv / "bin" / "sglang", "#!/usr/bin/env bash\nexit 0\n")
        make_executable(self.venv / "bin" / "python", "#!/usr/bin/env bash\nexit 0\n")
        self.models = base / "models root"
        self.next_model = self.models / "Flash Next [v1]"
        self.dense_model = self.models / "Qwen3.8-27B-FP8 (block)"
        self.draft_model = self.models / "DFlash2 $draft"
        self.custom_model = self.models / "custom-quant $'odd'"  # spaces, $ and quotes
        for directory, payload in (
                (self.next_model, NEXT_ARCH),
                (self.dense_model, DENSE_ARCH),
                (self.draft_model, DENSE_ARCH),
                (self.custom_model, {"architectures": ["CustomQuantForCausalLM"]}),
        ):
            directory.mkdir(parents=True)
            (directory / "config.json").write_text(json.dumps(payload))
        self.cache = base / "cache"
        self.nixl = base / "nixl"
        self.cache.mkdir()
        self.nixl.mkdir()
        self.config_path = base / "pennyroyal.env"

    def mkdtemp(self) -> Path:
        import tempfile
        return Path(self._stack.enter_context(
            tempfile.TemporaryDirectory(prefix="pennyroyal-test-")))


class GpuDiscoveryTests(unittest.TestCase):
    def test_present(self):
        gpus, note = pc.discover_gpus(runner=lambda: (
            "0, NVIDIA RTX PRO 6000\n1, NVIDIA Secondary\n"))
        self.assertEqual(note, "")
        self.assertEqual([gpu.index for gpu in gpus], ["0", "1"])
        self.assertIn("RTX PRO 6000", gpus[0].label())

    def test_absent_tool_falls_back_to_manual_input(self):
        with mock.patch("shutil.which", return_value=None):
            gpus, note = pc.discover_gpus()
        self.assertEqual(gpus, [])
        self.assertIn("nvidia-smi was not found", note)

    def test_failing_query_is_reported_not_raised(self):
        def boom() -> str:
            raise RuntimeError("driver busy")
        gpus, note = pc.discover_gpus(runner=boom)
        self.assertEqual(gpus, [])
        self.assertIn("driver busy", note)
        gpus, note = pc.discover_gpus(runner=lambda: "no gpu rows here")
        self.assertEqual(gpus, [])
        self.assertIn("manually", note)


class SetupSessionTests(FixtureMixin):
    """The wizard writes the same file the non-interactive path validates."""

    def run_setup(self, answers: list[str], existing: str | None = None,
                  config_path: Path | None = None,
                  home: Path | None = None) -> tuple[int, str, Path]:
        path = config_path or self.config_path
        if existing is not None:
            path.write_text(existing)
        out = io.StringIO()
        prompt = ps.Prompt(stdin=io.StringIO("".join(line + "\n" for line
                                                     in answers)),
                           stdout=out)
        code = ps.run_session("native", path, {}, prompt, self.repo,
                              home=home or self.base / "isolated-home")
        return code, out.getvalue(), path

    def basics(self, target: str, *, draft: str | None = None) -> list[str]:
        """Answers for REPO_ROOT, VENV_PATH, TARGET[_DRAFT], CACHE_BASE,
        NIXL_STORAGE_BASE, GPU, and the NIXL budget; defaults stay blank."""
        answers = ["", str(self.venv), target]
        if draft is not None:
            answers.append(draft)
        return answers + ["", str(self.nixl), "0", "0", ""]

    def test_full_flow_writes_a_file_the_launcher_accepts(self):
        code, output, path = self.run_setup(["next"] + self.basics(
            str(self.next_model)) + ["n", "y"])
        self.assertEqual(code, 0, output)
        self.assertIn("next command:", output)
        # Cache defaults came from the isolated home, never the real account.
        self.assertIn(str(self.base / "isolated-home"), output)
        self.assertNotIn(str(Path.home()),
                         output.split("next command:")[0])
        saved = pc.read_env_file(path)
        self.assertEqual(saved["TARGET_MODEL"], str(self.next_model))
        self.assertEqual(saved[pc.PROFILE_KEY], "next")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        plan = pc.build_plan("native", pc.load_config("native", path, {},
                                                      self.repo), {})
        self.assertEqual(plan.errors, [])

    def test_cancellation_and_eof_write_nothing(self):
        with self.assertRaises(ps.Cancelled):
            self.run_setup(["next", "q"])
        self.assertFalse(self.config_path.exists())
        with self.assertRaises(ps.Cancelled):
            self.run_setup(["next"])          # EOF at the first free answer
        self.assertFalse(self.config_path.exists())

    def test_no_confirmation_means_no_write(self):
        code, output, path = self.run_setup(["next"] + self.basics(
            str(self.next_model)) + ["n", "n"])
        self.assertEqual(code, 3, output)
        self.assertFalse(path.exists())
        self.assertIn("nothing was written", output)

    def test_rerun_loads_choices_and_preserves_unknown_keys(self):
        existing = (f"{pc.PROFILE_KEY}=next\n"
                    f"TARGET_MODEL={pc.quote_value(str(self.next_model))}\n"
                    f"CUSTOM_TOOL={pc.quote_value('keep $this # here')}\n"
                    "MAX_RUNNING_REQUESTS=6\n")
        code, output, path = self.run_setup(
            [""] + self.basics("") + ["n", "y"], existing=existing)
        self.assertEqual(code, 0, output)
        saved = pc.read_env_file(path)
        self.assertEqual(saved["CUSTOM_TOOL"], "keep $this # here")
        self.assertEqual(saved["MAX_RUNNING_REQUESTS"], "6")
        self.assertIn("unrecognized key kept", output)

    def test_existing_file_needs_consent_to_replace(self):
        # A valid saved file: the save gate passes and the replace confirmation
        # alone decides; declining leaves the file byte-for-byte untouched.
        existing = (f"{pc.PROFILE_KEY}=next\n"
                    f"TARGET_MODEL={pc.quote_value(str(self.next_model))}\n"
                    f"CACHE_BASE={pc.quote_value(str(self.cache))}\n"
                    f"NIXL_STORAGE_BASE={pc.quote_value(str(self.nixl))}\n")
        code, output, path = self.run_setup(
            ["", "", str(self.venv), "", "", "", "0", "0", "", "n", "n"],
            existing=existing)
        self.assertEqual(code, 3, output)
        self.assertIn("Replace", output)
        self.assertEqual(path.read_text(), existing)

    def test_invalid_answers_are_reasked_until_valid(self):
        code, output, path = self.run_setup(
            ["bogus-profile", "27b"] + self.basics(str(self.dense_model),
                                                   draft=str(self.draft_model))
            + ["n", "y"])
        self.assertEqual(code, 0, output)
        self.assertEqual(pc.read_env_file(path)[pc.PROFILE_KEY], "27b")
        self.assertIn("unknown profile", output)

    def test_written_file_survives_spaces_and_dollars(self):
        tricky = str(self.custom_model)
        code, output, path = self.run_setup(["next"] + self.basics(tricky)
                                             + ["n", "y"])
        self.assertEqual(code, 0, output)
        self.assertEqual(pc.read_env_file(path)["TARGET_MODEL"], tricky)

    # --- Review regression cases: validate before saving -------------------

    def test_wizard_custom_config_printed_command_reloads_that_file(self):
        # Fresh Sol's reproduction: ./configure-penny --config <custom> must
        # print run-penny --config <that file>, not a bare run-penny that would
        # read the user's default instead of the file just written. A relative
        # selection resolves once, so the printed command also works from /.
        custom_dir = self.base / "chosen"
        custom_dir.mkdir()
        custom = custom_dir / "user-selected.env"
        capture = self.base / "capture"
        # The printed command names <REPO_ROOT>/run-penny, so give the fixture
        # repo the real thin launcher and the real utility it delegates to;
        # this executes the shipped scripts, not a reimplementation.
        (self.repo / "scripts").symlink_to(SCRIPTS.parent)
        shutil.copy2(ROOT / "run-penny", self.repo / "run-penny")
        stub = self.repo / "configs/pennyroyal/serve-flash-next-frspec.sh"
        stub.write_text("#!/usr/bin/env bash\n"
                        'printf "argv=%s\\0" "$0" > "$CAPTURE"\n'
                        'printf "TARGET_MODEL=%s\\0" "${TARGET_MODEL-}" '
                        '>> "$CAPTURE"\n')
        stub.chmod(0o755)
        out = io.StringIO()
        prompt = ps.Prompt(stdin=io.StringIO("".join(
            line + "\n" for line in ["next"] + self.basics(
                str(self.next_model)) + ["n", "y"])), stdout=out)
        with contextlib.chdir(custom_dir):
            # Selected as a relative path from its own directory: the same
            # file must be what the wizard saves, names, and prints.
            code = ps.run_session("native", Path("user-selected.env"), {},
                                  prompt, self.repo,
                                  home=self.base / "isolated-home")
        output = out.getvalue()
        self.assertEqual(code, 0, output)
        self.assertTrue(custom.is_file())
        self.assertIn(str(custom), output)
        self.assertNotIn("user-selected.env\n", output.split("next command:")[0]
                         .replace(str(custom), ""))  # no bare relative echo
        printed = output.split("next command:")[1].strip().splitlines()[0]
        run = subprocess.run(printed, shell=True, capture_output=True,
                             text=True, check=False, cwd="/",
                             env=dict(os.environ, CAPTURE=str(capture)))
        self.assertEqual(run.returncode, 0, run.stderr)
        fields = dict(item.split("=", 1) for item in
                      capture.read_text().split("\0") if item)
        self.assertEqual(fields["argv"],
                         str(self.repo / "configs/pennyroyal"
                              / "serve-flash-next-frspec.sh"))
        self.assertEqual(fields["TARGET_MODEL"], str(self.next_model))

    def test_saved_blank_survives_a_rerun_as_an_explicit_suppression(self):
        # A blank saved on purpose must be written back as `KEY=`, so after a
        # rerun it still suppresses an inherited value instead of quietly
        # turning back into 'absent' (which would mean 'inherit').
        existing = (f"{pc.PROFILE_KEY}=next\n"
                    f"TARGET_MODEL={pc.quote_value(str(self.next_model))}\n"
                    f"CACHE_BASE={pc.quote_value(str(self.cache))}\n"
                    f"NIXL_STORAGE_BASE={pc.quote_value(str(self.nixl))}\n"
                    "MAX_RUNNING_REQUESTS=\n")
        code, output, path = self.run_setup(
            ["", "", str(self.venv), "", "", "", "0", "0", "", "n", "y"],
            existing=existing)
        self.assertEqual(code, 0, output)
        saved = pc.read_env_file(path)
        self.assertIn("MAX_RUNNING_REQUESTS", saved)
        self.assertEqual(saved["MAX_RUNNING_REQUESTS"], "")
        # ... and it still behaves as a suppression after reload.
        plan = pc.build_plan("native", pc.load_config("native", path, {},
                                                      self.repo),
                             {"MAX_RUNNING_REQUESTS": "8"},
                             repo_root=self.repo)
        self.assertEqual(plan.env["MAX_RUNNING_REQUESTS"], "")
        self.assertIn("blank", plan.origins["MAX_RUNNING_REQUESTS"])

    def test_new_optional_blank_stays_absent_so_inheritance_keeps_working(self):
        # The mirror case: an optional key never saved is written as absent,
        # and absence keeps meaning 'inherit from the environment'.
        code, output, path = self.run_setup(
            ["next"] + self.basics(str(self.next_model)) + ["n", "y"])
        self.assertEqual(code, 0, output)
        saved = pc.read_env_file(path)
        self.assertNotIn("MAX_RUNNING_REQUESTS", saved)
        plan = pc.build_plan("native", pc.load_config("native", path, {},
                                                      self.repo),
                             {"MAX_RUNNING_REQUESTS": "8"},
                             repo_root=self.repo)
        self.assertEqual(plan.env["MAX_RUNNING_REQUESTS"], "8")
        self.assertEqual(plan.origins["MAX_RUNNING_REQUESTS"],
                         "inherited environment")

    def test_invalid_proposal_is_not_saved_and_is_not_reported_ready(self):
        existing = (f"{pc.PROFILE_KEY}=next\n"
                    "TARGET_MODEL=/definitely-missing-penny-model\n")
        code, output, path = self.run_setup(
            ["next", "", str(self.venv), "", "", str(self.nixl), "0", "0", "",
             "n", "y"],
            existing=existing)
        self.assertEqual(code, 3, output)
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(), existing)
        self.assertIn("Not ready to save", output)
        self.assertIn("/definitely-missing-penny-model", output)
        self.assertNotIn("Saved ", output)
        self.assertNotIn("Next command:", output)

    def test_invalid_proposal_can_be_corrected_in_session(self):
        existing = (f"{pc.PROFILE_KEY}=next\n"
                    "TARGET_MODEL=/definitely-missing-penny-model\n")
        code, output, path = self.run_setup(
            ["next", "", str(self.venv), "", "", str(self.nixl), "0", "0", "",
             "n",
             "TARGET_MODEL", str(self.next_model),   # fix at the gate
             "y"],
            existing=existing)
        self.assertEqual(code, 0, output)
        self.assertEqual(pc.read_env_file(path)["TARGET_MODEL"],
                         str(self.next_model))

    def test_invalid_proposal_can_be_saved_anyway_after_warning(self):
        code, output, path = self.run_setup(
            ["next", "", str(self.venv), "/definitely-missing-penny-model", "",
             str(self.nixl), "0", "0", "", "n", "s", "y"])
        self.assertEqual(code, 0, output)
        self.assertIn("Saving despite the errors", output)
        self.assertIn("ERROR", output)  # still visible, never claimed ready

    def test_mixup_checkpoint_blocks_save_as_clear_error(self):
        # A dense checkpoint on the next profile is the parent's mixup case.
        code, output, path = self.run_setup(
            ["next", "", str(self.venv), str(self.dense_model), "",
             str(self.nixl), "0", "0", "", "n", ""])
        self.assertEqual(code, 3, output)
        self.assertIn("clear mixup", output)
        self.assertFalse(path.exists())




if __name__ == "__main__":
    unittest.main(verbosity=2)
