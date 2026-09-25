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

    def basics(self, target: str, *, draft: str | None = None,
               hicache: str = "") -> list[str]:
        """Answers for REPO_ROOT, VENV_PATH, TARGET[_DRAFT], CACHE_BASE,
        NIXL_STORAGE_BASE, GPU, the NIXL byte budget, the RAM (HiCache) size,
        and the media menu; defaults stay blank. hicache '' means 'keep the
        profile's recipe default'."""
        answers = ["", str(self.venv), target]
        if draft is not None:
            answers.append(draft)
        return answers + ["", str(self.nixl), "0", "0", hicache, ""]

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
            ["", "", str(self.venv), "", "", "", "0", "0", "", "", "n", "n"],
            existing=existing)
        self.assertEqual(code, 3, output)
        self.assertIn("Replace", output)
        self.assertEqual(path.read_text(), existing)

    def test_invalid_menu_selection_is_reasked_until_valid(self):
        # The numbered profile menu prints BEFORE the question; a junk answer
        # re-asks (no silent fallback) and selecting "3" picks 27b. The media
        # preprocessing row is a menu too: entering it gives cpu, never a
        # cuda id inferred from the host GPU index.
        code, output, path = self.run_setup(
            ["not-a-number", "3"] + self.basics(str(self.dense_model),
                                                draft=str(self.draft_model))
            + ["n", "y"])
        self.assertEqual(code, 0, output)
        self.assertEqual(pc.read_env_file(path)[pc.PROFILE_KEY], "27b")
        self.assertIn("Pick one of the listed numbers", output)
        menu = output.split("Model profile:")[1].split("Choice (")[0]
        self.assertLess(menu.index("[1] next"), menu.index("[3] 27b"))
        self.assertIn("Media preprocessing device", output)
        self.assertEqual(pc.read_env_file(path)["SGLANG_MM_PREPROCESS_DEVICE"],
                         "cpu")

    def test_enter_keeps_the_numbered_default_everywhere(self):
        # Blank lines at every menu keep the documented default: profile next,
        # media cpu, advanced section declined, save confirmed.
        code, output, path = self.run_setup(
            [""] + self.basics(str(self.next_model)) + ["", ""])
        self.assertEqual(code, 0, output)
        saved = pc.read_env_file(path)
        self.assertEqual(saved[pc.PROFILE_KEY], "next")
        self.assertEqual(saved["SGLANG_MM_PREPROCESS_DEVICE"], "cpu")
        self.assertNotIn("MAX_RUNNING_REQUESTS", saved)
        self.assertIn("Saved ", output)

    def test_gpu_menu_numbers_map_to_device_indexes_not_positions(self):
        # Choice [1] may mean GPU 0: the row text names the real index and the
        # saved value is that index, while Enter preserves the existing saved
        # GPU and the manual row accepts a UUID.
        with mock.patch.object(pc, "discover_gpus", return_value=(
                [pc.Gpu(index="0", name="NVIDIA RTX PRO 6000"),
                 pc.Gpu(index="1", name="NVIDIA Secondary")], "")):
            gpu_answers = self.basics(str(self.next_model))
            gpu_answers[5] = "2"          # menu row 2 = GPU index 1
            code, output, path = self.run_setup(
                ["next"] + gpu_answers + ["n", "y"])
            self.assertEqual(code, 0, output)
            self.assertIn("[1] GPU index 0 — NVIDIA RTX PRO 6000", output)
            self.assertIn("[2] GPU index 1 — NVIDIA Secondary", output)
            self.assertIn("[3] type another index or UUID", output)
            self.assertEqual(pc.read_env_file(path)["GPU"], "1")
            # Manual row accepts a UUID verbatim.
            gpu_answers = self.basics(str(self.next_model))
            gpu_answers[5] = "3"          # manual row, then the UUID
            code, output, path = self.run_setup(
                ["next"] + gpu_answers[:5] +
                ["3", "GPU-abcdef01-2345-6789-abcd-ef0123456789"] +
                gpu_answers[6:] + ["n", "y"])
            self.assertEqual(code, 0, output)
            self.assertEqual(
                pc.read_env_file(path)["GPU"],
                "GPU-abcdef01-2345-6789-abcd-ef0123456789")
        # With an existing GPU=1 saved, Enter on the menu preserves it.
        existing = (f"{pc.PROFILE_KEY}=next\n"
                    f"TARGET_MODEL={pc.quote_value(str(self.next_model))}\n"
                    f"CACHE_BASE={pc.quote_value(str(self.cache))}\n"
                    f"NIXL_STORAGE_BASE={pc.quote_value(str(self.nixl))}\n"
                    "GPU=1\n")
        with mock.patch.object(pc, "discover_gpus", return_value=(
                [pc.Gpu(index="0", name="NVIDIA RTX PRO 6000")], "")):
            gpu_answers = self.basics("")
            gpu_answers[5] = ""           # Enter keeps the saved GPU=1
            code, output, path = self.run_setup(
                [""] + gpu_answers + ["n", "y"], existing=existing)
            self.assertEqual(code, 0, output)
            # The saved GPU is not a listed menu row here (only index 0 is
            # visible), so Enter honestly keeps the current value.
            self.assertIn("Enter keeps the current value (1)", output)
            self.assertEqual(pc.read_env_file(path)["GPU"], "1")

    def test_container_mode_shows_the_same_numbered_menus_and_valid_file(self):
        # Config parity: the container wizard offers the identical numbered
        # menus (profile, fixed choices, yes/no gates) and its saved .env-style
        # file validates through the shared parser like the native one.
        host_root = self.base / "host models"
        host_root.mkdir()
        (self.base / "cache").mkdir(exist_ok=True)
        (self.base / "nixl").mkdir(exist_ok=True)
        # Give the fixture repo a compose file and the profile's default
        # target under the mount so the shared validator passes, exactly as a
        # real starter .env would.
        compose = self.repo / pc.COMPOSE_RELPATH
        compose.parent.mkdir(parents=True, exist_ok=True)
        compose.write_text("services: {}\n")
        (host_root / pc.CONTAINER_DEFAULT_TARGET["next"].removeprefix(
            "/models/")).mkdir(parents=True)
        names = [n for n in ps.ordered_names("container")
                 if n != "DRAFT_MODEL"]
        given = {"HOST_MODELS_ROOT": str(host_root),
                 "HOST_CACHE_BASE": str(self.base / "cache"),
                 "HOST_NIXL_STORAGE_BASE": str(self.base / "nixl")}
        # After the fields: the advanced gate (Enter keeps its default 'no')
        # and the save gate (Enter keeps its default 'yes').
        answers = ["1"] + [given.get(n, "") for n in names] + ["", ""]
        out = io.StringIO()
        prompt = ps.Prompt(stdin=io.StringIO("\n".join(answers) + "\n"),
                           stdout=out)
        code = ps.run_session("container", self.config_path, {}, prompt,
                              self.repo, home=self.base / "isolated-home")
        output = out.getvalue()
        self.assertEqual(code, 0, output)
        self.assertIn("Model profile:\n  [1] next", output)
        self.assertIn("Media preprocessing device (cpu or cuda:N):\n  [1] cpu",
                      output)
        self.assertIn("  [1] yes\n  [2] no", output)
        saved = pc.read_env_file(self.config_path)
        self.assertEqual(saved[pc.PROFILE_KEY], "next")
        self.assertEqual(saved["NVIDIA_GPU"], "0")
        plan = pc.build_plan("container", pc.load_config("container",
                                                         self.config_path, {},
                                                         self.repo), {},
                             repo_root=self.repo)
        self.assertEqual(plan.errors, [])
        # Cancel at the save gate writes nothing, same as native.
        cancel = self.base / "cancel.env"
        answers = (["1"] + [given.get(n, "") for n in names]
                   + ["", "2"])  # save gate: pick row 2 = no
        out2 = io.StringIO()
        code = ps.run_session("container", cancel, {}, ps.Prompt(
            stdin=io.StringIO("\n".join(answers) + "\n"), stdout=out2),
            self.repo, home=self.base / "isolated-home")
        self.assertEqual(code, 3, out2.getvalue())
        self.assertFalse(cancel.exists())

    def test_malformed_saved_gpu_is_named_not_crashed(self):
        # Parent's repro: saved GPU='bad gpu' + discovered GPU 0, then Enter.
        # The old code assigned the junk value straight from the menu and the
        # later _candidate_config validation raised an uncaught ConfigError.
        # Now the shared validator rejects the saved value (naming the rule),
        # Enter falls back to the documented default, and the file stays valid.
        existing = (f"{pc.PROFILE_KEY}=next\n"
                    f"TARGET_MODEL={pc.quote_value(str(self.next_model))}\n"
                    f"CACHE_BASE={pc.quote_value(str(self.cache))}\n"
                    f"NIXL_STORAGE_BASE={pc.quote_value(str(self.nixl))}\n"
                    f"GPU={pc.quote_value('bad gpu')}\n")
        with mock.patch.object(pc, "discover_gpus", return_value=(
                [pc.Gpu(index="0", name="NVIDIA RTX PRO 6000")], "")):
            answers = self.basics("")
            answers[5] = ""            # Enter at the GPU menu
            code, output, path = self.run_setup(
                [""] + answers + ["n", "y"], existing=existing)
            self.assertEqual(code, 0, output)
            self.assertIn("GPU may not contain spaces, quotes, or '$'", output)
            self.assertIn("Enter will use the default instead", output)
            self.assertEqual(pc.read_env_file(path)["GPU"], "0")
            # Retry path: the operator can still override with a typed value.
            answers = self.basics("")
            answers[5] = "2"           # manual row, then an explicit UUID
            code, output, path = self.run_setup(
                [""] + answers[:5] + ["2", "GPU-deadbeef-0000"]
                + answers[6:] + ["n", "y"], existing=existing)
            self.assertEqual(code, 0, output)
            self.assertEqual(pc.read_env_file(path)["GPU"],
                             "GPU-deadbeef-0000")
            # Cancellation at the GPU menu still writes nothing.
            stale = self.base / "stale-gpu.env"
            answers = self.basics("")
            answers[5] = "q"
            with self.assertRaises(ps.Cancelled):
                self.run_setup([""] + answers[:6] + ["q"], existing=existing,
                               config_path=stale)
            self.assertEqual(stale.read_text(), existing)

    def test_valid_off_list_saved_uuid_stays_the_enter_default(self):
        # The rejection above must not regress the preserved case: a valid
        # UUID of a device that is not in the discovered list is still a legal
        # saved value and Enter keeps it verbatim.
        existing = (f"{pc.PROFILE_KEY}=next\n"
                    f"TARGET_MODEL={pc.quote_value(str(self.next_model))}\n"
                    f"CACHE_BASE={pc.quote_value(str(self.cache))}\n"
                    f"NIXL_STORAGE_BASE={pc.quote_value(str(self.nixl))}\n"
                    "GPU=GPU-abcdef01-2345-6789-abcd-ef0123456789\n")
        with mock.patch.object(pc, "discover_gpus", return_value=(
                [pc.Gpu(index="0", name="NVIDIA RTX PRO 6000")], "")):
            answers = self.basics("")
            answers[5] = ""            # Enter
            code, output, path = self.run_setup(
                [""] + answers + ["n", "y"], existing=existing)
            self.assertEqual(code, 0, output)
            self.assertNotIn("may not contain", output)
            self.assertIn("Enter keeps the current value", output)
            self.assertEqual(
                pc.read_env_file(path)["GPU"],
                "GPU-abcdef01-2345-6789-abcd-ef0123456789")

    def test_yes_no_gates_are_numbered_and_cancel_without_writes(self):
        # The save gate is a menu too: choosing 'no' writes nothing, and the
        # advanced gate's default is 'no'.
        code, output, path = self.run_setup(
            ["next"] + self.basics(str(self.next_model)) + ["n", "2"])
        self.assertEqual(code, 3, output)
        self.assertFalse(path.exists())
        self.assertIn("nothing was written", output)
        self.assertIn("  [1] yes\n  [2] no", output)

    def test_media_menu_shows_cpu_model_gpu_saved_and_manual_rows(self):
        base = self.basics(str(self.next_model))
        # Menu rows carry human labels (never the __manual__ sentinel):
        # [1] cpu, [2] cuda:0 (the model's GPU), [3] another GPU. Selecting
        # the manual row asks for a NUMERIC logical index and stores cuda:N;
        # 'cuda:1' typed by hand also lands on cuda:1; junk is re-asked.
        code, output, path = self.run_setup(
            ["next"] + base[:8] + ["3", "junk", "1"] +
            ["y",                       # yes to the advanced section
             "",                        # PLE placement menu: Enter keeps ram
             "/nvme-unused-with-ram",   # free-text path stays free text
             "true",                    # online FP8 chosen from its menu
             "",                        # forward tools: Enter keeps true
             "", "", "", "",            # capacity/build jobs: Enter = blank
             "/opt/nixl",               # advanced free-text paths below...
             str(self.venv / "bin" / "sglang"),
             str(self.venv / "bin" / "python"),
             "y"])                       # ...then save
        self.assertEqual(code, 0, output)
        self.assertIn("[1] cpu — media preprocessing on the CPU", output)
        self.assertIn("[2] cuda:0 — GPU 0, the model's GPU", output)
        self.assertIn("[3] another GPU", output)
        self.assertNotIn("__manual__", output)
        self.assertIn("NOT the host GPU index", output)
        self.assertIn("must be cpu or cuda:N, got 'cuda:junk'", output)
        saved = pc.read_env_file(path)
        self.assertEqual(saved["SGLANG_MM_PREPROCESS_DEVICE"], "cuda:1")
        # A boolean advanced key is also a menu: [1]/'true' selected true.
        self.assertEqual(saved["SGLANG_SM120_ONLINE_MXFP8"], "true")
        self.assertIn("[1] true\n  [2] false", output)

    def test_media_menu_keeps_saved_device_and_cancels_without_writes(self):
        # A saved cuda:2 (beyond the fixed rows) gets its own row and Enter
        # keeps it; 'q' anywhere on the media flow cancels with no writes;
        # an edited-out junk value is rejected and re-asked, not kept.
        existing = (f"{pc.PROFILE_KEY}=next\n"
                    f"TARGET_MODEL={pc.quote_value(str(self.next_model))}\n"
                    f"CACHE_BASE={pc.quote_value(str(self.cache))}\n"
                    f"NIXL_STORAGE_BASE={pc.quote_value(str(self.nixl))}\n"
                    "SGLANG_MM_PREPROCESS_DEVICE=cuda:2\n")
        code, output, path = self.run_setup(
            [""] + self.basics("") + ["n", "y"], existing=existing)
        self.assertEqual(code, 0, output)
        self.assertIn("[3] cuda:2 — the device saved in this file", output)
        self.assertEqual(pc.read_env_file(path)["SGLANG_MM_PREPROCESS_DEVICE"],
                         "cuda:2")
        # 'q' at the media menu cancels through the shared _read contract and
        # the existing file stays byte-for-byte as it was.
        with self.assertRaises(ps.Cancelled):
            self.run_setup([""] + self.basics("")[:8] + ["q"],
                           existing=existing)
        self.assertEqual(path.read_text(), existing)
        # A junk saved value cannot be kept by Enter: the menu rejects and
        # re-asks, so the invalid value never reaches the file again (the
        # extra blank line feeds the second, now-valid, menu prompt).
        broken = existing.replace("cuda:2", "GPUs")
        # First Enter would keep the junk saved value; it is rejected, the
        # second Enter is refused outright, and only an explicit pick passes.
        code, output, path = self.run_setup(
            [""] + self.basics("")[:8] + ["", "", "1"] + ["n", "y"],
            existing=broken)
        self.assertEqual(code, 0, output)
        self.assertIn("must be cpu or cuda:N, got 'GPUs'", output)
        self.assertEqual(pc.read_env_file(path)["SGLANG_MM_PREPROCESS_DEVICE"],
                         "cpu")

    def test_saved_container_prefixed_profile_preselects_the_right_row(self):
        # Older files may hold PENNYROYAL_PROFILE=container:next. The wizard
        # normalizes it through validate_profile BEFORE the menu default and
        # the profile-target lookup, and saves the canonical name.
        existing = (f"{pc.PROFILE_KEY}=container:next\n"
                    f"TARGET_MODEL={pc.quote_value(str(self.next_model))}\n"
                    f"CACHE_BASE={pc.quote_value(str(self.cache))}\n"
                    f"NIXL_STORAGE_BASE={pc.quote_value(str(self.nixl))}\n")
        code, output, path = self.run_setup(
            [""] + self.basics("") + ["n", "y"], existing=existing)
        self.assertEqual(code, 0, output)
        self.assertIn("Enter keeps [1] next", output)
        self.assertEqual(pc.read_env_file(path)[pc.PROFILE_KEY], "next")

    def test_invalid_saved_profile_forces_a_choice_instead_of_crashing(self):
        # Enter must not return the junk saved value or crash the later
        # CONTAINER_DEFAULT_TARGET lookup: the wizard explains, re-asks, and
        # only proceeds after a valid selection.
        existing = f"{pc.PROFILE_KEY}=container:weird\n"
        # First Enter returns the menu's stated fallback ('next' here is not
        # yet a valid saved value, so choose returns it and validation runs);
        # the re-ask consumes the second answer '1' before the fields start.
        code, output, path = self.run_setup(
            ["1"] + self.basics(str(self.next_model)) + ["n", "y"],
            existing=existing)
        self.assertEqual(code, 0, output)
        self.assertIn("Saved profile is unusable", output)
        self.assertIn("unknown profile", output)
        self.assertEqual(pc.read_env_file(path)[pc.PROFILE_KEY], "next")
        # Cancelling at the re-ask writes nothing.
        stale = self.base / "stale.env"
        stale.write_text(existing)
        with self.assertRaises(ps.Cancelled):
            self.run_setup(["q"], existing=existing, config_path=stale)
        self.assertEqual(stale.read_text(), existing)

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
            ["", "", str(self.venv), "", "", "", "0", "0", "", "", "n", "y"],
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
        # The validation gate is a numbered menu; picking the cancel row
        # (Enter keeps it) leaves the file byte-for-byte untouched.
        code, output, path = self.run_setup(
            ["next"] + self.basics("") + ["n", ""],
            existing=existing)
        self.assertEqual(code, 3, output)
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(), existing)
        self.assertIn("Not ready to save", output)
        self.assertIn("/definitely-missing-penny-model", output)
        self.assertIn("[1] re-enter TARGET_MODEL", output)
        self.assertIn("[2] save despite these errors", output)
        self.assertIn("[3] cancel; nothing is written", output)
        self.assertNotIn("Saved ", output)
        self.assertNotIn("Next command:", output)

    def test_invalid_proposal_can_be_corrected_in_session(self):
        existing = (f"{pc.PROFILE_KEY}=next\n"
                    "TARGET_MODEL=/definitely-missing-penny-model\n")
        code, output, path = self.run_setup(
            ["next"] + self.basics("") + ["n",
             "1", str(self.next_model),   # menu row: re-enter TARGET_MODEL
             "y"],
            existing=existing)
        self.assertEqual(code, 0, output)
        self.assertEqual(pc.read_env_file(path)["TARGET_MODEL"],
                         str(self.next_model))

    def test_invalid_proposal_can_be_saved_anyway_after_warning(self):
        code, output, path = self.run_setup(
            ["next"] + self.basics("/definitely-missing-penny-model") +
            ["n", "2", "y"])  # row 2 = save anyway
        self.assertEqual(code, 0, output)
        self.assertIn("Saving despite the errors", output)
        self.assertIn("ERROR", output)  # still visible, never claimed ready

    def test_normal_setup_omits_the_installation_plumbing_questions(self):
        # A first run asks about models, caches, and the GPU only: the sglang
        # and python programs, the image, the runtime identity, the NIXL
        # prefix, and the compile jobs wait for the advanced section.
        code, output, path = self.run_setup(
            ["next"] + self.basics(str(self.next_model)) + ["n", "y"])
        self.assertEqual(code, 0, output)
        for absent in ("SGLANG_EXE", "PYTHON", "PENNYROYAL_IMAGE",
                       "USER_ID", "GROUP_ID", "NIXL_PREFIX",
                       "PENNY_BUILD_JOBS"):
            self.assertNotIn(absent, output.split("Review")[0])
        self.assertIn("Where your Pennyroyal folder is", output)
        self.assertIn("Your Python environment folder", output)
        saved = pc.read_env_file(path)
        self.assertNotIn("SGLANG_EXE", saved)

    def test_labels_keep_ram_gigabytes_disk_gibibytes_and_compiled_cache_apart(self):
        # Three different 'caches' exist and the wizard never calls them the
        # same thing: the RAM (HiCache) tier is decimal GB (1e9 bytes), the
        # NIXL budget is a GiB disk cap whose 0 means unlimited, and the
        # compiled-cache folder is neither.
        code, output, path = self.run_setup(
            ["next"] + self.basics(str(self.next_model)) + ["n", "y"])
        self.assertEqual(code, 0, output)
        # The configurator says what it is: an early BETA, not a finished UI.
        self.assertIn("Pennyroyal setup (BETA)", output)
        self.assertIn("separate from the PLE embedding table", output)
        asked = output.split("Review")[0]
        ram = asked.split("RAM (HiCache) cache size in GB")[1].split("\n")[0]
        self.assertIn("1e9 bytes, not GiB", asked)
        self.assertIn("blank = the recipe's default", ram)
        self.assertIn("Disk budget for the persistent NIXL cache, in GiB",
                      asked)
        self.assertIn("0 = unlimited budget, not an off switch", asked)
        self.assertIn("Folder for the compiled and runtime caches", asked)
        self.assertIn("not the RAM model cache", asked)
        self.assertIn("separate from the PLE embedding table", asked)

    def test_small_hicache_size_is_saved_and_only_junk_is_reasked(self):
        # 1 GB is a real choice (the pool code warns about a smaller-than-device
        # pool instead of clamping), while 0/negative/fractional/nonnumeric is
        # re-asked without ever losing the blank default.
        for size in ("1", "2"):
            code, output, path = self.run_setup(
                ["next"] + self.basics(str(self.next_model), hicache=size)
                + ["n", "y"])
            self.assertEqual(code, 0, output)
            saved = pc.read_env_file(path)
            self.assertEqual(saved["PENNY_HICACHE_SIZE_GB"], size)
            plan = pc.build_plan("native", pc.load_config("native", path, {},
                                                          self.repo), {},
                                 repo_root=self.repo)
            self.assertEqual(plan.env["PENNY_HICACHE_SIZE_GB"], size)
        fields = self.basics(str(self.next_model))
        code, output, path = self.run_setup(
            ["next"] + fields[:7]
            + ["0", "-1", "1.5", "two", ""]   # junk is re-asked, Enter = blank
            + fields[8:] + ["n", "y"],
            config_path=self.base / "junk-hicache.env")
        self.assertEqual(code, 0, output)
        for junk in ("0", "-1", "1.5", "two"):
            self.assertIn(f"must be a positive integer, got '{junk}'", output)
        # Enter after the re-asks is still the documented blank: a never-saved
        # optional key stays absent from the file, which means 'the recipe
        # decides' and exports nothing to the recipe.
        self.assertNotIn("PENNY_HICACHE_SIZE_GB", pc.read_env_file(path))
        plan = pc.build_plan("native", pc.load_config("native", path, {},
                                                      self.repo), {},
                             repo_root=self.repo)
        self.assertNotIn("PENNY_HICACHE_SIZE_GB", plan.env)
        self.assertIn("RAM cache (HiCache): 32 GB as --hicache-size",
                      "\n".join(plan.summary))

    def test_saved_installation_overrides_survive_a_normal_run_and_advance_next(self):
        # Honouring existing saved overrides: an advanced key the declined
        # section never asked about stays byte-identical, and when the operator
        # does open the advanced section the saved value is the offered default.
        existing = (f"{pc.PROFILE_KEY}=next\n"
                    f"TARGET_MODEL={pc.quote_value(str(self.next_model))}\n"
                    f"CACHE_BASE={pc.quote_value(str(self.cache))}\n"
                    f"NIXL_STORAGE_BASE={pc.quote_value(str(self.nixl))}\n"
                    "PENNY_HICACHE_SIZE_GB=2\n"
                    "PENNY_BUILD_JOBS=8\n")
        code, output, path = self.run_setup(
            [""] + self.basics("") + ["n", "y"], existing=existing)
        self.assertEqual(code, 0, output)
        saved = pc.read_env_file(path)
        self.assertEqual(saved["PENNY_HICACHE_SIZE_GB"], "2")
        self.assertEqual(saved["PENNY_BUILD_JOBS"], "8")
        # Enter on the shown RAM size keeps the saved 2 GB.
        self.assertIn("variable PENNY_HICACHE_SIZE_GB", output)
        # Opening the advanced section offers the saved jobs count as default.
        # Eleven answers cover the advanced keys (two menus, the capacity
        # knobs, the install paths); every one of them presses Enter.
        code, output, path = self.run_setup(
            [""] + self.basics("") + ["y",
                                      "",                       # PLE menu: ram
                                      "/nvme-unused-with-ram",  # snapshot path
                                      "false",                  # online FP8
                                      "true",                   # unknown tools
                                      "", "", "", "",           # capacity + jobs
                                      "/opt/nixl",              # NIXL prefix
                                      str(self.venv / "bin" / "sglang"),
                                      str(self.venv / "bin" / "python")] + ["y"],
            existing=existing)
        self.assertEqual(code, 0, output)
        self.assertIn("Build jobs for first-start compilation (empty = recipe "
                      "default) [8]", output)
        self.assertEqual(pc.read_env_file(path)["PENNY_BUILD_JOBS"], "8")

    def test_mixup_checkpoint_blocks_save_as_clear_error(self):
        # A dense checkpoint on the next profile is the parent's mixup case.
        code, output, path = self.run_setup(
            ["next"] + self.basics(str(self.dense_model)) +
            ["n", ""])  # Enter keeps the cancel row
        self.assertEqual(code, 3, output)
        self.assertIn("clear mixup", output)
        self.assertFalse(path.exists())




if __name__ == "__main__":
    unittest.main(verbosity=2)
