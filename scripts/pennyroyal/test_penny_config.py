"""CPU-only tests for the saved Pennyroyal configuration and its entry points.

Run from anywhere with:  python3 scripts/pennyroyal/test_penny_config.py

Everything here uses temporary fixtures; no live caches, models, GPUs, Docker,
network, or package installs are touched.
"""

from __future__ import annotations

import ast
import json
import math
import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "pennyroyal"
sys.path.insert(0, str(SCRIPTS))

import penny_config as pc  # noqa: E402

# Metadata shapes copied from the qualified checkpoints on the parent host.
NEXT_ARCH = {"architectures": ["Qwen4ExpForConditionalGeneration"],
             "model_type": "qwen4_exp",
             "text_config": {"model_type": "qwen4_exp_text"}}
DENSE_ARCH = {"architectures": ["Qwen3_5ForConditionalGeneration"],
              "model_type": "qwen3_5",
              "text_config": {"model_type": "qwen3_5_text"}}


def _decode_compose_scalar(text: str) -> str:
    """Undo docker compose's own YAML/quoting in `config` output.

    Compose re-escapes a literal $ as $$ and doubles apostrophes when it prints
    an environment value, which is output formatting rather than the stored
    value; the .env parse must still be the value we wrote.
    """
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        body = text[1:-1]
        if text[0] == "'":
            body = body.replace("''", "'")
        return body.replace("$$", "$")
    return text.replace("$$", "$")


def compose_command() -> list[str] | None:
    """Real Compose CLI when present; None when absent (no installs, no error).

    The parent host has no docker CLI at all, so a missing executable must
    mean 'unavailable' rather than an exception: FileNotFoundError marks the
    candidate dead and the search continues (docker compose, then the
    standalone docker-compose binary) before returning None, which makes the
    dependent tests skip cleanly there while the real renderer checks still
    run on hosts that have Compose.
    """
    for candidate in (["docker", "compose"], ["docker-compose"]):
        try:
            probe = subprocess.run(candidate + ["version"],
                                   capture_output=True, text=True, check=False)
        except (FileNotFoundError, OSError):
            continue
        if probe.returncode == 0:
            return candidate
    return None


def make_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)


class FixtureMixin(unittest.TestCase):
    """One reusable layout: repo with stub recipes, venv, model dirs, caches."""

    def setUp(self) -> None:
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
        self.custom_model = self.models / "custom-quant $'odd'"
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

    def native_env(self, **overrides: str) -> dict[str, str]:
        env = {
            "REPO_ROOT": str(self.repo),
            "VENV_PATH": str(self.venv),
            "TARGET_MODEL": str(self.next_model),
            "CACHE_BASE": str(self.cache),
            "NIXL_STORAGE_BASE": str(self.nixl),
        }
        env.update({key: value for key, value in overrides.items()
                    if value is not None})
        for key, value in list(env.items()):
            if value == "":
                del env[key]
        return env

    def write_native_config(self, values: dict[str, str],
                            profile: str = "next") -> Path:
        body = pc.serialize_env(
            [("basic", [(key, value) for key, value in values.items()])],
            header=(f"{pc.PROFILE_KEY}={pc.quote_value(profile)}",))
        self.config_path.write_text(body)
        return self.config_path

    def native_plan(self, values: dict[str, str], profile: str = "next",
                    environ: dict[str, str] | None = None) -> pc.Plan:
        path = self.write_native_config(values, profile=profile)
        config = pc.load_config("native", path, environ or {}, self.repo)
        return pc.build_plan("native", config, environ or {})

    def messages(self, plan: pc.Plan, level: str) -> str:
        return "\n".join(issue.message for issue in plan.issues
                         if issue.level == level)


class ParsingTests(unittest.TestCase):
    def test_parses_as_data_without_shell_execution(self):
        parsed = pc.parse_env_text(
            "# comment\n"
            "PLAIN=value\n"
            "export EXPORTED=yes\n"
            "QUOTED='keep $HOME and #hash'\n"
            'DOLLAR="literal $VAR"\n'
            "TRAILING=value # inline comment\n")
        self.assertEqual(parsed["PLAIN"], "value")
        self.assertEqual(parsed["EXPORTED"], "yes")
        self.assertEqual(parsed["QUOTED"], "keep $HOME and #hash")
        self.assertEqual(parsed["DOLLAR"], "literal $VAR")
        self.assertEqual(parsed["TRAILING"], "value")
        self.assertNotIn("comment", parsed)

    def test_invalid_syntax_and_duplicates_are_rejected(self):
        with self.assertRaises(pc.ConfigError):
            pc.parse_env_text("not_an_assignment")
        with self.assertRaises(pc.ConfigError):
            pc.parse_env_text("A=1\nA=2")
        with self.assertRaises(pc.ConfigError):
            pc.parse_env_text("9BAD=1")

    def test_quotes_and_dollars_survive_a_save_load_round_trip(self):
        values = {
            "SPACED_PATH": "/models/Flash Next [v1]",
            "DOLLAR": "/models/$cost $(whoami) 'quoted' `tick` group",
            "BACKSLASH": "C:\\path\\to",
            "HASH": "value # not a comment",
            "EMPTY": "",
        }
        body = pc.serialize_env([("", list(values.items()))])
        self.assertEqual(pc.parse_env_text(body), values)

    def test_apostrophe_values_survive_both_readers(self):
        values = {
            "AP": "it's a $5 [path] #1",
            "BS": "win C:\\path\\to",
            "DQ": 'quote " inside and $HOME',
        }
        body = pc.serialize_env([("", list(values.items()))])
        self.assertEqual(pc.parse_env_text(body), values)
        compose = compose_command()
        if compose is None:
            self.skipTest("docker compose CLI not installed on this host")
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env").write_text(body)
            (root / "docker-compose.yaml").write_text(
                "services:\n  p:\n    image: busybox\n    environment:\n"
                + "".join(f"      {k}: ${{{k}:-}}\n" for k in values))
            env = {k: v for k, v in os.environ.items()
                   if k not in values and k != "HOME"}
            env["HOME"] = "/SHOULD_NOT_APPEAR"
            run = subprocess.run(compose + ["--env-file", str(root / ".env"),
                                            "-f", str(root / "docker-compose.yaml"),
                                            "config"],
                                 capture_output=True, text=True, env=env,
                                 check=False)
            self.assertEqual(run.returncode, 0, run.stderr)
        produced = {}
        for line in run.stdout.splitlines():
            stripped = line.strip()
            for key in values:
                if stripped.startswith(f"{key}: "):
                    produced[key] = _decode_compose_scalar(stripped[len(key) + 2:])
        self.assertEqual(produced, values)

    def test_validation_accepts_documented_forms(self):
        specs = pc.specs_for("native")
        self.assertEqual(pc.validate_value(specs["SGLANG_SM120_ONLINE_MXFP8"],
                                           "YES", "t"), "true")
        self.assertEqual(pc.validate_value(specs["SGLANG_MM_PREPROCESS_DEVICE"],
                                           "cuda:1", "t"), "cuda:1")
        for bad in ("cuda", "gpu0", "cuda:x"):
            with self.assertRaises(pc.ConfigError):
                pc.validate_value(specs["SGLANG_MM_PREPROCESS_DEVICE"], bad, "t")
        for bad in ("0", "-1", "abc", "1.5"):
            with self.assertRaises(pc.ConfigError):
                pc.validate_value(specs["MAX_RUNNING_REQUESTS"], bad, "t")
        self.assertEqual(pc.validate_value(
            pc.specs_for("native")["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"], "0.5", "t"),
            "0.5")
        for bad in ("-1", "12abc"):
            with self.assertRaises(pc.ConfigError):
                pc.validate_value(
                    pc.specs_for("native")["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"],
                    bad, "t")


class DiscoveryTests(FixtureMixin):
    def test_explicit_config_wins_over_environment_and_default(self):
        explicit = self.base / "explicit.env"
        with mock.patch.dict(os.environ, {"PENNYROYAL_CONFIG": "/tmp/from-env"}):
            found, origin = pc.discover_config_path("native", {
                "PENNYROYAL_CONFIG": "/tmp/from-env"}, self.repo, explicit)
        self.assertEqual((found, origin), (explicit, "--config"))
        found, origin = pc.discover_config_path("native", {
            "PENNYROYAL_CONFIG": "/tmp/from-env"}, self.repo)
        self.assertEqual((str(found), origin), ("/tmp/from-env",
                                                "PENNYROYAL_CONFIG"))
        found, origin = pc.discover_config_path("native", {}, self.repo)
        self.assertEqual(origin, "default location")
        self.assertEqual(found.name, "pennyroyal.env")
        container, _ = pc.discover_config_path("container", {}, self.repo)
        self.assertEqual(container, self.repo / pc.COMPOSE_RELPATH.replace(
            "compose.yaml", ".env"))

    def test_missing_file_uses_defaults_instead_of_failing(self):
        config = pc.load_config("native", self.base / "nope.env", {}, self.repo)
        self.assertEqual(config.profile, "next")
        self.assertEqual(config.values, {})

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(pc.ConfigError):
            pc.validate_profile("27b-pro-max", "native")
        self.assertEqual(pc.validate_profile("container:27b", "container"), "27b")


def runtime_max_cache_gb_parser():
    """The ACTUAL runtime budget parser, loaded without sglang's heavy deps.

    python/.../nixl_utils.py imports sglang (orjson etc.), which a no-install
    CPU test must not require, so extract just this pure function from the
    shipped source and execute it. This keeps the assertion honest: plan
    values are judged by the code the server really uses, not a lookalike.
    """
    source = (ROOT / "python/sglang/srt/mem_cache/storage/nixl"
              / "nixl_utils.py").read_text()
    tree = ast.parse(source)
    fn = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef)
              and node.name == "_parse_max_cache_gb")
    from typing import Any  # stdlib; satisfies the function's own annotations
    namespace = {"math": math, "Any": Any}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "nixl_utils.py",
                 "exec"), namespace)
    return namespace["_parse_max_cache_gb"]


class NativePlanTests(FixtureMixin):
    def test_plan_uses_saved_values_and_recipe_path(self):
        plan = self.native_plan(self.native_env(GPU="1"))
        self.assertEqual(plan.argv,
                         [str(self.repo / "configs/pennyroyal/"
                                  "serve-flash-next-frspec.sh")])
        self.assertEqual(plan.env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(plan.env["TARGET_MODEL"], str(self.next_model))
        self.assertEqual(plan.env["CACHE_BASE"], str(self.cache))
        self.assertEqual(plan.errors, [])
        self.assertIn("profile: next", plan.summary[1])

    def test_saved_file_wins_over_inherited_environment(self):
        plan = self.native_plan(self.native_env(),
                                environ={"TARGET_MODEL": "/ambient/model",
                                         "MAX_RUNNING_REQUESTS": "9"})
        self.assertEqual(plan.env["TARGET_MODEL"], str(self.next_model))
        self.assertEqual(plan.origins["TARGET_MODEL"], "saved file")
        # Unset recipe-specific keys fall through to the inherited environment.
        self.assertEqual(plan.env["MAX_RUNNING_REQUESTS"], "9")
        self.assertEqual(plan.origins["MAX_RUNNING_REQUESTS"],
                         "inherited environment")

    def test_saved_blank_quota_resets_to_the_documented_zero(self):
        # The runtime reads SGLANG_HICACHE_NIXL_MAX_CACHE_GB through an
        # EnvStr-style scalar: unlike the shell's ${VAR:-default}, an exported
        # empty value reaches _parse_max_cache_gb('') and raises. So a saved
        # blank for a key we document must export the ACTUAL default (0),
        # never the raw empty string — while the file keeps the blank.
        parse = runtime_max_cache_gb_parser()
        self.assertRaises(ValueError, parse, "",
                          "SGLANG_HICACHE_NIXL_MAX_CACHE_GB")
        values = self.native_env()
        values["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"] = ""
        plan = self.native_plan(values,
                                environ={"SGLANG_HICACHE_NIXL_MAX_CACHE_GB":
                                         "200"})
        self.assertEqual(plan.env["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"], "0")
        self.assertEqual(
            plan.config.values["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"], "")
        self.assertIn("reset to documented default",
                      plan.origins["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"])
        self.assertEqual(
            parse(plan.env["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"],
                  "SGLANG_HICACHE_NIXL_MAX_CACHE_GB"), 0.0)
        self.assertEqual(self.messages(plan, "error"), "")
        # Absent still inherits: ambient 200 reaches the real parser intact.
        plan = self.native_plan(self.native_env(),
                                environ={"SGLANG_HICACHE_NIXL_MAX_CACHE_GB":
                                         "200"})
        self.assertEqual(plan.env["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"], "200")
        self.assertEqual(
            parse(plan.env["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"],
                  "SGLANG_HICACHE_NIXL_MAX_CACHE_GB"), 200.0)
        # A saved nonblank value is untouched by the reset logic.
        values = self.native_env()
        values["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"] = "12"
        plan = self.native_plan(values,
                                environ={"SGLANG_HICACHE_NIXL_MAX_CACHE_GB":
                                         "200"})
        self.assertEqual(plan.env["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"], "12")

    def test_saved_blank_suppresses_the_inherited_environment(self):
        # John's confirmed case: MAX_RUNNING_REQUESTS='' saved + ambient 8.
        # The plan must export the blank (recipe ${VAR:-4} then yields 4),
        # not omit the key and let 8 leak through to the recipe.
        plan = self.native_plan({**self.native_env(),
                                 "MAX_RUNNING_REQUESTS": ""},
                                environ={"MAX_RUNNING_REQUESTS": "8"})
        # No utility default -> export the blank so the recipe's ${VAR:-4}
        # fires; the file keeps the raw blank so the choice survives reloads.
        self.assertEqual(plan.env["MAX_RUNNING_REQUESTS"], "")
        self.assertEqual(plan.config.values["MAX_RUNNING_REQUESTS"], "")
        self.assertIn("blank", plan.origins["MAX_RUNNING_REQUESTS"])
        # A key never saved keeps the inherited value, exactly as before.
        plan = self.native_plan(self.native_env(),
                                environ={"MAX_RUNNING_REQUESTS": "8"})
        self.assertEqual(plan.env["MAX_RUNNING_REQUESTS"], "8")
        self.assertEqual(plan.origins["MAX_RUNNING_REQUESTS"],
                         "inherited environment")
        # A blank nvme choice is treated as empty for validation too.
        plan = self.native_plan({**self.native_env(),
                                 "PENNY_PLE_BACKEND": "nvme",
                                 "PENNY_PLE_NVME_MODEL": ""},
                                environ={"PENNY_PLE_NVME_MODEL":
                                         str(self.dense_model)})
        self.assertIn("PENNY_PLE_NVME_MODEL", self.messages(plan, "error"))

    def test_unset_capacity_keys_stay_unset_for_the_recipe(self):
        plan = self.native_plan(self.native_env())
        self.assertNotIn("MAX_RUNNING_REQUESTS", plan.env)
        self.assertNotIn("MAX_TOTAL_TOKENS", plan.env)
        self.assertEqual(plan.env["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"], "0")

    def test_profile_switch_requires_the_draft(self):
        plan = self.native_plan(self.native_env(), profile="27b")
        self.assertTrue(any("DRAFT_MODEL" in issue.message
                            for issue in plan.errors))
        plan = self.native_plan(self.native_env(
            TARGET_MODEL=str(self.dense_model),
            DRAFT_MODEL=str(self.draft_model)), profile="27b")
        self.assertEqual(plan.errors, [])
        self.assertEqual(plan.env["DRAFT_MODEL"], str(self.draft_model))

    def test_recipe_choice_follows_profile(self):
        for profile, recipe in pc.RECIPE_BY_PROFILE.items():
            dense = profile == "27b"
            plan = self.native_plan(
                self.native_env(
                    TARGET_MODEL=str(self.dense_model if dense
                                     else self.next_model),
                    DRAFT_MODEL=str(self.draft_model) if dense else None),
                profile=profile)
            self.assertEqual(Path(plan.argv[0]).name, recipe)

    def test_next_model_on_27b_profile_is_a_clear_mixup_error(self):
        plan = self.native_plan({"REPO_ROOT": str(self.repo),
                                 "VENV_PATH": str(self.venv),
                                 "TARGET_MODEL": str(self.next_model),
                                 "DRAFT_MODEL": str(self.draft_model),
                                 "CACHE_BASE": str(self.cache),
                                 "NIXL_STORAGE_BASE": str(self.nixl)},
                                profile="27b")
        self.assertIn("clear mixup", self.messages(plan, "error"))

    def test_27b_model_on_next_profile_is_a_clear_mixup_error(self):
        plan = self.native_plan({**self.native_env(),
                                 "TARGET_MODEL": str(self.dense_model)})
        self.assertIn("clear mixup", self.messages(plan, "error"))

    def test_next_plain_shares_the_next_family(self):
        plan = self.native_plan({**self.native_env(),
                                 "TARGET_MODEL": str(self.next_model)},
                                profile="next-plain")
        self.assertEqual(plan.errors, [])
        self.assertEqual(self.messages(plan, "warn"), "")

    def test_draft_checkpoint_metadata_is_checked_too(self):
        # A dense draft on 27b is fine; a Next draft is a clear mixup error.
        plan = self.native_plan({**self.native_env(),
                                 "TARGET_MODEL": str(self.dense_model),
                                 "DRAFT_MODEL": str(self.draft_model)},
                                profile="27b")
        self.assertEqual(plan.errors, [])
        plan = self.native_plan({**self.native_env(),
                                 "TARGET_MODEL": str(self.dense_model),
                                 "DRAFT_MODEL": str(self.next_model)},
                                profile="27b")
        self.assertIn("clear mixup", self.messages(plan, "error"))

    def test_unknown_architecture_and_missing_config_are_not_rejected(self):
        plan = self.native_plan({**self.native_env(),
                                 "TARGET_MODEL": str(self.custom_model)})
        self.assertEqual(plan.errors, [])
        self.assertEqual(self.messages(plan, "warn"), "")
        odd = self.models / "custom quant no config"
        odd.mkdir()
        plan = self.native_plan({**self.native_env(),
                                 "TARGET_MODEL": str(odd)})
        self.assertEqual(plan.errors, [])
        self.assertIn("config.json is missing", self.messages(plan, "warn"))

    def test_missing_writable_roots_report_the_exact_mkdir_command(self):
        plan = self.native_plan({**self.native_env(),
                                 "CACHE_BASE": str(self.cache / "not-yet" / "deep"),
                                 "NIXL_STORAGE_BASE": str(self.nixl)})
        self.assertEqual(plan.errors, [])
        self.assertIn(f"mkdir -p {str(self.cache / 'not-yet' / 'deep')}",
                      self.messages(plan, "warn"))

    def test_roots_that_are_files_or_missing_volume_are_errors(self):
        blocker = self.base / "blocker"
        blocker.write_text("x")
        plan = self.native_plan({**self.native_env(),
                                 "NIXL_STORAGE_BASE": str(blocker)})
        self.assertIn("not-a-directory", self.messages(plan, "error"))

    def test_nvme_backend_requires_the_prepared_snapshot(self):
        plan = self.native_plan({**self.native_env(),
                                 "PENNY_PLE_BACKEND": "nvme"})
        self.assertIn("PENNY_PLE_NVME_MODEL", self.messages(plan, "error"))
        plan = self.native_plan({**self.native_env(),
                                 "PENNY_PLE_BACKEND": "nvme",
                                 "PENNY_PLE_NVME_MODEL": str(self.next_model)})
        self.assertEqual(plan.errors, [])

    def test_missing_runtime_reports_the_origin_of_the_path(self):
        plan = self.native_plan({**self.native_env(),
                                 "VENV_PATH": str(self.base / "gone")})
        self.assertIn("saved file", self.messages(plan, "error"))

    def test_invalid_saved_values_fail_before_launch(self):
        self.write_native_config({**self.native_env(), "MAX_RUNNING_REQUESTS": "0"})
        with self.assertRaises(pc.ConfigError):
            pc.load_config("native", self.config_path, {}, self.repo)

    def test_gpu_index_alone_is_allowed_but_garbage_is_not(self):
        self.assertEqual(pc.validate_value(pc.specs_for("native")["GPU"], "1", "t"),
                         "1")
        self.assertEqual(pc.validate_value(pc.specs_for("native")["GPU"],
                                           "GPU-abcdef", "t"), "GPU-abcdef")
        for bad in ("0 1", "0,1", "card 0"):
            with self.assertRaises(pc.ConfigError):
                pc.validate_value(pc.specs_for("native")["GPU"], bad, "t")

    def test_unknown_keys_are_preserved_and_exported(self):
        values = dict(self.native_env())
        values["MY_ADVANCED"] = "keep $this"
        plan = self.native_plan(values)
        self.assertEqual(plan.env["MY_ADVANCED"], "keep $this")
        self.assertEqual(plan.config.unknown, {"MY_ADVANCED": "keep $this"})

    def test_check_and_show_config_exercise_the_real_cli(self):
        self.write_native_config(self.native_env())
        run = subprocess.run([sys.executable, str(SCRIPTS / "penny_config.py"),
                              "--mode", "native", "--config",
                              str(self.config_path), "--show-config"],
                             capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("launch command:", run.stdout)
        # --check never imports the model stack.
        run = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r); import penny_config as pc; "
             "rc = pc.main(['--mode','native','--config',%r,'--check']); "
             "print('TORCH_LOADED' if 'torch' in sys.modules else 'NO_TORCH', rc)"
             % (str(SCRIPTS), str(self.config_path))],
            capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("NO_TORCH 0", run.stdout)

    def test_bad_value_exits_nonzero_with_a_readable_reason(self):
        self.config_path.write_text("MAX_RUNNING_REQUESTS=zero\n")
        run = subprocess.run([sys.executable, str(SCRIPTS / "penny_config.py"),
                              "--mode", "native", "--config",
                              str(self.config_path), "--check"],
                             capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 2)
        self.assertIn("positive integer", run.stderr)


class LauncherTests(FixtureMixin):
    def test_run_penny_delegates_saved_environment_to_a_fake_recipe(self):
        capture = self.base / "capture"
        stub = self.repo / "configs/pennyroyal/serve-flash-next-frspec.sh"
        stub.write_text("#!/usr/bin/env bash\n"
                        "printf 'argv=%s\\0' \"$0\" > \"$CAPTURE\"\n"
                        'for n in REPO_ROOT SGLANG_EXE PYTHON TARGET_MODEL '
                        'CACHE_BASE NIXL_STORAGE_BASE CUDA_VISIBLE_DEVICES '
                        'SGLANG_HICACHE_NIXL_MAX_CACHE_GB MY_KEY; do '
                        'printf "%s=%s\\0" "$n" "${!n-}" >> "$CAPTURE"; done\n')
        stub.chmod(0o755)
        self.write_native_config({**self.native_env(), "GPU": "1",
                                 "MY_KEY": "value with spaces $literal"})
        env = dict(os.environ, CAPTURE=str(capture))
        run = subprocess.run([str(ROOT / "run-penny"), "--config",
                              str(self.config_path)], capture_output=True,
                             text=True, check=False, env=env, cwd=str(ROOT))
        self.assertEqual(run.returncode, 0, run.stderr)
        entries = [item.decode() for item in capture.read_bytes().split(b"\0")
                   if item]
        self.assertEqual(entries[0].split("=", 1)[1],
                         str(self.repo / "configs/pennyroyal/"
                                  "serve-flash-next-frspec.sh"))
        fields = dict(entry.split("=", 1) for entry in entries[1:])
        self.assertEqual(fields["TARGET_MODEL"], str(self.next_model))
        self.assertEqual(fields["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(fields["MY_KEY"], "value with spaces $literal")
        self.assertEqual(fields["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"], "0")
        self.assertIn("Pennyroyal configuration", run.stderr)

    def test_run_penny_blank_saved_key_reaches_recipe_as_empty(self):
        # The executed environment must carry KEY='' (suppressing an exported
        # ambient 8), letting the recipe's own ${KEY:-4} pick 4 — while an
        # unsaved key still passes the ambient value through unchanged.
        capture = self.base / "capture"
        stub = self.repo / "configs/pennyroyal/serve-flash-next-frspec.sh"
        stub.write_text("#!/usr/bin/env bash\n"
                        'printf "raw=${MAX_RUNNING_REQUESTS-<unset>} '
                        'effective=${MAX_RUNNING_REQUESTS:-4} '
                        'second=${MAX_TOTAL_TOKENS-<unset>}\n" > "$CAPTURE"\n')
        stub.chmod(0o755)
        self.write_native_config({**self.native_env(),
                                  "MAX_RUNNING_REQUESTS": ""})
        env = dict(os.environ, CAPTURE=str(capture),
                   MAX_RUNNING_REQUESTS="8", MAX_TOTAL_TOKENS="99")
        run = subprocess.run([str(ROOT / "run-penny"), "--config",
                              str(self.config_path)], capture_output=True,
                             text=True, check=False, env=env, cwd=str(ROOT))
        self.assertEqual(run.returncode, 0, run.stderr)
        captured = capture.read_text().strip()
        self.assertIn("raw= effective=4", captured)   # blank exported -> default
        self.assertIn("second=99", captured)          # unsaved -> inherited
        # Without the saved blank line, the ambient value must still reach it.
        self.write_native_config(self.native_env())
        run = subprocess.run([str(ROOT / "run-penny"), "--config",
                              str(self.config_path)], capture_output=True,
                             text=True, check=False, env=env, cwd=str(ROOT))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("raw=8 effective=8", capture.read_text())

    def test_run_penny_show_config_never_executes_the_recipe(self):
        capture = self.base / "never"
        stub = self.repo / "configs/pennyroyal/serve-flash-next-frspec.sh"
        stub.write_text(f'#!/usr/bin/env bash\nprintf ran > "{capture}"\n')
        stub.chmod(0o755)
        self.write_native_config(self.native_env())
        run = subprocess.run([str(ROOT / "run-penny"), "--config",
                              str(self.config_path), "--show-config"],
                             capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("launch command:", run.stdout)
        self.assertFalse(capture.exists())

    def test_run_penny_check_stops_before_launch(self):
        # No saved file and no .venv in the checkout: validation must fail
        # without executing anything.
        run = subprocess.run([str(ROOT / "run-penny"), "--config",
                              str(self.config_path), "--check"],
                             capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 1)
        self.assertIn("TARGET_MODEL", run.stderr)
        run = subprocess.run([str(ROOT / "run-penny"), "--help"],
                             capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0)
        self.assertIn("--show-config", run.stdout)
        self.assertIn("--mode {native,container}", run.stdout)


class ContainerPlanTests(FixtureMixin):
    def setUp(self) -> None:
        super().setUp()
        self.compose_dir = self.repo / "docker" / "pennyroyal"
        self.compose_dir.mkdir(parents=True)
        # Use the real repository Compose file so `config` renders the same
        # interpolation the deployment relies on.
        (self.compose_dir / "compose.yaml").write_text(
            (ROOT / pc.COMPOSE_RELPATH).read_text())
        self.host_root = self.base / "host models"
        self.host_root.mkdir()
        (self.host_root / "RadixArk-Qwen3.8-Flash-Next-NVFP4").mkdir()
        (self.host_root / "Qwen3.8-27B-FP8").mkdir()
        (self.host_root / "Qwen3.8-27B-DFlash2").mkdir()
        self.env_dir = self.repo / "docker" / "pennyroyal"
        self.env_file = self.env_dir / ".env"

    def container_plan(self, values: dict[str, str], profile: str = "next",
                       environ: dict[str, str] | None = None) -> pc.Plan:
        self.env_file.write_text(pc.serialize_env(
            [("", sorted(values.items()))],
            header=(f"{pc.PROFILE_KEY}={pc.quote_value(profile)}",)))
        config = pc.load_config("container", self.env_file, environ or {},
                                self.repo)
        return pc.build_plan("container", config, environ or {},
                             repo_root=self.repo)

    def base_values(self) -> dict[str, str]:
        (self.base / "cache").mkdir(exist_ok=True)
        (self.base / "nixl").mkdir(exist_ok=True)
        return {"HOST_MODELS_ROOT": str(self.host_root),
                "HOST_CACHE_BASE": str(self.base / "cache"),
                "HOST_NIXL_STORAGE_BASE": str(self.base / "nixl")}

    def test_default_target_follows_the_selected_profile(self):
        plan = self.container_plan(self.base_values(), profile="next")
        self.assertEqual(plan.env["TARGET_MODEL"],
                         "/models/RadixArk-Qwen3.8-Flash-Next-NVFP4")
        self.assertNotIn("DRAFT_MODEL", plan.env)
        plan = self.container_plan(self.base_values(), profile="27b")
        self.assertEqual(plan.env["TARGET_MODEL"], "/models/Qwen3.8-27B-FP8")
        self.assertEqual(plan.env["DRAFT_MODEL"],
                         "/models/Qwen3.8-27B-DFlash2")
        self.assertEqual(plan.errors, [])

    def test_explicit_paths_always_win_over_profile_defaults(self):
        values = {**self.base_values(),
                  "TARGET_MODEL": "/models/Qwen3.8-27B-FP8"}
        plan = self.container_plan(values, profile="next")
        self.assertEqual(plan.env["TARGET_MODEL"], "/models/Qwen3.8-27B-FP8")

    def test_paths_outside_the_models_mount_are_rejected(self):
        values = {**self.base_values(), "TARGET_MODEL": "/srv/elsewhere/model"}
        plan = self.container_plan(values)
        self.assertIn("must live under /models/", self.messages(plan, "error"))

    def test_container_path_maps_onto_the_host_mount(self):
        values = {**self.base_values(), "TARGET_MODEL": "/models/not-on-host"}
        plan = self.container_plan(values)
        message = self.messages(plan, "error")
        self.assertIn("/models/not-on-host", message)
        self.assertIn(str(self.host_root / "not-on-host"), message)

    def test_relative_host_paths_are_rejected(self):
        values = {**self.base_values(), "HOST_MODELS_ROOT": "relative/models"}
        plan = self.container_plan(values)
        self.assertIn("absolute host path", self.messages(plan, "error"))

    def test_compose_command_is_absolute_and_environment_matches_the_file(self):
        plan = self.container_plan(self.base_values(), profile="27b")
        self.assertEqual(plan.argv[0:2], ["docker", "compose"])
        self.assertIn("--env-file", plan.argv)
        self.assertEqual(plan.argv[plan.argv.index("-f") + 1],
                         str(self.compose_dir / "compose.yaml"))
        self.assertEqual(plan.argv[plan.argv.index("--env-file") + 1],
                         str(self.env_file))
        self.assertTrue(plan.next_command.endswith(
            " ".join(pc.quote_command_arg(arg) for arg in plan.argv)))
        self.assertIn("TARGET_MODEL=", plan.next_command)
        self.assertEqual(plan.forced_env["TARGET_MODEL"], "/models/Qwen3.8-27B-FP8")
        self.assertEqual(shlex.split(plan.next_command)[
            shlex.split(plan.next_command).index("docker"):], plan.argv)
        self.assertEqual(plan.env["PENNYROYAL_PROFILE"], "27b")
        self.assertEqual(plan.env["NVIDIA_GPU"], "0")
        self.assertEqual(plan.env["PENNYROYAL_PORT"], "8001")
        self.assertEqual(plan.env["USER_ID"], "1000")
        self.assertEqual(plan.env["PENNYROYAL_IMAGE"], pc.DEFAULT_IMAGE)

    def test_container_blank_quota_resets_to_zero_for_the_runtime(self):
        # Same contract in the container: the .env gets the real default so
        # the entrypoint's parser never sees '' (which it rejects), while an
        # ambient 200 cannot override the saved reset decision.
        parse = runtime_max_cache_gb_parser()
        plan = self.container_plan({**self.base_values(),
                                    "SGLANG_HICACHE_NIXL_MAX_CACHE_GB": ""},
                                   environ={"SGLANG_HICACHE_NIXL_MAX_CACHE_GB":
                                            "200"})
        self.assertEqual(plan.env["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"], "0")
        self.assertEqual(plan.forced_env["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"],
                         "0")
        self.assertEqual(
            parse(plan.forced_env["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"],
                  "SGLANG_HICACHE_NIXL_MAX_CACHE_GB"), 0.0)
        self.assertIn("SGLANG_HICACHE_NIXL_MAX_CACHE_GB=0",
                      plan.next_command)
        # Human-readable summary: lines stay separate and show the reset value.
        self.assertIn("NIXL byte budget: 0 GiB (0 = no budget)", plan.summary)
        self.assertIn("runtime identity: 1000:1000", plan.summary)
        self.assertTrue(any(line.startswith("API port: ")
                            for line in plan.summary))

    def test_container_blank_override_ships_in_the_printed_command(self):
        # An empty saved override must ride in the printed command as KEY=''
        # so compose's ${KEY:-default} fires despite a conflicting export.
        plan = self.container_plan({**self.base_values(),
                                    "MAX_RUNNING_REQUESTS": ""},
                                   environ={"MAX_RUNNING_REQUESTS": "8"})
        self.assertEqual(plan.env["MAX_RUNNING_REQUESTS"], "")
        self.assertEqual(plan.forced_env["MAX_RUNNING_REQUESTS"], "")
        self.assertIn("MAX_RUNNING_REQUESTS=''", plan.next_command)
        with mock.patch.dict(os.environ, {"MAX_RUNNING_REQUESTS": "8"}):
            rendered = self.run_printed_command(plan)
        self.assertIn('MAX_RUNNING_REQUESTS: "4"', rendered)
        self.assertNotIn('"8"', rendered)

    def test_saved_values_win_and_ambient_values_pass_through(self):
        plan = self.container_plan({**self.base_values(), "PENNYROYAL_PORT": "8123"},
                                   environ={"SGLANG_FORWARD_UNKNOWN_TOOLS": "false"})
        self.assertEqual(plan.env["PENNYROYAL_PORT"], "8123")
        self.assertEqual(plan.env["SGLANG_FORWARD_UNKNOWN_TOOLS"], "false")

    def run_printed_command(self, plan: Plan) -> str:
        """Run the printed command verbatim from / with `up -d` -> `config`.

        The ambient shell deliberately lacks (and sometimes conflicts with) the
        saved keys, so this proves the printed command alone delivers the plan.
        The compose renderer comes from compose_command(), so a host with only
        the standalone docker-compose binary runs that instead of a hardcoded
        'docker compose' pair.
        """
        compose_cmd = compose_command()
        if compose_cmd is None:
            self.skipTest("docker compose CLI not installed on this host")
        tokens = shlex.split(plan.next_command)
        body_start = len(tokens) - len(plan.argv)
        assignments = tokens[:body_start]
        self.assertEqual(tokens[body_start:], plan.argv)
        # Compare canonical parses: printed assignments must decode to the plan.
        decoded = dict(item.split("=", 1) for item in assignments)
        self.assertEqual(decoded, plan.forced_env)
        # Same -f/--env-file arguments as printed, with `up -d` replaced by
        # `config`: render only, no daemon, no service start.
        command = (list(compose_cmd) + plan.argv[2:plan.argv.index("up")]
                   + ["config"])
        env = {key: value for key, value in os.environ.items()
               if key not in pc.COMPOSE_CRITICAL_KEYS}
        env.update(decoded)
        run = subprocess.run(command, capture_output=True, text=True,
                             check=False, cwd="/", env=env)
        self.assertEqual(run.returncode, 0, run.stderr)
        return run.stdout

    def test_saved_root_wins_over_conflicting_ambient_env(self):
        plan = self.container_plan(self.base_values(),
                                   environ={"HOST_MODELS_ROOT": "/bad/ambient",
                                            "TARGET_MODEL": "/models/ambient"})
        # The plan validates the saved values, and the printed command must
        # force them even though the ambient shell exports the conflicting ones.
        self.assertEqual(plan.env["HOST_MODELS_ROOT"], str(self.host_root))
        self.assertEqual(plan.env["TARGET_MODEL"], "/models/ambient")
        self.assertEqual(plan.origins["HOST_MODELS_ROOT"], "saved file")
        self.assertEqual(plan.forced_env["HOST_MODELS_ROOT"], str(self.host_root))
        self.assertNotIn("/bad/ambient", plan.next_command)
        rendered = self.run_printed_command(plan)
        self.assertIn(str(self.host_root), rendered)
        self.assertNotIn("/bad/ambient", rendered)
        self.assertIn("/models/ambient", rendered)

    def test_27b_unset_target_reaches_compose_via_printed_command(self):
        plan = self.container_plan(self.base_values(), profile="27b")
        # The compose file itself defaults TARGET_MODEL to the Next path, so
        # only the forced prefix can deliver the profile-appropriate default.
        self.assertEqual(plan.forced_env["TARGET_MODEL"],
                         "/models/Qwen3.8-27B-FP8")
        self.assertEqual(plan.forced_env["DRAFT_MODEL"],
                         "/models/Qwen3.8-27B-DFlash2")
        rendered = self.run_printed_command(plan)
        self.assertIn("TARGET_MODEL: /models/Qwen3.8-27B-FP8", rendered)
        self.assertIn("DRAFT_MODEL: /models/Qwen3.8-27B-DFlash2", rendered)
        self.assertNotIn("RadixArk", rendered)

    def test_missing_compose_cli_is_detected_not_raised(self):
        # The parent host has no docker CLI at all; probing must answer
        # 'unavailable' (None) instead of raising FileNotFoundError, and honour
        # a standalone docker-compose when the docker plugin is absent.
        real = subprocess.run

        def fake(cmd, *args, **kwargs):
            if cmd[0] not in ("docker", "docker-compose"):
                return real(cmd, *args, **kwargs)
            if cmd[0] == "docker" and cmd[1:2] == ["compose"]:
                raise FileNotFoundError("docker")
            if cmd[0] == "docker":
                raise FileNotFoundError("docker")
            return real(["true"], *args[1:], **kwargs)
        with mock.patch("subprocess.run", side_effect=fake):
            self.assertEqual(compose_command(), ["docker-compose"])

        def all_missing(cmd, *args, **kwargs):
            if cmd[0] in ("docker", "docker-compose"):
                raise FileNotFoundError(cmd[0])
            return real(cmd, *args, **kwargs)
        with mock.patch("subprocess.run", side_effect=all_missing):
            self.assertIsNone(compose_command())

    def test_manual_compose_operation_still_works(self):
        # Without the prefix, Compose uses .env then compose.yaml defaults,
        # which is the documented manual path; we must not rewrite the file.
        compose = compose_command()
        if compose is None:
            self.skipTest("docker compose CLI not installed on this host")
        self.container_plan(self.base_values(), profile="27b")
        # The fixture compose.yaml is a stub, so point at the real repo file to
        # prove manual operation resolves the same .env without any prefix.
        run = subprocess.run(compose + ["--env-file", str(self.env_file),
                                        "-f", str(ROOT / pc.COMPOSE_RELPATH),
                                        "config"], capture_output=True,
                             text=True, check=False, cwd=str(ROOT),
                             env={**{key: value for key, value in os.environ.items()
                                     if key not in pc.COMPOSE_CRITICAL_KEYS},
                                  "HOST_MODELS_ROOT": str(self.host_root),
                                  "HOST_CACHE_BASE": str(self.host_root),
                                  "HOST_NIXL_STORAGE_BASE": str(self.host_root)})
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn(str(self.host_root), run.stdout)
        self.assertIn("/models/Qwen3.8-27B-DFlash2", run.stdout)

    def test_models_root_needs_readability_not_writability(self):
        readonly_root = self.base / "readonly models"
        for name in ("RadixArk-Qwen3.8-Flash-Next-NVFP4", "Qwen3.8-27B-FP8",
                     "Qwen3.8-27B-DFlash2"):
            (readonly_root / name).mkdir(parents=True)
        os.chmod(readonly_root, 0o500)  # r-x: mounted read-only anyway
        values = {**self.base_values(), "HOST_MODELS_ROOT": str(readonly_root)}
        plan = self.container_plan(values)
        try:
            self.assertEqual(plan.errors, [])
        finally:
            os.chmod(readonly_root, 0o700)

    def test_cache_ownership_mismatch_is_reported_concretely(self):
        foreign = self.base / "foreign cache"
        foreign.mkdir()
        os.chmod(foreign, 0o755)
        owner = pc.path_owner(str(foreign))
        # As the configuring user the directory is writable, so the plan reports
        # the container identity assumption instead of silently trusting or
        # rejecting it, and never chowns or sudos.
        plan = self.container_plan({**self.base_values(),
                                    "HOST_CACHE_BASE": str(foreign),
                                    "USER_ID": "65534", "GROUP_ID": "65534"})
        self.assertEqual(plan.errors, [])
        message = self.messages(plan, "warn")
        self.assertIn("65534:65534", message)
        self.assertIn(owner, message)
        self.assertIn("chown", message)

    def test_native_caches_checked_against_this_account_only(self):
        # Native launches run as the current user, so no container-identity
        # wording ever appears there.
        plan = self.native_plan(self.native_env())
        self.assertEqual(self.messages(plan, "warn"), "")

    def test_printed_native_command_equals_the_validated_plan(self):
        # A launcher run from an unrelated directory with explicit --config and
        # --profile must print a command that reloads exactly this plan.
        self.write_native_config(self.native_env())
        run = subprocess.run([sys.executable, str(SCRIPTS / "penny_config.py"),
                              "--mode", "native", "--config",
                              str(self.config_path), "--profile", "27b",
                              "--show-config"],
                             capture_output=True, text=True, check=False, cwd="/")
        self.assertIn("next command:", run.stdout)
        printed = run.stdout.split("next command:")[1].strip().splitlines()[0]
        self.assertEqual(shlex.split(printed),
                         [str(self.repo / "run-penny"), "--config",
                          str(self.config_path.absolute()), "--profile", "27b"])

    def test_relative_selected_config_prints_an_absolute_reload_path(self):
        # --config (or PENNYROYAL_CONFIG) given as a relative path must not
        # leak into the printed --env-file, where a later cwd would read a
        # different file; the selected path is resolved at load time.
        self.env_file.write_text(pc.serialize_env(
            [("basic", list(self.base_values().items()))],
            header=(f"{pc.PROFILE_KEY}=next",)))
        import contextlib
        with contextlib.chdir(self.env_dir):
            config = pc.load_config("container", Path(".env"), {}, self.repo)
            plan = pc.build_plan("container", config, {}, repo_root=self.repo)
        self.assertEqual(
            plan.argv[plan.argv.index("--env-file") + 1], str(self.env_file))
        self.assertTrue(Path(plan.argv[plan.argv.index("--env-file") + 1])
                        .is_absolute())
        # Running the printed command from / still resolves the same mounts.
        rendered = self.run_printed_command(plan)
        self.assertIn(str(self.host_root), rendered)

    def test_missing_env_file_is_a_warning_not_a_crash(self):
        config = pc.load_config("container", self.env_dir / ".env", {}, self.repo)
        plan = pc.build_plan("container", config, {}, repo_root=self.repo)
        self.assertIn("does not exist yet", self.messages(plan, "warn"))


class AtomicWriteTests(unittest.TestCase):
    def test_permissions_are_preserved_and_no_temp_file_remains(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pennyroyal.env"
            path.write_text("A=1\n")
            path.chmod(0o640)
            pc.write_env_file(path, "A=2\n")
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            self.assertEqual(pc.read_env_file(path), {"A": "2"})
            self.assertEqual(sorted(entry.name for entry in Path(temp).iterdir()),
                             ["pennyroyal.env"])


class BuildEnvTests(unittest.TestCase):
    def test_helper_exports_the_existing_defaults_and_respects_presets(self):
        helper = SCRIPTS / "build-env.sh"
        script = (f'set -euo pipefail\nsource "{helper}"\n'
                  'for name in CUDA_HOME CC CXX TORCH_CUDA_ARCH_LIST '
                  'PENNY_BUILD_JOBS MAX_JOBS FLASHINFER_NVCC_THREADS '
                  'TORCHINDUCTOR_COMPILE_THREADS; do '
                  'printf "%s=%s\\n" "$name" "${!name}"; done')
        run = subprocess.run(["bash", "-c", script], capture_output=True,
                             text=True, check=True,
                             env={"PATH": "/usr/bin:/bin"})
        values = dict(line.split("=", 1) for line in run.stdout.splitlines())
        self.assertEqual(values["PENNY_BUILD_JOBS"], "4")
        self.assertEqual(values["MAX_JOBS"], "4")
        self.assertEqual(values["TORCH_CUDA_ARCH_LIST"], "12.0")
        self.assertEqual(values["FLASHINFER_NVCC_THREADS"], "1")
        run = subprocess.run(["bash", "-c", script], capture_output=True,
                             text=True, check=True,
                             env={"PATH": "/usr/bin:/bin",
                                  "PENNY_BUILD_JOBS": "8"})
        values = dict(line.split("=", 1) for line in run.stdout.splitlines())
        self.assertEqual(values["MAX_JOBS"], "8")


if __name__ == "__main__":
    unittest.main(verbosity=2)
