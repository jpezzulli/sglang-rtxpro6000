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
import re
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


# Shell variables the generated run.sh reads as fallbacks; a developer's own
# export must never change what a test expects the launch to be.
LAUNCH_ENV_KEYS = ("PENNYROYAL_IMAGE", "PENNYROYAL_STARTUP",
                   "PENNYROYAL_NIXL_CONFIG", "PENNYROYAL_PORT",
                   "PENNYROYAL_USER", "HOST_MODELS_ROOT", "HOST_CACHE_BASE",
                   "HOST_NIXL_STORAGE_BASE", "NVIDIA_GPU", "TARGET_MODEL",
                   "DRAFT_MODEL", "NIXL", "USER_ID", "GROUP_ID", "TP_SIZE",
                   "PENNYROYAL_PROFILE")


def docker_stub(directory: Path) -> tuple[Path, Path]:
    """A docker on PATH that records its argv instead of talking to a daemon."""
    capture = directory / f"docker-argv-{os.urandom(4).hex()}"
    bin_dir = directory / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ': > "$DOCKER_CAPTURE"\n'
        'for argument in "$@"; do printf "%s\\n" "$argument" '
        '>> "$DOCKER_CAPTURE"; done\n')
    stub.chmod(0o755)
    return bin_dir, capture


class _Cancelled(Exception):
    """Stand-in for the wizard's cancellation (its 'q'/EOF Cancelled signal)."""


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

    def test_hicache_size_and_nixl_budget_labels_keep_their_units_apart(self):
        # The RAM (HiCache) knob is decimal GB (SGLang sizes the host pool at
        # size * 1e9 bytes), the NIXL knob is a GiB disk budget whose 0 means
        # 'no cap', and neither is the compiled-cache folder.
        specs = pc.specs_for("native")
        hicache = specs["PENNY_HICACHE_SIZE_GB"]
        nixl = specs["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"]
        self.assertIn("1e9 bytes, not GiB", hicache.prompt)
        self.assertEqual(hicache.kind, "positive-int")
        self.assertIn("qualified default", hicache.description)
        self.assertIn("GiB", nixl.prompt)
        self.assertIn("unlimited", nixl.prompt)
        self.assertNotIn("HiCache", specs["CACHE_BASE"].prompt)
        self.assertNotIn("HiCache", specs["CACHE_BASE"].description)
        # Both cache-size knobs stay in the main wizard: the RAM tier is
        # user-chosen, not an advanced afterthought.
        self.assertFalse(hicache.advanced)
        self.assertFalse(nixl.advanced)
        # A positive integer is honored at any size; the documented junk is
        # rejected the same way the capacity knobs already are.
        for value in ("1", "2", "96", "1024"):
            self.assertEqual(pc.validate_value(hicache, value, "t"), value)
        for bad in ("0", "-1", "1.5", "32abc", " 4"):
            with self.assertRaises(pc.ConfigError):
                pc.validate_value(hicache, bad, "t")
        # The NIXL budget keeps accepting a small decimal and 0.
        for value in ("0", "0.5", "200"):
            self.assertEqual(pc.validate_value(nixl, value, "t"), value)

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


class HiCacheSizePlanTests(FixtureMixin):
    """The chosen RAM cache size reaches the recipe without touching anything
    else: an unset key keeps the profile's qualified literal, and the saved
    blank keeps suppressing an inherited value instead of inventing a default.
    """

    def summary_has(self, plan: pc.Plan, prefix: str) -> str:
        return "\n".join(line for line in plan.summary
                        if line.startswith(prefix))

    def test_unset_size_leaves_the_recipe_default_alone_per_profile(self):
        for profile, default in (("next", "32"), ("next-plain", "32"),
                                 ("27b", "96")):
            with self.subTest(profile=profile):
                dense = profile == "27b"
                plan = self.native_plan(
                    self.native_env(
                        TARGET_MODEL=str(self.dense_model if dense
                                         else self.next_model),
                        DRAFT_MODEL=str(self.draft_model) if dense else None),
                    profile=profile)
                self.assertEqual(plan.errors, [])
                # Nothing is exported: the recipe's own literal still decides.
                self.assertNotIn("PENNY_HICACHE_SIZE_GB", plan.env)
                self.assertTrue(self.summary_has(
                    plan, f"RAM cache (HiCache): {default} GB as "
                          f"--hicache-size"))

    def test_explicit_small_size_is_exported_and_reported(self):
        for size in ("1", "2"):
            with self.subTest(size=size):
                plan = self.native_plan({**self.native_env(),
                                         "PENNY_HICACHE_SIZE_GB": size})
                self.assertEqual(plan.errors, [])
                self.assertEqual(plan.env["PENNY_HICACHE_SIZE_GB"], size)
                self.assertTrue(self.summary_has(
                    plan, f"RAM cache (HiCache): {size} GB as --hicache-size"))
                # The environment is the only plumbing: the recipe path and its
                # argv stay exactly what they were.
                self.assertEqual(
                    Path(plan.argv[0]).name,
                    "serve-flash-next-frspec.sh")

    def test_saved_blank_resets_to_the_recipe_default_and_still_suppresses(self):
        # Blank means 'the recipe decides': no utility default exists, so the
        # exported value is the empty string and the recipe's own
        # ${PENNY_HICACHE_SIZE_GB:-32} fires — while an ambient 8 cannot leak
        # past the saved blank.
        plan = self.native_plan({**self.native_env(),
                                 "PENNY_HICACHE_SIZE_GB": ""},
                                environ={"PENNY_HICACHE_SIZE_GB": "8"})
        self.assertEqual(plan.env["PENNY_HICACHE_SIZE_GB"], "")
        self.assertIn("blank", plan.origins["PENNY_HICACHE_SIZE_GB"])
        self.assertEqual(self.messages(plan, "error"), "")
        self.assertTrue(self.summary_has(
            plan, "RAM cache (HiCache): 32 GB as --hicache-size"))

    def test_ambient_small_size_reaches_the_recipe_when_not_saved(self):
        plan = self.native_plan(self.native_env(),
                                environ={"PENNY_HICACHE_SIZE_GB": "2"})
        self.assertEqual(plan.env["PENNY_HICACHE_SIZE_GB"], "2")
        self.assertEqual(plan.origins["PENNY_HICACHE_SIZE_GB"],
                         "inherited environment")

    def test_container_passes_the_size_as_one_more_environment_value(self):
        # The Compose file only gains a pass-through: its own default stays
        # empty so the recipe's qualified literal still answers when the
        # operator chose nothing, and no mount, port, or image setting moves.
        compose_text = (ROOT / pc.COMPOSE_RELPATH).read_text()
        self.assertIn("PENNY_HICACHE_SIZE_GB: ${PENNY_HICACHE_SIZE_GB:-}\n",
                      compose_text)
        self.assertIn("SGLANG_HICACHE_NIXL_MAX_CACHE_GB: "
                      "${SGLANG_HICACHE_NIXL_MAX_CACHE_GB:-0}", compose_text)
        self.assertNotIn("version:", compose_text)
        compose_dir = self.repo / "docker" / "pennyroyal"
        compose_dir.mkdir(parents=True, exist_ok=True)
        (compose_dir / "compose.yaml").write_text(compose_text)
        env_file = compose_dir / ".env"
        (self.base / "hm").mkdir()
        values = {"HOST_MODELS_ROOT": str(self.base / "hm"),
                  "HOST_CACHE_BASE": str(self.cache),
                  "HOST_NIXL_STORAGE_BASE": str(self.nixl),
                  "PENNY_HICACHE_SIZE_GB": "2"}
        env_file.write_text(pc.serialize_env(
            [("", sorted(values.items()))],
            header=(f"{pc.PROFILE_KEY}=next",)))
        # The fixture repo needs the shipped examples to generate from.
        (compose_dir / "launch").symlink_to(ROOT / pc.LAUNCH_RELPATH)
        config = pc.load_config("container", env_file, {}, self.repo)
        plan = pc.build_plan("container", config, {}, repo_root=self.repo)
        self.assertEqual(plan.env["PENNY_HICACHE_SIZE_GB"], "2")
        self.assertEqual(plan.forced_env["PENNY_HICACHE_SIZE_GB"], "2")
        # The size is a line in the operator's own startup file, not an
        # environment variable the container has to be told about.
        startup = dict(pc.container_launch_files(plan))[
            plan.launch_dir / "config" / "start-flash-next-frspec.sh"]
        self.assertIn("\nHICACHE_SIZE_GB=2\n", startup)
        self.assertNotIn("PENNY_HICACHE_SIZE_GB", " ".join(plan.argv))
        self.assertTrue(self.summary_has(
            plan, "RAM cache (HiCache): 2 GB as --hicache-size"))


class RecipeHiCacheSizeTests(unittest.TestCase):
    """The shipped recipes read the chosen size and reject unusable ones.

    Only the guard block and the launch block run (bash, no GPU, no model, no
    install): an unset or blank choice must keep the profile's qualified
    literal, which is what the launch argv contained before this knob existed.
    """

    RECIPES = {"serve-flash-next.sh": "32",
               "serve-flash-next-frspec.sh": "32",
               "serve-qwen38-27b-dflash2.sh": "96"}

    def source(self, recipe: str) -> str:
        return (ROOT / "configs" / "pennyroyal" / recipe).read_text()

    def guard_block(self, recipe: str) -> str:
        lines = self.source(recipe).splitlines()
        start = next((index for index, line in enumerate(lines)
                      if line.startswith("HICACHE_SIZE_GB=")), None)
        if start is None:
            raise AssertionError(f"no HiCache size guard in {recipe}")
        end = next(index for index in range(start, len(lines))
                   if lines[index] == "fi")
        return "\n".join(lines[start:end + 1]) + "\n"

    def run_guard(self, block: str, value: str | None
                  ) -> subprocess.CompletedProcess:
        env = {key: item for key, item in os.environ.items()
               if key != "PENNY_HICACHE_SIZE_GB"}
        if value is not None:
            env["PENNY_HICACHE_SIZE_GB"] = value
        return subprocess.run(["bash", "-c",
                               f"set -euo pipefail\n{block}"
                               'printf "%s\\n" "$HICACHE_SIZE_GB"'],
                              capture_output=True, text=True, check=False,
                              env=env)

    def test_unset_keeps_the_profile_literal_and_a_choice_is_honored(self):
        for recipe, qualified in self.RECIPES.items():
            with self.subTest(recipe=recipe):
                block = self.guard_block(recipe)
                self.assertIn(f'"${{PENNY_HICACHE_SIZE_GB:-{qualified}}}"',
                              block)
                for value, expected in ((None, qualified), ("", qualified),
                                        ("1", "1"), ("2", "2"),
                                        ("512", "512")):
                    run = self.run_guard(block, value)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertEqual(run.stdout.strip(), expected)

    def test_invalid_sizes_fail_before_the_recipe_starts(self):
        for recipe in self.RECIPES:
            with self.subTest(recipe=recipe):
                block = self.guard_block(recipe)
                for bad in ("0", "-1", "1.5", "two", "32 --hicache-ratio 4"):
                    run = self.run_guard(block, bad)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn("must be a positive integer", run.stderr)

    # The scalars the final launch block still expands, so the real block can
    # run against a capturing stub sglang (same fixture idea the startup
    # summary tests use: bash only, nothing installed, no GPU, no model).
    LAUNCH_SCALARS = {
        "SCRIPT_DIR": str(ROOT / "configs" / "pennyroyal"),
        "TARGET_MODEL": "/models/target model [v1]",
        "DRAFT_MODEL": "/models/draft model (small)",
        "TP_SIZE": "1",
        "COMPUTE_DTYPE": "bfloat16",
        "KV_DTYPE": "fp8_e4m3",
        "TARGET_KV_DTYPE": "fp8_e4m3",
        "DRAFT_KV_DTYPE": "fp8_e5m2",
        "CONTEXT_LENGTH": "524288",
        "TARGET_OVERRIDES": '{"path":"target value"}',
        "DRAFT_OVERRIDES": '{"path":"draft value"}',
        "PAGE_SIZE": "64",
        "PREFILL_CHUNK_SIZE": "4096",
        "MAMBA_SSM_DTYPE": "bfloat16",
        "MAMBA_CONV_DTYPE": "bfloat16",
        "MAMBA_TRACK_INTERVAL": "64",
        "MAX_MAMBA_CACHE_SIZE": "24",
        "MAX_RUNNING_REQUESTS": "4",
        # The integrated release sources the default from reasoning-effort.sh.
        "DEFAULT_CHAT_TEMPLATE_KWARGS": '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}',
        "NIXL_CONFIG": "/configs/NIXL config [qualified].toml",
        "CHAT_TEMPLATE": "/templates/chat template (tools).jinja",
        "IMAGE_PROCESSOR_BACKEND": "pil",
        "TOKEN_MAP": "/maps/token map [FR Spec].pt",
        "DRAFT_TOKENS": "8",
        "DRAFT_WINDOW_SIZE": "2048",
        "PLE_NAMESPACE_ARGS": "",
    }

    def launch_argv(self, recipe: str, chosen: str | None) -> tuple[list[str],
                                                                    str]:
        source = self.source(recipe).splitlines()
        start = next(index for index, line in enumerate(source)
                     if line.startswith("launch_args=(serve"))
        block = "\n".join(source[start:])
        # Drop the summary source/exec tail and print the array instead, so a
        # stub binary is never needed and nothing is executed.
        block = block.replace('source "$SCRIPT_DIR/startup-summary.sh"\n', "")
        self.assertIn('exec "$SGLANG_EXE" "${launch_args[@]}"', block)
        block = block.replace('exec "$SGLANG_EXE" "${launch_args[@]}"',
                              'printf \'%s\\0\' "${launch_args[@]}"')
        lines = ["set -euo pipefail",
                 "pennyroyal_startup_summary() { :; }"]
        lines.extend(f"{name}={shlex.quote(value)}"
                     for name, value in self.LAUNCH_SCALARS.items())
        lines.append("TOKEN_CAP_ARGS=(--max-total-tokens 824384)")
        lines.append("PLE_ARGS=(--ple-offload-embedding)")
        if chosen is not None:
            lines.append(f"HICACHE_SIZE_GB={shlex.quote(chosen)}")
        else:
            # No choice at all: the recipe's own guard must produce the
            # qualified literal (the environment below never defines it).
            lines.extend(self.guard_block(recipe).splitlines())
        lines.append(block)
        run = subprocess.run(["bash"], input="\n".join(lines) + "\n",
                             capture_output=True, text=True, check=False,
                             env={key: value for key, value in os.environ.items()
                                  if key != "PENNY_HICACHE_SIZE_GB"})
        self.assertEqual(run.returncode, 0, run.stderr)
        argv = [item for item in run.stdout.split("\0") if item]
        return argv, run.stdout

    def test_explicit_size_reaches_the_actual_flag_without_moving_anything(self):
        # The acceptance pair: unset keeps the qualified literal in the argv we
        # really build, and a chosen 1 GB / 2 GB reaches --hicache-size while
        # every other argument keeps its exact place.
        for recipe, qualified in self.RECIPES.items():
            with self.subTest(recipe=recipe):
                base_argv, _ = self.launch_argv(recipe, None)
                self.assertEqual(base_argv[base_argv.index("--hicache-size") + 1],
                                 qualified)
                self.assertIn("--enable-hierarchical-cache", base_argv)
                for size in ("1", "2"):
                    argv, _ = self.launch_argv(recipe, size)
                    at = argv.index("--hicache-size")
                    self.assertNotEqual(at, -1)
                    self.assertEqual(argv[at + 1], size)
                    # Only that one slot differs from the untouched argv.
                    self.assertEqual(argv[:at + 1] + [qualified] + argv[at + 2:],
                                     base_argv)

    def test_only_the_hicache_flag_reads_the_new_variable(self):
        # Every other launch flag keeps its literal: the diff is the size and
        # nothing else, so hierarchy/NIXL/speculation/backend defaults are
        # untouched and no cache mode or off switch appeared.
        for recipe in self.RECIPES:
            with self.subTest(recipe=recipe):
                source = self.source(recipe)
                self.assertIn('--hicache-size "$HICACHE_SIZE_GB"', source)
                self.assertNotIn("--hicache-size 32", source)
                self.assertNotIn("--hicache-size 96", source)
                self.assertIn("--enable-hierarchical-cache", source)
                self.assertIn("--hicache-storage-backend nixl", source)
                # The chosen size is read into one local and used once: the
                # guard reads the environment exactly once and only the flag
                # consumes the local, so no other flag can drift with it.
                self.assertEqual(len(re.findall(
                    r'(?<!PENNY_)HICACHE_SIZE_GB(?![A-Z])', source)), 4)
                self.assertIn('  echo "PENNY_HICACHE_SIZE_GB must be a positive '
                              'integer number of GB', source)


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
        # The configurator's own saved file, not the manual Compose .env: the
        # generated launch files carry the settings themselves, so nothing has
        # to read this file when the container starts.
        self.assertEqual(container,
                         Path.home() / pc.CONTAINER_CONFIG_RELPATH)
        self.assertNotIn("compose.yaml", str(container))
        self.assertEqual(pc.default_config_path(
                             "container", self.repo, home=Path("/fixture-home")),
                         Path("/fixture-home/.config/pennyroyal"
                              "/pennyroyal-container.env"))

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

    def test_every_zero_spelling_of_the_nixl_budget_reads_as_no_cap(self):
        # Accepted zero spellings are the same unlimited budget to the runtime
        # parser, so the summary must never call any of them a GiB cap. A value
        # that is not a number at all (hand-edited input) must still be named
        # without crashing the summary.
        no_cap = ("NIXL disk budget: 0 GiB = no cap; the persistent NIXL "
                  "cache stays enabled and keeps growing")
        parse = runtime_max_cache_gb_parser()
        nixl = pc.specs_for("native")["SGLANG_HICACHE_NIXL_MAX_CACHE_GB"]
        for spelling in ("0", "0.0", "0.00", ".0", "00", "0."):
            with self.subTest(spelling=spelling):
                self.assertEqual(pc.validate_value(nixl, spelling, "t"),
                                 spelling)
                self.assertEqual(parse(spelling, spelling), 0.0)
                plan = self.native_plan({**self.native_env(),
                        "SGLANG_HICACHE_NIXL_MAX_CACHE_GB": spelling})
                self.assertEqual(plan.errors, [])
                self.assertIn(no_cap, plan.summary)
        # A real cap is still reported as a cap, and junk never raises.
        plan = self.native_plan({**self.native_env(),
                "SGLANG_HICACHE_NIXL_MAX_CACHE_GB": "0.5"})
        self.assertIn("NIXL disk budget: 0.5 GiB cap on the persistent NIXL "
                      "cache dirs (the cache stays enabled)", plan.summary)
        for junk in ("junk", "nan", "inf", "200 GiB"):
            with self.subTest(junk=junk):
                self.assertIn(" GiB cap", pc._nixl_budget_line(junk))

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

    # --- WSL2 host-memory workaround (SGLANG_HICACHE_TORCH_PINNED_ALLOC) -----

    def wsl2_spec(self):
        spec = pc.specs_for("native")["SGLANG_HICACHE_TORCH_PINNED_ALLOC"]
        self.assertEqual(spec.kind, "bool")
        self.assertEqual(spec.default, "false")
        self.assertTrue(spec.advanced)
        return spec

    def test_wsl2_flag_is_saved_normalized_and_exported_to_the_recipe(self):
        self.wsl2_spec()
        for answer, wanted in (("yes", "true"), ("1", "true"), ("on", "true"),
                              ("no", "false"), ("0", "false"), ("off", "false"),
                              ("true", "true"), ("FALSE", "false")):
            plan = self.native_plan({**self.native_env(),
                                     "SGLANG_HICACHE_TORCH_PINNED_ALLOC": answer})
            self.assertEqual(plan.env["SGLANG_HICACHE_TORCH_PINNED_ALLOC"], wanted)
            self.assertIn(f"SGLANG_HICACHE_TORCH_PINNED_ALLOC={wanted}",
                          pc.launch_display(plan))

    def test_unset_wsl2_flag_off_by_default_and_inherits_only_when_unsaved(self):
        plan = self.native_plan(self.native_env())
        self.assertEqual(plan.env["SGLANG_HICACHE_TORCH_PINNED_ALLOC"], "false")
        plan = self.native_plan(self.native_env(),
                               environ={"SGLANG_HICACHE_TORCH_PINNED_ALLOC": "true"})
        self.assertEqual(plan.env["SGLANG_HICACHE_TORCH_PINNED_ALLOC"], "true")

    def test_explicit_false_wsl2_flag_beats_an_inherited_true_natively(self):
        plan = self.native_plan(
            {**self.native_env(), "SGLANG_HICACHE_TORCH_PINNED_ALLOC": "false"},
            environ={"SGLANG_HICACHE_TORCH_PINNED_ALLOC": "true"})
        self.assertEqual(plan.env["SGLANG_HICACHE_TORCH_PINNED_ALLOC"], "false")
        self.assertEqual(plan.origins["SGLANG_HICACHE_TORCH_PINNED_ALLOC"],
                         "saved file")

    def test_invalid_wsl2_flag_fails_clearly(self):
        spec = self.wsl2_spec()
        for bad in ("maybe", "2", "true false", "yes please"):
            with self.assertRaises(pc.ConfigError) as caught:
                pc.validate_value(spec, bad, "test")
            self.assertIn("SGLANG_HICACHE_TORCH_PINNED_ALLOC must be true or "
                          f"false, got {bad!r}", str(caught.exception))
        self.write_native_config({**self.native_env(),
                                  "SGLANG_HICACHE_TORCH_PINNED_ALLOC": "maybe"})
        run = subprocess.run([sys.executable, str(SCRIPTS / "penny_config.py"),
                              "--mode", "native", "--config",
                              str(self.config_path), "--check"],
                             capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 2)
        self.assertIn("SGLANG_HICACHE_TORCH_PINNED_ALLOC must be true or false",
                      run.stderr)

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
                        'SGLANG_HICACHE_NIXL_MAX_CACHE_GB '
                        'SGLANG_HICACHE_TORCH_PINNED_ALLOC MY_KEY; do '
                        'printf "%s=%s\\0" "$n" "${!n-}" >> "$CAPTURE"; done\n')
        stub.chmod(0o755)
        self.write_native_config({**self.native_env(), "GPU": "1",
                                 "SGLANG_HICACHE_TORCH_PINNED_ALLOC": "yes",
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
        # The WSL2 workaround is normalized on save and reaches the recipe.
        self.assertEqual(fields["SGLANG_HICACHE_TORCH_PINNED_ALLOC"], "true")
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
        FixtureMixin.setUp(self)   # shared with the generation checks below
        self.compose_dir = self.repo / "docker" / "pennyroyal"
        self.compose_dir.mkdir(parents=True)
        # The beta configurator generates the launch files from the shipped
        # examples, so the fixture repo carries them (by reference).
        (self.compose_dir / "launch").symlink_to(
            ROOT / "docker" / "pennyroyal" / "launch")
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
                "HOST_NIXL_STORAGE_BASE": str(self.base / "nixl"),
                "LAUNCH_DIR": str(self.base / "launch out")}

    def generated(self, plan: pc.Plan) -> dict[Path, str]:
        """The ordinary files a saved container setup would write."""
        return dict(pc.container_launch_files(plan))

    def run_printed_command(self, plan: pc.Plan) -> list[str]:
        """Run the printed command for real, with a docker that only records.

        The generated run.sh is the file an operator starts, so the command the
        plan prints must reach docker with exactly the validated image, mounts,
        port and GPU. The ambient shell deliberately lacks the saved keys (see
        LAUNCH_ENV_KEYS), so nothing but the printed command can supply them.
        No daemon, model or GPU is involved here.
        """
        pc.write_container_files(plan)
        bin_dir, capture = docker_stub(self.base)
        env = {key: value for key, value in os.environ.items()
               if key not in LAUNCH_ENV_KEYS}
        env.update({"PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "DOCKER_CAPTURE": str(capture),
                    "HOME": str(self.base / "isolated-home")})
        run = subprocess.run(plan.next_command, shell=True, cwd="/",
                             capture_output=True, text=True, check=False,
                             env=env)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        return capture.read_text().splitlines()

    def test_default_target_follows_the_selected_profile(self):
        plan = self.container_plan(self.base_values(), profile="next")
        self.assertEqual(plan.env["TARGET_MODEL"],
                         "/models/RadixArk-Qwen3.8-Flash-Next-NVFP4")
        self.assertEqual(Path(plan.argv[plan.argv.index("--startup") + 1]).name,
                         "start-flash-next-frspec.sh")
        plan = self.container_plan(self.base_values(), profile="next-plain")
        self.assertEqual(Path(plan.argv[plan.argv.index("--startup") + 1]).name,
                         "start-flash-next.sh")
        self.assertEqual(Path(plan.argv[plan.argv.index("--nixl-config") + 1]).name,
                         "nixl-posix.toml")
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

    def test_launch_command_names_the_generated_run_sh_and_its_options(self):
        plan = self.container_plan(self.base_values(), profile="27b")
        launch_dir = self.base / "launch out"
        self.assertEqual(Path(plan.argv[0]), launch_dir / "run.sh")
        for option, wanted in (
                ("--startup", str(launch_dir / "config" / "start-27b-dflash2.sh")),
                ("--image", pc.DEFAULT_IMAGE),
                ("--models", str(self.host_root)),
                ("--cache", str(self.base / "cache")),
                ("--port", "8001"),
                ("--gpu", "0"),
                ("--user", "1000:1000"),
                ("--nixl-root", str(self.base / "nixl")),
                ("--nixl-config", str(launch_dir / "config" / "nixl-posix.toml"))):
            self.assertEqual(plan.argv[plan.argv.index(option) + 1], wanted,
                             option)
        # The printed command is exactly that argv: run.sh options beat its own
        # settings, so the launch cannot drift onto an ambient value.
        self.assertEqual(shlex.split(plan.next_command), plan.argv)
        self.assertEqual(plan.env["PENNYROYAL_PROFILE"], "27b")
        self.assertEqual(plan.env["TARGET_MODEL"], "/models/Qwen3.8-27B-FP8")
        # No release number is pinned in the tests: the generated run.sh
        # must carry whatever the default and the saved file resolve to.
        self.assertIn(pc.DEFAULT_IMAGE,
                      self.generated(plan)[self.base / "launch out" / "run.sh"])

    def test_printed_command_reaches_docker_with_the_validated_launch(self):
        plan = self.container_plan({**self.base_values(),
                                    "PENNYROYAL_PORT": "8099",
                                    "NVIDIA_GPU": "1"},
                                   environ={"HOST_MODELS_ROOT": "/bad/ambient",
                                            "PENNYROYAL_PORT": "9999"})
        argv = self.run_printed_command(plan)
        self.assertEqual(argv[0], "run")
        self.assertEqual(argv[argv.index("--user") + 1], "1000:1000")
        self.assertEqual(argv[argv.index("--publish") + 1], "8099:8001")
        self.assertEqual(argv[argv.index("--gpus") + 1], '"device=1"')
        volumes = [argv[item_index + 1] for item_index, item in enumerate(argv)
                   if item == "--volume"]
        self.assertEqual(volumes, [
            f"{self.base / 'launch out' / 'config'}:/config:ro",
            f"{self.host_root}:/models:ro",
            f"{self.base / 'cache'}:/cache",
            f"{self.base / 'nixl'}:/nixl",
        ])
        self.assertNotIn("/bad/ambient", " ".join(argv))
        self.assertEqual(argv[-3:], ["exec", "bash",
                                     "/config/start-flash-next-frspec.sh"])

    def test_container_blank_quota_resets_to_zero_for_the_runtime(self):
        # Same contract in the container: the generated launch gets the real
        # default so the entrypoint's parser never sees '' (which it rejects),
        # while an ambient 200 cannot override the saved reset decision.
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
        # The startup script never mentions the budget, so run.sh has to carry
        # it: nothing outside the launch directory is read at container start.
        self.assertIn("  -e SGLANG_HICACHE_NIXL_MAX_CACHE_GB=0",
                      self.generated(plan)[self.base / "launch out" / "run.sh"])
        # Human-readable summary: lines stay separate and show the reset value.
        self.assertIn("NIXL disk budget: 0 GiB = no cap; the persistent NIXL "
                      "cache stays enabled and keeps growing", plan.summary)
        self.assertIn("runtime identity: 1000:1000", plan.summary)
        self.assertTrue(any(line.startswith("API port: ")
                            for line in plan.summary))

    def test_container_blank_override_keeps_the_scripts_qualified_value(self):
        # A saved blank is 'the recipe decides': the generated startup script
        # keeps its shipped qualified value rather than an empty string, and a
        # conflicting export never reaches it either.
        plan = self.container_plan({**self.base_values(),
                                    "MAX_RUNNING_REQUESTS": ""},
                                   environ={"MAX_RUNNING_REQUESTS": "8"})
        self.assertEqual(plan.env["MAX_RUNNING_REQUESTS"], "")
        startup = self.generated(plan)[
            self.base / "launch out" / "config" / "start-flash-next-frspec.sh"]
        self.assertIn("\nMAX_RUNNING_REQUESTS=4\n", startup)
        self.assertNotIn("MAX_RUNNING_REQUESTS=8", startup)

    def test_saved_values_win_and_ambient_values_pass_through(self):
        plan = self.container_plan({**self.base_values(), "PENNYROYAL_PORT": "8123"},
                                   environ={"SGLANG_FORWARD_UNKNOWN_TOOLS": "false"})
        self.assertEqual(plan.env["PENNYROYAL_PORT"], "8123")
        self.assertEqual(plan.env["SGLANG_FORWARD_UNKNOWN_TOOLS"], "false")

    def test_saved_root_wins_over_conflicting_ambient_env(self):
        (self.host_root / "ambient").mkdir()
        plan = self.container_plan(self.base_values(),
                                   environ={"HOST_MODELS_ROOT": "/bad/ambient",
                                            "TARGET_MODEL": "/models/ambient"})
        # The plan validates the saved values, and the printed command must
        # force them even though the ambient shell exports the conflicting ones.
        self.assertEqual(plan.env["HOST_MODELS_ROOT"], str(self.host_root))
        self.assertEqual(plan.env["TARGET_MODEL"], "/models/ambient")
        self.assertEqual(plan.origins["HOST_MODELS_ROOT"], "saved file")
        self.assertNotIn("/bad/ambient", plan.next_command)
        argv = self.run_printed_command(plan)
        self.assertIn(f"{self.host_root}:/models:ro", argv)
        self.assertNotIn("/bad/ambient", " ".join(argv))
        startup = self.generated(plan)[
            self.base / "launch out" / "config" / "start-flash-next-frspec.sh"]
        self.assertIn("TARGET_MODEL=/models/ambient", startup)

    def test_27b_profile_defaults_land_in_the_generated_startup_script(self):
        plan = self.container_plan(self.base_values(), profile="27b")
        self.assertEqual(plan.env["TARGET_MODEL"], "/models/Qwen3.8-27B-FP8")
        self.assertEqual(plan.env["DRAFT_MODEL"], "/models/Qwen3.8-27B-DFlash2")
        startup = self.generated(plan)[
            self.base / "launch out" / "config" / "start-27b-dflash2.sh"]
        self.assertIn("TARGET_MODEL=/models/Qwen3.8-27B-FP8", startup)
        self.assertIn("DRAFT_MODEL=/models/Qwen3.8-27B-DFlash2", startup)
        self.assertNotIn("RadixArk", startup)
        # The host path with a space survives the generated file and docker.
        argv = self.run_printed_command(plan)
        self.assertIn(f"{self.host_root}:/models:ro", argv)

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
                                     if key not in LAUNCH_ENV_KEYS},
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

    def test_relative_selected_config_is_resolved_once(self):
        # A relative --config (or PENNYROYAL_CONFIG) must be resolved at load,
        # so the file the plan names and validates is the same one a later cwd
        # would not find.
        self.env_file.write_text(pc.serialize_env(
            [("basic", list(self.base_values().items()))],
            header=(f"{pc.PROFILE_KEY}=next",)))
        import contextlib
        with contextlib.chdir(self.env_dir):
            config = pc.load_config("container", Path(".env"), {}, self.repo)
            plan = pc.build_plan("container", config, {}, repo_root=self.repo)
        self.assertEqual(config.path, self.env_file)
        self.assertIn(str(self.env_file), "\n".join(plan.summary))
        argv = self.run_printed_command(plan)
        self.assertIn(f"{self.host_root}:/models:ro", argv)

    def test_missing_env_file_is_a_warning_not_a_crash(self):
        config = pc.load_config("container", self.env_dir / ".env", {}, self.repo)
        plan = pc.build_plan("container", config, {}, repo_root=self.repo)
        self.assertIn("not exist yet", self.messages(plan, "warn"))
        # Nothing generated yet is a note about the next step, not an error:
        # the only errors are the roots this fixture never set.
        self.assertIn("no generated launch yet", self.messages(plan, "warn"))
        self.assertEqual({issue.key for issue in plan.errors},
                         {"HOST_MODELS_ROOT", "HOST_CACHE_BASE",
                          "HOST_NIXL_STORAGE_BASE"})


class ContainerLaunchGenerationTests(FixtureMixin):
    """The generated files, and the disk-tier choice that has to agree in both.

    run.sh decides whether /nixl is mounted; the startup script decides whether
    the NIXL storage backend is passed at all. One saved switch drives both, so
    this shares the container fixture and its helpers instead of a second one.
    """

    setUp = ContainerPlanTests.setUp
    container_plan = ContainerPlanTests.container_plan
    base_values = ContainerPlanTests.base_values
    generated = ContainerPlanTests.generated
    run_printed_command = ContainerPlanTests.run_printed_command

    def test_disk_tier_is_on_by_default_in_both_generated_files(self):
        plan = self.container_plan(self.base_values())
        self.assertEqual(plan.env["NIXL"], "on")
        files = self.generated(plan)
        self.assertIn("\nNIXL=on\n", files[self.base / "launch out" / "run.sh"])
        startup = files[self.base / "launch out" / "config"
                        / "start-flash-next-frspec.sh"]
        self.assertIn("\nNIXL=on\n", startup)
        # on is the qualified default: the NIXL TOML is written next to it.
        self.assertTrue((self.base / "launch out" / "config"
                         / "nixl-posix-frspec.toml") in files)
        self.assertIn("--nixl-root", plan.argv)

    def test_disk_tier_off_aligns_the_mount_and_the_runtime_arguments(self):
        for profile, script in (("next", "start-flash-next-frspec.sh"),
                                ("next-plain", "start-flash-next.sh"),
                                ("27b", "start-27b-dflash2.sh")):
            with self.subTest(profile=profile):
                plan = self.container_plan(
                    {**self.base_values(), "NIXL": "off",
                     "HOST_NIXL_STORAGE_BASE": ""}, profile=profile)
                # No root is required, so nothing about a missing one is held
                # against the launch, and nothing in it is deleted either.
                self.assertEqual(plan.errors, [])
                self.assertNotIn("NIXL", self.messages(plan, "warn"))
                self.assertIn("--no-nixl", plan.argv)
                self.assertNotIn("--nixl-root", plan.argv)
                self.assertNotIn("--nixl-config", plan.argv)
                files = self.generated(plan)
                run_sh = files[self.base / "launch out" / "run.sh"]
                startup = files[self.base / "launch out" / "config" / script]
                self.assertIn("\nNIXL=off\n", run_sh)
                self.assertIn("\nNIXL=off\n", startup)
                # The backend config the startup script drops is not shipped
                # either, and the RAM tier line it keeps is untouched.
                self.assertFalse(any(path.name.startswith("nixl-")
                                    for path in files))
                self.assertNotIn("SGLANG_HICACHE_NIXL_MAX_CACHE_GB", run_sh)
                self.assertIn("--enable-hierarchical-cache", startup)
                self.assertIn("GPU radix cache and host-RAM HiCache stay on",
                              "\n".join(plan.summary))
                self.assertEqual(self.run_printed_command(plan)
                                 [-3:], ["exec", "bash", f"/config/{script}"])

    def test_disk_tier_off_is_not_an_unlimited_budget(self):
        # A 0 budget means the cache is on with no cap; only NIXL=off removes
        # it, and the summary must never call the two the same thing.
        plan = self.container_plan({**self.base_values(), "NIXL": "off",
                                    "SGLANG_HICACHE_NIXL_MAX_CACHE_GB": "0"})
        self.assertNotIn("NIXL disk budget", "\n".join(plan.summary))
        plan = self.container_plan({**self.base_values(), "NIXL": "on",
                                    "SGLANG_HICACHE_NIXL_MAX_CACHE_GB": "0"})
        summary = "\n".join(plan.summary)
        self.assertIn("NIXL disk tier: on", summary)
        self.assertIn("NIXL disk budget: 0 GiB = no cap", summary)

    def test_qualified_27b_knobs_stay_with_their_recipe(self):
        # The 27b recipe pins its capacity, TP and PLE placement in its own
        # launch line, so a saved choice there cannot move the server: the plan
        # refuses it and the generated file keeps the shipped literals instead of
        # pretending the operator's value is in charge.
        plan = self.container_plan({**self.base_values(),
                                    "MAX_RUNNING_REQUESTS": "6",
                                    "TP_SIZE": "2"}, profile="27b")
        self.assertIn("is not a setting of the 27b profile",
                      self.messages(plan, "error"))
        startup = self.generated(plan)[
            self.base / "launch out" / "config" / "start-27b-dflash2.sh"]
        self.assertNotIn("MAX_RUNNING_REQUESTS=6", startup)
        self.assertIn("\nTP_SIZE=1\n", startup)
        self.assertNotIn("--nvme-ple", plan.argv)
        # On the Next profile the same saved TP is a real setting and is written.
        plan = self.container_plan({**self.base_values(), "TP_SIZE": "2"})
        self.assertEqual(plan.errors, [])
        startup = self.generated(plan)[
            self.base / "launch out" / "config" / "start-flash-next-frspec.sh"]
        self.assertIn("\nTP_SIZE=2\n", startup)

    def test_nvme_ple_keeps_its_own_independent_choice(self):
        (self.base / "ple snapshot").mkdir()
        plan = self.container_plan({**self.base_values(), "NIXL": "off",
                                    "PENNY_PLE_BACKEND": "nvme",
                                    "PENNY_PLE_NVME_MODEL":
                                        str(self.base / "ple snapshot")})
        self.assertIn("--nvme-ple", plan.argv)
        run_sh = self.generated(plan)[self.base / "launch out" / "run.sh"]
        self.assertIn("\nNVME_PLE=on\n", run_sh)
        self.assertIn("\nNIXL=off\n", run_sh)
        # io_uring stays permitted for the PLE reader without the disk tier.
        self.assertIn("seccomp=unconfined", " ".join(self.run_printed_command(plan)))

    def test_host_paths_with_spaces_and_dollars_survive_generation(self):
        odd = self.base / "share $models [v1]"
        odd.mkdir()
        (odd / "RadixArk-Qwen3.8-Flash-Next-NVFP4").mkdir()
        values = {**self.base_values(), "HOST_MODELS_ROOT": str(odd)}
        plan = self.container_plan(values)
        self.assertEqual(plan.errors, [])
        run_sh = self.generated(plan)[self.base / "launch out" / "run.sh"]
        self.assertIn(f"HOST_MODELS_ROOT={shlex.quote(str(odd))}", run_sh)
        argv = self.run_printed_command(plan)
        self.assertIn(f"{odd}:/models:ro", argv)

    def test_a_broken_template_is_named_and_never_guessed(self):
        empty = self.base / "no-templates"
        (empty / "configs" / "pennyroyal").mkdir(parents=True)
        config = pc.load_config("container", self.env_file, {}, self.repo)
        plan = pc.build_plan("container", config, {}, repo_root=empty)
        self.assertIn("launch template is missing", self.messages(plan, "error"))
        with self.assertRaises(pc.ConfigError):
            pc.container_launch_files(plan)

    def test_saving_twice_does_not_clobber_an_edited_file(self):
        plan = self.container_plan(self.base_values())
        pc.write_container_files(plan)
        run_sh = self.base / "launch out" / "run.sh"
        original = run_sh.read_text()
        run_sh.write_text(original + "\n# operator edit\n")
        asked: list[str] = []
        outcomes = dict(pc.write_container_files(
            plan, confirm=lambda text: asked.append(text) or False))
        self.assertEqual(len(asked), 1, asked)
        self.assertIn("run.sh", asked[0])
        self.assertEqual(outcomes[run_sh], "kept your edited copy")
        self.assertEqual(run_sh.read_text(), original + "\n# operator edit\n")
        # A file whose content already matches is never rewritten at all.
        startup = (self.base / "launch out" / "config"
                   / "start-flash-next-frspec.sh")
        started = startup.stat().st_mtime_ns
        self.assertEqual(dict(pc.write_container_files(plan))[startup],
                         "unchanged")
        self.assertEqual(startup.stat().st_mtime_ns, started)
        # Confirming the overwrite is what replaces the operator's edit.
        run_sh.write_text(original + "\n# operator edit\n")
        outcomes = dict(pc.write_container_files(plan, confirm=lambda text: True))
        self.assertEqual(outcomes[run_sh], "written")
        self.assertEqual(run_sh.read_text(), original)


    def test_saved_forward_unknown_tools_reaches_the_container(self):
        # A saved false must arrive as false in the launched process, not be
        # rewritten to the script's qualified default on the way.
        plan = self.container_plan({**self.base_values(),
                                    "SGLANG_FORWARD_UNKNOWN_TOOLS": "false"},
                                   environ={"SGLANG_FORWARD_UNKNOWN_TOOLS":
                                            "true"})
        self.assertEqual(plan.env["SGLANG_FORWARD_UNKNOWN_TOOLS"], "false")
        run_sh = self.generated(plan)[self.base / "launch out" / "run.sh"]
        self.assertIn("  -e SGLANG_FORWARD_UNKNOWN_TOOLS=false", run_sh)
        startup = self.generated(plan)[
            self.base / "launch out" / "config" / "start-flash-next-frspec.sh"]
        # The mounted script keeps the value it was handed instead of exporting
        # its own literal over it.
        self.assertIn('export SGLANG_FORWARD_UNKNOWN_TOOLS='
                      '"${SGLANG_FORWARD_UNKNOWN_TOOLS:-true}"', startup)
        self.assertNotIn("export SGLANG_FORWARD_UNKNOWN_TOOLS=true\n", startup)
        argv = self.run_printed_command(plan)
        # docker is handed the saved value as its own -e argument.
        self.assertIn("-e", argv)
        self.assertIn("SGLANG_FORWARD_UNKNOWN_TOOLS=false", argv)
        self.assertNotIn("SGLANG_FORWARD_UNKNOWN_TOOLS=true", argv)

    def test_unsupported_27b_settings_are_refused_not_ignored(self):
        # The 27b recipe pins 4/24, TP1 and RAM PLE in its own launch line, so a
        # saved choice there would be forwarded and then quietly dropped.
        for name, value in (("MAX_RUNNING_REQUESTS", "8"),
                            ("MAX_MAMBA_CACHE_SIZE", "48"),
                            ("MAX_TOTAL_TOKENS", "262144"),
                            ("TP_SIZE", "2"),
                            ("PENNY_PLE_BACKEND", "nvme")):
            with self.subTest(name=name):
                plan = self.container_plan({**self.base_values(), name: value},
                                           profile="27b")
                messages = self.messages(plan, "error")
                self.assertIn(f"{name}={value}", messages)
                self.assertIn("is not a setting of the 27b profile", messages)
                if value != "262144":
                    # Where the recipe does have a literal, name it: the
                    # operator learns what actually runs.
                    self.assertIn("launches with", messages)

    def test_27b_accepts_what_its_recipe_actually_uses(self):
        # Compatible saved values (the profile's own numbers) and an unset key
        # stay silent: nothing here invents 27b tuning or blocks an older file.
        plan = self.container_plan({**self.base_values(),
                                    "MAX_RUNNING_REQUESTS": "4",
                                    "MAX_MAMBA_CACHE_SIZE": "24",
                                    "TP_SIZE": "1"}, profile="27b")
        self.assertEqual(plan.errors, [])
        plan = self.container_plan(self.base_values(), profile="27b")
        self.assertEqual(plan.errors, [])
        # The Next profiles read every one of these, so they stay available.
        plan = self.container_plan({**self.base_values(),
                                    "MAX_RUNNING_REQUESTS": "8",
                                    "MAX_MAMBA_CACHE_SIZE": "48",
                                    "MAX_TOTAL_TOKENS": "262144",
                                    "TP_SIZE": "2"})
        self.assertEqual(plan.errors, [])
        startup = self.generated(plan)[
            self.base / "launch out" / "config" / "start-flash-next-frspec.sh"]
        self.assertIn("\nMAX_RUNNING_REQUESTS=8\n", startup)
        self.assertIn("\nTP_SIZE=2\n", startup)
        run_sh = self.generated(plan)[self.base / "launch out" / "run.sh"]
        self.assertIn("  -e MAX_TOTAL_TOKENS=262144", run_sh)

    def test_nvme_ple_snapshot_is_checked_like_a_model_path(self):
        # The snapshot is read inside the container, so an arbitrary host
        # directory has to fail here and not at the first cache miss.
        plan = self.container_plan({**self.base_values(),
                                    "PENNY_PLE_BACKEND": "nvme",
                                    "PENNY_PLE_NVME_MODEL": "/srv/elsewhere/ple"})
        self.assertIn("PENNY_PLE_NVME_MODEL must live under /models/",
                      self.messages(plan, "error"))
        plan = self.container_plan({**self.base_values(),
                                    "PENNY_PLE_BACKEND": "nvme",
                                    "PENNY_PLE_NVME_MODEL": "/models/not-prepared"})
        self.assertIn("does not exist under the models mount",
                      self.messages(plan, "error"))
        plan = self.container_plan({**self.base_values(),
                                    "PENNY_PLE_BACKEND": "nvme"})
        self.assertIn("PENNY_PLE_BACKEND=nvme needs PENNY_PLE_NVME_MODEL",
                      self.messages(plan, "error"))
        prepared = self.host_root / "flash-next-ple"
        prepared.mkdir(exist_ok=True)
        plan = self.container_plan({**self.base_values(),
                                    "PENNY_PLE_BACKEND": "nvme",
                                    "PENNY_PLE_NVME_MODEL":
                                        "/models/flash-next-ple"})
        self.assertEqual(plan.errors, [])
        self.assertIn("--nvme-ple", plan.argv)
        run_sh = self.generated(plan)[self.base / "launch out" / "run.sh"]
        self.assertIn("\nNVME_PLE=on\n", run_sh)
        self.assertIn("  -e PENNY_PLE_NVME_MODEL=/models/flash-next-ple", run_sh)

    def test_nvme_ple_stays_independent_of_the_disk_tier(self):
        # Turning the disk tier off must not ask for a NIXL root the PLE reader
        # never uses, and must not drop the io_uring permission it does need.
        (self.host_root / "flash-next-ple").mkdir(exist_ok=True)
        plan = self.container_plan({**self.base_values(), "NIXL": "off",
                                    "HOST_NIXL_STORAGE_BASE": "",
                                    "PENNY_PLE_BACKEND": "nvme",
                                    "PENNY_PLE_NVME_MODEL":
                                        "/models/flash-next-ple"})
        self.assertEqual(plan.errors, [])
        self.assertIn("--no-nixl", plan.argv)
        self.assertIn("--nvme-ple", plan.argv)
        files = self.generated(plan)
        run_sh = files[self.base / "launch out" / "run.sh"]
        self.assertIn("  -e PENNY_PLE_NVME_MODEL=/models/flash-next-ple", run_sh)
        self.assertFalse(any(path.name.startswith("nixl-") for path in files))


# --- generated launch files are written as one set -------------------------


    def test_generated_launch_keeps_every_knob_the_compose_path_forwards(self):
        # The manual compose file is the existing contract for what a container
        # may be told. Anything it forwards has to arrive by one of the two
        # generated routes, or the beta path silently loses a supported setting.
        compose = (ROOT / pc.COMPOSE_RELPATH).read_text()
        block = compose.split("environment:", 1)[1].split("volumes:", 1)[0]
        forwarded = [line.split(":")[0].strip() for line in block.splitlines()
                     if line.strip() and not line.strip().startswith("#")
                     and ":" in line]
        # Settings the startup script states in its own block instead.
        owned_by_script = {
            "TARGET_MODEL", "DRAFT_MODEL", "CACHE_BASE", "NIXL_STORAGE_BASE",
            "PENNY_HICACHE_SIZE_GB", "PENNY_PLE_BACKEND",
            "SGLANG_MM_PREPROCESS_DEVICE", "TP_SIZE", "MAX_RUNNING_REQUESTS",
            "MAX_MAMBA_CACHE_SIZE", pc.PROFILE_KEY.upper()}
        missing = [name for name in forwarded
                   if name not in owned_by_script
                   and name not in pc.CONTAINER_PASSTHROUGH_KEYS]
        self.assertEqual(missing, [], f"dropped knobs: {missing}")

        # And the two that are neither managed keys nor script settings still
        # reach the container as a saved or inherited value would.
        plan = self.container_plan({**self.base_values(),
                                    "PENNY_REASONING_EFFORT": "high",
                                    "NCCL_P2P_DISABLE": "1"},
                                   environ={"PENNY_REASONING_EFFORT": "low"})
        self.assertEqual(plan.env["PENNY_REASONING_EFFORT"], "high")
        run_sh = self.generated(plan)[self.base / "launch out" / "run.sh"]
        self.assertIn("  -e PENNY_REASONING_EFFORT=high", run_sh)
        self.assertIn("  -e NCCL_P2P_DISABLE=1", run_sh)
        argv = self.run_printed_command(plan)
        self.assertIn("PENNY_REASONING_EFFORT=high", argv)
        self.assertIn("NCCL_P2P_DISABLE=1", argv)

    def test_a_saved_key_nothing_reads_is_named_not_promise(self):
        # plan.env carries unknown keys for the native launcher; in the
        # container they cannot silently become part of the launch.
        plan = self.container_plan({**self.base_values(),
                                    "MY_OWN_KNOB": "yes"})
        self.assertEqual(plan.errors, [])
        self.assertIn("MY_OWN_KNOB=yes is saved but no Pennyroyal container "
                      "launch file reads it", self.messages(plan, "warn"))

    def test_the_nixl_toml_is_only_required_while_the_tier_is_mounted(self):
        # A checkout without the profile's NIXL TOML still plans an off launch:
        # nothing in that path reads the file, so it must not be demanded.
        empty = self.base / "partial-templates"
        (empty / "docker" / "pennyroyal" / "launch" / "config").mkdir(
            parents=True)
        for relative in ("run.sh", "config/start-flash-next.sh"):
            source = ROOT / pc.LAUNCH_RELPATH / relative
            target = empty / "docker" / "pennyroyal" / "launch" / relative
            target.write_text(source.read_text())
        values = {**self.base_values(), "NIXL": "off",
                  "HOST_NIXL_STORAGE_BASE": ""}
        path = self.base / "partial.env"
        path.write_text(pc.serialize_env([("", sorted(values.items()))],
                                         header=(f"{pc.PROFILE_KEY}=next-plain",)))
        config = pc.load_config("container", path, {}, empty)
        plan = pc.build_plan("container", config, {}, repo_root=empty)
        self.assertEqual([issue.message for issue in plan.errors], [])
        self.assertIn("\nNIXL=off\n",
                      dict(pc.container_launch_files(plan))[
                          plan.launch_dir / "run.sh"])
        values["NIXL"] = "on"
        path.write_text(pc.serialize_env([("", sorted(values.items()))],
                                         header=(f"{pc.PROFILE_KEY}=next-plain",)))
        on_plan = pc.build_plan(
            "container", pc.load_config("container", path, {}, empty), {},
            repo_root=empty)
        self.assertIn("launch template is missing", self.messages(on_plan,
                                                                 "error"))


class NativeDiskTierPlanTests(FixtureMixin):
    """The same on/off choice on the native path: the recipe decides nothing
    about the disk tier by itself when the saved settings say otherwise."""

    def test_disk_tier_is_on_by_default(self):
        plan = self.native_plan(self.native_env())
        self.assertEqual(plan.env["NIXL"], "on")
        self.assertIn("NIXL disk tier: on", "\n".join(plan.summary))
        self.assertIn("NIXL disk budget: 0 GiB = no cap", "\n".join(plan.summary))

    def test_off_needs_no_nixl_root_and_keeps_everything_else(self):
        plan = self.native_plan({**self.native_env(), "NIXL": "off",
                                 "NIXL_STORAGE_BASE": ""})
        self.assertEqual(plan.errors, [])
        self.assertEqual(plan.env["NIXL"], "off")
        summary = "\n".join(plan.summary)
        self.assertIn("NIXL disk tier: off", summary)
        self.assertNotIn("NIXL disk budget", summary)
        # The RAM tier and the recipe choice are untouched by the switch.
        self.assertIn("RAM cache (HiCache): 32 GB as --hicache-size", summary)
        self.assertEqual(Path(plan.argv[0]).name, "serve-flash-next-frspec.sh")

    def test_off_does_not_report_a_missing_nixl_root(self):
        blocker = self.base / "nixl blocker"
        blocker.write_text("x")
        plan = self.native_plan({**self.native_env(), "NIXL": "off",
                                 "NIXL_STORAGE_BASE": str(blocker)})
        self.assertEqual(self.messages(plan, "error"), "")
        plan = self.native_plan({**self.native_env(), "NIXL": "on",
                                 "NIXL_STORAGE_BASE": str(blocker)})
        self.assertIn("not-a-directory", self.messages(plan, "error"))

    def test_gpu_index_and_capacity_still_reach_the_recipe(self):
        plan = self.native_plan({**self.native_env(), "NIXL": "off",
                                 "TP_SIZE": "2", "GPU": "1"})
        self.assertEqual(plan.env["TP_SIZE"], "2")
        self.assertEqual(plan.env["CUDA_VISIBLE_DEVICES"], "1")

    # --- WSL2 host-memory workaround (SGLANG_HICACHE_TORCH_PINNED_ALLOC) -----

    def test_compose_file_carries_the_wsl2_pinned_alloc_with_an_off_default(self):
        text = (ROOT / pc.COMPOSE_RELPATH).read_text()
        self.assertIn(
            "SGLANG_HICACHE_TORCH_PINNED_ALLOC: "
            "${SGLANG_HICACHE_TORCH_PINNED_ALLOC:-false}", text)
        self.assertIn("#SGLANG_HICACHE_TORCH_PINNED_ALLOC=false",
                      (ROOT / "docker/pennyroyal/.env.example").read_text())

    def test_saved_wsl2_flag_reaches_the_rendered_container_environment(self):
        plan = self.container_plan({**self.base_values(),
                                    "SGLANG_HICACHE_TORCH_PINNED_ALLOC": "true"},
                                   environ={"SGLANG_HICACHE_TORCH_PINNED_ALLOC":
                                            "false"})
        # The saved file wins over the inherited shell value.
        self.assertEqual(plan.env["SGLANG_HICACHE_TORCH_PINNED_ALLOC"], "true")
        self.assertEqual(plan.forced_env["SGLANG_HICACHE_TORCH_PINNED_ALLOC"],
                         "true")
        self.assertIn("SGLANG_HICACHE_TORCH_PINNED_ALLOC=true", plan.next_command)
        rendered = self.run_printed_command(plan)
        self.assertIn('SGLANG_HICACHE_TORCH_PINNED_ALLOC: "true"', rendered)

    def test_explicit_false_wsl2_flag_beats_an_inherited_true(self):
        plan = self.container_plan(
            {**self.base_values(), "SGLANG_HICACHE_TORCH_PINNED_ALLOC": "false"},
            environ={"SGLANG_HICACHE_TORCH_PINNED_ALLOC": "true"})
        self.assertEqual(plan.env["SGLANG_HICACHE_TORCH_PINNED_ALLOC"], "false")
        self.assertIn('SGLANG_HICACHE_TORCH_PINNED_ALLOC: "false"',
                      self.run_printed_command(plan))

    def test_unset_wsl2_flag_keeps_the_inherited_value_or_the_off_default(self):
        # Not saved at all: the inherited value is carried to the container.
        plan = self.container_plan(
            self.base_values(),
            environ={"SGLANG_HICACHE_TORCH_PINNED_ALLOC": "true"})
        self.assertEqual(plan.env["SGLANG_HICACHE_TORCH_PINNED_ALLOC"], "true")
        self.assertIn('SGLANG_HICACHE_TORCH_PINNED_ALLOC: "true"',
                      self.run_printed_command(plan))
        # Nothing saved and nothing inherited: the documented off default.
        plan = self.container_plan(self.base_values())
        self.assertEqual(plan.env["SGLANG_HICACHE_TORCH_PINNED_ALLOC"], "false")

    def test_saved_blank_wsl2_flag_resets_to_off_and_suppresses_the_shell(self):
        plan = self.container_plan(
            {**self.base_values(), "SGLANG_HICACHE_TORCH_PINNED_ALLOC": ""},
            environ={"SGLANG_HICACHE_TORCH_PINNED_ALLOC": "true"})
        self.assertEqual(plan.env["SGLANG_HICACHE_TORCH_PINNED_ALLOC"], "false")
        self.assertIn("documented default", plan.origins[
            "SGLANG_HICACHE_TORCH_PINNED_ALLOC"])



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


    def test_a_blocked_destination_fails_the_whole_save_set(self):
        # The save writes settings, run.sh and the startup script as one set:
        # an existing directory where a file belongs must stop every one of
        # them, not land the first two and raise on the third.
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings, launch = root / "pennyroyal.env", root / "launch"
            (launch / "config").mkdir(parents=True)
            run_sh = launch / "run.sh"
            run_sh.write_text("#!/bin/sh\nold\n")
            run_sh.chmod(0o755)
            startup = launch / "config" / "start-flash-next.sh"
            startup.mkdir()                      # the operator's obstruction
            with self.assertRaises(pc.ConfigError) as caught:
                pc.commit_files([(settings, "NIXL=off\n"),
                                 (run_sh, "#!/bin/sh\nnew\n"),
                                 (startup, "#!/usr/bin/env bash\n")],
                                private=(settings,))
            self.assertIn("start-flash-next.sh", str(caught.exception))
            self.assertFalse(settings.exists(), "settings must not land")
            self.assertEqual(run_sh.read_text(), "#!/bin/sh\nold\n")
            # Nothing half-applied and no temporary litter anywhere in the set.
            self.assertEqual(sorted(p.name for p in root.rglob("*") if p.is_file()),
                             ["run.sh"])
            self.assertEqual([p.name for p in startup.iterdir()], [])

    def test_replacement_uses_the_documented_modes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config").mkdir()
            existing = root / "kept.env"
            existing.write_text("A=1\n")
            existing.chmod(0o640)
            fresh = root / "fresh.env"
            script = root / "config" / "start-flash-next.sh"
            toml = root / "config" / "nixl-posix.toml"
            pc.commit_files([(existing, "A=2\n"), (fresh, "A=1\n"),
                             (script, "#!/usr/bin/env bash\n"),
                             (toml, "backend = \"posix\")\n")],
                            private=(fresh,))
            self.assertEqual(existing.stat().st_mode & 0o777, 0o640,
                             "an existing file keeps its own permissions")
            self.assertEqual(fresh.stat().st_mode & 0o777, 0o600)
            self.assertEqual(script.stat().st_mode & 0o777, 0o755)
            self.assertEqual(toml.stat().st_mode & 0o777, 0o644)
            self.assertEqual(dict(pc.commit_files([(existing, "A=2\n")])),
                             {existing: "unchanged"},
                             "matching content is left alone")

    def test_a_failed_replacement_restores_what_was_already_moved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = root / "a.sh", root / "b.sh"
            first.write_text("old a\n")
            second.write_text("old b\n")
            real_replace = pc.os.replace
            attempts = {"n": 0}

            def flaky(src, dst):
                attempts["n"] += 1
                if attempts["n"] == 2:
                    raise OSError(21, "Is a directory")
                return real_replace(src, dst)

            pc.os.replace = flaky
            try:
                with self.assertRaises(pc.ConfigError) as caught:
                    pc.commit_files([(first, "new a\n"), (second, "new b\n")])
            finally:
                pc.os.replace = real_replace
            self.assertIn("b.sh", str(caught.exception))
            self.assertEqual(first.read_text(), "old a\n",
                             "the file already replaced has to come back")
            self.assertEqual(second.read_text(), "old b\n")
            self.assertEqual(sorted(p.name for p in root.iterdir()),
                             ["a.sh", "b.sh"])

    def test_a_rollback_renames_the_backup_so_read_only_and_crlf_survive(self):
        # Restoring has to rename the previous file back, not write text over
        # it: a 0444 destination cannot be rewritten at all, and a file whose
        # lines end CRLF must keep those bytes rather than a normalized copy.
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kept = root / "a.sh"
            blocked = root / "b.sh"
            kept.write_bytes(b"old a\r\nsecond\r\n")
            kept.chmod(0o444)
            blocked.write_bytes(b"old b\n")
            real_replace = pc.os.replace
            attempts = {"n": 0}

            def flaky(src, dst):
                attempts["n"] += 1
                if attempts["n"] == 2:
                    raise OSError(21, "Is a directory")
                return real_replace(src, dst)

            pc.os.replace = flaky
            try:
                with self.assertRaises(pc.ConfigError) as caught:
                    pc.commit_files([(kept, "new a\n"), (blocked, "new b\n")])
            finally:
                pc.os.replace = real_replace
            self.assertIn("b.sh", str(caught.exception))
            self.assertEqual(kept.read_bytes(), b"old a\r\nsecond\r\n",
                             "the previous bytes come back exactly, CRLF and all")
            self.assertEqual(kept.stat().st_mode & 0o777, 0o444,
                             "a read-only file stays read-only through the "
                             "rollback")
            self.assertEqual(blocked.read_bytes(), b"old b\n")
            self.assertEqual(sorted(path.name for path in root.iterdir()),
                             ["a.sh", "b.sh"],
                             "no backup or temporary file survives the refusal")

    def test_identical_content_with_crlf_is_left_alone_as_unchanged(self):
        # Comparison happens on bytes, so a file that already holds the payload
        # is not rewritten (and its CRLF endings are not silently normalized).
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_bytes(b"backend = \"posix\"\r\n")
            before = path.stat()
            self.assertEqual(dict(pc.commit_files([(path, 'backend = "posix"\r\n')])),
                             {path: "unchanged"})
            self.assertEqual(path.read_bytes(), b"backend = \"posix\"\r\n")
            self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(path.stat().st_mode & 0o777, before.st_mode & 0o777)

    def test_a_full_disk_refuses_the_save_and_leaves_no_temporary(self):
        # The temporary has to be registered before the write and fsync, or
        # ENOSPC escapes with an invisible file left in the operator's folder.
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = root / "pennyroyal.env"
            run_sh = root / "run.sh"
            run_sh.write_bytes(b"#!/bin/sh\nold\n")
            real_fsync = pc.os.fsync

            def no_space(fd):
                raise OSError(28, "No space left on device")

            pc.os.fsync = no_space
            try:
                with self.assertRaises(pc.ConfigError) as caught:
                    pc.commit_files([(settings, "NIXL=off\n"),
                                     (run_sh, "#!/bin/sh\nnew\n")],
                                    private=(settings,))
            finally:
                pc.os.fsync = real_fsync
            self.assertIn("No space left on device", str(caught.exception))
            self.assertFalse(settings.exists())
            self.assertEqual(run_sh.read_bytes(), b"#!/bin/sh\nold\n")
            self.assertEqual([path.name for path in root.iterdir()], ["run.sh"],
                             "nothing hidden, nothing half-written")


    def test_the_same_destination_named_twice_is_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run.sh"
            with self.assertRaises(pc.ConfigError):
                pc.commit_files([(path, "a\n"), (path, "b\n")])
            self.assertFalse(path.exists())


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


class ContainerGenerationConsentTests(FixtureMixin):
    """Consent covers the related set, so a decline cannot half-apply it."""

    def setUp(self) -> None:
        super().setUp()
        (self.repo / "docker" / "pennyroyal").mkdir(parents=True, exist_ok=True)
        (self.repo / "docker" / "pennyroyal" / "launch").symlink_to(
            ROOT / "docker" / "pennyroyal" / "launch")
        self.host_root = self.base / "host models"
        self.host_root.mkdir()
        (self.host_root / "RadixArk-Qwen3.8-Flash-Next-NVFP4").mkdir()
        (self.base / "cache").mkdir(exist_ok=True)
        (self.base / "nixl").mkdir(exist_ok=True)
        self.launch_dir = self.base / "launch out"

    def plan(self, nixl: str) -> pc.Plan:
        values = {"HOST_MODELS_ROOT": str(self.host_root),
                  "HOST_CACHE_BASE": str(self.base / "cache"),
                  "HOST_NIXL_STORAGE_BASE": str(self.base / "nixl"),
                  "LAUNCH_DIR": str(self.launch_dir), "NIXL": nixl}
        path = self.base / f"{nixl}.env"
        path.write_text(pc.serialize_env([("", sorted(values.items()))],
                                         header=(f"{pc.PROFILE_KEY}=next",)))
        return pc.build_plan("container", pc.load_config("container", path, {},
                                                         self.repo), {},
                             repo_root=self.repo)

    def tracked(self) -> list[Path]:
        return [self.launch_dir / "run.sh",
                self.launch_dir / "config" / "start-flash-next-frspec.sh"]

    def test_declining_leaves_the_whole_set_exactly_as_it_was(self):
        # The repro: run.sh wants the new disk-tier choice, the startup script
        # was edited by hand. Consent is for the set, before anything is
        # written, so a decline cannot pair run.sh's new choice with the old
        # startup file -- the two would disagree about the mount and the flags.
        on = self.plan("on")
        pc.write_container_files(on)
        before = {path: path.read_text() for path in self.tracked()}
        assert "NIXL=on" in before[self.launch_dir / "run.sh"]
        startup = self.launch_dir / "config" / "start-flash-next-frspec.sh"
        startup.write_text(before[startup] + "\n# my own edit\n")
        before = {path: path.read_text() for path in self.tracked()}
        asked: list[str] = []
        outcomes = dict(pc.write_container_files(
            self.plan("off"),
            confirm=lambda text: asked.append(text) or False))
        # One question naming every file that would be replaced, asked once.
        self.assertEqual(len(asked), 1, asked)
        self.assertIn("run.sh", asked[0])
        self.assertIn("start-flash-next-frspec.sh", asked[0])
        for path, text in before.items():
            self.assertEqual(path.read_text(), text, path)
        self.assertEqual(set(outcomes.values()), {"kept your edited copy"})
        # Nothing half-applied: the launcher still matches the kept script.
        self.assertIn("NIXL=on", (self.launch_dir / "run.sh").read_text())

    def test_cancel_at_the_question_writes_nothing(self):
        on = self.plan("on")
        pc.write_container_files(on)
        edited = self.launch_dir / "run.sh"
        edited.write_text(edited.read_text() + "\n# mine\n")
        before = {path: path.read_text() for path in self.tracked()}

        def cancel(_text: str) -> bool:
            # The wizard's Prompt raises this on 'q' or end of input.
            raise _Cancelled()

        with self.assertRaises(_Cancelled):
            pc.write_container_files(self.plan("off"), confirm=cancel)
        for path, text in before.items():
            self.assertEqual(path.read_text(), text, path)

    def test_confirming_replaces_the_whole_set_together(self):
        pc.write_container_files(self.plan("on"))
        startup = self.launch_dir / "config" / "start-flash-next-frspec.sh"
        startup.write_text(startup.read_text() + "\n# mine\n")
        outcomes = dict(pc.write_container_files(self.plan("off"),
                                                confirm=lambda _text: True))
        self.assertEqual(set(outcomes.values()), {"written"})
        self.assertIn("NIXL=off", (self.launch_dir / "run.sh").read_text())
        self.assertIn("NIXL=off", startup.read_text())
        self.assertNotIn("# mine", startup.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
