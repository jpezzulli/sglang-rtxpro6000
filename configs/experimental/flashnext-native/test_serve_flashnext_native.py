#!/usr/bin/env python3
"""CPU-only guards for the experimental Flash-Next native launch recipe.

Runs with any Python 3.11+ from the repository root (or under pytest):

    python3 configs/experimental/flashnext-native/test_serve_flashnext_native.py

The recipe is a hand-mapped port of the v3.1.4 Pennyroyal Flash-Next launch.
These checks keep that mapping honest on this base: every flag it passes must
exist on this checkout's CLI, features this base does not have must stay out
of it (documented, not silently dropped), and the template, parser, YaRN,
served-id and cache-identity settings must survive the port. The recipe is
EXECUTED against a recording fake CLI, so assertions run on the emitted argv
and the real bash expansions, not on literals. The cache-identity checks drive
the recipe's real FP8 identity adapter with an interpreter whose torch reports a
chosen device, because that answer -- not the requested Boolean -- decides the
namespace. Nothing here imports sglang, torch or CUDA in the test process and
nothing touches a GPU.
"""

import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SCRIPT = (HERE / "serve-flashnext-native.sh").read_text()
NAMESPACE_HELPER = (REPO_ROOT / "scripts/pennyroyal/derive_namespace.py").read_text()
FP8_IDENTITY_HELPER = (HERE / "online_fp8_identity.py").read_text()
# the launcher's own non-sglang flag surface: the namespace helper's and the FP8
# identity adapter's argparse plus git rev-parse --short
_ADD_ARGUMENT = re.compile(r'add_argument\(\s*"(--[a-z-]+)"')
HELPER_FLAGS = (
    set(_ADD_ARGUMENT.findall(NAMESPACE_HELPER))
    | set(_ADD_ARGUMENT.findall(FP8_IDENTITY_HELPER))
    | {"--field", "--short"}
)


def sglang_cli_flags() -> set[str]:
    """The kebab-case flags this checkout declares, as server_args builds them."""
    flags: set[str] = set()
    fields = REPO_ROOT / "python/sglang/srt/arg_groups/fields"
    for path in fields.glob("*.py"):
        flags.update(
            "--" + name.replace("_", "-")
            for name in re.findall(r"^    ([a-z_0-9]+): A\[", path.read_text(), re.M)
        )
    for path in (REPO_ROOT / "python/sglang/srt").rglob("*.py"):
        text = path.read_text()
        flags.update(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', text))
        flags.update(re.findall(r'aliases=\["(--[a-z0-9-]+)"', text))
    return flags


def uncommented(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def recipe_flags(text: str) -> set[str]:
    return set(re.findall(r"(?<![\w=-])(--[a-z][a-z0-9-]*)", uncommented(text)))


def make_fake_model(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"architectures":["Qwen4ExpForConditionalGeneration"]}'
    )
    (model / "model.safetensors.index.json").write_text(
        '{"metadata":{},"weight_map":{"layers.0":"model-00001.safetensors"}}'
    )
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (model / "model-00001.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"\x00\x00\x00\x00"
    )
    return model


class TestServeRecipeStatic(unittest.TestCase):
    def test_every_sglang_flag_exists_on_this_base(self):
        missing = (recipe_flags(SCRIPT) - HELPER_FLAGS) - sglang_cli_flags()
        self.assertEqual(
            missing,
            set(),
            "recipe passes flags this base does not declare: " + " ".join(sorted(missing)),
        )

    def test_unsupported_features_stay_out_rather_than_silently_dropped(self):
        body = uncommented(SCRIPT)
        self.assertNotIn("--gdn-mtp-cache-mode", body)
        self.assertNotIn("--tp ", body)
        self.assertIn("--tp-size", body)
        self.assertIn("--speculative-draft-model-quantization unquant", body)
        self.assertNotIn("adaptive-next-v3.0.2", body)

    def test_qualified_launch_shape_is_preserved(self):
        body = uncommented(SCRIPT)
        for flag in (
            "--quantization modelopt_fp4",
            "--kv-cache-dtype",
            "--json-model-override-args",
            "--reasoning-parser qwen3",
            "--tool-call-parser qwen3_coder",
            "--default-chat-template-kwargs",
            "--mamba-radix-cache-strategy extra_buffer",
            "--linear-attn-decode-backend flashinfer",
            "--enable-hierarchical-cache",
            "--hicache-write-policy write_through",
            "--hicache-io-backend kernel",
            "--hicache-mem-layout page_first",
            "--speculative-algorithm NEXTN",
            "--speculative-adaptive",
            "--speculative-adaptive-config",
            "--speculative-token-map",
            "--hicache-storage-backend nixl",
        ):
            self.assertIn(flag, body, flag)

    def test_compiler_caches_are_roots_under_one_configurable_base(self):
        for env in (
            "HF_HOME",
            "TORCH_HOME",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "CUDA_CACHE_PATH",
            "FLASHINFER_WORKSPACE_BASE",
            "SGLANG_CACHE_DIR",
            "TILELANG_CACHE_DIR",
        ):
            self.assertRegex(
                SCRIPT,
                rf'{env}="[^"]*\$CACHE_BASE/[a-z/]+',
                f"{env} escaped CACHE_BASE",
            )
        self.assertIn(
            'mkdir -p "$CACHE_BASE"/{huggingface,torch,torchinductor,triton,cuda,'
            "flashinfer,sglang/jit,tilelang}",
            SCRIPT,
        )

    def test_gdn_mode_is_the_qualified_default_with_a_kill_switch(self):
        self.assertIn(
            'export FLASHINFER_GDN_FP16_ACCUM_MMA="${FLASHINFER_GDN_FP16_ACCUM_MMA:-1}"',
            SCRIPT,
        )
        self.assertIn("gdn_fp16_accum_mma=$GDN_FP16_ACCUM_MMA", SCRIPT)

    def test_fp8_identity_hashes_what_is_resident_not_what_was_requested(self):
        body = uncommented(SCRIPT)
        self.assertIn('--field "online_fp8_precision=$ONLINE_FP8_PRECISION"', body)
        self.assertNotIn("online_fp8_requested_mode", body)
        # Resolved through the recipe's own adapter, not an inline guess.
        self.assertIn('FP8_IDENTITY_HELPER="$SCRIPT_DIR/online_fp8_identity.py"', body)
        self.assertIn('--requested "$ONLINE_FP8_MODE"', body)

    def test_adaptive_policies_only_use_supported_knobs(self):
        supported = {
            "candidate_steps",
            "ema_alpha",
            "update_interval",
            "warmup_batches",
            "up_hysteresis",
            "down_hysteresis",
            "ceiling_coeff",
        }
        for name in ("adaptive-next.json", "adaptive-next-fixed-w4.json"):
            cfg = json.loads((HERE / name).read_text())
            for key, entry in cfg.items():
                if key == "_comment":
                    continue
                self.assertTrue(key.isdigit(), key)
                self.assertTrue(entry["candidate_steps"])
                self.assertLessEqual(set(entry), supported, name)


class TestEmittedArgv(unittest.TestCase):
    """Execute the recipe with a recording fake CLI and judge the argv it emits."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = make_fake_model(self.root)
        self.dump = self.root / "argv"
        recorder = self.root / "record.sh"
        recorder.write_text(
            '#!/usr/bin/env bash\nfor arg in "$@"; do printf "%s\\0" "$arg" >> "$ARGV_DUMP"; done\n'
        )
        recorder.chmod(recorder.stat().st_mode | stat.S_IEXEC)
        self.recorder = recorder

    def tearDown(self):
        self.tmp.cleanup()

    def launch(self, **env):
        environ = {
            **os.environ,
            "SGLANG_EXE": str(self.recorder),
            "PYTHON": sys.executable,
            "TARGET_MODEL": str(self.model),
            "CACHE_BASE": str(self.root / "cache"),
            "NIXL": "off",
            "ARGV_DUMP": str(self.dump),
        }
        environ.update({k.upper(): str(v) for k, v in env.items()})
        result = subprocess.run(
            ["bash", str(HERE / "serve-flashnext-native.sh")],
            capture_output=True,
            text=True,
            env=environ,
        )
        self.dump.touch()
        argv = self.dump.read_text().split("\0") if self.dump.stat().st_size else []
        argv = [a for a in argv if a != ""]
        self.dump.unlink(missing_ok=True)
        return result, argv

    @staticmethod
    def value_after(argv, flag):
        return argv[argv.index(flag) + 1]

    def test_default_serves_the_maintained_client_id(self):
        result, argv = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.value_after(argv, "--served-model-name"), "pennyroyal")

    def test_served_model_name_is_configurable(self):
        _, argv = self.launch(served_model_name="flashnext")
        self.assertEqual(self.value_after(argv, "--served-model-name"), "flashnext")

    def test_default_yarn_override_reaches_the_cli(self):
        _, argv = self.launch()
        rope = json.loads(self.value_after(argv, "--json-model-override-args"))
        rope = rope["text_config"]["rope_parameters"]
        self.assertEqual(rope["rope_type"], "yarn")
        self.assertEqual(rope["factor"], 2.0)
        self.assertEqual(rope["original_max_position_embeddings"], 262144)

    def test_caller_model_override_survives_the_default_expansion(self):
        # bash re-splits ${VAR:-{json}} once VAR is set and appends the tail
        # as literal text; the default must live in its own variable.
        custom = '{"text_config":{"test":1}}'
        _, argv = self.launch(model_override_json=custom)
        self.assertEqual(self.value_after(argv, "--json-model-override-args"), custom)

    def test_fr_spec_is_opt_in_and_named(self):
        result, argv = self.launch()
        self.assertIn("full-vocabulary NEXTN draft baseline", result.stdout)
        self.assertNotIn("--speculative-token-map", argv)
        token_map = self.root / "flash-next-64k.pt"
        token_map.write_bytes(b"\x00map")
        _, argv = self.launch(token_map=token_map)
        self.assertEqual(self.value_after(argv, "--speculative-token-map"), str(token_map))

    def test_bringup_is_the_minimal_boot(self):
        result, argv = self.launch(bringup=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--speculative-algorithm", "--enable-hierarchical-cache", "--speculative-token-map"):
            self.assertNotIn(flag, argv)
        self.assertIn("--quantization", argv)
        self.assertEqual(self.value_after(argv, "--served-model-name"), "pennyroyal")

    def test_bringup_ignores_full_mode_only_inputs(self):
        # An obsolete width policy, a stale token map and a bogus host-tier
        # size are all invisible to the minimal boot: neither is in its argv,
        # so none may abort or hash during a smoke boot.
        result, argv = self.launch(
            bringup=1,
            mtp_policy="obsolete",
            token_map=self.root / "does-not-exist.pt",
            penny_hicache_size_gb="zero",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--speculative-algorithm", "--speculative-adaptive-config", "--speculative-token-map", "--enable-hierarchical-cache", "--hicache-size", "--hicache-storage-backend"):
            self.assertNotIn(flag, argv)
        self.assertNotIn("FR-Spec:", result.stdout)

    def test_full_mode_still_refuses_stale_full_mode_inputs(self):
        for env, message in (
            ({"mtp_policy": "obsolete"}, "MTP_POLICY must be tiered or fixed-w4"),
            ({"token_map": str(self.root / "does-not-exist.pt")}, "token map is not readable"),
            ({"penny_hicache_size_gb": "zero"}, "PENNY_HICACHE_SIZE_GB must be a positive integer"),
        ):
            result, _ = self.launch(**env)
            self.assertNotEqual(result.returncode, 0, env)
            self.assertIn(message, result.stderr, env)

    def test_mm_preprocess_device_parking_fails_clearly(self):
        result, _ = self.launch(sglang_mm_preprocess_device="cuda:1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not wired", result.stderr)

    def test_width_policy_difference_is_announced_not_claimed(self):
        result, argv = self.launch(mtp_policy="fixed-w4")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("does not implement the live v3.0.2", result.stdout)
        self.assertEqual(self.value_after(argv, "--speculative-num-steps"), "3")
        result, argv = self.launch()
        self.assertEqual(self.value_after(argv, "--speculative-num-steps"), "7")

    def _fake_device(self, capability):
        """A PYTHON whose torch reports one chosen device.

        The recipe's identity adapter asks the serving interpreter what device
        it is on, and the answer decides the namespace -- so the check has to
        drive that question instead of trusting whichever GPU (if any) happens
        to host the test worker. ``None`` stands for "no CUDA at all"; the fake
        also blocks the installed ``sglang`` package so the adapter resolves
        through this checkout's copy.
        """
        stub = self.root / f"device-{capability}"
        stub.mkdir(exist_ok=True)
        (stub / "sitecustomize.py").write_text(
            "import sys, types\n"
            "cuda = types.SimpleNamespace("
            f"is_available=lambda: {capability is not None}, "
            f"get_device_capability=lambda: {capability!r})\n"
            "torch = types.ModuleType('torch')\n"
            "torch.__version__ = '0.0+test-device'\n"
            "torch.cuda = cuda\n"
            "torch.bfloat16 = 'bfloat16'\n"
            "sys.modules['sglang'] = None\n"
            "sys.modules['torch'] = torch\n"
        )
        return stub

    def _namespace(self, device=None, **env):
        if device is not None:
            env["pythonpath"] = device
        env = {**env, "nixl": "on", "nixl_storage_base": self.root / "nixl"}
        result, _ = self.launch(**env)
        self.assertEqual(result.returncode, 0, result.stderr)
        for line in result.stdout.splitlines():
            if line.startswith("NIXL FILE namespace:"):
                return line.split(": ", 1)[1]
        self.fail("no namespace line in " + result.stdout)

    def test_nixl_identity_separates_token_map(self):
        token_map = self.root / "map.pt"
        token_map.write_bytes(b"\x00map")
        plain = self._namespace()
        mapped = self._namespace(token_map=token_map)
        self.assertNotEqual(plain, mapped)

    def test_nixl_identity_merges_automatic_and_explicit_fp8(self):
        """Eligible checkpoint on SM120: one resident representation, one namespace.

        The recipe used to hash the requested mode, so ``unset`` and ``true`` --
        the same E4M3 weights and per-row scales -- landed in separate NIXL
        namespaces and a qualified disk tier was re-read for a change of spelling.
        """
        sm120 = self._fake_device((12, 0))
        self.assertEqual(
            self._namespace(device=sm120),
            self._namespace(device=sm120, sglang_sm120_online_mxfp8="true"),
        )

    def test_nixl_identity_merges_automatic_off_and_explicit_false(self):
        """Where eligibility does not hold, automatic and explicit false are the
        same untouched BF16 checkpoint, and must share its namespace."""
        grid = self._fake_device((9, 0))
        self.assertEqual(
            self._namespace(device=grid),
            self._namespace(device=grid, sglang_sm120_online_mxfp8="false"),
        )

    def test_nixl_identity_separates_the_two_resident_formats(self):
        self.assertNotEqual(
            self._namespace(device=self._fake_device((12, 0))),
            self._namespace(device=self._fake_device((9, 0))),
        )

    def test_fp8_identity_announces_the_requested_mode_as_a_diagnostic(self):
        result, _ = self.launch(
            nixl="on",
            nixl_storage_base=self.root / "nixl",
            pythonpath=self._fake_device((12, 0)),
            sglang_sm120_online_mxfp8="true",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Online FP8: resident=rowwise_fp8 requested=true", result.stdout)

    def test_explicit_fp8_on_unsupported_hardware_aborts_before_a_namespace(self):
        result, _ = self.launch(
            nixl="on",
            nixl_storage_base=self.root / "nixl",
            pythonpath=self._fake_device((9, 0)),
            sglang_sm120_online_mxfp8="true",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires exactly SM120", result.stderr)
        self.assertIn("Cannot name the online-FP8 precision", result.stderr)
        self.assertNotIn("NIXL FILE namespace:", result.stdout)


if __name__ == "__main__":
    unittest.main()
