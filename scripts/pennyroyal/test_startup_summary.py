import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "configs" / "pennyroyal" / "startup-summary.sh"
BASE = "5d1a31074028a16f3fba468ed144125cc1222e18"
LAUNCHERS = (
    "serve-flash-next.sh",
    "serve-flash-next-frspec.sh",
    "serve-qwen38-27b-dflash2.sh",
)


class StartupSummaryTest(unittest.TestCase):
    def summary(self, *args: str, env: dict[str, str] | None = None) -> str:
        command = 'source "$1"; shift; pennyroyal_startup_summary "$@"'
        run_env = os.environ.copy()
        for name in (
            "PENNY_PLE_BACKEND",
            "SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR",
            "SGLANG_MM_PREPROCESS_DEVICE",
            "SGLANG_SM120_ONLINE_MXFP8",
        ):
            run_env.pop(name, None)
        if env:
            run_env.update(env)
        return subprocess.check_output(
            ["bash", "-c", command, "bash", str(SUMMARY), *args],
            cwd=ROOT,
            env=run_env,
            text=True,
        )

    def test_nextn_frspec_host_ple_cpu_and_quoted_paths(self):
        output = self.summary(
            "serve",
            "--model-path",
            "/models/Flash Next [qualified]",
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-token-map",
            "/maps/token map (v1).json",
            "--tp=2",
            "--kv-cache-dtype=fp8_e4m3",
            "--context-length=524288",
            "--max-total-tokens=824384",
            "--max-running-requests=4",
            "--max-mamba-cache-size=24",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=1",
            "--speculative-num-draft-tokens=4",
            "--ple-offload-embedding",
            "--enable-hierarchical-cache",
            "--hicache-size=32",
            "--hicache-storage-backend=nixl",
            "--image-processor-backend=sglang",
            env={
                "SGLANG_SM120_ONLINE_MXFP8": "true",
                "SGLANG_MM_PREPROCESS_DEVICE": "cpu",
                "SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR": "/cache/NIXL pool [one]",
            },
        )

        self.assertIn("Profile: Qwen3.8 Flash-Next / native NEXTN MTP | TP: 2", output)
        self.assertIn("Model: /models/Flash Next [qualified]", output)
        self.assertIn("FR-Spec: on | Online FP8: true | KV dtype: fp8_e4m3", output)
        self.assertIn("Context: 524288 tokens | KV cap: 824384", output)
        self.assertIn("Max running requests: 4 | Mamba slots: 24", output)
        self.assertIn("PLE: host RAM | HiCache: true | Host tier: 32 GiB", output)
        self.assertIn("Storage backend: nixl | NIXL location: /cache/NIXL pool [one]", output)
        self.assertIn("Media preprocessing: CPU (sglang)", output)

    def test_nextn_without_frspec_reports_automatic_cap_and_secondary_gpu(self):
        output = self.summary(
            "serve",
            "--model-path=/models/next",
            "--speculative-algorithm=NEXTN",
            "--tp=1",
            "--image-processor-backend=transformers",
            env={"SGLANG_MM_PREPROCESS_DEVICE": "cuda:2"},
        )

        self.assertIn("FR-Spec: off", output)
        self.assertIn("Context: automatic tokens | KV cap: automatic", output)
        self.assertIn("PLE: no host offload requested", output)
        self.assertIn("Media preprocessing: secondary GPU (cuda:2) (transformers)", output)

    def test_nextn_nvme_ple_and_main_gpu(self):
        output = self.summary(
            "serve",
            "--model-path",
            "/models/next",
            "--speculative-algorithm",
            "NEXTN",
            "--hicache-storage-backend",
            "nixl",
            env={
                "PENNY_PLE_BACKEND": "nvme",
                "SGLANG_MM_PREPROCESS_DEVICE": "cuda:0",
            },
        )

        self.assertIn("PLE: NVMe (prepared model overlay)", output)
        self.assertIn("Media preprocessing: main GPU (cuda:0)", output)

    def test_dflash_marks_ple_and_online_fp8_not_applicable(self):
        output = self.summary(
            "serve",
            "--model-path=/models/target model",
            "--speculative-draft-model-path=/models/draft model",
            "--speculative-algorithm=DFLASH",
            "--kv-cache-dtype=fp8_e4m3",
            "--speculative-draft-kv-cache-dtype=fp8_e5m2",
            env={
                "PENNY_PLE_BACKEND": "nvme",
                "SGLANG_SM120_ONLINE_MXFP8": "true",
            },
        )

        self.assertIn("Profile: Qwen3.8-27B / DFlash2", output)
        self.assertIn("Model: /models/target model", output)
        self.assertIn("Draft model: /models/draft model", output)
        self.assertIn("FR-Spec: off | Online FP8: not applicable | KV dtype: fp8_e4m3", output)
        self.assertIn("Draft KV dtype: fp8_e5m2", output)
        self.assertIn("PLE: not applicable", output)

    def test_all_launcher_argv_match_the_base_exec_blocks(self):
        # The comparison needs the base commit's blobs. A shallow or partial
        # clone may not carry that object at all; say so instead of reporting
        # a recipe regression that never happened.
        probe = subprocess.run(["git", "cat-file", "-t", BASE], cwd=ROOT,
                               capture_output=True, text=True)
        if probe.returncode != 0:
            self.skipTest(f"base commit {BASE[:10]} is not available here")
        for launcher in LAUNCHERS:
            with self.subTest(launcher=launcher), tempfile.TemporaryDirectory() as temp:
                temp_root = Path(temp)
                base_capture = temp_root / "base.argv"
                current_capture = temp_root / "current.argv"
                executable_dir = temp_root / "mock executable [dir]"
                executable_dir.mkdir()
                executable = executable_dir / "capture launcher"
                executable.write_text(
                    '#!/usr/bin/env bash\nprintf \'%s\\0\' "$@" > "$CAPTURE_PATH"\n'
                )
                executable.chmod(0o755)

                relative = f"configs/pennyroyal/{launcher}"
                base_source = subprocess.check_output(
                    ["git", "show", f"{BASE}:{relative}"], cwd=ROOT, text=True
                )
                current_source = (ROOT / relative).read_text()
                base_block = self._from_last_line(base_source, 'exec "$SGLANG_EXE" serve')
                current_block = self._from_last_line(current_source, "launch_args=(serve")

                base_argv, base_output = self._run_launch_block(
                    base_block, executable, base_capture
                )
                current_argv, current_output = self._run_launch_block(
                    current_block, executable, current_capture,
                    # The chosen RAM cache size is now a variable the recipe
                    # guards; with no choice made it must expand to the same
                    # qualified literal the base block still hardcodes, so the
                    # two argvs stay byte-identical.
                    hicache_size=self._qualified_hicache_size(current_source),
                )
                self.assertNotIn("Pennyroyal startup — requested settings", base_output)
                self.assertIn("Pennyroyal startup — requested settings", current_output)
                self.assertEqual(base_argv, current_argv)

    @staticmethod
    def _qualified_hicache_size(current_source: str) -> str:
        import re

        match = re.search(
            r'^HICACHE_SIZE_GB="\$\{PENNY_HICACHE_SIZE_GB:-([0-9]+)\}"$',
            current_source,
            re.MULTILINE,
        )
        if not match:
            raise AssertionError("the recipe lost its qualified HiCache default")
        return match.group(1)

    @staticmethod
    def _from_last_line(source: str, prefix: str) -> str:
        lines = source.splitlines()
        matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
        if not matches:
            raise AssertionError(f"missing final block starting with {prefix!r}")
        return "\n".join(lines[matches[-1] :]) + "\n"

    def _run_launch_block(
        self, block: str, executable: Path, capture: Path,
        hicache_size: str = "",
    ) -> tuple[list[str], str]:
        scalar_values = {
            "SGLANG_EXE": str(executable),
            "SCRIPT_DIR": str(ROOT / "configs" / "pennyroyal"),
            "TARGET_MODEL": "/models/target model [v1]",
            "DRAFT_MODEL": "/models/draft model (small)",
            "TP_SIZE": "2",
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
            "MAMBA_TRACK_INTERVAL": "256",
            # The Next recipes read these from request-capacity.sh, whose
            # defaults are the literals the older block still hardcodes, so
            # giving the fixture the same values compares identical argv.
            "MAX_RUNNING_REQUESTS": "4",
            "MAX_MAMBA_CACHE_SIZE": "24",
            # The integrated release sources this unchanged default separately.
            "DEFAULT_CHAT_TEMPLATE_KWARGS": '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}',
            "NIXL_CONFIG": "/configs/NIXL config [qualified].json",
            "CHAT_TEMPLATE": "/templates/chat template (tools).jinja",
            "IMAGE_PROCESSOR_BACKEND": "sglang",
            "TOKEN_MAP": "/maps/token map [FR Spec].npy",
            "DRAFT_TOKENS": "8",
            "DRAFT_WINDOW_SIZE": "32",
        }
        if hicache_size:
            scalar_values["HICACHE_SIZE_GB"] = hicache_size
        lines = ["set -euo pipefail"]
        lines.extend(
            f"{name}={shlex.quote(value)}" for name, value in scalar_values.items()
        )
        lines.extend(
            (
                "TOKEN_CAP_ARGS=(--max-total-tokens 824384)",
                "PLE_ARGS=(--ple-offload-embedding)",
                f"export CAPTURE_PATH={shlex.quote(str(capture))}",
                block,
            )
        )
        result = subprocess.run(
            ["bash"],
            cwd=ROOT,
            input="\n".join(lines),
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        payload = capture.read_bytes()
        self.assertTrue(payload.endswith(b"\0"))
        return [item.decode() for item in payload[:-1].split(b"\0")], result.stdout


if __name__ == "__main__":
    unittest.main()
