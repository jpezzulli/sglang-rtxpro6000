import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HELPER = REPO / "configs" / "pennyroyal" / "ple-backend.sh"
NEXT_RECIPES = (
    REPO / "configs" / "pennyroyal" / "serve-flash-next-frspec.sh",
    REPO / "configs" / "pennyroyal" / "serve-flash-next.sh",
)
GUARD = (
    REPO
    / "tools"
    / "ple_nvme"
    / "ssd_stream"
    / "src"
    / "sglang_ssd_stream"
    / "pennyroyal-source.json"
)


def _run_helper(body: str, **updates: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in (
        "MAX_TOTAL_TOKENS",
        "PENNY_PLE_BACKEND",
        "SGLANG_PLUGINS",
        "SGLANG_SM120_ONLINE_MXFP8",
    ):
        env.pop(name, None)
    env.update(updates)
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; REPO_ROOT="$1"; TARGET_MODEL=original; '
            'source "$2"; PAGE_SIZE=64; ' + body,
            "bash",
            str(REPO),
            str(HELPER),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_ram_mode_and_online_fp8_defaults_are_inert():
    result = _run_helper(
        'printf "%s\\n" "$TARGET_MODEL" "${PLE_ARGS[*]}" '
        '"${#PLE_NAMESPACE_ARGS[@]}" "$PLE_OFFLOAD_EMBEDDING" '
        '"$SGLANG_SM120_ONLINE_MXFP8"'
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "original",
        "--ple-offload-embedding",
        "0",
        "true",
        "false",
    ]


def test_token_caps_are_optional_aligned_and_frspec_has_qualified_default():
    absent = _run_helper('configure_max_total_tokens; echo "${#TOKEN_CAP_ARGS[@]}"')
    default = _run_helper(
        'configure_max_total_tokens 824384; echo "${TOKEN_CAP_ARGS[*]}"'
    )
    explicit = _run_helper(
        'configure_max_total_tokens 824384; echo "${TOKEN_CAP_ARGS[*]}"',
        MAX_TOTAL_TOKENS="1048576",
    )

    assert absent.returncode == 0 and absent.stdout.strip() == "0"
    assert default.returncode == 0
    assert default.stdout.strip() == "--max-total-tokens 824384"
    assert explicit.returncode == 0
    assert explicit.stdout.strip() == "--max-total-tokens 1048576"

    for invalid, message in (
        ("0", "positive integer"),
        ("word", "positive integer"),
        ("824385", "aligned to page size 64"),
    ):
        result = _run_helper(
            "configure_max_total_tokens 824384", MAX_TOTAL_TOKENS=invalid
        )
        assert result.returncode != 0
        assert message in result.stderr


def test_nvme_mode_has_no_hidden_token_cap(tmp_path):
    plugin = tmp_path / "plugin"
    (plugin / "sglang_ssd_stream").mkdir(parents=True)
    (plugin / "sglang_ssd_stream" / "plugin.py").touch()
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/sh\nprintf '%064d\\n' 0\n")
    fake_python.chmod(0o755)

    result = _run_helper(
        'printf "%s\\n" "$TARGET_MODEL" "${#PLE_ARGS[@]}" '
        '"$PLE_OFFLOAD_EMBEDDING" "$SGLANG_PLUGINS" '
        '"${PLE_NAMESPACE_ARGS[*]}"',
        PENNY_PLE_BACKEND="nvme",
        PENNY_PLE_NVME_MODEL="/prepared",
        PENNY_PLE_PLUGIN_DIR=str(plugin),
        PYTHON=str(fake_python),
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:4] == ["/prepared", "0", "false", "ssd_stream"]
    assert "ple_backend=nvme" in lines[4]
    assert "--max-total-tokens" not in result.stdout


def test_online_fp8_switch_requires_literal_boolean():
    enabled = _run_helper(
        'echo "$SGLANG_SM120_ONLINE_MXFP8"',
        SGLANG_SM120_ONLINE_MXFP8="true",
    )
    invalid = _run_helper(":", SGLANG_SM120_ONLINE_MXFP8="1")

    assert enabled.returncode == 0 and enabled.stdout.strip() == "true"
    assert invalid.returncode != 0
    assert "must be true or false" in invalid.stderr


def test_next_recipes_expose_qualified_media_devices_without_gpu_access():
    base_env = dict(
        os.environ,
        TARGET_MODEL="/missing-model",
        CACHE_BASE="/missing-cache",
        NIXL_STORAGE_BASE="/missing-nixl",
        SGLANG_EXE="/missing-sglang",
        PYTHON="/missing-python",
    )
    for recipe in NEXT_RECIPES:
        for device in (None, "cpu", "cuda:0", "cuda:1"):
            env = base_env.copy()
            if device is None:
                env.pop("SGLANG_MM_PREPROCESS_DEVICE", None)
            else:
                env["SGLANG_MM_PREPROCESS_DEVICE"] = device
            result = subprocess.run(
                ["bash", str(recipe)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            assert result.returncode != 0
            assert "Required executable missing" in result.stderr
            assert "Choose SGLANG_MM_PREPROCESS_DEVICE" not in result.stderr


def test_next_recipes_initialize_cache_environment_before_nvme_preflight(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for filename in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        (source / filename).write_text("{}")
    plugin = tmp_path / "plugin" / "sglang_ssd_stream"
    plugin.mkdir(parents=True)
    (plugin / "plugin.py").touch()
    cache = tmp_path / "cache"
    nixl = tmp_path / "nixl"
    home = tmp_path / "readonly-home"
    home.mkdir(mode=0o500)
    observed = tmp_path / "cache-observed"
    fake_sglang = tmp_path / "sglang"
    fake_sglang.write_text("#!/bin/sh\nexit 0\n")
    fake_sglang.chmod(0o755)
    python_wrapper = tmp_path / "python"
    python_wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        '[[ "$HF_HOME" == "$CACHE_BASE/huggingface" ]]\n'
        '[[ "$XDG_CACHE_HOME" == "$CACHE_BASE" ]]\n'
        '[[ "$TORCH_HOME" == "$CACHE_BASE/torch" ]]\n'
        '[[ "$TORCHINDUCTOR_CACHE_DIR" == "$CACHE_BASE/torchinductor" ]]\n'
        '[[ "$TRITON_CACHE_DIR" == "$CACHE_BASE/triton" ]]\n'
        '[[ "$CUDA_CACHE_PATH" == "$CACHE_BASE/cuda" ]]\n'
        '[[ "$FLASHINFER_WORKSPACE_BASE" == "$CACHE_BASE/flashinfer" ]]\n'
        '[[ "$SGLANG_CACHE_DIR" == "$CACHE_BASE/sglang" ]]\n'
        '[[ "$SGLANG_JIT_CACHE_DIR" == "$CACHE_BASE/sglang/jit" ]]\n'
        '[[ -d "$SGLANG_JIT_CACHE_DIR" ]]\n'
        'printf ready > "$CACHE_OBSERVED"\n'
        "printf '%064d\\n' 0\n"
    )
    python_wrapper.chmod(0o755)

    for recipe in NEXT_RECIPES:
        observed.unlink(missing_ok=True)
        result = subprocess.run(
            ["bash", str(recipe)],
            env={
                **os.environ,
                "HOME": str(home),
                "TARGET_MODEL": str(source),
                "CACHE_BASE": str(cache),
                "CACHE_OBSERVED": str(observed),
                "NIXL_STORAGE_BASE": str(nixl),
                "PENNY_PLE_BACKEND": "nvme",
                "PENNY_PLE_NVME_MODEL": str(tmp_path / "prepared"),
                "PENNY_PLE_PLUGIN_DIR": str(plugin.parent),
                "SGLANG_EXE": str(fake_sglang),
                "PYTHON": str(python_wrapper),
            },
            text=True,
            capture_output=True,
            check=False,
        )
        assert observed.read_text() == "ready", (recipe, result.stderr)


def test_source_guard_matches_every_hooked_publication_module():
    guard = json.loads(GUARD.read_text())
    assert guard["source"].startswith("Pennyroyal v2.5.0")
    assert "sglang.srt.models.qwen4_exp" in guard["modules"]

    for module, expected in guard["modules"].items():
        path = REPO / "python" / Path(*module.split("."))
        if path.with_suffix(".py").is_file():
            path = path.with_suffix(".py")
        else:
            path = path / "__init__.py"
        assert path.is_file(), module
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, module
