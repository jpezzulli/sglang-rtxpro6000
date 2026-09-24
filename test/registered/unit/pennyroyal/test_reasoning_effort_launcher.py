"""CPU-only launcher checks: PENNY_REASONING_EFFORT convenience + TP plumbing.

PR#18 is launcher-only policy: the shared server code never reads the
variable (guarded below against server_args/environ drift), the recipes build
their single qualified --default-chat-template-kwargs through the shared
reasoning-effort.sh JSON builder, unset/empty keeps the qualified medium
default byte-for-byte, accepted OpenAI tiers rewrite just that key (no
float/fraction extension), and an invalid tier fails at launch. Per-request
precedence stays a serving-side rule the launcher cannot change. The Compose
service forwards the knob (plus TP_SIZE/NCCL_P2P_DISABLE) and the entrypoint
normalizes empty passthroughs away so unset behaves identically.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
CONFIGS = REPO / "configs" / "pennyroyal"
TP_GUARD = CONFIGS / "tp-devices.sh"
COMPOSE = REPO / "docker" / "pennyroyal" / "compose.yaml"
ENTRYPOINT = REPO / "docker" / "pennyroyal" / "entrypoint.sh"
ENV_EXAMPLE = REPO / "docker" / "pennyroyal" / ".env.example"
BASE_SHA = "11fd1f9b2e79fb3d2e07bfcad088aa6c07dbb2fb"
RECIPES = (
    "serve-flash-next.sh",
    "serve-flash-next-frspec.sh",
    "serve-qwen38-27b-dflash2.sh",
)
QUALIFIED_DEFAULT_KWARGS = (
    '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}'
)


def _run_helper(body: str, **env_updates: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in ("PENNY_REASONING_EFFORT", "TP_SIZE", "NCCL_P2P_DISABLE"):
        env.pop(name, None)
    env.update(env_updates)
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1/reasoning-effort.sh"; ' + body,
            "bash",
            str(CONFIGS),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


# ---------------- shared launcher JSON builder ----------------


def test_unset_or_empty_builds_the_qualified_medium_json():
    for value in (None, "", "   "):
        env = {} if value is None else {"PENNY_REASONING_EFFORT": value}
        result = _run_helper(
            'printf "%s" "$DEFAULT_CHAT_TEMPLATE_KWARGS"', **env
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == QUALIFIED_DEFAULT_KWARGS, value


@pytest.mark.parametrize(
    ("value", "tier"),
    [("xhigh", "xhigh"), ("XHIGH", "xhigh"), (" none ", "none"), ("high", "high")],
)
def test_valid_tiers_rewrite_only_the_effort_key(value, tier):
    result = _run_helper(
        'printf "%s" "$DEFAULT_CHAT_TEMPLATE_KWARGS"',
        PENNY_REASONING_EFFORT=value,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": tier,
    }


@pytest.mark.parametrize("value", ["medium ", " high ", "MAX", "Xhigh"])
def test_trimmed_and_cased_tiers_still_build(value):
    result = _run_helper(
        'printf "%s" "$DEFAULT_CHAT_TEMPLATE_KWARGS"',
        PENNY_REASONING_EFFORT=value,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["reasoning_effort"] == value.strip().lower()


@pytest.mark.parametrize(
    "value",
    # No float/fraction extension in the launcher: PR#18 stays tier-only.
    ["ultrathink", "1.0", "-1", "0.42", "0.99", "x high", "0..5", "medium2"],
)
def test_invalid_or_fractional_values_fail_at_launch(value):
    result = _run_helper(
        'printf "survived %s\\n" "$DEFAULT_CHAT_TEMPLATE_KWARGS"',
        PENNY_REASONING_EFFORT=value,
    )
    assert result.returncode != 0
    assert "PENNY_REASONING_EFFORT" in result.stderr
    assert "survived" not in result.stdout


# ---------------- PR#18 stays out of the shared server ----------------


def test_server_side_has_no_penny_reasoning_effort_policy():
    """Launcher-only: the shared CLI/env surface must be exactly upstream's."""
    from sglang.srt import environ as environ_module
    from sglang.srt import server_args
    from sglang.srt.environ import envs

    assert not hasattr(envs, "PENNY_REASONING_EFFORT")
    for module_path in (
        Path(server_args.__file__),
        Path(environ_module.__file__),
    ):
        assert "PENNY_REASONING_EFFORT" not in module_path.read_text(), module_path
    # The shared files are byte-identical to the base revision.
    import subprocess

    for relative in ("python/sglang/srt/server_args.py", "python/sglang/srt/environ.py"):
        diff = subprocess.run(
            ["git", "diff", "--quiet", BASE_SHA, "--", relative],
            cwd=REPO,
            capture_output=True,
            check=False,
        )
        assert diff.returncode == 0, f"{relative} drifted from base"


def test_qualified_launcher_still_produces_json_the_server_parses():
    """No server change is required: the built JSON loads through ServerArgs."""
    from sglang.srt.server_args import prepare_server_args

    args = prepare_server_args(
        [
            "--model-path",
            "/nonexistent-penny-dummy",
            "--default-chat-template-kwargs",
            QUALIFIED_DEFAULT_KWARGS,
        ]
    )
    assert args.default_chat_template_kwargs == json.loads(QUALIFIED_DEFAULT_KWARGS)
    args = prepare_server_args(
        [
            "--model-path",
            "/nonexistent-penny-dummy",
            "--default-chat-template-kwargs",
            '{"enable_thinking":true,"preserve_thinking":true,'
            '"reasoning_effort":"xhigh"}',
        ]
    )
    assert args.default_chat_template_kwargs["reasoning_effort"] == "xhigh"


# ---------------- recipes ----------------


def test_every_recipe_pins_the_qualified_medium_and_sources_the_helper_first():
    for name in RECIPES:
        source = (CONFIGS / name).read_text()
        helper = source.index('source "$SCRIPT_DIR/reasoning-effort.sh"')
        launch = source.index("launch_args=(serve")
        assert helper < launch, name
        # One shared builder instead of three literals: the recipes forward
        # the built JSON, and the qualified default lives in the helper.
        assert (
            '--default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS"' in source
        ), name
        assert "reasoning_effort" not in source, name
        assert "PENNY_REASONING_EFFORT" not in source.replace(
            'source "$SCRIPT_DIR/reasoning-effort.sh"', ""
        ), name
    helper_source = (CONFIGS / "reasoning-effort.sh").read_text()
    # The builder's template keeps the other two keys pinned and defaults
    # the effort to the qualified medium before any operator override.
    assert (
        '{"enable_thinking":true,"preserve_thinking":true,'
        '"reasoning_effort":"${PENNY_REASONING_EFFORT_NORMALIZED}"}'
        in helper_source
    )
    assert 'PENNY_REASONING_EFFORT_NORMALIZED="medium"' in helper_source


def test_helper_default_byte_for_byte_matches_the_qualified_record():
    result = _run_helper('printf "%s" "$DEFAULT_CHAT_TEMPLATE_KWARGS"')
    assert result.stdout == QUALIFIED_DEFAULT_KWARGS


# ---------------- Compose / entrypoint ----------------


def test_compose_forwards_optional_knobs_with_empty_defaults():
    service = yaml.safe_load(COMPOSE.read_text())["services"]["pennyroyal"]
    environment = service["environment"]
    assert environment["PENNY_REASONING_EFFORT"] == "${PENNY_REASONING_EFFORT:-}"
    assert environment["TP_SIZE"] == "${TP_SIZE:-1}"
    assert environment["NCCL_P2P_DISABLE"] == "${NCCL_P2P_DISABLE:-}"
    # Preserved invariants: the qualified context/pool/precision/tool knobs
    # the container was built with remain untouched.
    assert environment["SGLANG_FORWARD_UNKNOWN_TOOLS"] == (
        "${SGLANG_FORWARD_UNKNOWN_TOOLS:-true}"
    )


def _entrypoint_normalizer_snippet() -> str:
    lines = ENTRYPOINT.read_text().splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("for optional in")
    )
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "done")
    return "\n".join(lines[start : end + 1])


def test_entrypoint_normalizes_empty_passthroughs():
    snippet = _entrypoint_normalizer_snippet()
    assert "PENNY_REASONING_EFFORT" in snippet
    assert "TP_SIZE" in snippet
    assert "NCCL_P2P_DISABLE" in snippet
    result = subprocess.run(
        ["bash", "-c", snippet + '\nprintf "<%s><%s><%s>\\n" '
         '"${PENNY_REASONING_EFFORT-}" "${TP_SIZE-}" "${NCCL_P2P_DISABLE-}"'],
        env={
            **os.environ,
            "PENNY_REASONING_EFFORT": "",
            "TP_SIZE": "",
            "NCCL_P2P_DISABLE": "",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    # Empty strings became unset (all three print as empty).
    assert result.stdout == "<><><>\n"
    # A real value survives the normalizer.
    result = subprocess.run(
        ["bash", "-c", snippet + '\nprintf "<%s>\\n" "${NCCL_P2P_DISABLE-}"'],
        env={**os.environ, "NCCL_P2P_DISABLE": "1", "PENNY_REASONING_EFFORT": "",
             "TP_SIZE": "2"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout == "<1>\n"


def test_env_example_documents_the_optional_knobs():
    text = ENV_EXAMPLE.read_text()
    for line in ("PENNY_REASONING_EFFORT=", "TP_SIZE=", "NCCL_P2P_DISABLE="):
        assert line in text, line


# ---------------- TP vs visible-GPU guard (CPU-only, faked device count) ---


def _run_tp_guard(
    tmp_path,
    device_count: str,
    tp_size: str,
    preprocess: str = "cpu",
    cuda_visible_devices: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the launcher guard against a fake $PYTHON reporting ``device_count``.

    No GPU is needed: the guard reads the same torch-visible device count the
    server would see, so a stub that just echoes the number is exact.
    """
    stub = tmp_path / "fake-python.sh"
    stub.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' '{device_count}'\n")
    stub.chmod(0o755)
    env = os.environ.copy()
    env["PYTHON"] = str(stub)
    if cuda_visible_devices is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; '
            'pennyroyal_check_tp_devices "$2" "$3"',
            "bash",
            str(TP_GUARD),
            tp_size,
            preprocess,
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_tp_guard_keeps_the_portable_single_gpu_default(tmp_path):
    result = _run_tp_guard(tmp_path, "1", "1")
    assert result.returncode == 0, result.stderr
    # Base compose reserves exactly one GPU; that stays the default.
    service = yaml.safe_load(COMPOSE.read_text())["services"]["pennyroyal"]
    devices = service["deploy"]["resources"]["reservations"]["devices"]
    assert len(devices) == 1 and devices[0]["device_ids"] == ["${NVIDIA_GPU:-0}"]


def test_tp_guard_accepts_tp2_when_two_devices_are_visible(tmp_path):
    result = _run_tp_guard(tmp_path, "2", "2", cuda_visible_devices="0,1")
    assert result.returncode == 0, result.stderr


def test_tp_guard_rejects_tp2_with_one_visible_device(tmp_path):
    result = _run_tp_guard(tmp_path, "1", "2")
    assert result.returncode != 0
    assert "TP_SIZE=2 needs at least 2 visible CUDA device(s)" in result.stderr
    assert "only 1 is visible" in result.stderr
    # Actionable: names the file, the exact complete reservation fragment with
    # two explicit ids, and the fact that TP_SIZE never grants GPUs.
    assert "compose.yaml" in result.stderr
    assert 'device_ids: ["0", "1"]' in result.stderr
    assert "does not grant GPU access" in result.stderr
    assert "NCCL_P2P_DISABLE" not in result.stderr  # no unrelated advice


def test_tp_guard_counts_a_dedicated_preprocessing_gpu(tmp_path):
    # cuda:1 outside the TP1 model range is a second device the launch needs.
    result = _run_tp_guard(tmp_path, "1", "1", preprocess="cuda:1")
    assert result.returncode != 0
    assert "TP_SIZE=1 needs at least 2 visible CUDA device(s)" in result.stderr
    assert _run_tp_guard(tmp_path, "2", "1", preprocess="cuda:1").returncode == 0
    # A preprocessor sharing a model GPU (N < TP_SIZE) adds no requirement.
    assert (
        _run_tp_guard(tmp_path, "2", "2", preprocess="cuda:1").returncode == 0
    )
    assert _run_tp_guard(tmp_path, "1", "1", preprocess="cpu").returncode == 0


def test_tp_guard_fails_closed_when_the_device_count_is_unreadable(tmp_path):
    result = _run_tp_guard(tmp_path, "n/a", "2")
    assert result.returncode != 0
    assert "visible CUDA device count" in result.stderr


def test_recipes_source_and_run_the_tp_guard():
    for name in RECIPES:
        source = (CONFIGS / name).read_text()
        assert 'source "$SCRIPT_DIR/tp-devices.sh"' in source, name
        call = source.index("pennyroyal_check_tp_devices")
        cudvd = source.index("export CUDA_DEVICE_ORDER")
        assert cudvd < call, name
        # Never silently ignore the request: the guard sits before the launch.
        assert call < source.index("launch_args=(serve"), name
        # Its $PYTHON probe must see the durable cache environment and must
        # not preempt it: the guard runs after the cache exports and before
        # the recipe's next Python invocation (NVMe preflight / version hash).
        assert source.index('export HF_HOME="$CACHE_BASE/huggingface"') < call, name
        next_python = min(
            source.index(marker)
            for marker in ('source "$SCRIPT_DIR/ple-backend.sh"', '"$PYTHON" -c')
            if marker in source
        )
        assert call < next_python, name
    # TP_SIZE does not grant access is said where operators read it.
    readme = (REPO / "docker" / "pennyroyal" / "README.md").read_text()
    run_md = (REPO / "RUN.md").read_text()
    for text in (readme, run_md):
        assert 'device_ids: ["0", "1"]' in text
        assert "does not grant GPU access" in text


def test_frspec_recipe_tp_size_defaults_to_qualified_one_and_validates():
    source = (CONFIGS / "serve-flash-next-frspec.sh").read_text()
    assert 'TP_SIZE="${TP_SIZE:-1}"' in source
    assert "TP_SIZE must be a positive integer" in source
    guard = source[
        source.index('TP_SIZE="${TP_SIZE:-1}"') : source.index("COMPUTE_DTYPE=")
    ]
    for value, expect_ok in (("1", True), ("2", True), ("0", False), ("x", False)):
        env = os.environ.copy()
        env.pop("TP_SIZE", None)
        if value is not None:
            env["TP_SIZE"] = value
        result = subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + guard + "printf ok"],
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        assert (result.returncode == 0) == expect_ok, (value, result.stderr)
