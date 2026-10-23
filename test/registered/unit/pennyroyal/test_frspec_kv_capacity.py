"""CPU-only checks for the KV capacity the Flash-Next launchers request.

The FR-Spec launchers used to hand configure_max_total_tokens their own
824384-token preset, so an operator who set nothing still got a fixed KV pool
instead of the memory-based sizing the engine profiles. The preset is gone:
unset or blank MAX_TOTAL_TOKENS must omit --max-total-tokens on every
Flash-Next launch path, while a deliberately supplied, page-aligned cap still
travels into the server argv and an invalid one still stops the launch. The
checks execute the launchers' own launch blocks and compare recorded argv, not
recipe text.
"""

import os
import shlex
import subprocess
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
CONFIGS = REPO / "configs" / "pennyroyal"
LAUNCH = REPO / "docker" / "pennyroyal" / "launch" / "config"
# Every Flash-Next launch path: native and container, FR-Spec and plain.
LAUNCHERS = (
    CONFIGS / "serve-flash-next.sh",
    CONFIGS / "serve-flash-next-frspec.sh",
    LAUNCH / "start-flash-next.sh",
    LAUNCH / "start-flash-next-frspec.sh",
)
FRSPEC_RECIPE = CONFIGS / "serve-flash-next-frspec.sh"

# The launcher reads these as already-decided scalars. They are literals on
# purpose: this checks the token-cap segment of the argv a launcher builds, and
# a recipe that renames one of them fails loudly instead of silently.
BLOCK_SCALARS = {
    "REPO_ROOT": str(REPO),
    "IMAGE_CONFIGS": str(CONFIGS),
    "TARGET_MODEL": "/models/Flash Next [qualified]",
    "TP_SIZE": "1",
    "COMPUTE_DTYPE": "bfloat16",
    "KV_DTYPE": "fp8_e4m3",
    "MAMBA_SSM_DTYPE": "bfloat16",
    "MAMBA_CONV_DTYPE": "bfloat16",
    "MAMBA_TRACK_INTERVAL": "64",
    "CONTEXT_LENGTH": "524288",
    "PAGE_SIZE": "64",
    "PREFILL_CHUNK_SIZE": "4096",
    "HICACHE_SIZE_GB": "32",
    "TARGET_OVERRIDES": '{"text_config":{"rope_parameters":{}}}',
    "IMAGE_PROCESSOR_BACKEND": "pil",
    "CHAT_TEMPLATE": str(CONFIGS / "templates" / "froggeric-v22.5.jinja"),
    "TOKEN_MAP": str(CONFIGS / "frspec" / "flash-next-64k.pt"),
    "SGLANG_SM120_ONLINE_MXFP8": "",
}
# Never inherited from the developer's shell: each test decides these itself.
CONTROLLED = (
    "MAX_TOTAL_TOKENS",
    "MAX_RUNNING_REQUESTS",
    "MAX_MAMBA_CACHE_SIZE",
    "PENNY_REASONING_EFFORT",
    "PENNY_PLE_BACKEND",
    "SGLANG_SM120_ONLINE_MXFP8",
)


def launch_argv(launcher: Path, **env_updates: str) -> subprocess.CompletedProcess[str]:
    """Build one launcher's real launch argv against a recording server.

    The preamble ahead of the launch block needs a real checkpoint, the pinned
    FR-Spec tokenizer and working import probes, none of which exist on a CPU
    runner, so the block is executed on its own: the shared capacity, reasoning
    and PLE helpers are sourced for real, the launcher's own
    configure_max_total_tokens call runs verbatim, and the exec records argv
    instead of booting a server.
    """
    source = launcher.read_text().splitlines()
    cap_call = next(s for s in source if s.startswith("configure_max_total_tokens"))
    start = next(i for i, s in enumerate(source) if s.startswith("launch_args=(serve"))
    block = "\n".join(source[start:]).replace(
        'exec "$SGLANG_EXE" "${launch_args[@]}"',
        """printf '%s\\0' "${launch_args[@]}\"""",
    )
    assert block != "\n".join(source[start:]), launcher
    lines = [
        "set -euo pipefail",
        # The summary reads argv; the argv itself is the subject here.
        "pennyroyal_startup_summary() { :; }",
        "HICACHE_STORAGE_ARGS=()",
        *[f"{name}={shlex.quote(value)}" for name, value in BLOCK_SCALARS.items()],
        'SCRIPT_DIR="$REPO_ROOT/configs/pennyroyal"',
        # The width policy is sourced for real, so the extracted block carries the
        # launch width and policy file of the qualified tiered default (the three
        # selections are launched end to end in test_prebuilt_container_launch.py).
        'source "$SCRIPT_DIR/mtp-policy.sh"',
        'source "$SCRIPT_DIR/request-capacity.sh"',
        'source "$SCRIPT_DIR/reasoning-effort.sh"',
        'source "$SCRIPT_DIR/ple-backend.sh"',
        # The opt-in cap call exactly as the launcher writes it, with the
        # same PAGE_SIZE its helper aligns against.
        cap_call,
        block,
    ]
    env = {name: value for name, value in os.environ.items() if name not in CONTROLLED}
    return subprocess.run(
        ["bash"],
        input="\n".join(lines) + "\n",
        env={**env, **BLOCK_SCALARS, **env_updates},
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def argv_of(result: subprocess.CompletedProcess[str]) -> list[str]:
    assert result.returncode == 0, result.stderr
    return [item for item in result.stdout.split("\0") if item]


def caps(argv: list[str]) -> list[str]:
    return [
        argv[index + 1]
        for index, value in enumerate(argv)
        if value == "--max-total-tokens"
    ]


def test_unset_or_blank_capacity_leaves_the_pool_to_the_engine():
    for launcher in LAUNCHERS:
        for value in (None, ""):  # unset, and saved-but-blank
            updates = {} if value is None else {"MAX_TOTAL_TOKENS": value}
            argv = argv_of(launch_argv(launcher, **updates))
            assert "--max-total-tokens" not in argv, (launcher, value, caps(argv))


def test_an_explicit_page_aligned_cap_still_reaches_the_server():
    for launcher in LAUNCHERS:
        argv = argv_of(launch_argv(launcher, MAX_TOTAL_TOKENS="1048576"))
        assert caps(argv) == ["1048576"], (launcher, caps(argv))
        # Only the cap was requested: context and page size stay as qualified.
        assert argv[argv.index("--context-length") + 1] == "524288", launcher
        assert argv[argv.index("--page-size") + 1] == "64", launcher


def test_the_old_preset_is_now_an_ordinary_number():
    # Below, equal to, and above the removed preset: every page-aligned value
    # is an ordinary request, and none of them is privileged.
    for cap in ("64", "824320", "824384", "824448"):
        argv = argv_of(launch_argv(FRSPEC_RECIPE, MAX_TOTAL_TOKENS=cap))
        assert caps(argv) == [cap], cap


def test_an_unaligned_or_non_numeric_cap_still_stops_the_launch():
    for value, message in (
        ("0", "positive integer"),
        ("-64", "positive integer"),
        ("word", "positive integer"),
        ("824385", "aligned to page size 64"),
    ):
        result = launch_argv(FRSPEC_RECIPE, MAX_TOTAL_TOKENS=value)
        assert result.returncode != 0, (value, result.stdout)
        assert message in result.stderr, (value, result.stderr)


if __name__ == "__main__":  # pytest is the real driver; this is a smoke check
    import inspect

    for name, case in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(case):
            case()
            print(f"{name} ok")
