"""CPU-only checks for the prebuilt-image launch examples in
docker/pennyroyal/launch: the host run.sh builds one plain docker command from
the operator's own files, and the image entrypoint hands a mounted startup
script to the image's own SGLang with the qualified flags, helpers, namespace
and pinned assets intact.

No GPU, no container runtime, no registry and no model: docker is a capturing
stub and the server binary is a capturing script that records its argv.
"""

import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
LAUNCH = REPO / "docker" / "pennyroyal" / "launch"
ENTRYPOINT = REPO / "docker" / "pennyroyal" / "entrypoint.sh"
CONFIGS = LAUNCH / "config"
RECIPES = REPO / "configs" / "pennyroyal"
PENNY_VERSION = (RECIPES / "pennyroyal.version").read_text().strip()
# Never inherited from the developer's shell: these are exactly what each test
# decides, so a stray export cannot change the expected launch.
CONTROLLED = (
    "PENNYROYAL_IMAGE",
    "PENNYROYAL_STARTUP",
    "PENNYROYAL_NIXL_CONFIG",
    "PENNYROYAL_PORT",
    "PENNYROYAL_USER",
    "HOST_MODELS_ROOT",
    "HOST_CACHE_BASE",
    "HOST_NIXL_STORAGE_BASE",
    "NVIDIA_GPU",
    "TARGET_MODEL",
    "DRAFT_MODEL",
    "CACHE_BASE",
    "NIXL_STORAGE_BASE",
    "NIXL_CONFIG",
    "NIXL",
    "PENNY_HICACHE_SIZE_GB",
    "PENNY_PLE_BACKEND",
    "TP_SIZE",
    "MAX_RUNNING_REQUESTS",
    "MAX_MAMBA_CACHE_SIZE",
    "MAX_TOTAL_TOKENS",
    "SGLANG_MM_PREPROCESS_DEVICE",
    "PENNY_MM_PREPROCESS_CACHE_DIR",
    "PENNY_MM_PREPROCESS_CACHE_SIZE_MB",
    "SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR",
    "FLASHINFER_GDN_FP16_ACCUM_MMA",
    "PENNY_MTP_POLICY",
    "TILELANG_CACHE_DIR",
    "CUDA_VISIBLE_DEVICES",
    "FAKE_CUDA_DEVICES",
)


def clean_env(**updates) -> dict[str, str]:
    env = {name: value for name, value in os.environ.items() if name not in CONTROLLED}
    return {**env, **updates}


def argv_after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def values_after(args: list[str], flag: str) -> list[str]:
    return [args[index + 1] for index, value in enumerate(args) if value == flag]


def qualified_defaults() -> dict[str, str]:
    # The recipes read these optional knobs from the environment; the mounted
    # scripts state them as settings. Pin the environment to what the recipes
    # would default to, so the two paths can be compared.
    return {
        "PENNY_HICACHE_SIZE_GB": "",
        "PENNY_PLE_BACKEND": "",
        "TP_SIZE": "",
        "MAX_RUNNING_REQUESTS": "",
        "MAX_MAMBA_CACHE_SIZE": "",
        "MAX_TOTAL_TOKENS": "",
        "SGLANG_MM_PREPROCESS_DEVICE": "",
        "PENNY_MM_PREPROCESS_CACHE_DIR": "",
        "PENNY_MM_PREPROCESS_CACHE_SIZE_MB": "",
    }


# --------------------------------------------------------------------- run.sh


def fake_docker(tmp_path: Path) -> tuple[Path, Path]:
    """A docker on PATH that records its argv instead of talking to a daemon."""
    capture = tmp_path / f"docker-argv-{os.urandom(4).hex()}"
    directory = tmp_path / "bin"
    directory.mkdir(exist_ok=True)
    stub = directory / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ': > "$DOCKER_CAPTURE"\n'
        'for argument in "$@"; do printf "%s\\n" "$argument" >> "$DOCKER_CAPTURE"; done\n'
    )
    stub.chmod(0o755)
    return directory, capture


def host_dirs(tmp_path: Path):
    roots = {}
    for name in ("models", "cache", "nixl"):
        directory = tmp_path / "host share" / name
        directory.mkdir(parents=True, exist_ok=True)
        roots[name] = directory
    return roots["models"], roots["cache"], roots["nixl"]


def run_launcher(
    tmp_path: Path,
    *options: str,
    cwd: Path | None = None,
    env: dict | None = None,
):
    directory, capture = fake_docker(tmp_path)
    result = subprocess.run(
        ["bash", str(LAUNCH / "run.sh"), *options],
        cwd=cwd,
        env=clean_env(
            PATH=f"{directory}:{os.environ['PATH']}",
            DOCKER_CAPTURE=str(capture),
            HOME=str(tmp_path / "home"),
            **(env or {}),
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    argv = capture.read_text().splitlines() if capture.exists() else None
    return result, argv


def test_launcher_mounts_directories_and_runs_the_mounted_script(tmp_path):
    startup = tmp_path / "my scripts" / "start up.sh"
    startup.parent.mkdir()
    startup.write_text("#!/usr/bin/env bash\nexit 0\n")
    nixl_config = tmp_path / "other share" / "nixl conf.toml"
    nixl_config.parent.mkdir()
    nixl_config.write_text("use_direct_io = true\n")
    models, cache, nixl = host_dirs(tmp_path)

    result, argv = run_launcher(
        tmp_path,
        "--startup",
        str(startup),
        "--nixl-config",
        str(nixl_config),
        "--models",
        str(models),
        "--cache",
        str(cache),
        "--nixl-root",
        str(nixl),
        "--image",
        "pennyroyal:test",
        "--port",
        "8099",
        "--gpu",
        "0,1",
        "--user",
        "1000:1000",
    )

    assert result.returncode == 0, result.stderr
    assert argv is not None
    assert argv[0] == "run"
    # Directories, never single files: an editor that writes a new file and
    # renames it must still be visible on the next start.
    assert values_after(argv, "--volume") == [
        f"{startup.parent}:/config:ro",
        f"{models}:/models:ro",
        f"{cache}:/cache",
        f"{nixl}:/nixl",
        f"{nixl_config.parent}:/nixl-config:ro",
    ]
    assert argv_after(argv, "--gpus") == '"device=0,1"'
    # docker/cli (opts/gpus.go) reads the value with encoding/csv, so the
    # literal inner quotes are part of the contract: unquoted device=0,1 is
    # parsed as device=0 plus a count of 1, which selects the wrong devices.
    assert next(csv.reader([argv_after(argv, "--gpus")])) == ["device=0,1"]
    assert next(csv.reader(["device=0,1"])) == ["device=0", "1"]  # the wrong shape
    assert argv_after(argv, "--publish") == "8099:8001"
    assert argv_after(argv, "--user") == "1000:1000"
    assert argv_after(argv, "--shm-size") == "16g"
    assert values_after(argv, "--ulimit") == ["memlock=-1:-1"]
    assert values_after(argv, "--security-opt") == ["seccomp=unconfined"]
    # The image's existing exec capability: no --entrypoint, no CMD profile.
    assert argv[-3:] == ["exec", "bash", "/config/start up.sh"]
    assert argv[argv.index("pennyroyal:test") + 1] == "exec"
    # A config outside the script's directory arrives as a container path.
    assert values_after(argv, "-e") == ["NIXL_CONFIG=/nixl-config/nixl conf.toml"]


def test_next_profiles_default_the_accepted_gdn_mode(tmp_path):
    # The accepted FlashInfer GDN fix is opted in per Next profile, in the
    # launched process's own environment: the image recipe and the operator's
    # mounted startup file both default it, an explicit opt-out survives, and
    # the 27B profile never sees the variable at all. The mode also separates the
    # persisted NIXL identity, because the two kernels write different numbers.
    mounted = prepared_config(tmp_path, "start-flash-next.sh") / "start-flash-next.sh"
    plain = prepared_config(tmp_path, "start-27b-dflash2.sh") / "start-27b-dflash2.sh"
    for startup, via_entrypoint in (
        (mounted, True),
        (RECIPES / "serve-flash-next.sh", False),
    ):
        result, _, server_env, _ = launch(
            tmp_path, startup, through_entrypoint=via_entrypoint
        )
        assert result.returncode == 0, (startup, result.stderr)
        assert server_env["gdn_fp16_accum_mma"] == "1", startup
        assert server_env["namespace"] != "unset", startup
        modes = {"default": server_env["namespace"]}
        for label, value in (("explicit on", "1"), ("off", "0"), ("unenabled", "true")):
            result, _, env, _ = launch(
                tmp_path,
                startup,
                through_entrypoint=via_entrypoint,
                env={"FLASHINFER_GDN_FP16_ACCUM_MMA": value},
            )
            assert result.returncode == 0, (startup, result.stderr)
            assert env["gdn_fp16_accum_mma"] == value, env
            modes[label] = env["namespace"]
        # Only the literal 1 enables the kernels, and only the resolved mode is
        # cache identity: a value that does not enable must not fork the root.
        assert modes["explicit on"] == modes["default"] != modes["off"], modes
        assert modes["unenabled"] == modes["off"], modes
        result, _, opted_out, _ = launch(
            tmp_path,
            startup,
            through_entrypoint=via_entrypoint,
            env={"FLASHINFER_GDN_FP16_ACCUM_MMA": "0"},
        )
        assert result.returncode == 0, (startup, result.stderr)
        assert opted_out["gdn_fp16_accum_mma"] == "0", startup
    for startup, via_entrypoint in (
        (plain, True),
        (RECIPES / "serve-qwen38-27b-dflash2.sh", False),
    ):
        result, _, server_env, _ = launch(
            tmp_path, startup, through_entrypoint=via_entrypoint
        )
        assert result.returncode == 0, (startup, result.stderr)
        assert server_env["gdn_fp16_accum_mma"] == "unset", startup
        # The 27B namespace is untouched by the Next-only knob.
        result, _, opted_out, _ = launch(
            tmp_path,
            startup,
            through_entrypoint=via_entrypoint,
            env={"FLASHINFER_GDN_FP16_ACCUM_MMA": "0"},
        )
        assert result.returncode == 0, (startup, result.stderr)
        assert opted_out["namespace"] == server_env["namespace"], startup


def test_launcher_default_startup_keeps_the_script_own_nixl_config(tmp_path):
    models, cache, nixl = host_dirs(tmp_path)
    result, argv = run_launcher(
        tmp_path,
        "--models",
        str(models),
        "--cache",
        str(cache),
        "--nixl-root",
        str(nixl),
    )
    assert result.returncode == 0, result.stderr
    assert argv[-3:] == ["exec", "bash", "/config/start-flash-next-frspec.sh"]
    assert values_after(argv, "--volume")[0] == f"{CONFIGS}:/config:ro"
    assert values_after(argv, "--volume")[3] == f"{nixl}:/nixl"
    # Nothing was named, so the selected script keeps the config it names.
    assert "-e" not in argv


def test_the_baked_image_command_starts_the_native_mtp_recipe(tmp_path):
    # A bare container start hands the Dockerfile CMD to the entry point, so the
    # baked default has to name the maintained native-MTP profile. An explicit
    # DFlash2 choice still has to reach its own routed recipe.
    dockerfile = (ENTRYPOINT.parent / "Dockerfile").read_text()
    cmd_lines = [line for line in dockerfile.splitlines() if line.startswith("CMD [")]
    assert cmd_lines == ['CMD ["next"]'], cmd_lines

    def dispatch(profile_argument: str | None):
        root, capture, env_capture = image_root(tmp_path)
        capture.unlink(missing_ok=True)
        env_capture.unlink(missing_ok=True)
        cache = tmp_path / "cache root"
        nixl = tmp_path / "nixl root"
        for directory in (cache, nixl):
            directory.mkdir(exist_ok=True)
        result = subprocess.run(
            [
                "bash",
                str(ENTRYPOINT),
                *([profile_argument] if profile_argument else []),
            ],
            env=clean_env(
                REPO_ROOT=str(root),
                CACHE_BASE=str(cache),
                NIXL_STORAGE_BASE=str(nixl),
                SGLANG_CAPTURE=str(capture),
                SGLANG_ENV_CAPTURE=str(env_capture),
                HOME=str(tmp_path / "home"),
                TARGET_MODEL=str(target_checkpoint(tmp_path)),
                DRAFT_MODEL=str(draft_checkpoint(tmp_path)),
                **{"FAKE_CUDA_DEVICES": "2", **qualified_defaults()},
            ),
            text=True,
            capture_output=True,
            check=False,
        )
        argv = capture.read_text().splitlines() if capture.exists() else []
        return result, argv

    result, argv = dispatch("next-dflash2")
    assert result.returncode == 0, result.stderr
    assert argv_after(argv, "--speculative-algorithm") == "DFLASH"
    assert argv_after(argv, "--speculative-num-steps") == "5"
    assert argv_after(argv, "--speculative-num-draft-tokens") == "6"
    assert argv_after(argv, "--speculative-adaptive-config").endswith(
        "adaptive-dflash2.json"
    )
    assert argv_after(argv, "--speculative-token-map").endswith("flash-next-64k.pt")
    assert "--cuda-graph-bs-decode" not in argv

    result, argv = dispatch("next")
    # The FR-Spec recipe pins the qualified target tokenizer, so the stand-in
    # checkpoint stops it inside that recipe — which is the dispatch proof, and
    # the server line is never reached.
    assert result.returncode != 0
    assert "qualified FR-Spec tokenizer" in result.stderr, result.stderr
    assert argv == []

    # A missing argument selects the same maintained native-MTP default.
    result, argv = dispatch(None)
    assert result.returncode != 0
    assert "qualified FR-Spec tokenizer" in result.stderr, result.stderr
    assert argv == []

    result, argv = dispatch("next-plain")
    assert result.returncode == 0, result.stderr
    assert argv_after(argv, "--speculative-algorithm") == "NEXTN"
    assert argv_after(argv, "--speculative-num-steps") == "7"
    assert "--speculative-token-map" not in argv


def test_launcher_resolves_relative_host_paths(tmp_path):
    # A --volume value without a leading '/' names a volume, not a host
    # directory, so relative operator paths must reach docker fully resolved --
    # including when the path itself contains a space.
    work = tmp_path / "operator dir"
    (work / "models").mkdir(parents=True)
    (work / "cache").mkdir(parents=True)
    (work / "nixl root").mkdir(parents=True)
    (work / "cfg dir" / "start up.sh").parent.mkdir()
    (work / "cfg dir" / "start up.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (work / "cfg dir" / "nixl conf.toml").write_text("use_direct_io = true\n")

    result, argv = run_launcher(
        tmp_path,
        "--startup",
        "cfg dir/start up.sh",
        "--nixl-config",
        "cfg dir/nixl conf.toml",
        "--models",
        "models",
        "--cache",
        "cache",
        "--nixl-root",
        "nixl root",
        cwd=work,
    )
    assert result.returncode == 0, result.stderr
    assert values_after(argv, "--volume") == [
        f"{work / 'cfg dir'}:/config:ro",
        f"{work / 'models'}:/models:ro",
        f"{work / 'cache'}:/cache",
        f"{work / 'nixl root'}:/nixl",
    ]
    assert argv[-3:] == ["exec", "bash", "/config/start up.sh"]
    assert values_after(argv, "-e") == ["NIXL_CONFIG=/config/nixl conf.toml"]


def nixl_volumes(argv: list[str]) -> list[str]:
    return [item for item in values_after(argv, "--volume") if item.endswith(":/nixl")]


def test_launcher_nixl_off_skips_the_mount_but_not_nvme_io_uring(tmp_path):
    models, cache, _ = host_dirs(tmp_path)
    common = ("--models", str(models), "--cache", str(cache))

    result, argv = run_launcher(tmp_path, "--no-nixl", *common)
    assert result.returncode == 0, result.stderr
    assert nixl_volumes(argv) == []
    assert "--security-opt" not in argv

    result, argv = run_launcher(tmp_path, "--no-nixl", "--nvme-ple", *common)
    assert result.returncode == 0, result.stderr
    # The independent NVMe PLE reader needs io_uring even with NIXL switched off.
    assert values_after(argv, "--security-opt") == ["seccomp=unconfined"]
    assert nixl_volumes(argv) == []


def test_launcher_refuses_missing_or_unreadable_inputs(tmp_path):
    models, cache, nixl = host_dirs(tmp_path)
    common = (
        "--models",
        str(models),
        "--cache",
        str(cache),
        "--nixl-root",
        str(nixl),
    )

    absent = tmp_path / "absent.sh"
    result, argv = run_launcher(tmp_path, "--startup", str(absent), *common)
    assert result.returncode != 0 and argv is None
    assert "Startup script is missing or unreadable" in result.stderr

    # A directory is not a startup script, and no image default fills in.
    result, argv = run_launcher(tmp_path, "--startup", str(models), *common)
    assert result.returncode != 0 and argv is None
    assert "Startup script is missing or unreadable" in result.stderr

    unreadable = tmp_path / "unreadable.sh"
    unreadable.write_text("exit 0\n")
    unreadable.chmod(0o000)
    if not os.access(unreadable, os.R_OK):  # skip when root or ACLs allow the read
        result, argv = run_launcher(tmp_path, "--startup", str(unreadable), *common)
        assert result.returncode != 0 and argv is None
        assert "Startup script is missing or unreadable" in result.stderr
    unreadable.chmod(0o644)

    absent_toml = tmp_path / "absent.toml"
    result, argv = run_launcher(tmp_path, *common, "--nixl-config", str(absent_toml))
    assert result.returncode != 0 and argv is None
    assert "NIXL config is missing or unreadable" in result.stderr

    missing_nixl_root = tmp_path / "no-nixl-here"
    result, argv = run_launcher(
        tmp_path, *common, "--nixl-root", str(missing_nixl_root)
    )
    assert result.returncode != 0 and argv is None
    assert "NIXL root is missing or not writable" in result.stderr

    cache.chmod(0o500)
    try:
        result, argv = run_launcher(tmp_path, *common, "--cache", str(cache))
        assert result.returncode != 0 and argv is None
        assert "Cache root is missing or not writable" in result.stderr
    finally:
        cache.chmod(0o700)

    result, argv = run_launcher(tmp_path, *common, "--bogus")
    assert result.returncode != 0 and argv is None
    assert "Unknown option" in result.stderr


# --------------------------------------------------- image + mounted startup


def make_checkpoint(path: Path, marker: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"marker": marker}))
    (path / "tokenizer.json").write_text(json.dumps({"marker": marker}))
    weights = path / "model.safetensors"
    if not weights.is_file():
        header = json.dumps(
            {"tensor": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
        ).encode()
        weights.write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"marker": marker}})
    )
    return path


def target_checkpoint(tmp_path: Path) -> Path:
    return make_checkpoint(tmp_path / "model-share", "target")


def draft_checkpoint(tmp_path: Path) -> Path:
    return make_checkpoint(tmp_path / "model-share" / "draft", "draft")


def image_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A stand-in /opt/pennyroyal: the real helpers reached through symlinks,
    the venv binaries replaced by recorders, one commit for the namespace
    identity. Built once so two launches in a test share the same identity."""
    root = tmp_path / "opt pennyroyal"
    capture = root / "server-argv"
    env_capture = root / "server-env"
    if (root / ".venv" / "bin" / "sglang").exists():
        return root, capture, env_capture
    (root / ".venv" / "bin").mkdir(parents=True)
    # A probe-accurate stand-in: like torch, it reports device_count() through
    # CUDA_VISIBLE_DEVICES, and only FAKE_CUDA_DEVICES says how many GPUs
    # Docker granted the container. A launcher that narrows the visible list
    # therefore really does hide a device from the TP guard.
    fake_python = root / ".venv" / "bin" / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'code="${*}"\n'
        'if [[ "$code" != *device_count* ]]; then\n'
        "  printf 2.9.0\n"
        "  exit 0\n"
        "fi\n"
        'if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then\n'
        '  IFS=, read -r -a requested <<< "$CUDA_VISIBLE_DEVICES"\n'
        '  count="${#requested[@]}"\n'
        "else\n"
        '  count="${FAKE_CUDA_DEVICES:-1}"\n'
        "fi\n"
        'if [[ "$code" == *range* ]]; then\n'
        '  printf "%s" "$(seq -s, 0 $((count - 1)))"\n'
        "else\n"
        '  printf "%s" "$count"\n'
        "fi\n"
    )
    fake_python.chmod(0o755)
    sglang = root / ".venv" / "bin" / "sglang"
    sglang.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ': > "$SGLANG_CAPTURE"\n'
        ': > "$SGLANG_ENV_CAPTURE"\n'
        'for argument in "$@"; do printf "%s\\n" "$argument" >> "$SGLANG_CAPTURE"; done\n'
        'printf "namespace=%s\\n" "${SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
        'printf "cache=%s\\n" "${SGLANG_CACHE_DIR:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
        'printf "tilelang=%s\\n" "${TILELANG_CACHE_DIR:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
        'printf "cuda=%s\\n" "${CUDA_VISIBLE_DEVICES:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
        # The settings the launcher can forward with -e, recorded as the server
        # actually receives them: a mounted script that exports a fixed value
        # over an operator's saved choice shows up here, not in the plan.
        'printf "forward_unknown_tools=%s\\n" "${SGLANG_FORWARD_UNKNOWN_TOOLS:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
        'printf "mm_device=%s\\n" "${SGLANG_MM_PREPROCESS_DEVICE:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
        'printf "max_running_requests=%s\\n" "${MAX_RUNNING_REQUESTS:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
        'printf "ple_backend=%s\\n" "${PENNY_PLE_BACKEND:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
        'printf "nccl_p2p_disable=%s\\n" "${NCCL_P2P_DISABLE:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
        'printf "gdn_fp16_accum_mma=%s\\n" "${FLASHINFER_GDN_FP16_ACCUM_MMA:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
    )
    sglang.chmod(0o755)
    (root / "configs").symlink_to(REPO / "configs")
    (root / "scripts").symlink_to(REPO / "scripts")
    (root / "VERSION").write_text("stand-in image\n")
    for step in (
        ["git", "init", "-q", str(root)],
        ["git", "-C", str(root), "config", "user.email", "t@example.com"],
        ["git", "-C", str(root), "config", "user.name", "T"],
        ["git", "-C", str(root), "add", "VERSION"],
        ["git", "-C", str(root), "commit", "-qm", "stand-in image"],
    ):
        subprocess.run(step, check=True)
    return root, capture, env_capture


# The FR-Spec launchers hash the checkpoint tokenizer and refuse anything but the
# qualified one, which a synthetic checkpoint cannot be. Reporting that one digest
# lets the FR-Spec routes be checked on the argv they actually build; every other
# digest (the FR-Spec map, the chat template) is still computed for real.
QUALIFIED_TOKENIZER_SHA256 = (
    "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
)


def qualified_tokenizer_path(tmp_path: Path) -> str:
    directory = tmp_path / "qualified tokenizer bin"
    directory.mkdir(exist_ok=True)
    real = shutil.which("sha256sum")
    stub = directory / "sha256sum"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'for argument in "$@"; do\n'
        '  case "$argument" in\n'
        "    */tokenizer.json)\n"
        f"      printf '{QUALIFIED_TOKENIZER_SHA256}  {{}}\\n' \"$argument\"\n"
        "      exit 0 ;;\n"
        "  esac\n"
        "done\n"
        f'exec "{real}" "$@"\n'
    )
    stub.chmod(0o755)
    return str(directory)


def qualified_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {
        "PATH": f"{qualified_tokenizer_path(tmp_path)}:{os.environ['PATH']}",
        **extra,
    }


def prepared_config(
    tmp_path: Path,
    *names: str,
    with_nixl_config: bool = True,
    nixl: str = "on",
    media: str | None = None,
    tp: str | None = None,
    policy: str | None = None,
    mm_cache: str | None = None,
    mm_budget: str | None = None,
) -> Path:
    """Copy the shipped examples into the operator's own directory and edit the
    settings there, which is exactly what an operator does before starting."""
    directory = tmp_path / "launch config"
    directory.mkdir(exist_ok=True)
    if with_nixl_config:
        for toml in ("nixl-posix.toml", "nixl-posix-frspec.toml"):
            (directory / toml).write_text((CONFIGS / toml).read_text())
    target = str(target_checkpoint(tmp_path))
    draft = str(draft_checkpoint(tmp_path))
    for name in names:
        text = (CONFIGS / name).read_text()
        for old, new in (
            (
                'TARGET_MODEL="/models/RadixArk-Qwen3.8-Flash-Next-NVFP4"',
                f'TARGET_MODEL="{target}"',
            ),
            ('TARGET_MODEL="/models/Qwen3.8-27B-FP8"', f'TARGET_MODEL="{target}"'),
            (
                'DRAFT_MODEL="/models/incoai-Qwen3.8-Flash-Next-DFlash2"',
                f'DRAFT_MODEL="{draft}"',
            ),
            ('DRAFT_MODEL="/models/Qwen3.8-27B-DFlash2"', f'DRAFT_MODEL="{draft}"'),
            ("NIXL=on\n", f"NIXL={nixl}\n"),
        ):
            if old in text or new == old:
                assert old in text, (name, old)
            text = text.replace(old, new)
        if media is not None:
            old = "export SGLANG_MM_PREPROCESS_DEVICE=cpu\n"
            assert old in text, name
            text = text.replace(old, f"export SGLANG_MM_PREPROCESS_DEVICE={media}\n")
        if tp is not None:
            assert "TP_SIZE=1\n" in text, name
            text = text.replace("TP_SIZE=1\n", f"TP_SIZE={tp}\n")
        if mm_cache is not None or mm_budget is not None:
            # The settings block, not the shell, decides these there too: this is
            # the edit ./configure-penny --container makes for an operator.
            assert "PENNY_MM_PREPROCESS_CACHE_DIR=\n" in text, name
            assert "PENNY_MM_PREPROCESS_CACHE_SIZE_MB=\n" in text, name
            if mm_cache is not None:
                text = text.replace(
                    "PENNY_MM_PREPROCESS_CACHE_DIR=\n",
                    f"PENNY_MM_PREPROCESS_CACHE_DIR={shlex.quote(mm_cache)}\n",
                )
            if mm_budget is not None:
                text = text.replace(
                    "PENNY_MM_PREPROCESS_CACHE_SIZE_MB=\n",
                    f"PENNY_MM_PREPROCESS_CACHE_SIZE_MB={mm_budget}\n",
                )
        if policy is not None:
            # The settings block, not the shell, is where a container operator
            # chooses the width policy: this is the edit ./configure-penny makes.
            assert "PENNY_MTP_POLICY=tiered\n" in text, name
            text = text.replace(
                "PENNY_MTP_POLICY=tiered\n", f"PENNY_MTP_POLICY={policy}\n"
            )
        assert f'TARGET_MODEL="{target}"' in text, name
        assert f"NIXL={nixl}\n" in text, name
        (directory / name).write_text(text)
    return directory


def launch(
    tmp_path: Path,
    startup: Path | None,
    *,
    through_entrypoint: bool = True,
    keep_nixl_root: bool = False,
    nixl_root: Path | None = None,
    env: dict | None = None,
    profile: str | None = None,
):
    root, capture, env_capture = image_root(tmp_path)
    capture.unlink(missing_ok=True)
    env_capture.unlink(missing_ok=True)
    cache = tmp_path / "cache root"
    cache.mkdir(exist_ok=True)
    nixl = tmp_path / "nixl root"
    nixl.mkdir(exist_ok=True)
    if nixl_root is not None:
        nixl = nixl_root
    if profile is not None:
        # The built-in profile route: the entry point plus a container command,
        # which is exactly what a bare 'docker run' of the image starts with.
        command = ["bash", str(ENTRYPOINT), profile]
    elif through_entrypoint:
        command = ["bash", str(ENTRYPOINT), "exec", "bash", str(startup)]
    else:
        command = ["bash", str(startup)]
    result = subprocess.run(
        command,
        env=clean_env(
            REPO_ROOT=str(root),
            CACHE_BASE=str(cache),
            # keep_nixl_root leaves the image's own /nixl default in place.
            NIXL_STORAGE_BASE="/nixl" if keep_nixl_root else str(nixl),
            SGLANG_EXE=str(root / ".venv" / "bin" / "sglang"),
            PYTHON=str(root / ".venv" / "bin" / "python"),
            SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR="",
            HOME=str(tmp_path / "home"),
            SGLANG_CAPTURE=str(capture),
            SGLANG_ENV_CAPTURE=str(env_capture),
            TARGET_MODEL=str(target_checkpoint(tmp_path)),
            DRAFT_MODEL=str(draft_checkpoint(tmp_path)),
            # Two GPUs granted by Docker unless the test says otherwise, and a
            # per-test override of any qualified default (one dict, so an
            # override cannot collide with the default it replaces).
            **{"FAKE_CUDA_DEVICES": "2", **qualified_defaults(), **(env or {})},
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    argv = capture.read_text().splitlines() if capture.exists() else None
    recorded = env_capture.read_text().splitlines() if env_capture.exists() else []
    server_env = dict(line.split("=", 1) for line in recorded)
    return result, argv, server_env, (root, cache, nixl)


def test_examples_are_editable_startup_files_with_container_paths(tmp_path):
    assert (LAUNCH / "run.sh").is_file()
    assert sorted(path.name for path in CONFIGS.glob("start-*.sh")) == [
        "start-27b-dflash2.sh",
        "start-flash-next-dflash2.sh",
        "start-flash-next-frspec.sh",
        "start-flash-next.sh",
    ]
    for name in (
        "start-flash-next.sh",
        "start-flash-next-frspec.sh",
        "start-flash-next-dflash2.sh",
        "start-27b-dflash2.sh",
    ):
        text = (CONFIGS / name).read_text()
        assert 'TARGET_MODEL="/models/' in text, name
        assert 'REPO_ROOT="${REPO_ROOT:-/opt/pennyroyal}"' in text, name
        # Helpers, the pinned template and the pinned map stay inside the image.
        assert 'source "$IMAGE_CONFIGS/chat-template.sh"' in text, name
        assert "TARGET_OVERRIDES='" in text, name
        # The disk tier is a setting inside the normal file, on by default, and
        # it gates only the NIXL root, config, helper and backend arguments.
        assert "NIXL=on\n" in text, name
        assert "HICACHE_STORAGE_ARGS=()" in text, name
        gate = '--hicache-mem-layout page_first "${HICACHE_STORAGE_ARGS[@]}"'
        assert gate in text, name
    assert (CONFIGS / "nixl-posix.toml").is_file()
    assert (CONFIGS / "nixl-posix-frspec.toml").is_file()


def test_mounted_startup_reaches_the_server_unchanged(tmp_path):
    config = prepared_config(tmp_path, "start-flash-next.sh")
    result, argv, server_env, (root, cache, nixl) = launch(
        tmp_path, config / "start-flash-next.sh"
    )
    assert result.returncode == 0, result.stderr
    assert argv is not None and argv[0] == "serve"
    # Qualified launch flags, exactly as the in-image recipe passes them.
    for flag, value in (
        ("--quantization", "modelopt_fp4"),
        ("--kv-cache-dtype", "fp8_e4m3"),
        ("--mem-fraction-static", "0.981"),
        ("--context-length", "524288"),
        ("--page-size", "64"),
        ("--mamba-track-interval", "64"),
        ("--max-running-requests", "4"),
        ("--max-mamba-cache-size", "24"),
        ("--hicache-size", "32"),
        ("--hicache-storage-backend", "nixl"),
        ("--speculative-algorithm", "NEXTN"),
        ("--speculative-num-draft-tokens", "8"),
    ):
        assert argv_after(argv, flag) == value, flag
    assert "--ple-offload-embedding" in argv  # RAM PLE stays on
    assert argv_after(argv, "--chat-template").endswith(
        "configs/pennyroyal/templates/froggeric-v22.5.jinja"
    )
    assert json.loads(argv_after(argv, "--default-chat-template-kwargs")) == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "medium",
    }
    # The NIXL operational config travels from the mounted directory to SGLang.
    assert argv_after(argv, "--hicache-storage-backend-extra-config") == (
        f"@{config / 'nixl-posix.toml'}"
    )
    assert argv_after(argv, "--model-path") == str(tmp_path / "model-share")
    # The writable roots, the derived namespace and the GPU view Docker granted
    # are the launcher's own; nothing here hides a granted device.
    assert str(root) in result.stdout  # the runtime really is the image's
    assert server_env["cuda"] == "0,1"
    assert (cache / "sglang" / "jit").is_dir()
    namespace = Path(server_env["namespace"])
    assert namespace.is_dir() and str(namespace).startswith(str(nixl))
    assert (namespace / "namespace-identity.json").is_file()
    assert server_env["cache"] == str(cache / "sglang")


def test_tilelang_kernels_stay_on_the_durable_mount(tmp_path):
    # An arbitrary --user gets HOME=/, so a launch path that leaves the TileLang
    # cache to ~ aborts at its first import (#65); the documented default UID
    # keeps the kernels in the disposable image layer instead. Checked in the
    # launched process's own environment, for both profiles and both routes, and
    # an explicit operator path stays the operator's.
    for startup, recipe in (
        ("start-flash-next.sh", RECIPES / "serve-flash-next.sh"),
        ("start-27b-dflash2.sh", RECIPES / "serve-qwen38-27b-dflash2.sh"),
    ):
        config = prepared_config(tmp_path, startup)
        for path, via_entrypoint in ((config / startup, True), (recipe, False)):
            result, _, server_env, (_, cache, _) = launch(
                tmp_path, path, through_entrypoint=via_entrypoint
            )
            assert result.returncode == 0, (path, result.stderr)
            assert server_env["tilelang"] == str(cache / "tilelang"), path
            assert (cache / "tilelang").is_dir(), path
        chosen = tmp_path / "operator kernels"
        result, _, server_env, _ = launch(
            tmp_path, config / startup, env={"TILELANG_CACHE_DIR": str(chosen)}
        )
        assert result.returncode == 0, result.stderr
        assert server_env["tilelang"] == str(chosen), server_env
        assert chosen.is_dir()


def test_startup_names_the_release_once_on_every_route(tmp_path):
    # Where an operator looks when a launch dies in preflight: the release and
    # its source, once, before any preflight step, on the mounted container route
    # and the native recipe route, for both model profiles. The stand-in image
    # root is a tagless single-commit checkout, like the published image.
    for startup, recipe in (
        ("start-flash-next.sh", RECIPES / "serve-flash-next.sh"),
        ("start-27b-dflash2.sh", RECIPES / "serve-qwen38-27b-dflash2.sh"),
    ):
        config = prepared_config(tmp_path, startup)
        for path, via_entrypoint in ((config / startup, True), (recipe, False)):
            result, _, _, (root, _, _) = launch(
                tmp_path, path, through_entrypoint=via_entrypoint
            )
            assert result.returncode == 0, (path, result.stderr)
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--short=10", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            banner = f"Penny Royal {PENNY_VERSION} | source {revision}"
            lines = result.stdout.splitlines()
            assert lines[0] == banner, (path, lines[:3])
            assert lines.count(banner) == 1, (path, result.stdout)


def test_mounted_startup_27b_keeps_its_profile(tmp_path):
    config = prepared_config(tmp_path, "start-27b-dflash2.sh")
    result, argv, _, _ = launch(tmp_path, config / "start-27b-dflash2.sh")
    assert result.returncode == 0, result.stderr
    assert argv[0] == "serve"
    for flag, value in (
        ("--speculative-algorithm", "DFLASH"),
        ("--speculative-num-draft-tokens", "8"),
        ("--speculative-draft-window-size", "2048"),
        ("--speculative-draft-kv-cache-dtype", "fp8_e4m3"),
        ("--hicache-size", "96"),
        ("--hicache-storage-backend", "nixl"),
        ("--max-mamba-cache-size", "24"),
        ("--mamba-track-interval", "256"),
        ("--decode-attention-backend", "trtllm_mha"),
    ):
        assert argv_after(argv, flag) == value, flag
    assert argv_after(argv, "--speculative-draft-model-path").endswith("draft")
    assert argv_after(argv, "--hicache-storage-backend-extra-config") == (
        f"@{config / 'nixl-posix.toml'}"
    )


def test_disk_tier_off_is_a_setting_inside_every_example(tmp_path):
    nixl = tmp_path / "nixl root"
    nixl.mkdir(parents=True, exist_ok=True)
    (nixl / "old-namespace" / "bucket").mkdir(parents=True)
    (nixl / "old-namespace" / "bucket" / "entry").write_bytes(b"cached")
    for name in ("start-flash-next.sh", "start-27b-dflash2.sh"):
        # No NIXL toml was supplied and only the image's own /nixl default
        # exists, so this launch cannot touch the disk tier at all.
        config = prepared_config(tmp_path, name, nixl="off", with_nixl_config=False)
        result, argv, server_env, _ = launch(
            tmp_path, config / name, keep_nixl_root=True
        )
        assert result.returncode == 0, (name, result.stderr)
        for flag in (
            "--hicache-storage-backend",
            "--hicache-storage-prefetch-policy",
            "--hicache-storage-backend-extra-config",
        ):
            assert flag not in argv, (name, flag)
        # The GPU radix cache, the host-RAM tier and the profile stay intact.
        assert "--enable-hierarchical-cache" in argv, name
        assert argv_after(argv, "--hicache-io-backend") == "kernel", name
        assert argv_after(argv, "--hicache-mem-layout") == "page_first", name
        assert argv_after(argv, "--hicache-write-policy") == "write_through", name
        assert argv_after(argv, "--hicache-size") in ("32", "96"), name
        assert argv_after(argv, "--speculative-algorithm") in ("NEXTN", "DFLASH"), name
        assert "--ple-offload-embedding" in argv or name != "start-flash-next.sh", name
        assert server_env["namespace"] == "unset", name
    # No NIXL work also means nothing created in, or deleted from, the root.
    assert (nixl / "old-namespace" / "bucket" / "entry").read_bytes() == b"cached"
    assert sorted(path.name for path in nixl.iterdir()) == ["old-namespace"]


def test_disk_tier_off_in_the_frspec_example_needs_no_nixl_root(tmp_path):
    config = prepared_config(
        tmp_path,
        "start-flash-next-frspec.sh",
        nixl="off",
        with_nixl_config=False,
    )
    result, argv, _, _ = launch(
        tmp_path, config / "start-flash-next-frspec.sh", keep_nixl_root=True
    )
    # The launch reaches the pinned tokenizer instead of failing on a NIXL root
    # or config, so the disk tier really is out of this path.
    assert result.returncode != 0 and argv is None
    assert "Mount a writable directory at /nixl" not in result.stderr
    assert "NIXL config missing" not in result.stderr
    assert "tokenizer differs from the qualified FR-Spec tokenizer" in result.stderr


def test_container_startup_keeps_every_granted_gpu_visible(tmp_path):
    # The exec entrypoint skips the built-in profile's device discovery, so the
    # startup script must keep the whole Docker-granted set visible; narrowing
    # it to 0..TP_SIZE-1 hides the optional cuda:N media processor.
    plain = prepared_config(tmp_path, "start-flash-next.sh")
    result, _, server_env, _ = launch(
        tmp_path, plain / "start-flash-next.sh", env={"CUDA_VISIBLE_DEVICES": "1"}
    )
    assert result.returncode == 0, result.stderr
    assert server_env["cuda"] == "1"  # an explicit selection still wins

    dedicated = prepared_config(tmp_path, "start-flash-next.sh", media="cuda:1")
    startup = dedicated / "start-flash-next.sh"
    result, argv, server_env, _ = launch(tmp_path, startup)
    assert result.returncode == 0, result.stderr
    assert server_env["cuda"] == "0,1"
    assert argv_after(argv, "--tp") == "1"
    assert argv_after(argv, "--image-processor-backend") == "torchvision"

    # The TP guard still refuses a device Docker never granted.
    result, argv, _, _ = launch(tmp_path, startup, env={"FAKE_CUDA_DEVICES": "1"})
    assert result.returncode != 0 and argv is None
    assert "needs at least 2 visible CUDA device(s)" in result.stderr


def test_missing_nixl_config_is_not_replaced_by_an_image_default(tmp_path):
    config = prepared_config(tmp_path, "start-flash-next.sh", with_nixl_config=False)
    result, argv, _, _ = launch(tmp_path, config / "start-flash-next.sh")
    assert result.returncode != 0 and argv is None
    assert "NIXL config missing" in result.stderr
    assert str(config / "nixl-posix.toml") in result.stderr


def test_frspec_startup_keeps_its_pinned_assets(tmp_path):
    # The stand-in checkpoint cannot match the pinned FR-Spec tokenizer, so the
    # mounted recipe must stop rather than serve an unqualified tokenizer.
    config = prepared_config(tmp_path, "start-flash-next-frspec.sh")
    result, argv, _, _ = launch(tmp_path, config / "start-flash-next-frspec.sh")
    assert result.returncode != 0 and argv is None
    assert "Verifying the pinned FR-Spec map" in result.stdout
    assert "tokenizer differs from the qualified FR-Spec tokenizer" in result.stderr


def test_mounted_recipe_launches_exactly_like_the_image_recipe(tmp_path):
    """Same environment in, same server argv out -- for both NIXL profiles."""
    for startup, recipe in (
        ("start-flash-next.sh", RECIPES / "serve-flash-next.sh"),
        ("start-27b-dflash2.sh", RECIPES / "serve-qwen38-27b-dflash2.sh"),
    ):
        config = prepared_config(tmp_path, startup)
        nixl_config = config / "nixl.toml"
        nixl_config.write_text((CONFIGS / "nixl-posix.toml").read_text())
        mounted_result, mounted_argv, _, (root, _, _) = launch(
            tmp_path, config / startup, env={"NIXL_CONFIG": str(nixl_config)}
        )
        recipe_result, recipe_argv, _, _ = launch(
            tmp_path,
            recipe,
            through_entrypoint=False,
            env={"NIXL_CONFIG": str(nixl_config)},
        )
        assert mounted_result.returncode == 0, mounted_result.stderr
        assert recipe_result.returncode == 0, recipe_result.stderr
        assert mounted_argv is not None and recipe_argv is not None, recipe
        assert "--hicache-storage-backend" in recipe_argv
        # The mounted copy reaches the helpers through the stand-in image root;
        # the recipe sees the same files at their repository paths.
        normalised = [
            item.replace(f"{root}/configs/", f"{REPO}/configs/")
            for item in mounted_argv
        ]
        assert normalised == recipe_argv, recipe


STORAGE_FLAGS = (
    "--hicache-storage-backend",
    "--hicache-storage-prefetch-policy",
    "--hicache-storage-backend-extra-config",
)


def without_storage_args(argv: list[str]) -> list[str]:
    """The launch argv with the three disk-tier arguments and their values cut."""
    kept: list[str] = []
    skip = False
    for item in argv:
        if skip:
            skip = False
            continue
        if item in STORAGE_FLAGS:
            skip = True
            continue
        kept.append(item)
    return kept


def test_forwarded_settings_survive_the_mounted_script_for_every_profile(tmp_path):
    # What the operator saved is what the server sees: the launcher can forward
    # these with -e, and the mounted startup files keep their qualified default
    # only when nothing was forwarded. Checked in the launched process's own
    # environment, not in a plan dictionary.
    profiles = ("start-flash-next.sh", "start-27b-dflash2.sh")
    for name in profiles:
        config = prepared_config(tmp_path, name)
        startup = config / name
        default_result, default_argv, default_env, _ = launch(tmp_path, startup)
        assert default_result.returncode == 0, (name, default_result.stderr)
        # Nothing forwarded: the qualified defaults stay exactly as qualified.
        assert default_env["forward_unknown_tools"] == "true", name
        assert default_env["nccl_p2p_disable"] == "unset", name
        assert '"reasoning_effort":"medium"' in argv_after(
            default_argv, "--default-chat-template-kwargs"
        ), name
        result, argv, server_env, _ = launch(
            tmp_path, startup, env={"SGLANG_FORWARD_UNKNOWN_TOOLS": "false"}
        )
        assert result.returncode == 0, (name, result.stderr)
        assert server_env["forward_unknown_tools"] == "false", (
            name,
            server_env["forward_unknown_tools"],
        )
        # The value is not merely exported: the tool parser still qualifies.
        assert argv_after(argv, "--tool-call-parser") == "qwen3_coder", name
    # The FR-Spec example cannot launch against the stand-in checkpoint (its
    # pinned tokenizer guard stops it, as it should), so the same guarantee is
    # read from the file the operator edits instead.
    frspec = (CONFIGS / "start-flash-next-frspec.sh").read_text()
    assert (
        "export SGLANG_FORWARD_UNKNOWN_TOOLS=" '"${SGLANG_FORWARD_UNKNOWN_TOOLS:-true}"'
    ) in frspec
    assert "export SGLANG_FORWARD_UNKNOWN_TOOLS=true" not in frspec
    # The 27b recipe pins its capacity pair, TP and PLE placement in the launch
    # line, so an ambient value there is decoration: the saved settings must be
    # refused up front (penny_config's profile check), not quietly ignored. Its
    # KV cap is the exception — unset stays automatic and an explicit cap
    # travels to the server.
    result, argv, server_env, _ = launch(
        tmp_path,
        RECIPES / "serve-qwen38-27b-dflash2.sh",
        through_entrypoint=False,
        env={
            "MAX_RUNNING_REQUESTS": "8",
            "MAX_MAMBA_CACHE_SIZE": "48",
            "MAX_TOTAL_TOKENS": "262144",
            "TP_SIZE": "2",
        },
    )
    assert result.returncode == 0, result.stderr
    assert argv_after(argv, "--max-running-requests") == "4"
    assert argv_after(argv, "--max-mamba-cache-size") == "24"
    assert argv_after(argv, "--tp") == "1"
    assert argv_after(argv, "--max-total-tokens") == "262144"


def test_a_generated_launch_forwards_the_saved_value_end_to_end(tmp_path):
    """Generated files, docker stub and the image stand-in: saved false stays false.

    This is the chain the review asked for: the configurator's generated run.sh
    must hand the saved value to docker, and the generated startup script must
    let it reach the launched server instead of exporting its own default. No
    daemon or GPU here: docker is a recorder and the server is a capturing stub.
    """
    sys.path.insert(0, str(REPO / "scripts" / "pennyroyal"))
    import penny_config as pc  # noqa: E402

    host_root = tmp_path / "model-share"
    cache = tmp_path / "cache root"
    nixl = tmp_path / "nixl root"
    for directory in (host_root, cache, nixl):
        directory.mkdir(parents=True, exist_ok=True)
    model = make_checkpoint(host_root / "penny-model", "generated")
    saved = {
        "HOST_MODELS_ROOT": str(host_root),
        "HOST_CACHE_BASE": str(cache),
        "HOST_NIXL_STORAGE_BASE": str(nixl),
        "LAUNCH_DIR": str(tmp_path / "generated"),
        "SGLANG_FORWARD_UNKNOWN_TOOLS": "false",
        # Saved but not one of the wizard's own keys: docker compose forwards
        # these today, so the generated launch must not lose them either.
        "PENNY_REASONING_EFFORT": "high",
        "NCCL_P2P_DISABLE": "1",
        "FLASHINFER_GDN_FP16_ACCUM_MMA": "0",
        "TARGET_MODEL": "/models/penny-model",
    }
    env_file = tmp_path / "container.env"
    env_file.write_text(
        pc.serialize_env(
            [("", sorted(saved.items()))], header=(f"{pc.PROFILE_KEY}=next-plain",)
        )
    )
    plan = pc.build_plan(
        "container", pc.load_config("container", env_file, {}, REPO), {}, repo_root=REPO
    )
    assert [issue.message for issue in plan.errors] == []
    pc.write_container_files(plan)
    run_sh = plan.launch_dir / "run.sh"
    assert run_sh.is_file(), plan.launch_dir

    # 1. Host side: the printed command reaches docker with the saved value.
    directory, capture = fake_docker(tmp_path)
    result = subprocess.run(
        ["bash", str(run_sh)],
        env=clean_env(
            PATH=f"{directory}:{os.environ['PATH']}",
            DOCKER_CAPTURE=str(capture),
            HOME=str(tmp_path / "home"),
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    argv = capture.read_text().splitlines()
    index = argv.index("-e")
    forwarded = [value for value in argv[index:] if value.startswith("SGLANG_")]

    # 2. Container side: the same generated script, run by the image's own
    # exec path with exactly the -e pairs docker was handed, exports what it
    # received (paths rewritten the way an operator rewrites them before the
    # first start).
    passed_through = dict(
        argv[index + 1].split("=", 1)
        for index, value in enumerate(argv)
        if value == "-e" and "=" in argv[index + 1]
    )
    startup = plan.launch_dir / "config" / "start-flash-next.sh"
    generated = startup.read_text()
    assert "TARGET_MODEL=/models/penny-model" in generated, generated
    startup.write_text(
        generated.replace("TARGET_MODEL=/models/penny-model", f'TARGET_MODEL="{model}"')
    )
    result, server_argv, server_env, _ = launch(tmp_path, startup, env=passed_through)
    assert result.returncode == 0, result.stderr
    assert server_env["forward_unknown_tools"] == "false", server_env
    assert server_env["cache"] == str(cache / "sglang"), server_env
    # The launcher-side reasoning tier reached the chat-template kwargs the
    # server is started with, and the NCCL workaround reached its environment.
    assert '"reasoning_effort":"high"' in argv_after(
        server_argv, "--default-chat-template-kwargs"
    ), server_argv
    assert server_env["nccl_p2p_disable"] == "1", server_env
    # The saved GDN opt-out survives the generated startup file's own default,
    # which is the only reason to forward it.
    assert server_env["gdn_fp16_accum_mma"] == "0", server_env
    # The profile's qualified flags are untouched by any of this.
    assert argv_after(server_argv, "--speculative-algorithm") == "NEXTN"

    # Nothing forwarded either: the same generated file keeps its qualified
    # default of 1 rather than honouring an absent setting.
    unforwarded = {
        k: v for k, v in passed_through.items() if k != "FLASHINFER_GDN_FP16_ACCUM_MMA"
    }
    assert "FLASHINFER_GDN_FP16_ACCUM_MMA" not in unforwarded
    result, _, default_env, _ = launch(tmp_path, startup, env=unforwarded)
    assert result.returncode == 0, result.stderr
    assert default_env["gdn_fp16_accum_mma"] == "1", default_env
    # Saved 0 and nothing saved are different representations of the recurrent
    # prefill computation, so they must not land on one persisted NIXL root.
    assert server_env["namespace"] != "unset", server_env
    assert default_env["namespace"] != server_env["namespace"], (
        default_env,
        server_env,
    )


def test_native_recipes_take_the_same_disk_tier_switch(tmp_path):
    # The native recipes carry the same on/off choice the mounted startup files
    # do: off drops the three storage-backend arguments (and every NIXL
    # requirement), and keeps the GPU radix cache, the host-RAM tier, the model
    # and the speculation flags exactly as they were. No data is deleted: the
    # namespace derivation simply never runs.
    for recipe in ("serve-flash-next.sh", "serve-qwen38-27b-dflash2.sh"):
        path = RECIPES / recipe
        on_result, on_argv, on_env, _ = launch(tmp_path, path, through_entrypoint=False)
        assert on_result.returncode == 0, (recipe, on_result.stderr)
        assert "--hicache-storage-backend" in on_argv, recipe
        assert on_env["namespace"] != "unset", recipe
        off_result, off_argv, off_env, _ = launch(
            tmp_path,
            path,
            through_entrypoint=False,
            env={"NIXL": "off"},
            nixl_root=tmp_path / "no nixl root here",
        )
        assert off_result.returncode == 0, (recipe, off_result.stderr)
        for flag in STORAGE_FLAGS:
            assert flag not in off_argv, (recipe, flag)
        # Everything except those three arguments (and their values) is the
        # same argv, and no NIXL path survived into it.
        assert without_storage_args(on_argv) == off_argv, recipe
        assert not [item for item in off_argv if "nixl" in item.lower()], recipe
        assert "--enable-hierarchical-cache" in off_argv, recipe
        assert off_env["namespace"] == "unset", recipe
        assert off_env["cache"] == on_env["cache"], recipe

    # A value that is neither on nor off is a mistake, not a new mode.
    bogus, argv, _, _ = launch(
        tmp_path,
        RECIPES / "serve-flash-next.sh",
        through_entrypoint=False,
        env={"NIXL": "sometimes"},
    )
    assert bogus.returncode != 0 and argv is None
    assert "NIXL must be on or off" in bogus.stderr


# ------------------------------------------------------------- MTP width policy

# One setting, three approved policies. The name maps to a bundled policy file
# and to the launch width that can carry it, and the pair comes from the image's
# configs/pennyroyal/mtp-policy.sh, so the native recipe and its mounted copy
# cannot disagree about what either selection means.
MTP_POLICIES = (
    # exported value, approved policy file, steps, draft tokens
    ("tiered", "adaptive-next.json", "7", "8"),
    ("adaptive", "adaptive-next-v3.0.2.json", "7", "8"),
    ("fixed-w4", "adaptive-next-fixed-w4.json", "3", "4"),
)


def test_the_mtp_policy_selection_reaches_the_real_launch_argv(tmp_path):
    # Not an enum parse: the server binary records the argv it was handed, for
    # the native recipe (which reads the exported value) and for the same
    # selection written into the mounted startup script (which is where a
    # container launch keeps the setting, beside the capacity knobs).
    identities = {}
    for value, asset, steps, draft_tokens in MTP_POLICIES + (
        (None, "adaptive-next.json", "7", "8"),
    ):
        mounted = (
            prepared_config(tmp_path, "start-flash-next.sh", policy=value or "tiered")
            / "start-flash-next.sh"
        )
        for startup, via_entrypoint, env in (
            (
                RECIPES / "serve-flash-next.sh",
                False,
                {} if value is None else {"PENNY_MTP_POLICY": value},
            ),
            (mounted, True, {}),
        ):
            result, argv, server_env, _ = launch(
                tmp_path, startup, through_entrypoint=via_entrypoint, env=env
            )
            assert result.returncode == 0, (value, startup, result.stderr)
            # The startup report names the same selection the argv carries.
            assert f"MTP width policy: {value or 'tiered'}" in result.stdout, value
            assert argv_after(argv, "--speculative-num-steps") == steps, value
            assert argv_after(argv, "--speculative-eagle-topk") == "1", value
            assert (
                argv_after(argv, "--speculative-num-draft-tokens") == draft_tokens
            ), value
            assert argv_after(argv, "--speculative-adaptive-config").endswith(
                asset
            ), value
            assert "--speculative-adaptive" in argv, value
            # The selection is cache identity, so no policy can be served out of
            # another one's persisted namespace.
            identities[value or "unset"] = server_env["namespace"]
    assert identities["unset"] == identities["tiered"], identities
    assert len(set(identities.values())) == 3, identities

    # A value exported in front of the mounted startup script changes nothing:
    # the settings block decides there, exactly as it decides TP_SIZE and the
    # HiCache tier, so a saved or hand-edited launch cannot be steered by an
    # inherited shell.
    shipped = prepared_config(tmp_path, "start-flash-next.sh") / "start-flash-next.sh"
    _, exported_argv, _, _ = launch(
        tmp_path, shipped, through_entrypoint=True, env={"PENNY_MTP_POLICY": "fixed-w4"}
    )
    _, plain_argv, _, _ = launch(tmp_path, shipped, through_entrypoint=True)
    assert exported_argv == plain_argv
    assert argv_after(plain_argv, "--speculative-adaptive-config").endswith(
        "adaptive-next.json"
    )

    # The two DFlash2 recipes and the 27B recipe route their own widths and never
    # read the setting: an exported value there changes neither their launch line
    # nor the cache identity they persist under (the knob is native-only, not a
    # reason these profiles cannot be selected).
    for startup in (
        RECIPES / "serve-flash-next-dflash2.sh",
        prepared_config(tmp_path, "start-flash-next-dflash2.sh")
        / "start-flash-next-dflash2.sh",
        RECIPES / "serve-qwen38-27b-dflash2.sh",
    ):
        via_entrypoint = startup.name.startswith("start-")
        _, baseline_argv, baseline_env, _ = launch(
            tmp_path, startup, through_entrypoint=via_entrypoint
        )
        _, argv, server_env, _ = launch(
            tmp_path,
            startup,
            through_entrypoint=via_entrypoint,
            env={"PENNY_MTP_POLICY": "fixed-w4"},
        )
        assert argv == baseline_argv, startup
        assert server_env["namespace"] == baseline_env["namespace"], startup
        if startup.name == "serve-flash-next-dflash2.sh":
            # The routed Flash-Next profile keeps its own widest block and file.
            assert argv_after(argv, "--speculative-num-steps") == "5", startup
            assert argv_after(argv, "--speculative-num-draft-tokens") == "6", startup
            assert argv_after(argv, "--speculative-adaptive-config").endswith(
                "adaptive-dflash2.json"
            ), startup


def test_an_unusable_mtp_policy_stops_the_launch_with_a_reason(tmp_path):
    # Refused on both routes: an inherited value reaching the recipe, and a value
    # written into the startup script by hand or by setup.
    mounted = (
        prepared_config(tmp_path, "start-flash-next.sh", policy="w6")
        / "start-flash-next.sh"
    )
    for startup, via_entrypoint, env in (
        (RECIPES / "serve-flash-next.sh", False, {"PENNY_MTP_POLICY": "w6"}),
        (mounted, True, {}),
    ):
        result, argv, _, _ = launch(
            tmp_path, startup, through_entrypoint=via_entrypoint, env=env
        )
        assert result.returncode != 0 and argv is None, startup
        assert "PENNY_MTP_POLICY must be tiered, adaptive or fixed-w4" in result.stderr


def test_the_launcher_adds_no_environment_of_its_own(tmp_path):
    # run.sh inherits nothing and forwards only what the generated settings put
    # in its docker arguments, so the width policy a container runs is the one in
    # the startup script: no ambient shell can reach it.
    startup = prepared_config(tmp_path, "start-flash-next.sh") / "start-flash-next.sh"
    models, cache, nixl = host_dirs(tmp_path)
    options = [
        "--startup",
        str(startup),
        "--models",
        str(models),
        "--cache",
        str(cache),
        "--nixl-root",
        str(nixl),
        "--image",
        "pennyroyal:test",
    ]
    for exported in ({}, {"PENNY_MTP_POLICY": "fixed-w4"}):
        result, argv = run_launcher(tmp_path, *options, env=exported)
        assert result.returncode == 0, result.stderr
        assert "-e" not in argv, (exported, argv)


def test_a_generated_launch_runs_the_policy_the_setup_saved(tmp_path):
    """Generated files, the docker recorder and the image stand-in.

    The whole chain the acceptance asks for: what ./configure-penny answered is
    what the launch runs, an inherited shell value cannot steer it, and the
    settings line the save wrote is the one place the choice lives.
    """
    sys.path.insert(0, str(REPO / "scripts" / "pennyroyal"))
    import penny_config as pc  # noqa: E402

    host_root = tmp_path / "model share"
    cache = tmp_path / "cache root"
    nixl = tmp_path / "nixl root"
    for directory in (host_root, cache, nixl):
        directory.mkdir(parents=True)
    model = make_checkpoint(host_root / "penny-model", "generated")
    saved = {
        "HOST_MODELS_ROOT": str(host_root),
        "HOST_CACHE_BASE": str(cache),
        "HOST_NIXL_STORAGE_BASE": str(nixl),
        "LAUNCH_DIR": str(tmp_path / "generated"),
        "TARGET_MODEL": "/models/penny-model",
        "PENNY_MTP_POLICY": "adaptive",
    }
    env_file = tmp_path / "container.env"
    env_file.write_text(
        pc.serialize_env(
            [("", sorted(saved.items()))], header=(f"{pc.PROFILE_KEY}=next-plain",)
        )
    )
    # The save runs with a different policy in the shell: saved wins, as it does
    # for every other setting, so a stray export cannot rewrite the folder.
    plan = pc.build_plan(
        "container",
        pc.load_config("container", env_file, {}, REPO),
        {"PENNY_MTP_POLICY": "fixed-w4"},
        repo_root=REPO,
    )
    assert [issue.message for issue in plan.errors] == []
    assert plan.env["PENNY_MTP_POLICY"] == "adaptive", plan.env
    assert any(
        "MTP width policy: adaptive" in line and "saved file" in line
        for line in plan.summary
    ), plan.summary
    pc.write_container_files(plan)
    run_sh = plan.launch_dir / "run.sh"
    startup = plan.launch_dir / "config" / "start-flash-next.sh"
    generated = startup.read_text()
    assert "PENNY_MTP_POLICY=adaptive" in generated, generated
    startup.write_text(
        generated.replace("TARGET_MODEL=/models/penny-model", f'TARGET_MODEL="{model}"')
    )

    directory, capture = fake_docker(tmp_path)

    def launch_generated(**exported):
        run = subprocess.run(
            ["bash", str(run_sh)],
            env=clean_env(
                PATH=f"{directory}:{os.environ['PATH']}",
                DOCKER_CAPTURE=str(capture),
                HOME=str(tmp_path / "home"),
                **exported,
            ),
            text=True,
            capture_output=True,
            check=False,
        )
        assert run.returncode == 0, run.stderr + run.stdout
        argv = capture.read_text().splitlines()
        assert not [
            value for value in argv if value.startswith("PENNY_MTP_POLICY=")
        ], argv
        # And the container side, by the image's own exec path: the generated
        # startup file carries the setting, so nothing needs to be forwarded.
        result, server_argv, _, _ = launch(tmp_path, startup, env=exported)
        assert result.returncode == 0, result.stderr
        return argv_after(server_argv, "--speculative-adaptive-config"), argv_after(
            server_argv, "--speculative-num-steps"
        )

    policy_file, steps = launch_generated()
    assert policy_file.endswith("adaptive-next-v3.0.2.json") and steps == "7", (
        policy_file,
        steps,
    )
    # An inherited value does not steer a generated launch either.
    policy_file, steps = launch_generated(PENNY_MTP_POLICY="fixed-w4")
    assert policy_file.endswith("adaptive-next-v3.0.2.json") and steps == "7", (
        policy_file,
        steps,
    )
    # The documented way to change it is the setting itself: edit the line the
    # save wrote (or run setup again) and the same launch switches.
    startup.write_text(
        startup.read_text().replace(
            "PENNY_MTP_POLICY=adaptive", "PENNY_MTP_POLICY=fixed-w4"
        )
    )
    policy_file, steps = launch_generated()
    assert policy_file.endswith("adaptive-next-fixed-w4.json") and steps == "3", (
        policy_file,
        steps,
    )


# ------------------------------------- optional shared image preprocessing cache

# One directory plus one MiB budget, and the pair means the same thing on every
# public launch path: the four native recipes (which read the exported pair, the
# route the built-in container profiles and Compose take) and their four mounted
# startup copies (whose settings block decides, the way ./configure-penny writes
# it). Both the FR-Spec pair are checked here: the recipe with the pinned
# tokenizer cannot complete against a stand-in checkpoint, but the pair is still
# resolved, and refused, before that check runs.
CACHE_PROFILES = (
    ("serve-flash-next.sh", "start-flash-next.sh"),
    ("serve-flash-next-frspec.sh", "start-flash-next-frspec.sh"),
    ("serve-flash-next-dflash2.sh", "start-flash-next-dflash2.sh"),
    ("serve-qwen38-27b-dflash2.sh", "start-27b-dflash2.sh"),
)


def test_the_image_cache_pair_reaches_the_real_argv_on_every_launch_path(tmp_path):
    directory = "/mm cache with space/shared"
    for recipe_name, startup_name in CACHE_PROFILES:
        for startup, via_entrypoint in (
            (RECIPES / recipe_name, False),
            (
                prepared_config(
                    tmp_path, startup_name, mm_cache=directory, mm_budget="3072"
                )
                / startup_name,
                True,
            ),
        ):
            # The recipe route reads the exported pair; the mounted copy reads
            # the same pair from its own settings block, which is what a
            # ./configure-penny --container save writes.
            env = qualified_env(tmp_path)
            if not via_entrypoint:
                env.update(
                    {
                        "PENNY_MM_PREPROCESS_CACHE_DIR": directory,
                        "PENNY_MM_PREPROCESS_CACHE_SIZE_MB": "3072",
                    }
                )
            result, argv, _, _ = launch(
                tmp_path, startup, through_entrypoint=via_entrypoint, env=env
            )
            assert result.returncode == 0, (startup, result.stderr, result.stdout)
            assert f"Image preprocessing cache: shared at {directory} | 3072 MiB" in (
                result.stdout
            ), (startup, result.stdout)
            assert argv_after(argv, "--mm-preprocess-cache-dir") == directory, startup
            assert argv_after(argv, "--mm-preprocess-cache-size-mb") == "3072", startup


def image_root_without(tmp_path: Path, *helper_names: str) -> Path:
    """The stand-in image one release older: this tree with those helpers gone.

    A mounted startup file is read from whichever image the operator chose, and a
    retained image pin is a documented choice, so a new launch file inside an older
    image is a real launch rather than a hypothetical.
    """
    root, _, _ = image_root(tmp_path)
    (root / "configs").unlink()
    (root / "configs" / "pennyroyal").mkdir(parents=True)
    for entry in (REPO / "configs" / "pennyroyal").iterdir():
        if entry.name not in helper_names:
            (root / "configs" / "pennyroyal" / entry.name).symlink_to(entry)
    return root


def test_an_image_without_the_helper_launches_unchanged_and_says_why_when_asked(
    tmp_path,
):
    # The pair is off by default, and the shared helper is a new file: an image
    # pinned to v3.1.5 does not carry it. Every mounted startup file therefore has
    # to launch that image exactly as it launched before these settings existed,
    # and refuse by name - rather than quietly ignoring - a cache it cannot build.
    image_root_without(tmp_path, "mm-preprocess-cache.sh")
    for startup_name in (
        "start-flash-next.sh",
        "start-flash-next-frspec.sh",
        "start-flash-next-dflash2.sh",
        "start-27b-dflash2.sh",
    ):
        startup = prepared_config(tmp_path, startup_name) / startup_name
        result, argv, _, _ = launch(tmp_path, startup, env=qualified_env(tmp_path))
        assert result.returncode == 0, (startup_name, result.stderr)
        assert "--mm-preprocess-cache-dir" not in argv, startup_name
        assert "--mm-preprocess-cache-size-mb" not in argv, startup_name
        assert "Image preprocessing cache: off" in result.stdout, startup_name

    # Asked for, and refused with the file that is missing named, because a
    # silently ignored cache setting is how an operator ends up wondering why
    # nothing changed.
    startup = (
        prepared_config(
            tmp_path,
            "start-flash-next.sh",
            mm_cache="/cache/mm work",
            mm_budget="256",
        )
        / "start-flash-next.sh"
    )
    result, argv, _, _ = launch(tmp_path, startup, env=qualified_env(tmp_path))
    assert result.returncode != 0, argv
    assert "mm-preprocess-cache.sh" in result.stderr, result.stderr
    assert "v3.1.6" in result.stderr, result.stderr


def test_the_built_in_image_profiles_take_the_same_pair(tmp_path):
    # The route a bare `docker run` of the image takes: the entrypoint plus a
    # profile name, with no mounted startup file to carry the settings. The pair
    # has to reach the server argv there too, on all four built-in profiles.
    directory = "/mm cache with space/shared"
    for profile in ("next", "next-plain", "next-dflash2", "27b"):
        result, argv, _, _ = launch(
            tmp_path,
            None,
            profile=profile,
            env=qualified_env(
                tmp_path,
                **{
                    "PENNY_MM_PREPROCESS_CACHE_DIR": directory,
                    "PENNY_MM_PREPROCESS_CACHE_SIZE_MB": "3072",
                },
            ),
        )
        assert result.returncode == 0, (profile, result.stderr, result.stdout)
        assert argv_after(argv, "--mm-preprocess-cache-dir") == directory, profile
        assert argv_after(argv, "--mm-preprocess-cache-size-mb") == "3072", profile

    # Nothing exported: the built-in profile launches without either flag, which
    # is the launch every existing image user gets after this update.
    result, argv, _, _ = launch(
        tmp_path, None, profile="next-plain", env=qualified_env(tmp_path)
    )
    assert result.returncode == 0, result.stderr
    assert "--mm-preprocess-cache-dir" not in argv
    assert "--mm-preprocess-cache-size-mb" not in argv
    assert "Image preprocessing cache: off" in result.stdout, result.stdout

    # An explicit blank, which is what saving `none` writes into a settings file:
    # Compose forwards it as an empty passthrough and the entrypoint has to read
    # that as 'turned off', not as 'unset, so let an ambient value through'.
    result, argv, _, _ = launch(
        tmp_path,
        None,
        profile="next-plain",
        env=qualified_env(
            tmp_path,
            **{
                "PENNY_MM_PREPROCESS_CACHE_DIR": "",
                "PENNY_MM_PREPROCESS_CACHE_SIZE_MB": "",
            },
        ),
    )
    assert result.returncode == 0, result.stderr
    assert "--mm-preprocess-cache-dir" not in argv
    assert "--mm-preprocess-cache-size-mb" not in argv
    assert "Image preprocessing cache: off" in result.stdout, result.stdout


def test_a_budget_without_a_directory_is_still_the_per_worker_cache(tmp_path):
    # The setting pair has to keep the old behaviour for anyone who only ever
    # sets the budget: one flag, no directory, and an explicit 0 that means the
    # per-worker cache is off. Nothing is asked for, nothing is launched, and the
    # report says so.
    startup = prepared_config(tmp_path, "start-flash-next.sh") / "start-flash-next.sh"
    for budget in ("0", "256"):
        for target, via_entrypoint, env in (
            (
                RECIPES / "serve-flash-next.sh",
                False,
                {"PENNY_MM_PREPROCESS_CACHE_SIZE_MB": budget},
            ),
            (startup, True, {}),
        ):
            if via_entrypoint:
                target = (
                    prepared_config(tmp_path, "start-flash-next.sh", mm_budget=budget)
                    / "start-flash-next.sh"
                )
            result, argv, _, _ = launch(
                tmp_path, target, through_entrypoint=via_entrypoint, env=env
            )
            assert result.returncode == 0, (budget, target, result.stderr)
            assert argv_after(argv, "--mm-preprocess-cache-size-mb") == budget
            assert "--mm-preprocess-cache-dir" not in argv
            assert (
                f"Image preprocessing cache: no shared directory | {budget} MiB"
                in result.stdout
            ), result.stdout

    # Neither setting at all: no flag, and the report says the cache is off. That
    # is the launch every existing saved file and startup script still has.
    result, argv, _, _ = launch(
        tmp_path, RECIPES / "serve-flash-next.sh", through_entrypoint=False
    )
    assert result.returncode == 0, result.stderr
    assert "--mm-preprocess-cache-size-mb" not in argv
    assert "--mm-preprocess-cache-dir" not in argv
    assert "Image preprocessing cache: off" in result.stdout, result.stdout


def test_a_directory_without_a_positive_budget_stops_the_launch(tmp_path):
    # Refused on both routes, before the model loads, with the reason naming both
    # settings: a shared store without a budget is undefined, not unlimited. The
    # mounted copy is refused from its own settings block, so neither a
    # hand-edited nor a generated launch folder can carry the broken pair, and
    # the FR-Spec route is refused before its pinned-tokenizer check.
    directory = {"PENNY_MM_PREPROCESS_CACHE_DIR": "/dev/shm/mm"}
    zeroed = {**directory, "PENNY_MM_PREPROCESS_CACHE_SIZE_MB": "0"}
    relative = {**directory, "PENNY_MM_PREPROCESS_CACHE_SIZE_MB": "512"}
    relative["PENNY_MM_PREPROCESS_CACHE_DIR"] = "relative/mm"
    plain = (
        prepared_config(tmp_path, "start-flash-next.sh", mm_cache="/dev/shm/mm")
        / "start-flash-next.sh"
    )
    frspec = (
        prepared_config(tmp_path, "start-flash-next-frspec.sh", mm_cache="/dev/shm/mm")
        / "start-flash-next-frspec.sh"
    )
    for startup, via_entrypoint, env, reason in (
        (RECIPES / "serve-flash-next.sh", False, directory, "needs a positive"),
        (RECIPES / "serve-flash-next.sh", False, zeroed, "needs a positive"),
        (RECIPES / "serve-flash-next.sh", False, relative, "must be an absolute"),
        (plain, True, {}, "needs a positive"),
        (frspec, True, {}, "needs a positive"),
    ):
        result, argv, _, _ = launch(
            tmp_path, startup, through_entrypoint=via_entrypoint, env=env
        )
        assert result.returncode != 0 and argv is None, (startup, env, result.stdout)
        assert reason in result.stderr, (startup, env, result.stderr)


def test_a_generated_launch_keeps_the_image_cache_pair_it_was_saved(tmp_path):
    """Saved settings, generated files, the docker recorder and the image stand-in.

    The pair a setup answered is the pair the launch runs: it is written into the
    generated startup script, an inherited shell value cannot steer it, editing
    the line the save wrote is how it changes, and saving nothing leaves the
    shipped blank line, which is off.
    """
    sys.path.insert(0, str(REPO / "scripts" / "pennyroyal"))
    import penny_config as pc  # noqa: E402

    host_root = tmp_path / "model share"
    cache = tmp_path / "cache root"
    nixl = tmp_path / "nixl root"
    for directory in (host_root, cache, nixl):
        directory.mkdir(parents=True)
    model = make_checkpoint(host_root / "penny-model", "generated")
    saved = {
        "HOST_MODELS_ROOT": str(host_root),
        "HOST_CACHE_BASE": str(cache),
        "HOST_NIXL_STORAGE_BASE": str(nixl),
        "LAUNCH_DIR": str(tmp_path / "generated"),
        "TARGET_MODEL": "/models/penny-model",
        "PENNY_MM_PREPROCESS_CACHE_DIR": "/cache/mm work",
        "PENNY_MM_PREPROCESS_CACHE_SIZE_MB": "1024",
    }
    env_file = tmp_path / "container.env"
    env_file.write_text(
        pc.serialize_env(
            [("", sorted(saved.items()))], header=(f"{pc.PROFILE_KEY}=next-plain",)
        )
    )
    # A different pair in the caller's shell: the saved settings win, as they do
    # for every other setting, so no stray export can rewrite the launch folder.
    plan = pc.build_plan(
        "container",
        pc.load_config("container", env_file, {}, REPO),
        {
            "PENNY_MM_PREPROCESS_CACHE_DIR": "/dev/shm/from-the-shell",
            "PENNY_MM_PREPROCESS_CACHE_SIZE_MB": "4096",
        },
        repo_root=REPO,
    )
    assert [issue.message for issue in plan.errors] == []
    assert plan.env["PENNY_MM_PREPROCESS_CACHE_DIR"] == "/cache/mm work", plan.env
    pc.write_container_files(plan)
    startup = plan.launch_dir / "config" / "start-flash-next.sh"
    generated = startup.read_text()
    assert "PENNY_MM_PREPROCESS_CACHE_DIR='/cache/mm work'" in generated, generated
    assert "PENNY_MM_PREPROCESS_CACHE_SIZE_MB=1024" in generated, generated
    startup.write_text(
        generated.replace("TARGET_MODEL=/models/penny-model", f'TARGET_MODEL="{model}"')
    )

    def server_arguments(**exported):
        # The generated script, run by the image's own exec path, with whatever
        # the caller's shell happens to carry.
        result, server_argv, _, _ = launch(tmp_path, startup, env=exported)
        assert result.returncode == 0, result.stderr
        return server_argv

    argv = server_arguments()
    assert argv_after(argv, "--mm-preprocess-cache-dir") == "/cache/mm work", argv
    assert argv_after(argv, "--mm-preprocess-cache-size-mb") == "1024", argv
    # An inherited value does not steer a generated launch either: the settings
    # block the save wrote is the one place the pair lives.
    argv = server_arguments(
        PENNY_MM_PREPROCESS_CACHE_DIR="/dev/shm/from-the-shell",
        PENNY_MM_PREPROCESS_CACHE_SIZE_MB="4096",
    )
    assert argv_after(argv, "--mm-preprocess-cache-dir") == "/cache/mm work", argv
    assert argv_after(argv, "--mm-preprocess-cache-size-mb") == "1024", argv
    # The documented way to change it is the setting the save wrote.
    startup.write_text(
        startup.read_text()
        .replace(
            "PENNY_MM_PREPROCESS_CACHE_DIR='/cache/mm work'",
            "PENNY_MM_PREPROCESS_CACHE_DIR=",
        )
        .replace(
            "PENNY_MM_PREPROCESS_CACHE_SIZE_MB=1024",
            "PENNY_MM_PREPROCESS_CACHE_SIZE_MB=",
        )
    )
    argv = server_arguments()
    assert "--mm-preprocess-cache-dir" not in argv, argv
    assert "--mm-preprocess-cache-size-mb" not in argv, argv


def test_native_frspec_disk_tier_off_needs_no_nixl_root(tmp_path):
    # The FR-Spec recipe reaches its pinned-tokenizer check instead of failing
    # on a NIXL root, config or namespace helper it no longer needs -- and the
    # root it does not need does not even have to exist.
    result, argv, _, _ = launch(
        tmp_path,
        RECIPES / "serve-flash-next-frspec.sh",
        through_entrypoint=False,
        env={"NIXL": "off"},
        nixl_root=tmp_path / "no nixl root here",
    )
    assert result.returncode != 0 and argv is None
    assert "NIXL config missing" not in result.stderr
    assert "Required executable missing" not in result.stderr
    assert "tokenizer differs from the qualified FR-Spec tokenizer" in (result.stderr)


if __name__ == "__main__":  # pytest is the real driver; this is a smoke check
    import inspect
    import tempfile

    for name, case in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(case):
            with tempfile.TemporaryDirectory() as temp:
                case(Path(temp))
            print(f"{name} ok")
