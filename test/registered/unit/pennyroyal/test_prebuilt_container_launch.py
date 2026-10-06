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
    "SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR",
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


def run_launcher(tmp_path: Path, *options: str, cwd: Path | None = None):
    directory, capture = fake_docker(tmp_path)
    result = subprocess.run(
        ["bash", str(LAUNCH / "run.sh"), *options],
        cwd=cwd,
        env=clean_env(
            PATH=f"{directory}:{os.environ['PATH']}",
            DOCKER_CAPTURE=str(capture),
            HOME=str(tmp_path / "home"),
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
        'printf "cuda=%s\\n" "${CUDA_VISIBLE_DEVICES:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
        # The settings the launcher can forward with -e, recorded as the server
        # actually receives them: a mounted script that exports a fixed value
        # over an operator's saved choice shows up here, not in the plan.
        'printf "forward_unknown_tools=%s\\n" "${SGLANG_FORWARD_UNKNOWN_TOOLS:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
        'printf "mm_device=%s\\n" "${SGLANG_MM_PREPROCESS_DEVICE:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
        'printf "max_running_requests=%s\\n" "${MAX_RUNNING_REQUESTS:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
        'printf "ple_backend=%s\\n" "${PENNY_PLE_BACKEND:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
        'printf "nccl_p2p_disable=%s\\n" "${NCCL_P2P_DISABLE:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
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


def prepared_config(
    tmp_path: Path,
    *names: str,
    with_nixl_config: bool = True,
    nixl: str = "on",
    media: str | None = None,
    tp: str | None = None,
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
        assert f'TARGET_MODEL="{target}"' in text, name
        assert f"NIXL={nixl}\n" in text, name
        (directory / name).write_text(text)
    return directory


def launch(
    tmp_path: Path,
    startup: Path,
    *,
    through_entrypoint: bool = True,
    keep_nixl_root: bool = False,
    nixl_root: Path | None = None,
    env: dict | None = None,
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
    command = (
        ["bash", str(ENTRYPOINT), "exec", "bash", str(startup)]
        if through_entrypoint
        else ["bash", str(startup)]
    )
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
        "start-flash-next-frspec.sh",
        "start-flash-next.sh",
    ]
    for name in (
        "start-flash-next.sh",
        "start-flash-next-frspec.sh",
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
        ("--speculative-num-draft-tokens", "4"),
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
    # The 27b recipe pins its own capacity, TP and PLE placement in the launch
    # line, so an ambient value there is decoration: the saved settings must be
    # refused up front (penny_config's profile check), not quietly ignored.
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
    assert "--max-total-tokens" not in argv


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
        # these two today, so the generated launch must not lose them either.
        "PENNY_REASONING_EFFORT": "high",
        "NCCL_P2P_DISABLE": "1",
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
    pc.write_container_files(plan, confirm=lambda text: True)
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
    # The profile's qualified flags are untouched by any of this.
    assert argv_after(server_argv, "--speculative-algorithm") == "NEXTN"


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
