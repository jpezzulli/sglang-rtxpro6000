import hashlib
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang_ssd_stream import __version__, config, plugin

REQUIRED_HOOK_TARGETS = (
    "sglang.srt.server_args.ServerArgs.from_cli_args",
    "sglang.srt.models.qwen3_5.make_layers",
    "sglang.srt.models.qwen4_exp.Qwen4ExpPLELayer",
    "sglang.srt.models.qwen4_exp.Qwen4ExpForConditionalGeneration.load_weights",
    (
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
        "DecodeCudaGraphRunner.execute"
    ),
    (
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
        "DecodeCudaGraphRunner.load_batch"
    ),
)


def test_corrected_adapter_version_is_explicit():
    assert __version__ == "0.2.0+pennyroyal2"


@pytest.fixture
def isolated_hook_registry(monkeypatch):
    original_apply_hooks = HookRegistry.__dict__["apply_hooks"]
    monkeypatch.setattr(HookRegistry, "_hooks", defaultdict(list))
    monkeypatch.setattr(HookRegistry, "_patched", set())
    monkeypatch.delattr(
        HookRegistry, "_pennyroyal_nvme_hooks_registered", raising=False
    )
    monkeypatch.delattr(HookRegistry, "_pennyroyal_nvme_hooks_applied", raising=False)
    try:
        yield
    finally:
        HookRegistry.apply_hooks = original_apply_hooks


def test_ram_registration_does_nothing(monkeypatch):
    monkeypatch.delenv("PENNY_PLE_BACKEND", raising=False)
    monkeypatch.setattr(
        plugin, "_register_pennyroyal", lambda: pytest.fail("RAM activated SSD")
    )
    plugin.register()
    monkeypatch.setenv("PENNY_PLE_BACKEND", "ram")
    plugin.register()


def test_explicit_nvme_failure_is_fatal(monkeypatch):
    monkeypatch.setenv("PENNY_PLE_BACKEND", "nvme")

    def broken():
        raise RuntimeError("bad source")

    monkeypatch.setattr(plugin, "_register_pennyroyal", broken)
    with pytest.raises(SystemExit, match="bad source"):
        plugin.register()


def test_payload_hash_is_checked(tmp_path):
    table = tmp_path / "table.bin"
    table.write_bytes(b"abcd")
    entry = config.SSDStreamTable(
        0, table, hashlib.sha256(b"abcd").hexdigest(), "float8_e4m3fn", 1, 4, 0, 4
    )
    artifact = config.SSDStreamConfig(tmp_path / "ssd-stream.json", (entry,))
    config.verify_tables(artifact)
    table.write_bytes(b"abce")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        config.verify_tables(artifact)


def test_source_guard_rejects_modified_module(monkeypatch, tmp_path):
    fake = tmp_path / "module.py"
    fake.write_text("changed source")
    from types import SimpleNamespace

    # Import before patching find_spec so dependency discovery isn't mocked.
    from sglang.srt.plugins import hook_registry  # noqa: F401

    monkeypatch.setattr(
        plugin.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(fake))
    )
    with pytest.raises(RuntimeError, match="SHA-256"):
        plugin._register_pennyroyal()


@pytest.mark.parametrize("failed_target", REQUIRED_HOOK_TARGETS)
def test_each_required_hook_application_failure_is_fatal(
    monkeypatch, isolated_hook_registry, failed_target
):
    applied = []

    def fault_injected_apply(cls, target, hooks):
        if target == failed_target:
            raise RuntimeError("injected hook failure")
        applied.append(target)

    monkeypatch.setattr(
        HookRegistry, "_apply_target", classmethod(fault_injected_apply)
    )
    plugin._register_pennyroyal()

    with pytest.raises(SystemExit, match="required hooks failed"):
        HookRegistry.apply_hooks()
    assert failed_target not in HookRegistry._patched
    assert set(applied) == set(REQUIRED_HOOK_TARGETS) - {failed_target}


def test_required_hook_enforcement_is_idempotent(
    monkeypatch, isolated_hook_registry
):
    applied = []

    def record_apply(cls, target, hooks):
        applied.append(target)

    monkeypatch.setattr(HookRegistry, "_apply_target", classmethod(record_apply))
    plugin._register_pennyroyal()
    plugin._register_pennyroyal()

    assert {
        target: len(HookRegistry._hooks[target]) for target in REQUIRED_HOOK_TARGETS
    } == {target: 1 for target in REQUIRED_HOOK_TARGETS}
    HookRegistry.apply_hooks()
    HookRegistry.apply_hooks()
    assert set(applied) == set(REQUIRED_HOOK_TARGETS)
    assert len(applied) == len(REQUIRED_HOOK_TARGETS)


def test_ram_launcher_helper_preserves_arguments():
    repo = Path(__file__).resolve().parents[4]
    helper = repo / "configs/pennyroyal/ple-backend.sh"
    env = dict(
        os.environ,
        PENNY_PLE_BACKEND="ram",
        TARGET_MODEL="original",
        PYTHONPATH="sentinel",
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n" "$TARGET_MODEL" "$PYTHONPATH" "${PLE_ARGS[*]}" "${#PLE_NAMESPACE_ARGS[@]}"',
            "bash",
            str(helper),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "original",
        "sentinel",
        "--ple-offload-embedding",
        "0",
    ]


def test_invalid_launcher_mode_fails():
    repo = Path(__file__).resolve().parents[4]
    helper = repo / "configs/pennyroyal/ple-backend.sh"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"', "bash", str(helper)],
        env=dict(os.environ, PENNY_PLE_BACKEND="typo"),
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
