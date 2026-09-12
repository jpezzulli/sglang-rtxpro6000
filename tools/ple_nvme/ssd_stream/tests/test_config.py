import hashlib
import json
from argparse import Namespace

import pytest
from sglang_ssd_stream import config


def _artifact(tmp_path, *, table_bytes=b"a" * 160, path="ple/table.bin"):
    root = tmp_path / "model"
    table = root / path
    table.parent.mkdir(parents=True)
    table.write_bytes(table_bytes)
    manifest = root / "ssd-stream.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "sglang-ssd-stream",
                "version": 1,
                "tables": [
                    {
                        "layer": 0,
                        "path": path,
                        "sha256": hashlib.sha256(table_bytes).hexdigest(),
                        "dtype": "float8_e4m3fn",
                        "rows": 1,
                        "columns": 160,
                        "row_start": 0,
                        "bytes": 160,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root, manifest, table


def test_manifest_resolves_immutable_table(tmp_path):
    _, manifest, table = _artifact(tmp_path)

    resolved = config.load_manifest(manifest)

    assert resolved.manifest_path == manifest.resolve()
    assert resolved.table_for_layer(0).path == table.resolve()
    assert resolved.table_for_layer(0).sha256 == hashlib.sha256(b"a" * 160).hexdigest()


def test_manifest_keeps_huggingface_snapshot_as_relative_root(tmp_path):
    blob_root = tmp_path / "blobs"
    blob_root.mkdir()
    table_blob = blob_root / "table"
    table_blob.write_bytes(b"a" * 160)
    manifest_blob = blob_root / "manifest"
    manifest_blob.write_text(
        json.dumps(
            {
                "format": "sglang-ssd-stream",
                "version": 1,
                "tables": [
                    {
                        "layer": 0,
                        "path": "ple/table.bin",
                        "sha256": "1" * 64,
                        "dtype": "float8_e4m3fn",
                        "rows": 1,
                        "columns": 160,
                        "bytes": 160,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    (snapshot / "ple").mkdir(parents=True)
    (snapshot / "ssd-stream.json").symlink_to(manifest_blob)
    (snapshot / "ple" / "table.bin").symlink_to(table_blob)

    resolved = config.load_manifest(snapshot / "ssd-stream.json")

    assert resolved.manifest_path == snapshot / "ssd-stream.json"
    assert resolved.table_for_layer(0).path == table_blob


def test_cli_pins_huggingface_snapshot_without_changing_table_identity(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("SGLANG_SSD_STREAM_CONFIG", raising=False)
    _, manifest, _ = _artifact(tmp_path)
    commit = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
    monkeypatch.setattr(
        config,
        "_resolve_artifact",
        lambda model_path, revision: (manifest, commit),
    )
    namespace = Namespace(
        model_path="garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream",
        revision=None,
        ple_offload_embedding=False,
        cpu_offload_gb=0,
        offload_group_size=-1,
        disable_cuda_graph=False,
        disable_decode_cuda_graph=False,
        disable_prefill_cuda_graph=False,
        cuda_graph_backend_decode=None,
        cuda_graph_backend_prefill=None,
        speculative_algorithm=None,
        startup_weight_load_mode="normal",
    )

    config.configure_cli(namespace)

    resolved = config.get_config()
    assert namespace.revision == commit
    assert namespace.ple_offload_embedding is False
    assert config.grouped_cpu_offload_enabled() is False
    assert resolved.table_for_layer(0).sha256 == hashlib.sha256(b"a" * 160).hexdigest()


def test_cpu_offload_requires_graphs_and_mtp_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("SGLANG_SSD_STREAM_CONFIG", raising=False)
    _, manifest, _ = _artifact(tmp_path)
    monkeypatch.setattr(
        config,
        "_resolve_artifact",
        lambda model_path, revision: (manifest, "a" * 40),
    )
    namespace = Namespace(
        model_path="prepared",
        revision=None,
        ple_offload_embedding=True,
        cpu_offload_gb=0,
        offload_group_size=1,
        disable_cuda_graph=False,
        disable_decode_cuda_graph=False,
        disable_prefill_cuda_graph=False,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        speculative_algorithm=None,
        startup_weight_load_mode="normal",
    )

    config.configure_cli(namespace)

    assert namespace.ple_offload_embedding is False
    assert config.grouped_cpu_offload_enabled() is True

    namespace.speculative_algorithm = "NEXTN"
    with pytest.raises(ValueError, match="does not support MTP"):
        config.configure_cli(namespace)


def test_manifest_rejects_wrong_table_size(tmp_path):
    _, manifest, _ = _artifact(tmp_path, table_bytes=b"short")

    with pytest.raises(ValueError, match="has 5 bytes, expected 160"):
        config.load_manifest(manifest)


def test_manifest_rejects_parent_path(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    manifest = root / "ssd-stream.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "sglang-ssd-stream",
                "version": 1,
                "tables": [
                    {
                        "layer": 0,
                        "path": "../table.bin",
                        "sha256": "1" * 64,
                        "dtype": "float8_e4m3fn",
                        "rows": 1,
                        "columns": 160,
                        "bytes": 160,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must stay inside"):
        config.load_manifest(manifest)
