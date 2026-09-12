#!/usr/bin/env python3
"""CPU-only NVMe PLE launcher preflight; print its namespace identity."""
# Malformed artifact values are consistently reported as ValueError.
# ruff: noqa: TRY004

import argparse
import hashlib
import importlib.metadata
import json
import struct
from pathlib import Path

import prepare_ple_nvme as preparer


def _load_ssd_stream_entrypoint(expected_register):
    entries = importlib.metadata.entry_points(group="sglang.srt.plugins")
    matches = [entry for entry in entries if entry.name == "ssd_stream"]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one ssd_stream entry point in sglang.srt.plugins, "
            f"found {len(matches)}"
        )
    entry = matches[0]
    distribution = entry.dist.name if entry.dist is not None else None
    if (
        distribution != "sglang-ssd-stream"
        or entry.value != "sglang_ssd_stream.plugin:register"
    ):
        raise ValueError(
            "ssd_stream entry point has wrong identity: "
            f"distribution={distribution}, value={entry.value}"
        )
    try:
        loaded = entry.load()
    except Exception as exc:
        raise ValueError(f"cannot load ssd_stream entry point: {exc}") from exc
    if loaded is not expected_register:
        raise ValueError(
            "ssd_stream entry point did not load sglang_ssd_stream.plugin.register"
        )


def _tensor_sha256(tensor: preparer.TensorLocation) -> str:
    digest = hashlib.sha256()
    remaining = tensor.nbytes
    with tensor.path.open("rb") as stream:
        stream.seek(tensor.offset)
        while remaining:
            chunk = stream.read(min(remaining, preparer._COPY_BYTES))
            if not chunk:
                raise ValueError(f"retained tensor ended early: {tensor.name}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _weight_map(index: dict, path: Path) -> dict[str, str]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{path} has no weight_map object")
    for name, raw_path in weight_map.items():
        if not isinstance(name, str):
            raise ValueError(f"{path} weight_map tensor names must be strings")
        preparer._safe_relative_path(raw_path)
    return weight_map


def _verify_prepared_weights(
    source: Path, prepared: Path, manifest: dict
) -> tuple[set[Path], set[Path]]:
    source_index_path = source / preparer.INDEX_NAME
    prepared_index_path = prepared / preparer.INDEX_NAME
    source_index = preparer._read_json(source_index_path)
    prepared_index = preparer._read_json(prepared_index_path)
    source_map = _weight_map(source_index, source_index_path)
    prepared_map = _weight_map(prepared_index, prepared_index_path)

    removed_names = {
        name for name in source_map if preparer._ple_coordinates(name) is not None
    }
    expected_map = {
        name: raw_path
        for name, raw_path in source_map.items()
        if name not in removed_names
    }
    if prepared_map != expected_map:
        missing = sorted(set(expected_map) - set(prepared_map))
        unexpected = sorted(set(prepared_map) - set(expected_map))
        changed = sorted(
            name
            for name in set(expected_map) & set(prepared_map)
            if expected_map[name] != prepared_map[name]
        )
        raise ValueError(
            "Prepared weight_map differs from source minus PLE tensors; "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )

    affected_relatives = {
        preparer._safe_relative_path(source_map[name]) for name in removed_names
    }
    if any(relative.parent != Path(".") for relative in affected_relatives):
        raise ValueError("Source PLE safetensors must be at the checkpoint root")
    provenance = manifest.get("source")
    if not isinstance(provenance, dict):
        raise ValueError("NVMe PLE manifest has no source provenance")
    raw_affected = provenance.get("ple_safetensors")
    if not isinstance(raw_affected, list):
        raise ValueError("NVMe PLE manifest has no source PLE file list")
    affected_provenance = {}
    for entry in raw_affected:
        if not isinstance(entry, dict):
            raise ValueError("NVMe PLE source file entry is not an object")
        relative = preparer._safe_relative_path(entry.get("path"))
        if relative in affected_provenance:
            raise ValueError(f"duplicate NVMe PLE source file entry: {relative}")
        affected_provenance[relative] = entry
    if set(affected_provenance) != affected_relatives:
        raise ValueError("NVMe PLE source file list differs from source weight_map")

    source_headers = {}
    removed_bytes = 0
    for relative in sorted(affected_relatives):
        source_path = source / relative
        header = preparer._read_safetensors_header(source_path)
        source_headers[relative] = header
        entry = affected_provenance[relative]
        if (
            entry.get("bytes") != source_path.stat().st_size
            or entry.get("header_sha256") != header.header_sha256
        ):
            raise ValueError(f"Source PLE safetensors changed: {relative}")
        locations = {tensor.name: tensor for tensor in header.tensors}
        expected_removed = {
            name
            for name in removed_names
            if preparer._safe_relative_path(source_map[name]) == relative
        }
        actual_removed = {
            tensor.name
            for tensor in header.tensors
            if preparer._ple_coordinates(tensor.name) is not None
        }
        if actual_removed != expected_removed:
            missing = sorted(expected_removed - actual_removed)
            unexpected = sorted(actual_removed - expected_removed)
            raise ValueError(
                "Source PLE safetensors differs from its weight_map; "
                f"missing={missing}, unexpected={unexpected}"
            )
        removed_bytes += sum(locations[name].nbytes for name in expected_removed)

    expected_index = dict(source_index)
    expected_index["weight_map"] = expected_map
    metadata = source_index.get("metadata")
    if isinstance(metadata, dict) and "total_size" in metadata:
        total_size = metadata["total_size"]
        if (
            not isinstance(total_size, int)
            or isinstance(total_size, bool)
            or total_size < removed_bytes
        ):
            raise ValueError("Source index metadata.total_size is invalid")
        expected_index["metadata"] = {
            **metadata,
            "total_size": total_size - removed_bytes,
        }
    if prepared_index != expected_index:
        raise ValueError("Prepared weight index differs from deterministic output")

    rewritten_relatives = set()
    for relative, source_header in source_headers.items():
        retained = tuple(
            tensor for tensor in source_header.tensors if tensor.name not in removed_names
        )
        if not retained:
            continue
        rewritten_relatives.add(relative)
        prepared_path = prepared / relative
        prepared_header = preparer._read_safetensors_header(prepared_path)
        encoded_header = preparer._encoded_safetensors_header(source_header, retained)
        expected_header_sha = hashlib.sha256(
            struct.pack("<Q", len(encoded_header)) + encoded_header
        ).hexdigest()
        if prepared_header.header_sha256 != expected_header_sha:
            raise ValueError(
                f"Prepared retained tensor header differs from source: {relative}"
            )
        if prepared_path.stat().st_size != (
            8 + len(encoded_header) + sum(tensor.nbytes for tensor in retained)
        ):
            raise ValueError(
                f"Prepared retained tensor file has wrong size: {relative}"
            )
        if len(prepared_header.tensors) != len(retained):
            raise ValueError(
                f"Prepared retained tensor set differs from source: {relative}"
            )
        for source_tensor, prepared_tensor in zip(
            retained, prepared_header.tensors, strict=True
        ):
            if (
                prepared_tensor.name != source_tensor.name
                or prepared_tensor.dtype != source_tensor.dtype
                or prepared_tensor.shape != source_tensor.shape
                or prepared_tensor.nbytes != source_tensor.nbytes
                or _tensor_sha256(prepared_tensor) != _tensor_sha256(source_tensor)
            ):
                raise ValueError(
                    "Prepared retained tensor differs from source: "
                    f"{source_tensor.name}"
                )
    return affected_relatives, rewritten_relatives


def _manifest_table_relatives(manifest: dict) -> set[Path]:
    raw_tables = manifest.get("tables")
    if not isinstance(raw_tables, list) or not raw_tables:
        raise ValueError("NVMe PLE manifest has no tables")
    relatives = set()
    for table in raw_tables:
        if not isinstance(table, dict):
            raise ValueError("NVMe PLE table entry is not an object")
        relative = preparer._safe_relative_path(table.get("path"))
        if relative.parent != Path("ple"):
            raise ValueError(f"NVMe PLE table must be directly inside ple/: {relative}")
        if relative in relatives:
            raise ValueError(f"duplicate NVMe PLE table path: {relative}")
        relatives.add(relative)
    return relatives


def _verify_source_overlay(
    source: Path,
    prepared: Path,
    affected_relatives: set[Path],
    rewritten_relatives: set[Path],
    table_relatives: set[Path],
) -> None:
    """Bind every ordinary asset as one unit, including future loader metadata.

    The source checkpoint remains immutable by operator precondition while this
    prepared overlay is used; only the preparer's explicit local outputs differ.
    """
    for reserved in (preparer.MANIFEST_NAME, "ple"):
        path = source / reserved
        if path.exists() or path.is_symlink():
            raise ValueError(f"Source checkpoint contains reserved asset: {reserved}")

    affected_names = {relative.name for relative in affected_relatives}
    ordinary = {
        entry.name: entry
        for entry in source.iterdir()
        if entry.name != preparer.INDEX_NAME
        and entry.name not in affected_names
    }
    local_names = {
        preparer.INDEX_NAME,
        preparer.MANIFEST_NAME,
        "ple",
        *(relative.name for relative in rewritten_relatives),
    }
    expected_names = set(ordinary) | local_names
    actual_names = {entry.name for entry in prepared.iterdir()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            "Prepared source overlay asset set differs from deterministic output; "
            f"missing={missing}, unexpected={unexpected}"
        )

    for name, source_path in ordinary.items():
        prepared_path = prepared / name
        try:
            source_target = source_path.resolve(strict=True)
            prepared_target = prepared_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"Prepared source overlay asset is invalid: {name}") from exc
        if not prepared_path.is_symlink() or prepared_target != source_target:
            raise ValueError(
                f"Prepared source overlay asset does not resolve to source: {name}"
            )

    for relative in {
        Path(preparer.INDEX_NAME),
        Path(preparer.MANIFEST_NAME),
        *rewritten_relatives,
    }:
        path = prepared / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Prepared rewritten asset is invalid: {relative}")
    ple_root = prepared / "ple"
    if ple_root.is_symlink() or not ple_root.is_dir():
        raise ValueError("Prepared rewritten asset is invalid: ple")
    expected_tables = {relative.name for relative in table_relatives}
    actual_tables = {entry.name for entry in ple_root.iterdir()}
    if actual_tables != expected_tables:
        missing = sorted(expected_tables - actual_tables)
        unexpected = sorted(actual_tables - expected_tables)
        raise ValueError(
            "Prepared PLE table asset set differs from manifest; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative in table_relatives:
        table = prepared / relative
        if table.is_symlink() or not table.is_file():
            raise ValueError(f"Prepared PLE table asset is invalid: {relative}")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(source: Path, prepared: Path) -> str:
    from sglang_ssd_stream import __version__ as loaded_version
    from sglang_ssd_stream import plugin
    from sglang_ssd_stream.config import load_manifest

    version = importlib.metadata.version("sglang-ssd-stream")
    expected_version = "0.2.0+pennyroyal2"
    if version != expected_version or loaded_version != expected_version:
        raise ValueError(
            "Install Pennyroyal's optional reader "
            f"{expected_version}, found distribution={version}, "
            f"loaded_package={loaded_version}"
        )
    manifest_path = prepared / "ssd-stream.json"
    manifest = json.loads(manifest_path.read_text())
    provenance = manifest.get("source", {})
    if Path(provenance.get("path", "")).resolve() != source.resolve():
        raise ValueError("Prepared PLE artifact belongs to a different TARGET_MODEL")
    for filename, field in (
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "index_sha256"),
    ):
        if sha256(source / filename) != provenance.get(field):
            raise ValueError(
                f"Source checkpoint changed: {filename}; prepare a new artifact"
            )
    load_manifest(manifest_path)
    resolved_source = source.resolve()
    resolved_prepared = prepared.resolve()
    affected, rewritten = _verify_prepared_weights(
        resolved_source, resolved_prepared, manifest
    )
    _verify_source_overlay(
        resolved_source,
        resolved_prepared,
        affected,
        rewritten,
        _manifest_table_relatives(manifest),
    )
    _load_ssd_stream_entrypoint(plugin.register)
    plugin._register_pennyroyal()  # Source-hash and native-extension import checks.
    return sha256(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(check(args.source, args.prepared))
    except Exception as error:  # noqa: BLE001 -- CLI boundary, every error is fatal
        parser.exit(1, f"NVMe PLE preflight failed: {error}\n")
