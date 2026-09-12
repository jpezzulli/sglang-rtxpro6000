#!/usr/bin/env python3
"""Prepare a local Qwen4-Exp checkpoint for SSD-backed PLE lookup."""
# Malformed serialized checkpoint values are consistently reported as ValueError.
# ruff: noqa: TRY004

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "ssd-stream.json"
_MAX_HEADER_BYTES = 128 * 1024 * 1024
_COPY_BYTES = 8 * 1024 * 1024
_PLE_SHARD = re.compile(
    r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.ple\.ple_embedding\."
    r"ngram_embedding)\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_WEIGHT_FRAGMENT = ".ple.ple_embedding.ngram_embedding."
_PLE_DTYPES = {
    "BF16": ("bfloat16", 2),
    "F8_E4M3": ("float8_e4m3fn", 1),
}


@dataclass(frozen=True)
class TensorLocation:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclass(frozen=True)
class SafetensorsFile:
    path: Path
    header_size: int
    header_sha256: str
    metadata: object | None
    tensors: tuple[TensorLocation, ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_COPY_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"weight-map path must be a string, got {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"weight-map path must stay inside the checkpoint: {value}")
    return Path(*relative.parts)


def _read_safetensors_header(path: Path) -> SafetensorsFile:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            raw_size = stream.read(8)
            if len(raw_size) != 8:
                raise ValueError(f"truncated safetensors size header in {path}")
            header_size = struct.unpack("<Q", raw_size)[0]
            if header_size == 0 or header_size > _MAX_HEADER_BYTES:
                raise ValueError(
                    f"unreasonable safetensors header size {header_size} in {path}"
                )
            raw_header = stream.read(header_size)
            if len(raw_header) != header_size:
                raise ValueError(f"truncated safetensors JSON header in {path}")
    except OSError as exc:
        raise ValueError(f"cannot read safetensors header from {path}: {exc}") from exc

    try:
        header = json.loads(
            raw_header.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors JSON header in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object in {path}")

    data_size = file_size - 8 - header_size
    if data_size < 0:
        raise ValueError(f"safetensors header exceeds file size in {path}")
    locations: list[TensorLocation] = []
    for name, raw in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"invalid tensor metadata for {name} in {path}")
        dtype = raw.get("dtype")
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if not isinstance(dtype, str):
            raise ValueError(f"invalid dtype for {name} in {path}")
        if not isinstance(shape, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in shape
        ):
            raise ValueError(f"invalid shape for {name} in {path}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in offsets
            )
        ):
            raise ValueError(f"invalid data offsets for {name} in {path}")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise ValueError(f"out-of-range data offsets for {name} in {path}")
        locations.append(
            TensorLocation(
                name=name,
                path=path,
                dtype=dtype,
                shape=tuple(shape),
                offset=8 + header_size + start,
                nbytes=end - start,
            )
        )

    ranges = sorted(
        (location.offset - 8 - header_size, location.nbytes, location.name)
        for location in locations
    )
    expected_start = 0
    for start, nbytes, name in ranges:
        if start != expected_start:
            raise ValueError(f"non-contiguous tensor data before {name} in {path}")
        expected_start += nbytes
    if expected_start != data_size:
        raise ValueError(f"unreferenced tensor data at the end of {path}")

    return SafetensorsFile(
        path=path,
        header_size=header_size,
        header_sha256=hashlib.sha256(raw_size + raw_header).hexdigest(),
        metadata=header.get("__metadata__"),
        tensors=tuple(locations),
    )


def _copy_range(
    source: BinaryIO,
    destination: BinaryIO,
    offset: int,
    length: int,
    digest: Any | None = None,
) -> None:
    source.seek(offset)
    remaining = length
    while remaining:
        chunk = source.read(min(remaining, _COPY_BYTES))
        if not chunk:
            raise ValueError("source safetensors payload ended during a bounded copy")
        destination.write(chunk)
        if digest is not None:
            digest.update(chunk)
        remaining -= len(chunk)


def _encoded_safetensors_header(
    source: SafetensorsFile, retained: tuple[TensorLocation, ...]
) -> bytes:
    header: dict[str, object] = {}
    if source.metadata is not None:
        header["__metadata__"] = source.metadata
    offset = 0
    for tensor in retained:
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.nbytes],
        }
        offset += tensor.nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % 8
    return encoded + b" " * padding


def _rewrite_safetensors(
    destination: Path,
    source: SafetensorsFile,
    retained: tuple[TensorLocation, ...],
) -> None:
    header = _encoded_safetensors_header(source, retained)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.path.open("rb") as input_stream, destination.open("xb") as output:
        output.write(struct.pack("<Q", len(header)))
        output.write(header)
        for tensor in retained:
            _copy_range(input_stream, output, tensor.offset, tensor.nbytes)


def _config_ple_layers(config: dict[str, Any]) -> tuple[dict[int, int], int]:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("config.json has no text_config object")
    raw_layer_ids = text_config.get("ple_layer_ids")
    if not isinstance(raw_layer_ids, list) or not raw_layer_ids:
        raise ValueError("config.json has no text_config.ple_layer_ids")
    if not all(
        isinstance(layer_id, int) and not isinstance(layer_id, bool) and layer_id > 0
        for layer_id in raw_layer_ids
    ):
        raise ValueError("text_config.ple_layer_ids must contain positive integers")
    if len(set(raw_layer_ids)) != len(raw_layer_ids):
        raise ValueError("text_config.ple_layer_ids contains duplicates")
    split_parts = text_config.get("split_ngram_parts")
    if (
        not isinstance(split_parts, int)
        or isinstance(split_parts, bool)
        or split_parts <= 0
    ):
        raise ValueError("text_config.split_ngram_parts must be a positive integer")

    # Qwen4ExpLayerExtensionMixin selects checkpoint layer (ple_layer_id - 1)
    # and derives ple_layer_index from the sorted configured absolute IDs.
    return {
        absolute_id - 1: ple_index
        for ple_index, absolute_id in enumerate(sorted(raw_layer_ids))
    }, split_parts


def _ple_coordinates(name: str) -> tuple[str, int, int] | None:
    match = _PLE_SHARD.fullmatch(name)
    if match is not None:
        return (
            match.group("prefix"),
            int(match.group("layer")),
            int(match.group("shard")),
        )
    if _PLE_WEIGHT_FRAGMENT in name and name.endswith(".weight"):
        raise ValueError(
            f"unsupported PLE weight layout (expected ngram_embedding.shard_N.weight): {name}"
        )
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _prepare(source: Path, output: Path) -> Path:
    source = source.expanduser().resolve(strict=True)
    if not source.is_dir():
        raise ValueError(f"source is not a directory: {source}")
    output = Path(os.path.abspath(output.expanduser()))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent is not a directory: {output.parent}")
    # Parent aliases must not bypass the source-untouched boundary. Keep the
    # final component unresolved so an existing output symlink stays rejected.
    output = output.parent.resolve(strict=True) / output.name
    if source == output or source in output.parents:
        raise ValueError("output must not be inside the source checkpoint")
    if (source / MANIFEST_NAME).exists():
        raise ValueError(f"source is already prepared ({MANIFEST_NAME} exists)")
    if (source / "ple").exists():
        raise ValueError("source already contains the reserved ple artifact directory")

    config_path = source / "config.json"
    index_path = source / INDEX_NAME
    config_sha256 = _sha256_file(config_path)
    index_sha256 = _sha256_file(index_path)
    config = _read_json(config_path)
    index = _read_json(index_path)
    layer_to_ple_index, split_parts = _config_ple_layers(config)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{INDEX_NAME} has no weight_map object")

    ple_indexed_paths: dict[str, Path] = {}
    for name, raw_path in weight_map.items():
        if not isinstance(name, str):
            raise ValueError("weight_map tensor names must be strings")
        relative = _safe_relative_path(raw_path)
        coordinates = _ple_coordinates(name)
        if coordinates is not None and relative.parent != Path("."):
            raise ValueError("PLE safetensors must be at the checkpoint root")
        path = source / relative
        if not path.is_file():
            raise ValueError(f"weight_map references a missing file: {relative}")
        if coordinates is not None:
            ple_indexed_paths[name] = path

    headers: dict[Path, SafetensorsFile] = {}
    for path in sorted(set(ple_indexed_paths.values())):
        headers[path] = _read_safetensors_header(path)

    header_locations: dict[str, TensorLocation] = {}
    tensor_owner: dict[str, Path] = {}
    for path, header in headers.items():
        for tensor in header.tensors:
            if tensor.name in header_locations:
                raise ValueError(
                    f"tensor occurs in multiple safetensors files: {tensor.name}"
                )
            header_locations[tensor.name] = tensor
            tensor_owner[tensor.name] = path

    for name, path in ple_indexed_paths.items():
        if name not in header_locations:
            raise ValueError(
                f"weight_map tensor is missing from its safetensors file: {name}"
            )
        if tensor_owner[name] != path:
            raise ValueError(f"weight_map points {name} at the wrong safetensors file")

    grouped: dict[int, dict[int, TensorLocation]] = {
        ple_index: {} for ple_index in layer_to_ple_index.values()
    }
    prefixes: dict[int, str] = {}
    removed_names: set[str] = set()
    for name, tensor in header_locations.items():
        coordinates = _ple_coordinates(name)
        if coordinates is None:
            continue
        prefix, model_layer, shard_index = coordinates
        if name not in weight_map:
            raise ValueError(f"PLE shard is absent from weight_map: {name}")
        if layer_to_ple_index.get(model_layer) is None:
            expected = sorted(layer_to_ple_index)
            raise ValueError(
                f"PLE shard uses model layer {model_layer}, expected one of {expected}"
            )
        ple_index = layer_to_ple_index[model_layer]
        previous_prefix = prefixes.setdefault(ple_index, prefix)
        if previous_prefix != prefix:
            raise ValueError(f"multiple PLE tensor prefixes map to layer {ple_index}")
        if shard_index in grouped[ple_index]:
            raise ValueError(f"duplicate PLE shard {shard_index} for layer {ple_index}")
        grouped[ple_index][shard_index] = tensor
        removed_names.add(name)

    expected_shards = set(range(split_parts))
    for ple_index, shards in grouped.items():
        actual_shards = set(shards)
        if actual_shards != expected_shards:
            missing = sorted(expected_shards - actual_shards)
            unexpected = sorted(actual_shards - expected_shards)
            raise ValueError(
                f"PLE layer {ple_index} shard set mismatch; missing={missing}, "
                f"unexpected={unexpected}"
            )

    affected_paths = {
        tensor.path for shards in grouped.values() for tensor in shards.values()
    }
    for path in affected_paths:
        if path.parent != source:
            raise ValueError("PLE safetensors must be at the checkpoint root")

    table_specs: list[dict[str, Any]] = []
    removed_bytes = 0
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        for entry in source.iterdir():
            if entry.name in {INDEX_NAME, MANIFEST_NAME, "ple"}:
                continue
            if entry in affected_paths:
                continue
            (temporary / entry.name).symlink_to(
                entry.resolve(), target_is_directory=entry.is_dir()
            )

        ple_root = temporary / "ple"
        ple_root.mkdir()
        for ple_index in sorted(grouped):
            shards = grouped[ple_index]
            ordered = [shards[index] for index in sorted(shards)]
            dtype = ordered[0].dtype
            if dtype not in _PLE_DTYPES:
                raise ValueError(
                    f"unsupported PLE dtype for layer {ple_index}: {dtype}"
                )
            manifest_dtype, element_size = _PLE_DTYPES[dtype]
            if any(tensor.dtype != dtype for tensor in ordered):
                raise ValueError(f"mixed PLE dtypes for layer {ple_index}")
            if any(len(tensor.shape) != 2 for tensor in ordered):
                raise ValueError(f"PLE shards must be rank-2 for layer {ple_index}")
            if len({tensor.shape for tensor in ordered}) != 1:
                raise ValueError(
                    f"PLE shards must have equal shapes for layer {ple_index}"
                )
            columns = ordered[0].shape[1]
            if columns <= 0 or any(tensor.shape[1] != columns for tensor in ordered):
                raise ValueError(f"inconsistent PLE shard widths for layer {ple_index}")
            for tensor in ordered:
                expected_bytes = tensor.shape[0] * columns * element_size
                if tensor.shape[0] <= 0 or tensor.nbytes != expected_bytes:
                    raise ValueError(
                        f"PLE shard {tensor.name} has {tensor.nbytes} bytes; "
                        f"expected {expected_bytes}"
                    )

            table_path = ple_root / f"layer-{ple_index}.bin"
            digest = hashlib.sha256()
            nbytes = 0
            rows = 0
            open_streams: dict[Path, BinaryIO] = {}
            try:
                with table_path.open("xb") as output_stream:
                    for tensor in ordered:
                        input_stream = open_streams.get(tensor.path)
                        if input_stream is None:
                            input_stream = tensor.path.open("rb")
                            open_streams[tensor.path] = input_stream
                        _copy_range(
                            input_stream,
                            output_stream,
                            tensor.offset,
                            tensor.nbytes,
                            digest,
                        )
                        nbytes += tensor.nbytes
                        rows += tensor.shape[0]
            finally:
                for stream in open_streams.values():
                    stream.close()
            removed_bytes += nbytes
            table_specs.append(
                {
                    "layer": ple_index,
                    "path": table_path.relative_to(temporary).as_posix(),
                    "sha256": digest.hexdigest(),
                    "dtype": manifest_dtype,
                    "rows": rows,
                    "columns": columns,
                    "row_start": 0,
                    "bytes": nbytes,
                }
            )

        for path in sorted(affected_paths):
            retained = tuple(
                tensor
                for tensor in headers[path].tensors
                if tensor.name not in removed_names
            )
            if retained:
                _rewrite_safetensors(temporary / path.name, headers[path], retained)

        output_index = dict(index)
        output_index["weight_map"] = {
            name: raw_path
            for name, raw_path in weight_map.items()
            if name not in removed_names
        }
        metadata = output_index.get("metadata")
        if isinstance(metadata, dict) and "total_size" in metadata:
            total_size = metadata["total_size"]
            if not isinstance(total_size, int) or isinstance(total_size, bool):
                raise ValueError("index metadata.total_size must be an integer")
            if total_size < removed_bytes:
                raise ValueError(
                    "index metadata.total_size is smaller than PLE payload"
                )
            output_index["metadata"] = {
                **metadata,
                "total_size": total_size - removed_bytes,
            }
        _write_json(temporary / INDEX_NAME, output_index)

        source_weights = []
        for path in sorted(affected_paths):
            source_weights.append(
                {
                    "path": path.relative_to(source).as_posix(),
                    "bytes": path.stat().st_size,
                    "header_sha256": headers[path].header_sha256,
                }
            )
        manifest = {
            "format": "sglang-ssd-stream",
            "version": 1,
            "preparer": {"format": "pennyroyal-ple-nvme", "version": 1},
            "source": {
                "path": str(source),
                "config_sha256": config_sha256,
                "index_sha256": index_sha256,
                "ple_safetensors": source_weights,
            },
            "tables": table_specs,
        }
        if (
            _sha256_file(config_path) != config_sha256
            or _sha256_file(index_path) != index_sha256
        ):
            raise ValueError("source config or index changed during preparation")
        # The manifest is deliberately last: even the private build directory
        # never advertises a usable artifact before all payloads and index edits exist.
        _write_json(temporary / MANIFEST_NAME, manifest)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output appeared during preparation: {output}")
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an atomic symlink-overlay checkpoint with raw SSD Stream PLE tables"
        )
    )
    parser.add_argument(
        "--source", required=True, type=Path, help="local source checkpoint"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="new prepared checkpoint directory"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prepared = _prepare(args.source, args.output)
    except (OSError, ValueError) as exc:
        print(f"prepare_ple_nvme: {exc}", file=sys.stderr)
        return 1
    print(prepared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
