"""Prepared SSD Stream artifact discovery and runtime configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_RUNTIME_CONFIG_ENV = "SGLANG_SSD_STREAM_CONFIG"
_MANIFEST_NAME = "ssd-stream.json"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DTYPES = {"bfloat16", "float8_e4m3fn"}


@dataclass(frozen=True)
class SSDStreamTable:
    layer: int
    path: Path
    sha256: str
    dtype: str
    rows: int
    columns: int
    row_start: int
    nbytes: int


@dataclass(frozen=True)
class SSDStreamConfig:
    manifest_path: Path
    tables: tuple[SSDStreamTable, ...]

    def table_for_layer(self, layer: int) -> SSDStreamTable:
        matches = [table for table in self.tables if table.layer == layer]
        if len(matches) != 1:
            raise RuntimeError(
                f"SSD Stream manifest must define exactly one table for layer {layer}"
            )
        return matches[0]


def _snapshot_commit(path: Path) -> str | None:
    parts = path.resolve().parts
    for index, part in enumerate(parts[:-1]):
        candidate = parts[index + 1]
        if part == "snapshots" and _COMMIT.fullmatch(candidate):
            return candidate
    return None


def _relative_artifact_path(root: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise TypeError("PLE table path must be a string")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"PLE table path must stay inside the model artifact: {value}")
    return root.joinpath(*relative.parts)


def load_manifest(path: Path) -> SSDStreamConfig:
    # Keep the snapshot path itself: Hugging Face manifests are symlinks into
    # its blob store, while their relative table paths live beside the symlink.
    manifest_path = Path(os.path.abspath(path.expanduser()))
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read SSD Stream manifest {manifest_path}: {exc}"
        ) from exc

    if payload.get("format") != "sglang-ssd-stream" or payload.get("version") != 1:
        raise ValueError(f"unsupported SSD Stream manifest format in {manifest_path}")
    raw_tables = payload.get("tables")
    if not isinstance(raw_tables, list) or not raw_tables:
        raise ValueError(f"SSD Stream manifest has no tables: {manifest_path}")

    root = manifest_path.parent
    tables: list[SSDStreamTable] = []
    layers: set[int] = set()
    for raw in raw_tables:
        if not isinstance(raw, dict):
            raise TypeError("each SSD Stream table entry must be an object")
        try:
            layer = int(raw["layer"])
            digest = raw["sha256"]
            dtype = raw["dtype"]
            rows = int(raw["rows"])
            columns = int(raw["columns"])
            row_start = int(raw.get("row_start", 0))
            nbytes = int(raw["bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid SSD Stream table entry: {raw}") from exc
        table_path = _relative_artifact_path(root, raw.get("path"))
        if layer < 0 or layer in layers:
            raise ValueError(f"invalid or duplicate SSD Stream layer: {layer}")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"invalid SHA-256 for SSD Stream layer {layer}")
        if dtype not in _DTYPES:
            raise ValueError(f"unsupported SSD Stream dtype for layer {layer}: {dtype}")
        if rows <= 0 or columns <= 0 or row_start < 0:
            raise ValueError(f"invalid SSD Stream shape for layer {layer}")
        element_size = 1 if dtype == "float8_e4m3fn" else 2
        expected_bytes = rows * columns * element_size
        if nbytes != expected_bytes:
            raise ValueError(
                f"SSD Stream layer {layer} declares {nbytes} bytes, expected "
                f"{expected_bytes} from its shape and dtype"
            )
        try:
            actual_bytes = table_path.stat().st_size
        except OSError as exc:
            raise ValueError(
                f"cannot access SSD Stream table {table_path}: {exc}"
            ) from exc
        if actual_bytes != nbytes:
            raise ValueError(
                f"SSD Stream table {table_path} has {actual_bytes} bytes, expected {nbytes}"
            )
        tables.append(
            SSDStreamTable(
                layer=layer,
                path=table_path.resolve(),
                sha256=digest,
                dtype=dtype,
                rows=rows,
                columns=columns,
                row_start=row_start,
                nbytes=nbytes,
            )
        )
        layers.add(layer)
    return SSDStreamConfig(manifest_path=manifest_path, tables=tuple(tables))


def _resolve_artifact(model_path: str, revision: str | None) -> tuple[Path, str | None]:
    local_path = Path(model_path).expanduser()
    if local_path.exists():
        root = local_path.resolve()
        manifest_path = root / _MANIFEST_NAME if root.is_dir() else root
        return manifest_path, None

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=model_path,
            revision=revision,
            allow_patterns=(_MANIFEST_NAME, "ple/*"),
        )
    )
    commit = _snapshot_commit(snapshot)
    if commit is None:
        raise RuntimeError(f"Hugging Face returned an unversioned snapshot: {snapshot}")
    return snapshot / _MANIFEST_NAME, commit


def configure_cli(*args, **kwargs):
    """Resolve the immutable prepared artifact and enable SSD streaming."""
    namespace = next(
        (value for value in reversed(args) if isinstance(value, argparse.Namespace)),
        None,
    )
    if namespace is None:
        raise RuntimeError("SGLang did not provide parsed server arguments")

    manifest_path, commit = _resolve_artifact(namespace.model_path, namespace.revision)
    config = load_manifest(manifest_path)
    verify_tables(config)
    if commit is not None:
        namespace.revision = commit
    namespace.ple_offload_embedding = False
    # The NVMe PLE reader is qualified only at the fixed W4 NEXTN window
    # (speculative_num_steps 3, speculative_num_draft_tokens 4): graph.py
    # sizes each replay's staged PLE rows to bs * captured_req_width off the
    # decode runner, and the community report against this adapter shows the
    # wider adaptive graphs crashing in prepare_ssd_stream_graph_replay ->
    # ticket.wait_for_launch before the row budget is reached. Rather than
    # silently fall back to RAM or trust an unqualified wider window, fail
    # loudly on any wider fixed window here. RAM PLE is the initial W8 path;
    # genuine W8+NVMe needs a re-qualified adapter/graph interaction.
    if namespace.speculative_algorithm is not None:
        # Adaptive spec decoding is not supported or in scope for this
        # adapter: speculative_hook resolves NEXTN to EAGLE and initializes
        # the adaptive params, whose default batch-1 candidate_steps [1,3,7]
        # (adaptive_spec_params.DEFAULT_ADAPTIVE_CONFIG) reach the wider
        # widths the now-multi-group QSA ring can serve -- exactly the
        # unqualified NVMe regime the community crash names. Refuse it
        # explicitly instead of parsing custom step policies here.
        if getattr(namespace, "speculative_adaptive", False):
            raise ValueError(
                "SSD Stream NVMe PLE does not support adaptive speculative "
                "decoding; the candidate-step policy can widen the verify "
                "window beyond the qualified fixed W4. Run NVMe with "
                "--speculative-adaptive off at the fixed W4 window, or use "
                "RAM PLE for the wider fixed windows."
            )
        # This is a BEFORE hook on ServerArgs.from_cli_args, so the raw
        # --speculative-num-draft-tokens is not yet the effective width:
        # arg_groups/speculative_hook.py later normalizes it to
        # speculative_num_steps + 1 whenever eagle_topk <= 1 (and auto-fills
        # it), so steps=7 with the width omitted or mismatched at 4 still
        # runs an 8-wide window. Refuse every reading of the arguments that
        # could resolve wider than the qualified W4 window; an all-unset
        # width keeps the historical accept (auto params were never
        # re-checked by this guard and the recipes pin explicit values).
        raw_tokens = getattr(namespace, "speculative_num_draft_tokens", None)
        raw_steps = getattr(namespace, "speculative_num_steps", None)
        raw_topk = getattr(namespace, "speculative_eagle_topk", None)
        effective_width = 0
        if raw_tokens is not None:
            effective_width = max(effective_width, int(raw_tokens))
        if raw_steps is not None and (raw_topk is None or int(raw_topk) <= 1):
            effective_width = max(effective_width, int(raw_steps) + 1)
        if effective_width > 4:
            raise ValueError(
                "SSD Stream NVMe PLE is qualified only at the fixed W4 NEXTN "
                f"window (effective draft-token width <= 4); got "
                f"{effective_width} from speculative_num_draft_tokens="
                f"{raw_tokens}, speculative_num_steps={raw_steps}, "
                f"speculative_eagle_topk={raw_topk}. Use RAM PLE for W8, or "
                "re-qualify the NVMe adapter's graph replay for the wider window."
            )
    if namespace.cpu_offload_gb > 0:
        raise ValueError(
            "per-parameter CPU offload is not safe for this model; use grouped offload"
        )
    cpu_offload = namespace.offload_group_size > 0
    if cpu_offload:
        decode_graph_disabled = (
            namespace.disable_cuda_graph
            or namespace.disable_decode_cuda_graph
            or namespace.cuda_graph_backend_decode == "disabled"
        )
        prefill_graph_disabled = (
            namespace.disable_cuda_graph
            or namespace.disable_prefill_cuda_graph
            or namespace.cuda_graph_backend_prefill == "disabled"
        )
        if not decode_graph_disabled or not prefill_graph_disabled:
            raise ValueError(
                "SSD Stream CPU offload requires disabled prefill and decode CUDA graphs"
            )
        if namespace.cuda_graph_backend_decode not in (None, "disabled"):
            raise ValueError("SSD Stream CPU offload cannot enable decode CUDA graphs")
        if namespace.cuda_graph_backend_prefill not in (None, "disabled"):
            raise ValueError("SSD Stream CPU offload cannot enable prefill CUDA graphs")
        if namespace.speculative_algorithm is not None:
            raise ValueError("SSD Stream CPU offload does not support MTP")
        if namespace.startup_weight_load_mode == "overlap":
            raise ValueError("SSD Stream CPU offload does not support overlap loading")
    os.environ[_RUNTIME_CONFIG_ENV] = json.dumps(
        {
            "manifest_path": str(config.manifest_path),
            "cpu_offload": cpu_offload,
        },
        separators=(",", ":"),
    )
    return args, kwargs


def verify_tables(config: SSDStreamConfig) -> None:
    """Validate bytes before workers can treat the manifest as model identity.

    The upstream manifest reader validates shape and digest syntax only. An
    external table is no longer covered by the safetensors index, so verify
    its payload once during server CLI setup, not on every gather.
    """
    for table in config.tables:
        digest = hashlib.sha256()
        with table.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != table.sha256:
            raise ValueError(
                f"SSD Stream table SHA-256 mismatch for layer {table.layer}"
            )


def get_config() -> SSDStreamConfig:
    try:
        values = json.loads(os.environ[_RUNTIME_CONFIG_ENV])
        manifest_path = Path(values["manifest_path"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid internal SSD Stream runtime configuration") from exc
    return load_manifest(manifest_path)


def grouped_cpu_offload_enabled() -> bool:
    try:
        values = json.loads(os.environ[_RUNTIME_CONFIG_ENV])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid internal SSD Stream runtime configuration") from exc
    return values.get("cpu_offload") is True
