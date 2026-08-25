#!/usr/bin/env python3
"""Derive a deterministic, representation-specific NIXL FILE cache root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "sglang-nixl-file-namespace-v2"
MANIFEST_NAME = "namespace-identity.json"
_FIELD_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.-]*$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safetensors_header_sha256(path: Path) -> str:
    """Hash tensor metadata without reading multi-gigabyte tensor payloads."""
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"invalid safetensors header in {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        if header_size > 128 * 1024 * 1024:
            raise ValueError(f"unreasonable safetensors header size in {path}")
        header = stream.read(header_size)
        if len(header) != header_size:
            raise ValueError(f"truncated safetensors header in {path}")
    return hashlib.sha256(raw_size + header).hexdigest()


def _checkpoint_content_identity(checkpoint: Path, weight: Path) -> dict[str, str]:
    """Return an authoritative content ID without rehashing large HF payloads.

    ``hf download --local-dir`` records the immutable repository revision and
    LFS SHA-256 next to each downloaded file. For checkpoints without that
    metadata, fall back to hashing the complete weight payload.
    """
    metadata_path = (
        checkpoint
        / ".cache"
        / "huggingface"
        / "download"
        / f"{weight.name}.metadata"
    )
    try:
        lines = metadata_path.read_text().splitlines()
    except OSError:
        lines = []
    if (
        len(lines) >= 2
        and re.fullmatch(r"[0-9a-f]{40}", lines[0])
        and re.fullmatch(r"[0-9a-f]{64}", lines[1])
    ):
        return {
            "source": "huggingface_download_metadata",
            "revision": lines[0],
            "sha256": lines[1],
        }
    return {
        "source": "full_file_sha256",
        "sha256": _sha256_file(weight),
    }


def checkpoint_identity(label: str, raw_path: str) -> dict[str, Any]:
    path = Path(raw_path).resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"{label} checkpoint is not a directory: {path}")

    metadata: dict[str, str] = {}
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "quantize_config.json",
    ):
        candidate = path / name
        if candidate.is_file():
            metadata[name] = _sha256_file(candidate)

    weights = []
    for weight in sorted(path.glob("*.safetensors")):
        stat = weight.stat()
        weights.append(
            {
                "name": weight.name,
                "size": stat.st_size,
                # Deliberately conservative: replacing/touching weights creates a
                # new namespace even when tensor geometry is unchanged.
                "mtime_ns": stat.st_mtime_ns,
                "header_sha256": _safetensors_header_sha256(weight),
                "content": _checkpoint_content_identity(path, weight),
            }
        )
    if not weights:
        raise ValueError(f"{label} checkpoint has no safetensors weights: {path}")

    return {
        "path": str(path),
        "metadata_sha256": metadata,
        "weights": weights,
    }


def git_identity(raw_path: str) -> dict[str, Any]:
    path = Path(raw_path).resolve(strict=True)

    def git(*args: str) -> bytes:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], stderr=subprocess.STDOUT
        )

    head = git("rev-parse", "HEAD").decode().strip()
    diff = git("diff", "--binary", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no").decode().splitlines()
    return {
        "path": str(path),
        "head": head,
        "tracked_worktree_clean": not status,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def parse_fields(values: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"field must be NAME=VALUE: {value!r}")
        name, field_value = value.split("=", 1)
        if not _FIELD_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid field name: {name!r}")
        if name in fields:
            raise ValueError(f"duplicate field: {name}")
        fields[name] = field_value
    return fields


def parse_models(values: list[str]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"model must be NAME=PATH: {value!r}")
        name, model_path = value.split("=", 1)
        if not _FIELD_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid model name: {name!r}")
        if name in models:
            raise ValueError(f"duplicate model: {name}")
        models[name] = checkpoint_identity(name, model_path)
    return models


def canonical_identity(
    fields: dict[str, str],
    models: dict[str, dict[str, Any]],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "fields": fields,
        "models": models,
        "runtime": runtime,
    }


def namespace_name(slug: str, identity: dict[str, Any]) -> tuple[str, str]:
    readable = _SLUG_RE.sub("_", slug.lower()).strip("_")
    if not readable:
        raise ValueError("slug must contain at least one letter or digit")
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    return f"{readable}_{digest[:12]}", digest


def ensure_manifest(root: Path, identity: dict[str, Any], digest: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_NAME
    manifest = {
        "identity_sha256": digest,
        "identity": identity,
    }
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()

    try:
        fd = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        try:
            current = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid namespace manifest: {manifest_path}") from exc
        if current != manifest:
            raise RuntimeError(
                f"namespace manifest does not match derived identity: {manifest_path}"
            )
        return

    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--git-repo", required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-identity", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        identity = canonical_identity(
            parse_fields(args.field),
            parse_models(args.model),
            git_identity(args.git_repo),
        )
        name, digest = namespace_name(args.slug, identity)
        root = Path(args.base_root).resolve() / name
        if not args.dry_run:
            ensure_manifest(root, identity, digest)
        if args.print_identity:
            print(json.dumps({"root": str(root), "identity": identity}, sort_keys=True))
        else:
            print(root)
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"derive_namespace: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
