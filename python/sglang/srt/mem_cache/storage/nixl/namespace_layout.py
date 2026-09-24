"""Fail-closed verification of a derived Pennyroyal NIXL FILE namespace root.

``scripts/pennyroyal/derive_namespace.py`` writes ``namespace-identity.json``
(schema ``sglang-nixl-file-namespace-v2``) into every derived root, hashing
tp_size among the representation fields. Cross-TP sharing of one root is
survivable for non-MLA models because the object name suffix carries
``_<rank>_<size>`` (TP1's ``_0_1`` never aliases TP2's ``_0_2``/``_1_2``),
and MLA-family models restrict storage writes to rank 0 entirely -- but it
is still a misconfiguration: each topology was qualified against its own
root. This check runs at storage construction and refuses the mismatch
(lifting no files) instead of silently reinterpreting what is on disk.

Lives outside ``hicache_nixl.py`` because that module imports the native
``nixl`` package at import time; keeping the manifest check here lets
CPU-only unit tests (and hosts without NIXL installed) exercise it.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

NAMESPACE_MANIFEST_NAME = "namespace-identity.json"


def verify_derived_namespace_layout(storage_dirs: List[str], tp_size: int) -> None:
    """Reject a derived namespace root whose pinned tp_size is not ours."""
    for base in storage_dirs:
        manifest_path = os.path.join(base, NAMESPACE_MANIFEST_NAME)
        if not os.path.exists(manifest_path):
            # Externally managed root (or a legacy pre-manifest directory):
            # nothing pinned to verify, behavior unchanged.
            continue
        try:
            with open(manifest_path, encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"NIXL FILE namespace manifest is unreadable (fail closed; no "
                f"cache data removed): {manifest_path}"
            ) from exc
        fields = manifest.get("identity", {}).get("fields", {})
        pinned = fields.get("tp_size")
        if pinned is None:
            continue
        try:
            pinned_tp_size = int(pinned)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"NIXL FILE namespace manifest has a non-integer tp_size "
                f"(fail closed): {manifest_path}"
            ) from None
        if pinned_tp_size != tp_size:
            raise RuntimeError(
                f"NIXL FILE namespace {base} is pinned to tp_size="
                f"{pinned_tp_size} but this instance runs tp_size={tp_size}; "
                "derive a separate namespace root per TP topology (no cache "
                "data was removed)"
            )
