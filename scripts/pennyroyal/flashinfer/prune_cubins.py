#!/usr/bin/env python3
"""Remove the pinned, proven-unsupported prebuilt cubins from flashinfer-cubin.

The accepted flashinfer-cubin 0.7.0.post1 wheel ships ~6.2 GiB of trtllm-gen
binaries for datacenter Blackwell and Rubin only. The pinned sources prove the
RTX targets (SM86/89/120) never load them:

* ``csrc/trtllm_batched_gemm_runner.cu`` / ``csrc/trtllm_gemm_runner.cu`` /
  ``csrc/trtllm_low_latency_gemm_runner.cu`` -- ``isArchCompatible()`` returns
  true only for smVersion 100, 103 and 107.
* ``include/flashinfer/trtllm/fmha/fmhaRunner.cuh`` -- the constructor accepts
  only ``kSM_100``, ``kSM_103`` and ``kSM_107``; SM120 attention uses the
  separate ``gen_trtllm_fmha_v2_sm120_module`` JIT provider instead.

So this helper deletes exactly the ``.cubin``/``.cubin.lock`` files whose names
carry the ``sm100a``/``sm100f``/``sm103a``/``sm107a`` family tokens inside the
three trtllm-gen payload trees (``fmha/trtllm-gen``, ``batched_gemm-*``,
``gemm-*``) of the pinned commit layout, and drops their lines from the
distribution RECORD so the installed distribution stays consistent. Every
other recorded byte is retained: the deep-gemm kernels and ``checksums.txt``
have no architecture token and no dispatch proof, and no filename is swept by
arch name alone. Those kernels belong to the trtllm-gen backends that SM100,
SM103 and SM107 hosts do dispatch, so no download recovery is claimed to be
impossible for a datacenter GPU; the proof behind this pruning is the other
half -- no retained RTX target can reach them -- and it is scoped to the RTX
packaging path, changing no dispatch behavior and no upstream header or
checksum.

The paths are taken from the official RECORD, deletion never follows a
symlink at any level of a candidate's directory chain, and every recorded
candidate is checked -- chain, file type and size against its RECORD entry
-- before the first unlink: an unexpected version, a tampered, symlinked or
otherwise non-regular file, or an unrecorded candidate fails clearly with
nothing removed. Repeat runs change nothing. ``--check`` is the read-only
form and fails while the unsupported payload is still installed.
``install.py`` runs this step in both installation sequences and in the
container build stage, so a new install is pruned the moment its wheels land
rather than after some first kernel request. SM100/103/107 hosts do dispatch
these upstream kernels, and they install the same whole wheel this tool was
written against; the pruning is scoped to the RTX packaging path, whose
retained targets never reach those dispatchers. No dispatch behavior and no
upstream header or checksum are changed.

Usage:

    python scripts/pennyroyal/flashinfer/prune_cubins.py             # prune
    python scripts/pennyroyal/flashinfer/prune_cubins.py --check     # verify
    python scripts/pennyroyal/flashinfer/prune_cubins.py --site DIR   # site-packages

It needs no GPU and imports nothing from the environment; --site defaults to
the site directory of the interpreter running this script.
"""

import argparse
import json
import os
import re
import sys
from importlib.util import find_spec
from pathlib import Path

PREFIX = "FlashInfer cubin pruning"
PACKAGE = "flashinfer_cubin"
PINNED_VERSION = "0.7.0.post1"

# One file directly under one of the three trtllm-gen payload trees of one
# pinned commit directory of the package (paths as RECORD records them).
TREES = re.compile(
    rf"\A{PACKAGE}/cubins/[0-9a-f]{{40}}/"
    r"(?:fmha/trtllm-gen|batched_gemm-[^/]+|gemm-[^/]+)/([^/]+)\Z"
)
# The family tokens the pinned runners dispatch: fmha kernel names start
# fmhaSm100a|fmhaSm100f|fmhaSm103a|fmhaSm107a..., gemm cubin names end
# _sm100a|_sm100f|_sm103a|_sm107a. Nothing named sm86, sm89 or sm120 matches.
UNSUPPORTED_ARCH = re.compile(
    r"(?:\AfmhaSm10[037][af]?Kernel)|(?:_sm10[037][af]?\.cubin(?:\.lock)?\Z)"
)


def fail(message: str) -> RuntimeError:
    return RuntimeError(f"{PREFIX}: {message}")


def _dist_info(site: Path) -> Path | None:
    """The one flashinfer-cubin distribution of a site directory, at the pin."""
    infos = sorted(site.glob(f"{PACKAGE}-*.dist-info"))
    if not infos:
        return None
    if len(infos) > 1:
        raise fail(f"{site} holds {len(infos)} flashinfer-cubin distributions")
    version = ""
    for line in (infos[0] / "METADATA").read_text().splitlines():
        if line.startswith("Version: "):
            version = line[len("Version: ") :]
    if version != PINNED_VERSION:
        raise fail(
            f"{infos[0]} is flashinfer-cubin {version}, not the pinned "
            f"{PINNED_VERSION}; prune or upgrade it first, then rerun this step"
        )
    record = infos[0] / "RECORD"
    if infos[0].is_symlink() or record.is_symlink():
        raise fail(f"{infos[0]} distribution metadata is a symlink")
    if not record.is_file():
        raise fail(f"{infos[0]} has no RECORD; cannot prune it consistently")
    return infos[0]


def _record_entries(record: Path) -> list[tuple[str, str, int]]:
    """(line, rel path, size) per RECORD line; the line text is kept verbatim."""
    entries = []
    for raw in record.read_text().splitlines():
        parts = raw.split(",")
        size = int(parts[-1]) if len(parts) == 3 and parts[-1] else -1
        rel = ",".join(parts[:-2]) if len(parts) >= 3 else raw
        entries.append((raw, rel, size))
    return entries


def _unsupported(rel: str) -> bool:
    match = TREES.match(rel)
    return (
        match is not None
        and match.group(1).endswith((".cubin", ".cubin.lock"))
        and UNSUPPORTED_ARCH.search(match.group(1)) is not None
    )


def _fail_guard(site: Path, rel: str) -> None:
    """Refuse any symlinked, non-directory or non-regular step of one path.

    Checked on every recorded candidate, whether or not the file itself is
    present, so a symlinked payload directory (which ``os.walk`` skips but a
    later unlink would follow out of the install) fails closed like a
    symlinked leaf, and an interrupted run with the chain already gone does
    not.
    """
    parts = rel.split("/")
    cur = site
    for i, part in enumerate(parts):
        cur = cur / part
        if cur.is_symlink():
            raise fail(f"{cur} is a symlink; refusing to follow it")
        if i != len(parts) - 1:
            if cur.exists() and not cur.is_dir():
                raise fail(f"{cur} is not a directory")
    if cur.exists() and not cur.is_file():
        raise fail(f"{cur} is not a regular file")


def _plan(site: Path, info: Path) -> tuple[list[str], int, int]:
    """Validate every recorded candidate; return (RECORD lines to drop,
    installed candidates, payload bytes).

    A recorded candidate's full path must be free of symlinks, its recorded
    parent directories must be real directories, and an installed candidate
    must be a regular file of exactly the recorded size; every unsupported
    file found by walking the package must also be recorded. All checks run
    before the first deletion, so any failure leaves the install untouched.
    """
    recorded = {}
    for line, rel, size in _record_entries(info / "RECORD"):
        if _unsupported(rel):
            if rel.startswith("/") or ".." in Path(rel).parts:
                raise fail(f"RECORD lists {rel!r}, which escapes {PACKAGE}/")
            recorded[rel] = (line, size)
    present = set()
    for rel in sorted(recorded):
        _fail_guard(site, rel)
        target = site / rel
        if target.exists():
            status = target.lstat()
            if not status.st_mode & 0o170000 == 0o100000:
                raise fail(f"{target} is not a regular file")
            line, size = recorded[rel]
            if size < 0 or status.st_size != size:
                raise fail(
                    f"{target} size does not match its RECORD entry "
                    f"({status.st_size} != {size}); the payload is not the "
                    "pinned one"
                )
            present.add(rel)
    root = site / PACKAGE
    if root.is_dir():
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                rel = Path(dirpath, name).relative_to(site).as_posix()
                if _unsupported(rel) and rel not in recorded:
                    raise fail(
                        f"{site / rel} is an unsupported cubin not recorded "
                        "in RECORD; refusing to remove an unknown file"
                    )
    dropped = [recorded[rel][0] for rel in sorted(recorded)]
    payload = sum(recorded[rel][1] for rel in present)
    return dropped, len(present), payload


def prune(site: Path) -> dict:
    """Delete the unsupported payload of an installed flashinfer-cubin."""
    site = Path(site)
    info = _dist_info(site)
    if info is None:
        return {
            "status": "absent",
            "removed_files": 0,
            "removed_bytes": 0,
            "record_entries_removed": 0,
        }
    dropped, removed, payload = _plan(site, info)
    record = info / "RECORD"
    for _, rel, _size in _record_entries(record):
        if _unsupported(rel):
            (site / rel).unlink(missing_ok=True)
    if dropped:
        keep = set(dropped)
        lines = [
            line
            for line in record.read_text().splitlines()
            if line not in keep
        ]
        temporary = record.with_name(record.name + ".tmp")
        temporary.write_text("\n".join(lines) + "\n")
        os.replace(temporary, record)
    return {
        "status": "pruned",
        "removed_files": removed,
        "removed_bytes": payload,
        "record_entries_removed": len(dropped),
    }


def check(site: Path) -> dict:
    """Read-only: pass only when the unsupported payload is fully gone."""
    site = Path(site)
    info = _dist_info(site)
    if info is None:
        return {"status": "absent"}
    dropped, _removed, payload = _plan(site, info)
    if dropped:
        raise fail(
            f"unsupported trtllm-gen cubin payload is still present: "
            f"{len(dropped)} RECORD entries, {payload} bytes on disk; run "
            f"{__file__} to prune it"
        )
    return {"status": "pruned"}


def default_site() -> Path:
    spec = find_spec(PACKAGE)
    search = list(spec.submodule_search_locations or ()) if spec else []
    if not search:
        raise fail(
            f"{PACKAGE} is not importable here; pass --site its site directory"
        )
    return Path(search[0]).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="verify without changing anything"
    )
    parser.add_argument(
        "--site",
        type=Path,
        metavar="DIR",
        help="site-packages directory to act on (default: this interpreter's)",
    )
    args = parser.parse_args(argv)
    try:
        site = args.site.resolve() if args.site else default_site()
        summary = check(site) if args.check else prune(site)
        print(json.dumps(summary, indent=2))
    except (RuntimeError, OSError) as error:
        print(f"{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
