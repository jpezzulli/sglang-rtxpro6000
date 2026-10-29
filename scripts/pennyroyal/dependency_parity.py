#!/usr/bin/env python3
"""The qualified native dependency baseline, and the parity gate around it.

``docker/pennyroyal/native-baseline.json`` is the single source of truth for what
a release environment must contain: one entry per distribution of the qualified
native installation, at its exact version including any local build tag, with a
classification and the provenance of anything that is not a plain PyPI wheel.
Every version the release image installs, and every version the image check
verifies, comes from this file through this module; nothing keeps a second
version dictionary that can drift.

Classifications:

* ``runtime`` (the default) -- a shared runtime dependency of serving.
* ``build`` -- a packaging or source-build tool the image installs and ships.
* ``native-dev`` -- a development tool of the native machine only. It is named
  here so the image excludes it explicitly rather than by accident, and finding
  one installed is a parity failure, not an extra. Nothing installed may require
  one of these, which is why ``pyelftools`` is not in that class: it looks like a
  wheel-repair tool and the retained Humming wheel depends on it unconditionally.
* ``image-build-addition`` -- a package the image needs and the native baseline
  does not carry, pinned to the exact version the image already ships.
* ``source-build`` -- built from a pinned source tree during the image build
  (NIXL); parity checks the resulting version, and the build may report it as
  still pending until its own step has run.
* ``source-editable`` -- Pennyroyal itself, installed editable from the image's
  checkout. Its version is derived from the checked-out revision, so the stale
  metadata version of the native baseline is recorded as provenance only and is
  never pinned.
* ``unclassified`` -- written by ``refresh`` for a name it has never seen; it
  has to be classified before it can be installed or checked.

Usage:

    requirements [--index URL]       the pins to install, arbitrary equality
    indexes                          the indexes those pins come from
    check [--pending-source-build]   verify an environment against the baseline
    refresh < listing.json [--write] re-derive the pins from a qualified environment

``check`` reads the installed distributions of the interpreter running it, from
its own site directories only, so a plugin installed with ``--target`` or a path
pushed onto ``sys.path`` cannot pose as part of the environment. It fails on
Python or platform drift, on a missing or mismatched shared dependency, on an
extra distribution that the baseline does not name, and on a native-only
development tool that has been installed anyway.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "docker" / "pennyroyal" / "native-baseline.json"
PREFIX = "Dependency parity"

NATIVE_DEV = "native-dev"
SOURCE_BUILD = "source-build"
SOURCE_EDITABLE = "source-editable"
IMAGE_ADDITION = "image-build-addition"
UNCLASSIFIED = "unclassified"
# The FlashInfer cache shim requires all six provider wheels; this GPU family can
# load only SM120f of them, so the image installs the shim and that one provider
# and pip check reports the rest. These five names, exactly, are what the
# dependency check forgives for the shim version of the baseline.
OMITTED_FLASHINFER_PROVIDERS = ("sm100a", "sm103a", "sm80", "sm89", "sm90a")
# The one accepted requirement conflict, spelled the way pip spells it. Both sides
# are literal here so a later Humming move has to be recorded rather than waived
# by a pattern: the sglang metadata asks for 0.1.12, the qualified environment
# carries 0.1.10, and no Penny profile imports it.
ACCEPTED_HUMMING_REQUIREMENT = "humming-kernels[cu13]==0.1.12"
ACCEPTED_HUMMING_INSTALLED = "0.1.10"
# The index an entry with no ``index`` of its own is installed from, and the
# marker for every group at once.
DEFAULT_INDEX = "https://pypi.org/simple"
ALL_INDEXES = "*"
# What the image installs from a package index; source builds and the editable
# checkout have their own steps, and native-only tools get no step at all.
INDEX_CLASSES = ("runtime", "build", IMAGE_ADDITION)
CLASSIFICATIONS = INDEX_CLASSES + (
    NATIVE_DEV,
    SOURCE_BUILD,
    SOURCE_EDITABLE,
    UNCLASSIFIED,
)


class ParityError(RuntimeError):
    pass


def normalized(name: str) -> str:
    """PEP 503: the name as an installed distribution is looked up by."""
    return re.sub(r"[-_.]+", "-", name).lower()


def load(path: Path = MANIFEST_PATH) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != 1 or not isinstance(manifest.get("packages"), dict):
        raise ParityError(f"{path} is not a schema 1 dependency baseline")
    for name, entry in manifest["packages"].items():
        if name != normalized(name):
            raise ParityError(f"{path}: {name} is not a normalized package name")
        if not isinstance(entry.get("version"), str) or not entry.get("version"):
            raise ParityError(f"{path}: {name} has no exact version")
        cls = entry.get("class", "runtime")
        if cls not in CLASSIFICATIONS:
            raise ParityError(f"{path}: {name} has unknown classification {cls!r}")
        if cls == UNCLASSIFIED:
            raise ParityError(
                f"{path}: {name} is unclassified; give it a class before the pins are used"
            )
    return manifest


def requirements(manifest: dict, index: str = ALL_INDEXES) -> list[str]:
    """The pins of the frozen install, optionally narrowed to one index group.

    Entries carry no ``index`` when PyPI is the source, and name one when the
    qualified build came from elsewhere; each group is installed from that index
    alone. Every pin is arbitrary equality (``===``), because ``==0.4.6.post1`` is
    satisfied by ``0.4.6.post1+cu130`` as well: what an index happens to publish
    today is not what makes the release build exact, the pin is.
    """
    pins = [
        f"{name}==={record['version']}"
        for name, record in manifest["packages"].items()
        if record.get("class", "runtime") in INDEX_CLASSES
        and (index == ALL_INDEXES or record.get("index", DEFAULT_INDEX) == index)
    ]
    if index != ALL_INDEXES and not pins:
        raise ParityError(f"no baseline entry installs from {index}")
    return pins


def indexes(manifest: dict) -> list[str]:
    """Every index the baseline installs from, PyPI first, as one image loop.

    The recipe reads its index list here rather than carrying its own, so an entry
    that names a new index cannot be installed by nothing and missed silently.
    """
    found = sorted(
        {
            record.get("index", DEFAULT_INDEX)
            for record in manifest["packages"].values()
            if record.get("class", "runtime") in INDEX_CLASSES
        }
    )
    return [DEFAULT_INDEX] + [index for index in found if index != DEFAULT_INDEX]


def installed_versions(site_dirs: list[str] | None = None) -> dict[str, str]:
    """The distributions installed in these site directories, name -> version."""
    if site_dirs is None:
        paths = sysconfig.get_paths()
        site_dirs = [paths["purelib"], paths["platlib"]]
    found: dict[str, str] = {}
    for directory in site_dirs:
        site = Path(directory)
        if not site.is_dir():
            continue
        for info in sorted(site.glob("*.dist-info")):
            record = metadata.Distribution.at(info)
            name = record.metadata["Name"] or record.name
            found[normalized(name)] = record.version
    return found


def source_revision(revision: str = "") -> str:
    return (
        revision
        or subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def expected_version(manifest: dict, name: str, revision: str) -> str:
    """The version ``name`` must report, resolving the one derived identity."""
    record = manifest["packages"][normalized(name)]
    base = record.get("version_base")
    if not base:
        return record["version"]
    if len(revision) < 12:
        return f"{base}+g<revision>"
    return f"{base}+g{revision[:12]}"


def check(
    manifest: dict,
    actual: dict[str, str],
    revision: str = "",
    pending_source_build: bool = False,
    python: str = "",
    platform: str = "",
) -> dict:
    """Compare one environment against the baseline; problems, never exceptions.

    ``actual`` is the installed name -> version map, by default the one
    ``installed_versions`` reads out of this interpreter's own site directories.
    ``python`` and ``platform`` default to what this interpreter reports.
    """
    problems: list[str] = []
    pending: list[str] = []
    wanted = manifest["packages"]
    for name, record in wanted.items():
        cls = record.get("class", "runtime")
        if cls == NATIVE_DEV:
            if name in actual:
                problems.append(
                    f"{name} {actual[name]} is a native-only development tool; "
                    "the image does not install it"
                )
            continue
        if cls == UNCLASSIFIED:
            problems.append(f"{name} is unclassified in the baseline")
            continue
        want = expected_version(manifest, name, revision)
        if name not in actual:
            if cls == SOURCE_BUILD and pending_source_build:
                pending.append(f"{name}=={want}")
            else:
                problems.append(f"{name}=={want} is missing")
            continue
        if cls == SOURCE_EDITABLE and len(revision) < 12:
            # Its version is only meaningful once the checkout is known.
            continue
        if actual[name] != want:
            problems.append(
                f"{name} is {actual[name]}, the baseline pins {want}"
                + (
                    ""
                    if actual[name].split("+")[0] != want.split("+")[0]
                    else "; the local build tag differs"
                )
            )
    for name, version in sorted(actual.items()):
        if name not in wanted:
            problems.append(f"{name} {version} is not in the baseline")
    additions = sum(
        1 for record in wanted.values() if record.get("class") == IMAGE_ADDITION
    )
    if len(wanted) - additions != manifest["qualified"].get("distributions"):
        problems.append(
            f"the baseline names {len(wanted) - additions} baseline distributions "
            f"and records {manifest['qualified'].get('distributions')}"
        )
    return {
        "python": python or platform_python(),
        "expected_python": manifest["qualified"]["python"],
        "platform": platform or sysconfig.get_platform(),
        "expected_platform": manifest["qualified"]["platform"],
        "pending": pending,
        "problems": problems,
    }


def platform_python() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])


def report(result: dict) -> list[str]:
    """The environment drift, including interpreter drift, as messages."""
    problems = list(result["problems"])
    if result["python"] != result["expected_python"]:
        problems.insert(
            0,
            f"Python {result['python']} is not the qualified "
            f"{result['expected_python']}",
        )
    if result["platform"].replace("-", "_") != result["expected_platform"].replace(
        "-", "_"
    ):
        problems.insert(
            0,
            f"platform {result['platform']} is not the qualified "
            f"{result['expected_platform']}",
        )
    return problems


def accepted_pip_check_line(line: str, manifest: dict) -> bool:
    """Is this one ``pip check`` finding one the accepted baseline explains?

    Only two kinds, each spelled exactly as pip reports it: the cache shim's
    providers for the architectures this GPU family cannot load, and the deferred
    Humming pin. A wrong version, an omitted provider this target does load, a
    missing dependency of anything that is installed, and every unrelated finding
    come back false so the check fails.
    """
    shim = manifest["packages"].get("flashinfer-jit-cache")
    if shim:
        omitted = {
            f"flashinfer-jit-cache {shim['version']} requires "
            f"flashinfer-jit-cache-{arch}, which is not installed."
            for arch in OMITTED_FLASHINFER_PROVIDERS
        }
        if line in omitted:
            return True
    humming = manifest["packages"].get("humming-kernels", {})
    if humming.get("version") == ACCEPTED_HUMMING_INSTALLED:
        # The sglang side reports the version the image derives from its own
        # revision, so only the requirement and the installed Humming are literal.
        return bool(
            re.fullmatch(
                r"sglang \S+ has requirement "
                + re.escape(ACCEPTED_HUMMING_REQUIREMENT)
                + r", but you have humming-kernels "
                + re.escape(ACCEPTED_HUMMING_INSTALLED)
                + r"\.",
                line,
            )
        )
    return False


def accepted_pip_check_output(stdout: str, stderr: str, manifest: dict) -> bool:
    lines = stdout.strip().splitlines()
    if stderr.strip() or not lines:
        return False
    return all(accepted_pip_check_line(line, manifest) for line in lines)


def dumps(manifest: dict) -> str:
    """The baseline written back the way it is committed: one entry per line.

    A refreshed file then diffs by package name rather than reformatting itself,
    which is what makes the review of ``refresh --write`` readable.
    """
    lines = ["{", ' "schema": 1,', ' "qualified": {']
    lines += [
        f"  {json.dumps(key)}: {json.dumps(value)},"
        for key, value in manifest["qualified"].items()
    ]
    lines[-1] = lines[-1][:-1]
    lines += [" },", ' "packages": {']
    lines += [
        f"  {json.dumps(name)}: {json.dumps(record)},"
        for name, record in manifest["packages"].items()
    ]
    lines[-1] = lines[-1][:-1]
    lines += [" }", "}", ""]
    return "\n".join(lines)


def refresh(manifest: dict, listing: str) -> dict:
    """Apply the installed distributions of a qualified environment.

    The input is ``pip list --format=json``, one record per installed distribution
    with the version its own metadata reports: a freeze renders the cubin and
    NIXL wheels as URLs, omits the bootstrap tools, and would have the gate parse
    a version out of a path. Names the baseline has never seen come in
    ``unclassified``, which neither installs nor checks until a maintainer has
    classified them, and the diff of ``--write`` is what a reviewer reads before
    approving a refreshed baseline. The entries a listing cannot speak for - the
    editable checkout and the exact version of an image-only addition - keep what
    is recorded for them.
    """
    try:
        records = json.loads(listing)
    except ValueError as error:
        raise ParityError(
            f"refresh needs `pip list --format=json`, not a freeze ({error})"
        ) from error
    if not isinstance(records, list) or not all(
        isinstance(one, dict) and one.get("name") and one.get("version")
        for one in records
    ):
        raise ParityError("refresh needs `pip list --format=json`, not a freeze")
    packages: dict[str, dict] = {}
    for record in records:
        name = normalized(record["name"])
        before = manifest["packages"].get(name)
        entry = dict(before) if before else {"class": UNCLASSIFIED}
        if entry.get("class") != SOURCE_EDITABLE:
            entry["version"] = record["version"]
        packages[name] = entry
    for name, record in manifest["packages"].items():
        if name not in packages and record.get("class") in (
            SOURCE_EDITABLE,
            IMAGE_ADDITION,
        ):
            packages[name] = record
    missing = sorted(set(manifest["packages"]) - set(packages))
    refreshed = dict(manifest)
    refreshed["packages"] = packages
    refreshed["qualified"] = dict(
        manifest["qualified"],
        distributions=len(packages)
        - sum(1 for one in packages.values() if one.get("class") == IMAGE_ADDITION),
    )
    for name in missing:
        print(f"{PREFIX}: dropped {name}", file=sys.stderr)
    return refreshed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="The native dependency baseline: pins, checks, refresh."
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--source-revision", default="", help="checked-out revision")
    commands = parser.add_subparsers(dest="command", required=True)
    pins = commands.add_parser("requirements", help="the frozen install pins")
    pins.add_argument(
        "--index",
        default=ALL_INDEXES,
        metavar="URL",
        help=f"only the entries of one index, for example {DEFAULT_INDEX}",
    )
    commands.add_parser("indexes", help="the indexes the baseline installs from")
    checked = commands.add_parser("check", help="verify the running environment")
    checked.add_argument(
        "--pending-source-build",
        action="store_true",
        help="the source-built entries of the baseline (NIXL) are not installed yet",
    )
    refreshed = commands.add_parser(
        "refresh", help="apply `pip list --format=json` to the baseline"
    )
    refreshed.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    manifest = load(args.manifest)

    if args.command == "requirements":
        for pin in requirements(manifest, args.index):
            print(pin)
        return 0
    if args.command == "indexes":
        for index in indexes(manifest):
            print(index)
        return 0
    if args.command == "refresh":
        updated = refresh(manifest, sys.stdin.read())
        text = dumps(updated)
        if args.write:
            args.manifest.write_text(text)
            print(f"{PREFIX}: refreshed {args.manifest}", file=sys.stderr)
        else:
            print(text, end="")
        return 0
    result = check(
        manifest,
        installed_versions(),
        source_revision(args.source_revision),
        args.pending_source_build,
    )
    problems = report(result)
    if problems:
        for problem in problems:
            print(f"{PREFIX}: {problem}", file=sys.stderr)
        return 1
    pending = f", pending {', '.join(result['pending'])}" if result["pending"] else ""
    print(
        f"{PREFIX}: passed; {len(manifest['packages'])} baseline distributions on "
        f"Python {result['python']} {result['platform']}{pending}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ParityError as error:
        print(f"{PREFIX}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
