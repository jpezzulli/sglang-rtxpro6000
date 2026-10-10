#!/usr/bin/env python3
"""Saved Pennyroyal settings: parse, validate, resolve, and display them.

This module is the single shared source of truth for the non-interactive
launch path (``run-penny``, ``configure-penny --check``) and the interactive
front end. It is CPU-only on purpose: nothing here imports the model stack,
CUDA, or torch, so validation works on a machine with no GPU.

The saved file is a plain user-owned .env-style file: ``KEY=value`` lines plus
``#`` comments. It is never shell-sourced or eval'd; the reader unquotes
verbatim, so spaces, ``$``, and quotes survive a save/load round trip, and the
same quoting also survives the manual ``docker compose`` .env file.

Precedence (stated the same way in --help and the examples):
  1. an explicit ``--config PATH`` wins over default discovery;
  2. a key saved in the config file wins over the inherited environment;
  3. a recipe-specific key that is not saved keeps the value it inherited from
     the environment, or the recipe's own default when nothing is set here.
This utility never rewrites recipe defaults; unset means "the recipe decides".

An explicit save (``./configure-penny``) never replaces anything: it writes a
NEW timestamped settings file beside its input and, for ``--container``, a new
timestamped folder of ordinary launch files, then prints the command that uses
exactly those new outputs. Previous files stay untouched for the operator to
rename, move or delete themselves; there is no pointer, symlink or registry.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# Profiles, recipes, and the keys a saved configuration may hold
# --------------------------------------------------------------------------

PROFILE_KEY = "PENNYROYAL_PROFILE"
MODES = ("native", "container")
PROFILES = ("next", "next-plain", "27b")
RECIPE_BY_PROFILE = {
    "next": "serve-flash-next-frspec.sh",
    "next-plain": "serve-flash-next.sh",
    "27b": "serve-qwen38-27b-dflash2.sh",
}
PROFILE_LABEL = {
    "next": "Flash-Next NVFP4 + FR-Spec/native NEXTN",
    "next-plain": "Flash-Next NVFP4 + native NEXTN without FR-Spec",
    "27b": "Qwen3.8-27B FP8 + DFlash2",
}
# The --hicache-size literals the three serve recipes pass when the operator
# chose nothing. --hicache-size counts decimal GB (server_args.hicache_size is
# an INT and the host pool sizes itself at host_size * 1e9 bytes), not GiB.
PROFILE_HICACHE_SIZE_GB = {"next": "32", "next-plain": "32", "27b": "96"}
CONTAINER_MODELS_TARGET = "/models"
DEFAULT_IMAGE = "ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.3"
COMPOSE_RELPATH = "docker/pennyroyal/compose.yaml"
# The accepted manual container path: an ordinary host run.sh plus a mounted,
# user-editable directory of SGLang startup scripts and NIXL TOMLs. The beta
# configurator generates copies of these files when the operator saves; the
# generated launch itself needs no host Python, no source checkout, no Compose,
# no .env and no configurator at container boot.
LAUNCH_RELPATH = "docker/pennyroyal/launch"
STARTUP_BY_PROFILE = {
    "next": "start-flash-next-frspec.sh",
    "next-plain": "start-flash-next.sh",
    "27b": "start-27b-dflash2.sh",
}
NIXL_TOML_BY_PROFILE = {
    "next": "nixl-posix-frspec.toml",
    "next-plain": "nixl-posix.toml",
    "27b": "nixl-posix.toml",
}
# Runtime settings the mounted startup scripts deliberately leave to the
# container environment, so the generated run.sh forwards them with -e. This is
# exactly the list the manual docker/pennyroyal/compose.yaml environment block
# forwards beyond the knobs the startup file states itself: dropping one here
# would silently lose a setting the compose path already honours.
#
# FLASHINFER_GDN_FP16_ACCUM_MMA is the accepted private GDN override rather
# than a KeySpec: the wizard never asks for it, and the 27b recipe does not name
# it at all (Triton GDN), so a saved value there is unused, never refused.
CONTAINER_PASSTHROUGH_KEYS = (
    "SGLANG_HICACHE_NIXL_MAX_CACHE_GB",
    "SGLANG_HICACHE_TORCH_PINNED_ALLOC",
    "SGLANG_SM120_ONLINE_MXFP8",
    "SGLANG_FORWARD_UNKNOWN_TOOLS",
    "PENNY_PLE_NVME_MODEL",
    "MAX_TOTAL_TOKENS",
    "PENNY_REASONING_EFFORT",
    "NCCL_P2P_DISABLE",
    "FLASHINFER_GDN_FP16_ACCUM_MMA",
)
# Settings that only mean something while the disk tier is mounted, so a
# generated launch with the tier off does not carry them at all.
DISK_ONLY_KEYS = ("SGLANG_HICACHE_NIXL_MAX_CACHE_GB",)
# Knobs the 27b recipe fixes in its own launch line (None: it has no such flag)
# rather than reading from the environment, so a saved choice there cannot
# reach the server. Values copied from configs/pennyroyal/
# serve-qwen38-27b-dflash2.sh; the Next profiles read all of these from the
# environment and are not listed.
PROFILE_PINNED_KEYS = {
    "27b": {
        "MAX_RUNNING_REQUESTS": "4",
        "MAX_MAMBA_CACHE_SIZE": "24",
        "MAX_TOTAL_TOKENS": None,
        "TP_SIZE": "1",
        "PENNY_PLE_BACKEND": "ram",
    },
}
# Asked about for the Next profiles, meaningless for 27b, and harmless when an
# older file still carries it: hidden from the wizard, never refused.
PROFILE_HIDDEN_KEYS = {"27b": ("PENNY_PLE_NVME_MODEL",)}
NATIVE_CONFIG_RELPATH = ".config/pennyroyal/pennyroyal.env"
CONTAINER_CONFIG_RELPATH = ".config/pennyroyal/pennyroyal-container.env"
DEFAULT_CONTAINER_LAUNCH_DIR = "~/pennyroyal-container"
RECIPE_DIR_RELPATH = "configs/pennyroyal"
DEFAULT_NATIVE_CACHE_BASE = "~/.cache/pennyroyal"
DEFAULT_NATIVE_NIXL_BASE = "~/.local/share/pennyroyal/nixl"
TRUE_VALUES = ("1", "true", "yes", "y", "on")
FALSE_VALUES = ("0", "false", "no", "n", "off")
JSON_BEGIN = "=== PENNYROYAL_CONFIG_JSON ==="
JSON_END = "=== PENNYROYAL_CONFIG_END ==="

# Best-effort Next/27B sanity signal using the metadata the qualified
# checkpoints actually ship (architectures plus top-level and text_config
# model_type). next-plain is a Next profile. Unknown architectures stay valid
# so a supported custom quant checkpoint is never rejected by name; the
# recipes keep ownership of weight-format and tokenizer checks, and we never
# read all the weights here.
MODEL_METADATA = {
    "next": ("Qwen4ExpForConditionalGeneration", "qwen4_exp", "qwen4_exp_text"),
    "27b": ("Qwen3_5ForConditionalGeneration", "qwen3_5", "qwen3_5_text"),
}
# Profile families: next and next-plain share the Next checkpoint family.
PROFILE_FAMILY = {"next": "next", "next-plain": "next", "27b": "27b"}

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INT_RE = re.compile(r"^[0-9]+$")
_POSITIVE_INT_RE = re.compile(r"^[1-9][0-9]*$")
_NUMBER_RE = re.compile(r"^(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9._:/=-]+$")
_MM_DEVICE_RE = re.compile(r"^(?:cpu|cuda:[0-9]+)$")
_SPECIAL_COMMENT_RE = re.compile(r"(?:^|\s)#")


class ConfigError(Exception):
    """A saved or generated setting is unusable."""


@dataclass(frozen=True)
class KeySpec:
    name: str
    description: str
    kind: str = "str"
    choices: tuple[str, ...] = ()
    default: str = ""
    advanced: bool = False
    prompt: str = ""


SHARED_KEYS: tuple[KeySpec, ...] = (
    KeySpec(
        "TARGET_MODEL",
        "target checkpoint directory",
        "path",
        prompt="Downloaded target model directory",
    ),
    KeySpec(
        "DRAFT_MODEL",
        "draft checkpoint directory (27b profile)",
        "path",
        prompt="Downloaded draft model directory (27b profile only)",
    ),
    KeySpec(
        "SGLANG_HICACHE_NIXL_MAX_CACHE_GB",
        "soft byte budget for the NIXL FILE cache dirs, in GiB "
        "(0 = no cap: the cache stays on and keeps growing)",
        "nonneg-number",
        default="0",
        prompt="Disk budget for the persistent NIXL cache, in GiB "
        "(0 = unlimited budget, not an off switch)",
    ),
    KeySpec(
        "PENNY_HICACHE_SIZE_GB",
        "host-RAM HiCache tier in decimal GB (blank = the recipe's "
        "qualified default: 32 for the Next profiles, 96 for 27b)",
        "positive-int",
        prompt="RAM (HiCache) cache size in GB (1 GB = 1e9 bytes, not GiB; "
        "blank = the recipe's default for this profile)",
    ),
    KeySpec(
        "SGLANG_HICACHE_TORCH_PINNED_ALLOC",
        "pinned host memory for HiCache transfers under WSL2 (off keeps "
        "the default host-register allocator)",
        "bool",
        default="false",
        advanced=True,
        prompt="WSL2 host-memory workaround (true or false)",
    ),
    KeySpec(
        "NIXL",
        "persistent NIXL disk cache tier (on is the qualified default; off "
        "keeps the GPU radix cache and the host-RAM HiCache tier and deletes "
        "nothing)",
        "choice",
        ("on", "off"),
        default="on",
        prompt="Disk cache tier via NIXL (on or off)",
    ),
    KeySpec(
        "TP_SIZE",
        "tensor-parallel ranks (blank = the recipe's qualified " "TP1)",
        "positive-int",
        advanced=True,
        prompt="Tensor-parallel ranks (blank = recipe default)",
    ),
    KeySpec(
        "PENNY_PLE_BACKEND",
        "PLE table placement",
        "choice",
        ("ram", "nvme"),
        default="ram",
        advanced=True,
        prompt="PLE placement (ram or nvme)",
    ),
    KeySpec(
        "PENNY_PLE_NVME_MODEL",
        "prepared NVMe PLE snapshot directory",
        "path",
        advanced=True,
        prompt="Prepared NVMe PLE snapshot directory",
    ),
    KeySpec(
        "SGLANG_SM120_ONLINE_MXFP8",
        "online FP8 kernel path (blank = automatic: eligible Flash-Next SM120 "
        "launches use the accepted rowwise-FP8 paths on their own; a saved "
        "true/false is the private compatibility escape hatch)",
        "bool",
        advanced=True,
    ),
    KeySpec(
        "SGLANG_MM_PREPROCESS_DEVICE",
        "media preprocessing device",
        "mm-device",
        default="cpu",
        prompt="Media preprocessing device (cpu or cuda:N)",
    ),
    KeySpec(
        "SGLANG_FORWARD_UNKNOWN_TOOLS",
        "forward unknown tool names",
        "bool",
        default="true",
        advanced=True,
        prompt="Forward unknown tool names (true or false)",
    ),
    KeySpec(
        "MAX_RUNNING_REQUESTS",
        "admitted requests (empty = recipe default)",
        "positive-int",
        advanced=True,
        prompt="Max running requests (empty = recipe default)",
    ),
    KeySpec(
        "MAX_MAMBA_CACHE_SIZE",
        "recurrent-state slots (empty = recipe default)",
        "positive-int",
        advanced=True,
        prompt="Max mamba cache size (empty = recipe default)",
    ),
    KeySpec(
        "MAX_TOTAL_TOKENS",
        "shared KV token cap (empty = recipe default)",
        "positive-int",
        advanced=True,
        prompt="Max total tokens (empty = recipe default)",
    ),
)

NATIVE_KEYS: tuple[KeySpec, ...] = (
    KeySpec(
        "REPO_ROOT",
        "Pennyroyal repository directory",
        "path",
        prompt="Where your Pennyroyal folder is (the one holding run-penny)",
    ),
    KeySpec(
        "VENV_PATH",
        "virtualenv that contains bin/sglang",
        "path",
        prompt="Your Python environment folder (the one with bin/sglang)",
    ),
    KeySpec(
        "SGLANG_EXE",
        "sglang executable override",
        "path",
        advanced=True,
        prompt="Expert: the exact sglang program to run",
    ),
    KeySpec(
        "PYTHON",
        "python interpreter override",
        "path",
        advanced=True,
        prompt="Expert: the exact python program to run",
    ),
    KeySpec(
        "CACHE_BASE",
        "compiler and runtime cache root",
        "path",
        prompt="Folder for the compiled and runtime caches",
    ),
    KeySpec(
        "NIXL_STORAGE_BASE",
        "persistent NIXL FILE cache root",
        "path",
        prompt="Folder for the persistent NIXL cache on disk",
    ),
    KeySpec(
        "PENNY_BUILD_JOBS",
        "first-start compile jobs (empty = 4)",
        "positive-int",
        advanced=True,
        prompt="Build jobs for first-start compilation (empty = recipe default)",
    ),
    KeySpec(
        "NIXL_PREFIX",
        "NIXL install prefix outside the linker path",
        "path",
        advanced=True,
        prompt="Expert: where NIXL is installed, when that prefix is "
        "outside the normal linker path",
    ),
    KeySpec(
        "GPU",
        "GPU index or UUID used for the model",
        "gpu",
        default="0",
        prompt="GPU index or UUID",
    ),
)

CONTAINER_KEYS: tuple[KeySpec, ...] = (
    KeySpec(
        "PENNYROYAL_IMAGE",
        "container image reference",
        "token",
        default=DEFAULT_IMAGE,
        advanced=True,
        prompt="Expert: which container image to start",
    ),
    KeySpec(
        "LAUNCH_DIR",
        "folder holding the generated run.sh and its config/ " "startup files",
        "path",
        default=DEFAULT_CONTAINER_LAUNCH_DIR,
        prompt="Folder for your container run.sh and startup files",
    ),
    KeySpec(
        "HOST_MODELS_ROOT",
        "host directory mounted read-only at /models",
        "path",
        prompt="Host folder holding your downloaded models "
        "(mounted read-only at /models)",
    ),
    KeySpec(
        "HOST_CACHE_BASE",
        "writable host directory mounted at /cache",
        "path",
        prompt="Host folder for the compiled and runtime caches " "(mounted at /cache)",
    ),
    KeySpec(
        "HOST_NIXL_STORAGE_BASE",
        "writable host directory at /nixl",
        "path",
        prompt="Host folder for the persistent NIXL cache on disk "
        "(mounted at /nixl; unused when the disk tier is off)",
    ),
    KeySpec(
        "USER_ID",
        "numeric UID that owns the writable directories",
        "int",
        default="1000",
        advanced=True,
        prompt="Runtime UID (numeric owner of the writable directories)",
    ),
    KeySpec(
        "GROUP_ID",
        "numeric GID that owns the writable directories",
        "int",
        default="1000",
        advanced=True,
        prompt="Runtime GID (numeric owner of the writable directories)",
    ),
    KeySpec(
        "NVIDIA_GPU",
        "GPU index or UUID the container reserves",
        "gpu",
        default="0",
        prompt="GPU index or UUID",
    ),
    KeySpec(
        "PENNYROYAL_PORT",
        "host API port",
        "port",
        default="8001",
        prompt="Host API port",
    ),
)

# Keys whose names exist only for the entry point: the recipe or the generated
# launch file gets the equivalent value instead.
SHELL_ONLY_KEYS = ("COMPOSE_FILE", "GPU", "VENV_PATH", "LAUNCH_DIR")
KEY_SPECS = {spec.name: spec for spec in SHARED_KEYS}


def specs_for(mode: str) -> dict[str, KeySpec]:
    """Managed keys for a mode: shared keys plus the mode's own keys."""
    if mode not in MODES:
        raise ConfigError(f"unknown mode: {mode}")
    specs = dict(KEY_SPECS)
    for spec in (NATIVE_KEYS if mode == "native" else CONTAINER_KEYS):
        specs[spec.name] = spec
    return specs


# --------------------------------------------------------------------------
# .env-style parsing and serialization (no eval, no shell, no interpolation)
# --------------------------------------------------------------------------


def parse_env_text(text: str, source: str = "<config>") -> dict[str, str]:
    """Parse KEY=value lines verbatim; quoted values stay literal."""
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ConfigError(f"{source}:{number}: expected KEY=value: {raw!r}")
        if not _KEY_RE.match(key):
            raise ConfigError(f"{source}:{number}: invalid key: {key!r}")
        if key in values:
            raise ConfigError(f"{source}:{number}: duplicate key: {key}")
        values[key] = _unquote(value.strip(), f"{source}:{number}")
    return values


def _unquote(value: str, location: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        body = value[1:-1]
        if value[0] == "'":
            # Compose keeps single-quoted values literal; we match it exactly.
            return body
        return (
            body.replace('\\"', '"')
            .replace("\\$", "$")
            .replace("$$", "$")
            .replace("\\\\", "\\")
        )
    comment = _SPECIAL_COMMENT_RE.search(value)
    if comment:
        value = value[: comment.start()]
    return value.strip()


def quote_value(value: str) -> str:
    """Quote for the saved file so Docker Compose and we read the same value.

    Single quotes are literal in both readers. When the value contains an
    apostrophe we use double quotes with the escapes Compose itself honours:
    \\" for a quote, \\ and $$ for a literal backslash and dollar sign, so no
    interpolation happens in either reader. Plain values stay unquoted.
    """
    if value == "":
        return ""
    if re.search(r"[ \t#\"'`$\\|;&<>()]", value):
        if "'" not in value:
            return f"'{value}'"
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "$$")
        return f'"{escaped}"'
    return value


def quote_command_arg(value: str) -> str:
    """Quote for the commands we print. Display only; we never eval them."""
    return shlex.quote(value)


def serialize_env(
    sections: list[tuple[str, list[tuple[str, str]]]], header: tuple[str, ...] = ()
) -> str:
    lines: list[str] = list(header)
    for title, entries in sections:
        if not entries:
            continue
        if title:
            lines.extend(["", f"# {title}"])
        for key, value in entries:
            lines.append(f"{key}={quote_value(value)}")
    return "\n".join(lines) + "\n"


def read_env_file(path: Path) -> dict[str, str]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    return parse_env_text(text, str(path))


def save_stamp() -> str:
    """Local timestamp (sortable, filename-safe) naming one save's outputs."""
    return datetime.now().strftime("%Y%m%dT%H%M%S")


# What one generated save suffix looks like at the end of a name: the
# timestamp and the optional collision counter timestamped_new_path appends.
# Rerunning against the file/folder just saved (the wizard feeds its own
# output and the saved LAUNCH_DIR straight back in) strips this tail first, so
# a name carries ONE generated timestamp on its stable basename instead of
# stacking suffixes toward ENAMETOOLONG. No registry decides this.
_GENERATED_TAIL_RE = re.compile(r"-\d{8}T\d{6}(?:-\d+)?$")


def timestamped_new_path(path: Path, stamp: str, directory: bool = False) -> Path:
    """An unused path beside `path`, named after `stamp`, for a save's output.

    ``pennyroyal.env`` becomes ``pennyroyal-20261006T072130.env`` next to it;
    a launch folder becomes ``...-20261006T072130``. A name that already ends
    in one generated suffix (the previous save's output or its saved
    LAUNCH_DIR) is reduced to its stable basename first, so re-saving onto
    its own output never stacks timestamps. Repeated saves inside one second
    (or an operator-made file of the same name) step to ``-2``, ``-3``, ... so
    every save owns its own fresh output and nothing previous is ever reused.
    Nothing is created here; create_new_files has the final word, and a name
    that appears in the meantime is refused, never replaced.
    """
    path = Path(path)
    if directory:
        stem, suffix = _GENERATED_TAIL_RE.sub("", path.name), ""
    else:
        stem = _GENERATED_TAIL_RE.sub("", path.stem)
        suffix = path.suffix or ".env"
    base = f"{stem}-{stamp}"
    candidate = path.parent / f"{base}{suffix}"
    number = 1
    while candidate.exists():
        number += 1
        if number > 999:
            raise ConfigError(f"cannot find a free new name beside {path}")
        candidate = path.parent / f"{base}-{number}{suffix}"
    return candidate


def create_new_files(
    files: list[tuple[Path, str]], private: tuple[Path, ...] = ()
) -> list[Path]:
    """Create every (path, content) as a brand-new file, or leave nothing behind.

    A save touches only its own fresh paths: a destination that already exists
    is a refusal, not an overwrite, so a previous file is never even opened for
    writing. Each file is created with O_EXCL, written and fsynced; a create or
    write failure (a full disk, a vanished folder) deletes exactly the files -
    and the folders - this call made, then reports the reason. There is no
    staging, journal or backup: the only rollback is deleting what this call
    itself created. A path named in `private` is created 0600, a new .sh
    becomes 0755, anything else 0644.
    """
    paths = [Path(path) for path, _ in files]
    if len(set(paths)) != len(paths):
        raise ConfigError("this save names the same file twice")
    created: list[Path] = []
    folders: list[Path] = []
    try:
        for path, content in ((Path(path), text) for path, text in files):
            missing: list[Path] = []
            probe = path.parent
            while not probe.is_dir():
                missing.append(probe)
                probe = probe.parent
            for folder in reversed(missing):
                try:
                    folder.mkdir()
                except FileExistsError:
                    pass  # someone else made it; it is not ours to delete
                except OSError as exc:
                    # A regular file where a folder belongs, or a denied
                    # parent, reads like every other create failure here.
                    raise ConfigError(
                        f"cannot create {folder}: {exc.strerror or exc}"
                    ) from exc
                else:
                    folders.append(folder)
            if path.exists():
                raise ConfigError(f"refusing to replace {path}")
            mode = (
                0o600 if path in private else (0o755 if path.suffix == ".sh" else 0o644)
            )
            try:
                handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except OSError as exc:
                raise ConfigError(
                    f"cannot create {path}: {exc.strerror or exc}"
                ) from exc
            created.append(path)
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                path.chmod(mode)
            except OSError as exc:
                raise ConfigError(
                    f"cannot write {path}: {exc.strerror or exc}"
                ) from exc
    except BaseException:
        for done in created:
            done.unlink(missing_ok=True)
        for folder in reversed(folders):
            try:
                folder.rmdir()  # empty again unless the world changed meanwhile
            except OSError:
                pass
        raise
    return paths


# --------------------------------------------------------------------------
# Per-key validation
# --------------------------------------------------------------------------


def normalize_bool(value: str) -> str:
    lowered = value.lower()
    if lowered in TRUE_VALUES:
        return "true"
    if lowered in FALSE_VALUES:
        return "false"
    return value


def validate_value(spec: KeySpec, value: str, source: str) -> str:
    """Return the normalized value, or raise. Empty always means "not set"."""
    if value == "":
        return ""
    kind = spec.kind
    if kind == "bool":
        if value.lower() not in TRUE_VALUES + FALSE_VALUES:
            raise ConfigError(
                f"{source}: {spec.name} must be true or false, got {value!r}"
            )
        return normalize_bool(value.lower())
    if kind == "choice":
        if value not in spec.choices:
            raise ConfigError(
                f"{source}: {spec.name} must be one of "
                f"{', '.join(spec.choices)}, got {value!r}"
            )
        return value
    if kind == "mm-device":
        if not _MM_DEVICE_RE.match(value):
            raise ConfigError(
                f"{source}: {spec.name} must be cpu or cuda:N, " f"got {value!r}"
            )
        return value
    if kind == "int":
        if not _INT_RE.match(value):
            raise ConfigError(
                f"{source}: {spec.name} must be a non-negative "
                f"integer, got {value!r}"
            )
        return value
    if kind == "positive-int":
        if not _POSITIVE_INT_RE.match(value):
            raise ConfigError(
                f"{source}: {spec.name} must be a positive " f"integer, got {value!r}"
            )
        return value
    if kind == "nonneg-number":
        if not _NUMBER_RE.match(value) or float(value) < 0:
            raise ConfigError(
                f"{source}: {spec.name} must be a number >= 0, " f"got {value!r}"
            )
        return value
    if kind == "port":
        if not _POSITIVE_INT_RE.match(value) or not 1 <= int(value) <= 65535:
            raise ConfigError(
                f"{source}: {spec.name} must be a port between 1 "
                f"and 65535, got {value!r}"
            )
        return value
    if kind in ("token", "gpu"):
        if kind == "gpu" and _INT_RE.match(value):
            return value
        if not _TOKEN_RE.match(value):
            raise ConfigError(
                f"{source}: {spec.name} may not contain spaces, "
                f"quotes, or '$', got {value!r}"
            )
        return value
    if "\0" in value:
        raise ConfigError(f"{source}: {spec.name} contains a NUL byte")
    return value


def validate_profile(profile: str, mode: str = "native") -> str:
    """Canonical profile name; the mode is selected by the entry point."""
    name = profile.strip().removeprefix("container:")
    if name not in PROFILES:
        raise ConfigError(
            f"unknown profile {profile!r} for {mode} mode; choose "
            f"from {', '.join(PROFILES)}"
        )
    return name


def recipe_for(profile: str) -> str:
    return RECIPE_BY_PROFILE[profile]


# --------------------------------------------------------------------------
# Saved configuration + resolution
# --------------------------------------------------------------------------


@dataclass
class Issue:
    level: str  # "error" or "warn"
    message: str
    key: str = ""  # managed key this issue can be fixed by, when known


@dataclass
class Config:
    mode: str
    profile: str
    values: dict[str, str] = field(default_factory=dict)
    unknown: dict[str, str] = field(default_factory=dict)
    path: Optional[Path] = None
    source: str = "built-in defaults"
    # Profile as written in the file, so an override can be surfaced honestly.
    file_profile: str = ""
    # Where the profile came from and the explicit CLI profile, so the printed
    # next command is the same plan that was validated, from any directory.
    profile_origin: str = ""
    cli_profile: str = ""


def default_config_path(mode: str, repo_root: Path, home: Path | None = None) -> Path:
    """Native: ~/.config/pennyroyal/pennyroyal.env.

    Container: the configurator's own saved file beside the native one. The
    generated launch files carry the resolved settings themselves, so nothing
    has to read this file when the container starts.
    """
    if mode == "native":
        return (home or Path.home()) / NATIVE_CONFIG_RELPATH
    return (home or Path.home()) / CONTAINER_CONFIG_RELPATH


def discover_config_path(
    mode: str, environ: dict[str, str], repo_root: Path, explicit: Optional[Path] = None
) -> tuple[Path, str]:
    """--config wins, then PENNYROYAL_CONFIG, then the per-mode default."""
    if explicit is not None:
        return Path(explicit), "--config"
    from_env = environ.get("PENNYROYAL_CONFIG", "").strip()
    if from_env:
        return Path(from_env), "PENNYROYAL_CONFIG"
    return default_config_path(mode, repo_root), "default location"


def config_from_text(
    mode: str, text: str, path: Optional[Path] = None, source: str = "saved file"
) -> Config:
    """Validate one KEY=value body into a Config, the way a saved file is read.

    load_config defers here for the bytes on disk. The beta configurator calls
    this with the exact text it is about to write, so the settings it validates,
    the launch files it generates from them and the file a later launch reloads
    are one resolution of the same text: a knob the body omits is omitted in the
    generated launch too, never rendered from a half-answered wizard value.
    """
    specs = specs_for(mode)
    values: dict[str, str] = {}
    unknown: dict[str, str] = {}
    saved = parse_env_text(text, source)
    profile = saved.pop(PROFILE_KEY, "")
    file_profile = profile
    for name, value in saved.items():
        spec = specs.get(name)
        if spec is None:
            unknown[name] = value
            continue
        values[name] = validate_value(spec, value, source)
    return Config(
        mode=mode,
        profile=validate_profile(profile or default_profile(mode), mode),
        values=values,
        unknown=unknown,
        path=path,
        source=source,
        file_profile=file_profile,
        profile_origin="saved file" if file_profile else "",
    )


def load_config(
    mode: str, path: Optional[Path], environ: dict[str, str], repo_root: Path
) -> Config:
    """Read one saved file; a missing file falls back to defaults, no error."""
    used: Optional[Path] = None
    if path is None:
        config = config_from_text(mode, "", None, "built-in defaults")
    else:
        # Resolve once, at load: relative --config and PENNYROYAL_CONFIG values
        # stay meaningful from any directory, and the paths we print and hand to
        # the launch are the file that was actually read.
        used = Path(path).expanduser().absolute()
        if used.exists() and not used.is_file():
            raise ConfigError(f"config path is not a file: {used}")
        if used.is_file():
            config = config_from_text(mode, used.read_text(), used, str(used))
        else:
            config = config_from_text(
                mode, "", used, f"built-in defaults ({used} does not exist yet)"
            )
    if not config.file_profile:
        ambient = environ.get(PROFILE_KEY, "").strip()
        if ambient:
            config.profile = validate_profile(ambient, mode)
            config.profile_origin = f"{PROFILE_KEY} environment"
            config.source = f"{config.source} + {PROFILE_KEY} environment"
    return config


def default_profile(mode: str) -> str:
    return "next"


@dataclass
class Resolved:
    value: str
    origin: str  # "saved file", "inherited environment", or "utility default"


def resolve(config: Config, environ: dict[str, str], name: str) -> Resolved:
    """Documented precedence for one key.

    A key saved empty is a decision, not an absence: it always wins over the
    inherited environment. For a key the utility documents a default for, the
    blank resets to that default, because a plain saved blank can reach code
    that reads the variable directly (the runtime's EnvStr-style scalars do
    not fall back on an empty value the way the shell's ${VAR:-default} does);
    for a key with no utility default (the recipe-owned capacity knobs), the
    blank is exported as the empty string so the recipe's own ${VAR:-...}
    fallback fires. Only a key the file does not mention at all keeps the
    inherited value. config.values keeps the raw blank either way.
    """
    if name in config.values:
        saved = config.values[name]
        if saved != "":
            return Resolved(saved, "saved file")
        blank_spec = specs_for(config.mode).get(name)
        if blank_spec is not None and blank_spec.default:
            return Resolved(
                blank_spec.default,
                f"saved blank: reset to documented default " f"{blank_spec.default!r}",
            )
        return Resolved("", "saved file (blank: inherited value suppressed)")
    if name == "GPU":
        # The native recipes take CUDA_VISIBLE_DEVICES; accept either spelling
        # so an existing shell export keeps working without saving it twice.
        for alias in ("GPU", "NVIDIA_GPU", "CUDA_VISIBLE_DEVICES"):
            ambient = environ.get(alias, "").strip()
            if ambient:
                return Resolved(ambient, "inherited environment")
    else:
        ambient = environ.get(name, "").strip()
        if ambient:
            return Resolved(ambient, "inherited environment")
    spec = specs_for(config.mode).get(name)
    if spec is not None and spec.default:
        return Resolved(spec.default, "utility default")
    return Resolved("", "not set")


@dataclass
class Plan:
    config: Config
    env: dict[str, str]
    argv: list[str]
    origins: dict[str, str] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)
    next_command: str = ""
    repo_root: Optional[Path] = None
    # For the container: resolved values that must override the ambient shell.
    forced_env: dict[str, str] = field(default_factory=dict)
    # Where the generated container launch files live (container mode only).
    launch_dir: Optional[Path] = None

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "warn"]


def _expand(value: str, repo_root: Path) -> str:
    if not value:
        return value
    expanded = os.path.expanduser(value)
    if not os.path.isabs(expanded):
        expanded = str(Path(repo_root) / expanded)
    return os.path.normpath(expanded)


def describe_path_state(
    path: Path, writable: bool = False, readable: bool = False
) -> str:
    """Classify a path so 'not created yet' is distinguishable from invalid."""
    path = Path(path)
    if not str(path):
        return "not-set"
    if not path.exists():
        # 'missing-creatable' means mkdir -p by this user would fix it; plain
        # 'missing' means something is in the way (or the root is not ours).
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if path.is_absolute() and parent.is_dir() and os.access(parent, os.W_OK):
            return "missing-creatable"
        return "missing"
    if not path.is_dir():
        return "not-a-directory"
    if writable and not os.access(path, os.W_OK):
        return "not-writable"
    if readable and not os.access(path, os.R_OK):
        return "not-readable"
    return "ok"


def _check_root(
    plan: Plan, name: str, path: str, writable: bool, runtime_identity: str = ""
) -> str:
    """Validate a configured directory and name the concrete next step.

    A writable cache must accept writes from the identity that will actually
    run the workload: the configured container USER_ID/GROUP_ID, or this
    account when launching natively. os.access() only speaks for the current
    user, so when that identity differs we report the exact assumption instead
    of falsely approving or rejecting it. We never chown, sudo, mkdir, or
    delete anything here; ownership changes are the operator's own action.
    """
    state = describe_path_state(Path(path), writable=writable, readable=not writable)
    if state == "missing-creatable":
        plan.issues.append(
            Issue(
                "warn",
                f"{name} does not exist yet; create it "
                f"with: mkdir -p {quote_command_arg(path)} "
                f"(as the identity that must write it)",
                key=name,
            )
        )
    elif state == "not-writable":
        owner = _safe_owner(path)
        plan.issues.append(
            Issue(
                "error",
                f"{name} is not writable by this account: {path} (owned by {owner}). "
                "The runtime identity that writes it must own it; we never chown or "
                "sudo, so adjust ownership yourself and rerun --check.",
                key=name,
            )
        )
    elif state == "not-readable":
        plan.issues.append(
            Issue(
                "error", f"{name} is not readable by this " f"account: {path}", key=name
            )
        )
    elif state != "ok":
        plan.issues.append(
            Issue("error", f"{name} is unusable ({state}): {path}", key=name)
        )
    elif state == "ok" and runtime_identity and not _running_as(runtime_identity):
        plan.issues.append(
            Issue(
                "warn",
                f"{name} ({path}) is writable by this account, but the configured "
                f"container runtime identity is {runtime_identity} "
                f"(directory owner {_safe_owner(path)}). This check cannot "
                f"verify that identity's access; {name} must be owned or "
                f"group-writable by it. No chown or sudo is performed here.",
                key=name,
            )
        )
    return state


def _safe_owner(path: str) -> str:
    try:
        return path_owner(path)
    except OSError:
        return "unknown"


def _running_as(identity: str) -> bool:
    """True when the current process already is that numeric identity."""
    uid, _, gid = identity.partition(":")
    return (
        uid.isdigit()
        and int(uid) == os.getuid()
        and gid.isdigit()
        and int(gid) in os.getgroups()
    )


def _runtime_identity(plan: Plan) -> str:
    if plan.config.mode != "container":
        return ""
    return f"{plan.env.get('USER_ID', '1000')}:" f"{plan.env.get('GROUP_ID', '1000')}"


def path_owner(path: str) -> str:
    import grp
    import pwd

    info = os.stat(path)
    try:
        user = pwd.getpwuid(info.st_uid).pw_name
    except KeyError:
        user = str(info.st_uid)
    try:
        group = grp.getgrgid(info.st_gid).gr_name
    except KeyError:
        group = str(info.st_gid)
    return f"{user}:{group}"


def inspect_architecture(path: Path) -> tuple[tuple[str, ...], Optional[str]]:
    """Read identity metadata from config.json only; never hash or load weights.

    Collects architectures, model_type, and text_config.model_type — the three
    fields the qualified checkpoints actually carry.
    """
    config = Path(path) / "config.json"
    if not config.is_file():
        return (), "config.json is missing"
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return (), f"config.json could not be parsed ({exc})"
    found: list[str] = []
    architectures = payload.get("architectures")
    if isinstance(architectures, list):
        found.extend(item for item in architectures if isinstance(item, str))
    for key in ("model_type",):
        value = payload.get(key)
        if isinstance(value, str):
            found.append(value)
    text_config = payload.get("text_config")
    if isinstance(text_config, dict):
        value = text_config.get("model_type")
        if isinstance(value, str):
            found.append(value)
    return tuple(found), None


def architecture_issue(profile: str, role: str, path: Path) -> Optional[Issue]:
    names, note = inspect_architecture(path)
    if note:
        return Issue("warn", f"{role} {path}: {note}", key=_role_key(role))
    if not names:
        return None
    recognised = {
        candidate
        for candidate, known in MODEL_METADATA.items()
        for name in names
        if name in known
    }
    if not recognised:
        return None  # Unknown but supported custom checkpoint: the recipe decides.
    family = PROFILE_FAMILY[profile]
    if recognised == {family}:
        return None
    if len(recognised) == 1:
        found = next(iter(recognised))
        return Issue(
            "error",
            f"{role} {path} is a '{found}' checkpoint "
            f"({', '.join(names)}) but the selected profile is "
            f"'{profile}'; this is a clear mixup, not a custom quant. "
            "Fix the profile or the path.",
            key=_role_key(role),
        )
    return Issue(
        "warn",
        f"{role} {path}: metadata {', '.join(names)} is "
        f"ambiguous for the '{profile}' profile.",
    )


def _role_key(role: str) -> str:
    return "TARGET_MODEL" if role == "target" else "DRAFT_MODEL"


def _adopt(
    plan: Plan,
    config: Config,
    environ: dict[str, str],
    name: str,
    export_as: Optional[str] = None,
    expand: Optional[Path] = None,
) -> str:
    resolved = resolve(config, environ, name)
    plan.origins[name] = resolved.origin
    value = resolved.value
    if expand is not None:
        value = _expand(value, expand)
    # Explicitly saved values export even when empty; unset keys export nothing
    # and keep the environment the recipe already inherited. Blank therefore
    # suppresses an ambient value without ever unsetting keys we do not own.
    saved_explicit = value != "" or name in config.values
    if saved_explicit and (export_as is not None or name not in SHELL_ONLY_KEYS):
        plan.env[export_as or name] = value
    return value


def build_plan(
    mode: str, config: Config, environ: dict[str, str], repo_root: Optional[Path] = None
) -> Plan:
    if mode != config.mode:
        raise ConfigError(f"config mode mismatch: {mode} != {config.mode}")
    if mode == "native":
        return _plan_native(config, environ, repo_root)
    return _plan_container(config, environ, repo_root)


def _profile_line(config: Config) -> str:
    line = f"profile: {config.profile} ({PROFILE_LABEL[config.profile]})"
    if config.file_profile and config.file_profile != config.profile:
        line += (
            f" [overridden for this run; {config.path} saves " f"{config.file_profile}]"
        )
    elif config.profile_origin in (f"{PROFILE_KEY} environment", "utility default"):
        line += f" [from {config.profile_origin}]"
    return line


def _plan_native(
    config: Config, environ: dict[str, str], repo_root: Optional[Path]
) -> Plan:
    plan = Plan(config=config, env={}, argv=[])
    root_value = resolve(config, environ, "REPO_ROOT")
    plan.origins["REPO_ROOT"] = root_value.origin
    root = Path(_expand(root_value.value or str(discover_repo_root()), Path.cwd()))
    plan.repo_root = root
    recipe = root / RECIPE_DIR_RELPATH / recipe_for(config.profile)

    venv = _adopt(plan, config, environ, "VENV_PATH", expand=root)
    sglang_exe = resolve(config, environ, "SGLANG_EXE").value
    if not sglang_exe:
        sglang_exe = str(Path(venv or root / ".venv") / "bin" / "sglang")
    sglang_exe = _expand(sglang_exe, root)
    python_exe = resolve(config, environ, "PYTHON").value
    python_exe = _expand(python_exe or str(Path(sglang_exe).parent / "python"), root)
    plan.env["REPO_ROOT"] = str(root)
    plan.env["SGLANG_EXE"] = sglang_exe
    plan.env["PYTHON"] = python_exe

    cache_base = _adopt(plan, config, environ, "CACHE_BASE", expand=root)
    cache_base = cache_base or _expand(DEFAULT_NATIVE_CACHE_BASE, root)
    plan.env["CACHE_BASE"] = cache_base
    nixl_base = _adopt(plan, config, environ, "NIXL_STORAGE_BASE", expand=root)
    nixl_base = nixl_base or _expand(DEFAULT_NATIVE_NIXL_BASE, root)
    plan.env["NIXL_STORAGE_BASE"] = nixl_base

    target = _expand(_adopt(plan, config, environ, "TARGET_MODEL", expand=root), root)
    draft = _expand(_adopt(plan, config, environ, "DRAFT_MODEL", expand=root), root)
    for name in (
        "PENNY_PLE_BACKEND",
        "PENNY_PLE_NVME_MODEL",
        "SGLANG_SM120_ONLINE_MXFP8",
        "SGLANG_MM_PREPROCESS_DEVICE",
        "SGLANG_FORWARD_UNKNOWN_TOOLS",
        "MAX_RUNNING_REQUESTS",
        "MAX_MAMBA_CACHE_SIZE",
        "MAX_TOTAL_TOKENS",
        "SGLANG_HICACHE_NIXL_MAX_CACHE_GB",
        "PENNY_HICACHE_SIZE_GB",
        "SGLANG_HICACHE_TORCH_PINNED_ALLOC",
        "PENNY_BUILD_JOBS",
        "NIXL_PREFIX",
        "NIXL",
        "TP_SIZE",
    ):
        _adopt(plan, config, environ, name)
    gpu = _adopt(plan, config, environ, "GPU") or "0"
    plan.env["CUDA_VISIBLE_DEVICES"] = gpu
    for name, value in config.unknown.items():
        plan.env[name] = value

    if not root.is_dir():
        plan.issues.append(
            Issue("error", f"REPO_ROOT is not a directory: {root}", key="REPO_ROOT")
        )
    for label, path in (("sglang", sglang_exe), ("python", python_exe)):
        if not Path(path).is_file() or not os.access(path, os.X_OK):
            issuesource = plan.origins.get("VENV_PATH", "default")
            plan.issues.append(
                Issue(
                    "error",
                    f"{label} is not an executable " f"({issuesource}): {path}",
                    key="VENV_PATH",
                )
            )
    if not recipe.is_file():
        plan.issues.append(Issue("error", f"recipe is missing: {recipe}"))
    if not target:
        plan.issues.append(
            Issue(
                "error",
                "TARGET_MODEL is not set; add it to the "
                "config or run ./configure-penny",
                key="TARGET_MODEL",
            )
        )
    elif not Path(target).is_dir():
        plan.issues.append(
            Issue(
                "error",
                f"target model directory is missing: " f"{target}",
                key="TARGET_MODEL",
            )
        )
    else:
        issue = architecture_issue(config.profile, "target", Path(target))
        if issue:
            plan.issues.append(issue)
    if config.profile == "27b":
        if not draft:
            plan.issues.append(
                Issue(
                    "error",
                    "profile 27b needs DRAFT_MODEL " "(the DFlash2 checkpoint)",
                    key="DRAFT_MODEL",
                )
            )
        elif not Path(draft).is_dir():
            plan.issues.append(
                Issue(
                    "error",
                    f"draft model directory is missing: " f"{draft}",
                    key="DRAFT_MODEL",
                )
            )
        else:
            issue = architecture_issue(config.profile, "draft", Path(draft))
            if issue:
                plan.issues.append(issue)
    # Use the resolved values, not raw sources: a saved blank suppression must
    # count as empty here too, exactly as it does for the recipe environment.
    if plan.env.get("PENNY_PLE_BACKEND", "").strip() == "nvme":
        if not plan.env.get("PENNY_PLE_NVME_MODEL", "").strip():
            plan.issues.append(
                Issue(
                    "error",
                    "PENNY_PLE_BACKEND=nvme needs " "PENNY_PLE_NVME_MODEL",
                    key="PENNY_PLE_NVME_MODEL",
                )
            )
    _check_root(
        plan,
        "CACHE_BASE",
        cache_base,
        writable=True,
        runtime_identity=_runtime_identity(plan),
    )
    disk_tier = plan.env.get("NIXL", "on")
    _profile_pinned_issues(plan)
    # With the disk tier off the recipe needs no NIXL root at all, so a missing
    # or unwritable one must not hold the launch up (and nothing is deleted).
    if disk_tier == "on":
        _check_root(
            plan,
            "NIXL_STORAGE_BASE",
            nixl_base,
            writable=True,
            runtime_identity=_runtime_identity(plan),
        )

    plan.argv = [str(recipe)]
    plan.summary = [
        "mode: native",
        _profile_line(config),
        f"config: {config.source}",
        f"recipe: {recipe}",
        f"runtime: {sglang_exe}",
        f"target: {target or 'not set'} ({plan.origins.get('TARGET_MODEL', 'not set')})",
        f"draft: {draft or 'not used by this profile'}",
        f"cache root: {cache_base}",
        _disk_tier_line(disk_tier, nixl_base),
        f"GPU: {gpu}",
        _hicache_size_line(plan.env.get("PENNY_HICACHE_SIZE_GB", ""), config.profile),
    ]
    # A budget says nothing about a tier that is switched off.
    if disk_tier == "on":
        plan.summary.append(
            _nixl_budget_line(plan.env.get("SGLANG_HICACHE_NIXL_MAX_CACHE_GB", "0"))
        )
    launcher = root / "run-penny"
    # The printed command has to reload exactly this plan from anywhere, so it
    # names the resolved file that was actually read (wizard or CLI, explicit
    # or discovered) plus any profile override the caller asked for.
    args = [str(launcher)]
    if config.path is not None:
        args += ["--config", str(config.path)]
    if config.cli_profile:
        args += ["--profile", config.cli_profile]
    plan.next_command = " ".join(quote_command_arg(arg) for arg in args)
    return plan


def _disk_tier_line(nixl: str, root: str) -> str:
    """Say what the disk tier does for this launch, without implying a budget.

    Turning the tier off is not the same as a 0 (unlimited) budget: the GPU
    radix cache, the host-RAM HiCache tier, the model and the speculation
    settings all stay exactly as they are, and no cached data is removed.
    """
    if nixl == "off":
        return (
            "NIXL disk tier: off; no NIXL root, config or namespace is "
            "needed and nothing is deleted (GPU radix cache and host-RAM "
            "HiCache stay on)"
        )
    return f"NIXL disk tier: on ({root})"


def _profile_pinned_issues(plan: Plan) -> None:
    """Name a saved setting the selected profile's recipe cannot honour.

    The 27b recipe fixes its own capacity, TP and PLE placement in the launch
    line, so an operator choice there would be accepted, forwarded and then
    silently dropped by the server: the recipe's literal is what runs. Saying so
    is the fail-loud rule this project already uses for unsupported paths, and
    it never invents tuning the qualified profile does not have. A blank (or an
    unset key) keeps meaning 'the recipe decides' and stays silent.
    """
    pinned = PROFILE_PINNED_KEYS.get(plan.config.profile, {})
    for name, value in sorted(pinned.items()):
        chosen = (plan.env.get(name) or "").strip()
        if not chosen:
            continue
        if value is None:
            plan.issues.append(
                Issue(
                    "error",
                    f"{name}={chosen} is not a setting of the "
                    f"{plan.config.profile} profile: its recipe has no "
                    f"{name.replace('_', ' ').lower()} knob at all; leave it unset "
                    "for this profile",
                    key=name,
                )
            )
            continue
        if chosen != value:
            plan.issues.append(
                Issue(
                    "error",
                    f"{name}={chosen} is not a setting of the "
                    f"{plan.config.profile} profile: that recipe is qualified and "
                    f"launches with {name}={value}; leave it unset (or use "
                    f"{value}) rather than run on a value the recipe ignores",
                    key=name,
                )
            )


def profile_unavailable(profile: str, name: str) -> bool:
    """True when the wizard should not offer a knob this profile does not have.

    Broader than the pinned values alone: the prepared NVMe snapshot folder is
    part of the PLE placement the 27b recipe does not have either, but a value
    left in an older file is unused rather than wrong, so it is not refused.
    """
    if name in PROFILE_PINNED_KEYS.get(profile, {}):
        return True
    return name in PROFILE_HIDDEN_KEYS.get(profile, ())


def _hicache_size_line(chosen: str, profile: str) -> str:
    """Say which --hicache-size the recipe will pass, and in which unit.

    The unit is decimal GB (server_args.hicache_size is an INT consumed as
    host_size * 1e9 bytes), never GiB, and it is separate from the PLE
    embedding table, which lives in its own host-RAM/NVMe placement.
    """
    default = PROFILE_HICACHE_SIZE_GB[profile]
    if chosen:
        return (
            f"RAM cache (HiCache): {chosen} GB as --hicache-size "
            f"(1 GB = 1e9 bytes, not GiB; the recipe default "
            f"{default} GB is not used)"
        )
    return (
        f"RAM cache (HiCache): {default} GB as --hicache-size "
        f"(1 GB = 1e9 bytes, not GiB; the recipe default for this "
        f"profile; unset keeps it)"
    )


def _nixl_budget_line(budget: str) -> str:
    """Name the NIXL disk budget honestly: 0 means no cap, not cache off.

    Every numeric spelling of zero (0, 0.0, 0.00, .0, 00, 0.) is the same
    unlimited budget to the runtime, so the summary must not call any of them
    a cap. Anything that is not a number at all (a hand-edited value that
    failed validation elsewhere) never crashes the summary; it is shown as a
    cap and named by the plan's own error reporting.
    """
    try:
        unlimited = budget == "" or float(budget) == 0
    except ValueError:
        unlimited = False
    if unlimited:
        return (
            "NIXL disk budget: 0 GiB = no cap; the persistent NIXL cache "
            "stays enabled and keeps growing"
        )
    return (
        f"NIXL disk budget: {budget} GiB cap on the persistent NIXL "
        f"cache dirs (the cache stays enabled)"
    )


def _plan_container(
    config: Config, environ: dict[str, str], repo_root: Optional[Path]
) -> Plan:
    """Plan the ordinary prebuilt-container launch: run.sh plus mounted files.

    The settings resolve here; the launch files are generated from the shipped
    examples only when the operator saves in ./configure-penny, so a hand edit
    is never rewritten at launch time and the container itself needs no Python,
    no checkout, no Compose and no .env to start.
    """
    plan = Plan(config=config, env={}, argv=[])
    root = Path(_expand(str(repo_root or discover_repo_root()), Path.cwd()))
    plan.repo_root = root
    templates = root / LAUNCH_RELPATH
    saved_file = (
        Path(config.path) if config.path else default_config_path("container", root)
    )

    launch_dir = Path(
        _expand(
            _adopt(plan, config, environ, "LAUNCH_DIR")
            or _expand(DEFAULT_CONTAINER_LAUNCH_DIR, root),
            root,
        )
    )
    plan.launch_dir = launch_dir

    for name in (
        "HOST_MODELS_ROOT",
        "HOST_CACHE_BASE",
        "HOST_NIXL_STORAGE_BASE",
        "NIXL",
        "PENNY_PLE_BACKEND",
        "PENNY_PLE_NVME_MODEL",
        "SGLANG_SM120_ONLINE_MXFP8",
        "SGLANG_MM_PREPROCESS_DEVICE",
        "SGLANG_FORWARD_UNKNOWN_TOOLS",
        "MAX_RUNNING_REQUESTS",
        "MAX_MAMBA_CACHE_SIZE",
        "MAX_TOTAL_TOKENS",
        "SGLANG_HICACHE_NIXL_MAX_CACHE_GB",
        "PENNY_HICACHE_SIZE_GB",
        "SGLANG_HICACHE_TORCH_PINNED_ALLOC",
        "USER_ID",
        "GROUP_ID",
        "TP_SIZE",
    ):
        _adopt(plan, config, environ, name)
    # Settings the generated launch forwards but no KeySpec owns: the saved
    # file wins over the caller's shell, exactly like every other setting, and
    # an inherited value still reaches the running container, as Compose's
    # ${VAR:-} forwarding would.
    for name in CONTAINER_PASSTHROUGH_KEYS:
        if name in config.unknown:
            plan.env[name] = config.unknown[name]
            plan.origins[name] = "saved file"
        else:
            _adopt(plan, config, environ, name)
    # The saved names are the variables the generated run.sh writes, so no
    # translation happens here and the file stays the single source of truth.
    gpu = _adopt(plan, config, environ, "NVIDIA_GPU") or "0"
    port = _adopt(plan, config, environ, "PENNYROYAL_PORT") or "8001"
    image = _adopt(plan, config, environ, "PENNYROYAL_IMAGE") or DEFAULT_IMAGE
    for name, value in config.unknown.items():
        plan.env[name] = value
        if name not in CONTAINER_PASSTHROUGH_KEYS:
            # Nothing reads it: the mounted script names its own settings and
            # run.sh forwards the documented runtime knobs (the list the manual
            # Compose file forwards). Saying so beats a launch that quietly
            # ignores a key the operator put in the file.
            plan.issues.append(
                Issue(
                    "warn",
                    f"{name}={value} is saved but no Pennyroyal container "
                    "launch file reads it; the startup script owns its "
                    "settings and run.sh forwards "
                    f"{', '.join(CONTAINER_PASSTHROUGH_KEYS)}",
                    key=name,
                )
            )
    plan.env[PROFILE_KEY] = config.profile

    # An unset target follows the selected profile instead of always Next;
    # an explicit path from the file or the environment always wins.
    target = resolve(config, environ, "TARGET_MODEL")
    if target.value:
        plan.env["TARGET_MODEL"] = target.value
    else:
        plan.env["TARGET_MODEL"] = CONTAINER_DEFAULT_TARGET[config.profile]
    plan.origins["TARGET_MODEL"] = target.origin or "profile default"
    draft = resolve(config, environ, "DRAFT_MODEL")
    if draft.value:
        plan.env["DRAFT_MODEL"] = draft.value
    elif config.profile == "27b":
        plan.env["DRAFT_MODEL"] = CONTAINER_DEFAULT_DRAFT
    plan.origins["DRAFT_MODEL"] = draft.origin or (
        "profile default" if config.profile == "27b" else "not used"
    )

    disk_tier = plan.env.get("NIXL", "on")
    startup_name = STARTUP_BY_PROFILE[config.profile]
    toml_name = NIXL_TOML_BY_PROFILE[config.profile]
    # The shipped examples are the template for the generated launch; a missing
    # one is a broken checkout, and nothing here guesses a replacement. The NIXL
    # TOML is only part of the picture while the disk tier is mounted: requiring
    # it when the tier is off would ask for a file nothing reads.
    relative = [Path("run.sh"), Path("config") / startup_name]
    if disk_tier == "on":
        relative.append(Path("config") / toml_name)
    for missing in relative:
        if not (templates / missing).is_file():
            plan.issues.append(
                Issue("error", f"launch template is missing: " f"{templates / missing}")
            )
    if not Path(saved_file).is_file():
        plan.issues.append(
            Issue(
                "warn",
                f"container settings do not exist yet: "
                f"{saved_file}; create them with "
                f"./configure-penny --container",
            )
        )
    if not (launch_dir / "run.sh").is_file():
        plan.issues.append(
            Issue(
                "warn",
                f"no generated launch yet: save the "
                f"settings with ./configure-penny "
                f"--container to write "
                f"{launch_dir / 'run.sh'}",
            )
        )
    # The models root is bind-mounted read-only, so it only has to exist and be
    # readable here; the cache roots must be writable by the configured runtime
    # identity. We report that assumption; we never chown or sudo. With the
    # disk tier off there is no /nixl mount at all, so its host root is neither
    # required nor checked (and nothing in it is deleted).
    for name, writable in (
        ("HOST_MODELS_ROOT", False),
        ("HOST_CACHE_BASE", True),
        ("HOST_NIXL_STORAGE_BASE", True),
    ):
        if name == "HOST_NIXL_STORAGE_BASE" and disk_tier == "off":
            continue
        value = plan.env.get(name, "")
        if not value:
            plan.issues.append(Issue("error", f"{name} is not set", key=name))
            continue
        if not Path(value).is_absolute():
            plan.issues.append(
                Issue(
                    "error",
                    f"{name} must be an absolute host " f"path: {value}",
                    key=name,
                )
            )
            continue
        _check_root(
            plan,
            name,
            value,
            writable=writable,
            runtime_identity=_runtime_identity(plan) if writable else "",
        )
    # run.sh options win over its own settings, so the printed command carries
    # the validated values and cannot run on a stale or ambient one.
    plan.argv = [
        str(launch_dir / "run.sh"),
        "--startup",
        str(launch_dir / "config" / startup_name),
        "--image",
        image,
        "--models",
        plan.env.get("HOST_MODELS_ROOT", ""),
        "--cache",
        plan.env.get("HOST_CACHE_BASE", ""),
        "--port",
        port,
        "--gpu",
        gpu,
        "--user",
        f"{plan.env.get('USER_ID', '1000')}" f":{plan.env.get('GROUP_ID', '1000')}",
    ]
    if disk_tier == "on":
        plan.argv += [
            "--nixl-root",
            plan.env.get("HOST_NIXL_STORAGE_BASE", ""),
            "--nixl-config",
            str(launch_dir / "config" / toml_name),
        ]
    else:
        plan.argv.append("--no-nixl")
    if plan.env.get("PENNY_PLE_BACKEND") == "nvme":
        plan.argv.append("--nvme-ple")
    plan.forced_env = dict(plan.env)
    models_root = plan.env.get("HOST_MODELS_ROOT", "")
    if models_root:
        for name in ("TARGET_MODEL", "DRAFT_MODEL"):
            path_value = plan.env.get(name, "")
            if not path_value:
                continue
            issue = container_mount_issue(models_root, path_value, name)
            if issue:
                plan.issues.append(issue)
        # The prepared NVMe PLE snapshot is read inside the container like any
        # other model path, so it gets the same mount check: an arbitrary host
        # directory would be forwarded and then simply not exist there. The
        # choice stays independent of the disk tier above.
        if plan.env.get("PENNY_PLE_BACKEND") == "nvme":
            snapshot = plan.env.get("PENNY_PLE_NVME_MODEL", "")
            if not snapshot:
                plan.issues.append(
                    Issue(
                        "error",
                        "PENNY_PLE_BACKEND=nvme needs PENNY_PLE_NVME_MODEL "
                        "(the prepared snapshot under the /models mount)",
                        key="PENNY_PLE_NVME_MODEL",
                    )
                )
            else:
                issue = container_mount_issue(
                    models_root, snapshot, "PENNY_PLE_NVME_MODEL"
                )
                if issue:
                    plan.issues.append(issue)
    _profile_pinned_issues(plan)

    plan.next_command = " ".join(quote_command_arg(arg) for arg in plan.argv)
    plan.summary = [
        "mode: container",
        _profile_line(config),
        f"config: {config.source}",
        f"launch directory: {launch_dir}",
        f"generated startup script: config/{startup_name}",
        f"image: {image}",
        f"recipe in the image: configs/pennyroyal/{recipe_for(config.profile)}",
        f"host models root: {models_root or 'not set'} -> {CONTAINER_MODELS_TARGET}",
        f"target: {plan.env['TARGET_MODEL']} ({plan.origins['TARGET_MODEL']})",
        f"draft: {plan.env.get('DRAFT_MODEL', 'not used by this profile')}",
        f"writable caches: {plan.env.get('HOST_CACHE_BASE', 'not set')}",
        _disk_tier_line(disk_tier, plan.env.get("HOST_NIXL_STORAGE_BASE", "off")),
        f"runtime identity: {plan.env.get('USER_ID', '1000')}"
        f":{plan.env.get('GROUP_ID', '1000')}",
        f"API port: {port}",
        f"GPU: {gpu}",
        _hicache_size_line(plan.env.get("PENNY_HICACHE_SIZE_GB", ""), config.profile),
    ]
    if disk_tier == "on":
        plan.summary.append(
            _nixl_budget_line(plan.env.get("SGLANG_HICACHE_NIXL_MAX_CACHE_GB", "0"))
        )
    return plan


CONTAINER_DEFAULT_TARGET = {
    "next": f"{CONTAINER_MODELS_TARGET}/RadixArk-Qwen3.8-Flash-Next-NVFP4",
    "next-plain": f"{CONTAINER_MODELS_TARGET}/RadixArk-Qwen3.8-Flash-Next-NVFP4",
    "27b": f"{CONTAINER_MODELS_TARGET}/Qwen3.8-27B-FP8",
}
CONTAINER_DEFAULT_DRAFT = f"{CONTAINER_MODELS_TARGET}/Qwen3.8-27B-DFlash2"


def _read_template(path: Path, purpose: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read the {purpose} template {path}: {exc}") from exc


def _set_setting(text: str, template: str, lhs: str, value: str) -> str:
    """Rewrite one `LHS=...` line of a shipped launch file.

    Generation edits the same settings block the operator would edit by hand, so
    what lands on disk stays an ordinary runnable file. A setting the template
    does not carry is skipped rather than invented: that is the profile's own
    pinned value (the 27b recipe, for example, hard-codes its capacity and TP1),
    and the shipped file remains the truth for it.
    """
    pattern = re.compile(rf"^({re.escape(lhs)}=)[^\n]*$", re.MULTILINE)
    return pattern.sub(lambda match: f"{match.group(1)}{value}", text, count=1)


def _has_setting(text: str, lhs: str) -> bool:
    return re.search(rf"^{re.escape(lhs)}=", text, re.MULTILINE) is not None


def _forward_runtime_env(text: str, plan: Plan) -> str:
    """Name in run.sh the runtime settings the mounted script does not decide.

    These are knobs the startup script either never mentions, or mentions only
    as a default it honours (SGLANG_FORWARD_UNKNOWN_TOOLS): the docker arguments
    have to carry the operator's choice, because nothing reads the saved file
    when the container starts.
    """
    anchor = "docker_args=(\n  run --rm --init\n"
    if anchor not in text:
        raise ConfigError(
            "the run.sh template has no docker argument list to " "extend"
        )
    keys = [
        key
        for key in CONTAINER_PASSTHROUGH_KEYS
        if plan.env.get(key)
        and not (key in DISK_ONLY_KEYS and plan.env.get("NIXL", "on") == "off")
    ]
    lines = [f"  -e {shlex.quote(f'{key}={plan.env[key]}')}" for key in keys]
    if not lines:
        return text
    block = (
        "\n".join(
            [
                "  # Pennyroyal setup: runtime settings the mounted startup script "
                "leaves",
                "  # to the container environment, written here so the launch needs no "
                "config",
                "  # file, no host Python and no configurator at container start.",
            ]
            + lines
        )
        + "\n"
    )
    return text.replace(anchor, anchor + block, 1)


def _render_startup(template: Path, plan: Plan, disk_tier: str) -> str:
    text = _read_template(template, "startup script")
    pinned = PROFILE_PINNED_KEYS.get(plan.config.profile, {})
    choices = {
        "TARGET_MODEL": plan.env.get("TARGET_MODEL", ""),
        "DRAFT_MODEL": plan.env.get("DRAFT_MODEL", ""),
        "HICACHE_SIZE_GB": plan.env.get("PENNY_HICACHE_SIZE_GB", ""),
        "TP_SIZE": plan.env.get("TP_SIZE", ""),
        "PENNY_PLE_BACKEND": plan.env.get("PENNY_PLE_BACKEND", ""),
        "export SGLANG_MM_PREPROCESS_DEVICE": plan.env.get(
            "SGLANG_MM_PREPROCESS_DEVICE", ""
        ),
        "MAX_RUNNING_REQUESTS": plan.env.get("MAX_RUNNING_REQUESTS", ""),
        "MAX_MAMBA_CACHE_SIZE": plan.env.get("MAX_MAMBA_CACHE_SIZE", ""),
        # The disk tier always has to be stated: run.sh decides the mount
        # separately and the two choices must agree.
        "NIXL": disk_tier,
    }
    for lhs, value in choices.items():
        # A knob this profile's recipe pins is not the configurator's to write:
        # the shipped file stays the truth for it, so a generated script can
        # never claim a setting the launch line ignores (see the plan check in
        # _profile_pinned_issues, which refuses such a saved choice anyway).
        if lhs.removeprefix("export ") in pinned:
            continue
        if value and _has_setting(text, lhs):
            text = _set_setting(text, template.name, lhs, quote_command_arg(value))
    return text


def container_launch_files(plan: Plan) -> list[tuple[Path, str]]:
    """The ordinary files a saved container setup generates.

    Copies of the shipped docker/pennyroyal/launch examples carrying the
    validated settings: run.sh (image, GPU, port, mounts, disk tier) plus the
    profile's startup script, and its NIXL TOML only when the disk tier is on.
    """
    root = plan.repo_root or discover_repo_root()
    templates = Path(root) / LAUNCH_RELPATH
    launch_dir = plan.launch_dir or Path(
        _expand(plan.env.get("LAUNCH_DIR", "") or DEFAULT_CONTAINER_LAUNCH_DIR, root)
    )
    profile = plan.config.profile
    startup_name = STARTUP_BY_PROFILE[profile]
    toml_name = NIXL_TOML_BY_PROFILE[profile]
    disk_tier = plan.env.get("NIXL", "on")
    run_text = _read_template(templates / "run.sh", "run.sh")
    for lhs, value in (
        ("IMAGE", quote_command_arg(plan.env.get("PENNYROYAL_IMAGE", DEFAULT_IMAGE))),
        # $SCRIPT_DIR has to survive to the shell, so this one stays in
        # double quotes instead of the literal quoting the paths get.
        ("STARTUP", f'"$SCRIPT_DIR/config/{startup_name}"'),
        ("HOST_MODELS_ROOT", quote_command_arg(plan.env.get("HOST_MODELS_ROOT", ""))),
        ("HOST_CACHE_BASE", quote_command_arg(plan.env.get("HOST_CACHE_BASE", ""))),
        (
            "HOST_NIXL_STORAGE_BASE",
            quote_command_arg(plan.env.get("HOST_NIXL_STORAGE_BASE", "")),
        ),
        ("PORT", plan.env.get("PENNYROYAL_PORT", "8001")),
        ("GPU", plan.env.get("NVIDIA_GPU", "0")),
        (
            "RUN_AS",
            f"{plan.env.get('USER_ID', '1000')}" f":{plan.env.get('GROUP_ID', '1000')}",
        ),
        ("NIXL", disk_tier),
        ("NVME_PLE", "on" if plan.env.get("PENNY_PLE_BACKEND") == "nvme" else "off"),
    ):
        run_text = _set_setting(run_text, "run.sh", lhs, value)
    run_text = _forward_runtime_env(run_text, plan)
    files = [
        (launch_dir / "run.sh", run_text),
        (
            launch_dir / "config" / startup_name,
            _render_startup(templates / "config" / startup_name, plan, disk_tier),
        ),
    ]
    if disk_tier == "on":
        files.append(
            (
                launch_dir / "config" / toml_name,
                _read_template(templates / "config" / toml_name, "NIXL config"),
            )
        )
    return files


def write_container_files(plan: Plan) -> list[Path]:
    """Create the generated launch files (run.sh, startup script, NIXL TOML).

    Generation is a save-time act, never a launch-time one, and a save only
    ever writes fresh paths: an existing file is never replaced, so ./run.sh
    afterwards runs exactly the files the operator has in that folder.
    """
    return create_new_files(container_launch_files(plan))


def container_mount_issue(
    host_models_root: str, container_path: str, name: str
) -> Optional[Issue]:
    """Constrain container paths to the read-only /models mount."""
    if not container_path.startswith(f"{CONTAINER_MODELS_TARGET}/"):
        return Issue(
            "error",
            f"{name} must live under {CONTAINER_MODELS_TARGET}/ "
            f"because only the host models root is mounted "
            f"there: {container_path}",
        )
    relative = container_path[len(CONTAINER_MODELS_TARGET) :].lstrip("/")
    host_path = Path(host_models_root) / relative
    if not host_path.exists():
        return Issue(
            "error",
            f"{name} does not exist under the models mount: "
            f"{container_path} (expected host path {host_path})",
        )
    return None


# --------------------------------------------------------------------------
# Optional GPU discovery: nvidia-smi when present, manual entry otherwise
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Gpu:
    index: str
    name: str

    def label(self) -> str:
        return f"{self.index} — {self.name}"


def discover_gpus(
    runner: Optional[object] = None, timeout: float = 10.0
) -> tuple[list[Gpu], str]:
    """Best-effort readable GPU list. Never imports CUDA; never fails hard.

    Multi-GPU Compose topology overrides stay an advanced manual task; the
    caller explains that instead of guessing a new generator.
    """
    if runner is None:
        tool = shutil.which("nvidia-smi")
        if not tool:
            return [], (
                "nvidia-smi was not found; enter a GPU index or UUID " "manually."
            )
        runner_cmd: list[str] = [
            tool,
            "--query-gpu=index,name",
            "--format=csv,noheader",
        ]
        try:
            completed = subprocess.run(
                runner_cmd, capture_output=True, text=True, timeout=timeout, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return [], (
                f"nvidia-smi could not run ({exc}); enter a GPU index " "manually."
            )
        if completed.returncode != 0:
            return [], (
                "nvidia-smi reported no usable GPUs; enter a GPU index " "manually."
            )
        text = completed.stdout
    else:  # pragma: no cover - test seam
        try:
            text = runner()  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 - display the reason only
            return [], f"GPU query failed ({exc}); enter a GPU index manually."
    gpus: list[Gpu] = []
    for line in text.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) >= 2 and _INT_RE.match(fields[0]):
            gpus.append(Gpu(index=fields[0], name=", ".join(fields[1:])))
    if not gpus:
        return [], ("no GPUs were reported; enter a GPU index or UUID manually.")
    return gpus, ""


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def format_plan(plan: Plan, show_env: bool = False) -> str:
    lines = ["Pennyroyal configuration"]
    lines.extend(f"  {line}" for line in plan.summary)
    lines.append(f"  launch command: {launch_display(plan)}")
    # The command the operator should run next, with the exact config path and
    # profile that produced this plan, valid from any working directory.
    lines.append(f"  next command: {plan.next_command}")
    for issue in plan.errors:
        lines.append(f"  ERROR: {issue.message}")
    for issue in plan.warnings:
        lines.append(f"  note:  {issue.message}")
    if show_env:
        lines.append("  environment for the launch:")
        for key in sorted(plan.env):
            lines.append(f"    {key}={quote_value(plan.env[key])}")
    return "\n".join(lines)


def launch_display(plan: Plan) -> str:
    """What actually runs, including the values that make it reproducible.

    The recipe itself (or the generated run.sh) is named together with every
    environment value the validated plan supplies, so nothing runs on a secret
    ambient value that was not shown here first.
    """
    body = " ".join(quote_command_arg(arg) for arg in plan.argv)
    # Show every value the plan supplies, including the effective runtime
    # selection (REPO_ROOT, SGLANG_EXE, PYTHON), so no displayed launch hides
    # an inherited variable behind a default.
    prefix = " ".join(
        f"{key}={quote_command_arg(value)}" for key, value in sorted(plan.env.items())
    )
    return f"{prefix} {body}" if prefix else body


def plan_document(plan: Plan) -> dict[str, object]:
    return {
        "mode": plan.config.mode,
        "profile": plan.config.profile,
        "source": plan.config.source,
        "path": str(plan.config.path) if plan.config.path else None,
        "saved": {**plan.config.values, PROFILE_KEY: plan.config.profile},
        "unknown": plan.config.unknown,
        "origins": plan.origins,
        "environment": plan.env,
        "argv": plan.argv,
        "next_command": plan.next_command,
        "summary": plan.summary,
        "issues": [
            {"level": issue.level, "message": issue.message} for issue in plan.issues
        ],
    }


# --------------------------------------------------------------------------
# Command line: the non-interactive face behind --check / --show-config
# --------------------------------------------------------------------------

HELP_EPILOG = """\
Files and precedence:
  native config    ~/.config/pennyroyal/pennyroyal.env (0600, user-owned)
  container config ~/.config/pennyroyal/pennyroyal-container.env, the
                   configurator's own file: each save ./configure-penny
                   --container writes a NEW timestamped copy of these
                   settings plus a fresh timestamped folder holding an
                   ordinary run.sh, a startup script (and a NIXL TOML when
                   the disk tier is on); previous files are never replaced,
                   and the container then reads only those files --
                   no host Python, checkout, Compose or .env at container start.
  --config PATH wins over PENNYROYAL_CONFIG, which wins over the default above.
  A key saved in the file wins over an inherited environment variable. A key
  saved explicitly blank suppresses the inherited value: it resets to the
  documented default for keys that have one, and exports an empty value for
  recipe-owned keys so the recipe's own default fires. A key that is not
  saved keeps the value it inherited, or the recipe's own default.
  The file is never shell-sourced or eval'd; unknown keys are preserved and
  exported as-is. Validation here never imports the model stack or CUDA.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="penny_config.py",
        description="Validate and display the saved Pennyroyal configuration.",
        epilog=HELP_EPILOG,
    )
    parser.add_argument(
        "--mode", choices=MODES, default="native", help="native (default) or container"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="explicit saved-config path (wins over discovery)",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default=None,
        help="override the saved profile for this run",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate only; exit 1 on any error, no GPU needed",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="resolved configuration and launch command",
    )
    parser.add_argument(
        "--show-env",
        action="store_true",
        help="also list the environment the launch would use",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="print the resolved config as .env body text",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="machine-readable output between JSON markers",
    )
    parser.add_argument(
        "--launch", action="store_true", help=argparse.SUPPRESS
    )  # exec the validated plan here
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    environ = dict(os.environ)
    try:
        repo_root = discover_repo_root()
        path, _origin = discover_config_path(args.mode, environ, repo_root, args.config)
        config = load_config(args.mode, path, environ, repo_root)
        if args.profile:
            # An explicit CLI profile beats the saved file for this run only;
            # it is not written back anywhere.
            config.cli_profile = args.profile
            config.profile = validate_profile(args.profile, args.mode)
            config.profile_origin = "--profile"
        plan = build_plan(args.mode, config, environ)
    except ConfigError as exc:
        print(f"Pennyroyal config error: {exc}", file=sys.stderr)
        return 2
    if args.launch:
        # One validated plan, one real execution: this process becomes the
        # recipe (or docker compose) with exactly the resolved environment, so
        # the shell never parses a plan and nothing is eval'd anywhere. The
        # summary stays on stderr so it remains readable before the exec.
        print(format_plan(plan), file=sys.stderr, flush=True)
        if plan.errors:
            for issue in plan.errors:
                print(f"check failed: {issue.message}", file=sys.stderr)
            return 1
        os.execvpe(plan.argv[0], plan.argv, {**environ, **plan.env})
        return 1  # unreachable
    if args.json:
        print(JSON_BEGIN)
        print(json.dumps(plan_document(plan), indent=2, sort_keys=True))
        print(JSON_END)
    elif args.print_env:
        sys.stdout.write(
            serialize_env(
                [("saved", sorted(plan.config.values.items()))],
                header=(f"{PROFILE_KEY}={quote_value(plan.config.profile)}",),
            )
        )
    else:
        print(format_plan(plan, show_env=args.show_env))
    for issue in plan.errors:
        print(f"check failed: {issue.message}", file=sys.stderr)
    return 1 if plan.errors else 0


def discover_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
