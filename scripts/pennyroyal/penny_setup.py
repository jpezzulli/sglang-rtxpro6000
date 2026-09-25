#!/usr/bin/env python3
"""Optional interactive setup for Pennyroyal: ./configure-penny.

Stdlib-only terminal wizard. It produces exactly the same validated files as
the non-interactive path because it shares penny_config.py for the .env-style
format, validation, and command generation. Rerunning loads existing choices;
unrecognized advanced keys are preserved; cancelling, EOF, or a declined
confirmation writes nothing; an existing file is never replaced without consent.

This utility only writes configuration and prints the next command. It does not
download models, install packages, start or restart anything, load a model,
delete caches, change ownership, run sudo, or touch the network.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))

import penny_config as pc  # noqa: E402


class Cancelled(Exception):
    """The user (or EOF) ended the session before the save confirmation."""


class Prompt:
    """Small stdin/stdout helper. Empty stdin (EOF) and 'q' always cancel."""

    def __init__(self, stdin: Optional[TextIO] = None,
                 stdout: Optional[TextIO] = None,
                 assume_yes: bool = False) -> None:
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.assume_yes = assume_yes

    def say(self, text: str = "") -> None:
        print(text, file=self.stdout)

    def _read(self, label: str) -> str:
        self.stdout.write(label)
        self.stdout.flush()
        line = self.stdin.readline()
        if line == "":
            self.say("")
            raise Cancelled()
        answer = line.rstrip("\n").rstrip("\r").strip()
        if answer.lower() in {"q", "quit"}:
            raise Cancelled()
        return answer

    def ask(self, label: str, default: str = "", allow_empty: bool = False) -> str:
        suffix = f" [{default}]" if default else (" [optional]" if allow_empty
                                                  else "")
        while True:
            answer = self._read(f"{label}{suffix}: ")
            if answer:
                return answer
            if default:
                return default
            if allow_empty:
                return ""
            self.say("  Enter a value, or 'q' to cancel without saving.")

    def ask_validated(self, label: str, spec: pc.KeySpec, default: str = "",
                      allow_empty: bool = False) -> str:
        # Re-ask on an invalid answer; EOF always raises Cancelled, so the loop
        # cannot trap a piped session.
        while True:
            answer = self.ask(label, default=default, allow_empty=allow_empty)
            try:
                return pc.validate_value(spec, answer, "your answer")
            except pc.ConfigError as exc:
                self.say(f"  {exc}")

    def choose(self, label: str, options: list[tuple[str, str]],
               default: str = "", accept: Optional[dict[str, str]] = None,
               manual: Optional[tuple[str, str]] = None) -> str:
        """Numbered menu, printed BEFORE the question.

        options are (value, display); manual is one more (value, display) the
        caller interprets as a free-text entry (it returns that value). Enter
        keeps the current/default value, 'q' or EOF cancels, a number or the
        choice's own text selects, and anything else is re-asked.
        """
        items = list(options) + ([manual] if manual else [])
        self.say(label + ":")
        for number, (_, display) in enumerate(items, start=1):
            self.say(f"  [{number}] {display}")
        keep = "Enter keeps the default"
        for number, (value, _) in enumerate(items, start=1):
            if value == default:
                keep = f"Enter keeps [{number}] {value}"
                break
        else:
            if default:
                keep = f"Enter keeps the current value ({default})"
        while True:
            answer = self._read(f"  Choice ({keep}, 'q' cancels): ").lower()
            if not answer:
                return default
            if answer.isdigit() and 1 <= int(answer) <= len(items):
                return items[int(answer) - 1][0]
            for value, _ in items:
                if answer == value.lower():
                    return value
            hit = (accept or {}).get(answer)
            if hit is not None:
                return hit
            self.say("  Pick one of the listed numbers, or Enter for the "
                     "default; 'q' cancels without saving.")

    def confirm(self, label: str, default: bool = True) -> bool:
        if self.assume_yes:
            return True
        answer = self.choose(
            label, [("yes", "yes"), ("no", "no")],
            default="yes" if default else "no",
            accept={word: "yes" for word in pc.TRUE_VALUES + ("y", "yes")}
                   | {word: "no" for word in pc.FALSE_VALUES + ("n", "no")})
        return answer == "yes"


# Prompt order for the wizard: basic section first, advanced on request.
# The keys a normal first run should not have to think about (the sglang and
# python programs, the image, the runtime identity, the NIXL prefix, the
# capacity/FP8/PLE knobs) are all advanced=True in penny_config.py, so they
# only appear when the operator says yes to the advanced section.
_BASIC_ORDER = ("REPO_ROOT", "VENV_PATH", "COMPOSE_FILE", "PENNYROYAL_IMAGE",
                "HOST_MODELS_ROOT", "TARGET_MODEL", "DRAFT_MODEL",
                "HOST_CACHE_BASE", "HOST_NIXL_STORAGE_BASE", "CACHE_BASE",
                "NIXL_STORAGE_BASE", "GPU", "NVIDIA_GPU", "PENNYROYAL_PORT",
                "SGLANG_HICACHE_NIXL_MAX_CACHE_GB", "PENNY_HICACHE_SIZE_GB")
_ADVANCED_ORDER = ("USER_ID", "GROUP_ID", "PENNY_PLE_BACKEND",
                   "PENNY_PLE_NVME_MODEL", "SGLANG_SM120_ONLINE_MXFP8",
                   "SGLANG_MM_PREPROCESS_DEVICE", "SGLANG_FORWARD_UNKNOWN_TOOLS",
                   "MAX_RUNNING_REQUESTS", "MAX_MAMBA_CACHE_SIZE",
                   "MAX_TOTAL_TOKENS", "PENNY_BUILD_JOBS", "NIXL_PREFIX")
_ALWAYS_REQUIRED = ("TARGET_MODEL", "DRAFT_MODEL", "REPO_ROOT", "VENV_PATH",
                    "CACHE_BASE", "NIXL_STORAGE_BASE", "HOST_MODELS_ROOT",
                    "HOST_CACHE_BASE", "HOST_NIXL_STORAGE_BASE", "COMPOSE_FILE")


# One short plain-English line per question, printed before it. The technical
# variable name stays in the review listing and in the saved file; the wizard
# asks about what the thing is for instead of its exported spelling.
_EXPLANATIONS = {
    "REPO_ROOT": "the folder you unpacked Pennyroyal into; a first run keeps "
                 "the suggested path",
    "VENV_PATH": "the Python environment folder holding the sglang command; a "
                 "first run keeps the suggested path",
    "SGLANG_EXE": "the sglang program itself, only worth changing when that "
                  "folder layout is unusual",
    "PYTHON": "the python interpreter that starts the server, only worth "
              "changing for an unusual layout",
    "TARGET_MODEL": "the folder of the downloaded model you want to serve",
    "DRAFT_MODEL": "the smaller model the 27b profile drafts tokens with",
    "COMPOSE_FILE": "the compose file to run; the shipped one is suggested, so "
                    "a first run keeps it",
    "PENNYROYAL_IMAGE": "which container image to start; change it only to pin "
                        "a different published tag",
    "PENNYROYAL_PORT": "the port on this machine the API listens on",
    "CACHE_BASE": "the folder for compiled kernels and runtime files. That is "
                  "not the RAM model cache and nothing here deletes it",
    "NIXL_STORAGE_BASE": "the folder the persistent NIXL cache writes to on "
                         "disk",
    "HOST_MODELS_ROOT": "the folder on this machine that the container sees as "
                        "its read-only models folder",
    "HOST_CACHE_BASE": "the folder on this machine that the container uses for "
                       "compiled kernels and runtime files",
    "HOST_NIXL_STORAGE_BASE": "the folder on this machine the container uses "
                              "for its persistent NIXL cache on disk",
    "GPU": "which physical graphics card the model gets; the menu lists what "
           "the machine reported",
    "NVIDIA_GPU": "which physical graphics card the container may use",
    "SGLANG_MM_PREPROCESS_DEVICE": "where image and audio input is prepared; "
                                   "the CPU is the qualified default",
    "PENNY_HICACHE_SIZE_GB": "how much system RAM the KV cache's RAM tier may "
                             "use, in decimal GB (1 GB = 1e9 bytes, not GiB). "
                             "This is separate from the PLE embedding table, "
                             "which has its own RAM or NVMe placement, and "
                             "separate from the compiled cache folder above",
    "SGLANG_HICACHE_NIXL_MAX_CACHE_GB": "a cap in GiB on the persistent NIXL "
                                        "cache folder on disk. 0 means no cap: "
                                        "the NIXL cache stays enabled and is "
                                        "not turned off",
    "PENNY_PLE_BACKEND": "where the PLE embedding table lives (RAM by default, "
                         "or a prepared NVMe snapshot); it is unrelated to the "
                         "RAM cache size above",
    "PENNY_PLE_NVME_MODEL": "the prepared NVMe snapshot folder, when PLE lives "
                            "on NVMe",
    "SGLANG_SM120_ONLINE_MXFP8": "read the FP8 guide before switching this on",
    "SGLANG_FORWARD_UNKNOWN_TOOLS": "pass tool names this build does not know "
                                    "through to the model",
    "MAX_RUNNING_REQUESTS": "how many requests are admitted at once; leaving "
                            "it blank uses the recipe default",
    "MAX_MAMBA_CACHE_SIZE": "how many recurrent-state slots are kept; blank "
                            "uses the recipe default",
    "MAX_TOTAL_TOKENS": "the shared token cap for the KV pool; blank uses the "
                        "recipe default",
    "PENNY_BUILD_JOBS": "how many compile jobs the first start may run; blank "
                        "uses the recipe default",
    "NIXL_PREFIX": "where NIXL is installed when that is outside the normal "
                   "linker path",
    "USER_ID": "the numeric user that owns the writable folders inside the "
               "container",
    "GROUP_ID": "the numeric group that owns those folders",
    "PENNYROYAL_PROFILE": "which model recipe the saved file runs",
}

# Installer plumbing that only matters when the shipped layout is not the one
# on this machine, so it waits for the advanced section instead of interrupting
# a first run.
_ADVANCED_NOTE = ("the paths of the sglang and python programs, the "
                  "container image, the runtime user and group, the NIXL "
                  "install prefix, and the first-start compile jobs")


def _explain(prompt: Prompt, name: str, extra: str = "") -> None:
    """Print the human explanation of a question before asking it."""
    text = _EXPLANATIONS.get(name, "")
    if extra:
        text = f"{text} — {extra}" if text else extra
    if text:
        prompt.say(f"  {text}")


def ordered_names(mode: str, advanced: bool = False) -> list[str]:
    specs = pc.specs_for(mode)
    order = _ADVANCED_ORDER if advanced else _BASIC_ORDER
    names = [name for name in order
             if name in specs and bool(specs[name].advanced) == advanced]
    names += [name for name in specs
              if name not in order and bool(specs[name].advanced) == advanced]
    return names


# Sentinel choose() returns for its manual-entry row; the caller then asks
# free text. Values like these never collide with real settings.
_MANUAL = "__manual__"


def _ask_cuda_id(prompt: Prompt, spec: pc.KeySpec, default: str) -> str:
    """The 'another GPU' row: ask the LOGICAL CUDA index as a bare number.

    The saved value is still a cuda:N string, but typing 'cuda:N' invites the
    mistake this whole screen guards against: a host GPU index (what nvidia-smi
    lists) is not the CUDA id once CUDA_VISIBLE_DEVICES restricts the view, and
    this setup never maps one to the other. 'q'/EOF still cancels via _read.
    """
    shown = default.removeprefix("cuda:") if default.startswith("cuda:") else ""
    while True:
        answer = prompt.ask(
            "  logical CUDA index of that GPU (a number, e.g. 1 — NOT the "
            "host GPU index; this setup does not map between them)",
            default=shown)
        text = answer.strip().lower()
        number = (text[5:] if text.startswith("cuda:") else text)
        try:
            return pc.validate_value(spec, f"cuda:{number}", "your answer")
        except pc.ConfigError as exc:
            prompt.say(f"  {exc}")
            shown = ""


def _ask_fixed(prompt: Prompt, spec: pc.KeySpec, default: str) -> str:
    """Menu entry for a managed key whose answers are a closed set.

    Booleans and declared choices get the numbered menu. The media-preprocessing
    device gets cpu, the model GPU (cuda:0 = GPU 0), the saved device when it is
    some other cuda:N, and a manual 'another GPU' row — every row carries a
    human label (the manual sentinel is never displayed), and nothing is ever
    inferred from a host GPU index. A saved value outside the managed set (an
    edited file) is rejected by the shared validator and re-asked instead of
    being accepted on Enter. Paths and numeric settings stay free text.
    """
    _explain(prompt, spec.name, extra=f"variable {spec.name}")
    accept = None
    if spec.kind == "bool":
        rows = [("true", "true"), ("false", "false")]
        accept = ({word: "true" for word in pc.TRUE_VALUES}
                  | {word: "false" for word in pc.FALSE_VALUES})
    elif spec.kind == "mm-device":
        rows = [("cpu", "cpu — media preprocessing on the CPU"),
                ("cuda:0", "cuda:0 — GPU 0, the model's GPU")]
        if default.startswith("cuda:") and default != "cuda:0":
            rows.append((default, f"{default} — the device saved in this file"))
        rows.append((_MANUAL, "another GPU — type its logical CUDA index "
                             "(a number; the setup never maps the host GPU "
                             "index to a CUDA id)"))
    else:
        rows = [(c, c) for c in spec.choices]
    while True:
        answer = prompt.choose(spec.prompt or spec.name, rows,
                               default=default, accept=accept)
        if answer == _MANUAL:
            return _ask_cuda_id(prompt, spec, default)
        if not answer and default == "":
            # Nothing valid can be kept by Enter (the saved value was junk);
            # an explicit pick is required before anything is saved.
            prompt.say("  Pick one of the listed numbers for this setting.")
            continue
        try:
            return pc.validate_value(spec, answer, "your answer")
        except pc.ConfigError as exc:
            # A saved value outside the menu must not survive an Enter.
            prompt.say(f"  {exc}")
            default = ""


def _prompt_gpu(prompt: Prompt, answers: dict[str, str], name: str, spec,
                default: str, gpus: list[pc.Gpu], note: str) -> None:
    _explain(prompt, name, extra=f"variable {name}")
    # A saved file may hold anything under this key (age, hand edits), so the
    # shared validator decides whether the saved value may act as the menu's
    # Enter default or be assigned from it at all. An invalid one is named and
    # dropped — Enter then keeps the documented default instead of crashing the
    # later _candidate_config validation with an uncaught ConfigError. A valid
    # off-list value (e.g. a UUID of an unlisted device) is still preserved.
    if default:
        try:
            pc.validate_value(spec, default, "saved file")
        except pc.ConfigError as exc:
            prompt.say(f"  {exc}")
            prompt.say("  Enter will use the default instead; pick another "
                       "row or type a value to override it.")
            default = ""
    if gpus:
        # Menu numbers are UI positions; each row names the device index it
        # actually selects, so choice [1] can clearly mean GPU 0.
        prompt.say("  Multi-GPU Compose topology overrides stay an advanced "
                   "manual edit; this setup does not generate them.")
        answer = prompt.choose(
            "GPU index or UUID to use",
            [(gpu.index, f"GPU index {gpu.index} — {gpu.name}")
             for gpu in gpus],
            default=default or "0",  # the documented default is GPU 0
            manual=(_MANUAL, "type another index or UUID"))
        if answer != _MANUAL:
            # The menu can only return a listed index or the Enter default;
            # validate anyway so no path can bypass the shared rules.
            try:
                answers[name] = pc.validate_value(spec, answer, "your answer")
            except pc.ConfigError as exc:
                prompt.say(f"  {exc}")
            else:
                return
        label = "GPU index or UUID (manual entry)"
    else:
        if note:
            prompt.say(f"  {note}")
        label = "GPU index or UUID (manual entry)"
    answers[name] = prompt.ask_validated(label, spec, default=default or "0")


def run_session(mode: str, config_path: Path, environ: dict[str, str],
                prompt: Prompt, repo_root: Optional[Path] = None,
                home: Optional[Path] = None) -> int:
    repo_root = repo_root or pc.discover_repo_root()
    home = home or Path.home()
    # Resolve the chosen path once: the save target, the review messages, and
    # the printed next command must all name the identical absolute file, so a
    # relative --config still reloads correctly from any later directory.
    config_path = Path(config_path).expanduser().absolute()
    saved: dict[str, str] = {}
    preserved: dict[str, str] = {}
    existing_profile = ""
    if config_path.is_file():
        try:
            saved = pc.read_env_file(config_path)
        except pc.ConfigError as exc:
            prompt.say(f"Cannot read {config_path}: {exc}")
            return 2
        existing_profile = saved.pop(pc.PROFILE_KEY, "")
        specs = pc.specs_for(mode)
        preserved = {key: value for key, value in saved.items()
                     if key not in specs}
        saved = {key: value for key, value in saved.items() if key in specs}
        prompt.say(f"Existing configuration: {config_path}")
    else:
        prompt.say(f"No saved configuration yet: {config_path}")

    prompt.say("")
    prompt.say("Pennyroyal setup (BETA) writes one plain text file and starts "
               "nothing.")
    prompt.say("It will not download, install, start, or delete anything.")
    prompt.say("The PLE embedding table has its own placement, chosen in the "
               "advanced section; it is separate from the RAM (HiCache) cache "
               "size asked below, and from the compiled-cache folders.")
    prompt.say("Typing 'q' (or end of input) cancels; nothing is written until "
               "you confirm.")

    specs = pc.specs_for(mode)
    # A saved profile is normalized through the shared validator BEFORE it
    # becomes the menu default or a target lookup key: an older
    # 'container:next' file must preselect 'next', and a corrupt value must not
    # become a default that Enter would silently return (or crash the later
    # CONTAINER_DEFAULT_TARGET lookup). Enter on a fresh file keeps 'next'.
    menu_default = "next"
    if existing_profile:
        try:
            menu_default = pc.validate_profile(existing_profile, mode)
        except pc.ConfigError as exc:
            prompt.say(f"  Saved profile is unusable: {exc}")
            menu_default = ""
    # The numbered profile menu is printed before the question is asked.
    while True:
        # Enter is allowed only once a valid row is the stated default; an
        # unusable saved profile forces an explicit valid choice (choose()
        # has no empty-default problem because the loop always re-asks).
        _explain(prompt, pc.PROFILE_KEY,
                 extra=f"variable {pc.PROFILE_KEY}")
        candidate = prompt.choose(
            "Model profile",
            [(name, f"{name} — {pc.PROFILE_LABEL[name]}")
             for name in pc.PROFILES],
            default=menu_default or pc.PROFILES[0])
        try:
            profile = pc.validate_profile(candidate, mode)
            break
        except pc.ConfigError as exc:
            prompt.say(f"  {exc}")
            menu_default = ""  # an empty Enter must ask again, never crash

    gpus, gpu_note = pc.discover_gpus()
    answers: dict[str, str] = {}

    prompt.say("")
    prompt.say("Downloaded model locations, caches, and GPU")
    prompt.say("  Anything Pennyroyal can infer for a normal first run — the "
               "repository folder, the Python environment folder, the compose "
               "file, the cache folders, the graphics card, and the API port — "
               "is offered with that value already typed in; press Enter to "
               "keep it. A value already in this file wins over the "
               "suggestion. The remaining plumbing stays in the advanced "
               f"section: {_ADVANCED_NOTE}.")
    for name in ordered_names(mode):
        spec = specs[name]
        if spec.kind in ("bool", "choice", "mm-device"):
            answers[name] = _ask_fixed(prompt, spec,
                                       saved.get(name, "") or spec.default)
            continue
        default = saved.get(name, "")
        if name in ("GPU", "NVIDIA_GPU"):
            _prompt_gpu(prompt, answers, name, spec, default, gpus, gpu_note)
            continue
        if name == "TARGET_MODEL" and not default:
            default = (pc.CONTAINER_DEFAULT_TARGET[profile] if mode == "container"
                       else "")
        if name == "REPO_ROOT" and not default:
            default = str(repo_root)
        if name == "VENV_PATH" and not default:
            default = str(repo_root / ".venv")
        if name == "DRAFT_MODEL" and profile != "27b":
            continue  # this profile does not use a draft; the value is preserved
        if name == "COMPOSE_FILE" and not default:
            default = str(repo_root / pc.COMPOSE_RELPATH)
        if name == "NIXL_STORAGE_BASE" and not default and mode == "native":
            default = str(home / pc.DEFAULT_NATIVE_NIXL_BASE.removeprefix("~/"))
        if name == "CACHE_BASE" and not default and mode == "native":
            default = str(home / pc.DEFAULT_NATIVE_CACHE_BASE.removeprefix("~/"))
        label = spec.prompt or f"{name} ({spec.description})"
        extra = ""
        if name == "PENNY_HICACHE_SIZE_GB":
            extra = (f"blank keeps this profile's qualified default "
                     f"{pc.PROFILE_HICACHE_SIZE_GB[profile]} GB; anything from "
                     f"1 GB up is honored")
        _explain(prompt, name, extra=f"variable {name}{'; ' + extra if extra else ''}")
        default = default or spec.default
        allow_empty = bool(default) or name not in _ALWAYS_REQUIRED
        answers[name] = prompt.ask_validated(label, spec, default=default,
                                             allow_empty=allow_empty)
    if profile != "27b" and "DRAFT_MODEL" in saved:
        answers["DRAFT_MODEL"] = saved["DRAFT_MODEL"]

    if prompt.confirm("\nConfigure the advanced section (capacity, online FP8, "
                      "PLE placement, runtime identity, installation "
                      "overrides)?", default=False):
        prompt.say(f"  The advanced section covers {_ADVANCED_NOTE}, plus the "
                   "capacity, FP8, and PLE knobs. Your saved overrides for "
                   "them are offered as defaults, so Enter keeps what you "
                   "already chose.")
        for name in ordered_names(mode, advanced=True):
            if name in answers:
                continue
            spec = specs[name]
            default = saved.get(name, "") or spec.default
            if spec.kind in ("bool", "choice", "mm-device"):
                answers[name] = _ask_fixed(prompt, spec, default)
                continue
            label = spec.prompt or f"{name} ({spec.description})"
            _explain(prompt, name, extra=f"variable {name}")
            answers[name] = prompt.ask_validated(
                label, spec, default=default,
                allow_empty=bool(default) or spec.kind == "positive-int")

    # Keys the wizard never asked about (the declined advanced section) stay
    # exactly as saved, so rerunning cannot quietly drop them.
    asked = set(answers)
    for name, value in saved.items():
        if name not in asked:
            answers[name] = value

    changed = {name: value for name, value in answers.items()
               if saved.get(name, "") != value}
    prompt.say("")
    prompt.say("Review")
    for name in sorted(answers):
        marker = "*" if name in changed else " "
        prompt.say(f"  {marker} {name}={pc.quote_value(answers[name])}")
    for name, value in sorted(preserved.items()):
        prompt.say(f"    {name}={pc.quote_value(value)}  (unrecognized key kept)")
    prompt.say(f"  {len(answers) - len(changed)} value(s) unchanged; 'q' "
               "cancels without writing.")

    # Validate the proposal in memory before touching anything: an invalid path
    # must not overwrite a good file or claim success. The user may correct the
    # named key, cancel, or explicitly save a not-ready-but-valid state.
    attempt = 0
    while True:
        config = _candidate_config(mode, profile, answers, preserved, config_path)
        plan = pc.build_plan(mode, config, environ, repo_root=repo_root)
        if not plan.errors:
            break
        prompt.say("")
        prompt.say("Not ready to save — fix these or cancel:")
        for issue in plan.errors:
            prompt.say(f"  ERROR: {issue.message}")
        attempt += 1
        if attempt >= 4:
            prompt.say("Too many failed attempts; the existing file was left "
                       "untouched.")
            return 4
        correctable = [issue for issue in plan.errors if issue.key
                       and issue.key in specs
                       and (issue.key in answers or issue.key in preserved)]
        choices = sorted({issue.key for issue in correctable})
        options = [(key, f"re-enter {key}") for key in choices]
        options.append(("s", "save despite these errors (--check keeps "
                             "reporting them)"))
        options.append(("c", "cancel; nothing is written"))
        choice = prompt.choose("Fix one setting, save anyway, or cancel",
                               options, default="c")
        if choice == "s":
            prompt.say("Saving despite the errors; --check will keep reporting "
                       "them.")
            break
        match = next((issue for issue in correctable
                      if issue.key == choice), None)
        if match is None:
            prompt.say("Cancelled; nothing was written.")
            return 3
        spec = specs[match.key]
        current = answers.get(match.key, preserved.get(match.key, ""))
        if spec.kind in ("bool", "choice", "mm-device"):
            answers[match.key] = _ask_fixed(
                prompt, spec, current)
        else:
            answers[match.key] = prompt.ask_validated(
                f"  new value for {match.key}", spec, default=current,
                allow_empty=spec.kind == "positive-int")

    verb = "Replace" if config_path.exists() else "Save"
    if not prompt.confirm(f"\n{verb} {config_path}?", default=True):
        prompt.say("Cancelled; nothing was written.")
        return 3

    def entries(names: list[str]) -> list[tuple[str, str]]:
        # A blank saved on purpose (a deliberate suppression) must survive the
        # rerun as an explicit blank line; new optional blanks that were never
        # in the file stay omitted, so absence keeps meaning 'inherited'.
        return [(name, answers[name]) for name in names
                if name in answers
                and (answers[name] != "" or name in saved)]

    sections = [("basic", entries([name for name in ordered_names(mode)
                                   if not specs[name].advanced])),
                ("advanced (optional)",
                 entries(ordered_names(mode, advanced=True)))]
    if preserved:
        sections.append(("kept from the previous file", sorted(preserved.items())))
    header = (
        "# Pennyroyal saved configuration "
        f"({mode}); written by ./configure-penny.",
        "# Plain KEY=value lines. This file is never shell-sourced or eval'd.",
        "# Saved values win over the inherited environment; unset keys keep the",
        "# recipe defaults documented in the Pennyroyal guides.",
        f"{pc.PROFILE_KEY}={pc.quote_value(profile)}",
    )
    pc.write_env_file(config_path, pc.serialize_env(sections, header=header))
    prompt.say(f"Saved {config_path}")
    prompt.say("")
    prompt.say(pc.format_plan(plan))
    return 0


def _candidate_config(mode: str, profile: str, answers: dict[str, str],
                      preserved: dict[str, str],
                      config_path: Path) -> pc.Config:
    """A Config for the in-memory proposal, using the same shared validator."""
    specs = pc.specs_for(mode)
    values: dict[str, str] = {}
    unknown: dict[str, str] = {}
    merged = {**preserved, **answers}
    for name, value in merged.items():
        spec = specs.get(name)
        if spec is None:
            unknown[name] = value
            continue
        values[name] = pc.validate_value(spec, value, "proposed answers")
    config = pc.Config(mode=mode, profile=pc.validate_profile(profile, mode),
                       values=values, unknown=unknown, path=config_path,
                       source="proposed answers (not saved yet)",
                       file_profile=profile, profile_origin="setup answers")
    return config


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="configure-penny",
        description="Interactively write the saved Pennyroyal configuration.",
        epilog=pc.HELP_EPILOG)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--native", action="store_const", const="native",
                       dest="mode", help="configure the native launcher (default)")
    group.add_argument("--container", action="store_const", const="container",
                       dest="mode", help="configure the Compose service")
    parser.add_argument("--config", type=Path, default=None,
                        help="explicit config path instead of the default one")
    parser.add_argument("--yes", action="store_true",
                        help="accept the save confirmation (scripted runs)")
    parser.add_argument("--check", action="store_true",
                        help="validate the saved file and exit; writes nothing")
    parser.add_argument("--show-config", action="store_true",
                        help="show the resolved configuration and exit")
    parser.add_argument("--print-env", action="store_true",
                        help="print the saved file as .env body text and exit")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output between JSON markers")
    parser.set_defaults(mode="native")
    args = parser.parse_args(argv)

    environ = dict(os.environ)
    report = [option for option in ("--check", "--show-config", "--print-env",
                                   "--json")
              if getattr(args, option.lstrip("-").replace("-", "_"))]
    if report:
        argv = ["--mode", args.mode]
        if args.config is not None:
            argv += ["--config", str(args.config)]
        return pc.main(argv + report)
    repo_root = pc.discover_repo_root()
    config_path = (Path(args.config) if args.config is not None
                   else pc.discover_config_path(args.mode, environ, repo_root)[0])
    prompt = Prompt(assume_yes=args.yes)
    # Tests and scripted runs can isolate the home-directory defaults without
    # touching the real account.
    home = Path(environ.get("PENNYROYAL_TEST_HOME") or Path.home())
    try:
        return run_session(args.mode, config_path, environ, prompt, repo_root,
                           home=home)
    except Cancelled:
        prompt.say("\nCancelled; nothing was written.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
