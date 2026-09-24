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

    def confirm(self, label: str, default: bool = True) -> bool:
        if self.assume_yes:
            return True
        suffix = "[Y/n]" if default else "[y/N]"
        while True:
            answer = self._read(f"{label} {suffix}: ").lower()
            if not answer:
                return default
            if answer in pc.TRUE_VALUES + ("y",):
                return True
            if answer in pc.FALSE_VALUES + ("n",):
                return False
            self.say("  Answer yes or no.")


# Prompt order for the wizard: basic section first, advanced on request.
_BASIC_ORDER = ("REPO_ROOT", "VENV_PATH", "COMPOSE_FILE", "PENNYROYAL_IMAGE",
                "HOST_MODELS_ROOT", "TARGET_MODEL", "DRAFT_MODEL",
                "HOST_CACHE_BASE", "HOST_NIXL_STORAGE_BASE", "CACHE_BASE",
                "NIXL_STORAGE_BASE", "GPU", "NVIDIA_GPU", "PENNYROYAL_PORT",
                "SGLANG_HICACHE_NIXL_MAX_CACHE_GB")
_ADVANCED_ORDER = ("USER_ID", "GROUP_ID", "PENNY_PLE_BACKEND",
                   "PENNY_PLE_NVME_MODEL", "SGLANG_SM120_ONLINE_MXFP8",
                   "SGLANG_MM_PREPROCESS_DEVICE", "SGLANG_FORWARD_UNKNOWN_TOOLS",
                   "MAX_RUNNING_REQUESTS", "MAX_MAMBA_CACHE_SIZE",
                   "MAX_TOTAL_TOKENS", "PENNY_BUILD_JOBS", "NIXL_PREFIX")
_ALWAYS_REQUIRED = ("TARGET_MODEL", "DRAFT_MODEL", "REPO_ROOT", "VENV_PATH",
                    "CACHE_BASE", "NIXL_STORAGE_BASE", "HOST_MODELS_ROOT",
                    "HOST_CACHE_BASE", "HOST_NIXL_STORAGE_BASE", "COMPOSE_FILE")


def ordered_names(mode: str, advanced: bool = False) -> list[str]:
    specs = pc.specs_for(mode)
    order = _ADVANCED_ORDER if advanced else _BASIC_ORDER
    names = [name for name in order
             if name in specs and bool(specs[name].advanced) == advanced]
    names += [name for name in specs
              if name not in order and bool(specs[name].advanced) == advanced]
    return names


def _prompt_gpu(prompt: Prompt, answers: dict[str, str], name: str, spec,
                default: str, gpus: list[pc.Gpu], note: str) -> None:
    if gpus:
        prompt.say("  Visible GPUs (from nvidia-smi):")
        for gpu in gpus:
            prompt.say(f"    {gpu.label()}")
        prompt.say("  Multi-GPU Compose topology overrides stay an advanced "
                   "manual edit; this setup does not generate them.")
        label = "GPU index or UUID to use"
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
    prompt.say("Pennyroyal setup writes one plain text file and starts nothing.")
    prompt.say("Typing 'q' (or end of input) cancels; nothing is written until "
               "you confirm.")

    specs = pc.specs_for(mode)
    profile = ""
    while True:
        answer = prompt.ask("Model profile",
                            default=existing_profile or "next")
        try:
            profile = pc.validate_profile(answer, mode)
            break
        except pc.ConfigError as exc:
            prompt.say(f"  {exc}")
    prompt.say("")
    for name in pc.PROFILES:
        marker = "*" if name == profile else " "
        prompt.say(f"  {marker} {name}: {pc.PROFILE_LABEL[name]}")

    gpus, gpu_note = pc.discover_gpus()
    answers: dict[str, str] = {}

    prompt.say("")
    prompt.say("Downloaded model locations, caches, and GPU")
    for name in ordered_names(mode):
        spec = specs[name]
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
        default = default or spec.default
        allow_empty = bool(default) or name not in _ALWAYS_REQUIRED
        answers[name] = prompt.ask_validated(label, spec, default=default,
                                             allow_empty=allow_empty)
    if profile != "27b" and "DRAFT_MODEL" in saved:
        answers["DRAFT_MODEL"] = saved["DRAFT_MODEL"]

    if prompt.confirm("\nConfigure the advanced section (capacity, online FP8, "
                      "PLE placement, runtime identity)?", default=False):
        for name in ordered_names(mode, advanced=True):
            if name in answers:
                continue
            spec = specs[name]
            default = saved.get(name, "") or spec.default
            label = spec.prompt or f"{name} ({spec.description})"
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
        hint = (", ".join(choices) + ", or 's'") if choices else "'s'"
        choice = prompt.ask(
            f"Correct one of: {hint} (anything else cancels; nothing is "
            "written)", default="", allow_empty=True)
        if choice.lower() == "s":
            prompt.say("Saving despite the errors; --check will keep reporting "
                       "them.")
            break
        match = next((issue for issue in correctable
                      if issue.key == choice.strip()), None)
        if match is None:
            prompt.say("Cancelled; nothing was written.")
            return 3
        spec = specs[match.key]
        answers[match.key] = prompt.ask_validated(
            f"  new value for {match.key}", spec,
            default=answers.get(match.key, preserved.get(match.key, "")),
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
