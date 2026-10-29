"""CPU-only checks for the native dependency baseline and the image recipe.

Nothing here installs a release dependency: an environment under inspection is a
name to version map, the way ``check`` sees a real site directory, and the recipe
is checked by parsing the build stage and running its own checkout layer against a
throwaway root. The qualified baseline is the fixture, so these tests go red when
the pins, the classifications, the accepted warnings or the image recipe drift
apart from it.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).parent / "dependency_parity.py"
SPEC = importlib.util.spec_from_file_location("dependency_parity", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

DOCKERFILE = (ROOT / "docker" / "pennyroyal" / "Dockerfile").read_text()
BUILD_STAGE = DOCKERFILE.split("FROM base AS runtime")[0]
MANIFEST = MODULE.load(MODULE.MANIFEST_PATH)
QUALIFIED = MANIFEST["qualified"]
REVISION = "c6a033ad5fcb9f724c731c0fba7a486acdc1a91f"
# The qualified interpreter, named so these checks report package drift rather
# than the version of whatever Python happens to run them.
INTERPRETER = {"python": QUALIFIED["python"], "platform": QUALIFIED["platform"]}
INDEXES = (
    MODULE.DEFAULT_INDEX,
    "https://download.pytorch.org/whl/cu130",
    "https://flashinfer.ai/whl",
    "https://flashinfer.ai/whl/cu130",
)
EXCLUDED_TOOLS = (
    "auditwheel",
    "cfgv",
    "clang-format",
    "distlib",
    "identify",
    "iniconfig",
    "nodeenv",
    "platformdirs",
    "pluggy",
    "pre-commit",
    "pytest",
    "python-discovery",
    "uv",
    "virtualenv",
)


def steps():
    """The build stage as (instruction, payload) pairs, continuations joined.

    A comment cannot sit inside a Dockerfile continuation, so the lines that matter
    are the ones that start with an instruction or continue one.
    """
    found = []
    for line in BUILD_STAGE.splitlines():
        if line.startswith("#"):
            continue
        if found and not re.match(r"[A-Z]+( |$)", line):
            found[-1][1] += " " + line.strip()
            continue
        match = re.match(r"([A-Z]+)(?: (.*))?", line)
        if match:
            found.append([match.group(1), (match.group(2) or "").strip()])
    return [(verb, re.sub(r"\\\s*", " ", payload).strip()) for verb, payload in found]


def position(verb, needle, last=False):
    """Where in the layer order an instruction naming ``needle`` appears."""
    matches = [
        index
        for index, (kind, payload) in enumerate(steps())
        if kind == verb and needle in payload
    ]
    if not matches:
        raise AssertionError(f"no {verb} step mentions {needle!r}")
    return matches[-1] if last else matches[0]


def dist_info(site: Path, name: str, version: str, extra: tuple[str, ...] = ()):
    """A representative installed distribution, as a wheel leaves it."""
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        "\n".join((f"Name: {name}", f"Version: {version}", *extra)) + "\n"
    )
    (info / "RECORD").write_text("")
    return info


def git_remote(tmp_path: Path) -> tuple[str, str]:
    """A one-commit file:// remote and the revision the recipe would ask for."""
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "README.md").write_text("fixture\n")

    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(remote), *arguments],
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "fixture",
                "GIT_AUTHOR_EMAIL": "fixture@example",
                "GIT_COMMITTER_NAME": "fixture",
                "GIT_COMMITTER_EMAIL": "fixture@example",
            },
        )

    git("init", "-q", ".")
    git("add", "README.md")
    git("commit", "-q", "-m", "one")
    return str(remote), git("rev-parse", "HEAD").stdout.strip()


def qualified_environment(revision=REVISION, **overrides):
    """The environment the baseline describes, Pennyroyal derived from ``revision``."""
    actual = {}
    for name, record in MANIFEST["packages"].items():
        if record.get("class", "runtime") == "native-dev":
            continue
        actual[name] = MODULE.expected_version(MANIFEST, name, revision)
    actual.update(overrides)
    return actual


class DependencyParityTests(unittest.TestCase):
    def problems(self, actual, **kwargs):
        return MODULE.report(
            MODULE.check(MANIFEST, actual, REVISION, **{**INTERPRETER, **kwargs})
        )

    def test_qualified_baseline_is_parity_clean(self):
        self.assertEqual(self.problems(qualified_environment()), [])

    def test_shared_dependency_version_drift_fails(self):
        self.assertEqual(
            self.problems(qualified_environment(torch="2.14.0+cu130")),
            ["torch is 2.14.0+cu130, the baseline pins 2.13.0+cu130"],
        )

    def test_exact_local_suffix_is_enforced_in_both_directions(self):
        """`==0.4.6.post1` is satisfied by the CUDA sibling; the pin is not."""
        self.assertEqual(
            self.problems(
                qualified_environment(**{"sglang-kernel": "0.4.6.post1+cu130"})
            ),
            [
                "sglang-kernel is 0.4.6.post1+cu130, the baseline pins 0.4.6.post1; "
                "the local build tag differs"
            ],
        )
        self.assertEqual(
            self.problems(qualified_environment(torch="2.13.0")),
            [
                "torch is 2.13.0, the baseline pins 2.13.0+cu130; "
                "the local build tag differs"
            ],
        )

    def test_missing_runtime_dependency_fails(self):
        actual = qualified_environment()
        del actual["transformers"]
        self.assertEqual(self.problems(actual), ["transformers==5.12.1 is missing"])

    def test_unapproved_extra_package_fails(self):
        self.assertEqual(
            self.problems(qualified_environment(cryptography="46.0.0")),
            ["cryptography 46.0.0 is not in the baseline"],
        )

    def test_native_only_development_tool_is_not_installed(self):
        self.assertEqual(
            self.problems(qualified_environment(pytest="9.1.1")),
            [
                "pytest 9.1.1 is a native-only development tool; the image does not "
                "install it"
            ],
        )
        pins = MODULE.requirements(MANIFEST)
        for name in EXCLUDED_TOOLS:
            self.assertFalse(any(pin.startswith(f"{name}===") for pin in pins), name)
        self.assertEqual(
            sorted(
                name
                for name, record in MANIFEST["packages"].items()
                if record.get("class") == "native-dev"
            ),
            sorted(EXCLUDED_TOOLS),
        )

    def test_pyelftools_is_a_runtime_dependency_and_not_an_exclusion(self):
        """The retained Humming wheel asks for it, whatever the tool's name says."""
        self.assertNotIn("pyelftools", EXCLUDED_TOOLS)
        self.assertEqual(
            MANIFEST["packages"]["pyelftools"].get("class", "runtime"), "runtime"
        )
        self.assertIn("pyelftools===0.33", MODULE.requirements(MANIFEST))
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            dist_info(site, "humming-kernels", "0.1.10", ("Requires-Dist: pyelftools",))
            installed = MODULE.installed_versions([str(site)])
        self.assertEqual(installed, {"humming-kernels": "0.1.10"})
        problems = self.problems(installed)
        self.assertIn("pyelftools==0.33 is missing", problems)
        # And the requirement check may not forgive it either: only the two
        # baseline findings are accepted, whatever is missing.
        self.assertFalse(
            MODULE.accepted_pip_check_line(
                "humming-kernels 0.1.10 requires pyelftools, which is not installed.",
                MANIFEST,
            )
        )

    def test_the_published_image_is_red_against_the_qualified_baseline(self):
        """The 3.1.3 image resolved its own release dependencies, and shows here."""
        actual = qualified_environment(**{"sglang-kernel": "0.4.6.post1+cu130"})
        for drift in (
            "cryptography",
            "httpcore2",
            "httpx2",
            "opentelemetry-api",
            "truststore",
            "tzdata",
        ):
            actual[drift] = "1.0"
        problems = "\n".join(self.problems(actual))
        self.assertIn("sglang-kernel is 0.4.6.post1+cu130", problems)
        for drift in ("cryptography", "truststore", "tzdata"):
            self.assertIn(f"{drift} 1.0 is not in the baseline", problems)

    def test_derived_source_version_is_not_the_stale_native_metadata(self):
        stale = qualified_environment(sglang="0.5.19.dev980+gfe35e1d81")
        self.assertEqual(
            self.problems(stale),
            [
                "sglang is 0.5.19.dev980+gfe35e1d81, the baseline pins "
                f"0.5.19+g{REVISION[:12]}"
            ],
        )
        # With no revision to derive from, that one entry is not judged, and the
        # stale metadata version recorded in the baseline is provenance only.
        self.assertEqual(MODULE.check(MANIFEST, stale, **INTERPRETER)["problems"], [])

    def test_nixl_is_accounted_for_while_the_build_still_owes_it(self):
        actual = qualified_environment()
        del actual["nixl"], actual["nixl-cu13"]
        pending = MODULE.check(
            MANIFEST, actual, REVISION, pending_source_build=True, **INTERPRETER
        )
        self.assertEqual(
            sorted(pending["pending"]), ["nixl-cu13==1.4.0", "nixl==1.4.0"]
        )
        self.assertEqual(MODULE.report(pending), [])
        self.assertEqual(
            MODULE.report(MODULE.check(MANIFEST, actual, REVISION, **INTERPRETER)),
            ["nixl==1.4.0 is missing", "nixl-cu13==1.4.0 is missing"],
        )
        # The concession covers the source builds alone: while they are pending,
        # an ordinary missing package is still a missing package.
        del actual["transformers"]
        self.assertEqual(
            MODULE.report(
                MODULE.check(
                    MANIFEST,
                    actual,
                    REVISION,
                    pending_source_build=True,
                    **INTERPRETER,
                )
            ),
            ["transformers==5.12.1 is missing"],
        )

    def test_interpreter_drift_fails_the_gate(self):
        environment = qualified_environment()
        self.assertEqual(
            self.problems(environment, python="3.12.12"),
            ["Python 3.12.12 is not the qualified 3.12.13"],
        )
        self.assertEqual(
            self.problems(environment, platform="linux-aarch64"),
            ["platform linux-aarch64 is not the qualified linux-x86_64"],
        )

    def test_installed_names_are_normalized_as_distributions_report_them(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            for folder, name in (
                ("sgl_deep_ep-0.1.2.dist-info", "sgl-deep-ep"),
                ("ipython_pygments_lexers-1.1.1.dist-info", "IPython_Pygments_Lexers"),
                ("torch-2.13.0+cu130.dist-info", "torch"),
            ):
                (site / folder).mkdir()
                (site / folder / "METADATA").write_text(
                    f"Name: {name}\nVersion: 0.0.0\n"
                )
                (site / folder / "RECORD").write_text("")
            found = MODULE.installed_versions([str(site)])
        self.assertEqual(
            found,
            {
                "sgl-deep-ep": "0.0.0",
                "ipython-pygments-lexers": "0.0.0",
                "torch": "0.0.0",
            },
        )
        self.assertEqual(
            MODULE.expected_version(MANIFEST, "sgl_deep_ep", REVISION),
            MANIFEST["packages"]["sgl-deep-ep"]["version"],
        )

    def test_requirements_are_arbitrary_equality_and_group_by_index(self):
        pins = MODULE.requirements(MANIFEST)
        self.assertTrue(all("===" in pin for pin in pins), pins[:3])
        grouped = {index: MODULE.requirements(MANIFEST, index) for index in INDEXES}
        self.assertEqual(
            sorted(pins), sorted(pin for group in grouped.values() for pin in group)
        )
        self.assertEqual(MODULE.indexes(MANIFEST), list(INDEXES))
        for pins_of_index in grouped.values():
            self.assertTrue(pins_of_index)
        for name, record in MANIFEST["packages"].items():
            installed = record.get("class", "runtime") in MODULE.INDEX_CLASSES
            self.assertEqual(f"{name}==={record['version']}" in pins, installed, name)
        self.assertIn("types-pyyaml===6.0.12.20260906", grouped[MODULE.DEFAULT_INDEX])
        self.assertIn(
            "torch===2.13.0+cu130",
            grouped["https://download.pytorch.org/whl/cu130"],
        )
        self.assertIn(
            "flashinfer-jit-cache-sm120f===0.7.0.post1+cu130",
            grouped["https://flashinfer.ai/whl/cu130"],
        )
        self.assertIn(
            "flashinfer-cubin===0.7.0.post1", grouped["https://flashinfer.ai/whl"]
        )

    def test_a_group_that_names_no_package_is_an_error(self):
        with self.assertRaises(MODULE.ParityError) as caught:
            MODULE.requirements(MANIFEST, "https://example.invalid/whl")
        self.assertIn("no baseline entry installs from", str(caught.exception))

    def test_refresh_takes_real_pip_list_records(self):
        """A freeze would lose the direct-URL wheels and the bootstrap tools."""
        listing = json.dumps(
            [
                {"name": name, "version": record["version"]}
                for name, record in MANIFEST["packages"].items()
                if name not in ("sglang", "types-pyyaml", "flashinfer-cubin")
            ]
            + [
                {"name": "flashinfer_cubin", "version": "0.7.0.post1"},
                {"name": "nixl_cu13", "version": "1.4.0"},
                {"name": "pip", "version": "26.2.1"},
                {"name": "setuptools", "version": "84.0.0"},
                {"name": "torch", "version": "2.14.0+cu130"},
                {"name": "brand-new-thing", "version": "1.2.3"},
            ]
        )
        refreshed = MODULE.refresh(MANIFEST, listing)
        self.assertEqual(
            refreshed["packages"]["torch"],
            {**MANIFEST["packages"]["torch"], "version": "2.14.0+cu130"},
        )
        for name in ("nixl", "nixl-cu13", "flashinfer-cubin", "pip", "setuptools"):
            self.assertIn(name, refreshed["packages"], name)
        # Only the derived identity of the checkout is real, so a refresh keeps
        # the recorded provenance of the native source checkout.
        self.assertEqual(
            refreshed["packages"]["sglang"], MANIFEST["packages"]["sglang"]
        )
        self.assertEqual(
            refreshed["packages"]["types-pyyaml"], MANIFEST["packages"]["types-pyyaml"]
        )
        self.assertEqual(
            refreshed["packages"]["brand-new-thing"],
            {"version": "1.2.3", "class": "unclassified"},
        )
        self.assertEqual(refreshed["qualified"]["distributions"], 232)
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        handle.write(MODULE.dumps(refreshed))
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        with self.assertRaises(MODULE.ParityError) as caught:
            MODULE.load(Path(handle.name))
        self.assertIn("unclassified", str(caught.exception))
        # A freeze is the wrong input, and saying so beats parsing a URL.
        with self.assertRaises(MODULE.ParityError) as caught:
            MODULE.refresh(MANIFEST, "sglang @ file:///opt/sglang/src\nnixl==1.4.0\n")
        self.assertIn("pip list --format=json", str(caught.exception))

    def test_the_committed_baseline_survives_a_round_trip(self):
        """The file is data a maintainer reviews, not a formatted blob."""
        self.assertEqual(MODULE.dumps(MANIFEST), MODULE.MANIFEST_PATH.read_text())

    def test_the_commands_the_image_build_runs(self):
        def run(*arguments):
            return subprocess.run(
                [sys.executable, str(SOURCE), *arguments],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()

        self.assertEqual(run("indexes"), list(INDEXES))
        for index in INDEXES:
            self.assertEqual(
                run("requirements", "--index", index),
                MODULE.requirements(MANIFEST, index),
            )

    def test_the_recipe_gates_dependencies_before_it_compiles_anything(self):
        workdir = position("WORKDIR", "/opt/pennyroyal")
        checkout = position("RUN", "git -C /opt/pennyroyal init")
        install = position("RUN", '--index-url "$index" -r /tmp/penny-pins.txt')
        editable = position("RUN", "--no-deps --no-build-isolation-package sglang")
        packaging = position("RUN", "scripts/pennyroyal/flashinfer/install.py")
        nixl = position("RUN", "bash docker/pennyroyal/build-nixl.sh")
        self.assertTrue(workdir < checkout < install < editable < packaging < nixl)
        closure = position("RUN", "flashinfer/check_dependencies.py")
        parity = position("RUN", "dependency_parity.py check")
        final_closure = position("RUN", "flashinfer/check_dependencies.py", last=True)
        final_parity = position("RUN", "dependency_parity.py check", last=True)
        self.assertNotEqual(final_closure, closure)
        self.assertEqual(final_parity, nixl)
        # Both dependency gates run in one layer, the requirement check first, so a
        # missing dependency of an installed wheel costs one layer and not an hour
        # of nvcc; and they run again once the source builds have joined the set.
        self.assertEqual(closure, parity)
        payload = steps()[closure][1]
        self.assertLess(
            payload.index("flashinfer/check_dependencies.py"),
            payload.index("dependency_parity.py check"),
        )
        self.assertIn("--pending-source-build", payload)
        self.assertLess(closure, packaging)
        late = steps()[final_closure][1]
        self.assertEqual(final_closure, final_parity)
        for marker, follower in (
            ("tools/ple_nvme/install.sh", "flashinfer/check_dependencies.py"),
            ("flashinfer/check_dependencies.py", "dependency_parity.py check"),
            ("dependency_parity.py check", "-m pip freeze >"),
        ):
            self.assertLess(late.index(marker), late.index(follower), late)
        # NIXL is named as pending before its own build, and never after it.
        self.assertNotIn("--pending-source-build", late)
        install_payload = steps()[install][1]
        self.assertIn("dependency_parity.py indexes", install_payload)
        self.assertIn("--no-deps", install_payload)
        self.assertNotIn("--extra-index-url", install_payload)
        self.assertNotIn("constraints.txt", DOCKERFILE)
        self.assertEqual(DOCKERFILE.count("uv python install 3.12.13"), 1)
        self.assertIn("uv venv --python 3.12.13 /opt/pennyroyal/.venv", install_payload)
        self.assertIn(
            'PENNY_EXPECTED_SOURCE_REVISION="$SOURCE_REVISION" pennyroyal --check',
            DOCKERFILE,
        )

    def test_the_checkout_layer_creates_the_root_it_needs(self):
        """Run the recipe's own checkout command in a fresh empty root.

        The layer is copied out of the build stage verbatim, so this fails for the
        reason a fresh build would: `git init` on a directory nothing created.
        """
        root = Path(tempfile.mkdtemp())
        try:
            remote, revision = git_remote(root)
            body = steps()[position("RUN", "git -C /opt/pennyroyal init")][1].replace(
                "/opt/pennyroyal", str(root / "opt/pennyroyal")
            )
            env = {
                **os.environ,
                "SOURCE_REVISION": revision,
                "SOURCE_REPOSITORY": remote,
            }

            def run():
                return subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", body],
                    env=env,
                    capture_output=True,
                    text=True,
                )

            # Nothing else in the image creates the directory, so an absent root is
            # the failure this layer exists to avoid.
            failed = run()
            self.assertEqual(failed.returncode, 128, failed.stderr)
            self.assertIn("No such file or directory", failed.stderr)
            # What the recipe actually does: WORKDIR precedes it and makes the path.
            self.assertLess(
                position("WORKDIR", "/opt/pennyroyal"),
                position("RUN", "git -C /opt/pennyroyal init"),
            )
            (root / "opt" / "pennyroyal").mkdir(parents=True)
            succeeded = run()
            self.assertEqual(succeeded.returncode, 0, succeeded.stderr)
            self.assertEqual(
                subprocess.check_output(
                    ["git", "-C", str(root / "opt/pennyroyal"), "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
                revision,
            )
            self.assertTrue((root / "opt/pennyroyal/README.md").is_file())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_the_baseline_itself_still_says_what_the_release_agreed(self):
        self.assertEqual(QUALIFIED["python"], "3.12.13")
        self.assertEqual(QUALIFIED["platform"], "linux-x86_64")
        self.assertEqual(QUALIFIED["distributions"], 231)
        self.assertEqual(
            MANIFEST["packages"]["sglang-kernel"]["version"], "0.4.6.post1"
        )
        self.assertEqual(MANIFEST["packages"]["torch"]["version"], "2.13.0+cu130")
        self.assertEqual(MANIFEST["packages"]["nixl"]["class"], "source-build")
        self.assertEqual(MANIFEST["packages"]["sglang"]["class"], "source-editable")
        self.assertEqual(
            MANIFEST["packages"]["pyelftools"].get("class", "runtime"), "runtime"
        )
        self.assertEqual(
            MANIFEST["packages"]["flashinfer-cubin"]["origin"],
            "https://github.com/flashinfer-ai/flashinfer/releases/download/"
            "v0.7.0.post1/flashinfer_cubin-0.7.0.post1-py3-none-any.whl",
        )
        self.assertEqual(MODULE.ACCEPTED_HUMMING_INSTALLED, "0.1.10")
        self.assertEqual(
            MANIFEST["packages"]["humming-kernels"]["version"],
            MODULE.ACCEPTED_HUMMING_INSTALLED,
        )


if __name__ == "__main__":
    unittest.main()
