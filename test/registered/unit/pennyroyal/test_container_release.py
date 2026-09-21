"""CPU-only release routing and image-source checks; no registry or GPU access."""

import importlib.util
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
WORKFLOW = REPO / ".github/workflows/pennyroyal-container.yml"
REVISION = "a" * 40


def workflow():
    # BaseLoader preserves GitHub's `on` key instead of treating it as bool.
    return yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def route(tmp_path, **overrides):
    step = next(
        step
        for step in workflow()["jobs"]["image"]["steps"]
        if step.get("id") == "image_meta"
    )
    output = tmp_path / "outputs"
    env = {
        **os.environ,
        "EVENT_NAME": "workflow_dispatch",
        "RELEASE_TAG": "",
        "REF_TYPE": "branch",
        "REF_NAME": "pennyroyal-main-sm120-final",
        "PROMOTE_DIGEST": "",
        "MANUAL_IMAGE_TAG": "",
        "GITHUB_OUTPUT": str(output),
        **overrides,
    }
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    values = (
        dict(line.split("=", 1) for line in output.read_text().splitlines())
        if output.exists()
        else {}
    )
    return result, values


def test_published_release_trigger_and_manual_fallback():
    data = workflow()
    assert data["on"]["release"]["types"] == ["published"]
    assert "workflow_dispatch" in data["on"]
    assert "tags" not in data["on"].get("push", {})
    assert not data["on"]["workflow_dispatch"]["inputs"]["image_tag"].get("default")


@pytest.mark.parametrize(
    "tag",
    ["pennyroyal-v2.5.1", "v2.5.1", "pennyroyal-v2.5.1-rc.1", "pennyroyal-v2.3.1.1"],
)
def test_release_build_selects_matching_version(tmp_path, tag):
    result, values = route(tmp_path, EVENT_NAME="release", RELEASE_TAG=tag)
    assert result.returncode == 0, result.stderr
    assert values["image_tag"] == tag.removeprefix("pennyroyal-")
    assert (
        values["revision"]
        == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    )


def test_manual_tag_retry_builds_release_but_branch_build_does_not(tmp_path):
    result, values = route(tmp_path, REF_TYPE="tag", REF_NAME="pennyroyal-v2.5.1")
    assert result.returncode == 0, result.stderr
    assert values["image_tag"] == "v2.5.1"
    result, values = route(tmp_path)
    assert result.returncode == 0, result.stderr
    assert values["image_tag"] == ""


@pytest.mark.parametrize(
    "tag",
    [
        "latest",
        "pennyroyal-main-sm120-final",
        "v2.5",
        "v2.5.1\nextra=bad",
        "v2.5.1;echo bad",
        "v2.5.1-" + "x" * 128,
    ],
)
def test_invalid_release_names_fail_before_build(tmp_path, tag):
    result, _ = route(tmp_path, EVENT_NAME="release", RELEASE_TAG=tag)
    assert result.returncode != 0


def test_explicit_digest_promotion_is_retained(tmp_path):
    result, values = route(
        tmp_path, PROMOTE_DIGEST="sha256:" + "b" * 64, MANUAL_IMAGE_TAG="v2.5.1"
    )
    assert result.returncode == 0, result.stderr
    assert values["image_tag"] == "v2.5.1"


@pytest.mark.parametrize(
    "digest,tag", [("sha256:bad", "v2.5.1"), ("sha256:" + "b" * 64, ""), ("", "v2.5.1")]
)
def test_ambiguous_or_invalid_manual_inputs_fail(tmp_path, digest, tag):
    result, _ = route(tmp_path, PROMOTE_DIGEST=digest, MANUAL_IMAGE_TAG=tag)
    assert result.returncode != 0


def test_only_successful_build_or_explicit_digest_can_set_version_tag():
    steps = workflow()["jobs"]["image"]["steps"]
    build = next(step for step in steps if step.get("id") == "build")
    publish = next(step for step in steps if step.get("id") == "publish")
    assert steps.index(build) < steps.index(publish)
    assert publish["if"] == "steps.image_meta.outputs.image_tag != ''"
    assert "always()" not in publish["if"]
    assert (
        publish["env"]["DIGEST"]
        == "${{ inputs.promote_digest || steps.build.outputs.digest }}"
    )
    assert "steps.image_meta.outputs.revision" in build["with"]["build-args"]
    assert ":latest" not in WORKFLOW.read_text()


@pytest.fixture
def image_check(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "penny_image_check", REPO / "docker/pennyroyal/check_install.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda cmd, **kwargs: REVISION + "\n" if "rev-parse" in cmd else "",
    )
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(
            origin=str(tmp_path / "python/sglang/__init__.py")
        ),
    )
    return module, tmp_path


def test_image_source_matches_requested_commit(image_check):
    module, root = image_check
    assert module.check_source(root, REVISION) == REVISION


def test_image_source_rejects_wrong_commit(image_check):
    module, root = image_check
    with pytest.raises(RuntimeError, match="revision"):
        module.check_source(root, "b" * 40)


def test_image_source_rejects_import_from_another_checkout(image_check, monkeypatch):
    module, root = image_check
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin="/elsewhere/sglang/__init__.py"),
    )
    with pytest.raises(RuntimeError, match="import"):
        module.check_source(root, REVISION)


def test_image_source_rejects_tracked_modifications(image_check, monkeypatch):
    module, root = image_check
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda cmd, **kwargs: (
            REVISION if "rev-parse" in cmd else " M python/sglang/__init__.py"
        ),
    )
    with pytest.raises(RuntimeError, match="modified tracked"):
        module.check_source(root, REVISION)


def test_docker_build_runs_cpu_checks_with_expected_revision():
    dockerfile = (REPO / "docker/pennyroyal/Dockerfile").read_text()
    assert (
        'PENNY_EXPECTED_SOURCE_REVISION="$SOURCE_REVISION" pennyroyal --check'
        in dockerfile
    )
    assert "&& pennyroyal --help" in dockerfile
