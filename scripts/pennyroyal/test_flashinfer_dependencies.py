import importlib.util
import subprocess
import unittest
from pathlib import Path

SOURCE = Path(__file__).parent / "flashinfer" / "check_dependencies.py"
SPEC = importlib.util.spec_from_file_location("dependency_check", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

# The two findings a qualified baseline explains, spelled the way pip 26.2.1
# writes them: the cache shim's providers for architectures this GPU family
# cannot load, and the deferred Humming upgrade. In an image the same Humming
# finding names the version the checkout derives, so that is tested too.
OMITTED = (
    "flashinfer-jit-cache 0.7.0.post1+cu130 requires "
    "flashinfer-jit-cache-{arch}, which is not installed."
)
HUMMING = (
    "sglang 0.5.19.dev980+gfe35e1d81 has requirement "
    "humming-kernels[cu13]==0.1.12, but you have humming-kernels 0.1.10."
)
HUMMING_IMAGE = HUMMING.replace("0.5.19.dev980+gfe35e1d81", "0.5.19+gc6a033ad5fcb")


class DependencyAuditTests(unittest.TestCase):
    def result(self, output, code=1, error=""):
        return subprocess.CompletedProcess([], code, output, error)

    def known(self, arch="sm100a"):
        return OMITTED.format(arch=arch)

    def test_known_omitted_cache_wheels_are_allowed(self):
        report = "\n".join(
            OMITTED.format(arch=arch)
            for arch in ("sm100a", "sm103a", "sm80", "sm89", "sm90a")
        )
        self.assertTrue(MODULE.only_accepted_findings(self.result(report)))
        self.assertTrue(
            MODULE.only_accepted_findings(self.result(report + "\n" + HUMMING))
        )

    def test_deferred_humming_pin_is_allowed(self):
        for finding in (HUMMING, HUMMING_IMAGE):
            with self.subTest(finding=finding):
                self.assertTrue(MODULE.only_accepted_findings(self.result(finding)))

    def test_deferred_humming_pin_stays_the_recorded_pair(self):
        """Only 0.1.12 requested of 0.1.10 installed; nothing else is forgiven."""
        for finding in (
            HUMMING.replace("0.1.12", "0.1.13"),
            HUMMING.replace("0.1.10.", "0.1.9."),
            HUMMING.replace("has requirement", "requires"),
            HUMMING.replace("humming-kernels[cu13]==", "humming-kernels=="),
            HUMMING.replace("sglang ", "sglang-kernel "),
        ):
            with self.subTest(finding=finding):
                self.assertFalse(MODULE.only_accepted_findings(self.result(finding)))
        # A baseline whose Humming has moved cannot be waived by the old pair.
        moved = {
            "packages": {
                "flashinfer-jit-cache": {"version": "0.7.0.post1+cu130"},
                "humming-kernels": {"version": "0.1.11"},
            }
        }
        self.assertFalse(MODULE.parity().accepted_pip_check_output(HUMMING, "", moved))

    def test_both_accepted_findings_together_stay_accepted(self):
        self.assertTrue(
            MODULE.only_accepted_findings(
                self.result("\n".join((self.known(), HUMMING, HUMMING_IMAGE)))
            )
        )

    def test_a_missing_dependency_of_an_installed_wheel_is_never_forgiven(self):
        """pyelftools stays required: the accepted list does not grow to fit."""
        for finding in (
            "humming-kernels 0.1.10 requires pyelftools, which is not installed.",
            "sglang 0.5.19+gc6a033ad5fcb has requirement humming-kernels[cu13]"
            "==0.1.12, but you have humming-kernels 0.1.9.",
            "flashinfer-python 0.7.0.post1 has requirement apache-tvm-ffi<0.2,"
            ">=0.1.11, but you have apache-tvm-ffi 0.2.0.",
            "sglang requires torch, which is not installed.",
            self.known("sm120f"),
            self.known().replace("0.7.0.post1", "0.7.1"),
            "flashinfer-jit-cache 0.7.0.post1+cu130 has requirement "
            "flashinfer-jit-cache-sm100a==0.7.0.post1+cu130, but you have "
            "flashinfer-jit-cache-sm100a 0.6.17+cu130.",
        ):
            with self.subTest(finding=finding):
                self.assertFalse(
                    MODULE.only_accepted_findings(
                        self.result("\n".join((self.known(), finding)))
                    )
                )

    def test_the_wrapper_needs_the_findings_and_the_exit_code_it_claims(self):
        for result in (
            self.result(""),
            self.result(self.known(), code=2),
            self.result(self.known(), code=0),
            self.result(self.known(), error="pip failed"),
            self.result(f"{self.known()}\n\nsome noise\n"),
        ):
            self.assertFalse(MODULE.only_accepted_findings(result))


if __name__ == "__main__":
    unittest.main()
