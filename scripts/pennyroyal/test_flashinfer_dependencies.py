import importlib.util
import subprocess
import unittest
from pathlib import Path

SOURCE = Path(__file__).parent / "flashinfer" / "check_dependencies.py"
SPEC = importlib.util.spec_from_file_location("dependency_check", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DependencyAuditTests(unittest.TestCase):
    def result(self, output, code=1, error=""):
        return subprocess.CompletedProcess([], code, output, error)

    def test_known_omitted_cache_wheels_are_allowed(self):
        lines = [
            f"flashinfer-jit-cache 0.7.0.post1+cu130 requires "
            f"flashinfer-jit-cache-{arch}, which is not installed."
            for arch in ("sm100a", "sm103a", "sm80", "sm89", "sm90a")
        ]
        self.assertTrue(MODULE.only_omitted_providers(self.result("\n".join(lines))))

    def test_required_provider_and_other_failures_are_not_hidden(self):
        known = (
            "flashinfer-jit-cache 0.7.0.post1+cu130 requires "
            "flashinfer-jit-cache-sm100a, which is not installed."
        )
        failures = (
            known.replace("sm100a", "sm120f"),
            known.replace("0.7.0.post1", "0.7.1"),
            "sglang requires torch, which is not installed.",
            "flashinfer-jit-cache 0.7.0.post1+cu130 has requirement "
            "flashinfer-jit-cache-sm100a==0.7.0.post1+cu130, but you have "
            "flashinfer-jit-cache-sm100a 0.6.17+cu130.",
        )
        for failure in failures:
            with self.subTest(failure=failure):
                self.assertFalse(
                    MODULE.only_omitted_providers(self.result(known + "\n" + failure))
                )
        for result in (
            self.result(""),
            self.result(known, code=2),
            self.result(known, error="pip failed"),
        ):
            self.assertFalse(MODULE.only_omitted_providers(result))


if __name__ == "__main__":
    unittest.main()
