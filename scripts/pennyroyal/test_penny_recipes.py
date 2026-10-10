"""CPU-only checks that the approved 3.0 adaptive MTP default reaches every
public Flash-Next launch path and the generated container setup.

The runtime implements adaptive speculative decoding behind the existing
--speculative-adaptive flags; these checks propagate nothing themselves. They
pin what the shipped recipes and configurator must say, so a launch file that
still selects the old static steps=3/draft_tokens=4 profile, a cache namespace
that cannot see the adaptive policy, or a stale default image fails here
before an ordinary user discovers it at a server boot.
"""

import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "pennyroyal"
LAUNCH = ROOT / "docker" / "pennyroyal" / "launch"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import penny_config as pc  # noqa: E402

# The qualified approved policy; the shipped file and every recipe checksum
# check must carry exactly this identity.
ADAPTIVE_NEXT_SHA256 = "64070286310a0d3eeeb3ff3ae074d1ca0fed2937d2274a0adfcb8ace84bc01ee"
RELEASED_IMAGE = "ghcr.io/jpezzulli/sglang-rtxpro6000:v3.0.0"
TOKEN_MAP_SHA256 = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"

NEXT_LAUNCHERS = (
    CONFIGS / "serve-flash-next-frspec.sh",
    CONFIGS / "serve-flash-next.sh",
    LAUNCH / "config" / "start-flash-next-frspec.sh",
    LAUNCH / "config" / "start-flash-next.sh",
)
FRSPEC_LAUNCHERS = (
    CONFIGS / "serve-flash-next-frspec.sh",
    LAUNCH / "config" / "start-flash-next-frspec.sh",
)
PLAIN_LAUNCHERS = (
    CONFIGS / "serve-flash-next.sh",
    LAUNCH / "config" / "start-flash-next.sh",
)


class AdaptiveLaunchRecipeTest(unittest.TestCase):
    def text(self, path: Path) -> str:
        self.assertTrue(path.is_file(), f"launch recipe missing: {path}")
        return path.read_text()

    def test_the_shipped_policy_is_the_approved_file(self):
        policy = CONFIGS / "adaptive-next.json"
        digest = hashlib.sha256(policy.read_bytes()).hexdigest()
        self.assertEqual(digest, ADAPTIVE_NEXT_SHA256)

    def test_every_next_launcher_adaptive_widths_the_launch(self):
        for launcher in NEXT_LAUNCHERS:
            text = self.text(launcher)
            with self.subTest(launcher=str(launcher)):
                # Launch at the widest candidate (W8 at topk=1) so no
                # adaptive width can exceed the captured graphs.
                self.assertIn("--speculative-num-steps 7", text)
                self.assertIn("--speculative-eagle-topk 1", text)
                self.assertIn("--speculative-num-draft-tokens 8", text)
                self.assertIn("--speculative-adaptive", text)
                self.assertIn("--speculative-adaptive-config", text)
                # The old static 3/4 profile is gone from the launch line.
                self.assertNotIn("--speculative-num-steps 3", text)
                self.assertNotIn("--speculative-num-draft-tokens 4", text)

    def test_every_next_launcher_pins_the_policy_in_the_namespace(self):
        for launcher in NEXT_LAUNCHERS:
            text = self.text(launcher)
            with self.subTest(launcher=str(launcher)):
                # The runtime policy file is an image/repo asset, never a
                # copy of the host-mounted config directory.
                self.assertRegex(text, r"adaptive-next\.json")
                self.assertIn(ADAPTIVE_NEXT_SHA256, text)
                self.assertIn('--field "speculative_adaptive=true"', text)
                self.assertRegex(
                    text, r'--field "speculative_adaptive[a-z_]*_sha256=\$ADAPTIVE'
                )
                # Namespace fields follow the launch line, so an incompatible
                # old namespace can never alias the adaptive one.
                self.assertIn('--field "speculative_num_steps=7"', text)
                self.assertIn('--field "speculative_num_draft_tokens=8"', text)
                self.assertNotIn('--field "speculative_num_steps=3"', text)
                self.assertNotIn('--field "speculative_num_draft_tokens=4"', text)

    def test_frspec_keeps_the_map_and_tokenizer_checks(self):
        for launcher in FRSPEC_LAUNCHERS:
            text = self.text(launcher)
            with self.subTest(launcher=str(launcher)):
                self.assertIn("--speculative-token-map", text)
                self.assertIn(TOKEN_MAP_SHA256, text)
                self.assertIn("tokenizer.json", text)

    def test_the_non_frspec_alternative_stays_explicit(self):
        for launcher in PLAIN_LAUNCHERS:
            text = self.text(launcher)
            with self.subTest(launcher=str(launcher)):
                self.assertNotIn("--speculative-token-map", text)


class ContainerDefaultsTest(unittest.TestCase):
    def test_the_default_image_targets_the_current_release(self):
        self.assertEqual(pc.DEFAULT_IMAGE, RELEASED_IMAGE)
        # Every public container entry point agrees on the release: the
        # manual launcher and the manual Compose pair (a v2.5.3 image under
        # the new startup files would lack the approved adaptive default).
        for path in (
            LAUNCH / "run.sh",
            ROOT / "docker" / "pennyroyal" / "compose.yaml",
            ROOT / "docker" / "pennyroyal" / ".env.example",
        ):
            with self.subTest(path=str(path)):
                text = self.text(path)
                self.assertIn("ghcr.io/jpezzulli/sglang-rtxpro6000:v3.0.0", text)
                self.assertNotIn("v2.5.3", text)

    def test_generated_run_sh_carries_the_default_image(self):
        source = self.text(ROOT / "scripts" / "pennyroyal" / "penny_config.py")
        self.assertRegex(
            source,
            r'plan\.env\.get\("PENNYROYAL_IMAGE", DEFAULT_IMAGE\)',
        )

    def text(self, path: Path) -> str:
        self.assertTrue(path.is_file(), f"container file missing: {path}")
        return path.read_text()


if __name__ == "__main__":
    unittest.main()
