"""CPU-only checks that the approved fixed Next width policy reaches every
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

# The qualified tiered policy (the unchanged 3.1.1 default); every Next launcher
# reaches it through configs/pennyroyal/mtp-policy.sh, which carries the approved
# identity of each selectable policy.
ADAPTIVE_NEXT_SHA256 = (
    "3edaa87d66f535caae0bddb8f457f03fce0f6dc1f4e8091b20495a0dc9da41fb"
)
# The v3.0.2 acceptance-adaptive policy, restored byte-for-byte from that tag as
# the `adaptive` choice rather than re-typed.
ADAPTIVE_NEXT_V302_SHA256 = (
    "658158c29d2de8d2ee9c58ea8e7ac68a2dc70dd1ac1c46642e0a4302b798728f"
)
# The narrowest policy: W4 at every concurrency, launched at steps 3.
ADAPTIVE_NEXT_FIXED_W4_SHA256 = (
    "51644008674cd048adda2a38355a8302a032ce31c7f7d9dd9319b5cd4c3db3cd"
)
RELEASED_IMAGE = "ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.4"
TOKEN_MAP_SHA256 = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"
# The approved fixed width-routing policy for the Flash-Next DFlash2 profile:
# one candidate per tier, block 6 at batch 1 and block 4 at batch 2 or more.
ADAPTIVE_DFLASH2_SHA256 = (
    "596024aff550d363a8b25ec5b7b6f5bd9e375f5856c53e341fd4f186b791ae1a"
)

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
                # One setting picks the policy file AND the launch width that can
                # carry it, so the pair cannot drift apart between a native
                # recipe and its mounted copy: both read them from the selector.
                self.assertRegex(text, r'source ".*/mtp-policy\.sh"')
                self.assertIn('--speculative-num-steps "$PENNY_MTP_SPEC_STEPS"', text)
                self.assertIn("--speculative-eagle-topk 1", text)
                self.assertIn(
                    '--speculative-num-draft-tokens "$PENNY_MTP_DRAFT_TOKENS"', text
                )
                self.assertIn("--speculative-adaptive", text)
                self.assertIn('--speculative-adaptive-config "$ADAPTIVE_CONFIG"', text)
                # No launcher states a width of its own any more, and none keeps
                # the old static 3/4 launch line: the policy file does that now.
                self.assertNotIn("--speculative-num-steps 3", text)
                self.assertNotIn("--speculative-num-draft-tokens 4", text)

    def test_the_three_shipped_policies_keep_their_exact_bytes(self):
        # The shared helper is the one copy of the selection table, and the
        # digests below record the shipped content of each policy file: the
        # current default and the asset restored from the v3.0.2 tag byte for
        # byte. They are a review record, not a boot gate (see
        # test_an_edited_policy_changes_identity_not_the_boot_decision).
        import json

        approved = {
            "adaptive-next.json": ADAPTIVE_NEXT_SHA256,
            "adaptive-next-v3.0.2.json": ADAPTIVE_NEXT_V302_SHA256,
            "adaptive-next-fixed-w4.json": ADAPTIVE_NEXT_FIXED_W4_SHA256,
        }
        for asset, digest in approved.items():
            with self.subTest(asset=asset):
                data = json.loads((CONFIGS / asset).read_text())
                # Keys are batch-size floors; no adaptation knobs beyond them may
                # appear in a policy that is described as fixed routing.
                self.assertEqual(
                    sorted(key for key in data if not key.startswith("_")), ["1", "2"]
                )
                self.assertEqual(
                    hashlib.sha256((CONFIGS / asset).read_bytes()).hexdigest(), digest
                )
        self.assertEqual(
            json.loads((CONFIGS / "adaptive-next.json").read_text())["1"][
                "candidate_steps"
            ],
            [7],
        )
        self.assertEqual(
            json.loads((CONFIGS / "adaptive-next.json").read_text())["2"][
                "candidate_steps"
            ],
            [3],
        )
        v302 = json.loads((CONFIGS / "adaptive-next-v3.0.2.json").read_text())
        # The 3.0.2 policy is the EMA one: two C1 candidates plus its hold and
        # backoff, unchanged, and C2+ still W4.
        self.assertEqual(v302["1"]["candidate_steps"], [3, 7])
        self.assertEqual(v302["1"]["switch_grace_batches"], 40)
        self.assertEqual(v302["1"]["max_grace_batches"], 80)
        self.assertEqual(v302["2"]["candidate_steps"], [3])
        fixed = json.loads((CONFIGS / "adaptive-next-fixed-w4.json").read_text())
        self.assertEqual(fixed["1"]["candidate_steps"], [3])
        self.assertEqual(fixed["2"]["candidate_steps"], [3])

    def test_the_selector_and_the_configurator_agree(self):
        # penny_config describes the same table for --show-config, the wizard and
        # the generated launch files. Run the helper for real, in bash, and
        # compare: a copy that drifted would show a policy the launch never runs.
        import re
        import subprocess

        helper = CONFIGS / "mtp-policy.sh"
        self.assertTrue(helper.is_file())
        for name in ("tiered", "adaptive", "fixed-w4", ""):
            with self.subTest(policy=name):
                script = (
                    f'export PENNY_MTP_POLICY="{name}"'
                    if name
                    else "unset PENNY_MTP_POLICY"
                )
                out = subprocess.run(
                    [
                        "bash",
                        "-c",
                        f'{script}; source "{helper}" && printf "%s %s %s %s\\n" '
                        '"$PENNY_MTP_POLICY" "$PENNY_MTP_POLICY_FILE" '
                        '"$PENNY_MTP_SPEC_STEPS" "$PENNY_MTP_DRAFT_TOKENS"',
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.split()
                chosen = name or "tiered"
                asset, steps, draft_tokens = pc.MTP_POLICY_SPEC[chosen]
                self.assertEqual(out, [chosen, asset, steps, draft_tokens])
        refused = subprocess.run(
            ["bash", "-c", f'source "{helper}"'],
            env={"PENNY_MTP_POLICY": "w6", "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn(
            "PENNY_MTP_POLICY must be tiered, adaptive or fixed-w4", refused.stderr
        )
        # And the table is complete: nothing in the helper that the configurator
        # does not know about, or the reverse.
        named = set(re.findall(r"PENNY_MTP_POLICY_FILE=(\S+)", helper.read_text()))
        self.assertEqual(named, {spec[0] for spec in pc.MTP_POLICY_SPEC.values()})

    def test_every_next_launcher_pins_the_policy_in_the_namespace(self):
        for launcher in NEXT_LAUNCHERS:
            text = self.text(launcher)
            with self.subTest(launcher=str(launcher)):
                # The runtime policy file is an image/repo asset, never a
                # copy of the host-mounted config directory.
                self.assertRegex(
                    text, r'source ".*(IMAGE_CONFIGS|SCRIPT_DIR)/mtp-policy\.sh"'
                )
                self.assertIn('--field "speculative_adaptive=true"', text)
                self.assertRegex(
                    text, r'--field "speculative_adaptive[a-z_]*_sha256=\$ADAPTIVE'
                )
                # Namespace fields follow the launch line, so an incompatible
                # old namespace can never alias the adaptive one. The approved
                # identities themselves live in the shared selector, once.
                self.assertIn(
                    '--field "speculative_num_steps=$PENNY_MTP_SPEC_STEPS"', text
                )
                self.assertIn(
                    '--field "speculative_num_draft_tokens=$PENNY_MTP_DRAFT_TOKENS"',
                    text,
                )
                # The hash is namespace identity and nothing else: no launcher and
                # not the shared selector carries a POLICY checksum to refuse a
                # boot with (3.1.1 compared one hard-coded hash, which could not
                # survive a second supported policy). The FR-Spec map and
                # tokenizer guards are a different matter and stay.
                text_of = {
                    "launcher": text,
                    "selector": self.text(CONFIGS / "mtp-policy.sh"),
                }
                for label, body in text_of.items():
                    for digest in (
                        ADAPTIVE_NEXT_SHA256,
                        ADAPTIVE_NEXT_V302_SHA256,
                        ADAPTIVE_NEXT_FIXED_W4_SHA256,
                    ):
                        self.assertNotIn(digest, body, (launcher, label))
                    self.assertNotIn('[[ "$ADAPTIVE_SHA" ==', body, (launcher, label))

    def test_an_edited_policy_changes_identity_not_the_boot_decision(self):
        # A harmless edit to a bundled policy (a different comment) must not stop
        # a launch: the file's hash is what separates cache identities, so the
        # edit shows up as a fresh namespace instead of a checksum refusal.
        import json
        import shutil
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            assets = Path(temp) / "configs"
            shutil.copytree(CONFIGS, assets)
            policy = assets / "adaptive-next.json"
            before = hashlib.sha256(policy.read_bytes()).hexdigest()
            data = json.loads(policy.read_text())
            data["_comment"] = ["an operator note"]
            policy.write_text(json.dumps(data, indent=2) + "\n")
            self.assertNotEqual(hashlib.sha256(policy.read_bytes()).hexdigest(), before)

            def resolve(policy_name: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$1" && printf "%s %s %s %s\\n" '
                        '"$ADAPTIVE_CONFIG" "$PENNY_MTP_POLICY" '
                        '"$ADAPTIVE_SHA" "$PENNY_MTP_SPEC_STEPS"',
                        "bash",
                        str(assets / "mtp-policy.sh"),
                    ],
                    capture_output=True,
                    text=True,
                    env={"PENNY_MTP_POLICY": policy_name, "PATH": "/usr/bin:/bin"},
                )

            edited = resolve("tiered")
            self.assertEqual(edited.returncode, 0, edited.stderr)
            config, name, digest, steps = edited.stdout.split()
            self.assertEqual(name, "tiered")
            self.assertEqual(steps, "7")
            self.assertEqual(Path(config), assets / "adaptive-next.json")
            self.assertEqual(digest, hashlib.sha256(policy.read_bytes()).hexdigest())
            self.assertNotEqual(digest, ADAPTIVE_NEXT_SHA256)
            # The other selections still resolve from the same edited copy, and a
            # missing policy file is still named rather than guessed.
            (assets / "adaptive-next-fixed-w4.json").unlink()
            refused = resolve("fixed-w4")
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("MTP width policy is missing or unreadable", refused.stderr)

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
                # One literal on purpose: the entry points must not drift apart.
                self.assertIn(RELEASED_IMAGE, text)
                self.assertNotIn("v2.5.3", text)

    def test_the_release_asset_names_the_same_release(self):
        # The startup banner reads the version asset and the launch files read
        # the image default: one release line, or a v3.0.x image boots under a
        # banner naming something else.
        version = (ROOT / "configs" / "pennyroyal" / "pennyroyal.version").read_text()
        self.assertEqual(version.strip(), RELEASED_IMAGE.rsplit(":v", 1)[1])

    def test_generated_run_sh_carries_the_default_image(self):
        source = self.text(ROOT / "scripts" / "pennyroyal" / "penny_config.py")
        self.assertRegex(
            source,
            r'plan\.env\.get\("PENNYROYAL_IMAGE", DEFAULT_IMAGE\)',
        )

    def text(self, path: Path) -> str:
        self.assertTrue(path.is_file(), f"container file missing: {path}")
        return path.read_text()


class DFlash2RoutedWidthRecipeTest(unittest.TestCase):
    """The Flash-Next DFlash2 launch paths carry the qualified routed profile:
    the FR-Spec map, the fixed width policy, the widest routed launch width, and
    the graph settings that routing needs. Static text on purpose — the argv-
    level checks live in test_flash_next_dflash2_profile.py, which really runs
    the launch blocks."""

    DFLASH2_LAUNCHERS = (
        CONFIGS / "serve-flash-next-dflash2.sh",
        LAUNCH / "config" / "start-flash-next-dflash2.sh",
    )

    def text(self, path: Path) -> str:
        self.assertTrue(path.is_file(), f"launch recipe missing: {path}")
        return path.read_text()

    def test_the_shipped_routing_policy_is_the_approved_file(self):
        policy = CONFIGS / "adaptive-dflash2.json"
        self.assertEqual(
            hashlib.sha256(policy.read_bytes()).hexdigest(), ADAPTIVE_DFLASH2_SHA256
        )
        import json

        tiers = json.loads(policy.read_text())
        self.assertEqual(tiers["1"]["candidate_steps"], [5])
        self.assertEqual(tiers["2"]["candidate_steps"], [3])

    def test_every_dflash2_launcher_routes_widths_and_carries_the_map(self):
        for launcher in self.DFLASH2_LAUNCHERS:
            text = self.text(launcher)
            with self.subTest(launcher=str(launcher)):
                # Launch at the widest routed width (block 6 at topk 1) so no
                # routed width exceeds the captured allocations.
                self.assertIn("SPEC_STEPS=5", text)
                self.assertIn("DRAFT_TOKENS=6", text)
                self.assertIn("--speculative-num-steps", text)
                self.assertIn("--speculative-adaptive", text)
                self.assertIn("adaptive-dflash2.json", text)
                self.assertIn("--speculative-token-map", text)
                self.assertIn("frspec/flash-next-64k.pt", text)
                # Prefill graphs are off; the decode bucket list is left to the
                # runtime, which captures the single-request bucket alongside the
                # rest of the reachable batches. Forcing a one-bucket list here
                # would take the graphs away from the C2+ widths.
                self.assertIn("--disable-prefill-cuda-graph", text)
                self.assertNotIn("--cuda-graph-bs", text)
                # The routed width is not an operator knob any more.
                self.assertNotIn("SPECULATIVE_NUM_DRAFT_TOKENS", text)

    def test_every_dflash2_launcher_names_routing_in_the_namespace(self):
        for launcher in self.DFLASH2_LAUNCHERS:
            text = self.text(launcher)
            with self.subTest(launcher=str(launcher)):
                # Identity follows the computation: the map and the policy are
                # hashed in, so neither can be swapped while reusing the cache.
                self.assertIn('--field "speculative_adaptive=true"', text)
                self.assertIn(
                    '--field "speculative_adaptive_policy_sha256=$ADAPTIVE', text
                )
                self.assertRegex(text, r'--field "draft_token_map_sha256=\$TOKEN_MAP')
                self.assertIn('--field "speculative_num_steps=$SPEC_STEPS"', text)
                self.assertIn('--field "prefill_cuda_graph=disabled"', text)
                # No checksum gate: the hash is identity, not a boot refusal.
                self.assertNotIn(f"== {ADAPTIVE_DFLASH2_SHA256}", text)
                self.assertNotIn(f"== {TOKEN_MAP_SHA256}", text)

    def test_the_routed_recipes_keep_the_runtime_graph_bucket_list(self):
        # Focused regression check: neither public DFlash2 path may narrow decode
        # graph capture. A --cuda-graph-bs-decode list here would reduce the
        # captured buckets to the ones named, and the W4 tier at C2+ would boot
        # with its graphs disabled by the runtime.
        for launcher in self.DFLASH2_LAUNCHERS:
            with self.subTest(launcher=str(launcher)):
                self.assertNotIn("--cuda-graph-bs-decode", self.text(launcher))
                self.assertNotIn('--field "decode_graph_bs', self.text(launcher))

    def test_the_maintained_mtp_recipes_are_untouched_by_this_profile(self):
        # Maintained is not rewritten: the native-MTP launchers keep their own
        # static NEXTN widths, policy and memory fraction, and none of them
        # picks up the routed DFlash2 policy file.
        for launcher in NEXT_LAUNCHERS:
            text = self.text(launcher)
            with self.subTest(launcher=str(launcher)):
                self.assertIn("--speculative-algorithm NEXTN", text)
                self.assertIn('--speculative-num-steps "$PENNY_MTP_SPEC_STEPS"', text)
                self.assertRegex(
                    text, r'source ".*(SCRIPT_DIR|IMAGE_CONFIGS)/mtp-policy\.sh"'
                )
                self.assertIn('"$ADAPTIVE_CONFIG"', text)
                self.assertNotIn("adaptive-dflash2.json", text)
                # The retained native fraction: 0.981 on this revision, and the
                # measured 0.97 in the retained-MTP configuration update.
                self.assertRegex(text, r"--mem-fraction-static 0\.(981|97)\b")
                self.assertNotIn("--cuda-graph-bs", text)
        for launcher in self.DFLASH2_LAUNCHERS:
            self.assertIn("--mem-fraction-static 0.95", self.text(launcher))
            # The DFlash2 recipes route their own width and never read the
            # native selector, so its launch line and policy file are unchanged.
            self.assertNotIn("mtp-policy.sh", self.text(launcher))
            self.assertIn('--speculative-num-steps "$SPEC_STEPS"', self.text(launcher))


if __name__ == "__main__":
    unittest.main()
