# Penny Royal 3.1.2

This patch release gives the maintained native-MTP profiles one public
width-policy setting, fixes DFlash2 decode-checkpoint tracking, stops the
scheduler's idle sleep from delaying encoder-embedding handoffs, and labels
both DFlash2 profiles experimental / alpha. Model, profile and performance
defaults are unchanged from 3.1.1: the `tiered` width routing still launches
exactly what it launched before.

## Fixes and changes

- **One width-policy setting.** The native-MTP launch recipes hard-coded one
  policy file and a checksum that refused anything else, so returning to the
  3.0.2 acceptance-adaptive widths meant editing launch lines. `PENNY_MTP_POLICY`
  is now the single setting that selects both the bundled policy file and the
  launch width that can carry it: `tiered` (the unchanged default — block 8 at
  one running request, block 4 at two or more), `adaptive` (the v3.0.2 policy
  restored byte for byte), or `fixed-w4` (block 4 at every concurrency).
  Unset keeps the default, so an existing launch is untouched. The checksum
  gate is gone: the selected policy file's hash is now NIXL cache identity,
  not a boot refusal. The setting does not apply to the DFlash2 and 27B
  profiles, which route their own widths
  ([#118](https://github.com/jpezzulli/sglang-rtxpro6000/pull/118)).
- **DFlash2 decode checkpoints.** The checkpoint-crossing check compared
  pre-verify lengths against themselves, so the mask was always false and the
  backend received "no tracking" even when an accepted draft block clearly
  crossed the grid — while the live state still committed, letting result
  processing register or promote a checkpoint that was never written. The
  post-verify lengths are now derived from the commit lengths. Both DFlash
  profiles were affected
  ([#115](https://github.com/jpezzulli/sglang-rtxpro6000/issues/115),
  [#119](https://github.com/jpezzulli/sglang-rtxpro6000/pull/119)).
- **EPD idle sleep.** With `--sleep-on-idle` and language-only EPD separation,
  a request waiting for its encoder embeddings left the scheduler "idle", and
  the sleeper's poll did not watch the per-request embedding socket, so the
  request waited for the 1-second poll timeout. The scheduler now skips the
  sleep while embeddings are waiting
  ([#117](https://github.com/jpezzulli/sglang-rtxpro6000/pull/117)).

## DFlash2 is experimental / alpha

Penny's DFlash2 integration — both the Flash-Next `next-dflash2` profile and
the `27b` profile — is now labelled **experimental / alpha** at every
user-facing surface. This is a maturity label, not a toggle and not an
automatic fallback to native MTP; the profiles stay fully available to opt
into, and the maintained default stays native NEXTN MTP.

What that label means today, in two separate parts:

- **Known and documented, not fixed here:** the DFlash2 draft keeps its own
  host cache **in addition to** the target's `PENNY_HICACHE_SIZE_GB` budget.
  This extra allocation is observed, not theoretical: locally, a 32 GB target
  HiCache pool plus about 22 GB of draft cache is roughly 54 GB of caches
  alone before the PLE table and other process memory, and one contributor
  reported a 48 GB target HiCache budget plus a further 32.9 GB of draft
  cache with the host swapping. The actual amount scales with the target's
  logical slots and the draft's geometry, so size host RAM from what the boot
  reports rather than from the HiCache setting alone. 3.1.2 documents this
  bill; it does not fix or reduce the allocation. If DFlash2 demonstrates a
  speed advantage over native MTP, we’ll work on reducing this extra memory
  use.
- **Not yet validated:** the #115 fix is covered by a CPU regression over
  both profiles' checkpoint grids, but the on-GPU state scatter and an actual
  cache-resume were not exercised by it, on either profile.

## Update

Native users can follow [BUILD.md](BUILD.md). Container users should pull
`ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.2` and restart with their existing
external configuration. New host startup scripts and image v3.1.2 are a pair:
the 3.1.2 startup scripts require the `mtp-policy.sh` helper that only ships
in the v3.1.2 image, so replace your downloaded launch files and move to the
new image together ([container guide](docker/pennyroyal/README.md#edit-your-configuration)).
Fresh launch files and the beta configurator select the new image, but an
explicit older image pin in saved settings survives regeneration on purpose:
changing the default does not move your pin, so update it to `v3.1.2`
yourself when you upgrade. The public default launch is unchanged from
3.1.1; `adaptive` and `fixed-w4` are operator choices, not new defaults. The
[3.1 release notes](RELEASE_NOTES_3.1.md) contain the feature list and
performance results.

## Thanks

- [Jim Smith](https://github.com/jhsmith409) (`jhsmith409`) — the
  encoder-embedding idle-sleep fix (PR #117).
- [Lesilva](https://github.com/Lesilva) — the #115 report, proposed
  arithmetic and runtime evidence.
- [Penny Royal](mailto:Pennyroyal@agentmail.to) — width-policy setting,
  checkpoint fix, alpha labeling.

## Validation

Run on this candidate source (2026-10-23), CPU only:

- 20 focused config/setup/recipe/startup-summary tests with 33 subtests
  passed (policy selection, image defaults, release asset, generated and
  shipped launch lines; 170 unrelated cases deselected).
- 65 registered tests passed across the prebuilt-container launch checks
  (shell-to-recorded-server policy selection, parity, saved settings,
  namespace identity, invalid selections, DFlash2 isolation), container
  release routing, the startup version banner and the DFlash2
  checkpoint-tracking regression.
- The checkpoint regression is red on the release base and green with the
  fix. Pinned CI formatting (Black 26.1.0) passes on all changed files.

No GPU runtime, NVMe serving test, image build or benchmark was run for this
candidate. The reusable runtime qualification is the dated 3.1 evidence
(GPU-qualified, release-3.1.json and RESULTS.md); the 3.1.1 record is
CPU/static and image-package verification, not new GPU qualification. Both
profile and performance defaults are unchanged, and the RESULTS.md values
stand as dated history. See [release-3.1.2.json](release-3.1.2.json) for the
draft publication status.
