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
  restored byte for byte, including its 40/80-batch hold and backoff), or
  `fixed-w4` (block 4 at every concurrency). Unset keeps the default, so an
  existing launch is untouched. The checksum gate is gone: the selected policy
  file's hash now decides which NIXL cache namespace a launch uses, not whether
  it boots. The setting is absent from the DFlash2 and 27B recipes, which route
  their own widths ([#118](https://github.com/jpezzulli/sglang-rtxpro6000/pull/118)).
- **DFlash2 decode checkpoints.** The checkpoint-crossing check compared
  pre-verify lengths against themselves, so the mask was always false and the
  backend received "no tracking" even when an accepted draft block clearly
  crossed the grid — while the live state still committed, letting result
  processing register or promote a checkpoint that was never written. The
  post-verify lengths are now derived from the commit lengths, matching the
  native-MTP arithmetic reference. Both DFlash profiles were affected
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

Two things still need validation on real hardware:

- The #115 fix is covered by a CPU regression over both profiles' checkpoint
  grids; the on-GPU state scatter and an actual cache-resume were not
  exercised by it.
- The DFlash2 draft keeps its own host cache **in addition to** the target's
  `PENNY_HICACHE_SIZE_GB` budget: in our measured setup the draft cache added
  about 22 GB on top of the 32 GB HiCache pool — roughly 54 GB of caches alone
  before the PLE table and the rest of the process. A 48 GB host with the
  default 32 GB pool swapped. 3.1.2 documents this bill; it does not fix or
  reduce the allocation. Any allocator work here is conditional on first
  demonstrating speed gains for DFlash2 over native MTP, and until then size
  host RAM from what the boot reports rather than from the HiCache setting
  alone.

## Update

Native users can follow [BUILD.md](BUILD.md). Container users should pull
`ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.2` and restart with their existing
external configuration. Fresh launch files and the beta configurator select
the new image; an explicit older image pin in saved settings survives until
you change it. Replace your downloaded host startup files when you upgrade:
the 3.1.2 startup scripts require the `mtp-policy.sh` helper that ships with
the v3.1.2 image, so new startup files must not be pointed at an older image
([container guide](docker/pennyroyal/README.md#configure-and-start)). The
public default launch is unchanged from 3.1.1; `adaptive` and `fixed-w4` are
operator choices, not new defaults. The
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

Focused CPU regressions cover the new behavior: the width-policy selector and
generated/container launch lines, the mounted container startup scripts, the
startup summary, and unchanged non-MTP (DFlash2/27B) routes, plus the
checkpoint-tracking unit test over both profiles' grids. The fix base is red
on the crossing cases and green with the fix. No new benchmarks were run;
3.1/3.1.1 qualification and the recorded values in RESULTS.md stand as dated
history. No NVMe serving, GPU runtime or image build was exercised for this
candidate; see [release-3.1.2.json](release-3.1.2.json) for the draft
publication status.
