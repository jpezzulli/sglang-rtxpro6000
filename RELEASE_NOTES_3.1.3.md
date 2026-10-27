# Penny Royal 3.1.3

Published 2026-10-27 as
[`pennyroyal-v3.1.3`](https://github.com/jpezzulli/sglang-rtxpro6000/releases/tag/pennyroyal-v3.1.3).
The six correctness fixes and the maturity-label correction are merged as
[#127](https://github.com/jpezzulli/sglang-rtxpro6000/pull/127) through
[#133](https://github.com/jpezzulli/sglang-rtxpro6000/pull/133), with
documentation followups
[#134](https://github.com/jpezzulli/sglang-rtxpro6000/pull/134) and
[#135](https://github.com/jpezzulli/sglang-rtxpro6000/pull/135) on top. The
qualified candidate is
[`ef3258749`](https://github.com/jpezzulli/sglang-rtxpro6000/commit/ef3258749bbe022e38a5b18922a0085320a59928),
whose runtime-code parent is
[`1a6d20827`](https://github.com/jpezzulli/sglang-rtxpro6000/commit/1a6d2082766d3f11646bd972c1bc12effb304cb2):
that is the source the October 27 checks were run on, and it still carries the
3.1.2 version and image assets. The merged integrated runtime source is
[`f38baf2f8`](https://github.com/jpezzulli/sglang-rtxpro6000/commit/f38baf2f84337a7f86ee5d2b38ff95c868f05ae8);
the image was built from
[`ec1f64e85`](https://github.com/jpezzulli/sglang-rtxpro6000/commit/ec1f64e850f31f33decee6d925ab2d57a52341c3),
whose source tree is identical. Full identities - digest, manifest, sizes, build
and promotion runs - are in [release-3.1.3.json](release-3.1.3.json).

This patch release carries six accepted correctness fixes, and it corrects one
maturity label: **27B with DFlash2 is an established, qualified profile.** Only
the optional **Flash-Next DFlash2** profile (`next-dflash2`) stays experimental
/ alpha. Model, profile, precision and performance defaults are unchanged from
3.1.2: the launch you ran yesterday launches the same way tomorrow.

## Fixes

Each item below is the problem it solves and what changes for you.

- **Anthropic requests with thinking enabled and no budget (#124).** An
  Anthropic-format request that asked for thinking without a `budget_tokens`
  figure was rejected, even though "enabled, budget up to us" is the common
  shape of that request. An absent or null budget now uses the same
  reasoning-enabled conversion as every other route, while a budget you do
  supply is still validated and `disabled` / `adaptive` keep their existing
  rules. If your client sends `{"type": "enabled"}` with nothing else, it now
  gets an answer instead of a 400 ([#127](https://github.com/jpezzulli/sglang-rtxpro6000/pull/127)).
- **An explicit reasoning effort on the Responses API wins.**
  A `/v1/responses` request that named its own `reasoning.effort` could be
  overridden by the effort the server was launched with, so a client asking for
  `low` silently paid for `xhigh`. The request's own choice now takes precedence
  through the existing request-first template policy, and nested overrides, an
  omitted or null effort, summary-only schema defaults and explicit
  thinking-off controls behave exactly as before
  ([#128](https://github.com/jpezzulli/sglang-rtxpro6000/pull/128)).
- **Incomplete NIXL file restores are refused (#122).** The disk cache could
  report a successful restore for an object that was missing, zero-length or
  shorter than the host buffer it was filling — most visibly after an
  interrupted write or a partial cleanup. A restore now checks each physical
  object against the byte count it should cover before reading it, and a
  missing, short or unreadable object returns the ordinary failed-read result, so
  the request recomputes the prefix instead of continuing on partial state.
  Complete and padded objects stay valid, and prefetch workers no longer turn a
  metadata query error into a hard failure
  ([#129](https://github.com/jpezzulli/sglang-rtxpro6000/pull/129)).
- **Retained prefix keys in narrow speculative windows (#123).** The QSA index
  sized its retained-prefix ring too tightly for the narrow speculative windows
  produced by the optional `fixed-w4` width policy, so a retained prefix could be
  recompressed and lose the positions it was supposed to keep. The ring now
  carries two pending index-key groups for speculative widths from 2 up to the
  compression ratio, allocation still follows the launch maximum, and an
  undersized or unresolved verify geometry is refused by the existing guard
  rather than guessed. Group-partitioned layout from #40143 is preserved. This
  affects the `fixed-w4` choice; the public `tiered` default routes wider blocks
  and was not the reported case
  ([#130](https://github.com/jpezzulli/sglang-rtxpro6000/pull/130)).
- **Truncated hybrid-Mamba chunks end on a page boundary (#109).** A chunked
  prefill cut short by the KV budget — the normal case under four- and
  six-request pressure with 64-token pages — committed at an arbitrary token
  count. The first pass still donated a page-aligned checkpoint, but the tokens
  past it stayed private to that request, so the *next* chunk cached a state
  under a key up to a page shorter than the prefix it summarized: another
  request could resume from a state that had already folded in someone else's
  tokens. Truncated hybrid chunks are now floored to the page, a chunk with less
  than a page of budget is parked and retried on the existing deferral path, and
  untruncated tails and full-attention profiles are untouched
  ([#131](https://github.com/jpezzulli/sglang-rtxpro6000/pull/131)).
- **Admission stops staying shut on an already-empty batch (#126, selected
  part).** The scheduler latches "batch full" when there is no request-to-token
  row left to allocate, and it cleared that latch only when the running batch
  shrank or when an empty *prefill-only* batch came round. A batch that was
  **already empty** clears neither test: there is no request left in it to leave,
  so no shrink is ever seen, and the reset could not come from inside the batch.
  Meanwhile the rows that produced the latch can be handed back from outside it -
  closing an idle streaming session frees its KV and its row and touches nothing
  - so waiting requests starved with a free pool. Any empty batch now reopens
  admission. A batch that still holds requests resets only on a shrink, as
  before, so admission is not retried every step while the batch really is at
  capacity. The rest of #126 (ownership and capacity handling) is **not** part of
  this release
  ([#132](https://github.com/jpezzulli/sglang-rtxpro6000/pull/132)).

The QSA, NIXL, hybrid-prefill and admission fixes adapt accepted upstream work
(#46125, #46131, #45607 and #46643); the original authors and the adaptation are
recorded in the commit messages and in [PROVENANCE.md](PROVENANCE.md). Features
from #80 and the remaining #47 work are not included.

## DFlash2 maturity correction

3.1.2 labelled both DFlash2 profiles experimental / alpha. That was wrong for
27B: the 27B DFlash2 profile is the shipped 27B launch path, and the October 27
checks on this candidate served it end to end with reasoning, tools, images,
long-context retrieval and a fresh-process cache restore. It keeps its recipe and
fixed block-8 width. Only the Flash-Next DFlash2 profile, which is optional, has
seen less field use and was not requalified on this candidate, carries the
experimental / alpha label
([README](README.md#dflash2-maturity), [RUN](RUN.md#launch-27b-with-dflash2),
[limitations](LIMITATIONS.md),
[#133](https://github.com/jpezzulli/sglang-rtxpro6000/pull/133)).

The draft's host-memory bill is a separate matter, and it applies to both
DFlash2 profiles, so it stays documented on both. The launch measured for this
release allocated **79.45 GB of target KV plus 16.63 GB of Mamba state** — the
96 GB target/Mamba budget — **and a further 24.83 GB of draft sliding-window
cache**, about **120.91 GB of caches** before any other process memory. The
earlier observations stand beside it: a 32 GB target HiCache pool with about
22 GB of draft cache is roughly 54 GB of caches on Flash-Next, and one
contributor reported a 48 GB target budget plus 32.9 GB of draft cache with the
host swapping. **3.1.3 documents this bill; it changes no allocator and claims
no memory reduction.** Any future memory work depends on Flash-Next DFlash2
first demonstrating a speed benefit over native MTP.

## Published image

`ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.3`, digest
`sha256:cdba5f8d72098813454aa552bad35873d65b2ba8cd72f93f3ae3ed7057c0636a`,
built by
[run 40525831339](https://github.com/jpezzulli/sglang-rtxpro6000/actions/runs/40525831339)
from source
[`ec1f64e85`](https://github.com/jpezzulli/sglang-rtxpro6000/commit/ec1f64e850f31f33decee6d925ab2d57a52341c3)
and promoted without rebuilding by
[run 40549616706](https://github.com/jpezzulli/sglang-rtxpro6000/actions/runs/40549616706);
the release tag adds documentation only on top of that image source. Manifest,
config, exact sizes and the candidate image tag are in
[release-3.1.3.json](release-3.1.3.json).

The published image was verified on the real hardware on October 27, 2026: the
existing-image check passed with GPUs hidden and no network (pinned packages,
the accepted FlashInfer SM120 sources and AOT module, NIXL POSIX/FILE and DRAM
transport config, the NVMe PLE io_uring import, clean exact source), the 28
runtime, launch, policy and version files hash-matched, the 8 installed
NVMe-reader Python files matched the vendored source, the obsolete NVMe source
manifest was absent, and the 16 existing recipe tests passed inside the image. A
quick actual-GPU serving smoke then passed on the Flash-Next profile with the
public `tiered` default at TP1: model discovery and health, an exact READY
stop-token answer, and one tool-call turn that returned the two expected lookup
calls as valid JSON with a `tool_calls` finish and no prose, at context 524,288,
4 requests / 24 slots, 32 GB HiCache with NIXL, RAM preprocessing and local
diagnostics off. Native was then restored byte-for-byte and passed the same two
checks at `adaptive`, 6 requests / 36 slots and 4,000-token preprocessing. This
is image-serving evidence, not a new throughput qualification and not a 27B
container boot; no NVMe PLE serving qualification was run.

## Update

Native users can follow [BUILD.md](BUILD.md). Container users should pull
`ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.3` and restart with their existing
external configuration
([container guide](docker/pennyroyal/README.md)). Fresh launch files, Compose
defaults, `.env.example` and the beta configurator name the new image, but an
explicit image pin in your saved setup or in an edited `run.sh` survives on
purpose — changing the default does not move your pin, so update it to `v3.1.3`
yourself when you upgrade. The 3.1.3 startup files belong with the 3.1.3 image;
change them together
([configuration reference](CONFIGURATION.md#container-routing-settings)).

Nothing here changes a setting or a default: the public native-MTP width policy
stays `tiered` (block 8 at one running request, block 4 at two or more),
`adaptive` and `fixed-w4` remain operator choices, the launcher default for
reasoning effort remains `medium`, and local diagnostics stay off. The rows in
[RESULTS.md](RESULTS.md#penny-royal-313-october-27-2026) record what was
measured on this candidate and with which configuration; they are dated
observations, not a matched-baseline speed-up claim.

## Validation

Accepted on 2026-10-27 on the candidate source above. Reused evidence, not rerun
for the documentation-only parts of this release:

- Integrated CPU suite: 409 tests with 404 subtests passed, 1 optional S3 check
  skipped. Each of the six fixes was reviewed individually and its patch matched
  the accepted contribution exactly by patch-id. The focused regressions behind
  #109, #122, #123, #124 and the Responses-effort fix are red on the 3.1.2 base
  and green with the fix.
- Real-CUDA coverage of the QSA fix used the production shape with retained
  prefix residues of 1, 2 and 3 tokens, partial acceptance and fused
  compression; the serving suite passed 9/9.
- The maintained Flash-Next native-MTP profile and the 27B DFlash2 profile each
  passed 4/4 API probes, a sealed agentic run (6 turns, 5 tool calls), image
  checks, and correct 63,906-token and 489,921-token retrieval. The optional
  Flash-Next DFlash2 profile was not requalified on this candidate.
- Fresh-process 64K and 490K NIXL restores were correct on both of those
  profiles: each returned 553,792 NIXL-backed KV tokens and 4 Mamba states, and
  the 27B profile additionally 4,096 sliding-window tokens. Both restart restores
  are valid. Repeat replays of the same prompt answered from the device cache
  rather than four independent disk loads, so the disk-path evidence is the two
  restore runs. The #115 decode-checkpoint arithmetic fix stays covered by its
  CPU regression; an end-to-end restore is not a grid-scatter unit regression,
  and no such isolated GPU check exists here.
- Reasoning: all 36 measured requests completed naturally, with no confirmed
  loops and extended reasoning not penalized. Blind grading scored 95.36/100
  sequentially and 97.61/100 across three concurrent users on Flash-Next native
  MTP, and 98.67/100 and 99.23/100 on 27B DFlash2.
- Tool calls: Flash-Next native MTP 30/30 parseable, 29/30 strict-matcher, 26/30
  automatic, with the four automatic misses coming from scorer/date substring
  behaviour and a mock-router response rather than model call failures. 27B
  DFlash2 30/30 scheduled, exact and parseable, 28/30 automatic: the two misses
  are the invalid-date test cases, where the model issued the correct
  `validate_date` calls and then used its entire 512-token output ceiling on
  reasoning, so the final answer came back with an empty visible body. One
  invoice request also returned unwanted explanatory prose alongside a correct
  call. The separate agentic and vision checks passed.
- Dated native decode and time-to-first-token measurements for both profiles are
  in [RESULTS.md](RESULTS.md#penny-royal-313-october-27-2026).

Two observations are recorded as open rather than fixed:

- The first attempt at the 27B four-request screen on October 27, 2026 ended with
  an Xid 109 GPU freeze. The unchanged retry completed and is the measurement that
  was kept. The cause is unproven and 3.1.3 does not claim to fix it; the event
  stays under observation.
- Under host-cache pressure, state was evicted rather than retained purely on the
  host. Host checkpoint reclamation and the NIXL disk-budget cleaner were both
  active during that run, so pure-host retention was not isolated on this
  candidate.

No NVMe PLE serving qualification was run. The published image itself was
checked and smoke-tested on the real hardware on October 27, 2026, as recorded
above and in the machine-readable record.

## Thanks

- [John Pezzulli](https://github.com/jpezzulli) - the Responses-API reasoning
  effort precedence fix.
- [Penny Royal](mailto:Pennyroyal@agentmail.to) - the remaining five fixes,
  integration and this release's documentation.

The upstream authors whose corrections were adapted here are credited in the
commit messages and in [PROVENANCE.md](PROVENANCE.md).
