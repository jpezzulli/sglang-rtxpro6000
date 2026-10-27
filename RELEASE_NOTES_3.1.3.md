# Penny Royal 3.1.3 — draft

**Status: draft, not approved, not published; container build pending.** The
release tag, image digest and build/promotion identities are deliberately left
open here and are recorded in [release-3.1.3.json](release-3.1.3.json) as null
until the image lands and the release is approved. Qualified runtime source for
this release is [`ef3258749`](https://github.com/jpezzulli/sglang-rtxpro6000/commit/ef3258749bbe022e38a5b18922a0085320a59928),
whose runtime-code parent is
[`1a6d20827`](https://github.com/jpezzulli/sglang-rtxpro6000/commit/1a6d2082766d3f11646bd972c1bc12effb304cb2).

This patch release carries six accepted correctness fixes, and it corrects one
maturity label: **27B with DFlash2 is an established, qualified profile.** Only
the optional **Flash-Next DFlash2** profile (`next-dflash2`) stays experimental
/ alpha. Model, profile, precision and performance defaults are unchanged from
3.1.2: the launch you ran yesterday launches the same way tomorrow.

## Fixes

Each item below ships as one functional change with its own tests, and
publication happens through its own pull request at the release checkpoint.

- **Anthropic requests with thinking enabled and no budget (#124).** An
  Anthropic-format request that asked for thinking without a `budget_tokens`
  figure was rejected, even though "enabled, budget up to us" is the common
  shape of that request. An absent or null budget now uses the same
  reasoning-enabled conversion as every other route, while a budget you do
  supply is still validated and `disabled` / `adaptive` keep their existing
  rules. If your client sends `{"type": "enabled"}` with nothing else, it now
  gets an answer instead of a 400 ([`45ebf993c`](https://github.com/jpezzulli/sglang-rtxpro6000/commit/45ebf993c121e1a1b61376dcbdafdd21c6d0baac)).
- **An explicit reasoning effort on the Responses API wins.**
  A `/v1/responses` request that named its own `reasoning.effort` could be
  overridden by the effort the server was launched with, so a client asking for
  `low` silently paid for `xhigh`. The request's own choice now takes precedence
  through the existing request-first template policy, and nested overrides, an
  omitted or null effort, summary-only schema defaults and explicit
  thinking-off controls behave exactly as before
  ([`15885c599`](https://github.com/jpezzulli/sglang-rtxpro6000/commit/15885c59934117f311d90f8865bc30d7cc718e17)).
- **Incomplete NIXL file restores are refused (#122).** The disk cache could
  report a successful restore for an object that was missing, zero-length or
  shorter than the host buffer it was filling — most visibly after an
  interrupted write or a partial cleanup. A restore now checks each physical
  object against the byte count it should cover before reading it, and a
  missing, short or unreadable object returns the ordinary failed-read result, so
  the request recomputes the prefix instead of continuing on partial state.
  Complete and padded objects stay valid, and prefetch workers no longer turn a
  metadata query error into a hard failure
  ([`bef8cce23`](https://github.com/jpezzulli/sglang-rtxpro6000/commit/bef8cce23c16c690cebd96480829df473591b7c0)).
- **Retained prefix keys in narrow speculative windows (#123).** The QSA index
  kept its retained-prefix ring sized too tightly for the narrow speculative
  windows this release routes (fixed-W4 at every concurrency), so a retained
  prefix could be recompressed and lose the positions it was supposed to keep.
  The ring now carries two pending index-key groups for speculative widths from
  2 up to the compression ratio, allocation still follows the launch maximum,
  and an undersized or unresolved verify geometry is refused by the existing
  guard rather than guessed. Group-partitioned layout from #40143 is preserved
  ([`12476c9a3`](https://github.com/jpezzulli/sglang-rtxpro6000/commit/12476c9a3cefcf940e8e972190f8cf190d686583)).
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
  ([`ed54e16dc`](https://github.com/jpezzulli/sglang-rtxpro6000/commit/ed54e16dc7064e5b57d1fcbd4e15365a577817a2)).
- **Admission reopens when a running batch empties (#126, selected part).** The
  scheduler latches "batch full" when there is no request-to-token row left to
  allocate, and cleared that latch only when a batch shrank or when a
  *prefill-only* batch came back empty. A batch that emptied through ordinary
  decode was neither, and the rows that caused the latch can be handed back from
  outside it — closing an idle streaming session frees its KV and its row and
  touches nothing. Waiting requests then starved with a free pool. Any empty
  batch now reopens admission; non-empty batches still reset only on a shrink,
  so admission is not retried every step while the batch really is at capacity.
  The rest of #126 (ownership and capacity handling) is **not** part of this
  release ([`1a6d20827`](https://github.com/jpezzulli/sglang-rtxpro6000/commit/1a6d2082766d3f11646bd972c1bc12effb304cb2)).

The QSA, NIXL, hybrid-prefill and admission fixes adapt accepted upstream work:
sgl-project/sglang#46125 (andyl98), #46131 (AEGO-WRJ), #45607 (jongukc) and
#46643 (aryatsc). Source notices and authorship stay with their commits.
Features from #80 and the remaining #47 work are not included.

## DFlash2 maturity correction

3.1.2 labelled both DFlash2 profiles experimental / alpha. That was wrong for
27B: the 27B DFlash2 profile is the shipped 27B launch path, qualified on this
hardware for reasoning, tools, images, long-context retrieval and cache restore,
and it keeps its recipe and fixed block-8 width. Only the Flash-Next DFlash2
profile, which is optional and has seen less field use, carries the
experimental / alpha label
([README](README.md#dflash2-maturity), [RUN](RUN.md#launch-27b-with-dflash2),
[limitations](LIMITATIONS.md)).

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

## Update

Native users can follow [BUILD.md](BUILD.md). Container users should pull
`ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.3` once it is published and restart
with their existing external configuration; until the image is published, that
tag does not exist yet
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

Accepted on 2026-10-27 on the candidate source above:

- Integrated CPU suite: 409 tests with 404 subtests passed, 1 optional S3 check
  skipped. Each of the six fixes was reviewed individually and its patch matched
  the accepted contribution exactly by patch-id. The focused regressions behind
  #109, #122, #123, #124 and the Responses-effort fix are red on the 3.1.2 base
  and green with the fix.
- Real-CUDA coverage of the QSA fix used the production shape with retained
  prefix residues of 1, 2 and 3 tokens, partial acceptance and fused
  compression; the serving suite passed 9/9.
- Both profiles passed 4/4 API probes, a sealed agentic run (6 turns, 5 tool
  calls), image checks, and correct 63,906-token and 489,921-token retrieval.
  Fresh-process 64K and 490K NIXL restores were correct on both: each returned
  553,792 NIXL-backed KV tokens and 4 Mamba states, and the 27B profile
  additionally 4,096 sliding-window tokens. Both restart restores are valid.
  Repeat replays of the same prompt answered from the device cache rather than
  four independent disk loads, so the disk-path evidence is the two restore
  runs.
- Reasoning: all 36 measured requests completed naturally, with no confirmed
  loops and extended reasoning not penalized. Blind grading scored 95.36/100
  sequentially and 97.61/100 across three concurrent users on Flash-Next native
  MTP, and 98.67/100 and 99.23/100 on 27B DFlash2.
- Tool calls are green with named deviations, not universally. Flash-Next
  native MTP: 30/30 parseable, 29/30 strict-matcher, 26/30 automatic; the
  automatic misses include scorer/date substring behaviour and a mock-router
  response. 27B DFlash2: 30/30 scheduled, exact and parseable, 28/30 automatic —
  two answers carried an invalid date and exhausted their 512-token ceiling
  entirely inside reasoning with empty visible answers, and one invoice answer
  added unwanted explanatory prose. The separate agentic and vision checks
  passed.
- Dated native decode and time-to-first-token measurements for both profiles are
  in [RESULTS.md](RESULTS.md#penny-royal-313-october-27-2026). No new
  qualification was run for the documentation-only parts of this release.

Two things are recorded honestly as open rather than fixed:

- The first 27B four-request screen stopped with an Xid 109 GPU freeze. The
  unchanged retry after a host reboot completed and is the accepted row. The
  cause is unproven, so **no freeze fix is claimed**, and the detailed incident
  note stays an internal observation.
- Under host-cache pressure, state was evicted rather than retained purely on
  the host. Host checkpoint reclamation and the NIXL disk-budget cleaner were
  both active, so pure-host retention was not isolated on this candidate. This
  outcome is accepted; it is not a blocker and not a testing instruction.

No NVMe PLE serving qualification was run, and the GPU container smoke check is
still pending under separate publication coordination — nothing here claims it
passed.

## Thanks

- [John Pezzulli](https://github.com/jpezzulli) — the Responses-API reasoning
  effort precedence fix.
- [andyl98](https://github.com/andyl98), [AEGO-WRJ](https://github.com/AEGO-WRJ),
  [jongukc](https://github.com/jongukc) and [aryatsc](https://github.com/aryatsc)
  — the upstream corrections this release adapts.
- [Penny Royal](mailto:Pennyroyal@agentmail.to) — the remaining five fixes,
  integration and this release's documentation.
