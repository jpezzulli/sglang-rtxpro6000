# Cumulative changes and upstream status

The current runtime is the ordered range
`e7e78940168f..739aff3dc599`. All 27 runtime commits remain in source history,
including the graph-lifetime trial and its full revert. The integration base
is unchanged. The v2.1.1 upstream check fetched `main` at
`cdbfe90b4a31079859817c148ef4498240ec2580` on 2026-08-29; older PR-status
tables below retain their stated observation dates. Version 2.1.2 added the
two maintenance corrections below, without a rebase. v2.3 adds the
FR-Spec launch configuration using the same executable. v2.3.1 adds only the
GDN rounding correction described below.

“Local” does not mean a permanent fork requirement. It means the exact active
commit has not merged upstream. Where a PR has a later refined head, that is
shown separately rather than pretending the local commit and PR head are
identical.

## v2.3.1 — GDN rounding maintenance

Released September 6, 2026. If v2.3 is working well for you, there is no urgent
need to update. This is a small correctness fix, not a performance release.

Commit `739aff3dc5` adapts SGLang
[#36014](https://github.com/sgl-project/sglang/pull/36014) by
[V-aerus](https://github.com/V-aerus) (Hangshuai He), from head
`bf5d4227dde4e532b4668327a8d6dcc4d79d9c2b`.

**27B FP8/DFlash2:** speculative verification could round a GDN gate
differently from ordinary decoding, causing their recurrent states to differ.
The correction aligns verification and accepted-state/radix-boundary recovery
when packed decode and verification both use Triton. It preserves the
existing behavior of mixed backends, KDA, and alternate-device paths.
**Flash-Next NVFP4/native MTP:** its FlashInfer/WY path is unchanged and passed
the same runtime regressions.

We could not reproduce the reported failure on the live runtime. Four focused
SM120 GPU tests did reproduce the underlying numerical mismatch before the
correction. The final fix passed:

- 25 GDN GPU tests and 14 dispatcher/policy CPU tests with 32 subtests;
  adjacent KDA/stride coverage also passed during development.
- Three sequential reviews, with review findings addressed and no remaining
  correctness findings in the final review.
- Both profiles' structured-output and parallel-tool smoke checks, exact
  READY from approximately 64K prefill, all three needles in one approximately
  490K prompt, and 1,024-token C1/C4 decode.
- Identical restart restoration of 489,856 NIXL tokens with all three needles
  correct on both profiles. Context, full KV pools, and CUDA graphs were retained.

Launch settings and dependencies are unchanged. Full reasoning, the complete
tool suite, and vision were not repeated; existing performance tables retain
their original versions and measurements. These results establish the narrow
rounding correction, not every failure reported for speculative decoding.

## v2.3 — Flash-Next FR-Spec

Released September 6, 2026.

**Flash-Next NVFP4 with native NEXTN MTP:** FR-Spec reduces draft-head work
by scoring a 65,536-token subset. The target head, full target vocabulary,
verification, and acceptance policy are unchanged.

- Adds an FR-Spec launcher, token map, map manifest, and deterministic map builder.
- Keeps 524,288-token context, 824,384 KV tokens, four concurrent requests,
  24 Mamba slots, CUDA graphs, and 32 GiB HiCache/NIXL.
- Includes the token-map hash in the persistent-cache identity.
- Uses the same SGLang executable, kernels, and dependencies as v2.1.2.
- Leaves the supported 27B FP8/DFlash2 profile unchanged.

Credit: [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120)
for the Flash-Next reduced draft-vocabulary optimization. The Pennyroyal map
was generated from runtime source, excluding evaluation cases and retaining
all special tokens. [PROVENANCE.md](PROVENANCE.md#v23-fr-spec-provenance)
records the source revision and map hashes.

Tests covered Flash-Next reasoning, tools, vision, agent workflows, long
context, and NIXL persistence. Both profiles passed startup, prefill,
single/concurrent decode, and restart-restoration checks. Full 27B
reasoning/tool tests were not repeated; its runtime and launcher are unchanged.

See [RESULTS.md](RESULTS.md#pennyroyal-v23--flash-next-fr-spec) for the measured
speed increase and [LIMITATIONS.md](LIMITATIONS.md#v23-validation-scope)
for the test scope.

## Version 2.1.2: maintenance only

Version 2.1.2 contains **two correctness fixes, no new features, and no new
performance claims**. Launch settings, dependency versions, and existing
performance tables are unchanged.

| Upstream PR | Models affected | What was wrong in normal terms | What changed |
|---|---|---|---|
| [#37962](https://github.com/sgl-project/sglang/pull/37962) | **Both 27B FP8/DFlash2 and Flash-Next NVFP4/native MTP.** | Grammar-constrained sampling could run an unnecessary token-ID synchronization even with only one GPU in the selected group. Its first NCCL operation could allocate GPU memory late, after model/cache allocation had settled. | Skip synchronization for a one-rank group. Preserve the selected TP/attention-TP group and normal multi-rank MIN behavior. Construction reads coordinator metadata so mocked tests do not need an initialized distributed process group. |
| [#37408](https://github.com/sgl-project/sglang/pull/37408) | **Both profiles when streaming with the Qwen3 Coder tool parser.** | Whitespace between tool calls could be emitted as ordinary content before pending JSON argument fragments, breaking tool-call framing in affected clients. | Hold post-tool separator whitespace until the next tag or genuine prose resolves it. Preserve real prose and word spaces instead of globally dropping whitespace. |

These are adapted local corrections, not claims that the upstream PR heads
were cherry-picked verbatim. Both PRs were open when checked on 2026-09-05:
`#37962@a0c2ba07ef31`, `#37408@b4e45091db2c`.

The source history adds `e2c6f4a4f8` and `836206a0ad`. The first also tried the
separate #37448 graph-lifetime correction; the second fully reverted that
trial after its memory cost was rejected. **#37448 is not included in the
released runtime behavior.** The net source diff from v2.1.1 changes only
the sampler, Qwen3 Coder parser, and their focused tests.

Validation for this maintenance release:

- Exact-base tests reproduced both defects before correction.
- Sampler coverage: 117 CPU tests and 27 subtests passed; 16 ROCm-only tests
  skipped. Parser coverage: 220 tests and 20 subtests passed; 15 unrelated
  DeepSeek tokenizer cases excluded because their offline fixtures were
  unavailable.
- Two fresh sequential adversarial reviews completed. The final
  sampler/parser subset had no material findings.
- Both profiles passed streaming/non-streaming JSON-schema checks with greedy
  and sampled output, ordinary warmups, and incremental two/four parallel-tool
  checks for IDs, names, JSON arguments, and separator ordering.
- Both profiles passed ordinary short/long prefill, one three-needle long
  prompt, single-request and four-request decode, and actual NIXL restart
  restoration. Original graph coverage and cache capacity were retained.

The runtime checks used Orca 27B FP8 and a source-specific local ModelOpt NVFP4
conversion of Orca Flash-Next, not a new Radix checkpoint campaign. Full
reasoning, vision, the complete 30-case tool suite, external translators, and
multi-GPU execution were not rerun. Arbitrary prose/tool coalescing in an
external translator is not claimed fixed or qualified. This release does not
replace or refresh the historical performance results.

## Version 2.1.1: sampling and cache-restore correctness

Version 2.1.1 keeps both qualified launch shapes unchanged and adds two fixes.
Their model impact is intentionally explicit:

| Commit | Models affected | What was wrong in normal terms | What changed |
|---|---|---|---|
| `0e5d8e3793` | **Qwen3.8-27B with DFlash2 only.** Flash-Next does not use DFlash2. | DFlash2 verifies several proposed tokens at once. Accumulated additive sampling penalties—such as presence/frequency penalties and minimum-new-token EOS suppression—were read from an obsolete field, so the verify block could incorrectly skip them. | Verification now applies the real accumulated additive penalty to every token position in the proposed block and does not take the no-adjustment fast path while that penalty is active. This adapts open SGLang PR [#33869](https://github.com/sgl-project/sglang/pull/33869). |
| `fb1216c6c4` | **Both Qwen3.8-27B/DFlash2 and Flash-Next.** They share HiCache kernel transfers, page-first pools, overlap scheduling, and NIXL persistence. | A cache page could be restored on the HiCache copy stream while an earlier model forward pass was still writing that same GPU page. In addition, TVM-FFI JIT copy kernels did not automatically follow the torch transfer stream, so adding only a torch-stream fence did not cover every selected Penny path. The rare result could be a silently damaged restored prefix. | H2D load-back now waits behind the model forward stream. Kernel-backend D2H and H2D JIT copies are explicitly bound to the intended torch transfer stream, and the previous TVM-FFI per-thread stream is restored afterward so unrelated JIT work does not inherit the HiCache stream. This completes merged SGLang PR [#36738](https://github.com/sgl-project/sglang/pull/36738) for Penny's active JIT paths while replacing the needed portion of closed PR [#36572](https://github.com/sgl-project/sglang/pull/36572). |

Focused validation included 23 DFlash tests, 8 direct GPU HiCache stream/race
tests, 73 broader hybrid/NIXL tests with 2 skips, and 94 unified-cache
load-back tests. Three sequential adversarial reviews found no material issue.
Both supported models then completed ordinary 64K prefill, one 490K
three-needle prefill, one 1,024-token decode, four simultaneous 1,024-token
decodes, and post-restart NIXL restoration. Each restart restored 489,856
tokens and retained all three exact needles. The final Flash-Next canonical
namespace was separately seeded and restart-verified after promotion.

## Version 2.1.0: bounded upstream correctness sync

Version 2.1 retains the qualified launch shapes and adds three source commits:

| Commit | Upstream relationship | Function and evidence |
|---|---|---|
| `8de07a058f` | Adapted from merged PR [#36806](https://github.com/sgl-project/sglang/pull/36806) | Replaces the family-wide SM12x QSA gate with exact SM120 detection, excluding SM121/GB10 while leaving RTX PRO 6000 behavior unchanged. Exact-capability and resolver tests cover SM100, SM120, and other SM12x. |
| `23e51dddcb` | Adapts merged PR [#35821](https://github.com/sgl-project/sglang/pull/35821) | Skips empty Mamba radix checkpoints instead of inserting a stale ghost node, and bounds interval-track selection to accepted speculative tokens. Penny's fused CUDA kernel and KDA path receive the same correction. |
| `1ba0b2a1b5` | Local test maintenance | Updates QSA hybrid test doubles for the existing RecoverSSM constructor contract; no runtime behavior change. |

Focused validation: 72 QSA/Mamba/CUDA tests passed. Flash-Next retained its
824,384-token FP8 target/native-MTP pools, 24 Mamba slots, recovery graphs,
73,664-token prefix reuse, and post-restart NIXL restoration. The 27B
DFlash2 launcher retained 1,118,784-token FP8 target/draft pools, fused KV
materialization, graph capture, and 73,664-token prefix reuse. No CUDA errors
or retractions were observed. Full reasoning/tool suites were not rerun.

## 2026-08-24: Qwen3.8-27B/DFlash2 dated release

Tag: `qwen38-dflash2-pro6000-20260824`.

That release established the 27B architecture, DFlash2 performance, XQA target
verification, request-span measurement, and HiCache/NIXL persistence later
carried into the unified runtime. Its seven source commits were:

| Dated-release commit | Function | Current lineage |
|---|---|---|
| `560f06a94e` | independent target/draft Hugging Face overrides | represented by current `a8c4404ef8`; project PR #35583 closed unmerged |
| `b74d971054` | preserve draftless custom speculative hooks | represented by current `94f362d1f2` |
| `5b06cf9f42` | pass fingerprinted host compiler to NVCC | represented by current `512a95e329`; project PR #35584 open |
| `e9971ac6eb` | queue/prefill/post-prefill/forward elapsed spans | represented by current `730d244a08` |
| `ba600c682a` | Mooncake hybrid-I/O experiment | obsolete; Mooncake was rejected and is not in the active deployment |
| `2e79cdc030` | XQA packed mask for fixed-width speculative verification | represented by current `0f159cd545` |
| `8e197ed3af` | NIXL bounce-transfer and overlapping-registration corrections | split into current `8b786639e4` and `067c639c0a`; project PRs #36520/#36524 |

Core DFlash2 support was already upstream-derived: PR #35371 added local
convolution and the selector; PR #35496 added quantized target-`lm_head`
selection; PR #35663 added the Qwen3.8-27B recipe. The selected target's
`lm_head` is BF16, so #35496 is present but not the measured fast path.

The dated release preserved the controlled performance, reasoning,
long-context, and real-agentic evidence documented in
[RESULTS.md](RESULTS.md).

## 2026-08-27: unified runtime and Flash-Next

Tag: `sglang-rtxpro6000-20260827`.

The branch moved to the current upstream integration base, carried the required
27B behavior forward, and added day-one Flash-Next/Qwen4, QSA, native MTP,
RecoverSSM, complete hybrid-state persistence, and three-axis fused mRoPE.

### Ordered active runtime changes

| Local commit | Disposition / upstream relationship | Affected path | Why and focused evidence |
|---|---|---|---|
| `7e4c212f7d` | Adapted import from open PR [#36497](https://github.com/sgl-project/sglang/pull/36497) | Flash-Next/Qwen4, QSA, PLE, native MTP, HC | Day-zero model implementation. Imported QSA, HC, PLE, MTP, memory-pool and model tests; full Flash-Next qualification. |
| `c1da0eef56` | Local bounded SM120 wrapper enablement | Flash-Next QSA decode | Routes SM120 into FlashInfer's page-aligned QSA wrapper, which resolves to XQA on SM12x; focused dispatch tests plus live decode. This does not enable TRTLLM-Gen. |
| `8de07a058f` | Adapted from merged Qwen4 integration PR [#36806](https://github.com/sgl-project/sglang/pull/36806) | Flash-Next QSA decode architecture gate | Narrows the wrapper route to exact SM120 and excludes SM121/GB10; focused capability/resolver matrix and unchanged RTX PRO 6000 runtime. |
| `a8c4404ef8` | Local runtime version; related project PR [#35583](https://github.com/sgl-project/sglang/pull/35583) closed unmerged | All speculative draft config loading; needed by 27B DFlash2 | Target and draft need independent Hugging Face overrides. `test_draft_model_override_args.py`. |
| `94f362d1f2` | Local follow-up | Draftless custom speculation hooks | Preserves algorithm hooks when no draft checkpoint exists; covered by Flash-Next NEXTN startup/smoke, with no dedicated commit-local test. |
| `512a95e329` | Local runtime version; open project PR [#35584](https://github.com/sgl-project/sglang/pull/35584) | Native JIT build | Passes fingerprinted `CXX` to NVCC `-ccbin`; JIT cache/toolchain regression. |
| `730d244a08` | Local observability | All requests | Adds queue, initial-prefill, post-prefill, and total-forward spans; request-time-stat unit tests and agentic analysis. No kernel behavior change. |
| `0f159cd545` | Local bounded SM120 correction | 27B DFlash2 fixed-width target verify | Supplies XQA packed causal mask through CUDA-graph metadata; graph metadata tests and 1x/4x DFlash2 runs. |
| `8b786639e4` | Local runtime version; open project PR [#36520](https://github.com/sgl-project/sglang/pull/36520) has refined head | NIXL FILE path registration | Prevents overlapping backup/prefetch contexts from reusing active device IDs; allocator invariant, concurrent and nested-registration tests. |
| `067c639c0a` | Local runtime version; open project PR [#36524](https://github.com/sgl-project/sglang/pull/36524) has refined head | NIXL hybrid bounce transfers | Chunks sidecar transfers at staging capacity and validates contiguous result prefixes; get/set, malformed-vector, zero-copy and real POSIX tests. |
| `37a163406f` | Local reconciliation of PR #36497 with then-current main | Flash-Next imported surface | Preserves current runtime context, Mamba validation, Hopper GEMV and ratio-3 GDN behavior; adapted focused tests and repository format checks. |
| `22161283b6` | Local day-zero integration correction | Qwen4 native MTP | Reads the moved parallel configuration leaf through `.config`; model unit regression and successful native-MTP boot. |
| `95da38fb3b` | Local unit-scale fix; overlaps only part of broader open PR [#36644](https://github.com/sgl-project/sglang/pull/36644) | Flash-Next FP8 QSA sparse prefill | Dequantizes active FP8 tiles before Triton sparse attention; QSA numerical test plus exact 64K/490K runs. It is not the calibrated-scale PR. |
| `280825c3e2` | Adapted from open PR [#30967](https://github.com/sgl-project/sglang/pull/30967) with Penny Qwen4/SM120 integration | Flash-Next GDN MTP state | Removes intermediate speculative state and enables narrow SM120 WY output/recovery. 43 focused CPU tests, 149 skips, 34 subtests, and four direct SM120 parity/graph tests. |
| `516e42a2ee` | Local persistence integration | Flash-Next Qwen4 PLE + GDN HiCache | Persists all PLE state siblings with Mamba state. Transfer, load-timing and NIXL storage tests plus restart restore. |
| `e1a1572743` | Local persistence correction | Mixed-size PLE/Mamba NIXL bounce I/O | Aligns mixed state components to logical page boundaries; NIXL storage regressions and live restore. |
| `bcd51eeb0e` | Local ordering correction | PLE ngram state restore | Restores PLE ngram state before the next use; Mamba/PLE unit regression and restart continuation. |
| `1787f88569` | Local ordering correction | Deferred Mamba copy-on-write after HiCache load | Prevents COW from observing pre-load state; unit regression and long prefix-reuse continuation. |
| `7b5cfb728d` | Local persistence integration | Flash-Next compressed QSA index keys | Adds QSA side pool to hybrid persistence; focused pool-host unit tests and 490K restart/needle evidence. |
| `64ecd64924` | Integrated open PR [#35744](https://github.com/sgl-project/sglang/pull/35744) | Qwen3.5/3.8 multimodal fused QK RMSNorm+RoPE | Applies all three mRoPE axes instead of silently using temporal only. Extensive fused-kernel numerical/contract tests and real Flash-Next/27B vision validation. |
| `23e51dddcb` | Adapted from merged PR [#35821](https://github.com/sgl-project/sglang/pull/35821) | Mamba radix finish and speculative accepted-state tracking | Prevents zero-length ghost nodes and carries the accepted-step clamp into Penny's eager, fused CUDA, and KDA paths; CPU ghost-node and CUDA boundary parity tests. |
| `1ba0b2a1b5` | Local test maintenance | QSA hybrid test fixtures | Models the existing RecoverSSM constructor field in test doubles; completes the 72-test focused suite with no runtime change. |
| `0e5d8e3793` | Adapts open PR [#33869](https://github.com/sgl-project/sglang/pull/33869) | 27B DFlash2 target verification and accumulated sampling penalties | Applies `acc_additive_penalties` across every flattened verify token and disables the no-adjustment predicate while the accumulated penalty is active; exact-base red/green tests plus ordinary penalized 27B generation. |
| `fb1216c6c4` | Completes merged PR [#36738](https://github.com/sgl-project/sglang/pull/36738) for Penny's JIT paths; replaces the required scoped behavior from closed PR [#36572](https://github.com/sgl-project/sglang/pull/36572) | Shared HiCache D2H/H2D transfer streams and load-back ordering | Binds kernel-backend TVM-FFI transfers to the current torch stream with restoration on exit, then fences H2D load-back behind the forward stream. Direct GPU race tests and two-profile NIXL restart restoration cover the selected paths. |
| `e2c6f4a4f8` | Adapts PRs [#37962](https://github.com/sgl-project/sglang/pull/37962) and [#37408](https://github.com/sgl-project/sglang/pull/37408) | One-rank sampler synchronization and Qwen3 Coder streaming separators | Exact-base red/green tests, sequential review, and both-profile API/decode/prefill/restore regressions. This commit also contains a graph-lifetime trial, fully reverted by the next commit. |
| `836206a0ad` | Local disposition of PR [#37448](https://github.com/sgl-project/sglang/pull/37448) | Restores pre-existing graph behavior and tests | Reverts the rejected graph-lifetime trial exactly to the v2.1.1 base, leaving only the two maintenance corrections above. Original launch shapes were requalified. |
| `739aff3dc5` | Adapts PR [#36014](https://github.com/sgl-project/sglang/pull/36014) | Paired Triton GDN verification and accepted-state recovery | Aligns beta rounding with packed decode, including recovery and ReplaySSM propagation, while preserving mixed-backend contracts. Focused GPU reproduction, final review, and both-profile runtime/NIXL regressions passed. |

## PRs opened by this project

Status and head commits were queried directly from GitHub on 2026-08-27.

| PR | Status | Verified head | Runtime relationship |
|---|---|---|---|
| [#36520](https://github.com/sgl-project/sglang/pull/36520) — isolate overlapping NIXL path registrations | Open, ready for review | `514d25770569256733a989ec260980744382b68d` | Refined upstream form of local `8b786639e4` |
| [#36524](https://github.com/sgl-project/sglang/pull/36524) — bound NIXL hybrid bounce transfers | Open, ready for review | `e5b4a8efeade7ed21060cb968518d93f6984f2ba` | Refined upstream form of local `067c639c0a` |
| [#35584](https://github.com/sgl-project/sglang/pull/35584) — pin NVCC host compiler | Open, ready for review | `4afec645d024d29b66fca388eec7dde4272185c1` | Upstream form of local `512a95e329` |
| [#35583](https://github.com/sgl-project/sglang/pull/35583) — independent draft config overrides | Closed, unmerged | `133cd746fd1786af3b04653fd7790181c3b69394` | Related to local `a8c4404ef8`; runtime still needs the behavior |
| [#35586](https://github.com/sgl-project/sglang/pull/35586) — ragged-verify mRoPE | Closed, unmerged | `6cd30404c587981b9a8975ead0ae0dbc16a7bb47` | Transient DSpARK/ragged work; deliberately absent from this runtime |

## Other relevant upstream work

| PR | Status / head on 2026-08-27 | Use here |
|---|---|---|
| [#30967](https://github.com/sgl-project/sglang/pull/30967) RecoverSSM | Open; `71c5de2e74c8e9b048bb62ab9cdadedb8ea83100` | Architectural basis adapted by `280825c3e2` |
| [#35744](https://github.com/sgl-project/sglang/pull/35744) fused mRoPE | Open; `9b2e053ce0d203b368e68e915667295aea24df32` | Integrated by `64ecd64924` |
| [#36497](https://github.com/sgl-project/sglang/pull/36497) Flash-Next | Open; `7c66045d71f067c1c5da2b85baad3c47d9a19cb7` | Day-zero import and local reconciliation; its SM100-only QSA throughput statement is not attributed to SM120 |
| [#36644](https://github.com/sgl-project/sglang/pull/36644) FP8 QSA scales | Open; `67f705c55e30324f047decf773b0b82d04e1ecb0` | Broader than active unit-scale correction; not integrated |
| [#35821](https://github.com/sgl-project/sglang/pull/35821) Mamba radix ghost node / track bound | Merged to Qwen optimization lineage; `b81d082abb5caec5d43a731a19b2bb898e93e3b0` | Adapted by `23e51dddcb` across Penny's additional fused/KDA paths |
| [#36806](https://github.com/sgl-project/sglang/pull/36806) exact SM120 QSA route | Merged to `qwen4-main-squashed`; `99c9362e6685db579c469f6e0e566b08827b3477` | Adapted by `8de07a058f` |
| [#35371](https://github.com/sgl-project/sglang/pull/35371) DFlash2 local convolution/selector | Merged 2026-08-19; `e5a3e4d30fa7abda95bafd2d697f9f9c48566114` | Already in integration base; core 27B DFlash2 path |
| [#35496](https://github.com/sgl-project/sglang/pull/35496) quantized DFlash2 target head | Merged 2026-08-20; `1bb02535dcb0ab03d399fd25e1595076b1db409d` | Present in base; inactive for the BF16 target `lm_head` |
| [#35663](https://github.com/sgl-project/sglang/pull/35663) Qwen3.8-27B DFlash2 recipe | Merged 2026-08-20; `34f4be339b122bac9783ea44a23097eab31ea064` | Upstream recipe lineage |

## Deliberate exclusions

Earlier DSpARK compact-ragged experiments, Mooncake code/configuration, rejected
quantized-cache experiments, private launchers, model files, raw journals, and
machine-local system service wiring are not part of this release. Historical
source remains available through older tags rather than being copied into the
current branch.
