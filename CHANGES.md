# What changed since the frozen build

The published runtime is the ordered range
`e7e78940168f..64ecd64924fe`. All 19 commits remain in source history. Current
upstream `main` was fetched at `20a491d1d311553bbab3f22e19bbafb86ef3c0cc`;
none of the commits was patch-equivalent there on 2026-08-27.

“Local” does not mean a permanent fork requirement. It means the exact active
commit has not merged upstream. Where a PR has a later refined head, that is
shown separately rather than pretending the local commit and PR head are
identical.

## Ordered runtime changes

| Local commit | Disposition / upstream relationship | Affected path | Why and focused evidence |
|---|---|---|---|
| `7e4c212f7d` | Adapted import from open PR [#36497](https://github.com/sgl-project/sglang/pull/36497) | Flash-Next/Qwen4, QSA, PLE, native MTP, HC | Day-zero model implementation. Imported QSA, HC, PLE, MTP, memory-pool and model tests; full Flash-Next qualification. |
| `c1da0eef56` | Local bounded SM120 enablement | Flash-Next QSA decode | Routes the existing TRTLLM-Gen sparse-decode path on SM120; focused QSA dispatch tests plus live decode. |
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
| [#36497](https://github.com/sgl-project/sglang/pull/36497) Flash-Next | Open; `7c66045d71f067c1c5da2b85baad3c47d9a19cb7` | Day-zero import and local reconciliation |
| [#36644](https://github.com/sgl-project/sglang/pull/36644) FP8 QSA scales | Open; `67f705c55e30324f047decf773b0b82d04e1ecb0` | Broader than active unit-scale correction; not integrated |
| [#35371](https://github.com/sgl-project/sglang/pull/35371) DFlash2 local convolution/selector | Merged 2026-08-19; `e5a3e4d30fa7abda95bafd2d697f9f9c48566114` | Already in integration base; core 27B DFlash2 path |

## Deliberate exclusions

Earlier DSpARK compact-ragged experiments, Mooncake code/configuration, rejected
quantized-cache experiments, private launchers, model files, raw journals, and
machine-local system service wiring are not part of this release. Historical
source remains available through older tags rather than being copied into the
current branch.
