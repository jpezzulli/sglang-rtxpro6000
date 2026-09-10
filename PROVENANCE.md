# Provenance

## Source identity

| Item | Value |
|---|---|
| Upstream | `https://github.com/sgl-project/sglang.git` |
| Canonical branch | `pennyroyal-main-sm120-final` |
| Executable source HEAD | `cf811a8c5988dc87941c1442fdc8ba574a0400f7` |
| Integration base | `e7e78940168f3ba65c762a6f82fd8bc5b6ee04e3` |
| Latest runtime additions | Seven commits above v2.4.0 source, listed below; prior history is retained |
| Qualification dependency base | `sglang==0.5.19.dev492+g836206a0a` with the updated v2.4.1 Python/JIT source |
| Release | v2.4.1 |
| Git tag | `pennyroyal-v2.4.1` |

The release includes launch recipes, documentation, and measurement summaries.
v2.4.1 updates runtime source and setup guidance while retaining the dependency
stack, model settings, CPU image preprocessing by default and the pinned
[Froggeric v22.5 template](configs/pennyroyal/templates/README.md).
The earlier 27B dated release
remains reachable through `qwen38-dflash2-pro6000-20260824`; its useful results
and source lineage are also retained cumulatively in the current documentation.

For the v2.1.1 update, upstream `main` was inspected at
`cdbfe90b4a31079859817c148ef4498240ec2580` on 2026-08-29. Core Flash-Next
PR #36497 remained unmerged, so the runtime was not rebased. Version 2.1 adopts
the merged #35821 Mamba correction and the exact-SM120 QSA gate from #36806.
Version 2.1.1 adds the DFlash additive-penalty correction related to open PR
#33869 and completes merged HiCache load fencing from #36738 for Penny's active
TVM-FFI JIT transfer paths.

Version 2.1.2 adds only the one-rank sampler guard adapted from #37962 and the
Qwen3 Coder streaming separator correction adapted from #37408. The #37448
graph-lifetime trial remains in history with its complete revert; it contributes
no change to the released graph implementation. Dependencies and launch
settings are unchanged. [CHANGES.md](CHANGES.md) records the focused
two-profile regression scope; existing performance results are not refreshed.

## v2.4.1 source provenance

Seven source commits above `4aaf531cafd8` preserve the PLE adaptation and add
three narrow maintenance changes and selectable media preprocessing,
without rebasing or refreshing dependencies:

| Local commit | Upstream basis |
|---|---|
| `d9b856b2db34` | PLE hashing/gather adaptation from [#38701](https://github.com/sgl-project/sglang/pull/38701), with fallback stream-ordering corrections. |
| `3e822155a538` | Local one-warp PLE selection for measured SM120 FP8 verification shapes. |
| `50a260d297` | Grammar history trimming from [#38865](https://github.com/sgl-project/sglang/pull/38865), head `72cd382088fe3494eb52e6cdcee8ec2a4b97ce21`. |
| `6de97f90fa` | Streaming top-logprob rows from [#38759](https://github.com/sgl-project/sglang/pull/38759), head `fb848cbb749a3121c306ec9c43587561339e13a8`. |
| `1048ef671ff9` | Pinned metadata staging from [#38703](https://github.com/sgl-project/sglang/pull/38703), head `372bce7738605f15c6d71aaaba9a93873ea6c1a9`. |
| `79ec899ddab9` | Local CPU or explicit CUDA-device selection for JPEG decode and media tensor preprocessing, with cache-identity separation and unsupported-transport guards. |
| `cf811a8c5988` | Local CUDA-device validation and canonical spelling, so invalid selections fail clearly instead of falling through to JPEG fallback. |

No net PLE serving-speed gain was established. Earlier dated performance
measurements remain associated with their original source and workload.
Build/install the selected source using [BUILD.md](BUILD.md); the dependency
base package alone does not contain these changes.

Flash-Next core subsequently merged upstream through
[#37500](https://github.com/sgl-project/sglang/pull/37500) on September 8.
This release deliberately retains the qualified integration base; the older
PR-status entries below remain dated history, not current merge-status claims.

## v2.4.0 source provenance

The release adds eight reviewed source commits above `739aff3dc5`, ending at
`4aaf531cafd8`. QSA prefill preparation and short-extend bounds, no-bias routing
and GPU dependency ordering, dispatched-request cleanup, explicit Chat effort
precedence, and tool-markup handling are described with upstream credits in
[CHANGES.md](CHANGES.md#v240--maintenance-and-faster-flash-next-prefill).

The published runtime/test files match the source used for both-profile
checks. The package version in the table identifies the dependency environment,
not a prebuilt wheel containing the new changes. Build/install the release
source using [BUILD.md](BUILD.md). No new model weights, token map, compiler,
CUDA, PyTorch, FlashInfer or NIXL version is required by this update.

The prefill comparison uses source `739aff3dc5` before and `4aaf531cafd8` after,
the same Flash-Next ModelOpt NVFP4 checkpoint, and the same TP1 context/pool
configuration. It measures the combined update rather than attributing a
percentage to any one commit. [RESULTS.md](RESULTS.md#pennyroyal-v240--prefill-and-maintenance)
distinguishes server prefill from client time to first token.

## v2.3.1 maintenance provenance

That release's only runtime change was `739aff3dc5`, adapted from SGLang
[#36014](https://github.com/sgl-project/sglang/pull/36014) by V-aerus
(Hangshuai He), head `bf5d4227dde4e532b4668327a8d6dcc4d79d9c2b`.
It aligns paired Triton GDN verification and recovery with packed decode's
beta rounding. Flash-Next's selected FlashInfer/WY path is unchanged.

The published six-file source/test change is byte-identical to the reviewed
and tested change. Tests loaded these updated Python/Triton sources on the
existing v2.3 environment; no new prebuilt wheel is distributed. Build/install
the release source using [BUILD.md](BUILD.md). The base package version above
describes that test environment, not a wheel containing the new correction.

The reported failure was not reproduced live, but focused SM120 GPU tests
reproduced the underlying mismatch. Focused tests and both-profile runtime/
NIXL regressions passed; [CHANGES.md](CHANGES.md#v231--gdn-rounding-maintenance)
records the scope. No new performance claim or model qualification is added.

## v2.3 FR-Spec provenance

Pennyroyal's Flash-Next FR-Spec configuration builds on the reduced
draft-vocabulary work in Gabriel's
[`sglang-flashnext-sm120`](https://github.com/gabrielolympie/sglang-flashnext-sm120),
[source revision 67d2f9234fa45ae1339f0d53cd37cb695e9c6493](https://github.com/gabrielolympie/sglang-flashnext-sm120/tree/67d2f9234fa45ae1339f0d53cd37cb695e9c6493).
It uses SGLang's existing native-MTP FR-Spec implementation and a
Pennyroyal-generated token map.

### Token map

- Map SHA-256: `becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25`.
- Tokenizer SHA-256: `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
- 65,536 unique valid IDs, including all 21 special IDs and a 32,768-ID base.
- Frequency corpus: 7,537,952 bytes / 1,780,923 tokens of runtime source.
  Evaluation cases were excluded.
- [Map manifest](configs/pennyroyal/frspec/flash-next-64k.manifest.json):
  repository-relative source paths, source revision, and byte hashes.
- [Map builder](scripts/pennyroyal/frspec/build_token_map.py):
  deterministic token selection from a supplied tokenizer and corpus.

Corpus coverage measures token coverage of the source corpus, not held-out
model accuracy. The map changes only draft proposals; target vocabulary and
acceptance policy are unchanged.

The launcher checks map and tokenizer hashes and incorporates the map hash
into the NIXL namespace. Use the generated namespace instead of assigning an
existing cache directory manually. Model weights are not included.

## Checkpoints

Weights are not included. The qualified model identities were:

- Flash-Next target: `RadixArk/Qwen3.8-Flash-Next-NVFP4` at
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`;
- 27B target: a Qwen3.8-27B block-FP8 checkpoint in the
  `orcarouter/Qwen3.8-27B-Uncensored-FP8` family at
  `9228df5c6c9c509e1019f83b4e085cf643118bac`;
- 27B draft: `incoai/Qwen3.8-27B-DFlash2` at
  `adde41d8fde3a75dc905a7df0bd5088d2a44b5a1`.

The v2.1.2 maintenance tests covered both 27B FP8/DFlash2 and Flash-Next
NVFP4/native MTP. Full reasoning, tools, and vision were not repeated for
that update. Historical measurements retain their original dates and source
revisions; the v2.3 results are listed separately.

The launch recipes require local checkpoint paths. The namespace helper records
checkpoint metadata and Hugging Face revision/LFS content identity when
available, falling back to full weight hashing. A changed checkpoint therefore
selects a different persistent-cache namespace.

## Runtime lineage

Core Qwen3.8 DFlash2 support, including local convolution and candidate
selection from merged SGLang PR #35371 and quantized target-head support from
merged PR #35496, is already present in the upstream integration base. The
selected target keeps `lm_head` in BF16. Local source adds fixed-width XQA mask
handling, independent draft overrides, NIXL correctness fixes, three-axis
mRoPE, and request-span observability.

Flash-Next entered through the then-current net work of PR #36497 and was
reconciled against the active upstream base. Bounded Qwen4, QSA, RecoverSSM,
HiCache, PLE, and mRoPE corrections were layered on that same source. Exact
relationships and PR status are in [CHANGES.md](CHANGES.md).

PR #36497's QSA throughput statement originated while its resolver was
SM100-only. Penny's SM120 wrapper dispatch resolves to XQA, so the SM100
TRTLLM-Gen attribution is not part of this runtime's provenance.

## Evidence handling

This repository contains compact, non-private summaries only. Raw service
journals, prompts, conversations, model files, cache payloads, local hostnames,
tokens, and private endpoints are deliberately excluded. Detailed public test
logs and the maintained validation harness live in
[`jpezzulli/pennyroyal-validation`](https://github.com/jpezzulli/pennyroyal-validation).
