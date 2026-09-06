# Provenance

## Source identity

| Item | Value |
|---|---|
| Upstream | `https://github.com/sgl-project/sglang.git` |
| Canonical branch | `pennyroyal-main-sm120-final` |
| Executable source HEAD | `836206a0adc8ef7aaa49f652230d5577a25014a5` |
| Integration base | `e7e78940168f3ba65c762a6f82fd8bc5b6ee04e3` |
| Local runtime commits | 26, including the fully reverted graph trial |
| Installed package | `sglang==0.5.19.dev492+g836206a0a` |
| Current release tag | `pennyroyal-v2.3.0` |

Documentation commits add the public explanation, sanitized launch recipes,
and small evidence summaries on top of the executable source. They must not
modify `python/`, `rust/`, or kernel source. The earlier 27B dated release
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

## Version 2.3 FR-Spec provenance

The adoption came from Gabriel's
[`sglang-flashnext-sm120`](https://github.com/gabrielolympie/sglang-flashnext-sm120)
at [67d2f9234fa45ae1339f0d53cd37cb695e9c6493](https://github.com/gabrielolympie/sglang-flashnext-sm120/tree/67d2f9234fa45ae1339f0d53cd37cb695e9c6493).
SGLang already contains the native-MTP FR-Spec implementation. Penny's
contribution here is the independently generated map, integration with its
existing configuration/persistence envelope, and local qualification—not
invention of FR-Spec or a claim that Gabriel's entire patch stack was imported.

The published map is byte-identical to frozen `pennyroyal-frspec-rc1-20260905`:

- Map SHA-256: `becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25`.
- Tokenizer SHA-256: `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
- 65,536 unique valid IDs, including all 21 special IDs and a 32,768-ID base.
- Frequency corpus: 7,537,952 bytes / 1,780,923 tokens of runtime source;
  evaluation cases were not used. Corpus coverage is not held-out accuracy.
- [Map manifest](configs/pennyroyal/frspec/flash-next-64k.manifest.json) records
  exact source-file byte hashes; local paths were made repository-relative
  and the checkpoint display name generalized for publication.
- The source, installed wheel and all 1,327 frozen RC artifact checksums were
  verified before publication. The immutable local RC was not rewritten.

The v2.3 speed/quality campaign kept the same Flash-Next ModelOpt NVFP4
checkpoint throughout the baseline and FR arms. Local checkpoint placement and
conversion identity remain in the retained local evidence, not the release.
Model weights are not distributed. The bundled token map is not a checkpoint.
The public launcher is a sanitized reproduction, not the host's service file;
its source/representation-derived NIXL namespace need not equal the local RC
namespace. Never point it at an existing incompatible namespace manually.

## Checkpoints

Weights are not included. The qualified model identities were:

- Flash-Next target: `RadixArk/Qwen3.8-Flash-Next-NVFP4` at
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`;
- 27B target: a Qwen3.8-27B block-FP8 checkpoint in the
  `orcarouter/Qwen3.8-27B-Uncensored-FP8` family at
  `9228df5c6c9c509e1019f83b4e085cf643118bac`;
- 27B draft: `incoai/Qwen3.8-27B-DFlash2` at
  `adde41d8fde3a75dc905a7df0bd5088d2a44b5a1`.

The v2.1.2 maintenance checks used the Orca 27B FP8 profile above and a local
ModelOpt NVFP4 conversion of `orcarouter/Qwen3.8-Flash-Next-Uncensored` source
revision `8336e613ea508b13c2159bd0f68965d97a606b95`. This is a source-specific
local conversion, not the Radix checkpoint or a newly published model artifact.
Earlier full qualification remains attached to its original checkpoints and
source revisions; v2.1.2 does not claim a new full Radix qualification.

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
