# Provenance

## Source identity

| Item | Value |
|---|---|
| Upstream | `https://github.com/sgl-project/sglang.git` |
| Canonical branch | `pennyroyal-main-sm120-final` |
| Executable source HEAD | `64ecd64924fee338e3bf846a32167cd604186827` |
| Integration base | `e7e78940168f3ba65c762a6f82fd8bc5b6ee04e3` |
| Local runtime commits | 19 |
| Installed package | `sglang==0.5.19.dev485+g64ecd6492` |

Documentation commits add the public explanation, sanitized launch recipes,
and small evidence summaries on top of the executable source. They must not
modify `python/`, `rust/`, or kernel source. The earlier 27B dated release
remains reachable through `qwen38-dflash2-pro6000-20260824`; its useful results
and source lineage are also retained cumulatively in the current documentation.

Current upstream `main` was inspected at
`20a491d1d311553bbab3f22e19bbafb86ef3c0cc` on 2026-08-27. `git cherry`
reported no patch-equivalent for the 19 local commits. That is a source-state
observation, not a claim that upstream lacks related implementations.

## Checkpoints

Weights are not included. The qualified model identities were:

- Flash-Next target: `RadixArk/Qwen3.8-Flash-Next-NVFP4` at
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`;
- 27B target: a Qwen3.8-27B block-FP8 checkpoint in the
  `orcarouter/Qwen3.8-27B-Uncensored-FP8` family at
  `9228df5c6c9c509e1019f83b4e085cf643118bac`;
- 27B draft: `incoai/Qwen3.8-27B-DFlash2` at
  `adde41d8fde3a75dc905a7df0bd5088d2a44b5a1`.

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

PR #36497's approximately 35% QSA decode statement originated while its
resolver was SM100-only. Penny's SM120 wrapper dispatch resolves to XQA, so the
SM100 TRTLLM-Gen attribution is not part of this runtime's provenance.

## Evidence handling

This repository contains compact, non-private summaries only. Raw service
journals, prompts, conversations, model files, cache payloads, local hostnames,
tokens, and private endpoints are deliberately excluded. Detailed public test
logs and the maintained validation harness live in
[`jpezzulli/pennyroyal-validation`](https://github.com/jpezzulli/pennyroyal-validation).
