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

The publication commit adds documentation, sanitized launch recipes, and small
evidence summaries on top of the executable source. It must not modify
`python/`, `rust/`, or kernel source. The old frozen release remains reachable
through its historical tag; it is not duplicated inside this release.

Current upstream `main` was inspected at
`20a491d1d311553bbab3f22e19bbafb86ef3c0cc` on 2026-08-27. `git cherry`
reported no patch-equivalent for the 19 local commits. That is a source-state
observation, not a claim that upstream lacks related implementations.

## Checkpoints

Weights are not included. The qualified model identities were:

- Flash-Next target: `RadixArk/Qwen3.8-Flash-Next-NVFP4`;
- 27B target: a Qwen3.8-27B block-FP8 checkpoint in the
  `orcarouter/Qwen3.8-27B-Uncensored-FP8` family;
- 27B draft: `incoai/Qwen3.8-27B-DFlash2`.

The launch recipes require local checkpoint paths. The namespace helper records
checkpoint metadata and Hugging Face revision/LFS content identity when
available, falling back to full weight hashing. A changed checkpoint therefore
selects a different persistent-cache namespace.

## Runtime lineage

Core Qwen3.8 DFlash2 support, including the local-convolution and candidate
selector work from merged SGLang PR #35371, is already present in the upstream
integration base. The local source adds fixed-width XQA mask handling,
independent draft overrides, NIXL correctness fixes, and shared observability.

Flash-Next entered through the then-current net work of PR #36497 and was
reconciled against the active upstream base. Bounded Qwen4, QSA, RecoverSSM,
HiCache, PLE, and mRoPE corrections were layered on that same source. Exact
relationships and PR status are in [CHANGES.md](CHANGES.md).

## Evidence handling

This repository contains compact, non-private summaries only. Raw service
journals, prompts, conversations, model files, cache payloads, local hostnames,
tokens, and private endpoints are deliberately excluded. Detailed public test
logs and the maintained validation harness live in
[`jpezzulli/pennyroyal-validation`](https://github.com/jpezzulli/pennyroyal-validation).
