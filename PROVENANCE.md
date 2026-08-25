# Provenance

This repository is a frozen, known-good SGLang-derived runtime distribution.
It is not current SGLang `main`, a rebased patch series, or a claim that every
local commit is suitable for upstream. The executable Python and Rust source is
preserved exactly because that source produced the measurements in this repo.

## Source identity

| Item | Value |
|---|---|
| Upstream | `https://github.com/sgl-project/sglang.git` |
| Runtime branch | `pennyroyal-main-sm120-final` |
| Runtime source HEAD | `8e197ed3afc559f29562a2e7de9026f011f5d28f` |
| Merge base with upstream | `5a7b26c636deb2def43640bab6c63146dbe536dc` |
| Upstream reference fetched for this audit | `effe0d14d2e6815c1385df47258985c74d8a3d07` |
| Divergence on 2026-08-24 EDT | 7 commits ahead, 210 commits behind |
| Runtime worktree at capture | clean; no untracked non-ignored files |

Documentation and sanitized evidence are added after the runtime source commit.
They do not modify the executable tree. Before publication, this relationship
must remain true:

```bash
git diff --exit-code 8e197ed3afc559f29562a2e7de9026f011f5d28f -- python rust
```

## Local commit stack

All seven commits are preserved in history. `git cherry` found no
patch-equivalent commit in the fetched upstream reference, but that does not
mean every commit is still active or that upstream lacks related functionality.

| Commit | Role in this distribution | Active in the measured runtime? |
|---|---|---|
| `560f06a94e` | Independent target/draft Hugging Face config overrides | Yes. The nested Qwen3.8 target override and flat DFlash2 override differ. |
| `b74d971054` | Preserve draftless custom speculative-algorithm hooks | No known effect on this DFlash2 run, which has an explicit draft checkpoint. |
| `5b06cf9f42` | Pass the fingerprinted `CXX` to NVCC through `-ccbin` | Yes for build/JIT reproducibility on this GCC 15 + CUDA 13.3 host. |
| `e9971ac6eb` | Log queue, initial-prefill, post-prefill, and total-forward elapsed spans | Yes for measurement. It is observability, not a speed optimization. |
| `ba600c682a` | Mooncake hybrid-I/O corrections | No. Mooncake was rejected and removed from the deployed path; the code remains because history is frozen. |
| `2e79cdc030` | Supply the packed causal mask required by fixed-width XQA speculative verification | Yes. This preserves the SM120 TRTLLM-MHA/XQA target-verify path. |
| `8e197ed3af` | Bound bounce-backed NIXL transfers and isolate concurrent path-mode registrations | Yes for HiCache/NIXL persistence and concurrent restore correctness; not for GPU-resident decode speed. |

## DFlash2 lineage

Core DFlash2 support is not a private reimplementation in this tree. The
upstream base already contains the important Qwen3.8 DFlash2 work, including:

- `c14312a664` / SGLang PR #35371: local convolution and candidate selector;
- `1cf2b8c54d` / SGLang PR #35496: quantized target `lm_head` selector support;
- `d9f6861359` / SGLang PR #35663: Qwen3.8 DFlash2 recipe documentation;
- DFlash page-aligned draft reservation and `extra_buffer_lazy` Mamba support.

The selected target keeps `lm_head` in BF16, so the quantized-head path is
present but is not what makes this checkpoint fast. The local performance-
relevant addition is the fixed-width XQA verification mask, plus the
configuration that selects TRTLLM-MHA/XQA for target decode and verification.

Earlier local DSpARK compact-ragged and mRoPE experiments are not part of this
seven-commit distribution and are not represented as requirements for DFlash2.

## Installed runtime versus source checkout

The running service imports SGLang from:

```text
/opt/sglang/.venv/lib/python3.12/site-packages/sglang
```

The installed wheel metadata is
`sglang==0.5.18.dev891+g2e79cdc03`, built at commit `2e79cdc030`. The two NIXL
adapter files from the later `8e197ed3af` commit were then installed over that
wheel. Every Python file changed by the seven-commit stack was compared with
the source checkout and was byte-identical at capture. A clean build from this
repository avoids that historical two-stage installation while producing the
same executable source.

## Checkpoints

Weights are not included. Exact repository revisions and LFS SHA-256 values are
recorded in [evidence/models/checkpoints.json](evidence/models/checkpoints.json).

- Target: `orcarouter/Qwen3.8-27B-Uncensored-FP8` at
  `9228df5c6c9c509e1019f83b4e085cf643118bac`.
- Draft: `incoai/Qwen3.8-27B-DFlash2` at
  `adde41d8fde3a75dc905a7df0bd5088d2a44b5a1`.

Both model cards declare Apache-2.0. Users must obtain and review the model
cards themselves. The target is an abliterated/uncensored block-FP8 derivative;
that fact is material to quality and behavior comparisons.

## NIXL

NIXL was built from `ai-dynamo/nixl` commit
`aecbc3846d92c34c7507a58d776e1fda50ff4fba`, clean upstream `main` at build
time. It is a separate Apache-2.0 project and is not vendored here. Build
identity is in [evidence/environment/nixl-build.txt](evidence/environment/nixl-build.txt).

## Licensing

The preserved SGLang source and this repository use the existing Apache-2.0
license in [LICENSE](LICENSE). Existing third-party notices and component
licenses remain in the source tree. No model weights, NIXL source, cache data,
private prompts, or conversation transcripts are included.
