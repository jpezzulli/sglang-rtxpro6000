# SGLang on one RTX PRO 6000 Blackwell

This repository is the canonical source for a tuned SGLang runtime used on one
96 GB NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120). The same
patched runtime supports two independently qualified launch configurations:

- `RadixArk/Qwen3.8-Flash-Next-NVFP4` with native NEXTN MTP;
- Qwen3.8-27B FP8 with `incoai/Qwen3.8-27B-DFlash2`.

Flash-Next support is day-one integration work. It is heavily qualified on this
specific system, including 524K context, multimodal requests, reasoning, tools,
agentic workloads, speculative recovery, and persistent prefix restoration. It
can still contain model- or hardware-specific assumptions and should not be
read as a claim of general SM120 support.

## Published runtime

| Item | Value |
|---|---|
| Release tag | `sglang-rtxpro6000-20260827` |
| Release source | `64ecd64924fee338e3bf846a32167cd604186827` |
| Upstream integration base | `e7e78940168f3ba65c762a6f82fd8bc5b6ee04e3` |
| SGLang package | `0.5.19.dev485+g64ecd6492` |
| Python / PyTorch | `3.12.13` / `2.13.0+cu130` |
| CUDA / compiler | CUDA `13.3` (NVCC `13.3.73`) / GCC `15.3.1` |
| FlashInfer / NIXL | `0.6.17` / `1.4.0` |
| GPU / driver | RTX PRO 6000 96 GB, SM120 / `610.57.04` |

The executable tree is upstream SGLang plus the ordered local commits listed in
[CHANGES.md](CHANGES.md). Some changes were imported or adapted from open
upstream PRs; three portable fixes were submitted by this project. The source
history, rather than a parallel patch directory, is the authoritative runtime.

## Feature matrix

| Capability | Flash-Next | Qwen3.8-27B + DFlash2 |
|---|---|---|
| Weights / compute / KV | NVFP4 / BF16 / FP8 E4M3 | block-FP8 / BF16 / target+draft FP8 E4M3 |
| Speculation | Native NEXTN, 4 draft tokens | DFlash2, 8 draft tokens, 2,048-token window |
| Served context | 524,288, factor-2 YaRN | 524,288, factor-2 YaRN target+draft |
| Linear state | BF16 GDN, 24 Mamba slots | FP32 GDN, 24 Mamba slots, 5 retained states/path |
| Verification state | RecoverSSM `none`; WY output-only recovery | DFlash2 fused KV materialization and compact draft cache |
| Sparse attention | QSA Triton prefill, TRTLLM-Gen decode, shared MTP index | Not applicable |
| HiCache/NIXL | 32 GB host tier; packed target/native-MTP KV, GDN, PLE, QSA keys | 96 GB host tier; target KV, Mamba/GDN, DFlash2 sidecar |
| Persistent prefix reuse | Qualified across service restart | Qualified on the 27B runtime line |
| Multimodal | Vision and three-axis mRoPE qualified | Vision mRoPE corrected by the same runtime source |
| Reasoning / tools | Qualified | Qualified |

The two configurations share a runtime, not benchmark results. Flash-Next-only
QSA, PLE, and RecoverSSM behavior must not be inferred for the 27B model.

## Flash-Next final qualification

| Measurement | Result |
|---|---:|
| 64K cold prefill | 10,103.70 tok/s |
| ~490K cold prefill | 7,872.15 tok/s; 3/3 exact needles |
| 1x decode | 171.09 tok/s |
| 4x synchronized batch | 427.54 tok/s aggregate |
| Individual post-first-token streams | 115.13, 127.56, 126.64, 122.96 tok/s |
| MTP accepted length / rate | 2.58 / 52.74% |
| Reasoning | 97.49/100 across 139,863 completion tokens |
| Tools | 30/30 semantically correct; 27/30 literal-checker exact |
| Vision | Complete 1,024-token validation passed |
| Sealed agentic control | 148.80 tok/s |
| Natural 3,072-token decode | 162.05 tok/s |

The 427.54 tok/s figure is `4,096 completion tokens / 9.580393 s` for the
synchronized four-request batch, including TTFT and the batch tail. Each listed
stream rate uses that request's own post-first-token interval. The values are
therefore deliberately not additive.

After a service restart, HiCache/NIXL restored 63,808 of 63,864 tokens in the
64K case and 489,856 of 489,879 tokens in the long case. Only 56 and 23 tokens
were recomputed. The long restored-prefix request measured 62,040.60 effective
input tok/s and retained all three exact needles. This is restored-prefix
throughput, not cold model prefill throughput.

## Real agentic workload

This is separate from controlled qualification. A decontaminated window of 96
completed requests produced 100,666 tokens at 139.5 tok/s token-weighted,
153.5 tok/s median, and 155.1 tok/s arithmetic mean. The sustained completed-
request peak was 218.8 tok/s; instantaneous telemetry reached 247.0 tok/s for
one stream and briefly 543.1 tok/s with four concurrent requests. The 90K-279K
input band measured 138.7 tok/s weighted and 148.4 tok/s median. There was no
high-context cliff; verification frequency declined modestly and speculative
acceptance explained most visible variation. The first sample after a large
prefill was excluded because its telemetry window mixed prefill or idle time
with decode.

## Current Flash-Next capacity

- served context: 524,288 tokens;
- automatically sized GPU KV pool: 824,384 tokens;
- Mamba slots: 24;
- recovery graphs: captured for batch sizes 1-4 and active;
- endpoint model name: `pennyroyal`;
- final post-restart logs: no CUDA error, OOM, retraction, traceback, or
  exception.

See [MEMORY-AND-PERSISTENCE.md](MEMORY-AND-PERSISTENCE.md) for allocation and
restart details, including a candid allocator-estimator limitation.

## Reproduce or inspect

- [BUILD.md](BUILD.md) — native build and exact dependency envelope.
- [RUN.md](RUN.md) — the two sanitized launch configurations and validation.
- [BACKENDS.md](BACKENDS.md) — resolved implementations and narrow SM120 gates.
- [RESULTS.md](RESULTS.md) — controlled, persistent-cache, agentic, and 27B data.
- [CHANGES.md](CHANGES.md) — local source changes and verified upstream status.
- [PROVENANCE.md](PROVENANCE.md) — source, package, checkpoint, and release identity.
- [LIMITATIONS.md](LIMITATIONS.md) — known portability and evidence boundaries.

The reusable launchers are in [`configs/pennyroyal`](configs/pennyroyal), and
the deterministic NIXL namespace helper is in
[`scripts/pennyroyal`](scripts/pennyroyal). Model weights, private prompts, raw
conversation logs, persistent cache contents, and host-specific secrets are not
included.

## How I got here

The engineering path—including experiments, regressions, DFlash2 work, and the
HiCache/Mooncake/NIXL detour—is documented at
[msoexpert.com](https://msoexpert.com/articles/qwen38-dflash2-rtx-pro-6000/).

## License

The SGLang-derived source remains under the repository's Apache-2.0 license.
Third-party runtimes and model checkpoints retain their own licenses.
