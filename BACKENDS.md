# Resolved Backends and SM120 Enablement

These tables describe implementations observed in startup/runtime logs and
confirmed in this source. “Explicit” means the launcher selected a narrow
phase-specific backend; “source” means local dispatch enabled an already
available path. Requested flags alone were not treated as resolution evidence.

## Qwen3.8 Flash-Next

| Component or phase | Resolved implementation | Selection | Change / evidence |
|---|---|---|---|
| Target GDN decode | `FlashInferGDNKernel` | Explicit | BF16 launcher override; startup dispatcher table |
| Native-MTP draft decode | `FlashInferGDNKernel` | Explicit/shared dispatch | BF16 launcher override; draft graph capture |
| Target GDN prefill | `FlashInferGDNKernel` | Explicit | 64K/490K exact tests |
| Native-MTP draft extend | `FlashInferGDNKernel` | Explicit/shared dispatch | Native-MTP graph capture |
| Active target MTP verification | FlashInfer WY output-only | Source | `280825c3e2`, PR #30967 adaptation; state-parity suite |
| Ordinary/tree state-writing verification | `TritonGDNKernel` | Preserved fallback | SM120 FlashInfer full-state gate deliberately retained |
| Accepted-state recovery | FlashInfer WY output-only | Source | `280825c3e2`; recovery graphs BS 1-4 |
| QSA sparse prefill | Triton sparse GQA | Model path | `95da38fb3b` unit-scale FP8 tile fix; long prefill |
| QSA decode | TRTLLM-Gen | Source | `c1da0eef56` enables the supported SM120 sparse-decode dispatch |
| QSA top-k | `sgl-kernel` | Automatic | Startup args and live decode |
| QSA MTP index sharing | Enabled | Model path | Startup log and shared-index tests |
| Target MoE | FlashInfer CUTLASS | Automatic | SM120 modelopt-FP4 resolution in startup log |
| Native-MTP MoE | FlashInfer CUTLASS | Automatic | resolved speculative MoE backend |
| HyperConnection Mix | persistent Triton Mix | Automatic fallback | FlashInfer/CuTe path remains SM100-only |
| HyperConnection Combine | fused SGLang CUDA | Model path | imported Qwen4 kernel tests |
| Multimodal attention | `triton_attn` | Automatic | vision qualification |
| Sampling | FlashInfer | Automatic | startup args |
| Grammar | XGrammar | Automatic | startup args and tool suite |
| HiCache transfer | NIXL POSIX | Explicit | restart restoration |
| Persistent state | packed target/native-MTP KV, full GDN/PLE siblings, compressed QSA keys | Local integration | `516e42a2ee` through `7b5cfb728d` |

### How the SM120 gates were handled

1. **Launcher override.** Automatic linear-attention selection used an SM100
   helper and did not choose FlashInfer on SM120. The installed FlashInfer
   decode and prefill implementations already supported the required BF16
   state. Narrow `--linear-attn-decode-backend flashinfer` and
   `--linear-attn-prefill-backend flashinfer` overrides selected only those
   phases. Long prefill, decode, and exact needles qualified them.

2. **Narrow source enablement.** RecoverSSM needed WY output-only state output
   and accepted-state recovery in `gdn_mtp_cache_mode=none`. FlashInfer 0.6.17
   already shipped the `sm_120a` kernel; SGLang did not route SM120 into it.
   Commit `280825c3e2` opens only this output/recovery path and preserves Qwen4
   PLE accepted-state commit ordering. Direct SM120 tests cover every accepted
   length for four draft tokens, batch sizes 1-4, mixed acceptance, positions
   63/64/65, `extra_buffer`, continuation, simulated restore/retraction, and
   mutable CUDA-graph replay.

3. **Gate deliberately preserved.** Ordinary FlashInfer state-writing target
   verification was not globally enabled on SM120. Full/tree state-writing
   verification remains Triton. The runtime log label
   `FlashInferGDNKernel (none-mode WY output-only)` describes the active special
   mode; it is not evidence of broad FlashInfer verification support.

QSA decode had a similar wiring gap: the underlying TRTLLM-Gen sparse path was
available, while SGLang's dispatch excluded SM120. Commit `c1da0eef56` narrows
the supported architecture test and adds SM120 dispatch coverage. Commit
`95da38fb3b` fixes FP8 sparse-prefill tile interpretation for the active
unit-scale checkpoint. It does not carry the broader calibrated-scale work from
open PR #36644.

Gates intentionally left untouched include FlashInfer state-writing GDN
verification, FlashInfer HyperConnection Mix, inactive shared-expert fusion,
and auto-selected FP4/BF16 GEMM runners.

## Qwen3.8-27B with DFlash2

| Component or phase | Resolved implementation | Selection | Change / evidence |
|---|---|---|---|
| Target prefill attention | FlashInfer | Explicit hybrid backend | 64K/490K tests |
| Target decode / fixed-width verify | TRTLLM-MHA with XQA | Explicit + source mask fix | `0f159cd545`; graph metadata regression |
| DFlash2 draft attention | FlashInfer | Explicit | draft-runner startup line |
| DFlash2 local convolution | upstream DFlash2 path | Upstream base | merged PR #35371 |
| Candidate selector | folded into draft CUDA graph | Upstream base | startup graph line |
| DFlash fused KV materialization | Enabled | Upstream base | five-layer/8-head startup line |
| Target and draft KV | FP8 E4M3 | Explicit | 1,118,784-token startup pools |
| GDN decode/prefill/verify | Triton | Explicit/default | startup args and qualified run |
| Multimodal attention | `triton_attn` | Automatic | startup log; mRoPE fixed by `64ecd64924` |
| Sampling / grammar | FlashInfer / XGrammar | Automatic | startup args |
| HiCache transfer | NIXL POSIX | Explicit | persistent restore qualification |
| Persistent state | target KV, Mamba/GDN, DFlash2 sidecar | Local NIXL integration | `8b786639e4`, `067c639c0a` |

No separate local DeepGEMM SM120 patch is part of this 19-commit runtime stack.
The DFlash2 support used here comes from the upstream base plus the explicit
XQA mask and NIXL fixes above; this document does not infer a DeepGEMM backend
from older experiments.
