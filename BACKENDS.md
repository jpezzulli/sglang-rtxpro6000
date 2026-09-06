# Resolved Backends and SM120 Enablement

These tables describe implementations observed in startup/runtime logs and
confirmed in this source. “Explicit” means the launcher selected a narrow
phase-specific backend; “source” means local dispatch enabled an already
available path. Requested flags alone were not treated as resolution evidence.

## Version 2.3: existing FR-Spec path

The additional Flash-Next recipe selects SGLang's existing native-MTP token-map
path: a 65,536-row BF16 draft head proposes tokens, remapped to full target IDs.
Target vocabulary, target verification and acceptance policy stay unchanged.
No attention/GDN/MoE backend switch or BF16 low-M kernel is included; the
resolved backends below remain in use. This is isolated from the unchanged
27B/DFlash2 recipe. See [PROVENANCE.md](PROVENANCE.md#version-23-fr-spec-provenance)
for source credit and map identity.

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
| QSA sparse decode on SM120 | FlashInfer QSA wrapper resolving to XQA | Source/wrapper dispatch | `c1da0eef56`; direct backend probe and live decode |
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

QSA decode is a distinct case. Commit `c1da0eef56` lets SM120 enter
FlashInfer's page-aligned QSA wrapper, but the wrapper's SM12x dispatch selects
XQA—not TRTLLM-Gen. The throughput statement inherited from PR #36497 was
measured while the resolver was SM100-only and is not evidence for SM120. There
is no matched end-to-end XQA-versus-fallback percentage claim here.

Direct probing also established that TRTLLM-Gen is not merely hidden behind a
conservative gate. Forced selection reports `Unsupported architecture`; after
bypassing that guard, the CUDA driver rejects the exact QSA cubin with
`CUDA_ERROR_NO_BINARY_FOR_GPU (209)`. It targets SM100 and contains the
`sm10x_tcgen05` instruction family unavailable on SM120. Future support needs
kernel/compiler adaptation upstream of SGLang; see
[TensorRT-LLM #11799](https://github.com/NVIDIA/TensorRT-LLM/issues/11799) and
[FlashInfer #3628](https://github.com/flashinfer-ai/flashinfer/issues/3628).

Commit `95da38fb3b` separately fixes FP8 sparse-prefill tile interpretation for
the active unit-scale checkpoint. It does not carry the broader
calibrated-scale work from open PR #36644.

Gates intentionally left untouched include FlashInfer state-writing GDN
verification, FlashInfer HyperConnection Mix, inactive shared-expert fusion,
and auto-selected FP4/BF16 GEMM runners.

## Qwen3.8-27B with DFlash2

| Component or phase | Resolved implementation | Selection | Change / evidence |
|---|---|---|---|
| Target prefill attention | FlashInfer | Explicit hybrid backend | 64K/490K tests |
| Target decode attention | TRTLLM-MHA with XQA | Explicit hybrid decode backend | controlled and real agentic decode |
| Fixed-width target full-attention verify | TRTLLM-MHA/XQA with packed causal mask | Explicit + source mask fix | `0f159cd545`; graph metadata regression |
| DFlash2 draft attention | FlashInfer | Explicit | draft-runner startup line |
| DFlash2 local convolution | upstream DFlash2 path | Upstream base | merged PR #35371 |
| Candidate selector | folded into draft CUDA graph | Upstream base | startup graph line |
| DFlash fused KV materialization | Enabled | Upstream base | five-layer/8-head startup line |
| Target and draft KV | FP8 E4M3 | Explicit | 1,118,784-token startup pools |
| GDN decode/prefill/state verify | Triton | Resolved linear backend | startup args and qualified run |
| Target routed-expert MoE | Triton FP8 MoE | Auto resolves to Triton because A2A is `none` | server args plus `Fp8MoEMethod.create_moe_runner` |
| Target prefill / verify / draft graphs | Breakable prefill; full fixed-width verify | Automatic capture | startup graph-capture lines and live `cuda graph: True` |
| Multimodal attention | `triton_attn` | Automatic | startup log; mRoPE fixed by `64ecd64924` |
| Sampling / grammar | FlashInfer / XGrammar | Automatic | startup args |
| HiCache transfer | NIXL POSIX | Explicit | persistent restore qualification |
| Persistent state | target KV, Mamba/GDN, DFlash2 sidecar | Local NIXL integration | `8b786639e4`, `067c639c0a` |

No separate local DeepGEMM SM120 patch is part of this 19-commit runtime stack.
With `moe_runner_backend=auto` and A2A `none`, the active FP8 method's source
resolver selects Triton rather than DeepGEMM. The DFlash2 support used here
comes from the upstream base plus the explicit XQA mask and NIXL fixes above.
