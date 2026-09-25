# SGLang Blackwell SM120 backends for Qwen3.8

These tables map the SGLang backends resolved for Qwen3.8-27B/DFlash2 and
Qwen3.8 Flash-Next on NVIDIA RTX PRO 6000 Blackwell (SM120). Startup/runtime
logs and source dispatch establish the active implementation. “Explicit” means
the launcher selects a phase-specific backend; “source” means Pennyroyal adds
the dispatch path.

## v2.5.0 optional Flash-Next paths

The default Flash-Next backend resolution remains the v2.4.1 layout described
below. Two independent options change bounded parts of that profile:

| Option | Resolved implementation | What remains unchanged |
|---|---|---|
| `SGLANG_SM120_ONLINE_MXFP8=true` | Eligible otherwise-unquantized transformer projections use FlashInfer CUTLASS MXFP8 weights and dynamic activations. HyperConnection mix and `lm_head` use row-wise FP8 weights with per-output-row scales. | Checkpoint NVFP4 experts and routers, PLE, QSA, BF16 GDN/convolution state, FP8 KV, native NEXTN and FR-Spec mapping/scale alignment. QSA and GDN state formats stay unchanged. |
| `PENNY_PLE_BACKEND=nvme` | The attributed SSD Stream reader and existing PLE graph adapter stage rows from a prepared immutable local-NVMe table. | PLE values and precision, hash calculation, QSA, native MTP, attention, MoE and recurrent-state backends. RAM PLE remains the default. |

Online FP8 supports exact SM120 and recognized Flash-Next modules; other
hardware and selected-module shapes fail at startup. NVMe PLE changes table
placement while keeping the existing PLE values and math. Its plugin loads only
when selected.
See [FP8.md](FP8.md) and [NVME-PLE.md](NVME-PLE.md).

## v2.3 FR-Spec

A 65,536-row BF16 draft head proposes tokens through SGLang's native-MTP
FR-Spec path. Draft IDs are mapped back to full target IDs before
verification. Target vocabulary and acceptance policy are unchanged.

The attention, GDN, MoE, and verification backends below remain in use.
The 27B/DFlash2 launcher is unchanged. Source credit and map identity are
listed in [PROVENANCE.md](PROVENANCE.md#v23-fr-spec-provenance).

v2.4.0 keeps those backend selections. It reduces QSA prefill preparation,
corrects short-extend bounds, removes unnecessary softmax-router allocation,
and waits for GPU input dependencies before routing reads. The dense 27B model
uses its own attention and feed-forward paths.

## Qwen3.8 Flash-Next

| Component or phase | Resolved implementation | Selection | Change / evidence |
|---|---|---|---|
| Target GDN decode | `FlashInferGDNKernel` | Explicit | BF16 launcher override; startup dispatcher table |
| Native-MTP draft decode | `FlashInferGDNKernel` | Explicit/shared dispatch | BF16 launcher override; draft graph capture |
| Target GDN prefill | `FlashInferGDNKernel` | Explicit | 64K/490K exact tests |
| Native-MTP draft extend | `FlashInferGDNKernel` | Explicit/shared dispatch | Native-MTP graph capture |
| Active target MTP verification | FlashInfer WY output-only | Source | `280825c3e2`, PR #30967 adaptation; state-parity suite |
| Ordinary/tree state-writing verification | `TritonGDNKernel` | Preserved fallback | SM120 FlashInfer full-state gate remains in place |
| Accepted-state recovery | FlashInfer WY output-only | Source | `280825c3e2`; recovery graphs BS 1-4 |
| QSA sparse prefill | Triton sparse GQA | Model path | `95da38fb3b` unit-scale FP8 tile fix; long prefill |
| QSA sparse decode on SM120 | FlashInfer QSA wrapper resolving to XQA | Source/wrapper dispatch | `c1da0eef56`; direct backend probe and live decode |
| QSA top-k | `sgl-kernel` | Automatic | Startup args and live decode |
| QSA MTP index sharing | Enabled | Model path | Startup log and shared-index tests |
| Target MoE | FlashInfer CUTLASS | Automatic | SM120 modelopt-FP4 resolution in startup log |
| Native-MTP MoE | FlashInfer CUTLASS | Automatic | resolved speculative MoE backend |
| HyperConnection Mix | persistent Triton Mix | Automatic fallback | FlashInfer/CuTe path remains SM100-only |
| Optional online-FP8 projections | FlashInfer CUTLASS MXFP8 plus row-wise FP8 HC mix and output head | Exact-SM120 environment opt-in | real-kernel numerics, changed-input graph replay and live decode |
| HyperConnection Combine | fused SGLang CUDA | Model path | imported Qwen4 kernel tests |
| Multimodal attention | `triton_attn` | Automatic | vision qualification |
| Sampling | FlashInfer | Automatic | startup args |
| Grammar | XGrammar | Automatic | startup args and tool suite |
| HiCache transfer | NIXL POSIX | Explicit | restart restoration |
| Persistent state | packed target/native-MTP KV, full GDN/PLE siblings, compressed QSA keys | Local integration | `516e42a2ee` through `7b5cfb728d` |
| Optional PLE storage | attributed SSD Stream v0.2.0 reader with existing graph adapter | Explicit `PENNY_PLE_BACKEND=nvme` | full pool/graphs, long prefill, media/tools and NIXL restart restoration |

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

3. **State-writing gate retained.** Ordinary FlashInfer state-writing target
   verification was not globally enabled on SM120. Full/tree state-writing
   verification remains Triton. The runtime log label
   `FlashInferGDNKernel (none-mode WY output-only)` describes the active special
   mode; full/tree state-writing verification remains on Triton.

QSA decode is a distinct case. Commit `c1da0eef56` lets SM120 enter
FlashInfer's page-aligned QSA wrapper, whose SM12x dispatch selects XQA. PR
#36497 measured its throughput while the resolver was SM100-only; Pennyroyal's
SM120 results use XQA and contain no matched XQA-versus-fallback comparison.

Direct probing also established that TRTLLM-Gen is not merely hidden behind a
conservative gate. Forced selection reports `Unsupported architecture`; after
bypassing that guard, the CUDA driver rejects the exact QSA cubin with
`CUDA_ERROR_NO_BINARY_FOR_GPU (209)`. It targets SM100 and contains the
`sm10x_tcgen05` instruction family unavailable on SM120. Future support needs
kernel/compiler adaptation upstream of SGLang; see
[TensorRT-LLM #11799](https://github.com/NVIDIA/TensorRT-LLM/issues/11799) and
[FlashInfer #3628](https://github.com/flashinfer-ai/flashinfer/issues/3628).

Commit `95da38fb3b` fixes FP8 sparse-prefill tile interpretation for the active
unit-scale checkpoint. Open PR #36644 owns the broader calibrated-scale work.

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
| Target feed-forward layers | Dense FP8 MLP | 27B model architecture; no routed-expert MoE | model configuration and source |
| Target prefill / verify / draft graphs | Breakable prefill; full fixed-width verify | Automatic capture | startup graph-capture lines and live `cuda graph: True` |
| Multimodal attention | `triton_attn` | Automatic | startup log; mRoPE fixed by `64ecd64924` |
| Sampling / grammar | FlashInfer / XGrammar | Automatic | startup args |
| HiCache transfer | NIXL POSIX | Explicit | persistent restore qualification |
| Persistent state | target KV, Mamba/GDN, DFlash2 sidecar | Local NIXL integration | `8b786639e4`, `067c639c0a` |

The 27B profile is dense and uses none of Flash-Next's routed-expert
optimizations. Its measured stack has no separate local DeepGEMM SM120 patch.
DFlash2 support comes from the upstream base plus the XQA mask and NIXL fixes
above.

## Qualified configuration matrix

The matrix separates representation, datatype, backend, and capacity. “BF16
runtime dtype” covers unquantized tensors; quantized kernels retain the formats
listed in their own rows.

| Property | Qwen3.8-27B + DFlash2 | Qwen3.8 Flash-Next |
|---|---|---|
| Target checkpoint | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) recipe reference; retained measurements used the [orcarouter derivative](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8) | [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) |
| Target weight format | block FP8 E4M3, 128x128 blocks | ModelOpt NVFP4, group size 16 on selected Linear modules |
| Quantized-path activations | dynamic FP8 E4M3 | NVFP4 input activations on selected Linear modules |
| Runtime dtype for unquantized tensors | BF16; includes excluded layers and BF16 `lm_head` | Default: BF16 for otherwise-unquantized tensors. Optional online FP8 converts eligible transformer linears, HC mix weights and `lm_head`; BF16 GDN state and the existing NVFP4 expert, router and FP8 PLE-table formats remain unchanged. |
| Target KV datatype | FP8 E4M3 | FP8 E4M3 |
| Speculative KV datatype | DFlash2 draft: FP8 E4M3 | native-MTP: FP8 E4M3 |
| Recurrent/GDN SSM state | FP32 | BF16 |
| Convolution state | BF16 | BF16 |
| MoE backend | Not applicable: dense FP8 feed-forward layers | FlashInfer CUTLASS for target and native MTP |
| Target attention | FlashInfer prefill; TRTLLM-MHA/XQA decode and fixed-width verify | QSA Triton sparse prefill; FlashInfer QSA wrapper resolving to XQA for sparse decode; general attention FlashInfer |
| Linear/GDN attention | Triton decode, prefill, and state-writing verify | FlashInfer decode/prefill; WY output-only verify/recovery in `none` mode |
| Speculative backend | DFlash2, 8 draft tokens, 2,048-token window | native NEXTN, 3 steps, top-k 1, 4 draft tokens |
| Draft vocabulary | unchanged DFlash2 | 65,536-ID FR-Spec map introduced in v2.3; full target vocabulary unchanged |
| Served context | 524,288, factor-2 YaRN target and draft | 524,288, factor-2 YaRN |
| KV page size | 64 | 64 |
| Recipe default GPU KV capacity | 1,118,784 target and draft tokens | 824,384 target and native-MTP tokens |
| Mamba capacity | 24 slots; maximum 5 retained states/path | 24 slots by default; optional C6 uses 36; `extra_buffer`, tracking interval 64 |
| HiCache/NIXL | 96 GB configured host tier; target KV + Mamba/GDN + DFlash2 state | 32 GB configured host tier; packed target/native-MTP KV + GDN + PLE + QSA keys. PLE table placement is RAM by default or optional NVMe. |
| Maximum active requests | 4 | 4 by default; [optional 6](RUN.md#optional-six-request-flash-next-profile) |

The current 27B launcher uses the 24-slot/five-state setting qualified on
2026-08-26. The 2026-08-24 performance release used 16 slots, a three-state
path cap, and a 1,194,496-token KV pool. [RESULTS.md](RESULTS.md) keeps the two
campaigns and their allocations separate.
