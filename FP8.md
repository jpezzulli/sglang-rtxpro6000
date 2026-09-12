# Optional Online FP8 for Flash-Next

Pennyroyal v2.5.0 can convert selected Flash-Next weights to FP8 while loading
the model. This is an **opt-in, exact-SM120 path** intended for the qualified
single-GPU Flash-Next recipe. The default remains the checkpoint's original
precision layout.

On the measured RTX PRO 6000 configuration, online FP8 improved
single-request post-first-token throughput by **15.8–28.3%** across short,
128K and 490K contexts. It also left **7.52 GiB available after CUDA-graph
capture**, compared with 3.66 GiB in the earlier matching BF16-projection boot.
That is an observed difference of about 3.86 GiB, not a claim that this release
adds 7 GiB of serving capacity. See [RESULTS.md](RESULTS.md#pennyroyal-v250--optional-online-fp8)
for the exact measurements and timing boundaries.

## Enable it

Use the normal Flash-Next launcher and set the switch to the literal value
`true`:

```bash
export SGLANG_SM120_ONLINE_MXFP8=true
export TARGET_MODEL=/path/to/compatible-flash-next-checkpoint
configs/pennyroyal/serve-flash-next-frspec.sh
```

Unset the variable, or set it to `false`, to retain the original path. The
switch is included in the NIXL representation identity, so changing it selects
a separate cache namespace without deleting the old one.

The Flash-Next path requires CUDA on an exact SM120 GPU; unsupported hardware
or selected-module shapes fail startup. This is not a general FP8 mode for
other models. The 27B/DFlash2 launcher does not enable this conversion.

## What changes and what stays the same

“Online” means eligible BF16 weights are converted while the checkpoint loads;
it does not mean weights are repeatedly quantized during generation.

| Runtime component | With online FP8 enabled |
|---|---|
| Eligible otherwise-unquantized Flash-Next transformer projections | MXFP8 weights with dynamic MXFP8 activations through the qualified FlashInfer CUTLASS path |
| HyperConnection mix weights | Row-wise FP8 weights with one scale per output row; prefill materializes transient BF16 operands for the existing compiled path |
| Output head | Row-wise FP8 weights with aligned per-row scales; the FR-Spec draft shares the converted target head safely |
| NVFP4 experts and expert routing | Preserved from the checkpoint; not re-quantized by this option |
| PLE table and PLE projection | Preserved; RAM or optional NVMe placement is a separate choice |
| QSA representation | Preserved; this release does not introduce a new FP8 QSA format |
| GDN recurrent and convolution state | Preserved in BF16 |
| Target and native-MTP KV cache | Preserved in FP8 E4M3 |
| FR-Spec | Preserved, including the reduced-vocabulary map and scale alignment |

Conversion is bounded to recognized Flash-Next modules and is all-or-nothing
for each selected module. Missing scale metadata, unsupported tied head
weights, unavailable kernels, unexpected dtypes, or incompatible shapes stop
startup instead of silently falling back.

## Qualified scope

The v2.5.0 release qualification retained the normal 524,288-token context,
824,384-token GPU KV pool, page size 64, native NEXTN 3/1/4 shape, 24 Mamba
slots, CUDA graphs, FR-Spec, 32 GiB HiCache and NIXL persistence. Focused tests
covered real SM120 kernels, changed-input CUDA-graph replay, load-time weight
and scale invariants, HyperConnection shapes, the output head, and the
default-off 27B path.

Runtime checks included short/128K/490K single-request decode, concurrent long
requests, reasoning, ordinary tools, vision, an agent workflow, long-context
recall, and identical-restart NIXL restoration. A single cold 393,223-input-
token synthetic request recovered **64/64 opaque records exactly**.

With only the RTX PRO 6000 visible, model-GPU media preprocessing on logical
`cuda:0` also retained the 824,384-token pool and passed ten selected
image/video scenarios, including concurrent images and successive 208K-context
history turns. Minimum sampled free GPU memory was 1,897 MiB; this does not
guarantee every image shape or concurrency level.

The fully exercised compatible artifact scored 95.75/100 in one blinded
reasoning review, with no fatal cap. That is one local validation-suite sample,
not an independent benchmark, proof of improvement over the option-off
precision, or a score for the public RadixArk checkpoint.

The public recipe points to
[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4).
That checkpoint passed the same 64/64 exact-recall check with the v2.5.0 online
FP8 runtime. Its scope was that focused compatibility check, not a repeat of
the full performance, reasoning, tool and vision campaign. Other compatible
checkpoints must preserve the Qwen3.8 Flash-Next architecture, ModelOpt NVFP4
expert layout, tokenizer/FR-Spec mapping, native-MTP shape and expected
unquantized projection structure; a similar name is not enough.

## Interpretation and limits

- The reported decode rates exclude time to first token. Cold TTFT did **not**
  improve in the matching 128K and 490K samples.
- The short result is a three-run median. The long-context C1 rows are single
  observations, and the C4 outputs did not all have the same length.
- The observed standard-pool VRAM difference is useful headroom. The default
  FR-Spec cap remains 824,384. An explicit 1,000,000-token pool was separately
  qualified with RAM PLE, CPU media preprocessing and one visible GPU, leaving
  5.21 GiB after graphs and at least 1,187 MiB in the sampled media window. It
  does not increase served context or establish a speed improvement, and it
  was not combined with model-GPU media preprocessing.
- The ordinary tool suite was 29/30 semantically correct. In a separate probe,
  three of six quoted fully wrapped tool examples were executed when they
  should have remained text. The parser was unchanged, so the evidence neither
  establishes a clean all-tools pass nor ties those failures to online FP8.
- No bit-for-bit output-parity claim, universal model-quality claim, or
  cross-hardware speed claim is made.

## Credit

The implementation is directly inspired by
[`mratsim/sglang-qwen38fn-sm120-turbo`](https://github.com/mratsim/sglang-qwen38fn-sm120-turbo/tree/94a68214b77514bc26ef78cee4c01f128162d09b)
at commit `94a68214b77514bc26ef78cee4c01f128162d09b`, specifically patches
`0003`, `0007`, and `0008`. Pennyroyal adapts that approach to its existing
Flash-Next, FR-Spec, persistence and fail-loud boundaries; it is not presented
as independently originating the online-FP8 design.
