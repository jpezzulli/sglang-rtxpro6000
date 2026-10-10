# Qwen3.8 Flash-Next online FP8 on RTX PRO 6000

Pennyroyal converts eligible Flash-Next weights to FP8 while loading the
model **automatically** on the single-GPU NVIDIA RTX PRO 6000 Blackwell
(SM120) recipe: a fresh or default launch needs no kernel switch and answers
no kernel questions. The conversion engages only for a recognized Flash-Next
checkpoint on exact SM120; anything unsupported simply keeps the checkpoint's
original precision layout.

On the measured RTX PRO 6000 configuration, online FP8 improved
single-request post-first-token throughput by **15.8–28.3%** across short,
128K and 490K contexts. It also left **7.52 GiB available after CUDA-graph
capture**, compared with 3.66 GiB in the earlier matching BF16-projection boot.
The observed difference was about 3.86 GiB; 7.52 GiB is the total available
after graphs. See [RESULTS.md](RESULTS.md#pennyroyal-v250--optional-online-fp8)
for measurements and timing definitions.

## Launching it

Use the normal Flash-Next launcher; nothing else to set:

```bash
export TARGET_MODEL=/path/to/compatible-flash-next-checkpoint
configs/pennyroyal/serve-flash-next-frspec.sh
```

The saved private overrides remain for compatibility and debugging:
`SGLANG_SM120_ONLINE_MXFP8=false` pins the original path, and `=true` forces
the conversion even where automatic eligibility would decline. An explicit
request on unsupported hardware still fails at startup. These values are part
of the NIXL representation identity (effective precision, not the requested
Boolean), so precision changes select a separate cache namespace without
deleting the old one; the older `online_mxfp8=true` namespaces named the
mixed-MXFP8 build and cannot claim rowwise-FP8 caches.

The conversion supports recognized Flash-Next modules on exact SM120. Other
hardware and module shapes keep the original path automatically; explicit
requests for them fail loudly. The 27B/DFlash2 launcher keeps its existing
precision path.

### Donor W8A16 GEMV for the output heads

As part of the same accepted selection, the resident row-wise FP8 target
and FR-Spec draft heads route through the donor's low-row (1–16 tokens) Triton
W8A16 GEMV (`python/sglang/srt/layers/quantization/w8a16_gemv.py`, direct port
of aiueo52/sglang-rtxpro6000@5105985116eb00dea8e6138aabeb5363387cb9de). Weights,
row scales and the FR-Spec token map are used as they are; nothing is
re-quantized. Prefill, the 24-token verification batch and any layout the kernel
does not read stay on the original head kernel, so the two are A/B comparable.
It reserves about 16 MiB of split-K scratch per device when enabled, allocated
before CUDA-graph capture. `SGLANG_FP8_W8A16_GEMV_MAX_M` (default 16) can only
narrow the row budget. `SGLANG_FP8_W8A16_GEMV=false` is the private opt-out
that retreats to the rowwise kernel alone.

## What changes and what stays the same

“Online” describes a one-time conversion of eligible BF16 weights while the
checkpoint loads. Generation uses the converted weights.

| Runtime component | On an eligible Flash-Next SM120 launch |
|---|---|
| Eligible otherwise-unquantized Flash-Next transformer projections | Row-wise (per-output-channel) FP8 weights with dynamic activations — the donor-compatible representation, arithmetic unchanged from the accepted candidate |
| HyperConnection mix weights | Row-wise FP8 weights with one scale per output row; prefill materializes transient BF16 operands for the existing compiled path |
| Output head | Row-wise FP8 weights with aligned per-row scales; the FR-Spec draft shares the converted target head safely |
| NVFP4 experts and expert routing | Preserved from the checkpoint |
| PLE table and PLE projection | Preserved; RAM or optional NVMe placement is a separate choice |
| QSA representation | Preserved in its existing format |
| GDN recurrent and convolution state | Preserved in BF16 |
| Target and native-MTP KV cache | Preserved in FP8 E4M3 |
| FR-Spec | Preserved, including the reduced-vocabulary map and scale alignment |

Conversion is all-or-nothing for each recognized Flash-Next module. Missing
scale metadata, unsupported tied head weights, unavailable kernels, unexpected
dtypes, and incompatible shapes stop startup.

<a id="qualified-scope"></a>

## Tests and results

The v2.5.0 tests used the normal 524,288-token context,
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
history turns. Minimum sampled free GPU memory was 1,897 MiB. Image dimensions
and concurrency change the required headroom.

The RadixArk checkpoint scored 95.75/100 in one blinded local validation-suite
review, with no fatal cap. The review did not compare model quality between
precision modes.

The public recipe uses
[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4).
For other checkpoints, use the Qwen3.8 Flash-Next architecture, ModelOpt NVFP4
expert layout, tokenizer/FR-Spec mapping, native-MTP shape and expected
unquantized projection structure.

<a id="interpretation-and-limits"></a>

## Choosing your settings

- The reported decode rates exclude time to first token. Cold TTFT did **not**
  improve in the matching 128K and 490K samples.
- The short result is a three-run median. The long-context C1 rows are single
  observations, and the C4 outputs did not all have the same length.
- The observed standard-pool VRAM difference is useful headroom. The default
  FR-Spec cap remains 824,384. An explicit 1,000,000-token pool was separately
  tested with RAM PLE, CPU media preprocessing and one visible GPU, leaving
  5.21 GiB after graphs and at least 1,187 MiB in the sampled media window.
  Served context remains 524,288 tokens; no speed comparison was run. The test
  used CPU media preprocessing.
- The ordinary tool suite was 29/30 semantically correct. In a separate probe,
  three of six quoted fully wrapped tool examples were executed when they
  should have remained text. The parser was unchanged, so those failures do
  not isolate online FP8 as the cause.

## Credit

The implementation is directly inspired by
[`mratsim/sglang-qwen38fn-sm120-turbo`](https://github.com/mratsim/sglang-qwen38fn-sm120-turbo/tree/94a68214b77514bc26ef78cee4c01f128162d09b)
at commit `94a68214b77514bc26ef78cee4c01f128162d09b`, specifically patches
`0003`, `0007`, and `0008`. Pennyroyal adapts that approach to its existing
Flash-Next, FR-Spec, persistence, and fail-loud boundaries. The online-FP8
design originates with the credited work above.
