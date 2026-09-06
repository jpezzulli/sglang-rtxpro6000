# Run

Both model profiles use the same built source and expose the model as
`pennyroyal` on an OpenAI-compatible endpoint. They are sanitized reproductions
of the qualified launch shape, not drop-in system service files.

## Common setup

Create durable compiler caches and a NIXL FILE root on storage appropriate for
your system:

```bash
mkdir -p "$CACHE_BASE"/{huggingface,torch,torchinductor,triton,cuda,flashinfer,sglang/jit}
mkdir -p "$NIXL_STORAGE_BASE"
```

Required environment variables:

```bash
export REPO_ROOT=/path/to/this/checkout
export CACHE_BASE=/path/to/compiler-and-runtime-caches
export NIXL_STORAGE_BASE=/path/to/persistent-nixl-root
export TARGET_MODEL=/path/to/target-checkpoint
```

For 27B/DFlash2 also set:

```bash
export DRAFT_MODEL=/path/to/incoai-Qwen3.8-27B-DFlash2
```

The scripts expect the SGLang executable at `$REPO_ROOT/.venv/bin/sglang`.
Override `SGLANG_EXE` and `PYTHON` if the environment lives elsewhere. Review
the sample NIXL watermarks before first use.

## Launch Flash-Next

```bash
configs/pennyroyal/serve-flash-next.sh
```

The qualified shape uses ModelOpt NVFP4 weights and input activations on selected
Linear modules, BF16 for excluded/unquantized tensors and recurrent state, FP8
E4M3 target/native-MTP KV, native NEXTN, 524K YaRN, 24 Mamba slots,
RecoverSSM `none`, explicit FlashInfer GDN decode/prefill, 32 GB HiCache, and
NIXL POSIX persistence. This is not a claim that every kernel computes in BF16.

## Launch Flash-Next with FR-Spec (2.3)

Use the same setup and full Flash-Next shape above, then select:

```bash
configs/pennyroyal/serve-flash-next-frspec.sh
```

This recipe adds only `--speculative-token-map` to the serving arguments,
using the bundled 65,536-ID map. It checks the map and tokenizer hashes before
startup and includes the map hash in the NIXL representation identity. It does
not enable a new BF16 kernel, FP8 weight copies or relaxed acceptance.
`nixl-posix-frspec.toml` records the qualified dedicated-filesystem 85%/80%
cleaner watermarks; review them for your filesystem. The original non-FR and
27B recipes/config remain unchanged.

The public target model reference remains
`RadixArk/Qwen3.8-Flash-Next-NVFP4`. Model weights are not included. Tokenizer
hash compatibility protects token-ID meaning; it does not itself establish
quality or throughput for every checkpoint or host. Configuration and map
provenance are recorded in [PROVENANCE.md](PROVENANCE.md).

Confirm `speculative_token_map` in resolved server arguments, the reduced
65,536-row draft head, and the original full target vocabulary. Retain all
normal graph, state-pool and NIXL startup checks below. The extra BF16 draft
head uses 320 MiB; it fit the measured 824,384-token pool without changing the
context, static fraction, request limit or Mamba slots.

The deterministic map builder and focused tests are under
`scripts/pennyroyal/frspec/`. Rebuilding with a different corpus or tokenizer
creates a different experiment; it is not necessary to regenerate the bundled
qualified artifact. Changing a map must change the persistent namespace.

## Launch 27B with DFlash2

```bash
configs/pennyroyal/serve-qwen38-27b-dflash2.sh
```

The target uses block-FP8 E4M3 weights and dynamic FP8 activations on quantized
paths, with BF16 for unquantized tensors. Target/draft KV are FP8 E4M3; GDN SSM
state is FP32 and convolution state BF16. The shape uses eight DFlash2 draft
tokens, a 2,048-token draft window, TRTLLM-MHA/XQA target decode, FlashInfer
target prefill/draft attention, Triton FP8 MoE and GDN, 24 Mamba slots, five
retained states per path, 96 GB HiCache, and NIXL POSIX persistence.

## Startup checks

For Flash-Next, confirm log lines for:

- 824,384 target/native-MTP KV tokens and FP8 E4M3 dtypes;
- 24 Mamba slots and zero intermediate speculative SSM;
- `FlashInferGDNKernel` decode/prefill and
  `none-mode WY output-only` verification/recovery;
- QSA sparse decode through FlashInfer's wrapper resolving to XQA on SM120,
  `sgl-kernel` top-k, and MTP index sharing;
- target and native-MTP MoE resolved to FlashInfer CUTLASS;
- recovery graphs for batch sizes 1-4;
- attached KV, Mamba/PLE, and QSA HiCache pools.

For 27B, confirm:

- target and draft FP8 E4M3 pools;
- `Initialized DFLASH draft runner` with eight tokens and window 2,048;
- fused KV materialization;
- FlashInfer prefill/draft and TRTLLM-MHA target decode/verify;
- FP8 MoE resolved to Triton rather than DeepGEMM under A2A `none`;
- 24 Mamba slots, five retained states/path, and 1,118,784 KV tokens;
- target prefill, target verify, and draft verify graph capture.

## Smoke through the normal API

```bash
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"pennyroyal",
    "messages":[{"role":"user","content":"Explain why 17 is prime."}],
    "max_tokens":256,
    "stream":false
  }'
```

Run several ordinary thinking-enabled warmups before measuring. Do not abort a
request with terminal signals; use protocol-level cancellation or let it finish.

## Distinguish cache paths

- **Cold prefill:** no matching GPU radix prefix and no matching NIXL object;
  logs show most tokens as newly computed.
- **Radix reuse:** same server process retains the prefix in GPU/host state;
  logs show cached tokens without a service restart.
- **Persistent NIXL restore:** restart the service without deleting the selected
  namespace, submit the identical serialized prefix, and verify HiCache/NIXL
  load plus a large restored/cached count and only a small recomputed suffix.

Do not label a restored-prefix effective rate as cold prefill. The persistent
namespace must remain representation-compatible; the helper deliberately
selects a new directory after relevant configuration or source changes.
