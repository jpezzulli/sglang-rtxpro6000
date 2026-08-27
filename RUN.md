# Run

Both launch recipes use the same built source and expose the model as
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

The qualified shape is NVFP4 weights, BF16 compute/recurrent state, FP8 E4M3
KV, native NEXTN, 524K YaRN, 24 Mamba slots, RecoverSSM `none`, explicit
FlashInfer linear decode/prefill, 32 GB HiCache, and NIXL POSIX persistence.

## Launch 27B with DFlash2

```bash
configs/pennyroyal/serve-qwen38-27b-dflash2.sh
```

The qualified shape is BF16 compute, target/draft FP8 E4M3 KV, eight DFlash2
draft tokens, a 2,048-token draft window, TRTLLM-MHA/XQA target decode,
FlashInfer target prefill and draft attention, 24 Mamba slots, five retained
states per path, 96 GB HiCache, and NIXL POSIX persistence.

## Startup checks

For Flash-Next, confirm log lines for:

- 824,384 target/native-MTP KV tokens and FP8 E4M3 dtypes;
- 24 Mamba slots and zero intermediate speculative SSM;
- `FlashInferGDNKernel` decode/prefill and
  `none-mode WY output-only` verification/recovery;
- QSA TRTLLM-Gen decode, `sgl-kernel` top-k, and MTP index sharing;
- recovery graphs for batch sizes 1-4;
- attached KV, Mamba/PLE, and QSA HiCache pools.

For 27B, confirm:

- target and draft FP8 E4M3 pools;
- `Initialized DFLASH draft runner` with eight tokens and window 2,048;
- fused KV materialization;
- FlashInfer prefill/draft and TRTLLM-MHA target decode/verify;
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
