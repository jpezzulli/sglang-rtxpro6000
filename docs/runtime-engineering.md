# Runtime engineering map

The measured result comes from an upstream DFlash2 implementation, a small set
of preserved local runtime changes, and a deliberately specific launch shape.
Those categories should not be conflated.

## Active decode path

```text
Qwen3.8 block-FP8 target, BF16 compute, FP8 KV
  prefill: FlashInfer
  decode and speculative verification: TRTLLM-MHA / XQA on SM120
                         ^ local fixed-width XQA mask integration

incoai DFlash2 draft, unquantized weights, FP8 KV
  draft attention: FlashInfer
  gamma: 8; visible draft window: 2,048
  Mamba/GDN state: FP32 SSM + BF16 conv, 3 retained states/path
```

Core DFlash2 local convolution, candidate selection, page-aligned reservation,
and `extra_buffer_lazy` support came from SGLang upstream before this branch's
merge base. The local branch does not carry a private DFlash2 worker.

## Local source that matters to active execution

- Independent draft configuration keeps the Qwen3.8 VLM target's nested mRoPE
  override separate from the draft's flat YaRN override.
- The XQA change preallocates and forwards the packed causal mask needed by the
  fixed-width DFlash2 target-verification path on SM120.
- NVCC receives the same GCC 15 host compiler that is fingerprinted and used to
  link JIT artifacts.
- Request lifecycle timing exposes elapsed initial-prefill and post-prefill
  spans, enabling completed-request effective decode measurement.

The draftless-hook commit is retained history but inactive here. The Mooncake
commit is retained history but the backend is absent from the deployment.

## Persistence path

HiCache retains target KV, Mamba/GDN state, and DFlash2 sidecar state in host
memory and writes representation-specific entries through NIXL POSIX FILE.
The local NIXL correction bounds bounce-buffer transfers and gives concurrent
path registrations distinct device identities. This is persistence correctness
work, not a GPU decoder optimization.

The namespace helper is deployment code outside upstream SGLang. It is included
under `scripts/pennyroyal/` because switching incompatible page geometry,
dtypes, checkpoint content, or runtime source into the same FILE root can
otherwise reinterpret persistent cache data.

## Configuration that materially shapes performance

- TP1 on one 96 GB SM120 GPU;
- FP8 E4M3 target and draft KV;
- TRTLLM-MHA/XQA target decode and verify;
- FlashInfer prefill and draft attention;
- DFlash2 gamma 8 and 2,048-token window;
- 2,048-token chunked/max prefill;
- page size 64;
- `mem_fraction_static=0.94`;
- four admitted requests;
- CUDA graphs active;
- thinking enabled, with medium as the deployment default.

DFlash2 acceptance is content-dependent. Configuration makes the fast path
available; it does not guarantee seven or eight accepted tokens on every step.
