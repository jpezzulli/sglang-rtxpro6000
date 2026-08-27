# Limitations

## Evidence scope

- One NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB, SM120, TP=1.
- Exact checkpoint revisions and the recorded CUDA, PyTorch, FlashInfer, NIXL,
  GCC, and SGLang source stack.
- Flash-Next final qualification ran at `64ecd64924`. The 27B controlled and
  real-agentic campaigns ran on the earlier qualified runtime line retained in
  current history; later Flash-Next-specific commits were not all re-benchmarked
  on 27B.
- Both launchers admit 524,288 tokens, but the controlled needle prompts were
  approximately 490K. This is not proof for every possible 524K prompt,
  modality, sampling configuration, or concurrent schedule.

## Early Flash-Next integration

- Flash-Next support was produced immediately after model availability. It has
  substantial direct evidence on this machine but may still contain model- or
  hardware-specific assumptions.
- An upstream architecture gate can mean “not selected or wired on SM120,” not
  “the kernel is incompatible.” This runtime opens only paths directly
  qualified on SM120 and does not globally remove architecture checks.
- Ordinary FlashInfer state-writing GDN target verification remains disabled on
  SM120. Only the `none`-mode WY output/recovery path is enabled; tree/full-state
  verification remains Triton.
- FlashInfer HyperConnection Mix remains SM100-only; SM120 uses the persistent
  Triton implementation. Shared-expert fusion is not claimed active.
- The active QSA sparse-prefill correction covers a unit-scale FP8 cache. The
  broader calibrated-scale work in open PR #36644 is not included.
- The KV estimator still reserves the intermediate GDN speculative pool even
  when RecoverSSM `none` removes the physical allocation. The qualified `.981`
  setting exposes that recovered budget; it is not a general recommendation to
  raise memory fractions without measuring headroom.

## Performance interpretation

- Cold prefill, restored-prefix effective prefill, completed-request decode,
  per-stream post-first-token rates, synchronized aggregate throughput, and
  instantaneous server telemetry are different measurements.
- Favorable 250-300 tok/s DFlash2 and 543 tok/s Flash-Next samples are short
  telemetry windows, not sustained completed-request rates.
- DFlash2 and native-MTP acceptance vary with content and context and explain a
  large part of throughput variation.
- The 27B community comparison is directional. Checkpoint, runtime,
  speculation, power, client, prompt, duration, and cache/offload differ.
- No repeat-variance campaign was run for every natural-workload number and no
  “fastest” claim is made.
- Results from Flash-Next and 27B use different models, speculative mechanisms,
  state layouts, and allocations and must not be treated as one A/B arm.

## Reasoning, tools, and vision

- Reasoning scores come from the identified internal validation suite, not an
  independently preregistered external benchmark.
- The 27B xhigh and medium campaigns used different concurrency schedules and
  token distributions. Their score/speed tradeoff is not universal.
- Literal tool checkers can reject a correct tool selection because of response
  wording. Results therefore preserve automatic, exact-call, and reviewed
  semantic/discipline outcomes separately.
- The 27B target is uncensored/abliterated; its behavior does not characterize
  the official Qwen checkpoint's refusal or safety policy.
- Flash-Next passed a complete 1,024-token vision validation. The earlier 27B
  dated release's final performance campaign was text-focused, although the
  current runtime's fused mRoPE correction has direct numerical and vision
  evidence.

## HiCache/NIXL

- HiCache/NIXL provides prefix movement and restart reuse; it is not credited
  for GPU-resident decode throughput.
- Cleaner thresholds are whole-filesystem occupancy percentages, not an
  absolute cache-directory byte limit. Other filesystem data changes effective
  cache capacity; a dedicated mount gives the clearest semantics.
- Persistent cache is disposable. A missing or incomplete object may become a
  miss and ordinary recomputation; transactional persistence is not claimed.
- Representation namespaces isolate incompatible layouts, but inactive roots
  need an external lifecycle policy if many configurations accumulate.
- Service/process restart reuse was directly demonstrated. Full-machine reboot
  reuse was not repeated for every current namespace/configuration combination.
- Mooncake was rejected and removed. The fallback is the same runtime without
  hierarchical RAM/SSD caching, not Mooncake.

## Reproducibility

- The native procedure is reconstructed from the successful installed state and
  source metadata. It has not been repeated on a clean second host.
- JIT and CUDA-graph cache state affects startup and first-request behavior.
- No model weights, cache payloads, private prompts, hidden reasoning traces,
  hostnames, tokens, or service credentials are distributed.
