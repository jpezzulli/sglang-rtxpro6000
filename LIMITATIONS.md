# Limitations

## Evidence scope

- One NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB, SM120, TP=1.
- Exact checkpoint revisions and the recorded CUDA, PyTorch, FlashInfer, NIXL,
  GCC, and SGLang source stack.
- Earlier Radix Flash-Next qualification ran at `64ecd64924`. The 27B controlled and
  real-agentic campaigns ran on the earlier qualified runtime line retained in
  current history; later Flash-Next-specific commits were not all re-benchmarked
  on 27B.
- Both launchers admit 524,288 tokens, but the controlled needle prompts were
  approximately 490K. Coverage varies by prompt length, modality, sampling
  configuration, and concurrent schedule.

## v2.5.0 scope

### Online FP8

- The short C1 comparison has three samples per arm. The 128K and 490K C1
  values are single observations. Their post-first-token rates improved, but
  cold TTFT did not improve in the matching long samples.
- Post-graph available VRAM was 7.52 GiB versus 3.66 GiB in an earlier matching
  option-off boot. The difference is about 3.86 GiB; 7.52 GiB is the total
  remaining after graphs. Both comparison arms used the 824,384-token pool.
- The 824,384-token pool in these runs was the recipe's default cap at the
  time; the FR-Spec recipe now asks for none and keeps only an explicit
  `MAX_TOTAL_TOKENS` cap. The explicit 1,000,000-token option passed startup,
  graph, long-context, concurrency and CPU-media checks with online FP8 and one
  visible GPU. Other cap values and combinations require their own evidence.
- The 1,000,000-token option changes KV capacity, not the 524,288-token served-
  context limit. Its warmed C1 median was close to the earlier standard-pool
  observation, but the runs used different measurement windows. No speed
  comparison was run.
  Post-graph free memory was 5.21 GiB and minimum sampled media-window free
  memory was 1,187 MiB; neither is a universal headroom guarantee.
- The RadixArk checkpoint supplied the performance, reasoning, ordinary-tool,
  vision, and 64/64 exact long-recall results with online FP8.
- The ordinary tool suite was 29/30 semantically correct. More importantly, a
  separate quoted-markup probe executed three of six fully wrapped examples
  that should have remained text. Parser code was unchanged, so the probe does
  not isolate online FP8 as the cause.
- The option preserves BF16 GDN state and the checkpoint's existing NVFP4
  expert, router, FP8 PLE-table, QSA, and recurrent-state formats. Evidence
  covers exact SM120; bit-for-bit output parity was not measured.

### NVMe PLE

- The table is 47.68 GiB on SSD. Host `MemAvailable` was about 54–56 GiB higher
  in separate NVMe snapshots, but filesystem cache, process state and other host
  activity prevent attributing that full difference to the table.
- RAM and NVMe PLE retained the same 824,384-token pool and 524,288 context.
  Their repeated C4 medians were 428.90 and 369.73 tok/s respectively in a
  separate operational comparison. Different cache/JIT history and a slow
  second run in both arms limit the comparison to those observed medians.
- NVMe mode requires a prepared immutable overlay and the matching isolated
  reader. It fails startup on source, model, manifest or table mismatch and has
  no automatic RAM fallback.
- NVMe PLE and online FP8 can be selected together. The combined configuration
  passed direct live functional checks and repeated warmed C1/C4 observations,
  in a separate run from the RAM-versus-NVMe comparison. The measurements use
  different windows and do not form one causal comparison.
- Sharing a device with NIXL can introduce I/O contention. Evidence covers
  local SSD, Linux x86-64, Python 3.12, TP1 Flash-Next only—not TP2, network
  storage, CPU expert offload, or alternate speculative modes.
- Short host-wide observations measure available memory during the test. They
  do not isolate every source of compaction or prior host-memory pressure.

### Startup and media controls

- The built-in startup warmup compiles one stable JSON schema and exercises one
  grammar/mask path. Other schemas, long prefill, multimodal input, and
  concurrency shapes warm when used.
- CPU, `cuda:0` model-GPU, and `cuda:1` secondary-GPU preprocessing have direct
  qualified evidence. The v2.5.0 `cuda:0` check used only the RTX PRO 6000,
  retained the 824,384-token pool with online FP8, and passed ten selected
  image/video scenarios. Minimum sampled free GPU memory was 1,897 MiB. Image
  shape and concurrency change the required headroom.
- Model-GPU media preprocessing and an experimental larger
  `MAX_TOTAL_TOKENS` value have not been qualified together.

## v2.4.1 scope

This maintenance update made no serving-speed or model-quality comparison. The
PLE kernel improved in focused measurements without a measured net decode gain
on the Flash-Next profile. Native setup instructions were reconstructed from
the working installation; the container supplies the clean automated build.

## v2.4.0 scope

The prefill percentage compares the same server initial-prefill timing window
before and after the combined update, with one cold prompt at each measured
length. Older results use client TTFT. Warm prefill, decode speed, bit-for-bit
output parity, and general model quality were not compared. Both profiles
retained their context, pools, and NIXL restart-restoration behavior.

Tool-markup guards cover bare functions and undeclared wrapped names; fully
wrapped, declared tools inside quotations remain ambiguous. Long-workload
GPU/driver stalls and unrelated speculative-decoding issues remain outside
this update.

## v2.3.1 maintenance scope

The reported live failure was not reproduced here. Focused GPU tests reproduced
the GDN rounding defect, and both profiles passed the maintenance regressions
listed in [CHANGES.md](CHANGES.md#v231--gdn-rounding-maintenance). Full
reasoning, tool, vision, and performance suites were not repeated. A working
v2.3 installation does not need this maintenance update urgently.

## v2.3 validation scope

- Performance tests used one RTX PRO 6000 at TP1 and the same Flash-Next
  checkpoint across baseline and FR-Spec runs. The comparison includes two
  baseline boots and repeated FR-Spec measurements. The design was not
  randomized; workload and run-to-run variation affect the observed speedup.
- FR-Spec leaves target weights, vocabulary, and acceptance policy unchanged.
  Bit-for-bit sampled-output parity was not tested.
- Tool results were 27/30 literal checks, 29/30 exact calls/arguments, and
  30/30 parseable responses. The workflows completed with one redundant
  read-only call. Two response-quality issues are documented separately in
  the [validation report](https://github.com/jpezzulli/pennyroyal-validation/blob/main/results/qwen38-flash-next-frspec-20260905.md).
- The natural-decode test returned all 3,072 token IDs, but its automatic check
  failed because the collector expected a different field name. The collector
  also saves mutable message-list references, which limits analysis of saved
  request snapshots. The automatic result remains failed despite the complete
  returned token list.
- Both profiles passed startup, prefill, single/concurrent decode, and NIXL
  restart restoration. Full 27B reasoning/tool tests were not repeated;
  its runtime and launcher are unchanged.
- Missing-sidecar and corrupt-manifest fault injection were not repeated for
  v2.3. NIXL cache storage remains disposable, not transactional.

## Early Flash-Next integration

- Flash-Next support was produced immediately after model availability. It has
  substantial direct evidence on this machine but may still contain model- or
  hardware-specific assumptions.
- An upstream architecture gate can mean “not selected or wired on SM120,” not
  “the kernel is incompatible.” This runtime opens only paths directly
  exercised on SM120 and keeps the remaining architecture checks.
- Ordinary FlashInfer state-writing GDN target verification remains disabled on
  SM120. Only the `none`-mode WY output/recovery path is enabled; tree/full-state
  verification remains Triton.
- FlashInfer HyperConnection Mix remains SM100-only; SM120 uses the persistent
  Triton implementation. Shared-expert fusion remains disabled.
- The active QSA sparse-prefill correction covers a unit-scale FP8 cache. The
  broader calibrated-scale work in open PR #36644 is not included.
- The KV estimator still reserves the intermediate GDN speculative pool even
  when RecoverSSM `none` removes the physical allocation. The qualified `.981`
  setting exposes that recovered budget on the measured GPU. Other memory
  fractions require their own headroom measurement.

## Performance interpretation

- Cold prefill, restored-prefix effective prefill, completed-request decode,
  per-stream post-first-token rates, synchronized aggregate throughput, and
  instantaneous server telemetry are different measurements.
- Favorable instantaneous samples are short telemetry windows, not sustained
  completed-request rates.
- DFlash2 and native-MTP acceptance vary with content and context and explain a
  large part of throughput variation.
- The **Flash-Next DFlash2 profile (`next-dflash2`) is experimental / alpha**:
  available for experimentation, but still newer in the field, untested on
  two-GPU hardware, and not requalified on the October 27 candidate. The **27B
  DFlash2 profile is established and qualified** and carries no alpha label; on
  that candidate it served the reasoning, tool, image and long-context checks and
  restored a fresh-process cache end to end, and it remains 27B's speculative path
  here. The #115 decode-checkpoint (Mamba `extra_buffer` /
  `extra_buffer_lazy`) arithmetic is covered by its 3.1.2 CPU regression; an
  end-to-end restore is not a grid-scatter unit regression, and no isolated GPU
  check of that path exists on either DFlash2 profile. On both profiles the
  draft's extra host DRAM is a real cost that is separate from maturity - the
  recorded 27B launch spent 79.45 GB of target KV plus 16.63 GB of Mamba state and
  a further 24.83 GB of draft sliding-window cache, about 120.91 GB of caches
  before other process memory. Nothing here is a toggle or an automatic fallback to
  native MTP, and no allocator change is included. See
  [DFlash2 maturity](README.md#dflash2-maturity).
- The 27B community comparison is directional. Checkpoint, runtime,
  speculation, power, client, prompt, duration, and cache/offload differ.
- Repeat-variance coverage differs across the natural-workload measurements.
- Results from Flash-Next and 27B use different models, speculative mechanisms,
  state layouts, and allocations and must not be treated as one A/B arm.

## Reasoning, tools, and vision

- Reasoning scores come from the identified internal validation suite. The
  suite was not independently preregistered.
- The 27B xhigh and medium campaigns used different concurrency schedules and
  token distributions. Their score/speed tradeoff is not universal.
- Literal tool checkers can reject a correct tool selection because of response
  wording. Results therefore preserve automatic, exact-call, and reviewed
  semantic/discipline outcomes separately.
- The 27B target is uncensored/abliterated; its behavior characterizes that
  derivative rather than the official Qwen checkpoint's refusal or safety
  policy.
- Flash-Next passed a complete 1,024-token vision validation. The earlier 27B
  dated release's final performance campaign was text-focused, although the
  current runtime's fused mRoPE correction has direct numerical and vision
  evidence.

## HiCache/NIXL

- HiCache/NIXL provides prefix movement and restart reuse; it is not credited
  for GPU-resident decode throughput.
- Cleaner thresholds use whole-filesystem occupancy percentages. Other data on
  that filesystem changes effective cache capacity; a dedicated mount gives
  the clearest semantics.
- Persistent cache is disposable. A missing or incomplete object may become a
  miss and ordinary recomputation. It is not a transactional store.
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
