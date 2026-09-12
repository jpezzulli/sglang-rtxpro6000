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
  approximately 490K. This is not proof for every possible 524K prompt,
  modality, sampling configuration, or concurrent schedule.

## v2.5.0 scope

### Online FP8

- The short C1 comparison has three samples per arm. The 128K and 490K C1
  values are single observations. Their post-first-token rates improved, but
  cold TTFT did not improve in the matching long samples.
- Post-graph available VRAM was 7.52 GiB versus 3.66 GiB in an earlier matching
  option-off boot. The difference is about 3.86 GiB; 7.52 GiB is not the amount
  newly freed. The result does not qualify a larger KV pool.
- The primary FR-Spec recipe retains an 824,384-token default cap. The explicit
  1,000,000-token option passed startup, graph, long-context, concurrency and
  CPU-media checks with online FP8 and one visible GPU. Other cap values and
  combinations require their own evidence.
- The 1,000,000-token option changes KV capacity, not the 524,288-token served-
  context limit. Its warmed C1 median was close to the earlier standard-pool
  observation, but the runs were not a matched A/B and establish no speed gain.
  Post-graph free memory was 5.21 GiB and minimum sampled media-window free
  memory was 1,187 MiB; neither is a universal headroom guarantee.
- The public RadixArk reference checkpoint passed one 64/64 exact long-recall
  request with online FP8. The performance, reasoning, ordinary-tool and vision
  results were not rerun on that checkpoint and must not be attributed to it.
- The ordinary tool suite was 29/30 semantically correct. More importantly, a
  separate quoted-markup probe executed three of six fully wrapped examples
  that should have remained text. Parser code was unchanged, so neither a clean
  all-tools pass nor online-FP8 causality is claimed.
- The option preserves BF16 GDN state and the checkpoint's existing NVFP4
  expert, router and FP8 PLE-table formats. It does not establish a new FP8 QSA
  or recurrent-state format, bit-for-bit output parity, or portability beyond
  exact SM120.

### NVMe PLE

- The table is 47.68 GiB on SSD. Host `MemAvailable` was about 54–56 GiB higher
  in separate NVMe snapshots, but filesystem cache, process state and other host
  activity prevent attributing that full difference to the table.
- RAM and NVMe PLE retained the same 824,384-token pool and 524,288 context.
  Their repeated C4 medians were 428.90 and 369.73 tok/s respectively in a
  separate operational comparison. Different cache/JIT history and a slow
  second run in both arms prevent a universal performance ratio or no-cost
  claim.
- NVMe mode requires a prepared immutable overlay and the matching isolated
  reader. It fails startup on source, model, manifest or table mismatch and has
  no automatic RAM fallback.
- NVMe PLE and online FP8 can be selected together. The combined configuration
  passed direct live functional checks and repeated warmed C1/C4 observations,
  but this was not a fresh controlled RAM-versus-NVMe A/B. Its throughput and
  the separately collected option-specific gains/costs must not be treated as
  one causal comparison.
- Sharing a device with NIXL can introduce I/O contention. Evidence covers
  local SSD, Linux x86-64, Python 3.12, TP1 Flash-Next only—not TP2, network
  storage, CPU expert offload, or alternate speculative modes.
- Short host-wide observations do not prove that NVMe eliminates memory
  compaction or that PLE caused all prior host-memory pressure.

### Startup and media controls

- The built-in startup warmup compiles one stable JSON schema and exercises one
  grammar/mask path. It does not warm arbitrary schemas, long prefill,
  multimodal input or every concurrency shape.
- CPU, `cuda:0` model-GPU, and `cuda:1` secondary-GPU preprocessing have direct
  qualified evidence. The v2.5.0 `cuda:0` check used only the RTX PRO 6000,
  retained the 824,384-token pool with online FP8, and passed ten selected
  image/video scenarios. Minimum sampled free GPU memory was 1,897 MiB; this is
  not a guarantee for every image shape or concurrency level.
- Model-GPU media preprocessing and an experimental larger
  `MAX_TOTAL_TOKENS` value have not been qualified together.

## v2.4.1 scope

This maintenance update makes no additional serving-speed or model-quality
claim. The PLE kernel improved in focused measurements, but a net decode gain
was not established on the qualified Flash-Next profile. The clearer setup
instructions are not a claim of a fresh full build on a second machine.
Existing model, topology and persistence limits still apply.

## v2.4.0 scope

The prefill percentage compares the same server initial-prefill timing window
before and after the combined update, with one cold prompt at each measured
length. It is not directly comparable to older client-TTFT-based rates. No
warm-prefill percentage, decode speedup, bit-for-bit output parity, or general
model-quality improvement is claimed. Both profiles retained their context,
pools and NIXL restart-restoration behavior.

Tool-markup guards cover bare functions and undeclared wrapped names; they do
not distinguish every quotation of a fully wrapped, declared tool. This update
is not a claim that all long-workload GPU/driver stalls or every speculative
decoding issue are fixed. Hardware, checkpoint and workload limits below still
apply.

## v2.3.1 maintenance scope

The reported live failure was not reproduced here. Focused GPU tests reproduced
the GDN rounding defect, and both profiles passed the maintenance regressions
listed in [CHANGES.md](CHANGES.md#v231--gdn-rounding-maintenance). This is not a
claim that every speculative-decoding failure is resolved. Full reasoning,
tool, and vision suites were not repeated, and no new performance claim is made.
If v2.3 is working well for you, there is no urgent need to update.

## v2.3 validation scope

- Performance tests used one RTX PRO 6000 at TP1 and the same Flash-Next
  checkpoint across baseline and FR-Spec runs. The comparison includes two
  baseline boots and repeated FR-Spec measurements; it is not a randomized
  benchmark. Workload and run-to-run variation affect the speedup.
- FR-Spec leaves target weights, vocabulary, and acceptance policy unchanged.
  Bit-for-bit sampled-output parity was not tested.
- Tool results were 27/30 literal checks, 29/30 exact calls/arguments, and
  30/30 parseable responses. The workflows completed with one redundant
  read-only call. Two response-quality issues are documented separately in
  the [validation report](https://github.com/jpezzulli/pennyroyal-validation/blob/main/results/qwen38-flash-next-frspec-20260905.md).
- The natural-decode test returned all 3,072 token IDs, but its automatic check
  failed because the collector expected a different field name. The collector
  also saves mutable message-list references, which limits analysis of saved
  request snapshots. These are evaluator limitations, not additional passing
  automatic tests.
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
- Favorable instantaneous samples are short telemetry windows, not sustained
  completed-request rates.
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
