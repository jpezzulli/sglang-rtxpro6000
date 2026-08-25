# Limitations

## Scope of the evidence

- One NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB, SM120.
- Tensor parallelism 1 only.
- One exact target checkpoint and one exact DFlash2 draft revision.
- This frozen SGLang-derived source and its CUDA/PyTorch/FlashInfer/NIXL stack.
- A 450 W configured GPU power limit for the final qualification.
- No claim for another GPU, driver, CUDA release, compiler, TP topology, model,
  quantization, draft, or current upstream SGLang.

The runtime admits 524,288 tokens, but the controlled exact-needle test was
489,921 prompt tokens. This is not proof for every possible 524K prompt,
modality, sampling configuration, or concurrent schedule.

## Performance interpretation

- Completed-request effective decode, instantaneous telemetry, controlled
  ceiling tests, and concurrent aggregate throughput are different metrics.
- The 250-300 tok/s figures are favorable short telemetry windows, not sustained
  completed-request rates.
- The real-use log includes significant context and acceptance variation. Its
  overall median is 131.31 tok/s, while short-context median behavior is around
  165 tok/s and 340K+ behavior is around 100 tok/s.
- Seven requests overlapped other completed request intervals. Their poor rates
  are valid user experience under interference, but not isolated engine speed.
- DFlash2 acceptance varies by generated content. It is a major explanatory
  variable, not a fixed runtime property.
- Controlled community comparisons are directional because checkpoint,
  runtime, speculation, power, client, prompt, context, and duration differ.
- The public TP1 baseline is independently visible and hardware/topology
  compatible, but checkpoint, runtime, speculation, power, cache/offload,
  request harness, and output-duration differences prevent a strict A/B claim.
- No repeat-variance campaign was run for every number in the real-use sample.
- No claim of "fastest" is made.

## Reasoning and tools

- The xhigh and medium reasoning campaigns used different scheduling profiles.
- The qualitative score was produced from a blinded packet but was not an
  independent preregistered external benchmark.
- Medium was much faster and shorter but scored 2.453 points below xhigh in
  this suite. That does not establish universal equivalence.
- Tool-call results apply to the identified 30-invocation suite. Two automatic
  misses were literal-grader false negatives; one reviewed miss was substantive.
- The target checkpoint is explicitly uncensored/abliterated. These scores do
  not characterize the official Qwen checkpoint's refusal or safety behavior.

## HiCache/NIXL

- HiCache/NIXL provides persistent prefix behavior; it is not credited for
  active GPU decode throughput.
- The configured 96 GB host tier produced 120.63 GB of actual hybrid host
  allocation. A host with only 128 GB RAM is not equivalent.
- Cleaner thresholds are whole-filesystem percentages. The included 68/65
  values do not guarantee a 500 GB cache on another filesystem.
- Process/service restart reuse was demonstrated. Full-machine reboot reuse was
  not repeated after namespace schema v2; it remains an inference from NIXL
  path-only FILE state rather than a qualified result.
- NIXL path-mode files lack a demonstrated temporary-file publication marker
  and strong interrupted-write validation. Cache state is disposable, so a bad
  entry may become recomputation, but failure-atomic persistence is not claimed.
- Inactive namespace roots are isolated from the active cleaner and need a
  separate lifecycle policy if many representations accumulate.
- Mooncake was removed and is not a tested fallback in this distribution.

## Reproducibility boundaries

- The build procedure is reconstructed from the successful installed state and
  retained build metadata. It has not yet been independently executed from a
  blank second host.
- The measured installation was a wheel through commit `2e79cdc030` plus a
  byte-identical overlay of the two NIXL adapter files from `8e197ed3af`. A clean
  build from the frozen final tree is expected to be equivalent but has not yet
  been cross-host validated.
- CUDA graph and JIT cache state affects startup and early-request behavior.
- Vision was smoked earlier in this engineering lane, but the final published
  performance and reasoning qualification is text-focused; no final vision
  quality suite is claimed.
- No model weights, cache data, private prompts, or hidden reasoning traces are
  distributed here.
