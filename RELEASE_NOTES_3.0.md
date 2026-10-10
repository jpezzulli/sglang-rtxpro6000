# Penny Royal 3.0

*Draft — ready for publication.*

Penny Royal 3.0 brings adaptive speculative decoding, faster small-batch
kernels, and more room for conversation state to Qwen3.8 Flash-Next. The target
is RTX PRO 6000 Blackwell with one card (TP1) or two (TP2). Qwen3.8-27B FP8
with DFlash2 remains the second supported profile.

## Faster Flash-Next generation

- **Adaptive MTP is enabled by default, and adapts only at C=1.** One active
  request switches between four- and eight-token windows according to draft
  acceptance. At **C=2 and above**, draft width is fixed at four; speculative
  decoding continues. A server configured for C6 can still adapt while only
  one request is active. C counts requests, independently of TP1/TP2 GPU count.
  Per-width graph state and target tuning preserve state when the width changes.
  **FR-Spec stays enabled at every concurrency:** it narrows the draft vocabulary,
  while adaptive MTP selects draft length. The target keeps its full vocabulary.
- **FP8 and small-batch kernels select themselves.** Eligible SM120 launches
  use rowwise-FP8 dense weights and specialized projection/output-head kernels
  that read those weights directly. No copied benchmark flags are needed.
- **Less work in each draft step.** Dedicated HyperConnection mixing, BF16 MTP
  entry projection, NVFP4 draft-MoE, and packed-key routing kernels reduce work
  repeated for every generated token. The router preserves expert selection.
- **GDN improvements.** Fused normalization/output projection removes an
  intermediate memory round trip. Extend/prefill computes gating beta in FP32
  to avoid the previous BF16 rounding.

## More room for context

Attention setup no longer constructs a backend that QSA immediately replaces,
and compatible QSA decode paths share scratch space. These savings recovered
adaptive MTP's extra memory cost. On our one-card C6 setup, the GPU token pool
is **1,067,072 tokens**, with **4.194 GiB free at startup**.

The pool is shared across requests and cached prefixes. The per-request
context limit remains **524,288 tokens**.

## Fixes for agents and long sessions

- Reserve speculative Mamba buffers from their actual allocation requirements,
  leave context room for speculative lookahead, and retain state at the
  committed-token boundary.
- Keep useful host Mamba checkpoints, release abandoned session branches, and
  track shared-backup ownership. Session retention uses the client's session
  IDs and close calls.
- Preserve literal thinking tags in answer text after reasoning ends.
- Emit Responses reasoning-part events and report capped or failed turns with
  the correct terminal status.
- Count tool-parser outcomes in Prometheus without logging prompts or arguments.
- Reject an image request whose shared-memory segment has disappeared, release
  its feature buffers, and keep the scheduler serving other requests.
- Separate persistent cache namespaces for the effective FP8 representation
  and GDN mode, and correct WSL2 host-memory aliases and hybrid cache keys.

## Simpler installation and a slimmer release

- **Configuration lives outside the container.** Your host `run.sh` chooses the
  image, GPUs, ports and mounts. Your host startup file holds model and SGLang
  settings. Docker mounts that configuration directory read-only at `/config`,
  and the container runs the mounted file. Models and caches have separate
  mounts. Edit and restart to change settings; keep the same files when updating
  the image. No image rebuild or Compose setup is required.
- **Optional NIXL disk caching.** It stays on by default. Turning it off keeps
  GPU and RAM caching; PLE placement is independent.
- **Beta configurator.** Generates ordinary timestamped launch files, leaves
  previous files untouched, and prints the command for the files just saved.
- **FlashInfer 0.7.0 post1.** Matching CUDA 13 providers and the accepted SM120
  MoE/GDN sources are included. Interrupted native-extension builds recover on
  the next run, and build failures retain compiler diagnostics.
- **The release went on a diet.** We kept **RTX Blackwell** for RTX PRO 6000
  inference and **Ampere/Ada Lovelace** for image-processing sidecars, and
  trimmed the other GPU build targets. A slimmer release for easier downloads.
  CPU preprocessing remains available.

## Performance on one RTX PRO 6000

OrcaRouter Qwen3.8 Flash-Next ModelOpt NVFP4, FR-Spec plus adaptive MTP, 524K context limit,
normal non-greedy sampling. Rates come from SGLang's native generation logs.

| Workload | Mean tok/s | Sampled peak tok/s |
|---|---:|---:|
| Real coding workflow | **262.32** | **518.18** |
| Presentation workflow | **239.53** | **543.72** |
| Single request | 218.63 | 326.56 |
| Four concurrent requests, aggregate | 489.35 | 636.71 |
| Six concurrent requests, aggregate | 433.54 | 606.48 |

Reasoning is included. Prefill, tools, and idle time are measured separately.
The peaks come from the same measured windows as the means.

| Cold prompt | Mean input tok/s | Time to first token |
|---|---:|---:|
| 64K | 17,235.90 | 4.71 s |
| 490K | 8,839.75 | 60.37 s |

### Recorded improvement over 2.5.3

The saved native regular C1 mean rose from **186.79 to 218.63 tok/s** (**+17.0%**),
and the C6 GPU token pool grew **2.7%**, from **1,039,040 to 1,067,072 tokens**.
The context limit remains 524,288. These are the recorded release configurations:
medium reasoning in the October 6 baseline, xhigh in the October 10 run.
[Results](RESULTS.md#recorded-comparison-with-253) retains the request settings.

### Public code-edit score

The accepted adaptive screen reached **491.88 tok/s mean / 574.16 sampled peak**
on the public code-edit prompt with normal non-greedy sampling. Our saved 2.5.3
code-edit set averaged **336.57 tok/s** using greedy sampling. The 3.0 screen
used the accepted code with a 1,048,576-token pool before the final pool increase.

The prompt is a mechanical rewrite of supplied code, which gives speculative
decoding highly predictable text. It is reported separately from the real
coding workflow's **262.32 mean / 518.18 peak**, which includes model turns
that use tools and write and test code. [Code-edit settings](RESULTS.md#public-code-edit-benchmark)
are recorded alongside the measurements.

The reasoning suite scored **95.69/100**, with all nine scenarios completing.
All 30 tool responses were parseable; 28 passed automatically and two hit a
confirmed fixture-classifier error. Image input and RAM cache restoration
also passed. [Detailed results](RESULTS.md#penny-royal-30) and the
[machine-readable release notes](release-3.0.json) retain the exact conditions.

## Thanks

- **[aiueo52](https://github.com/aiueo52)** — [Flash-Next
  optimization work](https://github.com/aiueo52/flash-next-rtxpro6000) and the discussions that helped us bring it into Penny.
- **[Tyler Landis / LandOfLemons](https://github.com/LandOfLemons)** — session
  and host-cache work, WSL2 fixes, and integration feedback.
- **[Martin Kazimir Kaplan / cube4elements](https://github.com/cube4elements)**
  — the image shared-memory crash report and fix.
- **[untcoder2](https://github.com/untcoder2)** and
  **[StockSpecialist1707](https://www.reddit.com/user/StockSpecialist1707/)** —
  TP2 field testing and practical configuration feedback.
- **[palves](https://github.com/palves)** — container configuration feedback
  and shared launch customizations.

**[Penny Royal](mailto:Pennyroyal@agentmail.to) did almost all the coding for this release**,
including the ports and integration work. The engineering was a collaboration
between [John](https://github.com/jpezzulli), Penny and Codex. We worked through
the architecture, tuning and debugging together, right down to fused kernels,
memory layouts and milliseconds per decode cycle. Lots of back and forth
figuring out why something that *should* be faster wasn't, and deciding what
was actually worth keeping.
