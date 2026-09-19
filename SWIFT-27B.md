# Swift Qwen3.8-27B FP8 — preliminary validation

Swift 27B FP8 passed our reasoning validation and is a credible alternative
checkpoint for the Pennyroyal 27B/DFlash2 profile. In this run it generally
finished reasoning tasks with fewer tokens, although not every task was faster.
This is a preliminary model validation, not a new runtime release or a change
to the recommended Flash-Next configuration.

## Model and setup

We tested [d0xin/Swift-Qwen3.8-27B-FP8](https://huggingface.co/d0xin/Swift-Qwen3.8-27B-FP8),
a community compressed-tensors block-FP8 quantization of
[UkisAI's Swift-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-Qwen3.8-27b),
on September 18, 2026. It loaded without runtime code changes using Pennyroyal
v2.5.1 and [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2).

The configuration used one RTX PRO 6000 Blackwell 96 GB for inference, TP1,
FP8 target and KV, eight DFlash2 draft tokens, 524,288-token context, four
concurrent requests, 24 Mamba slots, and HiCache/NIXL. The actual GPU KV pool
was 1,119,552 tokens. Media preprocessing ran on a secondary RTX A4000;
the vision encoder remained on the model GPU.

## Reasoning: time to finish and tokens generated

The reference is the saved September 9 run of
[OrcaRouter Qwen3.8-27B Uncensored FP8](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8)
with DFlash2. Both runs used the same reasoning collector, medium reasoning
default and request schedule: C1 alone, C2–C4 together, C5–C7 together, then
C8 and its dependent correction. Measured requests did not impose a new
short output cap.

Wall time includes prefill and generation through the completed response.
Generated tokens include both reasoning and visible output, as reported by
the server. These are task-completion timings, not decode tokens per second.

| Reasoning case | Wall time: reference 27B → Swift | Generated tokens: reference 27B → Swift |
|---|---:|---:|
| C1 | 31.4 → 21.2 s | 4,512 → 3,207 |
| C2 | 27.0 → 18.2 s | 3,822 → 2,485 |
| C3 | 22.8 → 20.5 s | 4,722 → 4,258 |
| C4 | 74.3 → 83.4 s | 12,983 → 14,365 |
| C5 | 49.3 → 42.4 s | 6,939 → 6,046 |
| C6 | 13.9 → 11.1 s | 2,024 → 1,580 |
| C7 | 29.5 → 21.0 s | 5,968 → 4,179 |
| C8, both turns | 13.8 → 10.8 s | 2,629 → 2,085 |

Total generated tokens fell from **43,599 to 38,205**, about **12% fewer**.
Elapsed time for the measured reasoning suite fell from **169 to 158 seconds**,
about **6.5% less**. Because some requests ran concurrently, suite elapsed time
is not the sum of the table's rows. Warm-ups are excluded.

This is one run per checkpoint, and the reference used an older runtime.
It is not a controlled measurement of Swift tuning alone, nor evidence of
better reasoning quality. The benefit also depends on the task: C4 was slower,
and a separate short multi-step artifact workflow passed but took 9.9 seconds
with Swift versus 5.9 seconds with the reference.

## Other checks and the remaining gap

- Native tool selection and arguments were correct across all 30 tool-suite
  invocations; semantic review found two task-response misses.
- Vision, PNG/JPEG input and a static-video input check passed.
- Long-context recall returned all 64 needles at roughly 393K input tokens
  and all three separated needles at roughly 490K.
- The 64K prefill check and single-request/four-request 1,024-token decode
  checks completed successfully.

**NIXL restart reuse remains unverified for this checkpoint.** Cache writes
succeeded, but the disk cleaner evicted the saved test prefixes before restart.
The replayed answers were correct after recomputation; that does not establish
persistent-cache restoration.

The [validation repository](https://github.com/jpezzulli/pennyroyal-validation)
describes the reasoning and tool cases. For the existing launch configuration,
see the [27B/DFlash2 recipe](RUN.md#launch-27b-with-dflash2).
