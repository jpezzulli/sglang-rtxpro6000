# Results

All throughput is measured on one RTX PRO 6000 96 GB. Controlled requests used
the ordinary OpenAI-compatible chat path with thinking enabled. Results from
the two model configurations are intentionally separate.

## Flash-Next final run — source `64ecd64924`

| Test | Result |
|---|---:|
| 64K cold prefill | 10,103.70 tok/s |
| ~490K cold prefill | 7,872.15 tok/s |
| Long-context needles | 3/3 exact |
| 1x 1,024-token decode | 171.09 tok/s |
| 4x 1,024-token synchronized batch | 427.54 tok/s aggregate |
| Per-stream post-first-token decode | 115.13 / 127.56 / 126.64 / 122.96 tok/s |
| MTP mean accepted length | 2.58 |
| MTP acceptance rate | 52.74% |
| Reasoning | 97.49/100; 139,863 completion tokens |
| Tools | 30/30 semantic; 27/30 strict literal checker |
| Vision | complete 1,024-token validation passed |
| Sealed agentic control | 148.80 tok/s |
| Natural 3,072-token decode | 162.05 tok/s |

The four-request aggregate uses all 4,096 output tokens over the synchronized
9.580393-second batch. Per-stream rates use 1,023 tokens over each stream's own
post-first-token interval, so their arithmetic sum is not the aggregate metric.

### Persistent restart

| Prefix | Restored | Recomputed | Effective restored-prefix rate |
|---|---:|---:|---:|
| ~64K | 63,808 | 56 | not used as cold-prefill comparison |
| ~490K | 489,856 | 23 | 62,040.60 tok/s |

All three needles remained exact after the 490K restore. The effective rate
measures restoration plus a tiny computed suffix, not model execution over
490K cold tokens.

### Real agentic window

| Measurement | Result |
|---|---:|
| Requests / output tokens | 96 / 100,666 |
| Token-weighted / median / mean | 139.5 / 153.5 / 155.1 tok/s |
| Sustained completed-request peak | 218.8 tok/s |
| Instantaneous single / four-stream peak | 247.0 / 543.1 tok/s |
| 90K-279K weighted / median | 138.7 / 148.4 tok/s |

The first post-prefill telemetry sample was excluded because its window mixed
prefill or idle time with decode. The window showed no high-context cliff;
acceptance variation dominated the visible throughput spread.

## Qwen3.8-27B + DFlash2 qualified run

This is the verified 27B runtime-line evidence, not a reuse of Flash-Next data.

| Test | Result |
|---|---:|
| 64K prefill | 6,163 tok/s |
| ~490K prefill | 1,618 tok/s; three needles exact |
| 1x decode | 108.75 tok/s |
| 4x decode | 390.23 tok/s aggregate |
| Reasoning, xhigh | 98.26/100 |
| Reasoning, medium | 95.807/100 |

The final qualified 27B allocation used 1,118,784 target and draft KV tokens,
FP8 E4M3 for both pools, 24 Mamba slots, five retained states per path, and
FP32 recurrent state. Core DFlash2 support was in the upstream base; local XQA
and NIXL corrections remained active.

The maintained public validation logs and harness are at
[`jpezzulli/pennyroyal-validation`](https://github.com/jpezzulli/pennyroyal-validation).
