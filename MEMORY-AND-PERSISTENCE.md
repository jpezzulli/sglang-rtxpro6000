# Memory Recovery and Automatic Capacity Calculation

## v2.3 FR-Spec allocation and cache identity

The 65,536-row BF16 draft head adds 320 MiB; startup also has a larger
temporary allocation while selecting its rows. The measured configuration
retains 824,384 target/native-MTP KV tokens, 524,288-token context, four
concurrent requests, 24 Mamba slots, CUDA graphs, and 32 GiB HiCache.

The token-map content hash is part of the NIXL namespace. Changing or removing
the map selects a separate cache representation. Tests covered new cache
writes, restart reuse, switching between FR-Spec and non-FR configurations,
concurrent restoration, and two dependent turns on a restored 490K conversation.

The FR-Spec recipe uses 85%/80% filesystem-usage thresholds for the NIXL
cleaner. Review these thresholds for your storage. The existing non-FR and
27B sample configurations are unchanged.

## v2.5.0 Flash-Next precision and PLE placement

The two v2.5.0 options affect different memory surfaces:

| Option | GPU effect | Host/storage effect | Cache identity |
|---|---|---|---|
| Online FP8 | Selected otherwise-BF16 transformer projections, HyperConnection mix weights and `lm_head` occupy less resident GPU memory. A measured boot left 7.52 GiB available after graphs versus 3.66 GiB in the earlier matching option-off boot. | PLE placement and HiCache are unchanged. | `online_mxfp8` is part of the NIXL namespace. |
| NVMe PLE | The qualified 824,384-token KV pool and CUDA graphs were retained. | The 47.68 GiB FP8 PLE table lives in a prepared immutable SSD overlay instead of fixed pinned RAM; bounded staging buffers and reclaimable filesystem cache remain. | Backend and overlay-manifest identity select a separate NIXL namespace. |

The online-FP8 difference is about 3.86 GiB of post-graph available VRAM; the
7.52 GiB value is the total remaining, not the amount newly freed. The primary
FR-Spec recipe keeps 824,384 as its default cap. A separate
`MAX_TOTAL_TOKENS=1000000` run allocated the full requested pool, retained
524,288-token context and graphs, and passed the selected long-context,
concurrency and CPU-media checks. It left 5.21 GiB after graphs and a minimum
sampled 1,187 MiB during media work. This qualifies that explicit capacity
shape, not a larger context or speed gain. Other overrides and combining the
1,000,000-token pool with model-GPU media preprocessing remain unqualified.

NVMe PLE's structural change removes the fixed 51,200,245,760-byte table from
pinned host residency, less its bounded reader buffers. Separate snapshots
showed about 54–56 GiB more host memory available with NVMe, but process state,
filesystem cache and other host activity also changed; do not attribute the
entire observed delta to PLE. The original checkpoint and prepared table must
remain immutable while serving.

Online FP8 does not introduce a new persisted QSA or recurrent-state format.
The NVFP4 expert, router and FP8 PLE-table formats remain unchanged; GDN
recurrent state remains BF16, KV remains FP8 E4M3, and FR-Spec target/draft
scale alignment is preserved. See [FP8.md](FP8.md) and
[NVME-PLE.md](NVME-PLE.md).

## Flash-Next GPU allocation

| Configuration | Intermediate SSM | Mamba slots | GPU KV capacity |
|---|---:|---:|---:|
| BF16 before RecoverSSM | 1.05 GiB | 24 | 745,600 tokens |
| RecoverSSM `none` mode | 0 | 24 | 824,384 tokens |

Moving recurrent state to BF16 first reduced bytes per Mamba slot. Automatic
Mamba sizing then increased the slot count from 21 to 49 and consumed much of
that saving. The workload did not need 49 slots, so the launcher fixed the pool
at 24—above observed demand—and allowed the remaining budget to flow to KV.

The PR #30967 integration then removed the 1.05 GiB intermediate speculative
SSM pool. Accepted state is reconstructed by the FlashInfer WY output-only path
and committed with the required Qwen4 PLE siblings. The measured KV pool grew
by 78,784 tokens, from 745,600 to 824,384 (+10.6%). No fixed KV-token count was
supplied.

### Allocator nuance

The physical intermediate pool is genuinely absent, but the active
`kv_cache_configurator.py::_handle_max_mamba_cache` estimator skips its reserve
only for ReplaySSM, not for RecoverSSM `none`. No local source commit corrected
that estimator. The qualified launcher changed `--mem-fraction-static` from
`.97` to `.981`, approximately the recovered fraction of this GPU, so automatic
KV sizing could use the physically freed memory. This is why the observed
capacity increase is safe and measured, but it is not evidence that upstream
allocation accounting is complete.

The final startup allocation was:

- target FP8 KV: 4.72 GiB K + 4.72 GiB V;
- native-MTP FP8 KV: 0.39 GiB K + 0.39 GiB V;
- Mamba/PLE pool: 24 slots, BF16 recurrent and convolution state;
- intermediate speculative SSM: 0;
- recovery graphs: batch buckets 1-4.

Peak physical use was 95,817 of 97,887 MiB, leaving 1,475 MiB. The final full
suite recorded no OOM, CUDA error, or retraction.

## Qwen3.8-27B/DFlash2 GPU allocation

The dated 2026-08-24 performance release and the later current-launcher
confirmation used different recurrent-state safety margins:

| 27B configuration | Mamba slots | States/path | Target/draft KV capacity |
|---|---:|---:|---:|
| Dated performance release | 16 | 3 | 1,194,496 tokens |
| Current launcher confirmation | 24 | 5 | 1,118,784 tokens |

The current configuration intentionally spends part of the token-pool headroom
on recurrent-state concurrency. Its startup allocation was:

- target FP8 KV: 17.07 GiB K + 17.07 GiB V;
- DFlash2 FP8 KV: 5.34 GiB K + 5.34 GiB V;
- Mamba convolution state: 0.07 GiB;
- Mamba SSM state: 3.52 GiB;
- speculative intermediate SSM: 5.62 GiB;
- intermediate convolution window: 0.05 GiB.

The full suite used at most 10 of 24 Mamba entries. The five-state path cap is a
retained-prefix correctness/concurrency setting, not a decode optimization.

## HiCache and NIXL

The deployed hierarchy is:

```text
GPU radix state
  -> HiCache host RAM (page-first, kernel I/O, write-through)
  -> NIXL POSIX FILE storage (io_uring + O_DIRECT)
```

Optional NVMe-backed PLE is parallel to this cache hierarchy: it is the
read-only source for embedding rows, not a replacement for HiCache or the NIXL
prefix store.

Flash-Next uses a 32 GB configured host tier. Its hybrid pool persists packed
target/native-MTP KV, complete GDN state, Qwen4 PLE accepted/pending/ngram
siblings, and compressed QSA index keys. The 27B configuration uses a 96 GB
host setting for target KV, Mamba/GDN state, and the DFlash2 sidecar. These are
configuration values, not claims that all hybrid pool components sum to exactly
that number.

Two portable NIXL corrections are in this runtime and open upstream:

- `8b786639e4` / PR #36520 assigns overlapping path registrations distinct
  device IDs;
- `067c639c0a` / PR #36524 chunks bounce-backed hybrid transfers at staging
  capacity and fails closed on malformed result vectors.

Flash-Next additionally required the PLE/QSA persistence sequence
`516e42a2ee..7b5cfb728d`: preserve every state sibling, align mixed bounce I/O,
restore PLE ngram before use, defer Mamba copy-on-write until after load, and
persist compressed QSA keys.

## Representation namespaces

NIXL's selected FILE root contains the persistent payload; process-local radix
and registry metadata are rebuilt on startup. The included namespace helper
derives a readable directory plus a 12-character SHA-256 suffix from:

- checkpoint revision/content identity and relevant metadata;
- exact SGLang source HEAD and tracked diff;
- context, topology, page geometry, dtypes, speculative shape, attention and
  recurrent-state modes;
- PyTorch version and CUDA architecture.

The v2.5.0 Flash-Next launchers also distinguish online-FP8 state, PLE backend,
and prepared-overlay manifest identity when applicable. An explicit KV-token
cap is not part of the identity because it changes capacity rather than the
stored page representation. Representation-relevant changes must not reuse an
incompatible namespace.

An exact manifest inside the root must match the derived identity. Identical
configuration after restart selects the existing directory. A representation-
relevant change selects another directory; switching back selects and reuses
the original root. Cache content is disposable: missing or incomplete state is
allowed to become a miss and ordinary recomputation.

The NIXL cleaner operates on whole-filesystem occupancy percentages, not bytes
owned by one namespace. The sample 54.6/53.0 thresholds were qualified on a
2 TB filesystem with a known non-cache baseline and target roughly 220-250 GiB
of cache. They must be recalculated for another filesystem; a dedicated mount
provides the clearest semantics.

## Restart evidence

```text
490K restart: 489,856 / 489,879 input tokens restored; 23 recomputed; 3/3 needles exact
```

With v2.5.0 online FP8 selected, an identical saved 490K request prefetched and
loaded back 489,984 KV tokens plus two Mamba states, recomputing only a five-
token suffix, and returned all three keys. With NVMe PLE selected and online
FP8 off, four concurrent saved 64K/490K requests restored 553,728 KV tokens and
four Mamba states. These are service-restart checks; they do not make the cache
transactional or prove every configuration across a full-machine reboot.
