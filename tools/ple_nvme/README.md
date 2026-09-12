# Optional NVMe PLE for Flash-Next

`PENNY_PLE_BACKEND=ram` remains the default. Both Flash-Next launcher recipes
also accept `nvme`; the 27B launcher is unchanged. This changes storage, not
PLE precision or model weights. The FR-Spec recipe defaults to the qualified
824,384-token KV cap; either recipe accepts a positive, page-64-aligned
`MAX_TOTAL_TOKENS` override. The non-FR-Spec recipe leaves capacity automatic
when the variable is unset. Startup logs remain the source of truth for actual
capacity; the launcher does not silently shrink a requested cap.

The FP8 PLE table occupies approximately47.68GiB when pinned in RAM. Streaming
uses two16MiB row buffers, row-ID staging and a32MiB reader pool instead.
Filesystem page cache may still occupy RAM, but can be reclaimed. This is not
a promise of47GiB more `free` RAM or any decode-speed improvement.

## Prepare once

Use an existing complete local Flash-Next checkpoint. The preparation helper
copies only PLE table bytes into a new directory and links ordinary assets to
the original snapshot. Keep that original snapshot in place. Allow roughly48GiB
of SSD space plus any mixed-shard retained tensors; no download or quantization
is performed. The output directory must not already exist.

```bash
PYTHON="$PWD/.venv/bin/python" bash tools/ple_nvme/install.sh
.venv/bin/python scripts/pennyroyal/prepare_ple_nvme.py \
  --source /path/to/original-checkpoint \
  --output /path/on/local-nvme/flash-next-ple
```

The installer needs Rust/Cargo and uv. It builds the optional reader with four
jobs by default into `.ple-nvme`, not the main Python environment. Set
`CARGO_BUILD_JOBS` and `PENNY_PLE_PLUGIN_DIR` to override these. It refuses an
existing destination rather than overwriting an installed copy. Build tooling
may download dependencies; inference needs no network access to this snapshot.

## Select in the existing recipe

Retain the recipe's usual compiler-cache, NIXL, template and GPU settings:

```bash
export TARGET_MODEL=/path/to/original-checkpoint
export PENNY_PLE_BACKEND=nvme
export PENNY_PLE_NVME_MODEL=/path/on/local-nvme/flash-next-ple
bash configs/pennyroyal/serve-flash-next-frspec.sh
```

Use `serve-flash-next.sh` for the no-FR-Spec variant. Choose `ram` to return to
the original checkpoint path and pinned-table arguments. No automatic fallback
occurs when NVMe is explicitly selected. No NUMA or kernel policy is changed.
Media preprocessing defaults to CPU; `SGLANG_MM_PREPROCESS_DEVICE=cuda:0` and
`cuda:1` remain available when the chosen device is visible to the process.
Online SM120 MXFP8 is independently opt-in with the literal setting
`SGLANG_SM120_ONLINE_MXFP8=true`; its default is `false`.
The optional reader is imported only in NVMe mode; its registration also requires
the explicit mode, protecting RAM/27B startup from plugin auto-discovery.
The launcher requires Pennyroyal adapter build `0.2.0+pennyroyal2`; older local
adapter builds are rejected because they lack the complete hook-application guard.

The launcher checks source compatibility and source/overlay identity. Server
startup hashes the external PLE table before workers start; allow time for this
sequential read. An invalid artifact or incompatible runtime fails startup.
NVMe uses a separate NIXL namespace carrying its manifest identity; old caches
are neither deleted nor modified. The backing table and source snapshot must
remain immutable while serving. Place the table on local SSD storage. Sharing a
device with NIXL introduces possible I/O contention.

## Source and limits

`ssd_stream/` is an attributed adaptation of Garner McCloud's
[SSD Stream v0.2.0](https://github.com/garnermccloud/sglang-ssd-stream/tree/176a522ef9d6dbb5056ae1f467fe49af0f1258a5),
Apache-2.0, retaining its license. AntigravityAI's
[Pennyroyal field report](https://github.com/jpezzulli/sglang-rtxpro6000/issues/2)
established prior integration against Penny v2.3.0. Its later NVFP4-pinned PLE
branch is not this FP8 streaming path.

The reader, gather implementation and graph adapter are unchanged from that
release. Local changes are opt-in/fail-loud registration, exact Penny source
guards, payload checksum validation and launcher/artifact preparation. The
upstream CLI runtime installer is not exposed by this package; no replacement
QSA/MTP runtime payload is included. The adapter retains Penny's hash calculation
but replaces the pinned-table gather with SSD row staging.

This integration is limited to the qualified TP1 Flash-Next configuration on
Linux x86_64 with Python3.12. It does not claim support for TP2, CPU expert
offload, prefill graphs, or alternate speculative modes. NVMe reduced fixed host
residency in qualification, but measured decode throughput was lower than RAM;
it is a capacity/host-pressure option, not a speed-neutral default.
