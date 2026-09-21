# Qwen3.8 Flash-Next NVMe-backed PLE on RTX PRO 6000

Pennyroyal v2.5.0 can stream Flash-Next's large FP8 PLE embedding table from a
prepared local NVMe snapshot, removing the table's fixed pinned-RAM residency.
The feature is optional on the single-GPU NVIDIA RTX PRO 6000 Blackwell
(SM120) recipe; RAM-backed PLE remains the default.

The qualified table is 51,200,245,760 bytes (**47.68 GiB**) on SSD. NVMe mode
replaces its fixed pinned-RAM residency with bounded row buffers, row-ID
staging and the reader's small native page pool. Filesystem page cache remains
reclaimable RAM. Separate snapshots showed host `MemAvailable` about
**54–56 GiB higher** with NVMe. Process state, filesystem cache, and other host
activity also differed between snapshots.

## Requirements and boundaries

- Linux x86-64, Python 3.12 and the qualified TP=1 Flash-Next runtime.
- Local SSD/NVMe storage for the prepared overlay. Do not place it on a media
  aggregation filesystem.
- The original complete Flash-Next checkpoint must remain present and
  immutable during both preparation and serving; `TARGET_MODEL` continues to
  identify it.
- Rust/Cargo and `uv` are needed once to build the isolated reader.
- NVMe placement supports Flash-Next; the 27B/DFlash2 recipe uses its existing
  memory path.

The public reference checkpoint is
[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4).
Other checkpoints need the same qualified Flash-Next architecture and PLE table
format; the preparation and launcher identity checks are authoritative.

## Install the optional reader

From the release checkout:

```bash
PYTHON="$PWD/.venv/bin/python" bash tools/ple_nvme/install.sh
```

The [installer](tools/ple_nvme/install.sh) builds into `.ple-nvme`, separate
from the main Python environment. It uses four Cargo build jobs by default.
Override either value before running it if needed:

```bash
export CARGO_BUILD_JOBS=2
export PENNY_PLE_PLUGIN_DIR=/path/to/isolated-ple-plugin
PYTHON="$PWD/.venv/bin/python" bash tools/ple_nvme/install.sh
```

Build tooling may download dependencies. Serving from an already prepared
snapshot works offline.

The reader checks the runtime's source signatures. When upgrading to v2.5.1,
install the reader from the new checkout; its QSA signature has changed.
The installer preserves existing destinations, so choose a new directory and
keep that setting for launch:

```bash
export PENNY_PLE_PLUGIN_DIR="$PWD/.ple-nvme-v251"
PYTHON="$PWD/.venv/bin/python" bash tools/ple_nvme/install.sh
```

A previously prepared 48 GiB overlay can be reused when the normal preflight
accepts the source config/index/header identity, prepared ordinary-weight
mapping, and prepared-table checksum. Changing only the isolated reader
directory leaves the overlay valid. The container image includes the matching
reader, so this reinstall step is for native installations.

## Prepare the NVMe overlay once

The [preparation helper](scripts/pennyroyal/prepare_ple_nvme.py) extracts the
PLE bytes exactly and links ordinary checkpoint files to the original snapshot.
It works locally without downloading, quantizing, or modifying the source
checkpoint. The output directory must be new.

```bash
.venv/bin/python scripts/pennyroyal/prepare_ple_nvme.py \
  --source /path/to/original-flash-next-checkpoint \
  --output /path/on/local-nvme/flash-next-ple
```

Allow at least 48 GiB plus filesystem overhead and any retained mixed-shard
tensors. Keep the source immutable throughout preparation and serving, and
keep the prepared overlay immutable while serving.

## Launch with NVMe PLE

Retain the normal compiler-cache, NIXL and model settings, then select `nvme`:

```bash
export TARGET_MODEL=/path/to/original-flash-next-checkpoint
export PENNY_PLE_BACKEND=nvme
export PENNY_PLE_NVME_MODEL=/path/on/local-nvme/flash-next-ple
configs/pennyroyal/serve-flash-next-frspec.sh
```

The non-FR-Spec launcher accepts the same options. If the reader was installed
outside `.ple-nvme`, also set `PENNY_PLE_PLUGIN_DIR` to that exact directory.

Use `PENNY_PLE_BACKEND=ram`, or leave it unset, to use the original pinned-RAM
path. An explicit NVMe selection fails at startup on an error. The plugin loads
only in NVMe mode; RAM-backed Flash-Next and the 27B launcher do not discover
it automatically.

## Startup integrity and persistence

The launcher checks runtime-source compatibility, source config/index/header
identity, and the prepared weight mapping. Ordinary overlay files and
directories must remain linked to the original checkpoint, including generation,
image and video configuration; only the deliberate PLE/index rewrites are
excepted. The configured cache paths are active before preflight imports any
runtime dependencies. Startup also reads the prepared external table
sequentially to verify its checksum before model workers begin, so this stage
can take time. Each launch checks the prepared table rather than rehashing the
original 47.68 GiB PLE payload; the original source must remain immutable
during preparation and serving. A
changed prepared table, incompatible runtime, incomplete overlay, wrong source
checkpoint, or required hook-application failure stops startup.

NVMe mode receives a separate NIXL namespace containing the overlay manifest
identity. Existing RAM-mode namespaces are neither deleted nor modified.
Both standalone NVMe and combined online-FP8/NVMe modes passed identical-
restart checks with four concurrent saved 64K/490K responses.

The NVMe PLE table and NIXL prefix store serve different purposes and may share
an SSD only if its I/O budget is sufficient. Sharing introduces possible
contention; use separate fast local devices when predictable latency matters.

## Measured tradeoff

Both modes retained 524,288-token context, the 824,384-token KV pool, page
size 64, native NEXTN/FR-Spec graphs, 32 GiB HiCache and NIXL. These RAM/NVMe
measurements and the 553,728-token restore used online FP8 **off**. Combined
mode results are listed separately below.

| PLE placement | C4 aggregate samples | Median | Host-memory observation |
|---|---|---:|---|
| RAM | 428.90, 286.16, 434.92 tok/s | **428.90 tok/s** | About 67 GiB available after the separate RAM comparison |
| NVMe | 369.73, 259.86, 391.95 tok/s | **369.73 tok/s** | About 121–123 GiB available after NVMe workload checks |

Each C4 run used four simultaneous requests with exactly 1,024 completion
tokens per stream. Aggregate throughput includes TTFT and synchronized batch
makespan. Both modes had a slower second run with roughly five-second TTFT.
These were separate operational comparisons with different cache/JIT history,
rather than a randomized or fully cache-controlled A/B. NVMe reduced fixed
host residency and had a lower median in this comparison.

NVMe mode also passed cold 64K and 490K exact-retrieval prefills, schema/tools,
image and long image-history checks, CUDA-graph capture, and restart
restoration. The short host snapshots measure available memory during those
runs; they do not isolate the source of all host-memory pressure.

### Combined with online FP8

The two options were also exercised together with the standard 824,384-token
pool, 524,288-token context, NVMe PLE, CPU media preprocessing and one visible
RTX PRO 6000. Post-graph free GPU memory was 7.64 GiB. Nine
warmup/schema/tool requests, 64K and 490K retrieval, fixed-output C1/C4, and all
ten selected image/video scenarios passed.

Identical restart restored **553,728 storage/prefetch/KV tokens and four Mamba
states**, with 8,068,005,952 bytes loaded back. All four saved responses were
correct; synchronized client makespan was 9.90 seconds. These token counters
span two distinct prefixes, not one request exceeding the 524,288-token limit.

After basic and media warmup, three 1,024-output-token C1 runs measured 172.38,
175.71 and 171.52 tok/s post-first-token (median **172.38 tok/s**). Three C4
runs measured 448.30, 442.15 and 438.26 tok/s synchronized aggregate (median
**442.15 tok/s**), with four requests running together and TTFT around
0.67–0.71 seconds. The first C1 sample was 123.08 tok/s post-first-token; the
first C4 aggregate was 254.35 tok/s with roughly 7.6-second TTFT. These values
show material first-use variation.

These runs establish combined-option function and throughput. The earlier
207.12 tok/s online-FP8/RAM-PLE observation came from a different measurement
window and cannot serve as a matched RAM-versus-NVMe comparison.

## Source and credit

The optional reader is an attributed adaptation of Garner McCloud's
[SSD Stream v0.2.0](https://github.com/garnermccloud/sglang-ssd-stream/tree/176a522ef9d6dbb5056ae1f467fe49af0f1258a5)
at commit `176a522ef9d6dbb5056ae1f467fe49af0f1258a5`, licensed under Apache-2.0.
Its [license](tools/ple_nvme/ssd_stream/LICENSE) and
[NOTICE](tools/ple_nvme/ssd_stream/NOTICE) are retained with the vendored
source. The integration also acknowledges AntigravityAI's
[Pennyroyal field report](https://github.com/jpezzulli/sglang-rtxpro6000/issues/2).

Pennyroyal adds explicit opt-in/fail-loud registration, exact source and model
guards, table checksum validation, an offline exact-byte preparer, and launcher
integration. It uses the isolated reader path rather than the upstream
all-in-one CLI installer; QSA/MTP remain part of the Pennyroyal runtime. The
adapted reader identifies itself as `0.2.0+pennyroyal2`; the credited upstream
release remains SSD Stream v0.2.0.
