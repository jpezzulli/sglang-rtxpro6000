# Optional NVMe-Backed PLE for Flash-Next

Pennyroyal v2.5.0 can stream Flash-Next's large FP8 PLE embedding table from a
prepared local NVMe snapshot instead of pinning the table in host RAM. It is an
**opt-in RAM-saving feature**; RAM-backed PLE remains the default.

The qualified table is 51,200,245,760 bytes (**47.68 GiB**) on SSD. NVMe mode
replaces its fixed pinned-RAM residency with bounded row buffers, row-ID
staging and the reader's small native page pool. Filesystem page cache remains
reclaimable RAM. Observed host `MemAvailable` was about **54–56 GiB higher** in
separate snapshots, but not all of that difference can be attributed to the
PLE table.

## Requirements and boundaries

- Linux x86-64, Python 3.12 and the qualified TP=1 Flash-Next runtime.
- Local SSD/NVMe storage for the prepared overlay. Do not place it on a media
  aggregation filesystem.
- The original complete Flash-Next checkpoint must remain present and
  immutable during both preparation and serving; `TARGET_MODEL` continues to
  identify it.
- Rust/Cargo and `uv` are needed once to build the isolated reader.
- NVMe placement is not supported by the 27B/DFlash2 recipe.

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
snapshot does not require network access.

The installer deliberately refuses to overwrite an existing destination, and
v2.5.0 rejects earlier reader builds that lack its complete hook-application
enforcement. If you are replacing an earlier optional-reader test install,
select a new empty directory and keep that setting for launch:

```bash
export PENNY_PLE_PLUGIN_DIR="$PWD/.ple-nvme-v250"
PYTHON="$PWD/.venv/bin/python" bash tools/ple_nvme/install.sh
```

A previously prepared 48 GiB overlay can be reused when the v2.5.0 preflight
accepts the source config/index/header identity, prepared ordinary-weight
mapping and prepared-table checksum. It does not need to be rebuilt solely
because the isolated reader directory changed.

## Prepare the NVMe overlay once

The [preparation helper](scripts/pennyroyal/prepare_ple_nvme.py) extracts the
PLE bytes exactly and links ordinary checkpoint files to the original snapshot.
It does not download, quantize, or modify the source checkpoint. The output
directory must not already exist.

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
path. An explicit NVMe selection never falls back silently. The plugin is
imported only in NVMe mode, which keeps RAM-backed Flash-Next and the 27B
launcher isolated from plugin auto-discovery.

## Startup integrity and persistence

The launcher checks runtime-source compatibility, source config/index/header
identity, and the prepared weight mapping. Ordinary overlay files and
directories must remain linked to the original checkpoint, including generation,
image and video configuration; only the deliberate PLE/index rewrites are
excepted. The configured cache paths are active before preflight imports any
runtime dependencies. Startup
also reads the prepared external table sequentially to verify its checksum
before model workers begin, so this stage can take time. It does not rehash the
original 47.68 GiB PLE payload against the prepared table on every launch; the
original source's immutability is a preparation-and-serving precondition. A
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
measurements and the 553,728-token restore used online FP8 **off**; they are not
combined online-FP8-plus-NVMe performance results.

| PLE placement | C4 aggregate samples | Median | Host-memory observation |
|---|---|---:|---|
| RAM | 428.90, 286.16, 434.92 tok/s | **428.90 tok/s** | About 67 GiB available after the separate RAM comparison |
| NVMe | 369.73, 259.86, 391.95 tok/s | **369.73 tok/s** | About 121–123 GiB available after NVMe workload checks |

Each C4 run used four simultaneous requests with exactly 1,024 completion
tokens per stream. Aggregate throughput includes TTFT and synchronized batch
makespan. Both modes had a slower second run with roughly five-second TTFT.
These were separate operational comparisons, not a randomized or fully
cache-controlled A/B. NVMe is functional and reduces fixed host residency, but
the evidence does **not** support a universal “no speed cost” claim.

NVMe mode also passed cold 64K and 490K exact-retrieval prefills, schema/tools,
image and long image-history checks, CUDA-graph capture and restart restoration.
Short host-wide samples do not prove that it eliminates memory compaction or
attribute all memory pressure to Pennyroyal.

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

This is direct combined-option functionality and throughput evidence, not a
fresh controlled RAM-versus-NVMe A/B. The earlier 207.12 tok/s online-FP8/RAM
PLE C1 observation came from a different measurement window and is not a
matched comparison.

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
integration. It does not expose the upstream all-in-one CLI installer or ship a
replacement QSA/MTP runtime payload. The adapted reader identifies itself as
`0.2.0+pennyroyal2`; the credited upstream release remains SSD Stream v0.2.0.
