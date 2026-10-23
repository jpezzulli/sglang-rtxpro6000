# Penny Royal configuration reference

This is the complete list of user-facing settings for Penny Royal 3.1: what
each one does, its default, where you set it, what it interacts with, and
whether changing it costs you cache. It covers the same surface the launch
recipes, environment example, container startup files and the beta
configurator already use — nothing here is a new option.

The two primary setup routes are the [native build](BUILD.md) with the recipes
in `configs/pennyroyal` (guide: [RUN.md](RUN.md)) and the
[prebuilt container](docker/pennyroyal/README.md) with host `run.sh` plus host
startup files. The [beta configurator](CONFIGURE.md) is a separate optional
helper that writes the same ordinary files.

## Where settings live

| Route | File | Holds |
|---|---|---|
| Native | `~/.config/pennyroyal/pennyroyal.env` (or any `--config` path) | Saved `KEY=value` settings for `./run-penny`; plain lines read as data, never shell-sourced |
| Native | Your shell environment | Any variable below that you did not save; unset means the recipe decides |
| Native | `configs/pennyroyal/serve-*.sh` | The qualified recipes themselves — authoritative for inference flags |
| Container | `run.sh` | Image tag, GPU ids, port, mount roots, UID/GID, selected startup file, `--no-nixl` |
| Container | `config/start-*.sh` | Model paths and every recipe knob, exported before the launch line |
| Container | `config/nixl-*.toml` | NIXL cleaner watermarks and storage settings |
| Compose | `.env` next to `compose.yaml` | The legacy manual-Compose path; forwarded values are listed under each key |
| Configurator | Saved native/container settings | A subset of the keys below; every save writes new timestamped files and leaves your older files untouched |

Precedence, native and container alike: a key **saved** in your configuration
file wins over an inherited environment variable; `--config PATH` wins over
`PENNYROYAL_CONFIG`, which wins over the default location. A key you never set
keeps the recipe's qualified default. Nothing is rewritten behind your back.
`./run-penny --check` validates settings and paths without loading a model,
and `--show-config` prints a saved file's resolved values.

## Profile defaults

Select the profile with `PENNYROYAL_PROFILE`. Values marked *pinned* are fixed
in that recipe's launch line; a saved override for them does not reach the
server.

| | `next-dflash2` (experimental / alpha) | `27b` (experimental / alpha) | `next` (default) | `next-plain` |
|---|---|---|---|---|
| Target | OrcaRouter ModelOpt NVFP4 Flash-Next | Qwen3.8-27B FP8 | Flash-Next NVFP4 | Flash-Next NVFP4 |
| Draft | `incoai/Qwen3.8-Flash-Next-DFlash2`, BF16 unquantized | `incoai/Qwen3.8-27B-DFlash2` | none (in-checkpoint NEXTN head) | none |
| Speculation | DFLASH; FR-Spec 65,536-ID map; width routed: block **6** at one running request, **4** at two or more; accepted draft tokens verified by FlashInfer's SM120 kernel | DFLASH, fixed block 8 | NEXTN MTP + FR-Spec; width fixed by concurrency: block **8** at one running request, **4** at two or more | NEXTN MTP only |
| `--mem-fraction-static` | 0.95 | 0.92 (pinned) | 0.97 | 0.981 |
| Requests / Mamba slots | 6 / 36 | 4 / 24 (pinned) | 4 / 24 | 4 / 24 |
| Context | 524,288 (pinned) | 524,288 (pinned) | 524,288 (pinned) | 524,288 (pinned) |
| Draft window | 2,048 (pinned) | 2,048 (pinned) | — | — |
| KV pool | sized automatically; `MAX_TOTAL_TOKENS` optional | automatic | automatic | automatic |
| HiCache default | 32 decimal GB | 96 decimal GB | 32 | 32 |
| NIXL config | `nixl-posix.toml` | `nixl-posix.toml` | `nixl-posix-frspec.toml` | `nixl-posix.toml` |
| TP | 1 default; 2 accepted | 1 (pinned) | 1 default; 2 accepted | 1 default; 2 accepted |
| PLE placement | `ram` or `nvme` | `ram` only | `ram` or `nvme` | `ram` or `nvme` |

The verify width on every maintained Flash-Next profile is fixed by active
concurrency through a bundled policy file: the native-MTP recipes read
`configs/pennyroyal/adaptive-next.json` (block **8** at one running request,
block **4** at two or more; the recipes pin its SHA-256 `3edaa87d66f535caae0bddb8f457f03fce0f6dc1f4e8091b20495a0dc9da41fb`) and
the DFlash2 profile reads `configs/pennyroyal/adaptive-dflash2.json` (block 6
at one running request, block 4 at two or more). One candidate width per tier,
and each launch carries the widest routed block (`--speculative-num-steps 7
--speculative-num-draft-tokens 8` native; `--speculative-num-steps 5
--speculative-num-draft-tokens 6` DFlash2). The DFlash2 profile verifies
accepted draft tokens with FlashInfer's SM120 kernel
(`--linear-attn-verify-backend flashinfer`), fixed in the recipe and startup
files; there is no public toggle for it. The 3.0-era
`SPECULATIVE_NUM_DRAFT_TOKENS` setting therefore does not exist on this
profile — setup names it as unavailable rather than forwarding a value the
policy would ignore. The routed profile also keeps a batch-1 decode graph
bucket and runs with `--disable-prefill-cuda-graph`.

## Models, paths and identity

| Setting | Default | Notes |
|---|---|---|
| `TARGET_MODEL` | required | Target checkpoint directory (native: absolute or relative to `REPO_ROOT`; container: the path inside `/models`). |
| `DRAFT_MODEL` | required for `27b` and `next-dflash2` | Separate DFlash2 draft checkpoint. It is a distinct download from the target; the native-MTP profiles need no draft. |
| `PENNYROYAL_PROFILE` | `next` | New setups default to the maintained native-MTP Flash-Next profile with FR-Spec (`next-plain` is the same without the map). `next-dflash2` keeps its existing recipe and stays available as an **experimental / alpha** opt-in (see [README](README.md#dflash2-maturity)); saved profile choices are always preserved. |
| `REPO_ROOT` / `VENV_PATH` | native: your checkout and its `.venv` | Recipes use `$REPO_ROOT/.venv/bin/sglang` and its Python. |
| `SGLANG_EXE` / `PYTHON` | derived | Overrides for an environment outside the checkout. |
| `CACHE_BASE` | native `~/.cache/pennyroyal`; container `/cache` mount | Compiler/JIT and runtime caches; recipes create subdirectories. WSL2 note below. |
| `NIXL_STORAGE_BASE` | native `~/.local/share/pennyroyal/nixl`; container `/nixl` mount | Persistent NIXL root; required only while `NIXL=on`. |

<a id="capacity-and-memory"></a>

## Capacity and memory

| Setting | Default | Notes |
|---|---|---|
| `PENNY_HICACHE_SIZE_GB` | 32 (Next profiles) / 96 (`27b`) | Host-RAM HiCache tier in **decimal GB** (`size × 1e9` bytes), a positive integer ≥ 1. Blank keeps the profile default. Sizing below the default leaves less reused state; it does not disable HiCache or NIXL and does not touch the GPU pool. |
| `MAX_TOTAL_TOKENS` | unset — automatic | The KV pool is sized from the memory SGLang profiles at boot, inside the recipe's static memory fraction; there is no preset token cap any more. A saved value is a deliberate positive cap aligned to the recipe's page size (64); SGLang clamps it to what the machine fits and reports the resolved pool. |
| `MAX_RUNNING_REQUESTS` | per-profile table above | Admitted requests. Pinned to 4 on `27b`. |
| `MAX_MAMBA_CACHE_SIZE` | per-profile table above | Recurrent-state slots; pinned to 24 on `27b`. The documented six-request Flash-Next shape is 6 / 36. |
| `SGLANG_HICACHE_NIXL_MAX_CACHE_GB` | 0 (no cap) | Soft byte budget for the NIXL FILE cache, in GiB. This is a cleanup target (cleanup aims for 90% of the budget), not a hard quota, and it covers the active namespace only. **0 means uncapped, not disabled** — disabling is `NIXL=off`. Only meaningful while the disk tier is mounted. |
| `NIXL` | `on` | Disk tier. `off` keeps the GPU radix cache and the RAM HiCache tier and deletes nothing; the NIXL root, config and storage arguments simply are not used. In the container, the choice must agree with `run.sh` (`--no-nixl` skips the `/nixl` mount). |
| `PENNY_PLE_BACKEND` | `ram` | Flash-Next PLE table placement: RAM residency or an `nvme` snapshot streamed from SSD ([NVME-PLE.md](NVME-PLE.md)). Independent of `NIXL`; combining NVMe PLE with `NIXL=off` needs `run.sh --nvme-ple` for the io_uring permission. |
| `PENNY_PLE_NVME_MODEL` | none | Prepared NVMe PLE snapshot directory. Meaningless for `27b`. |

## GPUs, TP and media

| Setting | Default | Notes |
|---|---|---|
| `GPU` (native saved settings) / `NVIDIA_GPU` (Compose `.env`) / `--gpu` in `run.sh` | `0` | The inference GPU (index or UUID). Exported as `CUDA_VISIBLE_DEVICES` natively; Compose reserves exactly that device. |
| `TP_SIZE` | 1 | Tensor-parallel ranks, positive integer. TP1 is the qualified default on every profile; `27b` pins 1. **TP2 is expected to work; it has not been tested on two-GPU hardware.** The launcher refuses a TP size the visible GPUs cannot serve; expose both cards (`run.sh --gpu 0,1`, or edit the Compose device reservation) and keep a sidecar out of the TP pair. |
| `SGLANG_MM_PREPROCESS_DEVICE` | `cpu` | Media preprocessing: `cpu` (PIL, keeps JPEG work off the model GPU), or `cuda:N` for the model GPU (`N=0`) or a second GPU (`N≥1`, Torchvision backend). CUDA choices spend serving headroom; image dimensions and concurrency set the requirement. Backend pixel differences select a new persistence identity; older cache directories remain. |
| `NCCL_P2P_DISABLE` | unset | Container/Compose escape hatch for P2P/ACS isolation on some two-card hosts; empty leaves NCCL defaults untouched. |

## Speculation and precision escape hatches

| Setting | Default | Notes |
|---|---|---|
| `SGLANG_SM120_ONLINE_MXFP8` | unset (automatic) | Eligible Flash-Next SM120 launches select the accepted rowwise-FP8 path themselves. `false` pins the original checkpoint precision; `true` forces the conversion. A saved private compatibility override, not everyday tuning — see [FP8.md](FP8.md). |
| `FLASHINFER_GDN_FP16_ACCUM_MMA` | `1` on the Next recipes | FP16-accumulate MMA mode of the patched FlashInfer SM12x delta-rule prefill kernels; `0` runs the kernels as released. The resolved mode is part of the NIXL namespace, so the two modes use separate persistent roots and nothing is deleted. The `27b` recipe never sets it (Triton GDN). |
| `--speculative-adaptive-config` | bundled per-profile policy file | Fixed routing policy: `configs/pennyroyal/adaptive-next.json` on `next` and `next-plain` (block 8 at BS1, block 4 at BS2+; SHA-256 `3edaa87d66f535caae0bddb8f457f03fce0f6dc1f4e8091b20495a0dc9da41fb`, pinned by the recipes) and `configs/pennyroyal/adaptive-dflash2.json` on `next-dflash2` (block 6 at BS1, block 4 at BS2+). A DFLASH tier with two candidates is refused, not averaged. No operator environment key. |
| `SPECULATIVE_NUM_DRAFT_TOKENS` | removed | The 3.0-era block-width knob is gone on `next-dflash2`: the routing policy owns the width. `27b` keeps its own fixed block 8 in the recipe. |

## Reasoning and request behaviour

| Setting | Default | Notes |
|---|---|---|
| `PENNY_REASONING_EFFORT` | `medium` | Launcher default for the chat-template reasoning tier; accepts `none|minimal|low|medium|high|xhigh|max`, and an invalid tier stops the launch. A request's own `reasoning_effort` still takes precedence; with the bundled Froggeric v22.5 template, `high`, `xhigh` and `max` select the same xhigh instruction. |
| `SGLANG_FORWARD_UNKNOWN_TOOLS` | `true` | Forwards a tool call whose name is absent from the request's tool definitions to the client's executor so the model can correct it. Set `false` to opt out. Markdown fences stay text; forwarding never executes anything by itself. |
| Froggeric chat template | bundled, pinned | The template hash, preprocessing device and backend are NIXL namespace fields: changing representation selects a fresh cache identity and leaves older namespaces intact on disk. |

## Host and build controls

| Setting | Default | Notes |
|---|---|---|
| `SGLANG_HICACHE_TORCH_PINNED_ALLOC` | `false` | WSL2 workaround: allocate the HiCache host pool as pinned memory to survive WSL2's CUDA host-pointer aliasing. Leave off on native Linux. In a saved config it beats a shell export; in Compose set it in `.env`. |
| `PENNY_BUILD_JOBS` | 4 | Build/JIT job budget (`MAX_JOBS`, CMake, Cargo, FlashInfer ninja, Inductor threads). Affects compilation and first-start warmup only. |
| `NIXL_PREFIX` | unset | Adds `$NIXL_PREFIX/lib64` to `LD_LIBRARY_PATH` for non-system NIXL installs. |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | 4 | Host thread caps for the launched process; set larger to retain a deliberate configuration. |
| `NUMPY_MADVISE_HUGEPAGE` | `0` | Avoids observed huge-page compaction stalls during CPU image-array allocation; `1` restores NumPy's default. Read at import, so changing it needs a restart. System huge-page policy and GPU pools are untouched. |

### Container routing settings

| Setting | Default | Notes |
|---|---|---|
| `PENNYROYAL_IMAGE` / `--image` | `ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.1` | An explicit pin survives every later setup save: update it to the new tag to upgrade (in an edited `run.sh`, keep the `IMAGE=` assignment — the script runs under `set -u`; in saved setup settings or the Compose `.env` you may instead delete the line to follow the launcher default). Startup files belong to their release's image. |
| `PENNYROYAL_PORT` / `--port` | 8001 | Published API port. |
| `PENNYROYAL_USER` / `--user`, `USER_ID`/`GROUP_ID` | `1000:1000` | Numeric identity that must own the writable cache mounts. |
| `PENNYROYAL_STARTUP` / `--startup` | `config/start-flash-next-frspec.sh` | Selected host startup file; the whole directory is mounted read-only at `/config`. |
| `PENNYROYAL_NIXL_CONFIG` / `--nixl-config` | empty | Overrides the TOML the startup file names; omit to keep the one your startup script selects. |

Container saved setup settings pass these recipe keys into the container
environment when you set them: `SGLANG_HICACHE_NIXL_MAX_CACHE_GB`,
`SGLANG_HICACHE_TORCH_PINNED_ALLOC`, `SGLANG_SM120_ONLINE_MXFP8`,
`SGLANG_FORWARD_UNKNOWN_TOOLS`, `PENNY_PLE_NVME_MODEL`, `MAX_TOTAL_TOKENS`,
`PENNY_REASONING_EFFORT`, `NCCL_P2P_DISABLE`,
`FLASHINFER_GDN_FP16_ACCUM_MMA`. The disk-tier-only key is dropped entirely
from a launch generated with `NIXL=off`.

## Interactions worth knowing

- **Automatic KV pool sizing and the memory fraction.** The pool fits inside
  the recipe's `--mem-fraction-static`; the Flash-Next DFlash2 profile holds
  `.95` deliberately for image-workload headroom, and the retained native-MTP
  FR-Spec recipe holds its separately measured `.97`. Do not raise one to pay
  for the other.
- **NIXL off, two places.** `NIXL=off` in the startup file and `--no-nixl` in
  `run.sh` are separate choices that must agree, or the container mounts a
  root nothing uses (or the startup script wants a mount that isn't there).
- **Sidecar GPU and TP.** A preprocessing `cuda:N` index is not a TP card.
  With TP2 and a sidecar, expose three GPUs and keep the sidecar index outside
  the tensor-parallel pair.
- **WSL2 and cache mounts.** Pinned allocation plus writable cache directories
  owned by the running UID/GID; the container path keeps using the numeric
  identity.
- **Native-MTP profiles are maintained.** `next` and `next-plain` launch
  with their existing recipes and their fixed concurrency widths (block 8 at
  one running request, block 4 at two or more, from the pinned policy), remain
  the default for new setups, and keep receiving updates; `next-dflash2` is
  the **experimental / alpha** opt-in alternative, verified through FlashInfer's
  SM120 kernel, and carries the same maturity label as the `27b` DFlash2 profile
  (see [README](README.md#dflash2-maturity)).

## Restarts and cache effects

- **Every setting here is startup-time.** Change it, restart the server; the
  recipes and entrypoint read the environment once, and NumPy/FlashInfer read
  theirs at import.
- **The NIXL namespace is content-addressed.** Model identity, chat-template
  hash, media backend/device, precision representation and GDN accumulate mode
  are namespace fields. Any of them changing selects a fresh persistent root;
  old namespaces are left on disk, nothing is deleted, and the previous cache
  stays ordinary reclaimable storage.
- **A fresh namespace costs you one cold prefill** per prefix, not your data:
  the first request after any cache-identity change recomputes; later
  requests restore normally.
- **Upgrading the image or source never re-downloads weights** and never
  clears your saved configuration files; setup saves always write new
  timestamped outputs and preserve your explicit image pins.
