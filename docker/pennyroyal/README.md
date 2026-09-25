# Pennyroyal container

This Compose service runs the same Pennyroyal v2.5.2 source and launch recipes
as the native installation. The default is Flash-Next with FR-Spec. Native
installation remains supported and is documented in [`BUILD.md`](../../BUILD.md)
and [`RUN.md`](../../RUN.md).

Publishing v2.5.2 builds and uploads
`ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.2` through GitHub Actions. Check the
[Pennyroyal container workflow](https://github.com/jpezzulli/sglang-rtxpro6000/actions/workflows/pennyroyal-container.yml)
for availability. Python, the CUDA toolchain, NIXL POSIX, and prebuilt
FlashInfer kernels are included; the host supplies the NVIDIA driver. Native
installations remain independent of the container image.

## Prerequisites

- Linux x86-64, ordinary rootful Docker Engine with Compose v2, an NVIDIA driver
  compatible with the image's CUDA 13.3 toolkit, and the NVIDIA
  Container Toolkit configured for Docker.
- An RTX PRO 6000 Blackwell and the model files described in
  [`BUILD.md`](../../BUILD.md#reference-and-measured-checkpoints).
- Writable host directories for compiler/runtime caches and NIXL persistence.
  The runtime UID and GID must own them.
- A local filesystem suitable for NIXL POSIX O_DIRECT/io_uring storage.

Container and native profiles use the same host-memory settings:

| Profile | Configured host memory |
|---|---|
| Flash-Next | 32 GB HiCache by default, plus roughly 48 GiB for RAM-backed PLE |
| 27B/DFlash2 | 96 GB HiCache by default, plus runtime and draft allocations |

Leave additional room for loading, the runtime, and the operating system. See
[host memory and first start](../../RUN.md#host-memory-and-first-start). NVMe
PLE can move Flash-Next's fixed PLE table residency to local SSD.

The UID/GID examples below assume Docker without `userns-remap`. Rootless
Docker and remapped daemons use different host/container UID mappings. Set
bind-directory ownership for the daemon's mapping; host-wide user-namespace
settings can remain in place.

The service runs without privileged mode. It uses `seccomp=unconfined` because
NIXL POSIX needs io_uring, which Docker's default seccomp profile commonly
blocks. A custom profile that permits the required io_uring syscalls can
replace this setting.

<a id="configure-and-start"></a>

## Get the Compose files

Get the matching launch and Compose files from the release tag:

```bash
git clone --depth 1 --branch pennyroyal-v2.5.2 \
  https://github.com/jpezzulli/sglang-rtxpro6000.git pennyroyal
cd pennyroyal
```

This checkout supplies configuration and documentation; Docker pulls the
prebuilt image; no local SGLang build is involved.

## Guided setup

The optional setup assistant is **beta**; manual Compose setup is available below.

If Python 3 is available on the host, run:

```bash
./configure-penny --container
./configure-penny --container --check
```

Choose Next or 27B, enter your model and cache paths, and review the settings
before saving. The utility writes `docker/pennyroyal/.env`; it does not install
or start anything. Model paths inside the container begin with `/models`.
For example, `/srv/models/MyModel` on the host becomes `/models/MyModel` when
`HOST_MODELS_ROOT=/srv/models`.

Create the cache directories shown by the check, with the configured UID/GID,
then run the launch command printed by setup. That command includes the
selected configuration and can be run from any directory. Rerun setup to
change settings; stop and recreate the container to apply them.

Next needs one target checkpoint. The 27B profile also needs its DFlash2 draft.
Choose the HiCache RAM size and optional NIXL disk budget during setup, or set
`PENNY_HICACHE_SIZE_GB` and `SGLANG_HICACHE_NIXL_MAX_CACHE_GB` in `.env`.
HiCache accepts whole GB starting at 1; the disk budget uses GiB. A disk budget
of 0 means unlimited, not disabled. Both cache tiers remain enabled. See
[RAM sizing](../../RUN.md#choose-hicache-ram-size) and
[what the disk budget covers](../../RUN.md#limit-nixl-disk-use).

Prefer to edit the configuration yourself? Use the manual path below. No host
Python is needed for manual Compose setup.

<a id="profiles-and-checks"></a>

## Manual setup

```bash
cd docker/pennyroyal
cp .env.example .env
```

Set `PENNYROYAL_PROFILE` in `.env` to one of:

| Value | Recipe |
|---|---|
| `next` | Flash-Next with FR-Spec (default) |
| `next-plain` | Flash-Next without FR-Spec |
| `27b` | Qwen3.8-27B target with the DFlash2 draft |

For 27B, change **all three** settings in `.env`, using your actual downloaded
directory names below `/models`:

```dotenv
PENNYROYAL_PROFILE=27b
TARGET_MODEL=/models/Qwen3.8-27B-FP8
DRAFT_MODEL=/models/Qwen3.8-27B-DFlash2
```

Changing the profile alone leaves the example's Flash-Next target selected;
set all three values together for 27B. The two Next profiles use only
`TARGET_MODEL`.

Set the host paths and runtime identity in the same file:

| Variable | Purpose |
|---|---|
| `HOST_MODELS_ROOT` | Parent directory containing every referenced checkpoint path |
| `HOST_CACHE_BASE` | Writable compiler and runtime caches |
| `HOST_NIXL_STORAGE_BASE` | Writable persistent NIXL storage |
| `USER_ID` / `GROUP_ID` | Numeric owner of the writable cache directories |
| `NVIDIA_GPU` | Host GPU index or UUID used for the model |
| `PENNYROYAL_PORT` | Host API port; defaults to `8001` |

Model paths are container paths below `/models`. If checkpoint files contain
symlinks to sibling directories, mount their common parent as
`HOST_MODELS_ROOT`; links outside the bind mount will be broken.

Create the writable directories with the configured numeric identity. For the
example values:

```bash
sudo install -d -o 1000 -g 1000 \
  /var/cache/pennyroyal /srv/pennyroyal-nixl
```

## Start and verify

For manual setup, run these from `docker/pennyroyal`. Normal Compose rules
apply: exported shell variables take precedence over `.env`; unset conflicting
exports if you want to use the saved values.

```bash
docker compose config --quiet
docker compose pull
docker compose up -d
docker compose logs -f pennyroyal
```

Startup can take many minutes while weights load, extensions compile, graphs
capture, and caches initialize. The image health check allows a 20-minute
start period. Configuration errors stop the container without entering a
restart loop.

The API is published at `http://localhost:8001/v1` by default. After the log
reports readiness, check it with:

```bash
curl -fsS http://127.0.0.1:8001/health
```

Use `pennyroyal` as the model name in your client. A
[sample chat request](../../RUN.md#smoke-through-the-normal-api) is available
if you want to try the API directly.

Inspect or stop the service with:

```bash
docker compose ps
docker compose logs -f pennyroyal
docker compose down
```

`down` allows up to two minutes for shutdown, then removes the container and
network. The three bind-mounted host directories remain intact.

## Image checks and command boundary

The entrypoint exposes two non-serving checks. The CPU-only import check skips
device work:

```bash
docker run --rm ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.2 --help
docker run --rm ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.2 --check
```

Arbitrary commands require the explicit `exec` boundary:

```bash
docker compose run --rm pennyroyal exec .venv/bin/python --version
```

The v2.5.2 image build runs the automated CPU installation check shown above.
Both profiles were regression-tested natively on the release source. The fresh
container GPU qualification remains the v2.5.0 result: both profiles passed API
schema/tool checks, 64K prefill, 1,024-token C1/C4 decode, JPEG and static-video
checks, and NIXL reuse after container restart. Each restored 63,872 of 63,906
prompt tokens from storage and returned exact `READY`. That GPU serving test
used rootless Podman on one RTX PRO 6000; Docker Compose was checked separately.
The Next check used the RadixArk reference target; 27B used the measured FP8
checkpoint in [`BUILD.md`](../../BUILD.md#reference-and-measured-checkpoints).

## Optional settings

Set these values in `.env` before the first `docker compose up`. To apply a
change to a running deployment, update `.env`, then recreate the container:

```bash
docker compose up -d --force-recreate
```

Online FP8 is off by default. Read [`FP8.md`](../../FP8.md), then set
`SGLANG_SM120_ONLINE_MXFP8=true` to opt in. RAM-backed PLE is the default.

`PENNY_REASONING_EFFORT` is a launcher-level convenience (PR#18): unset
(default) keeps the recipes' qualified `medium` default chat-template
kwargs, and `none|minimal|low|medium|high|xhigh|max` rewrites just that
key before launch. It is launcher-only -- the server does not read it --
and an explicit per-request `reasoning_effort` always wins over the
default. An invalid value stops the container at launch.

`TP_SIZE=2` asks the Next recipes for two tensor-parallel ranks (the
qualified default is `TP_SIZE=1`), but `TP_SIZE` does not grant GPU access:
this compose.yaml reserves exactly one GPU under
`deploy.resources.reservations.devices`, and Compose users who want TP2 must
also edit that existing reservation to name two explicit ids — the complete
item is:

```yaml
            - driver: nvidia
              device_ids: ["0", "1"]
              capabilities: [gpu]
```

Otherwise the recipe fails at launch with the visible-device count it found
and this fragment, rather than hanging in NCCL or silently running TP1
(`SGLANG_MM_PREPROCESS_DEVICE=cuda:N` outside the model range counts as an
extra needed device). TP2 is experimental and not yet hardware-qualified
on this image: expect a separate NIXL namespace per TP size, a replicated
FR-Spec draft head, and one scheduler per GPU. Peer access is not verified
at startup; if NCCL hangs during transport init on a consumer-PCIe host,
`NCCL_P2P_DISABLE=1` can help isolate a P2P/ACS/IOMMU problem at a possible
throughput cost the startup log repeats; the image never sets it for you.

For the optional six-request Flash-Next profile, keep preprocessing on the CPU
and set these values in `.env`:

```dotenv
SGLANG_SM120_ONLINE_MXFP8=true
SGLANG_MM_PREPROCESS_DEVICE=cpu
MAX_RUNNING_REQUESTS=6
MAX_MAMBA_CACHE_SIZE=36
MAX_TOTAL_TOKENS=1048576
```

Then recreate the container:

```bash
docker compose up -d --force-recreate
```

The normal FR-Spec defaults remain four requests, 24 Mamba slots, and 824,384 KV
tokens. The native C6 check used online FP8, RAM PLE and A4000 preprocessing;
it retained 524,288 tokens per request and profiled 1,034,176 shared KV tokens
from the 1,048,576 request. This is one shared pool,
not six independent 524K contexts; SGLang may clamp it to the capacity available
on another system. CPU preprocessing requires no second GPU. The optional
secondary-GPU configuration below is another way to keep preprocessing off the
model GPU; neither path requires a dual-socket system or NUMA configuration.

For NVMe-backed PLE, read [`NVME-PLE.md`](../../NVME-PLE.md). The image already
contains the isolated reader, but a prepared overlay is still required. The
normal `/models` mount is read-only. For the one-time preparation, override it
as writable and create a new output directory; the helper preserves existing
destinations:

```bash
docker compose run --rm --no-deps \
  -v /srv/models:/models \
  pennyroyal exec .venv/bin/python scripts/pennyroyal/prepare_ple_nvme.py \
  --source /models/RadixArk-Qwen3.8-Flash-Next-NVFP4 \
  --output /models/flash-next-ple
```

Use the actual `HOST_MODELS_ROOT` in place of `/srv/models`. Keep the source
checkpoint immutable. After preparation, restore the normal read-only mount,
set `PENNY_PLE_BACKEND=nvme` and `PENNY_PLE_NVME_MODEL=/models/flash-next-ple`,
then start either Next profile. NVMe PLE is not supported by the 27B profile.

To use a second GPU only for media preprocessing, create `compose.override.yaml`:

```yaml
services:
  pennyroyal:
    environment:
      SGLANG_MM_PREPROCESS_DEVICE: cuda:1
    deploy:
      resources:
        reservations:
          devices: !override
            - driver: nvidia
              device_ids: ["0", "1"]
              capabilities: [gpu]
```

Inside the container, `cuda:0` remains the model GPU and `cuda:1` is the
secondary preprocessing GPU. `!override` requires Docker Compose 2.24.4 or
newer and replaces the default device reservation. The model stays on
`cuda:0`; only preprocessing uses `cuda:1`. Use GPU UUIDs in `device_ids` when
stable device selection matters.

### SELinux hosts

If your container engine enables SELinux confinement, UID/GID ownership alone
may not make the bind mounts readable. Add this per-container override to
`compose.override.yaml` when the model directory is shared with native
services:

```yaml
services:
  pennyroyal:
    security_opt:
      - label=disable
```

This disables SELinux separation for this container only. Host SELinux remains
enabled, the container remains unprivileged, and the model mount remains
read-only. Omit the override when the engine does not enforce SELinux labels.

## Release builds

Publishing a GitHub release builds its exact tagged source and uploads the
matching versioned image automatically. For example, `pennyroyal-v2.5.2`
produces `ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.2`. Draft releases and ordinary
branch pushes do not publish a release image. No moving `latest` tag is used.

The build checks package versions, the source revision and import location,
launcher syntax, the entrypoint and the NIXL POSIX plugin without a GPU. A
failed check fails the workflow and leaves the version tag unchanged. GPU
regression remains separate: routine source-only maintenance uses the tested
native runtime evidence, while changes to the container's dependency stack or
device handling warrant another GPU container check.

For a failed build, rerun the **Pennyroyal container** workflow in Actions.
Alternatively, run it manually against the release's Git tag with both inputs
empty. A manual run on a branch publishes only a `build-<commit>` image. The
optional `promote_digest` and `image_tag` inputs still allow an already checked
image to receive a version tag without another build. Release creation must use
the GitHub UI or a normal user/App credential; GitHub's automatic `GITHUB_TOKEN`
does not trigger another workflow when it creates a release.
