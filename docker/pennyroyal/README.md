# Run Penny Royal from the prebuilt container

Download the image, put your startup and configuration files on the host,
and run it. The container supplies the runtime; you control the model,
GPUs, mounts, and settings from ordinary files. No image build or Compose
setup is required.

The inference target is **RTX PRO 6000 Blackwell, TP1 or TP2**. An Ampere or Ada
Lovelace sidecar can handle image preprocessing, or you can use the CPU.
For a native install, use [BUILD.md](../../BUILD.md).

## What goes where

```text
Your host
  run.sh                  image, GPUs, port, user and host mount paths
  config/
    start-flash-next-frspec.sh  model and SGLang startup settings
    start-27b-dflash2.sh   alternative 27B + draft startup settings
    nixl-posix-frspec.toml      disk-cache settings
  model files             mounted read-only at /models
  compiler/runtime cache  mounted writable at /cache
  optional NIXL cache     mounted writable at /nixl

Container
  Penny Royal runtime, dependencies and bundled templates
  /config                 your selected startup directory, mounted read-only
```

The startup directory is mounted as a directory, so normal editor saves work.
Edit the host files and restart the container to apply changes. No configuration
is copied out of the image and no image rebuild is needed.

## Before you start

- Linux, Docker Engine, a compatible NVIDIA driver and NVIDIA Container Toolkit.
- Downloaded [target model files](../../BUILD.md#reference-and-measured-checkpoints).
  The 27B profile also needs its DFlash2 draft.
- Writable cache directories owned by the UID/GID you will run as.
- Enough host RAM: Flash-Next defaults to 32 GB HiCache plus about 48 GiB RAM
  PLE; 27B defaults to 96 GB HiCache. Leave room for loading and the OS.
- If using NIXL, a local filesystem supporting its O_DIRECT/io_uring path.

<a id="configure-and-start"></a>

## Get the launch files

Use matching source and image tags from the release notes. Download only the
small launch-file set; a full source checkout is optional.

```bash
RELEASE_REF='pennyroyal-v3.0.0'
BASE="https://raw.githubusercontent.com/jpezzulli/sglang-rtxpro6000/$RELEASE_REF/docker/pennyroyal/launch"
mkdir -p pennyroyal/config
cd pennyroyal
curl -fL "$BASE/run.sh" -o run.sh
for file in start-flash-next.sh start-flash-next-frspec.sh start-27b-dflash2.sh nixl-posix.toml nixl-posix-frspec.toml; do
  curl -fL "$BASE/config/$file" -o "config/$file"
done
chmod +x run.sh
```

The source files are also available here: [run.sh](launch/run.sh),
[recommended Flash-Next FR-Spec startup](launch/config/start-flash-next-frspec.sh),
[optional non-FR startup](launch/config/start-flash-next.sh),
[27B startup](launch/config/start-27b-dflash2.sh), and
[FR-Spec NIXL settings](launch/config/nixl-posix-frspec.toml).

## Edit your configuration

In `run.sh`, set the image tag, host model/cache folders, GPU selection, API
port and numeric UID/GID. Set `STARTUP` to the file for your chosen model.
Create the host directories and make them writable by that identity.

In the selected startup file, set `TARGET_MODEL` to its path **inside** the
container. For example, if `/srv/models` is mounted at `/models`, the host's
`/srv/models/MyCheckpoint` becomes `/models/MyCheckpoint`. The 27B startup
also needs `DRAFT_MODEL`. RAM cache size, tensor-parallel size, request capacity,
PLE placement and image preprocessing are in this startup file.

<a id="profiles-and-checks"></a>

| Profile | Startup file |
|---|---|
| Flash-Next, native NEXTN MTP with FR-Spec (recommended) | `config/start-flash-next-frspec.sh` |
| Flash-Next without FR-Spec (alternative) | `config/start-flash-next.sh` |
| 27B FP8 with DFlash2 | `config/start-27b-dflash2.sh` |

For TP2, expose both RTX PRO 6000 GPUs in `run.sh` and set `TP_SIZE=2` in the
startup file. If using a sidecar for image processing, expose that GPU too
and select its **container-visible** `cuda:N` index. The sidecar is not part
of the tensor-parallel inference pair.

FR-Spec remains on when adaptive MTP is enabled. It narrows the draft
vocabulary; adaptation changes draft length at C1. Concurrent batches keep
FR-Spec with fixed four-token drafts.

## Start and verify

Run from your launch-file folder, using the image tag from the release:

```bash
IMAGE='ghcr.io/jpezzulli/sglang-rtxpro6000:v3.0.0'
./run.sh --image "$IMAGE" --startup config/start-flash-next-frspec.sh
```

The script runs `docker run` in the foreground and shows the server logs.
Docker pulls the image if needed. Wait for API readiness, then check it from
a second terminal:

```bash
curl -fsS http://localhost:8001/health
curl -fsS http://localhost:8001/v1/models
```

Use the port configured in `run.sh` if you changed it. The OpenAI-compatible
model name is `pennyroyal`. See the [API smoke check](../../RUN.md#smoke-through-the-normal-api)
for a generation request.

Stop with Ctrl-C. Edit your host settings and run the same command to start
again. Your model files and persistent caches remain in their host folders.
To upgrade, select the new image tag and review the new release's sample
startup files before restarting.

## Optional disk caching

NIXL is **on by default**. To turn it off:

1. Set `NIXL=off` in your startup file.
2. Add `--no-nixl` to `run.sh` so it skips the disk-cache mount.

```bash
./run.sh --image "$IMAGE" --startup config/start-flash-next-frspec.sh --no-nixl
```

GPU radix caching and RAM HiCache stay active. Existing disk-cache data is
not deleted. NVMe PLE is separate: if using it with NIXL off, also pass
`--nvme-ple` so its io_uring access is retained. See [NVMe PLE](../../NVME-PLE.md).

The supplied launcher uses `seccomp=unconfined` for the io_uring paths used by
NIXL or NVMe PLE. It does not run privileged. Paths for other optional features,
such as an external PLE snapshot, must be mounted in `run.sh` as well.

## Optional beta configurator

With a source checkout and Python 3, [the beta configurator](../../CONFIGURE.md)
can generate the host launch files for you. It writes a new timestamped folder
and prints the command to start it. It is open for testing; the manual files
above are the normal setup path.

<a id="manual-setup"></a>
<a id="guided-setup"></a>
<a id="get-the-compose-files"></a>

Existing Compose deployments can continue to use their own configuration.
For 3.0, the documented setup is the ordinary `run.sh` and host startup files
above.

<!-- Compatibility anchors for earlier versions of this guide. -->
<a id="check-the-image-and-run-commands"></a>
<a id="image-checks-and-command-boundary"></a>
<a id="optional-settings"></a>
<a id="other-settings"></a>
<a id="pennyroyal-container"></a>
<a id="prerequisites"></a>
<a id="release-builds"></a>
<a id="selinux-hosts"></a>
<a id="wsl2"></a>
