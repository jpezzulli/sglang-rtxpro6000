# Configure Pennyroyal with ./configure-penny

`./configure-penny` is an optional terminal assistant, marked **beta**. It
asks you a short series of questions about your model, GPU, and cache
folders, explains each setting as it goes, and saves your answers in one
plain text file that the launchers read. It writes that file and tells you
what to run next; it does not install packages, download models, or start
anything.

## Before you start

- Native: finish the install in [BUILD.md](BUILD.md) and download your model
  first, so the Python environment with `bin/sglang` already exists.
- Container: work through the prerequisites in the
  [Docker guide](docker/pennyroyal/README.md) and have your model files on
  the host.
- Python 3 must be available as `python3`, or set `PENNY_PYTHON` to an
  interpreter. Nothing else needs to be installed for the wizard itself.
- A GPU is not required to configure or check a saved file.

## Native: configure, check, then launch

From your Pennyroyal folder:

```bash
./configure-penny
./run-penny --check
```

`--check` reads the file you just wrote and checks the saved paths and
settings for common problems, without touching a GPU or loading a model: a
model folder that does not exist, a cache folder you cannot write into, a
runtime that is missing.
Fix what it names — create the folder, or rerun `./configure-penny` and give
a different path — and check again. Once the check stops reporting errors,
start the server:

```bash
./run-penny
```

That runs the launch recipe for the profile you picked with the settings you
saved. [RUN.md](RUN.md) covers what the server prints while starting and how
to confirm it is ready.

## Container: configure, check, then run what setup printed

```bash
./configure-penny --container
./configure-penny --container --check
```

Answer the questions, and once the check is clean, run the one long command
setup printed for you. It lists your resolved settings and ends with
`docker compose -f .../compose.yaml --env-file .../.env up -d`, so it works
from any directory. Copy it exactly instead of typing a plain
`docker compose up -d` — your file and env-file names live in that line.

Inside the container your models appear under `/models`, which is the host
folder you gave as `HOST_MODELS_ROOT`. With `HOST_MODELS_ROOT=/srv/models`,
a checkpoint at `/srv/models/RadixArk-Qwen3.8-Flash-Next-NVFP4` is entered as
`/models/RadixArk-Qwen3.8-Flash-Next-NVFP4`.

## What it asks you

- **Profile**, as a numbered menu: `next` (Flash-Next with FR-Spec),
  `next-plain` (the same target without FR-Spec), or `27b` (Qwen3.8-27B with
  the DFlash2 draft). Enter keeps the highlighted choice, and 27B also asks
  for its draft folder.
- **Paths**: your Pennyroyal folder and Python environment folder (native),
  or the host folders for models, caches, and persistent NIXL storage
  (container). The wizard suggests a value where it can work one out, so you
  are usually confirming one path rather than inventing it.
- **Model folder**: the checkpoint you downloaded for the chosen profile.
- **GPU**: a numbered list of the cards `nvidia-smi` reports, or a typed
  index or UUID when no list is available.
- **API port** and the two cache sizes below.
- **Advanced settings**: offered at the end, and skipped by saying no. It
  covers capacity overrides, online FP8, and where the PLE table lives.
  [RUN.md](RUN.md), [FP8.md](FP8.md), and [NVME-PLE.md](NVME-PLE.md) explain
  those; skipping them keeps the qualified defaults.

You then see every value in one review, with a `*` beside the ones you
changed, and a final confirmation before anything is written. Typing `q` at
any point abandons the session and leaves your file untouched.

## Two cache sizes, two different units

- **RAM (HiCache) size** — decimal **GB** (1 GB = 1e9 bytes, not GiB). Leave
  it blank to keep the profile default: 32 GB for the Next profiles, 96 GB
  for 27B. [Choose HiCache RAM size](RUN.md#choose-hicache-ram-size) has the
  trade-offs.
- **NIXL disk budget** — **GiB**, for the persistent NIXL cache folder on
  disk. `0`, the default, means **no cap**: the cache stays enabled and keeps
  growing, it does not switch anything off. It is a cleanup target rather
  than a hard quota; see
  [Limit NIXL disk use](RUN.md#limit-nixl-disk-use).

Neither value limits the other, and neither disables HiCache or NIXL.

## Saved settings, changes, and restarts

| Use | Saved in |
|---|---|
| Native | `~/.config/pennyroyal/pennyroyal.env` |
| Container | `docker/pennyroyal/.env` |

Rerun `./configure-penny` whenever you want to change something — it starts
from your saved answers — or edit those files in any text editor. For
container settings rerun `./configure-penny --container`; the bare command
selects the native file. The container file is the same `.env` that Compose
reads.

A running server keeps the settings it booted with, so restart to apply a
change: stop and rerun `./run-penny`, or for the container rerun the command
setup printed, with `--force-recreate` added to the end.

## Separate saved configurations

The default file holds one profile. To keep more than one, save each to its
own file and pass that path to both sides:

```bash
./configure-penny --config /absolute/path/to/27b.env
./run-penny --config /absolute/path/to/27b.env
```

For the container, pass the same path to setup and its check, then use the
launch command that setup prints, because `run-penny` is native-only:

```bash
./configure-penny --container --config /absolute/path/to/27b.env
./configure-penny --container --check --config /absolute/path/to/27b.env
```
