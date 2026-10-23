# Penny Royal beta configurator

`./configure-penny` helps prepare ordinary launch and configuration files.
It is **optional and open for testing**. The manual
[native instructions](RUN.md) and [container instructions](docker/pennyroyal/README.md)
work without it. Please [report problems](https://github.com/jpezzulli/sglang-rtxpro6000/issues)
if you try it.

The assistant asks about your model, GPU, paths and cache settings, shows your
choices, then saves files and prints what to run. It does not download models,
install packages or start a server.

## Before you start

Have Python 3 and a Penny Royal checkout. For native use, finish the
[installation](BUILD.md) and download your checkpoints. For container use,
prepare Docker/NVIDIA support and model/cache directories as described in the
[container guide](docker/pennyroyal/README.md).

The inference target is RTX PRO 6000 Blackwell at TP1 or TP2. A sidecar GPU
can be selected for image preprocessing when present; CPU preprocessing is
also available.

## Native setup

From the checkout:

```bash
./configure-penny
```

Review the choices and save. Each save creates a **new timestamped configuration**
and prints its check and launch commands. Run those exact commands so the
new file is selected. For example, using the path printed by setup:

```bash
./run-penny --config /path/to/the-new-file.env --check
./run-penny --config /path/to/the-new-file.env
```

The check validates settings and paths without loading a model. Create missing
folders or correct the named setting, then run it again. You can inspect a
saved configuration with `./run-penny --config /path/to/file.env --show-config`.

## Container setup

```bash
./configure-penny --container
```

Saving creates a **new timestamped folder** containing the host `run.sh`,
startup/configuration files, and the saved settings. Use the printed command
to launch that folder. It uses ordinary Docker, not a generated Compose setup.

Host model paths and container model paths differ: if `/srv/models` is mounted
at `/models`, `/srv/models/MyCheckpoint` is `/models/MyCheckpoint` inside the
container. The configurator asks for the appropriate paths.

## Changing an existing configuration

```bash
./configure-penny --config /path/to/saved.env
```

For a container configuration, include `--container`. The selected file supplies
starting values. Saving writes a new timestamped output; it does not overwrite,
rename or back up the old one. Choose the new printed launch command, or rename
and manage the files yourself. Restart the server to apply changed settings.

## What the settings mean

- **Profile:** Flash-Next with native NEXTN MTP and the FR-Spec map
  (`next`, the default for new setups) or without it (`next-plain`),
  Flash-Next with the DFlash2 draft and the FR-Spec map
  (`next-dflash2`, **experimental / alpha**), or 27B with its DFlash2 draft
  (`27b`, also **experimental / alpha** — see
  [DFlash2 maturity](README.md#dflash2-maturity)).
  Every profile keeps its existing recipe, and an explicitly saved
  choice is never rewritten by the default. The two DFlash2 profiles need both
  model folders. FR-Spec and
  fixed concurrency width routing work together on `next-dflash2`; they are not
  competing profile choices. See
  [RUN.md](RUN.md#launch-flash-next-with-dflash2) for what the routed profile
  ships and pins.
- **GPUs:** choose the inference device(s), TP size and optional image-processing
  device. TP2 needs two RTX PRO 6000 inference GPUs; a sidecar remains separate.
- **HiCache:** RAM prefix-cache size in decimal GB. Profile defaults are 32 GB
  for Flash-Next and 96 GB for 27B.
- **KV pool:** automatic on every profile — leave `MAX_TOTAL_TOKENS` blank and
  the engine sizes the shared pool from the memory it profiles. A deliberate
  cap is preserved and reaches the server on all four profiles; it must be
  positive and aligned to the recipe's page size (64).
- **NIXL:** optional disk-cache tier, on by default. Disabling it preserves GPU
  and RAM caching.
- **NIXL disk budget:** GiB. Zero means unlimited disk budget, **not disabled**.
- **MTP width policy (`PENNY_MTP_POLICY`, the native-MTP profiles):** how wide
  the draft blocks are by concurrency — `tiered` (the qualified default: block 8
  at one running request, block 4 at two or more), `adaptive` (the v3.0.2 policy,
  which may move between the two at one request) or `fixed-w4` (block 4 at every
  concurrency). It is asked in the advanced section, it is saved like any other
  setting, and one value picks both the bundled policy file and the launch width
  that carries it. A container save writes your answer into the startup script's
  settings block as `PENNY_MTP_POLICY=<choice>`, so the generated launch carries
  its own policy with it. Nothing exported and nothing saved leaves the recipe's
  qualified default in charge. The DFlash2 profiles have no such question: they
  route their own widths and read nothing from this setting, so a value saved for
  them changes nothing.
- **PLE:** Flash-Next table placement in RAM or a prepared NVMe snapshot,
  independent of whether NIXL is on.
- **Other settings:** model paths, ports, context/request capacity, reasoning,
  and the existing WSL2 host-memory workaround.

For the full explanations, use [CONFIGURATION.md](CONFIGURATION.md),
[RUN.md](RUN.md), [FP8.md](FP8.md) and [NVME-PLE.md](NVME-PLE.md). Accepted
kernel selection is automatic; users do not need a menu of internal
optimization flags.

<a id="update-the-setup-files"></a>

## Updating the setup files

Use the setup files from the same release as your runtime or image. The old
v2.5.3 `setup1` tag remains an update for v2.5.3; it is not the 3.1 setup.
Keep the configuration you currently use, generate a new one with the new
checkout, and use its printed command when you are ready to switch.

A saved container setting that names an image explicitly (`PENNYROYAL_IMAGE=`)
survives every later save: setup keeps your choice rather than moving you to a
new release. Before regenerating with a new checkout, update that line to the new
image or delete it to take the new default, because generated startup files
belong to the image they were generated with and the new ones will not run on an
older image. Staying on an older image is fine: keep using that release's setup
files for it. Either way a save writes a fresh folder, and the files you already
have stay as they were.

<!-- Compatibility anchors for earlier versions of this guide. -->
<a id="configure-pennyroyal-with-configure-penny"></a>
<a id="container-configure-check-then-run-what-setup-printed"></a>
<a id="native-configure-check-then-launch"></a>
<a id="saved-settings-changes-and-restarts"></a>
<a id="separate-saved-configurations"></a>
<a id="two-cache-sizes-two-different-units"></a>
<a id="what-it-asks-you"></a>
