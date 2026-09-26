# BF16 NVMe PLE staging hotfix (v2.5.2)

A default Flash-Next prefill asks the NVMe PLE reader for one row per token and
ngram head: 4096 tokens x 16 heads = 65,536 rows of 160 columns. FP8 rows need
10 MiB and fit the 16 MiB staging budget; BF16 rows need 20 MiB, so the reader
raised `PLE lookup needs 65536 staging rows but the configured capacity is
52428` before any I/O. The fix scales the staging byte budget with the storage
element size, which keeps the FP8 row capacity (and the FP8 allocation)
untouched while BF16 tables get the same row capacity at double the bytes. The
two bounded staging slots and the fail-loud oversized lookup are unchanged;
RAM-PLE (`PENNY_PLE_BACKEND=ram`) does not load this file at all.

The hotfix touches only the installed NVMe reader file
`/opt/pennyroyal/.ple-nvme/sglang_ssd_stream/backend.py`. It does not change
the image or its tag, the model, the prepared PLE overlay, or any Compose
setting you already have.

## Which patch to use

Apply [`patches/bf16-ple-staging-hotfix-backend.patch`](patches/bf16-ple-staging-hotfix-backend.patch),
the runtime-only patch (`tools/ple_nvme/patches/` in the repository): its single
diff header is
`tools/ple_nvme/ssd_stream/src/sglang_ssd_stream/backend.py`, which is exactly
where step 1 below puts the extracted file, so nothing else lands in your work
directory. The capacity test
([`ssd_stream/tests/test_staging_capacity.py`](ssd_stream/tests/test_staging_capacity.py))
and this guide belong to the same repository change and are not part of the
runtime patch; applying them to the extracted copy would only add files that the
container never reads.

## Why that path

- `docker/pennyroyal/Dockerfile` builds the reader with
  `PYTHON=/opt/pennyroyal/.venv/bin/python bash tools/ple_nvme/install.sh`.
- `tools/ple_nvme/install.sh` installs with
  `uv pip install --target "$PENNY_PLE_PLUGIN_DIR" --no-deps .../ssd_stream`
  and defaults `PENNY_PLE_PLUGIN_DIR` to `$REPO_ROOT/.ple-nvme`.
- The image sets `REPO_ROOT=/opt/pennyroyal` and
  `docker/pennyroyal/entrypoint.sh` exports
  `PENNY_PLE_PLUGIN_DIR="$REPO_ROOT/.ple-nvme"`; the `nvme` branch of
  `configs/pennyroyal/ple-backend.sh` requires
  `$PENNY_PLE_PLUGIN_DIR/sglang_ssd_stream/plugin.py` and prepends that
  directory to `PYTHONPATH`.

So the runtime file is `/opt/pennyroyal/.ple-nvme/sglang_ssd_stream/backend.py`;
the checkout copy under `tools/ple_nvme/ssd_stream/src/` is not what the
container imports. `install.sh` copies the package files unchanged, so the
installed file is expected to be byte-identical to the tagged source file:
SHA-256 `7cd44a6fa813db99e30ed9bfd6704f24e28b36a4bc53d6dc776c4247baae8fe4`
(verified equal for `pennyroyal-v2.5.2` and the hotfix base revision
`3c2436a0879a7d368eaa26b1b38d2c1d8f57df63`). After this hotfix it must read
`6f25e0a01c90c73e42eec667a6fcea10cc94bee53967100542eca4f9f2bc6a81`. If the
extracted file has neither value, stop and report it instead of patching.

## Keep your own Compose file stack

`docker/pennyroyal` normally holds `compose.yaml`, `.env`, and, on setups that
need a second GPU or `security_opt: ["label=disable"]`, a
`compose.override.yaml`. Compose picks `compose.yaml` + `compose.override.yaml`
up automatically, but as soon as a command passes any `-f` the automatic
discovery stops: `-f compose.yaml -f compose.bf16-ple.override.yaml` silently
drops `compose.override.yaml`, and `-f compose.bf16-ple.override.yaml` alone
drops `compose.yaml` as well. The commands below therefore name your whole file
stack and your `.env` explicitly, in apply and in rollback. Substitute your real
files (add or remove `-f` entries, keep the order you use normally) and never
edit `compose.yaml`, `compose.override.yaml`, or `.env` for this hotfix.

## Apply

Steps 1-2 run in a private working directory; steps 3-4 run in
`docker/pennyroyal`, the directory that already holds your `compose.yaml`, your
`compose.override.yaml` if you have one, and `.env`. The image is never written
to.

```bash
mkdir -p ~/ple-bf16-hotfix/work && cd ~/ple-bf16-hotfix
# Put bf16-ple-staging-hotfix-backend.patch, the runtime-only patch under
# tools/ple_nvme/patches/, in this directory.

# 1. Extract the file the v2.5.2 image actually runs, at the patch's path.
docker create --name ple-hotfix \
  ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.2 --help >/dev/null
mkdir -p work/tools/ple_nvme/ssd_stream/src/sglang_ssd_stream
docker cp ple-hotfix:/opt/pennyroyal/.ple-nvme/sglang_ssd_stream/backend.py \
  work/tools/ple_nvme/ssd_stream/src/sglang_ssd_stream/backend.py
docker rm ple-hotfix
printf '%s  %s\n' \
  7cd44a6fa813db99e30ed9bfd6704f24e28b36a4bc53d6dc776c4247baae8fe4 \
  work/tools/ple_nvme/ssd_stream/src/sglang_ssd_stream/backend.py \
  | sha256sum --check || exit 1

# 2. Patch the extracted copy (dry run first). The runtime-only patch changes
#    that one file and creates nothing else under work/.
(cd work && patch -p1 --dry-run --forward < ../bf16-ple-staging-hotfix-backend.patch)
(cd work && patch -p1 --forward < ../bf16-ple-staging-hotfix-backend.patch)
printf '%s  %s\n' \
  6f25e0a01c90c73e42eec667a6fcea10cc94bee53967100542eca4f9f2bc6a81 \
  work/tools/ple_nvme/ssd_stream/src/sglang_ssd_stream/backend.py \
  | sha256sum --check || exit 1
```

```bash
# 3. Read-only bind of the patched file, as a separate override file.
cd path/to/docker/pennyroyal   # your existing compose project directory
mkdir -p ple-bf16-hotfix
cp ~/ple-bf16-hotfix/work/tools/ple_nvme/ssd_stream/src/sglang_ssd_stream/backend.py \
   ple-bf16-hotfix/backend.py
cat > compose.bf16-ple.override.yaml <<'YAML'
services:
  pennyroyal:
    volumes:
      - type: bind
        source: ${PLE_BF16_HOTFIX_DIR:-./ple-bf16-hotfix}/backend.py
        target: /opt/pennyroyal/.ple-nvme/sglang_ssd_stream/backend.py
        read_only: true
        bind:
          create_host_path: false
YAML
# Compose's own automatic override discovery, if it applies to your project,
# does not see this file: name it after your existing -f entries.
COMPOSE="docker compose --env-file .env -f compose.yaml"
[ -f compose.override.yaml ] && COMPOSE="$COMPOSE -f compose.override.yaml"
COMPOSE="$COMPOSE -f compose.bf16-ple.override.yaml"
```

Set `PLE_BF16_HOTFIX_DIR` to the absolute directory holding
`backend.py` if you keep it elsewhere. The bind source must be the file, not a
directory, and the containing directory must be readable by UID/GID 1000.

## Recreate and confirm

`$COMPOSE` is the whole file stack from step 3; the same variable is reused in
rollback without the hotfix override.

```bash
$COMPOSE config --quiet
$COMPOSE config | grep -A4 "ple-bf16-hotfix"          # plus your own mounts
$COMPOSE config | grep -A4 "security_opt\|device_ids"  # unchanged from before
$COMPOSE up -d --force-recreate
$COMPOSE exec pennyroyal sha256sum \
  /opt/pennyroyal/.ple-nvme/sglang_ssd_stream/backend.py
# expect 6f25e0a01c90c73e42eec667a6fcea10cc94bee53967100542eca4f9f2bc6a81
$COMPOSE logs -f pennyroyal
curl -fsS http://127.0.0.1:8001/health
```

Startup still takes minutes (weights, graph capture, caches); the image health
check allows a 20-minute start period. The reader is only imported when
`PENNY_PLE_BACKEND=nvme`, so a RAM-mode container ignores the mount, and
`PYTHONDONTWRITEBYTECODE=1` in the image means no `__pycache__` write is
attempted beside the read-only file. If SELinux confinement blocks the bind, use
the documented `security_opt: ["label=disable"]` container override in your own
`compose.override.yaml`, which these commands already carry.

## Rollback

```bash
ROLLBACK="docker compose --env-file .env -f compose.yaml"
[ -f compose.override.yaml ] && ROLLBACK="$ROLLBACK -f compose.override.yaml"
$ROLLBACK up -d --force-recreate   # hotfix override dropped, your stack kept
rm compose.bf16-ple.override.yaml
rm -rf ple-bf16-hotfix
```

Nothing in the image changed, so recreating with your original file stack (the
same `-f` list minus the hotfix override, still including
`compose.override.yaml`) restores the original reader. Rolling back with a bare
`docker compose -f compose.yaml` would instead restore only `compose.yaml` and
quietly lose the GPU and SELinux settings from `compose.override.yaml`. If you
prefer to keep the same service entry in your usual `compose.override.yaml`,
delete that entry and recreate with the `-f` list for your remaining files.

## Notes

- The v2.5.2 image and tag stay untouched, along with your existing Compose
  files, `.env`, model paths and NVMe overlay settings; the additions are the
  override file and the patched copy. No image or model download is needed if
  you already have `ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.2`.
- The override file was rendered with `docker compose --env-file .env -f
  compose.yaml -f compose.override.yaml -f compose.bf16-ple.override.yaml
  config` against this repository's `compose.yaml` plus a `compose.override.yaml`
  holding the README's `label=disable` and dual-GPU examples, in the fix
  workspace: the `/models` (read-only), `/cache` and `/nixl` mounts and the
  override's settings survive and the patched file appears as an extra read-only
  bind at the installed path. That is configuration rendering only; no container
  was started here (this host has no access to the v2.5.2 image and no GPU), so
  the extracted file identity and live mount remain to be confirmed on your host.
- Executed vs. source-only checks: the capacity tests, including the fixture
  isolation check, were executed on CPU (torch CPU build, no GPU/model activity)
  together with the other CPU-runnable tests of this directory
  (`test_cli.py`, `test_config.py`, `test_graph_hooks.py`, `test_offload.py`),
  and the runtime-only patch was applied and compared against both the base
  revision and the tagged v2.5.2 runtime file. `test_native_reader.py` needs the
  compiled reader and `test_pennyroyal.py` needs the SGLang source tree, so
  neither was run here. The Docker steps above were configuration rendering
  only: no image pull and no container were started, so the installed container
  path, the read-only bind and the serving behaviour are source-derived and stay
  your acceptance check.
