# Fix Swift's NVMe PLE error in Pennyroyal v2.5.2

If Swift loads but fails on a request with `PLE lookup needs ... staging rows`,
this fix gives its larger BF16 PLE table enough working space. You do not need
to rebuild Docker or download the model again. Other models using FP8 PLE keep
the same memory allocation.

These steps are for the standard Docker Compose setup from our
[container guide](../../docker/pennyroyal/README.md), using `compose.yaml`,
`.env`, and an optional `compose.override.yaml`. If you use different file names,
a custom launch command, or Portainer, ask in [issue #21](https://github.com/jpezzulli/sglang-rtxpro6000/issues/21)
for instructions that match your setup.

## 1. Open your Pennyroyal folder

Open a Bash terminal in the `docker/pennyroyal` folder of your Pennyroyal
checkout—the folder containing `compose.yaml` and your `.env` settings.
Keep this terminal open for the following steps.

## 2. Download the fix

Copy and paste this whole block. It downloads the fixed file and a small Docker
settings file. It also checks the download before you continue. Your existing
settings and model files are left alone.

```bash
(
  set -eu
  test -f compose.yaml
  test -f .env
  mkdir -p .penny-bf16-hotfix
  curl -fL https://raw.githubusercontent.com/jpezzulli/sglang-rtxpro6000/e6319ae461d9e54d7d777878a4bf7fc503a69f0f/tools/ple_nvme/ssd_stream/src/sglang_ssd_stream/backend.py \
    -o .penny-bf16-hotfix/backend.py
  printf '%s  %s\n' \
    6f25e0a01c90c73e42eec667a6fcea10cc94bee53967100542eca4f9f2bc6a81 \
    .penny-bf16-hotfix/backend.py | sha256sum --check
  curl -fL https://raw.githubusercontent.com/jpezzulli/sglang-rtxpro6000/pennyroyal-main-sm120-final/tools/ple_nvme/patches/compose.bf16-ple.yaml \
    -o compose.bf16-ple.yaml
  chmod 755 .penny-bf16-hotfix
  chmod 644 .penny-bf16-hotfix/backend.py
  echo 'Fix downloaded. Continue to step 3.'
)
```

Wait for **“Fix downloaded. Continue to step 3.”** If you see an error instead,
stop and share it in the issue.

## 3. Restart Pennyroyal with the fix

Finish any running conversation first. Paste this block in the same terminal.
It keeps your usual settings, including an existing `compose.override.yaml`,
and adds the fixed file for this container.

```bash
(
  set -eu
  set -- -f compose.yaml
  if [ -f compose.override.yaml ]; then
    set -- "$@" -f compose.override.yaml
  fi
  docker compose "$@" -f compose.bf16-ple.yaml config --quiet
  docker compose "$@" -f compose.bf16-ple.yaml up -d --force-recreate pennyroyal
)
docker compose logs -f pennyroyal
```

Wait for Pennyroyal to finish loading, then try the request again. Press
**Ctrl+C** to stop watching the logs; that does not stop Pennyroyal.

Use the step 3 restart block again if you later recreate the container and want
to keep this fix. The published `v2.5.2` image itself is unchanged.

## Undo it

From the same folder, run:

```bash
docker compose up -d --force-recreate pennyroyal
```

That restarts Pennyroyal using its original files and your usual settings,
without the hotfix. There is no need to delete or restore any model files.

## What has been checked

The focused checks pass for FP8 and BF16, and the fix matches the v2.5.2 reader.
The Docker settings were checked, but Swift has not been run with this packaged
fix here. Please report whether it resolves your request error.

For source installations or anyone who prefers applying a patch, the
[small patch file](patches/bf16-ple-staging-hotfix-backend.patch) is also available.
