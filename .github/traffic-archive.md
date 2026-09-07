# Permanent GitHub traffic archive

The workflow `.github/workflows/archive-traffic.yml` runs at **00:17 UTC daily**
(`17 0 * * *`) on `pennyroyal-main-sm120-final` and supports manual dispatch.
Scheduled execution can be delayed or dropped by GitHub. An independent persistent
timer on thegrid now ensures a daily capture (details below). Run manually with:

```sh
gh workflow run archive-traffic.yml --repo jpezzulli/sglang-rtxpro6000 --ref pennyroyal-main-sm120-final
```

Read the [archive summary](https://github.com/jpezzulli/sglang-rtxpro6000/tree/traffic-history).
The independent `traffic-history` branch contains only archive data, no runtime
source or workflows. Every update is one atomic Git Data API commit. No checkout,
clone, Git fetch, package installation, or third-party Action is used. The script
itself is retrieved using Contents API at the workflow's exact commit SHA.

## Files and data semantics

- `daily.json`: date-keyed daily views/clones with GitHub's original `count` and
  `uniques`, plus the last collection time for each date. Starts at 2026-08-24.
- `snapshots/YYYY-MM-DD/<UTC timestamp>-<run ID>-<attempt>.json`: immutable per-run
  snapshot of all four traffic responses, including rolling counts, rolling
  uniques, referrers and popular paths. Multiple runs on a day are retained.
- `raw/first-run/{views,clones,referrers,paths}.json`: original response bodies,
  byte-for-byte, including any prelaunch days exposed by GitHub.
- `raw/first-run/metadata.json`: first capture provenance, SHA-256 checksums,
  endpoint query strings and exact coverage/missing dates.
- `README.md`: cumulative daily counts, daily table, current rolling uniques,
  and current referrer/path tables. Previous snapshots remain in the archive.

All structured archive documents contain repository, launch date, UTC collection
time, schema version `1`, and API version `2022-11-28`. Unmodified raw API bodies
are associated with their metadata sidecar. Referrer/path endpoints return top
10 rolling-window lists, not daily breakdowns; `per=day` is supplied on all four
requests but is meaningful only for views/clones.

Each run retrieves and merges the entire returned daily window, replacing both
counts and daily uniques for overlapping dates. Older dates and snapshots are
preserved. Exact cumulative counts apply only through the latest exposed date
when every date since launch is present. Missing dates are listed without zero
filling; recent unexposed dates are labeled separately. Counts are GitHub-reported
activity, including any bots GitHub counts, not verified human activity.

Never add rolling totals or rolling unique values across snapshots. The summary
labels daily-unique sums as `sum_of_daily_uniques`; these are not a deduplicated
lifetime person count. GitHub exposes no identities for historical deduplication.

## Authentication and permissions

The job grants only `contents: write` to its built-in `GITHUB_TOKEN`. It exercises
that token against all four traffic endpoints and logs HTTP status only. If those
reads succeed, no separate token is needed. All archive writes always use
`GITHUB_TOKEN` so history updates do not trigger other workflows.

If GitHub returns HTTP 403, repository Actions secret **`TRAFFIC_READ_TOKEN`**
provides traffic read access. John explicitly selected reuse of his existing
GitHub CLI OAuth credential rather than creating a fine-grained token. The
credential is transferred directly from `gh auth token` into `gh secret set`
through a pipe; it must never appear in command arguments, logs, files or commits.
All archive writes continue to use `GITHUB_TOKEN`.

The current operator OAuth credential has `repo` access and the separate
`workflow` scope needed for the one-time workflow installation. Revoking this
OAuth authorization requires replacing the Actions secret. Authentication errors
fail the collector without changing history.

For a future separately scoped credential, GitHub documents **Administration:
read** for all four traffic endpoints; Contents write is not required on that
traffic-read credential. This is documentation, not a requirement to change
John's selected authentication method.

GitHub traffic permission reference:
https://docs.github.com/en/rest/metrics/traffic

## Integrity and failure behavior

A single repository-wide concurrency group serializes scheduled/manual runs;
`cancel-in-progress: false` lets active collections finish. The collector reads
history from an immutable commit SHA, creates a tree based on that history, checks
that the branch has not moved, and updates the ref with `force: false`. A concurrent
writer causes a loud failure; rerun to merge the new head. Never force-push history.

Authentication/HTTP/transport errors, malformed JSON, invalid or duplicate daily
records, empty daily arrays, all-zero windows that would erase positive overlapping
history, mismatched daily windows or totals, corrupt history,
and branch conflicts fail before replacing the branch. Missing history cannot be
silently recreated during a scheduled run. Empty referrer/path lists can be valid;
they are stored as new snapshots and never delete previous snapshots. Original
raw capture files are written only during the explicit seed operation.

The initial branch is an orphan archive-only commit. History commit messages also
carry `[skip ci]`. Combined with built-in-token writes and no workflows on the
history branch, archive updates do not start runtime build/test jobs. Runtime
source, existing workflows, tags, releases and local candidate work are untouched.

To test archive merge/integrity logic without GitHub access:

```sh
python3 .github/scripts/test_archive_traffic.py
```

The first capture exposed 2026-08-22 through 2026-09-04, including all dates from
launch on 2026-08-24. No launch-period dates had been discarded. August 24 contains
explicit GitHub-reported zero counts. September 5 and 6 were not yet exposed at
the initial 2026-09-06T01:14:03Z collection; they were not invented or counted.

## Independent timer and recovery on thegrid

GitHub's first 00:17 UTC schedule event produced no run by 2026-09-07T02:03Z,
although the workflow was active on the correct default branch. Manual dispatch
continued to succeed. The underlying GitHub scheduler cause is unconfirmed;
manual success alone does not verify GitHub's schedule delivery.

Thegrid therefore runs `traffic-archive-watchdog.timer` as `mrkaos`, independently
of Codex, Neomatrix, and any interactive session. User lingering was already enabled.
The timer starts a check 30 seconds after activation and at minutes **17 and 47
of every UTC hour**, with `Persistent=true` for calendar catch-up after downtime.
The target is one capture at or after **00:17 UTC each day**. Fresh days perform
only read-only API checks; this does not make twice-hourly archive commits.
The first activation commissions one real workflow run even if a manual capture
is already fresh, proving that unattended credentials and execution work.

The watchdog dispatches the existing GitHub workflow if collection is overdue.
It adopts a queued/running collector instead of dispatching another, waits up to
eight minutes, and verifies both workflow success and a fresh `daily.json` on
`traffic-history`. A still-pending run is resumed on the next timer tick. If a pending run was
deleted, a confirmed HTTP 404 plus a successful run listing retires that reference
so recovery can proceed; authentication errors never clear pending state. Failed
runs or transport/authentication failures leave history untouched, mark the local
service failed, and are retried on later timer ticks. Ambiguous dispatch responses
are reconciled against GitHub run history before another dispatch is attempted.
The existing GitHub workflow concurrency group serializes all collectors.

The watchdog reuses the operator's `gh` login directly and does not copy or print
a credential. It makes no Git clone/fetch requests and never writes history
itself. GitHub retains collection and all archive writes, using the existing
Actions secret and built-in write token. GitHub's original daily schedule remains
active as a second trigger; its delivery is not relied on for the guarantee.
Both schedulers still depend on GitHub API/Actions availability; recovery resumes
when the service is available. A thegrid outage is caught up after user-manager
startup; data absent from GitHub's retention window cannot be reconstructed.

Installed locations:

- `/opt/sglang/traffic-archive/traffic_watchdog.py`: deployed watchdog.
- `/opt/sglang/traffic-archive/state/verified.json`: last run verified by watchdog.
- `/opt/sglang/traffic-archive/state/pending.json`: in-flight dispatch, when present.
- `~/.config/systemd/user/traffic-archive-watchdog.{service,timer}`: user units.
- `.github/traffic-archive/`: portable unit templates; substitute `@INSTALL_DIR@`
  and `@REPOSITORY@` in the service template during installation.

Inspect without changing anything:

```sh
systemctl --user list-timers traffic-archive-watchdog.timer
systemctl --user status traffic-archive-watchdog.service
journalctl --user -u traffic-archive-watchdog.service --since today
cat /opt/sglang/traffic-archive/state/verified.json
```

A successful oneshot service normally shows `inactive (dead)` with exit status 0;
the timer remains `active (waiting)`. Check that combination, the journal, the
verified state, and the remote archive timestamp. Do not interpret a completed
oneshot's inactive state as a failed scheduler.

Focused watchdog tests: `python3 .github/scripts/test_traffic_watchdog.py`.
No runtime code or inference service participates in this mechanism.
