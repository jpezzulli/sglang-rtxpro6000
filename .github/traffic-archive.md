# Permanent GitHub traffic archive

The workflow `.github/workflows/archive-traffic.yml` runs at **00:17 UTC daily**
(`17 0 * * *`) on `pennyroyal-main-sm120-final` and supports manual dispatch.
Scheduled execution can be delayed by GitHub. Run manually with:

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
