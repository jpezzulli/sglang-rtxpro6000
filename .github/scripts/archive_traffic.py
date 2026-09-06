#!/usr/bin/env python3
"""Archive GitHub traffic using REST only; never clone or fetch Git objects."""
import argparse
import base64
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request

REPOSITORY = 'jpezzulli/sglang-rtxpro6000'
BRANCH = 'traffic-history'
LAUNCH = '2026-08-24'
API_VERSION = '2022-11-28'
SCHEMA_VERSION = 1
ENDPOINTS = {'views': 'views?per=day', 'clones': 'clones?per=day',
             'referrers': 'popular/referrers?per=day', 'paths': 'popular/paths?per=day'}


class ArchiveError(RuntimeError):
    pass


class APIError(ArchiveError):
    def __init__(self, status, method, path):
        self.status = status
        super().__init__(f'GitHub API {method} {path}: HTTP {status}')


class API:
    def __init__(self, token):
        if not token:
            raise ArchiveError('Authentication token missing')
        self.token = token

    def request(self, path, method='GET', data=None, raw=False):
        req = urllib.request.Request(
            f'https://api.github.com/repos/{REPOSITORY}/{path}',
            data=None if data is None else json.dumps(data).encode(), method=method,
            headers={'Authorization': f'Bearer {self.token}',
                     'Accept': 'application/vnd.github+json',
                     'X-GitHub-Api-Version': API_VERSION,
                     'Content-Type': 'application/json', 'User-Agent': 'traffic-archive'})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise APIError(exc.code, method, path) from None
        except (urllib.error.URLError, TimeoutError):
            raise ArchiveError(f'GitHub API transport failure: {method} {path}') from None
        if not body:
            raise ArchiveError(f'Empty API response: {path}')
        try:
            parsed = json.loads(body)
        except (ValueError, UnicodeError):
            raise ArchiveError(f'Malformed JSON: {path}') from None
        return body if raw else parsed


def require(condition, message):
    if not condition:
        raise ArchiveError(message)


def dates(start, end):
    day, stop = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    while day <= stop:
        yield day.isoformat()
        day += dt.timedelta(days=1)


def pair(value):
    require(isinstance(value, dict), 'Expected count/uniques object')
    for field in ('count', 'uniques'):
        require(type(value.get(field)) is int and value[field] >= 0,
                f'Invalid {field}')
    require(value['uniques'] <= value['count'], 'uniques exceeds count')
    return {k: value[k] for k in ('count', 'uniques')}


def validate(raw):
    require(set(raw) == set(ENDPOINTS), 'Incomplete collection')
    result = {}
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    for kind in ('views', 'clones'):
        value = raw[kind]
        pair(value)
        rows = value.get(kind)
        require(isinstance(rows, list) and rows, f'Empty or malformed {kind} daily series')
        found = {}
        for row in rows:
            counts = pair(row)
            timestamp = row.get('timestamp', '')
            require(isinstance(timestamp, str) and
                    re.fullmatch(r'\d{4}-\d{2}-\d{2}T00:00:00Z', timestamp), 'Invalid daily timestamp')
            day = timestamp[:10]
            dt.date.fromisoformat(day)
            require(day <= today and day not in found, 'Future or duplicate date')
            found[day] = counts
        require(1 <= len(found) <= 15, 'Unexpected daily window size')
        require(set(found) == set(dates(min(found), max(found))), 'Gap inside API daily window')
        require(sum(v['count'] for v in found.values()) == value['count'],
                f'{kind} aggregate does not match daily counts')
        require(max(v['uniques'] for v in found.values()) <= value['uniques'] <=
                sum(v['uniques'] for v in found.values()), f'Inconsistent {kind} rolling uniques')
        result[kind] = found
    require(result['views'].keys() == result['clones'].keys(), 'Views/clones windows differ')
    for kind, key in [('referrers', 'referrer'), ('paths', 'path')]:
        rows = raw[kind]
        require(isinstance(rows, list) and len(rows) <= 10, f'Malformed {kind}')
        seen = set()
        for row in rows:
            pair(row)
            require(isinstance(row.get(key), str) and row[key] and row[key] not in seen,
                    f'Invalid or duplicate {kind} entry')
            if kind == 'paths':
                require(isinstance(row.get('title'), str), 'Invalid popular-path title')
            seen.add(row[key])
        # [] is a legitimate top-list result; timestamped previous lists are never deleted.
    return result


def metadata(stamp):
    return {'schema_version': SCHEMA_VERSION, 'repository': REPOSITORY,
            'launch_date': LAUNCH, 'api_version': API_VERSION, 'collected_at': stamp}


def validate_history(history):
    for key, expected in metadata(history.get('collected_at')).items():
        require(history.get(key) == expected, f'History metadata mismatch: {key}')
    require(isinstance(history.get('collected_at'), str) and
            re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', history['collected_at']),
            'Invalid history collection timestamp')
    require(isinstance(history.get('days'), dict) and history['days'], 'Empty historical archive')
    for day, row in history['days'].items():
        require(dt.date.fromisoformat(day).isoformat() == day and day >= LAUNCH, 'Invalid history date')
        for kind in ('views', 'clones'):
            pair(row[kind])
        require(isinstance(row.get('last_collected_at'), str), 'Missing historical timestamp')


def merge(history, raw, stamp):
    series = validate(raw)
    if history is not None:
        validate_history(history)
        require(stamp >= history['collected_at'], 'Collection is older than latest archive')
    days = copy.deepcopy(history['days']) if history is not None else {}
    for kind in ('views', 'clones'):
        overlap = set(days) & set(series[kind])
        if overlap and raw[kind]['count'] == 0:
            require(not any(days[d][kind]['count'] > 0 for d in overlap),
                    f'Suspicious empty {kind} window would erase positive historical data')
    for day in series['views']:
        if day >= LAUNCH:
            days[day] = {kind: series[kind][day] for kind in ('views', 'clones')}
            days[day]['last_collected_at'] = stamp
    require(days, 'No launch-or-later data returned')
    last = max(days)
    missing = sorted(set(dates(LAUNCH, last)) - set(days))
    today = stamp[:10]
    pending = list(dates((dt.date.fromisoformat(last) + dt.timedelta(days=1)).isoformat(), today))
    result = metadata(stamp)
    result.update({'coverage': {'first_date': min(days), 'through_date': last,
                               'missing_dates_through_latest_exposed': missing,
                               'not_yet_exposed_dates': pending,
                               'complete_through_latest_exposed': not missing},
                   'days': dict(sorted(days.items()))})
    return result


def render(history, raw, snapshot_path):
    c = history['coverage']
    exact = c['complete_through_latest_exposed']
    lines = ['# GitHub traffic archive', '', f'Repository: `{REPOSITORY}`. Launch: **{LAUNCH}**.',
             f'Collected: **{history["collected_at"]}**. Coverage through **{c["through_date"]}** (UTC).', '',
             'Counts are exact sums of GitHub-reported daily counts for the covered dates; '
             'the newest dates may be partial and GitHub can revise the overlapping window.', '',
             '| Metric | Value |', '| --- | ---: |']
    for kind in ('views', 'clones'):
        total = sum(row[kind]['count'] for row in history['days'].values())
        label = f'Exact cumulative {kind} since launch through {c["through_date"]}' if exact else f'Known {kind} (incomplete launch coverage)'
        lines.append(f'| {label} | {total:,} |')
        lines.append(f'| Current rolling-window {kind} | {raw[kind]["count"]:,} |')
        label = 'unique visitors' if kind == 'views' else 'unique cloners'
        lines.append(f'| Current rolling-window {label} | {raw[kind]["uniques"]:,} |')
        su = sum(row[kind]['uniques'] for row in history['days'].values())
        lines.append(f'| {kind}: `sum_of_daily_uniques` | {su:,} |')
    lines += ['', '**No exact lifetime unique-person count is available.** Daily unique values '
              'and rolling-window unique values are GitHub metrics. Summing daily uniques does '
              'not deduplicate people across days or windows.', '',
              'Missing dates through latest exposed date: ' + (', '.join(c['missing_dates_through_latest_exposed']) or 'none') + '.',
              'Not yet exposed: ' + (', '.join(c['not_yet_exposed_dates']) or 'none') + '.', '',
              '## Daily history', '', '| UTC date | Views | Daily unique visitors | Clones | Daily unique cloners |',
              '| --- | ---: | ---: | ---: | ---: |']
    for day, row in history['days'].items():
        lines.append(f'| {day} | {row["views"]["count"]} | {row["views"]["uniques"]} | {row["clones"]["count"]} | {row["clones"]["uniques"]} |')
    for kind, key, title in [('referrers', 'referrer', 'Referring sites'), ('paths', 'path', 'Popular paths')]:
        lines += ['', f'## {title} — current rolling-window snapshot', '',
                  '| Source/path | Views | Unique visitors within this snapshot |', '| --- | ---: | ---: |']
        for row in raw[kind]:
            label = row[key].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '&#124;').replace('\n', ' ')
            lines.append(f'| {label} | {row["count"]} | {row["uniques"]} |')
        if not raw[kind]:
            lines.append('| No entries returned by GitHub | — | — |')
    lines += ['', f'[Latest full snapshot]({snapshot_path}) · [Daily JSON](daily.json) · '
              '[Immutable first-run raw responses](raw/first-run/)', '',
              'Snapshots retain the aggregate totals, daily arrays, referrers and popular paths as '
              'reported together in each collection. They overlap and must not be added together.', '',
              'Collection: daily at **00:17 UTC** (`17 0 * * *`) and manual dispatch. '
              'GitHub may delay scheduled execution. All collection and branch updates use REST APIs; '
              'there is no repository checkout, clone or Git fetch.', '']
    return '\n'.join(lines)


def encode(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n'


def collect(api):
    bodies = {kind: api.request('traffic/' + endpoint, raw=True) for kind, endpoint in ENDPOINTS.items()}
    raw = {kind: json.loads(body) for kind, body in bodies.items()}
    validate(raw)
    return raw, bodies


def read_history(api, allow_create=False):
    try:
        head = api.request(f'git/ref/heads/{BRANCH}')['object']['sha']
    except APIError as exc:
        if exc.status == 404 and allow_create:
            return None, None, None
        raise
    commit = api.request(f'git/commits/{head}')
    tree_sha = commit['tree']['sha']
    tree = api.request(f'git/trees/{tree_sha}')
    require(not tree.get('truncated'), 'Truncated history root tree')
    roots = {entry['path']: entry for entry in tree['tree']}
    require('daily.json' in roots and 'raw' in roots and 'snapshots' in roots,
            'Existing history branch is incomplete; refusing to reset it')
    blob = api.request('git/blobs/' + roots['daily.json']['sha'])
    require(blob['encoding'] == 'base64', 'Unsupported history blob encoding')
    history = json.loads(base64.b64decode(blob['content'], validate=False))
    validate_history(history)
    return head, tree_sha, history


def publish(api, head, tree_sha, files, message):
    entries = [{'path': path, 'mode': '100644', 'type': 'blob', 'content': content}
               for path, content in files.items()]
    payload = {'tree': entries}
    if tree_sha is not None:
        payload['base_tree'] = tree_sha
    tree = api.request('git/trees', 'POST', payload)['sha']
    commit = api.request('git/commits', 'POST', {'message': message + ' [skip ci]', 'tree': tree,
                                               'parents': [head] if head else []})['sha']
    if head:
        require(api.request(f'git/ref/heads/{BRANCH}')['object']['sha'] == head,
                'Branch-write conflict: history moved; rerun to merge the new head')
        api.request(f'git/refs/heads/{BRANCH}', 'PATCH', {'sha': commit, 'force': False})
    else:
        api.request('git/refs', 'POST', {'ref': f'refs/heads/{BRANCH}', 'sha': commit})
    require(api.request(f'git/ref/heads/{BRANCH}')['object']['sha'] == commit,
            'Post-write verification failed: unexpected history head')
    return commit


def archive(api, raw, bodies, stamp, auth, seed=False, original_metadata=None):
    head, tree_sha, previous = read_history(api, allow_create=seed)
    require(not seed or head is None, 'Seed is only allowed for a new history branch')
    history = merge(previous, raw, stamp)
    run_id = os.environ.get('GITHUB_RUN_ID', 'seed')
    attempt = os.environ.get('GITHUB_RUN_ATTEMPT', '1')
    snapshot_path = f'snapshots/{stamp[:10]}/{stamp.replace(":", "")}-{run_id}-{attempt}.json'
    snapshot = metadata(stamp)
    snapshot.update({'authentication': auth, 'run_id': run_id, 'run_attempt': attempt,
                     'endpoints': ENDPOINTS, 'traffic': raw, 'coverage': history['coverage']})
    summary = render(history, raw, snapshot_path)
    files = {'daily.json': encode(history), 'README.md': summary, snapshot_path: encode(snapshot)}
    if seed:
        meta = metadata(stamp)
        meta.update({'endpoints': ENDPOINTS, 'coverage': history['coverage'],
                     'original_capture_metadata': original_metadata,
                     'sha256': {k + '.json': hashlib.sha256(v).hexdigest() for k, v in bodies.items()}})
        files.update({f'raw/first-run/{kind}.json': body.decode('utf-8') for kind, body in bodies.items()})
        files['raw/first-run/metadata.json'] = encode(meta)
    sha = publish(api, head, tree_sha, files, f'Archive traffic collected {stamp}')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        Path(os.environ['GITHUB_STEP_SUMMARY']).write_text(summary)
    print(f'Archived {len(history["days"])} days through {history["coverage"]["through_date"]}; '
          f'history commit {sha}; authentication {auth}')
    if history['coverage']['missing_dates_through_latest_exposed']:
        print('::warning::Missing historical dates: ' + ', '.join(history['coverage']['missing_dates_through_latest_exposed']))
    return sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-directory', type=Path)
    args = parser.parse_args()
    if args.seed_directory:
        token = subprocess.run(['gh', 'auth', 'token'], capture_output=True, check=True).stdout.decode().strip()
        api = API(token)
        bodies = {kind: (args.seed_directory / (kind + '.json')).read_bytes() for kind in ENDPOINTS}
        raw = {kind: json.loads(body) for kind, body in bodies.items()}
        meta = json.loads((args.seed_directory / 'capture-metadata.json').read_text())
        require(not meta['errors'], 'Initial capture contains API failures')
        stamp = dt.datetime.strptime(meta['collected_at'], '%Y%m%dT%H%M%SZ').strftime('%Y-%m-%dT%H:%M:%SZ')
        archive(api, raw, bodies, stamp, 'operator gh OAuth (initial seed only)', seed=True, original_metadata=meta)
        return
    require(os.environ.get('GITHUB_REPOSITORY') == REPOSITORY, 'Unexpected repository')
    api = API(os.environ.get('GITHUB_TOKEN'))
    auth = 'GITHUB_TOKEN'
    # Exercise the built-in token against all four actual endpoints. A separate
    # operator-authorized token is used only for traffic reads after a permission rejection.
    failures = []
    bodies = {}
    for kind, endpoint in ENDPOINTS.items():
        try:
            bodies[kind] = api.request('traffic/' + endpoint, raw=True)
            print(f'GITHUB_TOKEN traffic/{endpoint}: HTTP 200')
        except APIError as exc:
            if exc.status != 403:
                raise
            failures.append(kind)
            print(f'GITHUB_TOKEN traffic/{endpoint}: HTTP 403')
    if failures:
        fallback = os.environ.get('TRAFFIC_READ_TOKEN')
        require(fallback, 'GITHUB_TOKEN cannot read traffic. Set Actions secret TRAFFIC_READ_TOKEN '
                'to the operator-authorized GitHub credential with traffic access. '
                'GITHUB_TOKEN continues to perform all Contents writes.')
        raw, bodies = collect(API(fallback))
        auth = 'TRAFFIC_READ_TOKEN (operator-authorized credential); GITHUB_TOKEN (Contents: write)'
    else:
        raw = {kind: json.loads(body) for kind, body in bodies.items()}
        validate(raw)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    archive(api, raw, bodies, stamp, auth)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # API errors contain only method, path and status; never emit credentials.
        print(f'::error::{type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
