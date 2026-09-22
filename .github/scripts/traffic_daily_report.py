#!/usr/bin/env python3
"""Email one daily Pennyroyal traffic report from verified GitHub archive data."""
import argparse
import base64
import copy
import datetime as dt
import json
import subprocess
import sys
from urllib.parse import quote
from zoneinfo import ZoneInfo

from traffic_watchdog import GitHub, UTC, now, required_cutoff, timestamp


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def dates(first, last):
    current = dt.date.fromisoformat(first)
    stop = dt.date.fromisoformat(last)
    while current <= stop:
        yield current.isoformat()
        current += dt.timedelta(days=1)


def read_archive(github, branch):
    head = github.api('git/ref/heads/' + quote(branch, safe=''))['object']['sha']
    tree = github.api('git/trees/' + head)
    require(not tree.get('truncated'), 'Truncated traffic-history root tree')
    entries = {entry['path']: entry for entry in tree['tree'] if entry['type'] == 'blob'}
    required = ('daily.json', 'package-downloads.json', 'repository-metrics.json')
    require(all(path in entries for path in required), 'Required traffic-history file missing')
    data = {}
    for path in required:
        blob = github.api('git/blobs/' + entries[path]['sha'])
        require(blob.get('encoding') == 'base64' and blob.get('content'),
                f'Malformed traffic-history blob: {path}')
        try:
            data[path] = json.loads(base64.b64decode(blob['content']))
        except (ValueError, UnicodeError):
            raise RuntimeError(f'Malformed traffic-history JSON: {path}') from None
    traffic, package, metrics = (data['daily.json'], data['package-downloads.json'],
                                 data['repository-metrics.json'])
    for name, value in [('traffic', traffic), ('package', package), ('repository metrics', metrics)]:
        require(value.get('repository') == github.repository and value.get('schema_version') == 1,
                f'Invalid {name} archive identity/schema')
        require(isinstance(value.get('days'), dict) and value['days'],
                f'Empty {name} archive')
    coverage = traffic.get('coverage')
    require(isinstance(coverage, dict) and coverage.get('complete_through_latest_exposed') is True and
            coverage.get('missing_dates_through_latest_exposed') == [],
            'Traffic archive coverage is incomplete')
    launch, through = traffic.get('launch_date'), coverage.get('through_date')
    require(isinstance(launch, str) and isinstance(through, str) and
            set(dates(launch, through)) == set(traffic['days']),
            'Traffic archive dates are incomplete or inconsistent')
    cutoff = required_cutoff(now())
    for name, value in [('traffic', traffic), ('package', package), ('repository metrics', metrics)]:
        require(timestamp(value.get('collected_at')) >= cutoff,
                f'Stale {name} archive; refusing to email a success report')
    return head, traffic, package, metrics


def counts(value, label):
    require(isinstance(value, dict), f'Invalid {label}')
    result = {}
    for field in ('count', 'uniques'):
        require(type(value.get(field)) is int and value[field] >= 0,
                f'Invalid {label} {field}')
        result[field] = value[field]
    require(result['uniques'] <= result['count'], f'Invalid {label} uniques')
    return result


def live_traffic(github):
    result = {}
    for kind in ('views', 'clones'):
        raw = github.api(f'traffic/{kind}?per=day')
        totals = counts(raw, f'rolling {kind}')
        rows = raw.get(kind)
        require(isinstance(rows, list) and rows, f'Empty rolling {kind} series')
        series = {}
        for row in rows:
            stamp = row.get('timestamp')
            require(isinstance(stamp, str) and stamp.endswith('T00:00:00Z'),
                    f'Invalid rolling {kind} timestamp')
            day = stamp[:10]
            dt.date.fromisoformat(day)
            require(day not in series, f'Duplicate rolling {kind} date')
            series[day] = counts(row, f'rolling {kind} row')
        require(sum(row['count'] for row in series.values()) == totals['count'],
                f'Inconsistent rolling {kind} total')
        result[kind] = {'total': totals, 'days': series}
    require(result['views']['days'].keys() == result['clones']['days'].keys(),
            'Rolling views/clones windows differ')
    return result


def current_days(traffic, rolling):
    days = copy.deepcopy(traffic['days'])
    for kind in ('views', 'clones'):
        overlap = set(days) & set(rolling[kind]['days'])
        if rolling[kind]['total']['count'] == 0:
            require(not any(days[day][kind]['count'] > 0 for day in overlap),
                    f'Suspicious empty rolling {kind} window would erase archived traffic')
    for day in rolling['views']['days']:
        days[day] = {
            'views': rolling['views']['days'][day],
            'clones': rolling['clones']['days'][day],
        }
    merged = dict(sorted(days.items()))
    require(set(dates(traffic['launch_date'], max(merged))) == set(merged),
            'Merged traffic dates are incomplete or inconsistent')
    return merged


def latest_metric(history, fields, label):
    day = max(history['days'])
    value = history['days'][day]
    for field in fields:
        require(type(value.get(field)) is int and value[field] >= 0,
                f'Invalid {label} {field}')
    return day, value


def previous_metric(history, fields):
    entries = sorted(history['days'].items())
    if len(entries) < 2:
        return {field: None for field in fields}
    value = entries[-2][1]
    return {field: value.get(field) for field in fields}


def top_rows(rows, key, label):
    require(isinstance(rows, list), f'Invalid {label} list')
    result = []
    for row in rows[:5]:
        require(isinstance(row.get(key), str) and row[key], f'Invalid {label} name')
        value = counts(row, label)
        result.append((row[key], value['count'], value['uniques']))
    return result


def delta(current, previous):
    return '—' if previous is None else f'{current - previous:+,}'


def report(github, branch):
    head, traffic, package, metrics = read_archive(github, branch)
    rolling = live_traffic(github)
    archived_totals = {
        'views': sum(row['views']['count'] for row in traffic['days'].values()),
        'clones': sum(row['clones']['count'] for row in traffic['days'].values()),
        'daily_unique_cloners': sum(row['clones']['uniques'] for row in traffic['days'].values()),
    }
    days = current_days(traffic, rolling)
    latest_day = max(days)
    latest = days[latest_day]
    totals = {
        'views': sum(row['views']['count'] for row in days.values()),
        'clones': sum(row['clones']['count'] for row in days.values()),
        'daily_unique_visitors': sum(row['views']['uniques'] for row in days.values()),
        'daily_unique_cloners': sum(row['clones']['uniques'] for row in days.values()),
    }
    _, adoption = latest_metric(metrics, ('stars', 'forks'), 'adoption')
    _, package_latest = latest_metric(package, ('total_downloads',), 'package')
    referrers = top_rows(github.api('traffic/popular/referrers?per=day'), 'referrer', 'referrer')
    paths = top_rows(github.api('traffic/popular/paths?per=day'), 'path', 'path')
    values = {
        'stars': adoption['stars'], 'forks': adoption['forks'],
        'views': totals['views'], 'clones': totals['clones'],
        'daily_unique_cloners': totals['daily_unique_cloners'],
        'package_downloads': package_latest['total_downloads'],
    }
    prior = {
        **previous_metric(metrics, ('stars', 'forks')),
        'package_downloads': previous_metric(package, ('total_downloads',))['total_downloads'],
        **archived_totals,
    }
    lines = [
        'Pennyroyal daily GitHub report', '',
        f'Data through: {latest_day} UTC. Archive commit: {head[:12]}.',
        f'Rolling window: {min(rolling["views"]["days"])} through {max(rolling["views"]["days"])} UTC.', '',
        'Adoption',
        f'- Stars: {values["stars"]:,} ({delta(values["stars"], prior.get("stars"))})',
        f'- Forks: {values["forks"]:,} ({delta(values["forks"], prior.get("forks"))})',
        f'- Container downloads: {values["package_downloads"]:,} '
        f'({delta(values["package_downloads"], prior.get("package_downloads"))})', '',
        'Traffic since launch',
        f'- Views: {values["views"]:,} ({delta(values["views"], prior.get("views"))})',
        f'- Clones: {values["clones"]:,} ({delta(values["clones"], prior.get("clones"))})',
        f'- sum_of_daily_unique_cloners: {values["daily_unique_cloners"]:,} '
        f'({delta(values["daily_unique_cloners"], prior.get("daily_unique_cloners"))})',
        '  GitHub does not expose a deduplicated lifetime cloner count.', '',
        f'{latest_day} activity',
        f'- Views: {latest["views"]["count"]:,} from {latest["views"]["uniques"]:,} daily unique visitors',
        f'- Clones: {latest["clones"]["count"]:,} from {latest["clones"]["uniques"]:,} daily unique cloners', '',
        'Current rolling window',
        f'- Views: {rolling["views"]["total"]["count"]:,}; unique visitors: {rolling["views"]["total"]["uniques"]:,}',
        f'- Clones: {rolling["clones"]["total"]["count"]:,}; unique cloners: {rolling["clones"]["total"]["uniques"]:,}', '',
        'Top referrers',
    ]
    lines += [f'- {name}: {count:,} views / {uniques:,} unique visitors'
              for name, count, uniques in referrers] or ['- None returned by GitHub']
    lines += ['', 'Top paths']
    lines += [f'- {name}: {count:,} views / {uniques:,} unique visitors'
              for name, count, uniques in paths] or ['- None returned by GitHub']
    lines += ['', f'https://github.com/{github.repository}/tree/{branch}']
    return '\n'.join(lines) + '\n', values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', required=True)
    parser.add_argument('--history-branch', default='traffic-history')
    parser.add_argument('--recipient', required=True)
    parser.add_argument('--timezone', default='America/New_York')
    parser.add_argument('--mail-command', default='/usr/bin/mail')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    local_date = now().astimezone(ZoneInfo(args.timezone)).date().isoformat()
    body, _ = report(GitHub(args.repository, 'archive-traffic.yml'), args.history_branch)
    subject = f'Pennyroyal daily GitHub report — {local_date}'
    if args.dry_run:
        print(body, end='')
        return
    result = subprocess.run([args.mail_command, '-s', subject, args.recipient], input=body,
                            text=True, timeout=60, check=False)
    if result.returncode:
        raise RuntimeError(f'Mail sender failed with exit {result.returncode}')
    print(f'Emailed daily traffic report for {local_date}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'TRAFFIC REPORT ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
