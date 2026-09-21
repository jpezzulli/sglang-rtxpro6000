#!/usr/bin/env python3
"""Archive public GitHub Container Registry download counters."""
import argparse
import copy
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote
import urllib.error
import urllib.request

from archive_traffic import API, APIError, ArchiveError, BRANCH, REPOSITORY, encode, publish, require

PACKAGE_OWNER = 'jpezzulli'
PACKAGE_NAME = 'sglang-rtxpro6000'
PACKAGE_PUBLISHED_DATE = '2026-09-15'
PACKAGE_URL = f'https://github.com/users/{PACKAGE_OWNER}/packages/container/package/{PACKAGE_NAME}'
PACKAGE_VERSION_BASE_URL = f'https://github.com/users/{PACKAGE_OWNER}/packages/container/{PACKAGE_NAME}'
PACKAGE_SCHEMA_VERSION = 1
USER_AGENT = 'sglang-rtxpro6000-download-archive'


def fetch_page(url):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise ArchiveError(f'Package page HTTP {exc.code}: {url}') from None
    except (urllib.error.URLError, TimeoutError):
        raise ArchiveError(f'Package page transport failure: {url}') from None
    require(body, f'Empty package page: {url}')
    try:
        body.decode('utf-8')
    except UnicodeError:
        raise ArchiveError(f'Package page is not UTF-8: {url}') from None
    return body


def number(value, label):
    require(isinstance(value, str) and
            re.fullmatch(r'(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)', value),
            f'Invalid {label}')
    return int(value.replace(',', ''))


def version_links(text):
    pattern = (rf'/users/{re.escape(PACKAGE_OWNER)}/packages/container/'
               rf'{re.escape(PACKAGE_NAME)}/(\d+)\?tag=([^"&]+)')
    found = {}
    for version_id, tag in re.findall(pattern, text):
        found.setdefault(version_id, []).append(unquote(html.unescape(tag)))
    return {version_id: sorted(set(tags)) for version_id, tags in found.items()}


def parse_index(body):
    text = body.decode('utf-8')
    require(f'ghcr.io/{PACKAGE_OWNER}/{PACKAGE_NAME}' in text, 'Wrong package page identity')
    total = re.search(r'Total downloads</span>\s*<h3 title="([0-9,]+)">([0-9,]+)</h3>', text)
    require(total and total.group(1) == total.group(2), 'Missing or inconsistent package total')
    marker = '<h3 class="f5">Recent tagged image versions</h3>'
    require(marker in text, 'Missing recent tagged versions section')
    section = text.split(marker, 1)[1].split('<div data-view-component="true" class="Box-footer">', 1)[0]
    rows = re.findall(r'<li data-view-component="true" class="Box-row">(.*?)</li>', section, re.S)
    require(rows, 'No tagged package versions returned')
    versions = {}
    for row in rows:
        links = version_links(row)
        require(len(links) == 1, 'Malformed package version row')
        version_id, tags = next(iter(links.items()))
        digest = re.search(r'value="(sha256:[0-9a-f]{64})"', row)
        downloads = re.search(r'\s([0-9][0-9,]*)\s*<span class="sr-only">Version downloads</span>', row)
        require(digest and downloads, 'Incomplete package version row')
        require(version_id not in versions, 'Duplicate package version ID')
        versions[version_id] = {
            'version_id': int(version_id), 'tags': tags, 'digest': digest.group(1),
            'downloads': number(downloads.group(1), 'version downloads'),
            'available': True,
            'html_url': f'{PACKAGE_VERSION_BASE_URL}/{version_id}',
        }
    result = {'total_downloads': number(total.group(1), 'total downloads'),
              'versions': versions}
    require(all(v['downloads'] <= result['total_downloads'] for v in versions.values()),
            'Version downloads exceed package total')
    return result


def parse_version(body, version_id, prior=None):
    text = body.decode('utf-8')
    links = version_links(text)
    require(version_id in links, f'Version page identity mismatch: {version_id}')
    downloads = re.search(
        r'Total downloads</span>\s*<span class="flex-auto text-right text-bold">([0-9,]+)</span>', text)
    digest = re.search(
        rf'value="FROM ghcr\.io/{re.escape(PACKAGE_OWNER)}/{re.escape(PACKAGE_NAME)}@'
        r'(sha256:[0-9a-f]{64})"', text)
    require(downloads and digest, f'Incomplete package version page: {version_id}')
    return {
        'version_id': int(version_id), 'tags': links[version_id], 'digest': digest.group(1),
        'downloads': number(downloads.group(1), 'version downloads'), 'available': True,
        'html_url': f'{PACKAGE_VERSION_BASE_URL}/{version_id}',
    }


def package_metadata(stamp, first_collected_at):
    return {
        'schema_version': PACKAGE_SCHEMA_VERSION,
        'repository': REPOSITORY,
        'package': f'ghcr.io/{PACKAGE_OWNER}/{PACKAGE_NAME}',
        'package_html_url': PACKAGE_URL,
        'package_published_date': PACKAGE_PUBLISHED_DATE,
        'first_collected_at': first_collected_at,
        'collected_at': stamp,
    }


def validate_history(history):
    require(isinstance(history, dict), 'Malformed package history')
    required = {
        'schema_version': PACKAGE_SCHEMA_VERSION,
        'repository': REPOSITORY,
        'package': f'ghcr.io/{PACKAGE_OWNER}/{PACKAGE_NAME}',
        'package_html_url': PACKAGE_URL,
        'package_published_date': PACKAGE_PUBLISHED_DATE,
    }
    for key, expected in required.items():
        require(history.get(key) == expected, f'Package history metadata mismatch: {key}')
    for field in ('first_collected_at', 'collected_at'):
        require(isinstance(history.get(field), str) and
                re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', history[field]),
                f'Invalid package history timestamp: {field}')
    require(isinstance(history.get('days'), dict) and history['days'], 'Empty package history')
    for day, observation in history['days'].items():
        require(dt.date.fromisoformat(day).isoformat() == day, 'Invalid package history date')
        validate_observation(observation)


def validate_version(value):
    require(isinstance(value, dict), 'Malformed package version')
    require(type(value.get('version_id')) is int and value['version_id'] > 0,
            'Invalid package version ID')
    require(isinstance(value.get('tags'), list) and
            all(isinstance(tag, str) and tag for tag in value['tags']), 'Invalid package tags')
    require(isinstance(value.get('digest'), str) and
            re.fullmatch(r'sha256:[0-9a-f]{64}', value['digest']), 'Invalid package digest')
    require(type(value.get('downloads')) is int and value['downloads'] >= 0,
            'Invalid package version downloads')
    require(type(value.get('available')) is bool, 'Invalid package availability')
    require(isinstance(value.get('last_observed_at'), str) and
            re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z',
                         value['last_observed_at']), 'Invalid package version observation time')
    require(isinstance(value.get('html_url'), str) and
            value['html_url'].startswith(PACKAGE_VERSION_BASE_URL + '/'),
            'Invalid package version URL')


def validate_observation(observation):
    require(isinstance(observation, dict), 'Malformed package observation')
    require(isinstance(observation.get('collected_at'), str) and
            re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', observation['collected_at']),
            'Invalid package observation timestamp')
    require(type(observation.get('total_downloads')) is int and
            observation['total_downloads'] >= 0, 'Invalid package total downloads')
    require(isinstance(observation.get('source_sha256'), str) and
            re.fullmatch(r'[0-9a-f]{64}', observation['source_sha256']), 'Invalid package source hash')
    require(isinstance(observation.get('versions'), dict), 'Invalid package versions')
    for version_id, version in observation['versions'].items():
        require(version_id == str(version.get('version_id')), 'Package version key mismatch')
        validate_version(version)


def merge(history, collected, page_body, stamp):
    if history is not None:
        validate_history(history)
        require(stamp >= history['collected_at'], 'Package collection is older than latest archive')
        first = history['first_collected_at']
        days = copy.deepcopy(history['days'])
    else:
        first = stamp
        days = {}
    versions = copy.deepcopy(dict(sorted(
        collected['versions'].items(), key=lambda item: int(item[0]))))
    for version in versions.values():
        if version['available'] or 'last_observed_at' not in version:
            version['last_observed_at'] = stamp
    observation = {
        'collected_at': stamp,
        'total_downloads': collected['total_downloads'],
        'source_sha256': hashlib.sha256(page_body).hexdigest(),
        'versions': versions,
    }
    validate_observation(observation)
    days[stamp[:10]] = observation
    result = package_metadata(stamp, first)
    result['days'] = dict(sorted(days.items()))
    return result, observation


def delta(current, previous):
    return current - previous if previous is not None else None


def render(history):
    days = list(history['days'].items())
    latest_day, latest = days[-1]
    previous = days[-2][1] if len(days) > 1 else None
    lines = [
        '# GHCR package download archive', '',
        f'Package: [`{history["package"]}`]({PACKAGE_URL}).',
        f'Published: **{PACKAGE_PUBLISHED_DATE}**. First captured: **{history["first_collected_at"]}**.',
        f'Latest capture: **{latest["collected_at"]}**.', '',
        '| Metric | Latest reported | Change from previous archived day |',
        '| --- | ---: | ---: |',
    ]
    total_delta = delta(latest['total_downloads'], previous['total_downloads'] if previous else None)
    total_change = f'{total_delta:+,}' if total_delta is not None else '—'
    lines.append(f'| Package total downloads | {latest["total_downloads"]:,} | '
                 f'{total_change} |')
    version_ids = sorted(latest['versions'], key=int)
    for version_id in version_ids:
        version = latest['versions'][version_id]
        old = previous['versions'].get(version_id, {}).get('downloads') if previous else None
        change = delta(version['downloads'], old) if version['available'] else None
        tags = ', '.join(f'`{html.escape(tag)}`' for tag in version['tags']) or '(untagged)'
        availability = ('' if version['available'] else
                        f' (last observed {version["last_observed_at"]}; unavailable)')
        change_text = f'{change:+,}' if change is not None else '—'
        lines.append(f'| Version {tags}{availability} | {version["downloads"]:,} | '
                     f'{change_text} |')
    lines += [
        '',
        'These are GitHub package download counters, not unique users, successful installations, '
        'or repository clones. Container clients may fetch an index, platform manifests, and layers; '
        'builds and tests can also contribute downloads.', '',
        '## Daily observations', '',
        '| UTC date | Collected at | Package downloads | Daily change |',
        '| --- | --- | ---: | ---: |',
    ]
    prior = None
    for day, observation in days:
        change = delta(observation['total_downloads'], prior)
        change_text = f'{change:+,}' if change is not None else '—'
        lines.append(f'| {day} | {observation["collected_at"]} | '
                     f'{observation["total_downloads"]:,} | '
                     f'{change_text} |')
        prior = observation['total_downloads']
    lines += [
        '',
        'GitHub exposes these counters on the public package pages. Its documented Packages REST '
        'responses provide package/version metadata but do not include download counters, so this '
        'collector validates and archives the public HTML representation. A markup change fails the '
        'package step loudly; the preceding repository-traffic step remains safely committed.', '',
    ]
    return '\n'.join(lines)


def read_history(api):
    head = api.request(f'git/ref/heads/{BRANCH}')['object']['sha']
    commit = api.request(f'git/commits/{head}')
    tree_sha = commit['tree']['sha']
    tree = api.request(f'git/trees/{tree_sha}')
    require(not tree.get('truncated'), 'Truncated history root tree')
    roots = {entry['path']: entry for entry in tree['tree']}
    history = None
    if 'package-downloads.json' in roots:
        blob = api.request('git/blobs/' + roots['package-downloads.json']['sha'])
        require(blob['encoding'] == 'base64', 'Unsupported package history blob encoding')
        import base64
        history = json.loads(base64.b64decode(blob['content'], validate=False))
        validate_history(history)
    return head, tree_sha, history


def collect(index_body, prior_history=None, fetch=fetch_page):
    index = parse_index(index_body)
    known = set(index['versions'])
    if prior_history:
        latest = prior_history['days'][max(prior_history['days'])]
        # Build/cache tags are transient and remain in their historical
        # observations. Only immutable semver release IDs need perpetual
        # refresh after they leave GitHub's recent-version list.
        known.update(version_id for version_id, value in latest['versions'].items()
                     if any(re.fullmatch(r'v[0-9]+(?:\.[0-9]+)+', tag)
                            for tag in value['tags']))
    versions = {}
    for version_id in sorted(known, key=int):
        page = fetch(f'{PACKAGE_VERSION_BASE_URL}/{version_id}')
        if page is None:
            previous = None
            if prior_history:
                previous = prior_history['days'][max(prior_history['days'])]['versions'].get(version_id)
            if previous is not None:
                version = copy.deepcopy(previous)
                version['available'] = False
            else:
                # The index itself is an authoritative observation. A transient
                # build tag can disappear between the index and detail requests;
                # preserve the counter we actually received instead of inventing
                # zero or blocking the package total.
                version = copy.deepcopy(index['versions'][version_id])
                version['available'] = False
        else:
            version = parse_version(page, version_id)
        if version['available']:
            require(version['downloads'] <= index['total_downloads'],
                    f'Version downloads exceed package total: {version_id}')
        versions[version_id] = version
    return {'total_downloads': index['total_downloads'], 'versions': versions}


def archive(api, index_body, stamp, seed=False):
    head, tree_sha, previous = read_history(api)
    require(not seed or previous is None, 'Package seed is only allowed before package history exists')
    require(seed or previous is not None,
            'Package history must be initialized from the preserved first baseline')
    # A seed must represent the preserved page at its capture time exactly.
    # Do not mix it with later version-detail requests.
    collected = parse_index(index_body) if seed else collect(index_body, previous)
    history, observation = merge(previous, collected, index_body, stamp)
    run_id = os.environ.get('GITHUB_RUN_ID', 'package-seed')
    attempt = os.environ.get('GITHUB_RUN_ATTEMPT', '1')
    snapshot_path = (f'package-snapshots/{stamp[:10]}/'
                     f'{stamp.replace(":", "")}-{run_id}-{attempt}.json')
    snapshot = package_metadata(stamp, history['first_collected_at'])
    snapshot.update(observation)
    files = {
        'package-downloads.json': encode(history),
        'PACKAGE-DOWNLOADS.md': render(history),
        snapshot_path: encode(snapshot),
    }
    if seed:
        files['raw/packages/first-run/package.html'] = index_body.decode('utf-8')
        files['raw/packages/first-run/metadata.json'] = encode({
            **package_metadata(stamp, stamp),
            'source_url': PACKAGE_URL,
            'sha256': hashlib.sha256(index_body).hexdigest(),
            'total_downloads': observation['total_downloads'],
            'versions': observation['versions'],
            'historical_counters_before_first_collection_available': False,
        })
    sha = publish(api, head, tree_sha, files, f'Archive package downloads collected {stamp}')
    print(f'Archived package total {observation["total_downloads"]}; '
          f'{len(observation["versions"])} versions; history commit {sha}')
    if previous:
        old = previous['days'][max(previous['days'])]
        if observation['total_downloads'] < old['total_downloads']:
            print('::warning::Package total download counter decreased')
    return sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-page', type=Path)
    parser.add_argument('--collected-at')
    args = parser.parse_args()
    if args.seed_page:
        require(args.collected_at is not None, '--collected-at is required with --seed-page')
        stamp = args.collected_at
        require(re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', stamp),
                'Invalid --collected-at timestamp')
        token = subprocess.run(['gh', 'auth', 'token'], capture_output=True,
                               check=True).stdout.decode().strip()
        archive(API(token), args.seed_page.read_bytes(), stamp, seed=True)
        return
    require(os.environ.get('GITHUB_REPOSITORY') == REPOSITORY, 'Unexpected repository')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    archive(API(os.environ.get('GITHUB_TOKEN')), fetch_page(PACKAGE_URL), stamp)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'::error::{type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
