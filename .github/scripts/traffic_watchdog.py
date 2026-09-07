#!/usr/bin/env python3
"""Ensure daily traffic collection using an independent persistent scheduler.

Only reads GitHub APIs and dispatches the existing workflow. Archive writes and
concurrency remain owned by that workflow; this process never clones a repo.
"""
import argparse
import base64
import datetime as dt
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import quote

UTC = dt.timezone.utc


def now():
    return dt.datetime.now(UTC)


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise RuntimeError('Timestamp must include timezone')
    return parsed.astimezone(UTC)


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def required_cutoff(current):
    # A capture before 00:17 must not suppress the collection due at 00:17.
    cutoff = current.replace(hour=0, minute=17, second=0, microsecond=0)
    return cutoff if current >= cutoff else cutoff - dt.timedelta(days=1)


class MissingRunResource(RuntimeError):
    pass


class GitHub:
    def __init__(self, repository, workflow):
        self.repository = repository
        self.workflow = workflow

    def command(self, args, result_only=False):
        result = subprocess.run(['/usr/bin/gh', *args], capture_output=True,
                                text=True, timeout=60, check=False)
        if result_only:
            return result
        if result.returncode:
            # Do not print credential-helper diagnostics or command environment.
            raise RuntimeError(f'GitHub CLI failed (exit {result.returncode}) for {args[0]}')
        return result.stdout

    def api(self, path):
        result = self.command(['api', '--include', '-H', 'X-GitHub-Api-Version: 2022-11-28',
                               f'repos/{self.repository}/{path}'.rstrip('/')], result_only=True)
        header, separator, body = result.stdout.partition('\n\n')
        status = re.match(r'HTTP/\S+ (\d{3})', header)
        if result.returncode:
            if status and status.group(1) == '404':
                raise MissingRunResource(f'GitHub API resource not found: {path}')
            raise RuntimeError(f'GitHub API failed: HTTP {status.group(1) if status else "unknown"}; {path}')
        if not separator or not status or status.group(1) != '200':
            raise RuntimeError(f'Malformed GitHub API response: {path}')
        return json.loads(body)

    def archive(self, branch):
        head = self.api('git/ref/heads/' + quote(branch, safe=''))['object']['sha']
        # Use Git blobs so a growing lifetime archive does not hit the Contents
        # API's one-megabyte inline-content limit. Pin all reads to one commit.
        tree = self.api('git/trees/' + head)
        if tree.get('truncated'):
            raise RuntimeError('Truncated history tree')
        entries = [x for x in tree['tree'] if x['path'] == 'daily.json' and x['type'] == 'blob']
        if len(entries) != 1:
            raise RuntimeError('Canonical daily archive missing')
        response = self.api('git/blobs/' + entries[0]['sha'])
        if response.get('encoding') != 'base64' or not response.get('content'):
            raise RuntimeError('Missing or malformed canonical archive')
        data = json.loads(base64.b64decode(response['content']))
        if (data.get('repository') != self.repository or data.get('schema_version') != 1
                or not isinstance(data.get('days'), dict) or not data['days']):
            raise RuntimeError('Canonical archive identity/schema/days invalid')
        collected = timestamp(data['collected_at'])
        if collected > now() + dt.timedelta(minutes=5):
            raise RuntimeError('Archive collection timestamp is in the future')
        return {'head': head, 'collected_at': collected.isoformat(),
                'through_date': data['coverage']['through_date']}

    def runs(self):
        return self.api('actions/workflows/' + quote(self.workflow, safe='') +
                        '/runs?per_page=30')['workflow_runs']

    def run(self, run_id):
        return self.api(f'actions/runs/{run_id}')

    def dispatch(self):
        branch = self.api('')['default_branch']
        output = self.command(['workflow', 'run', self.workflow, '--repo', self.repository,
                               '--ref', branch])
        match = re.search(re.escape(f'https://github.com/{self.repository}/actions/runs/') + r'(\d+)', output)
        return int(match.group(1)) if match else None


def reconcile(github, state_dir, branch, wait_seconds=480, poll_seconds=15):
    pending_path = state_dir / 'pending.json'
    verified_path = state_dir / 'verified.json'
    archive = github.archive(branch)
    pending = load_json(pending_path)
    verified = load_json(verified_path)
    if pending is None and verified is not None and timestamp(archive['collected_at']) >= required_cutoff(now()):
        print(f'Current: archive collected {archive["collected_at"]}; through {archive["through_date"]}', flush=True)
        return

    if pending is None:
        # Commissioning dispatches once even if a manual capture is already fresh,
        # proving that the timer's noninteractive credentials can execute the path.
        # An existing queued/running collector is adopted to avoid duplicate work.
        active = [r for r in github.runs() if r['status'] != 'completed']
        if active:
            run = max(active, key=lambda r: r['id'])
            pending = {'requested_at': run['created_at'], 'run_id': run['id']}
        else:
            pending = {'requested_at': now().replace(microsecond=0).isoformat(), 'run_id': None}
            # Persist intent before dispatch: an interrupted/lost HTTP response
            # must not cause a second collector to be submitted blindly.
            write_json(pending_path, pending)
            pending['run_id'] = github.dispatch()
            print(f'Dispatched traffic collector: run {pending["run_id"]}', flush=True)
        write_json(pending_path, pending)

    if pending['run_id'] is None:
        candidates = [r for r in github.runs() if r['event'] == 'workflow_dispatch'
                      and timestamp(r['created_at']) >= timestamp(pending['requested_at'])]
        if not candidates:
            # Wait for GitHub's run listing to settle before allowing a new attempt.
            # Clearing intent after 15 minutes bounds recovery from a failed dispatch.
            if now() - timestamp(pending['requested_at']) > dt.timedelta(minutes=15):
                pending_path.unlink()
            raise RuntimeError('Dispatch outcome unknown; waiting for run registration before retry')
        pending['run_id'] = max(candidates, key=lambda r: r['id'])['id']
        write_json(pending_path, pending)

    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            run = github.run(pending['run_id'])
        except MissingRunResource:
            # Only retire a 404 after a successful Actions listing confirms the
            # run is absent. Authentication/transport failures preserve intent.
            listed = github.runs()
            if any(r['id'] == pending['run_id'] for r in listed):
                raise RuntimeError('Pending run lookup and Actions listing disagree; retry later')
            pending_path.unlink()
            raise RuntimeError('Pending workflow run no longer exists; retry on next timer tick') from None
        if run['status'] == 'completed':
            if run['conclusion'] != 'success':
                pending_path.unlink()
                raise RuntimeError(f'Traffic run {run["id"]} ended {run["conclusion"]}; retry on next timer tick')
            archive = github.archive(branch)
            if timestamp(archive['collected_at']) < max(timestamp(pending['requested_at']), required_cutoff(now())):
                pending_path.unlink()
                raise RuntimeError(f'Traffic run {run["id"]} succeeded without a fresh archive')
            write_json(verified_path, {'verified_at': now().isoformat(), 'run_id': run['id'],
                                      'run_url': run['html_url'], 'archive': archive})
            pending_path.unlink()
            print(f'Verified automatic collection: {run["html_url"]}; '
                  f'history {archive["head"]}; collected {archive["collected_at"]}; '
                  f'through {archive["through_date"]}', flush=True)
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Traffic run {run["id"]} still {run["status"]}; retained for next timer tick')
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', required=True)
    parser.add_argument('--workflow', default='archive-traffic.yml')
    parser.add_argument('--history-branch', default='traffic-history')
    parser.add_argument('--state-dir', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repository):
        raise RuntimeError('Invalid owner/repository')
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / 'watchdog.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another traffic watchdog invocation is active', flush=True)
            return
        reconcile(GitHub(args.repository, args.workflow), args.state_dir, args.history_branch)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'TRAFFIC ARCHIVE ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
