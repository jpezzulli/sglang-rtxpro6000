"""Focused archive integrity checks; no network calls or repository checkout."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('archive', Path(__file__).with_name('archive_traffic.py'))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def sample(start='2026-08-24', end='2026-08-26', count=10):
    raw = {'referrers': [], 'paths': []}
    for kind in ('views', 'clones'):
        rows = [{'timestamp': d + 'T00:00:00Z', 'count': count, 'uniques': min(count, 3)}
                for d in a.dates(start, end)]
        raw[kind] = {'count': sum(r['count'] for r in rows), 'uniques': min(count, 3), kind: rows}
    return raw


class ArchiveTests(unittest.TestCase):
    def test_overlap_corrections_preserve_older_days_and_uniques(self):
        old = a.merge(None, sample(), '2026-08-27T01:00:00Z')
        untouched = copy.deepcopy(old)
        new = a.merge(old, sample('2026-08-25', '2026-08-27', 2), '2026-08-28T01:00:00Z')
        self.assertEqual(new['days']['2026-08-24']['views']['count'], 10)
        self.assertEqual(new['days']['2026-08-25']['views'], {'count': 2, 'uniques': 2})
        self.assertEqual(len(new['days']), 4)
        self.assertEqual(old, untouched)

    def test_missing_launch_dates_are_not_invented(self):
        raw = sample('2026-08-26', '2026-08-28')
        h = a.merge(None, raw, '2026-08-29T00:17:00Z')
        self.assertEqual(h['coverage']['missing_dates_through_latest_exposed'], ['2026-08-24', '2026-08-25'])
        self.assertEqual(h['coverage']['not_yet_exposed_dates'], ['2026-08-29'])
        self.assertNotIn('Exact cumulative', a.render(h, raw, 'snapshot.json'))

    def test_bad_responses_fail_without_mutating_history(self):
        old = a.merge(None, sample(), '2026-08-27T01:00:00Z')
        original = copy.deepcopy(old)
        cases = []
        r = sample(); r['views']['views'] = []; cases.append(r)
        r = sample(); r['views']['count'] = 999; cases.append(r)
        cases.append(sample(count=0))
        r = sample(); r['views']['views'][0]['count'] = -1; cases.append(r)
        r = sample(); r['views']['views'][0]['uniques'] = True; cases.append(r)
        r = sample(); r['views']['views'][1]['timestamp'] = r['views']['views'][0]['timestamp']; cases.append(r)
        r = sample(); r['paths'] = [{'path': '/x', 'count': 1, 'uniques': 1}]; cases.append(r)
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(a.ArchiveError):
                a.merge(old, raw, '2026-08-28T01:00:00Z')
            self.assertEqual(old, original)

    def test_stale_collection_fails(self):
        old = a.merge(None, sample(), '2026-08-27T01:00:00Z')
        with self.assertRaises(a.ArchiveError):
            a.merge(old, sample(), '2026-08-26T01:00:00Z')

    def test_daily_zero_is_retained_and_uniques_label_is_honest(self):
        raw = sample(count=0)
        h = a.merge(None, raw, '2026-08-27T01:00:00Z')
        summary = a.render(h, raw, 'snapshot.json')
        self.assertEqual(h['days']['2026-08-24']['views']['count'], 0)
        self.assertIn('sum_of_daily_uniques', summary)
        self.assertNotIn('lifetime_unique_users', summary)
        self.assertIn('Exact cumulative views', summary)

    def test_branch_race_fails_before_ref_update(self):
        class Fake:
            calls = []
            def request(self, path, method='GET', data=None):
                self.calls.append((path, method, data))
                if path == 'git/trees': return {'sha': 'tree'}
                if path == 'git/commits': return {'sha': 'newcommit'}
                return {'object': {'sha': 'otherwriter'}}
        api = Fake()
        with self.assertRaises(a.ArchiveError):
            a.publish(api, 'original', 'basetree', {'daily.json': '{}'}, 'test')
        self.assertFalse(any(call[1] == 'PATCH' for call in api.calls))

    def test_snapshot_paths_and_raw_bytes_survive_updates(self):
        raw = sample()
        bodies = {k: json.dumps(v).encode() for k, v in raw.items()}
        captured = []
        original_read, original_publish = a.read_history, a.publish
        a.read_history = lambda *args, **kwargs: (None, None, None)
        a.publish = lambda api, head, tree, files, message: captured.append(files) or 'sha'
        try:
            a.archive(None, raw, bodies, '2026-08-27T00:17:00Z', 'test', seed=True)
            for key, value in bodies.items():
                self.assertEqual(captured[0][f'raw/first-run/{key}.json'].encode(), value)
            self.assertEqual(len([p for p in captured[0] if p.startswith('snapshots/')]), 1)
        finally:
            a.read_history, a.publish = original_read, original_publish


if __name__ == '__main__':
    unittest.main()
