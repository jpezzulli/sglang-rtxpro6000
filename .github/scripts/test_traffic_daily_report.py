"""Focused daily-email report tests; no network and no mail delivery."""
import importlib.util
import datetime as dt
from pathlib import Path
import unittest
from unittest.mock import patch

scripts = Path(__file__).parent
spec = importlib.util.spec_from_file_location('report', scripts / 'traffic_daily_report.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class FakeGitHub:
    repository = 'jpezzulli/sglang-rtxpro6000'
    def __init__(self):
        self.blobs = {
            'daily': {'repository': self.repository, 'schema_version': 1,
                      'collected_at': '2026-09-21T01:00:00Z',
                      'launch_date': '2026-09-20',
                      'coverage': {'complete_through_latest_exposed': True,
                                   'missing_dates_through_latest_exposed': [],
                                   'through_date': '2026-09-20'},
                      'days': {'2026-09-20': {'views': {'count': 10, 'uniques': 4},
                                              'clones': {'count': 8, 'uniques': 3}}}},
            'package': {'repository': self.repository, 'schema_version': 1,
                        'collected_at': '2026-09-21T01:00:00Z',
                        'days': {'2026-09-19': {'total_downloads': 40},
                                 '2026-09-20': {'total_downloads': 42}}},
            'metrics': {'repository': self.repository, 'schema_version': 1,
                        'collected_at': '2026-09-21T01:00:00Z',
                        'days': {'2026-09-19': {'stars': 8, 'forks': 2},
                                 '2026-09-20': {'stars': 9, 'forks': 2}}},
            'prior_snapshot': {'traffic': {
                'referrers': [{'referrer': 'reddit.com', 'count': 3, 'uniques': 2}],
                'paths': [{'path': '/jpezzulli/sglang-rtxpro6000', 'count': 4, 'uniques': 3}],
            }},
        }
    def api(self, path):
        if path.startswith('git/ref/heads/'):
            return {'object': {'sha': 'head'}}
        if path == 'git/trees/head':
            return {'truncated': False, 'tree': [
                {'path': 'daily.json', 'type': 'blob', 'sha': 'daily'},
                {'path': 'package-downloads.json', 'type': 'blob', 'sha': 'package'},
                {'path': 'repository-metrics.json', 'type': 'blob', 'sha': 'metrics'},
            ]}
        if path == 'git/trees/head?recursive=1':
            return {'truncated': False, 'tree': [
                {'path': 'snapshots/2026-09-20/prior.json', 'type': 'blob', 'sha': 'prior_snapshot'},
                {'path': 'snapshots/2026-09-21/latest.json', 'type': 'blob', 'sha': 'latest_snapshot'},
            ]}
        if path.startswith('git/blobs/'):
            import base64, json
            return {'encoding': 'base64', 'content': base64.b64encode(
                json.dumps(self.blobs[path.rsplit('/', 1)[1]]).encode()).decode()}
        if path == 'traffic/views?per=day':
            return {'count': 15, 'uniques': 6, 'views': [
                {'timestamp': '2026-09-20T00:00:00Z', 'count': 10, 'uniques': 4},
                {'timestamp': '2026-09-21T00:00:00Z', 'count': 5, 'uniques': 3}]}
        if path == 'traffic/clones?per=day':
            return {'count': 10, 'uniques': 4, 'clones': [
                {'timestamp': '2026-09-20T00:00:00Z', 'count': 8, 'uniques': 3},
                {'timestamp': '2026-09-21T00:00:00Z', 'count': 2, 'uniques': 2}]}
        if path == 'traffic/popular/referrers?per=day':
            return [{'referrer': 'reddit.com', 'count': 5, 'uniques': 4}]
        if path == 'traffic/popular/paths?per=day':
            return [{'path': '/jpezzulli/sglang-rtxpro6000', 'count': 6, 'uniques': 5}]
        raise AssertionError(path)


class DailyReportTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch.object(r, 'now', return_value=dt.datetime(
            2026, 9, 21, 10, tzinfo=dt.timezone.utc))
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def test_report_merges_live_window_and_labels_uniques_honestly(self):
        body, values = r.report(FakeGitHub(), 'traffic-history')
        self.assertEqual(values, {'stars': 9, 'forks': 2, 'views': 15, 'clones': 10,
                                  'daily_unique_cloners': 5, 'package_downloads': 42})
        self.assertIn('Stars: 9 (+1)', body)
        self.assertIn('Clones: 10 (+2 on 2026-09-21)', body)
        self.assertIn('sum_of_daily_unique_cloners: 5 (+2 on 2026-09-21)', body)
        self.assertIn('does not expose a deduplicated lifetime cloner count', body)
        self.assertIn('reddit.com: 5 views / 4 unique visitors (+2 views, +2 uniques)', body)

    def test_first_report_uses_archived_daily_delta(self):
        body, _ = r.report(FakeGitHub(), 'traffic-history')
        self.assertIn('Stars: 9 (+1)', body)
        self.assertIn('Container downloads: 42 (+2)', body)
        self.assertIn('Views: 15 (+5 on 2026-09-21)', body)
        self.assertIn('Clones: 10 (+2 on 2026-09-21)', body)

    def test_incomplete_archive_fails_without_report(self):
        github = FakeGitHub()
        github.blobs['daily']['coverage']['complete_through_latest_exposed'] = False
        with self.assertRaises(RuntimeError):
            r.report(github, 'traffic-history')

    def test_stale_archive_fails_without_report(self):
        github = FakeGitHub()
        github.blobs['package']['collected_at'] = '2026-09-20T23:59:59Z'
        with self.assertRaises(RuntimeError):
            r.report(github, 'traffic-history')

    def test_pre_cutoff_capture_fails_without_report(self):
        github = FakeGitHub()
        github.blobs['daily']['collected_at'] = '2026-09-21T00:05:00Z'
        with self.assertRaises(RuntimeError):
            r.report(github, 'traffic-history')

    def test_missing_history_date_fails_without_report(self):
        github = FakeGitHub()
        github.blobs['daily']['coverage']['through_date'] = '2026-09-21'
        with self.assertRaises(RuntimeError):
            r.report(github, 'traffic-history')

    def test_gap_between_archive_and_live_window_fails_without_report(self):
        traffic = FakeGitHub().blobs['daily']
        rolling = {'views': {'total': {'count': 1, 'uniques': 1},
                             'days': {'2026-09-22': {'count': 1, 'uniques': 1}}},
                   'clones': {'total': {'count': 1, 'uniques': 1},
                              'days': {'2026-09-22': {'count': 1, 'uniques': 1}}}}
        with self.assertRaises(RuntimeError):
            r.current_days(traffic, rolling)

    def test_empty_live_overlap_cannot_erase_archived_traffic(self):
        traffic = FakeGitHub().blobs['daily']
        rolling = {'views': {'total': {'count': 0, 'uniques': 0},
                             'days': {'2026-09-20': {'count': 0, 'uniques': 0}}},
                   'clones': {'total': {'count': 8, 'uniques': 3},
                              'days': {'2026-09-20': {'count': 8, 'uniques': 3}}}}
        with self.assertRaises(RuntimeError):
            r.current_days(traffic, rolling)

    def test_malformed_previous_snapshot_fails_without_report(self):
        github = FakeGitHub()
        del github.blobs['prior_snapshot']['traffic']['paths']
        with self.assertRaises(RuntimeError):
            r.report(github, 'traffic-history')



if __name__ == '__main__':
    unittest.main()
