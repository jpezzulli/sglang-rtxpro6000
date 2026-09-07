import datetime as dt
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('watchdog', Path(__file__).with_name('traffic_watchdog.py'))
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
T = dt.datetime(2026, 9, 7, 2, 0, tzinfo=dt.timezone.utc)


class FakeGitHub:
    def __init__(self, collected='2026-09-06T01:35:52+00:00', status='completed', conclusion='success'):
        self.collected = collected
        self.status = status
        self.conclusion = conclusion
        self.dispatch_count = 0
        self.active = []
        self.latest = []
        self.update = True
        self.dispatch_failure = False

    def archive(self, branch):
        return {'head': 'history', 'collected_at': self.collected, 'through_date': '2026-09-05'}

    def runs(self):
        return self.active + self.latest

    def dispatch(self):
        self.dispatch_count += 1
        if self.dispatch_failure:
            raise RuntimeError('connection failed')
        return 123

    def run(self, run_id):
        if self.status == 'completed' and self.conclusion == 'success' and self.update:
            self.collected = T.isoformat()
        return {'id': run_id, 'status': self.status, 'conclusion': self.conclusion,
                'html_url': 'https://github.com/test/repo/actions/runs/' + str(run_id)}


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        self.clock = patch.object(w, 'now', return_value=T)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def verified(self):
        w.write_json(self.state / 'verified.json', {'run_id': 1})

    def test_deadline_uses_utc_calendar_day(self):
        self.assertEqual(w.required_cutoff(T.replace(hour=0, minute=16)).isoformat(), '2026-09-06T00:17:00+00:00')
        self.assertEqual(w.required_cutoff(T.replace(hour=0, minute=17)).isoformat(), '2026-09-07T00:17:00+00:00')

    def test_stale_archive_dispatches_and_verifies(self):
        self.verified()
        g = FakeGitHub()
        w.reconcile(g, self.state, 'traffic-history', wait_seconds=0)
        self.assertEqual(g.dispatch_count, 1)
        self.assertEqual(w.load_json(self.state / 'verified.json')['run_id'], 123)
        self.assertFalse((self.state / 'pending.json').exists())

    def test_fresh_archive_is_noop_after_commissioning(self):
        self.verified()
        g = FakeGitHub(T.isoformat())
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 0)

    def test_same_day_pre_deadline_capture_does_not_suppress_due_run(self):
        self.verified()
        g = FakeGitHub('2026-09-07T00:05:00+00:00')
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 1)

    def test_old_adopted_run_cannot_verify_stale_archive(self):
        g = FakeGitHub();g.update = False
        g.active = [{'id': 77, 'created_at': '2026-09-06T00:00:00Z', 'status': 'in_progress'}]
        with self.assertRaises(RuntimeError):
            w.reconcile(g, self.state, 'traffic-history')
        self.assertFalse((self.state / 'verified.json').exists())

    def test_first_timer_commissions_even_if_capture_fresh(self):
        g = FakeGitHub(T.isoformat())
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 1)

    def test_active_workflow_adopted_without_duplicate_dispatch(self):
        g = FakeGitHub()
        g.active = [{'id': 77, 'created_at': T.isoformat(), 'status': 'in_progress'}]
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 0)
        self.assertEqual(w.load_json(self.state / 'verified.json')['run_id'], 77)

    def test_timeout_retains_pending_and_resumes_same_run(self):
        g = FakeGitHub(status='queued')
        with self.assertRaises(RuntimeError):
            w.reconcile(g, self.state, 'traffic-history', wait_seconds=0)
        self.assertEqual(w.load_json(self.state / 'pending.json')['run_id'], 123)
        g.status = 'completed'
        w.reconcile(g, self.state, 'traffic-history', wait_seconds=0)
        self.assertEqual(g.dispatch_count, 1)

    def test_failure_preserves_verified_state_and_retries_next_tick(self):
        self.verified()
        g = FakeGitHub(conclusion='failure')
        with self.assertRaises(RuntimeError):
            w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(w.load_json(self.state / 'verified.json')['run_id'], 1)
        self.assertFalse((self.state / 'pending.json').exists())
        g.conclusion = 'success'
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 2)

    def test_success_without_archive_update_is_failure(self):
        g = FakeGitHub();g.update = False
        with self.assertRaises(RuntimeError):
            w.reconcile(g, self.state, 'traffic-history')
        self.assertFalse((self.state / 'verified.json').exists())

    def test_unknown_dispatch_persists_intent_and_adopts_registered_run(self):
        g = FakeGitHub();g.dispatch_failure = True
        with self.assertRaises(RuntimeError):
            w.reconcile(g, self.state, 'traffic-history')
        self.assertIsNone(w.load_json(self.state / 'pending.json')['run_id'])
        g.latest = [{'id': 88, 'status': 'completed', 'event': 'workflow_dispatch', 'created_at': T.isoformat()}]
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 1)
        self.assertEqual(w.load_json(self.state / 'verified.json')['run_id'], 88)

    def test_deleted_pending_run_retires_reference_and_recovers(self):
        g = FakeGitHub()
        w.write_json(self.state / 'pending.json', {'run_id': 99, 'requested_at': T.isoformat()})
        with patch.object(g, 'run', side_effect=w.MissingRunResource('404')):
            with self.assertRaises(RuntimeError):w.reconcile(g, self.state, 'traffic-history')
        self.assertFalse((self.state / 'pending.json').exists())
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 1)

    def test_lookup_authentication_failure_keeps_pending(self):
        g = FakeGitHub()
        w.write_json(self.state / 'pending.json', {'run_id': 99, 'requested_at': T.isoformat()})
        with patch.object(g, 'run', side_effect=RuntimeError('HTTP 403')):
            with self.assertRaises(RuntimeError):w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(w.load_json(self.state / 'pending.json')['run_id'], 99)
        self.assertEqual(g.dispatch_count, 0)

    def test_404_without_accessible_run_listing_keeps_pending(self):
        g = FakeGitHub()
        w.write_json(self.state / 'pending.json', {'run_id': 99, 'requested_at': T.isoformat()})
        with patch.object(g, 'run', side_effect=w.MissingRunResource('404')), patch.object(g, 'runs', side_effect=RuntimeError('HTTP 403')):
            with self.assertRaises(RuntimeError):w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(w.load_json(self.state / 'pending.json')['run_id'], 99)

    def test_api_distinguishes_not_found_from_auth_errors(self):
        from types import SimpleNamespace
        g = w.GitHub('test/repo', 'archive-traffic.yml')
        for status, error in [('404', w.MissingRunResource), ('403', RuntimeError), ('401', RuntimeError)]:
            result = SimpleNamespace(returncode=1, stdout='HTTP/2.0 ' + status + ' Error\n\n{}')
            with patch.object(g, 'command', return_value=result), self.assertRaises(error):
                g.api('actions/runs/99')
        result = SimpleNamespace(returncode=0, stdout='HTTP/2.0 200 OK\nX-Example: value\n\n{"id": 99}')
        with patch.object(g, 'command', return_value=result):
            self.assertEqual(g.api('actions/runs/99'), {'id': 99})

    def test_unknown_dispatch_does_not_retry_immediately(self):
        g = FakeGitHub();g.dispatch_failure = True
        with self.assertRaises(RuntimeError):w.reconcile(g, self.state, 'traffic-history')
        with self.assertRaises(RuntimeError):w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 1)

    def test_unknown_dispatch_expires_after_registration_grace(self):
        g = FakeGitHub()
        w.write_json(self.state / 'pending.json', {'run_id': None, 'requested_at': (T-dt.timedelta(minutes=20)).isoformat()})
        with self.assertRaises(RuntimeError):w.reconcile(g, self.state, 'traffic-history')
        self.assertFalse((self.state / 'pending.json').exists())
        self.assertEqual(g.dispatch_count, 0)


if __name__ == '__main__':
    unittest.main()
