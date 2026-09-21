"""Focused GHCR download archive tests; no network calls."""
import importlib.util
import json
from pathlib import Path
import unittest

scripts = Path(__file__).parent
spec = importlib.util.spec_from_file_location('package_archive', scripts / 'archive_package_downloads.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def index(total=189, version_downloads=78):
    return f'''ghcr.io/jpezzulli/sglang-rtxpro6000
<h3 class="f5">Recent tagged image versions</h3>
<li data-view-component="true" class="Box-row">
<a href="/users/jpezzulli/packages/container/sglang-rtxpro6000/1249529601?tag=v2.5.0">v2.5.0</a>
<a href="/users/jpezzulli/packages/container/sglang-rtxpro6000/1249529601?tag=build-cache">build-cache</a>
<span value="sha256:{'a' * 64}"></span>
 {version_downloads}<span class="sr-only">Version downloads</span>
</li><div data-view-component="true" class="Box-footer">
<span>Total downloads</span><h3 title="{total}">{total}</h3>'''.encode()


def version(downloads=78):
    return f'''<a href="/users/jpezzulli/packages/container/sglang-rtxpro6000/1249529601?tag=v2.5.0">v2.5.0</a>
<a href="/users/jpezzulli/packages/container/sglang-rtxpro6000/1249529601?tag=build-cache">build-cache</a>
<span value="FROM ghcr.io/jpezzulli/sglang-rtxpro6000@sha256:{'a' * 64}"></span>
<span>Total downloads</span><span class="flex-auto text-right text-bold">{downloads}</span>'''.encode()


class PackageArchiveTests(unittest.TestCase):
    def test_parses_package_and_version_pages(self):
        parsed = p.parse_index(index())
        self.assertEqual(parsed['total_downloads'], 189)
        self.assertEqual(parsed['versions']['1249529601']['downloads'], 78)
        detail = p.parse_version(version(), '1249529601')
        self.assertEqual(detail['tags'], ['build-cache', 'v2.5.0'])

    def test_malformed_or_wrong_package_fails(self):
        for body in [b'', index().replace(b'189</h3>', b'188</h3>'),
                     index().replace(b'jpezzulli', b'someoneelse'),
                     index().replace(b'title="189">189', b'title="1,,89">1,,89')]:
            with self.subTest(body=body), self.assertRaises(p.ArchiveError):
                p.parse_index(body)

    def test_collect_refreshes_known_versions_by_stable_id(self):
        calls = []
        def fetch(url):
            calls.append(url)
            return version(81)
        collected = p.collect(index(200, 80), fetch=fetch)
        self.assertEqual(collected['versions']['1249529601']['downloads'], 81)
        self.assertEqual(calls, [p.PACKAGE_VERSION_BASE_URL + '/1249529601'])

    def test_missing_old_version_retains_last_count_and_marks_unavailable(self):
        collected = p.collect(index(), fetch=lambda url: version())
        history, _ = p.merge(None, collected, index(), '2026-09-16T12:21:06Z')
        current = p.collect(index(200, 80), history, fetch=lambda url: None)
        self.assertEqual(current['versions']['1249529601']['downloads'], 78)
        self.assertFalse(current['versions']['1249529601']['available'])

    def test_unavailable_stale_version_can_exceed_corrected_package_total(self):
        collected = p.collect(index(), fetch=lambda url: version())
        history, _ = p.merge(None, collected, index(), '2026-09-16T12:21:06Z')
        current = p.collect(index(70, 60), history, fetch=lambda url: None)
        self.assertEqual(current['total_downloads'], 70)
        self.assertEqual(current['versions']['1249529601']['downloads'], 78)

    def test_new_version_disappearing_after_index_keeps_observed_counter(self):
        current = p.collect(index(), fetch=lambda url: None)
        self.assertEqual(current['versions']['1249529601']['downloads'], 78)
        self.assertFalse(current['versions']['1249529601']['available'])

    def test_daily_merge_replaces_same_day_and_preserves_older_days(self):
        collected = p.collect(index(), fetch=lambda url: version())
        history, _ = p.merge(None, collected, index(), '2026-09-16T12:21:06Z')
        revised = p.collect(index(195, 82), history, fetch=lambda url: version(82))
        history, _ = p.merge(history, revised, index(195, 82), '2026-09-16T14:00:00Z')
        self.assertEqual(len(history['days']), 1)
        self.assertEqual(history['days']['2026-09-16']['total_downloads'], 195)
        later = p.collect(index(210, 90), history, fetch=lambda url: version(90))
        history, _ = p.merge(history, later, index(210, 90), '2026-09-17T00:17:00Z')
        self.assertEqual(len(history['days']), 2)
        self.assertEqual(history['first_collected_at'], '2026-09-16T12:21:06Z')

    def test_merge_owns_nested_values_and_two_day_summary_is_valid(self):
        collected = p.collect(index(), fetch=lambda url: version())
        history, _ = p.merge(None, collected, index(), '2026-09-16T12:21:06Z')
        collected['versions']['1249529601']['downloads'] = 82
        collected['total_downloads'] = 200
        self.assertEqual(
            history['days']['2026-09-16']['versions']['1249529601']['downloads'], 78)
        history, _ = p.merge(history, collected, index(200, 82), '2026-09-17T00:17:00Z')
        summary = p.render(history)
        self.assertIn('| Package total downloads | 200 | +11 |', summary)
        self.assertIn('| Version `build-cache`, `v2.5.0` | 82 | +4 |', summary)

    def test_summary_marks_unavailable_version_as_last_observed(self):
        collected = p.collect(index(), fetch=lambda url: version())
        history, _ = p.merge(None, collected, index(), '2026-09-16T12:21:06Z')
        current = p.collect(index(200, 80), history, fetch=lambda url: None)
        history, _ = p.merge(history, current, index(200, 80), '2026-09-17T00:17:00Z')
        summary = p.render(history)
        self.assertIn('(last observed 2026-09-16T12:21:06Z; unavailable)', summary)
        self.assertNotIn('| Version `build-cache`, `v2.5.0` | 78 | +0 |', summary)

    def test_transient_build_id_is_not_refetched_after_leaving_index(self):
        first_index = index().replace(
            b'</li><div data-view-component="true" class="Box-footer">',
            f'''</li><li data-view-component="true" class="Box-row">
<a href="/users/jpezzulli/packages/container/sglang-rtxpro6000/999?tag=build-old">build-old</a>
<span value="sha256:{'b' * 64}"></span>
 2<span class="sr-only">Version downloads</span>
</li><div data-view-component="true" class="Box-footer">'''.encode())
        first = p.parse_index(first_index)
        history, _ = p.merge(None, first, first_index, '2026-09-16T12:21:06Z')
        calls = []
        current = p.collect(index(), history,
                            fetch=lambda url: calls.append(url) or version())
        self.assertNotIn('999', current['versions'])
        self.assertFalse(any(url.endswith('/999') for url in calls))

    def test_counter_decrease_is_preserved_not_invented(self):
        collected = p.collect(index(), fetch=lambda url: version())
        history, _ = p.merge(None, collected, index(), '2026-09-16T12:21:06Z')
        lower = p.collect(index(180, 70), history, fetch=lambda url: version(70))
        history, observation = p.merge(history, lower, index(180, 70), '2026-09-17T00:17:00Z')
        self.assertEqual(observation['total_downloads'], 180)
        self.assertEqual(history['days']['2026-09-16']['total_downloads'], 189)

    def test_seed_writes_raw_page_and_future_runs_do_not_replace_it(self):
        captured = []
        original_read, original_publish = p.read_history, p.publish
        p.read_history = lambda api: (None, 'tree', None)
        p.publish = lambda api, head, tree, files, message: captured.append(files) or 'sha'
        try:
            p.archive(None, index(), '2026-09-16T12:21:06Z', seed=True)
            self.assertEqual(captured[0]['raw/packages/first-run/package.html'].encode(), index())
            self.assertIn('package-downloads.json', captured[0])
            self.assertEqual(len([x for x in captured[0] if x.startswith('package-snapshots/')]), 1)
        finally:
            p.read_history, p.publish = original_read, original_publish

    def test_normal_run_refuses_to_create_unseeded_history(self):
        original_read = p.read_history
        p.read_history = lambda api: ('head', 'tree', None)
        try:
            with self.assertRaises(p.ArchiveError):
                p.archive(None, index(), '2026-09-16T12:22:00Z')
        finally:
            p.read_history = original_read

    def test_summary_labels_downloads_without_unique_user_claim(self):
        collected = p.collect(index(), fetch=lambda url: version())
        history, _ = p.merge(None, collected, index(), '2026-09-16T12:21:06Z')
        summary = p.render(history)
        self.assertIn('Package total downloads', summary)
        self.assertIn('not unique users', summary)
        self.assertNotIn('unique downloaders', summary)


if __name__ == '__main__':
    unittest.main()
