"""CPU coverage for host-only hybrid checkpoint retention.

Mamba state is consumed only at the selected match boundary.  A missing
intermediate checkpoint is therefore harmless when a later boundary has every
component needed for restore, while a missing terminal state or Full-KV hole is
not.  Under host pressure, prefer reclaiming those redundant intermediates
before deleting an older complete endpoint.
"""

import unittest
from array import array
from collections import defaultdict
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.full_component import (
    FullComponent,
)
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedLRUList,
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _HostBoundaryFixture:
    def __init__(self, page_size: int = 1, *, enable_sessions: bool = False):
        self.page_size = page_size
        self.root = UnifiedTreeNode((ComponentType.FULL, ComponentType.MAMBA))
        self.root.key = RadixKey(array("q"))
        self.root.component_data[ComponentType.FULL].value = []

        self.full = object.__new__(FullComponent)
        self.mamba = object.__new__(MambaComponent)
        def session_predicate(node):
            return node.component_data[ComponentType.MAMBA].session_ref > 0

        self.tree = SimpleNamespace(
            root_node=self.root,
            page_size=page_size,
            enable_hicache=True,
            enable_session_radix_cache=enable_sessions,
            components=(self.full, self.mamba),
            host_lru_lists={
                ComponentType.MAMBA: UnifiedLRUList(
                    ComponentType.MAMBA,
                    (ComponentType.FULL, ComponentType.MAMBA),
                    use_host_ptr=True,
                    is_referenced=session_predicate if enable_sessions else None,
                )
            },
            evictable_host_leaves=set(),
        )
        self.full.tree_core = self.tree
        self.mamba.tree_core = self.tree
        self.mamba.mamba_checkpoint_grid = page_size

    def add(
        self,
        parent,
        token: int,
        *,
        full: str = "host",
        mamba: str = "host",
        add_to_mamba_lru: bool = True,
        session_ref: int = 0,
    ):
        node = UnifiedTreeNode((ComponentType.FULL, ComponentType.MAMBA))
        node.parent = parent
        node.key = RadixKey(array("q", range(token, token + self.page_size)))
        parent.children[node.key.child_key(self.page_size)] = node

        for component_type, residence in (
            (ComponentType.FULL, full),
            (ComponentType.MAMBA, mamba),
        ):
            cd = node.component_data[component_type]
            value = torch.tensor([token], dtype=torch.int64)
            if residence == "device":
                cd.value = value
            elif residence == "host":
                cd.host_value = value
            elif residence != "missing":
                raise ValueError(residence)

        node.component_data[ComponentType.MAMBA].session_ref = session_ref

        if mamba == "host" and add_to_mamba_lru:
            self.tree.host_lru_lists[ComponentType.MAMBA].insert_mru(node)
        return node

    def tokens_to(self, node):
        chunks = []
        while node is not self.root:
            chunks.append(list(node.key.token_ids))
            node = node.parent
        return [token for chunk in reversed(chunks) for token in chunk]

    def match(self, node):
        return UnifiedTreeCore._match_prefix_helper(
            self.tree, RadixKey(array("q", self.tokens_to(node)))
        )


class TestMambaHostBoundaryMatching(unittest.TestCase):
    def test_missing_intermediate_reaches_later_complete_boundary(self):
        for page_size in (1, 64):
            with self.subTest(page_size=page_size):
                f = _HostBoundaryFixture(page_size)
                first = f.add(f.root, 100)
                f.add(first, 200, mamba="missing")
                terminal = f.add(next(iter(first.children.values())), 300)

                _, best, _, _, full_hit, _ = f.match(terminal)

                self.assertIs(best, terminal)
                self.assertEqual(full_hit, 3 * page_size)

    def test_missing_terminal_state_is_not_accepted(self):
        f = _HostBoundaryFixture()
        first = f.add(f.root, 10)
        terminal = f.add(first, 20, mamba="missing")

        _, best, _, _, full_hit, _ = f.match(terminal)

        self.assertIs(best, first)
        self.assertEqual(full_hit, 2)

    def test_full_kv_hole_stops_before_later_mamba_state(self):
        f = _HostBoundaryFixture()
        first = f.add(f.root, 10)
        hole = f.add(first, 20, full="missing", mamba="missing")
        terminal = f.add(hole, 30)

        _, best, _, _, full_hit, _ = f.match(terminal)

        self.assertIs(best, first)
        self.assertEqual(full_hit, 1)

    def test_query_ending_at_hole_does_not_borrow_descendant_state(self):
        f = _HostBoundaryFixture()
        first = f.add(f.root, 10)
        hole = f.add(first, 20, mamba="missing")
        f.add(hole, 30)

        _, best, _, _, full_hit, _ = f.match(hole)

        self.assertIs(best, first)
        self.assertEqual(full_hit, 2)


class TestMambaHostEvictionPreference(unittest.TestCase):
    def _terminal_and_redundant(self):
        f = _HostBoundaryFixture()
        # Insert terminal first so plain LRU would select and atomically delete it.
        terminal = f.add(f.root, 10)
        redundant = f.add(f.root, 20)
        later = f.add(redundant, 30)
        f.tree.evictable_host_leaves.add(terminal)
        return f, terminal, redundant, later

    def test_prefers_redundant_intermediate_over_older_terminal(self):
        f, terminal, redundant, _ = self._terminal_and_redundant()

        selected = f.mamba._select_host_eviction_candidate(
            f.tree.host_lru_lists[ComponentType.MAMBA]
        )

        self.assertIs(selected, redundant)
        self.assertIsNot(selected, terminal)

    def test_drive_reclaims_intermediate_without_deleting_terminal_full_kv(self):
        f, terminal, redundant, _ = self._terminal_and_redundant()
        evicted = []

        def evict_host_leaf(node, tracker, device_frees, host_frees):
            evicted.append(("leaf", node))
            tracker[ComponentType.MAMBA] += 1

        def evict_component(
            node,
            component,
            *,
            target,
            tracker,
            device_frees,
            host_frees,
        ):
            evicted.append(("component", node))
            f.tree.host_lru_lists[ComponentType.MAMBA].remove_node(node)
            node.component_data[ComponentType.MAMBA].host_value = None
            tracker[ComponentType.MAMBA] += 1

        f.tree._evict_host_leaf = evict_host_leaf
        f.tree._evict_component_and_detach_lru = evict_component
        f.tree._cascade_evict = lambda *args, **kwargs: None
        f.tree._update_evictable_leaf_sets = lambda node: None
        tracker = defaultdict(int)

        f.mamba.drive_host_eviction(1, tracker, defaultdict(list), defaultdict(list))

        self.assertEqual(evicted, [("component", redundant)])
        self.assertIsNotNone(
            terminal.component_data[ComponentType.FULL].host_value,
            "terminal Full KV must not be atomically deleted for Mamba pressure",
        )
        self.assertIsNone(redundant.component_data[ComponentType.MAMBA].host_value)

    def test_pinned_redundant_checkpoint_falls_back_to_terminal(self):
        f, terminal, redundant, _ = self._terminal_and_redundant()
        redundant.component_data[ComponentType.MAMBA].host_lock_ref = 1

        selected = f.mamba._select_host_eviction_candidate(
            f.tree.host_lru_lists[ComponentType.MAMBA]
        )

        self.assertIs(selected, terminal)

    def test_preference_does_not_cross_session_partition(self):
        f = _HostBoundaryFixture(enable_sessions=True)
        terminal = f.add(f.root, 10)
        redundant = f.add(f.root, 20, session_ref=1)
        f.add(redundant, 30, session_ref=1)

        selected = f.mamba._select_host_eviction_candidate(
            f.tree.host_lru_lists[ComponentType.MAMBA]
        )

        self.assertIs(selected, terminal)

    def test_session_cursor_is_removed_when_candidate_check_raises(self):
        f = _HostBoundaryFixture(enable_sessions=True)
        f.add(f.root, 10)
        host_lru = f.tree.host_lru_lists[ComponentType.MAMBA]

        def raise_boundary(_node):
            raise RuntimeError("boom")

        f.mamba._has_later_complete_boundary = raise_boundary

        with self.assertRaisesRegex(RuntimeError, "boom"):
            f.mamba._select_host_eviction_candidate(host_lru)

        pointer = host_lru._pt
        self.assertIsNone(host_lru.cursor.lru_prev[pointer])
        self.assertIsNone(host_lru.cursor.lru_next[pointer])

    def test_branch_checkpoint_is_not_classified_redundant(self):
        f = _HostBoundaryFixture()
        branch = f.add(f.root, 10)
        f.add(branch, 20)
        f.add(branch, 30)

        self.assertFalse(f.mamba._has_later_complete_boundary(branch))

    def test_descendant_needs_complete_component_consensus(self):
        f = _HostBoundaryFixture()
        checkpoint = f.add(f.root, 10)
        missing_mamba = f.add(checkpoint, 20, mamba="missing", add_to_mamba_lru=False)
        f.add(
            missing_mamba,
            30,
            full="missing",
            mamba="device",
            add_to_mamba_lru=False,
        )

        self.assertFalse(f.mamba._has_later_complete_boundary(checkpoint))


if __name__ == "__main__":
    unittest.main()
