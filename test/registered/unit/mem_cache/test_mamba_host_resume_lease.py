"""CPU checks for Mamba resume pins.

Pins have to follow the session tracker's enabled, closed, and generation
rules. These cases use stub nodes and a stub LRU, and they call the real
pin, close, and fork methods.
"""

import unittest
from collections import defaultdict
from types import SimpleNamespace

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    CacheTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.session_ref_tracker import (
    UnifiedSessionRefTracker,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MAMBA = ComponentType.MAMBA


class _LRU:
    def __init__(self):
        self.nodes = set()

    def in_list(self, node):
        return node in self.nodes

    def remove_node(self, node):
        self.nodes.discard(node)

    def insert_mru(self, node):
        self.nodes.add(node)

    def reset_node_mru(self, node):
        self.nodes.add(node)


class _Bridge:
    def __init__(self, comp):
        self.comp = comp

    def reset_session_state(self):
        return None

    def resolve_session_leaf(self, req, last_node):
        return self.comp.resolve_session_leaf(req, last_node)

    def register_session_leaf(self, session_id, leaf):
        return self.comp.register_session_leaf(session_id, leaf)

    def release_session(self, session_id):
        return self.comp.release_session(session_id)


def _cd(host=True):
    return SimpleNamespace(
        host_value=[1] if host else None,
        value=None,
        host_lock_ref=0,
        lock_ref=0,
        session_ref=0,
        session_ids=None,
    )


class _Node:
    def __init__(self, node_id, parent, host=True, tokens=()):
        self.id = node_id
        self.parent = parent
        self.key = list(tokens) if tokens else None
        self.children = {}
        self.component_data = {MAMBA: _cd(host)}

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, _Node) and other.id == self.id


def _node(node_id, parent, host=True, tokens=()):
    return _Node(node_id, parent, host, tokens)


class ResumeLeaseTests(unittest.TestCase):
    def setUp(self):
        self.freed = []
        self.comp = MambaComponent.__new__(MambaComponent)
        self.comp.component_type = MAMBA
        self.comp.mamba_checkpoint_grid = 1
        self.comp._resume_leases = {}
        self.comp._resume_pins = {}
        self.comp._pending_resume_backup = {}
        self.comp._session_leaves = defaultdict(set)
        pool = SimpleNamespace(size=4, available_size=lambda: 3)
        self.comp.cache = SimpleNamespace(
            session_refs=None,
            _free_values=lambda *_args, **_kwargs: None,
            host_pool_group=SimpleNamespace(get_pool=lambda _name: pool),
            enable_mamba_extra_buffer=False,
            req_to_token_pool=SimpleNamespace(
                mamba_ckpt_pool=None,
                free_mamba_cache=lambda *_a, **_k: None,
            ),
        )
        self.root = _Node(0, None, host=False)
        self.nodes = {0: self.root}
        lru = _LRU()
        self.comp.tree_core = SimpleNamespace(
            root_node=self.root,
            enable_session_radix_cache=True,
            host_lru_lists={MAMBA: lru},
            lru_lists={MAMBA: _LRU()},
            evictable_host_leaves=set(),
            _update_evictable_leaf_sets=lambda _node: None,
            _evict_component_and_detach_lru=self._evict,
            _cascade_evict=lambda *_args, **_kwargs: None,
            node_by_id=lambda node_id: self.nodes[node_id],
        )
        self.bridge = _Bridge(self.comp)
        self.tracker = UnifiedSessionRefTracker(
            components=(self.bridge,),
            tree_core=self.comp.tree_core,
            enable_session_radix_cache=True,
        )
        self.comp.cache.session_refs = self.tracker

    def _evict(self, node, _comp, **_kwargs):
        node.component_data[MAMBA].host_value = None
        self.freed.append(node.id)

    def _add(self, node):
        self.nodes[node.id] = node
        return node

    def _req(self, session_id, generation, streaming=False):
        session = SimpleNamespace(streaming=streaming, session_id=session_id)
        return SimpleNamespace(
            session_id=session_id,
            session=session,
            session_generation=generation,
        )

    def _insert(self, node):
        return SimpleNamespace(last_device_node=node.id, mamba_exist=False)

    def _finish(self, req, node):
        """cache_finished_req order: component cleanup, then session register."""
        req.last_node = node.id
        self.comp.cleanup_after_caching_req(
            req,
            is_finished=True,
            insert_result=self._insert(node),
            insert_params=None,
        )
        self.tracker.register_session_ref(req)

    def _open(self, session_id="s"):
        return self.tracker.open_radix_session(session_id)

    def _match(self, req, node):
        """Pin a matched checkpoint the way a match tick does."""
        params = MatchPrefixParams(key=RadixKey(token_ids=[]), req=req)
        self.comp.finalize_match_result_in_tree_core(
            MatchResult(
                device_indices=[],
                last_device_node=node,
                last_host_node=node,
                best_match_node=node,
            ),
            params,
            [],
            0,
        )

    def _backup(self, node):
        self.comp.commit_hicache_transfer(
            node,
            CacheTransferPhase.BACKUP_HOST,
            transfers=[
                SimpleNamespace(host_indices=SimpleNamespace(clone=lambda: [1]))
            ],
            cache_actions=[],
        )

    def test_disabled_session_cache_does_not_pin(self):
        self.tracker.enable_session_radix_cache = False
        self.comp.tree_core.enable_session_radix_cache = False
        node = self._add(_node(1, self.root))
        generation = 1
        self.tracker._session_generations["s"] = generation
        req = self._req("s", generation)
        self.comp._note_inserted_resume(req, self._insert(node))
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 0)
        self.assertEqual(self.comp._resume_pins, {})
        self.assertEqual(self.tracker.release_radix_session("s"), 0)

    def test_close_drops_the_pin_and_a_late_finish_does_not_restore_it(self):
        generation = self._open()
        node = self._add(_node(1, self.root))
        req = self._req("s", generation)
        self.comp._note_inserted_resume(req, self._insert(node))
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 1)
        self.tracker.release_radix_session("s")
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 0)
        self.assertNotIn("s", self.comp._resume_pins)

        self.comp._note_inserted_resume(req, self._insert(node))
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 0)
        self.tracker.release_radix_session("s")
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 0)

    def test_reopened_session_id_ignores_the_previous_generation(self):
        first = self._open()
        node = self._add(_node(1, self.root))
        old = self._req("s", first)
        self.comp._note_inserted_resume(old, self._insert(node))
        self.tracker.release_radix_session("s")
        second = self._open()
        self.assertNotEqual(first, second)

        self.comp._note_inserted_resume(old, self._insert(node))
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 0)

        fresh = self._req("s", second)
        self.comp._note_inserted_resume(fresh, self._insert(node))
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 1)
        self.comp._note_inserted_resume(old, self._insert(node))
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 1)

    def test_close_clears_match_and_commit_pins(self):
        generation = self._open()
        req = self._req("s", generation)
        match = self._add(_node(1, self.root))
        commit = self._add(_node(2, self.root))
        params = MatchPrefixParams(key=RadixKey(token_ids=[]), req=req)
        result = MatchResult(
            device_indices=[],
            last_device_node=match,
            last_host_node=match,
            best_match_node=match,
        )
        self.comp.finalize_match_result_in_tree_core(result, params, [], 0)
        self.comp._note_inserted_resume(req, self._insert(commit))
        self.assertEqual(match.component_data[MAMBA].host_lock_ref, 1)
        self.assertEqual(commit.component_data[MAMBA].host_lock_ref, 1)
        self.assertEqual(set(self.comp._resume_pins["s"]), {"match", "commit"})

        self.tracker.release_radix_session("s")
        self.assertEqual(match.component_data[MAMBA].host_lock_ref, 0)
        self.assertEqual(commit.component_data[MAMBA].host_lock_ref, 0)
        self.assertNotIn("s", self.comp._resume_pins)

    def test_fork_drops_the_abandoned_tail_and_keeps_the_shared_node(self):
        generation = self._open()
        fork = self._add(_node(1, self.root))
        old = self._add(_node(2, fork))
        new = self._add(_node(3, fork))
        fork.children = {2: old, 3: new}
        old.component_data[MAMBA].session_ids = {"s"}
        self.comp._session_leaves["s"].add(old)

        req = self._req("s", generation)
        self.assertEqual(self.comp._pin_session_id(req), "s")
        self.comp.register_session_leaf("s", new)

        self.assertIsNone(old.component_data[MAMBA].host_value)
        self.assertEqual(fork.component_data[MAMBA].host_value, [1])
        self.assertEqual(self.freed, [old.id])
        self.assertIn(new, self.comp._session_leaves["s"])
        self.assertNotIn(old, self.comp._session_leaves["s"])

    def test_late_host_backup_does_not_pin_a_closed_session(self):
        generation = self._open()
        node = self._add(_node(1, self.root, host=False))
        node.component_data[MAMBA].value = [1]
        req = self._req("s", generation)
        self.comp._note_inserted_resume(req, self._insert(node))
        self.assertEqual(self.comp._pending_resume_backup[node.id], {"s": generation})
        self.tracker.release_radix_session("s")
        self.assertNotIn(node.id, self.comp._pending_resume_backup)

        # A backup that still holds the old session id must not pin after close.
        self.comp._pending_resume_backup[node.id] = {"s": generation}
        node.component_data[MAMBA].host_value = [1]
        self._backup(node)
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 0)
        self.assertNotIn(node.id, self.comp._pending_resume_backup)

    def test_fork_cancels_a_pending_backup_on_a_device_only_abandoned_leaf(self):
        generation = self._open()
        req = self._req("s", generation)
        shared = self._add(_node(1, self.root))
        # The abandoned branch is deeper, so it also blocks a shallower pin.
        abandoned = self._add(_node(2, shared, host=False, tokens=[20, 30]))
        abandoned.component_data[MAMBA].value = [1]
        abandoned.component_data[MAMBA].session_ids = {"s"}
        shared.children = {2: abandoned}
        self.comp._session_leaves["s"].add(abandoned)

        self.comp._note_inserted_resume(req, self._insert(abandoned))
        self.assertEqual(
            self.comp._pending_resume_backup[abandoned.id], {"s": generation}
        )

        branch = self._add(_node(3, shared, tokens=[40]))
        shared.children = {2: abandoned, 3: branch}
        self.comp.register_session_leaf("s", branch)
        self.assertNotIn(abandoned.id, self.comp._pending_resume_backup)
        # Registering the new leaf hands the pin to the branch this session
        # actually resumes from, even though it is shallower.
        self.assertEqual(self.comp._resume_pins, {"s": {"commit": branch.id}})

        # The abandoned backup lands after the fork. Same open generation, so
        # only the cancelled ownership can keep it from stealing the pin.
        abandoned.component_data[MAMBA].host_value = [1]
        self.comp.commit_hicache_transfer(
            abandoned,
            CacheTransferPhase.BACKUP_HOST,
            transfers=[
                SimpleNamespace(host_indices=SimpleNamespace(clone=lambda: [1]))
            ],
            cache_actions=[],
        )
        self.assertEqual(self.comp._resume_pins, {"s": {"commit": branch.id}})
        self.assertEqual(abandoned.component_data[MAMBA].host_lock_ref, 0)

        # Repeating the note for the pinned checkpoint changes nothing.
        self.comp._note_inserted_resume(req, self._insert(branch))
        self.assertEqual(self.comp._resume_pins, {"s": {"commit": branch.id}})
        self.assertEqual(branch.component_data[MAMBA].host_lock_ref, 1)
        self.assertEqual(abandoned.component_data[MAMBA].host_lock_ref, 0)

    def test_two_sessions_sharing_a_device_only_node_keep_both_pending_owners(self):
        first = self._open("a")
        second = self._open("b")
        node = self._add(_node(1, self.root, host=False))
        node.component_data[MAMBA].value = [1]
        self.comp._note_inserted_resume(self._req("a", first), self._insert(node))
        self.comp._note_inserted_resume(self._req("b", second), self._insert(node))
        self.assertEqual(
            self.comp._pending_resume_backup[node.id], {"a": first, "b": second}
        )

        self.tracker.release_radix_session("b")
        self.assertEqual(self.comp._pending_resume_backup[node.id], {"a": first})

        node.component_data[MAMBA].host_value = [1]
        self.comp.commit_hicache_transfer(
            node,
            CacheTransferPhase.BACKUP_HOST,
            transfers=[
                SimpleNamespace(host_indices=SimpleNamespace(clone=lambda: [1]))
            ],
            cache_actions=[],
        )
        self.assertEqual(self.comp._resume_pins, {"a": {"commit": node.id}})
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 1)

        # A reopened id is a new incarnation and pins the checkpoint on its own.
        third = self._open("b")
        self.comp._note_inserted_resume(self._req("b", third), self._insert(node))
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 2)

        self.tracker.release_radix_session("a")
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 1)
        self.assertEqual(self.comp._resume_pins, {"b": {"commit": node.id}})

    def test_fork_cancels_only_the_forking_session_pending_owner(self):
        first = self._open("a")
        second = self._open("b")
        shared = self._add(_node(1, self.root))
        abandoned = self._add(_node(2, shared, host=False))
        abandoned.component_data[MAMBA].value = [1]
        abandoned.component_data[MAMBA].session_ids = {"a", "b"}
        shared.children = {2: abandoned}
        self.comp._session_leaves["a"].add(abandoned)
        self.comp._session_leaves["b"].add(abandoned)
        self.comp._note_inserted_resume(self._req("a", first), self._insert(abandoned))
        self.comp._note_inserted_resume(self._req("b", second), self._insert(abandoned))

        # b forks; a still resumes from the shared checkpoint, so the host copy
        # stays and only b's pending ownership goes.
        branch = self._add(_node(3, shared))
        shared.children = {2: abandoned, 3: branch}
        self.comp.register_session_leaf("b", branch)
        self.assertEqual(self.comp._pending_resume_backup[abandoned.id], {"a": first})
        self.assertEqual(abandoned.component_data[MAMBA].host_value, None)
        self.assertEqual(self.freed, [])

    def test_finished_turn_pins_the_shorter_new_branch_after_the_fork(self):
        """The deeper abandoned pin must not leave the new branch unprotected.

        cache_finished_req notes the insert before registering the session leaf,
        so the forward-depth guard rejects the shallower branch while the old
        deeper pin still stands, and the fork then drops that pin.
        """
        generation = self._open()
        req = self._req("s", generation)
        shared = self._add(_node(1, self.root, tokens=[5]))
        old = self._add(_node(2, shared, tokens=[10, 11, 12]))
        shared.children = {2: old}
        deeper = self._add(_node(4, old, host=False, tokens=[13, 14, 15]))
        deeper.component_data[MAMBA].value = [1]
        old.children = {4: deeper}

        # Turn 1 leaves a durable host checkpoint, turn 2 goes deeper device-only.
        self._finish(req, old)
        self._finish(req, deeper)
        self.assertEqual(self.comp._resume_pins["s"], {"commit": old.id})
        self.assertEqual(self.comp._pending_resume_backup[deeper.id], {"s": generation})

        # Another session shares the deeper pending checkpoint.
        other = self._open("b")
        self.comp._note_inserted_resume(self._req("b", other), self._insert(deeper))
        self.assertEqual(
            self.comp._pending_resume_backup[deeper.id], {"s": generation, "b": other}
        )

        # A match on the shared prefix is the only other lease this session holds.
        self._match(req, shared)

        # Turn 3 forks to a shallower branch that is already durable on host.
        branch = self._add(_node(3, shared, tokens=[20]))
        shared.children = {2: old, 3: branch}
        req.last_node = branch.id
        self.comp.cleanup_after_caching_req(
            req,
            is_finished=True,
            insert_result=self._insert(branch),
            insert_params=None,
        )
        # The deeper abandoned pin still stands at this point, so the new
        # branch holds nothing yet (monotonic depth guard).
        self.assertEqual(
            self.comp._resume_pins["s"], {"match": shared.id, "commit": old.id}
        )
        self.assertEqual(branch.component_data[MAMBA].host_lock_ref, 0)

        self.tracker.register_session_ref(req)

        self.assertEqual(
            self.comp._resume_pins["s"], {"match": shared.id, "commit": branch.id}
        )
        self.assertEqual(branch.component_data[MAMBA].host_lock_ref, 1)
        self.assertEqual(old.component_data[MAMBA].host_lock_ref, 0)
        self.assertIsNone(old.component_data[MAMBA].host_value)
        self.assertEqual(self.freed, [old.id])
        self.assertEqual(self.comp._session_leaves["s"], {branch})

        # The abandoned deeper backup lands late: s keeps the new branch, and
        # the other owner still gets its promised checkpoint.
        self._backup(deeper)
        self.assertEqual(
            self.comp._resume_pins["s"], {"match": shared.id, "commit": branch.id}
        )
        self.assertEqual(self.comp._resume_pins["b"], {"commit": deeper.id})
        self.assertEqual(deeper.component_data[MAMBA].host_lock_ref, 1)
        self.assertEqual(set(self.comp._resume_leases["s"]), {shared.id, branch.id})

    def test_fork_retires_its_ownership_under_another_owners_host_lock(self):
        """Evictability must not decide who still owns a resume checkpoint.

        Another session's pin stops the host drop at the shared ancestor, yet
        this session's obsolete pin and pending backup on that ancestor are
        abandoned too, or the shallower new branch can never take the pin.
        """
        first = self._open("s")
        s_req = self._req("s", first)
        second = self._open("b")
        b_req = self._req("b", second)
        shared = self._add(_node(1, self.root, tokens=[5]))
        old = self._add(_node(2, shared, tokens=[10, 11, 12, 13]))
        deeper = self._add(_node(4, old, host=False, tokens=[14, 15, 16, 17]))
        deeper.component_data[MAMBA].value = [1]
        shared.children = {2: old}
        old.children = {4: deeper}

        self._finish(s_req, old)
        self._finish(s_req, deeper)
        self._finish(b_req, old)
        self.comp._note_inserted_resume(b_req, self._insert(deeper))
        self._match(s_req, shared)

        self.assertEqual(
            self.comp._resume_pins,
            {
                "s": {"commit": old.id, "match": shared.id},
                "b": {"commit": old.id},
            },
        )
        self.assertEqual(old.component_data[MAMBA].host_lock_ref, 2)
        self.assertEqual(
            self.comp._pending_resume_backup[deeper.id], {"s": first, "b": second}
        )

        branch = self._add(_node(3, shared, tokens=[20]))
        shared.children = {2: old, 3: branch}
        self._finish(s_req, branch)

        # s moved to the shallower branch; b still resumes the shared ancestor,
        # so its host copy stays and only s's lock is dropped.
        self.assertEqual(
            self.comp._resume_pins,
            {
                "s": {"match": shared.id, "commit": branch.id},
                "b": {"commit": old.id},
            },
        )
        self.assertEqual(branch.component_data[MAMBA].host_lock_ref, 1)
        self.assertEqual(old.component_data[MAMBA].host_lock_ref, 1)
        self.assertEqual(old.component_data[MAMBA].host_value, [1])
        self.assertEqual(self.freed, [])
        self.assertEqual(self.comp._pending_resume_backup[deeper.id], {"b": second})
        self.assertEqual(self.comp._session_leaves["s"], {branch})
        self.assertEqual(set(self.comp._resume_leases["s"]), {shared.id, branch.id})

        # The abandoned pending callback lands late: it can only pin for b.
        self._backup(deeper)
        self.assertEqual(
            self.comp._resume_pins["s"], {"match": shared.id, "commit": branch.id}
        )
        self.assertEqual(self.comp._resume_pins["b"], {"commit": deeper.id})
        self.assertEqual(deeper.component_data[MAMBA].host_lock_ref, 1)

        # Closing b reclaims the host copies nobody resumes from any more, and
        # leaves s's two pins and their locks alone.
        self.tracker.release_radix_session("b")
        self.assertEqual(self.freed, [old.id, deeper.id])
        self.assertEqual(
            self.comp._resume_pins, {"s": {"match": shared.id, "commit": branch.id}}
        )
        self.assertEqual(self.comp._pending_resume_backup, {})
        self.assertEqual(branch.component_data[MAMBA].host_lock_ref, 1)
        self.assertEqual(shared.component_data[MAMBA].host_lock_ref, 1)

    def test_a_durable_re_insert_settles_every_pending_owner(self):
        first = self._open("a")
        second = self._open("b")
        node = self._add(_node(1, self.root, host=False))
        node.component_data[MAMBA].value = [1]
        self.comp._note_inserted_resume(self._req("a", first), self._insert(node))
        self.comp._note_inserted_resume(self._req("b", second), self._insert(node))
        self.assertEqual(
            self.comp._pending_resume_backup[node.id], {"a": first, "b": second}
        )
        node.component_data[MAMBA].host_value = [1]

        # Only a finishes again on the now durable node; b keeps its promise.
        self.comp._note_inserted_resume(self._req("a", first), self._insert(node))
        self.assertEqual(self.comp._resume_pins["a"], {"commit": node.id})
        self.assertEqual(self.comp._resume_pins["b"], {"commit": node.id})
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 2)
        self.assertNotIn(node.id, self.comp._pending_resume_backup)

    def test_fork_retires_its_leaf_registration_under_another_owners_pin(self):
        """Leaf coverage is abandoned ownership too, not eviction bookkeeping."""
        first = self._open("s")
        s_req = self._req("s", first)
        second = self._open("b")
        b_req = self._req("b", second)
        common = self._add(_node(1, self.root, tokens=[5]))
        old = self._add(_node(2, common, tokens=[10, 11, 12]))
        common.children = {2: old}
        self._finish(s_req, old)
        self._finish(b_req, old)
        cd = old.component_data[MAMBA]
        self.assertEqual(cd.session_ids, {"s", "b"})
        self.assertEqual(cd.session_ref, 2)
        self.assertEqual(cd.host_lock_ref, 2)

        branch = self._add(_node(3, common, tokens=[20]))
        common.children = {2: old, 3: branch}
        self._finish(s_req, branch)

        # The host drop stops at b's pin, but s has no claim left on old: it
        # keeps the host copy for b and stays in b's session partition only.
        self.assertEqual(
            self.comp._resume_pins,
            {"s": {"commit": branch.id}, "b": {"commit": old.id}},
        )
        self.assertEqual(self.comp._session_leaves["s"], {branch})
        self.assertEqual(cd.session_ids, {"b"})
        self.assertEqual(cd.session_ref, 1)
        self.assertEqual(cd.host_lock_ref, 1)
        self.assertEqual(cd.host_value, [1])
        self.assertEqual(self.freed, [])

        # Once the surviving owner closes, nothing resumes old any more and the
        # host copy is reclaimed while the current branch keeps its pin.
        self.tracker.release_radix_session("b")
        self.assertIsNone(cd.session_ids)
        self.assertEqual(cd.session_ref, 0)
        self.assertEqual(cd.host_lock_ref, 0)
        self.assertIsNone(cd.host_value)
        self.assertEqual(self.freed, [old.id])
        self.assertEqual(self.comp._resume_pins, {"s": {"commit": branch.id}})
        self.assertEqual(branch.component_data[MAMBA].host_lock_ref, 1)

    def test_fork_retires_its_leaf_registration_behind_an_ordinary_lock(self):
        generation = self._open()
        req = self._req("s", generation)
        common = self._add(_node(1, self.root, tokens=[5]))
        old = self._add(_node(2, common, tokens=[10, 11, 12]))
        old.component_data[MAMBA].lock_ref = 1
        common.children = {2: old}
        self._finish(req, old)

        branch = self._add(_node(3, common, tokens=[20]))
        common.children = {2: old, 3: branch}
        self._finish(req, branch)

        cd = old.component_data[MAMBA]
        self.assertEqual(self.comp._resume_pins, {"s": {"commit": branch.id}})
        self.assertEqual(self.comp._session_leaves["s"], {branch})
        self.assertIsNone(cd.session_ids)
        self.assertEqual(cd.session_ref, 0)
        # An ordinary lock keeps the host data, it does not keep the marker.
        self.assertEqual(cd.host_value, [1])
        self.assertEqual(self.freed, [])


if __name__ == "__main__":
    unittest.main()
