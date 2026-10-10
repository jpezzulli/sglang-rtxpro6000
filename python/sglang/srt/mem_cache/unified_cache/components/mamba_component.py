from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.unified_cache.cache_action import (
    FreeComponentDeviceSlot,
    FreeComponentHostSlot,
    MambaEvictExcessPathStates,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    LRURefreshPhase,
    PrepareLoadBackResult,
    PreparePrefetchResult,
    TreeComponent,
    get_and_increase_time_counter,
)
from sglang.srt.runtime_context import (
    get_exec,
    mamba_cache_chunk_size,
    mamba_checkpoint_grid,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_radix_cache import (
        NodeId,
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


logger = logging.getLogger(__name__)


class MambaComponent(TreeComponent):
    component_type = ComponentType.MAMBA

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

        assert isinstance(
            params.req_to_token_pool, HybridReqToTokenPool
        ), f"MambaComponent requires HybridReqToTokenPool, got {type(params.req_to_token_pool)}"
        if not params.enable_mamba_extra_buffer:
            assert (
                params.page_size == 1
            ), f"MambaComponent requires page_size=1 when mamba_extra_buffer is disabled, got {params.page_size}"
        super().__init__(cache, params)
        self.mamba_cache_chunk_size = mamba_cache_chunk_size()
        # params.page_size is the tree page the allocator actually uses, already
        # widened by dcp_size, so it is the one grid a checkpoint depth can land on.
        self.mamba_checkpoint_grid = mamba_checkpoint_grid(params.page_size)
        self.mamba_max_states_per_path = get_exec().mamba.mamba_max_states_per_path
        # HiCache state
        self._mamba_pool_host = None  # set to host mamba pool when HiCache enabled
        self._resume_leases: dict[str, dict[int, bool]] = {}
        self._resume_pins: dict[str, dict[str, int]] = {}
        # node id -> {session id: session generation} waiting for a durable
        # host backup. Several sessions can share one device-only checkpoint,
        # so keep every eligible owner and cancel them independently.
        self._pending_resume_backup: dict[int, dict[str, int]] = {}

    def needs_incremental_backup(self, node: UnifiedTreeNode) -> bool:
        data = node.component_data[self.component_type]
        return data.value is not None and data.host_value is None

    def _inc_session_coverage(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        cd = leaf.component_data[self.component_type]
        cd.session_ref += 1
        if cd.session_ref == 1:
            self._refresh_session_partition(leaf)

    def _dec_session_coverage(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        cd = leaf.component_data[self.component_type]
        assert cd.session_ref > 0
        cd.session_ref -= 1
        if cd.session_ref == 0:
            self._refresh_session_partition(leaf)

    def _advance_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        old_ancestor: Optional[UnifiedTreeNode],
    ) -> None:
        self._inc_session_coverage(session_id, leaf)
        if old_ancestor is not None:
            self._dec_session_coverage(session_id, old_ancestor)

    def _recede_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        fallback: Optional[UnifiedTreeNode],
    ) -> None:
        self._dec_session_coverage(session_id, leaf)
        if fallback is not None:
            self._inc_session_coverage(session_id, fallback)

    def register_session_leaf(
        self, session_id: str, leaf: Optional[UnifiedTreeNode]
    ) -> None:
        """Drop a branch this session left behind when the new leaf diverges.

        A continuing turn keeps the previous leaf as an ancestor, and that
        checkpoint stays. A fork leaves the previous leaf as a sibling. Its
        host Mamba, and the ancestors strictly below the shared node, will
        not be resumed.
        """
        if (
            self.tree_core is not None
            and self.tree_core.enable_session_radix_cache
            and leaf is not None
            and leaf is not self.tree_core.root_node
        ):
            for old in tuple(self._session_leaves.get(session_id, ())):
                if old is leaf or self._node_is_ancestor(old, leaf):
                    continue
                self._drop_abandoned_tail(
                    session_id,
                    old,
                    self._lowest_common_ancestor(old, leaf),
                    "fork",
                )
            # A fork drops the deeper abandoned pin, and cache_finished_req notes
            # the insert before registering the leaf, so the forward-depth guard
            # had not let this shallower durable checkpoint take the pin yet.
            if leaf.component_data[self.component_type].host_value is not None:
                self._pin_resume(session_id, leaf, "commit")
        super().register_session_leaf(session_id, leaf)

    def release_session(self, session_id: str) -> int:
        if self.tree_core is not None and self.tree_core.enable_session_radix_cache:
            for leaf in tuple(self._session_leaves.get(session_id, ())):
                stop = self._shared_prefix_stop(leaf, session_id)
                if stop is None:
                    continue
                self._drop_abandoned_tail(session_id, leaf, stop, "close")
        indexed = super().release_session(session_id)
        self._release_resume_leases(session_id)
        return indexed

    def _resume_tracker(self):
        return getattr(self.cache, "session_refs", None)

    def _pin_session_id(self, req: Optional[Req]) -> Optional[str]:
        tracker = self._resume_tracker()
        if tracker is None or req is None:
            return None
        return tracker.current_pin_session_id(req)

    def _try_node(self, node_id: int) -> Optional[UnifiedTreeNode]:
        try:
            return self.tree_core.node_by_id(node_id)
        except (KeyError, IndexError):
            return None

    def _lease_log(
        self, op: str, pin: str, session_id: str, node: UnifiedTreeNode
    ) -> None:
        held = sum(
            1 for locked in self._resume_leases.get(session_id, {}).values() if locked
        )
        logger.info(
            "mamba host lease op=%s pin=%s session=%s node=%s depth=%s held=%s",
            op,
            pin,
            session_id,
            node.id,
            self._token_depth(node),
            held,
        )

    def _pin_label(self, session_id: str, node_id: int) -> str:
        pins = self._resume_pins.get(session_id, {})
        names = [name for name, pinned in pins.items() if pinned == node_id]
        return "+".join(names) if names else "released"

    def _needed_resume_nodes(self, session_id: str) -> set[int]:
        return set(self._resume_pins.get(session_id, {}).values())

    def _resume_sessions_for(self, node: UnifiedTreeNode) -> list[str]:
        return [
            session_id
            for session_id, pins in self._resume_pins.items()
            if node.id in pins.values()
        ]

    def _host_lock_resume(
        self,
        session_id: str,
        node: UnifiedTreeNode,
        pin: str,
    ) -> bool:
        """Hold this session's Mamba host lock on one resume snapshot.

        Match scans repeat every scheduler tick, so a session locks a node once.
        The lock is this component's host_lock_ref only. acquire removes a
        host-only node from the host LRU, but the tree sanity check requires
        every host-only Mamba node to stay in that list. Eviction skips
        host_lock_ref, so put the node back after the lock is held.
        """
        if node is self.tree_core.root_node:
            return False
        cd = node.component_data[self.component_type]
        if cd.host_value is None:
            return False
        leases = self._resume_leases.setdefault(session_id, {})
        if not leases.get(node.id):
            self.acquire_component_lock(node, IncLockRefResult(), lock_host=True)
            if cd.host_lock_ref <= 0:
                return False
            leases[node.id] = True
        if cd.value is None:
            host_lru = self.tree_core.host_lru_lists[self.component_type]
            if host_lru.in_list(node):
                host_lru.reset_node_mru(node)
            else:
                host_lru.insert_mru(node)
        # A host lock makes the node not an H-leaf. Demote updates the leaf
        # set while the lock is held, so release has to put the node back.
        self.tree_core._update_evictable_leaf_sets(node)
        return True

    def _host_unlock_resume(
        self,
        session_id: str,
        node: UnifiedTreeNode,
        pin: str,
    ) -> None:
        leases = self._resume_leases.get(session_id)
        if not leases or node.id not in leases:
            return
        held = leases.pop(node.id)
        if not leases:
            self._resume_leases.pop(session_id, None)
        cd = node.component_data[self.component_type]
        if held and cd.host_lock_ref > 0:
            self.release_component_lock(node, None, lock_host=True)
            self.tree_core._update_evictable_leaf_sets(node)
        self._lease_log("release", pin, session_id, node)

    def _gc_resume(self, session_id: str) -> None:
        needed = self._needed_resume_nodes(session_id)
        leases = self._resume_leases.get(session_id)
        if not leases:
            return
        for node_id in list(leases):
            if node_id in needed:
                continue
            node = self._try_node(node_id)
            if node is None:
                leases.pop(node_id, None)
                continue
            self._host_unlock_resume(session_id, node, "released")

    def _pin_resume(self, session_id: str, node: UnifiedTreeNode, pin: str) -> None:
        """Lock node as this session's match or commit pin.

        The pin moves only forward in token depth, and only after the host
        lock is held. A shallower match tick, or an insert whose host backup
        is not durable yet, leaves the previous checkpoint locked.
        """
        if node is None or node is self.tree_core.root_node:
            return
        pins = self._resume_pins.setdefault(session_id, {})
        leases = self._resume_leases.get(session_id, {})
        already = pins.get(pin) == node.id and bool(leases.get(node.id))
        current_id = pins.get(pin)
        current = self._try_node(current_id) if current_id is not None else None
        dropped_dead = False
        if current_id is not None and current is None:
            pins.pop(pin, None)
            dropped_dead = True
        elif current is not None and self._token_depth(node) < self._token_depth(
            current
        ):
            return
        if not self._host_lock_resume(session_id, node, pin):
            if dropped_dead:
                self._gc_resume(session_id)
            return
        pins[pin] = node.id
        self._gc_resume(session_id)
        if not already:
            self._lease_log("acquire", pin, session_id, node)

    def _promote_pending_backup(self, node: UnifiedTreeNode, *, durable: bool) -> None:
        """Settle every owner waiting on this node's host backup.

        Pending ownership is per session, so a backup settles them all and one
        session's re-insert must not silently drop the others' promise.
        """
        owners = self._pending_resume_backup.pop(node.id, None) or {}
        tracker = self._resume_tracker()
        if not durable or tracker is None:
            return
        for session_id, generation in owners.items():
            if tracker.pin_still_current(session_id, generation):
                self._pin_resume(session_id, node, "commit")

    def _note_inserted_resume(
        self,
        req: Optional[Req],
        insert_result: Optional[InsertResult],
    ) -> None:
        """Pin the newest durable host checkpoint. Do not move the match pin."""
        session_id = self._pin_session_id(req)
        if session_id is None or insert_result is None:
            return
        node_ref = insert_result.last_device_node
        node_id = getattr(node_ref, "id", node_ref)
        try:
            node_id = int(node_id)
        except (TypeError, ValueError):
            return
        node = self._try_node(node_id)
        if node is None or node is self.tree_core.root_node:
            return
        cd = node.component_data[self.component_type]
        if cd.host_value is None and cd.value is None:
            return
        if cd.host_value is not None:
            self._promote_pending_backup(node, durable=True)
            self._pin_resume(session_id, node, "commit")
            return
        self._pending_resume_backup.setdefault(node.id, {})[
            session_id
        ] = req.session_generation

    def _release_resume_leases(self, session_id: str) -> None:
        self._resume_pins.pop(session_id, None)
        for node_id in list(self._resume_leases.get(session_id, {})):
            node = self._try_node(node_id)
            if node is None:
                leases = self._resume_leases.get(session_id)
                if leases is not None:
                    leases.pop(node_id, None)
                continue
            self._host_unlock_resume(session_id, node, "released")
        self._resume_leases.pop(session_id, None)
        for node_id, owners in list(self._pending_resume_backup.items()):
            owners.pop(session_id, None)
            if not owners:
                self._pending_resume_backup.pop(node_id, None)

    def _node_is_ancestor(
        self, ancestor: UnifiedTreeNode, node: UnifiedTreeNode
    ) -> bool:
        root = self.tree_core.root_node
        cur = node
        while cur is not None and cur is not root:
            if cur is ancestor:
                return True
            cur = cur.parent
        return False

    def _lowest_common_ancestor(
        self, left: UnifiedTreeNode, right: UnifiedTreeNode
    ) -> UnifiedTreeNode:
        seen: set[int] = set()
        cur: Optional[UnifiedTreeNode] = left
        while cur is not None:
            seen.add(id(cur))
            cur = cur.parent
        cur = right
        while cur is not None:
            if id(cur) in seen:
                return cur
            cur = cur.parent
        return self.tree_core.root_node

    def _shared_prefix_stop(
        self, leaf: UnifiedTreeNode, session_id: str
    ) -> Optional[UnifiedTreeNode]:
        """Nearest ancestor another branch or session still uses.

        None means this chain never forked. The whole chain may still be a
        shared prompt prefix, so host Mamba stays for ordinary eviction.
        """
        cur: Optional[UnifiedTreeNode] = leaf
        root = self.tree_core.root_node
        while cur is not None and cur is not root:
            parent = cur.parent
            if parent is None:
                return None
            if len(parent.children) > 1:
                return parent
            if parent is not root:
                cd = parent.component_data[self.component_type]
                session_ids = cd.session_ids or ()
                if any(other != session_id for other in session_ids):
                    return parent
                if any(
                    other != session_id for other in self._resume_sessions_for(parent)
                ):
                    return parent
            cur = parent
        return None

    def _drop_blocked(
        self,
        node: UnifiedTreeNode,
        session_id: str,
        origin: UnifiedTreeNode,
    ) -> bool:
        """A lock, or another session's pin or marker, ends the drop walk."""
        cd = node.component_data[self.component_type]
        if cd.lock_ref > 0:
            return True
        if any(other != session_id for other in self._resume_sessions_for(node)):
            return True
        session_ids = cd.session_ids or ()
        if any(other != session_id for other in session_ids):
            return True
        if node is not origin and session_id in session_ids:
            return True
        own_lock = bool(self._resume_leases.get(session_id, {}).get(node.id))
        return cd.host_lock_ref > int(own_lock)

    def _tail_below(
        self,
        leaf: UnifiedTreeNode,
        stop: UnifiedTreeNode,
        session_id: str,
    ) -> list[UnifiedTreeNode]:
        path: list[UnifiedTreeNode] = []
        cur: Optional[UnifiedTreeNode] = leaf
        root = self.tree_core.root_node
        while cur is not None and cur is not stop and cur is not root:
            if self._drop_blocked(cur, session_id, leaf):
                break
            path.append(cur)
            cur = cur.parent
        return path

    def _descendants_until_block(
        self, leaf: UnifiedTreeNode, session_id: str
    ) -> list[UnifiedTreeNode]:
        found: list[UnifiedTreeNode] = []
        pending = list(leaf.children.values())
        while pending:
            node = pending.pop()
            if self._drop_blocked(node, session_id, leaf):
                continue
            found.append(node)
            pending.extend(node.children.values())
        return found

    def _clear_session_leaf(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        leaves = self._session_leaves.get(session_id)
        if not leaves or leaf not in leaves:
            return
        cd = leaf.component_data[self.component_type]
        if cd.session_ids is None or session_id not in cd.session_ids:
            return
        if cd.session_ref > 0:
            self._dec_session_coverage(session_id, leaf)
        self._unmark_session_leaf(session_id, leaf)

    def _release_own_pin(self, session_id: str, node: UnifiedTreeNode) -> None:
        pins = self._resume_pins.get(session_id)
        if pins:
            for name, node_id in list(pins.items()):
                if node_id == node.id:
                    pins.pop(name, None)
            if not pins:
                self._resume_pins.pop(session_id, None)
        pending = self._pending_resume_backup.get(node.id)
        if pending is not None:
            pending.pop(session_id, None)
            if not pending:
                self._pending_resume_backup.pop(node.id, None)
        self._host_unlock_resume(session_id, node, "released")

    def _tombstone_mamba_host(
        self, session_id: str, nodes: list[UnifiedTreeNode]
    ) -> list[int]:
        device_frees: dict[ComponentType, list] = defaultdict(list)
        host_frees: dict[ComponentType, list] = defaultdict(list)
        tracker: dict[ComponentType, int] = defaultdict(int)
        dropped: list[int] = []
        try:
            for node in nodes:
                # This session no longer resumes from these nodes, so drop its
                # own pins and pending backup ownership first. A device-only
                # node has no host copy to tombstone, but a backup that lands
                # later would otherwise repin the abandoned branch.
                self._release_own_pin(session_id, node)
                cd = node.component_data[self.component_type]
                if cd.host_value is None:
                    continue
                if self._drop_blocked(node, session_id, nodes[0]):
                    continue
                if cd.host_value is None or cd.host_lock_ref > 0 or cd.lock_ref > 0:
                    continue
                self.tree_core._evict_component_and_detach_lru(
                    node,
                    self,
                    target=EvictLayer.HOST,
                    tracker=tracker,
                    device_frees=device_frees,
                    host_frees=host_frees,
                )
                # Leaf cascade flattens every component to the same priority
                # and would free KV host. Interior Mamba eviction does not.
                if node not in self.tree_core.evictable_host_leaves:
                    self.tree_core._cascade_evict(
                        node,
                        self,
                        tracker,
                        device_frees=device_frees,
                        host_frees=host_frees,
                        target=EvictLayer.HOST,
                    )
                self.tree_core._update_evictable_leaf_sets(node)
                dropped.append(node.id)
        finally:
            self.cache._free_values(device_frees, host_frees)
        return dropped

    def _abandoned_chain(
        self, leaf: UnifiedTreeNode, stop: UnifiedTreeNode
    ) -> list[UnifiedTreeNode]:
        """Every node below the fork point this session no longer resumes from.

        Deliberately blind to locks and other owners: whether a host copy is
        reclaimable right now is its own bounded eviction decision, and an
        ordinary lock must not decide who still owns a resume checkpoint.
        """
        chain: list[UnifiedTreeNode] = []
        cur: Optional[UnifiedTreeNode] = leaf
        root = self.tree_core.root_node
        while cur is not None and cur is not stop and cur is not root:
            chain.append(cur)
            cur = cur.parent
        pending = list(leaf.children.values())
        while pending:
            node = pending.pop()
            chain.append(node)
            pending.extend(node.children.values())
        return chain

    def _drop_abandoned_tail(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        stop: Optional[UnifiedTreeNode],
        reason: str,
    ) -> None:
        if stop is None or leaf is stop:
            return
        # Retire this session's pins, host leases, pending backup ownership and
        # leaf registration across the whole abandoned branch first, whatever
        # the host drop can reach. Another owner's pin or an ordinary lock keeps
        # the shared host data alive; it does not keep this session's obsolete
        # pin, its late backup callback or its marker in the session partition.
        for node in self._abandoned_chain(leaf, stop):
            self._release_own_pin(session_id, node)
        self._clear_session_leaf(session_id, leaf)
        path = self._tail_below(leaf, stop, session_id)
        if not path:
            return
        nodes = list(path)
        if path[0] is leaf:
            nodes.extend(self._descendants_until_block(leaf, session_id))
        dropped = self._tombstone_mamba_host(session_id, nodes)
        if not dropped:
            return
        used, total = self._mamba_host_slot_counts()
        logger.info(
            "mamba host drop reason=%s session=%s nodes=%s depth=%s used=%s total=%s stop=%s",
            reason,
            session_id,
            len(dropped),
            self._token_depth(leaf),
            used,
            total,
            stop.id,
        )

    def refresh_lru(
        self,
        phase: LRURefreshPhase,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
    ) -> None:
        # A match consumes only best_match_node's mamba state (cf. inc_lock_ref,
        # which locks just this node's mamba value), unlike Full whose whole matched
        # path is reused as prefix. Refreshing ancestors would keep a whole session's
        # states adjacent in the mamba LRU and evict cold sessions wholesale, so touch
        # only the used state. New leaf states enter the LRU via
        # commit_insert_component_data, so the insert walk (WALKDOWN) is a no-op here.
        ct = self.component_type
        match phase:
            case LRURefreshPhase.WALKDOWN:
                return
            case LRURefreshPhase.MATCH_END:
                if node.component_data[ct].value is not None:
                    self.tree_core.lru_lists[ct].reset_node_mru(node)
                host_lru = self.tree_core.host_lru_lists[ct]
                if node.component_data[ct].host_value is not None and host_lru.in_list(
                    node
                ):
                    host_lru.reset_node_mru(node)
            case LRURefreshPhase.INSERT_END:
                return
            case _:
                raise ValueError(f"Unknown LRURefreshPhase: {phase}")

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        ct = self.component_type
        if match_device_only:
            return lambda node: node.component_data[ct].value is not None

        # HiCache: evicted + backuped (host_value present) is also a valid match
        return lambda node: (
            node.component_data[ct].value is not None
            or node.component_data[ct].host_value is not None
        )

    def finalize_match_result_in_tree_core(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        last_node = result.best_match_node

        mamba_boundary_len = len(result.device_indices) + result.host_hit_length

        # Full KV may extend beyond the latest reusable Mamba state. The branching
        # point is the last Mamba-cache-chunk-aligned position within the Full-KV hit
        # that lies beyond the current Mamba boundary. With HiCache, incremental
        # persistence of a new branching state is currently write-through only;
        # write-back eviction may discard the device-only state.
        aligned_seqlen = (
            result.full_kv_hit_length // self.mamba_checkpoint_grid
        ) * self.mamba_checkpoint_grid
        branching_seqlen = (
            aligned_seqlen if aligned_seqlen > mamba_boundary_len else None
        )

        # HiCache: if mamba was evicted from device but has host backup,
        # ensure mamba_host_hit_length >= 1 so load_back is triggered.
        if self.has_host_value_only(last_node):
            result = result._replace(
                mamba_host_hit_length=max(result.mamba_host_hit_length, 1)
            )

        session_id = self._pin_session_id(params.req)
        if session_id is not None and last_node is not None:
            self._pin_resume(session_id, last_node, "match")

        return result._replace(mamba_branching_seqlen=branching_seqlen)

    def finalize_match_result_in_cache(
        self, params: MatchPrefixParams, result: MatchResult
    ) -> MatchResult:
        # Copy-on-write the matched device mamba state into a per-request slot.
        if not params.cow_mamba:
            return result
        src_index = self.tree_core.get_component_device_value(
            result.best_match_node, self.component_type
        )
        if src_index is None:
            return result
        req = params.req
        assert req is not None
        if req.mamba_pool_idx is None:
            dst_index = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
            if dst_index is None:
                # Pin the window via inc/dec_lock_ref so evict's SWA release
                # stops at this request's window boundary instead of walking to
                # root and over-decrementing locks held by other requests.
                lock_result = self.cache.inc_lock_ref(result.best_match_node)
                self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))
                dst_index = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
                self.cache.dec_lock_ref(
                    result.best_match_node, lock_result.to_dec_params()
                )
                assert dst_index is not None, "Can not alloc mamba cache"
            req.mamba_pool_idx = dst_index[0]
        req.mamba_cow_src_index = src_index
        req.mamba_needs_clear = False
        return result

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        assert params.mamba_value is not None
        if is_new_leaf:
            node.component_data[self.component_type].value = params.mamba_value
            self.tree_core.lru_lists[self.component_type].insert_mru(node)
            self.tree_core.component_evictable_size_[self.component_type] += len(
                params.mamba_value
            )
            self._emit_excess_path_states_eviction(node, cache_actions)
            return
        if node.component_data[self.component_type].value is None:
            node.component_data[self.component_type].value = params.mamba_value
            # move from host LRU to device LRU
            host_lru = self.tree_core.host_lru_lists[self.component_type]
            if host_lru.in_list(node):
                host_lru.remove_node(node)
            self.tree_core.lru_lists[self.component_type].insert_mru(node)
            self.tree_core.component_evictable_size_[self.component_type] += len(
                params.mamba_value
            )
            node.last_access_time = get_and_increase_time_counter()
            self._emit_excess_path_states_eviction(node, cache_actions)
            return
        self.tree_core.lru_lists[self.component_type].reset_node_mru(node)
        node.last_access_time = get_and_increase_time_counter()
        result.mamba_exist = True

    def _emit_excess_path_states_eviction(
        self,
        tail: UnifiedTreeNode,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        """Defer the path-cap eviction so it runs after the insert's BackupKV."""
        if self.mamba_max_states_per_path < 0:
            return
        cache_actions.append(MambaEvictExcessPathStates(tail.id))

    def _evict_excess_path_states(
        self,
        tail: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Evict shallow eligible device checkpoints beyond the path cap.

        Full KV and any existing host backup are retained. The tail, forks,
        locked nodes (including a pending backup chain's write-through locks),
        and device leaves are preserved, so the cap is a best-effort soft
        limit. Freed slots are collected into the caller's dicts.
        """
        cap = self.mamba_max_states_per_path
        if cap < 0:
            return

        ct = self.component_type
        holders = []
        node = tail
        while node is not None and node is not self.tree_core.root_node:
            if node.component_data[ct].value is not None:
                holders.append(node)
            node = node.parent

        excess = len(holders) - cap
        if excess <= 0:
            return

        tracker = {component: 0 for component in self.cache.tree_components}
        for node in reversed(holders):
            if excess <= 0 or node is tail:
                break
            if node.component_data[ct].lock_ref > 0 or len(node.children) != 1:
                continue
            if node in self.tree_core.evictable_device_leaves:
                continue
            self.tree_core._evict_component_and_detach_lru(
                node,
                self,
                device_frees,
                host_frees,
                target=EvictLayer.DEVICE,
                tracker=tracker,
            )
            self.tree_core._cascade_evict(node, self, tracker, device_frees, host_frees)
            excess -= 1

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        ct = self.component_type
        new_parent.component_data[ct].value = None
        new_parent.component_data[ct].lock_ref = 0
        new_parent.component_data[ct].session_ref = 0
        new_parent.component_data[ct].session_ids = None
        # HiCache: mamba host_value stays on child (mamba = leaf-only data)
        new_parent.component_data[ct].host_value = None
        new_parent.component_data[ct].host_lock_ref = 0

    def evict_component(
        self,
        node: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        cd = node.component_data[self.component_type]
        freed = 0
        host_freed = 0

        # Device layer
        if EvictLayer.DEVICE in target and cd.value is not None:
            device_frees[self.component_type].append(cd.value)
            freed = len(cd.value)
            self.tree_core.component_evictable_size_[self.component_type] -= freed
            cd.value = None

        # Host layer
        host_lru = self.tree_core.host_lru_lists[self.component_type]
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            host_frees[self.component_type].append(cd.host_value)
            cd.host_value = None
            if host_lru.in_list(node):
                host_lru.remove_node(node)

        # After device tombstone: if only host_value remains, insert into host LRU
        if (
            target is EvictLayer.DEVICE
            and cd.value is None
            and cd.host_value is not None
        ):
            sessions = self._resume_sessions_for(node)
            if sessions:
                for session_id in sessions:
                    self._host_lock_resume(
                        session_id, node, self._pin_label(session_id, node.id)
                    )
            elif not host_lru.in_list(node):
                host_lru.insert_mru(node)

        return freed, host_freed

    def _evict_device_start(self, request_cnt: int) -> None:
        """Begin the device-eviction walk from this component's LRU cursor."""
        self._evict_device_request_cnt = request_cnt
        if self.tree_core.enable_session_radix_cache:
            lru = self.tree_core.lru_lists[self.component_type]
            lru.cursor_begin()
            self._evict_device_cursor = lru.cursor_next()
        else:
            self._evict_device_cursor = self.tree_core.lru_lists[
                self.component_type
            ].get_lru_no_lock()

    def _evict_device_next_node(
        self,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> Optional[NodeId]:
        """Advance one device-eviction step and return a leaf, if selected.

        An internal tombstone is one complete step so the caller can apply its
        pending frees and recheck allocator capacity before the next mutation.
        If the previous node's eviction removed the cursor, the walk resumes
        from the partition sentinel with session refs on, else it restarts at
        the LRU tail.
        """
        ct = self.component_type
        lru = self.tree_core.lru_lists[ct]
        enabled = self.tree_core.enable_session_radix_cache
        if self._evict_device_cursor is not None and not lru.in_list(
            self._evict_device_cursor
        ):
            self._evict_device_cursor = (
                lru.cursor_next() if enabled else lru.get_lru_no_lock()
            )
        if (
            tracker[ct] >= self._evict_device_request_cnt
            or self._evict_device_cursor is None
            or not lru.in_list(self._evict_device_cursor)
        ):
            return None

        x = self._evict_device_cursor
        assert x.component_data[ct].value is not None
        if x in self.tree_core.evictable_device_leaves and (
            not enabled or self._can_evict_leaf_atomically(x)
        ):
            self._evict_device_cursor = (
                lru.cursor_next() if enabled else lru.get_prev_no_lock(x)
            )
            return x.id
        if not enabled:
            x_next = lru.get_prev_no_lock(x)
        self.tree_core._evict_component_and_detach_lru(
            x,
            self,
            target=EvictLayer.DEVICE,
            tracker=tracker,
            device_frees=device_frees,
            host_frees=host_frees,
        )
        self.tree_core._cascade_evict(
            x, self, tracker, device_frees=device_frees, host_frees=host_frees
        )
        self._evict_device_cursor = lru.cursor_next() if enabled else x_next
        return None

    def _evict_device_end(self) -> None:
        """Clear the device-eviction walk cursor state."""
        if self.tree_core.enable_session_radix_cache:
            self.tree_core.lru_lists[self.component_type].cursor_end()
        self._evict_device_cursor = None

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        ct = self.component_type
        if node is self.tree_core.root_node:
            return result
        cd = node.component_data[ct]
        value = cd.host_value if lock_host else cd.value
        # A node in skip_lock_node_ids was a tombstone when this lock was acquired.
        if value is None:
            result.skip_lock_node_ids.setdefault(ct, set()).add(node.id)
            return result

        if lock_host:
            if cd.host_lock_ref == 0:
                host_lru = self.tree_core.host_lru_lists[ct]
                if host_lru.in_list(node):
                    host_lru.remove_node(node)
            cd.host_lock_ref += 1
        else:
            if cd.lock_ref == 0:
                vlen = len(value)
                self.tree_core.component_evictable_size_[ct] -= vlen
                self.tree_core.component_protected_size_[ct] += vlen
            cd.lock_ref += 1
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:
        ct = self.component_type
        if node is self.tree_core.root_node:
            return
        cd = node.component_data[ct]
        skip_lock_node_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()
        if node.id in skip_lock_node_ids:
            return

        value = cd.host_value if lock_host else cd.value
        if lock_host:
            cd.host_lock_ref -= 1
            if cd.host_lock_ref == 0 and cd.value is None and cd.host_value is not None:
                host_lru = self.tree_core.host_lru_lists[ct]
                if not host_lru.in_list(node):
                    host_lru.insert_mru(node)
            return

        if cd.lock_ref > 0:
            if cd.lock_ref == 1:
                vlen = len(value)
                self.tree_core.component_evictable_size_[ct] += vlen
                self.tree_core.component_protected_size_[ct] -= vlen
            cd.lock_ref -= 1

    def _alloc_mamba_slot(self) -> torch.Tensor:
        """Allocate one mamba pool slot, evicting if necessary."""
        slot = self._try_alloc_mamba_slot()
        assert slot is not None, "Can not alloc mamba cache"
        return slot

    def _try_alloc_mamba_slot(self) -> Optional[torch.Tensor]:
        """Allocate one slot after eviction, or return None if none is evictable."""
        slot = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
        if slot is None:
            self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))
            slot = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
        return slot

    _last_skip_log_time = 0.0

    def _log_skipped_checkpoint(self, req: Req) -> None:
        now = time.monotonic()
        if now - MambaComponent._last_skip_log_time < 1.0:
            return
        MambaComponent._last_skip_log_time = now
        allocator = self.cache.req_to_token_pool.mamba_allocator
        logger.info(
            "mamba checkpoint skipped for rid=%s: no free or evictable slot "
            "(available=%d evictable=%d of %d)",
            req.rid,
            allocator.available_size(),
            self.tree_core.mamba_evictable_size(),
            getattr(allocator, "size", -1),
        )

    @property
    def int8_ckpt_pool(self):
        return getattr(self.cache.req_to_token_pool, "mamba_ckpt_pool", None)

    def _alloc_int8_ckpt_slot(self) -> torch.Tensor:
        slot = self.int8_ckpt_pool.alloc(1)
        if slot is None:
            self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
            slot = self.int8_ckpt_pool.alloc(1)
            assert slot is not None, "Can not alloc int8 mamba checkpoint slot"
        return slot

    def _commit_int8_checkpoint(self, active_slots: torch.Tensor) -> torch.Tensor:
        ckpt_slot = self._alloc_int8_ckpt_slot()
        self.int8_ckpt_pool.store_from_active(
            self.cache.req_to_token_pool.mamba_pool,
            active_slots.view(-1),
            ckpt_slot,
        )
        return ckpt_slot

    def _free_mamba_value(self, mamba_value: torch.Tensor) -> None:
        if self.int8_ckpt_pool is not None:
            self.int8_ckpt_pool.free(mamba_value)
        else:
            self.cache.req_to_token_pool.mamba_allocator.free(mamba_value)

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        if self.cache.enable_mamba_extra_buffer:
            cache_len = req.mamba_last_track_seqlen
        else:
            cache_len = token_ids_len
            # ReplaySSM (no_buffer): `temporal[slot]` lags the live state by the
            # slot's unflushed ring depth (`write_pos`), so on request finish cap
            # the donate to the last flush boundary (where temporal is current)
            # and reset the cursor, keeping the donated checkpoint consistent with
            # its key length. page_size is asserted == 1, so no realign. Mirrors
            # MambaRadixCache.cache_finished_req.
            if is_finished:
                write_pos_buf = (
                    self.cache.req_to_token_pool.mamba_pool.replayssm_write_pos
                )
                if write_pos_buf is not None:
                    cache_len -= int(write_pos_buf[req.mamba_pool_idx].item())
                    write_pos_buf[req.mamba_pool_idx] = 0

        if is_finished:
            if cache_len is None:
                cache_len = 0
            if self.cache.enable_mamba_extra_buffer:
                keep_idx = self.cache.req_to_token_pool.get_mamba_ping_pong_keep_idx(
                    req
                )
                active_value = (
                    req.mamba_ping_pong_track_buffer[keep_idx].unsqueeze(-1).clone()
                )
            else:
                active_value = req.mamba_pool_idx.unsqueeze(-1).clone()
            if self.int8_ckpt_pool is not None:
                insert_params.mamba_value = self._commit_int8_checkpoint(active_value)
            else:
                insert_params.mamba_value = active_value
            return cache_len
        else:
            if cache_len is None:
                return 0
            # An unfinished-request checkpoint is an optimization. If every
            # slot is protected (including by an in-flight host backup), keep
            # the request's live state and retry at the next chunk boundary.
            if self.int8_ckpt_pool is not None:
                if self.cache.enable_mamba_extra_buffer:
                    new_slot = self._try_alloc_mamba_slot()
                    if new_slot is None:
                        self._log_skipped_checkpoint(req)
                        return 0
                    src_active = (
                        self.cache.req_to_token_pool.donate_mamba_ping_pong_slot(
                            req, new_slot
                        )
                    )
                    mamba_value_donated = self._commit_int8_checkpoint(src_active)
                    self.cache.req_to_token_pool.mamba_allocator.free(src_active)
                else:
                    mamba_value_donated = self._commit_int8_checkpoint(
                        req.mamba_pool_idx.view(-1)
                    )
            elif self.cache.enable_mamba_extra_buffer:
                new_slot = self._try_alloc_mamba_slot()
                if new_slot is None:
                    self._log_skipped_checkpoint(req)
                    return 0
                mamba_value_donated = (
                    self.cache.req_to_token_pool.donate_mamba_ping_pong_slot(
                        req, new_slot
                    )
                )
            else:
                mamba_value_donated = self._try_alloc_mamba_slot()
                if mamba_value_donated is None:
                    self._log_skipped_checkpoint(req)
                    return 0
                # mamba_pool is a pure PHYSICAL store; translate both slot ids
                # virtual->physical (identity for the non-unified memory pool) first.
                translate = self.cache.req_to_token_pool.translate_mamba_indices
                self.cache.req_to_token_pool.mamba_pool.copy_from(
                    translate(req.mamba_pool_idx.unsqueeze(0)),
                    translate(mamba_value_donated),
                )
            insert_params.mamba_value = mamba_value_donated
            return cache_len

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: Optional[InsertResult] = None,
        insert_params: Optional[InsertParams] = None,
    ) -> None:
        self._note_inserted_resume(req, insert_result)
        if is_finished:
            mamba_value_inserted = (
                insert_result is not None and not insert_result.mamba_exist
            )
            pool = self.cache.req_to_token_pool

            if self.int8_ckpt_pool is not None:
                insert_value_unused = (
                    not mamba_value_inserted
                    and insert_params is not None
                    and insert_params.mamba_value is not None
                )
                if insert_value_unused:
                    self._free_mamba_value(insert_params.mamba_value)
                pool.free_mamba_cache(req)
                return

            if self.cache.enable_mamba_extra_buffer:
                keep_idx = (
                    pool.get_mamba_ping_pong_keep_idx(req)
                    if mamba_value_inserted
                    else None
                )
                pool.free_mamba_cache(
                    req, mamba_ping_pong_track_buffer_to_keep=keep_idx
                )
                return

            if not mamba_value_inserted:
                pool.free_mamba_cache(req)
        else:
            if insert_params.mamba_value is not None and (
                insert_result is None or insert_result.mamba_exist
            ):
                self._free_mamba_value(insert_params.mamba_value)
            req.mamba_last_track_seqlen = None

    # ---- HiCache Hooks ----

    def prepare_load_back(
        self,
        node_id: NodeId,
        *,
        req: Optional[Req] = None,
    ) -> PrepareLoadBackResult:
        if (
            req is None
            or req.mamba_pool_idx is not None
            or not self.tree_core.component_has_host_value_only(
                node_id, self.component_type
            )
        ):
            return PrepareLoadBackResult()
        dst = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
        if dst is None:
            self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))
            dst = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
            assert dst is not None, "Cannot alloc mamba for load_back"
        req.mamba_pool_idx = dst[0]
        return PrepareLoadBackResult(allocated_mamba_slot=dst)

    def finalize_load_back(
        self, req: Optional[Req], prep: PrepareLoadBackResult, success: bool
    ) -> None:
        # A called-off load-back returns the slot prepare allocated and clears req (the H->D copy never ran).
        if not success and prep.allocated_mamba_slot is not None:
            self.cache.req_to_token_pool.mamba_allocator.free(prep.allocated_mamba_slot)
            req.mamba_pool_idx = None

    def prepare_prefetch(
        self,
        node_id: NodeId,
        *,
        prefetch_tokens: int = 0,
    ) -> PreparePrefetchResult:
        host_indices = self.cache.host_pool_group.alloc(
            1,
            pool=PoolName.MAMBA,
            reclaim=lambda size: self.cache.evict_host(size, ComponentType.MAMBA),
        )
        if host_indices is None:
            return PreparePrefetchResult(alloc_failed=True)
        return PreparePrefetchResult(host_indices=host_indices)

    def build_hicache_transfers(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        *,
        mamba_pool_idx: Optional[torch.Tensor] = None,
        host_indices: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            cd = node.component_data[ct]
            if cd.value is None:
                return None
            return [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    device_indices=cd.value,
                )
            ]

        if phase == CacheTransferPhase.LOAD_BACK:
            transfers: list[PoolTransfer] = []

            cd = node.component_data[ct]
            if cd.value is not None:
                return None

            # restore single node if host_value exists
            if cd.host_value is not None and cd.value is None:
                transfers.append(
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        host_indices=cd.host_value,
                        nodes_to_load=[node.id],
                    )
                )

            # Per-request mamba CoW: H→D copy into the request's device slot pre-allocated on the caller side.
            cd = node.component_data[ct]
            if mamba_pool_idx is not None and cd.host_value is not None:
                transfers.append(
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        host_indices=cd.host_value,
                        device_indices=mamba_pool_idx.unsqueeze(0),
                    )
                )

            return transfers if transfers else None

        if phase == CacheTransferPhase.BACKUP_STORAGE:
            cd = node.component_data[ct]
            if cd.host_value is None or not node.hash_value:
                return None
            return [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    host_indices=cd.host_value,
                    keys=[node.hash_value[-1]],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]

        if phase == CacheTransferPhase.PREFETCH:
            assert host_indices is not None
            return [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    host_indices=host_indices,
                    keys=["__placeholder__"],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]

        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
        *,
        cache_actions: list[CacheAction | ComponentAction],
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            if transfers and transfers[0].host_indices is not None:
                cd = node.component_data[ct]
                if cd.host_value is None:
                    cd.host_value = transfers[0].host_indices.clone()
                self._promote_pending_backup(node, durable=cd.host_value is not None)

        elif phase == CacheTransferPhase.LOAD_BACK:
            if not transfers:
                return
            transfer = transfers[0]
            if transfer.device_indices is not None:
                cd = node.component_data[ct]
                cd.value = transfer.device_indices.clone()
                count = len(cd.value)
                # Move from host LRU to device LRU
                host_lru = self.tree_core.host_lru_lists[ct]
                if host_lru.in_list(node):
                    host_lru.remove_node(node)
                self.tree_core.lru_lists[ct].insert_mru(node)
                self.tree_core.component_evictable_size_[ct] += count

        elif phase == CacheTransferPhase.PREFETCH:
            if not transfers:
                return
            transfer = transfers[0]
            host_indices = transfer.host_indices
            loaded = (
                pool_storage_result is not None
                and pool_storage_result.extra_pool_hit_pages.get(PoolName.MAMBA, 0) >= 1
            )
            target_node = (
                self.tree_core.node_by_id(insert_result.inserted_host_node)
                if insert_result is not None
                and insert_result.inserted_host_node is not None
                else None
            )
            if (
                host_indices is None
                or target_node is None
                or not loaded
                or target_node.component_data[ct].host_value is not None
            ):
                cache_actions.append(
                    FreeComponentHostSlot(
                        [host_indices], component_type=ComponentType.MAMBA
                    )
                )
                if insert_result is not None:
                    insert_result.mamba_exist = True
                return

            target_node.component_data[ct].host_value = host_indices.clone()
            if target_node.component_data[ct].value is None:
                host_lru = self.tree_core.host_lru_lists[ct]
                if not host_lru.in_list(target_node):
                    host_lru.insert_mru(target_node)
            if insert_result is not None:
                insert_result.mamba_exist = False

    def _token_depth(self, node: UnifiedTreeNode) -> int:
        depth = 0
        root = self.tree_core.root_node
        cur = node
        while cur is not None and cur is not root:
            if cur.key is not None:
                depth += len(cur.key)
            cur = cur.parent
        return depth

    def _path_tail_depth(self, node: UnifiedTreeNode) -> int:
        cur = node
        while len(cur.children) == 1:
            cur = next(iter(cur.children.values()))
        return self._token_depth(cur)

    def _mamba_host_slot_counts(self) -> tuple[int, int]:
        pool = self.cache.host_pool_group.get_pool(PoolName.MAMBA)
        total = int(pool.size)
        return total - int(pool.available_size()), total

    def drive_host_eviction(
        self,
        num_tokens: int,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Evict mamba host resources.
        Internal nodes: private tombstone (free host mamba only).
        Host leaves: atomic eviction via _evict_host_leaf."""
        ct = self.component_type
        host_lru = self.tree_core.host_lru_lists[ct]
        while tracker[ct] < num_tokens:
            x, kind = self._select_host_eviction_candidate(host_lru)
            if x is None:
                break
            used, total = self._mamba_host_slot_counts()
            logger.info(
                "mamba host reclaim kind=%s depth=%s tail=%s children=%s "
                "session_ref=%s host_lock=%s lock=%s node=%s used=%s total=%s",
                kind,
                self._token_depth(x),
                self._path_tail_depth(x),
                len(x.children),
                self.session_ref(x),
                x.component_data[ct].host_lock_ref,
                x.component_data[ct].lock_ref,
                x.id,
                used,
                total,
            )
            cd = x.component_data[ct]
            if x in self.tree_core.evictable_host_leaves and (
                not self.tree_core.enable_session_radix_cache
                or self._can_evict_leaf_atomically(x)
            ):
                # Host leaf: atomic eviction (all components host + delete)
                self.tree_core._evict_host_leaf(x, tracker, device_frees, host_frees)
            else:
                # Internal (or a leaf a session still pins): tombstone Mamba + cascade
                assert cd.host_value is not None
                self.tree_core._evict_component_and_detach_lru(
                    x,
                    self,
                    target=EvictLayer.HOST,
                    tracker=tracker,
                    device_frees=device_frees,
                    host_frees=host_frees,
                )
                self.tree_core._cascade_evict(
                    x,
                    self,
                    tracker,
                    device_frees=device_frees,
                    host_frees=host_frees,
                    target=EvictLayer.HOST,
                )
                self.tree_core._update_evictable_leaf_sets(x)

    def _select_host_eviction_candidate(self, host_lru):
        """Prefer a redundant linear-path checkpoint within the LRU partition.

        A later complete component-consensus boundary contains the recurrent
        state needed to resume that longer prefix; its intermediate Mamba
        checkpoints are not part of the restore unit. Prefer the oldest such
        checkpoint before falling back to the ordinary oldest unlocked entry.
        Branch points are never classified as redundant because their state may
        still be the only reusable boundary for another branch.
        """
        enabled = self.tree_core.enable_session_radix_cache
        fallback = None
        fallback_partition = None
        selected = None
        cursor_started = False
        try:
            if enabled:
                host_lru.cursor_begin()
                cursor_started = True
                node = host_lru.cursor_next(host_lock=True)
            else:
                node = host_lru.get_lru_no_host_lock()

            while node is not None and host_lru.in_list(node):
                partition = self.session_ref(node) > 0 if enabled else None
                if fallback is None:
                    fallback = node
                    fallback_partition = partition
                elif partition != fallback_partition:
                    break

                if self._has_later_complete_boundary(node):
                    selected = node
                    break
                node = (
                    host_lru.cursor_next(host_lock=True)
                    if enabled
                    else host_lru.get_prev_no_host_lock(node)
                )
        finally:
            if cursor_started:
                host_lru.cursor_end()
        chosen = selected if selected is not None else fallback
        if chosen is None:
            return None, None
        if selected is not None:
            return chosen, "interior"
        cd = chosen.component_data[self.component_type]
        if chosen in self.tree_core.evictable_host_leaves:
            kind = "leaf"
        elif cd.host_lock_ref > 0 or cd.lock_ref > 0:
            kind = "locked"
        elif len(chosen.children) != 1:
            kind = "fork"
        else:
            kind = "fallback"
        return chosen, kind

    def _has_later_complete_boundary(self, node: UnifiedTreeNode) -> bool:
        """Whether a single-child continuation has a later reusable boundary."""
        root = self.tree_core.root_node
        validators = tuple(
            component.create_match_validator()
            for component in self.tree_core.components
        )

        # Stateful validators (notably SWA) must observe the same root-to-node
        # history they would see during match_prefix before examining descendants.
        ancestors = []
        cur = node
        while cur is not root:
            ancestors.append(cur)
            cur = cur.parent
        for ancestor in reversed(ancestors):
            if ancestor.evicted and not ancestor.backuped:
                return False
            for validator in validators:
                validator(ancestor)

        cur = node
        while len(cur.children) == 1:
            child = next(iter(cur.children.values()))
            # Match traversal cannot cross a Full-KV hole, regardless of aux state.
            if child.evicted and not child.backuped:
                return False
            if all([validator(child) for validator in validators]):
                return True
            cur = child
        return False

    def free_host_values(self, host_values: list[torch.Tensor]) -> None:
        if self._mamba_pool_host is None:
            return
        for host_value in host_values:
            self.cache.host_pool_group.free(host_value, pool=PoolName.MAMBA)

    def apply_component_action(self, action: ComponentAction) -> None:
        if isinstance(action, MambaEvictExcessPathStates):
            device_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)
            host_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)
            # Drain even if the walk raises so tombstoned slots are not leaked;
            # the walk runs behind the tree-core interface (Rust runs it natively).
            try:
                self.tree_core.evict_excess_path_states(
                    action.tail_node_id, device_frees, host_frees
                )
            finally:
                self.cache._free_values(device_frees, host_frees)
            return
        if isinstance(action, FreeComponentDeviceSlot):
            for indices in action.indices:
                self._free_mamba_value(indices)
            return
        if isinstance(action, FreeComponentHostSlot):
            for host_indices in action.host_indices:
                if host_indices is not None and host_indices.numel() > 0:
                    self.cache.cache_controller.append_host_mem_release(
                        extra_pools=[
                            PoolTransfer(name=PoolName.MAMBA, host_indices=host_indices)
                        ]
                    )
            return
        raise AssertionError(
            f"MambaComponent: unhandled ComponentAction {type(action).__name__}"
        )
