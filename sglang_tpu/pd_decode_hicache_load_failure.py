"""Clean up a PD decode HiCache load-back whose raiden DMA failed.

A failed host-to-device load leaves the restored device pages holding stale
bytes, while the tree already lists them as the node's device copy
(``commit_load_back`` runs before the DMA). Without this, the first poll of the
load raises and stops the server.

On a PD decode server (the only one that restores into the decode radix tree):

* ``TpuRaidenHybridCacheController.resolve_load_back`` settles, over every rank,
  whether a load failed; a DMA that failed on one rank fails the load on all.
* The restore poll marks the requests bound to a failed load FAILED instead of
  READY, so the decode transfer queue aborts them through its usual failure
  path, which also drops their lock on the restored node.
* ``loading_check`` holds a failed load until only the load itself still locks
  its nodes, lets upstream ack it (dropping the load's locks), then demotes the
  nodes it brought back, deepest first: the device pages are freed and the host
  copy stays cached. A node that another request still holds or that gained a
  device child cannot be demoted safely; that raises, as a failed load did.

Every step is decided from rank-combined state or from the tree, which is the
same on every rank, so all ranks make the same changes. Freeing a failed load's
pages relies on raiden reporting a failed DMA only after all of its copies have
ended (google/tpu-sync#1228) and on ``CombinedFutureEvent`` waiting for every
pool's DMA. Prefill servers are unchanged: a failed load still stops the server.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCacheTransferMixin,
    HiCacheRestoreResult,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang_tpu.raiden_hybrid_cache_controller import (
    LoadBackEvent,
    TpuRaidenHybridCacheController,
)

logger = logging.getLogger(__name__)

_is_installed = False


def _raiden_controller(tree_cache):
    controller = getattr(tree_cache, "cache_controller", None)
    if isinstance(controller, TpuRaidenHybridCacheController):
        return controller
    return None


def _loaded_nodes(tree_core) -> Dict[int, List[int]]:
    """Anchor node id -> the nodes its load-back brings back, ancestors first."""
    return tree_core.__dict__.setdefault("_tpu_load_back_nodes", {})


def _fail_restores_of_failed_loads(transfer_queue, decode_reqs) -> None:
    """A restore whose load failed is FAILED, not READY."""
    counter = transfer_queue.tree_cache.cache_controller.layer_done_counter
    for dr in decode_reqs:
        event = getattr(dr, "_tpu_load_back_event", None)
        if (
            event is None
            and dr.hicache_restored_node is not None
            and dr.hicache_load_consumer_index >= 0
        ):
            # Bound in this poll's phase C: the slot holds this load's event.
            event = counter.events[dr.hicache_load_consumer_index]
            dr._tpu_load_back_event = event
        if (
            dr.hicache_restore_status == HiCacheRestoreResult.READY
            and isinstance(event, LoadBackEvent)
            and event.resolved
            and event.failed
        ):
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED


def _held_only_by_the_load(tree_core, node_id: int) -> bool:
    lock_ref = tree_core._binding.inspect_get_component_device_lock_ref(
        node_id, int(ComponentType.FULL)
    )
    return lock_ref == 1


def _release_failed_loads(cache, controller) -> None:
    """Settle the head load acks; let a failed one be acked once only it holds
    its nodes. A collective: every rank walks the same acks and stops at the
    same one."""
    loaded = _loaded_nodes(cache.tree_core)
    for ack in controller.ack_load_queue:
        event = ack.finish_event
        if not controller.resolve_load_back(event, cache._all_reduce):
            return
        if not isinstance(event, LoadBackEvent) or not event.failed:
            continue
        if event.failure_handled:
            continue
        if not hasattr(cache.tree_core, "_binding"):
            # Only the Rust tree core can tell whether the nodes are still in use.
            raise event.local_error or RuntimeError(
                "raiden HiCache load-back failed on another rank"
            )
        nodes = [n for anchor in ack.node_ids for n in loaded.get(anchor, [anchor])]
        if not all(_held_only_by_the_load(cache.tree_core, n) for n in nodes):
            return  # a request still holds them; it is aborted on this poll
        event.failure_handled = True


def _roll_back(cache, anchor: int, nodes: List[int]) -> None:
    """Demote the nodes a failed load brought back, deepest first."""
    binding = cache.tree_core._binding
    tracker: dict = {}
    for node_id in reversed(nodes):
        if not binding.inspect_is_device_evictable_leaf(node_id):
            raise RuntimeError(
                f"raiden HiCache load-back of node {anchor} failed and node "
                f"{node_id} is still in use, so its device copy cannot be "
                "dropped"
            )
        cache._demote(node_id, tracker)
    logger.warning(
        "raiden HiCache load-back of node %d failed: %d node(s) rolled back to "
        "host-only",
        anchor,
        len(nodes),
    )


def _loading_check_with_cleanup(orig_loading_check, cache, finish_count) -> None:
    """Upstream ``loading_check``, with a failed load held, acked, then rolled back."""
    controller = cache.cache_controller
    _release_failed_loads(cache, controller)
    pending = list(controller.ack_load_queue)
    orig_loading_check(cache, finish_count)
    # Upstream acks loads from the head of the queue.
    acked = pending[: len(pending) - len(controller.ack_load_queue)]
    loaded = _loaded_nodes(cache.tree_core)
    for ack in acked:
        failed = isinstance(ack.finish_event, LoadBackEvent) and ack.finish_event.failed
        for anchor in ack.node_ids:
            nodes = loaded.pop(anchor, None)
            if failed:
                _roll_back(cache, anchor, nodes or [anchor])
    # Loads that never started leave a record behind; keep only live ones.
    for anchor in [a for a in loaded if a not in cache.ongoing_load_back]:
        del loaded[anchor]


def install() -> None:
    """Patch the decode restore poll, ``loading_check`` and load-back recording."""
    global _is_installed
    if _is_installed:
        return
    _is_installed = True

    orig_process = DecodeHiCacheTransferMixin._process_hicache_local_restores
    orig_loading_check = UnifiedRadixCache.loading_check

    def _process_hicache_local_restores(self, decode_reqs):
        controller = _raiden_controller(self.tree_cache)
        if controller is None:
            return orig_process(self, decode_reqs)
        # Set before the poll's phase C issues a load, so its event can fail
        # without raising and its nodes are recorded for the rollback.
        controller.handles_load_back_failures = True
        self.tree_cache.tree_core.__dict__["_tpu_record_load_back_nodes"] = True
        orig_process(self, decode_reqs)
        _fail_restores_of_failed_loads(self, decode_reqs)

    def loading_check(self, finish_count=None):
        controller = _raiden_controller(self)
        if controller is None or not controller.handles_load_back_failures:
            return orig_loading_check(self, finish_count)
        _loading_check_with_cleanup(orig_loading_check, self, finish_count)

    DecodeHiCacheTransferMixin._process_hicache_local_restores = (
        _process_hicache_local_restores
    )
    UnifiedRadixCache.loading_check = loading_check
    _record_load_back_nodes()


def _record_load_back_nodes() -> None:
    """Record, per load-back anchor, the nodes its load brings back.

    Only the Rust tree core can roll a failed load back (it alone reports
    whether a node is still in use); importing it builds its extension, so a
    server without it keeps a failed load fail-stop.
    """
    try:
        from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore
    except (ImportError, RuntimeError) as e:
        logger.info("Rust tree core unavailable (%s); failed load-backs stay fatal", e)
        return
    orig_build_load_back_spec = RustUnifiedTreeCore.build_load_back_spec

    def build_load_back_spec(self, node_id, req=None):
        kv_xfer, comp_xfers = orig_build_load_back_spec(self, node_id, req=req)
        if self.__dict__.get("_tpu_record_load_back_nodes"):
            _loaded_nodes(self)[node_id] = list(kv_xfer.nodes_to_load or [node_id])
        return kv_xfer, comp_xfers

    RustUnifiedTreeCore.build_load_back_spec = build_load_back_spec
