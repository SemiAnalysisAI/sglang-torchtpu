"""write_back must not drop a node's SWA device slots when it has no host copy.

`SWAComponent._evict_device_next_node` returns evictable device leaves to the
caller, which stages them to host and demotes them. For an *internal* node it
instead tombstones the SWA device slots in place, via
`_evict_component_and_detach_lru(..., EvictLayer.DEVICE)`.

Under `write_through` that is free: the host copy was taken while the page was
still resident. Under `write_back` the host copy is only taken when the *node*
is evicted, and an internal node never reaches that path -- so the tombstone
destroys the only copy. The prefix then has SWA on neither tier, the stateful
SWA match validator rejects it, `best_match_node` never advances, and no
load_back is issued. Nothing logs a warning; reuse silently drops to zero.

These tests pin the invariant rather than the symptom: no path may release a
component's device value under write_back while its host value is None.
"""

import unittest
from collections import defaultdict
from unittest import mock

import torch

from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    ComponentType,
    UnifiedTreeNode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CT = ComponentType.SWA
COMPONENTS = (ComponentType.FULL, ComponentType.SWA)


class _SingleNodeLru:
    """Minimal LRU holding one node; the walk only needs these four calls."""

    def __init__(self, node):
        self._node = node

    def in_list(self, node):
        return node is self._node

    def get_lru_no_lock(self):
        return self._node

    def get_prev_no_lock(self, node):
        return None

    def cursor_next(self):
        return None


def _run_walk(*, is_write_back: bool, has_host_copy: bool):
    """Drive one SWA eviction step over a node that is NOT a device leaf.

    Returns the mocked tree_core so the caller can assert whether the in-place
    tombstone (`_evict_component_and_detach_lru`) was taken.
    """
    from sglang.srt.mem_cache.unified_cache.components.swa import (
        SWAComponent,
    )

    node = UnifiedTreeNode(COMPONENTS)
    cd = node.component_data[CT]
    cd.value = torch.arange(8, dtype=torch.int64)
    cd.host_value = torch.arange(8, dtype=torch.int64) if has_host_copy else None

    core = mock.Mock()
    core.lru_lists = {CT: _SingleNodeLru(node)}
    core.enable_session_radix_cache = False
    # Empty on purpose: an internal node, so the walk takes the tombstone path
    # rather than returning the node for a proper staged eviction.
    core.evictable_device_leaves = set()
    core.is_write_back = is_write_back

    comp = SWAComponent.__new__(SWAComponent)
    comp.component_type = CT
    comp.tree_core = core
    comp._evict_device_cursor = node
    comp._evict_device_request_cnt = 1

    node_id = comp._evict_device_next_node(
        defaultdict(int), defaultdict(list), defaultdict(list)
    )
    return core, node, node_id


class TestWriteBackUnbackedSwaTombstone(CustomTestCase):
    # Open defect: this fails on main today. Marked xfail so it documents the
    # bug without reddening CI. When the fix lands pytest reports an UNEXPECTED
    # SUCCESS -- remove this marker then.
    @unittest.expectedFailure
    def test_write_back_does_not_tombstone_swa_without_a_host_copy(self):
        """The defect: write_back drops SWA device slots that have no host copy."""
        core, node, node_id = _run_walk(is_write_back=True, has_host_copy=False)

        core._evict_component_and_detach_lru.assert_not_called()
        self.assertIsNotNone(
            node.component_data[CT].value,
            "write_back released SWA device slots for a node with no host copy; "
            "the node is not a device leaf, so it never reaches the backup path "
            "and the prefix becomes silently unreloadable",
        )
        self.assertIsNone(node_id)

    def test_write_back_may_tombstone_swa_that_is_already_on_host(self):
        """Reclaim is still allowed once the host copy exists."""
        core, _, _ = _run_walk(is_write_back=True, has_host_copy=True)
        core._evict_component_and_detach_lru.assert_called_once()

    def test_write_through_tombstones_unbacked_swa_as_before(self):
        """write_through copies while resident, so this path stays untouched."""
        core, _, _ = _run_walk(is_write_back=False, has_host_copy=False)
        core._evict_component_and_detach_lru.assert_called_once()


if __name__ == "__main__":
    unittest.main()
