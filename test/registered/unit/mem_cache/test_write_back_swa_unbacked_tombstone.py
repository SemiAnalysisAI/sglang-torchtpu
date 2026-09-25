"""Under write_back, SWA eviction must not release an internal node's SWA
device slots before they have a host copy."""

import atexit
import unittest
from array import array
from unittest import mock

import torch

import sglang.srt.mem_cache.pool_host.base as pool_host_base
import sglang.srt.mem_cache.pool_host.common as pool_host_common
import sglang.srt.mem_cache.pool_host.mha as pool_host_mha
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InsertParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SWA = ComponentType.SWA
FULL_LAYER_IDS = [3]
SWA_LAYER_IDS = [0, 1, 2]
KV_SIZE = 256

# Nested inserts build root -> [1, 2] -> [3, 4] -> [5, 6]: two internal nodes
# and one device leaf.
PREFIXES = ([1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6])


def _transfer_kv_direct_cpu(
    src_layers, dst_layers, src_indices, dst_indices, page_size
):
    assert page_size == 1
    for src, dst in zip(src_layers, dst_layers, strict=True):
        dst[dst_indices] = src[src_indices]


class TestWriteBackUnbackedSwaTombstone(CustomTestCase):
    def setUp(self):
        # Host pools pin their buffers with cudaHostRegister, which needs a CUDA
        # driver. Pinning only affects copy speed, not what eviction releases.
        patcher = mock.patch.object(pool_host_common, "_cuda_host_register")
        patcher.start()
        self.addCleanup(patcher.stop)
        # The budget subtracts a fixed reserve from free RAM, which goes negative
        # on small CI runners; these pools need far less than the stubbed 1 GB.
        patcher = mock.patch.object(
            pool_host_base, "host_memory_budget_bytes", return_value=1024**3
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        # The layer-first direct D2H copy is a CUDA kernel; this is its CPU form.
        patcher = mock.patch.object(
            pool_host_mha,
            "transfer_kv_direct",
            _transfer_kv_direct_cpu,
            create=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _build_cache(self, write_policy: str):
        """Returns the cache and its SWA LRU tail, an internal device-only node."""
        override = get_context().override_server_args(
            page_size=1,
            hicache_io_backend="direct",
            hicache_mem_layout="layer_first",
            hicache_write_policy=write_policy,
        )
        server_args = override.install()
        self.addCleanup(override.restore)

        kv_pool = SWAKVPool(
            size=KV_SIZE,
            size_swa=KV_SIZE,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=64,
            swa_attention_layer_ids=SWA_LAYER_IDS,
            full_attention_layer_ids=FULL_LAYER_IDS,
            device="cpu",
        )
        allocator = SWATokenToKVPoolAllocator(
            size=KV_SIZE,
            size_swa=KV_SIZE,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        params = CacheInitParams(
            req_to_token_pool=ReqToTokenPool(
                size=4, max_context_len=64, device="cpu", enable_memory_saver=False
            ),
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=False,
            sliding_window_size=128,
            tree_components=(ComponentType.FULL, SWA),
        )
        cache = UnifiedRadixCache(params=params)
        cache.init_hicache(server_args, params)
        self.addCleanup(atexit.unregister, cache.shutdown)
        self.addCleanup(cache.release_host_resources)
        # No backup on insert or hit, so neither policy starts with a host copy.
        cache.write_through_threshold = 1 << 30

        for tokens in PREFIXES:
            cache.insert(
                InsertParams(
                    key=RadixKey(array("q", tokens)),
                    value=allocator.alloc(len(tokens)),
                )
            )
        # Read the target off the LRU: a match_prefix lookup would refresh it.
        node = cache.tree_core.lru_lists[SWA].get_lru_no_lock()
        swa = node.component_data[SWA]
        self.assertEqual(cache.tree_core.is_write_back, write_policy == "write_back")
        self.assertTrue(node.children)
        self.assertNotIn(node, cache.tree_core.evictable_device_leaves)
        self.assertIsNotNone(swa.value)
        self.assertIsNone(swa.host_value)
        return cache, node

    def _evict_swa_of(self, cache, node) -> EvictResult:
        """Request exactly the node's SWA size, so the walk stops after it."""
        return cache.evict(EvictParams(swa_num_tokens=len(node.key)))

    def test_write_back_frees_swa_without_losing_it(self):
        """write_back must free the requested SWA slots and keep the SWA on a tier."""
        cache, node = self._build_cache("write_back")

        result = self._evict_swa_of(cache, node)

        swa = node.component_data[SWA]
        # Open defect: unwrap once write_back backs up an internal node's SWA first.
        # CustomTestCase re-raises errors as Exception; expectedFailure hides crashes.
        with self.assertRaises(AssertionError):
            self.assertTrue(
                swa.value is not None or swa.host_value is not None,
                "write_back released SWA device slots that had no host copy",
            )
        # Skipping the node without freeing SWA elsewhere starves SWA allocation.
        self.assertGreaterEqual(result.swa_num_tokens_evicted, len(node.key))

    def test_write_back_may_release_swa_that_is_already_on_host(self):
        """Guards against a fix that stops write_back SWA eviction altogether."""
        cache, node = self._build_cache("write_back")
        cache._execute_and_commit_kv_backup(
            BackupKV(node_ids=[node.id]), write_back=True
        )
        cache.writing_check(write_back=True)
        self.assertIsNotNone(node.component_data[SWA].host_value)

        self._evict_swa_of(cache, node)

        swa = node.component_data[SWA]
        self.assertIsNone(swa.value)
        self.assertIsNotNone(swa.host_value)

    def test_write_through_releases_internal_swa(self):
        """Shows the eviction call reaches the internal node's SWA."""
        cache, node = self._build_cache("write_through")

        self._evict_swa_of(cache, node)

        self.assertIsNone(node.component_data[SWA].value)


if __name__ == "__main__":
    unittest.main()
