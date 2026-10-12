"""HiCache over raiden trusts a DMA future that cannot report failure.

With ``RAIDEN_ENABLE_ASYNC_DISPATCH=1`` (forced by ``RaidenKVTransfer``),
tpu-sync's ``d2h`` / ``h2d`` return a future that reports ``is_ready() ==
True``, ``ok() == True`` and ``error_message() == ""`` even when the copy
failed, so a failed copy is indistinguishable from a successful one. These
tests show what sglang does with such a future: it marks the radix node backed
up (or loaded) and later serves KV that was never copied.

Everything is the production decode stack for a sliding-window model (gpt-oss
style, full-attention KV offloaded, SWA KV kept on device, built through
``pd_decode_swa_hicache``'s full-only host stack strategy) running on CPU:
``SWAKVPool`` of ``TpuFusedMHATokenToKVPool``, ``SWATokenToKVPoolAllocator``
with the TPU paged allocator, ``TpuRaidenHostKVCache``,
``TpuRaidenHybridCacheController``, ``RaidenFutureEvent`` and the Python
``UnifiedRadixCache`` with write-through. Only ``RaidenKVTransfer`` (which
needs a TPU and tpu-sync) is replaced, by a CPU copy engine whose failed copies
return the same "success" future tpu-sync returns. The other shims stand in for
the TPU host: the platform's host-pool hook and the SWA allocator's TPU branch
are pointed at their TPU versions, the request is a namespace with the fields
load-back reads, and the cache is driven through its scheduler entry points
(insert, check_hicache_events, evict, match_prefix, init_load_back,
ready_to_load_host_cache) rather than a live scheduler.

The silent-failure tests document the current (buggy) behaviour: they pass by
asserting the corruption. Once the transfer path can detect a failed copy they
must be inverted.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit")

import contextlib
import logging
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import pd_decode_swa_hicache as full_only
from sglang.srt.hardware_backend.tpu import raiden_transfer
from sglang.srt.hardware_backend.tpu.allocator_tpu import (
    TpuPagedTokenToKVPoolAllocator,
)
from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang.srt.hardware_backend.tpu.kv_pool_host import TpuRaidenHostKVCache
from sglang.srt.hardware_backend.tpu.raiden_hybrid_cache_controller import (
    TpuRaidenHybridCacheController,
)
from sglang.srt.hardware_backend.tpu.raiden_transfer import (
    NOOP_EVENT,
    RaidenFutureEvent,
)
from sglang.srt.hardware_backend.tpu.tpu_platform import TpuSRTPlatform
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator import swa as swa_allocator_module
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

PAGE_SIZE = 16
# gpt-oss interleaves sliding-window and full-attention layers.
SWA_LAYER_IDS = [0, 2]
FULL_LAYER_IDS = [1, 3]
DEVICE_TOKENS = 16 * PAGE_SIZE
SLIDING_WINDOW = 2 * PAGE_SIZE
PREFIX_TOKENS = 4 * PAGE_SIZE
# Fills host pages the fake engine never wrote; random KV is never all this.
HOST_SENTINEL = -3.0
# What other requests leave in device pages after the evicted prefix frees them.
STALE_DEVICE_VALUE = 5.0


# ---------------------------------------------------------------------------
# Fake raiden engine (the only faked component)
# ---------------------------------------------------------------------------


class _SilentSuccessFuture:
    """What tpu-sync returns under async dispatch, whether or not the copy ran."""

    def is_ready(self) -> bool:
        return True

    def ok(self) -> bool:
        return True

    def error_message(self) -> str:
        return ""


class _FakeRaidenKVTransfer:
    """CPU stand-in for ``RaidenKVTransfer``: page copies between the fused
    device buffers and a per-layer host tensor. With ``fail_backups`` /
    ``fail_loads`` set, the copy is skipped but the future still says success.
    """

    instances: list["_FakeRaidenKVTransfer"] = []

    def __init__(self, device_pool, num_host_pages: int):
        self.device_pool = device_pool
        self.num_host_pages = num_host_pages
        self.layer_num = device_pool.layer_num
        # [num_host_pages, *fused page shape] per layer, like raiden's host pool.
        self.host = [
            torch.full(
                (num_host_pages, *self._device_buffer(i).shape[1:]),
                HOST_SENTINEL,
                dtype=self._device_buffer(i).dtype,
            )
            for i in range(self.layer_num)
        ]
        self.fail_backups = False
        self.fail_loads = False
        self.backup_calls: list[tuple[list[int], list[int]]] = []
        self.load_calls: list[tuple[list[int], list[int]]] = []
        _FakeRaidenKVTransfer.instances.append(self)

    def _device_buffer(self, i: int) -> torch.Tensor:
        return self.device_pool.get_fused_kv_buffer(self.device_pool.start_layer + i)

    def backup(self, device_page_ids, host_page_ids) -> RaidenFutureEvent:
        assert len(device_page_ids) == len(host_page_ids)
        if not device_page_ids:
            return NOOP_EVENT
        self.backup_calls.append((list(device_page_ids), list(host_page_ids)))
        if not self.fail_backups:
            for i in range(self.layer_num):
                self.host[i][host_page_ids] = self._device_buffer(i)[device_page_ids]
        return RaidenFutureEvent(_SilentSuccessFuture())

    def load(self, host_page_ids, device_page_ids) -> RaidenFutureEvent:
        assert len(device_page_ids) == len(host_page_ids)
        if not host_page_ids:
            return NOOP_EVENT
        self.load_calls.append((list(host_page_ids), list(device_page_ids)))
        if not self.fail_loads:
            for i in range(self.layer_num):
                self._device_buffer(i)[device_page_ids] = self.host[i][host_page_ids]
        return RaidenFutureEvent(_SilentSuccessFuture())


# ---------------------------------------------------------------------------
# Stack construction
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _full_only_patches_installed():
    """``pd_decode_swa_hicache.install()``, undone on exit."""
    originals = [
        (owner, name, getattr(owner, name))
        for owner, name, _ in full_only._patch_targets()
    ]
    full_only.install()
    try:
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)
        hybrid_pool_assembler._STRATEGIES.remove(full_only._FULL_ONLY_SWA_STRATEGY)
        full_only._is_installed = False


def _server_args() -> ServerArgs:
    return ServerArgs(
        model_path="dummy",
        page_size=PAGE_SIZE,
        enable_hierarchical_cache=True,
        hicache_io_backend="raiden",
        hicache_write_policy="write_through",
        hicache_ratio=2.0,
        hicache_size=0,
    )


class _Stack:
    """The decode-side cache stack, built the way the production decode does."""

    def __init__(self, test: unittest.TestCase):
        server_args = _server_args()
        publish(server_args, role="scheduler")

        self.req_to_token_pool = ReqToTokenPool(
            size=4,
            max_context_len=DEVICE_TOKENS,
            device="cpu",
            enable_memory_saver=False,
        )
        self.kvcache = SWAKVPool(
            size=DEVICE_TOKENS,
            size_swa=DEVICE_TOKENS,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            swa_attention_layer_ids=SWA_LAYER_IDS,
            full_attention_layer_ids=FULL_LAYER_IDS,
            device="cpu",
            token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
        )
        self.full_pool = self.kvcache.full_kv_pool
        self.allocator = SWATokenToKVPoolAllocator(
            DEVICE_TOKENS,
            DEVICE_TOKENS,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=self.kvcache,
            need_sort=False,
            req_to_token_pool=self.req_to_token_pool,
        )
        test.assertIsInstance(
            self.allocator.full_attn_allocator, TpuPagedTokenToKVPoolAllocator
        )
        self.params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=PAGE_SIZE,
            sliding_window_size=SLIDING_WINDOW,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
        )
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
            self.cache = UnifiedRadixCache(self.params)
        # Same as the upstream tree factory with HiCache on, inside the
        # full-only build that pd_decode_swa_hicache._build_kv_cache opens.
        with full_only._full_only_host_cache_build(None):
            self.cache.init_hicache(server_args, self.params)

        test.assertTrue(full_only.uses_full_only_host_cache(self.cache))
        self.controller = self.cache.cache_controller
        test.assertIsInstance(self.controller, TpuRaidenHybridCacheController)
        self.host_pool = self.controller.mem_pool_host.anchor_entry.host_pool
        test.assertIsInstance(self.host_pool, TpuRaidenHostKVCache)
        # One raiden engine, over the full-attention sub-pool only.
        (self.engine,) = _FakeRaidenKVTransfer.instances
        test.assertIs(self.host_pool.transfer, self.engine)
        test.assertIs(self.engine.device_pool, self.full_pool)

    # ----- KV helpers --------------------------------------------------------

    def pages(self, token_indices: torch.Tensor) -> list[int]:
        return (token_indices[::PAGE_SIZE] // PAGE_SIZE).tolist()

    def write_random_kv(self, pages: list[int]) -> list[torch.Tensor]:
        """Fill ``pages`` of every full-attention layer with random KV."""
        written = []
        for buf in self.full_pool.kv_buffer:
            kv = torch.randn((len(pages), *buf.shape[1:])).to(buf.dtype)
            buf[pages] = kv
            written.append(kv.clone())
        return written

    def read_kv(self, pages: list[int]) -> list[torch.Tensor]:
        return [buf[pages].clone() for buf in self.full_pool.kv_buffer]

    def reuse_free_device_pages(self) -> None:
        """Hand every free full-attention page to "other requests", which write
        their own KV into it, then free it again."""
        full_allocator = self.allocator.full_attn_allocator
        indices = full_allocator.alloc(full_allocator.available_size())
        for buf in self.full_pool.kv_buffer:
            buf[self.pages(indices)] = STALE_DEVICE_VALUE
        full_allocator.free(indices)

    # ----- cache-driving helpers ---------------------------------------------

    def alloc_request_kv(self, num_tokens: int) -> torch.Tensor:
        """Device slots for a fresh request without a cached prefix."""
        zero = torch.tensor([0], dtype=torch.int64)
        seq_len = torch.tensor([num_tokens], dtype=torch.int64)
        return self.allocator.alloc_extend(
            prefix_lens=zero,
            prefix_lens_cpu=zero,
            seq_lens=seq_len,
            seq_lens_cpu=seq_len,
            last_loc=torch.tensor([-1], dtype=torch.int64),
            extend_num_tokens=num_tokens,
        )

    @staticmethod
    def key(num_tokens: int) -> RadixKey:
        return RadixKey(array("q", range(1000, 1000 + num_tokens)))

    def insert_prefix(self, device_indices: torch.Tensor):
        """Cache a finished request's prefix, as release_kv_cache does."""
        self.cache.insert(
            InsertParams(key=self.key(len(device_indices)), value=device_indices)
        )

    def match(self, num_tokens: int):
        return self.cache.match_prefix(MatchPrefixParams(key=self.key(num_tokens)))

    def prefix_nodes(self, node_id) -> list:
        """The radix nodes from the root down to ``node_id``; the tree may split
        one inserted prefix into several nodes."""
        nodes = []
        node = self.cache.tree_core.node_by_id(node_id)
        while node.parent is not None:
            nodes.append(node)
            node = node.parent
        return nodes[::-1]

    def prefix_host_pages(self, node_id) -> list[int]:
        """Host pages sglang believes hold the prefix's full-attention KV."""
        host_value = torch.cat(
            [
                node.component_data[ComponentType.FULL].host_value
                for node in self.prefix_nodes(node_id)
            ]
        )
        return self.host_pool._to_page_ids(host_value)


def _make_req(last_node):
    """The request fields init_load_back and the SWA/Full load-back hooks read."""
    return SimpleNamespace(
        rid="restore",
        last_node=last_node,
        swa_host_hit_length=0,
        mamba_host_hit_length=0,
        kv=ReqKvInfo(),
    )


class _SilentDmaFailureBase(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        _FakeRaidenKVTransfer.instances.clear()
        reset_context()
        self.addCleanup(reset_context)
        for ctx in (
            patch.object(raiden_transfer, "RaidenKVTransfer", _FakeRaidenKVTransfer),
            # TpuSRTPlatform picks the raiden host pool; CPU's platform would not.
            patch(
                "sglang.srt.platforms.current_platform.get_mha_kv_pool_host_cls",
                lambda device_pool: TpuSRTPlatform.get_mha_kv_pool_host_cls(
                    None, device_pool
                ),
            ),
            # The TPU branches of the SWA allocator (TPU paged allocator,
            # host-resident full->swa mapping), as on a TPU host.
            patch.object(swa_allocator_module, "_is_tpu", True),
            patch.object(
                swa_allocator_module,
                "TpuPagedTokenToKVPoolAllocator",
                TpuPagedTokenToKVPoolAllocator,
                create=True,
            ),
            _full_only_patches_installed(),
        ):
            ctx.__enter__()
            self.addCleanup(ctx.__exit__, None, None, None)
        self.stack = _Stack(self)

    # ----- the scheduler-driven phases, each through the real entry points ---

    def cache_prefix_and_write_through(self):
        """Insert a prefix with known KV and let write-through back it up.

        Returns (node_id, device_pages, original_kv).
        """
        stack = self.stack
        device_indices = stack.alloc_request_kv(PREFIX_TOKENS)
        self.assertIsNotNone(device_indices)
        pages = stack.pages(device_indices)
        original = stack.write_random_kv(pages)

        stack.insert_prefix(device_indices)
        match = stack.match(PREFIX_TOKENS)
        self.assertEqual(len(match.device_indices), PREFIX_TOKENS)
        node_id = match.last_device_node

        # Write-through (threshold 1) queued the backup during insert, and the
        # node already counts as host-backed while the DMA is pending.
        nodes = stack.prefix_nodes(node_id)
        for node in nodes:
            self.assertTrue(node.backuped, f"node {node.id} not backed up")
        self.assertEqual(
            {node.write_through_pending_id for node in nodes},
            set(stack.cache.ongoing_write_through),
        )
        self.assertEqual(stack.engine.backup_calls, [])
        # The scheduler's per-step poll submits the DMA and drains its ack.
        stack.cache.check_hicache_events()
        self.assertEqual(len(stack.engine.backup_calls), 1)
        self.assertEqual(stack.engine.backup_calls[0][0], pages)
        return node_id, pages, original

    def assert_backup_acked(self, node_id):
        """sglang considers the backup complete and the node host-backed."""
        stack = self.stack
        self.assertEqual(stack.cache.ongoing_write_through, {})
        self.assertEqual(stack.controller.ack_write_queue, [])
        nodes = stack.prefix_nodes(node_id)
        self.assertEqual(sum(len(node.key) for node in nodes), PREFIX_TOKENS)
        for node in nodes:
            self.assertIsNone(node.write_through_pending_id)
            self.assertTrue(stack.cache.is_backuped(node.id))
        self.assertEqual(
            len(stack.prefix_host_pages(node_id)), PREFIX_TOKENS // PAGE_SIZE
        )

    def evict_to_host(self, node_id):
        """Device pressure demotes the backed-up node to host-only."""
        stack = self.stack
        result = stack.cache.evict(EvictParams(num_tokens=PREFIX_TOKENS))
        self.assertGreaterEqual(result.num_tokens_evicted, PREFIX_TOKENS)
        for node in stack.prefix_nodes(node_id):
            self.assertTrue(node.evicted, f"node {node.id} still holds device KV")
            self.assertTrue(node.backuped, f"node {node.id} lost its host copy")
        # The freed pages go to other requests before the prefix comes back.
        stack.reuse_free_device_pages()

    def restore_from_host(self, node_id):
        """Re-match the prefix (host hit) and load it back, as the PD decode
        restore loop does. Returns the device pages the prefix now uses."""
        stack = self.stack
        match = stack.match(PREFIX_TOKENS)
        self.assertEqual(len(match.device_indices), 0)
        self.assertEqual(match.host_hit_length, PREFIX_TOKENS)
        self.assertEqual(match.best_match_node, node_id)

        new_indices, restored_node = stack.cache.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=_make_req(match.last_device_node),
            )
        )
        self.assertEqual(restored_node, node_id)
        self.assertEqual(len(new_indices), PREFIX_TOKENS)

        consumer_index = stack.cache.ready_to_load_host_cache()
        self.assertGreaterEqual(consumer_index, 0)
        self.assertEqual(len(stack.engine.load_calls), 1)
        # The raiden controller's poll that PD decode uses to flip the
        # restore READY (tpu patches route is_load_back_event_done here).
        self.assertTrue(
            stack.controller.is_load_back_event_done(
                consumer_index, stack.cache._all_reduce
            )
        )
        # The forward's whole-batch gate on the reload.
        stack.controller.layer_done_counter.set_consumer(consumer_index)
        stack.controller.layer_done_counter.wait()
        stack.cache.check_hicache_events()

        self.assertEqual(stack.cache.ongoing_load_back, {})
        self.assertEqual(stack.controller.ack_load_queue, [])
        for node in stack.prefix_nodes(node_id):
            self.assertFalse(node.evicted, f"node {node.id} has no device KV")
        rematch = stack.match(PREFIX_TOKENS)
        self.assertEqual(len(rematch.device_indices), PREFIX_TOKENS)
        self.assertTrue(torch.equal(rematch.device_indices, new_indices))
        pages = stack.pages(new_indices)
        self.assertEqual(stack.engine.load_calls[0][1], pages)
        return pages


class TestRaidenHiCacheRoundTrip(_SilentDmaFailureBase):
    def test_backup_evict_restore_is_bit_exact(self):
        """Control: with working DMAs the restored KV equals the original."""
        node_id, _, original = self.cache_prefix_and_write_through()
        self.assert_backup_acked(node_id)
        self.evict_to_host(node_id)
        restored_pages = self.restore_from_host(node_id)

        for layer, (got, want) in enumerate(
            zip(self.stack.read_kv(restored_pages), original)
        ):
            self.assertTrue(torch.equal(got, want), f"layer {layer} differs")


class TestRaidenHiCacheSilentDmaFailure(_SilentDmaFailureBase):
    def test_silent_backup_failure_restores_garbage(self):
        """Documents current (buggy) behaviour: a backup DMA that failed but
        reported success is acked, the node is treated as host-backed, eviction
        drops the only good copy, and load-back puts never-written host memory
        on the device, all without any error."""
        self.stack.engine.fail_backups = True

        # (d) Nothing raises and nothing is logged at WARNING or above.
        with self.assertNoLogs(level=logging.WARNING):
            node_id, _, original = self.cache_prefix_and_write_through()
            # (a) The failed backup is acked like a successful one.
            self.assert_backup_acked(node_id)
            # The host pages the prefix points at were never written.
            host_pages = self.stack.prefix_host_pages(node_id)
            for layer_host in self.stack.engine.host:
                self.assertTrue(
                    bool((layer_host[host_pages] == HOST_SENTINEL).all())
                )

            # (b) Eviction frees the device copy, leaving only the bogus host copy.
            self.evict_to_host(node_id)

            # (c) Load-back restores the host garbage instead of the original KV.
            restored_pages = self.restore_from_host(node_id)
        for layer, (got, want) in enumerate(
            zip(self.stack.read_kv(restored_pages), original)
        ):
            self.assertFalse(torch.equal(got, want), f"layer {layer} intact")
            self.assertTrue(
                bool((got == HOST_SENTINEL).all()),
                f"layer {layer} does not hold the never-written host pages",
            )

    def test_silent_load_failure_leaves_stale_device_kv(self):
        """Documents current (buggy) behaviour: a load-back DMA that failed but
        reported success is acked, the node is marked device-resident, and its
        device pages keep whatever other requests left there."""
        node_id, _, original = self.cache_prefix_and_write_through()
        self.assert_backup_acked(node_id)
        # The backup really landed: the host copy is good.
        host_pages = self.stack.prefix_host_pages(node_id)
        for layer_host, want in zip(self.stack.engine.host, original):
            self.assertTrue(torch.equal(layer_host[host_pages], want))

        self.evict_to_host(node_id)
        self.stack.engine.fail_loads = True
        with self.assertNoLogs(level=logging.WARNING):
            restored_pages = self.restore_from_host(node_id)

        for layer, (got, want) in enumerate(
            zip(self.stack.read_kv(restored_pages), original)
        ):
            self.assertFalse(torch.equal(got, want), f"layer {layer} intact")
            self.assertTrue(
                bool((got == STALE_DEVICE_VALUE).all()),
                f"layer {layer} does not hold the stale device pages",
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
