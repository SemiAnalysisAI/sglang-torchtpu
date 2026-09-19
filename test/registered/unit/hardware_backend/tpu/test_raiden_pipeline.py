"""Layer subsets and transfer lifetimes for pipeline-parallel Raiden disaggregation."""

import concurrent.futures
import json
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.raiden.host_pool import (
    StagingAdmission,
    node_worker_rank,
    ranks_per_node,
)
from sglang.srt.disaggregation.raiden.pipeline import (
    PipelineTransfer,
    Receive,
    destination_blocks,
    layout_fingerprint,
    legacy_mirror_slots,
    legacy_slot_host_bytes,
    materialized_storage_layout,
    pool_manifest,
    source_spans,
    warn_if_staging_exceeds_premapped,
)
from sglang.srt.disaggregation.raiden.serde import MessageTag
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Buffer:
    dtype = "torch.float8_e4m3fn"

    def __init__(self, pages=64):
        self.shape = (pages, 128, 8, 2, 128)

    def element_size(self):
        return 1

    def view(self, dtype):
        return SimpleNamespace(dtype=str(dtype), shape=self.shape)


def groups(full=(0, 2), swa=(1, 3), pages=64):
    return SimpleNamespace(
        full=[Buffer(pages) for _ in full],
        swa=[Buffer(pages) for _ in swa],
        full_layer_ids=list(full),
        swa_layer_ids=list(swa),
    )


def transfer(producer=False, pp_size=1, pp_rank=0):
    pipe = PipelineTransfer.__new__(PipelineTransfer)
    pipe.manager = SimpleNamespace(
        _state_lock=threading.Lock(),
        _aborted={},
        _completed_send=set(),
        _completed_recv=set(),
        _native_failed=set(),
        _failed=set(),
        _norm_req_id=lambda value: value.removeprefix("resharded_transfer_"),
        _require_frames=Mock(),
        update_status=Mock(),
        pp_size=pp_size,
        pp_rank=pp_rank,
    )
    pipe.producer = producer
    pipe.lock = threading.RLock()
    pipe.engine = SimpleNamespace(poll_stats=Mock(return_value=([], [], [])))
    pipe.pools = pool_manifest(groups(full=(6,), swa=(7,)))
    pipe.client = Mock()
    pipe.clients = {}
    pipe.client_type = Mock(return_value=Mock())
    pipe.controller = "decode:30001"
    pipe.unit = SimpleNamespace(data_replica_idx=pp_rank)
    pipe.executor = Mock()
    pipe.executor.submit.return_value = concurrent.futures.Future()
    pipe.max_blocks = 4
    pipe.admission = StagingAdmission(8)
    pipe.sent, pipe.replies, pipe.receives = {}, {}, {}
    pipe.release_requests = {}
    pipe.done_send, pipe.done_recv, pipe.failed = set(), set(), set()
    pipe.retired, pipe.cleared = set(), set()
    pipe._send = Mock()
    return pipe


def receive(rank=0, pp_size=8, full=(20, 21), swa=(30,)):
    metadata = dict(
        controller="prefill:30001",
        rank=rank,
        pp_size=pp_size,
        full=2,
        swa=1,
        unit=dict(
            job_name="sglang",
            job_replica_id="prefill:30001",
            data_name="prefill",
            data_replica_idx=rank,
        ),
    )
    return Receive(
        list(full),
        list(swa),
        None,
        {rank},
        {rank: metadata},
        deadline=time.monotonic() + 120,
    )


class TestStorageLayout(unittest.TestCase):
    def test_fp8_dtype_alias_queries_the_existing_uint8_storage_view(self):
        storage = SimpleNamespace(dtype="torch.uint8")
        tensor = SimpleNamespace(
            dtype="torch.float8_e4m3fn", view=Mock(return_value=storage)
        )
        expected = ((4, 3, 2, 1, 0), ((8, 128),), 8)
        getter = Mock(side_effect=[None, expected])
        self.assertEqual(materialized_storage_layout(tensor, getter), expected)
        self.assertIs(getter.call_args_list[0].args[0], tensor)
        self.assertIs(getter.call_args_list[1].args[0], storage)
        self.assertEqual(str(tensor.view.call_args.args[0]), "torch.uint8")

    def test_materialized_fp8_base_is_queried_directly(self):
        tensor = SimpleNamespace(dtype="torch.float8_e4m3fn", view=Mock())
        expected = ((4, 3, 2, 1, 0), ((8, 128),), 8)
        self.assertEqual(
            materialized_storage_layout(tensor, lambda _: expected), expected
        )
        tensor.view.assert_not_called()

    def test_other_unmaterialized_dtypes_are_not_reinterpreted(self):
        tensor = SimpleNamespace(dtype="torch.bfloat16", view=Mock())
        self.assertIsNone(materialized_storage_layout(tensor, lambda _: None))
        tensor.view.assert_not_called()


class TestLayerPools(unittest.TestCase):
    def test_hybrid_tags_keep_global_layer_ids_and_local_storage_indices(self):
        pools = pool_manifest(groups(full=(8, 10), swa=(9, 11)))
        self.assertEqual(
            [p["tag"] for p in pools], ["fa.l8", "fa.l10", "swa.l9", "swa.l11"]
        )
        self.assertEqual([p["storage_index"] for p in pools], [0, 1, 2, 3])

    def test_duplicate_layer_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct global layer IDs"):
            pool_manifest(groups(full=(8, 8), swa=()))

    def test_stage_requires_a_full_attention_pool(self):
        with self.assertRaisesRegex(ValueError, "full-attention KV"):
            pool_manifest(groups(full=(), swa=(1,)))

    def test_every_layer_owns_all_token_pages_in_its_namespace(self):
        pools = pool_manifest(groups(full=(7, 8), swa=(9,)))
        spans = source_spans(pools, [100, 103], [12])
        self.assertEqual([s.block_ids for s in spans], [[100, 103], [100, 103], [12]])
        self.assertEqual([s.dst_space_version for s in spans], [1, 1, 1])
        for pool, entry in zip(pools, spans):
            stride = pool["block_stride_bytes"]
            count = len(entry.block_ids)
            self.assertEqual([s.dst_block_index for s in entry.spans], [0] * count)
            self.assertEqual(
                [s.dst_offset_bytes for s in entry.spans],
                [index * stride for index in range(count)],
            )
            self.assertEqual(entry.declared_bytes, count * stride)

    def test_global_spans_clip_full_and_swa_prefixes_by_bytes(self):
        pools = [
            dict(tag="fa.l8", block_stride_bytes=4096),
            dict(tag="swa.l9", block_stride_bytes=2048),
        ]
        spans = source_spans(pools, [100, 103, 109], [12, 16])
        destination = destination_blocks(pools, [23, 24], [30], 3, 2)
        self.assertEqual(destination["dst_skip_bytes"], [4096, 2048])
        # Global byte offsets rebase onto the uncached destination suffix.
        clipped = [
            [
                (
                    entry.block_ids[span.src_block_ordinal],
                    span.dst_offset_bytes - skip,
                    span.size_bytes,
                )
                for span in entry.spans
                if span.dst_offset_bytes >= skip
            ]
            for entry, skip in zip(spans, destination["dst_skip_bytes"])
        ]
        self.assertEqual(
            clipped, [[(103, 0, 4096), (109, 4096, 4096)], [(16, 0, 2048)]]
        )

    def test_destination_clips_full_prefix_and_window_independently(self):
        pools = pool_manifest(groups(full=(8,), swa=(9,)))
        result = destination_blocks(pools, [23, 24], [30], 4, 1)
        self.assertEqual(result["dst_device_block_ids"], [23, 24, 30])
        self.assertEqual(result["dst_block_counts"], [2, 1])
        self.assertEqual(
            result["dst_skip_bytes"], [2 * pools[0]["block_stride_bytes"], 0]
        )

    def test_destination_omits_the_decode_only_page(self):
        pools = pool_manifest(groups(full=(8,), swa=()))
        result = destination_blocks(pools, [23, 24, 25], [], 2, 0)
        self.assertEqual(result["dst_device_block_ids"], [23, 24])
        self.assertEqual(result["dst_skip_bytes"], [0])

    def test_fingerprint_matches_an_uneven_stage_subset(self):
        full = groups(full=tuple(range(62)), swa=(), pages=64)
        stage = groups(full=tuple(range(55, 62)), swa=(), pages=192)
        getter = lambda tensor: ((4, 3, 2, 1, 0), ((8, 128),), 8)
        self.assertEqual(
            layout_fingerprint(pool_manifest(full), full.full, getter),
            layout_fingerprint(pool_manifest(stage), stage.full, getter),
        )
        self.assertNotEqual(
            layout_fingerprint(pool_manifest(full), full.full, getter),
            layout_fingerprint(
                pool_manifest(stage),
                stage.full,
                lambda tensor: ((4, 3, 2, 1, 0), ((4, 128),), 8),
            ),
        )

    def test_pp_ranks_have_distinct_ports_and_share_the_host_budget(self):
        ranks = {
            node_worker_rank(0, 1, 0, 1, 0, pp_rank=stage, pp_size=8)
            for stage in range(8)
        }
        self.assertEqual(ranks, set(range(8)))
        self.assertEqual(ranks_per_node(1, 1, 1, 1, pp_size=8), 8)
        self.assertEqual(ranks_per_node(1, 1, 1, 2, pp_size=8), 4)


class TestLegacyPoolBudget(unittest.TestCase):
    @staticmethod
    def pools(extents):
        return [
            dict(
                storage_index=i,
                base_offset_bytes=0,
                block_stride_bytes=4096,
                num_blocks=n,
            )
            for i, n in enumerate(extents)
        ]

    def test_uniform_pools_use_logical_metadata_despite_physical_tile_padding(self):
        tensors = [Buffer(64), Buffer(64)]
        padded = lambda _: ((4, 3, 2, 1, 0), ((8, 128),), 8)
        self.assertEqual(
            legacy_slot_host_bytes(tensors, 4, padded), 128 * 8 * 2 * 128 * 2 * 4
        )

    def test_mixed_full_window_pools_charge_native_first_tile_padding(self):
        tensors = [Buffer(64), Buffer(9)]
        padded = lambda _: ((4, 3, 2, 1, 0), ((8, 128),), 8)
        self.assertEqual(
            legacy_slot_host_bytes(tensors, 4, padded), 128 * 8 * 8 * 128 * 2 * 4
        )
        unpadded = lambda _: ((4, 2, 3, 1, 0), ((8, 128),), 8)
        self.assertEqual(
            legacy_slot_host_bytes(tensors, 4, unpadded), 128 * 8 * 2 * 128 * 2 * 4
        )

    def test_unknown_and_non_block_major_layouts_are_rejected(self):
        tensors = [Buffer(64), Buffer(9)]
        with self.assertRaisesRegex(ValueError, "materialized page layouts"):
            legacy_slot_host_bytes(tensors, 4, lambda _: None)
        with self.assertRaisesRegex(ValueError, "block-major"):
            legacy_slot_host_bytes(tensors, 4, lambda _: ((0, 4, 3, 2, 1), (), 8))
        with self.assertRaisesRegex(ValueError, "two-dimensional tile"):
            legacy_slot_host_bytes(
                tensors, 4, lambda _: ((4, 3, 2, 1, 0), ((8, 8, 128),), 8)
            )

    def test_existing_uniform_buffers_cover_small_pools(self):
        slots, peak = legacy_mirror_slots(self.pools([2, 3]), 8 * 4096, 4, 32 * 4096)
        self.assertEqual((slots, peak), (4, 32 * 4096))

    def test_mixed_full_and_window_pools_include_replacement_peak(self):
        slots, peak = legacy_mirror_slots(self.pools([40, 2]), 8 * 4096, 4, 56 * 4096)
        self.assertEqual((slots, peak), (2, 56 * 4096))

    def test_a_mirror_that_cannot_fit_one_slot_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "including pool replacement"):
            legacy_mirror_slots(self.pools([40, 2]), 8 * 4096, 4, 47 * 4096)

    def test_two_growing_storages_bound_last_replacement(self):
        self.assertEqual(
            legacy_mirror_slots(self.pools([40, 60]), 8 * 4096, 4, 108 * 4096),
            (2, 108 * 4096),
        )

    def test_staging_above_the_premapped_buffer_warns(self):
        with self.assertLogs(
            "sglang.srt.disaggregation.raiden.pipeline", level="WARNING"
        ) as logs:
            self.assertTrue(
                warn_if_staging_exceeds_premapped(
                    staging_bytes=33 << 30, premapped_bytes=4 << 30
                )
            )
        self.assertIn("33.0 GiB", logs.output[0])
        self.assertIn("TPU_PREMAPPED_BUFFER_SIZE", logs.output[0])

    def test_staging_within_the_premapped_buffer_is_silent(self):
        self.assertFalse(
            warn_if_staging_exceeds_premapped(
                staging_bytes=32 << 30, premapped_bytes=36 << 30
            )
        )

    def test_aliased_or_offset_storage_is_rejected(self):
        pools = self.pools([20, 30])
        pools[1]["storage_index"] = 0
        with self.assertRaisesRegex(ValueError, "one pool per storage"):
            legacy_mirror_slots(pools, 8, 1, 100)
        pools = self.pools([20])
        pools[0]["base_offset_bytes"] = 4
        with self.assertRaisesRegex(ValueError, "zero storage offsets"):
            legacy_mirror_slots(pools, 4, 1, 100)


class TestPipelineLifetimes(unittest.TestCase):
    def test_invalid_native_uuid_is_rejected_before_registration(self):
        source = transfer(producer=True)
        destination = transfer()
        for room in (0, -1, 1 << 63):
            with self.assertRaisesRegex(ValueError, "positive signed 64-bit"):
                source.finalize_send(room, None)
            with self.assertRaisesRegex(ValueError, "positive signed 64-bit"):
                destination.register_decode(room, [], [], None, [])
        source.client.register_request_blocks.assert_not_called()

    def test_pp_destination_enumerates_all_source_work_units(self):
        pipe = transfer(pp_size=8, pp_rank=3)
        pipe.receives[7] = receive(rank=3)
        self.assertTrue(pipe.maybe_start_pull(7))
        args = pipe.executor.submit.call_args.kwargs
        self.assertEqual(
            [u.data_replica_idx for u in args["src_units"]], list(range(8))
        )
        self.assertEqual(args["transfer_pool_tags"], ["fa.l6", "swa.l7"])
        self.assertEqual(args["dst_units"], [pipe.unit])
        self.assertEqual(args["dst_mem_type"], 2)

    def test_matching_pipeline_rejects_different_stage_layer_subsets(self):
        pipe = transfer(pp_size=8, pp_rank=3)
        pipe.receives[7] = receive(rank=3)
        meta = pipe.receives[7].sources[3] | {"pool_tags": ["fa.l12", "swa.l13"]}
        with self.assertRaisesRegex(ValueError, "same layers per stage"):
            pipe.on_message(
                [MessageTag.PIPELINE_READY, b"7", json.dumps(meta).encode(), b""]
            )

    def test_dp_destination_waits_for_every_pipeline_source(self):
        pipe = transfer()
        request = receive()
        request.ranks = set(range(8))
        pipe.receives[7] = request
        self.assertFalse(pipe.maybe_start_pull(7))
        pipe.executor.submit.assert_not_called()

    def test_native_receive_completion_waits_for_submission_to_return(self):
        pipe = transfer()
        request = receive()
        pipe.receives[7] = request
        pipe.maybe_start_pull(7)
        pipe.engine.poll_stats.return_value = ([], ["resharded_transfer_7"], [])
        pipe.poll()
        self.assertTrue(pipe.active(7))
        self.assertNotIn("7", pipe.manager._completed_recv)
        self.assertIn(7, pipe.admission)
        request.future.set_result(True)
        pipe.poll()
        self.assertFalse(pipe.active(7))
        self.assertIn("7", pipe.manager._completed_recv)
        self.assertNotIn(7, pipe.admission)

    def test_known_missing_source_registration_releases_admission_for_retry(self):
        pipe = transfer()
        pipe.receives[7] = receive()
        pipe.maybe_start_pull(7)
        pipe.receives[7].future.set_exception(
            RuntimeError("Missing producer block registration")
        )
        pipe.poll()
        self.assertIsNone(pipe.receives[7].future)
        self.assertNotIn(7, pipe.admission)
        self.assertNotIn("7", pipe.manager._failed)

    def test_cancelled_source_registration_fails_without_native_transfer(self):
        pipe = transfer()
        pipe.receives[7] = receive()
        pipe.maybe_start_pull(7)
        pipe.receives[7].future.set_exception(
            RuntimeError("Request block registration was cancelled")
        )
        pipe.poll()
        self.assertFalse(pipe.active(7))
        self.assertNotIn(7, pipe.admission)
        self.assertIn("7", pipe.manager._failed)

    def test_expired_missing_registration_fails_without_native_transfer(self):
        pipe = transfer()
        pipe.receives[7] = receive()
        pipe.receives[7].deadline = 0
        pipe.maybe_start_pull(7)
        pipe.receives[7].future.set_exception(
            RuntimeError("Missing producer block registration")
        )
        pipe.poll()
        self.assertFalse(pipe.active(7))
        self.assertIn("7", pipe.manager._failed)

    def test_unknown_submission_error_propagates(self):
        pipe = transfer()
        pipe.receives[7] = receive()
        pipe.maybe_start_pull(7)
        pipe.receives[7].future.set_exception(RuntimeError("bad pool geometry"))
        with self.assertRaisesRegex(RuntimeError, "bad pool geometry"):
            pipe.poll()

    def test_aborted_claimed_source_keeps_its_pages(self):
        pipe = transfer(producer=True)
        pipe.sent[7] = dict(expires=time.monotonic() + 120)
        pipe.manager._aborted[7] = time.time()
        pipe.client.cancel_request_blocks_if_unclaimed.return_value = False
        pipe.poll()
        self.assertTrue(pipe.active(7))
        pipe.clear(7)
        self.assertIn(7, pipe.sent)
        pipe.engine.poll_stats.return_value = (["7"], [], [])
        pipe.poll()
        self.assertFalse(pipe.active(7))
        self.assertNotIn(7, pipe.sent)
        pipe.client.complete_request_blocks.assert_called_once_with("7", 7, pipe.unit)

    def test_mixed_pp_cache_hits_vote_only_for_the_completed_stage(self):
        cached = transfer(producer=True, pp_size=2, pp_rank=0)
        pulling = transfer(producer=True, pp_size=2, pp_rank=1)
        for pipe in (cached, pulling):
            pipe.sent[7] = dict(expires=time.monotonic() + 120)
        cached.client.complete_request_blocks.side_effect = RuntimeError(
            "Only an empty producer registration may complete before the request block snapshot is claimed"
        )
        cached.on_message([MessageTag.PIPELINE_RELEASE, b"7", b"2"])
        self.assertTrue(cached.active(7))
        self.assertTrue(pulling.active(7))
        cached.client.cancel_request_blocks_if_unclaimed.assert_not_called()
        cached.client.complete_request_blocks.side_effect = None
        cached.release_requests[7]["next_attempt"] = 0
        cached.poll()
        self.assertFalse(cached.active(7))
        self.assertTrue(pulling.active(7))
        cached.client.complete_request_blocks.assert_called_with("7", 7, cached.unit)
        cached.client.cancel_request_blocks_if_unclaimed.assert_not_called()
        pulling.engine.poll_stats.return_value = (["7"], [], [])
        pulling.poll()
        self.assertFalse(pulling.active(7))
        pulling.client.complete_request_blocks.assert_called_once_with(
            "7", 7, pulling.unit
        )
        pulling.client.cancel_request_blocks_if_unclaimed.assert_not_called()

    def test_all_pp_cache_hits_cancel_only_after_unclaimed_expiry(self):
        for rank in range(2):
            pipe = transfer(producer=True, pp_size=2, pp_rank=rank)
            pipe.sent[7] = dict(expires=time.monotonic() + 120)
            pipe.client.complete_request_blocks.side_effect = RuntimeError(
                "Only an empty producer registration may complete before the request block snapshot is claimed"
            )
            pipe.on_message([MessageTag.PIPELINE_RELEASE, b"7", b"2"])
            self.assertTrue(pipe.active(7))
            pipe.client.cancel_request_blocks_if_unclaimed.assert_not_called()
            pipe.sent[7]["expires"] = 0
            pipe.client.cancel_request_blocks_if_unclaimed.return_value = True
            pipe.poll()
            self.assertFalse(pipe.active(7))
            self.assertIn("7", pipe.manager._completed_send)
            self.assertNotIn("7", pipe.manager._failed)
            pipe.client.cancel_request_blocks_if_unclaimed.assert_called_once_with(
                "7", 7
            )

    def test_native_source_completion_retires_exactly_once(self):
        pipe = transfer(producer=True)
        pipe.sent[7] = dict(expires=time.monotonic() + 120)
        pipe.engine.poll_stats.return_value = (["7"], [], [])
        pipe.poll()
        pipe.engine.poll_stats.return_value = ([], [], [])
        pipe.poll()
        pipe.client.complete_request_blocks.assert_called_once_with("7", 7, pipe.unit)

    def test_full_prefix_hit_releases_source_without_starting_transfer(self):
        pipe = transfer()
        request = receive(full=(), swa=())
        request.endpoints = ("tcp://prefill:30002",)
        pipe.receives[7] = request
        self.assertTrue(pipe.maybe_start_pull(7))
        self.assertIn("7", pipe.manager._completed_recv)
        pipe.executor.submit.assert_not_called()
        pipe._send.assert_called_once_with(
            "tcp://prefill:30002", [MessageTag.PIPELINE_RELEASE, b"7", b"1"]
        )


class TestPipelineInitialization(unittest.TestCase):
    def setUp(self):
        override = get_context().override_server_args(
            speculative_algorithm=None, nnodes=1
        )
        override.install()
        self.addCleanup(override.restore)

    def test_resolved_speculative_mode_is_rejected(self):
        get_context().override("test.speculative", speculative_algorithm="EAGLE3")
        with self.assertRaisesRegex(ValueError, "does not support speculative"):
            self._initialize()

    def test_legacy_host_budget_uses_resolved_node_count(self):
        get_context().override("test.multihost", nnodes=2)
        with patch(
            "sglang.srt.disaggregation.raiden.pipeline.ranks_per_node",
            wraps=ranks_per_node,
        ) as local_ranks:
            self._initialize(legacy=True)
        self.assertEqual(local_ranks.call_args.args[3], 2)

    def _initialize(self, legacy=False):
        events = []
        buffers = groups(full=(0,), swa=())
        manager = SimpleNamespace(
            disaggregation_mode="decode",
            attn_tp_size=1,
            attn_dp_size=1,
            system_dp_size=1,
            attn_cp_size=1,
            pp_size=1,
            pp_rank=0,
            local_ip="10.0.0.1",
            _control_port=31000,
            _local_rank=0,
            _timeout_s=120,
            server_args=SimpleNamespace(speculative_algorithm=None, nnodes=1),
            kv_args=SimpleNamespace(page_size=128),
            _max_request_blocks=lambda: 4,
            _num_slots=lambda *args: 2,
        )

        def layout(tensor):
            self.assertTrue(tensor.materialized)
            events.append("layout")
            return ((4, 2, 3, 1, 0), ((8, 128),), 8)

        def synchronize(tensors, wait):
            events.append("materialize")
            self.assertTrue(wait)
            for tensor in tensors:
                tensor.materialized = True

        def engine(**kwargs):
            events.append("engine")
            for tensor in kwargs["kv_caches"]:
                tensor.materialized = True
            return SimpleNamespace(
                transfer_address="10.0.0.1:31001",
                listener_address="10.0.0.1:31002",
                register_pools=Mock(
                    side_effect=lambda *args, **kwargs: events.append("pools")
                ),
            )

        def register_pools(
            self, pools, *, staging_leases=0, staging_blocks_per_pool=None
        ):
            pass

        def legacy_register_pools(self, pools):
            pass

        engine.register_pools = legacy_register_pools if legacy else register_pools
        client = Mock()
        client.register_work_unit.side_effect = lambda **kwargs: events.append(
            "work_unit"
        )
        native = {
            "torch_tpu._internal.sync.sync": SimpleNamespace(synchronize=synchronize),
            "torch_tpu._internal.compile": SimpleNamespace(
                tpu_torch_compile=SimpleNamespace(
                    get_device_layout_if_materialized=layout
                )
            ),
            "tpu_sync.api.torch.kv_cache_manager": SimpleNamespace(
                KVCacheManager=engine
            ),
            "tpu_sync.api.torch.kv_cache_store": SimpleNamespace(
                RaidenId=lambda **kwargs: kwargs
            ),
            "tpu_sync.api.torch.reshard_client": SimpleNamespace(
                ReshardClient=lambda address: client
            ),
            "tpu_sync.api.torch.reshard_store": SimpleNamespace(
                ReshardStore=lambda *args, **kwargs: SimpleNamespace(
                    reshard_service_port=32000,
                    raiden_controller_address="10.0.0.1:32001",
                )
            ),
            "tpu_sync.rpc": SimpleNamespace(
                raiden_service_pb2=SimpleNamespace(
                    StartTransferRequest=SimpleNamespace(
                        DESCRIPTOR=SimpleNamespace(
                            fields_by_name={"wire_pool_indices": object()}
                        )
                    )
                )
            ),
        }
        with patch.dict(sys.modules, native):
            pipe = PipelineTransfer(manager, buffers)
        expected = ["engine", "pools", "layout", "work_unit"]
        if legacy:
            expected = ["materialize", "layout"] + expected
        self.assertEqual(events, expected)
        if legacy:
            pipe.engine.register_pools.assert_called_once_with(pipe.pools)
        else:
            pipe.engine.register_pools.assert_called_once_with(
                pipe.pools, staging_leases=2, staging_blocks_per_pool=[4]
            )
        pipe.executor.shutdown(wait=False)

    def test_layout_is_measured_after_native_materializes_the_cache(self):
        self._initialize()

    def test_legacy_pool_api_uses_checked_full_mirrors(self):
        self._initialize(legacy=True)

    def test_bootstrap_messages_wait_for_pool_initialization(self):
        from sglang.srt.disaggregation.raiden.conn import RaidenKVManager

        manager = RaidenKVManager.__new__(RaidenKVManager)
        manager._pipeline_enabled = True
        manager._pipeline_init_lock = threading.Lock()
        manager._pipeline_messages = []
        manager._pipeline = None
        message = [MessageTag.PIPELINE_REQ, b"7", b"tcp://decode:31000"]
        manager._dispatch(message)
        self.assertEqual(manager._pipeline_messages, [message])


if __name__ == "__main__":
    unittest.main()
