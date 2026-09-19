"""Layer-addressed Raiden transfers from a pipeline to DP or matching PP stages."""

from __future__ import annotations

import concurrent.futures
import hashlib
import inspect
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.raiden.host_pool import (
    StagingAdmission,
    ranks_per_node,
)
from sglang.srt.disaggregation.raiden.serde import (
    MessageTag,
    read_aux_blob,
    write_aux_blob,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)


def pool_manifest(groups: Any) -> list[dict]:
    """Name each local full-attention or sliding-window tensor by model layer."""
    pools = []
    for kind, tensors, layer_ids in (
        ("fa", groups.full, groups.full_layer_ids),
        ("swa", groups.swa, groups.swa_layer_ids),
    ):
        if len(tensors) != len(layer_ids):
            raise ValueError(f"Raiden {kind} pools need one global layer ID per tensor")
        for tensor, layer in zip(tensors, layer_ids):
            if len(tensor.shape) != 5:
                raise ValueError("Raiden pipeline transfer requires rank-5 fused KV")
            size = math.prod(int(n) for n in tensor.shape[1:]) * tensor.element_size()
            pools.append(
                dict(
                    tag=f"{kind}.l{int(layer)}",
                    storage_index=len(pools),
                    base_offset_bytes=0,
                    block_stride_bytes=size,
                    num_blocks=int(tensor.shape[0]),
                    dtype_tag=str(tensor.dtype).removeprefix("torch."),
                    regions=[
                        dict(
                            name="kv",
                            offset_bytes=0,
                            stride_bytes=size,
                            unit_bytes=size,
                            num_units=1,
                        )
                    ],
                )
            )
    if not groups.full:
        raise ValueError(
            "Raiden pipeline transfer needs full-attention KV in every stage"
        )
    tags = [pool["tag"] for pool in pools]
    if len(tags) != len(set(tags)):
        raise ValueError("Raiden pipeline transfer requires distinct global layer IDs")
    return pools


def materialized_storage_layout(tensor, getter):
    """Read the layout of the live storage backing an FP8 cache dtype view."""
    layout = getter(tensor)
    if layout is None and str(tensor.dtype) in (
        "torch.float8_e4m3fn",
        "torch.float8_e5m2",
    ):
        import torch

        # An FP8 dtype view can retain uint8 base storage. Matching that dtype
        # lets the layout getter inspect the native storage reference.
        layout = getter(tensor.view(torch.uint8))
    return layout


def layout_fingerprint(pools: Sequence[dict], tensors: Sequence[Any], getter) -> str:
    """Compare per-class page geometry and materialized layout across layer subsets."""
    classes = {}
    for pool, tensor in zip(pools, tensors):
        layout = materialized_storage_layout(tensor, getter)
        if layout is None:
            raise ValueError("Raiden pipeline KV must have a materialized TPU layout")
        major, tiles, bits = layout
        signature = dict(
            page_shape=[int(n) for n in tensor.shape[1:]],
            dtype=pool["dtype_tag"],
            minor_to_major=[int(n) for n in major],
            tiles=[[int(n) for n in tile] for tile in tiles],
            element_bits=int(bits or tensor.element_size() * 8),
        )
        kind = pool["tag"].split(".l")[0]
        if kind in classes and classes[kind] != signature:
            raise ValueError(f"Raiden {kind} layers have differing page layouts")
        classes[kind] = signature
    value = json.dumps(
        dict(schema="sglang-fused-kv-v1", classes=classes), sort_keys=True
    )
    return hashlib.sha256(value.encode()).hexdigest()


def source_spans(pools: Sequence[dict], full: list[int], swa: list[int]) -> list[Any]:
    """Describe whole pages in destination byte space for each locally owned layer."""
    result = []
    for pool in pools:
        blocks = swa if pool["tag"].startswith("swa.") else full
        stride = pool["block_stride_bytes"]
        spans = [
            SimpleNamespace(
                src_block_ordinal=index,
                src_offset_bytes=0,
                dst_block_index=0,
                dst_offset_bytes=index * stride,
                size_bytes=stride,
                src_stride_bytes=0,
                dst_stride_bytes=0,
                count=1,
            )
            for index in range(len(blocks))
        ]
        result.append(
            SimpleNamespace(
                tag=pool["tag"],
                block_ids=list(blocks),
                declared_bytes=len(blocks) * stride,
                dst_space_version=1,
                spans=spans,
            )
        )
    return result


def destination_blocks(
    pools: Sequence[dict],
    full: list[int],
    swa: list[int],
    source_full_count: int,
    source_swa_count: int,
) -> dict:
    """Clip a cached prefix and omit the decode-only trailing token page."""
    tags, blocks, counts, skips = [], [], [], []
    for pool in pools:
        is_swa = pool["tag"].startswith("swa.")
        local = list(swa if is_swa else full)
        remote_count = source_swa_count if is_swa else source_full_count
        if len(local) > remote_count:
            local = local[:remote_count]
        if not local:
            continue
        tags.append(pool["tag"])
        blocks.extend(local)
        counts.append(len(local))
        skips.append((remote_count - len(local)) * pool["block_stride_bytes"])
    return dict(
        transfer_pool_tags=tags,
        dst_device_block_ids=blocks,
        dst_block_counts=counts,
        dst_skip_bytes=skips,
    )


def legacy_slot_host_bytes(tensors, max_blocks, getter):
    """Match the legacy native constructor's uniform host-storage page size."""
    first = tensors[0]
    layout = materialized_storage_layout(first, getter)
    if layout is None:
        raise ValueError("Legacy Raiden pools require materialized page layouts")
    major, tiles, _ = layout
    major = tuple(int(n) for n in major)
    if sorted(major) != list(range(len(first.shape))) or major[-1] != 0:
        raise ValueError("Legacy Raiden pools require block-major page layouts")
    page_bytes = math.prod(first.shape[1:]) * first.element_size()
    uniform = all(
        tuple(tensor.shape) == tuple(first.shape)
        and tensor.element_size() == first.element_size()
        for tensor in tensors
    )
    if not uniform and tiles:
        tile = tuple(int(n) for n in tiles[0])
        if len(tile) != 2 or min(tile) <= 0:
            raise ValueError("Legacy Raiden pools require a two-dimensional tile")
        height, width = tile
        minor_width, minor_height = major[:2]
        elements = (
            ((int(first.shape[minor_width]) + width - 1) // width)
            * ((int(first.shape[minor_height]) + height - 1) // height)
            * height
            * width
        )
        for index, dimension in enumerate(first.shape):
            if index not in (0, minor_width, minor_height):
                elements *= int(dimension)
        page_bytes = elements * first.element_size()
    return int(page_bytes) * len(tensors) * max_blocks


def legacy_mirror_slots(pools, slot_bytes: int, ceiling: int, budget: float):
    """Bound full mirrors plus one temporary replacement of an initial storage."""
    if {pool["storage_index"] for pool in pools} != set(range(len(pools))):
        raise ValueError("Legacy Raiden pipeline pools need one pool per storage")
    if any(pool["base_offset_bytes"] != 0 for pool in pools):
        raise ValueError("Legacy Raiden pipeline pools need zero storage offsets")
    extents = [pool["block_stride_bytes"] * pool["num_blocks"] for pool in pools]
    extents = [((extent + 4095) // 4096) * 4096 for extent in extents]
    initial_per_slot = slot_bytes // len(pools)
    for slots in range(ceiling, 0, -1):
        initial = ((initial_per_slot * slots + 4095) // 4096) * 4096
        final = sum(max(initial, extent) for extent in extents)
        # The native allocator creates and copies a replacement before freeing
        # that storage's original buffer; other storages retain their mirrors.
        peak = final + (initial if any(n > initial for n in extents) else 0)
        if peak <= budget:
            return slots, peak
    raise ValueError(
        "Raiden pipeline full mirrors need at least "
        f"{peak / 1024**3:.2f} GiB per rank including pool replacement, but "
        f"SGLANG_RAIDEN_HOST_POOL_GB allows {budget / 1024**3:.2f} GiB; "
        "raise the host budget, lower max-total-tokens, or use a wheel with "
        "bounded pool staging"
    )


def warn_if_staging_exceeds_premapped(staging_bytes: int, premapped_bytes: int) -> bool:
    """Full-mirror staging is carved out of libtpu's premapped DMA buffer; staging
    past the buffer is DMA-mapped as each copy lands, and those mappings stall the
    decode steps. Returns whether the staging exceeds the buffer."""
    if staging_bytes <= premapped_bytes:
        return False
    logger.warning(
        "Raiden host staging of %.1f GiB per rank exceeds libtpu's premapped DMA "
        "buffer of %.1f GiB; set TPU_PREMAPPED_BUFFER_SIZE above the staging size.",
        staging_bytes / 1024**3,
        premapped_bytes / 1024**3,
    )
    return True


@dataclass
class Receive:
    full: list[int]
    swa: list[int]
    aux_index: int | None
    ranks: set[int]
    sources: dict[int, dict]
    future: concurrent.futures.Future | None = None
    accepted: bool = False
    deadline: float = 0.0
    endpoints: tuple[str, ...] = ()
    next_attempt: float = 0.0


class PipelineTransfer:
    """One named-pool engine per worker, sharing its pipeline's reshard controller."""

    def __init__(self, manager, groups):
        from torch_tpu._internal.compile import tpu_torch_compile
        from tpu_sync.api.torch.kv_cache_manager import KVCacheManager
        from tpu_sync.api.torch.kv_cache_store import RaidenId
        from tpu_sync.api.torch.reshard_client import ReshardClient
        from tpu_sync.api.torch.reshard_store import ReshardStore
        from tpu_sync.rpc import raiden_service_pb2

        from sglang.srt.distributed import get_pp_group

        self.manager = manager
        self.producer = manager.disaggregation_mode == DisaggregationMode.PREFILL
        if manager.attn_tp_size != 1 or manager.attn_cp_size != 1:
            raise ValueError(
                "Raiden pipeline transfer requires attention TP=1 and CP=1"
            )
        if self.producer and manager.pp_size <= 1:
            raise ValueError("Raiden pipeline transfer requires a pipeline producer")
        if get_spec().speculative_algorithm is not None:
            raise ValueError(
                "Raiden pipeline transfer does not support speculative decoding"
            )
        if (
            "wire_pool_indices"
            not in raiden_service_pb2.StartTransferRequest.DESCRIPTOR.fields_by_name
        ):
            raise RuntimeError(
                "Raiden pipeline transfer requires tpu-sync PP pool-resharding support"
            )
        self.lock = threading.RLock()
        self.pools = pool_manifest(groups)
        tensors = list(groups.full) + list(groups.swa)
        self.client_type = ReshardClient
        self.clients = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="raiden-pipeline-transfer"
        )
        self.sent = {}
        self.replies = {}
        self.release_requests = {}
        self.receives: dict[int, Receive] = {}
        self.cleared = set()
        self.done_send = set()
        self.done_recv = set()
        self.failed = set()
        self.retired = set()
        self.store = None
        addresses = None
        if manager.pp_rank == 0:
            self.store = ReshardStore(
                RaidenId(
                    job_name="sglang",
                    job_replica_id=f"{manager.local_ip}:{manager._control_port}",
                    data_name="pipeline",
                    data_replica_idx=0,
                ),
                store_server_ip=manager.local_ip,
            )
            addresses = (
                NetworkAddress(
                    manager.local_ip, self.store.reshard_service_port
                ).to_host_port_str(),
                self.store.raiden_controller_address,
            )
        if manager.pp_size > 1:
            addresses = get_pp_group().broadcast_object(addresses, src=0)
        self.controller, dispatch = addresses
        self.client = ReshardClient(self.controller)
        self.unit_fields = dict(
            job_name="sglang",
            job_replica_id=self.controller,
            data_name="prefill" if self.producer else "decode",
            data_replica_idx=manager.pp_rank,
        )
        self.unit = SimpleNamespace(**self.unit_fields)
        full_blocks = manager._max_request_blocks()
        swa_blocks = manager._swa_max_request_blocks() if groups.swa else 0
        self.max_blocks = max(full_blocks, swa_blocks)
        slots = manager._num_slots(tensors, self.max_blocks)
        parameters = inspect.signature(KVCacheManager.register_pools).parameters
        bounded_pools = {
            "staging_leases",
            "staging_blocks_per_pool",
        } <= parameters.keys()
        staging_options = {}
        staging_mode = "leases"
        mirror_peak = 0
        initial_slot_bytes = 0
        if not bounded_pools:
            if os.environ.get("RAIDEN_SHM_KEY"):
                raise ValueError(
                    "Legacy pipeline full mirrors do not support RAIDEN_SHM_KEY"
                )
            if set(parameters) != {"self", "pools"}:
                raise RuntimeError("Unsupported Raiden register_pools signature")
            from torch_tpu._internal.sync.sync import synchronize

            synchronize(tensors, wait=True)
            initial_slot_bytes = legacy_slot_host_bytes(
                tensors,
                self.max_blocks,
                tpu_torch_compile.get_device_layout_if_materialized,
            )
            local_ranks = ranks_per_node(
                manager.attn_tp_size,
                manager.attn_dp_size,
                manager.system_dp_size,
                get_parallel().nnodes,
                manager.attn_cp_size,
                manager.pp_size,
            )
            budget = (
                float(os.environ.get("SGLANG_RAIDEN_HOST_POOL_GB", "128"))
                * 1024**3
                / local_ranks
            )
            slots, mirror_peak = legacy_mirror_slots(
                self.pools, initial_slot_bytes, slots, budget
            )
            staging_mode = "full_mirrors"
            warn_if_staging_exceeds_premapped(
                staging_bytes=mirror_peak,
                premapped_bytes=int(os.environ.get("TPU_PREMAPPED_BUFFER_SIZE", "0")),
            )
        else:
            staging_options = dict(
                staging_leases=slots,
                staging_blocks_per_pool=[
                    swa_blocks if p["tag"].startswith("swa.") else full_blocks
                    for p in self.pools
                ],
            )
        self.admission = StagingAdmission(slots * self.max_blocks)
        self.engine = KVCacheManager(
            kv_caches=tensors,
            node_id=manager._local_rank,
            local_control_port=manager._control_port,
            max_blocks=self.max_blocks,
            num_slots=slots,
            timeout_s=manager._timeout_s,
            listener_port=0,
            parallelism=1,
            raiden_controller_address=dispatch,
            worker_id=f"worker_{manager.pp_rank}",
        )
        self.engine.register_pools(self.pools, **staging_options)
        fingerprint = layout_fingerprint(
            self.pools, tensors, tpu_torch_compile.get_device_layout_if_materialized
        )
        self.client.register_work_unit(
            unit=self.unit,
            shards=[self.engine.transfer_address],
            control_plane_rpc_address=self.engine.listener_address,
            pool_manifest=self.pools,
            layout_fingerprint=fingerprint,
            page_tokens=manager.kv_args.page_size,
            transfer_parallelism=manager.pp_size,
            transfer_rank=manager.pp_rank,
        )
        manager.raiden_endpoint = self.controller
        logger.info(
            "Raiden pipeline ready: mode=%s pp=%d/%d local_rank=%d "
            "page_tokens=%d staging=%s slots=%d native_slot_bytes=%d mirror_peak_bytes=%d "
            "layout_fingerprint=%s pools=%s",
            "prefill" if self.producer else "decode",
            manager.pp_rank,
            manager.pp_size,
            manager._local_rank,
            manager.kv_args.page_size,
            staging_mode,
            slots,
            initial_slot_bytes,
            mirror_peak,
            fingerprint,
            json.dumps(self.pools, sort_keys=True),
        )

    def _send(self, endpoint, frames):
        self.manager._send_multipart_locked(endpoint, frames)

    @staticmethod
    def _validate_room(room: int) -> None:
        if not 0 < room < (1 << 63):
            raise ValueError(
                "Raiden pipeline bootstrap_room must be a positive signed 64-bit "
                "integer for the native transfer UUID"
            )

    def finalize_send(self, room: int, aux_index: int | None) -> bool:
        self._validate_room(room)
        mgr = self.manager
        with self.lock, mgr._state_lock:
            if room in mgr._aborted:
                return False
            full = list(mgr._prefill.send_blocks.get(room, []))
            swa = list(mgr._prefill.send_swa_blocks.get(room, []))
            aux = (
                b""
                if aux_index is None
                else read_aux_blob(
                    mgr.kv_args.aux_data_ptrs, mgr.kv_args.aux_item_lens, aux_index
                )
            )
            self.sent[room] = dict(
                full=len(full),
                swa=len(swa),
                aux=aux,
                expires=time.monotonic() + mgr._timeout_s,
            )
            try:
                self.client.register_request_blocks(
                    req_id=str(room),
                    uuid=room,
                    unit=self.unit,
                    block_ids=full,
                    pool_spans=source_spans(self.pools, full, swa),
                )
            except RuntimeError as error:
                if "Request block registration was cancelled" not in str(error):
                    raise
                self.failed.add(room)
                self.retired.add(room)
                return False
            mgr.update_status(room, KVPoll.Transferring)
            self._reply(room)
            return True

    def _reply(self, room):
        if room not in self.sent or room not in self.replies:
            return
        item = self.sent[room]
        mgr = self.manager
        metadata = dict(
            controller=self.controller,
            unit=self.unit_fields,
            rank=mgr.pp_rank,
            pp_size=mgr.pp_size,
            pool_tags=[pool["tag"] for pool in self.pools],
            full=item["full"],
            swa=item["swa"],
        )
        self._send(
            self.replies[room],
            [
                MessageTag.PIPELINE_READY,
                str(room).encode(),
                json.dumps(metadata).encode(),
                item["aux"],
            ],
        )

    def register_decode(self, room, full, swa, aux_index, infos):
        self._validate_room(room)
        mgr = self.manager
        ranks = {int(info["pp_rank"]) for info in infos}
        if len(ranks) != len(infos):
            raise ValueError(
                "Raiden pipeline sources must have distinct pipeline ranks"
            )
        with self.lock:
            self.receives[room] = Receive(
                list(full),
                list(swa),
                aux_index,
                ranks,
                {},
                deadline=time.monotonic() + mgr._timeout_s,
                endpoints=tuple(
                    NetworkAddress(info["rank_ip"], int(info["rank_port"])).to_tcp()
                    for info in infos
                ),
            )
        endpoint = NetworkAddress(mgr.local_ip, mgr.rank_port).to_tcp()
        for info in infos:
            peer = NetworkAddress(info["rank_ip"], int(info["rank_port"])).to_tcp()
            self._send(
                peer, [MessageTag.PIPELINE_REQ, str(room).encode(), endpoint.encode()]
            )

    def on_message(self, msg):
        mgr = self.manager
        room = int(msg[1])
        with self.lock, mgr._state_lock:
            if room in mgr._aborted:
                return
            if msg[0] == MessageTag.PIPELINE_RELEASE:
                mgr._require_frames(msg, MessageTag.PIPELINE_RELEASE, 3)
                if not self.producer:
                    raise ValueError("A pipeline release must target a prefill worker")
                decode_pp_size = int(msg[2])
                if decode_pp_size not in (1, mgr.pp_size):
                    raise ValueError("Pipeline release has an incompatible stage count")
                self.release_requests[room] = dict(
                    pp_size=decode_pp_size, next_attempt=0.0
                )
                self._try_release(room)
                return
            if msg[0] == MessageTag.PIPELINE_REQ:
                mgr._require_frames(msg, MessageTag.PIPELINE_REQ, 3)
                if not self.producer:
                    raise ValueError("A pipeline request must target a prefill worker")
                self.replies[room] = msg[2].decode()
                self._reply(room)
                return
            mgr._require_frames(msg, MessageTag.PIPELINE_READY, 4)
            request = self.receives.get(room)
            if request is None:
                return
            meta = json.loads(msg[2])
            rank = int(meta["rank"])
            if rank not in request.ranks:
                raise ValueError(f"Unexpected pipeline source rank {rank}")
            pp_size = int(meta["pp_size"])
            expected = set(range(pp_size)) if mgr.pp_size == 1 else {mgr.pp_rank}
            if request.ranks != expected or mgr.pp_size not in (1, pp_size):
                raise ValueError(
                    "Raiden decode must use DP or the same pipeline stage count"
                )
            if mgr.pp_size > 1 and set(meta["pool_tags"]) != {
                pool["tag"] for pool in self.pools
            }:
                raise ValueError(
                    "Matching pipelines must own the same layers per stage"
                )
            request.sources[rank] = meta
            if request.aux_index is not None and msg[3]:
                write_aux_blob(
                    mgr.kv_args.aux_data_ptrs,
                    mgr.kv_args.aux_item_lens,
                    request.aux_index,
                    msg[3],
                )

    def maybe_start_pull(self, room):
        mgr = self.manager
        with self.lock, mgr._state_lock:
            if room in mgr._aborted:
                return False
            request = self.receives.get(room)
            if request is None or set(request.sources) != request.ranks:
                return False
            if request.accepted or request.future is not None:
                return True
            if time.monotonic() < request.next_attempt:
                return False
            sources = [request.sources[rank] for rank in sorted(request.ranks)]
            addresses = {source["controller"] for source in sources}
            geometries = {(source["full"], source["swa"]) for source in sources}
            if len(addresses) != 1 or len(geometries) != 1:
                raise ValueError(
                    "Raiden pipeline sources disagree on request geometry or controller"
                )
            address = addresses.pop()
            source_full, source_swa = geometries.pop()
            args = destination_blocks(
                self.pools, request.full, request.swa, source_full, source_swa
            )
            if not args["transfer_pool_tags"]:
                for endpoint in request.endpoints:
                    self._send(
                        endpoint,
                        [
                            MessageTag.PIPELINE_RELEASE,
                            str(room).encode(),
                            str(mgr.pp_size).encode(),
                        ],
                    )
                self.done_recv.add(room)
                mgr._completed_recv.add(str(room))
                return True
            if not self.admission.admit(room, self.max_blocks):
                return False
            if address not in self.clients:
                self.clients[address] = self.client_type(address)
            client = self.clients[address]
            mgr.update_status(room, KVPoll.Transferring)
            request.future = self.executor.submit(
                client.coordinate_transfer,
                src_units=[
                    SimpleNamespace(**(sources[0]["unit"] | {"data_replica_idx": rank}))
                    for rank in range(int(sources[0]["pp_size"]))
                ],
                dst_units=[self.unit],
                req_id=str(room),
                uuid=room,
                use_block_chunks=True,
                is_sender=True,
                dst_mem_type=2,
                src_controller_address=address,
                dst_controller_address=self.controller,
                **args,
            )
            return True

    def _try_release(self, room):
        release = self.release_requests.get(room)
        if (
            release is None
            or room not in self.sent
            or room in self.done_send | self.failed
            or time.monotonic() < release["next_attempt"]
        ):
            return
        if release["pp_size"] == 1:
            if not self.client.cancel_request_blocks_if_unclaimed(str(room), room):
                return
        else:
            try:
                self.client.complete_request_blocks(str(room), room, self.unit)
            except RuntimeError as error:
                if (
                    "Only an empty producer registration may complete before"
                    not in str(error)
                ):
                    raise
                release["next_attempt"] = time.monotonic() + 0.05
                return
        self.done_send.add(room)
        self.retired.add(room)

    def active(self, room):
        with self.lock:
            if self.producer:
                return room in self.sent and room not in self.done_send | self.failed
            request = self.receives.get(room)
            return request is not None and (
                request.future is not None
                or (request.accepted and room not in self.done_recv | self.failed)
            )

    def poll(self):
        mgr = self.manager
        with self.lock:
            sent, received, failed = self.engine.poll_stats()
            self.done_send.update(int(mgr._norm_req_id(r)) for r in sent)
            self.done_recv.update(int(mgr._norm_req_id(r)) for r in received)
            self.failed.update(int(mgr._norm_req_id(r)) for r in failed)
            if self.producer:
                for room in (self.done_send | self.failed) - self.retired:
                    if room in self.sent:
                        self.client.complete_request_blocks(str(room), room, self.unit)
                    self.retired.add(room)
                for room, item in self.sent.items():
                    self._try_release(room)
                    if room in self.done_send | self.failed:
                        continue
                    if room in mgr._aborted or time.monotonic() >= item["expires"]:
                        if self.client.cancel_request_blocks_if_unclaimed(
                            str(room), room
                        ):
                            if (
                                room in self.release_requests
                                and room not in mgr._aborted
                            ):
                                self.done_send.add(room)
                            else:
                                self.failed.add(room)
                            self.retired.add(room)
            else:
                for room, request in self.receives.items():
                    if request.future is not None and request.future.done():
                        error = request.future.exception()
                        if error is not None:
                            missing = "Missing producer block registration" in str(
                                error
                            )
                            cancelled = (
                                "Request block registration was cancelled" in str(error)
                            )
                            if not missing and not cancelled:
                                raise error
                            self.admission.release(room)
                            request.future = None
                            if (
                                cancelled
                                or room in mgr._aborted
                                or time.monotonic() >= request.deadline
                            ):
                                self.failed.add(room)
                            else:
                                request.next_attempt = time.monotonic() + 0.05
                            continue
                        if not request.future.result():
                            raise RuntimeError(
                                f"Raiden pipeline transfer {room} was not accepted"
                            )
                        request.accepted = True
                        request.future = None
                    if room in self.done_recv | self.failed and request.future is None:
                        self.admission.release(room)
            with mgr._state_lock:
                mgr._completed_send.update(str(room) for room in self.done_send)
                mgr._completed_recv.update(
                    str(room)
                    for room in self.done_recv
                    if room in self.receives and self.receives[room].future is None
                )
                mgr._native_failed.update(str(room) for room in self.failed)
                mgr._failed.update(str(room) for room in self.failed)
            for room in list(self.cleared):
                if not self.active(room):
                    self._forget(room)

    def clear(self, room):
        with self.lock:
            if self.active(room):
                self.cleared.add(room)
            else:
                self._forget(room)

    def _forget(self, room):
        self.sent.pop(room, None)
        self.replies.pop(room, None)
        self.release_requests.pop(room, None)
        self.receives.pop(room, None)
        self.done_send.discard(room)
        self.done_recv.discard(room)
        self.failed.discard(room)
        self.retired.discard(room)
        self.cleared.discard(room)
        self.admission.forget(room)
