"""Strided KV resharding schedules for heterogeneous attention-TP disaggregation.

When prefill attention-TP > decode attention-TP (e.g. 8TP prefill -> 4TP decode
under DP-attention), a decode rank's kv-head slice spans several prefill ranks.
Which prefill bytes land where depends on ``total_kv_heads``:

  * total_kv_heads >= src_tp (e.g. MHA, 8 kv-heads at 8TP): each prefill rank owns
    a DISTINCT kv-head range, and decode rank r CONCATENATES the ranges from its
    ``ratio = src_tp // dst_tp`` prefill ranks (writers land at offsets 0, w, 2w…).
  * total_kv_heads < src_tp (e.g. Qwen3 GQA, 4 kv-heads at 8TP): kv-heads are
    REPLICATED across prefill ranks, so a decode rank's head is available whole
    from any of its prefill sources; each writer lands the same head at the same
    offset (redundant but idempotent).

Both are captured by ``head_slice_params`` (the same math as sglang's
``common.staging_buffer.compute_head_slice_params``), reimplemented here so this
module stays free of torch/triton and the byte-math is unit-testable anywhere.

The fused KV pool stores one ``[num_blocks, page_size, kv_heads_packed,
kv_packing, padded_head_dim]`` tensor per layer, laid out as a sequence of
head-dim-length PLANES in order ``K_0, V_0, K_1, V_1, ...`` — head ``h``'s K+V
is always the contiguous plane pair at plane index ``2h`` (``kv_packing`` is
outer to ``padded_head_dim``), so a per-token slice of ``num_heads`` heads is a
contiguous ``num_heads * head_pair_bytes`` run. ``kv_packing = 32 // dtype_bits``
groups planes into 32-bit-aligned granules: for 16-bit KV a granule is exactly
one head's pair, while sub-16-bit KV (e.g. fp8, packing 4) packs two heads per
granule and pads a partly-filled last granule with dummy planes. The dummy
planes carry no payload — they widen the per-token stride
(``token_stride_bytes``) but are never inside a copied range. These builders
turn that geometry into ``raiden_service_pb2.StartTransferRequest`` push
schedules.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def reshard_ratio(src_tp: int, dst_tp: int) -> int:
    """Prefill ranks feeding one decode rank. Requires src_tp a multiple of dst_tp."""
    if src_tp % dst_tp != 0:
        raise ValueError(
            f"src attn TP ({src_tp}) must be a multiple of dst attn TP ({dst_tp})"
        )
    return src_tp // dst_tp


def decode_rank_for(prefill_rank: int, ratio: int) -> int:
    """Decode rank that prefill ``prefill_rank`` contributes to."""
    return prefill_rank // ratio


def writer_index(prefill_rank: int, ratio: int) -> int:
    """Position of ``prefill_rank`` among the ``ratio`` writers of its decode rank."""
    return prefill_rank % ratio


def source_ranks_for(decode_rank: int, ratio: int) -> List[int]:
    """Prefill ranks feeding decode rank ``decode_rank``, in writer order."""
    return list(range(decode_rank * ratio, (decode_rank + 1) * ratio))


def head_pair_bytes(fused_layer_tensor) -> int:
    """Bytes of one kv-head's contiguous K+V plane pair in a fused KV layer
    tensor.

    Two ``padded_head_dim`` planes (K and V), independent of the per-rank
    kv-head count and of ``kv_packing``, so it matches on prefill and decode.
    ``tensor.shape = [num_blocks, page_size, kv_heads_packed, kv_packing,
    padded_head_dim]``.
    """
    shape = fused_layer_tensor.shape
    return 2 * int(shape[4]) * fused_layer_tensor.element_size()


def kv_packing_of(fused_layer_tensor) -> int:
    """kv_packing (planes per 32-bit granule) of a fused KV layer tensor."""
    return int(fused_layer_tensor.shape[3])


def per_rank_head_count(fused_layer_tensor) -> int:
    """This rank's kv-head count from a fused KV layer tensor's plane capacity.

    Exact for 16-bit KV (``kv_packing == 2``: every granule is one head's K+V
    pair). For sub-16-bit KV the trailing dummy planes of a partly-filled last
    granule are indistinguishable from real heads, so this overcounts; the
    real count must then come from model config, not the tensor shape.
    """
    shape = fused_layer_tensor.shape
    return int(shape[2]) * int(shape[3]) // 2


def head_slice_params(
    src_tp: int,
    dst_tp: int,
    src_rank: int,
    dst_rank: int,
    total_kv_heads: int,
) -> Tuple[int, int, int, int, int]:
    """Head-slice mapping for one (src_rank -> dst_rank) contribution.

    Returns ``(src_head_start, num_heads, dst_head_start, src_heads_per_rank,
    dst_heads_per_rank)`` in kv-head units. Mirrors sglang's
    ``compute_head_slice_params`` and additionally returns both per-rank head
    counts (the src/dst per-token strides).
    """
    src_heads_per_rank = max(1, total_kv_heads // src_tp)
    dst_heads_per_rank = max(1, total_kv_heads // dst_tp)
    local_src = src_rank % src_tp
    dst_in_group = dst_rank % dst_tp
    if src_tp > dst_tp:
        src_head_start = 0
        num_heads = src_heads_per_rank
        src_replication = max(1, src_tp // max(1, total_kv_heads))
        unique_head_idx = local_src // src_replication
        dst_head_start = (unique_head_idx * src_heads_per_rank) % dst_heads_per_rank
    else:
        src_head_start = (dst_in_group * dst_heads_per_rank) % src_heads_per_rank
        num_heads = dst_heads_per_rank
        dst_head_start = 0
    return (
        src_head_start,
        num_heads,
        dst_head_start,
        src_heads_per_rank,
        dst_heads_per_rank,
    )


class ReshardGeometry:
    """Byte geometry for one (src_tp -> dst_tp) reshard over ``total_kv_heads``.

    ``hp_bytes`` is one head's K+V plane pair (``head_pair_bytes``);
    ``kv_packing`` is the fused pool's planes-per-granule (``kv_packing_of``),
    which sets the granule-aligned per-token stride on each side.
    """

    def __init__(
        self,
        src_tp: int,
        dst_tp: int,
        hp_bytes: int,
        total_kv_heads: int,
        kv_packing: int = 2,
    ):
        if hp_bytes % 2 != 0:
            raise ValueError(f"hp_bytes ({hp_bytes}) must be even (K+V planes)")
        if kv_packing < 2 or kv_packing % 2 != 0:
            raise ValueError(f"kv_packing ({kv_packing}) must be a positive even")
        self.src_tp = src_tp
        self.dst_tp = dst_tp
        self.ratio = reshard_ratio(src_tp, dst_tp)
        self.hp_bytes = hp_bytes
        self.total_kv_heads = total_kv_heads
        self.kv_packing = kv_packing

    def token_stride_bytes(self, heads_per_rank: int) -> int:
        """Per-token byte width of a rank holding ``heads_per_rank`` kv-heads.

        The rank's ``2 * heads`` planes rounded up to whole ``kv_packing``
        granules; the round-up covers the trailing dummy planes of a
        partly-filled last granule (sub-16-bit KV with fewer real planes than
        the granule holds), which pad the stride but carry no payload.
        """
        planes = 2 * heads_per_rank
        granules = -(-planes // self.kv_packing)
        return granules * self.kv_packing * (self.hp_bytes // 2)

    def entry_bytes(self, prefill_rank: int, decode_rank: int):
        """Byte (src_offset, size, src_stride, dst_offset, dst_stride) for the
        prefill_rank -> decode_rank contribution, per token.

        Offsets and size address real head pairs (head ``h`` starts at
        ``h * hp_bytes``); the strides are granule-aligned token widths, so
        dummy planes are skipped over, never copied.

        Every offset and the size must land on whole granules: the device pool
        stores a granule TILED — its ``kv_packing`` planes interleaved into
        32-bit lane words — so raw DMA bytes are in that tiled order, and only
        granule-aligned ranges hold the same bytes in tiled and plane order.
        A sub-granule range (one head of an fp8 two-head granule) is a byte
        interleave of several heads and cannot be expressed as a single-stride
        copy; such splits raise instead of building silently corrupt schedules.
        """
        sh_start, num_heads, dh_start, src_hpr, dst_hpr = head_slice_params(
            self.src_tp, self.dst_tp, prefill_rank, decode_rank, self.total_kv_heads
        )
        for name, heads in (
            ("src head offset", sh_start),
            ("dst head offset", dh_start),
            ("head count", num_heads),
        ):
            if (2 * heads) % self.kv_packing != 0:
                raise NotImplementedError(
                    f"raiden reshard: {name} of {heads} kv-head(s) splits a "
                    f"{self.kv_packing}-plane granule (device granules are "
                    "lane-tiled; sub-granule slices need a strided byte gather)."
                    " Requires granule-aligned head splits — for fp8 an even "
                    "kv-head count per rank on both sides."
                )
        return (
            sh_start * self.hp_bytes,  # src_offset
            num_heads * self.hp_bytes,  # size
            self.token_stride_bytes(src_hpr),  # src_stride (src per-token width)
            dh_start * self.hp_bytes,  # dst_offset
            self.token_stride_bytes(dst_hpr),  # dst_stride (dst per-token width)
        )


def _add_entry(
    schedule,
    *,
    peer: str,
    src_block_id: int,
    dst_block_id: int,
    dst_shard_idx: int,
    src_offset: int,
    dst_offset: int,
    size: int,
    src_stride: int,
    dst_stride: int,
    count: int,
) -> None:
    entry = schedule.entries.add()
    entry.dst_peer = peer
    entry.dst_shard_idx = dst_shard_idx
    entry.src_block_id = src_block_id
    entry.dst_block_id = dst_block_id
    entry.src_offset_bytes = src_offset
    entry.dst_offset_bytes = dst_offset
    entry.size_bytes = size
    entry.src_stride_bytes = src_stride
    entry.dst_stride_bytes = dst_stride
    entry.count = count


def build_sender_request(
    pb2,
    *,
    uuid: int,
    req_id: str,
    peer: str,
    prefill_rank: int,
    decode_rank: int,
    geom: ReshardGeometry,
    src_block_ids: Sequence[int],
    dst_block_ids: Sequence[int],
    live_tokens: Sequence[int],
):
    """Sender-side StartTransferRequest for one prefill rank pushing to one decode peer.

    ``src_block_ids``/``dst_block_ids`` are physical page ids on each side, paired
    by token order; ``live_tokens[i]`` is the token count (``count``) of page ``i``.
    """
    if not (len(src_block_ids) == len(dst_block_ids) == len(live_tokens)):
        raise ValueError(
            "src_block_ids, dst_block_ids, live_tokens must be equal length"
        )
    src_off, size, src_stride, dst_off, dst_stride = geom.entry_bytes(
        prefill_rank, decode_rank
    )
    req = pb2.StartTransferRequest()
    req.uuid = uuid
    req.req_id = req_id
    req.is_sender = True
    req.dst_mem_type = pb2.MEMORY_TYPE_DRAM  # push reads the sender host staging pool
    req.use_block_chunks = True
    schedule = req.shard_push_schedules[0]
    for src_b, dst_b, n in zip(src_block_ids, dst_block_ids, live_tokens):
        if n <= 0:
            continue
        _add_entry(
            schedule,
            peer=peer,
            src_block_id=int(src_b),
            dst_block_id=int(dst_b),
            dst_shard_idx=0,
            src_offset=src_off,
            dst_offset=dst_off,
            size=size,
            src_stride=src_stride,
            dst_stride=dst_stride,
            count=int(n),
        )
    return req


def build_receiver_request(
    pb2,
    *,
    uuid: int,
    req_id: str,
    peer: str,
    decode_rank: int,
    geom: ReshardGeometry,
    sources: List[Tuple[int, Sequence[int]]],
    dst_block_ids: Sequence[int],
    live_tokens: Sequence[int],
    expected_block_count: Optional[int] = None,
):
    """Receiver-side StartTransferRequest assembling all sources into decode HBM.

    ``sources`` is ``[(prefill_rank, src_block_ids), ...]`` for the prefill ranks
    feeding this decode rank; each gets its own shard schedule with its strided
    dst placement. The receiver selects a source's schedule by the pushing node's
    id (``header.remote_id``), so each schedule is keyed by its ``prefill_rank``
    (the sender engine's ``node_id``), not by writer position. ``dst_mem_type =
    HBM`` makes the receiver auto-H2D the assembled host staging into HBM.
    """
    req = pb2.StartTransferRequest()
    req.uuid = uuid
    req.req_id = req_id
    req.is_sender = False
    req.dst_mem_type = pb2.MEMORY_TYPE_HBM  # auto-H2D host staging -> decode HBM
    req.use_block_chunks = True
    total = 0
    for prefill_rank, src_block_ids in sources:
        src_off, size, src_stride, dst_off, dst_stride = geom.entry_bytes(
            prefill_rank, decode_rank
        )
        schedule = req.shard_push_schedules[prefill_rank]
        for src_b, dst_b, n in zip(src_block_ids, dst_block_ids, live_tokens):
            if n <= 0:
                continue
            _add_entry(
                schedule,
                peer=peer,
                src_block_id=int(src_b),
                dst_block_id=int(dst_b),
                dst_shard_idx=0,
                src_offset=src_off,
                dst_offset=dst_off,
                size=size,
                src_stride=src_stride,
                dst_stride=dst_stride,
                count=int(n),
            )
            total += 1
    req.expected_block_count = (
        total if expected_block_count is None else int(expected_block_count)
    )
    return req
