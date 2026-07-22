"""Host-only push tests of the reshard schedules against real raiden managers.

Feeds the schedules built by ``reshard_schedule`` (the exact builders conn.py
uses) into CPU-only raiden KV-cache managers (``HostKVCacheManager``) and
pushes real bytes over the loopback data plane:

  build_sender_request/build_receiver_request -> register_active_plan
  -> push_registered_plan -> GetBlockChunks -> WriteVExact -> host bytes

The schedule unit tests assert proto fields without moving bytes, and a
serving run with 1 kv-head per rank on both sides has ``size == dst_stride``,
so every entry degenerates to a contiguous copy. What neither covers --
interleaved strided placement, where a decode block is assembled from several
sources with ``dst_stride > size`` -- is exercised here bit-exactly.

Covers:
  * concat (total_kv_heads >= src_tp): two writers interleave distinct heads
    at dst offsets 0 and hp with dst_stride 2*hp. Uses decode_rank=1 so the
    receiver's schedule keying by sender node id (not writer index) is
    load-bearing.
  * replication (total_kv_heads < src_tp): both writers land the same head at
    the same offset -- redundant but idempotent.
  * mixed replication+concat at ratio 4 (8TP -> 2TP attention, 4 kv-heads):
    the 2TP x 4DP decode topology, four writers per decode rank.
  * fp8 packed layout (kv_packing=4): granule-aligned head splits move whole
    lane-tiled granules as opaque payload (dummy planes never read or
    written); sub-granule splits are refused at schedule build.
  * partial pages: tokens past ``live_tokens`` stay untouched.

Requires the tpu_raiden host bindings (no TPU or torch_tpu runtime):
  PYTHONPATH=$HOME/tpu-raiden python -m pytest test_reshard_host_push.py
"""

import unittest

from sglang.srt.disaggregation.raiden import reshard_schedule as rs

try:
    from tpu_raiden.api.torch import kv_cache_manager_host
    from tpu_raiden.rpc import raiden_service_pb2 as pb2

    _RAIDEN_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - environment-dependent
    kv_cache_manager_host = None
    pb2 = None
    _RAIDEN_IMPORT_ERROR = e

HP_BYTES = 512  # one kv-head's K+V plane pair: 2 * head_dim(128) * bf16
TOKENS_PER_BLOCK = 16
NUM_LAYERS = 3
NUM_HOST_BLOCKS = 4
DEST_NODE_ID = 100
POISON = 0xAA
SRC_DUMMY = 0x55  # source dummy-plane filler; must never reach a destination
ROOM = 0x51E135


def _head_pattern(layer, block, token, global_head, hp_bytes):
    """One kv-head's bytes for one token, addressed by GLOBAL head index.

    Sharding-independent ground truth: a source rank's block content and the
    expected assembled decode block are both concatenations of these runs, so
    the expectation never goes through the schedule math under test. Every
    coordinate feeds the byte value so a wrong block pairing, token stride,
    or head offset shows up as a byte mismatch, not just a shuffled layout.
    """
    return bytes(
        (global_head * 131 + layer * 37 + block * 17 + token * 7 + i) % 251
        for i in range(hp_bytes)
    )


def _block_payload(
    layer, block, global_heads, live_tokens, token_bytes, hp_bytes, pad_byte
):
    """Block bytes for a rank holding ``global_heads`` in ``token_bytes``-wide
    tokens: the real head pairs, then ``pad_byte`` dummy planes up to the
    granule-aligned token width. Tokens past ``live_tokens`` are POISON
    throughout.
    """
    out = bytearray()
    for token in range(TOKENS_PER_BLOCK):
        if token < live_tokens:
            tok = bytearray()
            for head in global_heads:
                tok += _head_pattern(layer, block, token, head, hp_bytes)
            tok += bytes([pad_byte]) * (token_bytes - len(tok))
            out += tok
        else:
            out += bytes([POISON]) * token_bytes
    return bytes(out)


def _prefill_heads(prefill_rank, src_tp, total_kv_heads):
    """Global kv-head indices held by one prefill rank (concat or replicated)."""
    if total_kv_heads >= src_tp:
        per_rank = total_kv_heads // src_tp
        return list(range(prefill_rank * per_rank, (prefill_rank + 1) * per_rank))
    replication = src_tp // total_kv_heads
    return [prefill_rank // replication]


def _decode_heads(decode_rank, dst_tp, total_kv_heads):
    per_rank = max(1, total_kv_heads // dst_tp)
    return list(range(decode_rank * per_rank, (decode_rank + 1) * per_rank))


@unittest.skipUnless(
    kv_cache_manager_host is not None,
    f"tpu_raiden host bindings unavailable: {_RAIDEN_IMPORT_ERROR}",
)
class ReshardHostPushTest(unittest.TestCase):
    """Pushes real bytes through raiden host managers with conn.py's schedules."""

    def _run_reshard(
        self,
        *,
        src_tp,
        dst_tp,
        total_kv_heads,
        decode_rank,
        src_block_ids,
        dst_block_ids,
        live_tokens,
        hp_bytes=HP_BYTES,
        kv_packing=2,
    ):
        geom = rs.ReshardGeometry(
            src_tp, dst_tp, hp_bytes, total_kv_heads, kv_packing=kv_packing
        )
        src_hpr = max(1, total_kv_heads // src_tp)
        dst_hpr = max(1, total_kv_heads // dst_tp)
        src_token = geom.token_stride_bytes(src_hpr)
        dst_token = geom.token_stride_bytes(dst_hpr)
        source_ranks = rs.source_ranks_for(decode_rank, geom.ratio)

        sources = {
            p: kv_cache_manager_host.HostKVCacheManager(
                num_layers=NUM_LAYERS,
                num_shards=1,
                slice_byte_size=TOKENS_PER_BLOCK * src_token,
                node_id=p,
                local_port=0,
                host_blocks=NUM_HOST_BLOCKS,
                parallelism=1,
            )
            for p in source_ranks
        }
        dest = kv_cache_manager_host.HostKVCacheManager(
            num_layers=NUM_LAYERS,
            num_shards=1,
            slice_byte_size=TOKENS_PER_BLOCK * dst_token,
            node_id=DEST_NODE_ID,
            local_port=0,
            host_blocks=NUM_HOST_BLOCKS,
            parallelism=1,
        )

        dst_slot = TOKENS_PER_BLOCK * dst_token
        for layer in range(NUM_LAYERS):
            for block in range(NUM_HOST_BLOCKS):
                dest.write_block_bytes(layer, block, bytes([POISON]) * dst_slot)
            for p, mgr in sources.items():
                heads = _prefill_heads(p, src_tp, total_kv_heads)
                for block, live in zip(src_block_ids, live_tokens):
                    mgr.write_block_bytes(
                        layer,
                        block,
                        _block_payload(
                            layer, block, heads, live, src_token, hp_bytes, SRC_DUMMY
                        ),
                    )

        peer = dest.transfer_address
        recv_req = rs.build_receiver_request(
            pb2,
            uuid=ROOM,
            req_id=str(ROOM),
            peer=peer,
            decode_rank=decode_rank,
            geom=geom,
            sources=[(p, list(src_block_ids)) for p in source_ranks],
            dst_block_ids=list(dst_block_ids),
            live_tokens=list(live_tokens),
        )
        # conn.py registers with dst_mem_type=HBM so the device manager
        # auto-H2Ds the assembled staging; the host manager has no device, so
        # receive into DRAM and verify the host staging bytes directly.
        recv_req.dst_mem_type = pb2.MEMORY_TYPE_DRAM
        dest.register_active_plan(ROOM, recv_req, is_sender=False)

        for p in source_ranks:
            send_req = rs.build_sender_request(
                pb2,
                uuid=ROOM,
                req_id=str(ROOM),
                peer=peer,
                prefill_rank=p,
                decode_rank=decode_rank,
                geom=geom,
                src_block_ids=list(src_block_ids),
                dst_block_ids=list(dst_block_ids),
                live_tokens=list(live_tokens),
            )
            sources[p].register_active_plan(ROOM, send_req, is_sender=True)
            for layer in range(NUM_LAYERS):
                sources[p].push_registered_plan(
                    ROOM,
                    peer,
                    list(src_block_ids),
                    list(dst_block_ids),
                    layer_idx=layer,
                    parallelism=1,
                )

        decode_heads = _decode_heads(decode_rank, dst_tp, total_kv_heads)
        for layer in range(NUM_LAYERS):
            for src_block, dst_block, live in zip(
                src_block_ids, dst_block_ids, live_tokens
            ):
                # Destination dummy planes are never written, so they must
                # still hold the dest poison — never the sources' SRC_DUMMY.
                expected = _block_payload(
                    layer, src_block, decode_heads, live, dst_token, hp_bytes, POISON
                )
                actual = dest.read_block_bytes(layer, dst_block)
                self.assertEqual(
                    actual,
                    expected,
                    msg=(
                        f"layer={layer} src_block={src_block} "
                        f"dst_block={dst_block} live={live}"
                    ),
                )
            for block in set(range(NUM_HOST_BLOCKS)) - set(dst_block_ids):
                self.assertEqual(
                    dest.read_block_bytes(layer, block),
                    bytes([POISON]) * dst_slot,
                    msg=f"untargeted dst block {block} modified (layer {layer})",
                )

    def test_concat_two_writers_interleave(self):
        # 8 kv-heads at 8TP -> 4TP: prefill ranks 2,3 hold distinct heads 2,3;
        # decode rank 1 interleaves them at dst offsets 0, hp (dst_stride 2*hp).
        # Second block is a partial page: its tail must stay poisoned.
        self._run_reshard(
            src_tp=8,
            dst_tp=4,
            total_kv_heads=8,
            decode_rank=1,
            src_block_ids=[1, 3],
            dst_block_ids=[2, 0],
            live_tokens=[TOKENS_PER_BLOCK, TOKENS_PER_BLOCK // 2],
        )

    def test_replication_two_writers_idempotent(self):
        # 4 kv-heads at 8TP -> 4TP (the Qwen3-30B shape): prefill ranks 2,3
        # both hold head 1; each pushes a full copy to the same offset.
        self._run_reshard(
            src_tp=8,
            dst_tp=4,
            total_kv_heads=4,
            decode_rank=1,
            src_block_ids=[0, 2],
            dst_block_ids=[3, 1],
            live_tokens=[TOKENS_PER_BLOCK, TOKENS_PER_BLOCK],
        )

    def test_mixed_replication_concat_ratio4(self):
        # 4 kv-heads at 8TP -> 2TP (the 2TP x 4DP decode topology): decode
        # rank 1 is fed by prefill ranks 4,5 (head 2) and 6,7 (head 3) --
        # replication AND strided concat in one schedule.
        self._run_reshard(
            src_tp=8,
            dst_tp=2,
            total_kv_heads=4,
            decode_rank=1,
            src_block_ids=[2],
            dst_block_ids=[1],
            live_tokens=[TOKENS_PER_BLOCK],
        )

    def test_fp8_sub_granule_split_refused(self):
        # fp8 with one real head per rank on either side: the head is half of
        # a lane-tiled device granule, which single-stride byte entries cannot
        # extract, so schedule building must refuse before any bytes move.
        with self.assertRaises(NotImplementedError):
            self._run_reshard(
                src_tp=8,
                dst_tp=4,
                total_kv_heads=8,
                decode_rank=1,
                src_block_ids=[1],
                dst_block_ids=[2],
                live_tokens=[TOKENS_PER_BLOCK],
                hp_bytes=256,
                kv_packing=4,
            )

    def test_fp8_granule_aligned_concat(self):
        # fp8, 16 kv-heads at 8TP -> 4TP: every rank's slice is whole granules
        # (source 1 granule/token, decode 2), writers land at granule offsets
        # 0 and 512. Granule-aligned entries move tiled granules as opaque
        # payload, so this geometry is valid on-device as well.
        self._run_reshard(
            src_tp=8,
            dst_tp=4,
            total_kv_heads=16,
            decode_rank=1,
            src_block_ids=[1, 3],
            dst_block_ids=[2, 0],
            live_tokens=[TOKENS_PER_BLOCK, TOKENS_PER_BLOCK // 2],
            hp_bytes=256,
            kv_packing=4,
        )


if __name__ == "__main__":
    unittest.main()
