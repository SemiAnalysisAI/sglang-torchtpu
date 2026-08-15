"""On-device test: raiden host pool packs MTP draft KV as tail layers.

Requires a TPU + a built tpu_raiden extension + torch_tpu. NOT a CI unit test
(see test_tpu_raiden_hybrid_offload_unit.py for the host-only logic). Run with:

    PYTHONPATH=/path/to/sglang-torchtpu/python \
        python3 test/manual/tpu/test_tpu_raiden_mtp_offload_ondevice.py

Builds one ``TpuRaidenHostKVCache`` over a 2-layer target pool with a 1-layer
MTP draft pool packed as a tail layer (``mtp_draft_device_pools``), then
drives a write round + a load round through the shared raiden engine and
checks bit-exact data movement on BOTH pools from one page-id list — the
packed contract: draft pages share the target's slot mappings.

Like test_tpu_raiden_roundtrip_ondevice.py, the KV buffers are seeded with
eager writes BEFORE the host pool registers them with raiden and never written
again (eager writes reallocate the PJRT buffer and would orphan the
registration; production updates KV in place via kernel donation). The load
round therefore targets DIFFERENT device pages instead of evicting the source
pages: source pages -> host -> destination pages, verify the destination.
"""

import time

import torch

from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang.srt.hardware_backend.tpu.kv_pool_host import TpuRaidenHostKVCache

DEVICE = "tpu:0"
PAGE_SIZE = 16
NUM_PAGES = 16
SRC_PAGES = [1, 2, 3]
DST_PAGES = [8, 9, 10]


def _poll(ev, what, timeout_s=30):
    t0 = time.monotonic()
    while not ev.query():
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(f"{what} did not complete within {timeout_s}s")
        time.sleep(0.001)


def _make_pool(layer_num):
    return TpuFusedMHATokenToKVPool(
        size=NUM_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=64,
        layer_num=layer_num,
        device=DEVICE,
        enable_memory_saver=False,
    )


def _seed(pool, tag):
    """Give every (layer, page) a distinct constant; returns expected values.

    Eager writes rebind the underlying PJRT buffers, so this must run BEFORE
    the host pool registers them with raiden. Constants stay below 256 so
    bfloat16 represents each one exactly (bit-exact compare after the DMAs).
    """
    expected = {}
    for layer in range(pool.layer_num):
        buf = pool.get_fused_kv_buffer(layer)
        pattern = torch.zeros(buf.shape, dtype=buf.dtype)
        for page in range(NUM_PAGES):
            pattern[page] = tag + 40 * layer + page
        buf.copy_(pattern.to(DEVICE))
        expected[layer] = pattern
    return expected


def _page_tokens(pages):
    return torch.tensor(
        [p * PAGE_SIZE + i for p in pages for i in range(PAGE_SIZE)],
        dtype=torch.int64,
    )


def _verify_moved(pool, expected, what):
    """DST_PAGES must now hold SRC_PAGES' data; all other pages untouched."""
    for layer in range(pool.layer_num):
        got = pool.get_fused_kv_buffer(layer).cpu()
        want = expected[layer].clone()
        want[DST_PAGES] = want[SRC_PAGES]
        assert torch.equal(got, want), (
            f"{what} layer {layer}: KV mismatch after backup->load "
            f"(dst pages {DST_PAGES} should hold src pages {SRC_PAGES} data)"
        )
    print(f"  {what}: bit-exact on all {pool.layer_num} layer(s)")


def test_packed_mtp_round_moves_both_pools():
    print("== raiden host-pool write+load round with packed MTP tail ==")
    target_dev = _make_pool(layer_num=2)
    draft_dev = _make_pool(layer_num=1)
    expected_target = _seed(target_dev, tag=1.0)
    expected_draft = _seed(draft_dev, tag=101.0)

    host_pool = TpuRaidenHostKVCache(
        target_dev,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=PAGE_SIZE,
        layout="fused_raiden",
        pool_label="kv",
        mtp_draft_device_pools=(draft_dev,),
    )
    assert host_pool.transfer_layer_num == 3
    assert host_pool.transfer.layer_num == 3

    host_indices = host_pool.alloc(len(SRC_PAGES) * PAGE_SIZE)
    assert host_indices is not None

    # Write round: SRC_PAGES of target AND draft -> host, one page-id list.
    write_event = host_pool.backup_from_device_all_layer(
        target_dev, host_indices, _page_tokens(SRC_PAGES), "raiden"
    )
    _poll(write_event, "write-back (target + mtp tail)")

    # Load round: host -> DST_PAGES of both pools, one page-id list.
    load_event = host_pool.load_all_layer(
        target_dev, host_indices, _page_tokens(DST_PAGES)
    )
    _poll(load_event, "load-back (target + mtp tail)")
    torch.tpu.synchronize()

    _verify_moved(target_dev, expected_target, "target pool (2 layers)")
    _verify_moved(draft_dev, expected_draft, "mtp draft pool (1 layer)")


if __name__ == "__main__":
    print(f"TPU chips: {torch.tpu.device_count()}")
    test_packed_mtp_round_moves_both_pools()
    print("\nON-DEVICE RAIDEN PACKED-MTP OFFLOAD TEST PASSED")
