"""Unit tests for raiden host staging slot sizing.

Pure arithmetic over tensor-shaped stubs, so these run anywhere -- no TPU /
torch_tpu / tpu_raiden required.

Run: python -m pytest test/manual/kv_transfer/test_raiden_slot_budget.py
"""

import unittest

from sglang.srt.disaggregation.raiden.host_pool import (
    slot_host_bytes,
    slots_within_budget,
)

GB = 1024**3


class _Buffer:
    """Stand-in for one layer's fused KV tensor ([num_blocks, page, ...])."""

    def __init__(self, shape, itemsize=2):
        self.shape = shape
        self._itemsize = itemsize

    def element_size(self):
        return self._itemsize


def _pool(num_layers, page_size=128, kv_heads=1, head_dim=128):
    # One rank's fused K+V pool: [num_blocks, page_size, 2, kv_heads, head_dim].
    return [
        _Buffer((4096, page_size, 2, kv_heads, head_dim)) for _ in range(num_layers)
    ]


class TestSlotHostBytes(unittest.TestCase):
    def test_counts_every_layer(self):
        page = 128 * 2 * 1 * 128 * 2
        self.assertEqual(slot_host_bytes(_pool(1), 64), page * 64)
        self.assertEqual(slot_host_bytes(_pool(48), 64), page * 48 * 64)

    def test_scales_with_max_blocks(self):
        small = slot_host_bytes(_pool(48), 64)
        large = slot_host_bytes(_pool(48), 2048)
        self.assertEqual(large, small * 32)

    def test_uses_the_widest_layer(self):
        mixed = _pool(2) + [_Buffer((4096, 128, 2, 16, 128))]
        widest = 128 * 2 * 16 * 128 * 2
        self.assertEqual(slot_host_bytes(mixed, 8), widest * 3 * 8)


class TestSlotsWithinBudget(unittest.TestCase):
    def test_divides_budget_by_slot_cost(self):
        self.assertEqual(slots_within_budget(16 * GB, 1 * GB, 64), 16)

    def test_clamps_to_the_ceiling(self):
        self.assertEqual(slots_within_budget(16 * GB, 1 * GB, 4), 4)

    def test_never_returns_zero(self):
        self.assertEqual(slots_within_budget(1 * GB, 8 * GB, 64), 1)

    def test_long_context_lowers_the_count_instead_of_overcommitting(self):
        # 8k context at page 128 is 64 blocks/slot; 256k context is 2048.
        budget = 16 * GB
        short = slots_within_budget(budget, slot_host_bytes(_pool(48), 64), 64)
        long = slots_within_budget(budget, slot_host_bytes(_pool(48), 2048), 64)
        self.assertEqual(short, 64)
        self.assertLess(long, short)
        self.assertGreaterEqual(long, 1)


if __name__ == "__main__":
    unittest.main()
