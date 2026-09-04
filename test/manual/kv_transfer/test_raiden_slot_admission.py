"""Unit tests for raiden host staging slot admission.

Pure bookkeeping, so these run anywhere -- no TPU / torch_tpu / tpu_raiden
required.

Run: python -m pytest test/manual/kv_transfer/test_raiden_slot_admission.py
"""

import unittest

from sglang.srt.disaggregation.raiden.host_pool import SlotAdmission


class TestSlotAdmission(unittest.TestCase):
    def test_admits_up_to_capacity(self):
        gate = SlotAdmission(4)
        self.assertTrue(all(gate.admit(room) for room in range(4)))
        self.assertEqual(len(gate), 4)

    def test_defers_past_capacity_instead_of_failing(self):
        gate = SlotAdmission(4)
        for room in range(4):
            gate.admit(room)
        self.assertFalse(gate.admit(99))
        self.assertEqual(len(gate), 4)

    def test_a_deferred_room_is_admitted_once_a_slot_frees(self):
        gate = SlotAdmission(2)
        gate.admit(1)
        gate.admit(2)
        self.assertFalse(gate.admit(3))
        gate.release(1)
        self.assertTrue(gate.admit(3))
        self.assertEqual(sorted(gate.rooms()), [2, 3])

    def test_readmitting_an_in_flight_room_is_idempotent(self):
        gate = SlotAdmission(1)
        self.assertTrue(gate.admit(7))
        self.assertTrue(gate.admit(7))
        self.assertEqual(len(gate), 1)

    def test_releasing_an_unknown_room_is_a_noop(self):
        gate = SlotAdmission(2)
        gate.admit(1)
        gate.release(42)
        self.assertEqual(gate.rooms(), [1])

    def test_capacity_is_at_least_one(self):
        self.assertTrue(SlotAdmission(0).admit(1))

    def test_concurrency_above_capacity_drains_in_waves(self):
        gate = SlotAdmission(4)
        pending = list(range(16))
        waves = 0
        while pending:
            admitted = [room for room in pending if gate.admit(room)]
            self.assertLessEqual(len(gate), 4)
            for room in admitted:
                gate.release(room)
                pending.remove(room)
            waves += 1
        self.assertEqual(waves, 4)

    def test_dynamic_admission_accounts_actual_full_blocks(self):
        gate = SlotAdmission(64, full_block_capacity=128)
        self.assertTrue(gate.admit("a", full_blocks=32))
        self.assertTrue(gate.admit("b", full_blocks=64))
        self.assertFalse(gate.admit("c", full_blocks=33))
        self.assertTrue(gate.admit("c", full_blocks=32))

    def test_dynamic_admission_retains_a_count_ceiling(self):
        gate = SlotAdmission(2, full_block_capacity=128)
        self.assertTrue(gate.admit("a", full_blocks=1))
        self.assertTrue(gate.admit("b", full_blocks=1))
        self.assertFalse(gate.admit("c", full_blocks=1))

    def test_dynamic_admission_reserves_full_and_swa_atomically(self):
        gate = SlotAdmission(
            8,
            full_block_capacity=128,
            swa_block_capacity=8,
        )
        self.assertTrue(gate.admit("a", full_blocks=32, swa_blocks=6))
        self.assertFalse(gate.admit("b", full_blocks=32, swa_blocks=3))
        self.assertEqual(gate.rooms(), ["a"])

    def test_dynamic_release_returns_both_block_budgets(self):
        gate = SlotAdmission(
            8,
            full_block_capacity=32,
            swa_block_capacity=4,
        )
        self.assertTrue(gate.admit("a", full_blocks=32, swa_blocks=4))
        self.assertFalse(gate.admit("b", full_blocks=1))
        gate.release("a")
        self.assertTrue(gate.admit("b", full_blocks=32, swa_blocks=4))

    def test_dynamic_readmission_does_not_double_charge(self):
        gate = SlotAdmission(8, full_block_capacity=32)
        self.assertTrue(gate.admit("a", full_blocks=32))
        self.assertTrue(gate.admit("a", full_blocks=32))
        self.assertEqual(len(gate), 1)

    def test_dynamic_admission_requires_a_block_count(self):
        gate = SlotAdmission(8, full_block_capacity=32)
        with self.assertRaises(ValueError):
            gate.admit("a")


if __name__ == "__main__":
    unittest.main()
