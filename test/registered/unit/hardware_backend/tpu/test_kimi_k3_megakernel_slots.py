"""Row bookkeeping for the Kimi-K3 megakernel decode slots."""

import random
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kimi_k3_megakernel.slots import RowMove, SlotTable, row_bucket

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRowBucket(CustomTestCase):
    def test_smallest_tile_that_fits(self):
        self.assertEqual([row_bucket(n) for n in range(1, 9)], [1, 2, 4, 4, 8, 8, 8, 8])
        with self.assertRaises(ValueError):
            row_bucket(9)


class TestSlotTable(CustomTestCase):
    def test_assign_is_dense_and_release_compacts(self):
        table = SlotTable(4)
        self.assertEqual([table.assign(r) for r in "abc"], [0, 1, 2])
        self.assertEqual(table.release("a"), RowMove(source=2, destination=0))
        self.assertEqual((table.row("c"), table.row("b")), (0, 1))
        self.assertIsNone(table.release("b"))
        self.assertEqual(len(table), 1)
        self.assertEqual(table.assign("d"), 1)

    def test_full_table_rejects(self):
        table = SlotTable(2)
        table.assign("a")
        table.assign("b")
        with self.assertRaises(RuntimeError):
            table.assign("c")
        with self.assertRaises(ValueError):
            SlotTable(3)

    def test_random_churn_keeps_rows_dense_and_moves_consistent(self):
        rng = random.Random(0)
        table = SlotTable(8)
        # Simulated device rows: row -> request whose state the row holds.
        device = {}
        live = []
        for step in range(5000):
            if live and (len(live) == 8 or rng.random() < 0.5):
                rid = live.pop(rng.randrange(len(live)))
                move = table.release(rid)
                if move is not None:
                    device[move.destination] = device[move.source]
                device.pop(len(live), None)
            else:
                rid = f"r{step}"
                row = table.assign(rid)
                device[row] = rid
                live.append(rid)
            self.assertEqual(sorted(table.rows_for(live)), list(range(len(live))))
            for rid in live:
                self.assertEqual(device[table.row(rid)], rid)


if __name__ == "__main__":
    unittest.main()
