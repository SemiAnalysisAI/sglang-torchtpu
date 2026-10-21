# SPDX-License-Identifier: Apache-2.0
"""Checks interval accounting only; this is not a kernel accuracy test."""
import unittest

from profile_analyze import outermost, union_ns


def event(device, start, end):
    return {"device": device, "start_ns": start, "end_ns": end}


class IntervalAccountingTest(unittest.TestCase):
    def test_nested_and_duplicate_events_do_not_add_work(self):
        parent = event("TPU:0", 10, 30)
        rows = [event("TPU:0", 15, 20), parent, dict(parent), event("TPU:0", 40, 50)]
        self.assertEqual(outermost(rows), [parent, rows[-1]])
        self.assertEqual(union_ns(rows), 30)

    def test_devices_are_not_deduplicated_against_each_other(self):
        rows = [event("TPU:0", 0, 20), event("TPU:1", 2, 10)]
        self.assertEqual(outermost(rows), rows)

    def test_partial_overlap_is_counted_once(self):
        self.assertEqual(union_ns([event("TPU:0", 0, 10), event("TPU:0", 8, 15)]), 15)


if __name__ == "__main__":
    unittest.main()
