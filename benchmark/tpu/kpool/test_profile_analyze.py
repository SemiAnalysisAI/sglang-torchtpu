# SPDX-License-Identifier: Apache-2.0
"""Checks interval accounting only; this is not a kernel accuracy test."""
import unittest

from profile_analyze import fusion_context, module_ops, outermost, union_ns


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

    def test_one_ns_rounding_does_not_drop_final_device_op(self):
        module = event("TPU:0", 10, 30)
        final = {**event("TPU:0", 20, 31), "category": "XLA Ops"}
        unrelated = {**event("TPU:0", 20, 32), "category": "XLA Ops"}
        self.assertEqual(module_ops([final, unrelated], module), [final])

    def test_short_hlo_fusion_members(self):
        hlo = """HloModule example
fused_computation {
  p = f32[4] parameter(0)
  ROOT r = f32[4] add(p, p)
}
ENTRY main {
  p2 = f32[4] parameter(0)
  fused = f32[4] fusion(p2), kind=kLoop, calls=fused_computation
  ROOT kernel = f32[4] custom-call(fused), custom_call_target="tpu_custom_call"
}
"""
        fusion, members, _ = fusion_context(hlo, "fused")
        self.assertEqual((fusion, members), ("fused", ["r (add)"]))
        self.assertEqual(fusion_context(hlo, "kernel")[:2], (None, []))


if __name__ == "__main__":
    unittest.main()
