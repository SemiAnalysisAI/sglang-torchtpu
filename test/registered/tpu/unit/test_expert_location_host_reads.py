# SPDX-License-Identifier: Apache-2.0
"""Building the physical-expert map must not read the map element by element.

`_compute_logical_to_all_physical_map` used to index `physical_to_logical_map`
one element at a time. `init_by_mapping` has already moved that tensor to the
accelerator, so each index was a device-to-host sync: num_layers *
num_physical_experts of them. At Kimi K3's shape (93 layers x 896 experts) that
is 83,328 syncs and about 30 minutes of startup before a single weight loads --
long enough to read as a hang, with every rank sitting in the same loop.

This runs at K3's shape on one chiplet and needs no model, so it catches a
regression in seconds instead of a 15-minute multi-host launch.
"""

import time
import unittest

import torch

from sglang.srt.eplb.expert_location import _compute_logical_to_all_physical_map
from sglang.test.test_utils import CustomTestCase

# K3's grid: the shape that made the cost visible.
NUM_LAYERS = 93
NUM_EXPERTS = 896
# One transfer plus Python loops is well under a second; 83k device syncs is
# minutes. Two orders of magnitude of headroom, so this is not a timing flake.
BUDGET_S = 20.0


class TestExpertLocationHostReads(CustomTestCase):
    def _build(self, device):
        physical_to_logical_map = (
            torch.arange(0, NUM_EXPERTS).repeat(NUM_LAYERS, 1).to(device)
        )
        t0 = time.monotonic()
        out = _compute_logical_to_all_physical_map(
            physical_to_logical_map,
            num_logical_experts=NUM_EXPERTS,
            ep_size=32,
            moe_ep_rank=None,
        )
        return out, time.monotonic() - t0

    def test_matches_cpu_and_stays_within_budget_on_tpu(self):
        expected, _ = self._build("cpu")
        actual, elapsed = self._build("tpu")

        # The function returns a tensor on the input's device, so compare on host.
        self.assertTrue(
            torch.equal(actual.cpu(), expected.cpu()),
            "the map built from a device tensor must match the CPU result",
        )
        self.assertLess(
            elapsed,
            BUDGET_S,
            f"building the map took {elapsed:.1f}s at {NUM_LAYERS}x{NUM_EXPERTS}; "
            "this is the per-element device read regressing, not slow hardware",
        )


if __name__ == "__main__":
    unittest.main()
