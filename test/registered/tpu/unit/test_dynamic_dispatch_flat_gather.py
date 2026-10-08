# SPDX-License-Identifier: Apache-2.0
"""The dynamic dispatch gather must stay flat, and must stay correct.

`_topk_ids_logical_to_physical_dynamic` reads the replica map with one flat
gather instead of the 2-D advanced index `map[rows, cols]` it used to use. The
two forms compute the same thing; they do not cost the same. On TPU the 2-D form
cost ~137 ms per decode step on Qwen3-30B-A3B at EP8 -- 14x the entire step --
and the flat form costs nothing measurable.

This pins the equivalence, so a future rewrite back to the readable 2-D form
fails here instead of quietly reintroducing a 14x regression.
"""

import unittest

import torch

from sglang.srt.eplb.expert_location_dispatch import (
    _topk_ids_logical_to_physical_dynamic,
)
from sglang.test.test_utils import CustomTestCase

NUM_LOGICAL = 16
NUM_REPLICAS = 3
NUM_TOKENS = 12
TOP_K = 4


class _Info:
    ep_dispatch_algorithm = "dynamic"
    rank_invariant = True

    def __init__(self, device):
        # logical i -> physical {i, i+16, i+32}: non-identity in every column, so
        # an implementation that drops the replica axis cannot pass.
        self.partial_logical_to_all_physical_map = torch.stack(
            [
                torch.arange(NUM_LOGICAL, dtype=torch.int32) + k * NUM_LOGICAL
                for k in range(NUM_REPLICAS)
            ],
            dim=-1,
        ).to(device)
        self.partial_logical_to_all_physical_map_num_valid = torch.full(
            (NUM_LOGICAL,), NUM_REPLICAS, dtype=torch.int32, device=device
        )


def _reference_2d(topk_ids, info):
    """The original 2-D advanced index, kept here only as the oracle."""
    flat = topk_ids.flatten()
    num_valid = info.partial_logical_to_all_physical_map_num_valid[flat]
    row_index = (
        torch.arange(flat.shape[0], dtype=num_valid.dtype, device=flat.device)
        // topk_ids.shape[-1]
    )
    chosen = row_index % num_valid
    out = info.partial_logical_to_all_physical_map[flat, chosen]
    return out.to(topk_ids.dtype).view(topk_ids.shape)


class TestDynamicDispatchFlatGather(CustomTestCase):
    def _check(self, device):
        info = _Info(device)
        topk_ids = torch.randint(
            0, NUM_LOGICAL, (NUM_TOKENS, TOP_K), dtype=torch.int32, device=device
        )

        actual = _topk_ids_logical_to_physical_dynamic(topk_ids, info)
        expected = _reference_2d(topk_ids, info)

        self.assertEqual(actual.shape, topk_ids.shape)
        self.assertEqual(actual.dtype, topk_ids.dtype)
        self.assertTrue(
            torch.equal(actual.cpu(), expected.cpu()),
            "the flat gather must match the 2-D advanced index it replaced",
        )

        # And the result must actually be a replica of the logical expert, so a
        # reference that is itself wrong cannot make this test vacuous.
        replicas = info.partial_logical_to_all_physical_map[topk_ids.long()]
        self.assertTrue(
            (replicas.cpu() == actual.cpu().long().unsqueeze(-1)).any(-1).all()
        )

    def test_matches_the_2d_index_on_cpu(self):
        self._check("cpu")

    def test_matches_the_2d_index_on_tpu(self):
        self._check("tpu")


if __name__ == "__main__":
    unittest.main()
