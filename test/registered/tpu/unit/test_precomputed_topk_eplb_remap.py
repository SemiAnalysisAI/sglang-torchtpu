# SPDX-License-Identifier: Apache-2.0
"""A router that precomputes its own top-k must still go through EPLB's remap.

`build_precomputed_topk_output` is the entry Kimi K3's TPU gate uses: the fused
gate kernel produces (weights, ids) itself, so `select_experts` -- and with it
the logical->physical remap -- never runs. That left K3 out of EPLB entirely:
the placement solver could move an expert and the dispatch would keep sending
tokens to the logical id.

Two properties are checked, and the second one is the easy one to get wrong:

* the remap is applied, so ids come back physical;
* it is applied BEFORE the distribution recorder sees them. `select_experts`
  records post-remap ids and the accumulator converts physical counts back to
  logical, so recording logical ids would be miscounted as physical.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.topk import TopKConfig, build_precomputed_topk_output
from sglang.test.test_utils import CustomTestCase

NUM_LOGICAL = 8
NUM_TOKENS = 4
TOP_K = 2


def _nontrivial_map():
    """logical i -> physical (i + NUM_LOGICAL): identity would hide a missing remap."""
    return torch.arange(NUM_LOGICAL, dtype=torch.int32) + NUM_LOGICAL


class _FakeInfo:
    """Static dispatch: one gather through a non-identity map."""

    ep_dispatch_algorithm = "static"

    def __init__(self):
        self.partial_logical_to_rank_dispatch_physical_map = _nontrivial_map()


class _FakeDynamicInfo:
    """Dynamic dispatch, which is what `--enable-eplb` actually selects here.

    Without an a2a backend `needs_rank_invariant_dispatch` holds, so the arg
    hooks resolve `ep_dispatch_algorithm` to "dynamic", not "static". Under
    `rank_invariant` that path picks the replica by token row -- pure tensor
    arithmetic, no `torch.randint` -- which is what keeps it traceable inside
    the compiled decoder.

    Two replicas per logical expert: logical i -> {i + 8, i + 16}.
    """

    ep_dispatch_algorithm = "dynamic"
    rank_invariant = True

    def __init__(self):
        self.partial_logical_to_all_physical_map = torch.stack(
            [
                torch.arange(NUM_LOGICAL, dtype=torch.int32) + NUM_LOGICAL,
                torch.arange(NUM_LOGICAL, dtype=torch.int32) + 2 * NUM_LOGICAL,
            ],
            dim=-1,
        )
        self.partial_logical_to_all_physical_map_num_valid = torch.full(
            (NUM_LOGICAL,), 2, dtype=torch.int32
        )


class TestPrecomputedTopKEplbRemap(CustomTestCase):
    def setUp(self):
        self.topk_ids = torch.randint(
            0, NUM_LOGICAL, (NUM_TOKENS, TOP_K), dtype=torch.int32
        )
        self.topk_weights = torch.rand(NUM_TOKENS, TOP_K)
        self.config = TopKConfig(top_k=TOP_K)

    def _run(self, remap_enabled, info_cls=None):
        seen = {}

        class _Recorder:
            def on_select_experts(self, topk_ids):
                seen["ids"] = topk_ids.clone()

        with patch(
            "sglang.srt.layers.moe.topk._eplb_remap_enabled",
            return_value=remap_enabled,
        ), patch(
            "sglang.srt.layers.moe.topk.ExpertLocationDispatchInfo.init_new",
            return_value=(info_cls or _FakeInfo)(),
        ), patch(
            "sglang.srt.layers.moe.topk.get_global_expert_distribution_recorder",
            return_value=_Recorder(),
        ):
            out = build_precomputed_topk_output(
                self.topk_weights, self.topk_ids, self.config, layer_id=0
            )
        return out, seen["ids"]

    def test_remap_applied_and_recorder_sees_physical_ids(self):
        out, recorded = self._run(remap_enabled=True)
        expected = _nontrivial_map()[self.topk_ids.long()]

        self.assertTrue(
            torch.equal(out.topk_ids.cpu(), expected),
            "the returned ids must be physical, not the router's logical ids",
        )
        self.assertTrue(
            torch.equal(recorded.cpu(), expected),
            "the recorder must see post-remap ids; select_experts records those "
            "and the accumulator converts physical counts back to logical",
        )

    def test_dynamic_dispatch_spreads_over_replicas(self):
        """`--enable-eplb` resolves to dynamic here, so cover that path too."""
        out, recorded = self._run(remap_enabled=True, info_cls=_FakeDynamicInfo)

        # Every id must land on one of its logical expert's two replicas.
        logical = self.topk_ids.long()
        replicas = _FakeDynamicInfo().partial_logical_to_all_physical_map[logical]
        got = out.topk_ids.cpu().long().unsqueeze(-1)
        self.assertTrue(
            (replicas == got).any(-1).all(),
            "each token must be dispatched to a replica of its logical expert",
        )
        self.assertTrue(
            torch.equal(recorded.cpu(), out.topk_ids.cpu()),
            "the recorder must see the same post-remap ids",
        )

    def test_identity_when_the_gate_is_off(self):
        out, recorded = self._run(remap_enabled=False)
        self.assertTrue(torch.equal(out.topk_ids.cpu(), self.topk_ids))
        self.assertTrue(torch.equal(recorded.cpu(), self.topk_ids))


if __name__ == "__main__":
    unittest.main()
