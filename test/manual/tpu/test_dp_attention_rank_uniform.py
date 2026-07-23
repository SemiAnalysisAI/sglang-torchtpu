# SPDX-License-Identifier: Apache-2.0
"""attn_rank0_scale must stay a runtime graph input under torch.compile.

If the per-rank 0/1 scale is folded into the traced graph as a Python
constant, each attention-TP rank compiles a different graph; XLA then
constant-folds the zero ranks' multiply and drops their collective
contributions, which desynchronizes the ranks' collective programs. These
tests compile a minimal consumer on CPU and assert the scale stays a lifted
input, byte-identical graph code across ranks, and that a cold cache under
compilation fails loudly instead of silently folding.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers import dp_attention


class AttnRank0ScaleRankUniformTest(unittest.TestCase):
    def setUp(self):
        dp_attention._ATTN_RANK0_SCALE.clear()
        torch._dynamo.reset()

    def _compiled_graph_code(self, rank: int, prewarm: bool = True) -> str:
        graphs = []

        def backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        with mock.patch.object(
            dp_attention, "get_attention_tp_rank", return_value=rank
        ):
            if prewarm:
                dp_attention.attn_rank0_scale_prewarm(
                    device=torch.device("cpu"), dtype=torch.float32
                )

            @torch.compile(backend=backend, fullgraph=True)
            def fn(x):
                return x + x * dp_attention.attn_rank0_scale(x)

            fn(torch.ones(4))
        return "\n".join(gm.code for gm in graphs)

    def test_scale_is_lifted_not_folded(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                self.setUp()
                code = self._compiled_graph_code(rank)
                self.assertNotIn("torch.tensor", code)

    def test_graph_code_identical_across_ranks(self):
        code0 = self._compiled_graph_code(0)
        self.setUp()
        code1 = self._compiled_graph_code(1)
        self.assertEqual(code0, code1)

    def test_scale_values_stay_per_rank(self):
        x = torch.ones(4)
        with mock.patch.object(dp_attention, "get_attention_tp_rank", return_value=0):
            self.assertEqual(dp_attention.attn_rank0_scale(x).item(), 1.0)
        dp_attention._ATTN_RANK0_SCALE.clear()
        with mock.patch.object(dp_attention, "get_attention_tp_rank", return_value=1):
            self.assertEqual(dp_attention.attn_rank0_scale(x).item(), 0.0)

    def test_cold_cache_under_compile_raises(self):
        with self.assertRaisesRegex(Exception, "cache is cold"):
            self._compiled_graph_code(1, prewarm=False)


if __name__ == "__main__":
    unittest.main()
