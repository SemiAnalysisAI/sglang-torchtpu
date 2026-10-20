"""DFlash's greedy draft tokens over a vocab-sharded LM head at a TP size the
Pallas distributed TopK has no kernel for (it covers TP 8 and 16; Kimi-K3
serves at TP32): every rank's shard best, gathered, must give exactly the
full-vocabulary argmax, ties included."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.dflash_utils_tpu import pick_global_best, shard_best

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _sharded_argmax(logits: torch.Tensor, tp: int, padded: int) -> torch.Tensor:
    vocab = logits.shape[1]
    per = -(-vocab // tp)
    bests = []
    for rank in range(tp):
        start = rank * per
        org = max(0, min(per, vocab - start))
        # A padded shard, as VocabParallelEmbedding lays it out.
        local = torch.full((logits.shape[0], per + padded), -1e30)
        local[:, :org] = logits[:, start : start + org]
        shard = SimpleNamespace(num_org_elements=org, org_vocab_start_index=start)
        bests.append(shard_best(local, shard))
    return pick_global_best(torch.stack(bests))


class TestGatheredArgmax(CustomTestCase):
    def test_matches_the_full_vocabulary_argmax(self):
        g = torch.Generator().manual_seed(0)
        for tp in (2, 4, 32):
            with self.subTest(tp=tp):
                logits = torch.randn(9, 1000, generator=g)
                got = _sharded_argmax(logits, tp, padded=8)
                torch.testing.assert_close(got, torch.argmax(logits, dim=-1))

    def test_ties_resolve_to_the_lowest_token_like_torch_argmax(self):
        logits = torch.zeros(3, 64)
        logits[0, [5, 40]] = 2.0  # tie across shards (tp 4: shards of 16)
        logits[1, [17, 18]] = 1.0  # tie inside one shard
        logits[2, [63, 0]] = 3.0
        got = _sharded_argmax(logits, 4, padded=0)
        torch.testing.assert_close(got, torch.argmax(logits, dim=-1))


if __name__ == "__main__":
    unittest.main()
