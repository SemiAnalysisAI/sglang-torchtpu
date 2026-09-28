"""Distribution and native integration checks for distributed chain verification."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import unittest

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.patches import apply_tpu_patches
from sglang.srt.hardware_backend.tpu.spec_op_warmup import (
    SpecOpWarmupContext,
    make_target_only_sampling_args,
)
from sglang.srt.hardware_backend.tpu.spec_sampling_verify import sample_chain
from sglang.test.test_utils import CustomTestCase


class TestDistributedVerify(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        apply_tpu_patches()
        cls.sample_op = staticmethod(sample_chain)

    def test_chain_distribution_and_endpoints(self):
        # A fixed proposal token accepted with probability p, otherwise sampled
        # from p with that token removed, must recover the target distribution.
        n = 32768
        rng = np.random.default_rng(71)
        # Candidate order deliberately differs from vocabulary order.
        ids = torch.tensor([9, 2, 7, 4], dtype=torch.int32)
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        values = weights.expand(n, 2, 4).contiguous()
        indices = ids.expand(n, 2, 4).contiguous()
        candidates = torch.full((n, 2), 7, dtype=torch.int32)
        coins = torch.from_numpy(rng.random((n, 2), dtype=np.float32))
        final = torch.from_numpy(rng.random(n, dtype=np.float32))
        tokens, counts = self.sample_op(
            *[x.to("tpu") for x in (values, indices, candidates, coins, final)]
        )
        tokens, counts = tokens.cpu(), counts.cpu()
        frequencies = torch.stack([(tokens[:, 0] == i).float().mean() for i in ids])
        torch.testing.assert_close(frequencies, weights, rtol=0, atol=0.012)
        self.assertLess(abs(float(counts.float().mean()) - 0.3), 0.012)
        self.assertTrue(torch.isin(tokens[:, 0], ids).all())
        # Rejected proposal has zero conditional residual mass, and zero/one
        # coins must never choose a zero-probability padding entry.
        values = (
            torch.tensor([[[0.0, 0.25, 0.0, 0.75, 0.0]]]).expand(2, 2, 5).contiguous()
        )
        indices = (
            torch.tensor([[[0, 3, 4, 8, 10]]], dtype=torch.int32)
            .expand(2, 2, 5)
            .contiguous()
        )
        candidates = torch.full((2, 2), 3, dtype=torch.int32)
        coins, final = torch.ones(2, 2), torch.tensor([0.0, 1.0])
        tokens, counts = self.sample_op(
            *[x.to("tpu") for x in (values, indices, candidates, coins, final)]
        )
        self.assertEqual(tokens.cpu()[:, 0].tolist(), [8, 8])
        self.assertEqual(counts.cpu().tolist(), [0, 0])

    def test_distinct_rows_and_intermediate_rejection(self):
        n = 32768
        weights = torch.tensor(
            [
                [0.4, 0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4, 0.1],
                [0.1, 0.2, 0.3, 0.4],
                [0.25, 0.25, 0.25, 0.25],
            ]
        )
        ids = torch.tensor([11, 3, 7, 2], dtype=torch.int32)
        values = weights.expand(n, 4, 4).contiguous()
        indices = ids.expand(n, 4, 4).contiguous()
        candidates = (
            torch.tensor([0, 11, 7, 2], dtype=torch.int32).expand(n, 4).contiguous()
        )
        rng = np.random.default_rng(381)
        coins = torch.from_numpy(rng.random((n, 4), dtype=np.float32))
        final = torch.from_numpy(rng.random(n, dtype=np.float32))
        tokens, counts = self.sample_op(
            *[x.to("tpu") for x in (values, indices, candidates, coins, final)]
        )
        tokens, counts = tokens.cpu(), counts.cpu()
        for depth in range(4):
            selected = tokens[counts >= depth, depth]
            frequencies = torch.stack([(selected == i).float().mean() for i in ids])
            torch.testing.assert_close(frequencies, weights[depth], rtol=0, atol=0.025)
        coins = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        _, counts = self.sample_op(
            *[
                x.to("tpu")
                for x in (
                    values[:4],
                    indices[:4],
                    candidates[:4],
                    coins,
                    torch.full((4,), 0.5),
                )
            ]
        )
        self.assertEqual(counts.cpu().tolist(), [0, 1, 2, 3])

    def test_distributed_logits_filter_and_compiled_verifier(self):
        from sglang.srt.hardware_backend.tpu.spec_sampling_distributed_verify import (
            distributed_verify,
        )
        from sglang.srt.hardware_backend.tpu.spec_sampling_verify import (
            filter_distributed_logits,
        )

        # Vary support per depth, including padded k=1 rows. Global IDs are
        # deliberately far from candidate positions.
        bs, width = 64, 4
        rows = bs * width
        logits = torch.linspace(2, -2, 64).expand(rows, -1).clone()
        logits[::4] = 0
        ids = torch.arange(1000, 1064, dtype=torch.int32).expand(rows, -1).clone()
        temps = torch.full((rows, 1), 0.7)
        ks = torch.tensor([50, 64, 2, 1], dtype=torch.int32).repeat(bs)
        ps = torch.tensor([0.89, 1.0, 1.0, 1.0]).repeat(bs)
        masked = (logits / temps).masked_fill(
            torch.arange(64)[None] >= ks[:, None], -torch.inf
        )
        reference = torch.softmax(masked.double(), -1)
        keep = reference.cumsum(-1) - reference < ps[:, None]
        reference *= keep
        reference /= reference.sum(-1, keepdim=True)
        tensors = [x.to("tpu") for x in (logits, ids, temps, ks, ps)]
        actual, actual_ids = filter_distributed_logits(*tensors)
        torch.testing.assert_close(
            actual.cpu(), reference.float(), rtol=2e-5, atol=2e-7
        )
        torch.testing.assert_close(actual_ids.cpu(), ids)

        args = list(
            make_target_only_sampling_args(
                SpecOpWarmupContext(
                    torch.device("tpu"),
                    bs,
                    width,
                    64,
                    64,
                    128,
                    torch.bfloat16,
                )
            )
        )
        args[7] = tensors[0].reshape(bs, width, 64)
        args[3] = torch.full((bs, width), 1000, dtype=torch.int64, device="tpu")
        coins = torch.zeros(bs, width)
        for row in range(bs):
            # Rows reject after 0, 1, or 2 drafts, or accept all three.
            if row % width < width - 1:
                coins[row, row % width] = 1
        args[5] = coins.to("tpu")
        args[4] = args[4].flip(0).contiguous()
        args[0].fill_(-99)
        with envs.SGLANG_ENABLE_TORCH_COMPILE.override(True):
            distributed_verify(*args[:8], *tensors[1:], 1.0, 1.0)
        counts = args[2].cpu()
        self.assertEqual(counts.tolist(), [0, 1, 2, 3] * (bs // width))
        valid = torch.arange(width)[None] <= counts[:, None]
        retrieve = args[4].cpu()
        torch.testing.assert_close(
            args[1].cpu(), torch.where(valid, retrieve, -1).int()
        )
        tokens = args[0].cpu()[retrieve]
        self.assertTrue((tokens[~valid] == -99).all())
        self.assertTrue(((tokens[valid] >= 1000) & (tokens[valid] < 1064)).all())
        # A rejected proposal must not be sampled again from the residual.
        for row in range(bs):
            if counts[row] < 3:
                self.assertNotEqual(tokens[row, counts[row]].item(), 1000)


if __name__ == "__main__":
    unittest.main()
