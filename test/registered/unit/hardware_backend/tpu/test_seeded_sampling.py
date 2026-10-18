# SPDX-License-Identifier: Apache-2.0
"""Stateless TPU sampling: shared hash bits, rounding boundaries and row identity."""

import random
import struct
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.experimental.speculative.spec_ops import (
    accept_drafts_sampled,
    sample_tokens,
)
from sglang_tpu.sampling.seeded import (
    hash_to_uniform,
    murmur_hash32,
    sample_categorical,
    sample_with_seed,
    verify_coins,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _reference_hash(seed, position, column):
    data = struct.pack(
        "<QII", seed & ((1 << 64) - 1), position & 0xFFFFFFFF, column & 0xFFFFFFFF
    )
    state = 0
    for (block,) in struct.iter_unpack("<I", data):
        block = block * 0xCC9E2D51 & 0xFFFFFFFF
        block = ((block << 15) | (block >> 17)) & 0xFFFFFFFF
        block = block * 0x1B873593 & 0xFFFFFFFF
        state ^= block
        state = ((state << 13) | (state >> 19)) & 0xFFFFFFFF
        state = (state * 5 + 0xE6546B64) & 0xFFFFFFFF
    state ^= len(data)
    state ^= state >> 16
    state = state * 0x85EBCA6B & 0xFFFFFFFF
    state ^= state >> 13
    state = state * 0xC2B2AE35 & 0xFFFFFFFF
    return state ^ (state >> 16)


class TestSeededSampling(CustomTestCase):
    def test_hash_preserves_seed_high_bits_and_unsigned_position_words(self):
        seeds = [0, 1, -1, -(1 << 63), (1 << 32) + 17]
        positions = [0, 31, (1 << 32) + 9, 19, (1 << 62) + 7]
        columns = [0, 1, 7, 31, (1 << 32) + 3]
        actual = murmur_hash32(
            torch.tensor(seeds), torch.tensor(positions), torch.tensor(columns)
        )
        expected = [
            [_reference_hash(seed, pos, col) for col in columns]
            for seed, pos in zip(seeds, positions)
        ]
        self.assertEqual(actual.tolist(), expected)

    def test_float32_coins_match_shared_float64_intermediate_at_boundaries(self):
        # 4294966655 exposes double rounding: rounding the rational directly
        # chooses the neighboring float32 instead of the shared two-stage result.
        maximum = (1 << 32) - 1
        values = {0, maximum, 4294966655}
        for exponent in range(33):
            center = 1 << exponent
            values.update(range(max(0, center - 1024), min(maximum + 1, center + 1024)))
        rng = random.Random(371)
        values.update(rng.randrange(maximum + 1) for _ in range(20000))
        hashed = torch.tensor(sorted(values), dtype=torch.int64)
        expected = (hashed.double() / maximum).float().clamp_max(1.0 - 2**-24)
        actual = hash_to_uniform(hashed)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.all((actual >= 0) & (actual < 1)))

    def test_seeded_draws_follow_request_identity_across_padding_and_reorder(self):
        weights = torch.tensor([[0.1, 0.0, 0.9], [0.8, 0.2, 0.0], [0.0, 0.3, 0.7]])
        seeds = torch.tensor([4, -17, 1 << 40])
        positions = torch.tensor([8, 102, 78])
        baseline = sample_with_seed(weights, seeds, positions)
        order = torch.tensor([2, 0, 1])
        padded = sample_with_seed(
            torch.cat([weights[order], torch.ones(1, 3)]),
            torch.cat([seeds[order], torch.zeros(1, dtype=torch.int64)]),
            torch.cat([positions[order], torch.zeros(1, dtype=torch.int64)]),
        )
        self.assertTrue(torch.equal(padded[:3], baseline[order]))
        # The draft sampler consumes the same per-request/position categorical draw.
        actual = sample_tokens(
            weights.log(),
            torch.ones(3),
            torch.full((3,), 3),
            sampling_seed=seeds,
            positions=positions,
        )
        self.assertTrue(torch.equal(actual, baseline))

    def test_inverse_cdf_skips_zero_mass_bins_at_both_edges(self):
        weights = torch.tensor([[0.0, 2.0, 0.0, 1.0, 0.0]]).repeat(3, 1)
        actual = sample_categorical(weights, torch.tensor([0.0, 0.5, 1.0 - 2**-24]))
        self.assertEqual(actual.tolist(), [1, 1, 3])

    def test_seeded_acceptance_does_not_consume_ambient_rng(self):
        target = torch.tensor(
            [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [1.0, 2.0, 0.0]]
        ).repeat(2, 1)
        drafts = torch.tensor([[0, 1], [1, 2]])
        draft_logits = torch.tensor([[[1.0, 2.0, 0.0], [2.0, 0.0, 1.0]]]).repeat(
            2, 1, 1
        )
        coins, final = verify_coins(torch.tensor([123, -9]), torch.tensor([10, 20]), 3)
        outputs = []
        with torch.random.fork_rng():
            for seed in (1, 999):
                torch.manual_seed(seed)
                before = torch.random.get_rng_state()
                outputs.append(
                    accept_drafts_sampled(
                        target,
                        drafts,
                        torch.tensor([2, 2]),
                        draft_logits,
                        torch.ones(2),
                        torch.full((2,), 3),
                        coins=coins,
                        final_sampling_coins=final,
                    )
                )
                self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for left, right in zip(*outputs):
            self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
