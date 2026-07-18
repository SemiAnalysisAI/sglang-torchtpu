import unittest

import torch

from sglang.srt.layers.vocab_parallel_embedding import (
    _get_masked_input_and_mask_rank_uniform,
    get_masked_input_and_mask,
)


class TestRankUniformVocabMask(unittest.TestCase):
    def _check(self, input_ids, org_start, org_end, org_padding, added_start, added_end):
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        added_offset = added_start - (org_end - org_start) - org_padding
        bounds = torch.tensor(
            [org_start, org_end, added_start, added_end, added_offset],
            dtype=torch.int64,
        )
        expected = get_masked_input_and_mask(
            input_ids, org_start, org_end, org_padding, added_start, added_end
        )
        actual = _get_masked_input_and_mask_rank_uniform(input_ids, bounds)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_rank_zero_original_vocab(self):
        self._check([0, 1, 31, 32, 63, 95], 0, 32, 0, 96, 96)

    def test_nonzero_rank_original_vocab(self):
        self._check([0, 31, 32, 63, 64, 95], 32, 64, 0, 96, 96)

    def test_added_vocab_with_original_padding(self):
        self._check([31, 32, 63, 95, 96, 99, 100], 32, 64, 8, 96, 100)


if __name__ == "__main__":
    unittest.main()
