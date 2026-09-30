# SPDX-License-Identifier: Apache-2.0

"""The same TPU embedding executable must accept every rank's shard bounds."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.tpu.compilation.direct_compile import direct_compile
from sglang.srt.layers import vocab_parallel_embedding as vocab
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit")


@unittest.skipUnless(hasattr(torch, "tpu") and torch.tpu.is_available(), "requires TPU")
class TestVocabParallelEmbedding(CustomTestCase):
    def test_pruned_executable_reuses_tensor_shard_bounds(self):
        # Only the local lookup is compiled, so a single TPU can exercise the
        # first, last, padded and empty shards without initializing collectives.
        with (
            patch.object(
                vocab,
                "get_parallel",
                return_value=SimpleNamespace(tp_rank=0, tp_size=16),
            ),
            patch.object(vocab, "get_tp_group", return_value=None),
            patch.object(vocab, "is_allocation_symmetric", return_value=False),
            patch.object(vocab, "use_symmetric_memory", return_value=nullcontext()),
        ):
            for size, original_size in ((1009, 1009), (1043, 1010), (17, 17)):
                with self.subTest(size=size, original_size=original_size):
                    self._check_shards(size, original_size)

    def _check_shards(self, size, original_size):
        # Match CPU construction followed by model placement as well as the
        # usual construction directly on a TPU. Nonpersistent metadata must move.
        with torch.device("cpu"):
            layer = vocab.VocabParallelEmbedding(
                size, 8, org_num_embeddings=original_size, params_dtype=torch.bfloat16
            )
        self.assertNotIn("_tpu_vocab_shard_bounds", layer.state_dict())
        layer = layer.to("tpu")
        self.assertEqual(layer._tpu_vocab_shard_bounds.device.type, "tpu")
        layer.register_buffer("unused", torch.ones(2, device="tpu"), persistent=False)
        weight = (
            torch.arange(layer.weight.numel(), device="cpu")
            .remainder(127)
            .reshape(layer.weight.shape)
            .to(torch.bfloat16)
            / 16
        )
        with torch.no_grad():
            layer.weight.copy_(weight.to("tpu"))
        compiled = direct_compile(
            layer._embed_local_shard,
            specialization_key=lambda ids: ids.dtype,
            model=layer,
            prune_unused_inputs=True,
        )

        for dtype in (torch.int32, torch.int64):
            # Exhaust the small vocabulary, including invalid ids on either side.
            tokens = torch.arange(-1, size + 1, dtype=dtype, device="cpu")
            ids = tokens.to("tpu")
            for rank in (0, 15, 0):
                shard = layer._get_indices(
                    layer.num_embeddings_padded,
                    layer.org_vocab_size_padded,
                    size,
                    original_size,
                    rank,
                    16,
                )
                bounds = torch.tensor(
                    [
                        shard.org_vocab_start_index,
                        shard.org_vocab_end_index,
                        shard.num_org_vocab_padding,
                        shard.added_vocab_start_index,
                        shard.added_vocab_end_index,
                    ],
                    dtype=torch.int64,
                    device="tpu",
                )
                layer._tpu_vocab_shard_bounds.copy_(bounds)
                expected = torch.zeros(
                    (len(tokens), 8), dtype=weight.dtype, device="cpu"
                )
                for row, token in enumerate(tokens.tolist()):
                    if shard.org_vocab_start_index <= token < shard.org_vocab_end_index:
                        expected[row] = weight[token - shard.org_vocab_start_index]
                    elif (
                        shard.added_vocab_start_index
                        <= token
                        < shard.added_vocab_end_index
                    ):
                        index = (
                            shard.num_org_elements_padded
                            + token
                            - shard.added_vocab_start_index
                        )
                        expected[row] = weight[index]
                with torch.no_grad():
                    actual = compiled(ids)
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

        self.assertEqual(len(compiled.cache), 2)
        for artifact in compiled.cache.values():
            self.assertEqual(len(artifact.model_inputs), 2)
            self.assertTrue(any(t is layer.weight for t in artifact.model_inputs))
            self.assertTrue(
                any(t is layer._tpu_vocab_shard_bounds for t in artifact.model_inputs)
            )


if __name__ == "__main__":
    unittest.main()
