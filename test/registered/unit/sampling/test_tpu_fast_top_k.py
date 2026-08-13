"""CPU-side dispatch tests for the opt-in TPU exact top-k sampler."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.tpu.top_k import tpu_top_k
from sglang.srt.layers import sampler as sampler_module
from sglang.srt.layers.sampler import Sampler
from sglang.test.test_utils import CustomTestCase


def _sampling_info(*, uniform_top_k):
    return SimpleNamespace(
        need_top_k_sampling=True,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        uniform_top_k=uniform_top_k,
        sampling_seed=None,
        top_ks=torch.tensor([50], dtype=torch.int32),
        top_ps=torch.ones(1),
        min_ps=torch.zeros(1),
    )


class TestTpuFastTopKDispatch(CustomTestCase):
    def test_uniform_top_k_uses_exact_tpu_path(self):
        sampler = SimpleNamespace(use_tpu_fast_top_k=True)
        probs = torch.rand(1, 128)
        positions = torch.tensor([7])
        info = _sampling_info(uniform_top_k=50)
        expected = torch.tensor([91], dtype=torch.int32)

        with (
            patch.object(
                sampler_module,
                "get_exec",
                return_value=SimpleNamespace(
                    kernel=SimpleNamespace(sampling_backend="pytorch")
                ),
            ),
            patch.object(
                sampler_module,
                "top_k_sampling_from_probs_tpu",
                return_value=expected,
            ) as fast_top_k,
        ):
            actual = Sampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=False
            )

        self.assertIs(actual, expected)
        fast_top_k.assert_called_once_with(probs, 50, None, positions)

    def test_mixed_top_k_falls_back_to_generic_sampler(self):
        sampler = SimpleNamespace(use_tpu_fast_top_k=True)
        probs = torch.rand(1, 128)
        positions = torch.tensor([7])
        info = _sampling_info(uniform_top_k=None)
        expected = torch.tensor([17], dtype=torch.int32)

        with (
            patch.object(
                sampler_module,
                "get_exec",
                return_value=SimpleNamespace(
                    kernel=SimpleNamespace(sampling_backend="pytorch")
                ),
            ),
            patch.object(sampler_module, "top_k_sampling_from_probs_tpu") as fast_top_k,
            patch.object(
                sampler_module,
                "top_k_top_p_min_p_sampling_from_probs_torch",
                return_value=expected,
            ) as generic_sampler,
        ):
            actual = Sampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=False
            )

        self.assertIs(actual, expected)
        fast_top_k.assert_not_called()
        generic_sampler.assert_called_once_with(
            probs,
            info.top_ks,
            info.top_ps,
            info.min_ps,
            False,
            None,
            positions,
        )

    def test_primitive_rejects_cpu_tensor_before_tpu_dependencies_load(self):
        with self.assertRaisesRegex(ValueError, "requires a TPU tensor"):
            tpu_top_k(torch.ones(1, 8), 4)


if __name__ == "__main__":
    import unittest

    unittest.main()
