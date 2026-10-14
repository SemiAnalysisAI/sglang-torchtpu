# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for TPU sampling-batch host metadata hooks."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks.sampling_batch_info import (
    _HOST_PARAMS_ATTR,
    copy_basic_sampling_params,
    filter_basic_sampling_params,
    initialize_basic_sampling_params,
    merge_basic_sampling_params,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(value: int, seed: int | None = None):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(
            temperature=float(value),
            top_p=1.0 - value / 100,
            top_k=value,
            min_p=value / 100,
            sampling_seed=seed,
        )
    )


def _info(batch_size: int) -> SamplingBatchInfo:
    return SamplingBatchInfo(
        temperatures=torch.ones(batch_size, 1),
        top_ps=torch.ones(batch_size),
        top_ks=torch.full((batch_size,), TOP_K_ALL, dtype=torch.int32),
        min_ps=torch.zeros(batch_size),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=32,
        device="cpu",
        penalizer_orchestrator=MagicMock(is_required=False),
    )


class TestTpuSamplingBatchInfoHook(CustomTestCase):
    def _initialize(self, info, values, *, deterministic=False):
        batch = SimpleNamespace(reqs=[_req(value, value + 100) for value in values])
        execution = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
        )
        with patch(
            "sglang_tpu.hooks.sampling_batch_info.get_exec", return_value=execution
        ):
            initialize_basic_sampling_params(info, batch, 32)

    def test_merge_rebuilds_sampling_tensors_from_host(self):
        left, right = _info(2), _info(1)
        self._initialize(left, (10, 20), deterministic=True)
        self._initialize(right, (30,), deterministic=True)

        handled = merge_basic_sampling_params(left, right)

        self.assertTrue(handled)
        self.assertEqual(left.temperatures.flatten().tolist(), [10.0, 20.0, 30.0])
        self.assertEqual(left.top_ks.tolist(), [10, 20, 30])
        self.assertEqual(left.sampling_seed.tolist(), [110, 120, 130])

    def test_filter_keeps_host_rows_aligned(self):
        info = _info(3)
        self._initialize(info, (10, 20, 30), deterministic=True)

        filter_basic_sampling_params(info, [2, 0], torch.tensor([2, 0]))

        params = getattr(info, _HOST_PARAMS_ATTR)
        self.assertEqual(params.top_ks, (30, 10))
        self.assertEqual(params.sampling_seed, (130, 110))

    def test_missing_host_state_falls_back_to_core_merge(self):
        left, right = _info(1), _info(1)
        self._initialize(left, (10,))

        self.assertFalse(merge_basic_sampling_params(left, right))
        self.assertIsNone(getattr(left, _HOST_PARAMS_ATTR))

    def test_forward_copy_preserves_immutable_host_state(self):
        info = _info(1)
        copied = _info(1)
        self._initialize(info, (10,))

        result = copy_basic_sampling_params(copied, info)

        self.assertIs(result, copied)
        self.assertIs(
            getattr(copied, _HOST_PARAMS_ATTR), getattr(info, _HOST_PARAMS_ATTR)
        )


if __name__ == "__main__":
    unittest.main()
