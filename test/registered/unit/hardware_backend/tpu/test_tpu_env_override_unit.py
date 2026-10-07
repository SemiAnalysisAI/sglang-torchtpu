"""Unit tests for TPU LIBTPU_INIT_ARGS env overrides.

Covers the override contract: default flags are appended, the VMEM
prefetch serialization applies only to small explicit KV pools, and a
caller-supplied value for the same flag name — even a different value —
is never clobbered or duplicated.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.env_override import (
    _ALL_REDUCE_COMBINER_OVERRIDES,
    _DYNAMIC_SMEM_OVERRIDES,
    _VMEM_PREFETCH_MIN_LAYERS,
    _VMEM_PREFETCH_OVERRIDES,
    _VMEM_PREFETCH_SAFE_MIN_TOKENS,
    apply_small_kv_pool_vmem_prefetch_override,
    apply_tpu_env_overrides,
)
from sglang.test.test_utils import CustomTestCase


class TestLibtpuInitArgsOverrides(CustomTestCase):
    def _apply(
        self,
        initial: str | None,
        max_total_tokens: int | None,
        num_hidden_layers: int = _VMEM_PREFETCH_MIN_LAYERS + 1,
    ) -> str:
        env = {} if initial is None else {"LIBTPU_INIT_ARGS": initial}
        with patch.dict(os.environ, env, clear=False):
            if initial is None:
                os.environ.pop("LIBTPU_INIT_ARGS", None)
            apply_tpu_env_overrides()
            apply_small_kv_pool_vmem_prefetch_override(
                max_total_tokens=max_total_tokens,
                num_hidden_layers=num_hidden_layers,
            )
            return os.environ.get("LIBTPU_INIT_ARGS", "")

    def test_small_pool_serializes_vmem_prefetches(self):
        # The device deadlocks large programs at small KV pools when VMEM
        # prefetches overlap; a small explicit pool must get the serializing
        # flags.
        args = self._apply(None, _VMEM_PREFETCH_SAFE_MIN_TOKENS - 1).split()
        for flag in _VMEM_PREFETCH_OVERRIDES:
            self.assertIn(flag, args)

    def test_large_or_unset_pool_keeps_default_schedule(self):
        # Unaffected deployments must keep the default prefetch schedule
        # (bit-identical numerics): no serializing flags at the boundary
        # value or when no explicit pool size is requested.
        for max_total_tokens in (None, _VMEM_PREFETCH_SAFE_MIN_TOKENS):
            args = self._apply(None, max_total_tokens).split()
            for flag in _VMEM_PREFETCH_OVERRIDES:
                self.assertNotIn(flag, args)

    def test_small_model_keeps_default_schedule(self):
        # Models below the measured layer threshold never deadlocked at any
        # pool size; capping their prefetches broke an unrelated small-pool
        # CI configuration, so the containment must not touch them.
        args = self._apply(
            None,
            _VMEM_PREFETCH_SAFE_MIN_TOKENS - 1,
            num_hidden_layers=_VMEM_PREFETCH_MIN_LAYERS - 1,
        ).split()
        for flag in _VMEM_PREFETCH_OVERRIDES:
            self.assertNotIn(flag, args)

    def test_user_value_for_same_flag_name_wins(self):
        # A user-tuned value must be preserved, and the default for the same
        # flag must not be appended alongside it (libtpu would see both).
        user_flag = "--xla_gf_vmem_max_outstanding_prefetches=4"
        args = self._apply(user_flag, _VMEM_PREFETCH_SAFE_MIN_TOKENS - 1).split()
        self.assertIn(user_flag, args)
        self.assertNotIn("--xla_gf_vmem_max_outstanding_prefetches=1", args)

    def test_unrelated_user_flags_are_preserved(self):
        user_flag = "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false"
        args = self._apply(user_flag, _VMEM_PREFETCH_SAFE_MIN_TOKENS - 1).split()
        self.assertIn(user_flag, args)
        for flag in _VMEM_PREFETCH_OVERRIDES:
            self.assertIn(flag, args)


_SC_ALL_GATHER_FLAG = "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"
_OPT_IN_ENVS = (
    envs.SGLANG_TPU_DISABLE_ALL_REDUCE_COMBINER,
    envs.SGLANG_TPU_ENABLE_DYNAMIC_SMEM_NEGOTIATION,
    envs.SGLANG_TPU_ENABLE_SC_ALL_GATHER_OFFLOAD_THRESHOLD,
)


class TestOptInLibtpuFlags(CustomTestCase):
    def _apply(self, enabled: bool, initial: str = "", tpu_type: str = "tpu7x-8"):
        with ExitStack() as stack:
            for env in _OPT_IN_ENVS:
                stack.enter_context(env.override(enabled))
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {"LIBTPU_INIT_ARGS": initial, "TPU_ACCELERATOR_TYPE": tpu_type},
                )
            )
            apply_tpu_env_overrides()
            return os.environ["LIBTPU_INIT_ARGS"].split()

    def test_flags_off_by_default(self):
        args = self._apply(False)
        for flag in _ALL_REDUCE_COMBINER_OVERRIDES + _DYNAMIC_SMEM_OVERRIDES:
            self.assertNotIn(flag, args)
        self.assertFalse(any(a.startswith(_SC_ALL_GATHER_FLAG) for a in args))

    def test_opt_in_appends_flags(self):
        args = self._apply(True)
        for flag in _ALL_REDUCE_COMBINER_OVERRIDES + _DYNAMIC_SMEM_OVERRIDES:
            self.assertIn(flag, args)
        self.assertIn(f"{_SC_ALL_GATHER_FLAG}={64 << 20}", args)

    def test_sc_all_gather_threshold_follows_chip(self):
        args = self._apply(True, tpu_type="v6e-8")
        self.assertIn(f"{_SC_ALL_GATHER_FLAG}={128 << 20}", args)
        args = self._apply(True, tpu_type="unknown-chip")
        self.assertFalse(any(a.startswith(_SC_ALL_GATHER_FLAG) for a in args))

    def test_user_value_wins_over_opt_in(self):
        user_flags = (
            f"--xla_jf_crs_combiner_threshold_in_bytes=1024 {_SC_ALL_GATHER_FLAG}=1"
        )
        args = self._apply(True, initial=user_flags)
        self.assertIn("--xla_jf_crs_combiner_threshold_in_bytes=1024", args)
        self.assertNotIn(_ALL_REDUCE_COMBINER_OVERRIDES[0], args)
        self.assertEqual(
            [a for a in args if a.startswith(_SC_ALL_GATHER_FLAG)],
            [f"{_SC_ALL_GATHER_FLAG}=1"],
        )


if __name__ == "__main__":
    unittest.main()
