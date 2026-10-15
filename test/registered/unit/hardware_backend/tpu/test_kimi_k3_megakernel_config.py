# SPDX-License-Identifier: Apache-2.0
"""Server-arg contract, runtime guards, env registry and compile-cache key of the
Kimi-K3 decode megakernel. CPU-only: the TPU env setup is mocked."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.model_executor.cuda_graph_config import default_cuda_graph_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.kimi_k3_megakernel.config import (
    megakernel_context,
    validate_megakernel_server_args,
)
from sglang_tpu.kimi_k3_megakernel.runner import (
    KimiK3MegakernelRunner,
    maybe_attach_decode_megakernel,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _mk_args(**overrides):
    """A valid megakernel launch: TP32/EP8, 8 rows, radix off, 9344 context."""
    fields = dict(
        tp_size=32,
        ep_size=8,
        pp_size=1,
        dp_size=1,
        enable_dp_attention=False,
        speculative_algorithm=None,
        disaggregation_mode="null",
        enable_hierarchical_cache=False,
        enable_mixed_chunk=False,
        max_running_requests=8,
        disable_radix_cache=True,
        context_length=9344,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestValidateMegakernelServerArgs(CustomTestCase):
    def test_noop_when_disabled(self):
        args = _mk_args(tp_size=1, max_running_requests=64, disable_radix_cache=False)
        with tpu_envs.SGLANG_TPU_K3_MEGAKERNEL.override(False):
            validate_megakernel_server_args(args)
        self.assertFalse(hasattr(args, "_resolved_overrides"))

    def test_valid_launch_passes_unchanged(self):
        args = _mk_args()
        with tpu_envs.SGLANG_TPU_K3_MEGAKERNEL.override(True):
            validate_megakernel_server_args(args)
        self.assertFalse(getattr(args, "_resolved_overrides", None))

    def test_topology_must_be_tp32_ep8(self):
        with tpu_envs.SGLANG_TPU_K3_MEGAKERNEL.override(True):
            with self.assertRaisesRegex(ValueError, "--tp-size 32 --ep-size 8"):
                validate_megakernel_server_args(_mk_args(ep_size=4))

    def test_features_that_read_sglang_pools_are_refused(self):
        cases = {
            "--speculative-algorithm": dict(speculative_algorithm="EAGLE"),
            "--pipeline-parallel-size": dict(pp_size=2),
            "--dp-size": dict(dp_size=2),
            "--enable-dp-attention": dict(enable_dp_attention=True),
            "--disaggregation-mode": dict(disaggregation_mode="decode"),
            "--enable-hierarchical-cache": dict(enable_hierarchical_cache=True),
            "--enable-mixed-chunk": dict(enable_mixed_chunk=True),
        }
        with tpu_envs.SGLANG_TPU_K3_MEGAKERNEL.override(True):
            for flag, override in cases.items():
                with self.subTest(flag=flag):
                    with self.assertRaisesRegex(ValueError, flag):
                        validate_megakernel_server_args(_mk_args(**override))

    def test_max_running_requests_capped_by_rows(self):
        with tpu_envs.SGLANG_TPU_K3_MEGAKERNEL.override(True):
            with self.assertRaisesRegex(ValueError, "--max-running-requests <= 8"):
                validate_megakernel_server_args(_mk_args(max_running_requests=16))
            with tpu_envs.SGLANG_TPU_K3_MEGAKERNEL_MAX_ROWS.override(4):
                args = _mk_args(max_running_requests=None)
                validate_megakernel_server_args(args)
                self.assertEqual(resolution_result(args, "max_running_requests"), 4)
            with tpu_envs.SGLANG_TPU_K3_MEGAKERNEL_MAX_ROWS.override(16):
                with self.assertRaisesRegex(ValueError, "MAX_ROWS must be in"):
                    validate_megakernel_server_args(_mk_args())

    def test_radix_cache_is_turned_off(self):
        args = _mk_args(disable_radix_cache=False)
        with tpu_envs.SGLANG_TPU_K3_MEGAKERNEL.override(True):
            validate_megakernel_server_args(args)
        self.assertTrue(resolution_result(args, "disable_radix_cache"))

    def test_context_must_fit_megakernel_state(self):
        with tpu_envs.SGLANG_TPU_K3_MEGAKERNEL.override(True):
            with self.assertRaisesRegex(ValueError, "--context-length or"):
                validate_megakernel_server_args(_mk_args(context_length=None))
            with tpu_envs.SGLANG_TPU_K3_MEGAKERNEL_CONTEXT.override(9344):
                args = _mk_args(context_length=None)
                validate_megakernel_server_args(args)
                self.assertEqual(resolution_result(args, "context_length"), 9344)
                # Rounded up to a 128-token tile, so 9344 itself and below fit.
                validate_megakernel_server_args(_mk_args(context_length=9300))
                with self.assertRaisesRegex(ValueError, "exceeds the megakernel"):
                    validate_megakernel_server_args(_mk_args(context_length=9500))

    def test_megakernel_context_rounds_to_cache_tiles(self):
        self.assertEqual(megakernel_context(9300), 9344)
        self.assertEqual(megakernel_context(9344), 9344)
        self.assertEqual(megakernel_context(100), 384)

    def test_platform_hook_runs_the_contract(self):
        """TpuSRTPlatform.apply_server_args_defaults must enforce the contract."""
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        args = _mk_args(
            max_running_requests=16,
            dtype=None,
            attention_backend=None,
            sampling_backend=None,
            disable_cuda_graph=False,
            cuda_graph_config=default_cuda_graph_config(),
            enable_torch_compile=False,
            disable_overlap_schedule=False,
            mem_fraction_static=None,
            dcp_size=1,
            attn_dp_size=1,
            enable_dp_lm_head=False,
            enable_dynamic_chunking=False,
            page_size=None,
            kv_cache_dtype="auto",
            use_ray=False,
            hicache_io_backend=None,
            max_total_tokens=None,
            disaggregation_transfer_backend="mooncake",
            optimistic_prefill_attempts=0,
            max_queued_requests=None,
        )
        with (
            tpu_envs.SGLANG_TPU_K3_MEGAKERNEL.override(True),
            patch("sglang_tpu.patches._prepare_tpu_env") as prepare_tpu_env,
        ):
            with self.assertRaisesRegex(ValueError, "--max-running-requests <= 8"):
                TpuSRTPlatform().apply_server_args_defaults(args)
        # Refused before any TPU environment is touched.
        prepare_tpu_env.assert_not_called()


def _runner(*, max_rows: int, context: int = 384) -> KimiK3MegakernelRunner:
    def op(*operands):
        return torch.zeros((operands[-1].shape[0], 64), dtype=torch.float32)

    return KimiK3MegakernelRunner(
        weights={"m_kb": torch.zeros((1, 1, 512, 768), dtype=torch.bfloat16)},
        weight_names=("m_kb",),
        decode_ops={b: op for b in (1, 2, 4, 8) if b <= max_rows},
        max_rows=max_rows,
        context=context,
        layers=4,
        vocab_size=64,
        device="cpu",
    )


def _decode_batch(seq_lens, *, reqs=True):
    n = len(seq_lens)
    lens = torch.tensor(seq_lens, dtype=torch.int64)
    idx = torch.arange(1, n + 1, dtype=torch.int64) if reqs else torch.zeros(n).long()
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        seq_lens=lens,
        seq_lens_cpu=lens,
        req_pool_indices=idx,
        req_pool_indices_cpu=idx,
        reqs=[SimpleNamespace(rid=f"r{i}") for i in range(n)] if reqs else None,
        req_to_token_pool=None,
        token_to_kv_pool=None,
    )


class TestRunnerDecodeGuards(CustomTestCase):
    def test_real_batch_above_rows_raises_instead_of_stale_fallback(self):
        runner = _runner(max_rows=2)
        with self.assertRaisesRegex(RuntimeError, "holds 2 rows"):
            runner.prepare_batch(_decode_batch([5, 6, 7]))

    def test_warmup_batch_above_rows_still_falls_back(self):
        runner = _runner(max_rows=2)
        runner.prepare_batch(_decode_batch([1, 1, 1, 1], reqs=False))
        self.assertEqual(len(runner.slot_table), 0)

    def test_position_past_state_context_raises(self):
        runner = _runner(max_rows=2, context=384)
        with self.assertRaisesRegex(RuntimeError, "past its 384-token state"):
            runner.prepare_batch(_decode_batch([385]))

    def test_generation_counter_must_stay_on_host(self):
        pool = SimpleNamespace(req_generation=torch.zeros(4, device="meta"))
        with self.assertRaisesRegex(AssertionError, "host-resident"):
            KimiK3MegakernelRunner._extract_live_rids_and_gens(
                SimpleNamespace(reqs=None), [1, 2], pool
            )
        pool = SimpleNamespace(req_generation=torch.tensor([0, 3, 5, 0]))
        rids, gens = KimiK3MegakernelRunner._extract_live_rids_and_gens(
            SimpleNamespace(reqs=None), [1, 2], pool
        )
        self.assertEqual(gens, [3, 5])
        self.assertEqual(rids, ["__pool_1_gen_3", "__pool_2_gen_5"])

    def test_attach_is_noop_when_disabled(self):
        model = SimpleNamespace(logits_processor=SimpleNamespace())
        with tpu_envs.SGLANG_TPU_K3_MEGAKERNEL.override(False):
            maybe_attach_decode_megakernel(model)
        self.assertFalse(hasattr(model, "_megakernel_runner"))
        self.assertFalse(hasattr(model.logits_processor, "use_tpu_distributed_top_k"))


class TestRegistries(CustomTestCase):
    def test_env_vars_declared_once_in_plugin_registry(self):
        from sglang.srt.environ import Envs

        names = {
            "SGLANG_TPU_K3_MEGAKERNEL",
            "SGLANG_TPU_K3_MEGAKERNEL_CHECKPOINT",
            "SGLANG_TPU_K3_MEGAKERNEL_CONTEXT",
            "SGLANG_TPU_K3_MEGAKERNEL_MAX_ROWS",
            "SGLANG_TPU_K3_MEGAKERNEL_RECIPE",
        }
        self.assertTrue(names <= set(type(tpu_envs).names()))
        self.assertFalse(names & set(vars(Envs)))

    def test_megakernel_sources_are_in_compile_cache_key(self):
        from sglang_tpu import compile_cache

        root = Path(compile_cache.__file__).resolve().parent
        files = compile_cache._iter_source_files(
            root / p for p in compile_cache.get_pallas_kernel_sources()
        )
        self.assertIn(
            root / "kimi_k3_megakernel" / "_vendor" / "kimi" / "decode_megakernel.py",
            files,
        )


if __name__ == "__main__":
    unittest.main()
