"""Unit tests for TorchTPU worker initialization.

Runs anywhere (no TPU required) — exercises pure Python env-var logic.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import os
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.benchmark import one_batch
from sglang.srt.distributed import bootstrap
from sglang.srt.managers.scheduler import (
    run_scheduler_process,
    run_scheduler_process_with_legacy_hook,
)
from sglang.srt.managers.scheduler_process import (
    compute_scheduler_parallel_ranks,
    initialize_scheduler_process_config,
)
from sglang.srt.platforms import current_platform
from sglang.srt.utils.patch_torch import (
    _compute_torch_tpu_numeric_rank,
    _validate_torch_tpu_rank_device,
    initialize_torch_tpu_worker,
)
from sglang.test.test_utils import CustomTestCase

_TORCHRUN_ENV_VARS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TPU_WORKER_HOSTNAMES",
)


@dataclass(frozen=True)
class _PortArgs:
    scheduler_input_ipc_name: str


class TestInitializeTorchTpuWorker(CustomTestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _TORCHRUN_ENV_VARS}
        for k in _TORCHRUN_ENV_VARS:
            os.environ.pop(k, None)
        init_backend_patcher = patch.object(current_platform, "init_backend")
        self.init_backend = init_backend_patcher.start()
        self.addCleanup(init_backend_patcher.stop)
        align_rank_patcher = patch(
            "sglang.srt.utils.patch_torch._align_torch_tpu_rank_with_device",
            side_effect=lambda world_size: int(os.environ["RANK"]),
        )
        self.align_rank = align_rank_patcher.start()
        self.addCleanup(align_rank_patcher.stop)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_skips_when_device_not_tpu(self):
        args = SimpleNamespace(
            device="cuda",
            tp_size=8,
            pp_size=1,
            nnodes=1,
            node_rank=0,
            use_ray=False,
        )
        self.assertEqual(initialize_torch_tpu_worker(args, tp_rank=3), (3, 0))
        for k in _TORCHRUN_ENV_VARS:
            self.assertNotIn(k, os.environ)
        self.init_backend.assert_not_called()
        self.align_rank.assert_not_called()

    def test_sets_torchrun_vars_for_tpu(self):
        os.environ["MASTER_ADDR"] = "10.0.0.1"
        args = SimpleNamespace(
            device="tpu",
            tp_size=4,
            pp_size=1,
            nnodes=2,
            node_rank=1,
            use_ray=False,
        )

        def assert_worker_env_is_ready():
            self.assertEqual(os.environ["RANK"], "2")
            self.assertEqual(os.environ["WORLD_SIZE"], "4")

        self.init_backend.side_effect = assert_worker_env_is_ready
        initialize_torch_tpu_worker(args, tp_rank=2)
        self.assertEqual(os.environ["RANK"], "2")
        self.assertEqual(os.environ["LOCAL_RANK"], "0")
        self.assertEqual(os.environ["WORLD_SIZE"], "4")
        self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "2")
        self.assertEqual(os.environ["MASTER_ADDR"], "10.0.0.1")
        self.assertEqual(os.environ["MASTER_PORT"], "29500")
        self.init_backend.assert_called_once_with()
        self.align_rank.assert_called_once_with(4)

    def test_multihost_requires_topological_rank0_master(self):
        args = SimpleNamespace(
            device="tpu",
            tp_size=4,
            pp_size=1,
            nnodes=2,
            node_rank=0,
            use_ray=False,
        )
        with self.assertRaisesRegex(RuntimeError, "topological rank-0 host"):
            initialize_torch_tpu_worker(args, tp_rank=0)
        self.init_backend.assert_not_called()

    def test_single_node_rank0(self):
        args = SimpleNamespace(
            device="tpu",
            tp_size=8,
            pp_size=1,
            nnodes=1,
            node_rank=0,
            use_ray=False,
        )
        initialize_torch_tpu_worker(args, tp_rank=0)
        self.assertEqual(os.environ["RANK"], "0")
        self.assertEqual(os.environ["WORLD_SIZE"], "8")
        self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "8")

    def test_pipeline_rank_is_derived_from_composite_device_id(self):
        args = SimpleNamespace(
            device="tpu",
            tp_size=4,
            pp_size=2,
            nnodes=1,
            node_rank=0,
            use_ray=False,
        )
        def align_to_device_four(world_size):
            os.environ["RANK"] = "4"
            return 4

        self.align_rank.side_effect = align_to_device_four
        self.assertEqual(
            initialize_torch_tpu_worker(args, tp_rank=2, pp_rank=0), (0, 1)
        )
        self.assertEqual(os.environ["RANK"], "4")
        self.assertEqual(os.environ["WORLD_SIZE"], "8")
        self.align_rank.assert_called_once_with(8)

    def test_master_addr_port_respect_existing(self):
        os.environ["MASTER_ADDR"] = "10.0.0.1"
        os.environ["MASTER_PORT"] = "12345"
        args = SimpleNamespace(
            device="tpu",
            tp_size=1,
            pp_size=1,
            nnodes=1,
            node_rank=0,
            use_ray=False,
        )
        initialize_torch_tpu_worker(args, tp_rank=0)
        self.assertEqual(os.environ["MASTER_ADDR"], "10.0.0.1")
        self.assertEqual(os.environ["MASTER_PORT"], "12345")

    def test_defers_device_validation_until_distributed_bootstrap(self):
        args = SimpleNamespace(
            device="tpu",
            tp_size=8,
            pp_size=1,
            nnodes=1,
            node_rank=0,
            use_ray=False,
        )
        with patch(
            "sglang.srt.utils.patch_torch._validate_torch_tpu_rank_device"
        ) as validate_rank:
            initialize_torch_tpu_worker(args, tp_rank=0)
        validate_rank.assert_not_called()


class TestOneBatchRankInitialization(CustomTestCase):
    def test_rejects_pipeline_parallelism_before_pjrt_init(self):
        with (
            patch.object(one_batch, "initialize_torch_tpu_worker") as initialize,
            self.assertRaisesRegex(ValueError, "does not support pipeline"),
        ):
            one_batch.load_model(
                SimpleNamespace(pp_size=2), port_args=None, gpu_id=0, tp_rank=0
            )
        initialize.assert_not_called()


class TestTorchTpuNumericRank(CustomTestCase):
    def test_uses_numeric_global_device_id(self):
        self.assertEqual(
            _compute_torch_tpu_numeric_rank(4, [0, 1, 4, 5, 6, 7, 2, 3], 8),
            4,
        )

    def test_requires_complete_contiguous_device_ids(self):
        with self.assertRaisesRegex(RuntimeError, "expected.*got"):
            _compute_torch_tpu_numeric_rank(4, [0, 1, 4, 5, 6, 7, 2, 2], 8)

    def test_every_rank_must_match_device(self):
        fake_tpu = SimpleNamespace(get_local_device_attributes=lambda: {"id": 4})
        with (
            patch.object(torch, "tpu", fake_tpu, create=True),
            self.assertRaisesRegex(RuntimeError, "global rank 2.*global device 4"),
        ):
            _validate_torch_tpu_rank_device(2, 0, 8)

    def test_computes_ep_rank_from_canonical_tp_rank(self):
        args = SimpleNamespace(
            tp_size=8,
            dp_size=1,
            enable_dp_attention=False,
            attn_cp_size=1,
            moe_dp_size=1,
            ep_size=8,
        )
        self.assertEqual(
            compute_scheduler_parallel_ranks(args, 4, None), (0, 0, 4, None)
        )

    def test_computes_attention_dp_and_cp_ranks_from_canonical_tp_rank(self):
        args = SimpleNamespace(
            tp_size=8,
            dp_size=2,
            enable_dp_attention=True,
            attn_cp_size=2,
            moe_dp_size=2,
            ep_size=2,
        )
        self.assertEqual(
            compute_scheduler_parallel_ranks(args, 6, 0), (1, 1, 1, 1)
        )

    def test_canonical_rank_drives_dp_port_and_all_derived_ranks(self):
        args = SimpleNamespace(
            tp_size=8,
            dp_size=2,
            enable_dp_attention=True,
            attn_cp_size=2,
            moe_dp_size=2,
            ep_size=2,
            ep_join_rank_offset=0,
        )
        port_args = _PortArgs("tcp://127.0.0.1:3100")
        with patch(
            "sglang.srt.managers.scheduler_process.initialize_torch_tpu_worker",
            return_value=(6, 0),
        ) as initialize_worker:
            config = initialize_scheduler_process_config(
                args,
                port_args,
                tp_rank=2,
                pp_rank=0,
                dp_rank=0,
                worker_ports=[3100, 3101],
            )

        initialize_worker.assert_called_once_with(args, 2, 0)
        self.assertEqual(config.tp_rank, 6)
        self.assertEqual(config.pp_rank, 0)
        self.assertEqual(config.attn_cp_rank, 1)
        self.assertEqual(config.moe_dp_rank, 1)
        self.assertEqual(config.moe_ep_rank, 1)
        self.assertEqual(config.dp_rank, 1)
        self.assertEqual(
            config.port_args.scheduler_input_ipc_name, "tcp://127.0.0.1:3101"
        )

    def test_legacy_hook_forwards_worker_ports_to_canonical_adapter(self):
        args = SimpleNamespace(
            tp_size=8,
            dp_size=2,
            enable_dp_attention=True,
            attn_cp_size=2,
            moe_dp_size=2,
            ep_size=2,
            ep_join_rank_offset=0,
        )
        port_args = _PortArgs("tcp://127.0.0.1:3100")
        writer = object()
        with patch(
            "sglang.srt.managers.scheduler."
            "run_scheduler_process_from_bootstrap_rank"
        ) as canonical_entrypoint:
            run_scheduler_process_with_legacy_hook(
                run_scheduler_process,
                args,
                port_args,
                0,
                2,
                0,
                0,
                writer,
                [3100, 3101],
            )

        canonical_entrypoint.assert_called_once_with(
            args, port_args, 0, 2, 0, 0, writer, [3100, 3101]
        )

    def test_validates_after_process_group_initializes_pjrt(self):
        order = []
        server_args = SimpleNamespace(
            enable_p2p_check=False,
            pre_warm_nccl=False,
        )
        ps = SimpleNamespace(
            gpu_id=0,
            tp_rank=0,
            tp_size=8,
            pp_rank=0,
            pp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            moe_ep_size=8,
            moe_dp_size=1,
            attn_dcp_size=1,
        )
        world_group = SimpleNamespace(world_size=8, cpu_group=None)
        parallel = SimpleNamespace(attn_tp_group="attn-tp")

        with (
            patch.object(bootstrap, "_resolve_backend", return_value="tpu"),
            patch.object(
                bootstrap, "get_available_gpu_memory", return_value=94.75
            ),
            patch.object(bootstrap, "monkey_patch_p2p_access_check"),
            patch.object(
                bootstrap,
                "_resolve_dist_init_method",
                return_value="tcp://127.0.0.1:1234",
            ),
            patch.object(bootstrap, "_set_all_reduce_flags"),
            patch.object(
                bootstrap,
                "_init_parallel_groups",
                side_effect=lambda **kwargs: order.append("process-group"),
            ),
            patch.object(
                bootstrap,
                "_validate_torch_tpu_rank_device",
                side_effect=lambda *ranks: order.append("validate"),
            ) as validate_rank,
            patch.object(bootstrap, "get_world_group", return_value=world_group),
            patch.object(bootstrap, "get_tp_group", return_value="tp"),
            patch.object(bootstrap, "get_pp_group", return_value="pp"),
            patch.object(bootstrap, "get_parallel", return_value=parallel),
        ):
            result = bootstrap.init_torch_distributed(
                server_args=server_args,
                model_config=SimpleNamespace(),
                device="tpu",
                ps=ps,
                dist_port=1234,
                is_draft_worker=False,
                local_omp_cpuid=None,
            )

        self.assertEqual(order, ["process-group", "validate"])
        validate_rank.assert_called_once_with(0, 0, 8)
        self.assertEqual(result.tp_group, "tp")

    def test_accepts_matching_rank_and_device(self):
        fake_tpu = SimpleNamespace(get_local_device_attributes=lambda: {"id": 4})
        with patch.object(torch, "tpu", fake_tpu, create=True):
            self.assertEqual(_validate_torch_tpu_rank_device(0, 1, 4), 4)

if __name__ == "__main__":
    unittest.main()
