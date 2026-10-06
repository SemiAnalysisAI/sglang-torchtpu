"""Unit tests for the TPU platform's pipeline-parallel launch validation.

Mocks the slicebuilder env setup, so it runs without TPU hardware.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.arg_groups.overrides import declare_resolution, resolution_result
from sglang.srt.hardware_backend.tpu import pp_stage_groups
from sglang.srt.hardware_backend.tpu.tpu_platform import TpuSRTPlatform
from sglang.srt.model_executor.cuda_graph_config import default_cuda_graph_config
from sglang.test.test_utils import CustomTestCase


def _args(**overrides):
    defaults = dict(
        dtype=None,
        attention_backend=None,
        sampling_backend=None,
        disable_cuda_graph=False,
        cuda_graph_config=default_cuda_graph_config(),
        enable_torch_compile=False,
        disable_overlap_schedule=False,
        mem_fraction_static=None,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        dcp_size=1,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        enable_dynamic_chunking=False,
        page_size=None,
        kv_cache_dtype="auto",
        use_ray=False,
        hicache_io_backend=None,
        max_total_tokens=None,
        speculative_algorithm=None,
        speculative_eagle_topk=None,
        speculative_use_rejection_sampling=False,
        pp_async_batch_depth=0,
        attn_cp_size=1,
        enable_prefill_cp=False,
        cp_strategy=None,
        moe_dp_size=1,
        tpu_compile_tracer="dynamo",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _cp_stage_args(**overrides):
    """Two-device stages that run prefill context parallelism."""
    fields = dict(
        pp_size=4,
        tp_size=2,
        attn_cp_size=2,
        enable_prefill_cp=True,
        cp_strategy="zigzag",
        moe_dp_size=2,
        enable_torch_compile=True,
        sampling_backend="pytorch",
    )
    fields.update(overrides)
    return _args(**fields)


def _apply(args):
    with patch(
        "sglang.srt.hardware_backend.tpu.patches._prepare_tpu_env"
    ) as prepare_tpu_env:
        TpuSRTPlatform().apply_server_args_defaults(args)
    return prepare_tpu_env


class TestPipelineParallelValidation(CustomTestCase):
    def test_pp8_tp1_is_accepted_and_sizes_the_slice_to_eight_devices(self):
        args = _args(pp_size=8, tp_size=1, enable_torch_compile=True)
        prepare_tpu_env = _apply(args)
        prepare_tpu_env.assert_called_once_with(8, setup_slicebuilder=True)
        self.assertIsNone(args.attention_backend)
        self.assertEqual(
            resolution_result(args, "attention_backend"), "pallas_tpu_rpa_v3"
        )
        self.assertTrue(args.enable_torch_compile)

    def test_pipeline_runs_at_async_batch_depth_one(self):
        """The ICI hand-off needs the last stage's outputs received before
        each forward; depth 0 exchanges them after the forward and deadlocks."""
        args = _args(pp_size=8, tp_size=1, enable_torch_compile=True)
        _apply(args)
        self.assertEqual(args.pp_async_batch_depth, 0)
        self.assertEqual(resolution_result(args, "pp_async_batch_depth"), 1)
        args = _args(
            pp_size=8, tp_size=1, enable_torch_compile=True, pp_async_batch_depth=2
        )
        with self.assertRaisesRegex(NotImplementedError, "pp-async-batch-depth 1"):
            _apply(args)
        args = _args(tp_size=8)
        _apply(args)
        self.assertEqual(args.pp_async_batch_depth, 0)

    def test_pipeline_validation_honors_prior_resolution(self):
        args = _args(enable_torch_compile=True)
        declare_resolution(args, "test.pipeline", pp_size=8)
        prepare_tpu_env = _apply(args)
        prepare_tpu_env.assert_called_once_with(8, setup_slicebuilder=True)
        self.assertEqual(args.pp_size, 1)
        self.assertEqual(args.pp_async_batch_depth, 0)
        self.assertEqual(resolution_result(args, "pp_async_batch_depth"), 1)

    def test_resolved_unsupported_depth_is_rejected_before_environment_setup(self):
        args = _args(pp_size=8)
        declare_resolution(args, "test.pipeline", pp_async_batch_depth=2)
        with patch(
            "sglang.srt.hardware_backend.tpu.patches._prepare_tpu_env"
        ) as prepare:
            with self.assertRaisesRegex(NotImplementedError, "pp-async-batch-depth 1"):
                TpuSRTPlatform().apply_server_args_defaults(args)
        prepare.assert_not_called()
        self.assertEqual(args.pp_async_batch_depth, 0)

    def test_tp_only_slice_size_is_unchanged(self):
        prepare_tpu_env = _apply(_args(tp_size=8))
        prepare_tpu_env.assert_called_once_with(8, setup_slicebuilder=True)

    def test_accepts_tensor_parallel_pair_stages(self):
        with patch.dict("os.environ", {}):
            prepare_tpu_env = _apply(
                _args(pp_size=8, tp_size=2, enable_torch_compile=True)
            )
        prepare_tpu_env.assert_called_once_with(16, setup_slicebuilder=True)

    def test_rejects_tensor_parallel_stages_wider_than_a_pair(self):
        with self.assertRaisesRegex(NotImplementedError, r"tensor-parallel pair"):
            _apply(_args(pp_size=2, tp_size=4, enable_torch_compile=True))

    def test_rejects_stages_wider_than_their_context_parallel_group(self):
        with self.assertRaisesRegex(
            NotImplementedError, r"--attn-cp-size equal to --tp-size"
        ):
            _apply(_cp_stage_args(tp_size=4, attn_cp_size=2))

    def test_accepts_context_parallel_stages(self):
        prepare_tpu_env = _apply(_cp_stage_args())
        prepare_tpu_env.assert_called_once_with(8, setup_slicebuilder=True)

    def test_context_parallel_stages_reject_an_fp8_kv_cache(self):
        """Context-parallel stages serve only a bf16 KV cache, so an explicit
        fp8 dtype fails at launch instead of serving."""
        with self.assertRaisesRegex(NotImplementedError, r"only a bf16 KV cache"):
            _apply(_cp_stage_args(kv_cache_dtype="fp8_e4m3"))

    def test_context_parallel_stages_default_to_a_bf16_kv_cache(self):
        """The TPU fp8 default for the RPA backends does not apply to
        context-parallel stages."""
        args = _cp_stage_args()
        _apply(args)
        self.assertEqual(resolution_result(args, "kv_cache_dtype"), "bfloat16")

    def test_rejects_speculative_decoding(self):
        with self.assertRaisesRegex(
            NotImplementedError, r"Speculative decoding.*pipeline parallelism"
        ):
            _apply(_args(pp_size=2, speculative_algorithm="EAGLE3"))

    def test_rejects_dynamic_chunking(self):
        with self.assertRaisesRegex(
            NotImplementedError, r"--enable-dynamic-chunking.*pipeline"
        ):
            _apply(_args(pp_size=2, enable_dynamic_chunking=True))

    def test_rejects_eager_forward(self):
        """An eager forward gets the received activations at the hand-off
        buffer's row count, not the batch's token count."""
        with self.assertRaisesRegex(NotImplementedError, "--enable-torch-compile"):
            _apply(_args(pp_size=2))

    def test_rejection_happens_before_the_slice_env_is_prepared(self):
        with patch(
            "sglang.srt.hardware_backend.tpu.patches._prepare_tpu_env"
        ) as prepare_tpu_env:
            with self.assertRaises(NotImplementedError):
                TpuSRTPlatform().apply_server_args_defaults(
                    _cp_stage_args(tp_size=4, attn_cp_size=2)
                )
        prepare_tpu_env.assert_not_called()

    def test_single_stage_keeps_every_flag(self):
        args = _args(pp_size=1, tp_size=8, enable_dynamic_chunking=True)
        _apply(args)
        self.assertTrue(args.enable_dynamic_chunking)


def _installed_make_standalone():
    try:
        from torch_tpu._internal.distributed.spmd_util import make_standalone
    except ImportError:
        return None
    return make_standalone


def _group(*ranks):
    return SimpleNamespace(
        unique_name=f"group{ranks}",
        ranks=list(ranks),
        world_size=len(ranks),
        device_group=SimpleNamespace(ranks=ranks),
    )


def _parallel(*, pp_size, tp_size, stage, **groups):
    """World rank ``stage.ranks[0]``: groups not named are one-device groups."""
    single = _group(stage.ranks[0])
    fields = dict(
        tp_group=stage,
        attn_tp_group=single,
        attn_cp_group=single,
        moe_dp_group=single,
        moe_ep_group=single,
        moe_tp_group=single,
    )
    fields.update(groups)
    return SimpleNamespace(pp_size=pp_size, tp_size=tp_size, pp_rank=1, **fields)


class TestStandaloneStageGroups(CustomTestCase):
    def _made_standalone(self, parallel):
        make_standalone = Mock()
        with patch.object(
            pp_stage_groups, "_load_make_standalone", return_value=make_standalone
        ):
            pp_stage_groups.make_stage_groups_standalone(parallel)
        return [call.args[0] for call in make_standalone.call_args_list]

    def test_every_multi_device_group_of_a_stage_is_made_standalone_once(self):
        # PP4 x PCP2 --moe-dp-size 2: one stage group under three names.
        stage = _group(2, 3)
        parallel = _parallel(
            pp_size=4,
            tp_size=2,
            stage=stage,
            attn_cp_group=stage,
            moe_dp_group=stage,
            pp_group=_group(0, 2, 4, 6),
        )
        self.assertEqual(self._made_standalone(parallel), [stage.device_group])
        # PP2 x PCP4 --moe-dp-size 2 --ep-size 2: expert groups inside the stage,
        # in the same order on every member.
        stage, experts = _group(4, 5, 6, 7), _group(4, 5)
        parallel = _parallel(
            pp_size=2,
            tp_size=4,
            stage=stage,
            attn_cp_group=stage,
            moe_dp_group=stage,
            moe_ep_group=experts,
        )
        self.assertEqual(
            self._made_standalone(parallel),
            [stage.device_group, experts.device_group],
        )
        # PP4 x TP2: the tensor-parallel pair is the stage's only group.
        stage = _group(2, 3)
        parallel = _parallel(
            pp_size=4, tp_size=2, stage=stage, attn_tp_group=stage, moe_tp_group=stage
        )
        self.assertEqual(self._made_standalone(parallel), [stage.device_group])

    def test_one_device_stages_and_one_stage_do_not_need_the_api(self):
        load = Mock(side_effect=AssertionError("loaded make_standalone"))
        layouts = (
            _parallel(pp_size=4, tp_size=1, stage=_group(1)),
            _parallel(
                pp_size=1, tp_size=2, stage=_group(0, 1), attn_cp_group=_group(0, 1)
            ),
            _parallel(pp_size=1, tp_size=1, stage=_group(0)),
        )
        with patch.object(pp_stage_groups, "_load_make_standalone", load):
            for parallel in layouts:
                pp_stage_groups.make_stage_groups_standalone(parallel)
        load.assert_not_called()

    def test_a_torch_tpu_without_standalone_subgroups_fails_at_startup(self):
        """A context-parallel pipeline on a torch_tpu without standalone
        subgroups stalls at its first stage collective, so it must not start."""
        stage = _group(2, 3)
        parallel = _parallel(pp_size=4, tp_size=2, stage=stage, attn_cp_group=stage)
        without_api = types.ModuleType("torch_tpu._internal.distributed.spmd_util")
        with patch.dict(
            sys.modules, {"torch_tpu._internal.distributed.spmd_util": without_api}
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"standalone process subgroups \(google-pytorch/torch_tpu#4387\)",
            ):
                pp_stage_groups.make_stage_groups_standalone(parallel)

    @unittest.skipUnless(
        _installed_make_standalone() is not None,
        "the installed torch_tpu has no standalone process subgroups "
        "(google-pytorch/torch_tpu#4387)",
    )
    def test_installed_api_takes_the_process_group_alone(self):
        params = inspect.signature(_installed_make_standalone()).parameters
        self.assertEqual(len(params), 1)


if __name__ == "__main__":
    unittest.main()
