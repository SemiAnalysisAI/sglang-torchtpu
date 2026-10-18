"""Kimi-K3 TPU decode megakernel: startup contract, row bookkeeping and handoff.

CPU-only: the handoff runs against the real TPU pool classes allocated on the
host, and the decode op is a recording stand-in.
"""

import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import Parameter

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.srt.layers.quantization.mxfp4_tpu_util import _e8m0_to_fp32
from sglang.srt.model_executor.cuda_graph_config import default_cuda_graph_config
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hooks.kimi_k3_megakernel import attach_kimi_k3_megakernel
from sglang_tpu.kimi_k3_megakernel.handoff import Handoff, HandoffSources
from sglang_tpu.kimi_k3_megakernel.policy import (
    STEP_INPUTS_ATTR,
    megakernel_forward,
    megakernel_policy,
)
from sglang_tpu.kimi_k3_megakernel.program import layer_counts, state_shapes
from sglang_tpu.kimi_k3_megakernel.runner import KimiK3MegakernelRunner
from sglang_tpu.kimi_k3_megakernel.slots import RowMove, SlotTable
from sglang_tpu.kimi_k3_megakernel.weights import (
    StackedMxfp4MoEMethod,
    e8m0_from_f32_scales,
    stack_routed_experts,
)
from sglang_tpu.kv_pool import TpuFusedMLATokenToKVPool
from sglang_tpu.mamba_pool import TpuMambaPool
from sglang_tpu.tpu_platform import TpuSRTPlatform

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_RESOLVER = "tpu.kimi_k3_megakernel"


def _launch(**overrides):
    """A valid megakernel launch as the platform hook sees it."""
    fields = dict(
        model_path="moonshotai/Kimi-K3",
        trust_remote_code=True,
        revision=None,
        json_model_override_args=None,
        tp_size=32,
        ep_size=8,
        pp_size=1,
        dp_size=1,
        attn_dp_size=1,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        dcp_size=1,
        attn_cp_size=1,
        enable_prefill_cp=False,
        speculative_algorithm=None,
        speculative_eagle_topk=None,
        speculative_use_rejection_sampling=False,
        disaggregation_mode="null",
        disaggregation_transfer_backend="mooncake",
        optimistic_prefill_attempts=0,
        max_queued_requests=None,
        enable_priority_scheduling=False,
        enable_hierarchical_cache=False,
        hicache_io_backend=None,
        enable_pdmux=False,
        enable_mixed_chunk=False,
        enable_lora=None,
        lora_paths=None,
        enable_return_hidden_states=False,
        enable_return_routed_experts=False,
        enable_deterministic_inference=False,
        enable_eplb=False,
        ep_num_redundant_experts=0,
        load_format="auto",
        enable_torch_compile=True,
        tpu_compile_tracer="dynamo",
        tpu_compile_bs=None,
        max_running_requests=None,
        context_length=9344,
        disable_radix_cache=True,
        attention_backend=None,
        kv_cache_dtype="auto",
        page_size=128,
        dtype=None,
        sampling_backend=None,
        disable_cuda_graph=False,
        cuda_graph_config=default_cuda_graph_config(),
        disable_overlap_schedule=False,
        mem_fraction_static=None,
        enable_dynamic_chunking=False,
        use_ray=False,
        max_total_tokens=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _set_enabled(test, enabled):
    """Set SGLANG_TPU_ENABLE_K3_MEGAKERNEL for one test.

    EnvField.override restores the variable only when its block exits
    normally; closing the stack from a cleanup restores it after a raising
    test too, so it cannot leak into other test files in the process.
    """
    stack = ExitStack()
    stack.enter_context(tpu_envs.SGLANG_TPU_ENABLE_K3_MEGAKERNEL.override(enabled))
    test.addCleanup(stack.close)


def _apply(args, architectures=("KimiK3ForConditionalGeneration",)):
    hf_config = SimpleNamespace(architectures=list(architectures), num_hidden_layers=93)
    with (
        patch("sglang_tpu.patches._prepare_tpu_env") as prepare_env,
        patch(
            "sglang.srt.utils.hf_transformers.config.get_config",
            return_value=hf_config,
        ),
    ):
        TpuSRTPlatform().apply_server_args_defaults(args)
    return prepare_env


def _declared_by_megakernel(args):
    return {
        field: value
        for source, declared in getattr(args, "_resolved_overrides", None) or ()
        if source == _RESOLVER
        for field, value in declared.items()
    }


class TestStartupContract(CustomTestCase):
    """SGLANG_TPU_ENABLE_K3_MEGAKERNEL launches are refused or resolved before weights load."""

    def setUp(self):
        _set_enabled(self, True)

    def test_valid_launch_resolves_rows_tiles_backend_and_kv(self):
        args = _launch()
        _apply(args)
        self.assertEqual(resolution_result(args, "max_running_requests"), 8)
        self.assertEqual(resolution_result(args, "tpu_compile_bs"), [1, 2, 4, 8])
        self.assertEqual(resolution_result(args, "attention_backend"), "pallas_tpu_mla")
        # bf16, not the platform's fp8 default for the Pallas backends.
        self.assertEqual(resolution_result(args, "kv_cache_dtype"), "bfloat16")

    def test_flag_off_declares_nothing(self):
        _set_enabled(self, False)
        args = _launch(tp_size=4, ep_size=4, disable_radix_cache=False)
        _apply(args, architectures=("Qwen3ForCausalLM",))
        self.assertEqual(_declared_by_megakernel(args), {})

    def test_each_unsupported_flag_is_refused_by_name(self):
        cases = {
            "--enable-mixed-chunk": dict(enable_mixed_chunk=True),
            "--speculative-algorithm": dict(speculative_algorithm="EAGLE"),
            "--decode-context-parallel-size": dict(dcp_size=2),
            "--attention-context-parallel-size": dict(attn_cp_size=2),
            "--pipeline-parallel-size": dict(pp_size=2),
            "--dp-size": dict(dp_size=2),
            "--attention-data-parallel-size": dict(attn_dp_size=2),
            "--disaggregation-mode": dict(disaggregation_mode="decode"),
            "--enable-hierarchical-cache": dict(enable_hierarchical_cache=True),
            "--enable-pdmux": dict(enable_pdmux=True),
            "--kv-cache-dtype fp8_e4m3": dict(kv_cache_dtype="fp8_e4m3"),
            "--enable-lora": dict(enable_lora=True),
            "--lora-paths": dict(lora_paths=["adapter"]),
            "--enable-return-hidden-states": dict(enable_return_hidden_states=True),
            "--enable-return-routed-experts": dict(enable_return_routed_experts=True),
            "--enable-deterministic-inference": dict(
                enable_deterministic_inference=True
            ),
            "--enable-eplb": dict(enable_eplb=True),
            "--ep-num-redundant-experts": dict(ep_num_redundant_experts=8),
            "--load-format dummy": dict(load_format="dummy"),
            "--tpu-compile-tracer direct": dict(tpu_compile_tracer="direct"),
            "--attention-backend pallas_tpu_rpa_v3": dict(
                attention_backend="pallas_tpu_rpa_v3"
            ),
            "--enable-torch-compile": dict(enable_torch_compile=False),
            "--context-length": dict(context_length=None),
            "--disable-radix-cache": dict(disable_radix_cache=False),
            "--tp-size 32 --ep-size 8": dict(ep_size=32),
        }
        for flag, override in cases.items():
            with self.subTest(flag=flag):
                args = _launch(**override)
                with self.assertRaisesRegex(ValueError, flag):
                    _apply(args)
                self.assertEqual(_declared_by_megakernel(args), {})

    def test_refusal_precedes_tpu_env_setup(self):
        with patch("sglang_tpu.patches._prepare_tpu_env") as prepare_env:
            with self.assertRaises(ValueError):
                _apply(_launch(enable_mixed_chunk=True))
        prepare_env.assert_not_called()

    def test_non_kimi_k3_model_is_refused(self):
        with self.assertRaisesRegex(ValueError, "serves Kimi-K3"):
            _apply(_launch(), architectures=("DeepseekV3ForCausalLM",))

    def test_max_running_requests_must_be_a_row_tile(self):
        for rows in (3, 5, 6, 7, 16):
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, "--max-running-requests in"):
                    _apply(_launch(max_running_requests=rows))
        args = _launch(max_running_requests=4)
        _apply(args)
        self.assertEqual(resolution_result(args, "tpu_compile_bs"), [1, 2, 4])

    def test_tpu_compile_bs_must_be_the_row_tiles(self):
        with self.assertRaisesRegex(ValueError, "--tpu-compile-bs 1 2 4 8"):
            _apply(_launch(tpu_compile_bs=[1, 4, 8]))
        args = _launch(tpu_compile_bs=[8, 4, 2, 1])
        _apply(args)
        self.assertNotIn("tpu_compile_bs", _declared_by_megakernel(args))


class TestSlotTable(CustomTestCase):
    def test_release_moves_the_last_row_into_the_hole(self):
        table = SlotTable(max_rows=4)
        for key in ((1, 1), (2, 1), (3, 1)):
            table.assign(key)
        self.assertEqual(table.release((1, 1)), RowMove(source=2, destination=0))
        self.assertEqual(table.rows_for([(3, 1), (2, 1)]), [0, 1])
        self.assertIsNone(table.release((2, 1)))
        self.assertEqual(table.assign((4, 1)), 1)

    def test_rows_are_bounded_and_tiles_only(self):
        table = SlotTable(max_rows=1)
        table.assign((1, 1))
        with self.assertRaisesRegex(RuntimeError, "rows are in use"):
            table.assign((2, 1))
        with self.assertRaisesRegex(ValueError, "max_rows"):
            SlotTable(max_rows=6)


class TestProgramLayout(CustomTestCase):
    def test_kimi_k3_layer_split(self):
        # Every fourth layer is MLA, plus the last of Kimi-K3's 93 layers.
        self.assertEqual(layer_counts(93), (69, 24))

    def test_mla_state_bytes_per_token(self):
        """The README's memory cost: 55,296 bytes per token per row per rank."""
        shapes = state_shapes(rows=1, context=384, layers=93)
        per_token = sum(torch.Size(shape).numel() * 2 // 384 for shape, _ in shapes[2:])
        self.assertEqual(per_token, 55_296)


class TestE8m0Scales(CustomTestCase):
    def test_inverts_the_tpu_scale_decode_for_every_code(self):
        codes = torch.arange(256, dtype=torch.uint8)
        self.assertTrue(torch.equal(e8m0_from_f32_scales(_e8m0_to_fp32(codes)), codes))

    def test_rejects_a_rounded_scale(self):
        with self.assertRaisesRegex(ValueError, "powers of two"):
            e8m0_from_f32_scales(torch.tensor([0.75], dtype=torch.float32))


# A Kimi-K3 prefix of 8 layers: 6 KDA layers, 2 MLA layers.
_LAYERS = 8
_KDA, _MLA = layer_counts(_LAYERS)
_CONTEXT = 384


def _pools(*, mla_dtype=torch.bfloat16, draft_tokens=None):
    params = KimiLinearCacheParams(
        shape=KimiLinearStateShape.create(
            tp_world_size=32, num_heads=96, head_dim=128, conv_kernel_size=4
        ),
        layers=list(range(_KDA)),
    )
    mamba = TpuMambaPool(
        size=4,
        spec_state_size=4,
        cache_params=params,
        mamba_layer_ids=list(range(_KDA)),
        device="cpu",
        speculative_num_draft_tokens=draft_tokens,
    )
    kv = TpuFusedMLATokenToKVPool(
        size=512,
        page_size=16,
        dtype=mla_dtype,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=_MLA,
        device="cpu",
        enable_memory_saver=False,
    )
    return mamba, kv


def _sources(mamba, kv, req_to_token, mamba_slot_of_req):
    return HandoffSources.from_pools(
        req_to_token=req_to_token,
        mamba_slot_of_req=mamba_slot_of_req,
        mamba_pool=mamba,
        kv_pool=kv,
        kda_layers=_KDA,
        mla_layers=_MLA,
    )


def _states(rows):
    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    return tuple(
        torch.zeros(shape, dtype=dtypes[dtype])
        for shape, dtype in state_shapes(rows, _CONTEXT, _LAYERS)
    )


class _PoolFixture(CustomTestCase):
    """Real TPU pools on the host, with request 2 cached in Mamba slot 3."""

    REQ, SLOT, CACHED = 2, 3, 200

    def setUp(self):
        torch.manual_seed(0)
        self.mamba, self.kv = _pools()
        self.req_to_token = torch.zeros((4, 600), dtype=torch.int64)
        # Scattered token locations past the padding page.
        self.locations = torch.randperm(496)[: self.CACHED] + 16
        self.req_to_token[self.REQ, : self.CACHED] = self.locations
        self.mamba_slot_of_req = torch.zeros(4, dtype=torch.int32)
        self.mamba_slot_of_req[self.REQ] = self.SLOT
        for buf in self.mamba.mamba_cache.conv[0]:
            buf.normal_()
        for buf in self.mamba.mamba_cache.temporal:
            buf.normal_()
        for layer in range(_MLA):
            self.kv.get_key_buffer(layer).normal_()
        self.m_kb = torch.randn((1, _MLA, 512, 768), dtype=torch.bfloat16) / 16
        self.sources = _sources(
            self.mamba, self.kv, self.req_to_token, self.mamba_slot_of_req
        )


class TestHandoffOnTpuPools(_PoolFixture):
    def test_copies_kda_state_and_expands_mla_cache(self):
        states = _states(rows=2)
        handoff = Handoff(states=states, sources=self.sources, m_kb=self.m_kb)
        handoff.copy_request(row=1, req_pool_idx=self.REQ, cached_len=self.CACHED)
        conv, recurrent, key, value = states

        for layer in range(_KDA):
            pool_conv = self.mamba.mamba_cache.conv[0][layer][self.SLOT, :, 0]
            for qkv in range(3):
                for head in range(3):
                    channels = slice(
                        qkv * 384 + head * 128, qkv * 384 + head * 128 + 128
                    )
                    self.assertTrue(
                        torch.equal(
                            conv[1, layer, :, qkv, head],
                            pool_conv[:, channels].to(torch.bfloat16),
                        )
                    )
            self.assertTrue(
                torch.equal(
                    recurrent[1, layer],
                    self.mamba.mamba_cache.temporal[layer][self.SLOT],
                )
            )

        for layer in range(_MLA):
            latent = self.kv.get_key_buffer(layer)[self.locations, 0].float()
            w_kb = self.m_kb[0, layer].float()
            for head in range(3):
                k_nope = latent[:, :512] @ w_kb[:, 256 * head : 256 * head + 128]
                v = latent[:, :512] @ w_kb[:, 256 * head + 128 : 256 * head + 256]
                expected_key = torch.cat(
                    (k_nope, latent[:, 512:576], torch.zeros(self.CACHED, 64)), dim=1
                )
                torch.testing.assert_close(
                    key[1, layer, head, : self.CACHED].float(),
                    expected_key.to(torch.bfloat16).float(),
                    atol=2e-2,
                    rtol=2e-2,
                )
                torch.testing.assert_close(
                    value[1, layer, head, : self.CACHED].float(),
                    v.to(torch.bfloat16).float(),
                    atol=2e-2,
                    rtol=2e-2,
                )
        # Past the cached tokens, and in every other row, the caches are zero.
        self.assertEqual(int(key[1, :, :, self.CACHED :].count_nonzero()), 0)
        self.assertEqual(int(value[1, :, :, self.CACHED :].count_nonzero()), 0)
        for state in states:
            self.assertEqual(int(state[0].count_nonzero()), 0)

    def test_move_row_copies_every_state(self):
        states = _states(rows=2)
        handoff = Handoff(states=states, sources=self.sources, m_kb=self.m_kb)
        handoff.copy_request(row=1, req_pool_idx=self.REQ, cached_len=self.CACHED)
        handoff.move_row(RowMove(source=1, destination=0))
        for state in states:
            self.assertTrue(torch.equal(state[0], state[1]))

    def test_refuses_pools_the_handoff_cannot_read(self):
        """Speculative checkpoint rows (#1460) and an fp8 latent pool would be
        read as if they were one bf16 row per slot."""
        mamba, _ = _pools(draft_tokens=2)
        with self.assertRaisesRegex(ValueError, "checkpoint_width"):
            _sources(mamba, self.kv, self.req_to_token, self.mamba_slot_of_req)
        _, kv = _pools(mla_dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "bf16"):
            _sources(self.mamba, kv, self.req_to_token, self.mamba_slot_of_req)


class _RecordingOp:
    def __init__(self, vocab):
        self.vocab = vocab
        self.calls = []

    def __call__(self, *operands):
        tokens, positions = operands[-2], operands[-1]
        self.calls.append((tokens.clone(), positions.clone()))
        # Row r's logits are its token id, so the output order is checkable.
        return tokens.float()[:, None].expand(-1, self.vocab).clone()


class TestRunner(_PoolFixture):
    def _runner(self, rows=4):
        ops = {tile: _RecordingOp(vocab=8) for tile in (1, 2, 4) if tile <= rows}
        runner = KimiK3MegakernelRunner(
            weights=[],
            states=_states(rows),
            m_kb=self.m_kb,
            decode_ops=ops,
            rank=torch.zeros(1, dtype=torch.int32),
            layers=_LAYERS,
        )
        runner._req_generation = torch.ones(4, dtype=torch.int64)
        runner._handoff = Handoff(
            states=runner.states, sources=self.sources, m_kb=self.m_kb
        )
        return runner, ops

    def test_three_requests_run_in_the_four_row_tile(self):
        """B2: a batch of 3 is padded to the 4-row tile; staging and the op must
        both use the tile, never the raw batch size (which has no op)."""
        runner, ops = self._runner()
        step = runner.prepare_decode(
            req_pool_indices=[1, 2, 3, 0], seq_lens=[10, 20, 30], padded_batch_size=4
        )
        self.assertEqual(step.positions.tolist(), [9, 19, 29, -1])
        logits = runner.forward_decode(torch.tensor([11, 12, 13, 0]), step)
        self.assertEqual(len(ops[4].calls), 1)
        self.assertEqual(logits[:3, 0].tolist(), [11.0, 12.0, 13.0])
        with self.assertRaisesRegex(RuntimeError, "--tpu-compile-bs"):
            runner.prepare_decode(
                req_pool_indices=[1, 2, 3], seq_lens=[11, 21, 31], padded_batch_size=3
            )

    def test_rows_follow_requests_across_finish_and_slot_reuse(self):
        runner, ops = self._runner()
        with patch.object(Handoff, "copy_request", autospec=True) as copy:
            runner.prepare_decode(
                req_pool_indices=[1, 2, 3, 0], seq_lens=[5, 6, 7], padded_batch_size=4
            )
            self.assertEqual(copy.call_count, 3)
            # Request 1 finished: request 3 moves from row 2 into row 0.
            step = runner.prepare_decode(
                req_pool_indices=[2, 3], seq_lens=[7, 8], padded_batch_size=2
            )
            self.assertEqual(copy.call_count, 3)
            self.assertEqual(step.input_to_row.tolist(), [1, 0])
            self.assertEqual(step.positions.tolist(), [7, 6])
            logits = runner.forward_decode(torch.tensor([22, 33]), step)
            self.assertEqual(logits[:, 0].tolist(), [22.0, 33.0])
            # Slot 2 was retracted and re-admitted: a new generation, a new handoff.
            runner._req_generation[2] = 2
            runner.prepare_decode(
                req_pool_indices=[2, 3], seq_lens=[9, 9], padded_batch_size=2
            )
            self.assertEqual(copy.call_count, 4)
            self.assertEqual(copy.call_args.kwargs["cached_len"], 8)

    def test_warmup_batches_take_no_rows(self):
        runner, _ = self._runner()
        step = runner.prepare_decode(
            req_pool_indices=[0, 0, 0, 0], seq_lens=[1, 1, 1, 1], padded_batch_size=4
        )
        self.assertEqual(len(runner.slot_table), 0)
        self.assertEqual(step.positions.tolist(), [-1, -1, -1, -1])

    def test_capacity_and_context_are_enforced(self):
        runner, _ = self._runner(rows=2)
        with self.assertRaisesRegex(RuntimeError, "holds 2 rows"):
            runner.prepare_decode(
                req_pool_indices=[1, 2, 3, 0], seq_lens=[5, 5, 5], padded_batch_size=4
            )
        with self.assertRaisesRegex(RuntimeError, "--context-length"):
            runner.prepare_decode(
                req_pool_indices=[1], seq_lens=[_CONTEXT + 1], padded_batch_size=1
            )

    def test_a_decoding_request_in_another_batch_raises(self):
        runner, _ = self._runner()
        runner.prepare_decode(req_pool_indices=[1], seq_lens=[5], padded_batch_size=1)
        runner.check_not_decoding(req_pool_indices=[2, 0])
        with self.assertRaisesRegex(RuntimeError, "non-decode batch"):
            runner.check_not_decoding(req_pool_indices=[1, 2])


class TestDecodeDispatch(CustomTestCase):
    def test_decode_runs_the_megakernel_and_other_modes_the_model(self):
        runner = SimpleNamespace(forward_decode=lambda ids, step: ids.float()[:, None])
        model_forward = unittest.mock.Mock(return_value="model output")
        forward = megakernel_forward(model_forward, runner)
        ids = torch.tensor([3, 4])
        decode = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        setattr(decode, STEP_INPUTS_ATTR, None)
        out = forward(ids, None, decode)
        self.assertIsInstance(out, LogitsProcessorOutput)
        self.assertEqual(out.next_token_logits[:, 0].tolist(), [3.0, 4.0])
        model_forward.assert_not_called()
        extend = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        self.assertEqual(forward(ids, None, extend), "model output")


class TestPolicyStep(CustomTestCase):
    """The policy hands the padded batch to the runner, mode by mode."""

    def _prepare(self, mode, runner):
        policy_cls = megakernel_policy(DecoderTpuCompilePolicy, runner)
        policy = policy_cls.__new__(policy_cls)
        batch = SimpleNamespace(
            forward_mode=mode,
            req_pool_indices_cpu=torch.tensor([5, 6, 0, 0]),
            seq_lens_cpu=torch.tensor([9, 3]),
            batch_size=4,
        )
        # The base policy pads the batch; here it already is.
        with patch.object(DecoderTpuCompilePolicy, "prepare_forward_batch"):
            policy.prepare_forward_batch(batch)
        return batch

    def test_decode_stages_the_padded_batch(self):
        runner = unittest.mock.Mock()
        batch = self._prepare(ForwardMode.DECODE, runner)
        runner.prepare_decode.assert_called_once_with(
            req_pool_indices=[5, 6, 0, 0], seq_lens=[9, 3], padded_batch_size=4
        )
        self.assertIs(
            getattr(batch, STEP_INPUTS_ATTR), runner.prepare_decode.return_value
        )

    def test_other_modes_must_not_carry_decoding_requests(self):
        runner = unittest.mock.Mock()
        self._prepare(ForwardMode.EXTEND, runner)
        runner.check_not_decoding.assert_called_once_with(req_pool_indices=[5, 6, 0, 0])
        runner.prepare_decode.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "MIXED"):
            self._prepare(ForwardMode.MIXED, runner)


class TestAttachHook(CustomTestCase):
    def _run(self, enabled):
        model_runner = SimpleNamespace(
            model="model",
            model_config=SimpleNamespace(model_path="/weights/k3", context_len=9344),
        )
        _set_enabled(self, enabled)
        attach = unittest.mock.Mock()
        # A stand-in module: the real one imports jax, which CPU CI lacks.
        with patch.dict(
            sys.modules,
            {"sglang_tpu.kimi_k3_megakernel.attach": SimpleNamespace(attach=attach)},
        ):
            attach_kimi_k3_megakernel(None, model_runner)
        return attach

    def test_attaches_only_when_enabled(self):
        self._run(False).assert_not_called()
        self._run(True).assert_called_once_with(
            "model", model_path="/weights/k3", context_length=9344
        )


def _moe_layer(seed):
    """A processed MXFP4 FusedMoE layer as the TPU loader leaves it."""
    generator = torch.Generator().manual_seed(seed)
    moe = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(moe)
    moe.w13_weight = Parameter(
        torch.randint(0, 255, (2, 16, 8), dtype=torch.uint8, generator=generator),
        requires_grad=False,
    )
    moe.w2_weight = Parameter(
        torch.randint(0, 255, (2, 4, 32), dtype=torch.uint8, generator=generator),
        requires_grad=False,
    )
    codes = torch.randint(
        100, 140, (2, 1, 1, 8), dtype=torch.uint8, generator=generator
    )
    moe.w13_weight_scale = Parameter(_e8m0_to_fp32(codes), requires_grad=False)
    moe.w2_weight_scale = Parameter(
        _e8m0_to_fp32(codes[..., :1].expand(2, 1, 1, 32).contiguous()),
        requires_grad=False,
    )
    moe.w13_weight_bias = None
    moe.w2_weight_bias = None
    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method.moe_runner_config = MoeRunnerConfig(activation="situ")
    moe.quant_method = method
    return moe


class TestExpertStacking(CustomTestCase):
    def test_layers_share_one_stack_and_read_their_own_layer(self):
        layers = [_moe_layer(seed) for seed in range(3)]
        originals = [
            (moe.w13_weight.data.clone(), moe.w2_weight.data.clone(), moe.w13_weight)
            for moe in layers
        ]
        experts = stack_routed_experts(layers, torch.device("cpu"))
        self.assertEqual(tuple(experts["expert_gu"].shape), (1, 3, 2, 16, 8))
        self.assertEqual(tuple(experts["expert_gus"].shape), (1, 3, 2, 1, 8))
        for index, (moe, (w13, w2, old_param)) in enumerate(zip(layers, originals)):
            self.assertTrue(torch.equal(experts["expert_gu"][0, index], w13))
            self.assertTrue(torch.equal(experts["expert_down"][0, index], w2))
            self.assertTrue(
                torch.equal(
                    experts["expert_gus"][0, index],
                    e8m0_from_f32_scales(moe.w13_weight_scale).squeeze(-2),
                )
            )
            # The per-layer copy is gone; the layer reads the shared stack.
            self.assertEqual(old_param.numel(), 0)
            self.assertIs(moe.w13_weight, layers[0].w13_weight)
            self.assertIsInstance(moe.quant_method, StackedMxfp4MoEMethod)
            self.assertEqual(moe.quant_method.layer_index.tolist(), [index])

    def test_refuses_experts_it_cannot_share(self):
        moe = _moe_layer(0)
        moe.quant_method = SimpleNamespace()
        with self.assertRaisesRegex(TypeError, "MXFP4"):
            stack_routed_experts([moe], torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
