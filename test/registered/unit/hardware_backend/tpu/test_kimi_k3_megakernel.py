"""Kimi-K3 TPU decode megakernel: row bookkeeping, handoff and expert sharing.

CPU-only: the handoff runs against the real TPU pool classes allocated on the
host, and the decode op is a recording stand-in.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import Parameter

from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.srt.layers.quantization.mxfp4_tpu_util import _e8m0_to_fp32
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
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

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


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
