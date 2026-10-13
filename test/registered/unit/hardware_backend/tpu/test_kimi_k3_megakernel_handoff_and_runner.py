# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Kimi-K3 megakernel weight stacking, state handoff, and runner."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kimi_k3_megakernel.handoff import (
    apply_row_move,
    build_padded_token_locs_and_mask,
    handoff_request_to_megakernel_row,
)
from sglang_tpu.kimi_k3_megakernel.runner import (
    KimiK3MegakernelRunner,
    allocate_decode_states,
)
from sglang_tpu.kimi_k3_megakernel.slots import RowMove
from sglang_tpu.kimi_k3_megakernel.weights import (
    e8m0_from_f32_scales,
    stack_sglang_moe_experts_in_place,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_fake_pools(
    *,
    num_reqs: int = 8,
    max_ctx: int = 384,
    kda_layers: int = 3,
    mla_layers: int = 1,
    num_tokens: int = 64,
):
    req_to_token = torch.zeros((num_reqs, max_ctx), dtype=torch.int32)
    req_generation = torch.ones((num_reqs,), dtype=torch.int32)
    req_index_to_mamba = torch.arange(1, num_reqs + 1, dtype=torch.int32)

    conv_buffers = [
        torch.randn((num_reqs + 2, 3, 1, 1152), dtype=torch.float32)
        for _ in range(kda_layers)
    ]
    recurrent_buffers = [
        torch.randn((num_reqs + 2, 3, 128, 128), dtype=torch.float32)
        for _ in range(kda_layers)
    ]
    # Simulate TpuFusedMLATokenToKVPool 640-wide padded buffer (512 c_kv + 64 k_pe + 64 pad)
    kv_buffer = [
        torch.randn((num_tokens // 4, 4, 1, 640), dtype=torch.bfloat16)
        for _ in range(mla_layers)
    ]

    mamba_cache = SimpleNamespace(conv=[conv_buffers], temporal=recurrent_buffers)
    mamba_pool = SimpleNamespace(mamba_cache=mamba_cache)
    req_to_token_pool = SimpleNamespace(
        req_to_token=req_to_token,
        req_generation=req_generation,
        req_index_to_mamba_index_mapping=req_index_to_mamba,
        mamba_pool=mamba_pool,
    )
    token_to_kv_pool = SimpleNamespace(kv_buffer=kv_buffer)
    return req_to_token_pool, token_to_kv_pool


class TestStackSglangMoeExpertsInPlace(CustomTestCase):
    def test_stacks_into_shared_5d_base_tensors(self):
        layers = []
        for i in range(3):
            w13 = torch.full((4, 16, 8), i + 1, dtype=torch.uint8)
            w2 = torch.full((4, 8, 16), (i + 1) * 3, dtype=torch.uint8)
            w13_s = torch.full((4, 2, 1, 8), 2.0 ** (i - 2), dtype=torch.float32)
            w2_s = torch.full((4, 1, 1, 16), 2.0 ** (i - 1), dtype=torch.float32)
            layers.append(
                SimpleNamespace(
                    w13_weight=torch.nn.Parameter(w13, requires_grad=False),
                    w2_weight=torch.nn.Parameter(w2, requires_grad=False),
                    w13_weight_scale=torch.nn.Parameter(w13_s, requires_grad=False),
                    w2_weight_scale=torch.nn.Parameter(w2_s, requires_grad=False),
                )
            )

        out = stack_sglang_moe_experts_in_place(layers, device="cpu")
        self.assertEqual(tuple(out["expert_gu"].shape), (1, 3, 4, 16, 8))
        self.assertEqual(tuple(out["expert_down"].shape), (1, 3, 4, 8, 16))
        self.assertEqual(tuple(out["expert_gus"].shape), (1, 3, 4, 2, 8))
        self.assertEqual(tuple(out["expert_ds"].shape), (1, 3, 4, 1, 16))
        for tensor in out.values():
            self.assertIsNone(tensor._base)

        for idx, layer in enumerate(layers):
            self.assertIs(layer.w13_weight, layers[0].w13_weight)
            self.assertIs(layer.w2_weight, layers[0].w2_weight)
            self.assertEqual(
                layer.w13_weight.data.data_ptr(), out["expert_gu"].data_ptr()
            )
            self.assertEqual(
                layer.w2_weight.data.data_ptr(), out["expert_down"].data_ptr()
            )
            self.assertEqual(int(layer._moe_layer_idx_tensor.item()), idx)
            self.assertTrue((out["expert_gu"][0, idx] == (idx + 1)).all())
            self.assertTrue((out["expert_down"][0, idx] == (idx + 1) * 3).all())
            expected_gus = e8m0_from_f32_scales(
                torch.tensor([2.0 ** (idx - 2)], dtype=torch.float32)
            ).item()
            self.assertTrue((out["expert_gus"][0, idx] == expected_gus).all())

    def test_runs_process_weights_after_loading_when_called_before_postprocess(self):
        layers = []
        calls = []

        def fake_process(layer):
            calls.append(layer.layer_id)
            layer.w13_weight.data = layer.w13_weight.data.permute(0, 2, 1).contiguous()
            layer.w2_weight.data = layer.w2_weight.data.permute(0, 2, 1).contiguous()
            layer.w13_weight_scale = torch.nn.Parameter(
                torch.full((4, 2, 1, 8), 1.0, dtype=torch.float32),
                requires_grad=False,
            )
            layer.w2_weight_scale = torch.nn.Parameter(
                torch.full((4, 1, 1, 16), 0.5, dtype=torch.float32),
                requires_grad=False,
            )

        for i in range(2):
            w13_raw = torch.full((4, 8, 16), i + 1, dtype=torch.uint8)
            w2_raw = torch.full((4, 16, 8), (i + 1) * 2, dtype=torch.uint8)
            w13_s_raw = torch.full((4, 8, 2), 127, dtype=torch.uint8)
            w2_s_raw = torch.full((4, 16, 1), 126, dtype=torch.uint8)
            layers.append(
                SimpleNamespace(
                    layer_id=i,
                    w13_weight=torch.nn.Parameter(w13_raw, requires_grad=False),
                    w2_weight=torch.nn.Parameter(w2_raw, requires_grad=False),
                    w13_weight_scale=torch.nn.Parameter(w13_s_raw, requires_grad=False),
                    w2_weight_scale=torch.nn.Parameter(w2_s_raw, requires_grad=False),
                    quant_method=SimpleNamespace(
                        process_weights_after_loading=fake_process
                    ),
                )
            )

        out = stack_sglang_moe_experts_in_place(layers, device="cpu")
        self.assertEqual(calls, [0, 1])
        self.assertEqual(tuple(out["expert_gu"].shape), (1, 2, 4, 16, 8))
        self.assertEqual(tuple(out["expert_down"].shape), (1, 2, 4, 8, 16))
        # Subsequent postprocess_weights pass must be a no-op
        for layer in layers:
            layer.quant_method.process_weights_after_loading(layer)
        self.assertEqual(calls, [0, 1])

    def test_e8m0_roundtrip_with_prepare_tpu_mxfp4_moe_weights(self):
        from sglang.srt.layers.quantization.mxfp4_tpu_util import (
            _e8m0_to_fp32,
            prepare_tpu_mxfp4_moe_weights,
        )

        all_normal = torch.arange(1, 255, dtype=torch.uint8)
        f32_scales = _e8m0_to_fp32(all_normal)
        self.assertIsNone(f32_scales._base)
        self.assertTrue(torch.equal(e8m0_from_f32_scales(f32_scales), all_normal))

        # Full Kimi-K3 aligned non-interleaved flow through prepare_tpu_mxfp4_moe_weights
        layers = []
        saved_params = []
        for i in range(2):
            moe = SimpleNamespace(
                w13_weight=torch.nn.Parameter(
                    torch.full((4, 256, 64), i + 1, dtype=torch.uint8),
                    requires_grad=False,
                ),
                w2_weight=torch.nn.Parameter(
                    torch.full((4, 128, 64), i + 2, dtype=torch.uint8),
                    requires_grad=False,
                ),
                w13_weight_scale=torch.nn.Parameter(
                    torch.full((4, 256, 4), 120 + i, dtype=torch.uint8),
                    requires_grad=False,
                ),
                w2_weight_scale=torch.nn.Parameter(
                    torch.full((4, 128, 4), 121 + i, dtype=torch.uint8),
                    requires_grad=False,
                ),
                w13_weight_bias=torch.nn.Parameter(
                    torch.zeros((4, 256), dtype=torch.bfloat16),
                    requires_grad=False,
                ),
                w2_weight_bias=torch.nn.Parameter(
                    torch.zeros((4, 128), dtype=torch.bfloat16),
                    requires_grad=False,
                ),
                moe_runner_config=SimpleNamespace(activation="silu"),
                quant_method=SimpleNamespace(
                    process_weights_after_loading=lambda l: prepare_tpu_mxfp4_moe_weights(
                        l, with_bias=False
                    )
                ),
            )
            saved_params.extend(
                [
                    moe.w13_weight,
                    moe.w2_weight,
                    moe.w13_weight_scale,
                    moe.w2_weight_scale,
                    moe.w13_weight_bias,
                    moe.w2_weight_bias,
                ]
            )
            layers.append(moe)
        out = stack_sglang_moe_experts_in_place(layers, device="cpu")
        for old_param in saved_params:
            self.assertEqual(old_param.numel(), 0)
            self.assertEqual(old_param.untyped_storage().nbytes(), 0)
        self.assertTrue((out["expert_gus"][0, 0] == 120).all())
        self.assertTrue((out["expert_gus"][0, 1] == 121).all())
        self.assertTrue((out["expert_ds"][0, 0] == 121).all())
        self.assertTrue((out["expert_ds"][0, 1] == 122).all())


class TestHandoffAndRowMove(CustomTestCase):
    def test_handoff_populates_target_row_and_masks_past_cached_len(self):
        context = 384
        layers = 4  # 3 KDA + 1 MLA
        states = allocate_decode_states(rows=4, context=context, layers=layers, device="cpu")
        req_pool, kv_pool = _make_fake_pools(
            num_reqs=4, max_ctx=context, kda_layers=3, mla_layers=1, num_tokens=64
        )
        # Seed request at req_pool_idx=2 with cached_len=5 tokens at KV slots [10, 11, 12, 13, 14]
        req_pool_idx = 2
        cached_len = 5
        req_pool.req_to_token[req_pool_idx, :cached_len] = torch.tensor(
            [10, 11, 12, 13, 14], dtype=torch.int32
        )
        m_kb = torch.randn((1, 1, 512, 768), dtype=torch.bfloat16)

        locs_dev, mask_dev = build_padded_token_locs_and_mask(
            req_pool.req_to_token,
            req_pool_idx=req_pool_idx,
            cached_len=cached_len,
            context=context,
            device="cpu",
        )
        mamba_slot = int(req_pool.req_index_to_mamba_index_mapping[req_pool_idx].item())
        handoff_request_to_megakernel_row(
            states,
            row_dev=torch.tensor([1], dtype=torch.long),
            mamba_slot_dev=torch.tensor([mamba_slot], dtype=torch.long),
            token_locs_dev=locs_dev,
            valid_mask_dev=mask_dev,
            conv_buffers=req_pool.mamba_pool.mamba_cache.conv[0],
            recurrent_buffers=req_pool.mamba_pool.mamba_cache.temporal,
            mla_kv_buffers=kv_pool.kv_buffer,
            m_kb=m_kb,
        )

        conv_state, rec_state, key_cache, value_cache = states
        # Row 0 untouched
        self.assertEqual(float(conv_state[0].abs().sum()), 0.0)
        self.assertEqual(float(key_cache[0].abs().sum()), 0.0)

        # Row 1 KDA matches mamba_slot
        for l_idx in range(3):
            expected_conv = (
                req_pool.mamba_pool.mamba_cache.conv[0][l_idx][mamba_slot]
                .reshape(3, 3, 3, 128)
                .to(torch.bfloat16)
            )
            self.assertTrue(torch.equal(conv_state[1, l_idx], expected_conv))
            expected_rec = req_pool.mamba_pool.mamba_cache.temporal[l_idx][mamba_slot]
            self.assertTrue(torch.equal(rec_state[1, l_idx], expected_rec))

        # Row 1 MLA valid positions [0, 5) populated; [5, context) zeroed; trailing 64 of key_cache zeroed
        self.assertGreater(float(key_cache[1, 0, :, :cached_len, :192].abs().sum()), 0.0)
        self.assertEqual(float(key_cache[1, 0, :, :cached_len, 192:].abs().sum()), 0.0)
        self.assertEqual(float(key_cache[1, 0, :, cached_len:, :].abs().sum()), 0.0)
        self.assertGreater(float(value_cache[1, 0, :, :cached_len, :].abs().sum()), 0.0)
        self.assertEqual(float(value_cache[1, 0, :, cached_len:, :].abs().sum()), 0.0)

        # Compaction move: row 1 -> row 0
        apply_row_move(states, RowMove(source=1, destination=0))
        for s in states:
            self.assertTrue(torch.equal(s[0], s[1]))


class TestKimiK3MegakernelRunner(CustomTestCase):
    def test_runner_lifecycle_handoff_compaction_and_permutation(self):
        context = 384
        layers = 4  # 3 KDA + 1 MLA
        vocab_size = 64
        m_kb = torch.randn((1, 1, 512, 768), dtype=torch.bfloat16)
        weights = {"m_kb": m_kb}

        recorded_calls = []

        def make_fake_decode_op(b: int):
            def op(*operands):
                tokens = operands[-2]
                positions = operands[-1]
                recorded_calls.append(
                    (b, tokens.clone().tolist(), positions.clone().tolist())
                )
                # Return deterministic logits encoding the megakernel row's token & position
                out = torch.zeros((b, vocab_size), dtype=torch.float32)
                for r in range(b):
                    out[r, 0] = float(tokens[r].item())
                    out[r, 1] = float(positions[r].item())
                return out

            return op

        decode_ops = {b: make_fake_decode_op(b) for b in (1, 2, 4, 8)}
        runner = KimiK3MegakernelRunner(
            weights=weights,
            weight_names=("m_kb",),
            decode_ops=decode_ops,
            max_rows=8,
            context=context,
            layers=layers,
            vocab_size=vocab_size,
            device="cpu",
        )

        req_pool, kv_pool = _make_fake_pools(
            num_reqs=8, max_ctx=context, kda_layers=3, mla_layers=1, num_tokens=64
        )
        for req_idx in (1, 2, 3):
            req_pool.req_to_token[req_idx, :10] = torch.arange(
                req_idx * 10, req_idx * 10 + 10, dtype=torch.int32
            )

        # 1. Warmup dummy batch (reqs=None, req_pool_indices all 0)
        warmup_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            seq_lens=torch.ones(2, dtype=torch.int64),
            seq_lens_cpu=torch.ones(2, dtype=torch.int64),
            req_pool_indices=torch.zeros(2, dtype=torch.int64),
            req_pool_indices_cpu=torch.zeros(2, dtype=torch.int64),
            reqs=None,
            req_to_token_pool=req_pool,
            token_to_kv_pool=kv_pool,
        )
        runner.prepare_batch(warmup_batch)
        self.assertTrue(runner._handoff_warmed)
        self.assertEqual(len(runner.slot_table), 0)

        # 2. Live step 1: 3 requests (rA at pool 1, rB at pool 2, rC at pool 3), padded to bucket 4
        batch1 = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            _tpu_real_b=3,
            seq_lens=torch.tensor([5, 7, 9, 1], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([5, 7, 9, 1], dtype=torch.int64),
            req_pool_indices=torch.tensor([1, 2, 3, 0], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([1, 2, 3, 0], dtype=torch.int64),
            reqs=[
                SimpleNamespace(rid="rA"),
                SimpleNamespace(rid="rB"),
                SimpleNamespace(rid="rC"),
            ],
            req_to_token_pool=req_pool,
            token_to_kv_pool=kv_pool,
        )
        runner.prepare_batch(batch1)
        self.assertEqual(runner.slot_table.rows_for(["rA", "rB", "rC"]), [0, 1, 2])
        out1 = runner.forward_decode(
            torch.tensor([101, 102, 103, 0], dtype=torch.int64), batch1
        )
        self.assertEqual(recorded_calls[-1], (4, [101, 102, 103, 0], [4, 6, 8, -1]))
        self.assertEqual(out1.next_token_logits[:3, 0].tolist(), [101.0, 102.0, 103.0])
        self.assertEqual(out1.next_token_logits[:3, 1].tolist(), [4.0, 6.0, 8.0])

        # 3. Live step 2: rA finishes! Remaining batch is [rB, rC].
        # Releasing rA moves rC (row 2) into row 0, so rB owns row 1 and rC owns row 0!
        c_conv_before = runner.states[0][2].clone()
        batch2 = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            _tpu_real_b=2,
            seq_lens=torch.tensor([8, 10], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([8, 10], dtype=torch.int64),
            req_pool_indices=torch.tensor([2, 3], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([2, 3], dtype=torch.int64),
            reqs=[
                SimpleNamespace(rid="rB"),
                SimpleNamespace(rid="rC"),
            ],
            req_to_token_pool=req_pool,
            token_to_kv_pool=kv_pool,
        )
        runner.prepare_batch(batch2)
        self.assertEqual(runner.slot_table.rows_for(["rB", "rC"]), [1, 0])
        # Verify rC's state was compacted from row 2 into row 0
        self.assertTrue(torch.equal(runner.states[0][0], c_conv_before))

        out2 = runner.forward_decode(
            torch.tensor([202, 203], dtype=torch.int64), batch2
        )
        # Megakernel row 0 has rC (token 203, pos 9); row 1 has rB (token 202, pos 7)
        self.assertEqual(recorded_calls[-1], (2, [203, 202], [9, 7]))
        # Output logits are unpermuted back to batch order [rB, rC]!
        self.assertEqual(out2.next_token_logits[:, 0].tolist(), [202.0, 203.0])
        self.assertEqual(out2.next_token_logits[:, 1].tolist(), [7.0, 9.0])

        # 4. Non-zero tp_rank shifts token IDs by tp_rank * (vocab_size // 32) on device
        runner_rank3 = KimiK3MegakernelRunner(
            weights=weights,
            weight_names=("m_kb",),
            decode_ops=decode_ops,
            max_rows=8,
            context=context,
            layers=layers,
            vocab_size=vocab_size,
            tp_rank=3,
            device="cpu",
        )
        self.assertIsNone(runner_rank3._vocab_offset_dev._base)
        self.assertEqual(runner_rank3._vocab_offset_dev.tolist(), [3 * (vocab_size // 32)])
        runner_rank3.prepare_batch(batch2)
        runner_rank3.forward_decode(
            torch.tensor([202, 203], dtype=torch.int64), batch2
        )
        # Rank 3 shifts tokens by -6 (3 * (64 // 32)); row 0 has rB (202 - 6 = 196), row 1 has rC (203 - 6 = 197)
        self.assertEqual(recorded_calls[-1], (2, [196, 197], [7, 9]))


if __name__ == "__main__":
    unittest.main()
