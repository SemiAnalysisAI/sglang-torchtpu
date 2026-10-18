"""Muse Spark on TPU matches an fp32 torch reference on a small NVFP4 checkpoint.

The checkpoint keeps the released expert shapes (4096 x 4096, block-16 NVFP4
with per-expert global scales) and, like the released export, BF16 experts in
the first layer. Prompt logprobs and greedy decode are checked against a
reference computed on the CPU from the same weights. Routing is pinned by the
expert bias: with random weights a near-tie top-k choice flips under rounding
and the difference spreads to later tokens through attention.
"""

import json
import math
import os
import tempfile
import unittest

import requests
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_tpu_ci(
    est_time=600,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "muse-spark-open", "accuracy", "logprob", "compile"],
)

TEXT_CONFIG = {
    "hidden_size": 2048,
    "head_dim": 64,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 3,
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
    # Full-attention layers carry no RoPE, as in the released config.
    "layer_rope_theta": [500000.0, 500000.0, 0],
    "mlp_layer_types": ["sparse"] * 3,
    "sliding_window": 128,
    "moe_hidden_size": 4096,
    "moe_intermediate_size": 4096,
    "intermediate_size": 4096,
    "num_local_experts": 16,
    "num_experts_per_tok": 8,
    "vocab_size": 4096,
    "max_position_embeddings": 4096,
    # Smaller than the released 5.47 so bf16 rounding is not amplified through
    # near-tie attention into differences larger than a real bug.
    "qk_scale_factor": 2.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
}
BF16_EXPERT_LAYERS = (0,)
EP_SIZE = 8
# MuseSparkTextConfig defaults the tiny config inherits.
OUTPUT_MULTIPLIER = 0.17677669529663687
SOFTCAP = 20.0
GROUP_SIZE = 16
E2M1_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Midpoints between neighbouring magnitudes: round to nearest.
E2M1_BOUNDARIES = (E2M1_MAGNITUDES[1:] + E2M1_MAGNITUDES[:-1]) / 2
E2M1_VALUES = torch.cat([E2M1_MAGNITUDES, -E2M1_MAGNITUDES])

SERVER_ARGS = [
    "--language-model-only",
    "--skip-tokenizer-init",
    "--tp-size",
    str(EP_SIZE),
    "--ep-size",
    str(EP_SIZE),
    "--attention-backend",
    "pallas_tpu_rpa_v3",
    "--kv-cache-dtype",
    "bfloat16",
    "--page-size",
    "64",
    "--context-length",
    "1024",
    "--chunked-prefill-size",
    "256",
    "--max-prefill-tokens",
    "256",
    "--max-running-requests",
    "4",
    "--mem-fraction-static",
    "0.5",
    "--disable-radix-cache",
    "--grammar-backend",
    "none",
    # The TPU distributed top-k sampler can halt at TP8; greedy needs no top-k.
    "--sampling-backend",
    "pytorch",
    "--enable-torch-compile",
    "--torch-compile-max-bs",
    "4",
    "--tpu-compile-tokens",
    "256",
    "--tpu-compile-bs",
    "1",
    "4",
]
# A cold start compiles the 4096 x 4096 expert graphs for about ten minutes.
SERVER_LAUNCH_TIMEOUT = 1800
# The second prompt spans several chunks and slides past the 128-token window.
PROMPT_LENGTHS = (37, 600)
MAX_NEW_TOKENS = 16


def _quantize_nvfp4(weight: torch.Tensor):
    """NVFP4 rows ``[N, K]``: packed E2M1 codes, e4m3 block scales, fp32 global scale.

    Also returns the weight as the TPU path dequantizes it (block scales folded
    with the global scale in bf16), in bf16, for the reference.
    """
    rows, cols = weight.shape
    global_scale = weight.abs().max() / (448.0 * 6.0)
    blocks = weight.reshape(rows, cols // GROUP_SIZE, GROUP_SIZE)
    block_scale = (blocks.abs().amax(-1) / 6.0 / global_scale).to(torch.float8_e4m3fn)
    folded = (block_scale.float() * global_scale).to(torch.bfloat16).float()
    scaled = blocks / folded.clamp(min=1e-30).unsqueeze(-1)
    magnitude = torch.bucketize(scaled.abs(), E2M1_BOUNDARIES)
    codes = (magnitude + 8 * (scaled < 0)).to(torch.uint8).reshape(rows, cols)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    dequantized = E2M1_VALUES[codes.long()].reshape(blocks.shape) * folded[..., None]
    return packed, block_scale, global_scale, dequantized.reshape(rows, cols).bfloat16()


def _randn(generator, *shape, std):
    return torch.randn(*shape, generator=generator) * std


def _make_expert_tensors(generator, prefix: str, quantized: bool):
    """Stacked expert tensors under checkpoint names, plus fp32 reference weights."""
    num_experts = TEXT_CONFIG["num_local_experts"]
    hidden = TEXT_CONFIG["moe_hidden_size"]
    inter = TEXT_CONFIG["moe_intermediate_size"]
    gate_up = _randn(generator, num_experts, 2 * inter, hidden, std=hidden**-0.5)
    down = _randn(generator, num_experts, hidden, inter, std=inter**-0.5)
    if not quantized:
        gate_up, down = gate_up.bfloat16(), down.bfloat16()
        tensors = {f"{prefix}.gate_up_proj": gate_up, f"{prefix}.down_proj": down}
        return tensors, (gate_up, down)

    names = ("", "_weight_scale", "_weight_scale_2")
    w13 = {n: [] for n in names}
    w2 = {n: [] for n in names}
    ref_gate_up, ref_down = [], []
    for e in range(num_experts):
        halves = [_quantize_nvfp4(half) for half in gate_up[e].chunk(2)]
        w13[""].append(torch.cat([h[0] for h in halves]))
        w13["_weight_scale"].append(torch.cat([h[1] for h in halves]))
        w13["_weight_scale_2"].append(torch.stack([h[2] for h in halves]))
        ref_gate_up.append(torch.cat([h[3] for h in halves]))
        packed, block_scale, global_scale, ref = _quantize_nvfp4(down[e])
        w2[""].append(packed)
        w2["_weight_scale"].append(block_scale)
        w2["_weight_scale_2"].append(global_scale)
        ref_down.append(ref)
    tensors = {}
    for n in names:
        tensors[f"{prefix}.gate_up_proj{n}"] = torch.stack(w13[n])
        tensors[f"{prefix}.down_proj{n}"] = torch.stack(w2[n])
    tensors[f"{prefix}.gate_up_proj_input_scale"] = torch.ones(num_experts, 2)
    tensors[f"{prefix}.down_proj_input_scale"] = torch.ones(num_experts)
    return tensors, (torch.stack(ref_gate_up), torch.stack(ref_down))


def _routing_bias(*, layer: int) -> torch.Tensor:
    """Bias that routes every token to the same top-k experts, one per EP rank.

    Sigmoid scores lie in (0, 1), so a bias of 2 decides the selection outright and
    rounding cannot flip it; the routing weights still come from the raw scores.
    Each layer uses a different expert on every rank.
    """
    num_experts = TEXT_CONFIG["num_local_experts"]
    per_rank = num_experts // EP_SIZE
    bias = torch.zeros(num_experts)
    bias[torch.arange(EP_SIZE) * per_rank + layer % per_rank] = 2.0
    return bias


def _make_checkpoint(model_dir: str) -> dict:
    """Write config.json and model.safetensors; return fp32 reference weights."""
    g = torch.Generator().manual_seed(0)
    c = TEXT_CONFIG
    hidden, moe_hidden = c["hidden_size"], c["moe_hidden_size"]
    q_width = c["num_attention_heads"] * c["head_dim"]
    kv_width = c["num_key_value_heads"] * c["head_dim"]
    tensors, ref = {}, {}

    def put(name, value, dtype=torch.bfloat16):
        value = value.to(dtype)
        tensors[name] = value
        ref[name] = value.float()

    lm = "model.language_model"
    put(f"{lm}.embed_tokens.weight", _randn(g, c["vocab_size"], hidden, std=1.0))
    put(f"{lm}.norm.weight", 1.0 + _randn(g, hidden, std=0.1))
    put("lm_head.weight", _randn(g, c["vocab_size"], hidden, std=0.25))
    for i in range(c["num_hidden_layers"]):
        p = f"{lm}.layers.{i}"
        for name, rows in (
            ("q", q_width),
            ("k", kv_width),
            ("v", kv_width),
            ("gate", q_width),
        ):
            put(
                f"{p}.self_attn.{name}_proj.weight",
                _randn(g, rows, hidden, std=hidden**-0.5),
            )
        put(
            f"{p}.self_attn.o_proj.weight",
            _randn(g, hidden, q_width, std=q_width**-0.5),
        )
        for name in (
            "input_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            put(f"{p}.{name}.weight", _randn(g, hidden, std=0.1))
        for name in ("post_attention_residual_gate", "post_feedforward_residual_gate"):
            put(f"{p}.{name}.gate", _randn(g, hidden, std=1.0))
        put(
            f"{p}.mlp.gate.weight",
            _randn(g, c["num_local_experts"], hidden, std=8 * hidden**-0.5),
            torch.float32,
        )
        put(
            f"{p}.mlp.gate.e_score_correction_bias",
            _routing_bias(layer=i),
            torch.float32,
        )
        put(
            f"{p}.mlp.pre_expert_proj.weight",
            _randn(g, moe_hidden, hidden, std=hidden**-0.5),
        )
        put(
            f"{p}.mlp.post_expert_proj.weight",
            _randn(g, hidden, moe_hidden, std=moe_hidden**-0.5),
        )
        put(f"{p}.mlp.pre_expert_norm.weight", _randn(g, moe_hidden, std=0.1))
        put(f"{p}.mlp.experts.post_expert_norm.weight", _randn(g, moe_hidden, std=0.1))
        experts, (gate_up, down) = _make_expert_tensors(
            g, f"{p}.mlp.experts", quantized=i not in BF16_EXPERT_LAYERS
        )
        tensors.update(experts)
        ref[f"{p}.experts.gate_up"], ref[f"{p}.experts.down"] = gate_up, down
    save_file(tensors, os.path.join(model_dir, "model.safetensors"))

    excluded = ["lm_head"] + [
        f"{lm}.layers.{i}.mlp.experts" for i in BF16_EXPERT_LAYERS
    ]
    for i in range(c["num_hidden_layers"]):
        excluded += [
            f"{lm}.layers.{i}.self_attn.{n}_proj" for n in ("q", "k", "v", "o", "gate")
        ]
        excluded += [f"{lm}.layers.{i}.mlp.{n}_expert_proj" for n in ("pre", "post")]
    config = {
        "architectures": ["MuseSparkForConditionalGeneration"],
        "model_type": "muse_spark",
        "dtype": "bfloat16",
        "text_config": dict(TEXT_CONFIG, model_type="muse_spark_text"),
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "group_size": GROUP_SIZE,
            "modules_to_not_convert": excluded,
        },
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)
    return ref


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.bfloat16().float()


def _rms(x: torch.Tensor, eps: float, gain=None) -> torch.Tensor:
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return out if gain is None else out * gain


def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    dim = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    freqs = positions[:, None].double() * inv_freq
    cos, sin = freqs.cos().float()[:, None], freqs.sin().float()[:, None]
    x1, x2 = x[..., : dim // 2], x[..., dim // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _attention(ref, p, i, x, positions):
    c = TEXT_CONFIG
    d, n_q, n_kv = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
    eps = 1e-5
    q = _bf16(x @ ref[f"{p}.self_attn.q_proj.weight"].T).unflatten(-1, (n_q, d))
    k = _bf16(x @ ref[f"{p}.self_attn.k_proj.weight"].T).unflatten(-1, (n_kv, d))
    v = _bf16(x @ ref[f"{p}.self_attn.v_proj.weight"].T).unflatten(-1, (n_kv, d))
    q, k = _bf16(_rms(q, eps)), _bf16(_rms(k, eps))
    if c["layer_rope_theta"][i] > 0:
        q = _bf16(_rope(q, positions, c["layer_rope_theta"][i]))
        k = _bf16(_rope(k, positions, c["layer_rope_theta"][i]))
    k = k.repeat_interleave(n_q // n_kv, dim=1)
    v = v.repeat_interleave(n_q // n_kv, dim=1)
    scale = TEXT_CONFIG["qk_scale_factor"] / math.sqrt(d)
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    t = x.shape[0]
    rel = torch.arange(t)[:, None] - torch.arange(t)[None, :]
    allowed = rel >= 0
    if c["layer_types"][i] == "sliding_attention":
        allowed &= rel < c["sliding_window"]
    scores = scores.masked_fill(~allowed, float("-inf"))
    out = _bf16(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v))
    out = _bf16(_rms(out, eps)).flatten(-2)
    gate = _bf16(x @ ref[f"{p}.self_attn.gate_proj.weight"].T)
    out = _bf16(out * torch.sigmoid(gate))
    return _bf16(out @ ref[f"{p}.self_attn.o_proj.weight"].T)


def _input_scaled_norm(x, weight, eps=1e-8):
    return _bf16(_rms(_bf16(x * (1.0 + weight)), eps))


def _moe(ref, p, x):
    c = TEXT_CONFIG
    logits = x @ ref[f"{p}.mlp.gate.weight"].T
    scores = torch.sigmoid(logits * OUTPUT_MULTIPLIER)
    bias = ref[f"{p}.mlp.gate.e_score_correction_bias"]
    _, ids = torch.topk(scores + bias, c["num_experts_per_tok"])
    weights = scores.gather(1, ids)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-15)
    h = _bf16(x @ ref[f"{p}.mlp.pre_expert_proj.weight"].T)
    h = _bf16(_rms(h, 1e-5, 1.0 + ref[f"{p}.mlp.pre_expert_norm.weight"]))
    norm_weight = ref[f"{p}.mlp.experts.post_expert_norm.weight"]
    out = torch.zeros_like(h)
    for e in range(c["num_local_experts"]):
        tokens, slots = (ids == e).nonzero(as_tuple=True)
        if tokens.numel() == 0:
            continue
        gate_up = h[tokens] @ ref[f"{p}.experts.gate_up"][e].float().T
        gate, up = gate_up.chunk(2, dim=-1)
        y = _bf16(F.silu(gate) * up) @ ref[f"{p}.experts.down"][e].float().T
        y = _input_scaled_norm(y, norm_weight) * weights[tokens, slots, None]
        out.index_add_(0, tokens, y)
    return _bf16(_bf16(out) @ ref[f"{p}.mlp.post_expert_proj.weight"].T)


def _residual_gate(stream, branch, gate):
    w = gate / 0.3
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return alpha * stream + beta * branch


def _reference_logprobs(ref, ids: list[int]) -> torch.Tensor:
    """Next-token log-probabilities ``[len(ids), vocab]`` of the reference model."""
    c, lm = TEXT_CONFIG, "model.language_model"
    positions = torch.arange(len(ids))
    stream = _bf16(_rms(ref[f"{lm}.embed_tokens.weight"][ids], 1e-5))
    for i in range(c["num_hidden_layers"]):
        p = f"{lm}.layers.{i}"
        x = _bf16(_rms(stream, 1e-5, 1.0 + ref[f"{p}.input_layernorm.weight"]))
        attn = _bf16(_rms(_attention(ref, p, i, x, positions), 1e-8))
        stream = _residual_gate(
            stream, attn, ref[f"{p}.post_attention_residual_gate.gate"]
        )
        x = _bf16(
            _rms(stream, 1e-5, 1.0 + ref[f"{p}.pre_feedforward_layernorm.weight"])
        )
        ffn = _input_scaled_norm(
            _moe(ref, p, x), ref[f"{p}.post_feedforward_layernorm.weight"]
        )
        stream = _residual_gate(
            stream, ffn, ref[f"{p}.post_feedforward_residual_gate.gate"]
        )
    h = _bf16(_rms(_bf16(stream), 1e-5, ref[f"{lm}.norm.weight"]))
    # As served: the scale applies to the bf16 logits, the cap in fp32.
    logits = _bf16(_bf16(h @ ref["lm_head.weight"].T) * OUTPUT_MULTIPLIER)
    logits = SOFTCAP * torch.tanh(logits / SOFTCAP)
    return logits.log_softmax(-1)


def _prompt(length: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, TEXT_CONFIG["vocab_size"], (length,), generator=g).tolist()


def _generate(prompts: list[list[int]]) -> list[dict]:
    response = requests.post(
        f"{DEFAULT_URL_FOR_TEST}/generate",
        json={
            "input_ids": prompts,
            "sampling_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
        },
        timeout=900,
    )
    response.raise_for_status()
    return response.json()


class TestMuseSparkTinyNvfp4(CustomTestCase):
    def test_matches_fp32_reference(self):
        model_dir = tempfile.TemporaryDirectory(prefix="muse_spark_tiny_")
        self.addCleanup(model_dir.cleanup)
        ref = _make_checkpoint(model_dir.name)
        prompts = [_prompt(n, seed) for seed, n in enumerate(PROMPT_LENGTHS)]

        # setdefault: mode B runs these suites with SGLANG_PLATFORM=tpu_oot.
        os.environ.setdefault("SGLANG_PLATFORM", "tpu")
        # Prompt logprobs are opt-in on TPU. The SparseCore MoE gather halts on the
        # per-slot path of BF16 experts. The server inherits both overrides.
        with (
            envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB.override(True),
            envs.SGLANG_TPU_FUSED_MOE_SC_GATHER.override(False),
        ):
            process = popen_launch_server(
                model_dir.name,
                DEFAULT_URL_FOR_TEST,
                timeout=SERVER_LAUNCH_TIMEOUT,
                other_args=SERVER_ARGS,
                device="tpu",
            )
        try:
            outputs = _generate(prompts)
        finally:
            kill_process_tree(process.pid, wait_timeout=60)

        for prompt, output in zip(prompts, outputs):
            meta = output["meta_info"]
            served = torch.tensor([lp for lp, _, _ in meta["input_token_logprobs"][1:]])
            generated = output["output_ids"]
            reference = _reference_logprobs(ref, prompt + generated)
            n = len(prompt)
            # Logprob of each prompt token given its prefix.
            expected = reference[: n - 1].gather(1, torch.tensor(prompt[1:])[:, None])[
                :, 0
            ]
            diff = (served - expected).abs()
            print(
                f"prompt {n}: median |dlogprob| {diff.median():.4f}, "
                f"p95 {diff.quantile(0.95):.4f}, max {diff.max():.4f}"
            )
            # The reference itself moves by a median of about 0.04 when its
            # inputs get 0.1% relative noise; a wrong norm, scale or layout
            # moves it by 0.3 or more.
            self.assertLess(diff.median().item(), 0.08)
            self.assertLess((diff > 0.5).float().mean().item(), 0.01)
            # Each greedy token is the reference's choice or within 0.5 nats of it.
            steps = reference[n - 1 : n - 1 + len(generated)]
            chosen = steps.gather(1, torch.tensor(generated)[:, None])[:, 0]
            gap = steps.max(-1).values - chosen
            print(f"prompt {n}: decode gaps {[round(x, 3) for x in gap.tolist()]}")
            self.assertLess(gap.max().item(), 0.5)


if __name__ == "__main__":
    unittest.main()
