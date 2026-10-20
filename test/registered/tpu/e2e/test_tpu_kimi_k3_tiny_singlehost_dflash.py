"""DFlash speculative decoding on Kimi-K3 (TPU experimental spec worker).

The experimental worker verifies a block draft through the KDA checkpointed
state path, which it already uses for DSpark. This serves the shared tiny
Kimi-K3 checkpoint (kimi_k3_tiny_harness.py) at TP4 with a tiny random-weight
DFlash draft built here: one Qwen3-style decoder layer over target layers 1, 4
and 7. TP4 also exercises the gathered argmax the draft takes where the Pallas
distributed TopK has no kernel (it covers TP 8 and 16; Kimi-K3 serves at TP32).

Speculative decoding is lossless in exact arithmetic. Here a verify step
scores its tokens through an extend-shaped forward, not the decode program,
and this random model flips near-ties at the first difference. So the check
is teacher-forced: both servers continue the same prefixes, and
- the first token (both servers prefill it) must always agree;
- the second token, the first verify step, must agree with plain decode as
  often as the plain server's own extend agrees with its decode.

Measured on tpu7x over these 96 prefixes: plain vs plain 100%, plain extend
vs plain decode 81.2%, DFlash verify vs plain decode 81.2%.
"""

import json
import os
import tempfile
import unittest

import requests
from kimi_k3_tiny_harness import PROMPTS, build_model, launch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=900,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "kimi-k3", "spec", "experimental-spec", "compile"],
)

PORT = 21000
BASE_URL = f"http://127.0.0.1:{PORT}"
MAX_NEW_TOKENS = 32
BLOCK_SIZE = 4
TARGET_LAYER_IDS = [1, 4, 7]


def build_draft(root: str, target: str) -> None:
    """A random DFlash draft for the tiny Kimi-K3, in the published layout
    (fc over the captured target layers, hidden_norm, Qwen3 decoder layers,
    norm; embedding and LM head come from the target)."""
    import torch
    from safetensors.torch import save_file

    text = json.load(open(os.path.join(target, "config.json")))["text_config"]
    hidden, vocab = text["hidden_size"], text["vocab_size"]
    heads, kv_heads, head_dim, inter = 8, 2, 128, 2048
    config = {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "block_size": BLOCK_SIZE,
        # Any id the target's vocabulary embeds.
        "dflash_config": {
            "mask_token_id": vocab - 1,
            "target_layer_ids": TARGET_LAYER_IDS,
        },
        "num_target_layers": text["num_hidden_layers"],
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_hidden_layers": 1,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "attention_bias": False,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 8192,
        "layer_types": ["full_attention"],
        "tie_word_embeddings": False,
        "vocab_size": vocab,
        "dtype": "bfloat16",
    }
    os.makedirs(root, exist_ok=True)
    json.dump(config, open(os.path.join(root, "config.json"), "w"), indent=2)
    g = torch.Generator().manual_seed(0)

    def rand(*shape):
        return (torch.randn(*shape, generator=g) * 0.02).to(torch.bfloat16)

    def ones(n):
        return torch.ones(n, dtype=torch.bfloat16)

    p = "layers.0."
    tensors = {
        "fc.weight": rand(hidden, len(TARGET_LAYER_IDS) * hidden),
        "hidden_norm.weight": ones(hidden),
        "norm.weight": ones(hidden),
        p + "input_layernorm.weight": ones(hidden),
        p + "post_attention_layernorm.weight": ones(hidden),
        p + "self_attn.q_proj.weight": rand(heads * head_dim, hidden),
        p + "self_attn.k_proj.weight": rand(kv_heads * head_dim, hidden),
        p + "self_attn.v_proj.weight": rand(kv_heads * head_dim, hidden),
        p + "self_attn.o_proj.weight": rand(hidden, heads * head_dim),
        p + "self_attn.q_norm.weight": ones(head_dim),
        p + "self_attn.k_norm.weight": ones(head_dim),
        p + "mlp.gate_proj.weight": rand(inter, hidden),
        p + "mlp.up_proj.weight": rand(inter, hidden),
        p + "mlp.down_proj.weight": rand(hidden, inter),
    }
    save_file(tensors, os.path.join(root, "model.safetensors"))


def _cuts(prompt: list[int]) -> list[int]:
    return sorted({max(1, (len(prompt) * i) // 24) for i in range(1, 25)})


PREFIXES = [prompt[:cut] for prompt in PROMPTS for cut in _cuts(prompt)]


def _generate(ids: list[int], max_new_tokens: int) -> tuple[list[int], dict]:
    response = requests.post(
        f"{BASE_URL}/generate",
        json={
            "input_ids": ids,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    response.raise_for_status()
    body = response.json()
    return body["output_ids"], body["meta_info"]


class TestKimiK3TinyDFlash(CustomTestCase):
    def test_dflash_verify_matches_plain_decoding(self):
        workdir = tempfile.TemporaryDirectory(prefix="kimi_k3_dflash_")
        self.addCleanup(workdir.cleanup)
        target = os.path.join(workdir.name, "target")
        draft = os.path.join(workdir.name, "draft")
        build_model(target)
        build_draft(draft, target)
        logs = workdir.name

        process, log = launch(target, PORT, [], f"{logs}/plain.log")
        try:
            plain = [tuple(_generate(prefix, 2)[0]) for prefix in PREFIXES]
            # The plain server's extend-vs-decode agreement on the same token.
            extend = [
                _generate(prefix + [t1], 1)[0][0]
                for prefix, (t1, _) in zip(PREFIXES, plain)
            ]
        finally:
            kill_process_tree(process.pid, wait_timeout=60)
            log.close()

        process, log = launch(
            target,
            PORT,
            [
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                draft,
                "--speculative-num-draft-tokens",
                str(BLOCK_SIZE),
                # The draft is dense MHA beside the MLA target, as for DSpark.
                "--speculative-draft-attention-backend",
                "pallas_tpu_rpa_v3",
            ],
            f"{logs}/dflash.log",
            env={"SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1"},
        )
        try:
            served = [_generate(prefix, 2) for prefix in PREFIXES]
            # One longer request to confirm drafts get verified, not skipped.
            _, meta = _generate(PROMPTS[0], MAX_NEW_TOKENS)
        finally:
            kill_process_tree(process.pid, wait_timeout=60)
            log.close()

        total = len(PREFIXES)
        first = sum(tuple(ids)[0] == p[0] for (ids, _), p in zip(served, plain))
        second = sum(tuple(ids) == p for (ids, _), p in zip(served, plain))
        extend_agree = sum(e == p[1] for e, p in zip(extend, plain))
        print(
            f"first token {first}/{total}; verify step {second}/{total}; "
            f"plain extend vs decode {extend_agree}/{total}; "
            f"spec_verify_ct {meta.get('spec_verify_ct')}"
        )
        self.assertGreater(meta.get("spec_verify_ct", 0), 0, "no draft verified")
        self.assertEqual(first, total)
        self.assertGreaterEqual(second, extend_agree - total // 10)


if __name__ == "__main__":
    unittest.main()
