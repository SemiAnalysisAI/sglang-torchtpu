# SPDX-License-Identifier: Apache-2.0
"""Reproduce the mixed fused-RS/distributed-TopK TPU core halt.

This uses a two-layer, dummy-weight Qwen3 MoE model so no checkpoint or
tokenizer is required.  The server warmup compiles and executes one static
prefill graph; success means that warmup reaches the health endpoint.

Run on one TPU v7x host (eight workers):

    SGLANG_PLATFORM=tpu python3 \
      test/manual/tpu/repro_fused_rs_distributed_topk_halt.py \
      --tokens 5248 --distributed-top-k

Use ``--tokens 5120 --distributed-top-k`` for the smaller-M control, or omit
``--distributed-top-k`` for the full-vocabulary control at the same shape.
"""

import argparse
import json
import tempfile
from pathlib import Path

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import popen_launch_server


def _model_config() -> dict:
    top_k = 8
    return {
        "architectures": ["Qwen3MoeForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "decoder_sparse_step": 1,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "model_type": "qwen3_moe",
        "moe_intermediate_size": 768,
        "norm_topk_prob": True,
        "num_attention_heads": 32,
        "num_experts": 128,
        "num_experts_per_tok": top_k,
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000000,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        # TP8 leaves 128 logits per rank, enough for the fixed K=50 warmup.
        "vocab_size": 1024,
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "modules_to_not_convert": [
                "model.layers.0.input_layernorm",
                "model.layers.0.mlp.gate",
                "model.layers.0.post_attention_layernorm",
                "model.layers.1.input_layernorm",
                "model.layers.1.mlp.gate",
                "model.layers.1.post_attention_layernorm",
            ],
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--distributed-top-k", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    top_k = 8
    routed_m = args.tokens * top_k
    process = None

    with tempfile.TemporaryDirectory(prefix="fused-rs-topk-halt-") as temp_dir:
        model_dir = Path(temp_dir)
        (model_dir / "config.json").write_text(
            json.dumps(_model_config()), encoding="utf-8"
        )

        # distributed_top_k_for_batch reads this value when the server process
        # imports the TPU sampler.  B=1 is intentional: it removes scheduler
        # concurrency and leaves only the compiled graph composition under test.
        min_top_k_batch = "1" if args.distributed_top_k else "2"
        env = {
            "JAX_PLATFORMS": "cpu",
            "SGLANG_TPU_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE": min_top_k_batch,
            "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false",
        }
        server_args = [
            "--load-format",
            "dummy",
            "--skip-tokenizer-init",
            "--dtype",
            "bfloat16",
            "--device",
            "tpu",
            "--attention-backend",
            "pallas_tpu_rpa_v3",
            "--moe-runner-backend",
            "pallas_tpu_fused_rs",
            "--tpu-fp8-runtime",
            "channelwise",
            "--quantization",
            "fp8",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--sampling-backend",
            "tpu",
            "--tp-size",
            "8",
            "--ep-size",
            "8",
            "--enable-torch-compile",
            "--tpu-compile-tracer",
            "dynamo",
            "--tpu-compile-tokens",
            str(args.tokens),
            "--tpu-compile-bs",
            "1",
            "--torch-compile-max-bs",
            "1",
            "--chunked-prefill-size",
            str(args.tokens),
            "--max-prefill-tokens",
            str(args.tokens),
            "--context-length",
            str(args.tokens * 2),
            "--max-running-requests",
            "1",
            "--max-total-tokens",
            str(args.tokens * 2),
            "--page-size",
            "128",
            "--mem-fraction-static",
            "0.3",
            "--disable-radix-cache",
            "--grammar-backend",
            "none",
        ]

        mode = "distributed-top-k" if args.distributed_top_k else "full-vocabulary"
        print(
            f"FUSED_RS_TOPK_REPRO_START mode={mode} tokens={args.tokens} "
            f"routed_m={routed_m}",
            flush=True,
        )
        try:
            process = popen_launch_server(
                model=str(model_dir),
                base_url="http://127.0.0.1:30000",
                timeout=args.timeout,
                other_args=server_args,
                env=env,
                device="tpu",
            )
            if process.poll() is not None:
                raise RuntimeError(f"server exited with status {process.returncode}")
            print(
                f"FUSED_RS_TOPK_REPRO_PASS mode={mode} tokens={args.tokens} "
                f"routed_m={routed_m}",
                flush=True,
            )
        finally:
            if process is not None:
                kill_process_tree(process.pid, wait_timeout=60)


if __name__ == "__main__":
    main()
