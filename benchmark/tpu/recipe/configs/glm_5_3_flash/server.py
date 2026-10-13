"""GLM-5.3-Flash text serving on one tpu7x host (2x2x1 = 4 chips x 2 cores = tp8),
tuned for 8k-in/1k-out decode latency. Callers copy CONFIG before overriding it.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/server.py
    # add --dry-run to print the resolved sglang.launch_server command.

Full attention runs densely: DSA selects index_topk = 2048 tokens, so dense is
exact up to a 2048-token context. 8k/1k needs ~9.3k, so this recipe raises the
dense limit; past 2048 tokens the attention is denser than the model's DSA.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

INPUT_TOKENS = 8192
OUTPUT_TOKENS = 1024
# Largest concurrency served; one server covers c = 1 .. CONCURRENCY.
CONCURRENCY = 1
CONTEXT_LENGTH = 10240
# The request pool holds slots back at higher concurrency: with
# max_running_requests = 16 only 14 requests ever ran at once. c = 1 is unaffected.
MAX_RUNNING = CONCURRENCY if CONCURRENCY == 1 else CONCURRENCY + 2
# Batch buckets to precompile; the launcher adds MAX_RUNNING if it is missing.
COMPILE_BS = [b for b in (1, 2, 4, 8, 16) if b <= CONCURRENCY] or [1]

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/GLM-5.3-Flash",
        "tp_size": 8,
        # Tensor-parallel experts: at small batch EP8 waits on the busiest rank
        # (~2.6 of a token's 8 experts); EP=1 gives every rank 1/8 of each expert,
        # and enables the streaming one-token MoE kernel.
        "ep_size": 1,
        "language_model_only": True,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": 8192,
        "max_prefill_tokens": 8192,
        # Every extend pads to the next bucket: with 8192 alone, a short chat
        # turn or the tail chunk of an 8k+ prompt costs a full 8k prefill.
        "tpu_compile_tokens": [1024, 2048, 4096, 8192],
        "max_running_requests": MAX_RUNNING,
        "torch_compile_max_bs": MAX_RUNNING,
        "tpu_compile_bs": COMPILE_BS,
        "max_total_tokens": max(2, MAX_RUNNING) * CONTEXT_LENGTH,
        # Linear-attention (KDA) state slots; one per running request.
        "max_mamba_cache_size": MAX_RUNNING + 2,
        # Only the 11 full-attention layers have a KV cache; bf16 matches the
        # dense attention path these numbers were measured on.
        "kv_cache_dtype": "bfloat16",
        "mem_fraction_static": 0.9,
        "disable_shared_experts_fusion": True,
        "watchdog_timeout": 1800,
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT": str(CONTEXT_LENGTH),
        "SGLANG_MAMBA_SSM_DTYPE": "float32",
        # Decode speed flags, each measured by interleaved A/B on 8k/1k c=1:
        # route-meta MoE path (single-token SMEM counting sort).
        "SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD": "512",
        # f32 MoE scales remove a per-step scale relayout.
        "SGLANG_TPU_FP8_MOE_FP32_SCALES": "1",
    }
)
