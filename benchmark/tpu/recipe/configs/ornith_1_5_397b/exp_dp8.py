"""Experiment: DP8 attention + EP8 MoE for 8k1k up to c128 (per haolin/qwen35-recipe-dp-attention)."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.ornith_1_5_397b.perf import CONFIG as _BASE

DP = 8
CONCURRENCY = 128
CONTEXT = 8192 + 1024 + 20
MAX_RUNNING = CONCURRENCY + 4
PER_RANK_BS = (MAX_RUNNING + DP - 1) // DP
PER_RANK_CONC = (CONCURRENCY + DP - 1) // DP
CHUNK_PER_RANK = 4096

CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    {
        # TPU DP attention needs the text-only entry (Qwen3_5MoeForCausalLM); the recipe
        # builds this dir with TEXT_ENTRY=1: flat text config, weights symlinked.
        "model_path": "/dev/shm/Ornith-1.5-397B-FP8-text",
        "context_length": CONTEXT,
        "max_running_requests": MAX_RUNNING,
        # sglang divides the chunk budget by dp_size; scale so 4096 is per rank.
        "chunked_prefill_size": CHUNK_PER_RANK * DP,
        "max_prefill_tokens": CHUNK_PER_RANK * DP,
        "tpu_compile_tokens": [CHUNK_PER_RANK],
        "tpu_compile_bs": [PER_RANK_BS],
        "torch_compile_max_bs": PER_RANK_BS,
        "max_total_tokens": PER_RANK_CONC * CONTEXT + PER_RANK_BS * 2048,
        "enable_dp_attention": True,
        "dp_size": DP,
        "enable_dp_lm_head": True,
        "enable_prefill_delayer": True,
        "prefill_delayer_queue_min_ratio": 0.5,
        "prefill_delayer_max_delay_ms": 2000,
        "prefill_delayer_max_delay_passes": 250,
    }
)
# MAX_RUNNING * topk = 1320 must clear the dense-permute threshold.
CONFIG["runtime_env"]["SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD"] = "2048"
