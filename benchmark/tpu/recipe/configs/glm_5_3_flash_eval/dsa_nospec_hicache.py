from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.dsa_nospec_radix import CONFIG as _BASE

# KV offload smoke: radix + Raiden HiCache in host RAM on the recipe's DSA, no spec.
# A small device pool (32k tokens, 16 KDA slots) so a handful of ~5k-token prompts
# overflows it and the early prefixes must come back from host.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    context_length=16384,
    max_total_tokens=32768,
    enable_hierarchical_cache=True,
    hicache_ratio=3,
    hicache_io_backend="raiden",
    hicache_mem_layout="page_first_fused",
    hicache_write_policy="write_through",
    enable_cache_report=True,
)
