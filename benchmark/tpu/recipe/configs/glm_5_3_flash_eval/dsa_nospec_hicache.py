from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.dsa_nospec_radix import CONFIG as _BASE

# KV offload smoke: radix + Raiden HiCache in host RAM on the recipe's DSA, no spec
# (same server args as gpt_oss_120b/pd_singlehost_kv_offload*).
CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    enable_hierarchical_cache=True,
    hicache_ratio=2,
    hicache_io_backend="raiden",
    hicache_mem_layout="page_first_fused",
    hicache_write_policy="write_through",
    enable_cache_report=True,
)
