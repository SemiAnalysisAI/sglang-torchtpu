"""GLM-5.3-Flash with the radix prefix cache, no speculative decoding, one tpu7x host.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/prefix_cache.py

KV_OFFLOAD adds a host-RAM tier (Raiden HiCache). Speculative decoding is not
supported with the radix cache on TPU yet.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as BASE_CONFIG

CONCURRENCY = 8
CONTEXT_LENGTH = 32768
# Past 131072 tokens the sparse-MLA cache writer leaves the SparseCore.
MAX_TOTAL_TOKENS = 131072
MAX_RUNNING = CONCURRENCY + 2
# KDA state slots: about 5 per running request, plus one per cached prefix.
# Too few does not fail; cache hits silently stop.
MAMBA_RATIO = 5
CACHED_PREFIX_SLOTS = 64
# Off until concurrent decode with offload is verified.
KV_OFFLOAD = False
HICACHE_RATIO = 2

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "disable_radix_cache": False,
        "context_length": CONTEXT_LENGTH,
        "max_running_requests": MAX_RUNNING,
        "torch_compile_max_bs": MAX_RUNNING,
        "tpu_compile_bs": [1, 2, 4, 8],
        "max_total_tokens": MAX_TOTAL_TOKENS,
        "max_mamba_cache_size": MAX_RUNNING * MAMBA_RATIO + CACHED_PREFIX_SLOTS,
        "enable_cache_report": True,
    }
)
if KV_OFFLOAD:
    CONFIG["server_args"].update(
        {
            "enable_hierarchical_cache": True,
            "hicache_ratio": HICACHE_RATIO,
            "hicache_io_backend": "raiden",
            "hicache_mem_layout": "page_first_fused",
            "hicache_write_policy": "write_through",
        }
    )
