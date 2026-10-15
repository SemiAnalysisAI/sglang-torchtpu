"""GLM-5.3-Flash with the radix prefix cache backed by host RAM (Raiden HiCache),
no speculative decoding, one tpu7x host.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/kv_offload.py

Every cached prefix is written through to host RAM. When the device pool evicts
it, a later hit loads it back by Raiden DMA: the 11 DSA layers' latent and
indexer pages, and the 34 KDA layers' state snapshot at the cache boundary.
A reload that carries KDA state waits for the in-flight forward first (one
device sync per host hit).

Host RAM: about HICACHE_RATIO x (the device KV pool + the KDA state pool). With
the prefix_cache sizing at ratio 2 that is roughly 3 GB of KV and 4 GB of KDA
state per rank.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.prefix_cache import CONFIG as _BASE

# Host pool size as a multiple of the device pools (KV tokens and KDA slots).
HICACHE_RATIO = 2

CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    {
        "enable_hierarchical_cache": True,
        "hicache_ratio": HICACHE_RATIO,
        "hicache_io_backend": "raiden",
        "hicache_mem_layout": "page_first_fused",
        "hicache_write_policy": "write_through",
        "enable_cache_report": True,
    }
)
