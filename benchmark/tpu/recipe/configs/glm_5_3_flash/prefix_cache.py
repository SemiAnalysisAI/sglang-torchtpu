"""GLM-5.3-Flash with the radix prefix cache backed by host RAM (Raiden HiCache),
no speculative decoding, one tpu7x host.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/prefix_cache.py

A cached prefix reuses the 11 DSA layers' latent and indexer pages and restores
the 34 KDA layers from a state snapshot taken at the cache boundary. Every cached
prefix is also written through to host RAM; when the device pool evicts it, a
later hit loads both back by Raiden DMA. A reload that carries KDA state waits
for the in-flight forward first (one device sync per host hit). KV_OFFLOAD
turns the host tier on (off by default; see below).

Speculative decoding (DFlash / MTP) is refused together with the radix cache on
TPU: verify windows take no KDA snapshots yet.

The KDA state pool must be sized for radix: each running request holds about
MAMBA_RATIO state slots (its live state plus checkpoints), and every cached
prefix keeps one snapshot slot. An undersized pool does not fail; it evicts
snapshots and cache hits silently stop.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as BASE_CONFIG

CONCURRENCY = 8
CONTEXT_LENGTH = 32768
# KV pool shared by running requests and cached prefixes. Up to 131072 tokens
# the sparse-MLA cache writer runs on the SparseCore; above that it falls back to
# the slower XLA scatter (mla_sparse_fused._SPARSECORE_WRITER_MAX_BYTES).
MAX_TOTAL_TOKENS = 131072
MAX_RUNNING = CONCURRENCY + 2
# State slots per running request under radix (live state + checkpoints).
MAMBA_RATIO = 5
# Cached prefixes that keep a KDA snapshot (one slot each, ~18 MB per rank).
CACHED_PREFIX_SLOTS = 64
# Host RAM tier: HICACHE_RATIO x the device pools (KV tokens and KDA slots).
# At ratio 2 that is about 3 GB of KV and 4 GB of KDA state per rank.
# Off by default for now: host offload passes one request at a time (evicted
# prefixes reload exactly), but two requests decoding together trip a
# recompile after warmup.
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
