"""GPT-OSS-120B with Raiden HiCache under an agentic multi-turn load.

The device KV pool is capped below the load's working set, so prefixes spill
to the host pool and come back on later turns. Environment knobs:
  HICACHE_WRITE_POLICY  write policy (default write_back)
  CHAT_TEMPLATE         path to a .jinja chat template; use a copy with a fixed
                        date, or every cached prefix changes at midnight
  DEVICE_POOL_TOKENS    device KV pool size (default 786432)

    HICACHE_WRITE_POLICY=write_through CHAT_TEMPLATE=$HOME/models/gpt-oss-fixed-date.jinja \\
        python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_120b/agentic_hicache.py
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG as SERVER_CONFIG

CONCURRENCY = 32
CONTEXT_LENGTH = 131072
# The EXTEND precompile OOMs past ~8k tokens per chunk (see server.py).
PREFILL_TOKENS = 8192
# Sized for agentic_multiturn_client.py at its defaults (20 sessions, restart at
# 72k prompt tokens). Each session keeps one main and up to five reminder
# prefix chains, a working set of about 4-5M tokens: the device pool holds a
# sixth of that and must evict.
DEVICE_POOL_TOKENS = int(os.environ.get("DEVICE_POOL_TOKENS", 786_432))
# GB per rank for the host pools. At ~9.2 KB per token per rank this holds
# ~6.3M full-attention tokens (8x the device pool, the whole working set) plus
# ~1.3M SWA tokens. Needs about 560 GB of host RAM across the 8 ranks.
HOST_POOL_GB = 70

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    {
        "host": "0.0.0.0",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "max_total_tokens": DEVICE_POOL_TOKENS,
        "swa_full_tokens_ratio": 0.2,
        "enable_hierarchical_cache": True,
        "hicache_io_backend": "raiden",
        "hicache_mem_layout": "page_first_fused",
        "hicache_size": HOST_POOL_GB,
        "hicache_write_policy": os.environ.get("HICACHE_WRITE_POLICY", "write_back"),
        "enable_cache_report": True,
        "enable_metrics": True,
    }
)
if os.environ.get("CHAT_TEMPLATE"):
    CONFIG["server_args"]["chat_template"] = os.environ["CHAT_TEMPLATE"]
# Keep SWA KV for every token, not only the last window: more SWA state per
# prefix to back up and reload, which is where the write policies differ most.
CONFIG["runtime_env"].update({"SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS": "0"})
