"""GLM-5.3-Flash Raiden P/D: one tpu7x host per engine, no speculative decoding.

Sample usage (from the repo root), one command per host, then a router:
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/pd.py:PREFILL_CONFIG
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/pd.py:DECODE_CONFIG
    python3 -m sglang_router.launch_router --pd-disaggregation --mini-lb \\
        --prefill http://<prefill-host>:30000 8998 \\
        --decode http://<decode-host>:30000 --host 0.0.0.0 --port 8000
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as _BASE

GROUP = {"hosts": 2, "prefill_engines": 1, "decode_engines": 1}

CONTEXT_LENGTH = 32768
MAX_RUNNING = 10

_COMMON = deepcopy(_BASE)
_COMMON["server_args"].update(
    {
        "host": "0.0.0.0",
        "context_length": CONTEXT_LENGTH,
        "max_total_tokens": 131072,
        "max_running_requests": MAX_RUNNING,
        "torch_compile_max_bs": MAX_RUNNING,
        "tpu_compile_bs": [1, 2, 4, 8],
        "max_mamba_cache_size": MAX_RUNNING + 6,
        "disaggregation_transfer_backend": "raiden",
        "skip_server_warmup": True,
    }
)
_COMMON["runtime_env"].update(
    {"SGLANG_RAIDEN_KV_PORT": os.environ.get("SGLANG_RAIDEN_KV_PORT", "31000")}
)

PREFILL_CONFIG = deepcopy(_COMMON)
PREFILL_CONFIG["server_args"].update(
    port=int(os.environ.get("PREFILL_PORT", "30000")),
    disaggregation_mode="prefill",
    disaggregation_bootstrap_port=int(os.environ.get("BOOTSTRAP_PORT", "8998")),
)
DECODE_CONFIG = deepcopy(_COMMON)
DECODE_CONFIG["server_args"].update(
    port=int(os.environ.get("DECODE_PORT", "30000")),
    disaggregation_mode="decode",
)
