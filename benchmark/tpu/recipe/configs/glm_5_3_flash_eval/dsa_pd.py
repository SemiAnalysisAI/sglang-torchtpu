"""PD smoke: GLM-5.3-Flash DSA, no spec, no radix, Raiden KV transfer, one host per engine.

Launch each engine with `--export PREFILL_CONFIG` / `--export DECODE_CONFIG` on its own
slice, then a mini-LB router pointing at both (PREFILL_HOST / DECODE_HOST).
"""
from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.dsa_nospec import CONFIG as _BASE

GROUP = {"hosts": 2, "prefill_engines": 1, "decode_engines": 1}

_COMMON = deepcopy(_BASE)
_COMMON["server_args"].update(
    host="0.0.0.0",
    disaggregation_transfer_backend="raiden",
    skip_server_warmup=True,
)
_COMMON["runtime_env"].update(
    {"SGLANG_RAIDEN_KV_PORT": "31000", "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": None}
)

PREFILL_CONFIG = deepcopy(_COMMON)
PREFILL_CONFIG["server_args"].update(
    port=30000, disaggregation_mode="prefill", disaggregation_bootstrap_port=8998
)
DECODE_CONFIG = deepcopy(_COMMON)
DECODE_CONFIG["server_args"].update(port=30000, disaggregation_mode="decode")
