"""GPT-OSS-120B Raiden P/D: one TP8/EP8 host for each engine.

Run the shared launcher with --config recipe/configs/gpt_oss_120b/pd_singlehost.py.
SGLANG_LAUNCHER_HOSTNAMES lists prefill,decode; SGLANG_LAUNCHER_RANK is 0 or 1.
The test's existing environment overrides remain available for this recipe.
"""

import os
import re
import shlex
import time
from copy import deepcopy
from pathlib import Path

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

E = os.environ.get
MODEL = E("GPT_OSS_MODEL") or E("MODEL") or "/weights/gpt-oss-120b"
GROUP = {"hosts": 2, "prefill_engines": 1, "decode_engines": 1}

_COMMON = deepcopy(_FP8_BASE_CONFIG)
_COMMON["runtime_envs"] = _COMMON.pop("runtime_env")
_ARGS = _COMMON["server_args"]
_ARGS.update(
    model_path=MODEL,
    served_model_name="gpt-oss-120b",
    host="0.0.0.0",
    tp_size=8,
    ep_size=8,
    page_size=64,
    quantization=None,
    kv_cache_dtype="fp8_e4m3",
    disable_radix_cache=False,
    mem_fraction_static=float(E("MEM_FRACTION_STATIC", "0.35")),
    max_total_tokens=393216,
    max_running_requests=int(E("MAX_RUNNING") or E("CONC", "64")),
    chunked_prefill_size=int(E("CHUNKED_PREFILL_SIZE", "4096")),
    context_length=int(
        E("CONTEXT_LEN") or (int(E("ISL", "8192")) + int(E("OSL", "1024")) + 20)
    ),
    enable_torch_compile=E("ENABLE_TORCH_COMPILE", "1") == "1",
    reasoning_parser="gpt-oss",
    sampling_backend="pytorch",
    skip_server_warmup=True,
    disaggregation_transfer_backend="raiden",
)
_ARGS["torch_compile_max_bs"] = int(
    E("TORCH_COMPILE_MAX_BS") or _ARGS["max_running_requests"]
)
_ARGS["max_prefill_tokens"] = int(
    E("MAX_PREFILL_TOKENS") or _ARGS["chunked_prefill_size"]
)
for _key in ("tpu_compile_tokens", "tpu_compile_bs"):
    if E(_key.upper()):
        _ARGS[_key] = [int(value) for value in E(_key.upper()).split()]
_DP = int(E("DP_SIZE") or 1)
if _DP < 1 or 8 % _DP:
    raise ValueError("DP_SIZE must be a positive divisor of TP8")
if _DP > 1:
    _ARGS.update(dp_size=_DP, enable_dp_attention=True, enable_dp_lm_head=8 // _DP > 1)
for _key in ("enable_metrics", "enable_cache_report", "disable_overlap_schedule"):
    if E(_key.upper()) == "1":
        _ARGS[_key] = True
if E("NUM_CONTINUOUS_DECODE_STEPS"):
    _ARGS["num_continuous_decode_steps"] = int(E("NUM_CONTINUOUS_DECODE_STEPS"))
if E("JSON_MODEL_OVERRIDE"):
    _ARGS["json_model_override_args"] = E("JSON_MODEL_OVERRIDE")
if E("KV_OFFLOAD", "0") == "1":
    _ARGS.update(
        enable_hierarchical_cache=True,
        hicache_mem_layout=E("HICACHE_MEM_LAYOUT", "page_first_fused"),
        hicache_io_backend=E("HICACHE_IO_BACKEND", "torch_tpu"),
    )
    if E("HICACHE_SIZE"):
        _ARGS["hicache_size"] = float(E("HICACHE_SIZE"))
    else:
        _ARGS["hicache_ratio"] = float(E("HICACHE_RATIO", "2.0"))
    if E("HICACHE_WRITE_POLICY"):
        _ARGS["hicache_write_policy"] = E("HICACHE_WRITE_POLICY")

# Runtime settings previously supplied by common_env.sh and launch_server.sh.
_ENV = _COMMON["runtime_envs"]
_ENV.update(
    HF_HUB_OFFLINE="0",
    TRANSFORMERS_OFFLINE="0",
    TORCH_LOGS="+recompiles",
    TPU_ENABLE_DIAGNOSTIC_COORDINATOR="0",
    TPU_DISABLE_DIAGNOSTIC_COORDINATOR="1",
    TPU_DIAGNOSTIC_HARVESTER_INTERVAL_SEC="86400",
    SGLANG_TPU_PRELOAD_TELEMETRY_STUB="1",
)
_ENV.update({key: E(key, default) for key, default in _ENV.items()})
_ENV.update(
    SGLANG_TPU_FUSED_MOE_SC_GATHER="0",
    SGLANG_RAIDEN_KV_PORT=E("SGLANG_RAIDEN_KV_PORT", "31000"),
    LB_PORT=E("LB_PORT", "8000"),
    HEALTH_TIMEOUT=E("WAIT_SECS", "3600"),
    MODEL=None,
)
_SLUG = re.sub(r"[^a-z0-9]+", "_", MODEL.lower()).strip("_")
_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
_RUN_DIR = E("RUN_DIR", f"./profile/{_SLUG}-tpu/{_TIMESTAMP}")
_ENV["RUN_DIR"] = _RUN_DIR
if E("ENABLE_COMPILE_CACHE", "1") == "1":
    _ENV.update(
        TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=E(
            "TORCH_TPU_TIER2_NAME", f"torchtpu_{_SLUG}"
        ),
        TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=E(
            "TORCH_TPU_TIER3_ROOT", str(Path.home() / ".cache/torch_tpu_compile_cache")
        ),
        TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK="1",
    )
if E("ENABLE_PROFILER_ENDPOINT", "1") == "1":
    _ENV["SGLANG_TORCH_PROFILER_DIR"] = E("PROFILER_DIR", _RUN_DIR)
if E("ENABLE_TORCH_COMPILE_DEBUG", "0") == "1":
    _ENV.update(
        TORCH_COMPILE_DEBUG="1",
        TORCH_COMPILE_DEBUG_DIR=E(
            "TORCH_COMPILE_DEBUG_DIR", _RUN_DIR + "/torch_compile_debug"
        ),
    )
if E("ENABLE_HLO_DUMP", "0") == "1":
    _DUMP_DIR = E("HLO_DUMP_DIR", f"./hlo_dump/{_SLUG}-tpu/{_TIMESTAMP}")
    _ENV["XLA_FLAGS"] = E(
        "XLA_FLAGS",
        f"--xla_dump_to={_DUMP_DIR} --xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_hlo_module_re=.*tt_jit.*",
    )

PREFILL_CONFIG = deepcopy(_COMMON)
PREFILL_CONFIG["server_args"].update(
    port=int(E("PREFILL_PORT", "30000")),
    disaggregation_mode="prefill",
    disaggregation_bootstrap_port=int(E("BOOTSTRAP_PORT", "8998")),
)
PREFILL_CONFIG["extra_args"] = shlex.split(E("EXTRA_PREFILL_ARGS", ""))

DECODE_CONFIG = deepcopy(_COMMON)
DECODE_CONFIG["server_args"].update(
    port=int(E("DECODE_PORT", "30000")),
    disaggregation_mode="decode",
    hicache_mem_layout="page_first_fused",
    hicache_io_backend="raiden",
)
DECODE_CONFIG["extra_args"] = shlex.split(E("EXTRA_DECODE_ARGS", ""))
