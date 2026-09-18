"""GPT-OSS-120B Raiden P/D: one TP8/EP8 host for each engine.

Run with --config recipe/configs/gpt_oss_120b/pd_singlehost.py.
SGLANG_LAUNCHER_HOSTNAMES lists prefill,decode; SGLANG_LAUNCHER_RANK is 0 or 1.

Server arguments combine the original test's model settings with launch_server.sh
sizing/compile options. Runtime envs carry common_env.sh defaults and optional
cache/profiling settings. The exports below add only the differences between roles.
Environment-based options are read when this recipe is imported; the launcher's
--runtime-envs overrides are applied separately to child processes.
"""

import os
import re
import shlex
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

GROUP = {"hosts": 2, "prefill_engines": 1, "decode_engines": 1}
MODEL_PATH = (
    os.environ.get("GPT_OSS_MODEL")
    or os.environ.get("MODEL")
    or "/weights/gpt-oss-120b"
)


def _build_server_args(env: Mapping[str, str], model_path: str) -> dict[str, Any]:
    """Preserve the test's fixed TPU settings and shell-launcher option defaults."""
    args = deepcopy(_FP8_BASE_CONFIG["server_args"])
    args.update(
        model_path=model_path,
        served_model_name="gpt-oss-120b",
        host="0.0.0.0",
        tp_size=8,
        ep_size=8,
        page_size=64,
        # Use checkpoint-provided MXFP4 weights, with FP8 KV and radix caching.
        quantization=None,
        kv_cache_dtype="fp8_e4m3",
        disable_radix_cache=False,
        mem_fraction_static=float(env.get("MEM_FRACTION_STATIC", "0.35")),
        max_total_tokens=393216,
        max_running_requests=int(env.get("MAX_RUNNING") or env.get("CONC", "64")),
        chunked_prefill_size=int(env.get("CHUNKED_PREFILL_SIZE", "4096")),
        context_length=int(
            env.get("CONTEXT_LEN")
            or (int(env.get("ISL", "8192")) + int(env.get("OSL", "1024")) + 20)
        ),
        reasoning_parser="gpt-oss",
        sampling_backend="pytorch",
        skip_server_warmup=True,
        disaggregation_transfer_backend="raiden",
    )

    # launch_server.sh derives compile limits from concurrency and prefill size.
    args.update(
        enable_torch_compile=env.get("ENABLE_TORCH_COMPILE", "1") == "1",
        torch_compile_max_bs=int(
            env.get("TORCH_COMPILE_MAX_BS") or args["max_running_requests"]
        ),
        max_prefill_tokens=int(
            env.get("MAX_PREFILL_TOKENS") or args["chunked_prefill_size"]
        ),
    )
    for env_name, arg_name in (
        ("TPU_COMPILE_TOKENS", "tpu_compile_tokens"),
        ("TPU_COMPILE_BS", "tpu_compile_bs"),
    ):
        if value := env.get(env_name):
            args[arg_name] = [int(token) for token in value.split()]

    # Attention DP divides this engine's eight TP ranks; MoE stays TP8/EP8.
    dp_size = int(env.get("DP_SIZE") or 1)
    if dp_size < 1 or 8 % dp_size:
        raise ValueError("DP_SIZE must be a positive divisor of TP8")
    if dp_size > 1:
        args.update(
            dp_size=dp_size,
            enable_dp_attention=True,
            enable_dp_lm_head=8 // dp_size > 1,
        )

    for env_name, arg_name in (
        ("ENABLE_METRICS", "enable_metrics"),
        ("ENABLE_CACHE_REPORT", "enable_cache_report"),
        ("DISABLE_OVERLAP_SCHEDULE", "disable_overlap_schedule"),
    ):
        if env.get(env_name) == "1":
            args[arg_name] = True
    if env.get("NUM_CONTINUOUS_DECODE_STEPS"):
        args["num_continuous_decode_steps"] = int(env["NUM_CONTINUOUS_DECODE_STEPS"])
    if env.get("JSON_MODEL_OVERRIDE"):
        args["json_model_override_args"] = env["JSON_MODEL_OVERRIDE"]

    # Optional host-memory KV offload, as in launch_server.sh. Decode's Raiden
    # host-pool layout/backend are fixed in DECODE_CONFIG below.
    if env.get("KV_OFFLOAD", "0") == "1":
        args.update(
            enable_hierarchical_cache=True,
            hicache_mem_layout=env.get("HICACHE_MEM_LAYOUT", "page_first_fused"),
            hicache_io_backend=env.get("HICACHE_IO_BACKEND", "torch_tpu"),
        )
        if env.get("HICACHE_SIZE"):
            args["hicache_size"] = float(env["HICACHE_SIZE"])
        else:
            args["hicache_ratio"] = float(env.get("HICACHE_RATIO", "2.0"))
        if env.get("HICACHE_WRITE_POLICY"):
            args["hicache_write_policy"] = env["HICACHE_WRITE_POLICY"]
    return args


def _build_runtime_envs(
    env: Mapping[str, str], model_path: str
) -> dict[str, str | None]:
    """Build child env defaults, fixed test settings, and optional artifact paths."""
    # common_env.sh defaults: an explicitly exported value wins for these keys.
    defaults = {
        **_FP8_BASE_CONFIG["runtime_env"],
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "TORCH_LOGS": "+recompiles",
        "TPU_ENABLE_DIAGNOSTIC_COORDINATOR": "0",
        "TPU_DISABLE_DIAGNOSTIC_COORDINATOR": "1",
        "TPU_DIAGNOSTIC_HARVESTER_INTERVAL_SEC": "86400",
        "SGLANG_TPU_PRELOAD_TELEMETRY_STUB": "1",
    }
    runtime_envs: dict[str, str | None] = {
        key: env.get(key, default) for key, default in defaults.items()
    }
    runtime_envs.update(
        # The original test disables the unsupported SparseCore MoE gather path.
        SGLANG_TPU_FUSED_MOE_SC_GATHER="0",
        SGLANG_RAIDEN_KV_PORT=env.get("SGLANG_RAIDEN_KV_PORT", "31000"),
        LB_PORT=env.get("LB_PORT", "8000"),
        HEALTH_TIMEOUT=env.get("WAIT_SECS", "3600"),
        # model_path above already honors GPT_OSS_MODEL before MODEL. Do not let
        # the generic launcher overwrite that choice with an inherited MODEL.
        MODEL=None,
    )

    # launch_server.sh names output and compile-cache directories by model.
    model_slug = re.sub(r"[^a-z0-9]+", "_", model_path.lower()).strip("_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = env.get("RUN_DIR", f"./profile/{model_slug}-tpu/{timestamp}")
    runtime_envs["RUN_DIR"] = run_dir
    if env.get("ENABLE_COMPILE_CACHE", "1") == "1":
        runtime_envs.update(
            TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=env.get(
                "TORCH_TPU_TIER2_NAME", f"torchtpu_{model_slug}"
            ),
            TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=env.get(
                "TORCH_TPU_TIER3_ROOT",
                str(Path.home() / ".cache/torch_tpu_compile_cache"),
            ),
            TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK="1",
        )

    # Profiling is enabled by default; compiler and HLO dumps are opt-in.
    if env.get("ENABLE_PROFILER_ENDPOINT", "1") == "1":
        runtime_envs["SGLANG_TORCH_PROFILER_DIR"] = env.get("PROFILER_DIR", run_dir)
    if env.get("ENABLE_TORCH_COMPILE_DEBUG", "0") == "1":
        runtime_envs.update(
            TORCH_COMPILE_DEBUG="1",
            TORCH_COMPILE_DEBUG_DIR=env.get(
                "TORCH_COMPILE_DEBUG_DIR", run_dir + "/torch_compile_debug"
            ),
        )
    if env.get("ENABLE_HLO_DUMP", "0") == "1":
        dump_dir = env.get("HLO_DUMP_DIR", f"./hlo_dump/{model_slug}-tpu/{timestamp}")
        runtime_envs["XLA_FLAGS"] = env.get(
            "XLA_FLAGS",
            f"--xla_dump_to={dump_dir} --xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_hlo_module_re=.*tt_jit.*",
        )
    return runtime_envs


# Independent role configs: mutating one engine must not change the other.
_COMMON_CONFIG = {
    "server_args": _build_server_args(os.environ, MODEL_PATH),
    "runtime_envs": _build_runtime_envs(os.environ, MODEL_PATH),
}
PREFILL_CONFIG = deepcopy(_COMMON_CONFIG)
PREFILL_CONFIG["server_args"].update(
    port=int(os.environ.get("PREFILL_PORT", "30000")),
    disaggregation_mode="prefill",
    disaggregation_bootstrap_port=int(os.environ.get("BOOTSTRAP_PORT", "8998")),
)
PREFILL_CONFIG["extra_args"] = shlex.split(os.environ.get("EXTRA_PREFILL_ARGS", ""))

DECODE_CONFIG = deepcopy(_COMMON_CONFIG)
DECODE_CONFIG["server_args"].update(
    port=int(os.environ.get("DECODE_PORT", "30000")),
    disaggregation_mode="decode",
    hicache_mem_layout="page_first_fused",
    hicache_io_backend="raiden",
)
DECODE_CONFIG["extra_args"] = shlex.split(os.environ.get("EXTRA_DECODE_ARGS", ""))
