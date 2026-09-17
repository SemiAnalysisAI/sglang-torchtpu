"""One-host-per-engine Raiden P/D with EAGLE3 on both engines."""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b import pd_singlehost

E = os.environ.get
DRAFT_MODEL = (
    E("GPT_OSS_DRAFT_MODEL") or E("DRAFT_MODEL") or "lmsys/EAGLE3-gpt-oss-120b-bf16"
)
GROUP = deepcopy(pd_singlehost.GROUP)
PREFILL_CONFIG = deepcopy(pd_singlehost.PREFILL_CONFIG)
DECODE_CONFIG = deepcopy(pd_singlehost.DECODE_CONFIG)

for _config in (PREFILL_CONFIG, DECODE_CONFIG):
    _args = _config["server_args"]
    _args.update(
        kv_cache_dtype="bfloat16",
        disable_radix_cache=True,
        mem_fraction_static=float(E("MEM_FRACTION_STATIC", "0.6")),
        max_total_tokens=8192,
        max_running_requests=16,
        context_length=int(E("CONTEXT_LEN", "8192")),
        max_prefill_tokens=int(E("MAX_PREFILL_TOKENS", "4096")),
        torch_compile_max_bs=int(E("TORCH_COMPILE_MAX_BS", "16")),
        tpu_compile_bs=[
            int(value) for value in E("TPU_COMPILE_BS", "1 2 4 16").split()
        ],
        tpu_compile_tokens=[
            int(value) for value in E("TPU_COMPILE_TOKENS", "512 2048 4096").split()
        ],
        watchdog_timeout=7200,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=DRAFT_MODEL,
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
    )
    if _args["model_path"].startswith("gs://"):
        _args["load_format"] = "runai_streamer"
    if DRAFT_MODEL.startswith("gs://"):
        _args["speculative_draft_load_format"] = "runai_streamer"
    # Spec uses CPU-tensor retraction; host-pool I/O is tested separately.
    for _key in ("hicache_mem_layout", "hicache_io_backend"):
        if _key in pd_singlehost.PREFILL_CONFIG["server_args"]:
            _args[_key] = pd_singlehost.PREFILL_CONFIG["server_args"][_key]
        else:
            _args.pop(_key, None)
    _env = _config["runtime_envs"]
    _env.update(
        DRAFT_MODEL=DRAFT_MODEL,
        SGLANG_RAIDEN_RESHARD="0",
        ENABLE_PROFILER_ENDPOINT="0",
        HEALTH_TIMEOUT=E("WAIT_SECS", "5400"),
    )
    _env.pop("SGLANG_TORCH_PROFILER_DIR", None)

DECODE_CONFIG["server_args"]["disaggregation_decode_retraction_backup"] = "cpu_tensor"
