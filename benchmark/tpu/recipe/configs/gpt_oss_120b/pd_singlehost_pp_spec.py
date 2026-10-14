"""One-host-per-engine Raiden P/D: a PP8 EAGLE3 prefill feeding a TP8/EP8 decode."""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b import pd_singlehost

DRAFT_MODEL = (
    os.environ.get("GPT_OSS_DRAFT_MODEL")
    or os.environ.get("DRAFT_MODEL")
    or "lmsys/EAGLE3-gpt-oss-120b-bf16"
)
CONCURRENCY = 64
PREFILL_TOKENS = 8192

GROUP = deepcopy(pd_singlehost.GROUP)
PREFILL_CONFIG = deepcopy(pd_singlehost.PREFILL_CONFIG)
DECODE_CONFIG = deepcopy(pd_singlehost.DECODE_CONFIG)

for _config in (PREFILL_CONFIG, DECODE_CONFIG):
    _config["server_args"].update(
        kv_cache_dtype="bfloat16",
        disable_radix_cache=True,
        mem_fraction_static=0.6,
        max_total_tokens=1187840,
        max_running_requests=CONCURRENCY,
        context_length=9280,
        chunked_prefill_size=PREFILL_TOKENS,
        max_prefill_tokens=PREFILL_TOKENS,
        torch_compile_max_bs=CONCURRENCY,
        tpu_compile_tokens=[PREFILL_TOKENS],
        tpu_compile_tracer="direct",
        trust_remote_code=True,
        watchdog_timeout=3600,
        dist_timeout=3600,
        disable_hybrid_swa_memory=True,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=DRAFT_MODEL,
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
    )
    if _config["server_args"]["model_path"].startswith("gs://"):
        _config["server_args"]["load_format"] = "runai_streamer"
    if DRAFT_MODEL.startswith("gs://"):
        _config["server_args"]["speculative_draft_load_format"] = "runai_streamer"
    _config["runtime_envs"].update(
        SGLANG_RAIDEN_KV_PORT=os.environ.get("SGLANG_RAIDEN_KV_PORT", "40000"),
        SGLANG_WARMUP_TIMEOUT="3600",
        HEALTH_TIMEOUT=os.environ.get("WAIT_SECS", "5400"),
    )

# Eight single-device stages; PP rejects the overlap schedule.
PREFILL_CONFIG["server_args"].update(
    tp_size=1, ep_size=1, pp_size=8, disable_overlap_schedule=True
)

# A non-PP decode must name its layer pools to pull from a PP producer.
DECODE_CONFIG["runtime_envs"]["SGLANG_RAIDEN_PIPELINE_TRANSFER"] = "1"
# Spec uses CPU-tensor retraction; host-pool I/O is tested separately.
DECODE_CONFIG["server_args"].pop("hicache_mem_layout")
DECODE_CONFIG["server_args"].pop("hicache_io_backend")
DECODE_CONFIG["server_args"]["disaggregation_decode_retraction_backup"] = "cpu_tensor"
