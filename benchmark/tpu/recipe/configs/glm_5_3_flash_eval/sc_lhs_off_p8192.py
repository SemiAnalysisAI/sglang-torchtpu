from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.c1_dsa_dflash_acc35_scmoe_p8192 import CONFIG as _BASE

# SparseCore-halt test: the halting 8192 config (MoE gathers on SparseCore, top-k on XLA) with
# XLA's latency-hiding scheduler off, so SparseCore programs are not overlapped by it.
CONFIG = deepcopy(_BASE)
args = CONFIG["runtime_env"]["LIBTPU_INIT_ARGS"]
assert "--xla_tpu_enable_latency_hiding_scheduler=true" in args
CONFIG["runtime_env"]["LIBTPU_INIT_ARGS"] = args.replace(
    "--xla_tpu_enable_latency_hiding_scheduler=true", "--xla_tpu_enable_latency_hiding_scheduler=false"
)
