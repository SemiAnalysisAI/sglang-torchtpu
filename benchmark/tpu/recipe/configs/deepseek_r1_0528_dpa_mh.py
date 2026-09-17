"""R1 DPA regression: TP16/EP16/DP2, attention TP8 per group."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_r1_0528_fp8kv_mh import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "dp_size": 2,
        "enable_dp_attention": True,
        "enable_dp_lm_head": True,
        # The global 8192-token prefill budget is divided between DP groups.
        "tpu_compile_tokens": [4096],
    }
)
