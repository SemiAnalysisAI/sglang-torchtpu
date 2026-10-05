"""Gemma-4-26B-A4B-it for the AIME26 nightly: a 64k thinking budget.

At 32k, 4% of the 26B's AIME26 answers hit the cap and score 0. A 64k context
holds half as many full requests in the same pool, so concurrency drops to 32.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gemma4_26b_a4b.accuracy import CONFIG as BASE_CONFIG

CONCURRENCY = 32
CONTEXT_LENGTH = 65536 + 2048  # 64k thinking budget + prompt

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "max_total_tokens": CONCURRENCY * CONTEXT_LENGTH,
    }
)
