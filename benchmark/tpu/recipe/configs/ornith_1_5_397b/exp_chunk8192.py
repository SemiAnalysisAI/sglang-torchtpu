"""Experiment: baseline TP8 with chunked prefill 8192."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.ornith_1_5_397b.perf import CONFIG as _BASE

CHUNK = 8192
CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    {
        "chunked_prefill_size": CHUNK,
        "max_prefill_tokens": CHUNK,
        "tpu_compile_tokens": [CHUNK],
    }
)
