"""Experiment: baseline TP8 plus --xla_tpu_scoped_vmem_limit_kib=32768 (in haolin's recipe, lost in perf.py)."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.ornith_1_5_397b.perf import CONFIG as _BASE

CONFIG = deepcopy(_BASE)
CONFIG["runtime_env"]["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_scoped_vmem_limit_kib=32768 " + _BASE["runtime_env"]["LIBTPU_INIT_ARGS"]
)
