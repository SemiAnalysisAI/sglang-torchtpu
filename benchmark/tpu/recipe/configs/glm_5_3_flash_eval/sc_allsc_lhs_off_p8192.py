from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.sc_lhs_off_p8192 import CONFIG as _BASE

# SparseCore-halt test: like sc_lhs_off_p8192, but run on a tree whose indexer top-k is back on
# the SparseCore (dsa_kpool_topk.py from 08bf7feae0), i.e. all three SparseCore users on.
CONFIG = deepcopy(_BASE)
