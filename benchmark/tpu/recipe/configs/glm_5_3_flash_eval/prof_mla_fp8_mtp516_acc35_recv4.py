from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.prof_mla_fp8_mtp516_acc35 import CONFIG as _BASE

# The accept-3.5 bench, receiving (and broadcasting) new requests every 4th scheduler iteration.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["scheduler_recv_interval"] = 4
