"""GPT-OSS-120B single-host decode-context-parallel gate.

DCP shards each request's KV by absolute token position across a slice of the
TP group (rank ``r`` owns position ``p`` iff ``p % dcp == r``) and merges the
per-rank partial attention by log-sum-exp, trading KV-head replication for KV
capacity. Nothing in the nightly exercised it, so a regression in the position
shard, the span write or the merge could only show up in someone's manual run.

Two gates on one boot, in this order:

* ``dcp`` asserts the mechanism -- that the engine really is running a group of
  two, that a cold read and a prefix-hit read of the same prompt decode
  identical text, and that a needle buried mid-context comes back. It runs
  first because it flushes the cache it reasons about.
* ``gsm8k`` scores the same boot against the same floor the other single-host
  gpt-oss lanes use. DCP moves KV, it does not change what the model computes,
  so a drop here is a corrupted merge rather than a quality trade.

Neither gate alone is sufficient: GSM8K would score the same on a server that
silently ignored ``--dcp-size``, and the mechanism gate says nothing about
whether long-context quality survived.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.dcp_probe import dcp_runner

# Bring-up plus the probe's four prompts read twice each plus 200 GSM8K
# examples -- the same envelope as the DPA lane.
register_tpu_ci(
    est_time=3600,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "dcp"],
)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

# Same floor as the other single-host gpt-oss quality lanes.
GSM8K_ACCURACY = 0.70
DCP_SIZE = 2

LAUNCH_ENV = {
    "MODEL": "/weights/gpt-oss-120b",
    "ISL": "4096",
    "OSL": "512",
    "CONCUR": "16",
    "CONTEXT_LEN": "8192",
    "TPU_COMPILE_TOKENS": "512 1024 2048 4096",
    "TPU_COMPILE_BS": "1 2 4 8 16",
    "MAX_TOTAL_TOKENS": "16384",
    "ENABLE_PROFILER_ENDPOINT": "0",
    # TP8 with DCP groups of 2: the eight ranks form four groups of adjacent
    # ranks, and each group owns a quarter of the KV heads -- two of gpt-oss's
    # eight -- with the two ranks in a group splitting the positions. Plain TP8
    # gives each rank one head over every position, so this is the same bytes
    # per rank re-partitioned: gpt-oss replicates no KV at TP8, so DCP buys it
    # no capacity and costs it two collectives per layer. This gate is here for
    # the mechanism, not for a win.
    "DCP_SIZE": str(DCP_SIZE),
    # The DCP backend requires RPA v3 (the only TPU kernel with the cache-only
    # and log-sum-exp modes the merge consumes) and torch.compile. Both are the
    # recipe's default; pinned so a recipe change cannot take the path this
    # gate exists to cover away from it.
    "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
    "ENABLE_TORCH_COMPILE": "1",
    # Prefix caching is what makes the probe's warm read a full cache hit, and
    # launch_server.sh defaults it off.
    "DISABLE_RADIX_CACHE": "0",
    # bf16 KV: the DCP span write puts each rank's own rows into the pool
    # itself rather than letting the kernel write them, and that path has only
    # been exercised at the cache's compute dtype. fp8 KV under DCP is a
    # separate gate to add once it has been measured.
    "KV_CACHE_DTYPE": "bfloat16",
    "EXTRA_ARGS": (
        "--reasoning-parser gpt-oss "
        f"--served-model-name {SERVED_MODEL_NAME} "
        "--tpu-compile-tracer dynamo "
        # One KV pool for every layer. gpt-oss is hybrid-SWA and DCP does carry
        # a second span write for the SWA sub-pool, but the single-pool shape
        # is the one DCP was measured against; the hybrid-SWA DCP path needs
        # its own gate rather than riding in on this one.
        "--disable-hybrid-swa-memory"
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-dcp-gsm8k",
    "variant": f"TP8+EP8+DCP{DCP_SIZE}",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-dcp-e2e",
    },
    # No dashboard key: this is feature validation at a config the accuracy
    # lane does not serve, so plotting its score beside that lane's would
    # compare two different servers.
    "gates": [
        {
            "dataset": "dcp",
            "label": "dcp",
            "runner": dcp_runner(
                test_name="gpt-oss-120b-dcp-gsm8k",
                dcp_size=DCP_SIZE,
            ),
        },
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "num_examples": 200,
            "num_threads": 16,
            "max_tokens": 512,
            "temperature": 0.0,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
