"""Cookbook-parity overrides: c <= 16 serving and simulated acceptance 3 (the B200 low-latency run)."""

CONTEXT_LENGTH = 10240
CONCURRENCY = 16
MAX_RUNNING = CONCURRENCY + 2


def apply(config):
    config["server_args"].update(
        {
            "max_running_requests": MAX_RUNNING,
            "torch_compile_max_bs": MAX_RUNNING,
            # Only the measured batch sizes; ramp-up batches pad to 16. Range ratio 1
            # makes every prompt exactly 8192 tokens, so one prefill bucket suffices.
            "tpu_compile_bs": [1, 16],
            "tpu_compile_tokens": [8192],
            "max_total_tokens": MAX_RUNNING * CONTEXT_LENGTH,
            "max_mamba_cache_size": MAX_RUNNING + 2,
        }
    )
    config["runtime_env"].update(
        {"SGLANG_SIMULATE_ACC_LEN": "3", "SGLANG_SIMULATE_ACC_METHOD": "match-expected"}
    )
    return config
