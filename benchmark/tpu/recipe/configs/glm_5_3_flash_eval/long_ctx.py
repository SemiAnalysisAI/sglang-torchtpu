"""Long-context eval overrides for the GLM-5.3-Flash recipes (AIME-style reasoning, c <= 8)."""

CONTEXT_LENGTH = 65536
CONCURRENCY = 8
MAX_RUNNING = CONCURRENCY + 2


def apply(config):
    config["server_args"].update(
        {
            "context_length": CONTEXT_LENGTH,
            "max_running_requests": MAX_RUNNING,
            "torch_compile_max_bs": MAX_RUNNING,
            "tpu_compile_bs": [1, 2, 4, 8],
            "max_total_tokens": MAX_RUNNING * CONTEXT_LENGTH,
            "max_mamba_cache_size": MAX_RUNNING + 2,
            "reasoning_parser": "glm45",
        }
    )
    # Attention stays the recipe's DSA; never raise the dense limit here.
    return config
