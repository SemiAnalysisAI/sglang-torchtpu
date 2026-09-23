"""Shared settings for TPU launch recipes."""

CONFIG = {
    "server_args": {
        "device": "tpu",
        "dtype": "bfloat16",
        "host": "127.0.0.1",
        "port": 30000,
        "grammar_backend": "none",
    },
    "runtime_env": {"SGLANG_PLATFORM": "tpu"},
}
