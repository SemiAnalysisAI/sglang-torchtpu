"""Shared settings for TPU launch recipes."""

CONFIG = {
    "server_args": {
        "device": "tpu",
        "dtype": "bfloat16",
        "host": "127.0.0.1",
        "port": 30000,
        "grammar-backend": "none",
    },
    "sglang_env": {"SGLANG_PLATFORM": "tpu"},
    "torch_tpu_env": {},
    "runtime_env": {},
}
