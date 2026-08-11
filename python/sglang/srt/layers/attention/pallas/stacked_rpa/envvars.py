# Ported from https://github.com/vllm-project/tpu-inference
# tpu_inference/kernels/experimental/stacked_rpa/envvars.py
# Commit 37ce4872252c5259d65fe6d818181a07a4646dac (2026-08-10)
import os


def stacked_rpa_bkv() -> int:
    return int(os.environ.get("STACKED_RPA_BKV", "0"))


def stacked_rpa_sw_bound() -> bool:
    return os.environ.get("STACKED_RPA_SW_BOUND", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
