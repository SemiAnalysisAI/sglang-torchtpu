# SPDX-License-Identifier: Apache-2.0
"""Minimal model-free repro for E4M3FN corruption in TPU Pallas RPA.

This injects exact finite float8_e4m3fn byte encodings into the single-token
RPA custom op, then compares the input byte with the byte persisted in the
aliased KV cache. It requires a TPU, torch_tpu, JAX/Pallas, and an SGLang
checkout containing the TPU RPA kernel. It does not require a model checkpoint
or a running server.

Run from the SGLang checkout:

  SGLANG_PLATFORM=tpu \
  PYTHONPATH="$PWD/python" \
  python3 test/manual/tpu/repro_pallas_e4m3fn_rpa_cache_write.py
"""

from __future__ import annotations

import json
import math
import os

os.environ.setdefault("SGLANG_PLATFORM", "tpu")

import torch
import torch_tpu  # noqa: F401

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    _build_pallas_kernel_op,
)
from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_v3 import RPA_V3_SPEC
from sglang.srt.hardware_backend.tpu.patches import apply_tpu_patches

HEAD_DIM = 128
BYTE_INDEX = 50
PAGE_SIZE = 64


def fp8_tensor_with_byte(byte: int) -> torch.Tensor:
    raw = torch.zeros((1, 1, HEAD_DIM), dtype=torch.uint8)
    raw[0, 0, BYTE_INDEX] = byte
    return raw.view(torch.float8_e4m3fn).to("tpu")


def run_one(op, byte: int) -> dict[str, object]:
    q = torch.zeros((1, 2, HEAD_DIM), dtype=torch.bfloat16, device="tpu")
    k = fp8_tensor_with_byte(byte)
    v = fp8_tensor_with_byte(0x00)
    cache = torch.zeros(
        (1, PAGE_SIZE, 1, 4, HEAD_DIM),
        dtype=torch.uint8,
        device="tpu",
    ).view(torch.float8_e4m3fn)

    # Confirm that the TorchTPU tensor still contains the intended byte before
    # entering Pallas. This rules out host-to-device transfer and the FP8 cast.
    pre_byte = int(k.cpu().view(torch.uint8)[0, 0, BYTE_INDEX])
    if pre_byte != byte:
        raise AssertionError(
            f"input changed before Pallas: expected 0x{byte:02x}, got 0x{pre_byte:02x}"
        )

    out = op(
        q,
        k,
        v,
        cache,
        torch.tensor([1], dtype=torch.int32, device="tpu"),  # kv_lens
        torch.tensor([0], dtype=torch.int32, device="tpu"),  # page_indices
        torch.tensor([0, 1], dtype=torch.int32, device="tpu"),  # cu_q_lens
        torch.tensor([0, PAGE_SIZE], dtype=torch.int32, device="tpu"),
        torch.tensor([1, 1, 1], dtype=torch.int32, device="tpu"),
        None,
        1.0 / math.sqrt(HEAD_DIM),
        None,
        None,
        1.0,
        1.0,
    )
    out.cpu()  # Synchronize the custom op before reading the aliased cache.

    persisted = cache.cpu()
    post_byte = int(persisted.view(torch.uint8)[0, 0, 0, 0, BYTE_INDEX])
    post_value = float(persisted[0, 0, 0, 0, BYTE_INDEX].float())
    return {
        "input_byte": f"0x{byte:02x}",
        "input_exponent": f"0x{(byte >> 3) & 0xF:x}",
        "persisted_byte": f"0x{post_byte:02x}",
        "persisted_value": post_value,
        "persisted_is_nan": math.isnan(post_value),
    }


def main() -> None:
    apply_tpu_patches()
    op = _build_pallas_kernel_op(RPA_V3_SPEC)

    # Raw-byte injection intentionally bypasses SGLang's +/-240 workaround so
    # the underlying Pallas behavior remains observable on patched source.
    # 0x77/+240 and 0xf7/-240 are the largest safe boundary encodings.
    # 0x78..0x7e and 0xf8..0xfe are finite E4M3FN encodings with exponent 0xf.
    test_bytes = [0x77, *range(0x78, 0x7F), 0xF7, *range(0xF8, 0xFF)]
    rows = [run_one(op, byte) for byte in test_bytes]
    for row in rows:
        print(json.dumps(row, allow_nan=True, sort_keys=True))

    by_input = {int(row["input_byte"], 16): row for row in rows}
    safe_preserved = all(
        by_input[byte]["persisted_byte"] == f"0x{byte:02x}" for byte in (0x77, 0xF7)
    )
    unsafe_became_nan = all(
        bool(by_input[byte]["persisted_is_nan"])
        for byte in [*range(0x78, 0x7F), *range(0xF8, 0xFF)]
    )
    summary = {
        "safe_boundary_preserved": safe_preserved,
        "exponent_0xf_became_nan": unsafe_became_nan,
        "reproduced": safe_preserved and unsafe_became_nan,
    }
    print("SUMMARY=" + json.dumps(summary, sort_keys=True))
    if not summary["reproduced"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
