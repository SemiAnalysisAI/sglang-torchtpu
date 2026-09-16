# SPDX-License-Identifier: Apache-2.0
"""MXINT8 storage geometry, independent of backend and kernel implementations."""

import torch

GROUP_SIZE = 32
EXPONENT_BIAS = 127
EXPONENT_DTYPE = torch.uint8
QUANT_DTYPE = torch.int8
ROW_ALIGNMENT = 32


def _validate_geometry(
    head_num: int, head_dim: int, page_size: int, group_size: int = GROUP_SIZE
) -> None:
    if group_size not in (16, 32, 64):
        raise ValueError("MXINT8 group_size must be 16, 32, or 64")
    if head_num <= 0 or head_dim <= 0 or head_dim % group_size:
        raise ValueError(
            "MXINT8 requires positive heads and head_dim divisible by group_size"
        )
    if group_size != GROUP_SIZE and head_dim not in (64, 128):
        raise ValueError("MXINT8 group_size 16/64 requires head_dim 64 or 128")
    if page_size < 128 or page_size & (page_size - 1):
        raise ValueError("MXINT8 page_size must be a power of two and at least 128")


def exponent_cache_shape(
    num_blocks: int,
    head_num: int,
    head_dim: int,
    page_size: int,
    group_size: int = GROUP_SIZE,
) -> tuple[int, int, int]:
    """[pages, padded rows, tokens]; row = (2 * head + K0V1) * G + group."""
    _validate_geometry(head_num, head_dim, page_size, group_size)
    if num_blocks <= 0:
        raise ValueError("MXINT8 num_blocks must be positive (including dummy page)")
    rows = 2 * head_num * (head_dim // group_size)
    rows = (rows + ROW_ALIGNMENT - 1) // ROW_ALIGNMENT * ROW_ALIGNMENT
    return num_blocks, rows, page_size


def cache_bytes_per_token(
    head_num: int, head_dim: int, page_size: int, group_size: int = GROUP_SIZE
) -> int:
    """Physical bytes per layer, excluding the extra dummy page's token count."""
    _, rows, _ = exponent_cache_shape(1, head_num, head_dim, page_size, group_size)
    aligned_dim = (head_dim + ROW_ALIGNMENT - 1) // ROW_ALIGNMENT * ROW_ALIGNMENT
    return 2 * head_num * aligned_dim + rows
