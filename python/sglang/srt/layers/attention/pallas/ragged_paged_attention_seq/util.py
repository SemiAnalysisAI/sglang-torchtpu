"""Utility functions for sequence-along-lane ragged paged attention."""

import jax.numpy as jnp


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    """Return a dtype's bit width across supported JAX versions."""
    try:
        from jax._src import dtypes

        if hasattr(dtypes, "bit_width"):
            return dtypes.bit_width(dtype)
    except (ImportError, AttributeError):
        pass

    return jnp.dtype(dtype).itemsize * 8


def get_dtype_packing(dtype):
    return 32 // get_dtype_bitwidth(dtype)


def next_power_of_2(x: int):
    """Return the smallest power of two greater than or equal to ``x``."""
    assert x > 0
    if x == 1:
        return 1
    return 1 << (x - 1).bit_length()
