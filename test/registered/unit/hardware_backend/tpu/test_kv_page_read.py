"""Unit tests for the logical TPU KV page gather compatibility helper."""

import numpy as np
import pytest
import torch
from sglang.srt.hardware_backend.tpu.kv_page_read import read_pages, read_pages_torch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_read_pages_torch_selects_cpu_pages_in_requested_order():
    kv = torch.arange(5 * 2 * 3, dtype=torch.int32).reshape(5, 2, 3)
    original = kv.clone()

    actual = read_pages_torch(torch.tensor([3, 1, 3], dtype=torch.int32), kv)

    assert torch.equal(actual, kv[[3, 1, 3]])
    assert torch.equal(kv, original)


def test_read_pages_torch_preserves_empty_shape_dtype_and_device():
    kv = torch.zeros((5, 2, 3), dtype=torch.bfloat16)

    actual = read_pages_torch(torch.empty((0,), dtype=torch.int32), kv)

    assert actual.shape == (0, 2, 3)
    assert actual.dtype == kv.dtype
    assert actual.device == kv.device


def test_read_pages_interpret_gathers_logical_values():
    jnp = pytest.importorskip("jax.numpy")
    kv = jnp.arange(5 * 2 * 3, dtype=jnp.int32).reshape(5, 2, 3)
    block_ids = jnp.asarray([4, 0, 2], dtype=jnp.int32)

    actual = read_pages(block_ids, kv, interpret=True)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(kv)[[4, 0, 2]])
