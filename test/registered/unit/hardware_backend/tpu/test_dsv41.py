"""Numerical and ownership contracts for the V4.1 TPU integration."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.dsv41.kernels.cache_rows import scatter_index_keys
from sglang_tpu.dsv41.kernels.csa2_cache import encode_csa2_kv
from sglang_tpu.dsv41.kernels.index_key_layout import gather_records, to_ordinary
from sglang_tpu.dsv41.kernels.mhc import delayed_post_pre_flat_jax, delayed_pre_flat_jax
from sglang_tpu.dsv41.mhc import pack_mix_weight
from sglang_tpu.dsv41.ops import hc_combine, pair_compress, round_index_fp8

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[5]
_spec = importlib.util.spec_from_file_location(
    "dsv41_hooks", ROOT / "sglang_tpu/hooks/dsv41.py"
)
_hooks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hooks)
cache_bytes, hook_specs = _hooks.cache_bytes, _hooks.hook_specs


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("index_page_size", [64, 128])
def test_index_pool_supports_full_page_score_reads(ratio, index_page_size):
    tree = ast.parse((ROOT / "sglang_tpu/dsv41/memory_pool.py").read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {"BF16IndexHalfplanePool", "DSV41TPUTokenToKVPool"}
    ]
    namespace = {"torch": torch, "DSV4TPUTokenToKVPool": object}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "memory_pool.py", "exec"),
        namespace,
    )
    owner = namespace["DSV41TPUTokenToKVPool"]()
    owner.page_size = 256
    size = (16384 + owner.page_size) // ratio
    pool = owner._make_indexer_pool(
        size, index_page_size, torch.bfloat16, 128, 2, "cpu", False, force_fp4=True
    )
    for bank in pool.k_buffer:
        assert bank.dtype == torch.int32 and bank.shape[1] == 128
        assert bank.shape[0] % (owner.page_size // ratio // 2) == 0
        assert bank.shape[0] * 2 >= size + index_page_size
        assert bank.shape[0] * 2 == pool.num_pages * pool.page_size
        assert torch.count_nonzero(bank) == 0
    assert pool.size == size and pool.page_size == index_page_size
    assert pool.get_kv_size_bytes() == (sum(bank.nbytes for bank in pool.k_buffer), 0)
    for view, bank in zip(pool.contiguous_page_row_buffers(), pool.k_buffer):
        assert view.shape == (pool.num_pages, index_page_size // 2 * 128)
        assert view.data_ptr() == bank.data_ptr()


@pytest.mark.parametrize("chunks", [(8,), (1, 3, 1, 3), (3, 2, 3), (1,) * 8])
def test_c2_chunk_boundaries(chunks):
    torch.manual_seed(14)
    kv, score = torch.randn(8, 32), torch.randn(8, 32)
    expected = (kv.view(4, 2, 32) * score.view(4, 2, 32).softmax(1)).sum(1)
    state = torch.zeros(7, 64)
    state[:, 32:] = float("-inf")
    actual, start = [], 0
    for size in chunks:
        positions = torch.arange(start, start + size)
        out = pair_compress(
            kv[start : start + size],
            score[start : start + size],
            torch.ones(size, dtype=torch.int32),
            positions,
            torch.ones(size, dtype=torch.bool),
            state,
            2,
        )
        actual.append(out[positions % 2 == 1])
        start += size
    torch.testing.assert_close(torch.cat(actual), expected)


def test_c2_batch_reorder_and_padding_do_not_overwrite_live_state():
    state = torch.zeros(7, 2)
    state[:, 1] = float("-inf")
    pair_compress(
        torch.tensor([[2.0], [7.0], [999.0]]),
        torch.zeros(3, 1),
        torch.tensor([0, 1, 0]),
        torch.tensor([0, 0, 0]),
        torch.tensor([True, True, False]),
        state,
        2,
    )
    out = pair_compress(
        torch.tensor([[9.0], [4.0]]),
        torch.zeros(2, 1),
        torch.tensor([1, 0]),
        torch.tensor([1, 1]),
        torch.tensor([True, True]),
        state,
        2,
    )
    torch.testing.assert_close(out, torch.tensor([[8.0], [3.0]]))


def test_fullgraph_c2_state_mutation():
    compiled = torch.compile(pair_compress, backend="eager", fullgraph=True)
    state = torch.zeros(5, 4)
    state[:, 2:] = float("-inf")
    for position, value in enumerate([2.0, 4.0, 8.0, 10.0]):
        out = compiled(
            torch.full((1, 2), value),
            torch.zeros(1, 2),
            torch.zeros(1, dtype=torch.int64),
            torch.tensor([position]),
            torch.ones(1, dtype=torch.bool),
            state,
            2,
        )
        if position % 2:
            torch.testing.assert_close(out, torch.full((1, 2), value - 1))


def test_hc_combine_and_low_ratio_storage_accounting():
    x = torch.arange(24, dtype=torch.float32).view(2, 12)
    pre = torch.tensor([[1.0, 2.0, 3.0], [0.2, 0.3, 0.4]])
    expected = sum(x[:, i * 4 : (i + 1) * 4] * pre[:, i, None] for i in range(3))
    torch.testing.assert_close(hc_combine(x, pre, 3, torch.float32), expected)
    cfg = SimpleNamespace(
        _is_tpu_dsv4=True,
        attn_head_dim=512,
        indexer_head_dim=128,
        stage_owner_layers={1: [1, 3], 2: [2, 4, 6]},
    )
    assert cache_bytes(100.0, cfg) == 100.0 + 772 * (2 + 3 / 2)


def test_hook_targets_exist():
    for target, _, _ in hook_specs():
        parts = target.split(".")
        for split in range(len(parts) - 1, 1, -1):
            path = ROOT / "python" / Path(*parts[:split]).with_suffix(".py")
            if path.exists():
                tree = ast.parse(path.read_text())
                for name in parts[split:]:
                    tree = next(
                        node
                        for node in tree.body
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                        and node.name == name
                    )
                break
        else:
            pytest.fail(f"Missing hook target {target}")


def test_index_halfplanes_preserve_sibling_and_drop_padding():
    rng = np.random.default_rng(7)
    values = jnp.asarray(rng.standard_normal((6, 128)), dtype=jnp.bfloat16)
    pool = jnp.zeros((4, 128), jnp.int32)
    pool = scatter_index_keys(pool, jnp.array([0, 1, 2], jnp.int32), values[:3])
    # Odd record 3 must not overwrite its even sibling 2. Invalid record -1
    # must not write owner zero or make an earlier value visible to padding.
    pool = scatter_index_keys(pool, jnp.array([3, -1, 7], jnp.int32), values[3:])
    ids = jnp.array([0, 1, 2, 3, 7, -1], jnp.int32)
    actual = np.asarray(to_ordinary(gather_records(pool, ids))).astype(np.float32)
    expected = np.asarray(values)[[0, 1, 2, 3, 5, 4]].astype(np.float32)
    expected[-1] = 0
    np.testing.assert_array_equal(actual, expected)


def test_csa2_quantization_has_one_scale_for_all_512_features():
    rng = np.random.default_rng(9)
    latent = jnp.asarray(rng.standard_normal((3, 512)), dtype=jnp.bfloat16)
    words, scales = encode_csa2_kv(latent)
    bits = np.asarray(words).astype(np.uint32)
    octets = np.stack([(bits >> (8 * p)) & 255 for p in range(4)], axis=1)
    import ml_dtypes

    actual = octets.astype(np.uint8).view(ml_dtypes.float8_e4m3fn).reshape(3, 512)
    x = np.asarray(latent).astype(np.float32)
    expected_scales = np.maximum(np.abs(x).max(1), 1e-4) / 448
    expected = (x / expected_scales[:, None]).astype(ml_dtypes.float8_e4m3fn)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(scales, expected_scales, rtol=1e-7)


def test_index_fp8_uses_power_of_two_scale_per_head():
    values = torch.tensor([0.25, 0.75, -1.25, 1.75, 448.0, 449.0])
    values = values.repeat(2, 2, 22)[..., :128].bfloat16()
    actual = round_index_fp8(values)
    # The row maximum is 448 in BF16, giving scale 1. This differs from FP4.
    expected = values.float().clamp(-448, 448).to(torch.float8_e4m3fn).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        round_index_fp8(torch.zeros_like(values)), torch.zeros_like(values)
    )


def test_shifted_mhc_carries_previous_gate_and_commits_before_engram():
    rng = np.random.default_rng(12)
    tokens, mult, dim = 3, 4, 32
    flat = jnp.asarray(rng.standard_normal((tokens, mult * dim)), jnp.bfloat16)
    fn = torch.from_numpy(
        rng.standard_normal((24, mult * dim)).astype(np.float32) * 0.05
    )
    packed = pack_mix_weight(fn)
    packed_jax = jnp.asarray(packed.float().numpy(), jnp.bfloat16)
    parts = packed[:, :72].float().split(24, dim=1)
    torch.testing.assert_close(sum(parts).T, fn, rtol=0, atol=0)
    scale, base = jnp.ones(3, jnp.float32), jnp.zeros(24, jnp.float32)
    previous = jnp.asarray([[1, 0, 0, 0]] * tokens, jnp.float32)
    constants = (1e-6, 1e-6, 1e-6, 2.0, 20)
    post, comb, value, new_pre = delayed_pre_flat_jax(
        flat, packed_jax, scale, base, previous, *constants
    )
    np.testing.assert_array_equal(value, flat[:, :dim])
    sublayer = jnp.asarray(rng.standard_normal((tokens, dim)), jnp.bfloat16)
    merged, next_post, next_comb, next_value, next_pre = delayed_post_pre_flat_jax(
        sublayer, flat, post, comb, packed_jax, scale, base, new_pre, *constants
    )
    streams = np.asarray(flat).astype(np.float32).reshape(tokens, mult, dim)
    expected = np.asarray(post) * np.asarray(sublayer).astype(np.float32)[:, None, :]
    for i in range(mult):
        expected += np.asarray(comb)[:, i, :, None] * streams[:, i, None, :]
    expected = np.asarray(jnp.asarray(expected, jnp.bfloat16)).astype(np.float32)
    np.testing.assert_array_equal(
        np.asarray(merged).astype(np.float32).reshape(expected.shape), expected
    )
    collapsed = (expected * np.asarray(new_pre)[:, :, None]).sum(1)
    np.testing.assert_allclose(
        np.asarray(next_value).astype(np.float32), collapsed, rtol=0.008, atol=0.008
    )
    # Engram consumes this committed residual. A later start retains new_pre,
    # and does not substitute the gate computed by the following sublayer.
    changed = (merged.astype(jnp.float32) + 0.5).astype(jnp.bfloat16)
    _, _, after_engram, _ = delayed_pre_flat_jax(
        changed, packed_jax, scale, base, new_pre, *constants
    )
    expected_engram = (
        np.asarray(changed).astype(np.float32).reshape(tokens, mult, dim)
        * np.asarray(new_pre)[:, :, None]
    ).sum(1)
    np.testing.assert_allclose(
        np.asarray(after_engram).astype(np.float32),
        expected_engram,
        rtol=0.008,
        atol=0.008,
    )


def test_epic_fp4_ties_to_even_and_native_n_packing():
    import jax

    from sglang_tpu.dsv41.quantization import _quantizer

    values = np.zeros((1, 512, 2), np.float32)
    # A peak of six gives a block scale of one. The .75 tie must round to
    # 1.0, not to the old SGLang bucketize path's 0.5.
    values[:, 0, :] = 6
    values[:, 1:8, :] = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])[None, :, None]
    packed, scale = _quantizer()(values)
    assert packed.shape == (1, 512, 1)
    decoded = jax.lax.bitcast_convert_type(packed, jnp.float4_e2m1fn).reshape(1, 512, 2)
    np.testing.assert_array_equal(
        np.asarray(decoded[0, 1:8, 0]).astype(np.float32), [0, 1, 1, 2, 2, 4, 4]
    )
    np.testing.assert_array_equal(scale, np.ones((1, 1, 2), np.float32))


def test_cpu_export_patches_common_and_mosaic_tpu_info_and_restores():
    from unittest.mock import patch

    from jax._src.pallas.mosaic import tpu_info as mosaic_info
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    try:
        from jax._src import tpu_info as common_info
    except ImportError:
        common_info = mosaic_info
    modules = (mosaic_info, common_info, pltpu)
    previous = tuple(module.get_tpu_info for module in modules)
    with patch(
        "sglang_tpu.pallas_export.get_pallas_tpu_device_kind", return_value="TPU7x"
    ):
        with tpu_info_for_cpu_export():
            infos = [module.get_tpu_info() for module in modules]
            assert all(info is infos[0] for info in infos)
            assert infos[0].generation == 7
        assert tuple(module.get_tpu_info for module in modules) == previous


def test_signed_sqrt_preserves_negative_zero_and_clamped_signs():
    from sglang_tpu.dsv41.ops import signed_sqrt

    dot = torch.tensor([0.0, -0.0, 1e-8, -1e-8, 4.0, -4.0, float("inf"), -float("inf")])
    expected = torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)
    actual = signed_sqrt(dot, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_engram_gate_matches_core_per_copy_normalization(dtype):
    from sglang_tpu.dsv41.ops import engram_gate

    path = ROOT / "python/sglang/srt/layers/engram.py"
    tree = ast.parse(path.read_text())
    node = next(
        node for node in tree.body if getattr(node, "name", None) == "engram_gate"
    )
    namespace = {"torch": torch, "_cuda_kernels": lambda _: False}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    core_gate = namespace["engram_gate"]
    torch.manual_seed(917)
    x = torch.randn(9, 4, 128).to(dtype)
    kv = torch.randn(9, 5 * 128).to(dtype)
    x[0] = 0
    kv[1, : 4 * 128] = 0
    q_weight = torch.randn(4, 128).to(dtype)
    k_weight = torch.randn(4, 128).to(dtype)
    args = (x, kv, q_weight, k_weight, 1e-6, 1e-6)
    expected = core_gate(*args)
    torch.testing.assert_close(engram_gate(*args), expected, rtol=0, atol=0)
    compiled = torch.compile(engram_gate, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(*args), expected, rtol=0, atol=0)
