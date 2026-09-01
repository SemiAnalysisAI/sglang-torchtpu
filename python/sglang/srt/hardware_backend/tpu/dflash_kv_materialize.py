"""Vectorized target-hidden to draft-KV materialization for TPU DFlash."""

from __future__ import annotations

import functools
import logging
import math
from bisect import bisect_left
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.speculative.dflash_utils import can_dflash_slice_qkv_weight

logger = logging.getLogger(__name__)


def _build_stacked_context_kv_params(
    layers: Iterable[nn.Module],
) -> dict | None:
    """Stack compatible draft-layer KV projection and K-norm parameters."""
    layers = list(layers)
    if not layers:
        return None

    attn0 = layers[0].self_attn
    eps = attn0.k_norm.variance_epsilon
    rotary = attn0.rotary_emb
    weights = []
    biases = []
    k_norm_weights = []
    for layer in layers:
        attn = layer.self_attn
        can_slice, _ = can_dflash_slice_qkv_weight(attn.qkv_proj)
        if not can_slice:
            return None
        if (
            attn.k_norm.variance_epsilon != eps
            or attn.kv_size != attn0.kv_size
            or attn.num_kv_heads != attn0.num_kv_heads
            or attn.head_dim != attn0.head_dim
        ):
            return None
        if attn.rotary_emb is not rotary and not torch.equal(
            attn.rotary_emb.cos_sin_cache, rotary.cos_sin_cache
        ):
            return None
        kv_slice = slice(attn.q_size, attn.q_size + 2 * attn.kv_size)
        weights.append(attn.qkv_proj.weight[kv_slice])
        biases.append(
            attn.qkv_proj.bias[kv_slice] if attn.qkv_proj.bias is not None else None
        )
        k_norm_weights.append(attn.k_norm.weight)

    has_bias = [bias is not None for bias in biases]
    if any(has_bias) and not all(has_bias):
        return None
    return {
        "weight": torch.cat(weights, dim=0).contiguous(),
        "bias": torch.cat(biases, dim=0).contiguous() if all(has_bias) else None,
        "k_norm_weight": torch.stack(k_norm_weights, dim=0).float(),
        "eps": eps,
        "num_layers": len(layers),
        "num_kv_heads": attn0.num_kv_heads,
        "head_dim": attn0.head_dim,
        "kv_size": attn0.kv_size,
        "rotary_emb": rotary,
    }


def _project_stacked_context_kv(
    ctx_hidden: torch.Tensor,
    positions: torch.Tensor,
    stacked: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project, K-normalize, and rotate all draft layers together."""
    tokens = ctx_hidden.shape[0]
    num_layers = stacked["num_layers"]
    num_kv_heads = stacked["num_kv_heads"]
    head_dim = stacked["head_dim"]
    kv_size = stacked["kv_size"]

    kv_all = F.linear(ctx_hidden, stacked["weight"], stacked["bias"])
    kv_all = kv_all.view(tokens, num_layers, 2, kv_size)
    k32 = (
        kv_all[:, :, 0, :]
        .reshape(tokens, num_layers, num_kv_heads, head_dim)
        .to(torch.float32)
    )
    variance = k32.pow(2).mean(dim=-1, keepdim=True)
    k32 = k32 * torch.rsqrt(variance + stacked["eps"])
    k32 = k32 * stacked["k_norm_weight"].view(1, num_layers, 1, head_dim)
    k_all = k32.to(ctx_hidden.dtype)

    k_flat = k_all.reshape(tokens, num_layers * kv_size)
    dummy_q = k_flat.new_empty(k_flat.shape)
    _, k_flat = stacked["rotary_emb"](positions, dummy_q, k_flat)
    k_all = (
        k_flat.view(tokens, num_layers, num_kv_heads, head_dim)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    v_all = (
        kv_all[:, :, 1, :]
        .view(tokens, num_layers, num_kv_heads, head_dim)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    return k_all, v_all


# Lazy JAX/Pallas imports keep CPU-only SGLang imports working.
_jax_import_error: BaseException | None = None
try:
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from torch_tpu._internal import pallas as ttpu_pallas
except Exception as error:  # noqa: BLE001  # pragma: no cover - optional TPU stack
    _jax_import_error = error
    jax = None  # type: ignore[assignment]
    pl = None  # type: ignore[assignment]
    pltpu = None  # type: ignore[assignment]
    ttpu_pallas = None  # type: ignore[assignment]


def _scatter_available() -> bool:
    return _jax_import_error is None


if _jax_import_error is None:

    def _scatter_kernel_body(
        num_chunks_ref,
        src_offsets_ref,
        dest_offsets_ref,
        valid_mask_ref,
        src_ref,
        dst_ref_in,
        dst_ref_out,
    ):
        del dst_ref_in

        def body(sem):

            @pl.loop(0, num_chunks_ref[0])
            def _start(index):

                @pl.when(valid_mask_ref[index] != 0)
                def _copy():
                    pltpu.make_async_copy(
                        src_ref.at[pl.ds(src_offsets_ref[index], 1)],
                        dst_ref_out.at[pl.ds(dest_offsets_ref[index], 1)],
                        sem,
                    ).start()

            @pl.loop(0, num_chunks_ref[0])
            def _wait(index):

                @pl.when(valid_mask_ref[index] != 0)
                def _copy():
                    pltpu.make_async_copy(
                        src_ref.at[pl.ds(src_offsets_ref[index], 1)],
                        dst_ref_out.at[pl.ds(dest_offsets_ref[index], 1)],
                        sem,
                    ).wait()

        pl.run_scoped(body, pltpu.SemaphoreType.DMA)

    def _jax_scatter_one_layer(
        num_chunks,
        src_offsets,
        dest_offsets,
        valid_mask,
        src,
        dst,
    ):
        return pl.pallas_call(
            _scatter_kernel_body,
            out_shape=jax.ShapeDtypeStruct(shape=dst.shape, dtype=dst.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                grid=(1,),
                num_scalar_prefetch=4,
                in_specs=[
                    pl.BlockSpec(memory_space=pl.ANY),
                    pl.BlockSpec(memory_space=pl.ANY),
                ],
                out_specs=pl.BlockSpec(memory_space=pl.ANY),
            ),
            input_output_aliases={5: 0},
        )(
            num_chunks,
            src_offsets,
            dest_offsets,
            valid_mask,
            src,
            dst,
        )

    # jax_op requires concrete annotation objects; postponed annotations would
    # otherwise turn inline ``jax.Array`` annotations into strings.
    _jax_scatter_one_layer.__annotations__ = {
        "num_chunks": jax.Array,
        "src_offsets": jax.Array,
        "dest_offsets": jax.Array,
        "valid_mask": jax.Array,
        "src": jax.Array,
        "dst": jax.Array,
        "return": jax.Array,
    }

    @functools.lru_cache(maxsize=1)
    def _make_scatter_one_layer_kernel():
        return ttpu_pallas.jax_op(
            "sglang_tpu_pallas::dflash_kv_scatter_one_layer",
            _jax_scatter_one_layer,
            donate_argnums=(5,),
        )

else:  # pragma: no cover - CPU-only import path
    _make_scatter_one_layer_kernel = None  # type: ignore[assignment]


def _multi_layer_scatter_into(
    srcs: list[torch.Tensor],
    dsts: list[torch.Tensor],
    dest_offsets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Scatter token rows into each layer cache through one fixed-schema op.

    ``dsts`` must be flattened token-major views that cover each entire cache
    allocation. Returned handles alias those donated destinations; callers must
    replace the old cache handles before the next attention invocation. A
    compiled caller statically unrolls the layer loop into one XLA graph.
    """
    if _jax_import_error is not None:
        raise RuntimeError(
            f"DFlash multi-layer TPU scatter is unavailable: {_jax_import_error!r}"
        )
    if len(srcs) != len(dsts):
        raise ValueError(f"src/dst layer mismatch: {len(srcs)} != {len(dsts)}")
    if not srcs:
        return []
    num_tokens = int(dest_offsets.numel())
    if num_tokens == 0:
        return list(dsts)
    device = dsts[0].device
    dest_offsets = dest_offsets.reshape(-1).to(device=device, dtype=torch.int32)
    if valid_mask is None:
        valid_mask = torch.ones((num_tokens,), dtype=torch.int32, device=device)
    else:
        valid_mask = valid_mask.reshape(-1).to(device=device, dtype=torch.int32)
        if int(valid_mask.numel()) != num_tokens:
            raise ValueError(
                "DFlash scatter valid-mask length mismatch: "
                f"{valid_mask.numel()} != {num_tokens}"
            )
    # Invalid rows are skipped by the Pallas predicate. Redirect their offset
    # to slot zero as an additional bounds-safety measure for speculative tails.
    dest_offsets = torch.where(
        valid_mask != 0, dest_offsets, torch.zeros_like(dest_offsets)
    )
    num_chunks = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    src_offsets = torch.arange(num_tokens, dtype=torch.int32, device=device)
    kernel = _make_scatter_one_layer_kernel()
    return [
        kernel(
            num_chunks,
            src_offsets,
            dest_offsets,
            valid_mask,
            src,
            dst,
        )
        for src, dst in zip(srcs, dsts)
    ]


def _pack_fused_kv_rows(
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    *,
    head_num: int,
    head_dim: int,
    padded_head_dim: int,
    num_kv_heads_x2: int,
    physical_tail: tuple[int, ...],
    store_dtype: torch.dtype,
) -> torch.Tensor:
    """Pack layer-major logical K/V into the RPA-v3 physical row layout."""
    num_layers, num_tokens = k_all.shape[:2]
    k_all = k_all.view(num_layers, num_tokens, head_num, head_dim)
    v_all = v_all.view(num_layers, num_tokens, head_num, head_dim)
    rows = torch.stack((k_all, v_all), dim=3).reshape(
        num_layers, num_tokens, head_num * 2, head_dim
    )
    dim_pad = padded_head_dim - head_dim
    head_pad = num_kv_heads_x2 - head_num * 2
    if dim_pad < 0 or head_pad < 0:
        raise ValueError(
            "DFlash logical KV exceeds TPU physical cache shape: "
            f"head_dim={head_dim}/{padded_head_dim}, "
            f"heads_x2={head_num * 2}/{num_kv_heads_x2}"
        )
    if dim_pad or head_pad:
        rows = F.pad(rows, (0, dim_pad, 0, head_pad))
    return rows.to(store_dtype).view(num_layers, num_tokens, *physical_tail)


def _select_prefill_token_bucket(
    num_tokens: int, token_buckets: tuple[int, ...]
) -> int | None:
    """Return the smallest configured static token bucket covering prefill."""
    bucket_index = bisect_left(token_buckets, num_tokens)
    return token_buckets[bucket_index] if bucket_index < len(token_buckets) else None


def _pad_prefill_materialize_inputs(
    target_hidden: torch.Tensor,
    positions: torch.Tensor,
    cache_loc: torch.Tensor,
    bucket_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable prefill inputs to one compile bucket plus a validity mask."""
    num_tokens = int(target_hidden.shape[0])
    if bucket_tokens < num_tokens:
        raise ValueError(
            f"DFlash prefill bucket {bucket_tokens} is smaller than {num_tokens} tokens"
        )
    pad_tokens = bucket_tokens - num_tokens
    if pad_tokens:
        target_hidden = F.pad(target_hidden, (0, 0, 0, pad_tokens))
        positions = F.pad(positions.reshape(-1), (0, pad_tokens))
        cache_loc = F.pad(cache_loc.reshape(-1), (0, pad_tokens))
    valid_mask = (
        torch.arange(bucket_tokens, dtype=torch.int32, device=target_hidden.device)
        < num_tokens
    )
    return target_hidden, positions, cache_loc, valid_mask


class TpuDFlashKVMaterializer:
    """One compiled target-to-KV graph with a donated cache update."""

    def __init__(
        self,
        *,
        draft_model,
        token_to_kv_pool,
        token_buckets: Iterable[int] | None = None,
    ) -> None:
        if not _scatter_available():
            raise RuntimeError(f"Pallas scatter import failed: {_jax_import_error!r}")
        if not draft_model.supports_fused_context_kv:
            raise ValueError("draft model requires per-layer context preparation")
        for layer_index, layer in enumerate(draft_model.layers):
            attention = layer.self_attn.attn
            for scale_name, scale in (
                ("k_scale", attention.k_scale),
                ("v_scale", attention.v_scale),
            ):
                if scale is not None and not math.isclose(float(scale), 1.0):
                    raise ValueError(
                        "non-unit cache scale is not supported by the TPU "
                        "vectorized writer: "
                        f"layer={layer_index}, {scale_name}={scale}"
                    )
        stacked = _build_stacked_context_kv_params(draft_model.layers)
        if stacked is None:
            raise ValueError("draft layers are not eligible for stacked context KV")
        pool = token_to_kv_pool
        if len(pool.kv_buffer) != stacked["num_layers"]:
            raise ValueError(
                "draft layer/cache count mismatch: "
                f"{stacked['num_layers']} != {len(pool.kv_buffer)}"
            )
        first_buffer = pool.kv_buffer[0]
        if first_buffer.ndim != 5 or any(
            tuple(buffer.shape[2:]) != tuple(first_buffer.shape[2:])
            for buffer in pool.kv_buffer
        ):
            raise ValueError("DFlash TPU fusion requires uniform 5D RPA-v3 buffers")
        if int(first_buffer.shape[1]) != int(pool.page_size) or int(
            first_buffer.shape[-1]
        ) != int(pool.padded_head_dim):
            raise ValueError(
                "DFlash TPU fusion requires the token-major RPA-v3 cache layout"
            )
        if pool.head_num != stacked["num_kv_heads"]:
            raise ValueError(
                f"draft KV head mismatch: {stacked['num_kv_heads']} != {pool.head_num}"
            )
        if pool.head_dim != stacked["head_dim"] or pool.v_head_dim != pool.head_dim:
            raise ValueError(
                "DFlash TPU fusion currently requires equal logical K/V head dims: "
                f"model={stacked['head_dim']}, pool_k={pool.head_dim}, "
                f"pool_v={pool.v_head_dim}"
            )
        self.stacked = stacked
        self.draft_model = draft_model
        self.pool = pool
        self.num_layers = stacked["num_layers"]
        self.token_buckets = tuple(
            sorted({int(bucket) for bucket in token_buckets or ()})
        )
        if any(bucket <= 0 for bucket in self.token_buckets):
            raise ValueError(
                f"DFlash prefill token buckets must be positive: {self.token_buckets}"
            )
        self.physical_tail = tuple(first_buffer.shape[2:])
        self.scatter_kernel = _make_scatter_one_layer_kernel()
        # The fixed-schema jax_op is called once per layer. Dynamo statically
        # unrolls that model-sized loop so projection, norm, RoPE, packing, and
        # every donated scatter remain in one full XLA graph.
        self.compiled_materialize = torch.compile(
            self._materialize_tensors,
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )

    def _materialize_tensors(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        valid_mask: torch.Tensor,
        dsts: list[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        ctx_hidden = self.draft_model.project_target_hidden(target_hidden)
        k_all, v_all = _project_stacked_context_kv(
            ctx_hidden=ctx_hidden,
            positions=positions,
            stacked=self.stacked,
        )
        rows = _pack_fused_kv_rows(
            k_all=k_all,
            v_all=v_all,
            head_num=self.pool.head_num,
            head_dim=self.pool.head_dim,
            padded_head_dim=self.pool.padded_head_dim,
            num_kv_heads_x2=self.pool.num_kv_heads_x2,
            physical_tail=self.physical_tail,
            store_dtype=self.pool.store_dtype,
        )

        num_tokens = cache_loc.numel()
        cache_loc = cache_loc.reshape(-1).to(dtype=torch.int32)
        valid_mask = valid_mask.reshape(-1).to(dtype=torch.int32)
        cache_loc = torch.where(valid_mask != 0, cache_loc, torch.zeros_like(cache_loc))
        num_chunks = torch.tensor(
            [num_tokens], dtype=torch.int32, device=target_hidden.device
        )
        src_offsets = torch.arange(
            num_tokens, dtype=torch.int32, device=target_hidden.device
        )
        return tuple(
            self.scatter_kernel(
                num_chunks,
                src_offsets,
                cache_loc,
                valid_mask,
                src,
                dst,
            )
            for src, dst in zip(rows.unbind(0), dsts)
        )

    def materialize(
        self,
        *,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        num_tokens = int(target_hidden.shape[0])
        if num_tokens == 0:
            return
        if int(cache_loc.numel()) != num_tokens:
            raise ValueError(
                f"DFlash cache/token length mismatch: {cache_loc.numel()} != {num_tokens}"
            )
        old_shapes = [tuple(buffer.shape) for buffer in self.pool.kv_buffer]
        dsts = [buffer.flatten(0, 1) for buffer in self.pool.kv_buffer]
        materialize_tensors = (
            self._materialize_tensors
            if torch.compiler.is_compiling()
            else self.compiled_materialize
        )
        bucket_tokens = (
            _select_prefill_token_bucket(
                num_tokens=num_tokens,
                token_buckets=self.token_buckets,
            )
            if valid_mask is None
            else None
        )
        if bucket_tokens is not None:
            (
                target_hidden,
                positions,
                cache_loc,
                valid_mask,
            ) = _pad_prefill_materialize_inputs(
                target_hidden=target_hidden,
                positions=positions,
                cache_loc=cache_loc,
                bucket_tokens=bucket_tokens,
            )
            new_dsts = list(
                materialize_tensors(
                    target_hidden,
                    positions,
                    cache_loc,
                    valid_mask,
                    dsts,
                )
            )
        elif valid_mask is not None:
            new_dsts = list(
                materialize_tensors(
                    target_hidden,
                    positions,
                    cache_loc,
                    valid_mask,
                    dsts,
                )
            )
        else:
            ctx_hidden = self.draft_model.project_target_hidden(target_hidden)
            k_all, v_all = _project_stacked_context_kv(
                ctx_hidden=ctx_hidden,
                positions=positions,
                stacked=self.stacked,
            )
            rows = _pack_fused_kv_rows(
                k_all=k_all,
                v_all=v_all,
                head_num=self.pool.head_num,
                head_dim=self.pool.head_dim,
                padded_head_dim=self.pool.padded_head_dim,
                num_kv_heads_x2=self.pool.num_kv_heads_x2,
                physical_tail=self.physical_tail,
                store_dtype=self.pool.store_dtype,
            )
            new_dsts = _multi_layer_scatter_into(
                srcs=list(rows.unbind(0)),
                dsts=dsts,
                dest_offsets=cache_loc,
                valid_mask=valid_mask,
            )
        # Donation keeps the physical allocations but invalidates the old
        # Python handles. Adopt the returned aliases before draft attention.
        self.pool.set_state(
            [
                new_dst.view(old_shape)
                for new_dst, old_shape in zip(new_dsts, old_shapes)
            ]
        )
