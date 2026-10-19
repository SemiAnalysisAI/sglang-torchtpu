# SPDX-License-Identifier: Apache-2.0
"""Row scatter into the fused TPU KV pool for the replicated-pool prefill.

``kv_write_rows(block_ids, row_dst, rows, kv)``: for the pages ``kv[block_ids]``
(read with the aliased page kernel), overwrite token row ``row_dst[i]`` of the
flattened page buffer with ``rows[i]`` and write the pages back in place.
``rows`` is already in the pool dtype, laid out like the kernel's new-KV
packing (``[k_h, v_h]`` per head, flattened), so no dtype reinterpretation
happens anywhere: a torch dtype view of fp8, a whole-pool view or an 8-bit
bitcast on TPU all lower to copies of the layer or element-wise byte slices
(100+ ms per step). JAX is imported only when the op is built.
"""

from __future__ import annotations

import torch

_OPS: dict = {}


def _build():
    import jax
    import jax.numpy as jnp
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang.srt.hardware_backend.tpu.kv_page_write import read_pages, write_pages
    from sglang.srt.hardware_backend.tpu.pallas_export import tpu_info_for_cpu_export

    def write_rows(block_ids, row_dst, rows, kv):
        n = block_ids.shape[0]
        page = kv.shape[1]
        pages = read_pages(block_ids, kv)  # [n, page, ...] pool dtype
        flat = pages.reshape(n * page, -1)
        flat = flat.at[row_dst].set(rows.astype(flat.dtype))
        return write_pages(block_ids, flat.reshape(n, *kv.shape[1:]), kv)

    rows_kernel = _torch_pallas.custom_jax_kernel(
        write_rows,
        name="jax_kv_write_rows",
        donate_argnums=(3,),
        input_output_aliases={3: 0},
        warn_deprecated=False,
    )

    @torch.library.custom_op("sglang_tpu::kv_write_rows", mutates_args=("kv",), device_types=["tpu"])
    def kv_write_rows(
        block_ids: torch.Tensor, row_dst: torch.Tensor, rows: torch.Tensor, kv: torch.Tensor
    ) -> None:
        with tpu_info_for_cpu_export():
            rows_kernel(block_ids, row_dst, rows, kv)

    @kv_write_rows.register_fake
    def _(block_ids: torch.Tensor, row_dst: torch.Tensor, rows: torch.Tensor, kv: torch.Tensor) -> None:
        return None

    _OPS["write_rows"] = kv_write_rows

    # ---- page-granular variant (layout aligned to the pool pages) --------
    def write_pages_from_rows(block_ids, page_src, boundary_pos, boundary_lead, rows, kv):
        """kv[block_ids[i]] = rows.pages[page_src[i]], except that page
        block_ids[boundary_pos[j]] keeps its first boundary_lead[j] rows."""
        page = kv.shape[1]
        n = block_ids.shape[0]
        pages = rows.astype(kv.dtype).reshape(-1, page, rows.shape[-1])[page_src]  # [n, page, E]
        keep = read_pages(block_ids[boundary_pos], kv).reshape(boundary_pos.shape[0], page, -1)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, (boundary_pos.shape[0], page, 1), 1)
        blended = jnp.where(row_ids < boundary_lead[:, None, None], keep, pages[boundary_pos])
        pages = pages.at[boundary_pos].set(blended)
        return write_pages(block_ids, pages.reshape(n, *kv.shape[1:]), kv)

    pages_kernel = _torch_pallas.custom_jax_kernel(
        write_pages_from_rows,
        name="jax_kv_write_pages_from_rows",
        donate_argnums=(5,),
        input_output_aliases={5: 0},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        "sglang_tpu::kv_write_pages_from_rows", mutates_args=("kv",), device_types=["tpu"]
    )
    def kv_write_pages_from_rows(
        block_ids: torch.Tensor,
        page_src: torch.Tensor,
        boundary_pos: torch.Tensor,
        boundary_lead: torch.Tensor,
        rows: torch.Tensor,
        kv: torch.Tensor,
    ) -> None:
        with tpu_info_for_cpu_export():
            pages_kernel(block_ids, page_src, boundary_pos, boundary_lead, rows, kv)

    @kv_write_pages_from_rows.register_fake
    def _(block_ids, page_src, boundary_pos, boundary_lead, rows, kv) -> None:
        return None

    _OPS["write_pages_from_rows"] = kv_write_pages_from_rows


def ensure_kv_row_ops():
    """Build the ops once (outside torch.compile); returns the registry."""
    if not _OPS:
        _build()
        from sglang.srt.hardware_backend.tpu.kv_page_write import (
            get_read_pages_op,
            get_write_pages_op,
        )

        get_read_pages_op()
        get_write_pages_op()
    return _OPS
