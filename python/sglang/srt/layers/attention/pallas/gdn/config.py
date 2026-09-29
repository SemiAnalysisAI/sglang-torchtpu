# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for SGLang-TorchTPU from vllm-project/tpu-inference
# `tpu_inference.kernels.gdn.v3.config` (Apache-2.0).
# Changes: adds the KDA attention mode (per-channel gate); reformatted to the
# repo's style.

import dataclasses
import enum
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class AttentionMode(enum.StrEnum):
    # Gated Delta Net: one scalar gate per value head.
    GDN = enum.auto()
    # Kimi Delta Attention: the gate is per-channel, so it carries
    # `kq_head_dim` values per value head instead of one.
    KDA = enum.auto()


class GDNMode(enum.StrEnum):
    BATCHED = enum.auto()
    PER_SEQ = enum.auto()

    def get_seq_tile_size(self, tile_size: int) -> int:
        if self == GDNMode.BATCHED:
            return tile_size
        return 1

    def get_chunk_size(self, tile_size: int) -> int:
        if self == GDNMode.BATCHED:
            return 1
        return tile_size


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dtypes:
    act_in: jnp.dtype
    act_out: jnp.dtype
    compute: jnp.dtype
    recurrent_state: jnp.dtype
    conv_state: jnp.dtype


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNConfig:
    mode: GDNMode
    dtypes: Dtypes
    batch_size: int
    dim_size: int
    kernel_size: int
    tile_size: int
    num_kq_heads: int
    num_v_heads: int
    kq_head_dim: int
    v_head_dim: int
    attention_mode: AttentionMode = AttentionMode.GDN
    # KDA only, and must be negative. Selects the bounded gate form over the
    # unbounded one; see `fused_conv1d_gdn` for the two and their validation.
    gate_lower_bound: float | None = None
    num_buffers: int = 2

    @property
    def chunk_size(self) -> int:
        return self.mode.get_chunk_size(self.tile_size)

    @property
    def seq_tile_size(self) -> int:
        return self.mode.get_seq_tile_size(self.tile_size)

    @property
    def prev_kernel_size(self) -> int:
        return self.kernel_size - 1

    @property
    def v_dim_size(self) -> int:
        return self.num_v_heads * self.v_head_dim

    @property
    def kq_dim_size(self) -> int:
        return self.num_kq_heads * self.kq_head_dim

    @property
    def v_per_kq_head(self) -> int:
        return self.num_v_heads // self.num_kq_heads

    @property
    def aligned_num_v_heads(self) -> int:
        tpu_info = pltpu.get_tpu_info()
        num_lanes = tpu_info.num_lanes
        return pl.cdiv(self.num_v_heads, num_lanes) * num_lanes

    @property
    def is_kda(self) -> bool:
        return self.attention_mode == AttentionMode.KDA

    @property
    def gate_dim(self) -> int:
        """Width of the gate activation `a`, in elements per token."""
        if self.is_kda:
            return self.num_v_heads * self.kq_head_dim
        return self.num_v_heads

    @property
    def aligned_gate_dim(self) -> int:
        num_lanes = pltpu.get_tpu_info().num_lanes
        return pl.cdiv(self.gate_dim, num_lanes) * num_lanes

    def get_kernel_name(self) -> str:
        suffix = "_bounded" if self.is_kda and self.gate_lower_bound is not None else ""
        return f"fused_conv1d_{self.attention_mode.value}_{self.mode.value}{suffix}"

    def get_metadata(self) -> dict[str, str | int | float]:
        cfgs_dict = dataclasses.asdict(self)
        ret = {}
        for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
            key = jax.tree_util.keystr(path, simple=True, separator=".")
            if not isinstance(val, str | int | float):
                val = str(val)
            ret[key] = val
        return ret

    def get_out_shape(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            (self.batch_size, self.num_v_heads, self.v_head_dim),
            self.dtypes.act_out,
        )

    def get_vmem_limit_bytes(self) -> int:
        tpu_info = pltpu.get_tpu_info()
        return int(0.7 * tpu_info.vmem_capacity_bytes)

    def get_scratch_shape_dict(self) -> dict[str, Any]:
        conv_shape = (self.seq_tile_size, self.prev_kernel_size, 1, self.dim_size)
        recurrent_shape = (
            self.seq_tile_size,
            self.num_v_heads,
            self.kq_head_dim,
            self.v_head_dim,
        )

        carry_conv_scratch = carry_recurrent_scratch = None
        # NOTE: Currently, batched mode only supports case where 1 seq = 1 tile.
        # Therefore, inter tile carry is not needed.
        if self.mode != GDNMode.BATCHED:
            carry_conv_scratch = pltpu.VMEM(conv_shape, jnp.float32)
            carry_recurrent_scratch = pltpu.VMEM(recurrent_shape, jnp.float32)

        return dict(
            carry_conv_scratch_ref=carry_conv_scratch,
            carry_recurrent_scratch_ref=carry_recurrent_scratch,
        )
