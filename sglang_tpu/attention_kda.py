# SPDX-License-Identifier: Apache-2.0
"""TPU KDA backend for GLM-5.3-Flash using the shared hybrid request/state metadata."""

# Keep runtime annotations: custom_jax_kernel inspects jax.Array identity.
import torch

from sglang.srt.environ import envs
from sglang_tpu.attention_gdn_pallas import TpuGDNAttnBackend

try:
    from torch_tpu._internal import pallas
except ImportError:
    pallas = None

if pallas is not None:
    from sglang_tpu.kernels.kda import ragged_kda

_KDA_OP = None


def build_kda_op():
    global _KDA_OP
    if _KDA_OP is not None:
        return _KDA_OP
    if pallas is None:
        raise ImportError("The TPU KDA custom op requires torch_tpu")

    kernel = pallas.custom_jax_kernel(
        ragged_kda,
        name="jax_sglang_tpu_kda",
        static_argnums=(11, 12, 13),
        donate_argnums=(3, 4),
        input_output_aliases={3: 0, 4: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        "sglang_tpu_pallas::kda_attention",
        mutates_args=("conv_state", "recurrent_state"),
        device_types=["tpu"],
    )
    def op(
        qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        conv_weight: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        num_heads: int,
        head_dim: int,
        lower_bound: float,
    ) -> torch.Tensor:
        _, _, output = kernel(
            qkv,
            a,
            b,
            conv_state,
            recurrent_state,
            conv_weight,
            a_log,
            dt_bias,
            query_start_loc,
            state_indices,
            seq_lens,
            num_heads,
            head_dim,
            lower_bound,
        )
        return output

    @op.register_fake
    def fake(
        qkv,
        a,
        b,
        conv_state,
        recurrent_state,
        conv_weight,
        a_log,
        dt_bias,
        query_start_loc,
        state_indices,
        seq_lens,
        num_heads,
        head_dim,
        lower_bound,
    ):
        return qkv.new_empty((qkv.shape[0], num_heads, head_dim))

    _KDA_OP = op
    return op


class TpuKDAAttnBackend(TpuGDNAttnBackend):
    """GLM-5.3-Flash KDA on the fused Conv1D + GDN v3 Pallas kernel in KDA mode.

    SGLANG_TPU_KDA_USE_REFERENCE_SCAN selects the lax.scan recurrence instead.
    """

    def __init__(self, model_runner):
        self._use_scan = envs.SGLANG_TPU_KDA_USE_REFERENCE_SCAN.get()
        super().__init__(
            model_runner, op_builder=build_kda_op if self._use_scan else None
        )

    def init_forward_metadata(self, forward_batch):
        if forward_batch.forward_mode.is_target_verify():
            raise NotImplementedError(
                "TpuKDAAttnBackend does not support speculative target verify"
            )
        super().init_forward_metadata(forward_batch)

    def _run(self, layer, mixed_qkv, a, b):
        if self._use_scan:
            return self._run_scan(layer, mixed_qkv, a, b)
        cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        tokens = mixed_qkv.shape[0]
        n_v, d_k, d_v = layer.num_v_heads, layer.head_k_dim, layer.head_v_dim
        lower_bound = layer.lower_bound
        out = self._op(
            mixed_qkv,
            # Raw beta [T, n_v] and forget gate [T, n_v * d_k]; the kernel
            # applies the sigmoid and the gate activation itself.
            b.reshape(tokens, n_v),
            a.reshape(tokens, n_v * d_k),
            cache.conv[0],
            cache.temporal,
            layer.conv_weights.unsqueeze(1),
            layer.bias,
            layer.A_log.reshape(n_v),
            # Per-channel under KDA; Mosaic rejects the flat [n_v * d_k] form.
            layer.dt_bias.reshape(n_v, d_k),
            self._query_start_loc,
            self._state_indices,
            self._distribution,
            self._seq_lens,
            layer.num_k_heads,
            n_v,
            d_k,
            d_v,
            layer.conv_weights.shape[-1],
            True,
            None if lower_bound is None else float(lower_bound),
        )
        return out.view(tokens, n_v, d_v)

    def _run_scan(self, layer, mixed_qkv, a, b):
        cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        return self._op(
            mixed_qkv,
            a,
            b,
            cache.conv[0],
            cache.temporal,
            layer.conv_weights,
            layer.A_log,
            layer.dt_bias,
            self._query_start_loc,
            self._state_indices,
            self._seq_lens,
            layer.num_k_heads,
            layer.head_k_dim,
            float(layer.lower_bound or 0.0),
        )
