# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash text serving on TPU.

Projection and checkpoint mappings adapted from SGLang PR #36507. This initial
implementation supports bounded-context text generation: below index_topk
tokens DSA selects the entire causal prefix, which can be evaluated exactly using
ordinary attention. KDA uses the JAX recurrence in hardware_backend/tpu.
"""

import logging
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.configs.glm5_next import Glm5NextTextConfig
from sglang.srt.configs.model_config import GLM5_NEXT_TPU_MLA_ROPE_PAD
from sglang.srt.distributed import divide
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang_tpu.kda_layout import pack_kda_weights
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils.common import set_weight_attrs

logger = logging.getLogger(__name__)


class TpuRMSNormGated(nn.Module):
    def __init__(self, size, eps, activation="sigmoid"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x, gate):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float() * torch.sigmoid(gate.float())).to(x.dtype)


class Glm5NextLinearAttention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        config: Glm5NextTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
        reduce_results: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.tp_size = get_parallel().tp_size
        head_shard_size = get_parallel().attn_tp_size
        head_shard_rank = get_parallel().attn_tp_rank

        self.hidden_size = hidden_size
        self.config = config
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.num_k_heads = config.linear_attn_config["num_heads"]
        self.num_v_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = config.linear_attn_config["head_dim"]
        self.head_v_dim = config.linear_attn_config["head_dim"]
        self.layer_idx = layer_idx
        self.prefix = prefix
        assert self.num_heads % head_shard_size == 0
        self.local_num_heads = divide(self.num_heads, head_shard_size)

        projection_size = self.head_dim * self.num_heads
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_k_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=head_shard_rank,
            tp_size=head_shard_size,
            prefix=f"{prefix}.qkv_proj",
        )

        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_a_proj",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
            tp_rank=head_shard_rank,
            tp_size=head_shard_size,
        )

        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.b_proj",
            tp_rank=head_shard_rank,
            tp_size=head_shard_size,
        )

        self.g_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_a_proj",
        )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_b_proj",
            tp_rank=head_shard_rank,
            tp_size=head_shard_size,
        )

        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, head_shard_size), dtype=torch.float32)
        )

        set_weight_attrs(
            self.dt_bias,
            {"weight_loader": sharded_weight_loader(0)},
        )

        self.qkv_conv1d = MergedColumnParallelLinear(
            input_size=self.conv_size,
            output_sizes=[projection_size, projection_size, projection_size],
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.qkv_conv1d",
            tp_rank=head_shard_rank,
            tp_size=head_shard_size,
        )
        # ColumnParallelLinear's loader cannot reshape conv1d weights, so add the
        # singleton dimension after construction.
        self.qkv_conv1d.weight.data = self.qkv_conv1d.weight.data.unsqueeze(1)

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        set_weight_attrs(
            self.A_log,
            {"weight_loader": sharded_weight_loader(2)},
        )

        self.o_norm = TpuRMSNormGated(
            self.head_dim, eps=rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            reduce_results=reduce_results,
            tp_rank=head_shard_rank,
            tp_size=head_shard_size,
        )

        conv_weights = self.qkv_conv1d.weight.squeeze(1)
        bias = self.qkv_conv1d.bias

        self.attn = RadixLinearAttention(
            layer_id=self.layer_idx,
            num_q_heads=self.local_num_heads,
            num_k_heads=self.local_num_heads,
            num_v_heads=self.local_num_heads,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

        self.attn.lower_bound = config.linear_attn_config.get("gate_lower_bound", None)
        # [conv | dt_bias | A_log] as one f32 kernel operand, built after load.
        self.attn.kda_packed_weights = None

    def forward_qkvbfg(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        qkv, _ = self.qkv_proj(hidden_states)

        beta = self.b_proj(hidden_states)[0]
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]

        return (
            qkv,
            beta,
            forget_gate,
            g_proj_states,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            return hidden_states

        mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg(
            hidden_states, forward_batch
        )

        if not forward_batch.forward_mode.is_decode():
            forget_gate = forget_gate.unsqueeze(0)
        beta = beta.unsqueeze(0)

        core_attn_out = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=forget_gate,
            b=beta,
        )

        norm_gate = g_proj_states.unflatten(-1, (-1, self.head_dim))
        core_attn_out = self.o_norm(core_attn_out, norm_gate)
        # The TPU backend returns [tokens, heads, dim], including a token axis
        # for batch-one decode.
        core_attn_out = core_attn_out.flatten(-2)

        return self.o_proj(core_attn_out)[0]


class Glm5NextDenseAttention(nn.Module):
    """Unabsorbed MLA, exact while all causal keys fit the indexer budget."""

    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.heads = divide(config.num_attention_heads, get_parallel().attn_tp_size)
        self.q_dim = config.qk_nope_head_dim
        self.v_dim = config.v_head_dim
        self.q_rank = config.q_lora_rank
        self.kv_rank = config.kv_lora_rank
        if config.qk_rope_head_dim != 0:
            raise ValueError("GLM-5.3-Flash TPU expects NoPE attention")
        self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
            config.hidden_size,
            self.q_rank + self.kv_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj_with_mqa",
        )
        self.q_a_layernorm = RMSNorm(self.q_rank, eps=config.rms_norm_eps)
        self.kv_a_layernorm = RMSNorm(self.kv_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_rank,
            config.num_attention_heads * self.q_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_rank,
            config.num_attention_heads * (self.q_dim + self.v_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.v_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.attn = self._make_attn(layer_id)

    def _make_attn(self, layer_id):
        return RadixAttention(
            self.heads,
            self.q_dim,
            self.q_dim**-0.5,
            num_kv_heads=self.heads,
            layer_id=layer_id,
            v_head_dim=self.v_dim,
        )

    def forward(self, hidden_states, forward_batch, **kwargs):
        q_latent, kv_latent = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
            [self.q_rank, self.kv_rank], dim=-1
        )
        q = self.q_b_proj(self.q_a_layernorm(q_latent))[0].view(
            -1, self.heads, self.q_dim
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_latent))[0].view(
            -1, self.heads, self.q_dim + self.v_dim
        )
        k, v = kv.split([self.q_dim, self.v_dim], dim=-1)
        out = self.attn(q, k.contiguous(), v.contiguous(), forward_batch)
        return self.o_proj(out.reshape(-1, self.heads * self.v_dim))[0]


class Glm5NextLatentAttention(Glm5NextDenseAttention):
    """Absorbed MLA: cache the normed 512-dim latent, fold W_UK into q and W_UV
    into the output, and attend as MQA over the latent (pallas_tpu_mla)."""

    rope_pad = GLM5_NEXT_TPU_MLA_ROPE_PAD

    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__(config, layer_id, quant_config, prefix)
        # forward views kv_b_proj.weight as [heads, q_dim + v_dim, kv_rank]; that
        # holds only for the unquantized [out, in] layout the checkpoint ships.
        assert isinstance(self.kv_b_proj.quant_method, UnquantizedLinearMethod)

    def _make_attn(self, layer_id):
        return RadixAttention(
            self.heads,
            self.kv_rank + self.rope_pad,
            self.q_dim**-0.5,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_rank,
        )

    def forward(self, hidden_states, forward_batch, **kwargs):
        q_latent, kv_latent = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
            [self.q_rank, self.kv_rank], dim=-1
        )
        q = self.q_b_proj(self.q_a_layernorm(q_latent))[0].view(
            -1, self.heads, self.q_dim
        )
        # kv_b_proj rows are per head [K (q_dim) | V (v_dim)] over the latent.
        w_kv = self.kv_b_proj.weight.view(self.heads, self.q_dim + self.v_dim, -1)
        q_absorbed = torch.einsum("thd,hdc->thc", q, w_kv[:, : self.q_dim])
        latent = self.kv_a_layernorm(kv_latent).unsqueeze(1)
        n = hidden_states.shape[0]
        q_pe = q.new_zeros(n, self.heads, self.rope_pad)
        k_pe = q.new_zeros(n, 1, self.rope_pad)
        out = self.attn(
            q_absorbed, latent, latent, forward_batch, q_rope=q_pe, k_rope=k_pe
        ).view(n, self.heads, self.kv_rank)
        out = torch.einsum("thc,hvc->thv", out, w_kv[:, self.q_dim :])
        return self.o_proj(out.reshape(n, self.heads * self.v_dim))[0]


class Glm5NextSparseAttention(Glm5NextLatentAttention):
    """DeepSeek Sparse Attention: the pooled (kpool) indexer picks each query's
    keys, and absorbed MLA attends over only those (pallas_tpu_mla_sparse)."""

    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__(config, layer_id, quant_config, prefix)
        from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool

        self.layer_id = layer_id
        self.indexer = IndexerKPool(
            hidden_size=config.hidden_size,
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            rope_head_dim=0,
            index_topk=config.index_topk,
            q_lora_rank=self.q_rank,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 10000.0),
            scale_fmt="ue8m0",
            layer_id=layer_id,
            prefix=f"{prefix}.indexer",
            quant_config=quant_config,
            skip_rope=True,
            config=config,
        )

    def forward(self, hidden_states, forward_batch, **kwargs):
        q_latent, kv_latent = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
            [self.q_rank, self.kv_rank], dim=-1
        )
        q_lora = self.q_a_layernorm(q_latent)
        topk_indices = self.indexer(
            x=hidden_states,
            q_lora=q_lora,
            positions=forward_batch.positions,
            forward_batch=forward_batch,
            layer_id=self.layer_id,
        )
        q = self.q_b_proj(q_lora)[0].view(-1, self.heads, self.q_dim)
        # kv_b_proj rows are per head [K (q_dim) | V (v_dim)] over the latent.
        w_kv = self.kv_b_proj.weight.view(self.heads, self.q_dim + self.v_dim, -1)
        q_absorbed = torch.einsum("thd,hdc->thc", q, w_kv[:, : self.q_dim])
        latent = self.kv_a_layernorm(kv_latent).unsqueeze(1)
        n = hidden_states.shape[0]
        q_pe = q.new_zeros(n, self.heads, self.rope_pad)
        k_pe = q.new_zeros(n, 1, self.rope_pad)
        out = self.attn(
            q_absorbed,
            latent,
            latent,
            forward_batch,
            q_rope=q_pe,
            k_rope=k_pe,
            topk_indices=topk_indices,
        ).view(n, self.heads, self.kv_rank)
        out = torch.einsum("thc,hvc->thv", out, w_kv[:, self.q_dim :])
        return self.o_proj(out.reshape(n, self.heads * self.v_dim))[0]


def _runs_dsa() -> bool:
    # The target runs sparse attention unless the dense escape hatch is set
    # (model_config.runs_dsa_densely); the MTP draft head always runs dense.
    return envs.SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT.get() is None


def _full_attention_cls(*, draft: bool = False):
    if not draft and _runs_dsa():
        return Glm5NextSparseAttention
    return (
        Glm5NextLatentAttention
        if envs.SGLANG_TPU_GLM_FLASH_MLA.get()
        else Glm5NextDenseAttention
    )


def hc_pre(x, fn, scale, base, config):
    """mHC equations from the checkpoint's Transformers reference, in fp32."""
    h = config.hc_mult
    flat = x.flatten(-2).float()
    flat = flat * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + config.rms_norm_eps
    )
    pre, post, comb = F.linear(flat, fn.float()).split([h, h, h * h], dim=-1)
    pre = torch.sigmoid(pre * scale[0] + base[:h]) + config.hc_eps
    post = 2 * torch.sigmoid(post * scale[1] + base[h : 2 * h])
    comb = (
        torch.softmax((comb * scale[2] + base[2 * h :]).view(-1, h, h), dim=-1)
        + config.hc_eps
    )
    comb = comb / (comb.sum(-2, keepdim=True) + config.hc_eps)
    for _ in range(config.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + config.hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + config.hc_eps)
    return (pre[..., None] * x.float()).sum(-2).to(x.dtype), post, comb


def hc_post(output, residual, post, comb):
    return (
        post[..., None] * output.float().unsqueeze(-2)
        + torch.einsum("tij,tid->tjd", comb, residual.float())
    ).to(residual.dtype)


class Glm5NextDecoderLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.config = config
        if config.is_kda_layer(layer_id):
            self.self_attn = Glm5NextLinearAttention(
                layer_id,
                config.hidden_size,
                config,
                quant_config,
                prefix=f"{prefix}.self_attn",
                reduce_results=True,
            )
        else:
            self.self_attn = _full_attention_cls()(
                config, layer_id, quant_config, f"{prefix}.self_attn"
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if layer_id < config.first_k_dense_replace:
            self.mlp = DeepseekV2MLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                quant_config,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )
        else:
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                layer_id=layer_id,
            )
        for site in ("attn", "ffn"):
            h = config.hc_mult
            for suffix, shape in (
                ("base", ((2 + h) * h,)),
                ("scale", (3,)),
                ("fn", ((2 + h) * h, h * config.hidden_size)),
            ):
                setattr(
                    self,
                    f"hc_{site}_{suffix}",
                    nn.Parameter(torch.empty(shape, dtype=torch.float32)),
                )


class Glm5NextModel(nn.Module):
    def __init__(self, config, quant_config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Glm5NextDecoderLayer(config, i, quant_config, f"model.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.start_layer, self.end_layer = 0, config.num_hidden_layers
        # Layers whose input residual stream DFlash conditions on.
        self.layers_to_capture: frozenset[int] = frozenset()
        from sglang_tpu.router_topk import register_router_topk_op

        register_router_topk_op()
        from sglang_tpu.glm_mhc import register_glm_mhc_ops

        register_glm_mhc_ops()

    def forward(self, input_ids, forward_batch, input_embeds=None):
        """Final hidden states, plus the captured aux states when DFlash asks."""
        x = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        aux = []
        hidden = self._forward_fused_mhc(x, forward_batch, aux)
        return (hidden, aux) if self.layers_to_capture else hidden

    def prepare_fused_mhc(self):
        """Pack each site's hc_base and hc_scale into the kernel's [1, VEC_WIDTH] row."""
        from sglang_tpu.kernels.glm_mhc import VEC_WIDTH

        for layer in self.layers:
            for site in ("attn", "ffn"):
                base = getattr(layer, f"hc_{site}_base").detach().float()
                scale = getattr(layer, f"hc_{site}_scale").detach().float()
                vec = torch.zeros(1, VEC_WIDTH, dtype=torch.float32, device=base.device)
                vec[0, : base.numel()] = base
                vec[0, base.numel() : base.numel() + 3] = scale
                layer.register_buffer(f"hc_{site}_vec", vec, persistent=False)
            # The kernel applies each sublayer's input RMSNorm; f32 weights once.
            for site, norm in (
                ("attn", layer.input_layernorm),
                ("ffn", layer.post_attention_layernorm),
            ):
                layer.register_buffer(
                    f"hc_{site}_norm_weight",
                    norm.weight.detach().float().contiguous(),
                    persistent=False,
                )

    def _forward_fused_mhc(self, x, forward_batch, aux):
        """Same math as Glm5NextDecoderLayer.forward, with each hc_post fused
        into the following hc_pre: streams are [T, hc * hidden]. Appends the
        stream-mean residual entering each captured layer to ``aux``."""
        from sglang_tpu.glm_mhc import glm_mhc_ops

        pre_op, seam_op, post_op = glm_mhc_ops()
        c = self.config
        h = c.hc_mult
        # Each site also applies the next sublayer's input RMSNorm in-kernel,
        # so layer_in is already normalized.
        eps = float(c.rms_norm_eps)
        site = (h, c.hc_sinkhorn_iters, eps, float(c.hc_eps), eps)
        res = x.unsqueeze(-2).expand(-1, h, -1).reshape(x.shape[0], -1)
        first = self.layers[0]
        layer_in, post, comb = pre_op(
            res, first.hc_attn_fn, first.hc_attn_vec, first.hc_attn_norm_weight, *site
        )
        for i, layer in enumerate(self.layers):
            out = layer.self_attn(layer_in, forward_batch)
            res, layer_in, post, comb = seam_op(
                out,
                res,
                post,
                comb,
                layer.hc_ffn_fn,
                layer.hc_ffn_vec,
                layer.hc_ffn_norm_weight,
                *site,
            )
            out = layer.mlp(layer_in, forward_batch)
            if i + 1 < len(self.layers):
                nxt = self.layers[i + 1]
                res, layer_in, post, comb = seam_op(
                    out,
                    res,
                    post,
                    comb,
                    nxt.hc_attn_fn,
                    nxt.hc_attn_vec,
                    nxt.hc_attn_norm_weight,
                    *site,
                )
                if i + 1 in self.layers_to_capture:
                    aux.append(res.view(-1, h, c.hidden_size).mean(-2))
            else:
                res = post_op(out, res, post, comb, h)
        return self.norm(res.view(-1, h, c.hidden_size).mean(-2))


def _load_flash_weights(params, weights, n_routed_experts):
    """Place checkpoint tensors (already renamed to parameter names) into params:
    routed experts, stacked projections, and q_a + kv_a fused into
    fused_qkv_a_proj_with_mqa. Returns the loaded parameter names."""
    stacked = [
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("qkv_conv1d", "q_conv1d", 0),
        ("qkv_conv1d", "k_conv1d", 1),
        ("qkv_conv1d", "v_conv1d", 2),
    ]
    expert_mapping = FusedMoE.make_expert_params_mapping(
        "gate_proj", "down_proj", "up_proj", n_routed_experts
    )
    cached = {}
    loaded = set()
    for name, weight in weights:
        if ".experts." in name:
            for target, source, expert_id, shard_id in expert_mapping:
                if source in name:
                    name = name.replace(source, target)
                    param = params[name]
                    param.weight_loader(
                        param, weight, name, shard_id=shard_id, expert_id=expert_id
                    )
                    loaded.add(name)
                    break
            else:
                raise KeyError(f"Unrecognized expert weight: {name}")
            continue
        for target, source, shard_id in stacked:
            candidate = name.replace(source, target)
            if source in name and candidate in params:
                params[candidate].weight_loader(params[candidate], weight, shard_id)
                loaded.add(candidate)
                break
        else:
            if ".q_a_proj." in name or ".kv_a_proj_with_mqa." in name:
                cached[name] = weight
                qname = name.replace("kv_a_proj_with_mqa", "q_a_proj")
                kvname = name.replace("q_a_proj", "kv_a_proj_with_mqa")
                if qname not in cached or kvname not in cached:
                    continue
                name = qname.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                weight = torch.cat([cached.pop(qname), cached.pop(kvname)], dim=0)
            if name.endswith(".A_log"):
                weight = weight.view(1, 1, -1, 1)
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
    if cached:
        raise ValueError(f"Unpaired attention projections: {list(cached)}")
    return loaded


class Glm5NextForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if not current_platform.is_tpu():
            raise NotImplementedError("This GLM-5.3-Flash implementation requires TPU")
        args = get_server_args()
        self.config = config.get_text_config()
        c = self.config
        dense_limit = c.index_topk
        unsafe_limit = envs.SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT.get()
        if unsafe_limit is not None and unsafe_limit > dense_limit:
            logger.warning(
                "GLM-5.3-Flash attends densely up to %d tokens, past index_topk=%d; "
                "outputs beyond index_topk do not match DSA",
                unsafe_limit,
                dense_limit,
            )
            dense_limit = unsafe_limit
        if not args.language_model_only:
            raise ValueError("GLM-5.3-Flash TPU requires --language-model-only")
        if getattr(c, "index_kpool", 1) > 1 and not getattr(
            c, "index_kpool_always_select_tail", False
        ):
            raise ValueError(
                "GLM-5.3-Flash TPU requires the checkpoint's always-select-tail setting"
            )
        if not _runs_dsa() and (
            args.context_length is None or args.context_length > dense_limit
        ):
            raise ValueError(
                f"Dense GLM-5.3-Flash requires --context-length <= {dense_limit}"
            )
        if not args.disable_radix_cache:
            raise ValueError(
                "GLM-5.3-Flash TPU currently requires --disable-radix-cache"
            )
        if args.speculative_algorithm not in (None, "DFLASH", "EAGLE", "NEXTN"):
            raise ValueError(
                "GLM-5.3-Flash TPU supports DFLASH and NEXTN (MTP) speculative decoding"
            )
        if args.enable_dp_attention or args.pp_size != 1:
            raise ValueError(
                "GLM-5.3-Flash TPU currently supports TP/EP text serving without DP/PP"
            )
        if not c.mhc or c.qk_rope_head_dim != 0:
            raise ValueError("Expected the GLM-5.3-Flash mHC/NoPE checkpoint")
        self.quant_config = quant_config
        if quant_config is not None:
            quant_config.update_packed_modules_mapping(self.packed_modules_mapping)
        self.model = Glm5NextModel(c, quant_config)
        self.lm_head = ParallelLMHead(
            c.vocab_size, c.hidden_size, quant_config=quant_config, prefix="lm_head"
        )
        self.logits_processor = LogitsProcessor(c)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        hidden = self.model(input_ids, forward_batch, input_embeds)
        aux = None
        if self.model.layers_to_capture:
            hidden, aux = hidden
        return self.logits_processor(
            input_ids, hidden, self.lm_head, forward_batch, aux
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        # Capturing before layer k + 1 gives the completed output of layer k,
        # so the last layer has no capture point.
        assert max(layer_ids) + 1 < len(self.model.layers), layer_ids
        self.model.layers_to_capture = frozenset(val + 1 for val in layer_ids)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        config = config.get_text_config()
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
        )

    def get_embed_and_head(self):
        """Shared with the MTP draft head, which has no embedding or lm_head of its own."""
        return self.model.embed_tokens.weight, self.lm_head.weight

    def load_weights(self, weights):
        """Load the language checkpoint, explicitly excluding vision, MTP and the inactive indexer."""
        params = dict(self.named_parameters())

        def language_layers():
            for name, weight in weights:
                name = name.replace("model.language_model.", "model.")
                if name.startswith("model.visual."):
                    continue
                if ".indexer." in name and not _runs_dsa():
                    continue
                if (
                    name.startswith("model.layers.")
                    and int(name.split(".")[2]) >= self.config.num_hidden_layers
                ):
                    continue
                yield name, weight

        loaded = _load_flash_weights(
            params, language_layers(), self.config.n_routed_experts
        )
        missing = set(params) - loaded
        if missing:
            raise ValueError(f"Missing GLM-5.3-Flash weights: {sorted(missing)}")
        # The linear-attention layer holds a view. Refresh after loaders that
        # rebind storage (the TPU FP8 preparation path uses this convention).
        for layer in self.model.layers:
            attn = layer.self_attn
            if isinstance(attn, Glm5NextLinearAttention):
                # The depthwise [dim, 1, K] parameter itself, not a view.
                attn.attn.conv_weights = attn.qkv_conv1d.weight
                attn.attn.kda_packed_weights = pack_kda_weights(
                    attn.qkv_conv1d.weight,
                    attn.dt_bias,
                    attn.A_log,
                    attn.local_num_heads,
                    attn.head_k_dim,
                ).to(attn.dt_bias.device)
        self.model.prepare_fused_mhc()


class Glm5NextMtpModel(nn.Module):
    """The checkpoint's MTP head (layer num_hidden_layers): one full-attention +
    MoE block with a plain residual stream (no mHC, no KDA), fed with
    eh_proj([enorm(embed(token)), hnorm(target hidden)])."""

    def __init__(self, config, quant_config):
        super().__init__()
        h = config.hidden_size
        # Replaced by the target's tensors (set_embed_and_head) before serving.
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, h)
        self.enorm = RMSNorm(h, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(h, eps=config.rms_norm_eps)
        # The checkpoint keeps eh_proj unquantized.
        self.eh_proj = ReplicatedLinear(2 * h, h, bias=False, quant_config=None)
        # The checkpoint prefix, so the quant ignore list (bf16 kv_b_proj) matches.
        self.decoder = Glm5NextMtpLayer(
            config, quant_config, f"model.layers.{config.num_hidden_layers}"
        )
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(h, eps=config.rms_norm_eps)

    # (input_ids, positions, forward_batch) is the signature the TPU draft
    # compile policy calls the backbone with; positions are unused (NoPE).
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        x = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        target_hidden = forward_batch.spec_info.hidden_states
        x = self.eh_proj(torch.cat([self.enorm(x), self.hnorm(target_hidden)], -1))[0]
        x = self.decoder(x, forward_batch)
        return self.shared_head.norm(x)


class Glm5NextMtpLayer(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        # The draft has its own KV pool, holding this one layer.
        self.self_attn = _full_attention_cls(draft=True)(
            config, 0, quant_config, f"{prefix}.self_attn"
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = DeepseekV2MoE(
            config=config, quant_config=quant_config, prefix=f"{prefix}.mlp", layer_id=0
        )

    def forward(self, x, forward_batch):
        x = x + self.self_attn(self.input_layernorm(x), forward_batch)
        return x + self.mlp(self.post_attention_layernorm(x), forward_batch)


class Glm5NextForConditionalGenerationNextN(nn.Module):
    """TPU MTP draft for NEXTN/EAGLE speculative decoding. The draft config
    arrives renamed to this architecture with linear_attn_config cleared."""

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config.get_text_config()
        c = self.config
        self.quant_config = quant_config
        if quant_config is not None:
            quant_config.update_packed_modules_mapping(
                Glm5NextForConditionalGeneration.packed_modules_mapping
            )
        self.model = Glm5NextMtpModel(c, quant_config)
        self.lm_head = ParallelLMHead(
            c.vocab_size, c.hidden_size, quant_config=quant_config, prefix="lm_head"
        )
        self.logits_processor = LogitsProcessor(c)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        hidden = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        config = config.get_text_config()
        return ModelConfigForExpertLocation(
            num_layers=1, num_logical_experts=config.n_routed_experts
        )

    def load_weights(self, weights):
        """Load only the MTP layer; the embedding and lm_head come from the target."""
        mtp = f"model.layers.{self.config.num_hidden_layers}."
        params = dict(self.named_parameters())
        shared = ("embed_tokens", "shared_head.head")
        own = ("eh_proj", "enorm", "hnorm", "shared_head.norm")

        def mtp_layer():
            for name, weight in weights:
                name = name.replace("model.language_model.", "model.")
                if not name.startswith(mtp) or ".indexer." in name:
                    continue
                rest = name[len(mtp) :]
                if rest.startswith(shared) or rest.startswith("hc_"):
                    continue
                if rest.startswith(own):
                    yield "model." + rest, weight
                else:
                    yield "model.decoder." + rest, weight

        loaded = _load_flash_weights(params, mtp_layer(), self.config.n_routed_experts)
        missing = {
            n
            for n in set(params) - loaded
            if not n.startswith(("model.embed_tokens.", "lm_head."))
        }
        if missing:
            raise ValueError(f"Missing GLM-5.3-Flash MTP weights: {sorted(missing)}")


# Replaces the upstream glm5_next implementation (and its MTP draft) on TPU.
EntryClass = (
    [Glm5NextForConditionalGeneration, Glm5NextForConditionalGenerationNextN]
    if current_platform.is_tpu()
    else []
)
