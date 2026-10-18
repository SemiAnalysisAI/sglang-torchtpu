# SPDX-License-Identifier: Apache-2.0
"""Muse Spark Open text generation on TPU.

The language model of ``MuseSparkForConditionalGeneration`` built from SGLang
layers that have TPU backends; the vision and audio towers are not loaded.

Each routed expert's output is RMS-normalized before the top-k weighted sum.
NVFP4 experts declare that norm to the fused-RS MoE kernel, which applies it with
the combine and the expert-parallel reduction. Experts the checkpoint keeps in
BF16 return per-slot outputs, and the norm, combine and reduction run here.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.muse_spark import MuseSparkConfig, MuseSparkTextConfig
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix

# The fused-RS kernel normalizes expert outputs with this epsilon.
_FUSED_RS_POST_EXPERT_NORM_EPS = 1e-8
_ROUTER_RENORM_EPS = 1e-15
_RESIDUAL_GATE_EPS = 1e-3

_EXPERT_TENSOR = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\.(?P<proj>gate_up_proj|down_proj)"
    r"(?P<kind>|_weight_scale|_weight_scale_2|_input_scale)$"
)
_STACKED_LINEARS = (
    # (checkpoint suffix, parameter suffix, shard id)
    (".self_attn.q_proj.weight", ".self_attn.qkv_proj.weight", "q"),
    (".self_attn.k_proj.weight", ".self_attn.qkv_proj.weight", "k"),
    (".self_attn.v_proj.weight", ".self_attn.qkv_proj.weight", "v"),
    (".mlp.gate_proj.weight", ".mlp.gate_up_proj.weight", 0),
    (".mlp.up_proj.weight", ".mlp.gate_up_proj.weight", 1),
)
_SKIPPED_SUFFIXES = (
    ".self_attn.q_scale",
    ".self_attn.k_scale",
    ".self_attn.v_scale",
    ".mlp.norm_weight",
    "embed_norm.weight",
)


def _rms_norm(
    x: torch.Tensor, *, eps: float, gain: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """RMS norm over the last dim in fp32, times ``gain``, cast back to ``x.dtype``."""
    x32 = x.float()
    out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    if gain is not None:
        out = out * gain
    return out.to(x.dtype)


def _head_rms_norm(x: torch.Tensor, *, head_dim: int, eps: float) -> torch.Tensor:
    return _rms_norm(x.unflatten(-1, (-1, head_dim)), eps=eps).flatten(-2)


def _split_precision_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """fp32 ``x @ weight.T`` for bf16 ``x`` and an fp32 ``weight``.

    TPU matmuls round fp32 operands to bf16 (measured). Splitting the weight into
    a bf16 part and a bf16 residual keeps every product exact in fp32.
    """
    x32 = x.float()
    high = weight.bfloat16().float()
    low = (weight - high).bfloat16().float()
    return F.linear(x32, high) + F.linear(x32, low)


class MuseSparkRMSNorm(nn.Module):
    """RMS norm with gain ``gain_center + weight``.

    The checkpoint stores the input, pre-feed-forward and pre-expert gains
    zero-centered (``gain_center=1``) and the final norm's gain as is.
    """

    def __init__(self, hidden_size: int, *, eps: float, gain_center: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.eps = eps
        self.gain_center = gain_center

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, eps=self.eps, gain=self.weight.float() + self.gain_center)


class MuseSparkInputScaledRMSNorm(nn.Module):
    """Scales by ``1 + weight`` first, then RMS-normalizes without a gain."""

    def __init__(self, hidden_size: int, *, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = (x.float() * (self.weight.float() + 1.0)).to(x.dtype)
        return _rms_norm(scaled, eps=self.eps)


class MuseSparkResidualGate(nn.Module):
    """Gated residual ``alpha * stream + beta * branch`` with learned per-channel gates."""

    def __init__(self, hidden_size: int, *, temperature: float):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.temperature = temperature

    def forward(self, stream: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        w = self.gate.float() / self.temperature
        beta = torch.sigmoid(w)
        alpha = torch.sqrt(
            torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=_RESIDUAL_GATE_EPS)
        )
        return alpha * stream + beta * branch.float()


class MuseSparkAttention(nn.Module):
    def __init__(self, config: MuseSparkTextConfig, *, layer_id: int, prefix: str):
        super().__init__()
        tp_size = get_parallel().tp_size
        if config.num_attention_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by the TP size")
        if (
            config.num_key_value_heads % tp_size
            and tp_size % config.num_key_value_heads
        ):
            raise ValueError("TP size and num_key_value_heads must divide one another")
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size
        q_width = config.num_attention_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=False,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.gate_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=q_width,
            bias=False,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            input_size=q_width,
            output_size=hidden_size,
            bias=False,
            reduce_results=True,
            prefix=add_prefix("o_proj", prefix),
        )
        rope_theta = config.layer_rope_theta[layer_id]
        # The HF checkpoint stores Q/K in the split-half RoPE layout.
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                is_neox_style=True,
            )
            if rope_theta > 0
            else None
        )
        is_sliding = config.layer_types[layer_id] == "sliding_attention"
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            # QK-norm is weightless, so the query scale folds into the softmax scale.
            config.qk_scale_factor * self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=config.sliding_window - 1 if is_sliding else -1,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = _head_rms_norm(q, head_dim=self.head_dim, eps=self.norm_eps)
        k = _head_rms_norm(k, head_dim=self.head_dim, eps=self.norm_eps)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        out = self.attn(q, k, v, forward_batch)
        out = _head_rms_norm(out, head_dim=self.head_dim, eps=self.norm_eps)
        gate, _ = self.gate_proj(hidden_states)
        out, _ = self.o_proj(out * torch.sigmoid(gate))
        return out


class MuseSparkMLP(nn.Module):
    def __init__(self, config: MuseSparkTextConfig, *, prefix: str):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            reduce_results=True,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        out, _ = self.down_proj(F.silu(gate) * up)
        return out


class MuseSparkRouter(nn.Module):
    """Sigmoid router; the checkpoint stores its weight and bias in fp32."""

    def __init__(self, config: MuseSparkTextConfig):
        super().__init__()
        num_experts = config.num_local_experts
        self.weight = nn.Parameter(
            torch.empty(num_experts, config.hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(num_experts, dtype=torch.float32), requires_grad=False
        )
        self.top_k = config.num_experts_per_tok
        self.logit_multiplier = config.output_multiplier
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor) -> StandardTopKOutput:
        logits = _split_precision_linear(hidden_states, self.weight)
        scores = torch.sigmoid(logits * self.logit_multiplier)
        # The bias only picks the experts; the weights come from the raw scores.
        _, topk_ids = torch.topk(
            scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False
        )
        topk_weights = scores.gather(1, topk_ids)
        topk_weights = topk_weights / (
            topk_weights.sum(dim=-1, keepdim=True) + _ROUTER_RENORM_EPS
        )
        return StandardTopKOutput(
            topk_weights=topk_weights * self.routed_scaling_factor,
            topk_ids=topk_ids.to(torch.int32),
            router_logits=logits,
        )


class MuseSparkMoE(nn.Module):
    """Routed experts in a ``moe_hidden_size`` space between two projections."""

    def __init__(
        self,
        config: MuseSparkTextConfig,
        *,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        parallel = get_parallel()
        if parallel.moe_tp_size != 1:
            raise ValueError(
                "Muse Spark normalizes each expert's full output, so experts cannot "
                "be split across ranks; set --ep-size equal to --tp-size."
            )
        if config.post_norm_eps != _FUSED_RS_POST_EXPERT_NORM_EPS:
            raise ValueError(
                "The TPU fused-RS MoE kernel normalizes expert outputs with eps "
                f"{_FUSED_RS_POST_EXPERT_NORM_EPS}, got post_norm_eps="
                f"{config.post_norm_eps}."
            )
        self.tp_size = parallel.tp_size
        expert_hidden_size = config.moe_hidden_size
        self.gate = MuseSparkRouter(config)
        # Both projections split their output columns over TP and gather, so no
        # rank holds or reads the full weights.
        self.pre_expert_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=expert_hidden_size,
            bias=False,
            gather_output=True,
            prefix=add_prefix("pre_expert_proj", prefix),
        )
        self.pre_expert_norm = MuseSparkRMSNorm(
            expert_hidden_size, eps=config.rms_norm_eps
        )
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            hidden_size=expert_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            top_k=config.num_experts_per_tok,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            activation="silu",
            no_combine=True,
        )
        post_expert_norm = MuseSparkInputScaledRMSNorm(
            expert_hidden_size, eps=config.post_norm_eps
        )
        self.experts.post_expert_norm = post_expert_norm
        # Contract read by the TPU fused-RS MoE path (sglang_tpu.fused_rs_sequence_parallel).
        self.experts.supports_tpu_fused_rs_post_expert_norm = True
        self.experts.tpu_fused_rs_post_expert_norm_weight = lambda: (
            post_expert_norm.weight
        )
        self.post_expert_proj = ColumnParallelLinear(
            input_size=expert_hidden_size,
            output_size=config.hidden_size,
            bias=False,
            gather_output=True,
            prefix=add_prefix("post_expert_proj", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        topk_output = self.gate(hidden_states)
        expert_input, _ = self.pre_expert_proj(hidden_states)
        expert_input = self.pre_expert_norm(expert_input)
        out = self.experts(expert_input, topk_output)
        if not self.experts.handles_moe_reduction:
            out = self._combine(out, topk_weights=topk_output.topk_weights)
        out, _ = self.post_expert_proj(out)
        return out

    def _combine(
        self, per_slot: torch.Tensor, *, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        """Norm each [tokens, top_k, hidden] slot, weight, sum, then reduce over EP."""
        if per_slot.dim() != 3:
            raise RuntimeError(
                "Muse Spark needs per-expert outputs from the MoE backend to apply "
                f"its post-expert norm; got shape {tuple(per_slot.shape)}."
            )
        # Slots of experts on other ranks are zero and stay zero through the norm.
        normed = self.experts.post_expert_norm(per_slot)
        out = (normed.float() * topk_weights.unsqueeze(-1)).sum(dim=1)
        out = out.to(per_slot.dtype)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out


class MuseSparkDecoderLayer(nn.Module):
    """Post-normed attention and FFN branches joined to an fp32 residual stream."""

    def __init__(
        self,
        config: MuseSparkTextConfig,
        *,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        self.branch_dtype = torch.get_default_dtype()
        self.post_attention_norm_eps = config.post_norm_eps
        self.input_layernorm = MuseSparkRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MuseSparkAttention(
            config, layer_id=layer_id, prefix=add_prefix("self_attn", prefix)
        )
        self.post_attention_residual_gate = MuseSparkResidualGate(
            hidden_size, temperature=config.residual_gate_temperature
        )
        self.pre_feedforward_layernorm = MuseSparkRMSNorm(
            hidden_size, eps=config.rms_norm_eps
        )
        if config.mlp_layer_types[layer_id] == "sparse":
            self.mlp = MuseSparkMoE(
                config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = MuseSparkMLP(config, prefix=add_prefix("mlp", prefix))
        self.post_feedforward_layernorm = MuseSparkInputScaledRMSNorm(
            hidden_size, eps=config.post_norm_eps
        )
        self.post_feedforward_residual_gate = MuseSparkResidualGate(
            hidden_size, temperature=config.residual_gate_temperature
        )

    def forward(
        self,
        positions: torch.Tensor,
        stream: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        attn_input = self.input_layernorm(stream).to(self.branch_dtype)
        attn_output = self.self_attn(positions, attn_input, forward_batch)
        attn_output = _rms_norm(attn_output, eps=self.post_attention_norm_eps)
        stream = self.post_attention_residual_gate(stream, attn_output)

        ffn_input = self.pre_feedforward_layernorm(stream).to(self.branch_dtype)
        ffn_output = self.post_feedforward_layernorm(self.mlp(ffn_input))
        return self.post_feedforward_residual_gate(stream, ffn_output)


class MuseSparkTextModel(nn.Module):
    def __init__(
        self,
        config: MuseSparkTextConfig,
        *,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.branch_dtype = torch.get_default_dtype()
        self.embed_norm_eps = config.rms_norm_eps
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            MuseSparkDecoderLayer(
                config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix(f"layers.{layer_id}", prefix),
            )
            for layer_id in range(config.num_hidden_layers)
        )
        self.norm = MuseSparkRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, gain_center=0.0
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = _rms_norm(
                self.embed_tokens(input_ids), eps=self.embed_norm_eps
            )
        stream = input_embeds.float()
        for layer in self.layers:
            stream = layer(positions, stream, forward_batch)
        return self.norm(stream.to(self.branch_dtype))


class MuseSparkModel(nn.Module):
    """Holds the language model under the checkpoint's ``model.language_model`` names."""

    def __init__(
        self,
        config: MuseSparkTextConfig,
        *,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.language_model = MuseSparkTextModel(
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )


class MuseSparkForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: MuseSparkConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        parallel = get_parallel()
        if parallel.pp_size != 1:
            raise ValueError("Muse Spark on TPU does not support pipeline parallelism")
        if parallel.attn_dp_enabled:
            raise ValueError("Muse Spark on TPU does not support DP attention")
        if not config.language_model_only:
            raise ValueError(
                "Muse Spark on TPU serves text only; launch with --language-model-only"
            )
        if quant_config is not None and quant_config.get_name() != "modelopt_fp4":
            raise ValueError(
                "Muse Spark on TPU serves BF16 and ModelOpt NVFP4 checkpoints, got "
                f"{quant_config.get_name()}"
            )
        text_config = config.text_config
        self.config = text_config
        self.model = MuseSparkModel(
            text_config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            prefix=add_prefix("lm_head", prefix),
        )
        if text_config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(
                self.model.language_model.embed_tokens
            )
        # Logits are scaled by output_multiplier, then tanh-capped at
        # final_logit_softcapping (read from the config by LogitsProcessor).
        self.logits_processor = LogitsProcessor(
            text_config, logit_scale=text_config.output_multiplier
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.model.language_model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.language_model.embed_tokens

    def get_attention_sliding_window_size(self) -> Optional[int]:
        if "sliding_attention" in self.config.layer_types:
            return self.config.sliding_window - 1
        return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        loaded = set()
        for name, loaded_weight in weights:
            if _is_skipped(name):
                continue
            if self._load_expert_tensor(params, name, loaded_weight):
                loaded.add(_EXPERT_TENSOR.match(name).group("prefix"))
                continue
            param_name, shard_id = _param_name_and_shard(name)
            param = params.get(param_name)
            if param is None:
                raise ValueError(
                    f"Checkpoint weight {name!r} has no Muse Spark text parameter"
                )
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                loader(param, loaded_weight)
            else:
                loader(param, loaded_weight, shard_id)
            loaded.add(param_name)
        self._check_all_loaded(params, loaded)

    def _load_expert_tensor(
        self, params: dict, name: str, loaded_weight: torch.Tensor
    ) -> bool:
        """Split a stacked ``[num_experts, ...]`` tensor into FusedMoE expert loads."""
        match = _EXPERT_TENSOR.match(name)
        if match is None:
            return False
        is_w13 = match.group("proj") == "gate_up_proj"
        kind = match.group("kind") or "_weight"
        param_name = f"{match.group('prefix')}.{'w13' if is_w13 else 'w2'}{kind}"
        param = params.get(param_name)
        if param is None:
            raise ValueError(
                f"Checkpoint weight {name!r} has no parameter {param_name!r}"
            )
        if loaded_weight.shape[0] != self.config.num_local_experts:
            raise ValueError(
                f"{name!r} holds {loaded_weight.shape[0]} experts, the config "
                f"{self.config.num_local_experts}"
            )
        for expert_id, expert_weight in enumerate(loaded_weight):
            for shard_id, shard in _expert_shards(expert_weight, is_w13=is_w13):
                param.weight_loader(
                    param, shard, param_name, shard_id=shard_id, expert_id=expert_id
                )
        return True

    def _check_all_loaded(self, params: dict, loaded: set) -> None:
        expert_prefixes = {
            name.rsplit(".", 1)[0] for name in params if name.endswith(".w13_weight")
        }
        missing = sorted(
            name
            for name in params
            if not any(name.startswith(p + ".") for p in expert_prefixes)
            and name not in loaded
        )
        missing += sorted(p for p in expert_prefixes if p not in loaded)
        if missing:
            raise ValueError(
                f"Muse Spark checkpoint is missing {len(missing)} tensors, "
                f"e.g. {missing[:5]}"
            )


def _is_skipped(name: str) -> bool:
    """Vision and audio towers, and tensors the text model derives or ignores."""
    if not (name.startswith("model.language_model.") or name == "lm_head.weight"):
        return True
    return (
        "rotary_emb" in name
        or "num_tokens_per_expert" in name
        or name.endswith(_SKIPPED_SUFFIXES)
    )


def _param_name_and_shard(name: str) -> Tuple[str, Optional[object]]:
    for checkpoint_suffix, param_suffix, shard_id in _STACKED_LINEARS:
        if name.endswith(checkpoint_suffix):
            return name[: -len(checkpoint_suffix)] + param_suffix, shard_id
    return name, None


def _expert_shards(expert_weight: torch.Tensor, *, is_w13: bool):
    """(shard id, tensor) pairs for one expert; gate_up stacks [gate, up]."""
    if not is_w13:
        return (("w2", expert_weight),)
    if expert_weight.dim() == 1:
        # Per-tensor scales: one value each for gate and up.
        return (("w1", expert_weight[0]), ("w3", expert_weight[1]))
    gate, up = expert_weight.chunk(2, dim=0)
    return (("w1", gate), ("w3", up))


EntryClass = [MuseSparkForConditionalGeneration]
