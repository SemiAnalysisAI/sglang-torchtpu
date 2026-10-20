# SPDX-License-Identifier: Apache-2.0
"""Build the Kimi-K3 megakernel on a loaded model and route decode to it.

Runs once per rank after the model loader finished, quant post-processing
included, and before the KV pools are sized, so the megakernel's weights and
states count as model memory. ``sglang_tpu/hooks/kimi_k3_megakernel.py`` calls
it.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.compilation.support_torch_compile import (
    get_torch_compile_policy,
    support_torch_compile,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.models.kimi_k3 import (
    KimiK3ForConditionalGeneration,
    KimiK3LinearForCausalLM,
    KimiK3MoE,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.envs import envs as tpu_envs

from ._vendor.kimi.load import selected_local_shapes
from .config import (
    EP_SIZE,
    TP_SIZE,
    check_ranks_agree,
    megakernel_context,
    row_buckets,
)
from .policy import megakernel_policy
from .program import make_decode_program, state_shapes
from .runner import KimiK3MegakernelRunner
from .weights import base_tensor, load_dense_families, stack_routed_experts

logger = logging.getLogger(__name__)

_STATE_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def allocate_states(
    *, rows: int, context: int, layers: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.zeros(shape, dtype=_STATE_DTYPES[dtype], device=device)
        for shape, dtype in state_shapes(rows, context, layers)
    )


def register_decode_op(
    *,
    weight_names: tuple[str, ...],
    batch_size: int,
    layers: int,
    vocab_size: int,
    tuned: bool,
):
    """The TorchTPU custom op running one ``batch_size``-row decode step."""
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang_tpu.pallas_export import tpu_info_for_cpu_export
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = get_pallas_mesh(("tp",), (TP_SIZE,))
    with tpu_info_for_cpu_export():
        step, input_specs, output_specs, donated, aliases = make_decode_program(
            mesh, weight_names, batch_size=batch_size, layers=layers, tuned=tuned
        )

    def fake_step(*operands: torch.Tensor) -> torch.Tensor:
        # Logits only: the state outputs are written back into the operands.
        return operands[-1].new_empty((batch_size, vocab_size), dtype=torch.float32)

    recipe = "tuned" if tuned else "plain"
    return cpu_export_jax_op(
        f"sglang_tpu_pallas::kimi_k3_megakernel_decode_b{batch_size}_l{layers}_{recipe}",
        step,
        mesh=mesh,
        input_partition_specs=input_specs,
        output_partition_specs=output_specs,
        donate_argnums=donated,
        input_output_aliases=aliases,
        fake_function=fake_step,
    )


def _language_model(model: torch.nn.Module) -> KimiK3LinearForCausalLM:
    if isinstance(model, KimiK3ForConditionalGeneration):
        return model.language_model
    if isinstance(model, KimiK3LinearForCausalLM):
        return model
    raise TypeError(
        f"the Kimi-K3 megakernel needs a Kimi-K3 model, got {type(model).__name__}"
    )


def _moe_layers(language_model: KimiK3LinearForCausalLM) -> list[FusedMoE]:
    moe_layers = [
        layer.mlp.experts
        for layer in language_model.model.layers
        if isinstance(layer.mlp, KimiK3MoE)
    ]
    # The kernel's layout: a dense MLP in layer 0, routed experts in every other.
    if len(moe_layers) != language_model.config.num_hidden_layers - 1:
        raise ValueError(
            f"the Kimi-K3 megakernel needs routed experts in layers 1.."
            f"{language_model.config.num_hidden_layers - 1}, found "
            f"{len(moe_layers)} MoE layers"
        )
    return moe_layers


def build_runner(
    language_model: KimiK3LinearForCausalLM, *, model_path: str, context_length: int
) -> KimiK3MegakernelRunner:
    parallel = get_parallel()
    if parallel.tp_size != TP_SIZE or parallel.moe_ep_size != EP_SIZE:
        raise ValueError(
            f"the Kimi-K3 megakernel needs TP{TP_SIZE} / EP{EP_SIZE}, got "
            f"TP{parallel.tp_size} / EP{parallel.moe_ep_size}"
        )
    config = language_model.config
    layers = config.num_hidden_layers
    device = language_model.lm_head.weight.device
    rows = get_schedule().max_running_requests
    context = megakernel_context(context_length)
    tuned = tpu_envs.SGLANG_TPU_OPT_K3_MEGAKERNEL_TUNED_RECIPE.get()
    # Before the expert restack: a disagreeing rank fails fast, not after it.
    local = (rows, context, tuned, layers, config.vocab_size)
    gathered = [None] * parallel.tp_size
    torch.distributed.all_gather_object(
        gathered, local, group=parallel.tp_group.cpu_group
    )
    check_ranks_agree(local, gathered)

    experts = stack_routed_experts(_moe_layers(language_model), device)
    current_platform.synchronize()
    dense = load_dense_families(
        model_path, rank=parallel.tp_rank, layers=layers, device=device
    )
    weight_names = tuple(selected_local_shapes(layers, vocab=config.vocab_size))
    weights = {**dense, **experts}
    decode_ops = {
        batch_size: register_decode_op(
            weight_names=weight_names,
            batch_size=batch_size,
            layers=layers,
            vocab_size=config.vocab_size,
            tuned=tuned,
        )
        for batch_size in row_buckets(rows)
    }
    runner = KimiK3MegakernelRunner(
        weights=[weights[name] for name in weight_names],
        states=allocate_states(
            rows=rows, context=context, layers=layers, device=device
        ),
        m_kb=weights["m_kb"],
        decode_ops=decode_ops,
        rank=base_tensor(
            torch.tensor([parallel.tp_rank], dtype=torch.int32, device=device)
        ),
        layers=layers,
    )
    current_platform.synchronize()
    logger.info(
        "Kimi-K3 megakernel attached on rank %d: rows=%d context=%d recipe=%s",
        parallel.tp_rank,
        rows,
        context,
        "tuned" if tuned else "plain",
    )
    return runner


def attach(model: torch.nn.Module, *, model_path: str, context_length: int) -> None:
    """Build the runner and annotate the language model with the megakernel policy."""
    language_model = _language_model(model)
    runner = build_runner(
        language_model, model_path=model_path, context_length=context_length
    )
    # The megakernel all-gathers full-vocabulary logits, which the sampler's
    # vocabulary-sharded top-k cannot take; prefill samples full-vocab too.
    language_model.logits_processor.use_tpu_distributed_top_k = False
    base = get_torch_compile_policy(language_model) or DecoderTpuCompilePolicy
    if not issubclass(base, DecoderTpuCompilePolicy):
        raise TypeError(f"the Kimi-K3 megakernel extends a decoder policy, got {base}")
    support_torch_compile(language_model, policy=megakernel_policy(base, runner))
