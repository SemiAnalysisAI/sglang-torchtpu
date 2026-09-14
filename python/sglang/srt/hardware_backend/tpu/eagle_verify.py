"""Owned verification outputs for the TPU EAGLE top-k-one path."""

from functools import lru_cache

import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.hardware_backend.tpu.spec_op_warmup import (
    make_verify_tree_greedy_args,
)


@lru_cache(maxsize=32)
def _simulate_count(value: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.int32, device=device)


def _make_verify_args(context):
    return (
        *make_verify_tree_greedy_args(context),
        torch.ones(context.physical_bs, dtype=torch.int64, device=context.device),
        _simulate_count(0, context.device),
    )


@compiled_op(warmup=_make_verify_args)
def verify_and_bookkeep_tpu(
    logits, candidates, retrieve_index, seq_lens, simulate_count
):
    from sglang.srt.speculative.eagle_utils import (
        _verify_tree_greedy_torch,
        fused_verify_bookkeeping,
    )

    predict, accept_index, accept_lens = _verify_tree_greedy_torch.__wrapped__(
        logits, candidates, retrieve_index
    )
    bs, width = retrieve_index.shape
    columns = torch.arange(width, device=logits.device)
    # Positive counts force real draft tokens; negative counts use fixed tokens.
    forced = simulate_count.abs()
    simulated_index = torch.where(
        columns[None, :] < forced,
        accept_index[:, :1] + columns[None, :],
        -1,
    ).to(torch.int32)
    draft_tokens = candidates.reshape(bs, width)
    next_columns = (columns + 1).clamp_max(width - 1)
    simulated_predict = torch.where(
        columns[None, :] < forced - 1,
        draft_tokens[:, next_columns].to(predict.dtype),
        predict.reshape(bs, width),
    ).flatten()
    simulated_predict = torch.where(simulate_count < 0, 100, simulated_predict)
    predict = torch.where(forced > 0, simulated_predict, predict)
    accept_lens = torch.where(forced > 0, forced, accept_lens)
    accept_index = torch.where(forced > 0, simulated_index, accept_index)
    new_seq_lens, bonus_tokens = fused_verify_bookkeeping.__wrapped__(
        seq_lens, accept_lens, predict, accept_index
    )
    return predict, accept_lens, accept_index, new_seq_lens, bonus_tokens


def try_verify_and_bookkeep_tpu(next_token_logits, candidates, verify_input, seq_lens):
    """Greedy top-k-1 verification with bookkeeping fused into one program.

    Called from the greedy branch of ``eagle_sample`` after penalties and
    grammar masks have been applied. Returns ``None`` when the batch must take
    the separate verify and bookkeeping helpers.
    """
    from sglang.srt.speculative.spec_utils import (
        SIMULATE_ACC_LEN,
        SIMULATE_ACC_METHOD,
        SIMULATE_ACC_TOKEN_MODE,
        sample_simulated_acc_len,
    )

    if verify_input.tree_topk != 1:
        return None
    count = 0
    if SIMULATE_ACC_LEN > 0:
        state = verify_input.padding_state
        if SIMULATE_ACC_TOKEN_MODE not in ("fixed", "real-draft-token") or (
            state is not None and state.logical_batch_size != state.physical_batch_size
        ):
            return None
        count = sample_simulated_acc_len(
            SIMULATE_ACC_LEN, SIMULATE_ACC_METHOD, verify_input.max_tree_depth
        )
        if SIMULATE_ACC_TOKEN_MODE == "fixed":
            count = -count
    return verify_and_bookkeep_tpu(
        next_token_logits,
        candidates.flatten(),
        verify_input.retrieve_index,
        seq_lens,
        _simulate_count(count, next_token_logits.device),
    )


# Bookkeeping produced by the fused program, keyed by the accept_lens tensor it
# belongs to. eagle_sample stores it; verify_bookkeeping_func claims it.
_pending_bookkeeping: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


def stash_fused_bookkeeping_tpu(accept_lens, new_seq_lens, bonus_tokens) -> None:
    global _pending_bookkeeping
    _pending_bookkeeping = (accept_lens, new_seq_lens, bonus_tokens)


def take_fused_bookkeeping_tpu(accept_lens):
    """Return the fused bookkeeping for ``accept_lens`` once, else ``None``."""
    global _pending_bookkeeping
    pending, _pending_bookkeeping = _pending_bookkeeping, None
    if pending is None or pending[0] is not accept_lens:
        return None
    return pending[1], pending[2]
