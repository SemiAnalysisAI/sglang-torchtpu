"""Spec-v2 helper kernels + their pure-PyTorch (TPU) fallbacks.

Kept import-light on purpose: this module pulls in only torch / triton and two
small utils, NOT ModelRunner or the distributed/platform stack. That lets the CPU
fallback tests (test/manual/tpu/test_spec_v2_kernels_torch.py) import and exercise
the torch fallbacks without initializing the TPU backend.

torch_tpu has no Triton, so on TPU the ``*_func`` wrappers run device-generic
torch ops instead of launching the kernel. Each fallback is a faithful
re-expression of the kernel's semantics (validated on CPU against an independent
reference in the test above).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.utils.common import is_tpu, next_power_of_2

_is_tpu = is_tpu()


@triton.jit
def assign_draft_cache_locs_page_size_1(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    copy_len = topk * speculative_num_steps
    out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps

    # Copy from req_to_token to out_cache_loc
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(token_pool + kv_start + copy_offset, mask=mask)
        tl.store(out_cache_ptr + copy_offset, data, mask=mask)


@triton.jit
def fill_bonus_tokens(
    accept_tokens,
    accept_lens,
    bonus_tokens_ptr,
    num_draft_tokens: tl.constexpr,
):
    # NOTE: we cannot fuse any in-place operations of `accept_lens` inside this kernel
    # because this kernel reads accept_lens
    pid = tl.program_id(axis=0)
    # `accept_lens` includes the bonus token; the last accepted slot is at -1.
    accept_len = tl.load(accept_lens + pid)

    bonus_token_idx = num_draft_tokens * pid + accept_len - 1
    bonus_token = tl.load(accept_tokens + bonus_token_idx)
    tl.store(bonus_tokens_ptr + pid, bonus_token)


@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)


@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE


# ----------------------------------------------------------------------------
# Pure-PyTorch fallbacks for the helper kernels above (TPU has no Triton).
# ----------------------------------------------------------------------------


def assign_draft_cache_locs_page_size_1_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    topk: int,
    speculative_num_steps: int,
):
    """Gather draft KV slots from req_to_token into out_cache_loc (page_size=1).

    For each request ``i``: out_cache_loc[i*L : (i+1)*L] = req_to_token[
    req_pool_indices[i], seq_lens[i] : seq_lens[i]+L], with L = topk*num_steps.
    """
    bs = seq_lens.shape[0]
    if _is_tpu:
        if bs == 0:
            return
        copy_len = topk * speculative_num_steps
        # Both advanced-index tensors must share an integer dtype: torch_tpu
        # lowers the gather to a stablehlo.concatenate of the coordinate tensors,
        # which rejects mixed i32/i64 (CUDA silently promotes). Force int64.
        offsets = torch.arange(copy_len, device=req_to_token.device, dtype=torch.int64)
        rows = req_pool_indices.to(torch.int64)[:, None]  # (bs, 1)
        cols = seq_lens.to(torch.int64)[:, None] + offsets[None, :]  # (bs, L)
        gathered = req_to_token[rows, cols]  # (bs, L)
        out_cache_loc.view(bs, copy_len).copy_(gathered.to(out_cache_loc.dtype))
        return

    assign_draft_cache_locs_page_size_1[(bs,)](
        req_pool_indices,
        req_to_token,
        seq_lens,
        out_cache_loc,
        req_to_token.shape[1],
        topk,
        speculative_num_steps,
    )


def fill_bonus_tokens_func(
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,
    num_draft_tokens: int,
):
    """bonus_tokens[i] = accept_tokens[num_draft_tokens*i + accept_lens[i] - 1].

    accept_lens includes the bonus token, so the last accepted slot is -1.
    """
    bs = accept_lens.shape[0]
    if _is_tpu:
        if bs == 0:
            return
        idx = (
            num_draft_tokens * torch.arange(bs, device=accept_lens.device)
            + accept_lens
            - 1
        )
        # The kernel treats accept_tokens as flat (num_draft_tokens*pid + off);
        # the caller passes a [bs, num_draft_tokens] tensor, so flatten to match.
        flat = accept_tokens.reshape(-1)
        bonus_tokens.copy_(flat[idx].to(bonus_tokens.dtype))
        return

    fill_bonus_tokens[(bs,)](accept_tokens, accept_lens, bonus_tokens, num_draft_tokens)


def fill_accepted_out_cache_loc_func(
    accept_index: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accepted_out_cache_loc: torch.Tensor,
):
    """Compact accepted KV slots to the front of accepted_out_cache_loc.

    For each position ``i`` with accept_index[i] != -1, write
    out_cache_loc[accept_index[i]] to accepted_out_cache_loc[k], where k is the
    number of accepted positions strictly before i. Static-shape on TPU: route
    non-accepted positions to a scratch dump slot that is then discarded (no
    data-dependent output length, so no D2H sync). accepted_out_cache_loc is
    assumed zero-initialised; trailing (unwritten) slots stay 0.
    """
    # The kernel iterates a flat index space of bs*num_draft_tokens; the caller
    # may pass a [bs, num_draft_tokens] accept_index, so flatten and size by numel.
    size = accept_index.numel()
    if _is_tpu:
        if size == 0:
            return
        ai = accept_index.reshape(-1)
        ocl = out_cache_loc.reshape(-1)
        mask = ai != -1
        mask_i = mask.to(torch.int64)
        dst = torch.cumsum(mask_i, 0) - mask_i  # exclusive prefix count
        src = ai.clamp(min=0)
        vals = ocl[src].to(accepted_out_cache_loc.dtype)
        dump = size  # discard slot for non-accepted positions
        target = torch.where(mask, dst, torch.full_like(dst, dump))
        buf = torch.zeros(
            size + 1,
            dtype=accepted_out_cache_loc.dtype,
            device=ai.device,
        )
        buf.scatter_(0, target, vals)
        accepted_out_cache_loc.copy_(buf[:size])
        return

    fill_accepted_out_cache_loc[(size,)](
        accept_index,
        out_cache_loc,
        accepted_out_cache_loc,
        next_power_of_2(size),
    )


def assign_extend_cache_locs_kernel_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
):
    """Ragged gather: out_cache_loc[cum : cum+len_i] = req_to_token[
    req_pool_indices[i], start_offset[i] : end_offset[i]], packed by request.

    The TPU path must stay recompile-stable: a host-side loop that reads the
    per-request slice bounds (``.tolist()`` + ``req_to_token[r, s:e]`` ->
    ``out_cache_loc[off:off+n]``) bakes those Python-int bounds as HLO constants,
    so every distinct extend composition (which varies with seq_lens) compiles a
    new graph. Instead, map each output position to its owning request with pure
    tensor ops (fixed shapes: [cap] and [bs]) so the gather compiles once per
    bucket. ``bs`` is small, so the [cap, bs] segment comparison is cheap.
    """
    bs = req_pool_indices.shape[0]
    if _is_tpu:
        if bs == 0:
            return
        device = out_cache_loc.device
        cap = out_cache_loc.shape[0]
        start64 = start_offset.to(torch.int64)
        rows64 = req_pool_indices.to(torch.int64)
        lengths = (end_offset.to(torch.int64) - start64).clamp(min=0)  # [bs]
        cu = torch.cumsum(lengths, 0)  # inclusive segment ends [bs]
        cu_excl = cu - lengths  # exclusive segment starts [bs]
        total = cu[-1]  # tensor scalar: number of real packed tokens
        j = torch.arange(cap, device=device)  # output positions [cap]
        # Owning request per output position = #{k : cu[k] <= j} (no host sync).
        seg = (j[:, None] >= cu[None, :]).to(torch.int64).sum(1).clamp(max=bs - 1)
        within = j - cu_excl[seg]
        # Both advanced-index coords int64 so torch_tpu lowers a clean gather.
        col = (start64[seg] + within).clamp(min=0, max=req_to_token.shape[1] - 1)
        gathered = req_to_token[rows64[seg], col].to(out_cache_loc.dtype)
        # Only the first ``total`` positions are real; leave the rest untouched.
        out_cache_loc.copy_(torch.where(j < total, gathered, out_cache_loc))
        return

    assign_extend_cache_locs[(bs,)](
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        req_to_token.shape[1],
        next_power_of_2(bs),
    )
