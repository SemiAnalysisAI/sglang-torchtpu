import math
from enum import IntEnum
from typing import List, Optional

import torch

from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, is_tpu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()
_is_tpu = is_tpu()

if _is_cuda or _is_hip or _is_musa:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    score_list = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

    return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


def build_tree_kernel_efficient(
    bonus_tokens: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    draft_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrieve_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    if _is_npu:
        torch.ops.npu.build_tree_kernel_efficient(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    elif _is_tpu:
        _build_tree_kernel_efficient_torch(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
    )


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    if _is_cuda or _is_hip or _is_musa:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
    elif _is_tpu:
        _verify_tree_greedy_torch(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
    return predicts, accept_index, accept_token_num


def _build_tree_kernel_efficient_torch(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int,
):
    """Pure-torch CPU reference for ``sgl_kernel.build_tree_kernel_efficient``.

    Matches the FULL_MASK CUDA semantics in
    ``sgl-kernel/csrc/speculative/eagle_utils.cu::build_tree_efficient``.
    Per-step batch size and draft token count are small (~bs * draft_token_num
    ~= a few hundred elements), so a Python loop on CPU plus a bulk upload is
    cheap compared to either a TPU kernel launch or a Pallas reimplementation.
    """
    if tree_mask_mode != int(TreeMaskMode.FULL_MASK):
        raise NotImplementedError(
            "Eagle3 TPU path only supports FULL_MASK tree_mask_mode "
            f"(got tree_mask_mode={tree_mask_mode}). Other modes are CUDA-only."
        )

    device = tree_mask.device
    bs = int(parent_list.shape[0])
    if bs == 0:
        return

    pl_cpu = parent_list.cpu()  # [bs, topk*(depth-1)+1]
    si_cpu = selected_index.cpu()  # [bs, draft_token_num - 1]
    vsl_cpu = verified_seq_len.cpu()  # [bs]

    # CPU-side outputs filled in place, then bulk-copied back to device.
    tree_mask_cpu = tree_mask.detach().cpu().clone()
    positions_cpu = positions.detach().cpu().clone()
    retr_idx_cpu = retrieve_index.detach().cpu().clone()
    retr_nxt_cpu = retrieve_next_token.detach().cpu().clone()
    retr_sib_cpu = retrieve_next_sibling.detach().cpu().clone()

    parent_stride = topk * (depth - 1) + 1
    cumulative_seq = 0

    for bid in range(bs):
        seq_len = int(vsl_cpu[bid].item())
        seq_tree_idx = (
            draft_token_num * draft_token_num * bid + cumulative_seq * draft_token_num
        )

        for tid in range(draft_token_num):
            row_base = seq_tree_idx + (seq_len + draft_token_num) * tid
            own_bit = row_base + seq_len  # token_tree_idx - 1
            tree_mask_cpu[own_bit] = True
            for i in range(draft_token_num - 1):
                tree_mask_cpu[own_bit + 1 + i] = False

            if tid == 0:
                positions_cpu[bid * draft_token_num] = seq_len
                for i in range(draft_token_num - 1, 0, -1):
                    current_token_idx = bid * draft_token_num + i
                    retr_idx_cpu[bid, i] = current_token_idx
                    parent_tb_idx = int(si_cpu[bid, i - 1].item()) // topk
                    parent_position = 0
                    if parent_tb_idx > 0:
                        parent_token_idx = int(pl_cpu[bid, parent_tb_idx].item())
                        found = False
                        while parent_position < draft_token_num:
                            if (
                                int(si_cpu[bid, parent_position].item())
                                == parent_token_idx
                            ):
                                parent_position += 1
                                found = True
                                break
                            parent_position += 1
                        if not found:
                            # Matches CUDA's warning-and-skip path on invalid
                            # tree edges (e.g., NaN logprobs upstream).
                            continue
                    if parent_position == draft_token_num:
                        continue

                    if int(retr_nxt_cpu[bid, parent_position].item()) == -1:
                        retr_nxt_cpu[bid, parent_position] = i
                    else:
                        origin_next_token = int(
                            retr_nxt_cpu[bid, parent_position].item()
                        )
                        retr_nxt_cpu[bid, parent_position] = i
                        retr_sib_cpu[bid, i] = origin_next_token
                retr_idx_cpu[bid, 0] = bid * draft_token_num
            else:
                cur_position = tid - 1
                position = 0
                token_tree_idx = own_bit + 1  # row_base + seq_len + 1
                while True:
                    position += 1
                    tree_mask_cpu[token_tree_idx + cur_position] = True
                    parent_tb_idx = int(si_cpu[bid, cur_position].item()) // topk
                    if parent_tb_idx == 0:
                        break
                    token_idx = int(pl_cpu[bid, parent_tb_idx].item())
                    next_pos = draft_token_num
                    for k in range(draft_token_num):
                        if int(si_cpu[bid, k].item()) == token_idx:
                            next_pos = k
                            break
                    if next_pos == draft_token_num:
                        break
                    cur_position = next_pos
                positions_cpu[bid * draft_token_num + tid] = position + seq_len

        cumulative_seq += seq_len

    # Bulk-copy CPU-side outputs back to the original device tensors. The
    # caller passes pre-allocated buffers (potentially aliased downstream),
    # so we preserve the in-place contract via ``copy_``.
    tree_mask.copy_(tree_mask_cpu.to(device))
    positions.copy_(positions_cpu.to(device))
    retrieve_index.copy_(retr_idx_cpu.to(device))
    retrieve_next_token.copy_(retr_nxt_cpu.to(device))
    retrieve_next_sibling.copy_(retr_sib_cpu.to(device))


def _verify_tree_greedy_torch(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
):
    """Pure-torch CPU reference for ``sgl_kernel.verify_tree_greedy``.

    Matches ``sgl-kernel/csrc/speculative/eagle_utils.cu::VerifyTreeGreedy``.
    Per-batch sibling walk through the verification tree; accepts the longest
    matching prefix against ``target_predict``. Writes ``predicts``,
    ``accept_index``, ``accept_token_num`` in place to match the CUDA
    contract.
    """
    bs = int(candidates.shape[0])
    if bs == 0:
        return

    num_draft_tokens = int(candidates.shape[1])
    num_spec_step = int(accept_index.shape[1])

    cand = candidates.cpu()
    ri = retrieve_index.cpu()
    rnt = retrieve_next_token.cpu()
    rns = retrieve_next_sibling.cpu()
    tp = target_predict.reshape(-1).cpu()

    predicts_cpu = predicts.detach().cpu().clone()
    accept_index_cpu = accept_index.detach().cpu().clone()
    accept_token_num_cpu = accept_token_num.detach().cpu().clone()

    for bx in range(bs):
        last_accepted = int(ri[bx, 0].item())
        accept_index_cpu[bx, 0] = last_accepted
        num_accepted = 0
        cur_index = 0
        for _ in range(1, num_spec_step):
            cur_index = int(rnt[bx, cur_index].item())
            advanced = False
            while cur_index != -1:
                draft_index = int(ri[bx, cur_index].item())
                draft_token_id = int(cand[bx, cur_index].item())
                target_token_id = int(tp[last_accepted].item())
                if draft_token_id == target_token_id:
                    predicts_cpu[last_accepted] = target_token_id
                    num_accepted += 1
                    accept_index_cpu[bx, num_accepted] = draft_index
                    last_accepted = draft_index
                    advanced = True
                    break
                cur_index = int(rns[bx, cur_index].item())
            if not advanced:
                break
        accept_token_num_cpu[bx] = num_accepted
        predicts_cpu[last_accepted] = int(tp[last_accepted].item())

    predicts.copy_(predicts_cpu.to(predicts.device))
    accept_index.copy_(accept_index_cpu.to(accept_index.device))
    accept_token_num.copy_(accept_token_num_cpu.to(accept_token_num.device))
