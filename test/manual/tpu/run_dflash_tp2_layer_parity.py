"""Manual TP=2 parity harness for the non-causal DFlash draft attention.

Launches two real ranks (``tensor_model_parallel_size=2``, ``tpu_dist`` backend),
instantiates the auto-sharded ``DFlashAttention`` on each TPU chip, drives the
real ``layer.forward`` through ``RadixAttention(ENCODER_ONLY)`` + the RPA-v3
backend (so ``o_proj``'s RowParallel all-reduce is exercised across ranks), and
compares rank 0's combined output to a dense full-head reference reconstructed
from the all-gathered per-rank weight shards.

WARNING / STATUS: a *standalone* two-rank ``mp.spawn`` launch WEDGES this
torch_tpu stack -- the spawn children hang on the ``tpu_dist`` collective
rendezvous and lock the chips until the processes are killed (consistent with
the project's TP>1 wedge notes). SGLang's real server brings TP up correctly
(the EAGLE3-TPU TP path works), so the authoritative multi-rank collective
validation for DFlash is the end-to-end server run (task16), not this script.
This file is kept as a documented harness; it is guarded behind
``DFLASH_TP2_MANUAL=1`` so it cannot accidentally wedge the device, and the
DFlash-specific TP=2 head-sharding correctness is covered safely in-process by
``test_dflash_layer_attention.py::TestDFlashAttentionLayerParityTP2Shard``.

Run (may wedge the TPU; only with a recovery plan):
    DFLASH_TP2_MANUAL=1 ./.venv/bin/python test/manual/tpu/run_dflash_tp2_layer_parity.py
"""

import os
import sys
import types

PAGE_SIZE = 16
PAGES_PER_SEQ = 18
HIDDEN = 512
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 128
TOLERANCE = 0.05
CASES = [(0, 4), (1, 4), (17, 4), (63, 4), (64, 4), (65, 4), (255, 4), (17, 16), (255, 16)]


def _config():
    return types.SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_scaling=None,
        is_neox_style=True,
        max_position_embeddings=40960,
    )


def _make_backend(page_size, pages_per_seq, device):
    from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_v3 import (
        TpuPallasRpaV3AttnBackend,
        _build_pallas_kernel_op,
    )
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

    import torch

    max_ctx = pages_per_seq * page_size
    req_to_token = (
        torch.arange(max_ctx, dtype=torch.int32, device=device).unsqueeze(0).clone()
    )
    backend = TpuPallasRpaV3AttnBackend.__new__(TpuPallasRpaV3AttnBackend)
    AttentionBackend.__init__(backend)
    backend.device = device
    backend.speculative_step_id = 0
    backend.page_size = page_size
    backend.req_to_token_pool = types.SimpleNamespace(req_to_token=req_to_token)
    backend._is_hybrid_swa = False
    backend.kv_cache_quant_dtype = None
    backend.forward_metadata = None
    backend._distribution_cache = {}
    backend._encoder_only_mask_cache = {}
    backend._op = _build_pallas_kernel_op()
    return backend


def _fused_cache(prefix_k, prefix_v, prefix_len, device):
    import torch
    from sglang.srt.layers.attention.pallas.util import get_dtype_packing

    P, PS, NKV, HD = PAGES_PER_SEQ, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM
    k_pages = torch.zeros(P, PS, NKV, HD, dtype=torch.bfloat16, device=device)
    v_pages = torch.zeros(P, PS, NKV, HD, dtype=torch.bfloat16, device=device)
    for pos in range(prefix_len):
        k_pages[pos // PS, pos % PS] = prefix_k[pos]
        v_pages[pos // PS, pos % PS] = prefix_v[pos]
    packing = get_dtype_packing(__import__("jax.numpy", fromlist=["bfloat16"]).bfloat16)
    inter = __import__("torch").stack([k_pages, v_pages], dim=3).reshape(P, PS, NKV * 2, HD)
    return inter.reshape(P, PS, (NKV * 2) // packing, packing, HD)


def _worker(rank, world_size, port, ret):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["TPU_VISIBLE_CHIPS"] = str(rank)

    import torch  # noqa: F401
    import torch_tpu  # noqa: F401
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.distributed.parallel_state import (
        get_tensor_model_parallel_group,
    )
    from sglang.srt.models.dflash import DFlashAttention, apply_qk_norm
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    device = torch.device("tpu", rank)
    torch.tpu.set_device(device)
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="meta-llama/Llama-3.1-8B-Instruct")
    )
    init_distributed_environment(
        world_size=world_size, rank=rank, local_rank=rank, backend="tpu_dist"
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    tp_group = get_tensor_model_parallel_group()

    # Per-rank distinct weights (so the full weight is the concatenation of
    # shards). Seed by rank so the two shards differ.
    torch.manual_seed(100 + rank)
    layer = DFlashAttention(_config(), layer_id=0)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn_like(p, dtype=torch.float32) * 0.05)
            p.requires_grad_(False)
    layer = layer.to(device).to(torch.bfloat16)
    backend = _make_backend(PAGE_SIZE, PAGES_PER_SEQ, device)

    NKV_LOCAL = layer.num_kv_heads
    NQ_LOCAL = layer.num_heads

    failures = []
    for ci, (prefix_len, block_size) in enumerate(CASES):
        kv_len = prefix_len + block_size
        # Inputs identical on both ranks (same seed, no rank offset).
        torch.manual_seed(500 + ci)
        prefix_k = torch.randn(prefix_len, NKV_LOCAL, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.1
        prefix_v = torch.randn(prefix_len, NKV_LOCAL, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.1
        hidden = torch.randn(block_size, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1
        positions = torch.arange(prefix_len, kv_len, dtype=torch.int64, device=device)

        kv_fused = _fused_cache(prefix_k, prefix_v, prefix_len, device)
        fb = types.SimpleNamespace(
            seq_lens=torch.tensor([kv_len], dtype=torch.int32, device=device),
            req_pool_indices=torch.tensor([0], dtype=torch.int64, device=device),
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor([block_size], dtype=torch.int32, device=device),
            spec_info=None,
            seq_lens_cpu=torch.tensor([kv_len], dtype=torch.int32),
            token_to_kv_pool=types.SimpleNamespace(get_fused_kv_buffer=lambda lid: kv_fused),
            attn_backend=backend,
        )
        backend.init_forward_metadata(fb)
        tp_out = layer.forward(positions, hidden, fb).float()  # RowParallel all-reduced

        # Dense full-head reference from the gathered per-rank weight shards.
        # qkv local rows = [q_local | k_local | v_local]; o_proj is input-sharded.
        qkv_w = layer.qkv_proj.weight.data  # [q_local+2*kv_local, hidden]
        o_w = layer.o_proj.weight.data  # [hidden, q_local*head_dim]
        qkv_all = tp_group.all_gather(qkv_w.contiguous(), dim=0)  # concat rank shards
        o_all = tp_group.all_gather(o_w.contiguous(), dim=1)
        ql, kl = NQ_LOCAL * HEAD_DIM, NKV_LOCAL * HEAD_DIM
        # Split each rank's local qkv into q/k/v and regroup across ranks.
        full_q, full_k, full_v = [], [], []
        for r in range(world_size):
            seg = qkv_all[r * (ql + 2 * kl) : (r + 1) * (ql + 2 * kl)]
            full_q.append(seg[:ql])
            full_k.append(seg[ql : ql + kl])
            full_v.append(seg[ql + kl : ql + 2 * kl])
        full_qkv = torch.cat(full_q + full_k + full_v, dim=0).float()
        full_o = o_all.float()  # [hidden, NQ*head_dim]

        with torch.no_grad():
            qkv = torch.nn.functional.linear(hidden.float(), full_qkv)
            QSZ = NUM_Q_HEADS * HEAD_DIM
            KSZ = NUM_KV_HEADS * HEAD_DIM
            q, k, v = qkv.split([QSZ, KSZ, KSZ], dim=-1)
            q, k = apply_qk_norm(
                q.to(torch.bfloat16), k.to(torch.bfloat16), layer.q_norm, layer.k_norm, HEAD_DIM
            )
            q, k = layer.rotary_emb(positions, q, k)
            q = q.view(block_size, NUM_Q_HEADS, HEAD_DIM).float()
            bk = k.view(block_size, NUM_KV_HEADS, HEAD_DIM).float()
            bv = v.view(block_size, NUM_KV_HEADS, HEAD_DIM).float()
            # Full prefix K/V = gather the per-rank kv-head shards.
            pk_all = tp_group.all_gather(prefix_k.contiguous(), dim=1).float()
            pv_all = tp_group.all_gather(prefix_v.contiguous(), dim=1).float()
            fk = torch.cat([pk_all, bk], dim=0)
            fv = torch.cat([pv_all, bv], dim=0)
            rep = NUM_Q_HEADS // NUM_KV_HEADS
            kr = fk.repeat_interleave(rep, dim=1)
            vr = fv.repeat_interleave(rep, dim=1)
            attn = torch.einsum("qhd,khd->hqk", q, kr) * layer.scaling
            attn = torch.softmax(attn, dim=-1)
            ctx = torch.einsum("hqk,khd->qhd", attn, vr).reshape(block_size, NUM_Q_HEADS * HEAD_DIM)
            ref = torch.nn.functional.linear(ctx, full_o)

        diff = float((tp_out - ref).abs().max())
        scale = max(1.0, float(ref.abs().max()))
        if rank == 0 and diff / scale >= TOLERANCE:
            failures.append((prefix_len, block_size, diff / scale))

    if rank == 0:
        ret.put(failures)


def main():
    if os.environ.get("DFLASH_TP2_MANUAL") != "1":
        print(
            "Refusing to run: a standalone 2-rank launch wedges this torch_tpu "
            "stack. Set DFLASH_TP2_MANUAL=1 to force (have a process-kill recovery "
            "plan). DFlash TP=2 head-sharding is covered safely by "
            "test_dflash_layer_attention.py::TestDFlashAttentionLayerParityTP2Shard; "
            "the full multi-rank collective is validated by the e2e server (task16)."
        )
        sys.exit(2)

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    ret = ctx.Queue()
    port = 29531
    procs = [ctx.Process(target=_worker, args=(r, 2, port, ret)) for r in range(2)]
    for p in procs:
        p.start()
    failures = ret.get(timeout=600)
    for p in procs:
        p.join(timeout=600)
    if failures:
        print("TP=2 layer parity FAILED:", failures)
        sys.exit(1)
    print(f"TP=2 layer parity PASSED for all {len(CASES)} cases.")


if __name__ == "__main__":
    main()
