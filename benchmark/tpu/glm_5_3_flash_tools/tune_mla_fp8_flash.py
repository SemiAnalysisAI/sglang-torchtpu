"""Block-size tuner for the vendored MLA v2 kernel with an fp8 KV cache (bf16 Q), standalone on one chip.

Sweeps (num_kv_pages_per_block, decode_batch_size) for the batched-decode slot and
(num_kv_pages_per_block, num_queries_per_block) for the MIXED slot, and prints paste-ready rows for
TUNED_BLOCK_SIZES_MLA["TPU v7"] keyed with kv dtype float8_e4m3fn.

  python tune_mla_fp8.py --heads 4 --page-size 128 --decode-bs 1,16,64,128 --kv-len 2048 --mixed-mnt 4096,8192 --mixed-kv-len 8192

For speculative verification, total query tokens are NOT the query length of
one request. Use --mixed-q-len 4 to benchmark mnt/4 independent requests.
These results are labeled VERIFY_ONLY: the production table currently lacks
a per-request query-length key, so do not overwrite general prefill entries.
"""
import argparse, time
from types import SimpleNamespace
import jax, jax.numpy as jnp, numpy as np
import ml_dtypes
import sys; sys.path.insert(0, "/dev/shm/home/qinghanc/work/glm-port-b")
from sglang_tpu.attention.pallas.mla_ragged_paged_attention_v2 import (
    align_to, get_dtype_packing, get_kv_cache_shape, mla_ragged_paged_attention)

KV_LORA, NOPE, ROPE = 512, 128, 64
SM_SCALE = float((NOPE + ROPE) ** -0.5)
MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)
def _cdiv(a, b): return (a + b - 1) // b

def mixed_case_shape(mnt, q_len=None):
    if mnt <= 0 or (q_len is not None and (q_len < 2 or mnt % q_len)):
        raise ValueError("mixed-mnt must be positive and divisible by mixed-q-len (>=2)")
    return (1, mnt) if q_len is None else (mnt // q_len, q_len)

def build_case(bs, q_len, kv_len, *, page_size, heads, kv_dtype, q_dtype=jnp.bfloat16, max_seqs=None, seed=0):
    kvp = get_dtype_packing(kv_dtype); nope_dim = align_to(KV_LORA, 128); rope_dim = align_to(ROPE, 128)
    rng = np.random.default_rng(seed)
    T = bs * q_len; pages_per_seq = _cdiv(kv_len, page_size)
    total_pages = bs * pages_per_seq + 8
    pool = rng.permutation(total_pages).astype(np.int32)
    np_kv = np.dtype(ml_dtypes.float8_e4m3fn) if kv_dtype == jnp.float8_e4m3fn else np.dtype(jnp.bfloat16)
    np_q = np.dtype(jnp.bfloat16) if q_dtype == jnp.bfloat16 else np.dtype(ml_dtypes.float8_e4m3fn)
    cache = np.zeros(get_kv_cache_shape(total_pages, page_size, nope_dim + rope_dim, kv_dtype), dtype=np_kv)
    n_slots = max(bs, max_seqs or bs)
    page_indices = np.zeros((n_slots * pages_per_seq,), dtype=np.int32)
    ql_nope = rng.standard_normal((T, heads, KV_LORA)).astype(np_q)
    q_pe = rng.standard_normal((T, heads, ROPE)).astype(np_q)
    new_kv_c = rng.standard_normal((T, KV_LORA)).astype(np_kv)
    new_k_pe = rng.standard_normal((T, ROPE)).astype(np_kv)
    for i in range(bs):
        seq_pages = pool[i * pages_per_seq:(i + 1) * pages_per_seq]
        page_indices[i * pages_per_seq:(i + 1) * pages_per_seq] = seq_pages
        prefix = kv_len - q_len; pos = np.arange(prefix)
        slots = (seq_pages[pos // page_size], (pos % page_size) // kvp, (pos % page_size) % kvp)
        cache[(*slots, slice(0, KV_LORA))] = rng.standard_normal((prefix, KV_LORA)).astype(np_kv)
        cache[(*slots, slice(nope_dim, nope_dim + ROPE))] = rng.standard_normal((prefix, ROPE)).astype(np_kv)
    kv_lens = np.zeros((n_slots,), np.int32); kv_lens[:bs] = kv_len
    cu_q = np.zeros((n_slots + 1,), np.int32); cu_q[1:bs + 1] = np.arange(1, bs + 1) * q_len; cu_q[bs + 1:] = cu_q[bs]
    cu_kv = np.arange(n_slots + 1, dtype=np.int32) * np.int32(pages_per_seq * page_size)
    dist = np.array([bs, bs, bs], np.int32) if q_len == 1 else np.array([0, 0, bs], np.int32)
    j = jnp.asarray
    return SimpleNamespace(ql_nope=j(ql_nope), q_pe=j(q_pe), new_kv_c=j(new_kv_c), new_k_pe=j(new_k_pe), cache=j(cache),
                           kv_lens=j(kv_lens), page_indices=j(page_indices), cu_q_lens=j(cu_q), cu_kv_lens=j(cu_kv), distribution=j(dist),
                           kv_bytes=bs * kv_len * (nope_dim + rope_dim) * np_kv.itemsize)

def time_kernel(case, *, nkv, nq, dbs, iters=15):
    def run(a, b, c, d, cache):
        return mla_ragged_paged_attention(a, b, c, d, cache, case.kv_lens, case.page_indices, case.cu_q_lens, case.cu_kv_lens,
                                          case.distribution, sm_scale=SM_SCALE, mask_value=MASK_VALUE,
                                          num_kv_pages_per_block=nkv, num_queries_per_block=nq, decode_batch_size=dbs)
    fn = jax.jit(run); args = (case.ql_nope, case.q_pe, case.new_kv_c, case.new_k_pe, case.cache)
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); jax.block_until_ready(fn(*args)); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2] * 1e3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--page-size", type=int, default=128)
    ap.add_argument("--decode-bs", default="1,16,64,128")
    ap.add_argument("--kv-len", type=int, default=2048)
    ap.add_argument("--mixed-mnt", default="")
    ap.add_argument("--mixed-q-len", type=int, default=None,
                    help="Queries per request for verification; default is one request with mnt queries")
    ap.add_argument("--mixed-kv-len", type=int, default=8192)
    ap.add_argument("--bkv", default="4,8,16,32,64")
    ap.add_argument("--dbs", default="1,2,4,8,16,32")
    ap.add_argument("--bq-mixed", default="32,64,128,256")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--kv-dtype", default="fp8", choices=["fp8", "bf16"])
    a = ap.parse_args()
    kvd = jnp.float8_e4m3fn if a.kv_dtype == "fp8" else jnp.bfloat16
    kv_name = jnp.dtype(kvd).name
    ints = lambda s: [int(x) for x in s.split(",") if x]
    for mnt in ints(a.mixed_mnt):
        _, q_len = mixed_case_shape(mnt, a.mixed_q_len)
        if q_len > a.mixed_kv_len:
            ap.error("mixed query length must not exceed mixed-kv-len")
    print(f"# device={jax.devices()[0].device_kind} n={len(jax.devices())} heads={a.heads} page={a.page_size} kv_dtype={kv_name}", flush=True)
    rows = []
    for bs in ints(a.decode_bs):
        case = build_case(bs, 1, a.kv_len, page_size=a.page_size, heads=a.heads, kv_dtype=kvd)
        best = None
        base = time_kernel(case, nkv=(3, 1, 1), nq=(1, 1, 16), dbs=min(4, bs) if bs % min(4, bs) == 0 else 1, iters=a.iters)
        print(f"# decode bs={bs} kv_len={a.kv_len}: hardcoded fallback (3,1,dbs4) {base:.3f} ms ({case.kv_bytes/base/1e6:.0f} GB/s)", flush=True)
        for bkv in ints(a.bkv):
            for dbs in ints(a.dbs):
                if dbs > bs or bs % dbs: continue
                try:
                    t = time_kernel(case, nkv=(bkv, 1, 1), nq=(1, 1, 16), dbs=dbs, iters=a.iters)
                except Exception as e:  # noqa
                    print(f"#   bkv={bkv} dbs={dbs} FAIL {type(e).__name__}: {str(e)[:80]}", flush=True); continue
                print(f"#   bkv={bkv} dbs={dbs} {t:.3f} ms", flush=True)
                if best is None or t < best[0]: best = (t, bkv, dbs)
        if best:
            t, bkv, dbs = best
            rows.append(f'        ("decode", "bfloat16", "{kv_name}", {a.heads}, 512, 64, {a.page_size}, {bs}): ({bkv}, 1, {dbs}),  # {t:.3f} ms vs fallback {base:.3f}')
            print(rows[-1], flush=True)
    for mnt in ints(a.mixed_mnt):
        bs, q_len = mixed_case_shape(mnt, a.mixed_q_len)
        case = build_case(bs, q_len, a.mixed_kv_len, page_size=a.page_size, heads=a.heads, kv_dtype=kvd)
        best = None
        base = time_kernel(case, nkv=(3, 1, 1), nq=(1, 1, 16), dbs=1, iters=max(3, a.iters // 3))
        print(f"# mixed bs={bs} q_len={q_len} mnt={mnt} kv_len={a.mixed_kv_len}: hardcoded fallback (1,16) {base:.3f} ms", flush=True)
        if a.mixed_q_len is not None:
            current = time_kernel(case, nkv=None, nq=None, dbs=1, iters=a.iters)
            print(f"# VERIFY_ONLY current production table: {current:.3f} ms", flush=True)
        for bkv in ints(a.bkv):
            for bq in ints(a.bq_mixed):
                if bq > mnt: continue
                try:
                    t = time_kernel(case, nkv=(3, 1, bkv), nq=(1, 1, bq), dbs=1, iters=max(3, a.iters // 3))
                except Exception as e:  # noqa
                    print(f"#   bkv={bkv} bq={bq} FAIL {type(e).__name__}: {str(e)[:80]}", flush=True); continue
                print(f"#   bkv={bkv} bq={bq} {t:.3f} ms", flush=True)
                if best is None or t < best[0]: best = (t, bkv, bq)
        if best:
            t, bkv, bq = best
            row = f'        ("mixed", "bfloat16", "{kv_name}", {a.heads}, 512, 64, {a.page_size}, {mnt}): ({bkv}, {bq}),  # {t:.3f} ms vs fallback {base:.3f}'
            if a.mixed_q_len is not None:
                row = f'# VERIFY_ONLY bs={bs} q_len={q_len} kv_len={a.mixed_kv_len}: {row.strip()}'
            rows.append(row)
            print(rows[-1], flush=True)
    print("\n# ===== paste into TUNED_BLOCK_SIZES_MLA['TPU v7'] =====")
    for r in rows: print(r)

if __name__ == "__main__":
    main()
