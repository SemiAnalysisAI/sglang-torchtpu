"""Per-round phase and component breakdown of an sglang CUDA torch-profiler decode trace (MTP/EAGLE).

usage: python gpu_round_breakdown.py <trace.json>
Phases come from the main-stream gpu_user_annotation spans (draft / step[VERIFY] / draft_extend);
kernels on any stream are assigned to the phase whose window contains their start.
"""
import collections, json, re, statistics, sys

ev = json.load(open(sys.argv[1]))["traceEvents"]
ga = sorted((e for e in ev if e.get("cat") == "gpu_user_annotation"), key=lambda e: e["ts"])
main = collections.Counter(e["tid"] for e in ga if e["name"] == "draft").most_common(1)[0][0]
spans = [e for e in ga if e["tid"] == main and e["name"] in ("draft", "draft_extend") or
         (e["tid"] == main and e["name"].startswith("step["))]
kern = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])

RULES = [
    ("all-reduce / comm", r"all_reduce|allreduce|all_gather|nccl|reduce_scatter|one_shot|two_shot"),
    ("MoE experts GEMM", r"^bmm_|moe::dev::finalize|activationDeepSeek|moe::dev::(?!routing)"),
    ("MoE routing", r"routing|topk(?!_transform)|tiny_n_gemm"),
    ("mHC", r"mhc|hc_prenorm|fused_eh_norm"),
    ("KDA", r"kda|layer_norm_gated|causal_conv|chunk_gated|fused_recurrent"),
    ("DSA indexer / top-k", r"kpool|mqa_logits|indexer|topk_transform"),
    ("MLA/DSA attention", r"fmha|flashmla|mla|attention|decode_attention"),
    ("dense FP8 GEMM", r"deep_gemm::sm100_fp8|per_token_group_quant|TgvGemm|ConvertToFloat8"),
    ("dense bf16 GEMM / lm_head", r"nvjet|cublas|cutlass_80_simt|splitKreduce|gemm"),
    ("norms", r"rmsnorm|RMSNorm|layernorm|norm"),
    ("activation", r"silu|gelu"),
    ("elementwise / copy / index", r"elementwise|index|Copy|copy|Fill|cat|memcpy|memset|reduce_kernel"),
]


def comp(name):
    for c, p in RULES:
        if re.search(p, name, re.I):
            return c
    return "other"


rounds = []
cur = None
for s in spans:
    if s["name"] == "draft":
        cur = {"start": s["ts"]}
        rounds.append(cur)
    if cur is not None:
        cur[s["name"].split("[")[0]] = (s["ts"], s["ts"] + s["dur"])
rounds = [r for r in rounds if {"draft", "step", "draft_extend"} <= r.keys()]
for a, b in zip(rounds, rounds[1:]):
    a["period"] = b["start"] - a["start"]
full = [r for r in rounds if "period" in r]
print(f"{len(full)} complete rounds; period median {statistics.median(r['period'] for r in full)/1000:.2f} ms")

phase_ms = collections.defaultdict(list)
busy = collections.defaultdict(list)
comp_ms = collections.defaultdict(lambda: collections.defaultdict(float))
for r in full:
    for ph in ("draft", "step", "draft_extend"):
        t0, t1 = r[ph]
        phase_ms[ph].append((t1 - t0) / 1000)
    end = r["start"] + r["period"]
    for k in kern:
        if not (r["start"] <= k["ts"] < end):
            continue
        ph = next((p for p in ("draft", "step", "draft_extend") if r[p][0] <= k["ts"] < r[p][1]), "between")
        comp_ms[ph][comp(k["name"])] += k["dur"] / 1000 / len(full)
    # main-stream busy time (union of kernel intervals) in the round
    iv = sorted((k["ts"], k["ts"] + k["dur"]) for k in kern if r["start"] <= k["ts"] < end)
    tot, ce = 0, None
    for a, b in iv:
        if ce is None or a > ce:
            tot += b - a; ce = b
        elif b > ce:
            tot += b - ce; ce = b
    busy["gpu busy (any stream)"].append(tot / 1000)
print("phase wall (GPU annotation) per round, median ms:")
for ph, v in phase_ms.items():
    print(f"  {ph:14s} {statistics.median(v):6.2f}")
print(f"  gpu busy union {statistics.median(busy['gpu busy (any stream)']):6.2f} of period")
for ph in ("draft", "step", "draft_extend", "between"):
    d = comp_ms[ph]
    if not d:
        continue
    print(f"\n[{ph}] kernel time per round {sum(d.values()):.2f} ms")
    for c, v in sorted(d.items(), key=lambda x: -x[1]):
        print(f"  {v:6.2f} ms  {c}")
