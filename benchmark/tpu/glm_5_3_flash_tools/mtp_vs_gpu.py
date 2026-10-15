"""Module-by-module MTP round comparison: TPU capture (xprof) vs GPU torch-profiler trace.

usage: python mtp_vs_gpu.py <tpu_capture_dir> <gpu_trace.json>
TPU: programs ranked by self time = verify, fused draft, draft extend (hlo_stats self time / rounds).
GPU: phases from main-stream gpu_user_annotation; steady rounds only (first round dropped).
"""
import collections, html, json, re, statistics, sys

COMPS = ["MoE expert GEMM", "MoE routing", "shared experts", "dense GEMM (attn/KDA proj)", "lm_head",
         "all-reduce", "mHC", "KDA kernel", "KDA o_norm / gate", "attention kernel", "DSA indexer + top-k",
         "norms", "elementwise / other"]


def tpu(D):
    ev = json.load(open(f"{D}/tools/trace_viewer.json"))["traceEvents"]
    tn = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "thread_name"}
    pn = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "process_name"}
    d0 = [p for p, n in pn.items() if n.endswith("/device:TPU:0")][0]
    mt = [t for (p, t), n in tn.items() if p == d0 and n == "XLA Modules"][0]
    per = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and e["pid"] == d0 and e["tid"] == mt: per[e["name"]].append(e)
    n = len(max(per.values(), key=lambda ms: sum(m["dur"] for m in ms)))
    t = json.load(open(f"{D}/tools/hlo_stats.json")); t = t[0] if isinstance(t, list) else t
    cols = [c.get("label") for c in t["cols"]]; ix = {c: i for i, c in enumerate(cols)}
    rows = [[c.get("v") for c in r["c"]] for r in t["rows"]]
    pt = collections.Counter()
    for r in rows: pt[r[ix["Program id"]]] += float(r[ix["Total self time (us)"]])
    progs = [p for p, _ in pt.most_common(3)]
    names = {progs[0]: "verify", progs[1]: "draft", progs[2]: "draft_extend"}

    def cat(r):
        s = html.unescape(str(r[ix["Source Info"]] or "")); fw = str(r[ix["Framework op name"]]); txt = str(r[ix["HLO op text"]])
        f = re.findall(r"python/sglang/(srt/[\w/]+\.py):(\d+)", s)
        file, line = (f[0][0], int(f[0][1])) if f else ("", 0)
        if "glm_mhc" in fw or "glm_mhc" in txt: return "mHC"
        if "gmm_v2" in fw or "moe_stream" in fw: return "MoE expert GEMM"
        if "fused_conv1d_kda" in fw: return "KDA kernel"
        if r[ix["HLO op category"]] == "all-reduce": return "all-reduce"
        if file.endswith(("route_meta.py", "router_topk_small_m.py", "moe/topk.py", "fused_moe_op.py", "fp8_moe.py", "patches.py")): return "MoE routing"
        if file.endswith(("fp8_linear.py", "unquant.py", "fp8_linear_small_m.py")): return "dense GEMM (attn/KDA proj)"
        if "mla_ragged" in file or "RPA" in fw: return "attention kernel"
        if file.endswith("glm5_next_tpu.py") and line < 70: return "KDA o_norm / gate"
        if file.endswith("layernorm.py"): return "norms"
        if file.endswith(("logits_processor.py", "vocab_parallel_embedding.py")): return "lm_head"
        return "elementwise / other"

    out = {ph: collections.Counter() for ph in names.values()}
    calls = collections.Counter()
    for r in rows:
        p = r[ix["Program id"]]
        if p in names:
            c = cat(r); out[names[p]][c] += float(r[ix["Total self time (us)"]]) / n / 1e3
            if names[p] == "verify": calls[c] += float(r[ix["#Occurrences"]]) / n
    return n, out, calls


def gpu(path):
    ev = json.load(open(path))["traceEvents"]
    ga = sorted((e for e in ev if e.get("cat") == "gpu_user_annotation"), key=lambda e: e["ts"])
    main = collections.Counter(e["tid"] for e in ga if e["name"] == "draft").most_common(1)[0][0]
    ph = [e for e in ga if e["tid"] == main and (e["name"] in ("draft", "draft_extend") or e["name"].startswith("step["))]
    starts = [e["ts"] for e in ph if e["name"] == "draft"]
    kern = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    side = collections.Counter(e["tid"] for e in kern).most_common(2)[1][0]

    def cat(e):
        n = e["name"]
        if re.search(r"all_reduce|all_gather|nccl", n): return "all-reduce"
        if re.search(r"^bmm_|moe::dev::finalize|activationDeepSeek", n): return "MoE expert GEMM"
        if re.search(r"routing|tiny_n_gemm", n): return "MoE routing"
        if re.search(r"mhc|hc_prenorm|fused_eh_norm", n): return "mHC"
        if "kda" in n: return "KDA kernel"
        if "layer_norm_gated" in n: return "KDA o_norm / gate"
        if re.search(r"kpool|mqa_logits|topk_transform", n): return "DSA indexer + top-k"
        if "fmha" in n: return "attention kernel"
        if e["tid"] == side and re.search(r"gemm|quant|silu|nvjet|cublas|splitK", n): return "shared experts"
        if "nvjet_sm103_tst_256x8" in n: return "lm_head"
        if re.search(r"gemm|quant|TgvGemm|nvjet|cublas|cutlass_80|splitK|Float8", n): return "dense GEMM (attn/KDA proj)"
        if re.search(r"norm", n, re.I): return "norms"
        if re.search(r"silu", n): return "shared experts" if e["tid"] == side else "elementwise / other"
        return "elementwise / other"

    out = {p: collections.Counter() for p in ("draft", "verify", "draft_extend")}
    calls = collections.Counter(); walls = collections.defaultdict(list)
    rounds = list(zip(starts, starts[1:]))[1:]
    for t0, t1 in rounds:
        win = {}
        for e in ph:
            if t0 <= e["ts"] < t1:
                p = "verify" if e["name"].startswith("step") else e["name"]
                win[p] = (e["ts"], e["ts"] + e["dur"]); walls[p].append(e["dur"] / 1e3)
        for k in kern:
            for p, (a, b) in win.items():
                if a <= k["ts"] < b:
                    c = cat(k); out[p][c] += k["dur"] / 1e3 / len(rounds)
                    if p == "verify": calls[c] += 1 / len(rounds)
    return len(rounds), out, calls, {p: statistics.median(v) for p, v in walls.items()}


nt, T, Tc = tpu(sys.argv[1])
ng, G, Gc, Gw = gpu(sys.argv[2])
print(f"rounds: TPU {nt}, GPU {ng} (steady)\n")
for p in ("draft", "verify", "draft_extend"):
    print(f"== {p}" + (f"   (GPU wall {Gw[p]:.2f} ms)" if p in Gw else ""))
    print(f"   {'component':28s} {'TPU ms':>7} {'GPU ms':>7} {'TPU-GPU':>8}" + ("   calls TPU/GPU" if p == "verify" else ""))
    for c in COMPS:
        a, b = T[p].get(c, 0.0), G[p].get(c, 0.0)
        if a < 0.005 and b < 0.005: continue
        extra = f"   {Tc[c]:5.0f}/{Gc[c]:<5.0f}" if p == "verify" else ""
        print(f"   {c:28s} {a:7.2f} {b:7.2f} {a-b:+8.2f}{extra}")
    print(f"   {'TOTAL kernel time':28s} {sum(T[p].values()):7.2f} {sum(G[p].values()):7.2f} {sum(T[p].values())-sum(G[p].values()):+8.2f}\n")
