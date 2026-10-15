"""Per-round component time of the two largest device programs (verify, draft) in TPU spec captures.

usage: python spec_compare.py <capture_dir> [<capture_dir> ...]
ms/round = program self time in hlo_stats / number of rounds (instances of the largest module).
"""
import collections, html, json, re, sys


def cat(r, ix):
    s = html.unescape(str(r[ix["Source Info"]] or "")); txt = str(r[ix["HLO op text"]]); fw = str(r[ix["Framework op name"]])
    f = re.findall(r"python/sglang/(srt/[\w/]+\.py):(\d+)", s)
    file, line = (f[0][0], int(f[0][1])) if f else ("", 0)
    if "glm_mhc" in fw or "glm_mhc" in txt or file.endswith("glm_mhc.py"): return "mHC"
    if "gmm_v2" in fw: return "MoE expert GEMMs"
    if "moe_stream" in fw or file.endswith("moe_stream_bs1.py"): return "MoE expert GEMMs"
    if "fused_conv1d_kda" in fw: return "KDA kernel"
    if r[ix["HLO op category"]] == "all-reduce": return "all-reduce"
    if file.endswith(("route_meta.py", "router_topk_small_m.py", "moe/topk.py", "fused_moe_op.py", "fp8_moe.py", "patches.py")): return "MoE routing"
    if file.endswith(("fp8_linear.py", "unquant.py", "fp8_linear_small_m.py")): return "dense linears"
    if "mla_ragged" in file or "RPA" in fw: return "attention"
    if file.endswith("glm5_next_tpu.py") and line < 70: return "KDA o_norm"
    if file.endswith("layernorm.py"): return "RMSNorm"
    if file.endswith("logits_processor.py") or file.endswith("vocab_parallel_embedding.py"): return "lm_head / embed"
    return "other"


def analyze(D):
    ev = json.load(open(f"{D}/tools/trace_viewer.json"))["traceEvents"]
    tn = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "thread_name"}
    pn = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "process_name"}
    d0 = [p for p, n in pn.items() if n.endswith("/device:TPU:0")][0]
    mt = [t for (p, t), n in tn.items() if p == d0 and n == "XLA Modules"][0]
    per = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and e["pid"] == d0 and e["tid"] == mt:
            per[re.sub(r"\(\d+\)", "", e["name"])].append(e)
    n_rounds = len(max(per.values(), key=lambda ms: sum(m["dur"] for m in ms)))
    t = json.load(open(f"{D}/tools/hlo_stats.json")); t = t[0] if isinstance(t, list) else t
    cols = [c.get("label") or c.get("id") for c in t["cols"]]; ix = {c: i for i, c in enumerate(cols)}
    rows = [[c.get("v") for c in r["c"]] for r in t["rows"]]
    pt = collections.Counter()
    for r in rows: pt[r[ix["Program id"]]] += float(r[ix["Total self time (us)"]])
    out = {}
    for rank, (prog, _) in enumerate(pt.most_common(2)):
        agg = collections.Counter()
        for r in rows:
            if r[ix["Program id"]] == prog:
                agg[cat(r, ix)] += float(r[ix["Total self time (us)"]]) / n_rounds / 1e3
        out[("verify", "draft")[rank]] = agg
    return n_rounds, out


res = {D.rstrip("/").split("/")[-1]: analyze(D) for D in sys.argv[1:]}
for prog in ("verify", "draft"):
    comps = sorted({c for _, o in res.values() for c in o[prog]}, key=lambda c: -max(o[prog][c] for _, o in res.values()))
    print(f"\n[{prog}] ms/round  " + "  ".join(f"{k[:28]:>28}" for k in res))
    for c in comps:
        print(f"  {c:18s}" + "  ".join(f"{o[prog][c]:28.2f}" for _, o in res.values()))
    print(f"  {'TOTAL':18s}" + "  ".join(f"{sum(o[prog].values()):28.2f}" for _, o in res.values()))
