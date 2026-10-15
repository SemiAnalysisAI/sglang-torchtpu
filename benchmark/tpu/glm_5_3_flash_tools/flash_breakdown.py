"""Grouped per-step decode breakdown for GLM-5.3-Flash captures.

usage: python flash_breakdown.py <capture_dir>
Graph time per step comes from the trace (median decode module); hlo_stats self
time is grouped by component and scaled to it.
"""
import collections, html, json, re, sys

D = sys.argv[1]
ev = json.load(open(f"{D}/tools/trace_viewer.json"))["traceEvents"]
tname = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "thread_name"}
pname = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "process_name"}
dev0 = [p for p, n in pname.items() if n.endswith("/device:TPU:0")][0]
mod_tid = [t for (p, t), n in tname.items() if p == dev0 and n == "XLA Modules"][0]
mods = sorted((e for e in ev if e.get("ph") == "X" and e["pid"] == dev0 and e["tid"] == mod_tid), key=lambda e: e["ts"])
per = collections.defaultdict(list)
for m in mods: per[re.sub(r"\(\d+\)", "", m["name"])].append(m)
dec = max(per.values(), key=lambda ms: sum(m["dur"] for m in ms))
durs = sorted(m["dur"] for m in dec); starts = [m["ts"] for m in dec]
gaps = sorted(b - a for a, b in zip(starts, starts[1:]))
graph_ms = durs[len(durs) // 2] / 1e3
print(f"decode graph {graph_ms:.2f} ms/step (median of {len(dec)}), step period {gaps[len(gaps)//2]/1e3:.2f} ms")

t = json.load(open(f"{D}/tools/hlo_stats.json")); t = t[0] if isinstance(t, list) else t
cols = [c.get("label") or c.get("id") for c in t["cols"]]; ix = {c: i for i, c in enumerate(cols)}
rows = [[c.get("v") for c in r["c"]] for r in t["rows"]]
pt = collections.Counter()
for r in rows: pt[r[ix["Program id"]]] += float(r[ix["Total self time (us)"]])
prog = pt.most_common(1)[0][0]; rows = [r for r in rows if r[ix["Program id"]] == prog]

def cat(r):
    s = html.unescape(str(r[ix["Source Info"]] or "")); txt = str(r[ix["HLO op text"]]); fw = str(r[ix["Framework op name"]])
    f = re.findall(r"python/sglang/(srt/[\w/]+\.py):(\d+)", s)
    file, line = (f[0][0], int(f[0][1])) if f else ("", 0)
    if "glm_mhc" in fw or "glm_mhc" in txt or file.endswith("glm_mhc.py"): return "mHC fused kernel"
    if "sc_ragged_gather" in txt or "sc_ragged_gather" in str(r[ix["HLO op name"]]): return "MoE: SparseCore token gather"
    if "gmm_v2" in fw: return "MoE: expert GEMMs (gmm_v2)"
    if "RPAd" in fw or "RPAm" in fw: return "full attn: RPA v3 kernel (11 layers, dense)"
    if "fused_conv1d_kda" in fw: return "KDA: Pallas kernel (34 layers)"
    if r[ix["HLO op category"]] == "all-reduce": return "all-reduce"
    if file.endswith("glm5_next_tpu.py") and 328 <= line <= 356: return "mHC torch reference"
    if file.endswith("glm5_next_tpu.py") and line < 70: return "KDA: gated o_norm"
    if file.endswith("glm5_next_tpu.py") and 440 <= line <= 500: return "mHC glue (model forward)"
    if file.endswith(("moe/topk.py", "fused_moe_op.py", "fp8_moe.py", "mxfp4_flashinfer_trtllm_moe.py", "patches.py")): return "MoE: routing/dispatch glue"
    if file.endswith(("fp8_linear.py", "unquant.py")): return "dense linears (fp8 + bf16)"
    if file.endswith("layernorm.py"): return "RMSNorm"
    if file.endswith("deepseek_v2.py"): return "MoE/MLP glue (deepseek_v2)"
    if not file: return "unattributed (no source)"
    return "other: " + file.rsplit("/", 1)[-1] + f":{line}"

agg = collections.defaultdict(lambda: [0.0, 0.0])
for r in rows:
    k = cat(r); agg[k][0] += float(r[ix["Total self time (us)"]]); agg[k][1] += float(r[ix["#Occurrences"]])
tot = sum(v[0] for v in agg.values()); steps = tot / (graph_ms * 1e3)
print(f"{'ms/step':>8} {'share':>6} {'ops/step':>9}  component")
for k, (us, o) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    print(f"{us/steps/1e3:8.2f} {100*us/tot:5.1f}% {o/steps:9.0f}  {k}")
print(f"{graph_ms:8.2f} 100.0% {sum(v[1] for v in agg.values())/steps:9.0f}  total")
