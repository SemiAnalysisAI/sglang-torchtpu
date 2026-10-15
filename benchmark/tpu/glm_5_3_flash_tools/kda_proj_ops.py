"""KDA-layer projection ops (35/34 per step) from hlo_stats: shape, category, per-step time."""
import collections, html, json, re, sys
D, graph_ms = sys.argv[1], float(sys.argv[2])
t = json.load(open(f"{D}/tools/hlo_stats.json")); t = t[0] if isinstance(t, list) else t
cols = [c.get("label") or c.get("id") for c in t["cols"]]; ix = {c: i for i, c in enumerate(cols)}
rows = [[c.get("v") for c in r["c"]] for r in t["rows"]]
pt = collections.Counter()
for r in rows: pt[r[ix["Program id"]]] += float(r[ix["Total self time (us)"]])
prog = pt.most_common(1)[0][0]; rows = [r for r in rows if r[ix["Program id"]] == prog]
tot = sum(float(r[ix["Total self time (us)"]]) for r in rows); steps = tot / (graph_ms * 1e3)
agg = collections.defaultdict(lambda: [0.0, 0.0])
for r in rows:
    s = html.unescape(str(r[ix["Source Info"]] or "")); f = re.findall(r"python/sglang/(srt/[\w/]+\.py):(\d+)", s)
    occ = float(r[ix["#Occurrences"]]) / steps
    if not (33 <= occ <= 36 or 67 <= occ <= 71): continue
    top = f"{f[0][0].rsplit('/',1)[-1]}:{f[0][1]}" if f else "-"
    if any(k in top for k in ("glm_mhc", "fp8_linear", "fused_moe", "topk", "route_meta", "parallel_state")): continue
    txt = str(r[ix["HLO op text"]]); out = re.search(r"=\s*(\w+\[[^\]]*\])", txt)
    k = f"{top:24s} {r[ix['HLO op category']]:<20} {out.group(1) if out else ''}"
    agg[k][0] += float(r[ix["Total self time (us)"]]); agg[k][1] += float(r[ix["#Occurrences"]])
total = 0
for k, (us, o) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:22]:
    total += us / steps
    print(f"{us/steps/1e3:7.3f} ms {o/steps:5.0f}/step {us/o:6.2f}us  {k[:110]}")
print(f"sum of listed: {total/1e3:.3f} ms")
