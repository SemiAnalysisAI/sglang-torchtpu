"""Per-step device-time breakdown of the decode program from hlo_stats.json.

usage: python hlo_breakdown.py <capture_dir> [steps]
Groups self time by HLO category and by source file:line (innermost sglang frame).
"""
import collections, html, json, re, sys

D = sys.argv[1]
t = json.load(open(f"{D}/tools/hlo_stats.json"))
t = t[0] if isinstance(t, list) else t
cols = [c.get("label") or c.get("id") for c in t["cols"]]
ix = {c: i for i, c in enumerate(cols)}
rows = [[c.get("v") for c in r["c"]] for r in t["rows"]]
prog_time = collections.Counter()
for r in rows:
    prog_time[r[ix["Program id"]]] += float(r[ix["Total self time (us)"]])
prog = prog_time.most_common(1)[0][0]
rows = [r for r in rows if r[ix["Program id"]] == prog]
steps = float(sys.argv[2]) if len(sys.argv) > 2 else collections.Counter(
    int(float(r[ix["#Occurrences"]])) for r in rows).most_common(1)[0][0]
total = sum(float(r[ix["Total self time (us)"]]) for r in rows)
print(f"decode program {prog}: {total/steps/1e3:.2f} ms/step over {steps:g} steps, {len(rows)} HLO ops")

def src(r):
    s = html.unescape(str(r[ix["Source Info"]] or ""))
    m = re.findall(r"python/sglang/(srt/[\w/]+\.py):(\d+)", s)
    return f"{m[0][0]}:{m[0][1]}" if m else "(no source)"

def show(title, key, n):
    agg = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        k = key(r); agg[k][0] += float(r[ix["Total self time (us)"]]); agg[k][1] += float(r[ix["#Occurrences"]])
    print(f"\n== {title}\n{'ms/step':>8} {'%':>5} {'ops/step':>8}  key")
    for k, (us, occ) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:n]:
        print(f"{us/steps/1e3:8.3f} {100*us/total:5.1f} {occ/steps:8.0f}  {k}")

show("by HLO category", lambda r: r[ix["HLO op category"]], 25)
show("by source line", src, 45)
show("by source file", lambda r: src(r).rsplit(":", 1)[0], 20)
