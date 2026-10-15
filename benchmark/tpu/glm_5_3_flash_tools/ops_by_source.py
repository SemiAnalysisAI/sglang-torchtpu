"""Device op time inside the biggest module, grouped by Python source line: python ops_by_source.py <dir> [top]."""
import bisect, collections, json, re, sys
D = sys.argv[1]; TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 40
ev = json.load(open(f"{D}/tools/trace_viewer.json"))["traceEvents"]
tn = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "thread_name"}
pn = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "process_name"}
dev0 = [p for p, n in pn.items() if n.endswith("/device:TPU:0")][0]
X = [e for e in ev if e.get("ph") == "X" and e["pid"] == dev0]
th = lambda e: tn.get((e["pid"], e["tid"]))
mods = [e for e in X if th(e) == "XLA Modules"]
big = max(mods, key=lambda m: m["dur"])
ops = [e for e in X if th(e) == "XLA Ops" and big["ts"] <= e["ts"] <= big["ts"] + big["dur"]]
src = sorted([e for e in X if th(e) == "Source code"], key=lambda e: e["ts"])
starts = [s["ts"] for s in src]
agg = collections.defaultdict(float); cnt = collections.Counter(); kinds = collections.defaultdict(collections.Counter)
for o in ops:
    i = bisect.bisect_right(starts, o["ts"] + 0.01) - 1
    line = "?"
    while i >= 0 and i > bisect.bisect_right(starts, o["ts"]) - 50:
        s = src[i]
        if s["ts"] + s["dur"] >= o["ts"] + o["dur"] - 0.01:
            line = re.sub(r".*/(glm-port[^/]*)/", "", s["name"]); break
        i -= 1
    agg[line] += o["dur"]; cnt[line] += 1; kinds[line][re.sub(r"\.\d+$", "", o["name"])[:30]] += o["dur"]
tot = sum(agg.values())
for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:TOP]:
    top = ", ".join(f"{n}:{d/1e3:.1f}" for n, d in kinds[k].most_common(3))
    print(f"{v/1e3:8.2f} ms {100*v/tot:5.1f}% {cnt[k]:5d}x  {k[:75]:75s} [{top}]")
