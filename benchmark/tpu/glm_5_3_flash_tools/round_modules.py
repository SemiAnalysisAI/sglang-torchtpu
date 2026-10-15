"""Per-round device-module summary for a DFlash decode-only capture: usage python round_modules.py <dir>."""
import json, re, collections, sys
D = sys.argv[1]
ev = json.load(open(f"{D}/tools/trace_viewer.json"))["traceEvents"]
tname = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "thread_name"}
pname = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "process_name"}
dev0 = [p for p, n in pname.items() if n.endswith("/device:TPU:0")][0]
mt = [t for (p, t), n in tname.items() if p == dev0 and n == "XLA Modules"][0]
mods = sorted((e for e in ev if e.get("ph") == "X" and e["pid"] == dev0 and e["tid"] == mt), key=lambda e: e["ts"])
agg = collections.defaultdict(lambda: [0, 0.0])
for m in mods:
    k = re.sub(r"\(\d+\)", "", m["name"])[:80]; agg[k][0] += 1; agg[k][1] += m["dur"]
verify = max(agg, key=lambda k: agg[k][1]); rounds = agg[verify][0]
span = (mods[-1]["ts"] + mods[-1]["dur"] - mods[0]["ts"]) / 1e3; busy = sum(v[1] for v in agg.values()) / 1e3
print(f"{rounds} rounds in {span:.1f} ms: {span/rounds:.2f} ms/round, device busy {100*busy/span:.0f}% (idle {(span-busy)/rounds:.2f} ms/round)")
for k, (n, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:10]:
    print(f"  {d/rounds/1e3:7.3f} ms/round  {n/rounds:5.1f}x  {k}")
