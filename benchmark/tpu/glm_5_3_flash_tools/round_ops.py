"""Per-round device op breakdown on TPU:0 (rounds = greedy-verify launches): python round_ops.py <capture_dir> [topN]."""
import bisect, collections, json, re, sys
D = sys.argv[1]; TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 40
ev = json.load(open(f"{D}/tools/trace_viewer.json"))["traceEvents"]
tn = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "thread_name"}
pn = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "process_name"}
dev0 = [p for p, n in pn.items() if n.endswith("/device:TPU:0")][0]
th = lambda e: tn.get((e["pid"], e["tid"]))
X = [e for e in ev if e.get("ph") == "X" and e["pid"] == dev0]
mods = sorted((e for e in X if th(e) == "XLA Modules"), key=lambda e: e["ts"])
ops = [e for e in X if th(e) == "XLA Ops"]
rounds = sum(1 for m in mods if "greedy_verify" in m["name"])
starts = [m["ts"] for m in mods]
# Label each big module by size order within a round: the largest copy_logits module is the verify.
def modname(m):
    n = re.sub(r"\(\d+\)", "", m["name"])
    return n[:70]
mod_of = []
for o in ops:
    i = bisect.bisect_right(starts, o["ts"]) - 1
    m = mods[i] if i >= 0 and o["ts"] <= mods[i]["ts"] + mods[i]["dur"] else None
    mod_of.append(m)
# classify big modules by duration: verify vs draft-extend
big = [m for m in mods if "copy_logits" in m["name"]]
thr = sorted(m["dur"] for m in big)[len(big) // 2] * 0.5 if big else 0
def label(m):
    if m is None: return "?"
    if "copy_logits" in m["name"]:
        return "VERIFY" if m["dur"] > thr else "DRAFT_EXTEND"
    if "build_topk1" in m["name"]: return "DRAFT_LOOP"
    return "small:" + modname(m)[7:40]
agg = collections.defaultdict(float); cnt = collections.Counter(); bymod = collections.defaultdict(float)
for o, m in zip(ops, mod_of):
    k = (label(m), re.sub(r"\.\d+$", "", o["name"]))
    agg[k] += o["dur"]; cnt[k] += 1; bymod[label(m)] += o["dur"]
print(f"{rounds} rounds; module wall per round:")
mw = collections.defaultdict(float)
for m in mods: mw[label(m)] += m["dur"]
for k, v in sorted(mw.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {v/rounds/1e3:7.3f} ms wall  {bymod[k]/rounds/1e3:7.3f} ms ops  {k}")
print("top ops per round:")
for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:TOP]:
    print(f"  {v/rounds/1e3:7.3f} ms  {cnt[k]/rounds:6.1f}x  {k[0]:13s} {k[1]}")
