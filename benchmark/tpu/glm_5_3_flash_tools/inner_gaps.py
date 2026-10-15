"""Idle gaps inside big modules on TPU:0, attributed to the op that ends right before the gap: python inner_gaps.py <dir>."""
import bisect, collections, json, re, sys
D = sys.argv[1]
ev = json.load(open(f"{D}/tools/trace_viewer.json"))["traceEvents"]
tn = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "thread_name"}
pn = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "process_name"}
dev0 = [p for p, n in pn.items() if n.endswith("/device:TPU:0")][0]
th = lambda e: tn.get((e["pid"], e["tid"]))
X = [e for e in ev if e.get("ph") == "X" and e["pid"] == dev0]
mods = sorted((e for e in X if th(e) == "XLA Modules"), key=lambda e: e["ts"])
ops = sorted((e for e in X if th(e) in ("XLA Ops",)), key=lambda e: e["ts"])
asy = sorted((e for e in X if th(e) == "Async XLA Ops"), key=lambda e: e["ts"])
rounds = sum(1 for m in mods if "greedy_verify" in m["name"])
for key in ("copy_logits", "build_topk1"):
    big = [m for m in mods if key in m["name"]]
    if key == "copy_logits":
        thr = sorted(m["dur"] for m in big)[len(big) // 2] * 0.5; big = [m for m in big if m["dur"] > thr]
    g = collections.defaultdict(list); tot = 0
    for m in big:
        s, t = m["ts"], m["ts"] + m["dur"]
        i = bisect.bisect_left([o["ts"] for o in ops], s) if False else None
        inside = [o for o in ops if s <= o["ts"] <= t] if len(big) < 0 else None
    starts = [o["ts"] for o in ops]
    for m in big:
        s, t = m["ts"], m["ts"] + m["dur"]
        i = bisect.bisect_left(starts, s); j = bisect.bisect_right(starts, t)
        inside = ops[i:j]; end = s; prev = "<module start>"
        for o in inside:
            if o["ts"] - end > 2:
                g[(re.sub(r"\.\d+$", "", prev), re.sub(r"\.\d+$", "", o["name"]))].append(o["ts"] - end); tot += o["ts"] - end
            if o["ts"] + o["dur"] > end:
                end = o["ts"] + o["dur"]; prev = o["name"]
        if t - end > 2:
            g[(re.sub(r"\.\d+$", "", prev), "<module end>")].append(t - end); tot += t - end
    print(f"== {key}: {len(big)} modules, idle inside {tot/rounds/1e3:.3f} ms/round")
    for k, v in sorted(g.items(), key=lambda kv: -sum(kv[1]))[:15]:
        import statistics; print(f"  {sum(v)/rounds/1e3:6.3f} ms/round {len(v)/rounds:6.1f}x  med {statistics.median(v):7.1f}us max {max(v):7.1f}us  {k[0]} -> {k[1]}")
