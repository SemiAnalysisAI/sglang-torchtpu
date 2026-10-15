"""Per-round device-only time on TPU:0: union of XLA op intervals between consecutive verify starts.

usage: python device_only.py <capture_dir> ...
Reports round period, time inside modules, and time inside ops (all bubbles removed).
"""
import collections, json, re, statistics, sys


def union(iv):
    tot, ce = 0.0, None
    for a, b in sorted(iv):
        if ce is None or a > ce: tot += b - a; ce = b
        elif b > ce: tot += b - ce; ce = b
    return tot


for D in sys.argv[1:]:
    ev = json.load(open(f"{D}/tools/trace_viewer.json"))["traceEvents"]
    tn = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "thread_name"}
    pn = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "process_name"}
    d0 = [p for p, n in pn.items() if n.endswith("/device:TPU:0")][0]
    tids = {n: t for (p, t), n in tn.items() if p == d0}
    X = [e for e in ev if e.get("ph") == "X" and e["pid"] == d0]
    mods = [e for e in X if e["tid"] == tids["XLA Modules"]]
    ops = [e for e in X if e["tid"] == tids.get("XLA Ops")]
    per = collections.defaultdict(list)
    for m in mods: per[re.sub(r"\(\d+\)", "", m["name"])].append(m)
    verify = sorted(max(per.values(), key=lambda ms: sum(m["dur"] for m in ms)), key=lambda m: m["ts"])
    rows = []
    for a, b in zip(verify, verify[1:]):
        t0, t1 = a["ts"], b["ts"]
        clip = lambda es: [(max(e["ts"], t0), min(e["ts"] + e["dur"], t1)) for e in es if e["ts"] < t1 and e["ts"] + e["dur"] > t0]
        rows.append(((t1 - t0) / 1e3, union(clip(mods)) / 1e3, union(clip(ops)) / 1e3))
    med = lambda i: statistics.median(r[i] for r in rows)
    print(f"{D.rstrip('/').split('/')[-1]}: {len(rows)} rounds | period {med(0):.2f} ms | in modules {med(1):.2f} ms "
          f"| device-only (ops) {med(2):.2f} ms | bubbles {med(0)-med(2):.2f} ms "
          f"(between modules {med(0)-med(1):.2f}, inside modules {med(1)-med(2):.2f})")
