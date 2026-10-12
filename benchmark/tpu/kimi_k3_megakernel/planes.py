import collections
import sys

from jax.profiler import ProfileData

pd = ProfileData.from_file(sys.argv[1])
for p in pd.planes:
    ls = {l.name: l for l in p.lines}
    info = []
    for ln in ("XLA Modules", "XLA Ops", "XLA TraceMe"):
        if ln in ls:
            ev = list(ls[ln].events)
            if ev:
                st = min(e.start_ns for e in ev)
                en = max(e.start_ns + e.duration_ns for e in ev)
                info.append(f"{ln}: {len(ev)} ev span {(en - st) / 1e6:.1f}ms")
    print(p.name, "|", "; ".join(info))
    if "XLA Modules" in ls:
        c = collections.Counter(e.name for e in ls["XLA Modules"].events)
        for n, k in c.most_common(6):
            print("    ", k, n[:120])
