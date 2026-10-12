"""Kimi-K3 decode profile breakdown from one rank's xplane.pb.

Finds the device plane, the XLA Modules timeline (one event per program launch), picks the
decode-step program (most frequent large module), and categorizes the XLA ops inside the
decode-step windows. Prints per-step averages.
"""

import collections
import re
import sys

from jax.profiler import ProfileData

path = sys.argv[1]
pd = ProfileData.from_file(path)
planes = [p for p in pd.planes if p.name.startswith("/device:TPU:")]
print("device planes:", [p.name for p in planes])
plane = planes[0]
lines = {l.name: l for l in plane.lines}
print("lines:", list(lines))
mods = [(e.start_ns, e.duration_ns, e.name) for e in lines["XLA Modules"].events]
cnt = collections.Counter(n for _, _, n in mods)
dur = collections.defaultdict(float)
for _, d, n in mods:
    dur[n] += d
print("modules by total time:")
for n, d in sorted(dur.items(), key=lambda kv: -kv[1])[:12]:
    print(
        f"  {d / 1e6:9.2f} ms  {cnt[n]:4d}x  avg {d / cnt[n] / 1e3:9.1f} us  {n[:110]}"
    )
# Decode program: the module with >= 10 launches and the largest average time.
cands = [(dur[n] / cnt[n], n) for n in cnt if cnt[n] >= 10]
avg, decode = max(cands)
win = sorted((s, s + d) for s, d, n in mods if n == decode)
print(f"\ndecode module: {decode[:120]}  launches {len(win)}  avg {avg / 1e6:.3f} ms")
# Step-to-step period between decode launches.
gaps = [win[i + 1][0] - win[i][0] for i in range(len(win) - 1)]
gaps.sort()
print(
    f"decode launch period: median {gaps[len(gaps) // 2] / 1e6:.3f} ms (p10 {gaps[len(gaps) // 10] / 1e6:.3f}, p90 {gaps[9 * len(gaps) // 10] / 1e6:.3f})"
)
CATS = [
    ("all-reduce", r"all-reduce"),
    (
        "all-gather/other collective",
        r"all-gather|reduce-scatter|collective-permute|all-to-all",
    ),
    ("MLA kernel", r"mla|ragged_paged_attention"),
    ("KDA/GDN kernel", r"gdn|kda|chunk_gated|recurrent"),
    (
        "MoE kernel (gmm/fused moe/route)",
        r"gmm|fused_moe|route_meta|sparse_core|ragged_gather|moe",
    ),
    ("attn-res / onorm kernels", r"attn_res|gated_rms|onorm"),
    ("other custom-call", r"custom-call|tpu_custom_call"),
    ("convolution/dot fusion (GEMV)", r"convolution|dot|fusion.*conv"),
    ("copy", r"^copy|%copy|copy-start|copy-done"),
    ("other fusion", r"fusion"),
    ("other", r""),
]
ops = sorted((e.start_ns, e.duration_ns, e.name) for e in lines["XLA Ops"].events)
agg = collections.defaultdict(float)
n_ops = collections.Counter()
top = collections.defaultdict(float)
j = 0
for s, e in win:
    while j < len(ops) and ops[j][0] < s:
        j += 1
    k = j
    while k < len(ops) and ops[k][0] < e:
        st, d, name = ops[k]
        for cat, pat in CATS:
            if re.search(pat, name, re.I):
                agg[cat] += d
                n_ops[cat] += 1
                break
        top[name] += d
        k += 1
steps = len(win)
print(f"\nper decode step (avg over {steps} steps), op time by category:")
tot = sum(agg.values())
for cat, _ in CATS:
    if agg[cat]:
        print(
            f"  {cat:34s} {agg[cat] / steps / 1e3:9.1f} us  ({n_ops[cat] / steps:6.1f} ops/step)  {100 * agg[cat] / tot:5.1f}%"
        )
print(f"  {'TOTAL op time':34s} {tot / steps / 1e3:9.1f} us")
print("\ntop ops by time per step:")
for name, d in sorted(top.items(), key=lambda kv: -kv[1])[:40]:
    print(f"  {d / steps / 1e3:8.1f} us  {name[:150]}")
