"""Compare greedy output dumps from gen_dump.py: where do texts first diverge?

usage: python cmp_dump.py nospec.json dflash.json
"""
import json, sys


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def show(label, a, b):
    i = first_diff(a, b)
    if i is None:
        return f"{label}: identical ({len(a)} chars)"
    ctx = lambda s: repr(s[max(0, i - 40):i + 40])  # noqa: E731
    return (f"{label}: diverge at char {i}/{len(a)} ({100 * i / max(len(a), 1):.0f}%)\n"
            f"      A: {ctx(a)}\n      B: {ctx(b)}")


ns, df = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for i, kind in enumerate(ns["kinds"]):
    print(f"===== prompt {i} ({kind}); DFlash accept seq {df['seq'][i]['tokens'] / max(df['seq'][i].get('verify') or 1, 1):.2f}"
          if df["seq"][i].get("verify") else f"===== prompt {i} ({kind})")
    print(show("  nospec bs1 vs nospec bs8 ", ns["seq"][i]["text"], ns["par"][i]["text"]))
    print(show("  nospec bs1 vs DFlash bs1 ", ns["seq"][i]["text"], df["seq"][i]["text"]))
    print(show("  nospec bs8 vs DFlash bs8 ", ns["par"][i]["text"], df["par"][i]["text"]))
