"""KV offload under load: python offload_stress.py [port] [n_docs] [rounds] [conc].

n_docs distinct ~5k-token prompts (more than the device pool keeps). Round 0 runs
each cold one at a time and records its greedy text. Later rounds send all of them
in a shuffled order, `conc` at a time, plus a shared-prefix variant per doc, so host
reloads, write-throughs and evictions overlap running forwards. Every repeat must
reproduce its cold text; the summary also reports how many tokens were cached.
"""
import json, random, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

PORT = sys.argv[1] if len(sys.argv) > 1 else "30000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 4
BASE = f"http://127.0.0.1:{PORT}"


def post(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=1800).read()
    return json.loads(raw) if body is not None else raw


def run(text):
    out = post("/generate", {"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": 24}})
    return out["text"], out["meta_info"]["prompt_tokens"], out["meta_info"].get("cached_tokens") or 0


def doc(seed):
    rnd = random.Random(seed)
    return f"Sensor log {seed}.\n" + "\n".join(
        f"Entry {i}: sensor {rnd.randrange(50)} read {rnd.randrange(1000)} at minute {rnd.randrange(60)}."
        for i in range(300)) + "\n\n"


QS = ("Question: what did entry 7 read? Answer with the number only.",
      "Question: what did entry 12 read? Answer with the number only.")
post("/flush_cache")
prompts = [doc(s) + q for s in range(N) for q in QS]
cold = {p: run(p)[0] for p in prompts}
print(f"cold: {len(cold)} prompts")
bad = 0
for r in range(1, ROUNDS + 1):
    order = prompts[:]
    random.Random(r).shuffle(order)
    with ThreadPoolExecutor(CONC) as ex:
        res = list(ex.map(run, order))
    mism = [(i, t) for (t, _, _), p, i in zip(res, order, range(len(order))) if t != cold[p]]
    bad += len(mism)
    cached = sum(c for _, _, c in res); total = sum(n for _, n, _ in res)
    print(f"round {r}: {len(order) - len(mism)}/{len(order)} match cold, cached {cached}/{total} tokens "
          f"({cached / total:.0%})" + (f", first mismatch {mism[0]}" if mism else ""))
print(f"{'PASS' if bad == 0 else 'FAIL'}: {bad} mismatches")
print("OFFLOAD_STRESS_DONE")
