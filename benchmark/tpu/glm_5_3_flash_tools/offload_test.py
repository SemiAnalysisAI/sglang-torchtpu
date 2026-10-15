"""KV offload (HiCache) smoke: python offload_test.py [port] [n_docs].

Sends n_docs distinct ~5k-token prompts cold (greedy), more than the device KV pool
and KDA state slots can keep, then re-sends each. With host offload the evicted
prefixes come back from host: every repeat should report cached_tokens close to its
prompt length and reproduce the cold text, with first-token top-5 logprob gaps at
MoE-noise level. Without offload the early ones would report cached_tokens=0.
"""
import json, random, sys, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "30000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
BASE = f"http://127.0.0.1:{PORT}"


def post(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=900).read()
    return json.loads(raw) if body is not None else raw


def run(text):
    out = post("/generate", {"text": text, "return_logprob": True, "top_logprobs_num": 5,
                             "sampling_params": {"temperature": 0, "max_new_tokens": 24}})
    m = out["meta_info"]
    return out["text"], m["prompt_tokens"], m.get("cached_tokens"), {t[1]: t[0] for t in m["output_top_logprobs"][0]}


def doc(seed):
    rnd = random.Random(seed)
    lines = [f"Entry {i}: sensor {rnd.randrange(50)} read {rnd.randrange(1000)} at minute {rnd.randrange(60)}."
             for i in range(300)]
    return f"Sensor log {seed}.\n" + "\n".join(lines) + "\n\nQuestion: what did entry 7 read? Answer with the number only."


post("/flush_cache")
prompts = [doc(s) for s in range(N)]
cold = [run(p) for p in prompts]
for i, (t, n, c, _) in enumerate(cold):
    print(f"cold {i}: prompt={n} cached={c} text={t!r}")
bad = 0
worst = 0.0
for i, p in enumerate(prompts):
    t, n, c, top = run(p)
    t0, _, _, top0 = cold[i]
    common = set(top) & set(top0)
    gap = max((abs(top[k] - top0[k]) for k in common), default=float("inf"))
    worst = max(worst, gap)
    ok = t == t0 and c is not None and c >= n // 2
    bad += not ok
    print(f"repeat {i}: prompt={n} cached={c} same_text={t == t0} top5_gap={gap:.4f} {'OK' if ok else 'BAD'}"
          + ("" if t == t0 else f" repeat={t!r}"))
print(f"{N - bad}/{N} repeats hit the cache and matched; worst top-5 gap {worst:.4f}")
print("OFFLOAD_TEST_DONE")
