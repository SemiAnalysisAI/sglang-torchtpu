"""Prefix-cache correctness: does a cached prefix give the same next-token distribution as a
fresh prefill? python prefix_logprob_test.py [port].

For each prompt: flush the cache, run it cold, run it again (prefix now cached), then a
variant that shares the prefix. Compares the first generated token's top-5 logprobs between
cold and repeat (the same request). A correct cache gives matching tokens and logprob gaps at
the level of bf16 noise (~1e-2); a broken restore gives different tokens or gaps of ~1 or more.
"""
import json, random, sys, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "30000"
BASE = f"http://127.0.0.1:{PORT}"


def post(path, body=None, timeout=900):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout)) if body is not None else urllib.request.urlopen(req, timeout=timeout).read()


def first_token(text):
    out = post("/generate", {"text": text, "return_logprob": True, "top_logprobs_num": 5,
                             "sampling_params": {"temperature": 0, "max_new_tokens": 8}})
    meta = out["meta_info"]
    top = {t[1]: t[0] for t in meta["output_top_logprobs"][0]}
    return out["text"], meta.get("cached_tokens"), top


def doc(seed, n_lines):
    rnd = random.Random(seed)
    lines = [f"Entry {i}: sensor {rnd.randrange(50)} read {rnd.randrange(1000)} at minute {rnd.randrange(60)}." for i in range(n_lines)]
    return "Sensor log.\n" + "\n".join(lines) + "\n\n"


import os
LINES = [int(x) for x in os.environ.get("LINES", "120,300,600").split(",")]
cases = [(doc(s, n), q) for s, n in zip(range(1, 100), LINES) for q in
         ("Question: what did entry 7 read? Answer with the number only.",)]
worst = 0.0
for i, (d, q) in enumerate(cases):
    post("/flush_cache", None)
    t_cold, c_cold, top_cold = first_token(d + q)
    t_rep, c_rep, top_rep = first_token(d + q)
    common = set(top_cold) & set(top_rep)
    gap = max((abs(top_cold[t] - top_rep[t]) for t in common), default=float("inf"))
    worst = max(worst, gap)
    print(f"case {i}: cached cold={c_cold} repeat={c_rep} same_text={t_cold == t_rep} "
          f"top1 cold={max(top_cold, key=top_cold.get)} repeat={max(top_rep, key=top_rep.get)} "
          f"max_logprob_gap={gap:.4f} cold={t_cold!r} repeat={t_rep!r}")
print(f"worst top-5 logprob gap: {worst:.4f}")
print("PREFIX_LOGPROB_DONE")
