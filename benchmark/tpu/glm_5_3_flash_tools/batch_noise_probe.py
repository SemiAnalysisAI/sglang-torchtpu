"""Is the split-prefill logprob gap explained by batch-shape numerics alone?
python batch_noise_probe.py [port]

For each prompt (cache flushed every time, so nothing is split or reused):
  alone  : the prompt by itself in its prefill
  batched: the same prompt sent together with an unrelated filler prompt, so it shares
           a larger prefill batch (different compiled shape, same math)
  split  : the same prompt after its first ~half was cached (two-piece prefill)
Prints max |delta| over the common top-5 first-token logprobs for alone-vs-alone (run
twice: determinism), alone-vs-batched (shape noise, no state carry) and alone-vs-split.
"""
import json, random, sys, threading, urllib.request

BASE = f"http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else '30000'}"


def post(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=900).read()
    return json.loads(raw) if body is not None else raw


def top5(text):
    out = post("/generate", {"text": text, "return_logprob": True, "top_logprobs_num": 5,
                             "sampling_params": {"temperature": 0, "max_new_tokens": 1}})
    return {t[1]: t[0] for t in out["meta_info"]["output_top_logprobs"][0]}, out["meta_info"].get("cached_tokens")


def gap(a, b):
    common = set(a) & set(b)
    return max((abs(a[k] - b[k]) for k in common), default=float("inf"))


def doc(seed, n):
    rnd = random.Random(seed)
    return f"Sensor log {seed}.\n" + "\n".join(
        f"Entry {i}: sensor {rnd.randrange(50)} read {rnd.randrange(1000)} at minute {rnd.randrange(60)}."
        for i in range(n)) + "\n\nQuestion: what did entry 7 read? Answer with the number only."


for seed, n in ((1, 120), (2, 300), (3, 600)):
    p = doc(seed, n)
    filler = doc(100 + seed, n)
    post("/flush_cache"); a1, _ = top5(p)
    post("/flush_cache"); a2, _ = top5(p)
    post("/flush_cache")
    out = post("/generate", {"text": [p, filler], "return_logprob": True, "top_logprobs_num": 5,
                             "sampling_params": {"temperature": 0, "max_new_tokens": 1}})
    b = {t[1]: t[0] for t in out[0]["meta_info"]["output_top_logprobs"][0]}
    post("/flush_cache"); top5(p[: len(p) // 2]); s, cached = top5(p)
    print(f"~{len(p) // 4} tok: alone-vs-alone={gap(a1, a2):.4f}  alone-vs-batched={gap(a1, b):.4f}  "
          f"alone-vs-split(cached {cached})={gap(a1, s):.4f}")
print("BATCH_NOISE_DONE")
