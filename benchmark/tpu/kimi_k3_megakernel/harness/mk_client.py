"""mk_client.py <cfg_dir> <base_url> <batch> <isl> <osl> <rounds> [validate]

Drive the megakernel's target-only server: optional greedy sanity prompts (batch 1 only),
then <rounds> batches of <batch> concurrent requests with <isl>-token random prompts and
<osl> output tokens (ignore_eos). Writes client.json; decode timing comes from the server's
'target_batch:' log lines (device time per batch), parsed by mk_driver.sh.
"""

import json
import random
import sys
import threading
import time
import urllib.request

cfg_dir, base, batch, isl, osl, rounds = (
    sys.argv[1],
    sys.argv[2],
    *map(int, sys.argv[3:7]),
)
validate = len(sys.argv) > 7 and sys.argv[7] == "validate"


def post(path, body, timeout=7200):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


out = {"batch": batch, "isl": isl, "osl": osl, "rounds": [], "validation": []}
if validate and batch == 1:
    for prompt, expect in (
        ("The capital of France is", "paris"),
        ("Question: What is 17 * 23? Answer:", "391"),
        ("def fibonacci(n):", "return"),
    ):
        t0 = time.time()
        r = post(
            "/v1/completions", {"prompt": prompt, "max_tokens": 48, "temperature": 0}
        )
        text = r["choices"][0]["text"]
        rec = {
            "prompt": prompt,
            "text": text,
            "pass": expect in text.lower(),
            "latency_s": round(time.time() - t0, 2),
        }
        out["validation"].append(rec)
        print("[MK-VALIDATE]", json.dumps(rec), flush=True)

for rnd in range(rounds):
    results = [None] * batch

    def one(i):
        rng = random.Random(1000 * rnd + i)
        prompt = [rng.randrange(1000, 150000) for _ in range(isl)]
        t0 = time.time()
        r = post(
            "/v1/completions",
            {"prompt": prompt, "max_tokens": osl, "temperature": 0, "ignore_eos": True},
        )
        results[i] = {"e2e_s": round(time.time() - t0, 3), "usage": r.get("usage")}

    ts = [threading.Thread(target=one, args=(i,)) for i in range(batch)]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    out["rounds"].append({"wall_s": round(time.time() - t0, 3), "requests": results})
    print("[MK-ROUND]", rnd, json.dumps(out["rounds"][-1]), flush=True)
json.dump(out, open(f"{cfg_dir}/client.json", "w"), indent=2)
