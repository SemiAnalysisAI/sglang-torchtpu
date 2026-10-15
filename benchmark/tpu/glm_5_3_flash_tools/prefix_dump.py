"""Dump the first generated token's top-5 logprobs for fixed prompts: python prefix_dump.py <out.json> [port] [--flush]."""
import json, random, sys, urllib.request
out_path = sys.argv[1]; PORT = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].isdigit() else "30000"
FLUSH = "--flush" in sys.argv; BASE = f"http://127.0.0.1:{PORT}"
def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=900).read()
def doc(seed, n):
    rnd = random.Random(seed)
    return "Sensor log.\n" + "\n".join(f"Entry {i}: sensor {rnd.randrange(50)} read {rnd.randrange(1000)} at minute {rnd.randrange(60)}." for i in range(n)) + "\n\n"
res = []
for s, n in ((1, 120), (2, 300), (3, 600)):
    if FLUSH: post("/flush_cache", {})
    o = json.loads(post("/generate", {"text": doc(s, n) + "Question: what did entry 7 read? Answer with the number only.",
        "return_logprob": True, "top_logprobs_num": 5, "sampling_params": {"temperature": 0, "max_new_tokens": 4}}))
    m = o["meta_info"]
    res.append({"prompt_tokens": m["prompt_tokens"], "cached": m.get("cached_tokens"), "text": o["text"],
                "top": {str(t[1]): t[0] for t in m["output_top_logprobs"][0]}})
json.dump(res, open(out_path, "w"), indent=1); print(json.dumps(res)[:600])
