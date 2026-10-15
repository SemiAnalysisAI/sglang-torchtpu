"""Basic prefix-cache check against a local server: python prefix_test.py [port].

Sends a ~6k-token document three times with greedy decoding:
  1. doc + question A   (cold: nothing cached)
  2. doc + question A   (same request: expect most tokens cached and identical text)
  3. doc + question B   (shared prefix: expect about the document's length cached)
Prints prompt/cached token counts and whether runs 1 and 2 produced identical text.
"""
import json, sys, time, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "30000"
URL = f"http://127.0.0.1:{PORT}/generate"
lines = [f"Record {i}: station {i % 37} logged {i * 7 % 101} units at hour {i % 24}." for i in range(400)]
doc = "Inventory log.\n" + "\n".join(lines) + "\n\n"
QA = "Question: which station logged units at hour 5 in record 5? Answer briefly."
QB = "Question: how many units did record 12 log? Answer briefly."


def gen(text):
    body = {"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": 48}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    out = json.load(urllib.request.urlopen(req, timeout=900))
    meta = out.get("meta_info", {})
    return out["text"], meta.get("prompt_tokens"), meta.get("cached_tokens"), time.time() - t


results = []
for name, q in (("cold A", QA), ("repeat A", QA), ("shared-prefix B", QB)):
    text, prompt, cached, secs = gen(doc + q)
    results.append(text)
    print(f"{name:16s} prompt_tokens={prompt} cached_tokens={cached} latency={secs:.2f}s text={text[:60]!r}")
print("repeat identical to cold:", results[0] == results[1])
print("PREFIX_TEST_DONE")
