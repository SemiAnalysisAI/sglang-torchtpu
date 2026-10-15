"""Greedy outputs for a fixed prompt set: python pd_check.py <url> <out.json> [ref.json].
With ref.json, prints per-prompt whether the output matches the reference."""
import json, random, sys, urllib.request
url, out = sys.argv[1], sys.argv[2]; ref = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
rnd = random.Random(5)
doc = "Inventory log.\n" + "\n".join(f"Record {i}: station {rnd.randrange(40)} logged {rnd.randrange(900)} units." for i in range(260)) + "\n\n"
prompts = ["The capital of France is", "Write one sentence about the ocean.",
           "List three prime numbers greater than 50.", doc + "Question: how many units did record 9 log? Answer briefly.",
           doc + "Question: which station logged record 100? Answer briefly."]
res = []
for p in prompts:
    body = {"text": p, "sampling_params": {"temperature": 0, "max_new_tokens": 48}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900))
    res.append({"prompt_tokens": r["meta_info"].get("prompt_tokens"), "text": r["text"]})
json.dump(res, open(out, "w"), indent=1)
for i, x in enumerate(res):
    m = "" if ref is None else ("MATCH" if x["text"] == ref[i]["text"] else f"DIFF ref={ref[i]['text'][:60]!r}")
    print(f"[{i}] {x['prompt_tokens']} tok -> {x['text'][:70]!r} {m}")
print("PD_CHECK_DONE")
