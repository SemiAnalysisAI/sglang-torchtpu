"""Greedy chat completions for a few fixed prompts; prints text, acceptance and a 4-gram repetition rate."""
import json, sys, urllib.request

URL = "http://127.0.0.1:30000"
PROMPTS = [
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a short story about a lighthouse keeper in about 200 words.",
    "What is 17 * 23? Show the steps.",
]


def rep4(text):
    w = text.split()
    grams = [tuple(w[i : i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(grams)) / max(len(grams), 1)


for p in PROMPTS:
    body = {"model": "default", "messages": [{"role": "user", "content": p}], "max_tokens": int(sys.argv[1]) if len(sys.argv) > 1 else 400, "temperature": 0}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    m = r["choices"][0]["message"]
    text = (m.get("reasoning_content") or "") + (m.get("content") or "")
    print("=" * 80, "\nPROMPT:", p, "\nusage:", r.get("usage"), " rep4=%.3f" % rep4(text))
    print(text[:1200])
