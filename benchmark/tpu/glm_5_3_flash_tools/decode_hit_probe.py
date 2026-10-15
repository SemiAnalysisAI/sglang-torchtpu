"""Does a finished request's decoded span become a cache hit? python decode_hit_probe.py [port].

For a few prompt lengths: generate 600 tokens greedily, then send prompt + output + ","
and print cached_tokens against the prompt length. cached > prompt means KDA state
snapshots were taken inside the decode span (the radix tree can only reuse a prefix
up to the last snapshot)."""
import json, random, sys, urllib.request

BASE = f"http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else '30000'}"


def post(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=900).read()
    return json.loads(raw) if body is not None else raw


post("/flush_cache")
for n_lines in (40, 150, 400):
    rnd = random.Random(n_lines)
    text = "Write a long, detailed story continuing this log.\n" + "\n".join(
        f"Day {i}: the crew moved {rnd.randrange(90)} crates." for i in range(n_lines))
    a = post("/generate", {"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": 600, "ignore_eos": True}})
    ids = post("/generate", {"text": text, "sampling_params": {"max_new_tokens": 1}, "return_text_in_logprobs": False})
    prompt = a["meta_info"]["prompt_tokens"]; out = a["meta_info"]["completion_tokens"]
    b = post("/generate", {"text": text + a["text"] + ",", "sampling_params": {"temperature": 0, "max_new_tokens": 8}})
    print(f"prompt={prompt} decoded={out} second_turn_prompt={b['meta_info']['prompt_tokens']} "
          f"cached={b['meta_info']['cached_tokens']} -> {'DECODE HIT' if b['meta_info']['cached_tokens'] > prompt + 1 else 'prompt only'}")
print("DECODE_PROBE_DONE")
