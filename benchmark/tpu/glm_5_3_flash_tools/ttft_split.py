"""Split TTFT into fixed overhead, tokenization and prefill on a running server."""
import json, random, time, urllib.request
U = "http://127.0.0.1:30000"
def post(p, b, stream=False):
    req = urllib.request.Request(U + p, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t = time.perf_counter(); r = urllib.request.urlopen(req, timeout=900)
    if stream:
        for raw in r:
            if raw.startswith(b"data:"):
                return time.perf_counter() - t
    r.read(); return time.perf_counter() - t
def ids(n, seed):
    rnd = random.Random(seed); return [rnd.randrange(1000, 100000) for _ in range(n)]
sp = {"max_new_tokens": 1, "temperature": 0}
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/weights/GLM-5.3-Flash")
text = tok.decode(ids(8192, 99))
t = time.perf_counter(); n = len(tok(text)["input_ids"]); tt = time.perf_counter() - t
print(f"local tokenize of the 8k text: {n} tokens in {tt*1e3:.1f} ms")
for name, body, stream in (
    ("16 ids", lambda s: {"input_ids": ids(16, s), "sampling_params": sp}, False),
    ("8192 ids", lambda s: {"input_ids": ids(8192, s), "sampling_params": sp}, False),
    ("8192 ids stream first chunk", lambda s: {"input_ids": ids(8192, s), "sampling_params": {**sp, "max_new_tokens": 8}, "stream": True}, True),
    ("8k text", lambda s: {"text": tok.decode(ids(8192, s)), "sampling_params": sp}, False),
):
    vals = [post("/generate", body(100 + i), stream) * 1e3 for i in range(4)]
    print(f"{name:30s} " + " ".join(f"{v:7.1f}" for v in vals) + " ms")
