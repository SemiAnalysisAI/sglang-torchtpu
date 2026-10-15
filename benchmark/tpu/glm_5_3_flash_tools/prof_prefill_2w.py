"""Two-window profile of one 8192-token prefill on a running server: python prof_prefill_2w.py <out_dir> [delay_ms].

The trace buffer holds only part of an 8k prefill, so the same request is profiled twice:
window "a" starts the profiler before the request (early layers); window "b" starts it
delay_ms after the request was sent (later layers). Captures land in <out_dir>/a and <out_dir>/b.
"""
import json, random, sys, threading, time, urllib.request

U = "http://127.0.0.1:30000"; OUT = sys.argv[1]; DELAY = float(sys.argv[2]) / 1000 if len(sys.argv) > 2 else 0.25


def post(p, b, t=900):
    req = urllib.request.Request(U + p, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=t)


def gen(seed):
    rnd = random.Random(seed); ids = [rnd.randrange(1000, 100000) for _ in range(8192)]
    t = time.perf_counter()
    post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}}).read()
    return time.perf_counter() - t


print("warm e2e (s):", [round(gen(s), 4) for s in (21, 22, 23)])
for window, delay in (("a", None), ("b", DELAY)):
    if delay is None:
        post("/start_profile", {"output_dir": f"{OUT}/{window}", "activities": ["TPU"], "profile_id": window}).read()
        print(f"window {window}: e2e {gen(31):.4f}s"); time.sleep(0.3)
    else:
        th = threading.Thread(target=lambda: print(f"window {window}: e2e {gen(32):.4f}s")); th.start()
        time.sleep(delay)
        post("/start_profile", {"output_dir": f"{OUT}/{window}", "activities": ["TPU"], "profile_id": window}).read()
        th.join(); time.sleep(0.3)
    post("/stop_profile", {}).read()
print("PROFILED")
