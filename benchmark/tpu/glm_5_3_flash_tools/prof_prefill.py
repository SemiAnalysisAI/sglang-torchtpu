"""Profile one 8192-token prefill (max_new_tokens=1) on a running server: python prof_prefill.py <out_dir> [activities]."""
import json, random, sys, time, urllib.request
U = "http://127.0.0.1:30000"; out = sys.argv[1]; ACT = sys.argv[2].split(",") if len(sys.argv) > 2 else ["TPU"]
def post(p, b, t=900):
    return urllib.request.urlopen(urllib.request.Request(U + p, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=t)
def gen(seed):
    rnd = random.Random(seed); ids = [rnd.randrange(1000, 100000) for _ in range(8192)]
    t = time.perf_counter()
    post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}}).read()
    return time.perf_counter() - t
print("warm e2e (s):", [round(gen(s), 4) for s in (11, 12, 13)])
post("/start_profile", {"output_dir": out, "activities": ACT, "profile_id": "prefill"}).read()
print("profiled e2e (s):", round(gen(14), 4))
time.sleep(0.3); post("/stop_profile", {}).read(); print("profiled")
