"""Capture ~0.8 s of steady c16 decode rounds on the running server (16 concurrent 8k random prompts)."""
import json, random, sys, threading, time, urllib.request

U = "http://127.0.0.1:30000"; out = sys.argv[1]


def post(p, b, t=1800):
    return urllib.request.urlopen(urllib.request.Request(U + p, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=t)


started = []


def run(seed):
    rnd = random.Random(seed); ids = [rnd.randrange(1000, 100000) for _ in range(8192)]
    with post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 2048, "ignore_eos": True, "temperature": 0}, "stream": True}) as r:
        for raw in r:
            if raw.startswith(b"data:") and seed not in started:
                started.append(seed)


ths = [threading.Thread(target=run, args=(s,)) for s in range(16)]
[t.start() for t in ths]
while len(started) < 16:
    time.sleep(0.2)
time.sleep(2.0)
post("/start_profile", {"output_dir": out, "activities": ["TPU"], "profile_id": "c16"}).read()
time.sleep(0.8); post("/stop_profile", {}).read()
[t.join() for t in ths]; print("profiled")
