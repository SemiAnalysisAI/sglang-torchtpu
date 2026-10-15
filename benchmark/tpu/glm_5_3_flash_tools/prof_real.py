"""Capture ~0.6 s of steady DFlash/MTP decode on a real-text 8k chat prompt (c1).

usage: python prof_real.py <out_dir> [activities, default TPU,CPU]
Same prompt construction as bench_real.py, so acceptance matches the real-text bench.
"""
import json, sys, threading, time, urllib.request

from bench_real import TASKS, URL, context

out = sys.argv[1]
act = sys.argv[2].split(",") if len(sys.argv) > 2 else ["TPU", "CPU"]


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


ctx = context("/dev/shm/home/qinghanc/work/glm-dsa-tpu", 30000)
body = {"messages": [{"role": "user", "content": ctx + "\n\n" + TASKS[0]}], "max_tokens": 1024,
        "temperature": 0.0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
first = threading.Event()


def run():
    with post("/v1/chat/completions", body) as r:
        for raw in r:
            if raw.startswith(b"data:"):
                first.set()


th = threading.Thread(target=run)
th.start()
first.wait()
time.sleep(1.0)
post("/start_profile", {"output_dir": out, "activities": act, "profile_id": "round"}).read()
time.sleep(0.6)
post("/stop_profile", {}).read()
th.join()
print("profiled", out)
