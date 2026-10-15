import json, urllib.request, time, random, threading, sys
U="http://127.0.0.1:30000"; out=sys.argv[1]; ACT=sys.argv[2].split(",") if len(sys.argv)>2 else ["TPU"]
def post(p,b,t=900): return urllib.request.urlopen(urllib.request.Request(U+p,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"}),timeout=t)
rnd=random.Random(2); ids=[rnd.randrange(1000,100000) for _ in range(8192)]
first=threading.Event()
def run():
    with post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 1024, "ignore_eos": True, "temperature": 0}, "stream": True}) as r:
        for raw in r:
            if raw.startswith(b"data:"): first.set()
th=threading.Thread(target=run); th.start(); first.wait(); time.sleep(0.3)
post("/start_profile", {"output_dir": out, "activities": ACT, "profile_id": "round"}).read()
time.sleep(0.6); post("/stop_profile", {}).read(); th.join(); print("profiled")
