"""8k1k at concurrency 1 against a local sglang server: timing, then a decode-phase xprof capture.

usage: python bench_8k1k.py --runs 2 --profile-steps 20 --out ~/work/prof-8k1k
Prompts are random token ids (exactly ISL tokens), ignore_eos, temperature 0.
"""
import argparse, json, random, statistics, threading, time, urllib.request

URL = "http://127.0.0.1:30000"

def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def run_once(isl, osl, seed, on_first_token=None):
    rnd = random.Random(seed)
    ids = [rnd.randrange(1000, 100000) for _ in range(isl)]
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": osl, "ignore_eos": True, "temperature": 0.0}, "stream": True}
    t0 = time.perf_counter(); first = None; n = 0; last_meta = None
    with post("/generate", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            msg = json.loads(line[5:])
            last_meta = msg.get("meta_info", {})
            if first is None:
                first = time.perf_counter()
                if on_first_token: on_first_token()
    end = time.perf_counter()
    out_toks = last_meta.get("completion_tokens", osl)
    return {"ttft_s": first - t0, "tpot_ms": (end - first) * 1000 / max(out_toks - 1, 1), "e2e_s": end - t0,
            "prompt_tokens": last_meta.get("prompt_tokens"), "completion_tokens": out_toks}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isl", type=int, default=8192); ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=2); ap.add_argument("--profile-steps", type=int, default=0)
    ap.add_argument("--out", default="/dev/shm/home/qinghanc/work/prof-8k1k")
    a = ap.parse_args()
    print("warmup:", json.dumps(run_once(a.isl, 64, seed=0)), flush=True)
    res = []
    for i in range(a.runs):
        r = run_once(a.isl, a.osl, seed=100 + i); res.append(r); print(f"run {i}:", json.dumps(r), flush=True)
    print("SUMMARY isl=%d osl=%d conc=1: TTFT median %.3f s, TPOT median %.2f ms (%.1f tok/s)" % (
        a.isl, a.osl, statistics.median(x["ttft_s"] for x in res), statistics.median(x["tpot_ms"] for x in res),
        1000 / statistics.median(x["tpot_ms"] for x in res)), flush=True)
    if a.profile_steps:
        def arm():
            # Decode is underway once the first token streams; the request outlives the capture (osl >> steps).
            time.sleep(2.0)
            with post("/start_profile", {"output_dir": a.out, "num_steps": a.profile_steps, "activities": ["TPU", "CPU"], "profile_id": "decode8k"}) as r:
                print("start_profile:", r.read().decode().strip(), flush=True)
        r = run_once(a.isl, a.osl, seed=999, on_first_token=lambda: threading.Thread(target=arm, daemon=True).start())
        print("profiled run:", json.dumps(r), flush=True)

if __name__ == "__main__":
    main()
