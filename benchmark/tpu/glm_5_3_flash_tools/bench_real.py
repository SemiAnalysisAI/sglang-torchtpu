"""Real-text c1 benchmark: ~8k tokens of repo docs + a task, streamed via chat.

usage: python bench_real.py --runs 2 [--osl 1024] [--docs DIR]
Reports TTFT, TPOT and (with spec decoding) accept length = tokens / verify steps.
"""
import argparse, glob, json, statistics, time, urllib.request

URL = "http://127.0.0.1:30000"
TASKS = [
    "Write a detailed technical summary of the documents above, section by section.",
    "Explain, step by step, how a new contributor should set up and test changes described above.",
    "List every configuration option mentioned above and explain what each one does.",
]


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)


def context(docs, chars):
    text = ""
    for f in sorted(glob.glob(f"{docs}/**/*.md", recursive=True)):
        text += open(f, errors="ignore").read() + "\n\n"
        if len(text) > chars:
            break
    return text[:chars]


def run_once(ctx, task, osl):
    body = {"messages": [{"role": "user", "content": ctx + "\n\n" + task}], "max_tokens": osl,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter(); first = None; usage = None; n = 0
    with post("/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            msg = json.loads(line[5:])
            if msg.get("usage"):
                usage = msg["usage"]
            if msg.get("choices") and msg["choices"][0]["delta"].get("content"):
                n += 1
                if first is None:
                    first = time.perf_counter()
    end = time.perf_counter()
    out = (usage or {}).get("completion_tokens", n)
    return {"ttft_s": first - t0, "tpot_ms": (end - first) * 1000 / max(out - 1, 1),
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2); ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--chars", type=int, default=30000)
    ap.add_argument("--docs", default="/dev/shm/home/qinghanc/work/glm-dsa-tpu")
    a = ap.parse_args()
    ctx = context(a.docs, a.chars)
    run_once(ctx[:2000], TASKS[0], 32)  # warmup
    res = []
    for i in range(a.runs):
        r = run_once(ctx, TASKS[i % len(TASKS)], a.osl); res.append(r)
        print(f"run {i}: {json.dumps(r)}", flush=True)
    info = json.load(urllib.request.urlopen(URL + "/get_server_info", timeout=60))
    acc = (info.get("internal_states") or [{}])[0].get("avg_spec_accept_length")
    print(f"SUMMARY real-text c1: prompt {res[0]['prompt_tokens']} tok, TPOT median "
          f"{statistics.median(r['tpot_ms'] for r in res):.2f} ms, server avg accept length {acc}")


if __name__ == "__main__":
    main()
