"""Concurrent 8k/1k benchmark: N simultaneous streamed requests, greedy, ignore_eos.

usage: python bench_conc.py --conc 8 [--real] [--isl 8192 --osl 1024]
--real uses repo docs + varied tasks via the chat API (realistic acceptance);
default is random token ids (comparable to bench_8k1k.py).
"""
import argparse, glob, json, random, statistics, threading, time, urllib.request

URL = "http://127.0.0.1:30000"


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=7200)


def docs(chars):
    text = ""
    for f in sorted(glob.glob("/dev/shm/home/qinghanc/work/glm-dsa-tpu/**/*.md", recursive=True)):
        text += open(f, errors="ignore").read() + "\n\n"
        if len(text) > chars:
            break
    return text


def one(i, a, ctx, out):
    t0 = time.perf_counter(); first = None; n = 0
    if a.real:
        # Raw text through /generate so every chunk reports its token count.
        start = (i * 7919) % max(1, len(ctx) - 30000)
        body = {"text": ctx[start:start + 24000] +
                f"\n\nTask {i}: write a detailed, section-by-section technical summary of the text above.\n",
                "sampling_params": {"max_new_tokens": a.osl, "ignore_eos": True, "temperature": 0.0}, "stream": True}
        path = "/generate"
    else:
        rnd = random.Random(1000 + i)
        body = {"input_ids": [rnd.randrange(1000, 100000) for _ in range(a.isl)],
                "sampling_params": {"max_new_tokens": a.osl, "ignore_eos": True, "temperature": 0.0}, "stream": True}
        path = "/generate"
    meta = None; trace = []
    try:
        r = post(path, body)
    except Exception as e:  # a rejected request must not look like a fast one
        out[i] = {"error": str(e)[:200]}
        return
    with r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            msg = json.loads(line[5:])
            meta = msg.get("meta_info", meta)
            now = time.perf_counter(); first = first or now
            trace.append((now, (meta or {}).get("completion_tokens", 0)))
    end = time.perf_counter()
    toks = (meta or {}).get("completion_tokens") or a.osl
    out[i] = {"ttft_s": first - t0, "tpot_ms": (end - first) * 1000 / max(toks - 1, 1), "tokens": toks,
              "end": end, "first": first, "trace": trace, "verify": (meta or {}).get("spec_verify_ct")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", type=int, default=8); ap.add_argument("--isl", type=int, default=8192)
    ap.add_argument("--osl", type=int, default=1024); ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    ctx = docs(400000) if a.real else ""
    # Warm the batch buckets with a short burst.
    warm = {}
    ts = [threading.Thread(target=one, args=(i, argparse.Namespace(**{**vars(a), "osl": 16, "isl": 1024}), ctx, warm)) for i in range(a.conc)]
    [t.start() for t in ts]; [t.join() for t in ts]
    out = {}
    t0 = time.perf_counter()
    ts = [threading.Thread(target=one, args=(i, a, ctx, out)) for i in range(a.conc)]
    [t.start() for t in ts]; [t.join() for t in ts]
    errors = [r["error"] for r in out.values() if "error" in r]
    if errors:
        print(f"ERRORS {len(errors)}/{a.conc}: {errors[0]}")
        out = {k: v for k, v in out.items() if "error" not in v}
    wall = max(r["end"] for r in out.values()) - t0
    tp = sorted(r["tpot_ms"] for r in out.values())
    toks = sum(r["tokens"] for r in out.values())
    # Steady state: every request past prefill and none finished yet.
    w0 = max(r["first"] for r in out.values()); w1 = min(r["end"] for r in out.values())

    def at(trace, t):
        n = 0
        for ts, c in trace:
            if ts > t:
                break
            n = c
        return n
    steady = sum(at(r["trace"], w1) - at(r["trace"], w0) for r in out.values()) / max(w1 - w0, 1e-9)
    ver = [r["verify"] for r in out.values() if r["verify"]]
    acc = toks / sum(ver) if ver else None
    for k, r in sorted(out.items()):
        print(f"  req {k:2d}: first +{r['first'] - t0:6.2f}s end +{r['end'] - t0:6.2f}s tokens {r['tokens']} chunks {len(r['trace'])}")
    print(f"STEADY conc={a.conc}: window {w1 - w0:.1f} s, decode {steady:.0f} tok/s total, "
          f"{steady / a.conc:.0f} tok/s/request = TPOT {a.conc * 1000 / max(steady, 1e-9):.2f} ms; accept {acc}")
    print(f"SUMMARY conc={a.conc} {'real' if a.real else 'random'}: TPOT p50 {statistics.median(tp):.2f} ms, "
          f"p90 {tp[int(0.9 * (len(tp) - 1))]:.2f} ms, TTFT p50 {statistics.median(r['ttft_s'] for r in out.values()):.2f} s, "
          f"output {toks / wall:.0f} tok/s over {wall:.1f} s")


if __name__ == "__main__":
    main()
