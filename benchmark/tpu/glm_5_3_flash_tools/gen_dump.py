"""Greedy outputs for a fixed prompt set, sequential (conc 1) and all-at-once (conc N).

usage: python gen_dump.py <out.json> [--osl 512]
Prompts: chat questions (chat template) and raw-text doc summaries like bench_conc --real.
"""
import argparse, glob, json, threading, urllib.request

URL = "http://127.0.0.1:30000"
QUESTIONS = [
    "Explain how a hash map works and its time complexity.",
    "Write a Python function that checks whether a string is a palindrome, with tests.",
    "What are the main causes of the French Revolution? Answer in a few paragraphs.",
    "Describe the difference between TCP and UDP.",
]


def post(path, body):
    r = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))


def prompts():
    ctx = ""
    for f in sorted(glob.glob("/dev/shm/home/qinghanc/work/glm-dsa-tpu/**/*.md", recursive=True)):
        ctx += open(f, errors="ignore").read() + "\n\n"
        if len(ctx) > 400000:
            break
    ps = [("chat", q) for q in QUESTIONS]
    for i in range(4):
        start = (i * 7919) % (len(ctx) - 24000)
        ps.append(("raw", ctx[start:start + 24000] + f"\n\nTask {i}: write a detailed, section-by-section technical summary of the text above.\n"))
    return ps


def run(kind, text, osl):
    if kind == "chat":
        out = post("/v1/chat/completions", {"messages": [{"role": "user", "content": text}], "max_tokens": osl,
                                            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}})
        return {"text": out["choices"][0]["message"]["content"], "tokens": out["usage"]["completion_tokens"]}
    out = post("/generate", {"text": text, "sampling_params": {"max_new_tokens": osl, "temperature": 0.0}})
    return {"text": out["text"], "tokens": out["meta_info"]["completion_tokens"],
            "verify": out["meta_info"].get("spec_verify_ct")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--osl", type=int, default=512)
    a = ap.parse_args()
    ps = prompts()
    seq = [run(k, t, a.osl) for k, t in ps]
    par = [None] * len(ps)

    def worker(i):
        par[i] = run(*ps[i], a.osl)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(len(ps))]
    [t.start() for t in ts]; [t.join() for t in ts]
    json.dump({"kinds": [k for k, _ in ps], "seq": seq, "par": par}, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
