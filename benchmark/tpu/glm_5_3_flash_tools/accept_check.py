"""Per-request DFlash acceptance on fixed real-text prompts (greedy): tokens / verify rounds.

usage: python accept_check.py [--osl 512]
"""
import argparse, glob, json, urllib.request

URL = "http://127.0.0.1:30000"
TASKS = [
    "Summarize the documents above section by section.",
    "Explain how a new contributor should set up and test the changes described above.",
    "Write a Python function that parses the configuration options mentioned above.",
    "The capital of France is",
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--osl", type=int, default=512)
    a = ap.parse_args()
    ctx = ""
    for f in sorted(glob.glob("/dev/shm/home/qinghanc/work/glm-dsa-tpu/**/*.md", recursive=True)):
        ctx += open(f, errors="ignore").read() + "\n\n"
        if len(ctx) > 12000:
            break
    tot_tok = tot_ver = 0
    for i, task in enumerate(TASKS):
        text = task if i == len(TASKS) - 1 else ctx[:12000] + "\n\n" + task + "\n"
        body = {"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": a.osl}}
        r = urllib.request.Request(URL + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        m = json.load(urllib.request.urlopen(r, timeout=900))["meta_info"]
        tok, ver = m["completion_tokens"], m.get("spec_verify_ct") or 0
        tot_tok += tok; tot_ver += ver
        print(f"task {i}: {tok} tokens / {ver} rounds = accept {tok / max(ver, 1):.2f}")
    print(f"ACCEPT overall {tot_tok / max(tot_ver, 1):.3f}")


if __name__ == "__main__":
    main()
