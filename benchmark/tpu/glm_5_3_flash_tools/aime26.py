"""AIME 2026 (MathArena/aime_2026, I + II, 30 problems) against a local server, with per-problem dumps.

usage: python aime26.py --tag dflash [--repeats 1] [--threads 8] [--max-tokens 60000]
Sampling follows GLM's recommended temperature 1.0 / top_p 0.95. Grading matches the repo's
AIME25 eval ("Answer: N" line, else the last \\boxed{}). Writes aime26-<tag>.json with every
response: reasoning and content lengths, finish reason, extracted and correct answers.
"""
import argparse, json, re, statistics, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset

from sglang.test.simple_eval_aime25 import QUERY_TEMPLATE, normalize_aime_answer
from sglang.test.simple_eval_common import ANSWER_PATTERN, BOXED_PATTERN

URL = "http://127.0.0.1:30000/v1/chat/completions"


def clean(ans):
    """Strip LaTeX wrapping GLM puts around the integer (``$070$``, ``\\boxed{70}``, ``70.``)."""
    if ans is None:
        return None
    a = ans.strip()
    a = re.sub(r"\\(boxed|text|mathbf|mathrm)\{([^}]*)\}", r"\2", a)
    a = a.replace("$", "").replace(",", "").replace("\\!", "").strip().rstrip(".").strip()
    m = re.search(r"-?\d+", a)
    return m.group(0) if m else a


def extract(text):
    m = re.search(ANSWER_PATTERN, text or "")
    if m:
        return m.group(1).strip()
    boxed = re.findall(BOXED_PATTERN, text or "")
    return boxed[-1].strip() if boxed else None


def solve(row, max_tokens):
    body = {"model": "default", "messages": [{"role": "user", "content": QUERY_TEMPLATE.format(question=row["question"])}],
            "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=21600))
    ch = r["choices"][0]; msg = ch["message"]
    content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
    got_c, got_r = extract(content), extract(reasoning)
    got = got_c if got_c is not None else None
    ok = normalize_aime_answer(clean(got)) == normalize_aime_answer(row["answer"])
    return {"idx": row["idx"], "answer": row["answer"], "extracted": got, "extracted_from_reasoning": got_r,
            "correct": ok, "finish": ch.get("finish_reason"), "completion_tokens": r["usage"]["completion_tokens"],
            "content_chars": len(content), "reasoning_chars": len(reasoning),
            "content_tail": content[-400:], "reasoning_tail": reasoning[-400:],
            "content": content, "reasoning": reasoning}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--max-tokens", type=int, default=60000)
    ap.add_argument("--problems", default="", help="comma-separated problem_idx subset")
    ap.add_argument("--reps", default="", help="comma-separated repeat indices to run (default: all of --repeats)")
    a = ap.parse_args()
    rows = load_dataset("MathArena/aime_2026", split="train")
    keep = {int(x) for x in a.problems.split(",") if x}
    base = [{"idx": int(r["problem_idx"]), "question": r["problem"], "answer": str(r["answer"])}
            for r in rows if not keep or int(r["problem_idx"]) in keep]
    reps = [int(x) for x in a.reps.split(",") if x] or list(range(a.repeats))
    probs = [dict(p, rep=k) for k in reps for p in base]
    # Resumable: every finished response is appended to the jsonl as it lands;
    # a restart skips (problem, repeat) pairs already there.
    jl = f"aime26-{a.tag}.jsonl"
    done = {}
    try:
        for line in open(jl):
            r = json.loads(line); done[(r["idx"], r["rep"])] = r
    except FileNotFoundError:
        pass
    todo = [p for p in probs if (p["idx"], p["rep"]) not in done]
    print(f"AIME26 {a.tag}: {len(done)} done, {len(todo)} to run, {a.threads} concurrent", flush=True)
    lock = threading.Lock(); t0 = time.time()

    def run(p):
        try:
            r = solve(p, a.max_tokens)
        except Exception as e:  # leave it for the next resume rather than kill the pool
            print(f"[{time.time() - t0:7.0f}s] p{p['idx']} rep{p['rep']} FAILED ({type(e).__name__}: {e}); left for resume", flush=True)
            return None
        r["rep"] = p["rep"]
        with lock:
            with open(jl, "a") as f:
                f.write(json.dumps(r) + "\n")
            done[(p["idx"], p["rep"])] = r
            n_ok = sum(x["correct"] for x in done.values())
            print(f"[{time.time() - t0:7.0f}s] {len(done)}/{len(probs)} done, running acc {n_ok / len(done):.3f}, "
                  f"p{p['idx']} rep{p['rep']} {'OK' if r['correct'] else 'X'} {r['completion_tokens']} tok {r['finish']}", flush=True)
        return r

    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(run, todo))
    res = list(done.values())
    n = len(res); score = sum(r["correct"] for r in res) / n
    per_problem = {}
    for r in res:
        per_problem.setdefault(r["idx"], []).append(r["correct"])
    out = {"tag": a.tag, "score": score, "n": n, "wall_s": time.time() - t0,
           "per_problem": {k: sum(v) / len(v) for k, v in sorted(per_problem.items())}, "results": res}
    json.dump(out, open(f"aime26-{a.tag}.json", "w"), indent=1)
    trunc = sum(r["finish"] == "length" for r in res)
    no_ans = sum(r["extracted"] is None for r in res)
    ans_in_reasoning_only = sum(r["extracted"] is None and r["extracted_from_reasoning"] is not None for r in res)
    print(f"AIME26 {a.tag}: score {score:.3f} (mean over {n}), mean {statistics.mean(r['completion_tokens'] for r in res):.0f} tok, "
          f"{trunc} truncated, {no_ans} without an answer ({ans_in_reasoning_only} answered only inside reasoning), "
          f"wall {out['wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
