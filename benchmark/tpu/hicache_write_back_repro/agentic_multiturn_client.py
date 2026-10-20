#!/usr/bin/env python3
"""Agentic multi-turn load for an OpenAI-compatible /v1/chat/completions server.

Traffic shape (all text is synthetic, generated from a fixed seed):
  * --sessions concurrent sessions. Each turn sends one "main" request (a long
    system prompt + the session's growing history), then --reminders-per-turn
    "reminder" requests in parallel (a shorter reminder system prompt + the same
    history + a short question). Each reminder kind has its own system prompt,
    so every kind is its own prefix chain.
  * Each turn appends a canned assistant message and a synthetic "tool result"
    to the history. A session restarts under a new id once its main prompt
    reaches --ctx-max tokens (as reported by the server). With enough sessions
    the working set overflows the device KV pool, so prefixes are evicted to
    host and reloaded (HiCache on) or recomputed (HiCache off).
  * Each session's first conversation starts at a seeded random turn in
    [0, --initial-turn-max]: its earlier turns are built but not sent. Session
    ages stay spread out, as in real traffic. Without the spread, every session
    grows and restarts in lockstep, and the working set swings between small and
    larger than the host pool.
  * A seeded --replay-fraction of reminders is selected for replay: re-sent with
    identical input --replay-delay seconds after the first reply.
  * --prefill-only sends max_tokens=1 on every request, so the run measures
    prefill and cache reuse with decode out of the picture. The history is the
    same either way: it holds a canned assistant line, never the model's reply.
  * --rotate-system-prompt-at changes every system prompt at that time, which
    invalidates every cached prefix at once. Replays keep their original input,
    so replays sent after the rotation still read old prefixes.

Closed loop: each session sends its next request when the previous one returns,
so a faster server gets further into each session's (fixed) prompt sequence.
No request starts after --duration; requests in flight at that point are waited
for and reported as the drain.

Writes one JSON line per request to stdout and a final {"summary": ...} line.
Exits non-zero if a worker thread crashed. Stdlib only.
"""

import argparse
import heapq
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request

REM_KINDS = ("skill", "todo", "memory", "verify", "goal")

p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
p.add_argument("--url", default="http://127.0.0.1:30000/v1/chat/completions")
p.add_argument("--model", default="default")
p.add_argument("--label", default="run", help="tag written into every JSON line")
p.add_argument("--sessions", type=int, default=20)
p.add_argument(
    "--duration", type=int, default=3600, help="seconds; no request starts after this"
)
p.add_argument(
    "--ctx-max",
    type=int,
    default=72_000,
    help="main prompt tokens at which a session restarts",
)
p.add_argument(
    "--initial-turn-max",
    type=int,
    default=20,
    help="first conversation of each session starts at a seeded random turn in [0, this]",
)
p.add_argument("--reminders-per-turn", type=int, default=3, help=f"0-{len(REM_KINDS)}")
p.add_argument("--seed", type=int, default=1001)
p.add_argument("--replay-fraction", type=float, default=0.1)
p.add_argument(
    "--replay-delay",
    type=int,
    default=600,
    help="seconds between a reminder and its replay",
)
p.add_argument(
    "--rotate-system-prompt-at",
    type=int,
    default=None,
    help="seconds after start at which every system prompt changes (mass prefix invalidation)",
)
p.add_argument(
    "--prefill-only",
    action="store_true",
    help="max_tokens=1 on every request; latency is then time to the first token",
)
p.add_argument("--main-max-tokens", type=int, default=192)
p.add_argument("--reminder-max-tokens", type=int, default=96)
p.add_argument("--reasoning-effort", default=None, help='e.g. "high" for gpt-oss')
p.add_argument(
    "--timeout", type=int, default=900, help="per-request HTTP timeout, seconds"
)
args = p.parse_args()
if args.prefill_only:
    args.main_max_tokens = args.reminder_max_tokens = 1
for name, ok in (
    ("--sessions", args.sessions > 0),
    ("--duration", args.duration > 0),
    ("--ctx-max", args.ctx_max > 0),
    ("--initial-turn-max", args.initial_turn_max >= 0),
    ("--reminders-per-turn", 0 <= args.reminders_per_turn <= len(REM_KINDS)),
    ("--replay-fraction", 0.0 <= args.replay_fraction <= 1.0),
    ("--replay-delay", args.replay_delay >= 0),
    ("--main-max-tokens", args.main_max_tokens > 0),
    ("--reminder-max-tokens", args.reminder_max_tokens > 0),
    ("--timeout", args.timeout > 0),
    (
        "--rotate-system-prompt-at",
        args.rotate_system_prompt_at is None or args.rotate_system_prompt_at > 0,
    ),
):
    if not ok:
        p.error(f"invalid {name}")

WORDS = (
    "alpha beta gamma delta request buffer index cache page token layer "
    "module config parse render stream socket thread handler metric value "
    "result error retry window commit branch patch review deploy rollout "
    "schema table column filter query offset length header payload"
).split()


def words(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def code_block(rng, lines):
    out = []
    for i in range(lines):
        a, b = rng.choice(WORDS), rng.choice(WORDS)
        out.append(
            f"    def {a}_{b}_{i}(self, {rng.choice(WORDS)}): "
            f"return self.{b}.get({rng.randint(0, 9999)}, '{rng.choice(WORDS)}')"
        )
    return "\n".join(out)


_g = random.Random(args.seed)
SYS = (
    "You are a coding agent working in a large repository. Follow the tool "
    "protocol exactly. Tools available:\n"
    + "\n".join(f"- tool_{i}: {words(_g, 60)}" for i in range(110))
)
REM_SYS = {
    k: (
        f"You are the {k} reminder agent. Read the main agent's transcript "
        f"and reply with at most two short sentences. Rules:\n"
        + "\n".join(f"{i}. {words(_g, 40)}" for i in range(45))
    )
    for k in REM_KINDS
}
REM_ASK = {
    k: f"Transcript ends here. As the {k} reminder agent, what single reminder, "
    f"if any, should the main agent get now? Answer briefly."
    for k in REM_KINDS
}

t_start = time.time()
deadline = t_start + args.duration


def system_prompt(kind):
    text = SYS if kind == "main" else REM_SYS[kind]
    rotate = args.rotate_system_prompt_at
    if rotate is not None and time.time() - t_start >= rotate:
        return "Revision 2 of these instructions.\n" + text
    return text


class HTTPFailure(Exception):
    pass


def post(messages, max_tokens):
    body = {
        "model": args.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    req = urllib.request.Request(
        args.url,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode("utf-8", "replace")
        raise HTTPFailure(f"HTTP {e.code}: {detail}") from None
    ch = d["choices"][0]
    content = ch["message"].get("content")
    u = d.get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    out = {
        "content": content if isinstance(content, str) else "",
        "finish": ch.get("finish_reason"),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "cached_tokens": det.get("cached_tokens"),
    }
    for k in ("prompt_tokens", "completion_tokens", "cached_tokens"):
        if out[k] is not None and not isinstance(out[k], int):
            raise ValueError(f"usage.{k} is {out[k]!r}")
    return out


lock = threading.Lock()
records = []
replay_queue = []  # heap of (due_time, request_id, item)
crashes = []


def run_req(kind, sid, turn, messages, max_tokens, replay_of=None):
    if (
        time.time() >= deadline
    ):  # the single admission check: no request starts after --duration
        return None
    rid = f"{sid}/t{turn}/{kind}"
    rec = {
        "label": args.label,
        "id": rid,
        "kind": kind,
        "sid": sid,
        "turn": turn,
        "start_s": round(time.time() - t_start, 3),
    }
    result = None
    try:
        result = post(messages, max_tokens)
        rec.update(
            {
                k: result[k]
                for k in (
                    "finish",
                    "prompt_tokens",
                    "completion_tokens",
                    "cached_tokens",
                )
            }
        )
        rec["head"] = result["content"][:160]
        if replay_of is not None:
            rec["replay_same_prefix"] = result["content"][:160] == replay_of[:160]
    except Exception as e:  # noqa: BLE001 - every failure is a counted error
        rec["error"] = repr(e)[:400]
        result = None
    rec["end_s"] = round(time.time() - t_start, 3)
    with lock:
        records.append(rec)
        print(json.dumps(rec), flush=True)
    return result


def guarded(target, *a):
    def body():
        try:
            target(*a)
        except BaseException as e:  # noqa: BLE001 - surfaced through the exit code
            with lock:
                crashes.append(f"{target.__name__}{a}: {e!r}"[:400])

    return body


def session_worker(idx):
    gen = 0
    while time.time() < deadline:
        sid = f"s{idx}g{gen}"
        rng = random.Random(f"{args.seed}-{idx}-{gen}")
        hist = [{"role": "user", "content": f"Task {sid}: " + words(rng, 300)}]
        turn, prompt_tokens = 0, 0
        if gen == 0:
            # Built exactly as sent turns are, so the session's prompt sequence does not change.
            for _ in range(
                random.Random(f"{args.seed}-offset-{idx}").randint(
                    0, args.initial_turn_max
                )
            ):
                hist = hist + [
                    {
                        "role": "assistant",
                        "content": f"Calling tool_{rng.randint(0, 109)} for step {turn}.",
                    },
                    {
                        "role": "user",
                        "content": "Tool result:\n"
                        + code_block(rng, rng.randint(60, 180)),
                    },
                ]
                rng.sample(REM_KINDS, args.reminders_per_turn)
                turn += 1
        while prompt_tokens < args.ctx_max and time.time() < deadline:
            r = run_req(
                "main",
                sid,
                turn,
                [{"role": "system", "content": system_prompt("main")}] + hist,
                args.main_max_tokens,
            )
            # Server-reported size; a failed request falls back to a per-turn estimate.
            prompt_tokens = (
                r["prompt_tokens"] if r and r["prompt_tokens"] else prompt_tokens + 2500
            )
            # A canned assistant turn keeps every arm's prefixes byte-identical.
            hist = hist + [
                {
                    "role": "assistant",
                    "content": f"Calling tool_{rng.randint(0, 109)} for step {turn}.",
                },
                {
                    "role": "user",
                    "content": "Tool result:\n" + code_block(rng, rng.randint(60, 180)),
                },
            ]
            kinds = rng.sample(REM_KINDS, args.reminders_per_turn)
            threads = []
            for kind in kinds:
                if time.time() >= deadline:
                    break
                msgs = (
                    [{"role": "system", "content": system_prompt(kind)}]
                    + hist
                    + [{"role": "user", "content": REM_ASK[kind]}]
                )
                rid = f"{sid}/t{turn}/{kind}"
                # Keyed by request id, so the selection does not depend on timing or failures.
                replay = (
                    random.Random(f"{args.seed}-replay-{rid}").random()
                    < args.replay_fraction
                )

                def one(kind=kind, msgs=msgs, rid=rid, replay=replay, turn=turn):
                    r = run_req(kind, sid, turn, msgs, args.reminder_max_tokens)
                    if r is not None and replay:
                        with lock:
                            heapq.heappush(
                                replay_queue,
                                (
                                    time.time() + args.replay_delay,
                                    rid,
                                    (kind, msgs, r["content"], sid, turn),
                                ),
                            )

                th = threading.Thread(target=guarded(one))
                th.start()
                threads.append(th)
            for th in threads:
                th.join()
            turn += 1
        gen += 1


def replay_worker():
    while time.time() < deadline:
        with lock:
            item = (
                heapq.heappop(replay_queue)
                if replay_queue and replay_queue[0][0] <= time.time()
                else None
            )
        if item is None:
            time.sleep(1)
            continue
        kind, msgs, first, sid, turn = item[2]
        run_req(
            "replay-" + kind, sid, turn, msgs, args.reminder_max_tokens, replay_of=first
        )


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else None


threads = [
    threading.Thread(target=guarded(session_worker, i)) for i in range(args.sessions)
]
threads += [threading.Thread(target=guarded(replay_worker)) for _ in range(2)]
for i, th in enumerate(threads):
    th.start()
    time.sleep(1.5 if i < args.sessions else 0)  # stagger session starts
for th in threads:
    th.join()

ok = [r for r in records if "error" not in r]
kind_of = lambda r: (
    "main"
    if r["kind"] == "main"
    else ("replay" if r["kind"].startswith("replay") else "reminder")
)
late = [r for r in records if r["end_s"] > args.duration]
buckets = {}
for r in ok:
    buckets[int(r["start_s"] // 600)] = buckets.get(int(r["start_s"] // 600), 0) + 1
replays = [r for r in ok if "replay_same_prefix" in r]
prompt = sum(r["prompt_tokens"] or 0 for r in ok)
cached = sum(r["cached_tokens"] or 0 for r in ok)
summary = {
    "label": args.label,
    "duration_s": args.duration,
    "sessions": args.sessions,
    # Local wall-clock start, the clock of a server log written on the same host.
    "start_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
    "requests": len(ok),
    "errors": len(records) - len(ok),
    "worker_crashes": crashes,
    "requests_per_s": round(len(ok) / args.duration, 3),
    "requests_per_10min": [buckets.get(i, 0) for i in range(-(-args.duration // 600))],
    "by_kind": {
        k: sum(1 for r in ok if kind_of(r) == k) for k in ("main", "reminder", "replay")
    },
    "drain": {
        "requests": len(late),
        "last_end_s": round(max((r["end_s"] for r in records), default=0), 1),
    },
    "session_generations": len({r["sid"] for r in records}),
    "prompt_tokens": prompt,
    "cached_tokens": cached,
    "cached_fraction": round(cached / prompt, 4) if prompt else None,
    "prefill_only": args.prefill_only,
    "input_tokens_per_s": round(prompt / args.duration, 1),
    "uncached_input_tokens_per_s": round((prompt - cached) / args.duration, 1),
    "finish_length_fraction": round(
        sum(1 for r in ok if r.get("finish") == "length") / len(ok), 3
    )
    if ok
    else None,
    "replay": {
        "selected_fraction": args.replay_fraction,
        "sent": len(replays),
        "reminders": sum(1 for r in ok if kind_of(r) == "reminder"),
        "first_160_chars_differ": sum(
            1 for r in replays if not r["replay_same_prefix"]
        ),
    },
    "latency_s": {
        k: {
            q: pct([r["end_s"] - r["start_s"] for r in ok if kind_of(r) == k], v)
            for q, v in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))
        }
        for k in ("main", "reminder", "replay")
    },
}
print(json.dumps({"summary": summary}), flush=True)
sys.exit(1 if crashes else 0)
