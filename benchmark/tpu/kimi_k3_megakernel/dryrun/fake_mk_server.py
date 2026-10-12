"""Fake megakernel OpenAI server for dry-running mk_client.py: batches exactly B requests
(equal prompt length), logs 'target_batch:' lines like demo_kimi_dspark.py."""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port, B, log_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
lock = threading.Lock()
pending = []
cond = threading.Condition(lock)


def batcher():
    while True:
        with cond:
            while len(pending) < B:
                cond.wait()
            batch = pending[:B]
            del pending[:B]
        lens = {len(p["tokens"]) for p in batch}
        assert len(lens) == 1, f"unequal prompt lengths {lens}"
        steps = max(p["max_tokens"] for p in batch)
        time.sleep(0.002 * steps)
        with open(log_path, "a") as f:
            f.write(
                f"target_batch: {B} requests, {sum(p['max_tokens'] for p in batch)} generated in {0.002 * steps:.6f} s ({steps} steps)\n"
            )
        for p in batch:
            p["done"].set()


threading.Thread(target=batcher, daemon=True).start()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["prompt"]
        tokens = (
            prompt if isinstance(prompt, list) else list(range(len(prompt.split())))
        )
        job = {
            "tokens": tokens,
            "max_tokens": body.get("max_tokens", 16),
            "done": threading.Event(),
        }
        with cond:
            pending.append(job)
            cond.notify_all()
        job["done"].wait()
        text = (
            " Paris 391 return" if isinstance(prompt, str) else " x" * job["max_tokens"]
        )
        out = {
            "choices": [{"text": text, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": job["max_tokens"],
            },
        }
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
