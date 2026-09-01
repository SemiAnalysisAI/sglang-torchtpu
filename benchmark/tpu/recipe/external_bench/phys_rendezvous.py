"""Exchange (physical worker number -> pod hostname) across a TPU slice.

Why: GKE maps a JobSet pod's completion index (TPU_WORKER_ID) to a PHYSICAL
slice host arbitrarily. Splitting a 2x2x4 slice into two 2x2x2 sub-slice
engines must group PHYSICALLY ADJACENT hosts -- a group like {phys0, phys2}
folds two hosts onto the same 2x2x2 coordinates and every collective dies at
compile with "replica id #N seen more than once" (observed: iterations 6 and 8
failed with interleaved placements; 4 and 7 passed with aligned ones). Each
host's physical number comes from its instance metadata (agent-worker-number);
this exchanges them so every pod can build the physically-ordered host list.

Pod index 0 (an arbitrary physical host) serves a registry; every pod registers
its (phys, hostname) and polls until all N are present. Prints N lines
"<phys> <hostname>" sorted by phys, then exits 0.

    phys_rendezvous.py --self-phys 2 --self-host $(hostname) \
        --server-host <pod0-hostname> --num-hosts 4 [--serve]
"""

import argparse
import http.server
import json
import sys
import threading
import time
import urllib.parse
import urllib.request

PORT = 29998


def _serve(num_hosts):
    entries = {}
    lock = threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            if url.path == "/register":
                q = urllib.parse.parse_qs(url.query)
                with lock:
                    entries[int(q["phys"][0])] = q["host"][0]
            with lock:
                done = len(entries) >= num_hosts
                body = json.dumps(entries).encode()
            self.send_response(200 if done or url.path == "/register" else 503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-phys", type=int, required=True)
    ap.add_argument("--self-host", required=True)
    ap.add_argument("--server-host", required=True)
    ap.add_argument("--num-hosts", type=int, default=4)
    ap.add_argument("--serve", action="store_true", help="this pod hosts the registry")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()

    if a.serve:
        _serve(a.num_hosts)

    register = (
        f"http://{a.server_host}:{PORT}/register?"
        f"phys={a.self_phys}&host={urllib.parse.quote(a.self_host)}"
    )
    poll = f"http://{a.server_host}:{PORT}/map"
    deadline = time.time() + a.timeout
    entries = {}
    while time.time() < deadline:
        try:
            urllib.request.urlopen(register, timeout=5).read()
            with urllib.request.urlopen(poll, timeout=5) as r:
                if r.status == 200:
                    entries = {int(k): v for k, v in json.loads(r.read()).items()}
                    if len(entries) >= a.num_hosts:
                        break
        except Exception:
            pass
        time.sleep(3)
    else:
        print(
            f"phys_rendezvous: only {len(entries)}/{a.num_hosts} hosts "
            f"registered within {a.timeout}s",
            file=sys.stderr,
        )
        return 1

    for phys in sorted(entries):
        print(f"{phys} {entries[phys]}")
    if a.serve:
        # Keep the registry answering while slower peers fetch the completed
        # map; they poll every 3s, so a short linger covers the stragglers.
        sys.stdout.flush()
        time.sleep(30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
