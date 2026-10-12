import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(sys.argv[1])


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(
            {"data": [{"id": "fake"}]}
            if self.path.startswith("/v1/models")
            else {"internal_states": [{}]}
        )

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/start_profile":
            out = body["output_dir"]
            os.makedirs(out + "/profile-TP-0", exist_ok=True)
            time.sleep(1)
            open(out + "/profile-TP-0/fake.xplane.pb", "w").write("x")
            print("start_profile", body, flush=True)
            return self._send({"ok": True})
        if self.path == "/generate":
            n = len(body["input_ids"])
            m = body["sampling_params"]["max_new_tokens"]
            time.sleep(2)
            return self._send(
                {
                    "text": "x",
                    "meta_info": {
                        "prompt_tokens": n,
                        "completion_tokens": m,
                        "e2e_latency": 2.0,
                    },
                }
            )
        self._send({})


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
