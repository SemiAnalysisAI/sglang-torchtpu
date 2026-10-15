#!/bin/bash
# Unattended A/B queue for GLM-5.3-Flash 8k1k on one TPU host.
# usage: exp_queue.sh <queue-file>    (run detached; results -> glm-run/exp-results.jsonl)
# Queue file: one experiment per line, "name|ENV=val ENV2=val ..." (env passed to run-glm53-flash.sh).
# Per playbook: records weight memory (catches toy runs), p50 TPOT over 3 runs, and spread.
set -uo pipefail
W=/dev/shm/home/qinghanc/work; cd $W
Q=${1:?queue file}; OUT=$W/glm-run/exp-results.jsonl
# Persistent copy: only /work/<name> survives the end of a tpuq reservation.
KEEP=$(ls -d /work/* 2>/dev/null | head -1); [[ -n $KEEP ]] && KEEP=$KEEP/glm-flash-exp && mkdir -p $KEEP
keep() { [[ -n ${KEEP:-} ]] && cp -f "$@" $KEEP/ 2>/dev/null; }
. $W/../venv/bin/activate
stop() { $W/stop_sglang.sh TERM >/dev/null; sleep 12; $W/stop_sglang.sh KILL >/dev/null; sleep 3; }
while IFS='|' read -r name envs; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  log=$W/glm-run/exp-$name.log
  echo "[$(date -u +%H:%M:%S)] start $name ($envs)"
  stop
  ( env $envs setsid nohup ./run-glm53-flash.sh > $log 2>&1 < /dev/null & )
  for i in $(seq 240); do grep -qE "fired up|Scheduler hit an exception|Check failed|died during" $log 2>/dev/null && break; sleep 10; done
  if ! grep -q "fired up" $log; then
    echo "{\"name\":\"$name\",\"env\":\"$envs\",\"error\":\"$(grep -m1 -oE 'Scheduler hit an exception.{0,160}|Check failed.{0,160}|died during.{0,80}' $log | tr '"' "'")\"}" >> $OUT
    echo "[$(date -u +%H:%M:%S)] FAILED $name"; keep $OUT $log; continue
  fi
  python - "$name" "$envs" "$OUT" <<'EOF'
import json, statistics, subprocess, sys, urllib.request
name, envs, out = sys.argv[1:]
def post(path, body):
    r = urllib.request.Request("http://127.0.0.1:30000" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
sanity = post("/generate", {"text": "The capital of France is", "sampling_params": {"temperature": 0, "max_new_tokens": 16}})["text"]
info = json.load(urllib.request.urlopen("http://127.0.0.1:30000/get_server_info", timeout=60))
weight_gb = (info.get("internal_states") or [{}])[0].get("memory_usage", {}).get("weight")
res = subprocess.run(["python", "bench_8k1k.py", "--runs", "3"], capture_output=True, text=True).stdout
runs = [json.loads(l.split(":", 1)[1]) for l in res.splitlines() if l.startswith("run ")]
tpot = [r["tpot_ms"] for r in runs]; ttft = [r["ttft_s"] for r in runs]
rec = {"name": name, "env": envs, "tpot_p50_ms": statistics.median(tpot) if tpot else None,
       "tpot_runs": tpot, "ttft_p50_s": statistics.median(ttft) if ttft else None,
       "weight_gb": weight_gb, "sanity": sanity}
open(out, "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec))
EOF
  keep $OUT $log
done < "$Q"
stop
echo "[$(date -u +%H:%M:%S)] queue done"
