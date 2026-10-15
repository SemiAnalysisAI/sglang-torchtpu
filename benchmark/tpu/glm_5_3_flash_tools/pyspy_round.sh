#!/bin/bash
# Usage: pyspy_round.sh <venv> <out> -- py-spy the TP0 scheduler during an 8k1k c1 decode (server must be up).
W=/dev/shm/home/qinghanc/work; PY=/dev/shm/home/qinghanc/$1/bin/python; OUT=$2
PID=$(grep -l "^sglang::scheduler_TP0" /proc/[0-9]*/cmdline 2>/dev/null | head -1 | cut -d/ -f3); echo "TP0 pid $PID"
$PY - <<'PYEOF' &
import json, urllib.request, random
rnd = random.Random(2); ids = [rnd.randrange(1000, 100000) for _ in range(8192)]
req = urllib.request.Request("http://127.0.0.1:30000/generate", data=json.dumps({"input_ids": ids, "sampling_params": {"max_new_tokens": 1024, "ignore_eos": True, "temperature": 0}}).encode(), headers={"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=900).read()
PYEOF
sleep 1
py-spy record --nonblocking -r 997 -d 2 -f raw -p $PID -o $OUT
wait; echo PYSPY_DONE
