#!/bin/bash
# Usage: nohup bash run-chunk512.sh < /dev/null > chunk512.log 2>&1 &
# psdr: DSA no radix, 512-token prefill chunks; dump the same prompts; compare with the
# 8192 (dump-cold.json) and 1024 (dump-chunk.json) dumps.
S=$(cd "$(dirname "$0")" && pwd); W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
C=gke_sglang-torchtpu_us-central1-c_tpu-cluster; PS=tpuq-qinghan-de1c-w-0-0-6fn6n
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }
k $PS bash $W/stop_wait.sh < /dev/null
k $PS tee $W/glm-port/benchmark/tpu/recipe/configs/glm_5_3_flash_eval/dsa_nospec_chunk512.py < $S/eval_configs/dsa_nospec_chunk512.py > /dev/null
k $PS env HOME=$H setsid -f bash $W/tpu-boot-only.sh dsa_nospec_chunk512 ctl-chunk512 glm-port venv-port < /dev/null > /dev/null 2>&1
until k $PS grep -q BOOT_DONE $W/glm-run/boot-ctl-chunk512.out < /dev/null 2>/dev/null; do sleep 30; done
echo "$(date +%T) psdr: $(k $PS grep -av PrivateUse1 $W/glm-run/boot-ctl-chunk512.out < /dev/null | tr '\n' ' ' | cut -c1-200)"
k $PS $H/venv-port/bin/python $W/prefix_dump.py $W/glm-run/dump-chunk512.json < /dev/null > /dev/null
k $PS cat $W/glm-run/dump-chunk512.json < /dev/null > $S/dump-chunk512.json
python3 - $S/dump-cold.json $S/dump-chunk.json $S/dump-chunk512.json <<'PY'
import json, sys
names = ["8192", "1024", "512"]; d = [json.load(open(p)) for p in sys.argv[1:4]]
def gap(x, y):
    c = set(x["top"]) & set(y["top"]); return max(abs(x["top"][t] - y["top"][t]) for t in c)
for i in range(3):
    print(f"case {i} ({d[0][i]['prompt_tokens']} tok): " + "  ".join(f"{names[a]} vs {names[b]}: {gap(d[a][i], d[b][i]):.4f}" for a, b in ((0, 1), (0, 2), (1, 2))))
PY
echo "$(date +%T) CHUNK512_DONE"
