#!/bin/bash
# Usage: nohup bash run-chunk-control.sh < /dev/null > chunk-control.log 2>&1 &
# x5kk (radix server, still up): dump cold logprobs (cache flushed per prompt).
# psdr: boot DSA no-spec, radix off, 1024-token prefill chunks; dump the same prompts.
S=$(cd "$(dirname "$0")" && pwd); W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
C=gke_sglang-torchtpu_us-central1-c_tpu-cluster; PX=tpuq-qinghan-4d14-w-0-0-m27gm; PS=tpuq-qinghan-de1c-w-0-0-6fn6n
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }
for P in $PX $PS; do k $P tee $W/prefix_dump.py < $S/prefix_dump.py > /dev/null; done
k $PS tee $W/glm-port/benchmark/tpu/recipe/configs/glm_5_3_flash_eval/dsa_nospec_chunk1024.py < $S/eval_configs/dsa_nospec_chunk1024.py > /dev/null
k $PS env HOME=$H setsid -f bash $W/tpu-boot-only.sh dsa_nospec_chunk1024 ctl-chunk1024 glm-port venv-port < /dev/null > /dev/null 2>&1
echo "$(date +%T) x5kk cold (unsplit) dump:"; k $PX $H/venv-port-b/bin/python $W/prefix_dump.py $W/glm-run/dump-cold.json 30000 --flush < /dev/null
k $PX cat $W/glm-run/dump-cold.json < /dev/null > $S/dump-cold.json
until k $PS grep -q BOOT_DONE $W/glm-run/boot-ctl-chunk1024.out < /dev/null 2>/dev/null; do sleep 30; done
echo "$(date +%T) psdr: $(k $PS grep -av PrivateUse1 $W/glm-run/boot-ctl-chunk1024.out < /dev/null | tr '\n' ' ')"
k $PS $H/venv-port/bin/python $W/prefix_dump.py $W/glm-run/dump-chunk.json < /dev/null
k $PS cat $W/glm-run/dump-chunk.json < /dev/null > $S/dump-chunk.json
python3 - $S/dump-cold.json $S/dump-chunk.json <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for i, (x, y) in enumerate(zip(a, b)):
    common = set(x["top"]) & set(y["top"])
    gap = max(abs(x["top"][t] - y["top"][t]) for t in common) if common else float("inf")
    print(f"case {i} ({x['prompt_tokens']} tok): unsplit vs 1024-chunked max top-5 gap {gap:.4f} text {x['text']!r} vs {y['text']!r}")
PY
echo "$(date +%T) CHUNK_CONTROL_DONE"
