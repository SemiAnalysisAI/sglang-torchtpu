#!/bin/bash
# Usage: nohup bash run-gap-bisect.sh < /dev/null > gap-bisect.log 2>&1 &
# Prefill chunk-boundary logprob gap, cold vs cache-hit on the same server:
#   x5kk: DSA + radix (server already up), short prompts (~150-700 tokens)
#   psdr: DIAGNOSTIC dense MLA + radix, prompts from ~150 tokens to ~5k
S=$(cd "$(dirname "$0")" && pwd); W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
C=gke_sglang-torchtpu_us-central1-c_tpu-cluster; PX=tpuq-qinghan-4d14-w-0-0-m27gm; PS=tpuq-qinghan-de1c-w-0-0-6fn6n
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }
for P in $PX $PS; do k $P tee $W/prefix_logprob_test.py < $S/prefix_logprob_test.py > /dev/null; done
k $PS bash $W/stop_wait.sh < /dev/null
k $PS tee $W/glm-port/benchmark/tpu/recipe/configs/glm_5_3_flash_eval/dbg_dense_radix.py < $S/eval_configs/dbg_dense_radix.py > /dev/null
k $PS env HOME=$H setsid -f bash $W/tpu-boot-only.sh dbg_dense_radix gap-dense glm-port venv-port < /dev/null > /dev/null 2>&1
echo "$(date +%T) == x5kk DSA+radix, short prompts:"
k $PX env LINES=8,20,40 $H/venv-port-b/bin/python $W/prefix_logprob_test.py < /dev/null
until k $PS grep -q BOOT_DONE $W/glm-run/boot-gap-dense.out < /dev/null 2>/dev/null; do sleep 30; done
echo "$(date +%T) psdr: $(k $PS grep -av PrivateUse1 $W/glm-run/boot-gap-dense.out < /dev/null | tr '\n' ' ' | cut -c1-300)"
k $PS grep -aq "attends densely" $W/glm-run/gap-dense.log < /dev/null && echo "(confirmed dense attention on psdr)"
echo "$(date +%T) == psdr dense+radix:"
k $PS env LINES=8,40,120,300 $H/venv-port/bin/python $W/prefix_logprob_test.py < /dev/null
echo "$(date +%T) GAP_BISECT_DONE"
