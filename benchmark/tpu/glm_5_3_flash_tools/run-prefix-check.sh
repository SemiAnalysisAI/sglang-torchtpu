#!/bin/bash
# Usage: nohup bash run-prefix-check.sh < /dev/null > prefix-check.log 2>&1 &
# x5kk: stop leftovers, boot DSA no-spec with radix, run prefix_logprob_test.py.
S=$(cd "$(dirname "$0")" && pwd); W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
C=gke_sglang-torchtpu_us-central1-c_tpu-cluster; P=tpuq-qinghan-4d14-w-0-0-m27gm; PS=tpuq-qinghan-de1c-w-0-0-6fn6n
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }
k $PS bash $W/stop_wait.sh < /dev/null; k $P bash $W/stop_wait.sh < /dev/null
k $P tee $W/prefix_logprob_test.py < $S/prefix_logprob_test.py > /dev/null
k $P tee $W/glm-port-b/benchmark/tpu/recipe/configs/glm_5_3_flash_eval/dsa_nospec_radix.py < $S/eval_configs/dsa_nospec_radix.py > /dev/null
k $P env HOME=$H setsid -f bash $W/tpu-boot-only.sh dsa_nospec_radix pfx-check glm-port-b venv-port-b < /dev/null > /dev/null 2>&1
echo "$(date +%T) booting pfx-check on x5kk"
until k $P grep -q BOOT_DONE $W/glm-run/boot-pfx-check.out < /dev/null 2>/dev/null; do sleep 30; done
echo "$(date +%T) $(k $P grep -av PrivateUse1 $W/glm-run/boot-pfx-check.out < /dev/null | tr '\n' ' ')"
k $P grep -q BOOT_OK $W/glm-run/boot-pfx-check.out < /dev/null && k $P $H/venv-port-b/bin/python $W/prefix_logprob_test.py < /dev/null
echo "$(date +%T) PREFIX_CHECK_DONE"
