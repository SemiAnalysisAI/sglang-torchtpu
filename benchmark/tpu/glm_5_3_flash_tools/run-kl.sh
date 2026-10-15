#!/bin/bash
# Usage: nohup bash run-kl.sh < /dev/null > kl.log 2>&1 &
# KL consistency on one server each: x5kk DSA+radix, psdr dense+radix (diagnostic reference).
S=$(cd "$(dirname "$0")" && pwd); W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
C=gke_sglang-torchtpu_us-central1-c_tpu-cluster; PX=tpuq-qinghan-4d14-w-0-0-m27gm; PS=tpuq-qinghan-de1c-w-0-0-6fn6n
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }
run() {  # pod tree venv config tag
  local P=$1 T=$2 V=$3 CFG=$4 TAG=$5
  k $P bash $W/stop_wait.sh < /dev/null
  for f in kl_dsa_radix.py kl_dense_radix.py kl_dsa_radix_big.py kl_dense_radix_big.py dbg_dense_radix.py dsa_nospec_radix.py; do
    k $P tee $W/$T/benchmark/tpu/recipe/configs/glm_5_3_flash_eval/$f < $S/eval_configs/$f > /dev/null; done
  k $P tee $W/kl_run.py < $S/kl_run.py > /dev/null
  k $P env HOME=$H setsid -f bash $W/tpu-boot-only.sh $CFG $TAG $T $V < /dev/null > /dev/null 2>&1
  until k $P grep -q BOOT_DONE $W/glm-run/boot-$TAG.out < /dev/null 2>/dev/null; do sleep 30; done
  echo "$(date +%T) $TAG: $(k $P grep -av PrivateUse1 $W/glm-run/boot-$TAG.out < /dev/null | tr '\n' ' ' | cut -c1-250)"
  k $P grep -q BOOT_OK $W/glm-run/boot-$TAG.out < /dev/null || return
  k $P env HOME=$H PYTHONPATH=$W/$T/python $H/$V/bin/python $W/kl_run.py 30000 32 512 < /dev/null 2>&1 \
    | grep -E "RESULT|FAILED|Error|Running test" | sed "s/^/$TAG: /"
}
run $PX glm-port-b venv-port-b kl_dsa_radix_big kl-dsa2 &
run $PS glm-port venv-port kl_dense_radix_big kl-dense2 &
wait; echo "$(date +%T) KL_ALL_DONE"
