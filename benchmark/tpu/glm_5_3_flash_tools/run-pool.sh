#!/bin/bash
# Usage: nohup bash run-pool.sh < /dev/null > pool.log 2>&1 &
# Boot DSA no-spec with a 65536-token pool on psdr and 131072 on x5kk; report verdicts.
S=$(cd "$(dirname "$0")" && pwd); W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
C=gke_sglang-torchtpu_us-central1-c_tpu-cluster
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }
one() { local P=$1 T=$2 V=$3 N=$4
  k $P bash $W/stop_wait.sh < /dev/null
  k $P tee $W/$T/benchmark/tpu/recipe/configs/glm_5_3_flash_eval/pool_dsa_$N.py < $S/eval_configs/pool_dsa_$N.py > /dev/null
  k $P env HOME=$H setsid -f bash $W/tpu-boot-only.sh pool_dsa_$N pool-$N $T $V < /dev/null > /dev/null 2>&1
  until k $P grep -q BOOT_DONE $W/glm-run/boot-pool-$N.out < /dev/null 2>/dev/null; do sleep 30; done
  echo "$(date +%T) pool $N: $(k $P grep -av PrivateUse1 $W/glm-run/boot-pool-$N.out < /dev/null | tr '\n' ' ' | cut -c1-200)"
  k $P grep -a -m1 "KV Cache is allocated" $W/glm-run/pool-$N.log < /dev/null | cut -c1-200; }
one tpuq-qinghan-de1c-w-0-0-6fn6n glm-port venv-port 262144 &
one2() { local P=$1 T=$2 V=$3
  k $P bash $W/stop_wait.sh < /dev/null
  for f in kl_dsa_radix_big16.py kl_dsa_radix_big.py kl_dsa_radix.py dsa_nospec_radix.py; do k $P tee $W/$T/benchmark/tpu/recipe/configs/glm_5_3_flash_eval/$f < $S/eval_configs/$f > /dev/null; done
  k $P env HOME=$H setsid -f bash $W/tpu-boot-only.sh kl_dsa_radix_big16 kl-big16 $T $V < /dev/null > /dev/null 2>&1
  until k $P grep -q BOOT_DONE $W/glm-run/boot-kl-big16.out < /dev/null 2>/dev/null; do sleep 30; done
  echo "$(date +%T) kl-big16: $(k $P grep -av PrivateUse1 $W/glm-run/boot-kl-big16.out < /dev/null | tr '\n' ' ' | cut -c1-200)"; }
one2 tpuq-qinghan-4d14-w-0-0-m27gm glm-port-b venv-port-b &
wait; echo "$(date +%T) POOL_DONE"
