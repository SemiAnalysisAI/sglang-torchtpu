#!/bin/bash
# Verify the pool-size writer switch: psdr DSA 262144-token pool; x5kk AIME DFlash+DSA 330k context (3.3M pool).
S=$(cd "$(dirname "$0")" && pwd); R=$S/../../glm-dsa-tpu-rebase; W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc; C=gke_sglang-torchtpu_us-central1-c_tpu-cluster
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }
one() { local P=$1 T=$2 V=$3 CFG=$4 TAG=$5
  k $P bash $W/stop_wait.sh < /dev/null
  for f in $(cd $R && git diff --name-only 369d483ac1 HEAD -- python sglang_tpu benchmark/tpu/recipe/configs/glm_5_3_flash); do k $P tee $W/$T/$f < $R/$f > /dev/null; done
  k $P tee $W/$T/sglang_tpu/attention/pallas/mla_sparse/kernel.py < $R/sglang_tpu/attention/pallas/mla_sparse/kernel.py > /dev/null
  ( cd $S/eval_configs && COPYFILE_DISABLE=1 tar cf - --no-xattrs *.py ) | k $P tar xf - -C $W/$T/benchmark/tpu/recipe/configs/glm_5_3_flash_eval
  k $P env HOME=$H setsid -f bash $W/tpu-boot-only.sh $CFG $TAG $T $V < /dev/null > /dev/null 2>&1
  until k $P grep -q BOOT_DONE $W/glm-run/boot-$TAG.out < /dev/null 2>/dev/null; do sleep 30; done
  echo "$(date +%T) $TAG: $(k $P grep -av PrivateUse1 $W/glm-run/boot-$TAG.out < /dev/null | tr '\n' ' ' | cut -c1-220)"
  k $P grep -a -m1 "KV Cache is allocated" $W/glm-run/$TAG.log < /dev/null | cut -c1-160; }
one tpuq-qinghan-de1c-w-0-0-6fn6n glm-port venv-port pool_dsa_262144 fix-262k &
one tpuq-qinghan-4d14-w-0-0-m27gm glm-port-b venv-port-b aime_dflash_dsa_330k fix-aime330k &
wait; echo "$(date +%T) BIGPOOL_DONE"
