#!/bin/bash
# Usage: nohup bash run-offload.sh <pod> <tree> <venv> [tag] < /dev/null > offload.log 2>&1 &
# KV offload smoke on one slice: sync HEAD's changed sources, boot dsa_nospec_hicache,
# run offload_test.py, then print the server's HiCache/raiden lines and first error.
S=$(cd "$(dirname "$0")" && pwd); R=$S/../../glm-dsa-tpu-rebase; W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
C=gke_sglang-torchtpu_us-central1-c_tpu-cluster; P=$1; T=$2; V=$3; TAG=${4:-offload}
k() { kubectl --context $C -n tpuq exec -i "$P" -c main -- "$@"; }
log() { echo "$(date +%T) $*"; }
k bash $W/stop_wait.sh < /dev/null
for f in $(cd $R && git diff --name-only 369d483ac1 HEAD -- python sglang_tpu benchmark/tpu/recipe/configs/glm_5_3_flash); do k tee $W/$T/$f < $R/$f > /dev/null; done
( cd $S/eval_configs && COPYFILE_DISABLE=1 tar cf - --no-xattrs *.py ) | k tar xf - -C $W/$T/benchmark/tpu/recipe/configs/glm_5_3_flash_eval
for f in tpu-boot-only.sh tpu_env.sh offload_test.py offload_stress.py; do k tee $W/$f < $S/$f > /dev/null; done
log "synced HEAD $(cd $R && git rev-parse --short HEAD) to $P"
k env HOME=$H setsid -f bash $W/tpu-boot-only.sh dsa_nospec_hicache $TAG $T $V < /dev/null > /dev/null 2>&1
until k grep -q BOOT_DONE $W/glm-run/boot-$TAG.out < /dev/null 2>/dev/null; do sleep 30; done
log "boot: $(k grep -av PrivateUse1 $W/glm-run/boot-$TAG.out < /dev/null | tr '\n' ' ' | cut -c1-400)"
k grep -a -m6 -iE "hierarchical KV host pool|Raiden KV transfer ready|hybrid pool stack|host restore" $W/glm-run/$TAG.log < /dev/null | cut -c1-260
if k grep -q BOOT_OK $W/glm-run/boot-$TAG.out < /dev/null; then
  if [ "$TEST" = stress ]; then k $H/$V/bin/python $W/offload_stress.py 30000 12 3 4 < /dev/null 2>&1 | tail -12
  else k $H/$V/bin/python $W/offload_test.py 30000 ${N_DOCS:-10} < /dev/null 2>&1 | tail -25; fi
fi
log "server errors (if any):"
k grep -a -m1 -A60 "Scheduler hit an exception\|Traceback" $W/glm-run/$TAG.log < /dev/null | grep -E "File \"/dev|Error|raise " | tail -8 | cut -c1-260
log "OFFLOAD_DONE"
