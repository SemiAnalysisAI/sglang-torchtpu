#!/bin/bash
# Usage: attach-offload.sh <pod> <venv> <tag> [smoke|stress]: wait for a boot started by run-offload.sh, then test.
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc; C=gke_sglang-torchtpu_us-central1-c_tpu-cluster; P=$1; V=$2; TAG=$3
k() { kubectl --context $C -n tpuq exec -i "$P" -c main -- "$@"; }
k tee $W/offload_stress.py < $(dirname "$0")/offload_stress.py > /dev/null
until k grep -q BOOT_DONE $W/glm-run/boot-$TAG.out < /dev/null 2>/dev/null; do sleep 30; done
echo "$(date +%T) boot: $(k grep -av PrivateUse1 $W/glm-run/boot-$TAG.out < /dev/null | tr '\n' ' ' | cut -c1-400)"
k grep -a -m6 -iE "hierarchical KV host pool|Raiden KV transfer ready|hybrid pool stack|host restore" $W/glm-run/$TAG.log < /dev/null | cut -c1-260
if k grep -q BOOT_OK $W/glm-run/boot-$TAG.out < /dev/null; then
  if [ "$4" = stress ]; then k $H/$V/bin/python $W/offload_stress.py 30000 12 3 4 < /dev/null 2>&1 | tail -12
  else k $H/$V/bin/python $W/offload_test.py 30000 10 < /dev/null 2>&1 | tail -25; fi
fi
echo "$(date +%T) server errors (if any):"
k grep -a -m1 -A60 "Scheduler hit an exception\|Traceback" $W/glm-run/$TAG.log < /dev/null | grep -E "File \"/dev|Error|raise " | tail -8 | cut -c1-260
echo "$(date +%T) OFFLOAD_DONE"
