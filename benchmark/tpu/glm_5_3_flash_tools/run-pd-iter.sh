#!/bin/bash
# Usage: nohup bash run-pd-iter.sh < /dev/null > pd-iter.log 2>&1 &
# One PD iteration on the bootstrapped pair (prefill qinghan-9c1e, decode qinghan-29b5): sync the
# branch's changed sources at HEAD, boot both engines, start the router, send pd_check prompts.
# Prints the first error of an engine that fails.
S=$(cd "$(dirname "$0")" && pwd); R=$S/../../glm-dsa-tpu-rebase; W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
C=gke_sglang-torchtpu_us-central1-c_tpu-cluster
PS=tpuq-qinghan-9c1e-w-0-0-86j66; PX=tpuq-qinghan-29b5-w-0-0-l8njd
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }
launch() { k "$1" env HOME=$H setsid -f bash "${@:2}" < /dev/null > /dev/null 2>&1; }
log() { echo "$(date +%T) $*"; }
for P in $PS $PX; do
  k $P bash $W/stop_wait.sh < /dev/null
  k $P bash -c 'for f in /proc/[0-9]*/cmdline; do grep -qs launch_router $f && kill ${f//[^0-9]/} 2>/dev/null; done; true' < /dev/null
  for f in $(cd $R && git diff --name-only 369d483ac1 HEAD -- python sglang_tpu benchmark/tpu/recipe/configs/glm_5_3_flash); do k $P tee $W/glm-port/$f < $R/$f > /dev/null; done
  ( cd $S/eval_configs && COPYFILE_DISABLE=1 tar cf - --no-xattrs *.py ) | k $P tar xf - -C $W/glm-port/benchmark/tpu/recipe/configs/glm_5_3_flash_eval
  for f in run-cfg-port.sh stop_wait.sh stop_sglang.sh pd_check.py; do k $P tee $W/$f < $S/$f > /dev/null; done
done
log "synced HEAD $(cd $R && git rev-parse --short HEAD)"
PS_IP=$(kubectl --context $C -n tpuq get pod $PS -o jsonpath='{.status.podIP}'); PX_IP=$(kubectl --context $C -n tpuq get pod $PX -o jsonpath='{.status.podIP}')
TAG=pd$(date +%H%M)
launch $PS $W/run-cfg-port.sh "glm_5_3_flash_eval/dsa_pd.py:PREFILL_CONFIG" $TAG-prefill glm-port venv-port
launch $PX $W/run-cfg-port.sh "glm_5_3_flash_eval/dsa_pd.py:DECODE_CONFIG" $TAG-decode glm-port venv-port
log "launched $TAG: prefill $PS_IP, decode $PX_IP"
ready() { while :; do k $1 grep -aq "fired up" $W/glm-run/$2.log < /dev/null 2>/dev/null && return 0
  k $1 grep -aq "Traceback" $W/glm-run/$2.log < /dev/null 2>/dev/null && return 1; sleep 20; done; }
err() { k $1 grep -a -m1 -A60 "Scheduler hit an exception\|Traceback" $W/glm-run/$2.log < /dev/null | grep -E "File \"/dev|Error|raise " | tail -6 | cut -c1-260; }
okp=1; okd=1
ready $PS $TAG-prefill || { okp=0; log "prefill engine failed:"; err $PS $TAG-prefill; }
ready $PX $TAG-decode || { okd=0; log "decode engine failed:"; err $PX $TAG-decode; }
if [ $okp = 1 ] && [ $okd = 1 ]; then
  log "both engines up; router + requests"
  k $PX env HOME=$H setsid -f bash -c "$H/venv-port/bin/python -m sglang_router.launch_router --pd-disaggregation --mini-lb --prefill http://$PS_IP:30000 8998 --decode http://$PX_IP:30000 --host 0.0.0.0 --port 8000 > $W/glm-run/$TAG-router.log 2>&1" < /dev/null > /dev/null 2>&1
  until k $PX grep -q "Application startup complete" $W/glm-run/$TAG-router.log < /dev/null 2>/dev/null; do sleep 10; done
  log "router up"
  k $PX timeout 900 $H/venv-port/bin/python $W/pd_check.py http://127.0.0.1:8000 $W/glm-run/$TAG-out.json < /dev/null 2>&1 | tail -8
  k $PX cat $W/glm-run/$TAG-out.json < /dev/null > $S/pd-out.json 2>/dev/null
  log "errors after requests (if any):"; err $PX $TAG-decode; err $PS $TAG-prefill
  log "raiden warnings:"; k $PX grep -a -m3 "RaidenKVManager.*raised\|Raiden.*fail" $W/glm-run/$TAG-decode.log < /dev/null | cut -c1-220
fi
log "PD_ITER_DONE $TAG"
