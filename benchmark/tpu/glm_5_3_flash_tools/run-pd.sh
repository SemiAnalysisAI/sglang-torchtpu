#!/bin/bash
# Usage: nohup bash run-pd.sh < /dev/null > pd.log 2>&1 &
# PD on the two new slices (qinghan-9c1e prefill, qinghan-29b5 decode): wait for the hosts,
# bootstrap both at HEAD, boot the engines, router, same prompts. The single-server reference
# comes from x5kk's DSA server once the KL run finishes.
S=$(cd "$(dirname "$0")" && pwd); R=$S/../../glm-dsa-tpu-rebase; W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
C=gke_sglang-torchtpu_us-central1-c_tpu-cluster; PREF=tpuq-qinghan-4d14-w-0-0-m27gm
pod_of() { tpuq status 2>/dev/null | grep -A1 "^  $1 " | grep -o "tpuq-$1-w-0-0-[a-z0-9]*" | head -1; }
until [ -n "$(pod_of qinghan-9c1e)" ] && [ -n "$(pod_of qinghan-29b5)" ]; do sleep 30; done
PS=$(pod_of qinghan-9c1e); PX=$(pod_of qinghan-29b5)
until kubectl --context $C -n tpuq exec $PS -c main -- true < /dev/null 2>/dev/null && kubectl --context $C -n tpuq exec $PX -c main -- true < /dev/null 2>/dev/null; do sleep 20; done
echo "$(date +%T) hosts up: prefill $PS, decode $PX; bootstrapping"
( bash $S/bootstrap-port-pod.sh $PS < /dev/null > $S/pd-bootstrap-prefill.log 2>&1 ) &
( bash $S/bootstrap-port-pod.sh $PX < /dev/null > $S/pd-bootstrap-decode.log 2>&1 ) &
wait; echo "$(date +%T) bootstrapped: $(tail -1 $S/pd-bootstrap-prefill.log) / $(tail -1 $S/pd-bootstrap-decode.log)"
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }
launch() { k "$1" env HOME=$H setsid -f bash "${@:2}" < /dev/null > /dev/null 2>&1; }
log() { echo "$(date +%T) $*"; }
for P in $PX $PS; do k $P tee $W/pd_check.py < $S/pd_check.py > /dev/null; done
for spec in $PX:glm-port $PS:glm-port; do P=${spec%%:*}; T=${spec#*:}
  for f in $(cd $R && git diff --name-only 369d483ac1 HEAD -- python sglang_tpu benchmark/tpu/recipe/configs/glm_5_3_flash); do k $P tee $W/$T/$f < $R/$f > /dev/null; done
  ( cd $S/eval_configs && COPYFILE_DISABLE=1 tar cf - --no-xattrs *.py ) | k $P tar xf - -C $W/$T/benchmark/tpu/recipe/configs/glm_5_3_flash_eval
  for f in run-cfg-port.sh stop_wait.sh stop_sglang.sh; do k $P tee $W/$f < $S/$f > /dev/null; done
done
log "synced HEAD $(cd $R && git rev-parse --short HEAD)"
PS_IP=$(kubectl --context $C -n tpuq get pod $PS -o jsonpath='{.status.podIP}'); PX_IP=$(kubectl --context $C -n tpuq get pod $PX -o jsonpath='{.status.podIP}')
launch $PS $W/run-cfg-port.sh "glm_5_3_flash_eval/dsa_pd.py:PREFILL_CONFIG" pd-prefill glm-port venv-port
launch $PX $W/run-cfg-port.sh "glm_5_3_flash_eval/dsa_pd.py:DECODE_CONFIG" pd-decode glm-port venv-port
log "PD launched: prefill psdr $PS_IP, decode x5kk $PX_IP"
ready() { while :; do k $1 grep -aq "fired up" $W/glm-run/$2.log < /dev/null 2>/dev/null && return 0
  k $1 grep -aq "Traceback" $W/glm-run/$2.log < /dev/null 2>/dev/null && return 1; sleep 20; done; }
err() { k $1 grep -a -m1 -B2 -A30 "Traceback" $W/glm-run/$2.log < /dev/null | grep -E "File \"/dev|Error|raise " | tail -5 | cut -c1-260; }
ready $PS pd-prefill || { log "prefill engine failed:"; err $PS pd-prefill; }
ready $PX pd-decode || { log "decode engine failed:"; err $PX pd-decode; }
if k $PS grep -aq "fired up" $W/glm-run/pd-prefill.log < /dev/null && k $PX grep -aq "fired up" $W/glm-run/pd-decode.log < /dev/null; then
  log "both engines up; starting router"
  launch $PX $H/venv-port/bin/python -m sglang_router.launch_router --pd-disaggregation --mini-lb \
    --prefill "http://$PS_IP:30000" 8998 --decode "http://$PX_IP:30000" --host 0.0.0.0 --port 8000
  sleep 40
  log "PD outputs:"
  k $PX $H/venv-port/bin/python $W/pd_check.py http://127.0.0.1:8000 $W/glm-run/pd-out.json < /dev/null 2>&1 | tail -8
  k $PX cat $W/glm-run/pd-out.json < /dev/null > $S/pd-out.json
  until grep -q KL_ALL_DONE $S/kl.log 2>/dev/null; do sleep 30; done
  k $PREF tee $W/pd_check.py < $S/pd_check.py > /dev/null
  k $PREF tee $W/pd-out.json < $S/pd-out.json > /dev/null
  log "single-server DSA (x5kk) vs PD:"
  k $PREF $H/venv-port-b/bin/python $W/pd_check.py http://127.0.0.1:30000 $W/glm-run/pd-ref.json $W/pd-out.json < /dev/null 2>&1 | tail -8
  log "engine errors after requests (if any):"; err $PX pd-decode; err $PS pd-prefill
fi
log "PD_DONE"
