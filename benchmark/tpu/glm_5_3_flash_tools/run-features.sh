#!/bin/bash
# Usage: nohup bash run-features.sh < /dev/null > features.log 2>&1 &
# Smoke tests for GLM-5.3-Flash DSA (no spec) on two slices:
#   phase 1, in parallel:
#     x5kk (4d14): prefix cache  -- radix on; prefix_test.py (cold / repeat / shared prefix)
#     psdr (de1c): KV offload     -- radix + Raiden HiCache in host RAM; prefix_test.py if it boots
#   phase 2: PD -- prefill engine on psdr, decode engine on x5kk, Raiden transfer, mini-LB on x5kk
# Every outcome (boot verdict, error line, test output) is printed here.
S=$(cd "$(dirname "$0")" && pwd); R=$S/../../glm-dsa-tpu-rebase
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc; C=gke_sglang-torchtpu_us-central1-c_tpu-cluster
PX=tpuq-qinghan-4d14-w-0-0-m27gm; PS=tpuq-qinghan-de1c-w-0-0-6fn6n   # x5kk, psdr
TX=glm-port-b; VX=venv-port-b; TS=glm-port; VS=venv-port
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }
launch() { k "$1" env HOME=$H setsid -f bash "${@:2}" < /dev/null > /dev/null 2>&1; }
log() { echo "$(date +%T) $*"; }

sync_tree() {  # HEAD sources changed on this branch, eval configs, helper scripts
  local P=$1 T=$2
  for f in $(cd $R && git diff --name-only 369d483ac1 HEAD -- python sglang_tpu benchmark/tpu/recipe/configs/glm_5_3_flash); do
    k $P tee $W/$T/$f < $R/$f > /dev/null
  done
  ( cd $S/eval_configs && COPYFILE_DISABLE=1 tar cf - --no-xattrs *.py ) \
    | k $P tar xf - -C $W/$T/benchmark/tpu/recipe/configs/glm_5_3_flash_eval
  for f in tpu-boot-only.sh run-cfg-port.sh stop_wait.sh stop_sglang.sh prefix_test.py; do
    k $P tee $W/$f < $S/$f > /dev/null
  done
}
verdict() {  # $1 pod, $2 tag -> waits for BOOT_DONE, prints the verdict, returns 0 on BOOT_OK
  until k $1 grep -q BOOT_DONE $W/glm-run/boot-$2.out < /dev/null 2>/dev/null; do sleep 30; done
  log "$2: $(k $1 grep -av PrivateUse1 $W/glm-run/boot-$2.out < /dev/null | tr '\n' ' ' | cut -c1-400)"
  k $1 grep -a -m3 -E "ValueError|NotImplementedError|RuntimeError|AssertionError" $W/glm-run/$2.log < /dev/null | cut -c1-300
  k $1 grep -q BOOT_OK $W/glm-run/boot-$2.out < /dev/null
}
wait_ready() {  # $1 pod, $2 log -> 0 when fired up, 1 on traceback / silent 25 min
  while :; do
    k $1 grep -aq "fired up" $W/glm-run/$2.log < /dev/null 2>/dev/null && return 0
    k $1 grep -aq "Traceback" $W/glm-run/$2.log < /dev/null 2>/dev/null && return 1
    age=$(k $1 stat -c %Y $W/glm-run/$2.log < /dev/null 2>/dev/null); [ -n "$age" ] && [ $(( $(date +%s) - age )) -gt 1500 ] && return 1
    sleep 30
  done
}
first_error() { k $1 grep -a -m3 -E "Error|raise |Exception" $W/glm-run/$2.log < /dev/null | grep -v "UserWarning" | cut -c1-300; }

sync_tree $PX $TX; sync_tree $PS $TS
# SMOKE-ONLY (uncommitted): allow --language-model-only with PD on the slices.
SP=/private/tmp/claude-502/-Users-qinghan-chen-Desktop-sglang-jax-workdir-glm-dsa-tpu/a2719a6b-8303-4d9e-9a9f-33f0b4ad3e47/scratchpad/model_hook.py
k $PX tee $W/$TX/python/sglang/srt/arg_groups/model_hook.py < $SP > /dev/null
k $PS tee $W/$TS/python/sglang/srt/arg_groups/model_hook.py < $SP > /dev/null
log "synced HEAD $(cd $R && git rev-parse --short HEAD) to both slices"

# ---- phase 1 ----
launch $PX $W/tpu-boot-only.sh dsa_nospec_radix feat-prefix $TX $VX
[ -z "$SKIP_HICACHE" ] && launch $PS $W/tpu-boot-only.sh dsa_nospec_hicache feat-hicache $TS $VS
log "phase 1 launched: x5kk feat-prefix${SKIP_HICACHE:-, psdr feat-hicache}"
( if verdict $PX feat-prefix; then log "prefix test:"; k $PX $H/$VX/bin/python $W/prefix_test.py < /dev/null; fi ) &
[ -z "$SKIP_HICACHE" ] && ( if verdict $PS feat-hicache; then log "KV offload: prefix test (watch cached_tokens):"; k $PS $H/$VS/bin/python $W/prefix_test.py < /dev/null; fi ) &
wait
log "phase 1 done"

# ---- phase 2: PD ----
k $PX bash $W/stop_wait.sh < /dev/null; k $PS bash $W/stop_wait.sh < /dev/null
PS_IP=$(kubectl --context $C -n tpuq get pod $PS -o jsonpath='{.status.podIP}')
PX_IP=$(kubectl --context $C -n tpuq get pod $PX -o jsonpath='{.status.podIP}')
launch $PS $W/run-cfg-port.sh "glm_5_3_flash_eval/dsa_pd.py:PREFILL_CONFIG" feat-pd-prefill $TS $VS
launch $PX $W/run-cfg-port.sh "glm_5_3_flash_eval/dsa_pd.py:DECODE_CONFIG" feat-pd-decode $TX $VX
log "phase 2 launched: prefill on psdr ($PS_IP), decode on x5kk ($PX_IP)"
okp=0; okd=0
wait_ready $PS feat-pd-prefill && okp=1 || { log "PD prefill engine failed:"; first_error $PS feat-pd-prefill; }
wait_ready $PX feat-pd-decode && okd=1 || { log "PD decode engine failed:"; first_error $PX feat-pd-decode; }
log "PD engines: prefill_up=$okp decode_up=$okd"
if [ $okp = 1 ] && [ $okd = 1 ]; then
  launch $PX $H/$VX/bin/python -m sglang_router.launch_router --pd-disaggregation --mini-lb \
    --prefill "http://$PS_IP:30000" 8998 --decode "http://$PX_IP:30000" --host 0.0.0.0 --port 8000
  sleep 30
  log "PD request through the router:"
  k $PX $H/$VX/bin/python -c '
import json, urllib.request
body = {"text": "The capital of France is", "sampling_params": {"temperature": 0, "max_new_tokens": 32}}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600)
print(json.load(r))' < /dev/null 2>&1 | tail -3
  log "decode log errors (if any):"; first_error $PX feat-pd-decode
  log "prefill log errors (if any):"; first_error $PS feat-pd-prefill
fi
log "FEATURES_DONE"
