#!/bin/bash
# Usage: nohup bash run-decode-probe.sh <pod> <tree> <venv> < /dev/null > decode-probe.log 2>&1 &
# Boot dsa_nospec_radix (KDA slots sized for radix) and run decode_hit_probe.py.
S=$(cd "$(dirname "$0")" && pwd); R=$S/../../glm-dsa-tpu-rebase; W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
C=gke_sglang-torchtpu_us-central1-c_tpu-cluster; P=$1; T=$2; V=$3; TAG=decprobe
k() { kubectl --context $C -n tpuq exec -i "$P" -c main -- "$@"; }
k bash $W/stop_wait.sh < /dev/null
for f in $(cd $R && git diff --name-only 369d483ac1 HEAD -- python sglang_tpu benchmark/tpu/recipe/configs/glm_5_3_flash); do k tee $W/$T/$f < $R/$f > /dev/null; done
( cd $S/eval_configs && COPYFILE_DISABLE=1 tar cf - --no-xattrs *.py ) | k tar xf - -C $W/$T/benchmark/tpu/recipe/configs/glm_5_3_flash_eval
for f in tpu-boot-only.sh tpu_env.sh decode_hit_probe.py; do k tee $W/$f < $S/$f > /dev/null; done
k env HOME=$H setsid -f bash $W/tpu-boot-only.sh pc_recipe $TAG $T $V < /dev/null > /dev/null 2>&1
until k grep -q BOOT_DONE $W/glm-run/boot-$TAG.out < /dev/null 2>/dev/null; do sleep 30; done
echo "$(date +%T) boot: $(k grep -av PrivateUse1 $W/glm-run/boot-$TAG.out < /dev/null | tr '\n' ' ' | cut -c1-300)"
k grep -q BOOT_OK $W/glm-run/boot-$TAG.out < /dev/null && k $H/$V/bin/python $W/decode_hit_probe.py < /dev/null 2>&1 | tail -6
k grep -a -m3 "mamba_track_interval\|track_interval" $W/glm-run/$TAG.log < /dev/null | cut -c1-200
echo "$(date +%T) DECODE_PROBE_RUN_DONE"
