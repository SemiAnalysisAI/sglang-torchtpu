#!/bin/bash
# psdr: boot DFlash+DSA sized for c16, then 8k1k cookbook bench at c1, c4, c8, c16.
S=$(cd "$(dirname "$0")" && pwd); R=$S/../../glm-dsa-tpu-rebase; W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
C=gke_sglang-torchtpu_us-central1-c_tpu-cluster; P=tpuq-qinghan-de1c-w-0-0-6fn6n
k() { kubectl --context $C -n tpuq exec -i "$P" -c main -- "$@"; }
k bash $W/stop_wait.sh < /dev/null
for f in $(cd $R && git diff --name-only 369d483ac1 HEAD -- python sglang_tpu benchmark/tpu/recipe/configs/glm_5_3_flash); do k tee $W/glm-port/$f < $R/$f > /dev/null; done
k tee $W/glm-port/sglang_tpu/attention/pallas/mla_sparse/kernel.py < $R/sglang_tpu/attention/pallas/mla_sparse/kernel.py > /dev/null
k tee $W/glm-port/benchmark/tpu/recipe/configs/glm_5_3_flash_eval/conc_dsa_dflash_acc35.py < $S/eval_configs/conc_dsa_dflash_acc35.py > /dev/null
for f in bench-cb.sh tpu-boot-only.sh; do k tee $W/$f < $S/$f > /dev/null; done
k env HOME=$H setsid -f bash $W/tpu-boot-only.sh conc_dsa_dflash_acc35 conc-dsa glm-port venv-port < /dev/null > /dev/null 2>&1
until k grep -q BOOT_DONE $W/glm-run/boot-conc-dsa.out < /dev/null 2>/dev/null; do sleep 30; done
echo "$(date +%T) $(k grep -av PrivateUse1 $W/glm-run/boot-conc-dsa.out < /dev/null | tr '\n' ' ' | cut -c1-200)"
if k grep -q BOOT_OK $W/glm-run/boot-conc-dsa.out < /dev/null; then
  for c in 1 4 8 16; do
    k env HOME=$H bash $W/bench-cb.sh conc-dsa $c < /dev/null 2>&1 | grep -E "Successful|Output token thr|Accept|Median TTFT|Median TPOT" | tr -s ' ' | tr '\n' ' ' | sed "s/^/c$c: /"; echo
  done
  k grep -a -c "Traceback\|halted" $W/glm-run/conc-dsa.log < /dev/null | sed 's/^/errors in server log: /'
fi
echo CONC_DONE
