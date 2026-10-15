#!/bin/bash
# x5kk: DSA prefix-cache KL (131k pool, 24 prompts). psdr: 262k pool with sparse-gather padding at page 0.
S=$(cd "$(dirname "$0")" && pwd); W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc; C=gke_sglang-torchtpu_us-central1-c_tpu-cluster
D=/private/tmp/claude-502/-Users-qinghan-chen-Desktop-sglang-jax-workdir-glm-dsa-tpu/a2719a6b-8303-4d9e-9a9f-33f0b4ad3e47/scratchpad
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }
( P=tpuq-qinghan-4d14-w-0-0-m27gm; T=glm-port-b; V=venv-port-b
  k $P bash $W/stop_wait.sh < /dev/null
  for f in kl_dsa_radix_131k.py kl_dsa_radix.py dsa_nospec_radix.py; do k $P tee $W/$T/benchmark/tpu/recipe/configs/glm_5_3_flash_eval/$f < $S/eval_configs/$f > /dev/null; done
  k $P tee $W/kl_run.py < $S/kl_run.py > /dev/null
  k $P env HOME=$H setsid -f bash $W/tpu-boot-only.sh kl_dsa_radix_131k kl-dsa131 $T $V < /dev/null > /dev/null 2>&1
  until k $P grep -q BOOT_DONE $W/glm-run/boot-kl-dsa131.out < /dev/null 2>/dev/null; do sleep 30; done
  echo "$(date +%T) kl-dsa131: $(k $P grep -av PrivateUse1 $W/glm-run/boot-kl-dsa131.out < /dev/null | tr '\n' ' ' | cut -c1-200)"
  k $P grep -q BOOT_OK $W/glm-run/boot-kl-dsa131.out < /dev/null && k $P env HOME=$H PYTHONPATH=$W/$T/python $H/$V/bin/python $W/kl_run.py 30000 24 512 < /dev/null 2>&1 | grep -E "RESULT|FAILED|Error" | sed "s/^/kl-dsa131: /" ) &
( P=tpuq-qinghan-de1c-w-0-0-6fn6n; T=glm-port; V=venv-port
  k $P bash $W/stop_wait.sh < /dev/null
  k $P tee $W/$T/sglang_tpu/attention/pallas/mla_sparse/kernel.py < $D/pad0/kernel.py > /dev/null
  k $P env HOME=$H setsid -f bash $W/tpu-boot-only.sh pool_dsa_262144 pool-pad0 $T $V < /dev/null > /dev/null 2>&1
  until k $P grep -q BOOT_DONE $W/glm-run/boot-pool-pad0.out < /dev/null 2>/dev/null; do sleep 30; done
  echo "$(date +%T) 262k + padding at page 0: $(k $P grep -av PrivateUse1 $W/glm-run/boot-pool-pad0.out < /dev/null | tr '\n' ' ' | cut -c1-200)"
  k $P bash -c "cd $W/$T && git -c safe.directory='*' status >/dev/null 2>&1; true" < /dev/null
  k $P tee $W/$T/sglang_tpu/attention/pallas/mla_sparse/kernel.py < $S/../../glm-dsa-tpu-rebase/sglang_tpu/attention/pallas/mla_sparse/kernel.py > /dev/null ) &
wait; echo "$(date +%T) KL3_DONE"
