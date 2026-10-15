#!/bin/bash
# Usage: nohup bash run-dsa-final.sh > dsa-final.log 2>&1 &
# Final DSA try (DFlash + DSA with the SparseCore fixes: indexer top-k on XLA, MoE gathers on
# XLA, no Triton on TPU), one job per slice:
#   slice 1 (4d14): 8k1k c1 at simulated accept 3  -> TTFT/TPOT + round profile
#   3131          : 8k1k c1 at real acceptance      -> real-text and random TPOT
#   4836          : AIME26 avg@4, 320k budget, 330k context, 8 concurrent
# Tier-2 cache is off in the configs; checkpoint files are dropped from the page cache every
# 2 min while booting. Results land in each slice's $W/glm-run/ and in this log.
S=$(cd "$(dirname "$0")" && pwd); W=/dev/shm/home/qinghanc/work; C=gke_sglang-torchtpu_us-central1-c_tpu-cluster
P1=tpuq-qinghan-4d14-w-0-0-m27gm; P2=tpuq-qinghan-3131-w-0-0-7bnjs; P3=tpuq-qinghan-4836-w-0-0-d29h5
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }

for spec in $P1:glm-port-b $P2:glm-port $P3:glm-port; do
  P=${spec%%:*}; T=${spec#*:}
  ( cd $S/eval_configs && COPYFILE_DISABLE=1 tar cf - --no-xattrs *.py ) \
    | k $P tar xf - -C $W/$T/benchmark/tpu/recipe/configs/glm_5_3_flash_eval
  for f in tpu-mtp-run.sh tpu-real-run.sh tpu-aime.sh bench-cb.sh bench_real.py prof_round.py aime26.py drop_weight_cache.py; do
    k $P tee $W/$f < $S/$f > /dev/null
  done
done

k $P1 env HOME=/dev/shm/home/qinghanc setsid -f bash $W/tpu-mtp-run.sh c1_dsa_dflash_acc3 glm-port-b venv-port-b final-acc3 < /dev/null > /dev/null 2>&1
k $P2 env HOME=/dev/shm/home/qinghanc setsid -f bash $W/tpu-real-run.sh real_dsa_dflash glm-port venv-port final-real < /dev/null > /dev/null 2>&1
k $P3 env HOME=/dev/shm/home/qinghanc setsid -f bash $W/tpu-aime.sh aime_dflash_dsa_330k glm-port venv-port aime-dflash-dsa 8 0,1,2,3 < /dev/null > /dev/null 2>&1
echo "$(date +%T) launched: $P1 final-acc3, $P2 final-real, $P3 aime-dflash-dsa"

# Drop checkpoint page cache every 2 min for the first hour (boot + compile).
for i in $(seq 30); do
  for P in $P1 $P2 $P3; do k $P python3 $W/drop_weight_cache.py /weights /work/qinghan/weights > /dev/null 2>&1; done
  sleep 120
done
echo "$(date +%T) page-cache loop done"
