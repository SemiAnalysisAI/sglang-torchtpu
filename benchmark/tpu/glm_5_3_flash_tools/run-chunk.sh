#!/bin/bash
# Usage: nohup bash run-chunk.sh < /dev/null > chunk.log 2>&1 &
# DFlash + DSA 8k1k c1 (simulated accept 3.5), MoE gathers back on the SparseCore, indexer top-k on XLA:
#   slice 1 (4d14): prefill chunked at 1024 tokens
#   3131          : 8192 (control, same settings)
#   4836          : 512
#   de1c          : bootstrap first, then 2048
# Each run: boot -> 8k1k c1 bench -> round profile ($W/glm-run/run-<tag>.out on that slice).
S=$(cd "$(dirname "$0")" && pwd); R=$S/../../glm-dsa-tpu-rebase
W=/dev/shm/home/qinghanc/work; C=gke_sglang-torchtpu_us-central1-c_tpu-cluster
P1=tpuq-qinghan-4d14-w-0-0-m27gm; P4=tpuq-qinghan-de1c-w-0-0-6fn6n
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }
launch() { k "$1" env HOME=/dev/shm/home/qinghanc setsid -f bash "${@:2}" < /dev/null > /dev/null 2>&1; }
sync_tree() {
  local P=$1 T=$2
  for f in sglang_tpu/tpu_platform.py sglang_tpu/attention/pallas/dsa_kpool_topk.py \
           sglang_tpu/dsv4/streamindex_topk/streamindex_topk.py sglang_tpu/dsa_indexer_tpu.py \
           python/sglang/srt/mem_cache/allocation.py benchmark/tpu/recipe/configs/glm_5_3_flash/server.py \
           benchmark/tpu/recipe/configs/glm_5_3_flash/mtp.py benchmark/tpu/recipe/configs/glm_5_3_flash/dflash.py \
           python/sglang/srt/configs/model_config.py python/sglang/srt/arg_groups/speculative_hook.py; do
    k $P tee $W/$T/$f < $R/$f > /dev/null
  done
  ( cd $S/eval_configs && COPYFILE_DISABLE=1 tar cf - --no-xattrs *.py ) \
    | k $P tar xf - -C $W/$T/benchmark/tpu/recipe/configs/glm_5_3_flash_eval
  for f in tpu-mtp-run.sh tpu-boot-only.sh bench-cb.sh prof_round.py prof_prefill.py drop_weight_cache.py; do
    k $P tee $W/$f < $S/$f > /dev/null
  done
}

P2=tpuq-qinghan-3131-w-0-0-7bnjs; P3=tpuq-qinghan-4836-w-0-0-d29h5
# All four in parallel.
sync_tree $P1 glm-port-b; sync_tree $P2 glm-port; sync_tree $P3 glm-port
launch $P1 $W/tpu-mtp-run.sh c1_dsa_dflash_acc35_scmoe_p1024 glm-port-b venv-port-b c-p1024
launch $P2 $W/tpu-mtp-run.sh c1_dsa_dflash_acc35_scmoe_p8192 glm-port venv-port c-p8192
launch $P3 $W/tpu-mtp-run.sh c1_dsa_dflash_acc35_scmoe_p512 glm-port venv-port c-p512
echo "$(date +%T) launched: slice1 c-p1024, 3131 c-p8192, 4836 c-p512"

( bash $S/bootstrap-port-pod.sh $P4 < /dev/null > $S/chunk-de1c-bootstrap.log 2>&1
  sync_tree $P4 glm-port
  launch $P4 $W/tpu-mtp-run.sh c1_dsa_dflash_acc35_scmoe_p2048 glm-port venv-port c-p2048
  echo "$(date +%T) de1c: bootstrapped, launched c-p2048" ) &

for i in $(seq 40); do
  for P in $P1 $P2 $P3 $P4; do k $P python3 $W/drop_weight_cache.py /weights /work/qinghan/weights < /dev/null > /dev/null 2>&1; done
  sleep 120
done
wait
echo "$(date +%T) chunk round done"
