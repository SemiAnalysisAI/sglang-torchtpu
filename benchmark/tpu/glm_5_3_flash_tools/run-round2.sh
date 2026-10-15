#!/bin/bash
# Usage: nohup bash run-round2.sh > round2.log 2>&1 &
# Four slices in parallel (all with the SparseCore fixes and the MTP draft-pool fix):
#   4d14 (slice 1): DFlash + DSA, 8k1k c1, simulated accept 3.5  -> TTFT/TPOT + profile
#   3131          : MTP 5/1/6 + DSA, 8k1k c1, simulated accept 3.5 -> TTFT/TPOT + profile
#   4836          : prefix cache (radix) basic test, DSA no-spec
#   de1c          : bootstrap, then KV offload (hierarchical cache) basic test, DSA no-spec
# Each slice's results land in $W/glm-run/; this log records launches and the two basic tests.
S=$(cd "$(dirname "$0")" && pwd); R=$S/../../glm-dsa-tpu-rebase
W=/dev/shm/home/qinghanc/work; C=gke_sglang-torchtpu_us-central1-c_tpu-cluster
P1=tpuq-qinghan-4d14-w-0-0-m27gm; P2=tpuq-qinghan-3131-w-0-0-7bnjs; P3=tpuq-qinghan-4836-w-0-0-d29h5; P4=tpuq-qinghan-de1c-w-0-0-6fn6n
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }
launch() { k "$1" env HOME=/dev/shm/home/qinghanc setsid -f bash "${@:2}" < /dev/null > /dev/null 2>&1; }
sync_tree() {  # $1 pod, $2 tree: HEAD sources that changed today, eval configs, scripts
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
  for f in tpu-mtp-run.sh tpu-boot-only.sh bench-cb.sh prof_round.py prefix_test.py drop_weight_cache.py; do
    k $P tee $W/$f < $S/$f > /dev/null
  done
}

sync_tree $P1 glm-port-b; sync_tree $P2 glm-port; sync_tree $P3 glm-port
launch $P1 $W/tpu-mtp-run.sh c1_dsa_dflash_acc35 glm-port-b venv-port-b r2-dflash-acc35
launch $P2 $W/tpu-mtp-run.sh prof_dsa_mtp516_acc35 glm-port venv-port r2-mtp-acc35
launch $P3 $W/tpu-boot-only.sh dsa_nospec_radix r2-prefix glm-port venv-port
echo "$(date +%T) launched: $P1 r2-dflash-acc35, $P2 r2-mtp-acc35, $P3 r2-prefix"

# de1c: fresh slice, full bootstrap at HEAD, then the KV offload boot.
( bash $S/bootstrap-port-pod.sh $P4 > $S/round2-de1c-bootstrap.log 2>&1
  sync_tree $P4 glm-port
  launch $P4 $W/tpu-boot-only.sh dsa_nospec_hicache r2-hicache glm-port venv-port
  echo "$(date +%T) de1c bootstrapped, launched r2-hicache" ) &

# Basic tests: wait for each boot, then report.
( until k $P3 grep -q BOOT_DONE $W/glm-run/boot-r2-prefix.out 2>/dev/null; do sleep 30; done
  echo "$(date +%T) r2-prefix boot: $(k $P3 grep -av PrivateUse1 $W/glm-run/boot-r2-prefix.out | tr '\n' ' ' | cut -c1-300)"
  k $P3 grep -q BOOT_OK $W/glm-run/boot-r2-prefix.out && k $P3 /dev/shm/home/qinghanc/venv-port/bin/python $W/prefix_test.py ) &
( sleep 600; until k $P4 grep -q BOOT_DONE $W/glm-run/boot-r2-hicache.out 2>/dev/null; do sleep 30; done
  echo "$(date +%T) r2-hicache boot: $(k $P4 grep -av PrivateUse1 $W/glm-run/boot-r2-hicache.out | tr '\n' ' ' | cut -c1-400)"
  k $P4 grep -a -m3 -iE "hierarch|hicache|not support|ValueError" $W/glm-run/r2-hicache.log | cut -c1-300
  k $P4 grep -q BOOT_OK $W/glm-run/boot-r2-hicache.out && k $P4 /dev/shm/home/qinghanc/venv-port/bin/python $W/prefix_test.py ) &

# Drop checkpoint page cache every 2 min for the first hour on all four slices.
for i in $(seq 30); do
  for P in $P1 $P2 $P3 $P4; do k $P python3 $W/drop_weight_cache.py /weights /work/qinghan/weights > /dev/null 2>&1; done
  sleep 120
done
wait
echo "$(date +%T) round 2 done"
