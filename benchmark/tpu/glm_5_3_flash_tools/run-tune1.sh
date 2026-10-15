#!/bin/bash
# Usage: nohup bash run-tune1.sh < /dev/null > tune1.log 2>&1 &
# TTFT tuning round 1 on a63aea39f4 (DSA, all three ops on the SparseCore, 256-token gather chunks).
# All DFlash 8k1k c1 at simulated accept 3.5:
#   slice 1 (4d14): DENSE attention, same tree          (dense_dflash_acc35)    -> the floor
#   3131          : DSA at HEAD                         (c1_dsa_dflash_acc35)   -> then a 4096-token full trace
#   4836          : DSA, sparse-attention batch 32      (c1_dsa_dflash_acc35 + bz32 wrapper)
#   de1c          : DSA, indexer without row chunking   (c1_dsa_dflash_acc35 + nochunk indexer)
# Launches are 90 s apart so the four weight loads don't all hit shared storage at once.
S=$(cd "$(dirname "$0")" && pwd); R=$S/../../glm-dsa-tpu-rebase
D=/private/tmp/claude-502/-Users-qinghan-chen-Desktop-sglang-jax-workdir-glm-dsa-tpu/f68e81d4-98d8-430f-82ae-8f66edad9cca/scratchpad/dbg
W=/dev/shm/home/qinghanc/work; C=gke_sglang-torchtpu_us-central1-c_tpu-cluster
P1=tpuq-qinghan-4d14-w-0-0-m27gm; P2=tpuq-qinghan-3131-w-0-0-7bnjs; P3=tpuq-qinghan-4836-w-0-0-d29h5; P4=tpuq-qinghan-de1c-w-0-0-6fn6n
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }
launch() { k "$1" env HOME=/dev/shm/home/qinghanc setsid -f bash "${@:2}" < /dev/null > /dev/null 2>&1; }
sync_tree() {
  local P=$1 T=$2
  for f in sglang_tpu/tpu_platform.py sglang_tpu/attention/pallas/dsa_kpool_topk.py sglang_tpu/attention_pallas_mla_sparse.py \
           sglang_tpu/attention/pallas/mla_sparse_fused.py \
           sglang_tpu/dsv4/streamindex_topk/streamindex_topk.py sglang_tpu/dsa_indexer_tpu.py \
           python/sglang/srt/mem_cache/allocation.py benchmark/tpu/recipe/configs/glm_5_3_flash/server.py \
           benchmark/tpu/recipe/configs/glm_5_3_flash/mtp.py benchmark/tpu/recipe/configs/glm_5_3_flash/dflash.py \
           python/sglang/srt/configs/model_config.py python/sglang/srt/arg_groups/speculative_hook.py; do
    k $P tee $W/$T/$f < $R/$f > /dev/null
  done
  ( cd $S/eval_configs && COPYFILE_DISABLE=1 tar cf - --no-xattrs *.py ) \
    | k $P tar xf - -C $W/$T/benchmark/tpu/recipe/configs/glm_5_3_flash_eval
  for f in tpu-mtp-run.sh tpu-boot-only.sh bench-cb.sh prof_round.py prof_prefill.py prof_prefill_n.py; do
    k $P tee $W/$f < $S/$f > /dev/null
  done
  for f in xplane_kinds.py xplane_grep.py; do k $P tee $W/$f < $D/$f > /dev/null; done
}

sync_tree $P1 glm-port-b; sync_tree $P2 glm-port; sync_tree $P3 glm-port; sync_tree $P4 glm-port
k $P3 tee $W/glm-port/sglang_tpu/attention/pallas/mla_sparse_fused.py < $D/bz32/mla_sparse_fused.py > /dev/null
k $P4 tee $W/glm-port/sglang_tpu/attention/pallas/dsa_kpool_topk.py < $D/nochunk/dsa_kpool_topk.py > /dev/null
echo "$(date +%T) synced; 4836 bz32=$(k $P3 grep -c 'attention_kernel_batch_size=32' $W/glm-port/sglang_tpu/attention/pallas/mla_sparse_fused.py < /dev/null) de1c nochunk=$(k $P4 grep -c '1 << 30' $W/glm-port/sglang_tpu/attention/pallas/dsa_kpool_topk.py < /dev/null)"

launch $P2 $W/tpu-mtp-run.sh c1_dsa_dflash_acc35 glm-port venv-port t-head; echo "$(date +%T) 3131 t-head"; sleep 90
launch $P1 $W/tpu-mtp-run.sh dense_dflash_acc35 glm-port-b venv-port-b t-dense; echo "$(date +%T) slice1 t-dense"; sleep 90
launch $P3 $W/tpu-mtp-run.sh c1_dsa_dflash_acc35 glm-port venv-port t-bz32; echo "$(date +%T) 4836 t-bz32"; sleep 90
launch $P4 $W/tpu-mtp-run.sh c1_dsa_dflash_acc35 glm-port venv-port t-nochunk; echo "$(date +%T) de1c t-nochunk"
echo "$(date +%T) launched all"
