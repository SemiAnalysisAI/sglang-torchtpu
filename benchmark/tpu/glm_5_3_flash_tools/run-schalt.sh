#!/bin/bash
# Usage: nohup bash run-schalt.sh < /dev/null > schalt.log 2>&1 &
# SparseCore-halt fix attempts, all DFlash + DSA 8k1k c1 at simulated accept 3.5, each one change
# from the halting 8192 control (MoE gathers on SparseCore, indexer top-k on XLA):
#   slice 1 (4d14): latency-hiding scheduler off, top-k on XLA          (sc_lhs_off_p8192)
#   3131          : ALL on SparseCore, sparse-attention gather chunk 512 (c1_dsa_dflash_acc35_scmoe_p8192)
#   4836          : ALL on SparseCore, sparse-attention gather chunk 256 (c1_dsa_dflash_acc35_scmoe_p8192)
#   de1c          : ALL on SparseCore + scheduler off                    (sc_allsc_lhs_off_p8192)
# Each run: boot -> 8k1k c1 bench -> round profile; verdicts in $W/glm-run/run-<tag>.out.
S=$(cd "$(dirname "$0")" && pwd); R=$S/../../glm-dsa-tpu-rebase
W=/dev/shm/home/qinghanc/work; C=gke_sglang-torchtpu_us-central1-c_tpu-cluster
P1=tpuq-qinghan-4d14-w-0-0-m27gm; P2=tpuq-qinghan-3131-w-0-0-7bnjs; P3=tpuq-qinghan-4836-w-0-0-d29h5; P4=tpuq-qinghan-de1c-w-0-0-6fn6n
k() { kubectl --context $C -n tpuq exec -i "$1" -c main -- "${@:2}"; }
launch() { k "$1" env HOME=/dev/shm/home/qinghanc setsid -f bash "${@:2}" < /dev/null > /dev/null 2>&1; }
sync_tree() {
  local P=$1 T=$2
  for f in sglang_tpu/tpu_platform.py sglang_tpu/attention/pallas/dsa_kpool_topk.py sglang_tpu/attention_pallas_mla_sparse.py \
           sglang_tpu/dsv4/streamindex_topk/streamindex_topk.py sglang_tpu/dsa_indexer_tpu.py \
           python/sglang/srt/mem_cache/allocation.py benchmark/tpu/recipe/configs/glm_5_3_flash/server.py \
           benchmark/tpu/recipe/configs/glm_5_3_flash/mtp.py benchmark/tpu/recipe/configs/glm_5_3_flash/dflash.py \
           python/sglang/srt/configs/model_config.py python/sglang/srt/arg_groups/speculative_hook.py; do
    k $P tee $W/$T/$f < $R/$f > /dev/null
  done
  ( cd $S/eval_configs && COPYFILE_DISABLE=1 tar cf - --no-xattrs *.py ) \
    | k $P tar xf - -C $W/$T/benchmark/tpu/recipe/configs/glm_5_3_flash_eval
  for f in tpu-mtp-run.sh tpu-boot-only.sh bench-cb.sh prof_round.py prof_prefill.py; do
    k $P tee $W/$f < $S/$f > /dev/null
  done
}

sync_tree $P1 glm-port-b; sync_tree $P2 glm-port; sync_tree $P3 glm-port; sync_tree $P4 glm-port
D=/private/tmp/claude-502/-Users-qinghan-chen-Desktop-sglang-jax-workdir-glm-dsa-tpu/f68e81d4-98d8-430f-82ae-8f66edad9cca/scratchpad/dbg
# 3131, 4836, de1c: indexer top-k back on the SparseCore (the kpool step before topk_impl="xla").
for P in $P2 $P3 $P4; do
  ( cd $R && git show 08bf7feae0:sglang_tpu/attention/pallas/dsa_kpool_topk.py ) \
    | k $P tee $W/glm-port/sglang_tpu/attention/pallas/dsa_kpool_topk.py > /dev/null
done
# 3131 / 4836: sparse-attention gather in 512 / 256-token chunks (one SparseCore program each).
k $P2 tee $W/glm-port/sglang_tpu/attention/pallas/mla_sparse_fused.py < $D/gchunk512/mla_sparse_fused.py > /dev/null
k $P3 tee $W/glm-port/sglang_tpu/attention/pallas/mla_sparse_fused.py < $D/gchunk256/mla_sparse_fused.py > /dev/null
for P in $P2 $P3 $P4; do
  echo "$(date +%T) $P xla-topk=$(k $P grep -c 'topk_impl="xla"' $W/glm-port/sglang_tpu/attention/pallas/dsa_kpool_topk.py < /dev/null) gchunk=$(k $P grep -o 'min([0-9]*, max(64' $W/glm-port/sglang_tpu/attention/pallas/mla_sparse_fused.py < /dev/null)"
done

launch $P1 $W/tpu-mtp-run.sh sc_lhs_off_p8192 glm-port-b venv-port-b s-lhsoff
launch $P2 $W/tpu-mtp-run.sh c1_dsa_dflash_acc35_scmoe_p8192 glm-port venv-port s-allsc-g512
launch $P3 $W/tpu-mtp-run.sh c1_dsa_dflash_acc35_scmoe_p8192 glm-port venv-port s-allsc-g256
launch $P4 $W/tpu-mtp-run.sh sc_allsc_lhs_off_p8192 glm-port venv-port s-allsc-lhsoff
echo "$(date +%T) launched: slice1 s-lhsoff, 3131 s-allsc-g512, 4836 s-allsc-g256, de1c s-allsc-lhsoff"
