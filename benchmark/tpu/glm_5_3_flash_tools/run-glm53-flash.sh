#!/bin/bash
# GLM-5.3-Flash 8k1k, single-host TP8/EP8.
# KDA=pallas (default) | scan  -- KDA kernel: chunked Pallas GDN v3 KDA mode or lax.scan reference.
# OVERLAP=1|0, CHUNK=8192 by default.
set -euo pipefail
export HOME=/dev/shm/home/qinghanc
. $HOME/venv/bin/activate
cd $HOME/work/glm-dsa-tpu/benchmark/tpu
export FORCE_SINGLE_HOST=1 TPU_TOPOLOGY=2x2x1 TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 \
       TPU_WORKER_HOSTNAMES=localhost TPU_WORKER_ID=0 TPU_PROCESS_ADDRESSES=localhost:8471 TPU_ACCELERATOR_TYPE=tpu7x-4
export TPU_LOG_DIR=$HOME/tpu_logs HF_HOME=$HOME/.cache/hf XDG_CACHE_HOME=$HOME/.cache TMPDIR=$HOME/tmp
mkdir -p $TMPDIR $TPU_LOG_DIR
KDA=${KDA:-pallas}
export ENABLE_COMPILE_CACHE=1 TORCH_TPU_TIER2_NAME=qinghanc_glm_flash_${KDA}_moe${MOE_FLAGS:-0}_t${MOE_THRESHOLD:-d}_f${FP8_SMALL_M:-0}_ep${EP:-8}_sc${SC_OFFLOAD:-0}_x${EXP_TAG:-0}_l${LAYERS:-all}
export PYTHON=$HOME/venv/bin/python HOST=127.0.0.1 PORT=30000
export SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT=${CONTEXT_LEN:-10240} SGLANG_MAMBA_SSM_DTYPE=float32
[[ $KDA == scan ]] && export SGLANG_TPU_KDA_USE_REFERENCE_SCAN=1
# MOE_FLAGS=1: the GLM-5.3 MoE speed flags (misc/profiles/glm/OPTIMIZATION.md).
if [[ ${MOE_FLAGS:-0} == 1 ]]; then
  export SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD=3 SGLANG_TPU_FP8_MOE_SKIP_PAD_EXPERT=1 \
    SGLANG_TPU_FP8_MOE_BS1_ROUTING=1 SGLANG_TPU_GLM_ROUTING_GATHER=1 \
    SGLANG_TPU_GLM_GMM_TWO_RHS_BUFFERS=1 SGLANG_TPU_GLM_FUSED_EXPERTS=1
fi
# MOE_THRESHOLD=512 enables the single-token route-meta MoE path (needs > m_pad * topk = 16).
[[ -n ${MOE_THRESHOLD:-} ]] && export SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD=$MOE_THRESHOLD
# FP8_SMALL_M=1: single-kernel W8A8 dense linears for decode. EP=1: tensor-parallel MoE.
[[ ${FP8_SMALL_M:-0} == 1 ]] && export SGLANG_TPU_FP8_LINEAR_SMALL_M_KERNEL=1
# SC_OFFLOAD=allreduce|all: flip common_env.sh's SparseCore collective-offload flags (playbook 3.7).
if [[ -n ${SC_OFFLOAD:-} ]]; then
  base=$(bash -c '. '"$HOME"'/work/glm-dsa-tpu/benchmark/tpu/common_env.sh >/dev/null 2>&1; echo "$LIBTPU_INIT_ARGS"')
  base=${base/offload_all_reduce=false/offload_all_reduce=true}
  if [[ $SC_OFFLOAD == all ]]; then
    base=${base/offload_all_gather=false/offload_all_gather=true}
    base=${base/offload_reduce_scatter=false/offload_reduce_scatter=true}
  fi
  export LIBTPU_INIT_ARGS=$base
fi
# MAX_RUNNING / TPU_COMPILE_BS ("1 2 4 8 16") / MAX_TOTAL_TOKENS / MAMBA_CACHE: concurrency > 1.
# LAYERS=N: truncate the model (host-overhead floor experiments only).
[[ -n ${LAYERS:-} ]] && EXTRA_ARGS="${EXTRA_ARGS:-} --json-model-override-args {\"num_hidden_layers\":$LAYERS}"
CHUNK=${CHUNK:-8192}
exec env MODEL=${MODEL:-/weights/GLM-5.3-Flash} TP_SIZE=8 EP_SIZE=${EP:-8} \
  CONTEXT_LEN=${CONTEXT_LEN:-10240} CHUNKED_PREFILL_SIZE=$CHUNK MAX_PREFILL_TOKENS=$CHUNK MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-20480} \
  PAGE_SIZE=128 MAX_RUNNING=${MAX_RUNNING:-1} TORCH_COMPILE_MAX_BS=${MAX_RUNNING:-1} TPU_COMPILE_TOKENS=$CHUNK TPU_COMPILE_BS="${TPU_COMPILE_BS:-1}" \
  MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.90} KV_CACHE_DTYPE=bfloat16 DISABLE_RADIX_CACHE=1 \
  DISABLE_OVERLAP_SCHEDULE=$(( 1 - ${OVERLAP:-1} )) ENABLE_PROFILER_ENDPOINT=0 \
  EXTRA_ARGS="--language-model-only --watchdog-timeout 1800 --max-mamba-cache-size ${MAMBA_CACHE:-4} --disable-shared-experts-fusion ${EXTRA_ARGS:-}" \
  bash launch_server.sh
