#!/usr/bin/env bash
# Per-host driver: run mk_bridge_spike.py (megakernel inside torch_tpu workers via SGLang's
# jax-op bridge) for each config in $PLAN under torchrun, 4 hosts x 8 processes.
# Env: PLAN, OUT, NHOSTS, JOBSET_JOB_INDEX, JS_NAME, POD_NAMESPACE.
set -u
IDX=${JOBSET_JOB_INDEX}
# shellcheck disable=SC1090
source "$PLAN"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.host${IDX}.log") 2>&1
echo "[spike] host ${IDX} plan=${PLAN} configs='${CONFIGS}' $(date -u)"
cp "$PLAN" "$OUT/plan.sh" 2>/dev/null || true
pip install --no-deps -q /mnt/workspace/dagent_wheels/torch_tpu-0.1.1+release.2026.9.22.20260927223741-cp312-cp312-manylinux_2_31_x86_64.whl "tpu-sync-torch==0.0.1.dev20261008082921" "libtpu==0.0.47" "jax==0.10.2" "jaxlib==0.10.2" \
  || pip install --no-deps -q /mnt/workspace/dagent_wheels/torch_tpu-0.1.1+release.2026.9.22.20260927223741-cp312-cp312-manylinux_2_31_x86_64.whl "tpu-sync-torch==0.0.1.dev20261002213151" "libtpu==0.0.47" "jax==0.10.2" "jaxlib==0.10.2"
pip install -q "tiktoken>=0.14.0" > /dev/null 2>&1
python3 -c 'import jax, torch, torch_tpu; print("[spike] jax", jax.__version__, "torch", torch.__version__)'
export PYTHONPATH="${SG_SRC}/python:${SG_SRC}:${MK_SRC}"
export MK_SRC MK_CKPT=/mnt/kimik3-ssd/kimik3
eval "$(python3 -m torch_tpu._internal.distributed.launchers.multihost_wrapper)"
echo "[spike] NODE_RANK=$NODE_RANK NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT TOPO=$TORCH_TPU_TOPOLOGY"
k=0
for CFG in $CONFIGS; do
  k=$((k + 1))
  IFS=: read -r NAME B CTX POS <<< "$CFG"
  echo "[spike] === $NAME batch=$B context=$CTX position=$POS $(date -u +%T)"
  MK_BATCH=$B MK_CTX=$CTX MK_POS=$POS MK_STEPS=${MK_STEPS:-30} timeout "${CONFIG_TIMEOUT_S:-5400}" \
    torchrun --nnodes="$NNODES" --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" \
      --master_port=$((MASTER_PORT + k)) --nproc_per_node=8 /mnt/workspace/dagent/mkspike/mk_bridge_spike.py \
      > "$OUT/${NAME}.host${IDX}.log" 2>&1
  echo "[spike] $NAME rc=$? $(grep -h '\[spike\]' "$OUT/${NAME}.host${IDX}.log" | tail -4 | tr '\n' '|')"
done
echo "[spike] done $(date -u)"
