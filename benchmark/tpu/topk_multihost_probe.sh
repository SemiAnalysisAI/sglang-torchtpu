#!/usr/bin/env bash
# Isolate distributed TopK device operations on one or more TPU v7 hosts.
set -euo pipefail

export JAX_PLATFORMS=cpu
eval "$(python3 -m torch_tpu._internal.distributed.launchers.multihost_wrapper)"
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES
export TORCH_TPU_XPROF_SESSION_ID NNODES NODE_RANK MASTER_ADDR MASTER_PORT

echo "topk probe: nnodes=${NNODES} node_rank=${NODE_RANK} topology=${TORCH_TPU_TOPOLOGY} args=$*"
exec python3 -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node=8 \
  test/manual/tpu/topk_multihost_probe.py "$@"
