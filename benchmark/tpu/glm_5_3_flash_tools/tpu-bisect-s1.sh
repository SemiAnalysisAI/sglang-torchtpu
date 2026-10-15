#!/bin/bash
# Slice 1 queue: D2 (XLA attention), 4096-token prefill (HEAD kernels), D1 (XLA cache writer).
W=/dev/shm/home/qinghanc/work; F=$W/glm-port-b/sglang_tpu/attention/pallas/mla_sparse_fused.py
export HOME=/dev/shm/home/qinghanc
cp $W/bisect/d2_mla_sparse_fused.py $F; bash $W/tpu-boot-only.sh prof_dsa_mtp516_acc35 bisect-D2-xlaattn glm-port-b venv-port-b
cp $W/bisect/head_mla_sparse_fused.py $F
bash $W/tpu-boot-only.sh prof_dsa_mtp516_p4096 bisect-P4096 glm-port-b venv-port-b
cp $W/bisect/d1_mla_sparse_fused.py $F
bash $W/tpu-boot-only.sh prof_dsa_mtp516_acc35 bisect-D1-xlawriter glm-port-b venv-port-b
cp $W/bisect/head_mla_sparse_fused.py $F
