# Bridge spike (2026-10-12): Inferact Kimi-K3 megakernel decode step inside torch_tpu workers via
# SGLang's cpu_export_jax_op bridge, synthetic weights, SGLang's stack (jax 0.10.2, libtpu 0.0.47).
# CONFIGS NAME:BATCH:CONTEXT:POSITION.
MK_SRC=/mnt/workspace/dagent/snap/megakernels
SG_SRC=/mnt/workspace/dagent/snap/p1-heads
CONFIGS="b1c1k:1:1152:128 b1c8k:1:9344:8192 b4c2k:4:2176:1024"
CONFIG_TIMEOUT_S=2400
