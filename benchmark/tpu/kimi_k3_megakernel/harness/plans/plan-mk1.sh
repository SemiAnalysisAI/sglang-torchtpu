# Inferact Kimi-K3 megakernel, standalone target-only greedy decode on one 2x2x4 slice (snapshot
# megakernels @ 4048f08). Configs NAME:BATCH:CONTEXT:ISL:OSL:ROUNDS. b1c1k repeats the published
# setup (1-token prompts, context 1152); b1c8k / b4c1k match our SGLang 8k/1k c1 and 1k/1k c4.
MK_SRC=/mnt/workspace/dagent/snap/megakernels
CONFIGS="b1c1k:1:1152:1:1024:3 b1c8k:1:9344:8192:1024:2 b4c1k:4:2176:1024:1024:2"
CONFIG_TIMEOUT_S=3000
