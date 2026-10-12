# Combined run, SGLang phase (2026-10-12): current main (with #1149/#1150/#1151 merged) + the 5 open Kimi-K3 PRs
# (snapshot p1-heads). Real Kimi-K3 2.7T, TP32/EP32, one 2x2x4 slice. Per variant: validate, one
# TPOT sweep pass, then xprof captures of decode at bs1 (8k prompt) and bs4 (1k prompts).
# allon: #1090 + #1093 + #1083 flags (best config); offa: flags off.
VARIANTS="allon"
GPQA_VARIANTS=""
CLIENT_CODE=/mnt/workspace/dagent/snap/p1-heads
SWEEP="1:8192:1024:8k1k 4:1024:1024:1k1k"
E2E_PHASES1=validate
E2E_PHASES2=logscan
PROFILE_SPECS="1:8192 4:1024"
PROFILE_STEPS=20
export CONTEXT_LEN=9344 MAX_RUNNING=64 TORCH_COMPILE_MAX_BS=64 TPU_COMPILE_BS="1 4 16 64" TPU_COMPILE_TOKENS=1024 CHUNKED_PREFILL_SIZE=1024 MAX_PREFILL_TOKENS=1024 MEM_FRACTION_STATIC=0.75
unset MAX_TOTAL_TOKENS
STD_ENV="SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD=512 SGLANG_TPU_EAGER_MODE=DEFER_AND_FUSE SGLANG_TPU_CHECK_RECOMPILATION=1"
declare -A CODE STACK ENVS ARGS
for v in $VARIANTS; do
  CODE[$v]=/mnt/workspace/dagent/snap/p1-heads
  STACK[$v]=main
done
ENVS[offa]="$STD_ENV"
ENVS[allon]="$STD_ENV SGLANG_TPU_ENABLE_K3_FUSED_ONORM=1 SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL=1 SGLANG_TPU_ENABLE_UNSAFE_GUARD_SKIP=1"
