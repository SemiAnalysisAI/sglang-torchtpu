#!/bin/bash
# Quick Qwen3-1.7B EAGLE3 benchmark with torch.compile on TPU.
# Uses minimal compile buckets for fast startup.
#
# Usage:
#   ./bench_qwen3_compile.sh
#   SKIP_BASELINE=1 ./bench_qwen3_compile.sh   # EAGLE3 only
#   NUM_QUESTIONS=5 ./bench_qwen3_compile.sh    # fewer questions
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
export DRAFT_MODEL="${DRAFT_MODEL:-AngelSlim/Qwen3-1.7B_eagle3}"
export COMPILE=1
export COMPILE_BS="${COMPILE_BS:-1}"
export COMPILE_TOKENS="${COMPILE_TOKENS:-256 1024}"
export CONTEXT_LEN="${CONTEXT_LEN:-4096}"
# Keep output short so both MT-Bench turns fit within CONTEXT_LEN.
# System prompt (~100 tokens) + turn 1 answer + turn 2 question (~50 tokens)
# + turn 2 answer must all fit in CONTEXT_LEN.
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export NUM_QUESTIONS="${NUM_QUESTIONS:-5}"
export WARMUP_ROUNDS="${WARMUP_ROUNDS:-1}"

export SKIP_BASELINE="${SKIP_BASELINE:-1}"

exec bash "$SCRIPT_DIR/bench_eagle_e2e.sh"
