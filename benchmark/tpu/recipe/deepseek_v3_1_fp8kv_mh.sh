#!/usr/bin/env bash
# DeepSeek-V3.1 thinking mode on a two-host TP16/EP16 slice.
set -euo pipefail
export MODEL=${MODEL:-/weights/DeepSeek-V3.1}
export REASONING_PARSER=deepseek-v3
# Fail before allocating/loading shards if the shared CI disk is not hydrated.
if [[ "$MODEL" == /weights/* && ! -f "$MODEL/config.json" ]]; then
  echo "Missing DeepSeek-V3.1 checkpoint: $MODEL (hydrate the CI weights disk first)" >&2
  exit 1
fi
exec bash "$(dirname "$0")/deepseek_r1_0528_fp8kv_mh.sh" "$@"
