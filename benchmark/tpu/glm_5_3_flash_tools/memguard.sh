#!/bin/bash
# Stop sglang before the pod's working set nears its memory limit (/dev/shm counts too);
# a breach evicts the pod. Working set = memory.current - inactive_file, which is what
# eviction looks at; easily reclaimed inactive page cache is excluded. Above SOFT, first
# drop the checkpoint files from the page cache (they are in HBM after load) and re-check.
LIMIT_GIB=${1:-820}; SOFT_GIB=${2:-760}
# Detach from the launching kubectl exec, or it waits on our stdout forever.
exec < /dev/null > /dev/null 2>&1
W=/dev/shm/home/qinghanc/work
WEIGHTS="/weights /work/qinghan/weights"
ws_gib() {
  local cur inactive
  cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes)
  inactive=$(awk '$1 == "inactive_file" {print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null)
  echo $(( (cur - ${inactive:-0}) / 1073741824 ))
}
last_drop=0
while :; do
  ws=$(ws_gib)
  if [ $ws -ge $SOFT_GIB ] && [ $(( $(date +%s) - last_drop )) -ge 60 ]; then
    out=$(python3 $W/drop_weight_cache.py $WEIGHTS 2>&1 | tail -1); last_drop=$(date +%s)
    echo "$(date) memguard: working set $ws GiB >= soft $SOFT_GIB, $out, now $(ws_gib) GiB" >> $W/glm-run/memguard.log
    ws=$(ws_gib)
  fi
  if [ $ws -ge $LIMIT_GIB ]; then
    echo "$(date) memguard: working set $ws GiB >= $LIMIT_GIB, stopping sglang" >> $W/glm-run/memguard.log
    bash $W/stop_sglang.sh > /dev/null 2>&1
  fi
  sleep 5
done
