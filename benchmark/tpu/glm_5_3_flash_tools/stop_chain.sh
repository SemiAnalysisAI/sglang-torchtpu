#!/bin/bash
# Kill AIME chains and drivers (tpu-finish-s1.sh, tpu-aime.sh, aime26.py); no pkill in this image.
# Run it as its own command: a shell whose command line contains these patterns gets killed too.
for f in /proc/[0-9]*; do
  pid=${f#/proc/}; [ "$pid" = "$$" ] && continue
  cmd=$(tr "\0" " " < $f/cmdline 2>/dev/null)
  case "$cmd" in *tpu-finish-s1.sh*|*tpu-aime.sh*|*aime26.py*|*tpu-dsa-s1.sh*|*tpu-dsa-8k1k.sh*|*tpu-mtp-run.sh*|*tpu-real-run.sh*|*tpu-dsa-s1.sh*|*tpu-dsa-8k1k.sh*|*tpu-mtp-run.sh*|*tpu-real-run.sh*) kill $pid 2>/dev/null && echo "killed $pid ${cmd:0:60}";; esac
done
