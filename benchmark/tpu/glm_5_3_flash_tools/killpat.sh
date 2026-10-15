#!/bin/bash
# Usage: killpat.sh <substring> -- kill processes whose command line contains it (no pkill in this image).
for f in /proc/[0-9]*; do
  pid=${f#/proc/}; [ "$pid" = "$$" ] && continue
  cmd=$(tr "\0" " " < $f/cmdline 2>/dev/null)
  case "$cmd" in *killpat.sh*) ;; *"$1"*) kill $pid 2>/dev/null && echo "killed $pid ${cmd:0:60}";; esac
done
