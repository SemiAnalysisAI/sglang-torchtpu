#!/bin/bash
# Kill sglang servers/launchers (no pkill in this image). $1 = signal (default TERM).
# Run it as its own command: a shell whose command line contains these patterns gets killed too.
sig=${1:-TERM}
for f in /proc/[0-9]*; do
  pid=${f#/proc/}; [ "$pid" = "$$" ] && continue
  cmd=$(tr "\0" " " < $f/cmdline 2>/dev/null)
  case "$cmd" in *sglang.launch_server*|*sglang::*|*run-glm53-flash*|*launch_server.sh*) kill -$sig $pid 2>/dev/null && echo "killed $pid ${cmd:0:60}";; esac
done
