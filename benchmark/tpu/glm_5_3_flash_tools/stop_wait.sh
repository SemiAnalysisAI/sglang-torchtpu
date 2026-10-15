#!/bin/bash
# Stop sglang and block until no sglang process remains (a crashed server holds /dev/vfio for ~60 s).
W=/dev/shm/home/qinghanc/work
bash $W/stop_sglang.sh > /dev/null
while grep -lqsE "sglang.launch_server|sglang::" /proc/[0-9]*/cmdline 2>/dev/null; do sleep 5; done
sleep 5
