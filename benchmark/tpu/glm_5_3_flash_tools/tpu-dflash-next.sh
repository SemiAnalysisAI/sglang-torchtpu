#!/bin/bash
# Slice 2: once DFlash AIME26 repeats 2-3 finish, run repeats 0-1 against the same live server.
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
export HOME=$H
until grep -q AIME_DONE $W/glm-run/aime-aime-dflash-b.out 2>/dev/null; do sleep 30; done
exec >> $W/glm-run/aime-aime-dflash.out 2>&1
echo "== DFlash repeats 0-1 on the live slice-2 server"
cd /work/qinghan/aime26 && $H/venv-port/bin/python $W/aime26.py --tag aime-dflash --repeats 8 --reps 0,1 --threads 32 --max-tokens 320000
echo AIME_DONE
