#!/bin/bash
# Boot the best DFlash recipe (MLA bf16 latent), bench real-text c1, then capture a real-text profile.
W=/dev/shm/home/qinghanc/work
exec > $W/glm-run/prof-dflash.out 2>&1
. $W/tpu_env.sh
cfg=mla_bf16_dflash
bash $W/stop_wait.sh
bash $W/run-cfg.sh glm_5_3_flash_eval/$cfg.py prof-$cfg
until grep -qE "fired up|Traceback" $W/glm-run/prof-$cfg.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/prof-$cfg.log && { echo "BOOT FAILED"; exit 1; }
cd $W
$HOME/venv/bin/python bench_real.py --runs 2 2>&1 | grep -E "^run|SUMMARY"
$HOME/venv/bin/python prof_real.py $W/prof-dflash-real TPU,CPU
find $W/prof-dflash-real -name "*.xplane.pb" -exec ls -la {} \;
echo PROF_DONE
