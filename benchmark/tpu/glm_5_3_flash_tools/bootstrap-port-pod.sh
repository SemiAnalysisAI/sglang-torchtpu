#!/bin/bash
# Usage: bootstrap-port-pod.sh <pod> -- base venv, ported tree + its venv, eval configs, scripts, memguard.
set -euo pipefail
P=$1; H=/dev/shm/home/qinghanc; W=$H/work; S=$(cd "$(dirname "$0")" && pwd)
K=(kubectl --context gke_sglang-torchtpu_us-central1-c_tpu-cluster -n tpuq exec -i "$P" -c main --)
"${K[@]}" mkdir -p $H
"${K[@]}" tar xzf - -C $H < $S/../glm-dsa-tpu/ci-tp16/base.tgz
# Logs and profiles go to shared storage (see tpu_env.sh); $W/glm-run and $W/prof link there.
"${K[@]}" mkdir -p $W
( cd $S && COPYFILE_DISABLE=1 tar cf - --no-xattrs tpu_env.sh ) | "${K[@]}" tar xf - -C $W
"${K[@]}" bash -c ". $W/tpu_env.sh && ln -sfn \$POD_DIR/glm-run $W/glm-run && ln -sfn \$POD_DIR/prof $W/prof && ls -l $W"
bash $S/deploy-port.sh "$P"
"${K[@]}" mkdir -p $W/glm-port/benchmark/tpu/recipe/configs/glm_5_3_flash_eval
( cd $S/eval_configs && COPYFILE_DISABLE=1 tar cf - --no-xattrs *.py ) | "${K[@]}" tar xf - -C $W/glm-port/benchmark/tpu/recipe/configs/glm_5_3_flash_eval
( cd $S && COPYFILE_DISABLE=1 tar cf - --no-xattrs tpu_env.sh stop_wait.sh stop_sglang.sh run-cfg-port.sh memguard.sh \
    prof_round.py prof_real.py bench_real.py bench_8k1k.py dsa_sample.py ) | "${K[@]}" tar xf - -C $W
"${K[@]}" /usr/local/bin/python3 -m venv --system-site-packages $H/venv-port
"${K[@]}" $H/venv-port/bin/pip install -q --no-deps --no-build-isolation -e $W/glm-port/python 2>&1 | tail -2
"${K[@]}" $H/venv-port/bin/pip install -q --no-deps --no-build-isolation -e $W/glm-port 2>&1 | tail -2
"${K[@]}" $H/venv-port/bin/python -c "import sglang, sglang_tpu; print(sglang.__file__, sglang_tpu.__file__)"
"${K[@]}" setsid -f bash $W/memguard.sh 820 < /dev/null > /dev/null 2>&1
echo bootstrapped "$P"
