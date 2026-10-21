#!/usr/bin/env bash
set -euo pipefail
# The capture runtime is installed by falcon_task.py. XProf only runs on CPU.
python -m pip install xprof-nightly==2.25.0a20261020
mkdir -p "${ARTIFACT_LOCAL_DIR:?}/rank-${FALCON_RANK:-0}/analysis"
python -m pip freeze > "${ARTIFACT_LOCAL_DIR:?}/rank-${FALCON_RANK:-0}/analysis/requirements-resolved.txt"
python benchmark/tpu/kpool/locate_xprof_skill.py
printf '%s\n' "$PWD" > /tmp/kpool-analysis-source
printf '%s\n' "$(command -v python)" > /tmp/kpool-analysis-python
echo ANALYSIS_READY
sleep 7200
