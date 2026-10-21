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
if [ -n "${KPOOL_CAPTURE_ROOT:-}" ]; then
  if [ "${KPOOL_ANALYSIS_WORKERS:-1}" -gt 1 ]; then
    python benchmark/tpu/kpool/profile_matrix.py --capture-root "$KPOOL_CAPTURE_ROOT" \
      --workers "$KPOOL_ANALYSIS_WORKERS" \
      --output "${ARTIFACT_LOCAL_DIR:?}/rank-${FALCON_RANK:-0}/analysis/results"
  else
    python benchmark/tpu/kpool/profile_analyze.py --capture-root "$KPOOL_CAPTURE_ROOT" \
      --output "${ARTIFACT_LOCAL_DIR:?}/rank-${FALCON_RANK:-0}/analysis/results"
  fi
fi
sleep 7200
