#!/usr/bin/env bash
set -euo pipefail
out="${ARTIFACT_LOCAL_DIR:?}/rank-${FALCON_RANK:-0}/device-profile"
mkdir -p "$out"
# All variants use the already verified profiler-compatible runtime.
python -m pip install jax==0.11.2 jaxlib==0.11.2
# Resolve the intended wheel explicitly; do not inherit the image's libtpu.
export TPU_LIBRARY_PATH="$(python -c 'import libtpu; print(libtpu.get_library_path())')"
export LIBTPU_INIT_ARGS="--xla_xprof_register_llo_debug_info=true"
echo PHASE_BASELINE_PROFILER_WIRING
set +e
python benchmark/tpu/kpool/profile_smoke.py --output "$out/abi-original" > "$out/abi-original.log" 2>&1
abi_status=$?
set -e
cat "$out/abi-original.log"
if [ "$abi_status" -ne 0 ]; then
  if ! grep -q 'Unexpected PLUGIN_Profiler_Api size: expected 80, got 104' "$out/abi-original.log"; then
    exit "$abi_status"
  fi
  echo PHASE_CORRECTED_PROFILER_RUNTIME
  # The profiler is linked to libjax_common, so replace the matched JAX pair.
  # Every compared variant is subsequently compiled and sampled in this venv.
  python -m pip install jax==0.11.2 jaxlib==0.11.2
  python benchmark/tpu/kpool/profile_smoke.py --output "$out/abi-corrected" > "$out/abi-corrected.log" 2>&1
  cat "$out/abi-corrected.log"
fi
python -m pip freeze > "$out/requirements-resolved.txt"
git fetch --depth=1 origin 985b4f8c9f2151cfae7c110156f7433892e1ee60
git fetch --depth=1 origin 61edd0e421a877158f7bfdfdfd2ae6133173cbfc
git fetch --depth=1 origin de72e465f066024c2e678b63de7d4cd73212a404
echo PHASE_DEVICE_CAPTURE
timeout 7200 python benchmark/tpu/kpool/benchmark.py --output "$out/benchmark" \
  --cases decode prefill --decode-pipeline --decode-query-lengths 1 6 --warmup 10 --repeats 100 --hlo-text "$@"
echo DEVICE_CAPTURE_COMPLETE
