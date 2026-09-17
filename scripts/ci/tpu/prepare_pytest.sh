#!/usr/bin/env bash
# Source before uv/pytest in the existing TPU image. Keep its installed serving
# stack; uv runs without syncing a project or resolving TPU dependencies.
# Match the uv version in requirements/uv.lock.json.
python3 -m pip install --quiet --disable-pip-version-check --no-deps uv==0.12.9 || exit 1

# Bazel supplies this as a built data dependency. The source-based CI image
# supplies a C compiler, so build the same artifact before starting pytest.
if [[ -z "${LIBTPU_TELEMETRY_STUB:-}" ]]; then
    _tpu_pytest_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    cc -O2 -fPIC -shared "${_tpu_pytest_root}/benchmark/tpu/libtpu_telemetry_stub.c" \
        -o /tmp/libtpu_telemetry_stub.so || exit 1
    export LIBTPU_TELEMETRY_STUB=/tmp/libtpu_telemetry_stub.so
    unset _tpu_pytest_root
fi
