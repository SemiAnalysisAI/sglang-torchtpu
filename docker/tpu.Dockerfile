# syntax=docker/dockerfile:1.7
#
# SGLang on Google Cloud TPU (tpu7x) via torch_tpu
# Builds on top of sglang-tpu-base (which provides all pinned dependencies).
#
ARG BASE_IMAGE=us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci/sglang-tpu-base:latest
FROM ${BASE_IMAGE} AS final

WORKDIR /sgl-workspace/sglang

# Copy source trees
COPY README.md LICENSE ./
COPY python    ./python
COPY test      ./test
COPY scripts   ./scripts
COPY benchmark ./benchmark

# Python-only CI uses the same prebuilt telemetry workaround as Bazel runfiles.
# Keep compilation out of server startup.
RUN cc -O2 -fPIC -shared benchmark/tpu/libtpu_telemetry_stub.c \
    -o benchmark/tpu/libtpu_telemetry_stub.so

# Swap the TPU dependency manifest into place
RUN cd python && rm -f pyproject.toml && cp pyproject_tpu.toml pyproject.toml

# Install SGLang in editable mode without reinstalling dependencies
# Fast build on top of base
RUN pip install --no-deps -e ./python

CMD ["/bin/bash"]
