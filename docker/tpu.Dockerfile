# syntax=docker/dockerfile:1.7
#
# SGLang on Google Cloud TPU (tpu7x) via torch_tpu
# Builds on top of sglang-tpu-base (which provides all pinned dependencies).
#
ARG BASE_IMAGE=us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci/sglang-tpu-base:latest

########################################################
# Gateway Builder
########################################################
# Builds sgl-model-gateway in isolation so Python-only changes don't trigger a
# full Rust recompilation. Mirrors the gateway_builder stage in docker/Dockerfile.
FROM ${BASE_IMAGE} AS gateway_builder

WORKDIR /build

# Copy ONLY the gateway source (not the full repo)
COPY sgl-model-gateway /build/sgl-model-gateway

# The TPU base carries only git/curl/build-essential. smg-grpc-client's build
# script compiles .proto files, so it needs protoc, and --features
# vendored-openssl builds OpenSSL from source, which needs perl and pkg-config.
# docker/gateway.Dockerfile installs the same set.
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config protobuf-compiler perl libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && protoc --version

# Install Rust, build the Python bindings wheel, then drop the toolchain so none
# of it reaches the final image.
RUN --mount=type=cache,target=/root/.cache/pip \
    curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && python3 -m pip install maturin \
    && cd /build/sgl-model-gateway/bindings/python \
    && ulimit -n 65536 && maturin build --release --features vendored-openssl --out /build/gateway_wheels \
    && rm -rf /root/.cargo /root/.rustup /build/sgl-model-gateway/target \
              /build/sgl-model-gateway/bindings/python/target

########################################################
# Final image
########################################################
FROM ${BASE_IMAGE} AS final

WORKDIR /sgl-workspace/sglang

# Copy source trees
COPY README.md LICENSE ./
COPY python    ./python
COPY test      ./test
COPY scripts   ./scripts
COPY benchmark ./benchmark

# Swap the TPU dependency manifest into place
RUN cd python && rm -f pyproject.toml && cp pyproject_tpu.toml pyproject.toml

# Install SGLang in editable mode without reinstalling dependencies
# Fast build on top of base
RUN pip install --no-deps -e ./python

# Install the sglang-router built from this tree. pyproject_tpu.toml no longer
# declares it, so this is the only source of the package; --force-reinstall
# covers a base image that still carries the old registry wheel.
COPY --from=gateway_builder /build/gateway_wheels /tmp/gateway_wheels
RUN pip install --no-cache-dir --force-reinstall /tmp/gateway_wheels/*.whl \
    && rm -rf /tmp/gateway_wheels

# The launcher degrades to a hard error if the native extension is missing, so
# assert it here rather than discovering it on a TPU node mid-test.
RUN python -c "import sglang_router.sglang_router_rs as rs; from sglang_router.router import Router; print('sglang_router OK:', rs.__file__)"

CMD ["/bin/bash"]
