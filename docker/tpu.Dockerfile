# syntax=docker/dockerfile:1.7
#
# SGLang on Google Cloud TPU (tpu7x) via torch_tpu
# Builds on top of sglang-tpu-base (which provides all pinned dependencies).
#
ARG BASE_IMAGE=us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci/sglang-tpu-base:latest

# experimental/sgl-router, upstream's single-model router, as a standalone
# binary. The P/D tests select it with SGLANG_TPU_PD_ROUTER=sgl_router (see
# python/sglang/test/tpu/pd_router.py). Built from THIS tree so the router the
# nightly exercises is the one the sync brought in, not a registry artifact.
# Own stage, copying only the router directory, so Python-only changes do not
# invalidate the Rust layer. Toolchain pinned to the crate's rust-toolchain.toml.
ARG SGL_ROUTER_RUST_VERSION=1.92
FROM rust:${SGL_ROUTER_RUST_VERSION}-bookworm AS sgl_router_builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work/sgl-router
# Static PCRE2 so the binary does not depend on the base image's libpcre2.
ENV PCRE2_SYS_STATIC=1
COPY experimental/sgl-router ./
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/work/sgl-router/target \
    cargo build --release --locked --bin sgl-router \
    && install -m 0755 target/release/sgl-router /usr/local/bin/sgl-router

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

# The router binary; fail the image build, not a TPU job, if it does not run.
COPY --from=sgl_router_builder /usr/local/bin/sgl-router /usr/local/bin/sgl-router
RUN sgl-router --help > /dev/null

CMD ["/bin/bash"]
