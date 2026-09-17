# syntax=docker/dockerfile:1.7
#
# SGLang on Google Cloud TPU (tpu7x) via torch_tpu
# Builds on top of sglang-tpu-base (which provides all pinned dependencies).
#
ARG BASE_IMAGE=us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci/sglang-tpu-base:latest
FROM ${BASE_IMAGE} AS final

# Match requirements/uv.lock.json; test jobs use the image's installed Python stack.
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /usr/local/bin/

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

# Verify that uv runs against the Python environment installed in this image.
RUN uv run --no-project --no-managed-python python3 -c "import sglang, pytest"

CMD ["/bin/bash"]
