#!/bin/bash
set -e

# Build the sglang TPU image (docker/tpu.Dockerfile).
#
# Mints two short-lived tokens and passes them as BuildKit secrets (so they never
# land in an image layer): a gcloud OAuth token for the private torch-tpu wheel
# registry, and a GitHub token (gh) to clone the sglang/torch_tpu sources at main.
#
# Usage examples:
#   1. Build the prod variant with defaults:
#        ./docker/build_tpu_image.sh
#   2. Build the dev variant with a custom tag:
#        ./docker/build_tpu_image.sh --target dev -t sglang-tpu-dev
#   3. Build against a specific base image:
#        ./docker/build_tpu_image.sh -b <base-image>

# Default values
IMAGE_TAG="sglang-tpu-local"
BASE_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu-base:nightly-latest"
TARGET="prod"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    -b|--base-image)
      BASE_IMAGE="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Get script / repo directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../" && pwd)"

echo "===> Building sglang TPU image..."
echo "Image Tag:    $IMAGE_TAG"
echo "Base Image:   $BASE_IMAGE"
echo "Build Target: $TARGET"

# Mint a short-lived access token for the private torch-tpu registry.
# Prefer an explicit ACCESS_TOKEN, then gcloud, then the GCE metadata server.
if [ -z "${ACCESS_TOKEN:-}" ]; then
  if command -v gcloud >/dev/null 2>&1; then
    ACCESS_TOKEN="$(gcloud auth print-access-token)"
    export ACCESS_TOKEN
  elif command -v curl >/dev/null 2>&1; then
    METADATA_TOKEN_JSON="$(curl -sf -H "Metadata-Flavor: Google" \
      "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
      || true)"
    ACCESS_TOKEN="$(printf '%s' "${METADATA_TOKEN_JSON}" | \
      sed -nE 's/.*"access_token"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p')"
    if [ -n "${ACCESS_TOKEN}" ]; then
      export ACCESS_TOKEN
    fi
  fi
fi

if [ -z "${ACCESS_TOKEN:-}" ]; then
  echo "Missing ACCESS_TOKEN and gcloud is unavailable. Private registry authentication will fail."
  exit 1
fi

# Mint a GitHub token to clone the sglang / torch_tpu sources at build time
# (the dev and prod stages clone them). Prefer an explicit GITHUB_TOKEN, else gh.
if [ -z "${GITHUB_TOKEN:-}" ]; then
  if command -v gh >/dev/null 2>&1; then
    GITHUB_TOKEN="$(gh auth token 2>/dev/null || true)"
    export GITHUB_TOKEN
  fi
fi

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "Missing GITHUB_TOKEN and gh is unavailable. The dev/prod source clones will fail."
  exit 1
fi

DOCKER_ARGS=(
  --progress=plain
  -f "${SCRIPT_DIR}/tpu.Dockerfile"
  -t "${IMAGE_TAG}"
  --target "${TARGET}"
  --build-arg BASE_IMAGE="${BASE_IMAGE}"
  --secret "id=gcloud_token,env=ACCESS_TOKEN"
  --secret "id=github_token,env=GITHUB_TOKEN"
)

# Run docker build from the repo root (build context = repo).
DOCKER_BUILDKIT=1 docker build "${DOCKER_ARGS[@]}" "${REPO_ROOT}"

echo "===> Done!"
