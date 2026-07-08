# TPU Docker image

`docker/tpu.Dockerfile` builds sglang for Google Cloud TPUs (torch_tpu backend).
Its structure mirrors torchtpu-vllm's `docker/Dockerfile`: it builds on
torch_tpu's deps-only `torch-tpu-base` image, then layers `base`, `ci`, `dev`,
and `prod` variants on top.

`torch-tpu` (and `libtpu`) are pinned in `python/pyproject_tpu.toml`;
`torch`/`jax`/`jaxlib` come warm from `torch-tpu-base`. The pinned `torch-tpu`
wheel is installed on top from the private torch-tpu virtual registry, so the
torch_tpu version is pinned in the repo, not floated by the base-image tag.

The image lays everything out under **`/workspace`**:
`/workspace/sglang` (the sglang app) and `/workspace/torch_tpu` (a torch_tpu
**source** checkout for reference/patching — the *installed* torch_tpu stays the
pinned wheel). Both are **cloned fresh at `main`** during the build (overridable
via `--build-arg SGLANG_REF=... / TORCH_TPU_REF=...`), so the image bakes
canonical main rather than whatever local branch the build ran from.

> [!NOTE]
> Cloning `main` suits a standalone / nightly image. For PR CI (build+test the
> commit under test) switch the sglang app source to `COPY .` so the image is the
> checked-out ref — see the "app source vs CI" discussion in the build runbook.

## Building

Use the helper script, which mints a short-lived OAuth token for the private
registry and passes it as a BuildKit secret (so it never lands in a layer):

```bash
# prod variant (default)
./docker/build_tpu_image.sh

# dev variant with a custom tag
./docker/build_tpu_image.sh --target dev -t sglang-tpu-dev
```

Requires `gcloud` (or an `ACCESS_TOKEN` env var) with read access to the
torch-tpu registry.

## Image targets

### `base`
`torch-tpu-base` + the pinned TPU dependency set reconciled and the pinned
torch_tpu wheel installed (plus the Rust toolchain, mirroring the TPU CI install
flow). Does not install sglang itself — only the dependencies from
`pyproject_tpu.toml`.

### `ci`
`base` + CI tooling: `google-cloud-cli`, `gcsfuse`, `lsof`, `psmisc`. Does not
install sglang itself.

### `dev`
`ci` + developer tooling (`vim`, `tmux`, `htop`, `less`, `tree`, `ag`, `zsh`,
`git-lfs`) + sglang cloned at `main` under `/workspace/sglang`, installed editable
with test extras (`uv pip install -e "python[dev_tpu]"`), plus the torch_tpu
source under `/workspace/torch_tpu`. Intended for development and tests.

### `prod`
`base` + sglang cloned at `main`, installed regularly (`python[srt_tpu]`), plus
the torch_tpu source. Lean image for runtime use — no dev tools.

## Build arguments / secrets

* `BASE_IMAGE` — base image (default: torch_tpu's `torch-tpu-base:nightly-latest`,
  which ships torch/jax/libtpu warm and ABI-compatible with the torch_tpu wheels).
* `SGLANG_REF` / `TORCH_TPU_REF` — git refs to clone (default `main`).
* `SGLANG_REPO` / `TORCH_TPU_REPO` — repo URLs (defaults: the public GitHub repos).
* BuildKit secrets: `gcloud_token` (torch-tpu wheel registry) and `github_token`
  (clone the sources). See `docker/build_tpu_image.sh` and the build runbook.
