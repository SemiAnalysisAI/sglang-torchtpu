# Bazel phase-one build

This repository uses Bazel 8.6.0, matching `google-pytorch/torch_tpu`.
Use Bazelisk or an explicit Bazel 8.6.0 binary so `.bazelversion` is honored.

The first TPU end-to-end target is:

```shell
bazel test --config=tpu \
  //test/registered/tpu/e2e:test_tpu_gpt_oss_120b_singlehost_spec
```

It builds `torch_tpu` and its native extension from the pinned commit in
`git@github.com:google-pytorch/torch_tpu.git`. The resulting shared libraries
use Bazel runfiles and embedded runtime search paths; no library-path override
is required. The test itself requires a single TPU v7 host, network access,
and the GPT-OSS-120B weights at `/weights/gpt-oss-120b`.

For development against the neighboring source checkout, run:

```shell
bazel test --config=tpu \
  --override_module=torch_tpu=../torch_tpu \
  //test/registered/tpu/e2e:test_tpu_gpt_oss_120b_singlehost_spec
```

## Local Bazel storage

`.bazelrc` imports `user.bazelrc` last. That file is deliberately ignored by
Git so every developer can choose machine-local storage. This checkout starts
with:

```text
startup --output_user_root=/dev/shm/bazel-sglang-torchtpu
```

Change it to a directory under `/filestore/srcs` if `/dev/shm` is too small
for the source build. No PATH or Python overrides are needed anywhere: build
actions and tests use the hermetic Python toolchain from MODULE.bazel.

## Updating Python dependencies

The lock is generated from `python/pyproject_tpu.toml` (plus the `test`
extra), the same file the TPU container images install from. Regenerate with:

```shell
bazel run //requirements:regenerate_lock
```

which runs `uv pip compile` into `requirements/requirements_lock_3_12.txt`
using the hermetic uv binary pinned by `rules_uv` in MODULE.bazel — the same
uv on every machine, nothing required on PATH. (The underlying
`requirements/regenerate_lock.sh` still works standalone with a locally
installed uv.)
The TPU stack itself (`torch-tpu`, `libtpu`, `jax`, `jaxlib`,
`tpu-raiden-torch`) is excluded via `requirements/tpu_stack_excludes.txt`:
Bazel builds torch_tpu from the pinned source module and takes libtpu/jax
from torch_tpu's own pip hub, so resolving them from PyPI would only fight the
private-registry credentials they need. Edit `pyproject_tpu.toml`, not the
lock, to change a dependency.

## Fine-grained BUILD files for python/sglang

`python/sglang/**/BUILD.bazel` and `python/cycles.bzl` are generated, not
handwritten. sglang's import graph has real cycles, so the generator collapses
each strongly connected component into one `cycle_N` py_library (sources
gathered through per-directory `cycle_N_files` filegroups, since a glob cannot
cross a package boundary); acyclic modules keep their own per-directory
targets. Regenerate after adding/removing modules or changing imports:

```shell
bazel run //bazel:gen_sglang_build
```

Only the **TPU serving closure** gets BUILD files — directories statically
reachable from the launch entry points plus the dynamic roots
(`DYNAMIC_ROOTS` in `bazel/gen_sglang_build.py`: models, speculative,
multimodal, attention/quantization layers, hardware backends, kernel ops).
The rest of the tree (CUDA kernel zoos, `multimodal_gen`, ...) stays
Bazel-free; a stale BUILD file outside the closure is removed on regen. A
closure module that starts importing outside the closure fails the generator
loudly — extend `DYNAMIC_ROOTS` or `NON_CLOSURE_DEPS` deliberately.

`python/BUILD.bazel` is handwritten: it loads the generated cycles and defines
the `:sglang` aggregator (the TPU serving closure) that tests depend on.
