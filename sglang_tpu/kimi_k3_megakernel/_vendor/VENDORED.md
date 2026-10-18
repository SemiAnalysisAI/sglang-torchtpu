# Vendored: Inferact/tpu-megakernels

| | |
|---|---|
| Upstream | https://github.com/Inferact/tpu-megakernels |
| Commit | `4048f0820aa4ff8787f707ca9d99b2bada9751aa` (2026-09-24) |
| License | Apache-2.0, copied unchanged to [`LICENSE`](LICENSE). Upstream ships no `NOTICE` file and no per-file copyright lines. |
| Upstream JAX pin | `jax[tpu]==0.11.1` (upstream `pyproject.toml`). SGLang runs these files on its own pin (`python/pyproject_tpu.toml`, jax 0.10.2); upstream's CPU component tests pass there. |

## Files

Copied byte for byte from the commit above, at the same relative paths:

| File | Used by SGLang for |
|---|---|
| `kimi/decode_megakernel.py` | `decode_local`, the per-rank decode step |
| `kimi/__init__.py` | Reference config and helpers that `decode_megakernel.py` and `load.py` import |
| `kimi/load.py` | `Checkpoint`, `selected_local_shapes`, `_family_layers` (dense weight families). The rest of the loader is unused but kept to stay diffable |
| `collectives32.py` | In-kernel TP32 collectives, imported by `decode_megakernel.py` |
| `pool_alias.py` | Raw VMEM views, imported by `decode_megakernel.py` |

SGLang-owned files in this directory: `__init__.py` (makes the tree a package),
`LICENSE` (upstream's) and this file.

## Patch policy

- The pristine import is one commit; every SGLang change is a later commit
  touching only this directory, one concern per commit, listed below.
- Every changed hunk carries a `# SGLANG-PATCH: <why>` comment.
- No formatter or linter runs here: `.pre-commit-config.yaml` excludes the
  directory, so `diff` against upstream shows only the patches.
- To re-sync, copy the upstream files at the new commit, update the table
  above, and re-apply the patches below one by one.

## Patches

In commit order:

1. **Package-relative imports** (`kimi/decode_megakernel.py`). Upstream runs
   from its repository root and imports `collectives32`, `kimi` and
   `pool_alias` as top-level modules.
2. **Python 3.10 syntax** (`pool_alias.py`). One f-string reused its outer
   quote character, which only Python 3.12 parses; SGLang supports 3.10.
3. **Document and guard the process-wide JAX patch** (`pool_alias.py`).
   Importing the module replaces the private
   `jax._src.pallas.mosaic.lowering._reshape_memref` for every kernel in the
   process. A header now states what it replaces, why, the JAX versions it
   targets and when it is installed, and the import fails if the private
   function's four-argument signature changes.
