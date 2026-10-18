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
4. **Decode rows [0, B) of R >= B state rows** (`kimi/decode_megakernel.py`).
   SGLang keeps one set of state arrays with `max_rows` rows and runs the
   compiled tile `B` over rows `[0, B)`. `stack()` reads, writes and drains
   only those rows, and both it and `make_decode` accept states with more rows
   than the tile.
5. **uint8 MXFP4 expert weights** (`kimi/decode_megakernel.py`). SGLang's
   prefill MoE keeps the packed routed experts as uint8; the megakernel reads
   the same tensors. `stack()` treats a uint8 `expert_gu` / `expert_down` as
   uint32 words (4x fewer sublanes) and bitcasts the HBM refs in the body.
6. **Padded rows** (`kimi/decode_megakernel.py`). SGLang pads a batch of `n`
   live rows up to the compiled tile and passes position `-1` for the padding.
   Those rows route no experts (the MoE routing masks them out of local
   ownership, and the shared-route expert prefetch also requires agreeing
   ownership), and every position read is clamped to 0 so cache reads and
   writes stay in bounds. Padding rows are never owned by a request, so their
   state contents do not matter.
7. **Caller-supplied rank** (`kimi/decode_megakernel.py`). `decode_local`
   takes an optional `rank`. Its own `jax.lax.axis_index("tp")` sits outside
   `pallas_call` and lowers to `stablehlo.partition_id`, which torch_tpu's
   export bridge partitions a second time and crashes on. SGLang passes each
   device's rank as a traced value read from a `P("tp")`-sharded operand, so
   the replicated token operand holds the same values on every device.
8. **Compact state updates** (`kimi/decode_megakernel.py`). With
   `return_state_updates=True` (default False, upstream behavior), `stack()`
   takes the states read-only and returns only the `[B, ...]` KDA rows and the
   written `[B, mla, 3, 128, D]` MLA tiles, with no custom-call aliasing;
   `decode_local` writes them into the donated states with
   `dynamic_update_slice` (`_apply_state_updates`). libtpu copies every output
   of a communicating custom call that aliases an entry parameter, which cost
   two full state copies per step (7.7 GiB, 5.2 ms at 8 rows) when the states
   were aliased through `stack()`.

Upstream's own environment reads are left as they are: `TP32_PROFILE_SCOPES`
(`collectives32.py`, profiler scope names) and `K3_FP8_BLOCK` (`kimi/load.py`,
the FP8 expert reblock SGLang does not use). Neither is part of SGLang's
compile-cache key, so leave them unset when serving.
