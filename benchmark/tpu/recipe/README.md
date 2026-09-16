# TPU serving recipes

Per-model serving configurations, as Python modules under `configs/`. Each
module builds a `CONFIG` dict — `server_args` plus the `runtime_env` the server
needs — by copying a shared base (`configs/base.py`) and overriding the fields
that are specific to the model and host shape. These are the launch examples
for each model; CI drives the same modules, so what you run is what is tested.

## Launch

Run from the repo root. `--config` takes the path relative to `benchmark/tpu`:

```bash
python3 -m benchmark.tpu.recipe.bench_matrix.serve_group \
    --config recipe/configs/gpt_oss_120b/server.py
```

Add `--dry-run` to print the resolved `sglang.launch_server` command — with the
inherited base args filled in — without launching. This is the fastest way to
see exactly what a recipe sets before committing a host to it:

```bash
python3 -m benchmark.tpu.recipe.bench_matrix.serve_group \
    --config recipe/configs/gpt_oss_120b/server.py --dry-run
```

## Single-host serve configs

The per-model examples most people want. Each has a `Sample usage` docstring.

| config | model | shape |
|---|---|---|
| `configs/gpt_oss_120b/server.py` | GPT-OSS-120B (MXFP4 MoE) | 1 host, tp8 / ep8 |
| `configs/qwen3_coder_480b/server.py` | Qwen3-Coder-480B (FP8 MoE) | 1 host, tp8 / ep8 |
| `configs/qwen3_coder_30b_a3b_fp8.py` | Qwen3-Coder-30B (FP8 MoE) | 1 host, tp8 / ep8 |
| `configs/qwen3_0_6b.py` | Qwen3-0.6B (tracer e2e) | single chip |

## Layout

- `configs/base.py` — `_BASE_CONFIG` and `_FP8_BASE_CONFIG`: the shared TPU
  serving defaults every recipe starts from. Model configs `deepcopy` a base
  and `update` it, so a field not mentioned in a recipe comes from here.
- `configs/<model>/...` — grouped families: `gpt_oss_120b/` (single-host serve
  plus the two-host `multihost_base` for the spec variants), `qwen3_coder_480b/`
  (serve, bench, and the `nightly/` agentic P/D sweep), `mimo/`, `bench_matrix/`.
- `configs/*_mh.py` — multi-host variants, launched by the CI JobSet across
  hosts rather than by hand.
- `bench_matrix/` — the concurrency/prefix-cache sweep machinery
  (`serve_group.py` is the launcher above; `bench_leg.py`, `run_sweep.py` drive
  the matrix legs).

## Customizing

The `.py` configs are fixed values, not env-parameterized like the old shell
recipes. To vary a recipe, edit the fields in its `server_args.update({...})`
block (or copy the module and override), then `--dry-run` to confirm the
resolved command before launching.
