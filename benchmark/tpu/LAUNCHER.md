# Unified TPU launcher

`launch_server.py` starts servers and P/D routers from Python recipe modules or
JSON configuration. It sets `runtime_envs` on child processes before they import
SGLang and owns their readiness checks and process-group cleanup.

## Launch a server

From the repository root:

```bash
python3 -m benchmark.tpu.launch_server \
    --config recipe/configs/qwen3_coder_30b_a3b_fp8.py
```

The Bazel binary supplies the Python interpreter, dependencies, recipe modules,
and prebuilt telemetry library:

```bash
bazel run //benchmark/tpu:launch_server -- \
    --config recipe/configs/qwen3_coder_30b_a3b_fp8.py --dry-run
```

`--dry-run` prints the resolved commands without starting servers. `--config`
also accepts a JSON object or `@path/to/config.json` containing `server_args`
and optional `runtime_envs` and `extra_args`. `--runtime-envs` accepts a JSON
dictionary or `@path/to/env.json`; its values override the recipe environment.
String values set variables and `null` removes them from the child environment.

## Single-host P/D tests

The GPT-OSS-120B P/D recipes use one full TP8/EP8 host per engine, with a separate
prefill pod and decode pod. The prefill pod also runs the router. Their Python
modules export `GROUP`, `PREFILL_CONFIG`, and `DECODE_CONFIG`:

- `recipe/configs/gpt_oss_120b/pd_singlehost.py`: baseline Raiden P/D.
- `recipe/configs/gpt_oss_120b/pd_singlehost_spec.py`: EAGLE3 on both engines.
- `recipe/configs/gpt_oss_120b/pd_singlehost_decode_radix.py`: decode radix cache.

The prefill fixture supplies `SGLANG_LAUNCHER_HOSTNAMES` in prefill,decode order
and `SGLANG_LAUNCHER_RANK=0`. These variables describe the serving group without
rewriting slice-local TPU ranks. The decode service starts independently from
the same module, without needing the prefill hostname:

```bash
python3 -m benchmark.tpu.launch_server \
    --config recipe/configs/gpt_oss_120b/pd_singlehost_spec.py:DECODE_CONFIG
```

Selecting `PREFILL_CONFIG` similarly starts only the prefill engine, without a
router. Running the group recipe on prefill starts the router after both engines
are healthy.

## Test lifecycle

The three P/D integration cases inherit
[`TPULauncherTestCase`](launcher_test_case.py). Its `setUpClass` starts the shared
launcher and waits for the router. All test methods share that serving group;
`tearDownClass` stops the local launcher and children. The existing `KEEP_UP=1`
debug option keeps them running. The decode service pod runs until JobSet teardown.

Subclasses set `config_module`, `base_url`, and optional runtime, logging, and
readiness attributes. Override `get_runtime_envs()` for topology discovered at
setup time. `check_server_configuration()` performs checks before evaluation;
the EAGLE3 case uses it for its existing server checks. Setup failures also stop
the launcher.

`RecipeLauncher` in `sglang.test.tpu.eval_support` delegates Python-config
startup, readiness, and cleanup to the same launcher while retaining servers
across evaluation gates. Existing shell recipes remain supported. Multihost
tests and JobSet launch paths retain their existing launchers.

## Telemetry library outside Bazel

The P/D recipes preload the existing libtpu telemetry workaround. Bazel supplies
the compiled library through runfiles. For launches outside Bazel, supply a
prebuilt library through `LIBTPU_TELEMETRY_STUB`; for example:

```bash
bazel build //benchmark/tpu:libtpu_telemetry_stub.so
export LIBTPU_TELEMETRY_STUB="$PWD/bazel-bin/benchmark/tpu/libtpu_telemetry_stub.so"
```
