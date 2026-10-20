# Reproducing the write_back copy-ahead hit-rate loss on one TPU7x host

This guide reproduces, with open components only, a cache hit-rate loss in sglang-torchtpu's HiCache
`write_back` copy-ahead window, and the fix for it. It is written to be followed step by step, by an
engineer or an agent, on a single tpu7x-8 host.

Four server configurations ("arms") serve the same synthetic agentic multi-turn load for one hour each:

| Arm | sglang-torchtpu commit | Write policy | Copy-ahead window |
|---|---|---|---|
| A | `deec4fbada` (main) | write_through | off |
| B | `deec4fbada` (main) | write_back | off |
| C | `deec4fbada` (main) | write_back | on (default) |
| D | `041f406ea1`, the first commit of [sgl-project/sglang-torchtpu#1586](https://github.com/sgl-project/sglang-torchtpu/pull/1586) | write_back | on (default), with the fix |

What to look for: C has a lower cache hit rate than B from the first ten minutes, and D brings it back
to B's level. [Expected results](#9-expected-results) gives the numbers and what counts as reproduced.

## 1. Prerequisites

- One tpu7x-8 host: 4 chips, 8 TensorCores. Every arm uses TP8/EP8.
- Host RAM: the host KV pools take about 560 GB (70 GB per rank, 8 ranks), plus the server processes.
  Plan for at least 700 GB free. With less, edit the `HOST_POOL_GB` constant in
  `agentic_hicache.py` (it is not an environment variable); the comparison then
  changes, because the host pool no longer holds the whole working set.
- Disk: about 61 GB for the model, plus room for logs and the compile cache.
- Python 3.12 in a virtual environment (the version measured), `git`, `curl`, and a C compiler (`gcc --version` or `clang --version` must
  work): the launcher builds a small telemetry stub with it at start-up.
- Network access to GitHub, PyPI, download.pytorch.org and Hugging Face.
  `openai/gpt-oss-120b` is public; no token is needed.
- Time: one hour of load per arm, plus server start-up. The arms can run one after another on the same
  host; this guide assumes that.

## 2. Install sglang-torchtpu

This package lives on the `yifeixu/hicache-write-back-repro` branch of sglang-torchtpu. Keep it in its
own clone, so switching the server checkout between arms does not touch it:

```bash
git clone --branch yifeixu/hicache-write-back-repro https://github.com/sgl-project/sglang-torchtpu.git ~/repro-pkg
cd ~/repro-pkg/benchmark/tpu/hicache_write_back_repro
```

Create a Python 3.12 virtual environment, then write a small file that every later block sources, so
each block also works in a fresh shell (an agent's shell may reset between commands). Run this from
this package's directory, so the file records its path as `PKG`:

```bash
python3.12 -m venv ~/repro-venv
cat > ~/repro_env.sh <<ENV
export REPO=\$HOME/sglang-torchtpu PKG="$(pwd)" MODELS=\$HOME/models
. \$HOME/repro-venv/bin/activate
cd "\$REPO"
ENV
cat ~/repro_env.sh   # PKG must be this package's directory
```

Every block below starts with `. ~/repro_env.sh; ARM=<letter>` and writes its files as
`server_$ARM.log`, `client_$ARM.jsonl` and `metrics_$ARM.txt` in the checkout.

Follow the repository's "From source" install, then check out the commit for the arm. The install is
editable, so switching commits switches the code; restart the server after a switch.

```bash
git clone https://github.com/sgl-project/sglang-torchtpu.git ~/sglang-torchtpu
. ~/repro_env.sh
git fetch origin 041f406ea1044528805864d31b42e60b07c7bc56    # arm D's commit
git checkout deec4fbada                                     # arms A, B, C
cp python/pyproject_tpu.toml python/pyproject.toml
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e python/
pip install --no-deps -e .
git checkout -- python/pyproject.toml
```

For arm D:

```bash
. ~/repro_env.sh; git checkout 041f406ea1044528805864d31b42e60b07c7bc56
```

The only runtime difference
between the two commits is `sglang_tpu/hicache_write_back_copy_ahead.py` (the commit also adds a unit
test); nothing needs reinstalling. If that commit is ever unreachable, apply this package's
`arm_d_full_pin.patch` on `deec4fbada` instead. #1586 has later commits that were not measured here; use
`041f406ea1` exactly.

Copy this package's server config into the checkout. Keep it there across checkouts:
`git checkout` leaves an untracked file in place.

```bash
. ~/repro_env.sh; cp $PKG/agentic_hicache.py benchmark/tpu/recipe/configs/gpt_oss_120b/
```

## 3. Model

```bash
. ~/repro_env.sh
python3 -c 'import os; from huggingface_hub import snapshot_download; snapshot_download("openai/gpt-oss-120b", local_dir=os.path.expandvars("$MODELS/gpt-oss-120b"), ignore_patterns=["original/*", "metal/*"])'
```

## 4. Fixed-date chat template

The gpt-oss chat template writes the current date into every system prompt. At midnight (server-local
time) every prompt changes, so every cached prefix becomes useless at once, and all arms lose their
hits together. Use a copy of the template with a fixed date:

```bash
. ~/repro_env.sh
sed 's/strftime_now("%Y-%m-%d")/"2026-01-01"/' $MODELS/gpt-oss-120b/chat_template.jinja \
  > $MODELS/gpt-oss-fixed-date.jinja
grep -c 'Current date: " + "2026-01-01"' $MODELS/gpt-oss-fixed-date.jinja   # must print 1
```

## 5. Start the server

From the repository root. Every arm sets the model, the fixed-date template and its policy variables:

| Arm | Environment |
|---|---|
| A | `HICACHE_WRITE_POLICY=write_through SGLANG_TPU_HICACHE_WRITE_BACK_COPY_AHEAD_TOKENS=0` |
| B | `HICACHE_WRITE_POLICY=write_back SGLANG_TPU_HICACHE_WRITE_BACK_COPY_AHEAD_TOKENS=0 SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM=1` |
| C | `HICACHE_WRITE_POLICY=write_back SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM=1` |
| D | `HICACHE_WRITE_POLICY=write_back SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM=1` (at commit `041f406ea1`) |

- On main, the copy-ahead window is on by default for `write_back`. Arm B turns it off with
  `SGLANG_TPU_HICACHE_WRITE_BACK_COPY_AHEAD_TOKENS=0`. Arm A sets the same variable; it has no effect
  under `write_through`.
- `SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM=1` keeps write_back's host duplicate reclaim out of the
  comparison in every write_back arm.

Arm C, for example:

```bash
. ~/repro_env.sh; ARM=C
MODEL=$MODELS/gpt-oss-120b CHAT_TEMPLATE=$MODELS/gpt-oss-fixed-date.jinja \
HICACHE_WRITE_POLICY=write_back SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM=1 \
  nohup python3 -m benchmark.tpu.launch_server \
  --config recipe/configs/gpt_oss_120b/agentic_hicache.py > server_$ARM.log 2>&1 < /dev/null &
echo $! > server.pid
```

Write the `--config` path exactly as shown: the launcher resolves `recipe/...` against
`benchmark/tpu/` and reads any other value as JSON. Add `--dry-run` to print the resolved
`sglang.launch_server` command without starting it. The config caps the device KV pool at 786,432
tokens, sets 70 GB per rank of host pool (`--hicache-size 70`), the Raiden HiCache backend, an SWA pool
at 0.2 of the full pool, and keeps SWA KV for every token
(`SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=0`).

Wait until the server answers. The first start compiles kernels and can take 20 minutes or more. This
gives up after an hour, or as soon as the launcher exits, and then stops the launcher:

```bash
. ~/repro_env.sh; ARM=C
ready=0; end=$((SECONDS + 3600))
while [ $SECONDS -lt $end ]; do
  curl -sf --max-time 10 http://127.0.0.1:30000/health > /dev/null && { ready=1; break; }
  kill -0 "$(cat server.pid)" 2>/dev/null || break
  sleep 20
done
if [ $ready = 1 ]; then echo ready; else
  echo "server not ready; see server_$ARM.log"; kill "$(cat server.pid)" 2>/dev/null
fi
```

Continue with the arm only if this prints `ready`. The pid file lets a later shell stop the server.

Check that the arm is configured as intended before starting the load:

| Arm | Check | Expected |
|---|---|---|
| A | `grep -c "copy-ahead" server_$ARM.log` | `0` |
| B | `grep -c "copy-ahead off: evictions back up synchronously" server_$ARM.log` | at least `1` |
| C, D | `grep -c "HiCache write_back copy-ahead: 32768 tokens" server_$ARM.log` | at least `1` |
| C | `grep -c "def _pin_full_host" sglang_tpu/hicache_write_back_copy_ahead.py` | `0` |
| D | `grep -c "def _pin_full_host" sglang_tpu/hicache_write_back_copy_ahead.py` | `1` |

and `git rev-parse HEAD` prints the arm's commit.

## 6. Run the load

`agentic_multiturn_client.py` uses only the Python standard library. Run it on the server host, so its
start time is in the same clock as the server log:

```bash
. ~/repro_env.sh; ARM=C
nohup python3 $PKG/agentic_multiturn_client.py --label $ARM --duration 3600 \
  > client_$ARM.jsonl 2> client_$ARM.err < /dev/null &
```

It runs for a little over an hour. Poll until the last line is the summary:

```bash
. ~/repro_env.sh; ARM=C
tail -n 1 client_$ARM.jsonl | grep -q '^{"summary"' && echo done || echo running
```

The defaults are the measured setting: 20 sessions, sessions restart at 72k prompt tokens, first
conversations start at a seeded random turn in 0-20, 3 reminders per turn, 10% of reminders selected
for replay. The prompt sequence of each session is fixed by the seed. The load is closed-loop, so a
faster server gets further through each session's sequence. The traffic shape:

- Main request: a system prompt of about 7.2k tokens plus the session history.
- Each main request is followed by 3 parallel reminder requests: a reminder system prompt of about
  1.9k tokens, the same history, and a short question. Each of the five reminder kinds has its own
  system prompt, so each session caches up to six prefix chains.
- Each turn adds about 2.5k tokens of history (1.3-3.8k). The largest main prompt is about 75k tokens.
- `--rotate-system-prompt-at SECONDS` (off by default) changes every system prompt at that time, to
  invalidate all cached prefixes at once. Replays keep their original input.

When it finishes, the last line of `client_$ARM.jsonl` is the summary. Its fields:

| Field | Meaning |
|---|---|
| `requests`, `errors` | requests that returned, requests that failed; all requests start before `duration_s` |
| `requests_per_s` | `requests / duration_s` |
| `requests_per_10min` | requests by start time |
| `drain` | requests still running at `duration_s`, and when the last one ended |
| `session_generations` | conversations started (a session restarts at `--ctx-max`) |
| `cached_fraction` | cached prompt tokens over prompt tokens, as reported by the server |
| `finish_length_fraction` | replies that hit `max_tokens`; most gpt-oss replies do, which is expected |
| `replay` | replays sent, and how many differ in their first 160 characters from the first reply. Differences are expected even at temperature 0, because cached and recomputed KV give slightly different logits. |
| `latency_s` | p50 / p90 / p99 per request kind |
| `worker_crashes` | must be empty; the client exits non-zero otherwise |

## 7. Collect

Right after the client finishes, before stopping the server:

```bash
. ~/repro_env.sh; ARM=C
curl -s http://127.0.0.1:30000/metrics > metrics_$ARM.txt
```

Then stop the server, wait for it to exit, and start the next arm:

```bash
. ~/repro_env.sh
kill "$(cat server.pid)"; while kill -0 "$(cat server.pid)" 2>/dev/null; do sleep 5; done
```

## 8. Compare

```bash
. ~/repro_env.sh
python3 $PKG/compare.py A=client_A.jsonl:server_A.log:metrics_A.txt B=client_B.jsonl:server_B.log:metrics_B.txt \
  C=client_C.jsonl:server_C.log:metrics_C.txt D=client_D.jsonl:server_D.log:metrics_D.txt
```

`compare.py` (standard library only) prints the comparison table and the hit rate per 10 minutes. Its
decode column is the mean of the scheduler's sampled `gen throughput` values, not completed tokens over
the hour. The hit rate comes from the scheduler's `Prefill batch` lines of TP rank 0, inside the client's hour:
(`#hbm-hit-token` + `#host-hit-token`) / (those + `#new-token`). Eviction time is
`sglang:eviction_duration_seconds_sum` for TP rank 0 from `/metrics`. Other useful `/metrics` counters:
`sglang:hicache_backup_tokens_total`, `sglang:load_back_tokens_total`,
`sglang:num_retracted_requests_total`. These `/metrics` counters cover the whole server lifetime
(warm-up and drain included), not only the client's hour. `compare.py` warns if the client recorded
errors or worker crashes; a run with errors does not count.

## 9. Expected results

Measured on six tpu7x-8 hosts in parallel, one hour per arm, with C and D run twice. In the measured
build the copy-ahead module ran the same code as the commits above (it also carried a test-only switch
selecting the merged pin for arm C and the fixed pin for arm D); other cache code differed only in
features that are off by default, and other parts of the stack differed, so absolute throughput on
your build can differ. Full tables, including HiCache counters, are in [results.md](results.md).

| Arm | Requests | Hit % | Hit % per 10 min | Eviction s (TP0) |
|---|---|---|---|---|
| A | 4,973 | 68.9 | 70, 76, 67, 66, 63, 69 | 25 |
| B | 4,797 | 73.1 | 69, 79, 70, 78, 70, 72 | 1,880 |
| C | 2,235 / 2,217 | 10.7 / 10.6 | 9-15 in every bucket | 1,697 / 1,674 |
| D | 5,497 / 5,112 | 71.2 / 72.9 | 70, 76, 67, 75, 70, 68 (first run) | 732 / 600 |

Counts as reproduced when all of these hold:

1. **C loses hits.** C's hit rate over the hour is at least 20 points below B's, and below B's in every
   10-minute bucket.
2. **D restores them.** D's hit rate over the hour is within 5 points of B's.
3. **D keeps the window's benefit.** D's eviction time is less than half of B's, and D serves at least
   90% of B's requests (`requests_D >= 0.9 * requests_B`).
4. **No errors.** Every client summary shows 0 errors and no worker crashes.

Run to run, the same arm varied by about 7-8% in requests and 2 points in hit rate. The first ~2
minutes of each run are cold (every session's first request has a long uncached prompt).

## 10. Prefill-only variant

To measure prefill throughput and cache reuse with decode out of the picture, add `--prefill-only` to
the client. Every request then asks for one token (`max_tokens=1`). Nothing else changes: the client's
history holds a canned assistant line, never the model's reply, so the conversations and their prefix
reuse are the same as in the normal run. Run sections 5-7 with `ARM=PA`, `PB`, `PC` and `PD` (the
servers of arms A-D), and start the client with `--prefill-only`:

```bash
. ~/repro_env.sh; ARM=PC
nohup python3 $PKG/agentic_multiturn_client.py --label $ARM --duration 3600 --prefill-only \
  > client_$ARM.jsonl 2> client_$ARM.err < /dev/null &
```

Poll for the summary line as in section 6, then compare:

```bash
. ~/repro_env.sh
python3 $PKG/compare.py PA=client_PA.jsonl:server_PA.log:metrics_PA.txt PB=client_PB.jsonl:server_PB.log:metrics_PB.txt \
  PC=client_PC.jsonl:server_PC.log:metrics_PC.txt PD=client_PD.jsonl:server_PD.log:metrics_PD.txt
```

`compare.py` then also prints input tokens per second (all prompt tokens, and the uncached part) and
the latency per request kind, which with one output token is the time to first token. Measured, one run
per arm (PA-PD are arms A-D with `--prefill-only`):

| Arm | Requests | Hit % | Input tok/s (uncached) | Main TTFT p50 / p90 / p99 s | Eviction s (TP0) |
|---|---|---|---|---|---|
| PA | 5,877 | 67.9 | 63,404 (20,407) | 21.6 / 39.1 / 72.2 | 30 |
| PB | 6,202 | 72.3 | 67,081 (18,645) | 20.7 / 35.8 / 71.0 | 2,716 |
| PC | 2,131 | 9.5 | 23,211 (21,000) | 68.0 / 95.4 / 108.3 | 3,187 |
| PD | 5,797 | 72.0 | 62,502 (17,538) | 23.0 / 36.7 / 69.3 | 1,839 |

Hit % per 10 minutes: PA 10, 62, 78, 76, 74, 74; PB 10, 77, 78, 79, 79, 70; PC 8-11 in every bucket;
PD 10, 76, 78, 79, 78, 75. The first 10 minutes are nearly cold in every arm (in PB, 455 requests with
a mean prompt of 39k tokens, and only 314 requests in minutes 2-10, which looks like a server stall
rather than a cold cache); why this window is colder than in the normal run was not investigated.

Counts as reproduced when criteria 1, 2 and 4 above hold for PB, PC and PD, with criterion 1 judged on
the buckets after the first 10 minutes, and PD's eviction time is
below PB's with `requests_PD >= 0.9 * requests_PB` (measured: 68% of PB's eviction time, 93% of its
requests). A disaggregated prefill server (prefill/decode split) was not tested.

## 11. Pitfalls

- **Date rollover.** Without the fixed-date template, a run that crosses server-local midnight loses
  every cached prefix at that moment, in every arm. It looks like a cache collapse and is not one.
- **The window is on by default on main.** Checking out main and setting only `write_back` gives arm C,
  not arm B. Arm B needs `SGLANG_TPU_HICACHE_WRITE_BACK_COPY_AHEAD_TOKENS=0`.
- **Host memory.** The host pools are allocated at start-up. If the host cannot hold 560 GB of pools,
  the server fails to start or the system swaps. A smaller `HOST_POOL_GB` changes the result: once the
  host pool cannot hold the working set, all arms lose hits to capacity.
- **Session ages.** Keep `--initial-turn-max` at its default. With 0, all sessions grow and restart
  together, and the working set swings above the host pool at the same moment in every session. Hit
  rates then collapse in every arm, write_through included, for reasons unrelated to the write policy.
- **Clock and timezone.** `compare.py` aligns the server log with the client's start time in local
  time. If the client and the server run in different timezones, pass `--start-<ARM>` with the
  server-local start time.
- **One run at a time.** The arms need the whole host. Run them one after another, restarting the
  server between arms so each starts with an empty cache.
