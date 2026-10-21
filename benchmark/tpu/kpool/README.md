# Kpool device timing

`benchmark.py` captures XProf events. It does not measure host latency or
produce speedup claims. Numerical accuracy and serving tests are separate
validation gates.

**Current status: the expanded Q/B matrix fails exact-byte validation.** Of the
301 uniform shapes, only Q1/B32 decode and the 14 no-output prefill Q1/Q2 cases
pass. The other 286 shapes differ from the frozen JAX reference. This candidate
is not ready to replace the serving implementation. See the [full matrix](results/matrix.md),
[timing CSV](results/matrix.csv), and [accuracy cases](results/accuracy.csv).

The full 414-case exact-byte gate completed: 53 passed and 361 failed, with no
runtime exceptions. Performance was measured first; this accuracy gate ran
after all device timings were extracted. No kernel changes were made during
the matrix run. Serving integration and end-to-end validation remain pending.

| Mode | Measured shapes | Lower fused device p50 | Higher fused device p50 | Uniform exact-byte passes |
|---|---:|---:|---:|---:|
| Decode/verify | 224 | 223 | 1 | 1 |
| Prefill | 77 | 43 | 34 | 14 (no closed pools) |

These latency comparisons do not establish correctness. Q1/B32 decode has
JAX/fused module p50 of 25.744/14.1905 us and passes the uniform byte gate.
Q1/B64 is the only decode timing regression: 32.722/52.934 us. Prefill Q4/B16
regresses from 37.9035 to 64.710 us. Full percentiles and each shape's validation
status are in the linked tables. Each variant has 100 verified module events.

The targeted benchmark uses 14,849 raw-cache pages, donated pooled-cache output,
10 warmup calls, and 100 recorded calls per variant. It covers decode Q1/Q6 at
B1/B32 and prefill T8192. Each case alternates variant order between rounds.
All variants use one device and one runtime in the same process.

Use `falcon_task.py --mode device-profile` to render the capture task from a
pushed full Git SHA. The task first tests profiler wiring with the pinned
JAX/JAXlib 0.11.2 pair and libtpu 0.0.50. It installs this pair before the
profiler smoke check because JAX 0.10.2 hit an ABI 80/104 failure with this
profiler. Every comparison uses the same runtime. The captured manifest and
resolved requirements record it. Custom-call runtime instrumentation stays
disabled.

Use `falcon_task.py --mode profile-analysis --analysis-input PATH --profile-bucket BUCKET` for the CPU
analysis task. PATH is the mounted capture's `rank-0/device-profile` directory.
The task uses XProf 2.25.0a20261020 and keeps full traces, HLO, and event tables
remote. Collect `summary.json`, `samples.json`, and selected raw tool outputs.

Device module latency is `XLA Modules.end_ns - start_ns`. It includes the full
compiled computation. Pallas latency is the union of outermost matching
`XLA Ops` events inside that device module. It is a subset of module latency;
do not add it to module latency. Hardware-counter events do not count as ops.

XLA can reuse one executable name for equivalent variants. Host sample markers
identify call order only. The parser validates all marker names, alternating
order, counts, and executable identities, then pairs them with the serialized
device module events in order. It does not use host durations or host/device
time containment. Cross-clock offsets can make that containment incorrect.
Missing or extra module events fail the analysis instead of producing a partial
comparison. Event-to-module joins allow 1 ns for XProf integer timestamp rounding;
this does not change any measured duration. Every non-reference sample must
contain exactly one outer Pallas event. The full module span excludes gaps between separate calls.

The prior `perf_counter + block_until_ready` comparisons measured host dispatch
and synchronization. Those numbers do not establish device speedups or
regressions and must not be used for kernel optimization decisions.

## Q1 batch path

`compress_and_store(..., decode=True, decode_max_q=1)` selects the batch
implementation automatically only for B32, 16-row tiles, and the regular
pipeline. `decode_q1_batch=False` keeps the previous pipeline;
`decode_q1_batch=True` opts other Q1 shapes into the candidate. Other default
batch sizes, Q>1, prefill, serial mode, and the unrolled experiment keep their
existing graphs. The benchmark labels the old implementation
`previous_pipeline` and the candidate `q1_batch`.

The candidate starts raw and old-cache reads for independent requests, computes
one useful pool per request together, then starts the writes together. Raw
reads remain aligned 16-row transfers; the kernel selects four rows in VMEM.
A four-row HBM transfer at a nonzero pool offset failed the runtime bounds
check and is not used. Any shared destination quad selects the ordered RMW
fallback inside the same aliased call. Batch scratch is bounded to 32 requests.

The Q1 candidate also matches the *compiled* frozen reference's rounding:
pooling rounds to BF16, while Hadamard rotation remains F32 until FP8
quantization. In the tested JAX 0.11.2 executable, the reference's final
BF16-to-F32 round trip disappears. Explicitly retaining that rounding in Pallas
changed FP8 midpoint decisions. `diagnose_q1.py` compares intermediate probes
with the complete reference executable and saves its HLO remotely. These
results are specific to the pinned runtime; a runtime change requires repeating
the exact-byte gate.

Run `falcon_task.py --mode accuracy` after the performance checkpoint. It runs
`validate_q1.py`: entire cache bytes, including untouched records and padding,
must match the independent reference. Coverage includes B1/2/3/8/16/32, every
pool offset, fragmented pages, inactive requests, no closed pool, shared and
nonadjacent shared quads, zero/tiny values, and NaN/Inf. It checks the automatic
B32 dispatch separately. The old pipeline remains a diagnostic control; its
preexisting rounding mismatches are reported without weakening the new path's
exact-byte rule.

`check_isolation.py --q1-default --baseline f898371c2ce58b3fe9977dd76663b0b3625edcbf --output isolation.json`
checks 224 traced configurations. Only regular Q1/B32 changes. The changed
default graph must equal the explicitly selected candidate. These graph checks
do not establish numerical correctness for prefill or Q>1. The serving call
site remains unchanged; full-indexer and end-to-end validation are still pending.

### Verified checkpoint (2026-10-21)

The capture at `3249725411ececc90b2913b9e60728672bba18fd` and CPU analysis
use the runtime and sampling recipe above.
For independent Q1/B32 requests, XProf device medians are:

| Implementation | Full module (us) | Pallas event (us) |
|---|---:|---:|
| Frozen JAX reference | 25.7630 | N/A |
| Previous pipeline | 30.1275 | 24.6055 |
| Q1 batch candidate | 14.2075 | 6.3740 |

The module takes 52.84% less time than the previous pipeline (2.12x speedup)
and 44.85% less than the reference. Candidate module p10/p90 are
14.0969/14.4201 us; the previous pipeline has 29.9258/30.5310 us. Each group
has 100 complete device samples. All four decode cases have validated event
counts and attribution: 2,000 module samples and 1,600 Pallas samples. Q6/B1
and Q6/B32 reuse their previous executables. Shared-quad speed is not measured.

Automatic dispatch at `66949064d894638271ce096e831b3bbbe2c324eb` has the same
traced Q1/B32 graph as this capture. Accuracy validation passed all
14 candidate cases and all nine automatic B32 cases, with zero differing
cache bytes. This is module-level evidence on one TPU device, not a serving
or multi-device performance result. Prefill/Q>1 numerical gates and serving
integration remain open.

## Full Q/B coverage

The fixed dimensions remain K=4, D=128, raw page=128, and raw dtype=BF16.
`shape_matrix.py` declares 301 combinations:

- Decode/verify: every Q from 1 through 32, crossed with B=1/2/4/8/16/32/64
  (224 cases). T=B*Q matches fixed-width decode/verify token bucketing.
- Prefill: Q=1/2/4/.../8192, crossed with the same B values, subject to
  B*Q<=8192 (77 cases). T=max(64,B*Q). This is standalone coverage;
  a model's serving configuration can impose a larger minimum bucket.

Pass `--shape-matrix --matrix-section decode` or `prefill` to `benchmark.py`.
These runs compare only the frozen JAX reference and the current default fused
path. Each uses the same 10-warmup, 100-sample XProf recipe described above.
The older targeted benchmark retains its historical small-batch token padding;
use the full matrix for exact T=B*Q decode comparisons.

For Falcon, pass these arguments after `falcon_task.py --benchmark-args`.
Use `--analysis-workers 4` on the CPU analysis task. Collect its compact timing
result. Internal capture locations and task identifiers belong in private run
records and must be supplied at launch, never committed to this repository.

After the performance checkpoint, `--mode matrix-accuracy` runs 414 exact-byte
checks: every measured pair plus ragged lengths, padding, inactive requests,
page boundaries, shared destination quads, and zero/tiny/nonfinite values.
Coverage is declared by the script; it is not a claim that these checks passed.

Export public tables with:

```sh
python benchmark/tpu/kpool/report_matrix.py \
  --input PREFILL_COMPACT_JSON DECODE_COMPACT_JSON \
  --accuracy ACCURACY_JSON --output OUTPUT_DIRECTORY
```

The report includes full module p10/p50/p90, Pallas p50, sample counts, source
revisions and per-shape accuracy. Without an accuracy file, it labels every row
`not_run`. It rejects missing or duplicate shapes, wrong token buckets,
incomplete samples, and mismatched sources or runtimes. It exports no internal
task identifiers or storage paths.
