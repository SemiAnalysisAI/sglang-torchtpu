# Kpool device timing

`benchmark.py` captures XProf events. It does not measure host latency or
produce speedup claims. Numerical accuracy and serving tests are separate
validation gates.

The default matrix uses 14,849 raw-cache pages, donated pooled-cache output,
10 warmup calls, and 100 recorded calls per variant. It covers decode Q1/Q6 at
B1/B32 and prefill T8192. Each case alternates variant order between rounds.
All variants use one device and one runtime in the same process.

Use `falcon_task.py --mode device-profile` to render the capture task from a
pushed full Git SHA. The task first tests profiler wiring with the pinned
runtime in `requirements.txt`. On the specific profiler ABI 80/104 failure,
it installs the matched JAX/JAXlib 0.11.2 pair and runs every comparison on that
runtime with libtpu 0.0.50. The captured manifest and resolved requirements
record which runtime ran. Custom-call runtime instrumentation stays disabled.

Use `falcon_task.py --mode profile-analysis --analysis-input PATH` for the CPU
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
comparison. The full module span excludes gaps between separate calls.

The prior `perf_counter + block_until_ready` comparisons measured host dispatch
and synchronization. Those numbers do not establish device speedups or
regressions and must not be used for kernel optimization decisions.
