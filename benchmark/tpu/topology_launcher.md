# Coordinate-ordered multi-host launches

Use the opt-in topology launcher when saving compilation artifacts for reuse
on another allocation with the same TPU topology and software. It obtains each
host's coordinates through TorchTPU's `local_device_coords()` API, then uses one
coordinate-sorted host order for node ranks, the rendezvous leader, and every
SliceBuilder endpoint. Kubernetes pod indices identify participants during
discovery; they do not determine model shard ownership.

Run this on **every** host of a TPU7x 2x2x2 slice, using the same ordered host
list and a fresh shared rendezvous ID. Set `HOST_INDEX` to this host's position
in that input list, regardless of its physical placement:

```bash
uv run python -m sglang.srt.hardware_backend.tpu.topology_launcher \
  --hosts host-a,host-b --host-index "$HOST_INDEX" \
  --topology 2,2,2,2 --rendezvous-id experiment-001 \
  --manifest /tmp/experiment-001-launch.json \
  -- --model-path /models/model --tp 16 --ep 16
```

The launcher supplies `--device`, `--nnodes`, `--node-rank`, and
`--dist-init-addr`. Do not supply conflicting distributed flags. Allow TCP
traffic between hosts on the bootstrap port (29590), master port (29500),
and SliceBuilder ports (8070–8077); each has a launcher override.

Coordinate discovery runs in a temporary full-slice torchrun world because
the native API initializes PJRT and GKE requires SliceBuilder setup first.
This probe uses CPU Gloo for synchronization, loads no model or artifacts, and
must release every device before model workers start. It adds a bootstrap
launch to startup, not work to the inference hot path.
Missing, duplicate, out-of-bounds, or incomplete coordinates fail the launch;
there is no silent fallback to pod order. Existing launch commands are unchanged.

## Cache compatibility is still required

Coordinate ordering does **not** make arbitrary executable bytes portable:

- Use it on both the producer and consumer. Old artifacts compiled with another
  mapping may need recompilation.
- Keep the full ordered logical-rank-to-device assignment in the cache key, as
  well as the existing model, shapes, code, runtime, and topology keys. The
  launcher only normalizes host order; it preserves native local-device order.
- An assignment mismatch must cause a cache miss or rejection before execution.
  Do not remove checks or relabel serialized artifacts to force a hit.
- Host IPs may differ between allocations. Compare physical device assignments,
  not IP addresses, for executable compatibility.

The supported scope is a single complete slice of uniformly tiled hosts with
DNS names or IPv4 addresses. The default host tile is 2x2x1 chips with eight
logical devices. Other layouts require matching explicit `--chips-per-host`
and `--devices-per-host`. Ray and partial/multi-slice allocations are not
supported by this launcher.

The HTTP-serving host follows normalized node rank zero, which may be the
orchestrator's worker pod. A controller with a fixed leader Service must adapt
its routing before using server mode; changing ranks alone does not update
Kubernetes Services, Ray placement, or leader-local IPC. The manual torchrun
test does not validate those serving integrations.

## TPU regression test

`test/registered/e2e/hardware_backend/tpu/test_topology_artifact_transfer.py` contains a real TPU
`unittest.TestCase`. Run it simultaneously on every host of a dedicated slice:

```bash
JAX_PLATFORMS=cpu uv run python test/registered/e2e/hardware_backend/tpu/test_topology_artifact_transfer.py -v
```

Set `TPU_WORKER_HOSTNAMES` to the same input host list on every host,
`TPU_WORKER_ID` to its index in that list, and `TORCH_TPU_TOPOLOGY` to the full
device topology (for example `2,2,2,2`). In addition to the launcher ports,
allow port 29591 for test coordination. Missing multi-host configuration fails
the test rather than silently skipping TPU validation.

The test compiles on TPU, transfers serialized artifacts through a CPU store,
and restores in fresh TPU processes with the input host list reversed. Exact
collective outputs and zero region recompilations are required. A deliberately
noncanonical launch must reject the artifacts before loading. This is wired
into the multi-host feature-validation workflow as `topology-artifact-transfer`;
it cannot run in the single-host TPU PR gate. CPU tests separately cover the
host-ordering algorithm, but cannot establish executable correctness.

## Manual cross-allocation regression

The component test uses integer local arithmetic, an ordered all-gather, and
an all-reduce with independent exact expected outputs. It is not a Tier5
accuracy test. Replace the server arguments above with:

```bash
--mode torchrun -- \
  test/registered/e2e/hardware_backend/tpu/test_topology_artifact_transfer.py --worker \
  --action populate --artifacts /tmp/artifacts --output /tmp/source-results
```

After both source hosts finish, collect their `rank-*.bin` and `rank-*.json`
files. Copy the merged directory to both destination hosts and verify SHA256
hashes after transfer. Run the destination launcher with a fresh rendezvous ID,
its own host list, and `--action restore`. Check all ranks' result files: exact
outputs must pass and `cache_requests` must be zero. Restore also intercepts
the native direct-compilation entry point to detect accidental recompilation.

Repeat with the destination input host list reversed and its host indices
adjusted accordingly. Coordinate ordering must still produce the same
assignment and restore the same bytes. As a negative control, launch directly
with torchrun using a deliberately different physical assignment and
`--action expect-reject`: the test's compatibility manifest must reject the
artifact before execution. This manifest is test machinery, not a new
production disk-cache format.
