# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TPU-specific helpers for the Ray scheduler launch path.

These live in ``hardware_backend/tpu`` (rather than ``ray/engine.py``) so the
generic Ray engine stays free of TPU cluster-discovery specifics.
"""

from __future__ import annotations

import logging
import os

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logger = logging.getLogger(__name__)


def detect_tpu_nnodes() -> int:
    """Count alive Ray nodes that advertise the ``TPU`` resource.

    Used to auto-derive ``nnodes`` for TPU ray runs so the operator doesn't have
    to pass ``--nnodes`` -- the placement group and rank math then follow the live
    cluster. Returns 0 if nothing is detectable (caller keeps server_args.nnodes).
    """
    try:
        return sum(
            1
            for node in ray.nodes()
            if node.get("Alive") and float(node.get("Resources", {}).get("TPU", 0)) > 0
        )
    except Exception as e:  # pragma: no cover - defensive; fall back to server_args
        logger.warning("Failed to count TPU nodes from Ray: %s", e)
        return 0


def validate_full_host_tpu_allocation(nnodes: int, tpus_per_node: int) -> None:
    """Require topology probes to reserve every physical TPU on each host.

    Ray rewrites TPU visibility and host bounds whenever a task requests only a
    subset of a host's TPU resources. That makes physical coordinate discovery
    impossible, so multihost Ray currently supports only full-slice placement.
    """
    physical_counts = [
        int(node.get("Resources", {}).get("TPU", 0))
        for node in ray.nodes()
        if node.get("Alive") and float(node.get("Resources", {}).get("TPU", 0)) > 0
    ]
    if len(physical_counts) != nnodes or any(
        count != tpus_per_node for count in physical_counts
    ):
        raise RuntimeError(
            "Ray TPU topology discovery requires full-slice placement: expected "
            f"exactly {nnodes} TPU node(s) with {tpus_per_node} TPU(s) requested "
            f"per node, but the live physical TPU counts are {physical_counts}. "
            "Partial-host and partial-slice Ray placement would hide physical "
            "device coordinates."
        )


@ray.remote(num_cpus=0, num_gpus=0)
def _get_topological_rank_and_address():
    """Return this node's coordinate-derived rank and IP without retaining PJRT."""
    from sglang.srt.hardware_backend.tpu.topology import (
        probe_topological_node_rank,
    )

    result = probe_topological_node_rank()
    return int(result["node_rank"]), ray.util.get_node_ip_address()


def _resolve_bundle_order_from_probes(
    probes: list[tuple[int, str]], engine_ip: str
) -> tuple[list[int], str, str]:
    """Resolve Ray bundles and hosts from coordinate-derived node ranks.

    ``probes[bundle_idx]`` is ``(topological_node_rank, ip)``. The returned host
    list and bundle list are both ordered by physical topology, so scheduler
    rank ranges, slicebuilder endpoints, and ``MASTER_ADDR`` agree.

    The Engine must be on topological node 0 because SGLang co-locates its rank-0
    scheduler and tokenizer there.

    Pure function (no Ray calls) so the ordering + validation can be unit-tested.
    """
    nnodes = len(probes)

    rank_to_bundle: dict[int, int] = {}
    rank_to_ip: dict[int, str] = {}
    for bundle_idx, (node_rank, ip) in enumerate(probes):
        node_rank = int(node_rank)
        if node_rank in rank_to_bundle:
            raise RuntimeError(
                f"Ray TPU multihost: duplicate topological node rank {node_rank} "
                f"across bundles {rank_to_bundle[node_rank]} and {bundle_idx} "
                f"({probes})."
            )
        rank_to_bundle[node_rank] = bundle_idx
        rank_to_ip[node_rank] = ip

    if sorted(rank_to_bundle) != list(range(nnodes)):
        raise RuntimeError(
            "Ray TPU multihost: topological node ranks are not contiguous 0.."
            f"{nnodes - 1}; got {sorted(rank_to_bundle)} ({probes})."
        )

    bundle_for_node = [rank_to_bundle[k] for k in range(nnodes)]
    ordered_ips = [rank_to_ip[k] for k in range(nnodes)]
    rank0_node_ip = ordered_ips[0]

    if engine_ip != rank0_node_ip:
        raise RuntimeError(
            f"Ray TPU multihost: Engine runs on {engine_ip} but topological TPU "
            f"node 0 is on {rank0_node_ip}. Start the Ray head / Engine on "
            "topological node 0 so its tokenizer and rank-0 scheduler are "
            "co-located."
        )

    return bundle_for_node, rank0_node_ip, ",".join(ordered_ips)


def order_tpu_bundles_by_topology(
    placement_group, nnodes: int, tpus_per_node: int
) -> tuple[list[int], str, str]:
    """Probe the live placement group and order bundles by TPU coordinates.

    Each task reserves the bundle's full TPU allocation. Reserving only one TPU
    makes Ray narrow ``TPU_VISIBLE_CHIPS`` and the TPU host bounds, which hides
    the physical multihost coordinates that this probe needs. Coordinate
    discovery then runs in a short-lived child so PJRT is released before the
    scheduler actors start.
    """
    engine_ip = ray.util.get_node_ip_address()
    probes = ray.get(
        [
            _get_topological_rank_and_address.options(
                resources={"TPU": tpus_per_node},
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group, placement_group_bundle_index=i
                ),
            ).remote()
            for i in range(nnodes)
        ]
    )
    return _resolve_bundle_order_from_probes(probes, engine_ip)


# Config env namespaces the launcher (benchmark/tpu/launch_server.sh) exports at
# runtime and that a scheduler actor needs, but Ray drops: the actor only inherits
# the raylet's env (the container env at `ray start`), which predates those exports
# -- and worker nodes never run launch_server.sh at all. We forward these from the
# driver's os.environ via runtime_env (see _forwarded_driver_env). Allowlist, not
# denylist: missing an entry just leaves it dropped (status quo, safe), whereas
# forwarding a per-chip/per-node var would clobber Ray's per-actor value.
_FORWARD_ENV_PREFIXES = ("SGLANG_", "TORCH_", "HF_", "TRANSFORMERS_")
_FORWARD_ENV_NAMES = frozenset(
    {
        "JAX_PLATFORMS",
        "LIBTPU_INIT_ARGS",
        "XLA_FLAGS",
        "TPU_PREMAPPED_BUFFER_SIZE",
    }
)
# Never forward, even when a prefix matches: these are per-chip/per-node (Ray sets
# them per actor -- forwarding the driver's copy makes every actor grab the same
# chip and crash) or are the physically-ordered slice env injected separately
# below from server_args (which must win over any inherited copy).
_NEVER_FORWARD_ENV = frozenset(
    {
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
        "TPU_WORKER_HOSTNAMES",
        "TPU_VISIBLE_CHIPS",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "TPU_HOST_BOUNDS",
    }
)


def _forwarded_driver_env(environ=None) -> dict:
    """Pick the driver's config env vars that should be forwarded to scheduler actors.

    Returns ``{name: value}`` for every var in ``environ`` (default ``os.environ``)
    whose name is in ``_FORWARD_ENV_NAMES`` or matches a ``_FORWARD_ENV_PREFIXES``
    prefix, minus ``_NEVER_FORWARD_ENV``. ``environ`` is injectable for unit tests.
    """
    if environ is None:
        environ = os.environ
    forwarded = {}
    for name, value in environ.items():
        if name in _NEVER_FORWARD_ENV:
            continue
        if name in _FORWARD_ENV_NAMES or name.startswith(_FORWARD_ENV_PREFIXES):
            forwarded[name] = value
    return forwarded


# Case-insensitive substrings marking an env var whose *value* is a secret. Such
# vars are still forwarded to actors (e.g. HF_TOKEN, which the actor needs to auth
# to the Hugging Face Hub), but their values must be masked whenever we log the
# forwarded set -- otherwise a token lands in plaintext in the driver log.
_SECRET_ENV_SUBSTRINGS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "KEY", "CREDENTIAL")


def _is_secret_env(name: str) -> bool:
    upper = name.upper()
    return any(sub in upper for sub in _SECRET_ENV_SUBSTRINGS)


def redact_env_for_log(env: dict) -> str:
    """Format ``{name: value}`` as a ``k=v`` list for logging, masking secret values.

    A var whose name looks like a secret (see ``_SECRET_ENV_SUBSTRINGS``) is shown as
    ``NAME=***`` so operators can still see it is being forwarded without leaking the
    value. Keys are sorted for stable, greppable output.
    """
    return ", ".join(
        f"{name}={'***' if _is_secret_env(name) else value}"
        for name, value in sorted(env.items())
    )


def tpu_actor_runtime_env(server_args) -> dict:
    """Ray ``.options()`` kwargs injecting the env a scheduler actor needs.

    Ray does NOT propagate the driver's ``os.environ`` to actors (they inherit only
    the raylet's env), so ``runtime_env`` is the sole channel. Two sources merge
    into one ``env_vars`` dict:

    1. The launcher-set config the driver has but the actor lost (``LIBTPU_INIT_ARGS``,
       compile-cache, ``SGLANG_*`` / ``TORCH_*`` toggles), via ``_forwarded_driver_env``.
    2. The launcher-*computed* TPU slice env stashed on ``server_args``:
       ``TPU_WORKER_HOSTNAMES`` (multihost host list) or the single-host
       ``TORCH_TPU_SLICEBUILDER_ADDRESSES`` / ``TORCH_TPU_TOPOLOGY``, read by the
       actor's TPU bootstrap (``_prepare_tpu_env``).

    Source 2 always wins over any inherited copy of the same var (it is the
    physically-ordered / single-host-override value). Returns an empty dict when
    there is nothing to inject, so actors fall back to their own pod env.
    """
    env_vars = _forwarded_driver_env()
    # Computed slice env overrides any forwarded/inherited copy. Inject an empty
    # string too (single-host on a multi-host slice), so the actor overrides the
    # slice-wide value inherited from `ray start` rather than falling back to it --
    # only None (GPU / discovery failed) means "don't inject, use the pod env".
    if server_args.tpu_worker_hostnames is not None:
        env_vars["TPU_WORKER_HOSTNAMES"] = server_args.tpu_worker_hostnames
    if server_args.torch_tpu_slicebuilder_addresses:
        env_vars["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = (
            server_args.torch_tpu_slicebuilder_addresses
        )
    if server_args.torch_tpu_topology:
        env_vars["TORCH_TPU_TOPOLOGY"] = server_args.torch_tpu_topology
    if not env_vars:
        return {}
    return {"runtime_env": {"env_vars": env_vars}}
