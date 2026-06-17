"""Opt-in persistent compiled-artifact cache reuse with version-keyed invalidation (TPU).

When ``SGLANG_TPU_PERSIST_COMPILE_CACHE=<root>`` is set, the server points BOTH compile caches at a
stable ``<root>`` so compiled artifacts persist across restarts:

* the torch.compile inductor/triton caches (``<root>/inductor_cache``, ``<root>/triton_cache``), and
* the torch_tpu XLA-program cache (Tier-3 on-disk at ``<root>/torch_tpu_cache`` + a deterministic
  Tier-2 shared-memory name), so the compiled XLA programs are reused too (not just the FX lowering).

Reuse is guarded by a version-keyed manifest: if any of the torch / libtpu / torch_tpu versions, the
model/draft identities, the spec config, TP size, page size, the libtpu init flags, or the cache
identity differ from the cached manifest, every managed cache sub-dir is cleared and rebuilt (so a
stale artifact from a different toolchain/config is never served).

The env vars are set at the end of ``TpuSRTPlatform.apply_server_args_defaults`` — after the TPU
effective defaults are applied (so the manifest records the EFFECTIVE config: ``page_size`` forced to
1 for spec decode, ``dtype`` resolved, attention backend chosen) and before the TPU runtime
initializes its Tier-2/Tier-3 caches (so libtpu reads the cache env vars). The manifest identity is
built from the parsed ``server_args`` object, not from scraping ``sys.argv``. Default OFF — when the
env is unset, nothing is touched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from typing import Optional

logger = logging.getLogger(__name__)

# Opt-in root directory for the persistent compile cache. Unset/empty => disabled.
PERSIST_CACHE_ENV = "SGLANG_TPU_PERSIST_COMPILE_CACHE"

# Bump when the managed cache LAYOUT/policy changes (sub-dir names, Tier-2/3 wiring), so an old
# cache built by a different layout is invalidated even if every other key matches.
_CACHE_FORMAT_VERSION = 2

_MANIFEST_NAME = "sglang_tpu_cache_manifest.json"
# Cache sub-dirs this module owns; only these are cleared on a version mismatch (never the whole
# root, so an operator pointing at a shared dir cannot lose unrelated data).
_MANAGED_SUBDIRS = ("inductor_cache", "triton_cache", "torch_tpu_cache")

# torch_tpu cache env vars (read by libtpu at TPU-runtime init).
_TIER3_ROOT_ENV = "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT"
_TIER3_BACKUP_ENV = "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK"
_TIER2_NAME_ENV = "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"
# Tier-2 lives in shared memory; clear only the deterministic per-cache name, never the parent.
_TIER2_SHM_PARENT = "/dev/shm/torch_tpu_cache"


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return "absent"
    except Exception:
        return "unknown"


def collect_version_keys(server_args, tp_size: int) -> dict:
    """Build the FULL version/config keys that must match for a cache to be reusable, from the parsed
    and defaults-applied ``server_args`` (EFFECTIVE values — e.g. page_size already forced to 1 for
    spec decode, dtype already resolved to bfloat16). A change in ANY key can change the compiled
    artifacts (TPU programs / inductor lowering), so any difference invalidates the persistent cache.
    Includes the TPU precompile grid (tpu_compile_bs/tokens), the libtpu flags, and a cache-format
    version so a managed-layout change also invalidates."""
    import torch

    sa = server_args
    return {
        "cache_format_version": _CACHE_FORMAT_VERSION,
        "torch": getattr(torch, "__version__", "unknown"),
        "torch_tpu": _pkg_version("torch_tpu"),
        "libtpu": _pkg_version("libtpu"),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "model": getattr(sa, "model_path", None),
        "draft_model": getattr(sa, "speculative_draft_model_path", None),
        "spec_algo": getattr(sa, "speculative_algorithm", None),
        "spec_num_steps": getattr(sa, "speculative_num_steps", None),
        "spec_eagle_topk": getattr(sa, "speculative_eagle_topk", None),
        "spec_num_draft_tokens": getattr(sa, "speculative_num_draft_tokens", None),
        "tp_size": tp_size,
        "page_size": getattr(sa, "page_size", None),  # effective (defaults applied)
        "dtype": str(getattr(sa, "dtype", None)),
        "attention_backend": getattr(sa, "attention_backend", None),
        # The TPU precompile grid — these buckets define which shapes are compiled.
        "tpu_compile_bs": list(getattr(sa, "tpu_compile_bs", []) or []),
        "tpu_compile_tokens": list(getattr(sa, "tpu_compile_tokens", []) or []),
    }


def _tier2_name(root: str) -> str:
    """Deterministic Tier-2 (shared-memory) cache name, isolated per cache root."""
    return "sglang_" + hashlib.sha256(root.encode()).hexdigest()[:16]


def _first_diff(old: dict, new: dict) -> str:
    diffs = [k for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
    if not diffs:
        return "manifest equal but reuse not taken"
    k = diffs[0]
    return f"version-key mismatch: {k!r} cached={old.get(k)!r} current={new.get(k)!r}"


def _clear_managed(root: str) -> None:
    for sub in _MANAGED_SUBDIRS:
        shutil.rmtree(os.path.join(root, sub), ignore_errors=True)
    # Also clear the deterministic Tier-2 shared-memory dir for this root (never the parent).
    shutil.rmtree(
        os.path.join(_TIER2_SHM_PARENT, _tier2_name(root)), ignore_errors=True
    )


def validate_or_invalidate(root: str, version_keys: dict) -> tuple[bool, str]:
    """Validate the manifest at ``root`` against ``version_keys``. Returns ``(reused, reason)``. On a
    full match the caches are kept; on a missing manifest or ANY key mismatch every managed cache
    sub-dir (inductor/triton/torch_tpu) and the per-root Tier-2 shm dir are cleared and a fresh
    manifest is written, so the next compile rebuilds cleanly."""
    os.makedirs(root, exist_ok=True)
    mpath = os.path.join(root, _MANIFEST_NAME)
    existing: Optional[dict] = None
    if os.path.exists(mpath):
        try:
            existing = json.load(open(mpath)).get("keys")
        except Exception:
            existing = None

    want = json.loads(
        json.dumps(version_keys)
    )  # normalise tuples->lists for comparison
    if existing is not None and existing == want:
        return True, "manifest match: reusing persistent compile cache"

    reason = (
        "no manifest (fresh cache)" if existing is None else _first_diff(existing, want)
    )
    _clear_managed(root)
    with open(mpath, "w") as fh:
        json.dump({"keys": want}, fh, indent=2)
    return False, reason


def _set_cache_env(root: str) -> dict:
    """Point all managed caches at ``root`` and return the dirs set."""
    inductor = os.path.join(root, "inductor_cache")
    triton = os.path.join(root, "triton_cache")
    torch_tpu = os.path.join(root, "torch_tpu_cache")
    for d in (inductor, triton, torch_tpu):
        os.makedirs(d, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor
    os.environ["TRITON_CACHE_DIR"] = triton
    # torch_tpu Tier-3 (on-disk, persists across reboots) + Tier-2 (shm, deterministic name).
    os.environ[_TIER3_ROOT_ENV] = torch_tpu
    os.environ[_TIER3_BACKUP_ENV] = "1"
    os.environ[_TIER2_NAME_ENV] = _tier2_name(root)
    return {"inductor": inductor, "triton": triton, "torch_tpu": torch_tpu}


def setup_persistent_compile_cache(server_args, tp_size: int) -> Optional[dict]:
    """If the opt-in env is set, validate/invalidate the manifest and point ALL managed caches
    (inductor/triton + torch_tpu Tier-2/Tier-3) at the stable root. MUST be called AFTER the TPU
    effective defaults are applied (so the manifest records effective page_size/dtype/etc.) and
    BEFORE the TPU runtime initializes — i.e. at the end of ``apply_server_args_defaults``. Returns a
    status dict (or None when disabled). Best effort: a failure never blocks startup (falls back to
    default non-persistent behavior)."""
    root = os.environ.get(PERSIST_CACHE_ENV, "")
    if not root:
        return None
    try:
        keys = collect_version_keys(server_args, tp_size)
        reused, reason = validate_or_invalidate(root, keys)
        dirs = _set_cache_env(root)
        logger.info(
            "[persist-cache] %s root=%s (%s); torch_tpu Tier-3 root=%s, Tier-2 name=%s",
            "REUSE" if reused else "REBUILD",
            root,
            reason,
            dirs["torch_tpu"],
            os.environ[_TIER2_NAME_ENV],
        )
        return {
            "enabled": True,
            "reused": reused,
            "reason": reason,
            "root": root,
            **dirs,
        }
    except Exception as exc:  # never block startup for a cache setup issue
        logger.warning("[persist-cache] setup failed (%s); using default cache", exc)
        return {"enabled": True, "reused": False, "reason": f"error: {exc}"}
