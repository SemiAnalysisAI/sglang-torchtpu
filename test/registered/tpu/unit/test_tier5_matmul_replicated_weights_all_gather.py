# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for matmul_replicated_weights_all_gather on a real 2x2x2 (16 dev).

Validates the fused broadcast kernel, the topology-aware hypercube gather (``TestTopoGather`` +
``TestAutoTopoDispatch``), the large-T reorder path, and the size-dispatch entry point all equal
``all_gather(x) @ W`` within bf16 tolerance, plus reuse-safety of the fused kernel's setup barrier
(calling it repeatedly with the same collective_id). Also covers the PURE-PYTHON topo
routing-schedule (``TestBuildIciSchedule`` / ``TestSimulateAllGather`` / ``TestAntipodeCoreSplit``
-- these need no TPU; run off-TPU with ``JAX_PLATFORMS=cpu``, jax is only imported transitively).
"""

import functools
import importlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=1200, suite="stage-a-test-tpu-unit")
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.common import (
    make_hypercube_tp_mesh,
    make_ring_tp_mesh,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.matmul_replicated_weights_all_gather import (
    _matmul_topo as _sched,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.matmul_replicated_weights_all_gather.api import (
    matmul_all_gather,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.matmul_replicated_weights_all_gather._matmul_broadcast import (
    _pick_bn,
    _matmul_all_gather_broadcast,
)

# The dispatch's MODULE (for the topo/broadcast route spies). Imported via importlib because the
# package attribute ``matmul_all_gather`` is the re-exported FUNCTION, not this ``api`` submodule.
_mag = importlib.import_module(
    "sglang.srt.hardware_backend.tpu.kernels.tier5_sp.matmul_replicated_weights_all_gather.api"
)


def _is_real_2x2x2():
    devs = jax.devices()
    return len(devs) == 16 and all("TPU" in d.device_kind for d in devs)


def _vs_reference(
    kernel,
    m,
    k,
    n,
    n_calls=1,
    axis_name="tp",
    w_transposed=False,
    x_dtype=None,
    out_dtype=None,
    w_dtype=None,
):
    """``kernel(x, w, mesh, axis)`` (replicated W) == all_gather(x) @ W, multi-host.

    Runs ``n_calls`` times reusing the compiled executable to also exercise the
    collective_id-reuse path the setup barrier must survive. ``w_transposed`` -> ``w`` is
    ``[N, K]`` (ColumnParallelLinear ``[out, in]`` native layout); the reference contracts K.
    ``x_dtype`` generates ``x`` in that dtype (e.g. fp8, the online-FP8 unpooler input) and
    passes it to the kernel NATIVE; ``out_dtype`` is the kernel's output dtype -- the
    reference matches by widening the gathered ``x`` to ``w``'s dtype, as ``reorder_matmul`` does.
    ``w_dtype`` (default bf16) sets the weight dtype -- e.g. fp32 for the all-fp32 topo path.
    """
    if not _is_real_2x2x2():
        pytest.skip("needs a real 16-core tpu7x 2x2x2 slice (multi-host)")
    mesh = make_hypercube_tp_mesh(axis_name)
    xs = NamedSharding(mesh, P(axis_name, None))
    ws = NamedSharding(mesh, P(None, None))  # replicated weight
    w_shape = (n, k) if w_transposed else (k, n)

    @functools.partial(jax.jit, out_shardings=(xs, ws))
    def _gen():
        k1, k2 = jax.random.split(jax.random.key(0), 2)
        return (
            jax.random.normal(k1, (m, k), jnp.bfloat16).astype(x_dtype or jnp.bfloat16),
            jax.random.normal(k2, w_shape, jnp.bfloat16).astype(
                w_dtype or jnp.bfloat16
            ),
        )

    x, w = _gen()
    jax.block_until_ready((x, w))

    def _ref(xx, ww):
        # Match reorder_matmul: gather x natively, then widen to w's dtype at the matmul.
        g = jax.lax.all_gather(xx, axis_name, axis=0, tiled=True).astype(ww.dtype)
        return jnp.dot(g, ww.T) if w_transposed else jnp.dot(g, ww)

    ref = jax.jit(
        jax.shard_map(
            _ref,
            mesh=mesh,
            in_specs=(P(axis_name, None), P(None, None)),
            out_specs=P(None, None),
            check_vma=False,
        )
    )(x, w)
    ref_g = np.asarray(multihost_utils.process_allgather(ref, tiled=True), np.float32)

    fn = functools.partial(
        kernel,
        mesh=mesh,
        axis_name=axis_name,
        w_transposed=w_transposed,
        out_dtype=out_dtype,
    )
    for _ in range(n_calls):
        out = fn(x, w)
        if out_dtype is not None:
            assert out.dtype == jnp.dtype(out_dtype), (out.dtype, out_dtype)
        out_g = np.asarray(
            multihost_utils.process_allgather(out, tiled=True), np.float32
        )
        np.testing.assert_allclose(out_g, ref_g, rtol=0.01, atol=0.01)


def test_fp8_input_bf16_output():
    # Online-FP8 serve: the unpooler input x is fp8 (the fp8 RS-output comm) and the weight is
    # the [N, K] CPL layout. The kernels keep x fp8 on the wire for the gather/matmul and emit
    # bf16 via out_dtype (they otherwise follow x's dtype). Checks value correctness AND the
    # output dtype across the fused (small-T) kernel and the dispatched paths, up-proj K4096/N8192.
    f8, bf16 = jnp.float8_e4m3fn, jnp.bfloat16
    # fbcast (small-T, with the 16-row HBM-aligned local shard)
    _vs_reference(
        _matmul_all_gather_broadcast,
        m=256,
        k=4096,
        n=8192,
        w_transposed=True,
        x_dtype=f8,
        out_dtype=bf16,
    )
    # dispatch on the hypercube mesh (_vs_reference): naive (T=64, m_local<16), then topo for
    # T>=256 (full parity -- fp8 x + [N, K] weight -> bf16 output).
    for m in (64, 256, 2048):
        _vs_reference(
            matmul_all_gather,
            m=m,
            k=4096,
            n=8192,
            w_transposed=True,
            x_dtype=f8,
            out_dtype=bf16,
        )


def test_dispatch_broadcast_x_scale(monkeypatch):
    # Raw fp8 x + per-token online-FP8 scale through the AUTO dispatch at small T -> the broadcast
    # kernel, which now dequants x ITSELF (util.dequant_activation) rather than a dispatcher-level
    # _dequant. Must equal (x_fp8 * scale) @ W, and the broadcast kernel must be the one that runs.
    if not _is_real_2x2x2():
        pytest.skip("needs a real 16-core tpu7x 2x2x2 slice (multi-host)")
    calls = {"n": 0}
    real = _mag._matmul_all_gather_broadcast

    def _spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(_mag, "_matmul_all_gather_broadcast", _spy)

    mesh = make_hypercube_tp_mesh("tp")
    m, k, n = 256, 8192, 4096  # local T=16, output 2MB -> broadcast
    xs = NamedSharding(mesh, P("tp", None))
    ss = NamedSharding(mesh, P("tp"))  # per-token scale, token-sharded like x's rows
    ws = NamedSharding(mesh, P(None, None))

    @functools.partial(jax.jit, out_shardings=(xs, ss, ws))
    def _gen():
        k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
        xf = jax.random.normal(k1, (m, k), jnp.bfloat16).astype(jnp.float8_e4m3fn)
        scale = jax.random.uniform(k3, (m,), jnp.float32, minval=0.1, maxval=0.6)
        w = jax.random.normal(k2, (n, k), jnp.bfloat16)  # [N, K] (w_transposed)
        return xf, scale, w

    x, scale, w = _gen()
    jax.block_until_ready((x, scale, w))

    def _ref(xx, sc, ww):
        xd = (xx.astype(jnp.float32) * sc[:, None]).astype(ww.dtype)
        g = jax.lax.all_gather(xd, "tp", axis=0, tiled=True)
        return jnp.dot(g, ww.T)

    ref = jax.jit(
        jax.shard_map(
            _ref,
            mesh=mesh,
            in_specs=(P("tp", None), P("tp"), P(None, None)),
            out_specs=P(None, None),
            check_vma=False,
        )
    )(x, scale, w)
    out = matmul_all_gather(
        x,
        w,
        mesh=mesh,
        axis_name="tp",
        w_transposed=True,
        out_dtype=jnp.bfloat16,
        x_scale=scale,
    )
    ref_g = np.asarray(multihost_utils.process_allgather(ref, tiled=True), np.float32)
    out_g = np.asarray(multihost_utils.process_allgather(out, tiled=True), np.float32)
    np.testing.assert_allclose(out_g, ref_g, rtol=0.02, atol=0.02)
    assert calls["n"] >= 1, "small-T x_scale did not route to the broadcast kernel"


def test_fused_decode_shape():
    # Small-T shape with a 16-row local shard, required by packed bf16 HBM DMA.
    _vs_reference(_matmul_all_gather_broadcast, m=256, k=4096, n=8192)


def test_fused_reuse_safe():
    # Reuse the compiled kernel 8x -- the setup double-barrier must survive
    # collective_id reuse (no cross-call corruption).
    _vs_reference(_matmul_all_gather_broadcast, m=256, k=4096, n=8192, n_calls=8)


def test_fused_down_projection():
    # Down-projection decode (K=8192, N=4096): the fused (small-T) path on the
    # other shape we ship.
    _vs_reference(_matmul_all_gather_broadcast, m=256, k=8192, n=4096)


def test_dispatch_decode():
    # T=128 has only eight rows per TP16 rank and must use the XLA path.
    _vs_reference(matmul_all_gather, m=128, k=4096, n=8192)


def test_dispatch_large():
    # Dispatch picks the reorder + XLA-gather (large-T) path for T=4096.
    _vs_reference(matmul_all_gather, m=4096, k=4096, n=8192)


def test_dispatch_tiny_no_crash():
    # T=32 -> m_local=2 (<16): dispatch must NOT pick fused (Mosaic crashes there);
    # it should fall back to naive/ring and stay correct. Regression for the
    # m_local>=16 guard.
    _vs_reference(matmul_all_gather, m=32, k=4096, n=8192)
    _vs_reference(matmul_all_gather, m=64, k=4096, n=8192)


def test_dispatch_down_projection_large_k():
    # Down-projection with large K (K=8192): dispatch -> the pure-XLA reorder, which
    # tiles K itself (a Pallas matmul with bk=k would OOM VMEM here). ~3x at T=4096.
    _vs_reference(matmul_all_gather, m=2048, k=8192, n=4096)


def test_dispatch_multi_axis_token_gather(monkeypatch):
    # The token dim sharded over a GENUINE 2-live-axis product (dp=2 x tp=8), not a single
    # size-16 axis. Every path must gather the whole product natively via the mixed-radix
    # combined index and equal all_gather(x) @ W: the small-out BROADCAST and the mid-T TOPO
    # gather (both hand-rolled Pallas collectives over an axis TUPLE), plus the XLA reorder
    # (large T) and naive (tiny T). The 16 devices are reshaped into (dp=2, tp=8) in the RING
    # (gray-code) order -- deliberately NON-hypercube -- so this ALSO proves topo figures out the
    # physical topology from the coords on a multi-axis, non-hypercube mesh.
    if not _is_real_2x2x2():
        pytest.skip("needs a real 16-core tpu7x 2x2x2 slice (multi-host)")
    import numpy as _np

    from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.common import (
        hypercube_device_order,
    )

    hc = hypercube_device_order(jax.devices())
    devs = [
        hc[p ^ (p >> 1)] for p in range(16)
    ]  # gray-code (ring) order: NON-hypercube
    mesh = jax.sharding.Mesh(
        _np.array(devs).reshape(2, 8),
        ("dp", "tp"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    axes = ("dp", "tp")
    ws = NamedSharding(mesh, P(None, None))  # replicated weight

    # Spy the two hand-rolled Pallas kernels to assert the multi-axis product actually reaches
    # them (a silent reorder/naive fallback would still be correct but miss the new coverage).
    calls = {"topo": 0, "bcast": 0}
    real_topo, real_bcast = (
        _mag._matmul_all_gather_topo,
        _mag._matmul_all_gather_broadcast,
    )

    def _topo_spy(*a, **k):
        calls["topo"] += 1
        return real_topo(*a, **k)

    def _bcast_spy(*a, **k):
        calls["bcast"] += 1
        return real_bcast(*a, **k)

    monkeypatch.setattr(_mag, "_matmul_all_gather_topo", _topo_spy)
    monkeypatch.setattr(_mag, "_matmul_all_gather_broadcast", _bcast_spy)

    def _check(m, k, n):
        xs = NamedSharding(mesh, P(axes, None))  # tokens sharded over BOTH live axes

        @functools.partial(jax.jit, out_shardings=(xs, ws))
        def _gen():
            k1, k2 = jax.random.split(jax.random.key(0), 2)
            return (
                jax.random.normal(k1, (m, k), jnp.bfloat16),
                jax.random.normal(k2, (k, n), jnp.bfloat16),
            )

        x, w = _gen()
        jax.block_until_ready((x, w))

        def _ref(xx, ww):
            return jnp.dot(jax.lax.all_gather(xx, axes, axis=0, tiled=True), ww)

        ref = jax.jit(
            jax.shard_map(
                _ref,
                mesh=mesh,
                in_specs=(P(axes, None), P(None, None)),
                out_specs=P(None, None),
                check_vma=False,
            )
        )(x, w)
        out = matmul_all_gather(x, w, mesh=mesh, axis_name=axes)
        ref_g = _np.asarray(
            multihost_utils.process_allgather(ref, tiled=True), _np.float32
        )
        out_g = _np.asarray(
            multihost_utils.process_allgather(out, tiled=True), _np.float32
        )
        _np.testing.assert_allclose(out_g, ref_g, rtol=0.01, atol=0.01)

    # Down-proj T=256 -> broadcast (local T=16, output 2MB); up-proj T=512 -> topo;
    # T=4096 -> XLA reorder; T=32 -> naive.
    _check(256, 8192, 4096)
    assert calls["bcast"] >= 1, "multi-axis product did not reach the broadcast kernel"
    _check(512, 4096, 8192)
    assert calls["topo"] >= 1, "multi-axis product did not reach the topo gather"
    _check(4096, 4096, 8192)  # XLA reorder
    _check(32, 4096, 8192)  # naive


def test_pick_bn_targets_eight_128_aligned_tiles():
    # the two shapes we ship: N=8192 (up-proj) and N=4096 (down-proj)
    assert _pick_bn(8192) == 1024  # ~8 tiles
    assert _pick_bn(4096) == 512  # ~8 tiles of 512
    assert _pick_bn(8192) % 128 == 0 and _pick_bn(4096) % 128 == 0
    assert 8192 % _pick_bn(8192) == 0 and 4096 % _pick_bn(4096) == 0


# ---------------------------------------------------------------------------
# Transposed weight ([N, K] ColumnParallelLinear native layout). The unpooler weight
# is stored [out=N, in=K]; passing it as-is (w_transposed=True) instead of a materialized
# ``w.T`` lets every path contract K without a [N,K]->[K,N] transpose ``copy``.
# ---------------------------------------------------------------------------


def test_fused_transposed_weight():
    # Small-T fused broadcast with a native [N, K] weight == all_gather(x) @ W.T.
    _vs_reference(
        _matmul_all_gather_broadcast, m=256, k=4096, n=8192, w_transposed=True
    )


def test_dispatch_transposed_decode():
    # Dispatch (fused small-T path) with a native [N, K] weight, T=128.
    _vs_reference(matmul_all_gather, m=128, k=4096, n=8192, w_transposed=True)


def test_dispatch_transposed_large():
    # Dispatch (XLA reorder, large-T) with a native [N, K] weight, T=4096.
    _vs_reference(matmul_all_gather, m=4096, k=4096, n=8192, w_transposed=True)


def test_dispatch_transposed_tiny():
    # Dispatch (naive, tiny-T) with a native [N, K] weight, T=32.
    _vs_reference(matmul_all_gather, m=32, k=4096, n=8192, w_transposed=True)


def test_transposed_weight_avoids_copy_in_hlo():
    # Regression for the trace's copy.284: feeding a ``w.T`` VIEW to a Pallas BlockSpec that slices
    # physical [K, bn] tiles forces XLA to MATERIALIZE the whole [N, K] -> [K, N] weight transpose
    # every step. Passing w as [N, K] with w_transposed slices [bn, K] tiles + contracts K
    # in-kernel -> no [K, N] weight array is built. At T=256 the dispatch routes to topo (also a
    # [bn, K]-slicing Pallas gather), so this guards topo's transposed-weight path.
    if not _is_real_2x2x2():
        pytest.skip("needs a real 16-core tpu7x 2x2x2 slice (multi-host)")
    mesh = make_hypercube_tp_mesh("tp")
    m, k, n = 256, 4096, 8192  # mid-T up-proj -> both route to topo
    xs = NamedSharding(mesh, P("tp", None))
    ws = NamedSharding(mesh, P(None, None))

    @functools.partial(jax.jit, out_shardings=(xs, ws))
    def _gen():
        k1, k2 = jax.random.split(jax.random.key(0), 2)
        return (
            jax.random.normal(k1, (m, k), jnp.bfloat16),
            jax.random.normal(k2, (n, k), jnp.bfloat16),  # [N, K] native (out, in)
        )

    x, w_nk = _gen()
    jax.block_until_ready((x, w_nk))

    f_new = jax.jit(
        lambda xx, ww: matmul_all_gather(
            xx, ww, mesh=mesh, axis_name="tp", w_transposed=True
        )
    )
    f_old = jax.jit(  # the old integration materialized w.T then fed [K, N]
        lambda xx, ww: matmul_all_gather(xx, ww.T, mesh=mesh, axis_name="tp")
    )
    new_hlo = f_new.lower(x, w_nk).compile().as_text()
    old_hlo = f_old.lower(x, w_nk).compile().as_text()
    # [K, N] = [4096, 8192] can ONLY appear if the [N, K] weight is transpose-materialized:
    # x is [*, 4096], the output/N-tiles are [*, 8192]/[*, bn] -- none is [4096, 8192].
    assert "bf16[4096,8192]" in old_hlo, "expected the w.T path to materialize [K, N]"
    assert (
        "bf16[4096,8192]" not in new_hlo
    ), "w_transposed must avoid the [N,K]->[K,N] weight copy (trace's copy.284)"


def test_dispatch_size1_axes_native_topo(monkeypatch):
    """A tuple axis_name with size-1 axes + one live axis -- e.g. get_moe_expert_axis's
    ``(attn_dp=1, attn_dp_expert=1, expert=1, model=16)`` -- must hit the Pallas topo gather:
    the size-1 axes contribute a constant 0 digit in the mixed-radix combined index (no
    special-casing / stripping), so the product is effectively the single ``model`` axis and topo
    reads the physical topology from its coords. Regression for the MoE unpooler passing its
    EP-axis tuple straight to matmul_all_gather.
    """
    if not _is_real_2x2x2():
        pytest.skip("needs a real 16-core tpu7x 2x2x2 slice (multi-host)")
    calls = {"n": 0}
    real = _mag._matmul_all_gather_topo

    def _spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(_mag, "_matmul_all_gather_topo", _spy)

    # A deliberately NON-hypercube (ring/gray-code) order also proves topo routes by physical topo.
    base = make_ring_tp_mesh("tp")
    # Prepend a size-1 axis, preserving the device order on "tp".
    mesh = jax.sharding.Mesh(
        base.devices.reshape(1, base.devices.size), ("dummy", "tp")
    )
    ep = ("dummy", "tp")  # a size-1 axis + the live size-16 "tp"
    m, k, n = 256, 4096, 8192  # mid-T up-proj -> topo
    xs = NamedSharding(mesh, P(ep, None))
    ws = NamedSharding(mesh, P(None, None))

    @functools.partial(jax.jit, out_shardings=(xs, ws))
    def _gen():
        k1, k2 = jax.random.split(jax.random.key(0), 2)
        return (
            jax.random.normal(k1, (m, k), jnp.bfloat16),
            jax.random.normal(k2, (n, k), jnp.bfloat16),  # [N, K] -> w_transposed
        )

    x, w = _gen()
    jax.block_until_ready((x, w))

    def _ref(xx, ww):
        g = jax.lax.all_gather(xx, ep, axis=0, tiled=True).astype(ww.dtype)
        return jnp.dot(g, ww.T)

    ref = jax.jit(
        jax.shard_map(
            _ref,
            mesh=mesh,
            in_specs=(P(ep, None), P(None, None)),
            out_specs=P(None, None),
            check_vma=False,
        )
    )(x, w)
    out = matmul_all_gather(
        x, w, mesh=mesh, axis_name=ep, w_transposed=True, out_dtype=jnp.bfloat16
    )
    ref_g = np.asarray(multihost_utils.process_allgather(ref, tiled=True), np.float32)
    out_g = np.asarray(multihost_utils.process_allgather(out, tiled=True), np.float32)
    np.testing.assert_allclose(out_g, ref_g, rtol=0.01, atol=0.01)
    assert (
        calls["n"] >= 1
    ), "size-1 axes not handled natively -> topo missed (routed to reorder)"


class TestXlaPathXScale:
    """Raw fp8 x + per-token online-FP8 row scale through the two pure-XLA builders directly:
    ``_xla_all_gather_matmul`` (gather-then-matmul) and ``_xla_matmul_all_gather`` (matmul-then-gather).
    Both now dequant INSIDE ``_fn`` -- the gather-then-matmul path all-gathers the RAW fp8 x (keeping
    it fp8 on the wire) and dequants the gathered ``[T, K]`` activation; the matmul-then-gather path
    dequants the local ``[T/P, K]`` shard before its 1/P matmul. These paths' x_scale handling is
    otherwise only reachable via the size dispatch (tiny-T -> gather-then-matmul, large-T ->
    matmul-then-gather) and was untested directly. Real 16-core 2x2x2 (multi-host).
    """

    M, K, N = 256, 4096, 8192  # up-proj; w is [N, K] (w_transposed)

    @pytest.fixture
    def inputs(self):
        if not _is_real_2x2x2():
            pytest.skip("needs a real 16-core tpu7x 2x2x2 slice (multi-host)")
        mesh = make_hypercube_tp_mesh("tp")
        xs = NamedSharding(mesh, P("tp", None))
        ss = NamedSharding(mesh, P("tp"))  # per-token scale, sharded like x rows
        ws = NamedSharding(mesh, P(None, None))

        @functools.partial(jax.jit, out_shardings=(xs, ss, ws))
        def _gen():
            k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
            xf = jax.random.normal(k1, (self.M, self.K), jnp.bfloat16).astype(
                jnp.float8_e4m3fn
            )
            scale = jax.random.uniform(
                k3, (self.M,), jnp.float32, minval=0.1, maxval=0.6
            )
            w = jax.random.normal(k2, (self.N, self.K), jnp.bfloat16)  # [N, K]
            return xf, scale, w

        x, scale, w = _gen()
        jax.block_until_ready((x, scale, w))
        return mesh, x, scale, w

    @pytest.mark.parametrize(
        "builder_name", ["_xla_all_gather_matmul", "_xla_matmul_all_gather"]
    )
    def test_equals_scaled_reference(self, inputs, builder_name):
        # Each XLA path == (x_fp8 * scale) @ W within fp8 tolerance, output dtype bf16. Moving the
        # dequant inside _fn (after the gather / on the local shard) must not change the result.
        mesh, x, scale, w = inputs

        def _ref(xx, sc, ww):
            xd = (xx.astype(jnp.float32) * sc[:, None]).astype(ww.dtype)
            g = jax.lax.all_gather(xd, "tp", axis=0, tiled=True)
            return jnp.dot(g, ww.T)

        ref = jax.jit(
            jax.shard_map(
                _ref,
                mesh=mesh,
                in_specs=(P("tp", None), P("tp"), P(None, None)),
                out_specs=P(None, None),
                check_vma=False,
            )
        )(x, scale, w)
        builder = getattr(_mag, builder_name)
        fn = builder(mesh, "tp", w_transposed=True, out_dtype=jnp.bfloat16)
        out = fn(x, w, scale)
        assert out.dtype == jnp.bfloat16
        ref_g = np.asarray(
            multihost_utils.process_allgather(ref, tiled=True), np.float32
        )
        out_g = np.asarray(
            multihost_utils.process_allgather(out, tiled=True), np.float32
        )
        np.testing.assert_allclose(out_g, ref_g, rtol=0.02, atol=0.02)

    def test_gather_then_matmul_keeps_x_fp8_on_the_wire(self, inputs):
        # The gather-then-matmul path must all-gather the RAW fp8 x (the dequant happens AFTER the
        # gather), so a scaled fp8 x stays fp8 on the wire. Regression for dequanting up front, which
        # all-gathers a widened bf16 [T, K] -- 2x the bytes. K != N, so [M, K] is unambiguously the
        # gathered activation (not the [M, N] matmul output). HLO-string check like
        # ``test_transposed_weight_avoids_copy_in_hlo``.
        mesh, x, scale, w = inputs
        fn = _mag._xla_all_gather_matmul(
            mesh, "tp", w_transposed=True, out_dtype=jnp.bfloat16
        )
        hlo = fn.lower(x, w, scale).compile().as_text()
        act = rf"\[{self.M},{self.K}\]"  # the gathered [T, K] activation
        assert re.search(
            rf"f8e4m3fn{act}[^\n]*all-gather", hlo
        ), "gather-then-matmul must all-gather the raw fp8 x (fp8 stays on the wire)"
        assert not re.search(
            rf"bf16{act}[^\n]*all-gather", hlo
        ), "x must NOT be widened to bf16 before the all-gather"


class TestTopoGather:
    """The opt-in topology-aware output gather (``matmul_all_gather(..., gather_kind="topo")``)
    on a real 16-core 2x2x2. topo routes the output gather through the 2x2x2x2 hypercube
    store-and-forward schedule (``_matmul_topo._matmul_all_gather_topo``), reading the physical
    topology from the device coords so ANY mesh order works; it is correctness-preserving, so every
    case must still equal ``all_gather(x) @ W`` within bf16 tolerance. Driven through the public
    ``matmul_all_gather`` entrypoint (the opt-in knob). ``_vs_reference`` uses the hypercube mesh;
    ``test_ring_mesh_order`` additionally exercises a NON-hypercube (serve-like) order.
    """

    @staticmethod
    def _topo(x, w, **kw):
        return matmul_all_gather(x, w, gather_kind="topo", **kw)

    def test_up_proj_decode(self):
        _vs_reference(self._topo, m=256, k=4096, n=8192)

    def test_down_proj_decode(self):
        # down-proj K8192/N4096 -- the other shape we ship.
        _vs_reference(self._topo, m=256, k=8192, n=4096)

    def test_fp32(self):
        # all-fp32 path: o_vmem + matmul in f32.
        _vs_reference(
            self._topo, m=256, k=4096, n=8192, x_dtype=jnp.float32, w_dtype=jnp.float32
        )

    def test_reuse_safe(self):
        # reuse the compiled kernel 8x -- the setup double-barrier must survive
        # collective_id reuse (no cross-call corruption of the pipelined semaphores).
        _vs_reference(self._topo, m=256, k=4096, n=8192, n_calls=8)

    def test_larger_t(self):
        _vs_reference(self._topo, m=512, k=4096, n=8192)

    def test_ring_mesh_order(self):
        # NON-hypercube (serve-like ring/gray-code) mesh: topo must figure out the physical
        # topology from the device coords and still equal all_gather(x) @ W -- proves the routing
        # is mesh-ORDER-agnostic (the point of dropping the hypercube-ordering requirement).
        if not _is_real_2x2x2():
            pytest.skip("needs a real 16-core tpu7x 2x2x2 slice (multi-host)")
        mesh = make_ring_tp_mesh("tp")
        m, k, n = 256, 4096, 8192
        xs = NamedSharding(mesh, P("tp", None))
        ws = NamedSharding(mesh, P(None, None))

        @functools.partial(jax.jit, out_shardings=(xs, ws))
        def _gen():
            k1, k2 = jax.random.split(jax.random.key(0), 2)
            return (
                jax.random.normal(k1, (m, k), jnp.bfloat16),
                jax.random.normal(k2, (k, n), jnp.bfloat16),
            )

        x, w = _gen()
        jax.block_until_ready((x, w))

        def _ref(xx, ww):
            return jnp.dot(jax.lax.all_gather(xx, "tp", axis=0, tiled=True), ww)

        ref = jax.jit(
            jax.shard_map(
                _ref,
                mesh=mesh,
                in_specs=(P("tp", None), P(None, None)),
                out_specs=P(None, None),
                check_vma=False,
            )
        )(x, w)
        out = matmul_all_gather(x, w, mesh=mesh, axis_name="tp", gather_kind="topo")
        ref_g = np.asarray(
            multihost_utils.process_allgather(ref, tiled=True), np.float32
        )
        out_g = np.asarray(
            multihost_utils.process_allgather(out, tiled=True), np.float32
        )
        np.testing.assert_allclose(out_g, ref_g, rtol=0.01, atol=0.01)

    def test_transposed_weight(self):
        # [N, K] ColumnParallelLinear layout (w_transposed): topo slices natural [bn, K] row
        # tiles (no transpose copy) and contracts K -- same result as the [K, N] layout.
        _vs_reference(self._topo, m=256, k=4096, n=8192, w_transposed=True)

    def test_fp8_input_bf16_output(self):
        # Online-FP8 unpooler input: raw fp8 x -> bf16 output (out_dtype). topo widens x to W's
        # dtype at the matmul and gathers the bf16 OUTPUT, exactly like the other paths.
        _vs_reference(
            self._topo,
            m=256,
            k=4096,
            n=8192,
            x_dtype=jnp.float8_e4m3fn,
            out_dtype=jnp.bfloat16,
        )

    def test_transposed_fp8(self):
        # Both at once -- the real serve regime: fp8 x + [N, K] CPL weight -> bf16 output.
        _vs_reference(
            self._topo,
            m=256,
            k=4096,
            n=8192,
            w_transposed=True,
            x_dtype=jnp.float8_e4m3fn,
            out_dtype=jnp.bfloat16,
        )

    def test_x_scale_dequant(self):
        # Raw fp8 x + per-token online-FP8 row scale: topo dequants the local shard up front
        # (it gathers the OUTPUT, not x) and equals (x_fp8 * scale) @ W within fp8 tolerance.
        if not _is_real_2x2x2():
            pytest.skip("needs a real 16-core tpu7x 2x2x2 slice (multi-host)")
        mesh = make_hypercube_tp_mesh("tp")
        m, k, n = 256, 4096, 8192
        xs = NamedSharding(mesh, P("tp", None))
        ss = NamedSharding(
            mesh, P("tp")
        )  # per-token scale, token-sharded like x's rows
        ws = NamedSharding(mesh, P(None, None))

        @functools.partial(jax.jit, out_shardings=(xs, ss, ws))
        def _gen():
            k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
            xf = jax.random.normal(k1, (m, k), jnp.bfloat16).astype(jnp.float8_e4m3fn)
            scale = jax.random.uniform(k3, (m,), jnp.float32, minval=0.1, maxval=0.6)
            w = jax.random.normal(k2, (n, k), jnp.bfloat16)  # [N, K] (w_transposed)
            return xf, scale, w

        x, scale, w = _gen()
        jax.block_until_ready((x, scale, w))

        def _ref(xx, sc, ww):
            xd = (xx.astype(jnp.float32) * sc[:, None]).astype(ww.dtype)
            g = jax.lax.all_gather(xd, "tp", axis=0, tiled=True)
            return jnp.dot(g, ww.T)

        ref = jax.jit(
            jax.shard_map(
                _ref,
                mesh=mesh,
                in_specs=(P("tp", None), P("tp"), P(None, None)),
                out_specs=P(None, None),
                check_vma=False,
            )
        )(x, scale, w)

        out = matmul_all_gather(
            x,
            w,
            mesh=mesh,
            axis_name="tp",
            gather_kind="topo",
            w_transposed=True,
            out_dtype=jnp.bfloat16,
            x_scale=scale,
        )
        ref_g = np.asarray(
            multihost_utils.process_allgather(ref, tiled=True), np.float32
        )
        out_g = np.asarray(
            multihost_utils.process_allgather(out, tiled=True), np.float32
        )
        np.testing.assert_allclose(out_g, ref_g, rtol=0.02, atol=0.02)


class TestAutoTopoDispatch:
    """``matmul_all_gather(gather_kind="auto")`` auto-selects the topo gather for T>=256 on ANY
    16-core 2x2x2 slice (the measured-fastest path there). The gate is ``_is_2x2x2_slice`` -- topo
    reads the physical topology from the device coords, so any mesh order qualifies (hypercube AND
    the serve's ring order). Non-2x2x2 meshes fall through to the XLA reorder / naive.
    """

    def test_is_2x2x2_slice_accepts_any_order(self):
        # topo works on any 2x2x2 order now: both the hypercube AND the serve's ring order are
        # accepted (each is a resolvable 8-chip x 2-core 2x2x2 slice).
        if not _is_real_2x2x2():
            pytest.skip("needs a real 16-core tpu7x 2x2x2 slice (multi-host)")
        assert _mag._is_2x2x2_slice(make_hypercube_tp_mesh("tp"), ("tp",)) is True
        assert _mag._is_2x2x2_slice(make_ring_tp_mesh("tp"), ("tp",)) is True

    def test_auto_routes_to_topo_at_t256(self):
        # T=256 (out_bytes 4MB > 3MB, m_local=16) on a hypercube mesh -> topo is auto-selected.
        self._assert_routes(m=256, expect_topo=True)

    def test_auto_skips_topo_at_t128(self):
        # T=128 has only eight local rows, so both Pallas DMA paths are skipped.
        self._assert_routes(m=128, expect_topo=False)

    def test_auto_routes_to_topo_fp8_transposed(self):
        # Full parity: fp8 x + [N, K] weight -> bf16 output still auto-selects topo at T>=256
        # (no dtype/layout gate -- only the 2x2x2 mesh gates it).
        self._assert_routes(
            m=256,
            expect_topo=True,
            x_dtype=jnp.float8_e4m3fn,
            w_transposed=True,
            out_dtype=jnp.bfloat16,
        )

    def test_auto_routes_to_topo_on_ring_mesh(self):
        # The KEY new behavior: topo is auto-selected at T=256 on the serve's NON-hypercube ring
        # mesh too (it figures out the physical topology from the coords).
        self._assert_routes(m=256, expect_topo=True, mesh_fn=make_ring_tp_mesh)

    @staticmethod
    def _assert_routes(
        m,
        expect_topo,
        *,
        x_dtype=None,
        w_transposed=False,
        out_dtype=None,
        mesh_fn=None,
    ):
        if not _is_real_2x2x2():
            pytest.skip("needs a real 16-core tpu7x 2x2x2 slice (multi-host)")
        mesh = (mesh_fn or make_hypercube_tp_mesh)("tp")
        xs = NamedSharding(mesh, P("tp", None))
        ws = NamedSharding(mesh, P(None, None))
        w_shape = (8192, 4096) if w_transposed else (4096, 8192)

        @functools.partial(jax.jit, out_shardings=(xs, ws))
        def _gen():
            k1, k2 = jax.random.split(jax.random.key(0), 2)
            return (
                jax.random.normal(k1, (m, 4096), jnp.bfloat16).astype(
                    x_dtype or jnp.bfloat16
                ),
                jax.random.normal(k2, w_shape, jnp.bfloat16),
            )

        x, w = _gen()
        jax.block_until_ready((x, w))

        # Spy on the dispatch module's topo builder; restore in finally so pytest-split runs
        # (whole-suite) don't inherit a patched global.
        called = {"topo": False}
        orig = _mag._matmul_all_gather_topo

        @functools.wraps(orig)
        def _spy(*a, **kw):
            called["topo"] = True
            return orig(*a, **kw)

        _mag._matmul_all_gather_topo = _spy
        try:
            jax.block_until_ready(
                matmul_all_gather(
                    x,
                    w,
                    mesh=mesh,
                    axis_name="tp",
                    w_transposed=w_transposed,
                    out_dtype=out_dtype,
                )
            )
        finally:
            _mag._matmul_all_gather_topo = orig
        assert called["topo"] is expect_topo


# ---------------------------------------------------------------------------
# PURE-PYTHON topo routing-schedule tests (no TPU): exercise only the logical routing
# table + host-side simulator in ``_matmul_topo`` (``_sched``). The Pallas kernel that
# consumes this exact schedule is validated by ``TestTopoGather`` above on a real 2x2x2.
# An entry ``(link_mask, slot_xor)`` means device ``D`` sends its slot ``D ^ slot_xor`` to
# neighbour ``D ^ link_mask`` (receiver stores it at the same global slot id), so ``D`` gains
# the piece at offset ``link_mask ^ slot_xor``. See ``_matmul_topo.py`` for the derivation.
# ---------------------------------------------------------------------------


def _delivered(entry):
    """Offset (relative to the receiver) of the piece an entry delivers."""
    link_mask, slot_xor = entry
    return link_mask ^ slot_xor


class TestBuildIciSchedule:
    """The 3-step ICI neighbour-exchange routing table."""

    def test_has_three_steps(self):
        assert len(_sched.build_ici_schedule()) == 3

    def test_piece_counts_are_3_3_1(self):
        # The defining property of the 2x2x2 schedule: each chip pulls 3 pieces
        # in step 1 (distance-1 neighbours), 3 in step 2 (distance-2), and 1 in
        # step 3 (the antipode) -- 7 remote pieces total.
        table = _sched.build_ici_schedule()
        assert [len(step) for step in table] == [3, 3, 1]

    def test_only_uses_ici_links(self):
        # Every hop must be along an ICI dimension (x=2, y=4, z=8); never the
        # core-bridge (mask 1), which is reserved for the reconstruction phase.
        for step in _sched.build_ici_schedule():
            for link_mask, _slot_xor in step:
                assert link_mask in _sched.ICI_MASKS

    def test_step_k_delivers_distance_k_pieces(self):
        # Step k (0-indexed) must deliver pieces at Hamming distance k+1, and the
        # slot it forwards must be a piece it already owns (distance k).
        for k, step in enumerate(_sched.build_ici_schedule()):
            for entry in step:
                link_mask, slot_xor = entry
                assert bin(slot_xor).count("1") == k  # forwards a dist-k piece
                assert bin(_delivered(entry)).count("1") == k + 1  # gains dist-(k+1)
                assert link_mask & _delivered(entry)  # link is a set bit of target

    def test_step1_broadcasts_own_shard(self):
        step1 = _sched.build_ici_schedule()[0]
        assert sorted(step1) == sorted((m, 0) for m in _sched.ICI_MASKS)

    def test_step2_is_a_perfect_matching(self):
        # In step 2 the 3 links must carry 3 *distinct* distance-2 pieces (a
        # perfect matching over the 3 ICI links), otherwise a link is wasted.
        step2 = _sched.build_ici_schedule()[1]
        assert len({link for link, _ in step2}) == 3
        assert {_delivered(e) for e in step2} == {2 ^ 4, 2 ^ 8, 4 ^ 8}  # {6,10,12}

    def test_step3_reaches_the_antipode(self):
        step3 = _sched.build_ici_schedule()[2]
        assert len(step3) == 1
        assert _delivered(step3[0]) == 2 ^ 4 ^ 8  # 14, the distance-3 antipode


class TestSimulateAllGather:
    """Host-side simulation of running the schedule (with the split final step)
    on all 16 devices."""

    def test_ici_phase_gathers_all_same_parity_slots(self):
        # After the 3 ICI steps, each device holds every slot that shares its
        # core parity (its own core-plane's 8 half-shards) and nothing else.
        buffers = _sched.simulate_ici_phase()
        for dev in range(16):
            parity = dev & 1
            expected = {s for s in range(16) if (s & 1) == parity}
            assert buffers[dev] == expected, f"device {dev} gathered {buffers[dev]}"

    def test_bridge_phase_completes_full_all_gather(self):
        # After the core-bridge exchange every device holds all 16 half-shards.
        buffers = _sched.simulate_full()
        for dev in range(16):
            assert buffers[dev] == set(range(16)), f"device {dev}: {buffers[dev]}"

    def test_every_forward_uses_data_already_received(self):
        # A device may only forward a slot it has actually received in a prior
        # step; simulate_ici_phase(strict=True) raises if that is ever violated.
        _sched.simulate_ici_phase(strict=True)

    def test_intermediate_counts_are_4_7_8(self):
        # Cumulative slots held (including own) after each step: 1+3, +3, +1.
        assert _sched.simulate_ici_counts() == [4, 7, 8]


class TestAntipodeCoreSplit:
    """The final (antipode) step sends each on-chip core over a different ICI axis,
    so it uses 2 physical links at half width instead of 1 at full width. (The
    full-gather / data-dependency / piece-count proofs for the split live in
    ``TestSimulateAllGather``, since the simulator always applies the split.)"""

    def test_two_cores_use_different_axes(self):
        # core 0 and core 1 pull the antipode over distinct ICI links, and both
        # deliveries still land on the antipode (offset link ^ slot_xor).
        split = _sched.antipode_core_split()
        (l0, x0), (l1, x1) = split[0], split[1]
        assert l0 != l1  # different physical axes
        assert l0 ^ x0 == _sched.ANTIPODE_OFFSET
        assert l1 ^ x1 == _sched.ANTIPODE_OFFSET
