import json
import os
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch

from sglang.srt.hardware_backend.tpu import fp8_linear as lin
from sglang.srt.hardware_backend.tpu import quantized_matmul as qm
from sglang.srt.layers.moe.pallas.gmm_v2 import TileSizes, gmm_v2

r = Path(os.environ["R1_TEST_OUTPUT"])
results = []
# Validate native partial output blocks in the torch bridge, without allocating
# torch TPU tensors or hiding shape issues behind torch.compile.
original = qm.blockwise_quantized_matmul
seen = []


def capture(x, w, s):
    seen.append(s.clone())
    return torch.zeros((x.shape[0], w.shape[0]), dtype=x.dtype)


qm.blockwise_quantized_matmul = capture
n, k = 2112, 7168
s = torch.arange(((n + 127) // 128) * (k // 128), dtype=torch.float32).reshape(
    (n + 127) // 128, k // 128
)
y = lin.tpu_w8a8_block_fp8_linear(
    torch.zeros((1, k), dtype=torch.bfloat16),
    torch.zeros((n, k), dtype=torch.float8_e4m3fn),
    [128, 128],
    s,
)
assert y.shape == (1, n)
assert torch.equal(seen[-1], s.t().repeat_interleave(128, 1)[:, :n].unsqueeze(1))
results.append(dict(test="native_partial_N_bridge", shape=[n, k], passed=True))
qm.blockwise_quantized_matmul = original
rng = np.random.default_rng(881)
for m, n, k in [(1, 2112, 7168), (32, 448, 2048), (1, 128, 256)]:
    x = jnp.asarray(rng.normal(0, 0.3, (m, k)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(0, 30, (n, k)), jnp.float8_e4m3fn)
    scale = jnp.asarray(rng.uniform(0.001, 0.03, (k // 128, 1, n)), jnp.float32)
    t = time.time()
    out = qm._blockwise_quantized_matmul_kernel(
        x, w, scale, block_size=128, x_q_dtype=jnp.float8_e4m3fn
    )
    acc = jnp.zeros((m, n), jnp.float32)
    for b in range(k // 128):
        a = x[:, b * 128 : (b + 1) * 128].astype(jnp.float32)
        amax = jnp.max(jnp.abs(a), 1, keepdims=True)
        a_scale = jnp.where(amax == 0, 1.0, amax / 448.0)
        aq = jnp.clip(a / a_scale, -448, 448).astype(jnp.float8_e4m3fn)
        acc = (
            acc
            + jnp.matmul(
                aq, w[:, b * 128 : (b + 1) * 128].T, preferred_element_type=jnp.float32
            )
            * a_scale
            * scale[b]
        )
    ref = acc.astype(jnp.bfloat16)
    aa, bb = np.asarray(out, np.float32), np.asarray(ref, np.float32)
    rel = float(np.linalg.norm(aa - bb) / max(np.linalg.norm(bb), 1e-12))
    assert np.isfinite(aa).all() and rel < 0.01, rel
    results.append(
        dict(
            test="FP32_dense_acc",
            shape=[m, n, k],
            relative_l2=rel,
            seconds=time.time() - t,
            passed=True,
        )
    )
    print(results[-1], flush=True)
# Production R1 expands N-block scales; compare a blockwise representation
# against identical per-channel scales while retaining the same K=128 act groups.
x = jnp.asarray(rng.normal(0, 0.3, (16, 512)), jnp.bfloat16)
w = jnp.asarray(rng.normal(0, 30, (2, 512, 256)), jnp.float8_e4m3fn)
s = jnp.asarray(rng.uniform(0.001, 0.03, (2, 1, 1, 256)), jnp.bfloat16)
g = jnp.array([5, 11], jnp.int32)
a = gmm_v2(
    x, w, g, rhs_scale=jnp.repeat(s, 4, axis=1), tile_info=TileSizes(16, 256, 128)
)
b = gmm_v2(x, w, g, rhs_scale=s, tile_info=TileSizes(16, 256, 128))
np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
results.append(dict(test="FP32_MoE_block_vs_channel_representations", passed=True))
(r / "startup-smoke.json").write_text(json.dumps(results, indent=2))
print("PASSED", flush=True)
