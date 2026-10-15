"""Pallas GDN-v3 KDA mode vs the lax.scan reference at GLM-5.3-Flash TP8 per-rank shapes."""
import jax, jax.numpy as jnp, numpy as np
from sglang.srt.hardware_backend.tpu.kernels.kda import ragged_kda
from sglang.srt.layers.attention.pallas.gdn import fused_conv1d_gdn
from sglang.srt.layers.attention.pallas.gdn.config import AttentionMode

H, D, K, LB, SLOTS = 8, 128, 4, -5.0, 4
dim = 3 * H * D
key = iter(jax.random.split(jax.random.PRNGKey(0), 64))
r = lambda shape, s=1.0, dt=jnp.bfloat16: (jax.random.normal(next(key), shape) * s).astype(dt)
conv_w = r((dim, K), 0.5, jnp.float32)
a_log = r((H,), 0.5, jnp.float32); dt_bias = r((H, D), 0.5, jnp.float32)

def run_pallas(qkv, a, b, conv, ssm, qsl, idx, seq, decode):
    T = qkv.shape[0]; bs = idx.shape[0]
    dist = jnp.array([bs, bs, bs] if decode else [0, 0, bs], jnp.int32)
    (conv, ssm), out = fused_conv1d_gdn(qkv, b, a, conv, ssm, conv_w[:, None, :], None, a_log, dt_bias,
        qsl, idx, dist, seq, n_kq=H, n_v=H, d_k=D, d_v=D, kernel_size=K,
        attention_mode=AttentionMode.KDA, gate_lower_bound=LB)
    return conv, ssm, out.reshape(T, H, D)

def run_ref(qkv, a, b, conv, ssm, qsl, idx, seq):
    return ragged_kda(qkv, a, b, conv, ssm, conv_w, a_log.reshape(1, 1, H, 1), dt_bias.reshape(-1),
                      qsl, idx, seq, H, D, LB)

def cmp(name, x, y):
    x = np.asarray(x, np.float32); y = np.asarray(y, np.float32)
    err = np.abs(x - y).max(); rel = np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-9)
    print(f"{name}: max_abs={err:.3e} rel_l2={rel:.3e}", flush=True)
    return rel

# Prefill: two requests (300 fresh tokens, 211 tokens) padded to 512.
T = 512
lens = [300, 211]
qsl = jnp.array([0, 300, 511], jnp.int32); idx = jnp.array([1, 2], jnp.int32); seq = jnp.array([300, 211], jnp.int32)
qkv, a, b = r((T, dim)), r((T, H * D)), r((T, H))
z_conv = jnp.zeros((SLOTS, K - 1, dim), jnp.bfloat16); z_ssm = jnp.zeros((SLOTS, H, D, D), jnp.float32)
pc, ps, po = jax.jit(run_pallas, static_argnums=8)(qkv, a, b, z_conv, z_ssm, qsl, idx, seq, False)
rc, rs, ro = jax.jit(run_ref)(qkv, a, b, z_conv, z_ssm, qsl, idx, seq)
worst = max(cmp("prefill out", po[:511], ro[:511]), cmp("prefill ssm", ps[1:3], rs[1:3]), cmp("prefill conv", pc[1:3], rc[1:3]))
# Decode 8 steps from the prefilled state (both requests).
for step in range(8):
    q1, a1, b1 = r((2, dim)), r((2, H * D)), r((2, H))
    qsl_d = jnp.array([0, 1, 2], jnp.int32); seq = seq + 1
    pc, ps, po = jax.jit(run_pallas, static_argnums=8)(q1, a1, b1, pc, ps, qsl_d, idx, seq, True)
    rc, rs, ro = jax.jit(run_ref)(q1, a1, b1, rc, rs, qsl_d, idx, seq)
worst = max(worst, cmp("decode out (step 8)", po, ro), cmp("decode ssm (step 8)", ps[1:3], rs[1:3]))
print("PASS" if worst < 2e-2 else "FAIL", f"worst rel_l2={worst:.3e}")
