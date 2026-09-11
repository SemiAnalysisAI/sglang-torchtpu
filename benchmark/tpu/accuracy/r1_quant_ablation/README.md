# Temporary DeepSeek R1-0528 accuracy isolation

This worktree starts from the frozen production serving snapshot. Unset controls
preserve production behavior. These are diagnostic controls, not new defaults.

| Environment variable | Values | Scope |
|---|---|---|
| `SGLANG_R1_TEST_WEIGHTS` | `production` (default), `mla_bf16`, `native` | MLA-only direct checkpoint-to-BF16 weights, or that plus native generic dense/MoE block weights |
| `SGLANG_R1_TEST_MATH` | `production` (default), `fp32` | K=128 dense/MoE activation groups; FP32 dense block accumulator, MoE scales/activation quantization/partial sums, router logits and expert combination |
| `SGLANG_R1_TEST_KV` | `production` (default), `bf16` | R1 recipe selects BF16 KV instead of the baseline FP8 KV |

`native` changes generic matmul implementations, not just stored weight codes.
Use the same `MATH=fp32` setting on native and channelwise arms for a meaningful
weight comparison. Native MoE blocks also bound the activation group size.
The math setting keeps native FP8 dot operations; it does not establish GPU
bitwise equivalence or remove every hardware rounding difference.

The MLA treatment dequantizes original checkpoint blocks directly into BF16
`w_kc`/`w_vc` before generic linear post-processing. It avoids the legacy extra
tensorwise FP8 conversion. The unused/other `kv_b_proj` linear path may still be
processed later; the absorbed MLA forward uses the saved BF16 weights.

Experiment order (hypothesis ranking, not measured attribution):

1. Production baseline.
2. MLA-only BF16 weight reconstruction, the previously unremoved conversion.
3. Native weights with precise math, a broad weight-preservation reference.
4. BF16 KV alone.
5. MLA BF16 plus precise math: matched channelwise comparator for arm 3.
6. Native weights + precise math + BF16 KV: combined reference.

Each arm: the original fixed 11 historically failed AIME25 questions, original
`generic/math` prompt, temperature 0.6, top-p 0.95, 65536 generation tokens,
two samples per question, concurrency 8, identical server seed. Per-request
seeds remain unset because seeded sampling uses a different slow code path.
This is a diagnostic subset score, not a full AIME25 score. Preserve raw API
reasoning and final answers; no score-based early stopping or answer repair.

Before serving, run `mla_smoke.py` on CPU and `kernel_smoke.py` on an idle TPU
with `SGLANG_R1_TEST_MATH=fp32` and `R1_TEST_OUTPUT` set to an artifact directory.
The MLA test executes the actual post-loader method and compares its TPU
BF16 treatment against both direct checkpoint reconstruction and the CUDA
BF16 branch. Kernel smoke covers partial output blocks and matched block/
channel scale representations.
