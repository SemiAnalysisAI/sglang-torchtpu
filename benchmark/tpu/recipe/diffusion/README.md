# Diffusion recipes on Cloud TPU (`torch_tpu`)

Launch recipes for `sglang.multimodal_gen` on one tpu7x 2x2x1 host (8 TensorCores, `--num-gpus` counts cores).
All recipes use the Pallas splash attention backend, torch.compile with the `tpu` backend, a TP-folded resident
text encoder, the highest TPU clock and a 128 MiB scoped-VMEM limit for XLA fusions (`LIBTPU_INIT_ARGS`); see `launch.sh`. The Wan
recipes compile per transformer block (`--regional-compile`), which keeps each block's collectives inside its program,
and run Ulysses across the two cores of a chip with a K/V all-gather ring across chips.

| Recipe | Model | Cores | Parallelism |
| :--- | :--- | :---: | :--- |
| `flux2_klein_4b.sh` | `black-forest-labs/FLUX.2-klein-4B` | 1 or 4 | Ulysses = cores (24 heads) |
| `zimage_turbo.sh` | `Tongyi-MAI/Z-Image-Turbo` | 1 or 2 | Ulysses = cores (30 heads) |
| `wan22_ti2v_5b.sh` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 8 | CFG 2 x Ulysses 2 x ring 2 |
| `wan22_a14b.sh` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 8 | CFG 2 x Ulysses 2 x ring 2 |
| `ltx2.sh` | `Lightricks/LTX-2` | 8 | CFG 2 x Ulysses 4 |

Override `NUM_GPUS`, `PARALLEL_ARGS`, `MODEL_ARGS`, `PORT`, or append flags with `EXTRA_ARGS`.

## Performance

tpu7x 2x2x1, steady-state request (the first request compiles), seed 42. Shapes whose token count does not
divide by the sequence-parallel size (the two I2V rows) pad the sequence; attention masks the padding.

| Model | Request | Cores | E2E | Denoise step | VAE decode | Peak HBM / core |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: |
| FLUX.2-klein-4B | T2I 1024x1024, 4 steps | 1 | 0.96 s | 0.075 s | 0.33 s | 25.4 GB |
| FLUX.2-klein-4B | T2I 1024x1024, 4 steps | 4 | 0.78 s | 0.044 s | 0.33 s | 19.6 GB |
| Z-Image-Turbo | T2I 1024x1024, 9 steps | 1 | 1.46 s | 0.091 s | 0.32 s | 29.9 GB |
| Z-Image-Turbo | T2I 1024x1024, 9 steps | 2 | 1.21 s | 0.070 s | 0.33 s | 25.8 GB |
| Wan2.2-TI2V-5B | T2V 1280x704, 121 frames, 20 steps | 8 | 9.6 s | 0.274 s | 2.5 s | 30.5 GB |
| Wan2.2-TI2V-5B | I2V 960x928 (from a square image), 121 frames, 20 steps | 8 | 11.7 s | 0.331 s | 2.6 s | 65.7 GB |
| Wan2.2-T2V-A14B | T2V 1280x720, 81 frames, 20 steps | 8 | 58.6 s | 2.72 s | 2.2 s | 89.9 GB |
| Wan2.2-I2V-A14B | I2V 624x624, 81 frames, 20 steps | 8 | 17.9 s | 0.703 s | 1.2 s | 73.6 GB |
| LTX-2 | T2V+audio 1280x720, 121 frames, 40 steps | 8 | 23.3 s | 0.426 s | 2.9 s | 60.3 GB |

## Example request

```bash
curl -s -X POST http://localhost:30000/v1/videos -H "Content-Type: application/json" -d '{
  "prompt": "A red fox running through fresh snow, cinematic lighting",
  "size": "1280x720", "num_frames": 81, "num_inference_steps": 20, "guidance_scale": 4.0, "seed": 42
}'
```

Images use `/v1/images/generations` with `size`, `num_inference_steps`, and `guidance_scale`.
