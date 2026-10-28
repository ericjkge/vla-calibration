# Cannon validation — 2026-10-28

The runtime uses OpenPI's unquantized BF16/FP32 PyTorch implementation, not
integer quantization or all-FP32 inference. One episode is an integration check;
it does not reproduce the published LIBERO benchmark average.

## Environment and checkpoint

- OpenPI: `215abfb217dbac7d5f1273282331b9b1866c0479`.
- Original LIBERO: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
- Model environment: Python 3.11, PyTorch `2.7.1+cu126`, Transformers `4.53.2`
  with OpenPI's pinned patches.
- Simulator: Python 3.8.20, MuJoCo 2.3.7, robosuite 1.4.1, CPU Torch 1.11.0.
- Source checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`.
- Conversion job `55941761` completed with exit 0 in 10m40s; peak host memory
  was approximately 28.1 GiB. The resulting `model.safetensors` is about 6.8 GiB.
- The original checkpoint's normalization assets were copied and validated.

The checkpoint is at
`/n/home13/sophiezeng/vla-calibration/data/checkpoints/pi05_libero_pytorch`.
Setup instructions and launch commands are in [smoke-test.md](smoke-test.md).

## Successful episode

Job `55962026` completed with Slurm exit `0:0` and client exit 0 on
`holygpu8a29406` using an NVIDIA A100-SXM4-40GB with MIG disabled. Total job time
was 4m00s; peak host memory was approximately 11.4 GiB.

| Setting / result | Value |
| --- | --- |
| Repository commit | `946b1fd4616d8cdfff01775f74cfd96491de4169` |
| Suite / task / initial state | `libero_spatial` / `0` / `0` |
| Seed | `7` |
| Instruction | Pick up the black bowl between the plate and the ramekin and place it on the plate |
| Outcome | `status: completed`, `success: true` |
| Executed action steps | 78, following 10 settling steps |
| Policy calls | 16; 10-action predictions, replanning every 5 actions |
| Action dimensions | 7 |
| Denoising steps | 10 |
| Renderer | EGL |
| Client elapsed time | 115.16 seconds, including initialization |

The run saved `episode.mp4`, `episodes.jsonl`, `run.json`, `server.log`, and
`client.log` under
`/n/home13/sophiezeng/vla-calibration/data/results/smoke-20261028T195612Z-3009131/`.
A copy is in the local workspace at the same relative `data/results/` path.
Results remain outside Git. This validates the end-to-end observation → PyTorch
policy → 7D action → simulator loop for this one task and initial state.

## Resolved setup failure

GPU job `55948084` successfully loaded the policy and served its health check on
an NVIDIA A100-SXM4-40GB. The simulator then failed before stepping because BDDL
did not declare its `future` dependency. This is recorded as an infrastructure
error with `success: null`, not as a failed manipulation task.

The dependency is now pinned, and setup checks import `OffScreenRenderEnv`.
CPU job `55962024` passed that stronger check before the successful GPU retry.

## Reproduction

For this short run, the scheduler command selects a full A100 in the preemptible
partition rather than a MIG slice:

```bash
sbatch --partition=gpu_requeue --gres=gpu:nvidia_a100-sxm4-40gb:1 \
  scripts/cannon_smoke.sbatch
```

The default script still targets the regular `gpu` partition. Preempted runs and
startup/import errors must not be counted as benchmark task failures.
