# Robust Calibration for VLA Quantization

CS 2420 final project.

Post-training quantization (PTQ) makes vision-language-action (VLA) models practical for edge
deployment, but existing methods are evaluated only in-distribution, where quantized policies look
nearly lossless. Standard PTQ calibrates on clean frames, so under distribution shift (lighting,
camera viewpoint, background, object layout) activations leave the calibrated ranges and
quantization error grows.

We study how quantized π0.5 (SmoothQuant W4A8, GPTQ W4, QVLA at a matched average budget)
performs on LIBERO-Plus, then test two training-free fixes that change only the calibration
data:

1. **On-policy calibration:** recalibrate on states the quantized policy actually visits.
2. **Out-of-distribution calibration:** recalibrate on perturbed observations.

All quantizers are reimplemented here on shared activation-capture hooks and checked against
their official implementations. Quantization is simulated (quantize-dequantize) rather than run
with integer kernels, since we measure task success, not latency.

## Layout

```
pyproject.toml      lightweight development env (Python 3.11+): vlacal and tests
sim/                sim env (Python 3.8): original LIBERO + openpi-client
third_party/        pinned OpenPI and its nested original LIBERO submodule
scripts/            setup and experiment entrypoints
src/vlacal/
  policy/           π0.5 loading, activation hooks, policy server
  quant/            fake-quant primitives, Quantizer API, SmoothQuant / GPTQ / QVLA
  calib/            clean / OOD / on-policy calibration sets
  eval/             per-episode JSONL records, aggregation
configs/            experiment configs (quantizer × calibration source × eval split)
tests/
```

The policy and the simulator run as separate processes in separate environments: a websocket
policy server in the model env, and a LIBERO rollout client in the sim env (the same setup
openpi uses for LIBERO).

Checkpoints, calibration sets, and results live under `$VLACAL_DATA_DIR` (default `./data`),
outside git.

## First milestone: unquantized PyTorch π0.5 on one LIBERO episode

See [the Cannon and Linux setup guide](docs/smoke-test.md). The runner uses the
official `pi05_libero` checkpoint, OpenPI's BF16/FP32 inference precision, and the
original LIBERO benchmark pinned by OpenPI. It runs exactly one task and initial
state, and saves an MP4, per-episode JSONL, server/client logs, and run provenance.
This is an integration smoke test, not a reproduction of the full benchmark score.

Validated on Cannon on 2026-10-28: `libero_spatial` task 0, initial state 0,
seed 7 succeeded in 78 action steps with 16 policy calls. See the
[recorded environment and run details](docs/cannon-validation.md).

Three isolated environments avoid the upstream dependency conflicts:

- `.venv/`: lightweight development and tests (Mac or Linux).
- `.venv-model/`: Python 3.11, pinned OpenPI PyTorch runtime (Linux x86_64).
- `sim/.venv/`: Python 3.8, pinned LIBERO simulator and websocket client (Linux x86_64).

The first milestone uses original LIBERO through the OpenPI submodule. LIBERO-Plus
and quantization methods are not implemented yet. Do not install LIBERO-Plus over
the ID simulator environment when reproducing this baseline.

## Development

```bash
uv run pytest
uv run ruff check .
```

The model environment uses OpenPI's committed `uv.lock`; the root and simulator
environments have their own lockfiles. `uv run` at the root is for development;
the launch scripts invoke the model and simulator Python executables explicitly.
