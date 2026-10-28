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
pyproject.toml      model env (Python 3.11): vlacal, openpi, torch
sim/                sim env (Python 3.8): LIBERO-Plus + openpi-client rollout client
third_party/        git submodules: openpi, LIBERO-Plus
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
policy server in the model env, and a LIBERO-Plus rollout client in the sim env (the same setup
openpi uses for LIBERO).

Checkpoints, calibration sets, and results live under `$VLACAL_DATA_DIR` (default `./data`),
outside git.

## Setup

> Not yet tested. Dependencies and submodules have not been added.

Requires [uv](https://docs.astral.sh/uv/).

```bash
git submodule update --init --recursive

# Model env (repo root)
uv sync

# Sim env
cd sim && uv sync
```

## Development

```bash
uv run pytest
uv run ruff check .
```
