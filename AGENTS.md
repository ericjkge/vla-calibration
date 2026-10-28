# AGENTS.md

Context for coding agents working in this repo. Living notes, not polished docs. Update as
decisions change.

## What this project is

CS 2420 final project: **Robust Calibration for VLA Quantization.** Project doc (Google Doc,
first tab): https://docs.google.com/document/d/1x8Kt_hxKpZRo6hrYKqQAChF64CAdLnRY0djOs9hBzmM/edit

- PTQ for VLAs is only evaluated in-distribution, where quantized policies look near-lossless.
- Hypothesis: standard PTQ calibrates on clean frames → clipping ranges / scales / smoothing
  factors / Hessians are fit to in-distribution activations → under distribution shift
  (lighting, camera, background, layout, ...) activations leave those ranges, outlier channels
  move, quantization error grows. Quantized policy also drifts into states clean demos don't cover.
- Plan: measure quantized π0.5 on LIBERO (ID) and LIBERO-Plus (OOD), then two training-free fixes
  that only change the **calibration data**:
  1. on-policy calibration: recalibrate on states the *quantized* policy visits
  2. OOD calibration: recalibrate on perturbed observations
- Calibration data is THE experimental variable. Everything else stays fixed.

## Setup decisions (settled)

- Model: π0.5 via **openpi**, using its **PyTorch** implementation (not JAX, not LeRobot's pi05).
  Need to verify the PyTorch LIBERO checkpoint reproduces published success rate before trusting
  anything downstream.
- Quantizers: **SmoothQuant (W4A8), GPTQ (W4), QVLA**.
  - It's **QVLA** (ICLR 2026, https://github.com/AutoLab-SAI-SJTU/QVLA), NOT QuantVLA. The
    project doc changed from QuantVLA to QVLA. Old notes may still say QuantVLA.
  - QVLA is mixed precision → constrain it to a **fixed average budget** matching the others
    (avg 4-bit weights, 8-bit activations) so robustness gaps aren't just "more bits".
- **All quantizers are reimplemented from scratch** on our own activation-capture hooks. Do not
  pull in AutoGPTQ / GPTQModel / llm-compressor / QVLA's code as dependencies. DO cross-check
  against the official repos (tests comparing our output to theirs on toy layers):
  - SmoothQuant: https://github.com/mit-han-lab/smoothquant
  - GPTQ: https://github.com/IST-DASLab/gptq
  - QVLA: https://github.com/AutoLab-SAI-SJTU/QVLA (written for OpenVLA, port the idea)
- **Fake quantization** (quantize → dequantize in float). No int kernels. We measure task
  success, not latency.
- Keep small, sensitive parts at full precision (projectors, action in/out projections, etc.),
  like the VLA quant papers do. Exact layer list TBD.
- **Policy server / sim client split** (same as openpi's LIBERO example): websocket policy server
  in the model env, LIBERO-Plus rollout client in a separate env using `openpi-client`. Reason:
  LIBERO pins old robosuite/MuJoCo/numpy that conflict with torch/openpi.
- Results: local **per-episode JSONL** + aggregation script. No W&B.
- Large files (checkpoints, calib sets, results) under `$VLACAL_DATA_DIR` (default `./data`,
  gitignored). On RunPod this can point at a network volume.

## Repo layout / tooling

- **uv** for envs. `pyproject.toml` + `uv.lock` (lock not generated yet).
  - root = model env, Python 3.11, package `vlacal` (src layout: `src/vlacal/`)
  - `sim/` = sim env, Python 3.8, its own `pyproject.toml`, not a package
- `src/vlacal/policy` π0.5 loading, hooks, server · `quant` fake-quant primitives + Quantizer API
  + methods · `calib` clean/OOD/on-policy calib sets · `eval` JSONL records, aggregation
- `third_party/` → git submodules (openpi, LIBERO-Plus), not added yet
- `configs/` → experiment configs (quantizer × calib source × eval split)
- `scripts/` → entrypoints + `setup_model_env.sh` (openpi PyTorch needs extra setup steps,
  something like patching installed transformers + converting JAX ckpt; check openpi README)
- `uv run pytest`, `uv run ruff check .` (ruff excludes third_party)
- Mac is fine for unit tests on toy layers. Anything loading π0.5 or doing real rollouts → Linux
  GPU (RunPod A100 likely).

## Rough intended design (not built yet)

- `Quantizer.calibrate(model, calib_batches) -> model` common interface; the three calib
  strategies are just three data sources feeding it.
- On-policy calib = loop: quantize → roll out → collect visited states → recalibrate.
- Calib set size per doc: ~512 trajectories × 4–8 frames.
- Inputs: third-person image, wrist image, language instruction, robot state. Output: action
  chunk of 7D actions.

## Status (2026-10-28)

- First milestone implementation is in `docs/smoke-test.md`; check that guide and
  recorded run artifacts for runtime validation status rather than assuming success.
- OpenPI is pinned in `third_party/openpi`; original LIBERO is its nested submodule.
  LIBERO-Plus is deferred until the original ID smoke test is established.
- Root `.venv` is lightweight development; `.venv-model` uses OpenPI's frozen lock
  (excluding unused rerun-sdk for Cannon glibc compatibility); `sim/.venv` uses its
  own Python 3.8 lock. Setup and run scripts invoke explicit Python executables.
- Cannon is the initial runtime. Setup/conversion are CPU jobs, inference is a full
  A100 GPU job. Keep RepoDisco's conda environment independent.
- Next after validating the FP episode: fake-quant primitives + Quantizer API
  (+ RTN as sanity baseline) → SmoothQuant → GPTQ → QVLA.

## Open / TBD

- LIBERO-Plus repo link for the submodule (make sure it's the right one).
- LIBERO-Plus calib/test split: has to be defined by us. Leaning toward holding out whole
  perturbation types to avoid leakage, report seen-type vs unseen-type. Decide later.
- Compute provider details (RunPod?), decide later.
- Exact set of layers kept at FP.

## Background notes (from the doc)

- π0.5 = PaliGemma (SigLIP + Gemma 2B) + flow-matching action expert.
- SmoothQuant: per-channel s_j migrates activation outliers into weights; s_j fixed at runtime →
  likely sensitive to shift if outlier channels change.
- GPTQ: column-by-column quantization with Hessian-based error compensation; Hessian comes from
  calib inputs.
- QVLA: action-sensitivity-driven per-channel bit allocation + activation scales; calib data
  decides which channels get bits → allocation from clean data might protect the wrong channels
  under shift (interesting angle for the writeup).
- LIBERO: robosuite/MuJoCo, 4 suites × 10 tasks (Spatial, Object, Goal, Long).
- LIBERO-Plus: same 40 tasks + 7 perturbation types (camera, robot init state, language,
  lighting, background texture, sensor noise, object layout).
- Similar paper: https://github.com/ruanruan-andy/PIVOT-Q (not yet checked for reusable code).
