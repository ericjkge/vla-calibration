# PyTorch π0.5 / LIBERO smoke test

The target is one completed episode using the released `pi05_libero` policy,
without quantization. Full precision here means OpenPI's standard BF16/FP32
inference, not all-FP32. A single successful episode does not validate the
published 96.85% benchmark average.

## Cannon

Use the FASRC VPN and your usual web-portal shell, or authenticate from a terminal:

```bash
ssh sophiezeng@login.rc.fas.harvard.edu
```

The project is independent of RepoDisco's `repodisco` conda environment. Do not
install these dependencies into that environment. Clone the repo into a location
with at least 60 GiB free for environments, caches, simulator assets and checkpoints.
Large data defaults to `./data`; set `VLACAL_DATA_DIR` to suitable shared storage
before submitting jobs if desired. Keep that setting for both setup and evaluation.

```bash
git clone https://github.com/ericjkge/vla-calibration.git
cd vla-calibration
mkdir -p data/logs
# Install uv from its official distribution if it is not already on PATH.
curl -LsSf https://astral.sh/uv/0.13.6/install.sh | \
  env UV_INSTALL_DIR="$PWD/data/tools" UV_NO_MODIFY_PATH=1 sh
export PATH="$PWD/data/tools:$PATH"

# CPU job: environments, official checkpoint download, conversion.
setup_job=$(sbatch --parsable scripts/cannon_setup.sbatch)
# GPU job starts only if setup succeeds; defaults to a full A100 on gpu.
sbatch --dependency="afterok:$setup_job" scripts/cannon_smoke.sbatch
squeue -u "$USER"
```

The setup job requests 96 GiB CPU RAM because checkpoint conversion temporarily
holds multiple copies of weights. Evaluation requests one GPU, 8 CPUs, 64 GiB
host RAM, and 45 minutes. Adjust partition/account flags to your allocation.
Cannon's `gpu_test` uses A100 MIG slices; use a full GPU for the initial EGL run.
Both policy server and sim client run on the same compute node over loopback.
The server binds only to `127.0.0.1`; no public port or SSH tunnel is needed.

```bash
tail -f data/logs/setup-<job-id>.log
tail -f data/logs/smoke-<job-id>.log
sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

## Existing Linux GPU machine / interactive allocation

Run package installation and checkpoint conversion on a compute node, not a
Cannon login node. In an interactive allocation:

```bash
# On Cannon, provide CMake for the simulator's egl-probe extension.
module load cmake/3.28.3-fasrc01
bash scripts/setup_model_env.sh
bash scripts/setup_sim_env.sh
bash scripts/prepare_checkpoint.sh
bash scripts/run_smoke.sh --require-success
```

The default episode is `libero_spatial`, task 0, initial state 0, seed 7. Options:

```bash
bash scripts/run_smoke.sh --suite libero_object --task-id 2 --initial-state-id 0
```

The runner uses 256px renders, rotates both views 180 degrees, resizes/pads to
224px, constructs the upstream 8D state, requests 10-action chunks and executes
five actions before replanning. It waits 10 settling steps and uses the upstream
suite-specific episode horizon. No demonstration dataset is needed for this test.

## Outputs and failure handling

Each run creates `$VLACAL_DATA_DIR/results/smoke-<UTC timestamp>-<pid>/`:

- `episodes.jsonl`: one episode, task/seed, success, steps, policy metadata and errors.
- `episode.mp4`: front-camera replay, if rendering started.
- `server.log` and `client.log`: full subprocess output.
- `run.json`: commands, repository commit, Slurm job ID and process status.

Exit 0 means a completed episode (with `--require-success`, also task success).
Exit 2 means a completed episode that did not succeed when success was required.
Other failures are errors, not recorded as normal task failures. Server startup
and episode execution have bounded timeouts (20 and 15 minutes, respectively);
the owned server is stopped on exit. The initial policy-import check took about
3.5 minutes on Cannon's shared home filesystem, so the first load needs headroom.
Never count a timeout or import error as a valid benchmark failure.

## Compatibility decisions

- OpenPI is pinned as a submodule; its nested original LIBERO revision is pinned too.
- The model environment follows upstream's lock, excluding only `rerun-sdk`, an
  unused LeRobot visualization dependency whose wheel requires glibc 2.31.
  Cannon has glibc 2.28. This environment is for inference, not LeRobot visualization.
- Transformers 4.53.2 receives the pinned OpenPI patches by atomic file replacement,
  with uv copy mode, to avoid modifying hardlinked cache files.
- The converter explicitly copies `checkpoint/assets`, correcting the pinned
  converter's parent-directory asset lookup. Original normalization is preserved.
- The simulator uses CPU Torch 1.11 for reading initial states; PyTorch CUDA 2.7.1
  lives only in the model environment. MuJoCo 2.3.7 / robosuite 1.4.1 are pinned.
- JAX is CPU-only at runtime; the policy requires a PyTorch checkpoint and CUDA.
  Eager inference avoids compilation startup overhead for this small smoke test.
- EGL needs NVIDIA rendering libraries on the compute node. If EGL initialization
  fails, inspect `client.log`; `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` is an
  explicit CPU-rendering fallback only where the OSMesa system library is installed.

## Local verification

```bash
uv sync --locked
uv run pytest
uv run ruff check .
```

Tests cover image/state preprocessing, action validity, settling/replanning/horizon
semantics, error propagation, checkpoint assets, and cache-safe patch application.
They do not establish GPU model correctness or simulator rendering availability.
