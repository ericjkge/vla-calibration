"""Run exactly one LIBERO task/initial-state pair and write JSONL plus an MP4."""

import argparse
import json
import os
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from rollout import MAX_STEPS, run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--suite", choices=MAX_STEPS, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "task_id": args.task_id,
        "initial_state_id": args.initial_state_id,
        "seed": args.seed,
        "replan_steps": args.replan_steps,
        "max_steps": MAX_STEPS[args.suite],
        "status": "error",
        "success": None,
        "renderer": os.environ.get("MUJOCO_GL"),
    }
    env = None
    writer = None
    started = time.monotonic()
    exit_code = 1
    try:
        import imageio.v2 as imageio
        import numpy as np
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from openpi_client import image_tools
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        random.seed(args.seed)
        np.random.seed(args.seed)
        client = WebsocketClientPolicy(args.host, args.port)
        metadata = client.get_server_metadata()
        if (
            metadata.get("model") != "pi05_libero"
            or metadata.get("backend") != "pytorch"
            or "quantization" not in metadata
            or metadata["quantization"] is not None
        ):
            raise ValueError("Expected the unquantized vlacal pi05_libero PyTorch server")
        record["policy"] = metadata
        suite = benchmark.get_benchmark_dict()[args.suite]()
        if not 0 <= args.task_id < suite.n_tasks:
            raise ValueError("task-id is outside this suite")
        task = suite.get_task(args.task_id)
        initial_states = suite.get_task_init_states(args.task_id)
        if not 0 <= args.initial_state_id < len(initial_states):
            raise ValueError("initial-state-id is outside this task's initial states")
        record.update(task=task.name, instruction=task.language)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        video = args.output_dir / "episode.mp4"
        writer = imageio.get_writer(str(video), fps=10)

        def resize(image):
            return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))

        result = run_episode(
            env,
            client,
            initial_states[args.initial_state_id],
            task.language,
            resize,
            max_steps=MAX_STEPS[args.suite],
            replan_steps=args.replan_steps,
            on_frame=writer.append_data,
        )
        writer.close()
        writer = None
        record.update(result, status="completed", video=str(video.resolve()))
        exit_code = 2 if args.require_success and not result["success"] else 0
    except Exception:
        record["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        try:
            if writer is not None:
                writer.close()
            if env is not None:
                env.close()
        except Exception:
            record.update(status="error", error=traceback.format_exc())
            exit_code = 1
        record["elapsed_seconds"] = time.monotonic() - started
        with (args.output_dir / "episodes.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
