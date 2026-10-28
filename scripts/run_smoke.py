"""Own both subprocesses, bound readiness/runtime, and retain logs on failure."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def wait_for_server(process, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Policy server exited with code {process.returncode}; see server.log"
            )
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
    raise TimeoutError("Policy server startup timed out; see server.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--episode-timeout", type=int, default=900)
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server_command = [
        sys.executable,
        "-m",
        "vlacal.policy.serve",
        "--checkpoint",
        str(args.checkpoint),
        "--port",
        str(port),
        "--seed",
        str(args.seed),
    ]
    client_command = [
        str(root / "sim/.venv/bin/python"),
        str(root / "sim/smoke_episode.py"),
        "--port",
        str(port),
        "--output-dir",
        str(output),
        "--suite",
        args.suite,
        "--task-id",
        str(args.task_id),
        "--initial-state-id",
        str(args.initial_state_id),
        "--seed",
        str(args.seed),
    ]
    if args.require_success:
        client_command.append("--require-success")
    manifest = {
        "started_at": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "server_command": server_command,
        "client_command": client_command,
        "status": "error",
    }
    server = None
    exit_code = 1
    try:
        with (
            (output / "server.log").open("w") as server_log,
            (output / "client.log").open("w") as client_log,
        ):
            server = subprocess.Popen(server_command, stdout=server_log, stderr=subprocess.STDOUT)
            wait_for_server(server, port, args.startup_timeout)
            result = subprocess.run(
                client_command,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                timeout=args.episode_timeout,
                check=False,
            )
            exit_code = result.returncode
            manifest["status"] = "completed" if exit_code in (0, 2) else "error"
            manifest["client_exit_code"] = exit_code
    except Exception as error:
        manifest["error"] = str(error)
        print(str(error), file=sys.stderr)
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        (output / "run.json").write_text(json.dumps(manifest, indent=2))
    print(f"Smoke-test artifacts: {output}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
