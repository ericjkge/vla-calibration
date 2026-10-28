"""Serve the unquantized OpenPI PyTorch pi05_libero policy on a local websocket."""

import argparse
import dataclasses
import json
import logging
import os
import random
from pathlib import Path

from vlacal.policy.checkpoint import validate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--denoising-steps", type=int, default=10)
    args = parser.parse_args()
    validate_checkpoint(args.checkpoint)
    if args.denoising_steps < 1:
        parser.error("--denoising-steps must be positive")
    os.environ["JAX_PLATFORMS"] = "cpu"
    import numpy as np
    import torch
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Run inside a GPU compute allocation.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_config = config.get_config("pi05_libero")
    # Eager inference avoids a long compilation warm-up for a single episode.
    train_config = dataclasses.replace(
        train_config,
        model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
    )
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        pytorch_device="cuda",
        sample_kwargs={"num_steps": args.denoising_steps},
    )
    if not policy._is_pytorch_model:
        raise RuntimeError("Refusing to serve a JAX policy in the PyTorch smoke test")
    metadata = {
        "model": "pi05_libero",
        "backend": "pytorch",
        "quantization": None,
        "precision": "OpenPI BF16/FP32",
        "seed": args.seed,
        "denoising_steps": args.denoising_steps,
        "action_horizon": train_config.model.action_horizon,
        "action_dim": 7,
        "checkpoint": str(args.checkpoint.resolve()),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    manifest = args.checkpoint / "vlacal_checkpoint.json"
    if manifest.is_file():
        metadata["checkpoint_provenance"] = json.loads(manifest.read_text())
    logging.info("Policy ready: %s", json.dumps(metadata))
    with torch.inference_mode():
        WebsocketPolicyServer(
            policy, host=args.host, port=args.port, metadata=metadata
        ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
