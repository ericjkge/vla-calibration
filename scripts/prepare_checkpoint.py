"""Download the official LIBERO checkpoint and convert it with pinned OpenPI."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile

from vlacal.policy.checkpoint import copy_normalization_assets, validate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        validate_checkpoint(output)
        print(f"Reusing existing PyTorch checkpoint: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 30 * 1024**3:
        raise RuntimeError("Need at least 30 GiB free for checkpoint download and conversion")
    os.environ["JAX_PLATFORMS"] = "cpu"
    from openpi.shared import download

    uri = "gs://openpi-assets/checkpoints/pi05_libero"
    source = Path(download.maybe_download(uri))
    root = Path(__file__).resolve().parents[1]
    upstream = root / "third_party/openpi"
    # Upstream conversion needs substantial CPU RAM, but no GPU allocation.
    converter = runpy.run_path(str(upstream / "examples/convert_jax_model_to_pytorch.py"))
    with tempfile.TemporaryDirectory(prefix="pi05-conversion-", dir=output.parent) as temporary:
        converted = Path(temporary)
        converter["main"](
            checkpoint_dir=str(source), config_name="pi05_libero",
            output_path=str(converted), precision="bfloat16",
        )
        copy_normalization_assets(source, converted)
        validate_checkpoint(converted)
        manifest = {
            "source": uri,
            "config": "pi05_libero",
            "precision": "bfloat16",
            "quantization": None,
            "openpi_commit": subprocess.check_output(
                ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
            ).strip(),
            "converted_at": datetime.now(timezone.utc).isoformat(),
        }
        (converted / "vlacal_checkpoint.json").write_text(json.dumps(manifest, indent=2))
        # Only expose the final path after weights and normalization validate.
        converted.rename(output)
    print(f"Prepared PyTorch checkpoint: {output}")


if __name__ == "__main__":
    main()
