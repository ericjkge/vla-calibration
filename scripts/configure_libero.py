"""Write a project-specific LIBERO config before LIBERO's interactive first import."""

import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    benchmark = root / "third_party/openpi/third_party/libero/libero/libero"
    data = Path(os.environ.get("VLACAL_DATA_DIR", root / "data")).resolve()
    paths = {
        "benchmark_root": benchmark,
        "bddl_files": benchmark / "bddl_files",
        "init_states": benchmark / "init_files",
        "datasets": data / "datasets",
        "assets": benchmark / "assets",
    }
    for key in ("bddl_files", "init_states", "assets"):
        if not paths[key].is_dir():
            raise FileNotFoundError(f"Missing {key}: {paths[key]}; initialize the LIBERO submodule")
    paths["datasets"].mkdir(parents=True, exist_ok=True)
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", data / "libero-config"))
    config_dir.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML and requires no parser just to configure paths.
    target = config_dir / "config.yaml"
    target.write_text(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    print(f"LIBERO config: {target}")


if __name__ == "__main__":
    main()
