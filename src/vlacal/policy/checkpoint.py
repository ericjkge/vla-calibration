"""Small, dependency-free checkpoint checks shared by setup and serving."""

import json
import shutil
from pathlib import Path


def validate_checkpoint(directory: Path) -> None:
    weights = directory / "model.safetensors"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise FileNotFoundError(f"Missing PyTorch weights: {weights}; run prepare_checkpoint.sh")
    stats = list((directory / "assets").rglob("norm_stats.json"))
    if not stats:
        raise FileNotFoundError(f"Missing normalization assets in {directory / 'assets'}")
    for path in stats:
        json.loads(path.read_text())


def copy_normalization_assets(source: Path, destination: Path) -> None:
    # The pinned upstream converter searches source.parent/assets, but the
    # released pi05_libero checkpoint stores these in source/assets.
    assets = source / "assets"
    if not list(assets.rglob("norm_stats.json")):
        raise FileNotFoundError(f"Missing normalization statistics in {assets}")
    shutil.copytree(assets, destination / "assets", dirs_exist_ok=True)
