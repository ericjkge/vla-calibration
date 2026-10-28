"""Apply the pinned OpenPI patch without mutating uv's cached hardlinks."""

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import tempfile


def patch_tree(source: Path, destination: Path) -> int:
    count = 0
    for source_file in sorted(source.rglob("*.py")):
        target = destination / source_file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replacement breaks any hardlink, even in an older environment.
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            stream.write(source_file.read_bytes())
            temporary = Path(stream.name)
        temporary.chmod(0o644)
        os.replace(temporary, target)
        count += 1
    if not count:
        raise RuntimeError(f"No patch files found in {source}")
    return count


def main() -> None:
    if importlib.metadata.version("transformers") != "4.53.2":
        raise RuntimeError("The pinned OpenPI patch requires transformers==4.53.2")
    spec = importlib.util.find_spec("transformers")
    if spec is None or spec.origin is None:
        raise RuntimeError("transformers is not installed")
    root = Path(__file__).resolve().parents[1]
    source = root / "third_party/openpi/src/openpi/models_pytorch/transformers_replace"
    count = patch_tree(source, Path(spec.origin).parent)
    print(f"Applied {count} OpenPI Transformers patch files")


if __name__ == "__main__":
    main()
