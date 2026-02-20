from pathlib import Path


def build_index(regulations_dir: Path, output_dir: Path) -> dict[str, str]:
    """Create a deterministic index manifest for regulation files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    indexed_files = sorted(path.name for path in regulations_dir.glob('*') if path.is_file())
    return {
        "regulations_dir": str(regulations_dir),
        "index_output": str(output_dir),
        "indexed_files": ",".join(indexed_files),
    }
