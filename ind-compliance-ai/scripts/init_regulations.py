from datetime import datetime, timezone
from pathlib import Path


def init_regulations(base_dir: Path) -> Path:
    """Initialize regulation version registry for local corpora."""
    regulations_dir = base_dir / "data" / "regulations"
    regulations_dir.mkdir(parents=True, exist_ok=True)

    registry = regulations_dir / "registry.txt"
    if not registry.exists():
        registry.write_text(
            f"initialized_at={datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
    return registry


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    registry_file = init_regulations(root)
    print(f"Regulation registry ready: {registry_file}")
