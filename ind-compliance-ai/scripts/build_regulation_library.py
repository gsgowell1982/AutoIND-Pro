from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.regulation_ingestion import write_regulation_corpus_entry  # noqa: E402


def _resolve_default_source() -> Path:
    regulations_root = PROJECT_ROOT / "data" / "regulations"
    matches = [
        path
        for path in regulations_root.iterdir()
        if path.suffix.lower() == ".doc" and not path.name.startswith("~$")
    ]
    implementation_matches = [path for path in matches if "药品管理法实施条例" in path.name]
    if implementation_matches:
        return implementation_matches[0]
    if matches:
        return max(matches, key=lambda path: len(path.name))
    raise FileNotFoundError("No regulation .doc source found under data/regulations.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build normalized regulation document/clause/rule-candidate artifacts."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional source regulation path. Defaults to the implementation regulation doc under data/regulations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "regulations" / "normalized",
        help="Output directory for generated regulation artifacts.",
    )
    parser.add_argument(
        "--draft-rules-dir",
        type=Path,
        default=PROJECT_ROOT / "rules" / "regulation_drafts",
        help="Output directory for generated direct-rule draft catalogs.",
    )
    args = parser.parse_args()

    source_path = args.source or _resolve_default_source()
    outputs = write_regulation_corpus_entry(
        source_path,
        output_root=args.output_dir,
        draft_rules_root=args.draft_rules_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
