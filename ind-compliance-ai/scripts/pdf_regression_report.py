# Version: v1.0.11
# Optimization Summary:
# - Batch PDF regression report generator for parser quality auditing.
# - Produces per-document metrics and aggregate summary in JSON.
# - Supports directory scan, glob filtering, and deterministic metric fields.
# - Adds Markdown executive summary output for fast human review.
# - Include possible-missing-content diagnostic counters.

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parsers.pdf_parser import parse_pdf


def _discover_pdfs(input_path: Path, glob_pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".pdf" else []
    if not input_path.exists():
        return []
    return sorted([p for p in input_path.rglob(glob_pattern) if p.is_file() and p.suffix.lower() == ".pdf"])


def _table_conf_stats(table_asts: list[dict[str, Any]]) -> dict[str, float | None]:
    confs = [float(t.get("confidence", 0.0) or 0.0) for t in table_asts if t.get("confidence") is not None]
    if not confs:
        return {"min": None, "avg": None, "max": None}
    return {
        "min": round(min(confs), 3),
        "avg": round(sum(confs) / len(confs), 3),
        "max": round(max(confs), 3),
    }


def _doc_record(pdf_path: Path) -> dict[str, Any]:
    result = parse_pdf(pdf_path)
    metadata = result.get("metadata", {})
    table_asts = result.get("table_asts", []) or []

    return {
        "file": str(pdf_path),
        "page_count": int(metadata.get("page_count", 0) or 0),
        "table_count": int(metadata.get("table_count", 0) or 0),
        "raw_table_candidates": int(metadata.get("raw_table_candidates", 0) or 0),
        "accepted_table_candidates": int(metadata.get("accepted_table_candidates", 0) or 0),
        "rejected_table_candidates": int(metadata.get("rejected_table_candidates", 0) or 0),
        "continuation_table_count": int(metadata.get("continuation_table_count", 0) or 0),
        "cross_page_table_links": int(metadata.get("cross_page_table_links", 0) or 0),
        "review_required_table_count": int(metadata.get("review_required_table_count", 0) or 0),
        "low_confidence_table_count": int(metadata.get("low_confidence_table_count", 0) or 0),
        "possible_missing_content_table_count": int(metadata.get("possible_missing_content_table_count", 0) or 0),
        "possible_missing_content_candidate_count": int(metadata.get("possible_missing_content_candidate_count", 0) or 0),
        "continuation_similarity_avg": metadata.get("continuation_similarity_avg"),
        "continuation_similarity_min": metadata.get("continuation_similarity_min"),
        "continuation_similarity_max": metadata.get("continuation_similarity_max"),
        "table_confidence_stats": _table_conf_stats(table_asts),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "documents": 0,
            "total_pages": 0,
            "total_tables": 0,
            "total_raw_candidates": 0,
            "total_accepted_candidates": 0,
            "total_rejected_candidates": 0,
            "total_cross_page_links": 0,
            "total_review_required_tables": 0,
            "total_low_confidence_tables": 0,
            "acceptance_rate": None,
        }

    total_raw = sum(r["raw_table_candidates"] for r in records)
    total_accepted = sum(r["accepted_table_candidates"] for r in records)
    return {
        "documents": len(records),
        "total_pages": sum(r["page_count"] for r in records),
        "total_tables": sum(r["table_count"] for r in records),
        "total_raw_candidates": total_raw,
        "total_accepted_candidates": total_accepted,
        "total_rejected_candidates": sum(r["rejected_table_candidates"] for r in records),
        "total_cross_page_links": sum(r["cross_page_table_links"] for r in records),
        "total_review_required_tables": sum(r["review_required_table_count"] for r in records),
        "total_low_confidence_tables": sum(r["low_confidence_table_count"] for r in records),
        "total_possible_missing_content_tables": sum(r["possible_missing_content_table_count"] for r in records),
        "total_possible_missing_content_candidates": sum(r["possible_missing_content_candidate_count"] for r in records),
        "acceptance_rate": round(total_accepted / total_raw, 4) if total_raw else None,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    docs = report.get("documents", [])
    lines: list[str] = []
    lines.append("# PDF Regression Report")
    lines.append("")
    lines.append(f"- Generated At: `{report.get('generated_at')}`")
    lines.append(f"- Input: `{report.get('input')}`")
    lines.append(f"- Glob: `{report.get('glob')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Documents: `{summary.get('documents')}`")
    lines.append(f"- Total Pages: `{summary.get('total_pages')}`")
    lines.append(f"- Total Tables: `{summary.get('total_tables')}`")
    lines.append(f"- Raw Candidates: `{summary.get('total_raw_candidates')}`")
    lines.append(f"- Accepted Candidates: `{summary.get('total_accepted_candidates')}`")
    lines.append(f"- Rejected Candidates: `{summary.get('total_rejected_candidates')}`")
    lines.append(f"- Cross-Page Links: `{summary.get('total_cross_page_links')}`")
    lines.append(f"- Low-Confidence Tables: `{summary.get('total_low_confidence_tables')}`")
    lines.append(f"- Possible Missing Content Tables: `{summary.get('total_possible_missing_content_tables')}`")
    lines.append(f"- Missing Content Candidates: `{summary.get('total_possible_missing_content_candidates')}`")
    lines.append(f"- Acceptance Rate: `{summary.get('acceptance_rate')}`")
    lines.append("")
    lines.append("## Documents")
    lines.append("")
    lines.append("| File | Pages | Tables | Accepted/Raw | Rejected | CrossPage | LowConf | MissingTbl | MissingCand | ContSimAvg |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in docs:
        accepted = item.get("accepted_table_candidates")
        raw = item.get("raw_table_candidates")
        ratio = f"{accepted}/{raw}"
        lines.append(
            f"| {item.get('file')} | {item.get('page_count')} | {item.get('table_count')} | "
            f"{ratio} | {item.get('rejected_table_candidates')} | {item.get('cross_page_table_links')} | "
            f"{item.get('low_confidence_table_count')} | {item.get('possible_missing_content_table_count')} | "
            f"{item.get('possible_missing_content_candidate_count')} | {item.get('continuation_similarity_avg')} |"
        )

    errors = report.get("errors", [])
    if errors:
        lines.append("")
        lines.append("## Errors")
        lines.append("")
        for e in errors:
            lines.append(f"- `{e.get('file')}`: {e.get('error')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate batch PDF parser regression report.")
    parser.add_argument("--input", default=str(PROJECT_ROOT), help="PDF file or directory path.")
    parser.add_argument("--glob", default="*.pdf", help="Glob pattern used when input is a directory.")
    parser.add_argument("--output", default="", help="Output JSON path. Default writes to output/reports.")
    parser.add_argument("--markdown-output", default="", help="Optional output Markdown path.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    pdfs = _discover_pdfs(input_path, args.glob)
    if not pdfs:
        print(f"No PDF files found under: {input_path}")
        return 1

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for pdf_path in pdfs:
        try:
            records.append(_doc_record(pdf_path))
            print(f"OK: {pdf_path.name}")
        except Exception as exc:  # pragma: no cover - runtime safety
            errors.append({"file": str(pdf_path), "error": str(exc)})
            print(f"FAIL: {pdf_path.name} -> {exc}")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "glob": args.glob,
        "summary": _aggregate(records),
        "documents": records,
        "errors": errors,
    }

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT_ROOT / "output" / "reports" / f"pdf_regression_report_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written to: {out_path}")

    if args.markdown_output:
        md_path = Path(args.markdown_output).resolve()
    else:
        md_path = out_path.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    print(f"Markdown summary written to: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
