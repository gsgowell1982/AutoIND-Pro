from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any


ENGINE_NAME = "autoind-pro-smoke"
BENCHMARK_NEUTRAL_ENGINE_NAME = "autoind-pro-benchmark-neutral"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_benchmark_root() -> Path:
    return _repo_root().parent / "external_benchmarks" / "opendataloader-bench"


def _copy_subset(src_dir: Path, dst_dir: Path, doc_ids: list[str], suffix: str) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for doc_id in doc_ids:
        src = src_dir / f"{doc_id}{suffix}"
        if not src.exists():
            raise FileNotFoundError(f"Missing benchmark source file: {src}")
        shutil.copy2(src, dst_dir / src.name)


def _strip_autoind_wrapper(markdown: str, filename: str) -> str:
    lines = markdown.splitlines()
    start_index = 0
    for index, line in enumerate(lines):
        if line.startswith(f"## {filename} "):
            start_index = index + 1
            break
    body = lines[start_index:]
    while body and not body[0].strip():
        body.pop(0)
    while body and re_match_metadata_bullet(body[0]):
        body.pop(0)
    while body and not body[0].strip():
        body.pop(0)
    if body and body[0].strip().startswith("### Table of Contents"):
        body.pop(0)
        while body and body[0].strip():
            body.pop(0)
        while body and not body[0].strip():
            body.pop(0)
    filtered: list[str] = []
    for line in body:
        stripped = line.strip()
        if stripped in {"### 姝ｆ枃缁撴瀯鍖栧唴瀹?", "### Document Body"}:
            continue
        if stripped.startswith("![") and "](data:image/" in stripped:
            continue
        filtered.append(line)
    return "\n".join(filtered).strip() + "\n"


def re_match_metadata_bullet(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("- Estimated pages:") or stripped.startswith("- Parser strategy:")


def _parse_one_pdf(pdf_path: Path, *, projection: str = "autoind-body") -> tuple[str, float, str | None]:
    start = time.perf_counter()
    try:
        repo_root = str(_repo_root())
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from parsers.pdf_parser import parse_pdf

        previous_ocr_profile = os.environ.get("PDF_OCR_RUNTIME_PROFILE")
        if projection == "benchmark-neutral" and not previous_ocr_profile:
            os.environ["PDF_OCR_RUNTIME_PROFILE"] = "benchmark_image_evidence"
        try:
            parsed = parse_pdf(pdf_path)
        finally:
            if projection == "benchmark-neutral" and not previous_ocr_profile:
                os.environ.pop("PDF_OCR_RUNTIME_PROFILE", None)
        parsed.setdefault("filename", pdf_path.name)
        parsed.setdefault("source_type", "pdf")
        if projection == "benchmark-neutral":
            from scripts.opendataloader_benchmark_projection import (
                build_opendataloader_benchmark_markdown,
            )

            markdown = build_opendataloader_benchmark_markdown(parsed)
        else:
            from api.main import _build_document_body_markdown_sections

            markdown = "\n".join(
                _build_document_body_markdown_sections(
                    parsed,
                    embed_images=False,
                    table_export_mode="semantic_html",
                )
            )
            markdown = _strip_autoind_wrapper(markdown, pdf_path.name)
        elapsed = time.perf_counter() - start
        return markdown, elapsed, None
    except Exception as exc:  # pragma: no cover - smoke-report guard
        elapsed = time.perf_counter() - start
        return "", elapsed, f"{type(exc).__name__}: {exc}"


def _write_summary(
    engine_dir: Path,
    *,
    engine_name: str,
    projection: str,
    document_count: int,
    dataset_size: int,
    total_elapsed: float,
    failures: dict[str, str],
) -> None:
    scope = (
        "OpenDataLoader Bench full local dataset run."
        if document_count >= dataset_size
        else "OpenDataLoader Bench smoke subset only; not a full benchmark run."
    )
    summary = {
        "engine_name": engine_name,
        "engine_version": "current-autoind-pro-worktree",
        "projection": projection,
        "document_count": document_count,
        "total_elapsed": total_elapsed,
        "elapsed_per_doc": total_elapsed / document_count if document_count else None,
        "failed_documents": failures,
        "date": time.strftime("%Y-%m-%d"),
        "scope": scope,
    }
    (engine_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_evaluator(benchmark_root: Path, ground_truth_dir: Path, prediction_root: Path, *, engine_name: str) -> Path:
    validate_output_dir_under_benchmark_root(benchmark_root, ground_truth_dir)
    validate_output_dir_under_benchmark_root(benchmark_root, prediction_root)
    command = [
        "uv",
        "run",
        "src/evaluator.py",
        "--ground-truth-dir",
        str(ground_truth_dir.relative_to(benchmark_root)),
        "--prediction-root",
        str(prediction_root.relative_to(benchmark_root)),
        "--engine",
        engine_name,
        "--output-filename",
        "evaluation.smoke.json",
        "--log-level",
        "INFO",
    ]
    subprocess.run(command, cwd=benchmark_root, check=True)
    return prediction_root / engine_name / "evaluation.smoke.json"


def validate_output_dir_under_benchmark_root(benchmark_root: Path, output_dir: Path) -> None:
    resolved_benchmark_root = benchmark_root.resolve()
    resolved_output_dir = output_dir.resolve()
    try:
        resolved_output_dir.relative_to(resolved_benchmark_root)
    except ValueError as exc:
        raise ValueError(
            f"OpenDataLoader smoke output must be under --benchmark-root: "
            f"output={resolved_output_dir}, benchmark_root={resolved_benchmark_root}"
        ) from exc


def _load_eval(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_value(doc: dict[str, Any], key: str) -> float | None:
    value = (doc.get("scores") or {}).get(key)
    return float(value) if isinstance(value, int | float) else None


def _lowest_documents(eval_payload: dict[str, Any], metric: str, limit: int = 5) -> list[dict[str, Any]]:
    docs = []
    for doc in eval_payload.get("documents", []) or []:
        value = _score_value(doc, metric)
        if value is not None:
            docs.append({"document_id": doc.get("document_id"), metric: value})
    docs.sort(key=lambda item: item[metric])
    return docs[:limit]


def _markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _write_report(
    report_path: Path,
    *,
    doc_ids: list[str],
    eval_payload: dict[str, Any],
    failures: dict[str, str],
    output_dir: Path,
    elapsed: float,
    dataset_size: int,
    engine_name: str,
    projection: str,
) -> None:
    full_run = len(doc_ids) >= dataset_size
    title = (
        "OpenDataLoader Bench Full Evaluation - AutoIND-Pro"
        if full_run
        else "OpenDataLoader Bench Smoke Evaluation - AutoIND-Pro"
    )
    scope_line = (
        "This is the full local benchmark dataset run."
        if full_run
        else "This is a smoke subset report, not the full 200-document benchmark."
    )
    score = ((eval_payload.get("metrics") or {}).get("score") or {})
    counts = eval_payload.get("metrics") or {}
    rows = [
        ["Metric", "Value"],
        ["overall_mean", _fmt(score.get("overall_mean"))],
        ["nid_mean", _fmt(score.get("nid_mean"))],
        ["nid_s_mean", _fmt(score.get("nid_s_mean"))],
        ["teds_mean", _fmt(score.get("teds_mean"))],
        ["teds_s_mean", _fmt(score.get("teds_s_mean"))],
        ["mhs_mean", _fmt(score.get("mhs_mean"))],
        ["mhs_s_mean", _fmt(score.get("mhs_s_mean"))],
        ["nid_count", str(counts.get("nid_count"))],
        ["teds_count", str(counts.get("teds_count"))],
        ["mhs_count", str(counts.get("mhs_count"))],
        ["missing_predictions", str(counts.get("missing_predictions"))],
        ["total_smoke_elapsed_seconds", f"{elapsed:.2f}"],
        ["projection", projection],
    ]
    low_rows = [["Metric", "Lowest documents"]]
    for metric in ("overall", "nid", "teds", "mhs"):
        items = _lowest_documents(eval_payload, metric, limit=5)
        low_rows.append(
            [
                metric,
                ", ".join(f"{item['document_id']}={item[metric]:.3f}" for item in items),
            ]
        )
    failure_lines = ["None"] if not failures else [
        f"- `{doc_id}`: {message}" for doc_id, message in sorted(failures.items())
    ]
    report = f"""# {title}

Generated: `{time.strftime("%Y-%m-%d %H:%M:%S")}`

Scope:

- Dataset: OpenDataLoader Bench local checkout.
- Engine label: `{engine_name}`.
- Projection: `{projection}`.
- Sample count: `{len(doc_ids)}`.
- Documents: `{", ".join(doc_ids)}`.
- {scope_line}

## Aggregate Metrics

{_markdown_table(rows)}

## Lowest Scoring Samples

{_markdown_table(low_rows)}

## Parse Failures

{chr(10).join(failure_lines)}

## Artifacts

- Prediction markdown: `{output_dir / "prediction-smoke" / engine_name / "markdown"}`
- Evaluation JSON: `{output_dir / "prediction-smoke" / engine_name / "evaluation.smoke.json"}`
- Evaluation CSV: `{output_dir / "prediction-smoke" / engine_name / "evaluation.smoke.csv"}`
"""
    report_path.write_text(report, encoding="utf-8")


def _fmt(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, int | float) else "null"


def _write_document_csv(csv_path: Path, eval_payload: dict[str, Any]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["document_id", "overall", "nid", "nid_s", "teds", "teds_s", "mhs", "mhs_s"])
        for doc in eval_payload.get("documents", []) or []:
            scores = doc.get("scores") or {}
            writer.writerow([
                doc.get("document_id"),
                scores.get("overall"),
                scores.get("nid"),
                scores.get("nid_s"),
                scores.get("teds"),
                scores.get("teds_s"),
                scores.get("mhs"),
                scores.get("mhs_s"),
            ])


def run(args: argparse.Namespace) -> Path:
    benchmark_root = args.benchmark_root.resolve()
    output_dir = args.output_dir.resolve()
    validate_output_dir_under_benchmark_root(benchmark_root, output_dir)
    projection = str(getattr(args, "projection", "autoind-body") or "autoind-body")
    engine_name = BENCHMARK_NEUTRAL_ENGINE_NAME if projection == "benchmark-neutral" else ENGINE_NAME
    available_doc_ids = [path.stem for path in sorted((benchmark_root / "pdfs").glob("*.pdf"))]
    if not available_doc_ids:
        raise FileNotFoundError(f"No benchmark PDFs found under {benchmark_root / 'pdfs'}")
    requested_doc_ids = _requested_doc_ids(str(getattr(args, "doc_ids", "") or ""))
    if requested_doc_ids:
        available = set(available_doc_ids)
        missing = [doc_id for doc_id in requested_doc_ids if doc_id not in available]
        if missing:
            raise FileNotFoundError(f"Requested benchmark document IDs not found: {', '.join(missing)}")
        doc_ids = requested_doc_ids
    else:
        doc_ids = available_doc_ids[: args.count]
    dataset_size = len(available_doc_ids)

    pdf_subset = output_dir / "pdfs-smoke"
    gt_subset = output_dir / "ground-truth-smoke" / "markdown"
    prediction_root = output_dir / "prediction-smoke"
    engine_dir = prediction_root / engine_name
    markdown_dir = engine_dir / "markdown"

    _copy_subset(benchmark_root / "pdfs", pdf_subset, doc_ids, ".pdf")
    _copy_subset(benchmark_root / "ground-truth" / "markdown", gt_subset, doc_ids, ".md")
    if prediction_root.exists():
        shutil.rmtree(prediction_root)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    failures: dict[str, str] = {}
    start = time.perf_counter()
    for doc_id in doc_ids:
        pdf_path = pdf_subset / f"{doc_id}.pdf"
        markdown, elapsed, error = _parse_one_pdf(pdf_path, projection=projection)
        if error:
            failures[doc_id] = error
        (markdown_dir / f"{doc_id}.md").write_text(markdown, encoding="utf-8")
        print(f"{doc_id}: {elapsed:.2f}s" + (f" ERROR {error}" if error else ""))
    total_elapsed = time.perf_counter() - start
    _write_summary(
        engine_dir,
        engine_name=engine_name,
        projection=projection,
        document_count=len(doc_ids),
        dataset_size=dataset_size,
        total_elapsed=total_elapsed,
        failures=failures,
    )

    eval_path = _run_evaluator(benchmark_root, gt_subset, prediction_root, engine_name=engine_name)
    eval_payload = _load_eval(eval_path)
    report_name = "autoind_pro_full_report.md" if len(doc_ids) >= dataset_size else "autoind_pro_smoke_report.md"
    report_path = output_dir / report_name
    _write_report(
        report_path,
        doc_ids=doc_ids,
        eval_payload=eval_payload,
        failures=failures,
        output_dir=output_dir,
        elapsed=total_elapsed,
        dataset_size=dataset_size,
        engine_name=engine_name,
        projection=projection,
    )
    _write_document_csv(output_dir / "autoind_pro_smoke_documents.csv", eval_payload)
    return report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, default=_default_benchmark_root())
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--doc-ids",
        default="",
        help="Comma-separated benchmark document IDs. When provided, overrides --count.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_benchmark_root() / "autoind-smoke-output",
    )
    parser.add_argument(
        "--projection",
        choices=("autoind-body", "benchmark-neutral"),
        default="autoind-body",
        help="Evaluation-only output projection. benchmark-neutral is not production IND Markdown.",
    )
    return parser.parse_args()


def _requested_doc_ids(value: str) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for item in str(value or "").split(","):
        doc_id = item.strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
    return doc_ids


def main() -> None:
    report_path = run(_parse_args())
    print(f"REPORT {report_path}")


if __name__ == "__main__":
    main()
