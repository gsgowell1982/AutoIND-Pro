from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pymupdf
except Exception:  # pragma: no cover - environment guard
    pymupdf = None  # type: ignore[assignment]

from parsers.pdf.raster_table_evidence import (
    bbox_intersection_over_min as _bbox_intersection_over_min,
    bbox_iou as _bbox_iou,
    detect_raster_table_regions as _detect_raster_table_regions,
    merge_aligned_text_matrix_regions as _merge_aligned_text_matrix_regions,
)

ENGINE_NAME = "autoind-pro"


def summarize_omnidocbench_metrics(metric_payload: dict[str, Any]) -> dict[str, Any]:
    """Return the AutoIND-prioritized OmniDocBench metric view."""
    table_all = (((metric_payload.get("table") or {}).get("all") or {}))
    formula_all = (((metric_payload.get("display_formula") or {}).get("all") or {}))
    reading_all = (((metric_payload.get("reading_order") or {}).get("all") or {}))
    text_all = (((metric_payload.get("text_block") or {}).get("all") or {}))

    formula_edit = _nested_float(formula_all, "Edit_dist", "ALL_page_avg")
    reading_edit = _nested_float(reading_all, "Edit_dist", "ALL_page_avg")
    text_edit = _nested_float(text_all, "Edit_dist", "ALL_page_avg")
    return {
        "priority_order": [
            "teds_mean",
            "teds_s_mean",
            "formula_cdm_mean",
            "read_order_accuracy",
        ],
        "teds_mean": _nested_float(table_all, "TEDS", "all"),
        "teds_s_mean": _nested_float(table_all, "TEDS_structure_only", "all"),
        "formula_cdm_mean": _nested_float(formula_all, "CDM", "all"),
        "formula_edit_distance": formula_edit,
        "formula_edit_accuracy": _distance_to_accuracy(formula_edit),
        "read_order_edit_distance": reading_edit,
        "read_order_accuracy": _distance_to_accuracy(reading_edit),
        "text_block_edit_distance": text_edit,
        "core_text_mean": _distance_to_accuracy(text_edit),
    }


def run(args: argparse.Namespace) -> Path:
    benchmark_root = args.benchmark_root.resolve()
    output_dir = args.output_dir.resolve()
    dataset_json = args.dataset_json.resolve()
    image_dir = args.image_dir.resolve()
    omnidoc_python = args.omnidoc_python.resolve()

    if not dataset_json.exists():
        raise FileNotFoundError(f"Missing OmniDocBench dataset JSON: {dataset_json}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing OmniDocBench image directory: {image_dir}")
    if not omnidoc_python.exists():
        raise FileNotFoundError(f"Missing OmniDocBench Python runtime: {omnidoc_python}")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / time.strftime("run_%Y%m%d_%H%M%S")
    pdf_dir = run_dir / "pdfs"
    pred_dir = run_dir / "predictions"
    config_path = run_dir / "autoind_omnidocbench.yaml"
    report_path = run_dir / "autoind_omnidocbench_initial_report.md"
    summary_path = run_dir / "autoind_omnidocbench_summary.json"
    low_score_path = run_dir / "autoind_omnidocbench_low_scores.json"
    eval_dataset_path = run_dir / "autoind_omnidocbench_eval_dataset.json"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(dataset_json, args.count, category_filter=args.category_filter)
    _write_eval_dataset(eval_dataset_path, samples)
    parse_records = _generate_predictions(samples, image_dir=image_dir, pdf_dir=pdf_dir, pred_dir=pred_dir)
    _write_config(config_path, dataset_json=eval_dataset_path, pred_dir=pred_dir, include_cdm=not args.skip_cdm)
    metric_payload, eval_elapsed, eval_error = _run_official_eval(
        benchmark_root=benchmark_root,
        omnidoc_python=omnidoc_python,
        config_path=config_path,
    )
    summary = summarize_omnidocbench_metrics(metric_payload) if metric_payload else {}
    low_scores = _collect_low_scores(run_dir / "result", metric_payload)
    payload = {
        "dataset": {
            "name": "OmniDocBench",
            "scope": _dataset_scope(args.count, args.category_filter),
            "dataset_json": str(dataset_json),
            "eval_dataset_json": str(eval_dataset_path),
            "image_dir": str(image_dir),
            "sample_count": len(samples),
            "category_filter": args.category_filter or "",
            "full_dataset_expected_pages": 1651,
        },
        "engine": {
            "name": ENGINE_NAME,
            "parser": "AutoIND-Pro current worktree",
            "prediction_dir": str(pred_dir),
            "pdf_dir": str(pdf_dir),
        },
        "metrics": summary,
        "official_metric_payload": metric_payload,
        "low_scores": low_scores,
        "parse_records": parse_records,
        "runtime": {
            "eval_elapsed_seconds": round(eval_elapsed, 3),
            "eval_error": eval_error,
            "cdm_enabled": not args.skip_cdm,
        },
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    low_score_path.write_text(json.dumps(low_scores, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(report_path, payload)
    return report_path


def _run_table_region_evaluation(
    *,
    dataset_json: Path,
    image_dir: Path,
    output_dir: Path,
    count: int | None,
    category_filter: str = "table",
) -> Path:
    dataset_json = dataset_json.resolve()
    image_dir = image_dir.resolve()
    output_dir = output_dir.resolve()
    if not dataset_json.exists():
        raise FileNotFoundError(f"Missing OmniDocBench dataset JSON: {dataset_json}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing OmniDocBench image directory: {image_dir}")
    run_dir = output_dir / time.strftime("region_run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dataset_path = run_dir / "autoind_omnidocbench_table_region_eval_dataset.json"
    prediction_path = run_dir / "autoind_omnidocbench_table_region_predictions.json"
    summary_path = run_dir / "autoind_omnidocbench_table_region_summary.json"
    report_path = run_dir / "autoind_omnidocbench_table_region_report.md"

    samples = _load_samples(dataset_json, count, category_filter=category_filter)
    _write_eval_dataset(eval_dataset_path, samples)
    predictions: dict[str, list[dict[str, Any]]] = {}
    elapsed_records: list[float] = []
    for sample in samples:
        image_name = Path(str((sample.get("page_info") or {}).get("image_path") or "")).name
        if not image_name:
            continue
        start = time.perf_counter()
        predictions[image_name] = _detect_raster_table_regions(image_dir / image_name)
        elapsed_records.append(time.perf_counter() - start)
    metrics = _evaluate_table_region_detection(samples, predictions, iou_threshold=0.5)
    prediction_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "dataset": {
            "name": "OmniDocBench",
            "dataset_json": str(dataset_json),
            "eval_dataset_json": str(eval_dataset_path),
            "image_dir": str(image_dir),
            "sample_count": len(samples),
            "category_filter": category_filter,
        },
        "metrics": metrics,
        "runtime": {
            "mean_region_detection_seconds": round(fmean(elapsed_records), 6) if elapsed_records else None,
            "total_region_detection_seconds": round(sum(elapsed_records), 6),
        },
        "predictions_path": str(prediction_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_table_region_report(report_path, payload)
    return report_path


def _generate_predictions(
    samples: list[dict[str, Any]],
    *,
    image_dir: Path,
    pdf_dir: Path,
    pred_dir: Path,
) -> list[dict[str, Any]]:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from api.main import _build_document_body_markdown_sections
    from parsers.pdf_parser import parse_pdf

    records: list[dict[str, Any]] = []
    for sample in samples:
        image_name = Path(str((sample.get("page_info") or {}).get("image_path") or "")).name
        if not image_name:
            continue
        image_path = image_dir / image_name
        pdf_path = pdf_dir / f"{Path(image_name).stem}.pdf"
        pred_path = pred_dir / f"{Path(image_name).stem}.md"
        start = time.perf_counter()
        error = ""
        try:
            _image_to_pdf(image_path, pdf_path)
            parsed = parse_pdf(pdf_path)
            parsed.setdefault("filename", pdf_path.name)
            parsed.setdefault("source_type", "pdf")
            markdown = "\n".join(
                _build_document_body_markdown_sections(
                    parsed,
                    embed_images=False,
                    table_export_mode="semantic_html",
                )
            ).strip()
            pred_path.write_text(markdown + "\n", encoding="utf-8")
        except Exception as exc:  # pragma: no cover - report guard
            error = f"{type(exc).__name__}: {exc}"
            pred_path.write_text("", encoding="utf-8")
        records.append(
            {
                "image_name": image_name,
                "pdf_path": str(pdf_path),
                "prediction_path": str(pred_path),
                "elapsed_seconds": round(time.perf_counter() - start, 3),
                "error": error,
            }
        )
        print(f"{image_name}: {records[-1]['elapsed_seconds']:.2f}s" + (f" ERROR {error}" if error else ""))
    return records


def _image_to_pdf(image_path: Path, pdf_path: Path) -> None:
    if pymupdf is None:
        raise RuntimeError("PyMuPDF is required for OmniDocBench image-to-PDF conversion.")
    if not image_path.exists():
        raise FileNotFoundError(f"Missing OmniDocBench page image: {image_path}")
    doc = pymupdf.open()
    try:
        pix = pymupdf.Pixmap(str(image_path))
        page = doc.new_page(width=pix.width, height=pix.height)
        page.insert_image(page.rect, filename=str(image_path))
        doc.save(str(pdf_path))
    finally:
        doc.close()


def _run_official_eval(
    *,
    benchmark_root: Path,
    omnidoc_python: Path,
    config_path: Path,
) -> tuple[dict[str, Any], float, str]:
    result_dir = benchmark_root / "result"
    if result_dir.exists():
        shutil.rmtree(result_dir)
    start = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    command = [str(omnidoc_python), "pdf_validation.py", "--config", str(config_path)]
    completed = subprocess.run(
        command,
        cwd=benchmark_root,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - start
    log_path = config_path.parent / "official_eval.log"
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    metric_path = result_dir / f"{pred_dir_basename(config_path)}_metric_result.json"
    if completed.returncode != 0:
        return {}, elapsed, f"official evaluator exited {completed.returncode}; see {log_path}"
    if not metric_path.exists():
        candidates = sorted(result_dir.glob("*_metric_result.json"))
        if not candidates:
            return {}, elapsed, f"official evaluator produced no metric JSON; see {log_path}"
        metric_path = candidates[-1]
    copy_root = config_path.parent / "result"
    if copy_root.exists():
        shutil.rmtree(copy_root)
    if result_dir.exists():
        shutil.copytree(result_dir, copy_root)
    return json.loads(metric_path.read_text(encoding="utf-8")), elapsed, ""


def pred_dir_basename(config_path: Path) -> str:
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task = cfg["end2end_eval"]
    pred_path = Path(str(task["dataset"]["prediction"]["data_path"]))
    return f"{pred_path.name}_{task['dataset'].get('match_method', 'quick_match')}"


def _write_config(config_path: Path, *, dataset_json: Path, pred_dir: Path, include_cdm: bool) -> None:
    cdm_line = "\n      - CDM" if include_cdm else ""
    config = f"""end2end_eval:
  metrics:
    text_block:
      metric:
      - Edit_dist
    display_formula:
      metric:
      - Edit_dist{cdm_line}
      cdm_workers: 1
    table:
      metric:
      - TEDS
      - Edit_dist
      teds_workers: 4
      timeout_sec: 120
    reading_order:
      metric:
      - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: {dataset_json.as_posix()}
    prediction:
      data_path: {pred_dir.as_posix()}
    match_method: quick_match
    match_workers: 4
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
"""
    config_path.write_text(config, encoding="utf-8")


def _load_samples(dataset_json: Path, count: int | None, category_filter: str = "") -> list[dict[str, Any]]:
    samples = json.loads(dataset_json.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("OmniDocBench dataset JSON must be a list of page samples.")
    if category_filter:
        samples = [sample for sample in samples if _sample_has_category(sample, category_filter)]
    if count and count > 0:
        return samples[:count]
    return samples


def _write_eval_dataset(path: Path, samples: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")


def _sample_has_category(sample: dict[str, Any], category: str) -> bool:
    for item in sample.get("layout_dets") or []:
        if isinstance(item, dict) and item.get("category_type") == category:
            return True
    return False


def _collect_low_scores(result_dir: Path, metric_payload: dict[str, Any]) -> dict[str, Any]:
    low_scores: dict[str, Any] = {}
    table_path = _first_existing(result_dir, "*_table_per_table_TEDS.json")
    if table_path:
        table_scores = json.loads(table_path.read_text(encoding="utf-8"))
        rows = [
            {
                "sample": key,
                "teds": _as_float(value.get("TEDS")),
                "teds_s": _as_float(value.get("TEDS_structure_only")),
            }
            for key, value in table_scores.items()
            if isinstance(value, dict)
        ]
        rows.sort(key=lambda item: (item["teds"] if item["teds"] is not None else -1.0, item["sample"]))
        low_scores["lowest_tables"] = rows[:20]
    table_detail_path = _first_existing(result_dir, "*_table_result.json")
    if table_detail_path:
        table_details = _collect_low_table_details(table_detail_path)
        low_scores["lowest_table_details"] = table_details[:20]
        low_scores["table_prediction_presence"] = _summarize_table_prediction_presence(table_detail_path)
    reading_path = _first_existing(result_dir, "*_reading_order_per_page_edit.json")
    if reading_path:
        reading_scores = json.loads(reading_path.read_text(encoding="utf-8"))
        rows = [
            {
                "sample": key,
                "read_order_edit_distance": _as_float(value),
                "read_order_accuracy": _distance_to_accuracy(_as_float(value)),
            }
            for key, value in reading_scores.items()
        ]
        rows.sort(key=lambda item: (-(item["read_order_edit_distance"] or 0.0), item["sample"]))
        low_scores["lowest_reading_order"] = rows[:20]
    cdm_debug = (((metric_payload.get("display_formula") or {}).get("metric_debug") or {}).get("CDM") or {})
    if cdm_debug:
        low_scores["formula_cdm_exceptions"] = cdm_debug.get("exception_cases", [])[:20]
        low_scores["formula_cdm_exception_count"] = cdm_debug.get("exception_case_count")
    return low_scores


def _collect_low_table_details(table_detail_path: Path) -> list[dict[str, Any]]:
    details = json.loads(table_detail_path.read_text(encoding="utf-8"))
    if not isinstance(details, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in details:
        if not isinstance(item, dict):
            continue
        metric = item.get("metric") if isinstance(item.get("metric"), dict) else {}
        pred = str(item.get("pred") or "")
        sample = str(item.get("img_id") or item.get("image_name") or item.get("sample") or "")
        rows.append(
            {
                "sample": sample,
                "teds": _as_float(metric.get("TEDS")),
                "teds_s": _as_float(metric.get("TEDS_structure_only")),
                "edit_distance": _as_float(metric.get("Edit_dist")),
                "has_pred_table": "<table" in pred.lower(),
                "pred_preview": _preview_html(pred),
                "gt_preview": _preview_html(str(item.get("gt") or "")),
                "attributes": item.get("gt_attribute") if isinstance(item.get("gt_attribute"), list) else [],
            }
        )
    rows.sort(key=lambda row: (row["teds"] if row["teds"] is not None else -1.0, row["sample"]))
    return rows


def _summarize_table_prediction_presence(table_detail_path: Path) -> dict[str, Any]:
    details = json.loads(table_detail_path.read_text(encoding="utf-8"))
    if not isinstance(details, list):
        return {
            "total_tables": 0,
            "predicted_tables": 0,
            "missing_pred_tables": 0,
            "predicted_table_rate": None,
        }
    total = 0
    predicted = 0
    for item in details:
        if not isinstance(item, dict):
            continue
        total += 1
        if "<table" in str(item.get("pred") or "").lower():
            predicted += 1
    missing = total - predicted
    return {
        "total_tables": total,
        "predicted_tables": predicted,
        "missing_pred_tables": missing,
        "predicted_table_rate": round(predicted / total, 6) if total else None,
    }


def _evaluate_table_region_detection(
    samples: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    gt_records: list[dict[str, Any]] = []
    for sample in samples:
        image_name = Path(str((sample.get("page_info") or {}).get("image_path") or "")).name
        for item in sample.get("layout_dets") or []:
            if not isinstance(item, dict) or item.get("category_type") != "table":
                continue
            bbox = _poly_to_bbox(item.get("poly") or [])
            if bbox is None:
                continue
            gt_records.append({"image_name": image_name, "bbox": bbox})

    matched_prediction_keys: set[tuple[str, int]] = set()
    per_table: list[dict[str, Any]] = []
    matched = 0
    best_ious: list[float] = []
    for gt_index, gt in enumerate(gt_records):
        image_name = gt["image_name"]
        candidates = predictions.get(image_name) or []
        best_iou = 0.0
        best_index: int | None = None
        for pred_index, candidate in enumerate(candidates):
            pred_bbox = tuple(float(value) for value in candidate.get("bbox", (0, 0, 0, 0)))
            if len(pred_bbox) != 4:
                continue
            candidate_iou = _bbox_iou(tuple(gt["bbox"]), pred_bbox)
            if candidate_iou > best_iou:
                best_iou = candidate_iou
                best_index = pred_index
        is_match = best_iou >= iou_threshold and best_index is not None
        if is_match:
            matched += 1
            matched_prediction_keys.add((image_name, int(best_index)))
        best_ious.append(best_iou)
        per_table.append(
            {
                "gt_index": gt_index,
                "image_name": image_name,
                "gt_bbox": [float(value) for value in gt["bbox"]],
                "best_iou": round(best_iou, 6),
                "matched": is_match,
            }
        )

    predicted_count = sum(len(items) for items in predictions.values())
    false_positive_count = max(0, predicted_count - len(matched_prediction_keys))
    gt_count = len(gt_records)
    return {
        "gt_table_count": gt_count,
        "predicted_region_count": predicted_count,
        "matched_table_count": matched,
        "missed_table_count": gt_count - matched,
        "false_positive_count": false_positive_count,
        "recall": round(matched / gt_count, 6) if gt_count else None,
        "precision": round(matched / predicted_count, 6) if predicted_count else None,
        "mean_best_iou": round(fmean(best_ious), 6) if best_ious else None,
        "iou_threshold": iou_threshold,
        "per_table": per_table,
    }


def _poly_to_bbox(poly: list[Any]) -> tuple[float, float, float, float] | None:
    if len(poly) < 8:
        return None
    xs = [float(value) for value in poly[0::2]]
    ys = [float(value) for value in poly[1::2]]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    denominator = left_area + right_area - intersection
    return intersection / denominator if denominator > 0.0 else 0.0


def _bbox_intersection_over_min(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1.0, min(left_area, right_area))


def _first_existing(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    metrics = payload.get("metrics") or {}
    low_scores = payload.get("low_scores") or {}
    parse_records = payload.get("parse_records") or []
    errors = [item for item in parse_records if item.get("error")]
    elapsed_values = [float(item["elapsed_seconds"]) for item in parse_records if isinstance(item.get("elapsed_seconds"), (int, float))]
    estimated_full = fmean(elapsed_values) * 1651 if elapsed_values else None
    low_table_rows = [["Sample", "TEDS", "TEDS-S"]]
    for item in (low_scores.get("lowest_tables") or [])[:10]:
        low_table_rows.append([item.get("sample", ""), _fmt(item.get("teds")), _fmt(item.get("teds_s"))])
    low_table_detail_rows = [["Sample", "TEDS", "TEDS-S", "Pred Table", "Attributes"]]
    for item in (low_scores.get("lowest_table_details") or [])[:10]:
        low_table_detail_rows.append(
            [
                item.get("sample", ""),
                _fmt(item.get("teds")),
                _fmt(item.get("teds_s")),
                "yes" if item.get("has_pred_table") else "no",
                _format_table_attributes(item.get("attributes") or []),
            ]
        )
    table_presence = low_scores.get("table_prediction_presence") or {}
    low_ro_rows = [["Sample", "RO Edit", "RO Accuracy"]]
    for item in (low_scores.get("lowest_reading_order") or [])[:10]:
        low_ro_rows.append(
            [
                item.get("sample", ""),
                _fmt(item.get("read_order_edit_distance")),
                _fmt(item.get("read_order_accuracy")),
            ]
        )
    report = f"""# OmniDocBench Initial AutoIND Evaluation

Generated: `{time.strftime("%Y-%m-%d %H:%M:%S")}`

Scope:

- Dataset: `{payload['dataset']['dataset_json']}`
- Images: `{payload['dataset']['image_dir']}`
- Evaluated pages: `{payload['dataset']['sample_count']}` of expected full OmniDocBench `1651`
- Engine: `{payload['engine']['name']}`
- Prediction Markdown: `{payload['engine']['prediction_dir']}`
- CDM enabled: `{payload['runtime']['cdm_enabled']}`
- Official evaluator issue: `{payload['runtime']['eval_error'] or 'none'}`

## Priority Metrics

| Metric | Value | Priority |
| --- | ---: | --- |
| `teds_mean` | `{_fmt(metrics.get('teds_mean'))}` | 1 |
| `teds_s_mean` | `{_fmt(metrics.get('teds_s_mean'))}` | 2 |
| `formula_cdm_mean` | `{_fmt(metrics.get('formula_cdm_mean'))}` | 3 |
| `formula_edit_accuracy` | `{_fmt(metrics.get('formula_edit_accuracy'))}` | diagnostic fallback |
| `read_order_accuracy` | `{_fmt(metrics.get('read_order_accuracy'))}` | 4 |
| `core_text_mean` | `{_fmt(metrics.get('core_text_mean'))}` | AutoIND diagnostic |

## Low Table Samples

{_markdown_table(low_table_rows)}

## Low Table Diagnostics

{_markdown_table(low_table_detail_rows)}

Table prediction presence:

- GT tables: `{table_presence.get('total_tables', 'n/a')}`
- Predicted tables: `{table_presence.get('predicted_tables', 'n/a')}`
- Missing predicted tables: `{table_presence.get('missing_pred_tables', 'n/a')}`
- Predicted table rate: `{_fmt(table_presence.get('predicted_table_rate'))}`

## Low Reading Order Samples

{_markdown_table(low_ro_rows)}

## Runtime Notes

- Parse failures: `{len(errors)}`
- Mean AutoIND parse time/page: `{_fmt(fmean(elapsed_values) if elapsed_values else None)}` seconds
- Estimated parse-only time for 1651 pages from this subset: `{_fmt(estimated_full)}` seconds
- Formula-CDM requires the official Linux/Docker runtime with TeX Live, Ghostscript, and ImageMagick. Current Windows demo runs should treat `formula_edit_accuracy` as the usable interim formula signal and `formula_cdm_mean` as environment-limited when CDM exceptions are present.

## Initial Framework Reading

OmniDocBench adds pressure beyond the OpenDataLoader 200-sample set: page-image PDFs, colorful layouts, notes/newspapers, formula-heavy academic pages, and table annotations with HTML structure. For AutoIND, use it as an external diagnostic set only. The first optimization target remains evidence-layer region classification and table/formula semantic recovery, while OCR content parsing should be handled in the later OCR module checkpoint rather than mixed into table or reading-order rules.
"""
    report_path.write_text(report, encoding="utf-8")


def _write_table_region_report(report_path: Path, payload: dict[str, Any]) -> None:
    metrics = payload.get("metrics") or {}
    missed = [
        item
        for item in metrics.get("per_table", [])
        if isinstance(item, dict) and not item.get("matched")
    ][:12]
    missed_rows = [["Image", "Best IoU", "GT BBox"]]
    for item in missed:
        missed_rows.append(
            [
                item.get("image_name", ""),
                _fmt(item.get("best_iou")),
                _preview_html(json.dumps(item.get("gt_bbox", []), ensure_ascii=False), limit=120),
            ]
        )
    report = f"""# OmniDocBench Raster Table Region Evaluation

Generated: `{time.strftime("%Y-%m-%d %H:%M:%S")}`

Scope:

- Dataset: `{payload['dataset']['dataset_json']}`
- Images: `{payload['dataset']['image_dir']}`
- Evaluated pages: `{payload['dataset']['sample_count']}`
- Category filter: `{payload['dataset']['category_filter']}`
- Predictions: `{payload['predictions_path']}`

## Region Metrics

| Metric | Value |
| --- | ---: |
| `gt_table_count` | `{metrics.get('gt_table_count')}` |
| `predicted_region_count` | `{metrics.get('predicted_region_count')}` |
| `matched_table_count` | `{metrics.get('matched_table_count')}` |
| `missed_table_count` | `{metrics.get('missed_table_count')}` |
| `false_positive_count` | `{metrics.get('false_positive_count')}` |
| `recall@0.5` | `{_fmt(metrics.get('recall'))}` |
| `precision@0.5` | `{_fmt(metrics.get('precision'))}` |
| `mean_best_iou` | `{_fmt(metrics.get('mean_best_iou'))}` |

## Missed/Weak Tables

{_markdown_table(missed_rows)}

## Framework Reading

This is a region-only diagnostic for raster/image table evidence. It must remain separate from the protected text-layer table parser. A good region score means the next step can normalize image/OCR/cell evidence into shared `TableEvidence`; a weak score means the raster evidence-entry layer still needs stronger visual/OCR layout support before TEDS/TEDS-S is meaningful.
"""
    report_path.write_text(report, encoding="utf-8")


def _nested_float(payload: dict[str, Any], *keys: str) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _as_float(value)


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _distance_to_accuracy(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, round(1.0 - float(value), 6)))


def _fmt(value: Any) -> str:
    return f"{float(value):.6f}" if isinstance(value, (int, float)) else "null"


def _dataset_scope(count: int | None, category_filter: str) -> str:
    prefix = f"{category_filter}-filtered" if category_filter else "demo/subset"
    return prefix if count else f"{prefix}/provided dataset json"


def _preview_html(value: str, limit: int = 260) -> str:
    compact = " ".join(value.split())
    return compact[:limit]


def _format_table_attributes(attributes: list[Any]) -> str:
    if not attributes:
        return ""
    first = attributes[0] if isinstance(attributes[0], dict) else {}
    keys = ["with_span", "line", "language", "include_equation", "include_photo", "include_background"]
    return ", ".join(f"{key}={first[key]}" for key in keys if key in first)


def _markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join("---" for _ in rows[0]) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    default_benchmark = (
        Path.home()
        / ".config"
        / "superpowers"
        / "worktrees"
        / "AutoIND-Pro"
        / "opendataloader-bench-eval"
        / "external_benchmarks"
        / "OmniDocBench"
    )
    parser = argparse.ArgumentParser(description="Run AutoIND-Pro against OmniDocBench demo/subset data.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["end2end", "region-eval"],
        default="end2end",
        help="Run end-to-end Markdown scoring or raster table region evaluation.",
    )
    parser.add_argument("--benchmark-root", type=Path, default=default_benchmark)
    parser.add_argument("--dataset-json", type=Path, default=default_benchmark / "demo_data" / "omnidocbench_demo" / "OmniDocBench_demo.json")
    parser.add_argument("--image-dir", type=Path, default=default_benchmark / "demo_data" / "omnidocbench_demo" / "images")
    parser.add_argument("--omnidoc-python", type=Path, default=default_benchmark / ".venv-omnidocbench" / "Scripts" / "python.exe")
    parser.add_argument("--output-dir", type=Path, default=default_benchmark / "autoind-output")
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument(
        "--category-filter",
        default="",
        help="Only evaluate pages containing this OmniDocBench category_type, e.g. table.",
    )
    parser.add_argument("--skip-cdm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "region-eval":
        report = _run_table_region_evaluation(
            dataset_json=args.dataset_json,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            count=args.count,
            category_filter=args.category_filter or "table",
        )
    else:
        report = run(args)
    print(report)


if __name__ == "__main__":
    main()
