from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import re
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.upload_controller import ALLOWED_EXTENSIONS
from parsers.parser_registry import parse_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "samples" / "uploaded"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PARSED_MARKDOWN_DIR = PROJECT_ROOT / "output" / "parsed_markdown"
PARSED_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("ind_compliance.api")

JOB_STORE: dict[str, dict[str, Any]] = {}
FILE_STORE: dict[str, Path] = {}
STORE_LOCK = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in name)
    return sanitized or "uploaded_file"


def _module_label(filename: str, index: int) -> str:
    lowered = filename.lower()
    if "m1" in lowered or "module1" in lowered:
        return "M1"
    if "m2" in lowered or "module2" in lowered:
        return "M2"
    if "m3" in lowered or "module3" in lowered or "3.2" in lowered:
        return "M3"
    if "m4" in lowered or "module4" in lowered:
        return "M4"
    if "m5" in lowered or "module5" in lowered:
        return "M5"
    return f"DOC-{index + 1}"


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _normalize_text_preview(text: str, max_lines: int = 80) -> str:
    cleaned_lines = [" ".join(line.split()) for line in text.replace("\x00", "").splitlines()]
    filtered_lines = [line for line in cleaned_lines if line]
    return "\n".join(filtered_lines[:max_lines]).strip()


def _estimate_first_page_text(document: dict[str, Any]) -> str:
    source_type = str(document.get("source_type", "")).lower()
    if source_type == "pdf":
        pages = document.get("pages", [])
        if pages:
            return _normalize_text_preview(str(pages[0].get("text", "")))
    if source_type == "word":
        paragraphs = document.get("paragraphs", [])
        metadata = document.get("metadata", {})
        paragraph_count = len(paragraphs)
        page_count = metadata.get("page_count")
        if paragraph_count == 0:
            return ""
        if isinstance(page_count, int) and page_count > 0:
            per_page = max(1, round(paragraph_count / page_count))
            preview_count = min(max(per_page, 8), 40)
        else:
            preview_count = min(paragraph_count, 24)
        preview_text = "\n".join(str(item.get("text", "")) for item in paragraphs[:preview_count])
        return _normalize_text_preview(preview_text)
    if source_type == "presentation":
        slides = document.get("slides", [])
        if slides:
            return _normalize_text_preview(str(slides[0].get("text", "")))
    return _normalize_text_preview(str(document.get("text", "")))


def _count_tokens(text: str) -> int:
    whitespace_tokens = len(re.findall(r"\S+", text))
    cjk_char_tokens = len(re.findall(r"[\u4e00-\u9fff]", text))
    return max(whitespace_tokens, 0) + cjk_char_tokens


def _build_enterprise_metrics(
    parsed_documents: list[dict[str, Any]],
    file_records: list[dict[str, Any]],
    consistency_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_files = len(file_records)
    parsed_count = len(parsed_documents)
    total_chars = sum(len(str(item.get("text", ""))) for item in parsed_documents)
    total_tokens = sum(_count_tokens(str(item.get("text", ""))) for item in parsed_documents)

    estimated_pages = 0
    for document in parsed_documents:
        metadata = document.get("metadata", {})
        source_type = str(document.get("source_type", "")).lower()
        page_count = metadata.get("page_count")
        if isinstance(page_count, int) and page_count > 0:
            estimated_pages += page_count
        elif source_type == "pdf":
            estimated_pages += len(document.get("pages", []))
        elif source_type == "presentation":
            estimated_pages += len(document.get("slides", []))
        elif source_type == "word":
            paragraph_count = len(document.get("paragraphs", []))
            estimated_pages += max(1, round(paragraph_count / 24)) if paragraph_count else 0

    expected_fact_keys = {"drug_name", "dosage_form", "batch_number", "manufacturing_site", "strength"}
    captured_facts = sum(
        len({key for key in document.get("atomic_facts", {}) if key in expected_fact_keys})
        for document in parsed_documents
    )
    expected_total_facts = len(expected_fact_keys) * max(parsed_count, 1)

    consistent_rows = sum(1 for row in consistency_rows if row.get("is_consistent"))
    consistency_total = len(consistency_rows)

    parser_strategy_counter: dict[str, int] = {}
    for document in parsed_documents:
        parser_hint = str(document.get("metadata", {}).get("parser_hint", "unknown"))
        parser_strategy_counter[parser_hint] = parser_strategy_counter.get(parser_hint, 0) + 1

    parse_success_rate = _safe_ratio(parsed_count, total_files) * 100
    extraction_quality_score = (
        _safe_ratio(total_chars, max(parsed_count, 1) * 5000) * 40
        + _safe_ratio(captured_facts, expected_total_facts) * 35
        + _safe_ratio(consistent_rows, max(consistency_total, 1)) * 25
    )
    extraction_quality_score = min(extraction_quality_score, 100.0)

    return {
        "total_files": total_files,
        "parsed_count": parsed_count,
        "parse_success_rate": parse_success_rate,
        "estimated_pages": estimated_pages,
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "captured_facts": captured_facts,
        "expected_total_facts": expected_total_facts,
        "fact_capture_rate": _safe_ratio(captured_facts, expected_total_facts) * 100,
        "consistency_passed": consistent_rows,
        "consistency_total": consistency_total,
        "consistency_rate": _safe_ratio(consistent_rows, max(consistency_total, 1)) * 100,
        "quality_score": extraction_quality_score,
        "parser_distribution": parser_strategy_counter,
    }


def _build_ui_markdown(
    parsed_documents: list[dict[str, Any]],
    file_records: list[dict[str, Any]],
    consistency_rows: list[dict[str, Any]],
) -> str:
    metrics = _build_enterprise_metrics(parsed_documents, file_records, consistency_rows)

    lines = [
        "# Enterprise Parsing Executive Summary",
        "",
        f"- Generated at: {_utc_now()}",
        f"- Job scope: {metrics['total_files']} file(s), {metrics['parsed_count']} parsed successfully",
        "",
        "## Quantitative KPI Overview",
        "",
        "| Metric | Value | Interpretation |",
        "| --- | --- | --- |",
        (
            f"| Parse success rate | {metrics['parse_success_rate']:.1f}% "
            f"({metrics['parsed_count']}/{metrics['total_files']}) | Parser execution completeness |"
        ),
        f"| Estimated pages/slides | {metrics['estimated_pages']} | Approximate payload scale |",
        f"| Extracted text volume | {metrics['total_characters']} chars / {metrics['total_tokens']} tokens | Text capture throughput |",
        (
            f"| Atomic fact capture rate | {metrics['fact_capture_rate']:.1f}% "
            f"({metrics['captured_facts']}/{metrics['expected_total_facts']}) | Cross-module key-field readiness |"
        ),
        (
            f"| Consistency pass rate | {metrics['consistency_rate']:.1f}% "
            f"({metrics['consistency_passed']}/{metrics['consistency_total']}) | Cross-document alignment quality |"
        ),
        f"| Enterprise quality score | {metrics['quality_score']:.1f}/100 | Composite extraction reliability indicator |",
        "",
        "## Parser Distribution",
        "",
    ]
    for parser_name, count in sorted(metrics["parser_distribution"].items()):
        lines.append(f"- {parser_name}: {count} file(s)")
    if not metrics["parser_distribution"]:
        lines.append("- No parser output")

    lines.extend(
        [
            "",
            "## Document Inventory",
            "",
            "| File | Type | Estimated pages/slides | Extracted chars | Parser strategy |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for index, document in enumerate(parsed_documents):
        filename = str(document.get("filename", f"document-{index + 1}"))
        source_type = str(document.get("source_type", "unknown")).upper()
        metadata = document.get("metadata", {})
        parser_hint = str(metadata.get("parser_hint", "unknown"))
        page_count = metadata.get("page_count")
        if not isinstance(page_count, int) or page_count <= 0:
            if source_type == "PDF":
                page_count = len(document.get("pages", []))
            elif source_type == "PRESENTATION":
                page_count = len(document.get("slides", []))
            elif source_type == "WORD":
                paragraph_count = len(document.get("paragraphs", []))
                page_count = max(1, round(paragraph_count / 24)) if paragraph_count else 0
            else:
                page_count = 0
        extracted_chars = len(str(document.get("text", "")))
        lines.append(f"| {filename} | {source_type} | {page_count} | {extracted_chars} | {parser_hint} |")

    lines.extend(["", "## First-Page Preview", ""])
    if parsed_documents:
        primary_doc = parsed_documents[0]
        filename = str(primary_doc.get("filename", "document-1"))
        source_type = str(primary_doc.get("source_type", "unknown")).upper()
        first_page_preview = _estimate_first_page_text(primary_doc)
        lines.append(f"### {filename} ({source_type})")
        lines.append("")
        lines.append("```text")
        lines.append(first_page_preview or "No preview text available.")
        lines.append("```")
    else:
        lines.append("No parsed document available.")

    return "\n".join(lines).strip()


def _build_full_markdown(parsed_documents: list[dict[str, Any]]) -> str:
    lines = [
        "# IND Parse Snapshot (Full)",
        "",
        f"Generated at: {_utc_now()}",
        "",
    ]
    for index, document in enumerate(parsed_documents):
        filename = document.get("filename", f"document-{index + 1}")
        source_type = str(document.get("source_type", "unknown")).upper()
        lines.append(f"## {filename} ({source_type})")
        lines.append("")

        metadata = document.get("metadata", {})
        page_count = metadata.get("page_count")
        parser_hint = metadata.get("parser_hint")
        if page_count:
            lines.append(f"- Estimated pages: {page_count}")
        if parser_hint:
            lines.append(f"- Parser strategy: {parser_hint}")
        if page_count or parser_hint:
            lines.append("")

        facts = document.get("atomic_facts", {})
        if facts:
            lines.append("### Atomic Facts")
            for key, value in facts.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        if source_type == "PDF":
            lines.append("### PDF Pages")
            for page in document.get("pages", [])[:10]:
                page_number = page.get("page_number")
                block_count = page.get("block_count", 0)
                lines.append(f"- Page {page_number}: {block_count} text blocks")
            lines.append("")

        preview_text = str(document.get("text", "")).strip()
        if preview_text:
            lines.append("### Text Preview")
            lines.append("```text")
            lines.append(preview_text)
            lines.append("```")
            lines.append("")

    return "\n".join(lines).strip()


def _build_consistency_rows(parsed_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    module_fact_map: dict[str, dict[str, str]] = {}
    for index, document in enumerate(parsed_documents):
        module_name = _module_label(str(document.get("filename", "")), index)
        module_fact_map[module_name] = {
            key: str(value)
            for key, value in document.get("atomic_facts", {}).items()
            if str(value).strip()
        }

    all_fact_keys = sorted({key for facts in module_fact_map.values() for key in facts})
    rows: list[dict[str, Any]] = []
    for fact_key in all_fact_keys:
        module_values: list[dict[str, str]] = []
        present_values: set[str] = set()
        for module_name, facts in module_fact_map.items():
            value = facts.get(fact_key, "")
            module_values.append({"module": module_name, "value": value})
            if value:
                present_values.add(value)

        rows.append(
            {
                "fact": fact_key,
                "module_values": module_values,
                "is_consistent": len(present_values) <= 1,
            }
        )
    return rows


def _build_workbench(
    parsed_documents: list[dict[str, Any]],
    file_records: list[dict[str, Any]],
    consistency_rows: list[dict[str, Any]],
    markdown_download_url: str | None,
) -> dict[str, Any]:
    pdf_document: dict[str, Any] | None = None
    for document in parsed_documents:
        if document.get("source_type") == "pdf":
            pdf_document = {
                "file_id": document.get("file_id"),
                "filename": document.get("filename"),
                "file_url": f"/api/v1/files/{document.get('file_id')}",
                "pages": document.get("pages", []),
                "bounding_boxes": document.get("bounding_boxes", []),
            }
            break

    return {
        "pdf_document": pdf_document,
        "markdown": _build_ui_markdown(parsed_documents, file_records, consistency_rows),
        "full_markdown_download_url": markdown_download_url,
        "rule_checks": {
            "enabled": False,
            "items": [],
            "message": "Phase 1 keeps this panel as placeholder; AI rule checks start in the next phase.",
        },
    }


def _process_job(job_id: str) -> None:
    with STORE_LOCK:
        job = JOB_STORE.get(job_id)
        if job is None:
            return
        job["status"] = "processing"
        job["progress"] = 5
        job["updated_at"] = _utc_now()
        file_count = len(job["files"])
    logger.info("Job %s started processing (%s files)", job_id, file_count)

    parsed_documents: list[dict[str, Any]] = []
    failed_files = 0

    with STORE_LOCK:
        file_records = list(JOB_STORE[job_id]["files"])

    total_files = len(file_records) or 1
    for index, file_record in enumerate(file_records):
        with STORE_LOCK:
            job = JOB_STORE[job_id]
            for item in job["files"]:
                if item["file_id"] == file_record["file_id"]:
                    item["status"] = "processing"
                    item["progress"] = 10
            job["updated_at"] = _utc_now()

        try:
            parsed = parse_file(Path(file_record["path"]))
            parsed["file_id"] = file_record["file_id"]
            parsed["filename"] = file_record["filename"]
            parsed_documents.append(parsed)
            file_status = "completed"
            file_message = "Parsed successfully"
            file_progress = 100
            logger.info(
                "Job %s parsed file %s (%s)",
                job_id,
                file_record["filename"],
                file_record["suffix"],
            )
        except Exception as exc:  # pragma: no cover - defensive surface for parser errors
            failed_files += 1
            file_status = "failed"
            file_message = f"Parse failed: {exc}"
            file_progress = 100
            logger.exception("Job %s failed parsing file %s", job_id, file_record["filename"])

        processed_ratio = (index + 1) / total_files
        with STORE_LOCK:
            job = JOB_STORE[job_id]
            for item in job["files"]:
                if item["file_id"] == file_record["file_id"]:
                    item["status"] = file_status
                    item["message"] = file_message
                    item["progress"] = file_progress
            job["progress"] = 5 + int(processed_ratio * 80)
            job["updated_at"] = _utc_now()

    consistency_rows = _build_consistency_rows(parsed_documents)
    full_markdown_text = _build_full_markdown(parsed_documents)
    markdown_file_path = PARSED_MARKDOWN_DIR / f"{job_id}_parse_full.md"
    markdown_file_path.write_text(full_markdown_text, encoding="utf-8")
    markdown_download_url = f"/api/v1/jobs/{job_id}/markdown/download"

    with STORE_LOCK:
        job = JOB_STORE[job_id]
        final_file_records = [dict(item) for item in job["files"]]
        workbench = _build_workbench(
            parsed_documents=parsed_documents,
            file_records=final_file_records,
            consistency_rows=consistency_rows,
            markdown_download_url=markdown_download_url,
        )
        job["parsed_documents"] = parsed_documents
        job["consistency_rows"] = consistency_rows
        job["workbench"] = workbench
        job["full_markdown_path"] = str(markdown_file_path)
        job["progress"] = 100
        job["updated_at"] = _utc_now()
        if failed_files == len(file_records):
            job["status"] = "failed"
        elif failed_files > 0:
            job["status"] = "completed_with_warnings"
        else:
            job["status"] = "completed"
        final_status = job["status"]
    logger.info("Job %s finished with status=%s", job_id, final_status)


def create_app() -> FastAPI:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    app = FastAPI(title="IND Compliance AI", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/uploads")
    async def create_upload_job(
        background_tasks: BackgroundTasks,
        files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        job_id = uuid4().hex
        file_records: list[dict[str, Any]] = []
        logger.info("Upload request received. job_id=%s, files=%s", job_id, len(files))

        for incoming_file in files:
            filename = incoming_file.filename or "uploaded_file"
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {suffix}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
                )

            file_id = uuid4().hex
            target_path = UPLOAD_DIR / f"{file_id}_{_safe_filename(filename)}"
            target_path.write_bytes(await incoming_file.read())

            with STORE_LOCK:
                FILE_STORE[file_id] = target_path

            file_records.append(
                {
                    "file_id": file_id,
                    "filename": filename,
                    "suffix": suffix,
                    "path": str(target_path),
                    "status": "queued",
                    "message": "Queued",
                    "progress": 0,
                }
            )

        job_record = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "files": file_records,
            "parsed_documents": [],
            "workbench": None,
            "consistency_rows": [],
        }
        with STORE_LOCK:
            JOB_STORE[job_id] = job_record

        background_tasks.add_task(_process_job, job_id)
        logger.info("Job %s queued", job_id)
        return {
            "job_id": job_id,
            "status": "queued",
            "files": [
                {
                    "file_id": file_item["file_id"],
                    "filename": file_item["filename"],
                    "status": file_item["status"],
                    "progress": file_item["progress"],
                }
                for file_item in file_records
            ],
        }

    @app.get("/api/v1/jobs/{job_id}")
    def get_job_status(job_id: str) -> dict[str, Any]:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            return {
                "job_id": job["job_id"],
                "status": job["status"],
                "progress": job["progress"],
                "created_at": job["created_at"],
                "updated_at": job["updated_at"],
                "files": [
                    {
                        "file_id": file_item["file_id"],
                        "filename": file_item["filename"],
                        "status": file_item["status"],
                        "message": file_item["message"],
                        "progress": file_item["progress"],
                    }
                    for file_item in job["files"]
                ],
            }

    @app.get("/api/v1/jobs/{job_id}/workbench")
    def get_workbench(job_id: str) -> dict[str, Any]:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            if job["status"] in {"queued", "processing"}:
                raise HTTPException(status_code=409, detail="Job is still processing")
            logger.info("Workbench requested for job %s", job_id)
            return job["workbench"] or {}

    @app.get("/api/v1/jobs/{job_id}/consistency")
    def get_consistency(job_id: str) -> dict[str, Any]:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            if job["status"] in {"queued", "processing"}:
                raise HTTPException(status_code=409, detail="Job is still processing")
            logger.info("Consistency board requested for job %s", job_id)
            return {"rows": job["consistency_rows"]}

    @app.get("/api/v1/jobs/{job_id}/markdown/download")
    def download_full_markdown(job_id: str) -> FileResponse:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            markdown_path_str = str(job.get("full_markdown_path", "")).strip()
        if not markdown_path_str:
            raise HTTPException(status_code=404, detail="Full markdown file not found")
        markdown_path = Path(markdown_path_str)
        if not markdown_path.exists():
            raise HTTPException(status_code=404, detail="Full markdown file not found")
        logger.info("Full markdown download requested for job %s", job_id)
        return FileResponse(
            markdown_path,
            media_type="text/markdown; charset=utf-8",
            filename=f"{job_id}_full_parse.md",
        )

    @app.get("/api/v1/files/{file_id}")
    def get_file(file_id: str) -> FileResponse:
        with STORE_LOCK:
            file_path = FILE_STORE.get(file_id)
        if file_path is None or not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path)

    frontend_dist = PROJECT_ROOT / "ui" / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
    else:
        @app.get("/")
        def frontend_placeholder() -> JSONResponse:
            return JSONResponse(
                content={
                    "message": (
                        "Frontend dist not found. Run `npm install && npm run dev` in ui/frontend "
                        "or `npm run build` to let FastAPI serve static files."
                    )
                }
            )

    return app


app = create_app()
