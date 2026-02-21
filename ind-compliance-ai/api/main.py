from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
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


def _build_markdown(parsed_documents: list[dict[str, Any]]) -> str:
    lines = [
        "# IND Parse Snapshot",
        "",
        f"Generated at: {_utc_now()}",
        "",
    ]
    for index, document in enumerate(parsed_documents):
        filename = document.get("filename", f"document-{index + 1}")
        source_type = str(document.get("source_type", "unknown")).upper()
        lines.append(f"## {filename} ({source_type})")
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
            lines.append(preview_text[:2000])
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


def _build_workbench(parsed_documents: list[dict[str, Any]]) -> dict[str, Any]:
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
        "markdown": _build_markdown(parsed_documents),
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
    workbench = _build_workbench(parsed_documents)

    with STORE_LOCK:
        job = JOB_STORE[job_id]
        job["parsed_documents"] = parsed_documents
        job["consistency_rows"] = consistency_rows
        job["workbench"] = workbench
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
