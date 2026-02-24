from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _relative(path: Path, base: Path) -> str:
    return str(path.relative_to(base))


@dataclass(slots=True)
class RunContext:
    run_id: str
    run_dir: Path
    manifest_path: Path
    log_path: Path
    manifest: dict[str, Any]
    project_root: Path
    table_counter: int = 0
    image_counter: int = 0


def create_run_context(project_root: Path, job_id: str, file_records: list[dict[str, Any]]) -> RunContext:
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_id = f"run_{run_stamp}_{job_id[:8]}"
    run_dir = project_root / "runs" / run_id

    required_dirs = [
        run_dir / "artifacts" / "ast" / "pdf",
        run_dir / "artifacts" / "ast" / "docx",
        run_dir / "artifacts" / "ast" / "pptx",
        run_dir / "artifacts" / "ast" / "xml",
        run_dir / "artifacts" / "ast" / "ectd",
        run_dir / "artifacts" / "tables",
        run_dir / "artifacts" / "images",
        run_dir / "artifacts" / "normalized",
        run_dir / "artifacts" / "atomic_facts",
        run_dir / "output",
        run_dir / "logs",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    log_path = run_dir / "logs" / "pipeline.log"
    manifest = {
        "run_id": run_id,
        "job_id": job_id,
        "created_at": _utc_now(),
        "status": "processing",
        "inputs": [
            {
                "file_id": item.get("file_id"),
                "filename": item.get("filename"),
                "suffix": item.get("suffix"),
                "path": item.get("path"),
            }
            for item in file_records
        ],
        "versions": {
            "parser_version": "phase1-parser-v2",
            "rule_version": "phase1-rule-v1",
            "schema_version": "phase1-schema-v1",
        },
        "artifacts_index": {
            "ast": [],
            "tables": [],
            "images": [],
            "normalized": [],
            "atomic_facts": [],
        },
        "output_index": {},
        "error": None,
        "finished_at": None,
    }
    ectd_tree_path = run_dir / "artifacts" / "ast" / "ectd" / "ectd_tree.ast.json"
    _json_dump(
        ectd_tree_path,
        {
            "source_type": "ectd",
            "generated_at": _utc_now(),
            "tree": {"module_1": [], "module_2": [], "module_3": [], "module_4": [], "module_5": []},
        },
    )
    manifest["artifacts_index"]["ast"].append(_relative(ectd_tree_path, run_dir))
    _json_dump(manifest_path, manifest)
    log_path.write_text(f"{_utc_now()} | INFO | run created: {run_id}\n", encoding="utf-8")

    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        log_path=log_path,
        manifest=manifest,
        project_root=project_root,
    )


def append_run_log(context: RunContext, message: str, level: str = "INFO") -> None:
    with context.log_path.open("a", encoding="utf-8") as handler:
        handler.write(f"{_utc_now()} | {level} | {message}\n")


def _ast_bucket_for_suffix(suffix: str, source_type: str) -> str:
    lowered = suffix.lower()
    if lowered == ".pdf" or source_type == "pdf":
        return "pdf"
    if lowered in {".doc", ".docx"} or source_type == "word":
        return "docx"
    if lowered in {".ppt", ".pptx"} or source_type == "presentation":
        return "pptx"
    if lowered == ".xml" or source_type == "xml":
        return "xml"
    return "ectd"


def persist_document_artifacts(
    context: RunContext,
    doc_index: int,
    parsed_document: dict[str, Any],
    file_record: dict[str, Any],
) -> None:
    suffix = str(file_record.get("suffix", ""))
    source_type = str(parsed_document.get("source_type", "unknown"))
    ast_bucket = _ast_bucket_for_suffix(suffix, source_type)
    ast_path = context.run_dir / "artifacts" / "ast" / ast_bucket / f"doc_{doc_index + 1:03d}.ast.json"

    ast_payload = {
        "document_id": f"doc_{doc_index + 1:03d}",
        "source_file": file_record.get("filename"),
        "source_path": parsed_document.get("source_path"),
        "source_type": source_type,
        "metadata": parsed_document.get("metadata", {}),
        "document_ast": parsed_document.get("document_ast", {}),
        "text_excerpt": str(parsed_document.get("text", ""))[:2000],
    }
    _json_dump(ast_path, ast_payload)
    context.manifest["artifacts_index"]["ast"].append(_relative(ast_path, context.run_dir))

    for table in parsed_document.get("table_asts", []):
        table_id = str(table.get("table_id", "")).strip()
        if table_id:
            table_filename = f"{table_id}.json"
        else:
            context.table_counter += 1
            table_filename = f"tbl_{context.table_counter:03d}.json"
        table_path = context.run_dir / "artifacts" / "tables" / table_filename
        _json_dump(table_path, table)
        context.manifest["artifacts_index"]["tables"].append(_relative(table_path, context.run_dir))

    for image in parsed_document.get("image_blocks", []):
        context.image_counter += 1
        image_path = context.run_dir / "artifacts" / "images" / f"img_{context.image_counter:03d}.json"
        _json_dump(image_path, image)
        context.manifest["artifacts_index"]["images"].append(_relative(image_path, context.run_dir))

    _json_dump(context.manifest_path, context.manifest)


def persist_normalized_artifacts(context: RunContext, parsed_documents: list[dict[str, Any]]) -> None:
    normalized_path = context.run_dir / "artifacts" / "normalized" / "material_normalized.json"
    normalized_payload = {
        "generated_at": _utc_now(),
        "documents": [
            {
                "document_id": f"doc_{index + 1:03d}",
                "filename": document.get("filename"),
                "source_type": document.get("source_type"),
                "metadata": document.get("metadata", {}),
                "atomic_facts": document.get("atomic_facts", {}),
                "text_length": len(str(document.get("text", ""))),
                "image_count": len(document.get("image_blocks", [])),
                "table_count": len(document.get("table_asts", [])),
            }
            for index, document in enumerate(parsed_documents)
        ],
    }
    _json_dump(normalized_path, normalized_payload)
    context.manifest["artifacts_index"]["normalized"] = [_relative(normalized_path, context.run_dir)]

    atomic_path = context.run_dir / "artifacts" / "atomic_facts" / "atomic_facts.json"
    atomic_payload = {
        "generated_at": _utc_now(),
        "facts": [
            {
                "document_id": f"doc_{index + 1:03d}",
                "filename": document.get("filename"),
                "facts": document.get("atomic_facts", {}),
            }
            for index, document in enumerate(parsed_documents)
        ],
    }
    _json_dump(atomic_path, atomic_payload)
    context.manifest["artifacts_index"]["atomic_facts"] = [_relative(atomic_path, context.run_dir)]
    _json_dump(context.manifest_path, context.manifest)


def persist_run_outputs(
    context: RunContext,
    compliance_result: dict[str, Any],
    audit_log: dict[str, Any],
) -> None:
    compliance_path = context.run_dir / "output" / "compliance_result.json"
    audit_path = context.run_dir / "output" / "audit_log.json"
    _json_dump(compliance_path, compliance_result)
    _json_dump(audit_path, audit_log)
    context.manifest["output_index"] = {
        "compliance_result": _relative(compliance_path, context.run_dir),
        "audit_log": _relative(audit_path, context.run_dir),
    }
    _json_dump(context.manifest_path, context.manifest)


def finalize_run_context(
    context: RunContext,
    status: str,
    error: str | None = None,
) -> None:
    context.manifest["status"] = status
    context.manifest["error"] = error
    context.manifest["finished_at"] = _utc_now()
    _json_dump(context.manifest_path, context.manifest)
