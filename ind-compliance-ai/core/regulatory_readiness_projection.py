from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


_SOURCE_ORDER: tuple[dict[str, str], ...] = (
    {
        "regulation_id": "cn_ectd_validation_standard",
        "source_file": "eCTD验证标准.pdf",
        "recommended_product_role": "deterministic_validation_backbone",
        "default_triage": "deterministic",
        "automation_boundary": (
            "Keep deterministic validation-standard rules closed unless source artifacts or requirements change. "
            "Citation-only report statistics should remain non-runtime records."
        ),
    },
    {
        "regulation_id": "cn_ectd_technical_specification",
        "source_file": "eCTD技术规范.pdf",
        "recommended_product_role": "technical_spec_traceability_backbone",
        "default_triage": "prerequisite_required",
        "automation_boundary": (
            "Traceability is closed, but partial/deferred/citation-only evidence boundaries must not be inflated "
            "into inaccurate runtime judgments."
        ),
    },
    {
        "regulation_id": "reg_3454a11dabae",
        "source_file": "eCTD实施指南.pdf",
        "recommended_product_role": "operational_guidance_and_explanation",
        "default_triage": "human_review",
        "automation_boundary": (
            "Use mainly for implementation guidance and reviewer-facing explanations unless a clause passes "
            "four-way triage with local objective evidence."
        ),
    },
    {
        "regulation_id": "cn_drug_administration_law_implementation_regulation",
        "source_file": "中华人民共和国药品管理法实施条例.doc",
        "recommended_product_role": "legal_background_risk_guidance",
        "default_triage": "human_review",
        "automation_boundary": (
            "Do not pursue exhaustive rule automation for broad legal or administrative clauses; isolate only "
            "narrow dossier-decidable obligations with clear local evidence."
        ),
    },
    {
        "regulation_id": "cn_drug_registration_classification_and_dossier_requirements",
        "source_file": "药品注册分类及申报资料要求.doc",
        "recommended_product_role": "dossier_checklist_and_applicability_backbone",
        "default_triage": "prerequisite_required",
        "automation_boundary": (
            "Use as the next product-facing dossier checklist backbone, but require application type, "
            "registration class, product type, and submission-stage facts before hard applicability judgment."
        ),
    },
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _count_clauses(normalized_root: Path, regulation_id: str) -> int:
    payload = _load_json(normalized_root / f"{regulation_id}.clauses.json")
    return len(list(payload.get("clauses") or payload.get("items") or []))


def _count_rule_candidates(normalized_root: Path, regulation_id: str) -> tuple[int, int]:
    payload = _load_json(normalized_root / f"{regulation_id}.rule_candidates.json")
    candidates = list(payload.get("rule_candidates") or payload.get("candidates") or [])
    direct_ish = sum(
        1
        for candidate in candidates
        if candidate.get("recommended_rule_mode") in {"hard", "soft"} or candidate.get("automation_ready")
    )
    return len(candidates), direct_ish


def _count_requirements(normalized_root: Path, regulation_id: str) -> int:
    payload = _load_json(normalized_root / f"{regulation_id}.requirement_matrix.json")
    if "requirement_count" in payload:
        return int(payload.get("requirement_count") or 0)
    requirements = payload.get("requirements") or payload.get("requirement_matrix") or []
    if isinstance(requirements, dict):
        return len(requirements)
    return len(list(requirements))


def _count_rule_drafts(project_root: Path, regulation_id: str) -> int:
    payload = _load_json(project_root / "rules" / "regulation_drafts" / f"{regulation_id}.direct_rule_drafts.json")
    drafts = payload.get("rule_drafts") or payload.get("direct_rule_drafts") or payload.get("rules") or []
    return len(list(drafts))


def _walk_coverage_statuses(payload: Any) -> Counter[str]:
    statuses: Counter[str] = Counter()
    if isinstance(payload, dict):
        status = str(payload.get("coverage_status") or "").strip()
        if status:
            statuses[status] += 1
        for value in payload.values():
            statuses.update(_walk_coverage_statuses(value))
    elif isinstance(payload, list):
        for value in payload:
            statuses.update(_walk_coverage_statuses(value))
    return statuses


def _coverage_counts(normalized_root: Path, regulation_id: str) -> dict[str, int]:
    payload = _load_json(normalized_root / f"{regulation_id}.coverage_report.json")
    top_level_counts = payload.get("coverage_counts")
    if isinstance(top_level_counts, dict) and top_level_counts:
        return {str(key): int(value or 0) for key, value in top_level_counts.items()}
    return dict(_walk_coverage_statuses(payload))


def _coverage_status(regulation_id: str, counts: dict[str, int]) -> str:
    if regulation_id == "cn_ectd_validation_standard" and counts.get("covered") == 146:
        return "closed"
    if regulation_id == "cn_ectd_technical_specification" and counts.get("covered") == 20:
        return "traceability_closed"
    return "parsed_not_coverage_closed"


def _source_payload(project_root: Path, source: dict[str, str]) -> dict[str, Any]:
    normalized_root = project_root / "data" / "regulations" / "normalized"
    regulation_id = source["regulation_id"]
    counts = _coverage_counts(normalized_root, regulation_id)
    candidate_count, direct_ish_count = _count_rule_candidates(normalized_root, regulation_id)

    payload: dict[str, Any] = {
        "regulation_id": regulation_id,
        "source_file": source["source_file"],
        "coverage_status": _coverage_status(regulation_id, counts),
        "clause_count": _count_clauses(normalized_root, regulation_id),
        "covered_count": counts.get("covered", 0),
        "partial_count": counts.get("partially_covered", 0),
        "deferred_count": counts.get("deferred", 0),
        "citation_only_count": counts.get("citation_only_recorded", 0),
        "rule_candidate_count": candidate_count,
        "direct_ish_candidate_count": direct_ish_count,
        "requirement_count": _count_requirements(normalized_root, regulation_id),
        "direct_rule_draft_count": _count_rule_drafts(project_root, regulation_id),
        "recommended_product_role": source["recommended_product_role"],
        "default_triage": source["default_triage"],
        "automation_boundary": source["automation_boundary"],
    }
    if regulation_id == "cn_ectd_technical_specification":
        coverage_payload = _load_json(normalized_root / f"{regulation_id}.coverage_report.json")
        payload["traceability_gap_count"] = int(coverage_payload.get("traceability_gap_clause_count") or 0)
    return payload


def build_regulatory_readiness_projection(project_root: Path) -> dict[str, Any]:
    sources = [_source_payload(project_root, source) for source in _SOURCE_ORDER]
    closed_source_count = sum(
        1 for source in sources if source["coverage_status"] in {"closed", "traceability_closed"}
    )
    return {
        "schema_version": "regulatory-readiness-v1",
        "phase": "phase_a_demo_workbench",
        "phase_status": "in_progress",
        "recommended_next_action_code": "build_source_readiness_matrix_and_triage_projection",
        "recommended_next_action_label": "Build source readiness matrix and triage projection",
        "rule_triage_categories": [
            "deterministic",
            "prerequisite_required",
            "human_review",
            "out_of_scope_low_roi",
        ],
        "sources": sources,
        "summary": {
            "source_count": len(sources),
            "closed_source_count": closed_source_count,
            "demo_value_statement": (
                "Phase A should show deterministic eCTD validation, technical-spec evidence boundaries, "
                "dossier checklist applicability, prerequisite prompts, and human-review guidance."
            ),
        },
    }
