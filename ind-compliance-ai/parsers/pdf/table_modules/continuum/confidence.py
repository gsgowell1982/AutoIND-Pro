"""Phase 6: Confidence, Risk & Review Policy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..assembly import TableInstance
    from .continuity import ContinuityResult
    from .grid import GridStabilization
    from .identity import IdentityResolution
    from .nested import NestedStructure


# Patterns integrated from the legacy validation layer.
TOC_LINE_PATTERN = re.compile(r"[\.路鈥]{4,}\s*\d{1,3}\s*$")
REFERENCE_ROW_PATTERN = re.compile(r"^\s*(?:\d+\.\s+|[锛?]?\d+[)锛塢])")


@dataclass
class ConfidenceAssessment:
    """Phase 6 result."""

    overall_confidence: float
    risk_flags: list[str] = field(default_factory=list)
    review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)
    is_valid_table: bool = True
    toc_row_ratio: float = 0.0
    reference_like_ratio: float = 0.0


def assess_confidence(
    instance: "TableInstance",
    identity: "IdentityResolution",
    grid: "GridStabilization",
    continuity: "ContinuityResult",
    nested: "NestedStructure",
    context: dict[str, Any] | None = None,
) -> ConfidenceAssessment:
    """Assess confidence and review risk for a table candidate."""

    risk_flags: list[str] = []
    review_reasons: list[str] = []
    confidence_factors: list[float] = []

    title_block = context.get("title_block") if context else None
    section_hint = context.get("section_hint") if context else None
    toc_context = context.get("toc_context", False) if context else False
    grid_line_score = context.get("grid_line_score", 0.0) if context else 0.0

    # Factor 1: identity confidence.
    confidence_factors.append(identity.confidence)
    if identity.confidence < 0.6:
        risk_flags.append("low_identity_confidence")

    # Factor 2: grid stability.
    confidence_factors.append(grid.structure_score)
    if not grid.stable:
        risk_flags.append("unstable_grid")
        review_reasons.append("表格结构不稳定")

    # Factor 3: continuation quality.
    if continuity.is_continuation:
        if continuity.similarity_score < 0.5:
            risk_flags.append("low_continuation_similarity")
            review_reasons.append("续表相似度低")
        else:
            confidence_factors.append(continuity.similarity_score)

    # Factor 4: suspected nested structure complexity.
    if nested.has_nested:
        confidence_factors.append(0.8)
        risk_flags.append("suspected_nested_structure")
        review_reasons.append("疑似存在单元格内嵌套表格")

    # Factor 5: content coverage.
    non_null_ratio = sum(1 for cell in instance.cells if cell.get("text")) / max(1, len(instance.cells))
    confidence_factors.append(non_null_ratio)
    if non_null_ratio < 0.3:
        risk_flags.append("sparse_content")
        review_reasons.append("单元格内容稀疏")

    # Factor 6/7: TOC-like and reference-like ratios.
    toc_row_ratio = _calculate_toc_row_ratio(instance)
    reference_like_ratio = _calculate_reference_like_ratio(instance)

    is_valid = _validate_table_candidate(
        instance=instance,
        identity=identity,
        grid=grid,
        continuity=continuity,
        title_block=title_block,
        section_hint=section_hint,
        toc_context=toc_context,
        grid_line_score=grid_line_score,
        toc_row_ratio=toc_row_ratio,
        reference_like_ratio=reference_like_ratio,
    )

    overall_confidence = (
        sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    )

    if toc_context or toc_row_ratio >= 0.38:
        is_valid = False
        risk_flags.append("toc_like_table")

    review_required = (
        len(risk_flags) >= 2
        or overall_confidence < 0.5
        or any(
            flag in risk_flags
            for flag in [
                "unstable_grid",
                "low_continuation_similarity",
                "suspected_nested_structure",
            ]
        )
        or not is_valid
    )

    return ConfidenceAssessment(
        overall_confidence=min(1.0, overall_confidence),
        risk_flags=risk_flags,
        review_required=review_required,
        review_reasons=review_reasons,
        is_valid_table=is_valid,
        toc_row_ratio=toc_row_ratio,
        reference_like_ratio=reference_like_ratio,
    )


def _validate_table_candidate(
    instance: "TableInstance",
    identity: "IdentityResolution",
    grid: "GridStabilization",
    continuity: "ContinuityResult",
    title_block: dict[str, Any] | None,
    section_hint: dict[str, Any] | None,
    toc_context: bool,
    grid_line_score: float,
    toc_row_ratio: float,
    reference_like_ratio: float,
) -> bool:
    """Keep only candidates that still look like tables after validation."""

    del identity

    score = grid.structure_score
    row_count = instance.row_count
    col_count = instance.col_count

    if toc_context or toc_row_ratio >= 0.38:
        return False

    if title_block is None and continuity.is_continuation is False and reference_like_ratio >= 0.75:
        return False

    if (
        title_block is None
        and continuity.is_continuation is False
        and section_hint is None
        and grid_line_score < 0.3
        and reference_like_ratio >= 0.42
    ):
        return False

    if (
        title_block is None
        and continuity.is_continuation is False
        and grid_line_score < 0.2
        and col_count >= 4
        and row_count <= 8
        and reference_like_ratio >= 0.25
    ):
        return False

    if title_block is not None:
        if col_count == 1:
            return row_count >= 4 and score >= 0.15
        return score >= 0.25 and row_count >= 2 and col_count >= 2

    if continuity.is_continuation:
        if row_count >= 2 and col_count >= 2 and score >= 0.15:
            return True
        if row_count >= 3 and col_count >= 2:
            return score >= 0.12
        return row_count >= 4 and col_count >= 1 and score >= 0.15

    if section_hint is not None:
        section_text = str(section_hint.get("text", "")).strip().lower()
        if "术语表" in section_text or "glossary" in section_text:
            return row_count >= 4 and col_count >= 2 and score >= 0.25
        return row_count >= 4 and col_count >= 2 and score >= 0.55

    if grid_line_score >= 0.45:
        return row_count >= 4 and col_count >= 2 and score >= 0.56

    if instance.is_continuation and row_count >= 2 and col_count >= 2:
        return score >= 0.15

    strong_multi_col = row_count >= 5 and col_count >= 3 and score >= 0.85 and toc_row_ratio <= 0.15
    strong_two_col = row_count >= 6 and col_count >= 2 and score >= 0.85 and toc_row_ratio <= 0.1
    decent_table = row_count >= 4 and col_count >= 2 and score >= 0.7 and toc_row_ratio <= 0.2
    return strong_multi_col or strong_two_col or decent_table


def _calculate_toc_row_ratio(instance: "TableInstance") -> float:
    """Estimate TOC-likeness from row text patterns."""

    if not instance.row_texts:
        return 0.0

    toc_rows = 0.0
    for row_text in instance.row_texts:
        cleaned = row_text.strip() if row_text else ""
        if not cleaned:
            continue

        if TOC_LINE_PATTERN.search(cleaned):
            toc_rows += 1.0
            continue

        tokens = cleaned.split()
        if len(tokens) >= 3:
            if re.fullmatch(r"\d+(?:\.\d+){1,4}", tokens[0]) and re.fullmatch(r"\d{1,3}", tokens[-1]):
                middle = " ".join(tokens[1:-1])
                has_textual_middle = bool(re.search(r"[A-Za-z\u4e00-\u9fff]", middle))
                has_dot_leader = bool(re.search(r"[\.路鈥]{2,}", middle))
                if has_textual_middle and (has_dot_leader or len(middle) >= 3):
                    toc_rows += 0.75
                    continue

        if re.search(r"[\.路鈥]{2,}", cleaned) and re.search(r"\d{1,3}\s*$", cleaned):
            toc_rows += 0.65

    return min(1.0, toc_rows / max(1, len(instance.row_texts)))


def _calculate_reference_like_ratio(instance: "TableInstance") -> float:
    """Estimate how much a table looks like references rather than tabular data."""

    row_texts = [row for row in instance.row_texts if row and row.strip()]
    if not row_texts:
        return 0.0

    score = 0.0
    for row_text in row_texts:
        cleaned = row_text.strip()
        if not cleaned:
            continue

        row_score = 0.0
        lowered = cleaned.lower()

        if "——" in cleaned or ("《" in cleaned and "》" in cleaned):
            row_score += 0.9
        if REFERENCE_ROW_PATTERN.match(cleaned):
            row_score += 0.75
        if re.search(r"\bV\d+(?:\.\d+){1,3}\b", cleaned, re.IGNORECASE):
            row_score += 0.45
        if "ich" in lowered:
            row_score += 0.35
        if re.search(r"(specification|document|technical|change request)", lowered):
            row_score += 0.25

        score += min(1.0, row_score)

    return min(1.0, score / max(1, len(row_texts)))


__all__ = [
    "TOC_LINE_PATTERN",
    "REFERENCE_ROW_PATTERN",
    "ConfidenceAssessment",
    "assess_confidence",
]
