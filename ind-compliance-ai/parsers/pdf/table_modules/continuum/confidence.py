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
    header_candidate_count = len(context.get("header_candidates") or []) if context else 0
    raw_source = str(context.get("raw_source", "") or "") if context else ""
    raw_rule_or_box_support_count = int(context.get("raw_rule_or_box_support_count", 0) or 0) if context else 0
    raw_has_rule_or_box_support = bool(context.get("raw_has_rule_or_box_support", False)) if context else False

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
        header_candidate_count=header_candidate_count,
        toc_row_ratio=toc_row_ratio,
        reference_like_ratio=reference_like_ratio,
        raw_source=raw_source,
        raw_rule_or_box_support_count=raw_rule_or_box_support_count,
        raw_has_rule_or_box_support=raw_has_rule_or_box_support,
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
    header_candidate_count: int,
    toc_row_ratio: float,
    reference_like_ratio: float,
    raw_source: str = "",
    raw_rule_or_box_support_count: int = 0,
    raw_has_rule_or_box_support: bool = False,
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

    if _validate_structured_low_column_candidate(
        instance=instance,
        row_count=row_count,
        col_count=col_count,
        structure_score=score,
        raw_source=raw_source,
        raw_rule_or_box_support_count=raw_rule_or_box_support_count,
        raw_has_rule_or_box_support=raw_has_rule_or_box_support,
        grid_line_score=grid_line_score,
        toc_row_ratio=toc_row_ratio,
        reference_like_ratio=reference_like_ratio,
    ):
        return True

    if title_block is not None:
        if col_count == 1:
            return row_count >= 4 and score >= 0.15
        if row_count == 1 and col_count >= 2 and grid_line_score >= 0.45:
            return score >= 0.15
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

    if _validate_keyed_long_description_candidate(
        instance=instance,
        row_count=row_count,
        col_count=col_count,
        structure_score=score,
        raw_source=raw_source,
        toc_row_ratio=toc_row_ratio,
        reference_like_ratio=reference_like_ratio,
    ):
        return True

    if _validate_compact_wide_schema_candidate(
        instance=instance,
        row_count=row_count,
        col_count=col_count,
        structure_score=score,
        raw_source=raw_source,
        toc_row_ratio=toc_row_ratio,
        reference_like_ratio=reference_like_ratio,
    ):
        return True

    if grid_line_score >= 0.45:
        if row_count == 1 and col_count >= 2 and score >= 0.30 and reference_like_ratio < 0.25:
            return True
        return row_count >= 4 and col_count >= 2 and score >= 0.56

    if grid_line_score >= 0.30:
        if row_count >= 4 and col_count >= 2 and score >= 0.60 and header_candidate_count >= 2:
            return True

    if instance.is_continuation and row_count >= 2 and col_count >= 2:
        return score >= 0.15

    strong_multi_col = row_count >= 5 and col_count >= 3 and score >= 0.85 and toc_row_ratio <= 0.15
    strong_two_col = row_count >= 6 and col_count >= 2 and score >= 0.85 and toc_row_ratio <= 0.1
    decent_table = row_count >= 4 and col_count >= 2 and score >= 0.7 and toc_row_ratio <= 0.2
    return strong_multi_col or strong_two_col or decent_table


def _validate_compact_wide_schema_candidate(
    *,
    instance: "TableInstance",
    row_count: int,
    col_count: int,
    structure_score: float,
    raw_source: str,
    toc_row_ratio: float,
    reference_like_ratio: float,
) -> bool:
    """Accept short, wide schema tables common in nonclinical overviews."""

    if raw_source != "text_aligned_borderless_grid":
        return False
    if row_count < 3 or col_count < 5:
        return False
    if structure_score < 0.68:
        return False
    if toc_row_ratio >= 0.25 or reference_like_ratio >= 0.35:
        return False

    grid = instance.grid or instance.raw_grid or []
    if len(grid) < 3 or not isinstance(grid[0], list):
        return False

    header_cells = [str(cell or "").strip() for cell in grid[0][:col_count]]
    header_filled = [text for text in header_cells if text]
    compact_header_cells = [
        text
        for text in header_filled
        if len(text) <= 80
        and len(text.split()) <= 8
        and bool(re.search(r"[A-Za-z\u4e00-\u9fff]", text))
    ]
    header_numeric_cells = sum(1 for text in header_filled[1:] if _looks_like_table_value_atom(text))
    schema_opening = len(compact_header_cells) >= 3
    stub_value_opening = (
        len(header_filled) >= 4
        and bool(re.search(r"[A-Za-z\u4e00-\u9fff]", header_filled[0]))
        and header_numeric_cells >= 3
    )
    if not schema_opening and not stub_value_opening:
        return False

    dense_rows = 0
    data_rows = 0
    for row in grid[1:]:
        if not isinstance(row, list):
            continue
        values = [str(cell or "").strip() for cell in row[:col_count]]
        filled = [text for text in values if text]
        if len(filled) >= max(3, min(5, col_count - 1)):
            dense_rows += 1
        value_atoms = sum(1 for text in filled if _looks_like_table_value_atom(text))
        text_atoms = sum(1 for text in filled if re.search(r"[A-Za-z\u4e00-\u9fff]", text))
        if value_atoms >= 2 or (value_atoms >= 1 and text_atoms >= 1):
            data_rows += 1

    return dense_rows >= 2 and data_rows >= 2


def _looks_like_table_value_atom(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    if re.fullmatch(r"[-+*/#a-zA-Z]?", candidate):
        return False
    if re.search(r"\d", candidate):
        return True
    if candidate.lower() in {"yes", "no", "na", "n/a", "nd"}:
        return True
    return bool(re.fullmatch(r"[A-Z]{1,4}[-/]?[A-Z0-9]{0,6}", candidate))


def _validate_keyed_long_description_candidate(
    *,
    instance: "TableInstance",
    row_count: int,
    col_count: int,
    structure_score: float,
    raw_source: str,
    toc_row_ratio: float,
    reference_like_ratio: float,
) -> bool:
    """Accept sparse tables with compact key columns and long description cells."""
    if raw_source != "word_clustering":
        return False
    if row_count < 8 or col_count < 4:
        return False
    if structure_score < 0.40:
        return False
    if toc_row_ratio >= 0.20 or reference_like_ratio >= 0.30:
        return False

    grid = instance.grid or instance.raw_grid or []
    if len(grid) < row_count:
        return False

    header_rows = grid[: min(4, len(grid))]
    header_cells = [
        str(cell or "").strip()
        for row in header_rows
        if isinstance(row, list)
        for cell in row
        if str(cell or "").strip()
    ]
    if not _header_cells_look_like_long_description_schema(header_cells):
        return False

    keyed_rows = 0
    continuation_rows = 0
    long_description_cells = 0
    compact_key_cells = 0
    for row in grid[len(header_rows) :]:
        if not isinstance(row, list):
            continue
        values = [str(cell or "").strip() for cell in row[:col_count]]
        filled_indices = [idx for idx, text in enumerate(values) if text]
        if not filled_indices:
            continue
        for text in values:
            if _looks_like_compact_key_or_flag_cell(text):
                compact_key_cells += 1
            if _looks_like_long_description_cell(text):
                long_description_cells += 1
        if len(filled_indices) >= max(3, col_count - 1):
            keyed_rows += 1
            continue
        if len(filled_indices) == 1 and filled_indices[0] >= min(3, col_count - 2):
            continuation_rows += 1

    return (
        keyed_rows >= 2
        and continuation_rows >= 4
        and compact_key_cells >= 4
        and long_description_cells >= 4
    )


def _header_cells_look_like_long_description_schema(cells: list[str]) -> bool:
    if len(cells) < 4:
        return False
    compact_cells = sum(1 for text in cells if len(text) <= 64 and len(text.split()) <= 6)
    if compact_cells < max(3, len(cells) - 1):
        return False
    schema_terms = {
        "category",
        "description",
        "dose",
        "endpoint",
        "item",
        "jurisdiction",
        "ownership",
        "parameter",
        "permitted",
        "reference",
        "reporting",
        "requirement",
        "reservation",
        "restriction",
        "result",
        "route",
        "species",
        "status",
        "study",
        "type",
        "unit",
        "value",
    }
    hits = 0
    for text in cells:
        tokens = re.findall(r"[A-Za-z\u4e00-\u9fff]+", text.lower())
        if any(token in schema_terms for token in tokens):
            hits += 1
    return hits >= 2


def _looks_like_compact_key_or_flag_cell(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    if len(candidate) <= 3 and re.fullmatch(r"[A-Za-z0-9()+/\-]+", candidate):
        return True
    return len(candidate) <= 32 and len(candidate.split()) <= 3 and bool(re.search(r"[A-Za-z\u4e00-\u9fff]", candidate))


def _looks_like_long_description_cell(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    if len(re.findall(r"[\u4e00-\u9fff]", candidate)) >= 10:
        return True
    return len(candidate) >= 28 and len(candidate.split()) >= 4


def _validate_structured_low_column_candidate(
    *,
    instance: "TableInstance",
    row_count: int,
    col_count: int,
    structure_score: float,
    raw_source: str,
    raw_rule_or_box_support_count: int,
    raw_has_rule_or_box_support: bool,
    grid_line_score: float,
    toc_row_ratio: float,
    reference_like_ratio: float,
) -> bool:
    """Accept compact ruled/list tables that are not shaped like wide grids.

    This branch is intentionally evidence-driven rather than source-exclusive:
    the low-column detector only admits candidates with local line/box support
    or stable row structure, and this validator carries that evidence forward
    so one-column and two-column tables are not rejected by wide-data thresholds.
    """

    source_support = raw_source in {
        "structured_text_region",
        "visual_structure_grid",
        "embedded_image_ocr",
        "vector_ocr",
    }
    if not source_support:
        return False
    if col_count < 1 or col_count > 2:
        return False
    if toc_row_ratio >= 0.30 or reference_like_ratio >= 0.55:
        return False

    row_texts = [str(text or "").strip() for text in instance.row_texts if str(text or "").strip()]
    if len(row_texts) < row_count:
        row_texts = [str(text or "").strip() for text in instance.raw_row_texts if str(text or "").strip()]
    if len(row_texts) < 3:
        return False

    local_structure_support = (
        raw_has_rule_or_box_support
        or raw_rule_or_box_support_count >= 1
        or grid_line_score >= 0.30
    )
    if not local_structure_support:
        return False

    narrative_rows = sum(1 for text in row_texts if _looks_like_narrative_sentence(text))
    compact_rows = sum(1 for text in row_texts if len(text) <= 140 and not re.search(r"[.。！？!?；;]\s*$", text))
    marker_rows = sum(
        1
        for text in row_texts
        if re.match(r"^\s*(?:[#*+\-•]?\s*)?(?:#?\d+[:.)]|[A-Za-z][.)])\s+", text)
    )
    if narrative_rows >= max(2, len(row_texts) // 3):
        return False

    if col_count == 1:
        return (
            row_count >= 4
            and structure_score >= 0.55
            and (marker_rows >= 2 or compact_rows >= max(4, len(row_texts) - 1))
        )

    multi_cell_rows = 0
    for row in instance.grid or instance.raw_grid or []:
        if isinstance(row, list) and sum(1 for cell in row if str(cell or "").strip()) >= 2:
            multi_cell_rows += 1
    return (
        row_count >= 3
        and structure_score >= 0.55
        and multi_cell_rows >= 2
        and compact_rows >= max(2, len(row_texts) // 2)
    )


def _looks_like_narrative_sentence(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", candidate))
    if cjk_count >= 18:
        return True
    if cjk_count >= 10 and re.search(r"[，,。；;：:、]\s*$", candidate):
        return True
    if len(candidate) >= 90 and re.search(r"[.。！？!?；;，,：:]\s*$", candidate):
        return True
    if len(candidate.split()) >= 14 and re.search(r"[.。！？!?；;，,：:]\s*$", candidate):
        return True
    return False


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
