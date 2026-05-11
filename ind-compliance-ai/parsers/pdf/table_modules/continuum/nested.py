"""Phase 5: Nested Structure Detection - nested-table diagnostics.

Architecture:
    Table Instance + Cell Semantics -> Phase 5 -> Nested Structure

This phase currently implements a conservative, non-destructive first step:
- detect high-confidence suspected sub-tables inside merged cells
- keep ordinary merged title/header/description cells out of the nested path
- expose provenance-style diagnostics without changing business-facing table rows
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..assembly import TableInstance
    from .semantics import CellSemantics


@dataclass
class NestedStructure:
    """Phase 5 result for nested-structure diagnostics."""

    has_nested: bool
    nested_regions: list[dict[str, Any]] = field(default_factory=list)
    nesting_type: str | None = None


def detect_nested_structure(
    instance: "TableInstance",
    cell_semantics: dict[tuple[int, int], "CellSemantics"],
) -> NestedStructure:
    """Detect likely nested sub-table regions inside cells.

    The current implementation is intentionally narrow. A region is only marked
    when merged-cell geometry and the cell's own text strongly suggest an
    internal local table rather than a normal merged title/header/description.
    """

    del cell_semantics

    nested_regions: list[dict[str, Any]] = []
    for cell in instance.cells:
        region = _detect_suspected_nested_region(cell)
        if region:
            nested_regions.append(region)

    has_nested = bool(nested_regions)
    return NestedStructure(
        has_nested=has_nested,
        nested_regions=nested_regions,
        nesting_type="suspected_subtable" if has_nested else None,
    )


def _detect_suspected_nested_region(cell: dict[str, Any]) -> dict[str, Any] | None:
    rowspan = _safe_int(cell.get("rowspan"), default=1)
    colspan = _safe_int(cell.get("colspan"), default=1)
    text = str(cell.get("text") or "").strip()
    if not text:
        return None

    score = 0.0
    signals: list[str] = []

    if rowspan >= 2 and colspan >= 2:
        score += 0.5
        signals.append("merged_rows_and_cols")
    elif rowspan >= 3 or colspan >= 3:
        score += 0.2
        signals.append("large_merged_span")
    else:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 3:
        score += 0.15
        signals.append("multi_line_content")

    aligned_line_count = _aligned_token_count(lines)
    if aligned_line_count >= 2:
        score += 0.3
        signals.append("line_internal_alignment")

    if len(text) >= 80:
        score += 0.1
        signals.append("dense_content")

    if _looks_like_plain_title_or_section(text):
        score -= 0.35
        signals.append("title_like_text")

    if score < 0.75:
        return None

    region: dict[str, Any] = {
        "row": cell.get("row"),
        "col": cell.get("col"),
        "rowspan": rowspan,
        "colspan": colspan,
        "type": "suspected_nested_subtable",
        "confidence": round(min(score, 0.99), 2),
        "signals": signals,
        "text_preview": text[:160],
    }
    bbox = _coerce_bbox(cell.get("bbox"))
    if bbox is not None:
        region["bbox"] = bbox
    return region


def _aligned_token_count(lines: list[str]) -> int:
    aligned = 0
    for line in lines:
        tokens = [token.strip() for token in re.split(r"(?:\t+|\s{2,})", line) if token.strip()]
        if len(tokens) >= 2:
            aligned += 1
    return aligned


def _looks_like_plain_title_or_section(text: str) -> bool:
    compact = " ".join(text.split())
    if "\n" in text:
        return False
    return bool(re.match(r"^(?:Table|TABLE|Figure|FIGURE|Appendix|Section)\b", compact))


def _coerce_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "NestedStructure",
    "detect_nested_structure",
]
