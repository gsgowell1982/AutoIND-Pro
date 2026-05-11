"""Logical table AST construction for the PDF parser.

Architecture:
    Continuum Engine -> Logical Table AST

This layer shapes the final table payload while preserving three distinct views:
- `raw_*`: audit evidence as observed in the source table region
- `display_*`: opening rows plus data rows for human-oriented reconstruction
- business/default fields (`grid` / `row_texts` / `row_count`): data rows only
"""

from __future__ import annotations

# Version: v1.0.2
# Optimization Summary:
# - Preserve raw row-level audit evidence alongside semantic business rows.
# - Expose semantic row_count/grid/row_texts for downstream compliance analysis.
# - Separate default business rows from display-layer opening rows so internal
#   table titles remain metadata rather than business records.

from dataclasses import dataclass, field
from typing import Any

from .assembly import TableInstance
from .continuum import ContinuumResult


@dataclass
class LogicalTableAST:
    """Final logical table payload."""

    block_type: str = "table"
    table_id: str = ""
    page: int = 0
    bbox: list[float] = field(default_factory=list)

    header: list[dict[str, Any]] = field(default_factory=list)
    cells: list[dict[str, Any]] = field(default_factory=list)

    # Business-facing rows: data rows only.
    grid: list[list[str | None]] = field(default_factory=list)
    row_count: int = 0
    row_texts: list[str] = field(default_factory=list)

    # Explicit business aliases for downstream rule engines.
    data_grid: list[list[str | None]] = field(default_factory=list)
    data_row_count: int = 0
    data_row_texts: list[str] = field(default_factory=list)

    # Display-layer rows: preserve opening rows for UI/reconstruction.
    display_grid: list[list[str | None]] = field(default_factory=list)
    display_row_count: int = 0
    display_row_texts: list[str] = field(default_factory=list)

    # Raw audit evidence.
    raw_grid: list[list[str | None]] = field(default_factory=list)
    raw_row_count: int = 0
    raw_row_texts: list[str] = field(default_factory=list)
    structural_empty_rows: list[int] = field(default_factory=list)

    col_count: int = 0
    logical_row_count: int = 0
    logical_col_count: int = 0
    physical_row_count: int = 0
    physical_col_count: int = 0

    column_signature: list[float] = field(default_factory=list)
    column_hash: str = ""

    near_page_top: bool = False
    near_page_bottom: bool = False

    is_continuation: bool = False
    continued_from: str | None = None
    continued_to: list[str] = field(default_factory=list)
    header_inherited: bool = False
    title_inherited: bool = False
    continuation_source: dict[str, Any] | None = None

    title: str | None = None
    section_hint: str | None = None
    toc_context: bool = False
    grid_line_score: float = 0.0
    detection_method: str = "pymupdf_builtin"
    normalization_strategy: str = "column_clustering"

    title_row_index: int | None = None
    header_row_index: int | None = None
    data_start_row: int = 0

    confidence: float = 0.0
    risk_flags: list[str] = field(default_factory=list)
    review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)
    nested_structure: dict[str, Any] | None = None

    structure_score: float = 0.0
    toc_row_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to the payload expected by downstream parser consumers."""

        result = {
            "block_type": self.block_type,
            "table_id": self.table_id,
            "page": self.page,
            "bbox": self.bbox,
            "header": self.header,
            "cells": self.cells,
            "grid": self.grid,
            "row_count": self.row_count,
            "row_texts": self.row_texts,
            "data_grid": self.data_grid,
            "data_row_count": self.data_row_count,
            "data_row_texts": self.data_row_texts,
            "display_grid": self.display_grid,
            "display_row_count": self.display_row_count,
            "display_row_texts": self.display_row_texts,
            "raw_grid": self.raw_grid,
            "raw_row_count": self.raw_row_count,
            "raw_row_texts": self.raw_row_texts,
            "structural_empty_rows": self.structural_empty_rows,
            "col_count": self.col_count,
            "logical_row_count": self.logical_row_count,
            "logical_col_count": self.logical_col_count,
            "physical_row_count": self.physical_row_count,
            "physical_col_count": self.physical_col_count,
            "column_signature": self.column_signature,
            "column_hash": self.column_hash,
            "near_page_top": self.near_page_top,
            "near_page_bottom": self.near_page_bottom,
            "detection_method": self.detection_method,
            "structure_score": self.structure_score,
            "toc_row_ratio": self.toc_row_ratio,
            "data_start_row": self.data_start_row,
        }

        if self.is_continuation:
            result["is_continuation"] = True
        if self.continued_from:
            result["continued_from"] = self.continued_from
        if self.continued_to:
            result["continued_to"] = self.continued_to
        if self.header_inherited:
            result["header_inherited"] = True
        if self.title_inherited:
            result["title_inherited"] = True
        if self.continuation_source:
            result["continuation_source"] = self.continuation_source
        if self.title:
            result["title"] = self.title
        if self.section_hint:
            result["section_hint"] = self.section_hint
        if self.title_row_index is not None:
            result["title_row_index"] = self.title_row_index
        if self.header_row_index is not None:
            result["header_row_index"] = self.header_row_index
        if self.confidence > 0:
            result["confidence"] = round(self.confidence, 3)
        if self.risk_flags:
            result["risk_flags"] = self.risk_flags
        if self.review_required:
            result["review_required"] = True
        if self.review_reasons:
            result["review_reasons"] = self.review_reasons
        if self.nested_structure:
            result["nested_structure"] = self.nested_structure

        return result


def build_logical_ast(
    instance: TableInstance,
    continuum_result: ContinuumResult,
) -> LogicalTableAST:
    """Combine assembly output and continuum metadata into the final AST."""

    header_list = instance.header.cells.copy() if instance.header else []
    structure_score = _calculate_structure_score(instance)
    toc_row_ratio = _calculate_toc_row_ratio(instance)

    ast = LogicalTableAST(
        block_type="table",
        table_id=instance.table_id,
        page=instance.page_number,
        bbox=list(instance.bbox),
        header=header_list,
        cells=instance.cells,
        grid=instance.data_grid,
        row_count=instance.data_row_count,
        row_texts=instance.data_row_texts,
        data_grid=instance.data_grid,
        data_row_count=instance.data_row_count,
        data_row_texts=instance.data_row_texts,
        display_grid=instance.grid,
        display_row_count=instance.row_count,
        display_row_texts=instance.row_texts,
        raw_grid=instance.raw_grid,
        raw_row_count=instance.raw_row_count,
        raw_row_texts=instance.raw_row_texts,
        structural_empty_rows=instance.structural_empty_rows,
        col_count=instance.col_count,
        logical_row_count=instance.data_row_count,
        logical_col_count=instance.col_count,
        physical_row_count=instance.physical_row_count,
        physical_col_count=instance.physical_col_count,
        column_signature=instance.column_signature,
        column_hash=instance.column_hash,
        near_page_top=instance.near_page_top,
        near_page_bottom=instance.near_page_bottom,
        is_continuation=instance.is_continuation or continuum_result.continuity.is_continuation,
        header_inherited=instance.header.inherited if instance.header else False,
        title=instance.title,
        section_hint=instance.section_hint,
        toc_context=instance.toc_context,
        grid_line_score=instance.grid_line_score,
        detection_method=instance.detection_method,
        normalization_strategy=instance.normalization_strategy,
        title_row_index=instance.title_row_index,
        header_row_index=instance.header_row_index,
        data_start_row=instance.data_start_row,
        confidence=continuum_result.confidence.overall_confidence,
        risk_flags=continuum_result.confidence.risk_flags,
        review_required=continuum_result.confidence.review_required,
        review_reasons=continuum_result.confidence.review_reasons,
        nested_structure=_build_nested_structure_payload(continuum_result),
        structure_score=structure_score,
        toc_row_ratio=toc_row_ratio,
    )

    if continuum_result.continuity.is_continuation:
        ast.continued_from = continuum_result.continuity.parent_table_id
        if continuum_result.continuity.inherited_fields:
            ast.continuation_source = {
                "source_table_id": continuum_result.continuity.parent_table_id,
                "strategy": "continuum_engine",
                "similarity": continuum_result.continuity.similarity_score,
                "inherited_fields": continuum_result.continuity.inherited_fields,
            }

    return ast


def _build_nested_structure_payload(continuum_result: ContinuumResult) -> dict[str, Any] | None:
    nested = getattr(continuum_result, "nested", None)
    if not nested or not getattr(nested, "has_nested", False):
        return None

    regions = list(getattr(nested, "nested_regions", []) or [])
    if not regions:
        return None

    return {
        "has_nested": True,
        "nesting_type": getattr(nested, "nesting_type", None),
        "region_count": len(regions),
        "regions": regions,
    }


def _calculate_structure_score(instance: TableInstance) -> float:
    """Estimate structural strength using business rows, not opening metadata rows."""

    if not instance.cells:
        return 0.0

    effective_row_count = int(instance.data_row_count or instance.row_count or 0)
    effective_grid = list(instance.data_grid or instance.grid or [])

    non_null_cells = sum(1 for cell in instance.cells if cell.get("text"))
    total_slots = effective_row_count * instance.col_count
    coverage = non_null_cells / max(1, total_slots)

    aligned_rows = sum(1 for row in effective_grid if any(cell for cell in row))
    alignment = aligned_rows / max(1, len(effective_grid))

    numeric_cells = sum(
        1
        for cell in instance.cells
        if cell.get("text") and any(ch.isdigit() for ch in cell["text"])
    )
    numeric_ratio = numeric_cells / max(1, non_null_cells)

    return min(1.0, 0.45 * coverage + 0.35 * alignment + 0.2 * numeric_ratio)


def _calculate_toc_row_ratio(instance: TableInstance) -> float:
    """Estimate TOC-likeness from business rows only."""

    effective_row_texts = list(instance.data_row_texts or instance.row_texts or [])
    if not effective_row_texts:
        return 0.0

    import re

    toc_patterns = [
        r"\.{4,}\s*\d{1,3}\s*$",
        r"^\s*\d+(?:\.\d+){1,4}\s+\S+\s+\d{1,3}\s*$",
    ]

    toc_rows = 0.0
    for row_text in effective_row_texts:
        for pattern in toc_patterns:
            if re.search(pattern, row_text):
                toc_rows += 1.0
                break

    return min(1.0, toc_rows / len(effective_row_texts))


def ast_to_legacy_dict(ast: LogicalTableAST) -> dict[str, Any]:
    """Convert AST to dict for downstream compatibility."""

    return ast.to_dict()


def merge_ast_with_context(
    ast: LogicalTableAST,
    title_block: dict[str, Any] | None = None,
    section_hint_block: dict[str, Any] | None = None,
    toc_context: bool = False,
    grid_line_score: float = 0.0,
    continuation_hint: dict[str, Any] | None = None,
) -> LogicalTableAST:
    """Merge context-layer metadata into an existing AST."""

    if title_block:
        ast.title = title_block.get("text", "").strip()

    if section_hint_block:
        ast.section_hint = section_hint_block.get("text", "").strip()

    ast.toc_context = toc_context
    ast.grid_line_score = grid_line_score

    if continuation_hint:
        ast.continued_from = continuation_hint.get("table_id")
        ast.is_continuation = True

    return ast


__all__ = [
    "LogicalTableAST",
    "build_logical_ast",
    "ast_to_legacy_dict",
    "merge_ast_with_context",
    "_calculate_structure_score",
    "_calculate_toc_row_ratio",
]
