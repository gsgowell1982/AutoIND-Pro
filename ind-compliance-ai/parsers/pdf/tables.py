"""Table parsing for PDF documents - Unified Architecture.

统一架构:
    PDF
    └─ PyMuPDF (物理层) - raw text extraction
        └─ Raw Objects Layer (table_modules.raw_objects) - 原始证据提取
            └─ Normalization Layer (table_modules.normalization) - 物理证据规范化
                └─ Assembly Layer (table_modules.assembly) - 表格实例组装
                    └─ Continuum Engine (table_modules.continuum) - 6 Phase 处理
                        └─ AST Layer (table_modules.ast) - 逻辑表格 AST

6 Phase 处理流程:
    Phase 1: Table Identity Resolution - 表格身份判定
    Phase 2: Logical Grid Stabilization - 逻辑网格稳定
    Phase 3: Cross-Page Continuity - 跨页连续性
    Phase 4: Cell Semantics & State Machine - 单元格语义状态机
    Phase 5: Nested Structure Detection - 嵌套结构识别
    Phase 6: Confidence, Risk & Review Policy - 置信度评估

Usage:
    from parsers.pdf.tables import extract_tables_from_page

    tables = extract_tables_from_page(page, page_number, page_height)
"""

from __future__ import annotations

import re
from typing import Any

# ============================================================================
# New Architecture Imports
# ============================================================================

from .table_modules.raw_objects import (
    RawTableEvidence,
    RawSpan,
    RawCell,
    RawRow,
    RawChar,
    RawDrawing,
    DrawingType,
    extract_raw_evidence_from_pymupdf,
    extract_raw_evidence_from_words,
)

from .table_modules.normalization import (
    NormalizedTable,
    NormalizedCell,
    NormalizedRow,
    ColumnCluster,
    normalize_raw_evidence,
)

from .table_modules.assembly import (
    TableInstance,
    TableHeader,
    assemble_table_instance,
)

from .table_modules.continuum import (
    # Phase 1
    TableIdentity,
    IdentityResolution,
    resolve_table_identity,
    # Phase 2
    GridStabilization,
    stabilize_logical_grid,
    # Phase 3
    ContinuityResult,
    establish_cross_page_continuity,
    # Phase 4
    CellState,
    CellSemantics,
    analyze_cell_semantics,
    # Phase 5
    NestedStructure,
    detect_nested_structure,
    # Phase 6
    ConfidenceAssessment,
    assess_confidence,
    TOC_LINE_PATTERN,
    REFERENCE_ROW_PATTERN,
    # Pipeline
    ContinuumResult,
    run_continuum_engine,
    # Utilities
    column_similarity,
    header_similarity,
)

from .table_modules.postprocess import (
    stitch_cross_page_tables,
    can_merge_table_fragments_on_same_page,
    merge_two_table_fragments,
    merge_same_page_table_fragments,
    can_merge_tables_with_same_title,
    merge_same_title_tables,
    renumber_table_ids,
    deduplicate_cells,
    sort_cells_by_position,
)

from .table_modules.ast import (
    LogicalTableAST,
    build_logical_ast,
    ast_to_legacy_dict,
    merge_ast_with_context,
    _calculate_structure_score,
    _calculate_toc_row_ratio,
)

from .shared import _Word


# ============================================================================
# Patterns for Context Detection
# ============================================================================

TABLE_TITLE_PATTERN = re.compile(
    r"^\s*(?:附?表\s*[0-9A-Za-z一二三四五六七八九十零〇.\-]+|table\s*[0-9A-Za-z.\-]+)\s*(?:[.:：、\-]|\s+)",
    re.IGNORECASE,
)
TABLE_SECTION_HINT_PATTERN = re.compile(
    r"(术语表|词汇表|名词表|名词解释|参数表|清单|附录表|glossary)",
    re.IGNORECASE,
)
TOC_HEADING_PATTERN = re.compile(r"^\s*(目录|contents)\s*$", re.IGNORECASE)


# ============================================================================
# Main Entry Points
# ============================================================================

def extract_tables_from_page(
    page: Any,
    page_number: int,
    page_height: float,
    text_blocks: list[dict[str, Any]] | None = None,
    page_drawings: list[dict[str, Any]] | None = None,
    prev_tables: list[dict[str, Any]] | None = None,
    table_counter: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Extract tables from a single PDF page.

    使用统一的新架构处理流程：
    1. Raw Objects Extraction - 原始证据提取
    2. Normalization - 物理证据规范化
    3. Assembly - 表格实例组装
    4. Continuum Engine - 6 Phase 处理
    5. AST Building - 构建逻辑表格 AST

    Args:
        page: PyMuPDF page object
        page_number: Page number (1-indexed)
        page_height: Page height in points
        text_blocks: Optional pre-extracted text blocks for context
        page_drawings: Optional pre-extracted drawings for grid detection
        prev_tables: Tables from previous pages (for continuation detection)
        table_counter: Starting counter for table IDs

    Returns:
        Tuple of (list of table ASTs, updated table counter)
    """
    tables: list[dict[str, Any]] = []

    # Get parent context for continuation
    parent_col_count = None
    parent_header = None
    parent_bbox = None

    if prev_tables:
        for prev in reversed(prev_tables):
            if prev.get("near_page_bottom") and prev.get("page") == page_number - 1:
                parent_col_count = prev.get("col_count")
                parent_header = prev.get("header")
                # Extract parent bbox for column boundary inheritance
                # This ensures continuation tables align with parent's column structure
                if prev.get("bbox"):
                    parent_bbox = tuple(prev.get("bbox"))
                break

    page_width = page.rect.width if hasattr(page, 'rect') else 612.0

    # Strategy 1: PyMuPDF built-in detection
    raw_evidence = extract_raw_evidence_from_pymupdf(
        page=page,
        page_number=page_number,
        page_height=page_height,
        page_width=page_width,
    )

    if raw_evidence:
        table_ast = _process_raw_evidence(
            raw_evidence=raw_evidence,
            page=page,
            page_number=page_number,
            page_height=page_height,
            text_blocks=text_blocks,
            page_drawings=page_drawings,
            prev_tables=prev_tables,
            table_counter=table_counter,
            parent_col_count=parent_col_count,
            parent_header=parent_header,
            parent_bbox=parent_bbox,
        )

        if table_ast:
            tables.append(table_ast)
            table_counter += 1

    # Strategy 2: Word-clustering fallback
    if not tables:
        words = _extract_words_from_page(page)
        if words:
            raw_evidence_fallback = extract_raw_evidence_from_words(
                words=words,
                page_number=page_number,
                page_height=page_height,
                page_width=page_width,
            )

            if raw_evidence_fallback:
                table_ast = _process_raw_evidence(
                    raw_evidence=raw_evidence_fallback,
                    page=page,
                    page_number=page_number,
                    page_height=page_height,
                    text_blocks=text_blocks,
                    page_drawings=page_drawings,
                    prev_tables=prev_tables,
                    table_counter=table_counter,
                    parent_col_count=parent_col_count,
                    parent_header=parent_header,
                    parent_bbox=parent_bbox,
                )

                if table_ast:
                    tables.append(table_ast)
                    table_counter += 1

    return tables, table_counter


def _process_raw_evidence(
    raw_evidence: RawTableEvidence,
    page: Any,
    page_number: int,
    page_height: float,
    text_blocks: list[dict[str, Any]] | None,
    page_drawings: list[dict[str, Any]] | None,
    prev_tables: list[dict[str, Any]] | None,
    table_counter: int,
    parent_col_count: int | None,
    parent_header: list[dict[str, Any]] | None,
    parent_bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, Any] | None:
    """Process raw evidence through the full pipeline."""

    # Phase 1: Normalization
    # Pass parent_bbox for column boundary inheritance in continuation tables
    normalized = normalize_raw_evidence(
        raw_evidence=raw_evidence,
        parent_col_count=parent_col_count,
        parent_bbox=parent_bbox,
    )

    # Phase 2: Assembly
    table_id = f"tbl_{table_counter + 1:03d}"
    instance = assemble_table_instance(
        normalized=normalized,
        table_id=table_id,
        parent_header=parent_header,
    )

    # Phase 3: Get context for Continuum Engine
    context = _build_context(
        instance=instance,
        text_blocks=text_blocks,
        page_drawings=page_drawings,
        page_height=page_height,
    )

    # Phase 4: Continuum Engine
    prev_instances = _dict_to_instances(prev_tables) if prev_tables else []
    continuum_result = run_continuum_engine(instance, prev_instances, context)

    # Phase 5: Validation check
    if not continuum_result.confidence.is_valid_table:
        return None

    if continuum_result.confidence.overall_confidence < 0.3:
        return None

    # Phase 6: Build AST
    ast = build_logical_ast(instance, continuum_result)

    # Enrich with context
    if context.get("title_block"):
        ast.title = context["title_block"].get("text", "").strip()
    if context.get("section_hint"):
        ast.section_hint = context["section_hint"].get("text", "").strip()
    ast.toc_context = context.get("toc_context", False)
    ast.grid_line_score = context.get("grid_line_score", 0.0)

    # Post-process cells
    _postprocess_cells(ast)

    return ast.to_dict()


def _build_context(
    instance: TableInstance,
    text_blocks: list[dict[str, Any]] | None,
    page_drawings: list[dict[str, Any]] | None,
    page_height: float,
) -> dict[str, Any]:
    """Build context for Continuum Engine."""
    context = {
        "title_block": None,
        "section_hint": None,
        "toc_context": False,
        "grid_line_score": 0.0,
    }

    if not text_blocks:
        return context

    bbox = tuple(instance.bbox)

    # Find title
    candidates: list[tuple[float, dict[str, Any]]] = []
    for block in text_blocks:
        text = block.get("text", "").strip()
        if not text or not TABLE_TITLE_PATTERN.search(text):
            continue
        block_bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
        gap = bbox[1] - block_bbox[3]
        if gap < -8 or gap > 160:
            continue
        overlap = _horizontal_overlap_ratio(bbox, block_bbox)
        if overlap < 0.12:
            continue
        candidates.append((max(0, gap), block))

    if candidates:
        context["title_block"] = candidates[0][1]

    # Find section hint
    for block in text_blocks:
        text = block.get("text", "").strip()
        if not text or not TABLE_SECTION_HINT_PATTERN.search(text):
            continue
        block_bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
        gap = bbox[1] - block_bbox[3]
        if 0 <= gap <= 180:
            overlap = _horizontal_overlap_ratio(bbox, block_bbox)
            if overlap >= 0.1:
                context["section_hint"] = block
                break

    # Check TOC context
    table_top = bbox[1]
    for block in text_blocks:
        text = block.get("text", "").strip()
        if not text:
            continue
        block_bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
        if block_bbox[3] > table_top + 15:
            continue
        vertical_gap = table_top - block_bbox[3]
        if vertical_gap < 0 or vertical_gap > 220:
            continue
        if TOC_HEADING_PATTERN.search(text):
            context["toc_context"] = True
            break

    # Calculate grid score
    if page_drawings:
        count = 0
        for drawing in page_drawings:
            rect = drawing.get("rect")
            if rect:
                if (bbox[0] <= rect[0] <= bbox[2] and bbox[1] <= rect[1] <= bbox[3]):
                    count += 1
        context["grid_line_score"] = min(1.0, count / 8.0)

    return context


def _horizontal_overlap_ratio(bbox_a: tuple, bbox_b: tuple) -> float:
    """Calculate horizontal overlap ratio."""
    x_overlap = max(0, min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0]))
    width_a = bbox_a[2] - bbox_a[0]
    width_b = bbox_b[2] - bbox_b[0]
    if width_a <= 0 or width_b <= 0:
        return 0.0
    return x_overlap / min(width_a, width_b)


def _postprocess_cells(ast: LogicalTableAST) -> None:
    """Post-process cells in AST."""
    cells = ast.cells
    if not cells:
        return

    # Deduplicate
    seen: set[tuple[int, int]] = set()
    unique_cells: list[dict[str, Any]] = []
    for cell in cells:
        key = (cell.get("logical_row", cell.get("row", 0)), cell.get("col", 0))
        if key not in seen:
            seen.add(key)
            unique_cells.append(cell)
    ast.cells = unique_cells

    # Sort by position
    ast.cells = sorted(
        ast.cells,
        key=lambda c: (c.get("logical_row", c.get("row", 0)), c.get("col", 0))
    )


def extract_tables_from_document(
    doc: Any,
) -> list[dict[str, Any]]:
    """Extract all tables from a PDF document.

    This function:
    1. Extracts tables from each page
    2. Stitches cross-page continuation tables
    3. Merges same-page fragments
    4. Renumbers table IDs

    Args:
        doc: PyMuPDF document object

    Returns:
        List of all table ASTs from the document
    """
    all_tables: list[dict[str, Any]] = []
    table_counter = 0
    prev_tables: list[dict[str, Any]] = []
    page_heights: dict[int, float] = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        page_heights[page_num + 1] = page_height

        # Get context data
        text_blocks = _get_text_blocks(page)
        page_drawings = _get_drawings(page)

        tables, table_counter = extract_tables_from_page(
            page=page,
            page_number=page_num + 1,
            page_height=page_height,
            text_blocks=text_blocks,
            page_drawings=page_drawings,
            prev_tables=prev_tables,
            table_counter=table_counter,
        )

        # Merge same-page fragments
        if len(tables) > 1:
            tables, _ = merge_same_page_table_fragments(tables)

        all_tables.extend(tables)
        prev_tables = all_tables.copy()

    # Post-process across pages
    stitch_cross_page_tables(all_tables, page_heights)
    renumber_table_ids(all_tables)

    return all_tables


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_words_from_page(page: Any) -> list[_Word]:
    """Extract words from page as _Word objects."""
    words: list[_Word] = []
    try:
        for w in page.get_text("words"):
            if len(w) >= 5:
                words.append(_Word(
                    x0=w[0], y0=w[1], x1=w[2], y1=w[3],
                    text=w[4], block_no=0, line_no=0, word_no=0,
                ))
    except Exception:
        pass
    return words


def _get_text_blocks(page: Any) -> list[dict[str, Any]]:
    """Get text blocks from page."""
    try:
        blocks = page.get_text("dict", flags=11).get("blocks", [])
        result = []
        for b in blocks:
            if "lines" in b:
                text = " ".join(
                    span.get("text", "")
                    for line in b.get("lines", [])
                    for span in line.get("spans", [])
                )
                if text.strip():
                    result.append({
                        "text": text,
                        "bbox": b.get("bbox", (0, 0, 0, 0)),
                    })
        return result
    except Exception:
        return []


def _get_drawings(page: Any) -> list[dict[str, Any]]:
    """Get drawings from page."""
    try:
        return page.get_drawings()
    except Exception:
        return []


def _dict_to_instances(tables: list[dict[str, Any]]) -> list[TableInstance]:
    """Convert legacy dict tables to TableInstance objects."""
    instances = []
    for table in tables or []:
        instance = TableInstance(
            table_id=table.get("table_id", ""),
            page_number=table.get("page", 0),
            bbox=tuple(table.get("bbox", (0, 0, 0, 0))),
            col_count=table.get("col_count", 0),
            row_count=table.get("row_count", 0),
            cells=table.get("cells", []),
            grid=table.get("grid", []),
            row_texts=table.get("row_texts", []),
            physical_col_count=table.get("physical_col_count", table.get("col_count", 0)),
            physical_row_count=table.get("physical_row_count", table.get("row_count", 0)),
            near_page_top=table.get("near_page_top", False),
            near_page_bottom=table.get("near_page_bottom", False),
            is_continuation=table.get("is_continuation", False),
            column_signature=table.get("column_signature", []),
            column_hash=table.get("column_hash", ""),
            detection_method=table.get("detection_method", "pymupdf_builtin"),
        )
        instances.append(instance)
    return instances


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main entry points
    "extract_tables_from_page",
    "extract_tables_from_document",
    # Raw Objects Layer
    "RawTableEvidence",
    "RawSpan",
    "RawCell",
    "RawRow",
    "RawChar",
    "RawDrawing",
    "DrawingType",
    "extract_raw_evidence_from_pymupdf",
    "extract_raw_evidence_from_words",
    # Normalization Layer
    "NormalizedTable",
    "NormalizedCell",
    "NormalizedRow",
    "ColumnCluster",
    "normalize_raw_evidence",
    # Assembly Layer
    "TableInstance",
    "TableHeader",
    "assemble_table_instance",
    # Continuum Engine - Phase 1
    "TableIdentity",
    "IdentityResolution",
    "resolve_table_identity",
    # Continuum Engine - Phase 2
    "GridStabilization",
    "stabilize_logical_grid",
    # Continuum Engine - Phase 3
    "ContinuityResult",
    "establish_cross_page_continuity",
    # Continuum Engine - Phase 4
    "CellState",
    "CellSemantics",
    "analyze_cell_semantics",
    # Continuum Engine - Phase 5
    "NestedStructure",
    "detect_nested_structure",
    # Continuum Engine - Phase 6
    "ConfidenceAssessment",
    "assess_confidence",
    "TOC_LINE_PATTERN",
    "REFERENCE_ROW_PATTERN",
    # Continuum Engine - Pipeline
    "ContinuumResult",
    "run_continuum_engine",
    # Post-processing
    "column_similarity",
    "header_similarity",
    "stitch_cross_page_tables",
    "can_merge_table_fragments_on_same_page",
    "merge_two_table_fragments",
    "merge_same_page_table_fragments",
    "can_merge_tables_with_same_title",
    "merge_same_title_tables",
    "renumber_table_ids",
    "deduplicate_cells",
    "sort_cells_by_position",
    # AST Layer
    "LogicalTableAST",
    "build_logical_ast",
    "ast_to_legacy_dict",
    "merge_ast_with_context",
    # Patterns
    "TABLE_TITLE_PATTERN",
    "TABLE_SECTION_HINT_PATTERN",
    "TOC_HEADING_PATTERN",
]
