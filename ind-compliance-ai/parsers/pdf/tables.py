# Version: v1.2.0
# Optimization Summary:
# - Keep AST extraction non-destructive and surface optional diagnostics only.
# - Attach per-table possible-missing-content diagnostic payload.
# - Mark fully-null rows as structural-empty for enterprise auditability.
# - Add conservative parallel word-clustering candidate path for borderless-table scenarios.
# - Use strict bbox-overlap de-dup to protect already-correct PyMuPDF detections.
# - Add config-driven two-column guard and supplemental candidate strength scoring.

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

================================================================================
修复历史 (Fix History)
================================================================================

v1.0.7 - 第21页续表列映射错误修复
    问题: 第21页作为第20页续表，列映射错误导致数据错位
    原因: 第20页表格不在页面底部(下面有脚注)，续表未被正确识别
    修复:
        1. 放宽续表检测条件 - 即使上一页表格不在底部，当前表格在顶部+水平重叠>=50%也识别为续表
        2. 新增两层续表检测机制，确保与第18页修复逻辑兼容
    位置: extract_tables_from_page() 第182-221行

v1.0.6 - 第20页多识别一行修复
    问题: "文件夹"被拆分成两行
    原因: PDF文本换行导致的错误行拆分
    修复: 新增 _merge_split_rows() 函数，检测并合并被错误拆分的行
    位置: _postprocess_cells() 第451-452行, _merge_split_rows() 第455-592行

v1.0.5 - 第18页逻辑列数错误修复
    问题: 续表错误继承不相关表格的列数
    原因: 续表检测仅依赖页面位置，未验证表格相关性
    修复:
        1. 新增续表验证逻辑 - 检查 near_page_top 和 overlap_ratio >= 0.3
        2. 不满足条件时清除 parent_col_count，避免错误继承
    位置: _process_raw_evidence() 第290-316行

================================================================================
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
from .settings import get_pdf_parser_settings


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
    out_stats: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Extract tables from a single PDF page.

    Uses the unified table pipeline:
    1. Raw Objects Extraction
    2. Normalization
    3. Assembly
    4. Continuum Engine (6 phases)
    5. Logical AST building

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
    raw_candidate_count = 0
    rejected_count = 0
    page_width = page.rect.width if hasattr(page, "rect") else 612.0
    detection_policy = get_pdf_parser_settings().table_detection_policy

    # Strategy 1: PyMuPDF built-in detection (multi-table aware)
    pymupdf_table_count = 0
    try:
        detected_tables = page.find_tables()
        if detected_tables and hasattr(detected_tables, "tables") and detected_tables.tables:
            pymupdf_table_count = len(detected_tables.tables)
    except Exception:
        pymupdf_table_count = 0

    for table_index in range(pymupdf_table_count):
        raw_evidence = extract_raw_evidence_from_pymupdf(
            page=page,
            page_number=page_number,
            page_height=page_height,
            page_width=page_width,
            table_index=table_index,
        )
        if not raw_evidence:
            continue
        raw_candidate_count += 1

        parent_col_count, parent_header, parent_bbox = _resolve_parent_context(
            raw_evidence=raw_evidence,
            page_number=page_number,
            prev_tables=prev_tables,
        )

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
        else:
            rejected_count += 1

    # Strategy 2: Word-clustering supplemental candidate (borderless-table support)
    # Run even when Strategy 1 succeeds, but only append when clearly distinct.
    words = _extract_words_from_page(page)
    if words:
        raw_evidence_fallback = extract_raw_evidence_from_words(
            words=words,
            page_number=page_number,
            page_height=page_height,
            page_width=page_width,
        )

        if raw_evidence_fallback:
            raw_candidate_count += 1
            parent_col_count, parent_header, parent_bbox = _resolve_parent_context(
                raw_evidence=raw_evidence_fallback,
                page_number=page_number,
                prev_tables=prev_tables,
            )
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

            accept_supplemental = _can_accept_supplemental_candidate(
                raw_evidence=raw_evidence_fallback,
                words=words,
                page_width=page_width,
            )
            if table_ast and accept_supplemental and _is_distinct_from_existing_tables(
                table_ast,
                tables,
                overlap_threshold=detection_policy.supplemental_dedup_overlap_threshold,
            ):
                tables.append(table_ast)
                table_counter += 1
            elif table_ast:
                rejected_count += 1
            else:
                rejected_count += 1

    if out_stats is not None:
        out_stats["raw_candidates"] = out_stats.get("raw_candidates", 0) + raw_candidate_count
        out_stats["accepted"] = out_stats.get("accepted", 0) + len(tables)
        out_stats["rejected"] = out_stats.get("rejected", 0) + rejected_count

    return tables, table_counter


def _resolve_parent_context(
    raw_evidence: RawTableEvidence,
    page_number: int,
    prev_tables: list[dict[str, Any]] | None,
) -> tuple[int | None, list[dict[str, Any]] | None, tuple[float, float, float, float] | None]:
    """Resolve possible parent context for continuation validation."""
    parent_col_count = None
    parent_header = None
    parent_bbox = None

    if not prev_tables:
        return parent_col_count, parent_header, parent_bbox

    for prev in reversed(prev_tables):
        prev_page = prev.get("page")
        prev_near_bottom = prev.get("near_page_bottom")
        prev_bbox_value = prev.get("bbox")
        if prev_page != page_number - 1:
            continue

        is_continuation = False
        if prev_near_bottom:
            is_continuation = True
        elif raw_evidence.near_page_top and prev_bbox_value:
            current_bbox = raw_evidence.bbox
            prev_bbox_tuple = tuple(prev_bbox_value)
            x_overlap = max(0, min(current_bbox[2], prev_bbox_tuple[2]) - max(current_bbox[0], prev_bbox_tuple[0]))
            current_width = current_bbox[2] - current_bbox[0]
            prev_width = prev_bbox_tuple[2] - prev_bbox_tuple[0]
            overlap_ratio = x_overlap / min(current_width, prev_width) if min(current_width, prev_width) > 0 else 0
            if overlap_ratio >= 0.5:
                is_continuation = True

        if is_continuation:
            parent_col_count = prev.get("col_count")
            parent_header = prev.get("header")
            if prev_bbox_value:
                parent_bbox = tuple(prev_bbox_value)
            break

    return parent_col_count, parent_header, parent_bbox


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

    # Validate continuation before passing parent_col_count
    # This prevents false positives where unrelated tables inherit wrong column count
    effective_parent_col_count = parent_col_count
    effective_parent_bbox = parent_bbox

    if parent_col_count is not None and parent_bbox is not None and prev_tables:
        # Check if current table is really a continuation
        # by comparing column signatures
        current_bbox = raw_evidence.bbox

        # Check if current table is near page top (continuation indicator)
        near_top = raw_evidence.near_page_top

        # Check horizontal overlap with parent table
        current_width = current_bbox[2] - current_bbox[0]
        parent_width = parent_bbox[2] - parent_bbox[0]

        # Calculate horizontal overlap
        x_overlap = max(0, min(current_bbox[2], parent_bbox[2]) - max(current_bbox[0], parent_bbox[0]))
        overlap_ratio = x_overlap / min(current_width, parent_width) if min(current_width, parent_width) > 0 else 0

        # Check if tables are truly related
        # Conditions: must be near page top AND have significant horizontal overlap
        if not near_top or overlap_ratio < 0.3:
            # Not a valid continuation, clear parent info
            effective_parent_col_count = None
            effective_parent_bbox = None

    # Phase 1: Normalization
    # Pass parent_bbox for column boundary inheritance in continuation tables
    normalized = normalize_raw_evidence(
        raw_evidence=raw_evidence,
        parent_col_count=effective_parent_col_count,
        parent_bbox=effective_parent_bbox,
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

    result = ast.to_dict()
    structural_empty_rows = _collect_structural_empty_rows(result.get("grid", []))
    if structural_empty_rows:
        result["structural_empty_rows"] = structural_empty_rows
    if normalized.missing_content_candidates_count > 0:
        result["diagnostics"] = {
            "possible_missing_table_content": True,
            "candidate_count": normalized.missing_content_candidates_count,
            "candidates": normalized.missing_content_candidates,
            "supplement_writeback_enabled": normalized.supplement_writeback_enabled,
        }
    else:
        result["diagnostics"] = {
            "possible_missing_table_content": False,
            "candidate_count": 0,
            "candidates": [],
            "supplement_writeback_enabled": normalized.supplement_writeback_enabled,
        }

    return result


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


def _collect_structural_empty_rows(grid: list[list[str | None]]) -> list[int]:
    """Return 1-based row indices that are completely empty in the logical grid."""
    empty_rows: list[int] = []
    for row_idx, row in enumerate(grid or [], start=1):
        if not row:
            empty_rows.append(row_idx)
            continue
        if all((cell is None) or (isinstance(cell, str) and not cell.strip()) for cell in row):
            empty_rows.append(row_idx)
    return empty_rows


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

    # Merge incorrectly split rows
    _merge_split_rows(ast)


def _merge_split_rows(ast: LogicalTableAST) -> None:
    """合并被错误拆分的行

    检测并合并由于 PDF 文本换行导致的错误行拆分。

    例如：
    Row 5: ['m1', None, '符合 ICH 要求的模块一内容文件']
    Row 6: [None, '夹', '符合 ICH 要求的模块一内容文件\n夹']

    应该合并为：
    Row 5: ['m1', None, '符合 ICH 要求的模块一内容文件夹']
    Row 6: 删除
    """
    if not ast.grid or len(ast.grid) < 2:
        return

    rows_to_merge: list[tuple[int, int]] = []  # (source_row, target_row)

    for row_idx in range(len(ast.grid) - 1, 0, -1):  # 从后往前遍历
        current_row = ast.grid[row_idx]
        prev_row = ast.grid[row_idx - 1]

        # 检测是否是错误拆分的行
        # 条件: 当前行包含很短的内容（1-3字符），且是常见拆分词
        # 并且：当前行某列内容与前一行某列内容高度相似（前缀关系）

        # 检查是否有短内容拆分词
        has_short_split_word = False
        short_split_word_col = -1
        short_split_word_text = ""
        short_split_words = ['夹', '件', '文', '的', '等', '表', '书', '明']

        for col_idx, cell in enumerate(current_row):
            if cell and len(cell.strip()) <= 2 and cell.strip() in short_split_words:
                has_short_split_word = True
                short_split_word_col = col_idx
                short_split_word_text = cell.strip()
                break

        if not has_short_split_word:
            continue

        # 检查是否有内容续接关系（在其他列）
        has_continuation = False
        continuation_col = -1
        for col_idx in range(min(len(current_row), len(prev_row))):
            if col_idx == short_split_word_col:
                continue  # 跳过短词所在列

            current_cell = current_row[col_idx]
            prev_cell = prev_row[col_idx]

            if not current_cell or not current_cell.strip():
                continue

            current_text = current_cell.strip().replace('\n', ' ')

            if prev_cell:
                prev_text = prev_cell.strip().replace('\n', ' ')

                # 检查续接关系：当前行内容包含前一行内容
                if prev_text and current_text.startswith(prev_text):
                    has_continuation = True
                    continuation_col = col_idx
                    break

        if has_short_split_word and has_continuation:
            rows_to_merge.append((row_idx, row_idx - 1))

    # 执行合并（从后往前，避免索引变化）
    for source_row, target_row in rows_to_merge:
        current_row = ast.grid[source_row]
        prev_row = ast.grid[target_row]

        # 找到短拆分词所在列
        short_split_word_col = -1
        short_split_word_text = ""
        short_split_words = ['夹', '件', '文', '的', '等', '表', '书', '明']

        for col_idx, cell in enumerate(current_row):
            if cell and len(cell.strip()) <= 2 and cell.strip() in short_split_words:
                short_split_word_col = col_idx
                short_split_word_text = cell.strip()
                break

        # 找到有续接关系的列
        continuation_col = -1
        for col_idx in range(min(len(current_row), len(prev_row))):
            if col_idx == short_split_word_col:
                continue

            current_cell = current_row[col_idx]
            prev_cell = prev_row[col_idx]

            if not current_cell or not current_cell.strip():
                continue

            current_text = current_cell.strip().replace('\n', ' ')

            if prev_cell:
                prev_text = prev_cell.strip().replace('\n', ' ')
                if prev_text and current_text.startswith(prev_text):
                    continuation_col = col_idx
                    break

        # 步骤1: 先把短词追加到续接列（而不是短词列对应的 target）
        if short_split_word_text and continuation_col >= 0:
            target_cell = prev_row[continuation_col]
            if target_cell:
                # 检查是否需要追加
                target_text = target_cell.strip().replace('\n', ' ')
                # 只有当短词不在目标文本末尾时才追加
                if not target_text.endswith(short_split_word_text):
                    prev_row[continuation_col] = target_text + short_split_word_text

        # 步骤2: 清空源行
        ast.grid[source_row] = [None] * len(ast.grid[source_row])

        # 更新 cells
        for cell in ast.cells:
            if cell.get("logical_row", cell.get("row", 0)) == source_row:
                cell["text"] = None

    # 移除空行并更新 row_texts
    if rows_to_merge:
        # 过滤空行
        new_grid = [row for row in ast.grid if any(c for c in row if c and c.strip())]
        ast.grid = new_grid
        ast.row_count = len(new_grid)

        # 重建 row_texts
        ast.row_texts = []
        for row in new_grid:
            parts = []
            for cell in row:
                text = cell.replace('\n', ' ').strip() if cell else "null"
                parts.append(text if text else "null")
            ast.row_texts.append(" | ".join(parts))


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


def _bbox_overlap_ratio(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    """Overlap ratio based on min area, robust for containment duplicate checks."""
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter_area = inter_w * inter_h
    area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
    denom = min(area_a, area_b)
    if denom <= 0:
        return 0.0
    return inter_area / denom


def _is_distinct_from_existing_tables(
    candidate: dict[str, Any],
    existing_tables: list[dict[str, Any]],
    overlap_threshold: float = 0.30,
) -> bool:
    """Conservative same-page de-dup gate for supplemental candidates.

    Keep current correct detections stable by requiring low spatial overlap
    before accepting supplemental word-clustering candidates.
    """
    cand_bbox_raw = candidate.get("bbox")
    if not cand_bbox_raw or len(cand_bbox_raw) != 4:
        return False
    cand_bbox = tuple(float(v) for v in cand_bbox_raw)

    for existing in existing_tables:
        ex_bbox_raw = existing.get("bbox")
        if not ex_bbox_raw or len(ex_bbox_raw) != 4:
            continue
        ex_bbox = tuple(float(v) for v in ex_bbox_raw)
        overlap_ratio = _bbox_overlap_ratio(cand_bbox, ex_bbox)
        # Strong overlap indicates duplicate extraction of the same table.
        if overlap_ratio >= overlap_threshold:
            return False
    return True


def _can_accept_supplemental_candidate(
    raw_evidence: RawTableEvidence,
    words: list[_Word],
    page_width: float,
) -> bool:
    """Guard supplemental word-clustering candidates with config-driven rules."""
    policy = get_pdf_parser_settings().table_detection_policy
    if raw_evidence.physical_row_count < policy.min_rows_for_supplemental_candidate:
        return False
    if raw_evidence.physical_col_count < policy.min_cols_for_supplemental_candidate:
        return False

    score = _tabular_strength_score(raw_evidence)
    if _is_two_column_layout(words, page_width):
        if not policy.enable_two_column_guard:
            return score >= 0.55
        return score >= policy.two_column_min_tabular_score

    return score >= 0.45


def _is_two_column_layout(words: list[_Word], page_width: float) -> bool:
    """Heuristic two-column detector for literature-style pages."""
    policy = get_pdf_parser_settings().table_detection_policy
    if not policy.enable_two_column_guard:
        return False
    if not words or page_width <= 0:
        return False
    if len(words) < policy.two_column_min_words:
        return False

    mid = page_width / 2.0
    gutter_half = page_width * max(0.0, policy.two_column_gutter_ratio_min) / 2.0
    gutter_max_half = page_width * max(policy.two_column_gutter_ratio_max, policy.two_column_gutter_ratio_min) / 2.0
    left_count = 0
    right_count = 0
    gutter_count = 0
    for w in words:
        xc = (w.x0 + w.x1) / 2.0
        dist = abs(xc - mid)
        if dist <= gutter_half:
            gutter_count += 1
        elif xc < mid:
            left_count += 1
        else:
            right_count += 1

    side_total = left_count + right_count
    if side_total <= 0:
        return False
    balance = abs(left_count - right_count) / side_total
    gutter_ratio = gutter_count / max(1, len(words))
    return balance <= policy.two_column_balance_tolerance and gutter_ratio <= policy.two_column_gutter_ratio_max


def _tabular_strength_score(raw_evidence: RawTableEvidence) -> float:
    """Estimate whether a word-clustered candidate is likely a real table."""
    rows = raw_evidence.physical_row_count
    cols = raw_evidence.physical_col_count
    raw_data = raw_evidence.raw_data or []
    if rows <= 0 or cols <= 0 or not raw_data:
        return 0.0

    filled = 0
    total = rows * cols
    multi_cell_rows = 0
    consistent_rows = 0
    per_row_filled: list[int] = []
    for row in raw_data:
        row_fill = 0
        for cell in row:
            txt = str(cell or "").strip()
            if txt:
                filled += 1
                row_fill += 1
        per_row_filled.append(row_fill)
        if row_fill >= 2:
            multi_cell_rows += 1
        if row_fill >= max(1, cols // 2):
            consistent_rows += 1

    density = filled / max(1, total)
    multi_row_ratio = multi_cell_rows / max(1, rows)
    consistent_ratio = consistent_rows / max(1, rows)

    col_usage = 0
    for c in range(cols):
        if any(str((row[c] if c < len(row) else "") or "").strip() for row in raw_data):
            col_usage += 1
    col_usage_ratio = col_usage / max(1, cols)

    # Weighted score tuned for conservative supplemental acceptance.
    score = (
        0.30 * min(1.0, rows / 8.0)
        + 0.20 * min(1.0, cols / 4.0)
        + 0.20 * max(0.0, min(1.0, density))
        + 0.15 * multi_row_ratio
        + 0.10 * consistent_ratio
        + 0.05 * col_usage_ratio
    )
    return max(0.0, min(1.0, score))


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
