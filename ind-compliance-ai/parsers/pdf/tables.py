# Version: v1.3.4
# Optimization Summary:
# - Keep AST extraction non-destructive and surface optional diagnostics only.
# - Attach per-table possible-missing-content diagnostic payload.
# - Mark fully-null rows as structural-empty for enterprise auditability.
# - Add conservative parallel word-clustering candidate path for borderless-table scenarios.
# - Use strict bbox-overlap de-dup to protect already-correct PyMuPDF detections.
# - Add config-driven two-column guard and supplemental candidate strength scoring.
# - Generalize running-header recognition so narrow top-edge template headers with
#   version/page tokens remain non-blocking after visual text-line reconstruction.
# - Bring current-table local title and preceding-text evidence into early
#   continuation assessment so contradiction signals block false continuation.
# - Apply the same continuation assessment flow to the supplemental fallback path.
# - Persist local context signal into the table AST so later postprocess stages can
#   reuse the same semantic decision basis instead of re-inferring from thresholds.
# - Recognize compact running headers whose version token is attached directly to
#   adjacent CJK/OCR text, so reconstructed page headers stay non-blocking.
#
# v1.3.0 (2026-03-12):
# - Refactor continuation detection to use "evidence collection → evidence assessment" framework.
# - Step1: _resolve_parent_context() now collects evidence bundles instead of computing confidence.
# - Step2: New _assess_continuation_candidates() integrates evidence and computes assessment.
# - Add DecisionType for explicit decision type tracking.
# - Add multi-dimensional evidence scoring (layout, structure, context).
# - Improve generalization for different PDF layouts and parsing quality conditions.
#
# v1.2.1 (2026-03-12):
# - Add bbox correction for continuation tables when PyMuPDF detects shifted boundaries.
# - When a continuation table is detected, inherit parent table's bbox boundaries.
# - This ensures column mapping is calculated correctly based on aligned boundaries.
# - Add bbox_correction metadata for audit trail.

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

v1.1.0 - 增加领先段落文本识别并记录为 table 级 metadata，供 cross-page stitching guard 参考，防止段落隔断误判续表。
    修复: 让 stitch_cross_page_tables() 在没有明确“continued”提示时，读取 preceding_text_block 并据此阻断续表连线。
    位置: extract_tables_from_page() 第471-520行、stitch_cross_page_tables() 第60-200行
v1.1.1 - 在 guard 拦截续表后，用 header 候选重建列、再据该列数处理 row_texts，避免原始 header 被错误继承导致 null 列。
    说明：从页上方 words 提取 header 候选，group 并按列重建，然后在 _reset_continuation_flags() 重写 header/col_count 并记录 rebuild 标记。
    位置: _build_context() 第486-544行、_reset_continuation_flags() 第288-305行
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

from .types import (
    LayoutEvidence,
    StructureEvidence,
    ContextEvidence,
    EvidenceBundle,
    ContinuationAssessment,
    StructureValidation,
    DecisionType,
)
from .layout import is_two_column_layout

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
    analyze_table_opening_structure,
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
    split_internal_table_segments,
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

from .shared import _Word, _bbox_to_list, _bbox_union
from .settings import CrossPageStitchingThresholds, get_pdf_parser_settings
from .table_vector_ocr import extract_vector_ocr_table_candidates


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
TOC_TITLE_PATTERN = re.compile(r"\b(?:table of contents|contents|toc)\b|目录", re.IGNORECASE)
TOC_TITLE_ROW_PATTERN = re.compile(r"^\s*(?:table of contents|contents|toc|目录)\s*$", re.IGNORECASE)
PAGE_LOCATOR_PATTERN = re.compile(r"^(?:\d{1,4}|[ivxlcdm]{1,12})$", re.IGNORECASE)
OUTLINE_INDEX_TOKEN_PATTERN = re.compile(r"^(\d+(?:\.\d+)+)\b")
SINGLE_NUMERIC_OUTLINE_PREFIX_PATTERN = re.compile(r"^\s*(\d+)\.(?:\s+|$)")
INLINE_TOC_PAGE_LOCATOR_PATTERN = re.compile(
    r"^(?P<body>.+?)(?P<leader>(?:[.\u2026·•_ ]*[.\u2026·•_]{3,}[.\u2026·•_ ]*| {2,}))(?P<locator>\d{1,4}|[ivxlcdm]{1,12})$",
    re.IGNORECASE,
)
APPENDIX_OUTLINE_PREFIX_PATTERN = re.compile(
    r"^\s*(APPENDIX\s+[A-Z0-9]+)(?:\s*[:.\-])?(?:\s+|$)",
    re.IGNORECASE,
)
ROMAN_OUTLINE_PREFIX_PATTERN = re.compile(r"^\s*([ivxlcdm]+)\.(?:\s+|$)", re.IGNORECASE)
ALPHA_OUTLINE_PREFIX_PATTERN = re.compile(r"^\s*([A-Z])\.(?:\s+|$)")
INVENTORY_PATH_PATTERN = re.compile(r"(?:\.pdf\b|\\|/|_[0-9A-Za-z].*\.pdf\b)", re.IGNORECASE)

CONTINUATION_HINT_KEYWORDS = (
    "continued",
    "continuation",
    "\u7eed\u8868",
    "\u7eed\u9875",
)
LOCAL_TABLE_TITLE_PATTERN = re.compile(
    "^\\s*(?:(?:\u9644)?\u8868\\s*[0-9A-Za-z\u4e00-\u4e5d\u5341\u96f6\u3007.\\-]+|table\\s*[0-9A-Za-z.\\-]+)\\s*(?:[.:：、\\-]|\\s+)",
    re.IGNORECASE,
)


# ============================================================================
# Main Entry Points
# ============================================================================

def extract_tables_from_page(
    page: Any,
    page_number: int,
    page_height: float,
    text_blocks: list[dict[str, Any]] | None = None,
    page_drawings: list[dict[str, Any]] | None = None,
    layout_profile: dict[str, Any] | None = None,
    prev_tables: list[dict[str, Any]] | None = None,
    table_counter: int = 0,
    out_stats: dict[str, int] | None = None,
    out_toc_blocks: list[dict[str, Any]] | None = None,
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
    toc_outline_count = 0
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

    page_words = _extract_words_from_page(page)
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

        # Step1: 证据收集
        current_context = _build_current_table_context(
            raw_evidence=raw_evidence,
            text_blocks=text_blocks,
        )
        candidates = _resolve_parent_context(
            raw_evidence=raw_evidence,
            page_number=page_number,
            prev_tables=prev_tables,
            current_context=current_context,
        )
        
        # Step2: 证据评估
        assessment = _assess_continuation_candidates_v2(
            candidates=candidates,
            raw_evidence=raw_evidence,
        )
        
        # 从评估结果中获取父表信息
        parent_col_count = None
        parent_header = None
        parent_bbox = None
        parent_column_boundaries = None
        
        if assessment.is_continuation:
            parent_col_count = assessment.selected_parent_col_count
            parent_header = assessment.selected_parent_header
            parent_bbox = assessment.selected_parent_bbox
            parent_column_boundaries = assessment.selected_parent_column_boundaries

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
            parent_column_boundaries=parent_column_boundaries,
            assessment=assessment,
            current_context=current_context,
            words=page_words,
        )
        
        # 将评估结果附加到表格 AST
        if table_ast and assessment.is_continuation:
            table_ast["continuation_assessment"] = assessment.to_dict()

        if table_ast:
            table_counter += 1
            if table_ast.get("semantic_role") == "toc_outline":
                toc_outline_count += 1
                if out_toc_blocks is not None:
                    out_toc_blocks.append(_build_toc_block(table_ast))
            else:
                tables.append(table_ast)
        else:
            rejected_count += 1

    # Strategy 2: Vector-outline OCR fallback for titled tables with no text layer.
    occupied_bboxes = _collect_item_bboxes(tables + (out_toc_blocks or []))
    title_blocks = [
        block
        for block in (text_blocks or [])
        if _looks_like_table_title(str(block.get("text", "")).strip())
    ]
    vector_ocr_candidates = extract_vector_ocr_table_candidates(
        page=page,
        page_number=page_number,
        page_height=page_height,
        page_width=page_width,
        title_blocks=title_blocks,
        page_drawings=page_drawings or [],
        page_words=page_words,
        occupied_bboxes=occupied_bboxes,
    )
    for raw_evidence_ocr in vector_ocr_candidates:
        raw_candidate_count += 1
        current_context = _build_current_table_context(
            raw_evidence=raw_evidence_ocr,
            text_blocks=text_blocks,
        )
        candidates = _resolve_parent_context(
            raw_evidence=raw_evidence_ocr,
            page_number=page_number,
            prev_tables=prev_tables,
            current_context=current_context,
        )
        assessment = _assess_continuation_candidates_v2(
            candidates=candidates,
            raw_evidence=raw_evidence_ocr,
        )
        parent_col_count = None
        parent_header = None
        parent_bbox = None
        parent_column_boundaries = None
        if assessment.is_continuation:
            parent_col_count = assessment.selected_parent_col_count
            parent_header = assessment.selected_parent_header
            parent_bbox = assessment.selected_parent_bbox
            parent_column_boundaries = assessment.selected_parent_column_boundaries
        table_ast = _process_raw_evidence(
            raw_evidence=raw_evidence_ocr,
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
            parent_column_boundaries=parent_column_boundaries,
            assessment=assessment,
            current_context=current_context,
            words=page_words,
        )
        if table_ast and table_ast.get("semantic_role") == "toc_outline":
            rejected_count += 1
            continue
        if table_ast and _is_distinct_from_existing_tables(
            table_ast,
            tables,
            overlap_threshold=detection_policy.supplemental_dedup_overlap_threshold,
        ):
            table_counter += 1
            tables.append(table_ast)
            occupied_bboxes.append(tuple(float(v) for v in table_ast.get("bbox", (0.0, 0.0, 0.0, 0.0))))
            continue
        rejected_count += 1

    # Strategy 3: Word-clustering supplemental candidate (borderless-table support)
    # Run on residual page words after excluding already accepted regions, then keep
    # iterating over remaining structured groups under the same unified path.
    occupied_bboxes = _collect_item_bboxes(tables + (out_toc_blocks or []))
    supplemental_words = _exclude_words_in_occupied_regions(page_words, occupied_bboxes)
    while supplemental_words:
        raw_evidence_fallback = extract_raw_evidence_from_words(
            words=supplemental_words,
            page_number=page_number,
            page_height=page_height,
            page_width=page_width,
        )
        if not raw_evidence_fallback:
            break

        raw_candidate_count += 1
        current_context = _build_current_table_context(
            raw_evidence=raw_evidence_fallback,
            text_blocks=text_blocks,
        )
        raw_evidence_fallback = _expand_candidate_with_nearby_title_words(
            raw_evidence_fallback,
            page_words=page_words,
            current_context=current_context,
        )
        current_context = _build_current_table_context(
            raw_evidence=raw_evidence_fallback,
            text_blocks=text_blocks,
        )
        split_candidates = _split_two_column_composite_candidate(
            raw_evidence_fallback,
            current_context=current_context,
            text_blocks=text_blocks,
            layout_profile=layout_profile,
            page_width=page_width,
            page_words=page_words,
        )
        if split_candidates:
            consumed_split_words: list[Any] = []
            split_accepted = False
            existing_toc_blocks = out_toc_blocks or []
            for split_evidence in split_candidates:
                split_context = _build_current_table_context(
                    raw_evidence=split_evidence,
                    text_blocks=text_blocks,
                )
                split_evidence = _expand_candidate_with_nearby_title_words(
                    split_evidence,
                    page_words=page_words,
                    current_context=split_context,
                )
                split_context = _build_current_table_context(
                    raw_evidence=split_evidence,
                    text_blocks=text_blocks,
                )
                split_parent_candidates = _resolve_parent_context(
                    raw_evidence=split_evidence,
                    page_number=page_number,
                    prev_tables=prev_tables,
                    current_context=split_context,
                )
                split_assessment = _assess_continuation_candidates_v2(
                    candidates=split_parent_candidates,
                    raw_evidence=split_evidence,
                )
                parent_col_count = None
                parent_header = None
                parent_bbox = None
                parent_column_boundaries = None
                if split_assessment.is_continuation:
                    parent_col_count = split_assessment.selected_parent_col_count
                    parent_header = split_assessment.selected_parent_header
                    parent_bbox = split_assessment.selected_parent_bbox
                    parent_column_boundaries = split_assessment.selected_parent_column_boundaries
                split_table_ast = _process_raw_evidence(
                    raw_evidence=split_evidence,
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
                    parent_column_boundaries=parent_column_boundaries,
                    assessment=split_assessment,
                    current_context=split_context,
                    words=page_words,
                )
                accept_split = _can_accept_supplemental_candidate(
                    raw_evidence=split_evidence,
                    words=page_words,
                    page_width=page_width,
                    current_context=split_context,
                    page_drawings=page_drawings,
                    layout_profile=layout_profile,
                    assessment=split_assessment,
                )
                if split_table_ast and split_table_ast.get("semantic_role") == "toc_outline":
                    if _is_distinct_from_existing_tables(
                        split_table_ast,
                        tables + existing_toc_blocks,
                        overlap_threshold=detection_policy.supplemental_dedup_overlap_threshold,
                    ):
                        table_counter += 1
                        toc_outline_count += 1
                        split_accepted = True
                        consumed_split_words.extend(split_evidence.words)
                        if out_toc_blocks is not None:
                            out_toc_blocks.append(_build_toc_block(split_table_ast))
                    else:
                        rejected_count += 1
                    continue
                if split_table_ast and accept_split and _is_distinct_from_existing_tables(
                    split_table_ast,
                    tables,
                    overlap_threshold=detection_policy.supplemental_dedup_overlap_threshold,
                ):
                    table_counter += 1
                    tables.append(split_table_ast)
                    split_accepted = True
                    consumed_split_words.extend(split_evidence.words)
                    continue
                if split_table_ast:
                    rejected_count += 1

            if split_accepted and consumed_split_words:
                previous_word_count = len(supplemental_words)
                supplemental_words = _exclude_words_by_identity(
                    supplemental_words,
                    consumed_split_words,
                )
                if len(supplemental_words) == previous_word_count:
                    supplemental_words = _exclude_words_in_occupied_regions(
                        supplemental_words,
                        [candidate.bbox for candidate in split_candidates],
                    )
                    if len(supplemental_words) == previous_word_count:
                        break
                continue

        candidates = _resolve_parent_context(
            raw_evidence=raw_evidence_fallback,
            page_number=page_number,
            prev_tables=prev_tables,
            current_context=current_context,
        )
        assessment = _assess_continuation_candidates_v2(
            candidates=candidates,
            raw_evidence=raw_evidence_fallback,
        )
        parent_col_count = None
        parent_header = None
        parent_bbox = None
        parent_column_boundaries = None
        if assessment.is_continuation:
            parent_col_count = assessment.selected_parent_col_count
            parent_header = assessment.selected_parent_header
            parent_bbox = assessment.selected_parent_bbox
            parent_column_boundaries = assessment.selected_parent_column_boundaries
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
            parent_column_boundaries=parent_column_boundaries,
            assessment=assessment,
            current_context=current_context,
            words=page_words,
        )

        accept_supplemental = _can_accept_supplemental_candidate(
            raw_evidence=raw_evidence_fallback,
            words=page_words,
            page_width=page_width,
            current_context=current_context,
            page_drawings=page_drawings,
            layout_profile=layout_profile,
            assessment=assessment,
        )
        rejected_as_narrative_false_positive = _looks_like_two_column_narrative_false_positive(
            raw_evidence=raw_evidence_fallback,
            words=page_words,
            page_width=page_width,
            current_context=current_context,
            page_drawings=page_drawings,
            layout_profile=layout_profile,
            assessment=assessment,
        )
        existing_toc_blocks = out_toc_blocks or []
        words_to_consume = list(raw_evidence_fallback.words or [])
        if rejected_as_narrative_false_positive and not accept_supplemental:
            words_to_consume = _supplemental_rejection_peel_words(raw_evidence_fallback)
        if table_ast and table_ast.get("semantic_role") == "toc_outline":
            if _is_distinct_from_existing_tables(
                table_ast,
                tables + existing_toc_blocks,
                overlap_threshold=detection_policy.supplemental_dedup_overlap_threshold,
            ):
                table_counter += 1
                toc_outline_count += 1
                if out_toc_blocks is not None:
                    out_toc_blocks.append(_build_toc_block(table_ast))
            else:
                rejected_count += 1
        elif table_ast and accept_supplemental and _is_distinct_from_existing_tables(
            table_ast,
            tables,
            overlap_threshold=detection_policy.supplemental_dedup_overlap_threshold,
        ):
            table_counter += 1
            tables.append(table_ast)
        elif table_ast:
            rejected_count += 1
        else:
            rejected_count += 1

        previous_word_count = len(supplemental_words)
        supplemental_words = _exclude_words_by_identity(
            supplemental_words,
            words_to_consume,
        )
        if len(supplemental_words) == previous_word_count:
            candidate_bboxes = [raw_evidence_fallback.bbox]
            if rejected_as_narrative_false_positive and not accept_supplemental and raw_evidence_fallback.rows:
                peel_rows = list(raw_evidence_fallback.rows[: max(1, min(len(raw_evidence_fallback.rows), 2))])
                candidate_bboxes = [row.bbox for row in peel_rows if row.bbox]
            supplemental_words = _exclude_words_in_occupied_regions(
                supplemental_words,
                candidate_bboxes,
            )
            if len(supplemental_words) == previous_word_count:
                break

    if out_toc_blocks is not None and len(out_toc_blocks) > 1:
        toc_outline_count = len(_consolidate_same_page_toc_blocks(out_toc_blocks))
    elif out_toc_blocks is not None:
        toc_outline_count = len(out_toc_blocks)

    if out_stats is not None:
        out_stats["raw_candidates"] = out_stats.get("raw_candidates", 0) + raw_candidate_count
        out_stats["accepted"] = out_stats.get("accepted", 0) + len(tables)
        out_stats["toc_outlines"] = out_stats.get("toc_outlines", 0) + toc_outline_count
        out_stats["rejected"] = out_stats.get("rejected", 0) + rejected_count

    return tables, table_counter


def _build_current_table_context(
    raw_evidence: RawTableEvidence,
    text_blocks: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Collect current-table local context before continuation assessment."""
    opening_structure = analyze_table_opening_structure(
        raw_evidence.raw_data or [],
        raw_evidence.physical_col_count,
    )
    internal_title_text = str(opening_structure.title_text or "").strip()
    header_cells = list(opening_structure.header_cells)
    header_texts = list(opening_structure.header_texts)

    if not text_blocks:
        return {
            "title_block": None,
            "preceding_text_block": None,
            "section_hint": None,
            "title_text": internal_title_text,
            "preceding_text": "",
            "local_signal": "new_table_title" if internal_title_text else "none",
            "header_cells": header_cells,
            "header_texts": header_texts,
        }

    bbox = raw_evidence.bbox
    cfg = get_pdf_parser_settings().cross_page_stitching
    title_block = _find_title_block(text_blocks, bbox)
    section_hint = _find_section_hint_block(text_blocks, bbox)
    preceding_block = _find_preceding_text_block(text_blocks, bbox, cfg)
    if (
        title_block is None
        and raw_evidence.source == "word_clustering"
        and preceding_block is not None
        and raw_evidence.physical_row_count >= 4
        and raw_evidence.physical_col_count >= 3
        and _looks_like_structural_heading_text(str(preceding_block.get("text", "")).strip())
    ):
        title_block = preceding_block

    title_text = internal_title_text
    if not title_text and title_block:
        title_text = str(title_block.get("text", "")).strip()
    elif not title_text and preceding_block and _looks_like_table_title(str(preceding_block.get("text", "")).strip()):
        title_text = str(preceding_block.get("text", "")).strip()

    preceding_text = str(preceding_block.get("text", "")).strip() if preceding_block else ""
    local_signal = _classify_current_table_signal(
        title_block,
        preceding_block,
        section_hint,
        page_height=raw_evidence.page_height,
        page_width=raw_evidence.page_width,
        internal_title_text=internal_title_text,
    )

    return {
        "title_block": title_block,
        "preceding_text_block": preceding_block,
        "section_hint": section_hint,
        "title_text": title_text,
        "preceding_text": preceding_text,
        "local_signal": local_signal,
        "header_cells": header_cells,
        "header_texts": header_texts,
    }


def _find_title_block(
    text_blocks: list[dict[str, Any]] | None,
    bbox: tuple[float, float, float, float],
) -> dict[str, Any] | None:
    """Return the closest title block immediately above the table."""
    if not text_blocks:
        return None

    candidates: list[tuple[float, dict[str, Any]]] = []
    for block in text_blocks:
        text = str(block.get("text", "")).strip()
        if not text or not _looks_like_table_title(text):
            continue
        block_bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
        gap = bbox[1] - block_bbox[3]
        if gap < -8 or gap > 160:
            continue
        overlap = _horizontal_overlap_ratio(bbox, block_bbox)
        if overlap < 0.12:
            continue
        candidates.append((max(0.0, gap), block))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    title_block = candidates[0][1]
    return _merge_split_table_title_block(title_block, text_blocks, bbox)


def _merge_split_table_title_block(
    title_block: dict[str, Any],
    text_blocks: list[dict[str, Any]] | None,
    bbox: tuple[float, float, float, float],
) -> dict[str, Any]:
    if not text_blocks:
        return title_block

    title_text = str(title_block.get("text", "")).strip()
    if not re.fullmatch(r"(?i)table\s*[0-9A-Za-z.\-]+", title_text):
        return title_block

    title_bbox = tuple(title_block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(title_bbox) != 4:
        return title_block

    continuation_candidates: list[tuple[float, dict[str, Any]]] = []
    for block in text_blocks:
        if block is title_block:
            continue
        text = str(block.get("text", "")).strip()
        if not text or _looks_like_table_title(text):
            continue
        block_bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        if len(block_bbox) != 4:
            continue
        if block_bbox[1] < title_bbox[3] - 2:
            continue
        if block_bbox[3] > bbox[1] + 8:
            continue
        gap = block_bbox[1] - title_bbox[3]
        if gap < -2 or gap > 24:
            continue
        overlap = _horizontal_overlap_ratio(bbox, block_bbox)
        if overlap < 0.1:
            continue
        if len(text.split()) > 18:
            continue
        continuation_candidates.append((max(0.0, gap), block))

    if not continuation_candidates:
        return title_block

    continuation_candidates.sort(key=lambda item: item[0])
    continuation_block = continuation_candidates[0][1]
    merged_bbox = _bbox_union([list(title_bbox), list(continuation_block.get("bbox", []))])
    return {
        **title_block,
        "text": f"{title_text} {str(continuation_block.get('text', '')).strip()}".strip(),
        "bbox": merged_bbox,
        "source": "split_title_merge",
    }


def _promote_preceding_heading_to_structural_title(
    instance: TableInstance,
    preceding_block: dict[str, Any] | None,
    header_candidates: list[str],
) -> dict[str, Any] | None:
    """Promote a compact heading above a strong borderless grid as title support."""
    if instance.detection_method != "word_clustering":
        return None
    if preceding_block is None:
        return None
    if instance.row_count < 4 or instance.col_count < 3:
        return None
    if len(header_candidates) < min(instance.col_count, 3):
        return None

    text = str(preceding_block.get("text", "")).strip()
    if not _looks_like_structural_heading_text(text):
        return None
    return preceding_block


def _looks_like_structural_heading_text(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False

    compact_length = len(_compact_context_text(cleaned))
    token_count = len(cleaned.split())
    if compact_length < 3 or compact_length > 60:
        return False
    if token_count > 8:
        return False
    if any(mark in cleaned for mark in (";", "!", "?", "。", "；", "！", "？")):
        return False
    return True


def _find_section_hint_block(
    text_blocks: list[dict[str, Any]] | None,
    bbox: tuple[float, float, float, float],
) -> dict[str, Any] | None:
    """Return the nearest section/glossary heading relevant to the table."""
    if not text_blocks:
        return None

    for block in text_blocks:
        text = str(block.get("text", "")).strip()
        if not text or not TABLE_SECTION_HINT_PATTERN.search(text):
            continue
        block_bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
        gap = bbox[1] - block_bbox[3]
        if 0 <= gap <= 180:
            overlap = _horizontal_overlap_ratio(bbox, block_bbox)
            if overlap >= 0.1:
                return block
    return None


def _classify_current_table_signal(
    title_block: dict[str, Any] | None,
    preceding_block: dict[str, Any] | None,
    section_hint: dict[str, Any] | None,
    page_height: float,
    page_width: float,
    internal_title_text: str = "",
) -> str:
    """Classify the strongest local signal around the current table."""
    title_text = internal_title_text or (str(title_block.get("text", "")).strip() if title_block else "")
    preceding_text = str(preceding_block.get("text", "")).strip() if preceding_block else ""
    section_text = str(section_hint.get("text", "")).strip() if section_hint else ""

    if title_text:
        if _contains_continuation_hint(title_text):
            return "continuation_title"
        return "new_table_title"

    if preceding_text:
        block_bbox = tuple(preceding_block.get("bbox", (0.0, 0.0, 0.0, 0.0))) if preceding_block else (0.0, 0.0, 0.0, 0.0)
        if _is_running_header_block(preceding_text, block_bbox, page_height, page_width):
            return "running_header"
        if _contains_continuation_hint(preceding_text):
            return "continuation_title"
        if _looks_like_table_title(preceding_text):
            return "new_table_title"
        if section_text and _compact_context_text(section_text) == _compact_context_text(preceding_text):
            return "section_heading"
        return "narrative_barrier"

    if section_text:
        return "section_heading"

    return "none"


def _contains_continuation_hint(text: str) -> bool:
    normalized = str(text or "").lower()
    return any(keyword in normalized for keyword in CONTINUATION_HINT_KEYWORDS)


def _is_running_header_block(
    text: str,
    block_bbox: tuple[float, float, float, float],
    page_height: float,
    page_width: float,
) -> bool:
    """Detect page-template headers that should not act as narrative barriers."""
    if not text:
        return False
    cleaned = str(text).strip()
    top_ratio = (block_bbox[1] / page_height) if page_height > 0 else 1.0
    width_ratio = ((block_bbox[2] - block_bbox[0]) / page_width) if page_width > 0 else 0.0
    left_gap_ratio = (block_bbox[0] / page_width) if page_width > 0 else 1.0
    right_gap_ratio = ((page_width - block_bbox[2]) / page_width) if page_width > 0 else 1.0
    token_count = len(cleaned.split())
    has_sentence_punct = any(mark in cleaned for mark in ("。", "；", ";", "!", "?", "？"))
    has_period_sentence = ". " in cleaned or cleaned.endswith(".")
    starts_with_caption_label = bool(re.match(r"^\s*(?:figure|fig\.?|table)\b", cleaned, re.IGNORECASE))
    if starts_with_caption_label or ((has_sentence_punct or has_period_sentence) and token_count >= 6):
        return False
    has_version_token = bool(re.search(r"(?<![A-Za-z0-9])v?\d+(?:\.\d+)+(?![A-Za-z0-9])", cleaned, re.IGNORECASE))
    has_page_token = bool(re.search(r"\bpage\b|\d+\s*/\s*\d+", cleaned, re.IGNORECASE))
    compact_length = len(_compact_context_text(cleaned))
    edge_aligned = left_gap_ratio <= 0.18 or right_gap_ratio <= 0.18
    if top_ratio > 0.12:
        return False
    wide_template_header = width_ratio >= 0.45 and (has_version_token or has_page_token or not has_sentence_punct)
    narrow_edge_template_header = edge_aligned and compact_length <= 40 and (has_version_token or has_page_token)
    return wide_template_header or narrow_edge_template_header


def _looks_like_table_title(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    if LOCAL_TABLE_TITLE_PATTERN.search(candidate):
        return True
    return bool(re.fullmatch(r"(?i)table\s*[0-9A-Za-z.\-]+", candidate))


def _compact_context_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text or "").lower())


def _normalize_title_identity(text: str) -> str:
    normalized = str(text or "").lower()
    for keyword in CONTINUATION_HINT_KEYWORDS:
        normalized = normalized.replace(keyword, "")
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = normalized.replace("(cont.)", "")
    normalized = normalized.replace("(continued)", "")
    normalized = normalized.replace("(续)", "")
    return _compact_context_text(normalized)


def _looks_like_numeric_toc_content_stub(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    if re.search(r"[A-Za-z\u4e00-\u9fff]", candidate):
        return False
    numeric_tokens = re.findall(r"\d+(?:[./-]\d+)?", candidate)
    if len(numeric_tokens) < 2:
        return False
    normalized = re.sub(r"[\s,.;:()\-–—_/|]+", "", candidate)
    return bool(normalized) and normalized == "".join(numeric_tokens)


def _resolve_parent_context(
    raw_evidence: RawTableEvidence,
    page_number: int,
    prev_tables: list[dict[str, Any]] | None,
    current_context: dict[str, Any] | None = None,
) -> list[tuple[dict[str, Any], EvidenceBundle]]:
    """解析父表格上下文 - 证据收集阶段 (Step1)
    
    职责：
    - 遍历 prev_tables，识别候选父表
    - 收集多维度证据（页序、版式、结构）
    - 构建 evidence_bundle
    
    注意：不计算最终置信度，只收集证据
    
    Args:
        raw_evidence: 当前表格的原始证据
        page_number: 当前页码
        prev_tables: 之前页面的表格列表
        
    Returns:
        [(parent_info, evidence_bundle), ...] 候选列表
    """
    if not prev_tables:
        return []
    
    candidates: list[tuple[dict[str, Any], EvidenceBundle]] = []
    
    for prev in reversed(prev_tables):
        prev_page = prev.get("page")
        
        # 基本过滤：页码必须相邻或接近
        page_gap = page_number - prev_page
        if page_gap < 1 or page_gap > 2:  # 允许跨1-2页
            continue
        
        # 构建证据集合
        evidence = EvidenceBundle()
        
        # 填充父表基本信息
        evidence.parent_table_id = prev.get("table_id", "")
        evidence.parent_page = prev_page
        evidence.parent_bbox = tuple(prev.get("bbox", (0, 0, 0, 0)))
        
        # 填充当前表格基本信息
        evidence.current_bbox = raw_evidence.bbox
        evidence.current_page = page_number
        evidence.current_page_height = raw_evidence.page_height
        
        # 收集版式证据
        evidence.layout = _collect_layout_evidence(
            prev, raw_evidence, page_gap
        )
        
        # 收集结构证据
        evidence.structure = _collect_structure_evidence(
            prev,
            raw_evidence,
            current_context,
        )
        
        # 收集上下文证据
        evidence.context = _collect_context_evidence(
            prev, raw_evidence, current_context
        )
        
        # 基本有效性检查：至少页序连续
        if evidence.layout.page_sequential:
            candidates.append((prev, evidence))
    
    return candidates


def _collect_layout_evidence(
    prev_table: dict[str, Any],
    raw_evidence: RawTableEvidence,
    page_gap: int,
) -> LayoutEvidence:
    """收集版式证据"""
    evidence = LayoutEvidence()
    
    prev_bbox = tuple(prev_table.get("bbox", (0, 0, 0, 0)))
    curr_bbox = raw_evidence.bbox
    page_height = raw_evidence.page_height
    
    # 页序连续性
    evidence.page_sequential = (page_gap == 1)
    evidence.page_gap = page_gap
    
    # 版式位置关系
    evidence.prev_near_bottom = prev_table.get("near_page_bottom", False)
    evidence.curr_near_top = raw_evidence.near_page_top
    evidence.prev_bottom_ratio = prev_bbox[3] / page_height if page_height > 0 else 0
    evidence.curr_top_ratio = curr_bbox[1] / page_height if page_height > 0 else 0
    
    # 水平覆盖情况
    prev_width = prev_bbox[2] - prev_bbox[0]
    curr_width = curr_bbox[2] - curr_bbox[0]
    
    if prev_width > 0 and curr_width > 0:
        # 水平重叠
        x_overlap = max(0, min(prev_bbox[2], curr_bbox[2]) - max(prev_bbox[0], curr_bbox[0]))
        evidence.horizontal_overlap_ratio = x_overlap / min(prev_width, curr_width)
        
        # 边界差异
        evidence.left_boundary_diff = curr_bbox[0] - prev_bbox[0]
        evidence.right_boundary_diff = prev_bbox[2] - curr_bbox[2]
        
        # 宽度比例
        evidence.width_ratio = curr_width / prev_width
    
    return evidence


def _collect_structure_evidence(
    prev_table: dict[str, Any],
    raw_evidence: RawTableEvidence,
    current_context: dict[str, Any] | None = None,
) -> StructureEvidence:
    """收集结构证据"""
    evidence = StructureEvidence()
    
    # 列结构
    evidence.parent_col_count = prev_table.get("col_count", 0)
    evidence.current_physical_col_count = raw_evidence.physical_col_count
    evidence.current_max_content_cols = raw_evidence.max_content_cols_per_row
    
    # 列分布模式
    evidence.parent_column_signature = prev_table.get("column_signature", [])
    evidence.parent_column_boundaries = _extract_column_boundaries(prev_table)
    
    # 行模式
    evidence.parent_row_count = prev_table.get("row_count", 0)
    evidence.current_row_count = raw_evidence.physical_row_count
    
    # 表头信息
    evidence.parent_header = prev_table.get("header", [])
    evidence.parent_header_texts = [
        h.get("text", "") for h in evidence.parent_header if h.get("text")
    ]
    current_header = (current_context or {}).get("header_cells", []) if current_context else []
    evidence.current_header_texts = [
        str(item.get("text", "")).strip()
        for item in current_header
        if str(item.get("text", "")).strip()
    ]
    evidence.current_has_header_row = bool(evidence.current_header_texts)
    if evidence.parent_header and current_header:
        evidence.header_similarity_score = header_similarity(evidence.parent_header, current_header)
    
    return evidence


def _collect_context_evidence(
    prev_table: dict[str, Any],
    raw_evidence: RawTableEvidence,
    current_context: dict[str, Any] | None = None,
) -> ContextEvidence:
    """收集上下文证据"""
    evidence = ContextEvidence()
    
    # 标题信息
    evidence.parent_title = str(prev_table.get("title", "")).strip()
    evidence.title_available = bool(evidence.parent_title)
    
    # 章节信息
    evidence.parent_section_hint = str(prev_table.get("section_hint", "")).strip()
    
    # 续表提示（从 preceding_text_block 中检测）
    preceding_block = prev_table.get("preceding_text_block")
    if preceding_block:
        text = str(preceding_block.get("text", "")).lower()
        hint_keywords = ["续表", "续页", "continued", "continuation"]
        evidence.has_continuation_hint = _contains_continuation_hint(text)
        if evidence.has_continuation_hint:
            evidence.continuation_hint_text = text[:100]

    context_data = current_context or {}
    evidence.current_title = str(context_data.get("title_text", "")).strip()
    evidence.current_preceding_text = str(context_data.get("preceding_text", "")).strip()
    evidence.current_local_signal = str(context_data.get("local_signal", "none")).strip() or "none"
    evidence.current_has_new_table_title = evidence.current_local_signal == "new_table_title"
    evidence.current_has_continuation_hint = evidence.current_local_signal == "continuation_title"
    evidence.current_has_narrative_barrier = evidence.current_local_signal == "narrative_barrier"
    evidence.current_has_section_heading = evidence.current_local_signal == "section_heading"
    if evidence.current_has_continuation_hint:
        hint_text = evidence.current_title or evidence.current_preceding_text
        evidence.current_continuation_hint_text = hint_text[:100]

    current_title_key = _normalize_title_identity(evidence.current_title)
    parent_title_key = _normalize_title_identity(evidence.parent_title)
    if current_title_key and parent_title_key:
        evidence.titles_compatible = current_title_key == parent_title_key
        evidence.title_conflict = not evidence.titles_compatible
    elif current_title_key and not parent_title_key:
        evidence.title_conflict = evidence.current_has_new_table_title

    return evidence


def _extract_column_boundaries(table: dict[str, Any]) -> list[float]:
    """提取表格的列边界"""
    bbox = tuple(table.get("bbox", (0, 0, 0, 0)))
    col_count = int(table.get("col_count", 0) or 0)
    
    if col_count <= 0:
        return []

    observed_boundaries = _extract_observed_column_boundaries(table, col_count)
    if observed_boundaries:
        return observed_boundaries
    
    width = bbox[2] - bbox[0]
    if width <= 0:
        return []
    col_width = width / col_count
    
    return [bbox[0] + i * col_width for i in range(col_count + 1)]


def _extract_observed_column_boundaries(
    table: dict[str, Any],
    col_count: int,
) -> list[float]:
    """Build logical column boundaries from real cell x-spans when available."""
    bbox_values = table.get("bbox", (0.0, 0.0, 0.0, 0.0))
    if not isinstance(bbox_values, (list, tuple)) or len(bbox_values) < 4:
        return []
    try:
        table_x0 = float(bbox_values[0])
        table_x1 = float(bbox_values[2])
    except (TypeError, ValueError):
        return []
    if table_x1 <= table_x0:
        return []

    nominal_col_width = (table_x1 - table_x0) / max(1, col_count)
    max_single_col_evidence_width = nominal_col_width * 1.8
    col_ranges: list[list[tuple[float, float]]] = [[] for _ in range(col_count)]
    for cell in table.get("cells", []) or []:
        try:
            logical_col = int(cell.get("col", 0) or cell.get("logical_col", 0) or 0) - 1
        except (TypeError, ValueError):
            continue
        if logical_col < 0 or logical_col >= col_count:
            continue

        bbox = cell.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            x0 = float(bbox[0])
            x1 = float(bbox[2])
        except (TypeError, ValueError):
            continue
        if x1 <= x0:
            continue
        if (x1 - x0) > max_single_col_evidence_width:
            continue
        col_ranges[logical_col].append((x0, x1))

    if sum(1 for ranges in col_ranges if ranges) < max(2, col_count - 1):
        return []

    observed_min: list[float | None] = [
        min((x0 for x0, _ in ranges), default=None) for ranges in col_ranges
    ]
    observed_max: list[float | None] = [
        max((x1 for _, x1 in ranges), default=None) for ranges in col_ranges
    ]
    observed_center: list[float | None] = [
        ((observed_min[idx] + observed_max[idx]) / 2)
        if observed_min[idx] is not None and observed_max[idx] is not None
        else None
        for idx in range(col_count)
    ]

    boundaries: list[float] = [table_x0]
    for idx in range(col_count - 1):
        left_max = observed_max[idx]
        right_min = observed_min[idx + 1]
        left_center = observed_center[idx]
        right_center = observed_center[idx + 1]
        boundary: float | None = None
        if left_max is not None and right_min is not None:
            boundary = (float(left_max) + float(right_min)) / 2
        elif left_center is not None and right_center is not None:
            boundary = (float(left_center) + float(right_center)) / 2
        elif left_max is not None:
            boundary = float(left_max)
        elif right_min is not None:
            boundary = float(right_min)

        if boundary is None or boundary <= boundaries[-1]:
            return []
        boundaries.append(boundary)

    boundaries.append(table_x1)
    if len(boundaries) != col_count + 1:
        return []
    return boundaries


def _assess_continuation_candidates(
    candidates: list[tuple[dict[str, Any], EvidenceBundle]],
    raw_evidence: RawTableEvidence,
) -> ContinuationAssessment:
    """评估续表候选 - 证据评估阶段 (Step2)
    
    职责：
    - 整合多源证据
    - 计算综合置信度
    - 一致性评估
    - 选择最合理的父表
    
    Args:
        candidates: Step1 返回的候选列表
        raw_evidence: 当前表格的原始证据
        
    Returns:
        ContinuationAssessment 评估结果
    """
    assessment = ContinuationAssessment()
    
    if not candidates:
        assessment.decision_type = DecisionType.FALLBACK
        return assessment
    
    best_candidate = None
    best_evidence = None
    best_confidence = 0.0
    best_scores = {"layout": 0.0, "structure": 0.0, "context": 0.0}
    
    for parent, evidence in candidates:
        # 计算各维度评分
        layout_score = _calculate_layout_score(evidence)
        structure_score = _calculate_structure_score(evidence, raw_evidence)
        context_score = _calculate_context_score(evidence)
        
        # 综合置信度计算
        # 权重：结构 40%，版式 35%，上下文 25%
        overall = (
            0.40 * structure_score +
            0.35 * layout_score +
            0.25 * context_score
        )
        
        if overall > best_confidence:
            best_confidence = overall
            best_candidate = parent
            best_evidence = evidence
            best_scores = {
                "layout": layout_score,
                "structure": structure_score,
                "context": context_score,
            }
    
    # 决策：综合置信度阈值
    CONFIDENCE_THRESHOLD = 0.45
    
    if best_confidence >= CONFIDENCE_THRESHOLD and best_candidate:
        assessment.is_continuation = True
        assessment.overall_confidence = best_confidence
        assessment.layout_score = best_scores["layout"]
        assessment.structure_score = best_scores["structure"]
        assessment.context_score = best_scores["context"]
        
        # 填充父表信息
        assessment.selected_parent_id = best_candidate.get("table_id", "")
        assessment.selected_parent_bbox = best_evidence.parent_bbox
        assessment.selected_parent_col_count = best_evidence.structure.parent_col_count
        assessment.selected_parent_header = best_evidence.structure.parent_header
        assessment.selected_parent_column_boundaries = best_evidence.structure.parent_column_boundaries
        
        # 确定判定类型
        assessment.decision_type = _determine_decision_type(
            best_scores, best_evidence
        )
        
        # 记录决策因素
        assessment.decision_factors = _get_decision_factors(
            best_evidence, assessment
        )
        
        # 记录主要证据描述
        assessment.primary_evidence = _get_primary_evidence_description(
            assessment.decision_type, best_evidence, best_scores
        )
        
        # bbox 修正（如果需要）
        if best_evidence.layout.left_boundary_diff > 10.0 or best_evidence.layout.right_boundary_diff > 10.0:
            original_bbox = raw_evidence.bbox
            corrected_bbox = (
                min(raw_evidence.bbox[0], best_evidence.parent_bbox[0]),
                raw_evidence.bbox[1],
                max(raw_evidence.bbox[2], best_evidence.parent_bbox[2]),
                raw_evidence.bbox[3]
            )
            raw_evidence.bbox = corrected_bbox
            raw_evidence.bbox_correction = {
                "original_bbox": original_bbox,
                "corrected_bbox": corrected_bbox,
                "parent_bbox": best_evidence.parent_bbox,
                "left_diff": round(best_evidence.layout.left_boundary_diff, 2),
                "right_diff": round(best_evidence.layout.right_boundary_diff, 2),
                "reason": "continuation_bbox_inheritance"
            }
    
    return assessment


def _assess_continuation_candidates_v2(
    candidates: list[tuple[dict[str, Any], EvidenceBundle]],
    raw_evidence: RawTableEvidence,
) -> ContinuationAssessment:
    """Assess continuation candidates with contradiction-first local-context rules."""
    assessment = ContinuationAssessment()

    if not candidates:
        assessment.decision_type = DecisionType.FALLBACK
        return assessment

    best_candidate = None
    best_evidence = None
    best_confidence = 0.0
    best_scores = {"layout": 0.0, "structure": 0.0, "context": 0.0}

    for parent, evidence in candidates:
        contradiction_factors = _get_contradiction_factors(evidence)
        if contradiction_factors:
            continue

        layout_score = _calculate_layout_score(evidence)
        structure_score = _calculate_structure_score(evidence, raw_evidence)
        context_score = _calculate_context_score_with_local_context(evidence)

        if not _has_minimum_continuation_support(
            evidence,
            layout_score=layout_score,
            structure_score=structure_score,
            context_score=context_score,
        ):
            continue

        overall = (
            0.40 * structure_score +
            0.35 * layout_score +
            0.25 * context_score
        )

        if overall > best_confidence:
            best_confidence = overall
            best_candidate = parent
            best_evidence = evidence
            best_scores = {
                "layout": layout_score,
                "structure": structure_score,
                "context": context_score,
            }

    if best_confidence < 0.45 or not best_candidate or not best_evidence:
        assessment.decision_type = DecisionType.FALLBACK
        return assessment

    assessment.is_continuation = True
    assessment.overall_confidence = best_confidence
    assessment.layout_score = best_scores["layout"]
    assessment.structure_score = best_scores["structure"]
    assessment.context_score = best_scores["context"]
    assessment.selected_parent_id = best_candidate.get("table_id", "")
    assessment.selected_parent_bbox = best_evidence.parent_bbox
    assessment.selected_parent_col_count = best_evidence.structure.parent_col_count
    assessment.selected_parent_header = best_evidence.structure.parent_header
    assessment.selected_parent_column_boundaries = best_evidence.structure.parent_column_boundaries
    assessment.decision_type = _determine_decision_type(best_scores, best_evidence)
    assessment.decision_factors = _get_decision_factors(best_evidence, assessment)
    assessment.primary_evidence = _get_primary_evidence_description(
        assessment.decision_type,
        best_evidence,
        best_scores,
    )

    if best_evidence.layout.left_boundary_diff > 10.0 or best_evidence.layout.right_boundary_diff > 10.0:
        original_bbox = raw_evidence.bbox
        corrected_bbox = (
            min(raw_evidence.bbox[0], best_evidence.parent_bbox[0]),
            raw_evidence.bbox[1],
            max(raw_evidence.bbox[2], best_evidence.parent_bbox[2]),
            raw_evidence.bbox[3],
        )
        raw_evidence.bbox = corrected_bbox
        raw_evidence.bbox_correction = {
            "original_bbox": original_bbox,
            "corrected_bbox": corrected_bbox,
            "parent_bbox": best_evidence.parent_bbox,
            "left_diff": round(best_evidence.layout.left_boundary_diff, 2),
            "right_diff": round(best_evidence.layout.right_boundary_diff, 2),
            "reason": "continuation_bbox_inheritance",
        }

    return assessment


def _get_contradiction_factors(evidence: EvidenceBundle) -> list[str]:
    """Return strong contradiction signals that should block continuation."""
    context = evidence.context
    structure = evidence.structure
    factors: list[str] = []

    if context.current_has_section_heading and not context.current_has_continuation_hint:
        factors.append("section_heading_barrier")

    if context.current_has_narrative_barrier and not context.current_has_continuation_hint:
        factors.append("narrative_barrier")

    if context.title_conflict and not context.current_has_continuation_hint:
        factors.append("title_conflict")

    if context.current_has_new_table_title:
        if context.title_available and not context.titles_compatible:
            factors.append("local_new_table_title")
        elif not context.title_available:
            factors.append("local_new_table_title_without_parent_title")

    if (
        structure.current_has_header_row
        and structure.parent_header_texts
        and structure.header_similarity_score < 0.50
        and not context.current_has_continuation_hint
        and not context.has_continuation_hint
        and not context.titles_compatible
    ):
        factors.append("header_conflict")

    return factors


def _has_minimum_continuation_support(
    evidence: EvidenceBundle,
    layout_score: float,
    structure_score: float,
    context_score: float,
) -> bool:
    """Require structure continuity plus either local/context or layout support."""
    structure = evidence.structure
    has_structure_support = structure_score >= 0.60
    has_layout_support = (
        evidence.layout.prev_near_bottom and evidence.layout.curr_near_top
    ) or layout_score >= 0.70
    has_context_support = (
        evidence.context.current_has_continuation_hint
        or evidence.context.has_continuation_hint
        or evidence.context.titles_compatible
        or context_score >= 0.60
    )
    has_header_semantic_support = True
    if structure.current_has_header_row and structure.parent_header_texts:
        has_header_semantic_support = structure.header_similarity_score >= 0.66
    return has_structure_support and has_header_semantic_support and (has_context_support or has_layout_support)


def _calculate_context_score_with_local_context(evidence: EvidenceBundle) -> float:
    """Score context continuity while prioritizing current-table local evidence."""
    context = evidence.context

    if context.current_has_continuation_hint:
        return 1.0
    if context.title_conflict:
        return 0.0
    if context.current_has_new_table_title and not context.titles_compatible:
        return 0.0
    if context.current_has_narrative_barrier or context.current_has_section_heading:
        return 0.0
    if context.titles_compatible and context.current_title:
        return 0.9
    if context.has_continuation_hint:
        return 0.75
    if context.title_available:
        return 0.55
    return 0.40


def _calculate_layout_score(evidence: EvidenceBundle) -> float:
    """计算版式延续性评分"""
    layout = evidence.layout
    
    # 页序连续性（必要条件）
    if not layout.page_sequential:
        return 0.0
    
    # 版式位置关系评分
    position_score = 0.0
    if layout.prev_near_bottom and layout.curr_near_top:
        position_score = 1.0
    elif layout.prev_bottom_ratio >= 0.60 and layout.curr_top_ratio <= 0.35:
        position_score = 0.8
    elif layout.prev_bottom_ratio >= 0.50 and layout.curr_top_ratio <= 0.40:
        position_score = 0.6
    else:
        position_score = 0.3
    
    # 水平覆盖评分
    overlap_score = layout.horizontal_overlap_ratio
    
    # 边界对齐评分
    avg_width = (evidence.parent_bbox[2] - evidence.parent_bbox[0])
    if avg_width > 0:
        left_align_score = max(0, 1 - abs(layout.left_boundary_diff) / avg_width)
        right_align_score = max(0, 1 - abs(layout.right_boundary_diff) / avg_width)
        alignment_score = (left_align_score + right_align_score) / 2
    else:
        alignment_score = 0.5
    
    # 综合版式评分
    score = 0.40 * position_score + 0.35 * overlap_score + 0.25 * alignment_score
    
    return min(1.0, score)


def _calculate_structure_score(
    evidence: EvidenceBundle,
    raw_evidence: RawTableEvidence,
) -> float:
    """计算结构连续性评分"""
    structure = evidence.structure
    _ = raw_evidence
    
    # 列数匹配评分
    if structure.parent_col_count <= 0:
        return 0.0
    
    # 检查当前表的物理列是否能合理映射到父表的列数
    col_count_score = 0.0
    if structure.current_max_content_cols <= structure.parent_col_count:
        # 内容列数不超过父表列数，合理
        col_count_score = 1.0
    elif structure.current_max_content_cols <= structure.parent_col_count + 1:
        # 内容列数略多，可能是检测误差
        col_count_score = 0.7
    else:
        col_count_score = 0.3
    
    # 列签名相似度评分
    signature_score = 0.0
    if structure.parent_column_signature:
        # 基于父表列签名计算当前表的预期列签名
        signature_score = 0.8
    else:
        signature_score = 0.5
    
    # 表头可继承性评分
    header_score = 0.0
    if structure.parent_header_texts and structure.current_header_texts:
        header_score = structure.header_similarity_score
    elif structure.parent_header_texts:
        header_score = 0.55
    else:
        header_score = 0.3
    
    # 综合结构评分
    score = 0.40 * col_count_score + 0.30 * signature_score + 0.30 * header_score
    
    return min(1.0, score)


def _calculate_context_score(evidence: EvidenceBundle) -> float:
    """计算上下文一致性评分"""
    context = evidence.context
    
    # 标题兼容性评分
    title_score = 0.0
    if context.title_available:
        title_score = 0.8
    else:
        title_score = 0.5
    
    # 续表提示评分
    hint_score = 0.0
    if context.has_continuation_hint:
        hint_score = 1.0
    else:
        hint_score = 0.5
    
    # 综合上下文评分
    score = 0.60 * title_score + 0.40 * hint_score
    
    return min(1.0, score)


def _determine_decision_type(
    scores: dict[str, float],
    evidence: EvidenceBundle,
) -> DecisionType:
    """确定判定类型"""
    layout_score = scores["layout"]
    structure_score = scores["structure"]
    context_score = scores["context"]
    
    # 检查是否有明确的续表提示
    if evidence.context.current_has_continuation_hint or evidence.context.has_continuation_hint:
        return DecisionType.EXPLICIT_HINT
    
    # 检查是否为多证据组合
    high_scores = sum([
        layout_score >= 0.7,
        structure_score >= 0.7,
        context_score >= 0.7,
    ])
    
    if high_scores >= 2:
        return DecisionType.MULTI_EVIDENCE
    
    # 单一证据来源判断
    if structure_score >= layout_score and structure_score >= context_score:
        return DecisionType.STRUCTURE_CONSISTENCY
    elif layout_score >= context_score:
        return DecisionType.LAYOUT_CONTINUITY
    else:
        return DecisionType.CONTEXT_SEMANTIC


def _get_decision_factors(
    evidence: EvidenceBundle,
    assessment: ContinuationAssessment,
) -> list[str]:
    """获取决策因素列表"""
    factors = []
    
    if evidence.layout.page_sequential:
        factors.append("page_sequential")
    
    if evidence.layout.prev_near_bottom and evidence.layout.curr_near_top:
        factors.append("layout_continuity")
    
    if evidence.layout.horizontal_overlap_ratio >= 0.8:
        factors.append("high_horizontal_overlap")
    
    if assessment.structure_score >= 0.7:
        factors.append("structure_consistent")
    
    if evidence.structure.parent_header_texts:
        factors.append("header_inheritable")
    
    if evidence.context.current_has_continuation_hint or evidence.context.has_continuation_hint:
        factors.append("continuation_hint")

    if evidence.context.titles_compatible and evidence.context.current_title:
        factors.append("title_match")
    
    return factors


def _get_primary_evidence_description(
    decision_type: DecisionType,
    evidence: EvidenceBundle,
    scores: dict[str, float],
) -> str:
    """获取主要证据描述"""
    
    if decision_type == DecisionType.LAYOUT_CONTINUITY:
        return (
            f"版式连续性判定：父表在页面底部({evidence.layout.prev_bottom_ratio:.1%})，"
            f"当前表在页面顶部({evidence.layout.curr_top_ratio:.1%})，"
            f"水平重叠{evidence.layout.horizontal_overlap_ratio:.1%}"
        )
    elif decision_type == DecisionType.STRUCTURE_CONSISTENCY:
        return (
            f"结构一致性判定：父表{evidence.structure.parent_col_count}列，"
            f"当前表物理{evidence.structure.current_physical_col_count}列，"
            f"结构评分{scores['structure']:.2f}"
        )
    elif decision_type == DecisionType.CONTEXT_SEMANTIC:
        title_preview = evidence.context.parent_title[:20] if evidence.context.parent_title else "无"
        return (
            f"上下文语义判定：父表标题'{title_preview}...'，"
            f"上下文评分{scores['context']:.2f}"
        )
    elif decision_type == DecisionType.MULTI_EVIDENCE:
        return (
            f"多证据组合判定：版式{scores['layout']:.2f} + "
            f"结构{scores['structure']:.2f} + 上下文{scores['context']:.2f}"
        )
    elif decision_type == DecisionType.EXPLICIT_HINT:
        hint_preview = evidence.context.continuation_hint_text[:50] if evidence.context.continuation_hint_text else ""
        return f"明确续表提示：'{hint_preview}...'"
    else:
        return "降级判断：未找到明确证据"


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
    parent_column_boundaries: list[float] | None = None,  # 新增参数
    assessment: ContinuationAssessment | None = None,  # 新增参数
    current_context: dict[str, Any] | None = None,
    words: list[tuple[float, float, float, float, str]] | None = None,
) -> dict[str, Any] | None:
    """Process raw evidence through the full pipeline.
    
    v1.3.0 更新：
    - 接受 assessment 参数，使用 Step2 的评估结果
    - 接受 parent_column_boundaries 参数，用于 Step3 的列映射计算
    """

    # 从评估结果中获取父表信息
    effective_parent_col_count = parent_col_count
    effective_parent_bbox = parent_bbox
    effective_parent_column_boundaries = parent_column_boundaries
    
    # 如果有评估结果，使用评估结果
    if assessment and assessment.is_continuation:
        effective_parent_col_count = assessment.selected_parent_col_count
        effective_parent_bbox = assessment.selected_parent_bbox
        effective_parent_column_boundaries = assessment.selected_parent_column_boundaries

    # Phase 1: Normalization
    # Pass parent_bbox for column boundary inheritance in continuation tables
    normalized = normalize_raw_evidence(
        raw_evidence=raw_evidence,
        parent_col_count=effective_parent_col_count,
        parent_bbox=effective_parent_bbox,
        parent_column_boundaries=effective_parent_column_boundaries,
        assessment=assessment,
    )

    # Phase 2: Assembly
    table_id = f"tbl_{table_counter + 1:03d}"
    instance = assemble_table_instance(
        normalized=normalized,
        table_id=table_id,
        parent_header=parent_header,
        assessment=assessment,
    )

    # Phase 3: Get context for Continuum Engine
    context_words = words
    if not context_words:
        try:
            context_words = [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]
        except Exception:
            context_words = []
    context = _build_context(
        instance=instance,
        text_blocks=text_blocks,
        page_drawings=page_drawings,
        page_height=page_height,
        words=context_words,
    )

    # Phase 4: Continuum Engine
    prev_instances = _dict_to_instances(prev_tables) if prev_tables else []
    continuum_result = run_continuum_engine(instance, prev_instances, context)

    # Phase 5: Validation check
    confidence = continuum_result.confidence
    toc_only_invalid = (
        not confidence.is_valid_table
        and confidence.overall_confidence >= 0.3
        and "toc_like_table" in confidence.risk_flags
        and set(confidence.risk_flags).issubset({"toc_like_table"})
    )
    if not confidence.is_valid_table and not toc_only_invalid:
        return None

    if confidence.overall_confidence < 0.3:
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

    table_bbox = tuple(ast.bbox)
    result = ast.to_dict()
    result["header_candidates"] = context.get("header_candidates", [])
    if _should_override_header_with_candidates(result):
        result["header"] = [
            {"col": idx + 1, "text": str(text).strip()}
            for idx, text in enumerate(result.get("header_candidates") or [])
            if str(text).strip()
        ]
        result["header_rebuilt_by_context"] = True
    if current_context:
        result["local_context_signal"] = str(current_context.get("local_signal", "none")).strip() or "none"
    preceding_block = context.get("preceding_text_block")
    if preceding_block:
        block_bbox = tuple(preceding_block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        gap = round(max(0.0, table_bbox[1] - block_bbox[3]), 3)
        result["preceding_text_block"] = {
            "text": str(preceding_block.get("text", "")).strip(),
            "bbox": list(block_bbox),
            "gap": gap,
        }
    structural_empty_rows = result.get("structural_empty_rows") or _collect_structural_empty_rows(
        result.get("raw_grid", result.get("grid", []))
    )
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

    raw_rows = raw_evidence.raw_data or result.get("display_grid", []) or result.get("raw_grid", [])
    toc_row_hints = _build_toc_row_hints(raw_rows, raw_evidence.rows)
    semantic_role, semantic_signals = _classify_table_semantic_role(
        table_ast=result,
        raw_evidence=raw_evidence,
        context=context,
        toc_row_hints=toc_row_hints,
    )
    if semantic_role == "business_table" and _looks_like_publication_front_matter_table(
        result,
        context,
        text_blocks,
        page_number=page_number,
    ):
        return None
    if semantic_role == "business_table" and _looks_like_numbered_guidance_false_positive(result):
        return None
    if semantic_role == "business_table" and _looks_like_two_column_narrative_table_ast_false_positive(result):
        return None
    if semantic_role == "business_table" and _looks_like_formula_layout_false_positive(result):
        return None
    if semantic_role == "business_table" and _looks_like_formula_table_ast_false_positive(result):
        return None
    if toc_only_invalid and semantic_role != "toc_outline":
        return None
    result["semantic_role"] = semantic_role
    result["is_business_table"] = semantic_role == "business_table"
    result["semantic_signals"] = semantic_signals
    if semantic_role == "toc_outline":
        result["_toc_row_hints"] = toc_row_hints
        risk_flags = [flag for flag in result.get("risk_flags", []) if flag != "toc_like_table"]
        if risk_flags:
            result["risk_flags"] = risk_flags
        else:
            result.pop("risk_flags", None)
            result.pop("review_required", None)
            result.pop("review_reasons", None)

    return result


def _has_explicit_table_title_signal(
    table_ast: dict[str, Any],
    context: dict[str, Any],
) -> bool:
    title_candidates = [
        str(table_ast.get("title", "")).strip(),
        str(context.get("title_block", {}).get("text", "")).strip() if context.get("title_block") else "",
        str(context.get("preceding_text_block", {}).get("text", "")).strip() if context.get("preceding_text_block") else "",
    ]
    return any(
        candidate
        and _looks_like_table_title(candidate)
        and not TOC_TITLE_PATTERN.search(candidate)
        for candidate in title_candidates
    )


def _looks_like_publication_front_matter_table(
    table_ast: dict[str, Any],
    context: dict[str, Any],
    text_blocks: list[dict[str, Any]] | None,
    *,
    page_number: int,
) -> bool:
    if page_number != 1:
        return False
    if table_ast.get("is_continuation", False):
        return False
    if _has_explicit_table_title_signal(table_ast, context):
        return False

    row_text_blob = "\n".join(table_ast.get("raw_row_texts") or table_ast.get("row_texts") or [])
    row_text_compact = _compact_context_text(row_text_blob)
    table_markers = [
        "articlehistory",
        "received",
        "accepted",
        "availableonline",
        "keywords",
        "highlights",
    ]
    marker_count = sum(1 for marker in table_markers if marker in row_text_compact)
    if marker_count < 2:
        return False

    page_context_blob = " ".join(str(block.get("text", "") or "") for block in (text_blocks or [])[:40])
    page_context_compact = _compact_context_text(page_context_blob)
    page_markers = [
        "contentslistsavailable",
        "sciencedirect",
        "journalhomepage",
        "articleinfo",
        "abstract",
        "highlights",
    ]
    page_marker_count = sum(1 for marker in page_markers if marker in page_context_compact)
    return page_marker_count >= 2


def _looks_like_two_column_narrative_table_ast_false_positive(table_ast: dict[str, Any]) -> bool:
    if table_ast.get("is_continuation", False):
        return False
    if int(table_ast.get("col_count", 0) or 0) != 2:
        return False
    if int(table_ast.get("row_count", 0) or 0) < 4:
        return False
    if _looks_like_table_title(str(table_ast.get("title", "") or "").strip()):
        return False

    header_texts = [
        str(cell.get("text", "")).strip()
        for cell in table_ast.get("header", [])
        if str(cell.get("text", "")).strip()
    ]
    placeholder_header_count = sum(1 for text in header_texts if _is_placeholder_header_text(text))
    if len(header_texts) >= 2 and all(len(re.findall(r"[A-Za-z]+", text)) >= 5 for text in header_texts):
        return True
    if header_texts and placeholder_header_count == 0:
        return False

    data_grid = table_ast.get("data_grid") or table_ast.get("grid") or []
    sentence_like_rows = 0
    for row in data_grid:
        if len(row) < 2:
            continue
        left = str(row[0] or "").strip()
        right = str(row[1] or "").strip()
        left_words = re.findall(r"[A-Za-z]+", left)
        right_words = re.findall(r"[A-Za-z]+", right)
        if len(left_words) >= 5 and len(right_words) >= 5:
            sentence_like_rows += 1

    return sentence_like_rows >= max(3, (len(data_grid) + 1) // 2)


def _looks_like_numbered_guidance_false_positive(table_ast: dict[str, Any]) -> bool:
    if table_ast.get("is_continuation", False):
        return False
    if str(table_ast.get("title", "") or "").strip():
        return False
    if str(table_ast.get("detection_method", "") or "") != "word_clustering":
        return False
    if str(table_ast.get("local_context_signal", "") or "") != "narrative_barrier":
        return False
    if int(table_ast.get("col_count", 0) or 0) != 2:
        return False
    if int(table_ast.get("row_count", 0) or 0) < 4:
        return False

    header_texts = [
        str(cell.get("text", "")).strip()
        for cell in table_ast.get("header", [])
        if str(cell.get("text", "")).strip()
    ]
    if len(header_texts) != 2:
        return False
    if not _looks_like_outline_marker(header_texts[0]):
        return False
    if not _looks_like_structural_heading_text(f"{header_texts[0]} {header_texts[1]}"):
        return False

    data_grid = table_ast.get("data_grid") or table_ast.get("grid") or []
    numbered_rows = 0
    descriptive_right_rows = 0
    leading_narrative_left_rows = 0
    for row in data_grid:
        if len(row) < 2:
            continue
        left = str(row[0] or "").strip()
        right = str(row[1] or "").strip()
        if _looks_like_enumeration_marker(left):
            numbered_rows += 1
            if len(_compact_context_text(right)) >= 8:
                descriptive_right_rows += 1
            continue
        if not right and len(_compact_context_text(left)) >= 12:
            leading_narrative_left_rows += 1

    return (
        numbered_rows >= 2
        and descriptive_right_rows >= 2
        and leading_narrative_left_rows >= 1
    )


def _looks_like_outline_marker(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    return bool(re.fullmatch(r"\d+(?:\.\d+)+\.?", candidate))


def _looks_like_enumeration_marker(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    return bool(
        re.fullmatch(r"(?:\(?\d+\)?|[一二三四五六七八九十]+)[\.\u3001\uff0e\uff09)]?", candidate)
    )


def _looks_like_formula_layout_false_positive(table_ast: dict[str, Any]) -> bool:
    if table_ast.get("is_continuation", False):
        return False
    if str(table_ast.get("title", "") or "").strip():
        return False
    if int(table_ast.get("row_count", 0) or 0) < 3 or int(table_ast.get("row_count", 0) or 0) > 8:
        return False

    header_texts = [
        str(cell.get("text", "")).strip()
        for cell in table_ast.get("header", [])
        if str(cell.get("text", "")).strip()
    ]
    if not header_texts:
        return False

    formula_symbol_count = 0
    total_cells = 0
    sparse_rows = 0
    formula_like_rows = 0
    box_drawing_rows = 0
    null_token_rows = 0
    for row in (table_ast.get("raw_grid") or table_ast.get("grid") or []):
        non_empty = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
        if len(non_empty) <= max(1, len(row) // 2):
            sparse_rows += 1
        row_formula_like = False
        row_box_like = False
        for text in non_empty:
            total_cells += 1
            if re.search(r"[=∑λσμθΩ⎧⎨⎩⎪⎫⎬⎭{}()\[\]]", text):
                formula_symbol_count += 1
                row_formula_like = True
            if re.search(r"[⎧⎨⎩⎪⎫⎬⎭]", text):
                row_box_like = True
            if re.search(r"\bmin\b|\bmax\b|\blog\b|\bexp\b", text, re.IGNORECASE):
                row_formula_like = True
        if row_formula_like:
            formula_like_rows += 1
        if row_box_like:
            box_drawing_rows += 1
        if any(str(cell or "").strip().lower() == "null" for cell in row):
            null_token_rows += 1

    if total_cells == 0:
        return False

    formula_ratio = formula_symbol_count / total_cells
    header_formula_like = any(re.search(r"[=∑λσμθΩ⎧⎨⎩⎪{}()\[\]]", text) for text in header_texts)
    placeholder_header_count = sum(1 for text in header_texts if _is_placeholder_header_text(text))
    row_text_blob = "\n".join(table_ast.get("raw_row_texts") or table_ast.get("row_texts") or [])
    row_texts = [str(text or "").strip() for text in (table_ast.get("row_texts") or []) if str(text or "").strip()]
    leading_symbol_rows = sum(1 for text in row_texts if re.match(r"^[^A-Za-z0-9]+", text))
    if placeholder_header_count >= 2 and leading_symbol_rows >= 2 and " | null | " in row_text_blob:
        return True
    if placeholder_header_count >= 2 and re.search(r"[⎧⎨⎩⎪⎫⎬⎭]", row_text_blob):
        return True
    if placeholder_header_count >= 2 and formula_like_rows >= 2 and null_token_rows >= 3:
        return True
    return (
        (
            formula_ratio >= 0.22
            or formula_like_rows >= max(2, int(table_ast.get("row_count", 0) or 0) // 2)
            or box_drawing_rows >= 2
        )
        and (sparse_rows >= 1 or null_token_rows >= 2)
        and (header_formula_like or placeholder_header_count >= 1)
    )


def _looks_like_formula_table_ast_false_positive(table_ast: dict[str, Any]) -> bool:
    if str(table_ast.get("title", "") or "").strip():
        return False
    if int(table_ast.get("col_count", 0) or 0) < 4:
        return False
    if int(table_ast.get("row_count", 0) or 0) < 4:
        return False
    header_texts = [
        str(cell.get("text", "")).strip()
        for cell in (table_ast.get("header") or [])
        if str(cell.get("text", "")).strip()
    ]
    placeholder_header_count = sum(1 for text in header_texts if _is_placeholder_header_text(text))
    if placeholder_header_count < 2:
        return False
    row_texts = [str(text or "").strip() for text in (table_ast.get("row_texts") or []) if str(text or "").strip()]
    leading_symbol_rows = sum(1 for text in row_texts if re.match(r"^[^A-Za-z0-9]+", text))
    row_text_blob = "\n".join(row_texts)
    return leading_symbol_rows >= 2 and " | null | " in row_text_blob


def _should_override_header_with_candidates(table_ast: dict[str, Any]) -> bool:
    header_candidates = [
        str(text).strip()
        for text in (table_ast.get("header_candidates") or [])
        if str(text).strip()
    ]
    current_header = [
        str(cell.get("text", "")).strip()
        for cell in (table_ast.get("header") or [])
        if str(cell.get("text", "")).strip()
    ]
    if len(header_candidates) < 2 or len(header_candidates) != int(table_ast.get("col_count", 0) or 0):
        return False
    if not current_header or len(current_header) != len(header_candidates):
        return False

    current_numeric_like = sum(1 for text in current_header if re.search(r"\d", text))
    candidate_numeric_like = sum(1 for text in header_candidates if re.search(r"\d", text))
    candidate_schema_like = sum(1 for text in header_candidates if len(re.findall(r"[A-Za-z]+", text)) <= 4)
    if current_numeric_like <= candidate_numeric_like:
        return False
    if candidate_schema_like < max(2, len(header_candidates) - 1):
        return False
    return True


def _classify_table_semantic_role(
    table_ast: dict[str, Any],
    raw_evidence: RawTableEvidence,
    context: dict[str, Any],
    toc_row_hints: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    raw_rows = raw_evidence.raw_data or table_ast.get("display_grid", []) or table_ast.get("raw_grid", [])
    meaningful_rows = 0
    page_locator_rows = 0
    inventory_path_rows = 0
    hierarchical_rows = 0
    locator_indent_positions: list[int] = []
    outline_depths: list[int] = []

    row_hints = toc_row_hints or _build_toc_row_hints(raw_rows, raw_evidence.rows)
    if row_hints:
        meaningful_rows = len(row_hints)
        for row_hint in row_hints:
            if row_hint.get("has_inventory_path"):
                inventory_path_rows += 1
            if row_hint.get("page_locator") and not row_hint.get("has_inventory_path"):
                page_locator_rows += 1
                leading_anchor = row_hint.get("leading_anchor")
                if leading_anchor is not None:
                    locator_indent_positions.append(float(leading_anchor))
            row_outline_depth = int(row_hint.get("outline_depth", 0) or 0)
            row_level = int(row_hint.get("level", 1) or 1)
            if row_outline_depth >= 2 or row_level >= 2:
                hierarchical_rows += 1
            if row_outline_depth > 0:
                outline_depths.append(row_outline_depth)
    else:
        for row in raw_rows:
            filled_cells = _row_non_empty_cells(row)
            if not filled_cells:
                continue
            meaningful_rows += 1
            texts = [text for _, text in filled_cells]
            has_inventory_path = any(_looks_like_inventory_path(text) for text in texts)
            if has_inventory_path:
                inventory_path_rows += 1
            if _looks_like_page_locator(texts[-1]) and not has_inventory_path:
                page_locator_rows += 1
                locator_indent_positions.append(float(filled_cells[0][0]))
            row_outline_depth = max((_outline_token_depth(text) for text in texts[:2]), default=0)
            if row_outline_depth >= 2:
                hierarchical_rows += 1
            if row_outline_depth > 0:
                outline_depths.append(row_outline_depth)

    header_texts = [
        str(cell.get("text", "")).strip()
        for cell in table_ast.get("header", [])
        if str(cell.get("text", "")).strip()
    ]
    semantic_header_texts = [text for text in header_texts if not _is_placeholder_header_text(text)]
    has_schema_header = len(semantic_header_texts) >= 2
    page_schema_header = any(re.search(r"\bpage\b", text, re.IGNORECASE) for text in semantic_header_texts)
    toc_title_support = any(
        TOC_TITLE_PATTERN.search(candidate or "")
        for candidate in (
            table_ast.get("title"),
            context.get("title_block", {}).get("text") if context.get("title_block") else "",
            context.get("preceding_text_block", {}).get("text") if context.get("preceding_text_block") else "",
            header_texts[0] if len(header_texts) == 1 else "",
        )
    )
    explicit_table_title_support = _has_explicit_table_title_signal(table_ast, context)

    page_locator_ratio = (page_locator_rows / meaningful_rows) if meaningful_rows else 0.0
    inventory_path_ratio = (inventory_path_rows / meaningful_rows) if meaningful_rows else 0.0
    hierarchical_ratio = (hierarchical_rows / meaningful_rows) if meaningful_rows else 0.0
    locator_indent_level_count = len(set(locator_indent_positions))
    max_outline_depth = max(outline_depths, default=0)
    outline_ladder = locator_indent_level_count >= 2 or max_outline_depth >= 3 or hierarchical_ratio >= 0.18
    schema_header_blocks_toc = (
        has_schema_header
        and not page_schema_header
        and not toc_title_support
        and page_locator_ratio < 0.8
    )
    toc_outline = (
        not table_ast.get("is_continuation", False)
        and meaningful_rows >= 4
        and page_locator_ratio >= 0.45
        and inventory_path_ratio <= 0.15
        and outline_ladder
        and not schema_header_blocks_toc
        and not explicit_table_title_support
        and (
            toc_title_support
            or locator_indent_level_count >= 3
            or hierarchical_ratio >= 0.28
            or page_schema_header
            or not has_schema_header
        )
    )

    role = "toc_outline" if toc_outline else "business_table"
    reported_has_schema_header = has_schema_header if not toc_outline else False
    signals = {
        "meaningful_row_count": meaningful_rows,
        "page_locator_ratio": round(page_locator_ratio, 3),
        "inventory_path_ratio": round(inventory_path_ratio, 3),
        "hierarchical_ratio": round(hierarchical_ratio, 3),
        "locator_indent_level_count": locator_indent_level_count,
        "max_outline_depth": max_outline_depth,
        "toc_title_support": toc_title_support,
        "explicit_table_title_support": explicit_table_title_support,
        "has_schema_header": reported_has_schema_header,
        "page_schema_header": page_schema_header,
        "schema_header_blocks_toc": schema_header_blocks_toc,
        "semantic_role_reason": (
            "outline_navigation"
            if toc_outline
            else "business_table_default"
        ),
    }
    return role, signals


def _build_toc_block(table_ast: dict[str, Any]) -> dict[str, Any]:
    toc_block = dict(table_ast)
    toc_title = _derive_toc_title(table_ast)
    toc_row_hints = toc_block.pop("_toc_row_hints", None)
    raw_entries = _extract_raw_toc_entries(table_ast, toc_row_hints=toc_row_hints)
    toc_block["block_type"] = "toc"
    toc_block["semantic_role"] = "toc_outline"
    toc_block["is_business_table"] = False
    toc_block["toc_title"] = toc_title
    toc_block["title"] = toc_title
    toc_block["source_candidate_id"] = table_ast.get("table_id")
    toc_block["source_candidate_diagnostics"] = toc_block.get("diagnostics")
    _populate_toc_block_from_raw_entries(toc_block, raw_entries)
    return toc_block


def _extract_raw_toc_entries(
    table_ast: dict[str, Any],
    toc_row_hints: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    raw_rows = table_ast.get("raw_grid") or table_ast.get("display_grid") or []
    row_hints = toc_row_hints or _build_toc_row_hints(raw_rows)
    entries: list[dict[str, Any]] = []

    for row_hint in row_hints:
        row_index = int(row_hint.get("row_index", 0) or 0)
        outline_index = row_hint.get("outline_index")
        page_locator = row_hint.get("page_locator")
        page_locator_kind = _classify_page_locator_kind(page_locator)
        entries.append(
            {
                "row_index": row_index,
                "source_row_indices": [row_index],
                "level": int(row_hint.get("level", 1) or 1),
                "leading_anchor": row_hint.get("leading_anchor"),
                "outline_index": outline_index,
                "outline_depth": int(row_hint.get("outline_depth", 0) or 0),
                "text": str(row_hint.get("text") or "").strip(),
                "page_locator": page_locator,
                "page_locator_kind": page_locator_kind,
                "page_locator_value": _page_locator_value(page_locator, page_locator_kind),
                "sort_y0": float(row_hint.get("sort_y0", row_index) or row_index),
                "sort_x0": float(row_hint.get("sort_x0", 0.0) or 0.0),
                "source_kind": row_hint.get("source_kind", "table_row"),
                "bbox": list(row_hint.get("bbox", [])) if len(row_hint.get("bbox", []) or []) == 4 else None,
            }
        )

    return entries


def _sort_raw_toc_entries(raw_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(entry) for entry in raw_entries],
        key=lambda entry: (
            float(entry.get("sort_y0", entry.get("row_index", 0)) or 0.0),
            float(entry.get("sort_x0", entry.get("leading_anchor", 0.0)) or 0.0),
            int(entry.get("row_index", 0) or 0),
        ),
    )


def _relevel_raw_toc_entries(raw_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_entries = _sort_raw_toc_entries(raw_entries)
    leading_anchors = sorted(
        {
            round(float(entry["leading_anchor"]), 1)
            for entry in normalized_entries
            if entry.get("leading_anchor") is not None
        }
    )
    anchor_to_level = {anchor: index + 1 for index, anchor in enumerate(leading_anchors)}
    for entry in normalized_entries:
        leading_anchor = entry.get("leading_anchor")
        if leading_anchor is None:
            entry["level"] = int(entry.get("level", 1) or 1)
            continue
        entry["level"] = anchor_to_level.get(round(float(leading_anchor), 1), int(entry.get("level", 1) or 1))
    return normalized_entries


def _populate_toc_block_from_raw_entries(
    toc_block: dict[str, Any],
    raw_entries: list[dict[str, Any]],
    external_outline_parent_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prepared_raw_entries = _relevel_raw_toc_entries(raw_entries)
    normalized_entries = _normalize_toc_entries(prepared_raw_entries)
    entries, toc_diagnostics = _annotate_toc_diagnostics(
        normalized_entries,
        external_outline_parent_map=external_outline_parent_map,
    )
    entry_summary = _summarize_toc_entries(prepared_raw_entries, entries)
    toc_block["entries"] = entries
    toc_block["toc_diagnostics"] = toc_diagnostics
    toc_block["entry_count"] = len(entries)
    toc_block["raw_entry_row_count"] = entry_summary["raw_entry_row_count"]
    toc_block["wrapped_entry_count"] = entry_summary["wrapped_entry_count"]
    toc_block["max_entry_level"] = entry_summary["max_entry_level"]
    toc_block["max_outline_depth"] = entry_summary["max_outline_depth"]
    toc_block["page_locator_kinds"] = entry_summary["page_locator_kinds"]
    toc_block["missing_page_locator_count"] = entry_summary["missing_page_locator_count"]
    toc_block["_raw_entries"] = prepared_raw_entries
    _tighten_toc_block_bbox(toc_block)
    return toc_block


def _consolidate_same_page_toc_blocks(toc_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_blocks = sorted(
        list(toc_blocks or []),
        key=lambda toc_block: (
            int(toc_block.get("page", 0) or 0),
            float((toc_block.get("bbox", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])[1]),
            float((toc_block.get("bbox", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])[0]),
        ),
    )
    if not ordered_blocks:
        return []

    consolidated: list[dict[str, Any]] = []
    current_block = ordered_blocks[0]
    for next_block in ordered_blocks[1:]:
        if _can_consolidate_same_page_toc_blocks(current_block, next_block):
            current_block = _combine_same_page_toc_blocks(current_block, next_block)
            continue
        consolidated.append(current_block)
        current_block = next_block
    consolidated.append(current_block)

    toc_blocks[:] = consolidated
    return toc_blocks


def _can_consolidate_same_page_toc_blocks(previous_block: dict[str, Any], current_block: dict[str, Any]) -> bool:
    previous_page = int(previous_block.get("page", 0) or 0)
    current_page = int(current_block.get("page", 0) or 0)
    if previous_page <= 0 or current_page != previous_page:
        return False

    previous_bbox = list(previous_block.get("bbox", []) or [])
    current_bbox = list(current_block.get("bbox", []) or [])
    if len(previous_bbox) != 4 or len(current_bbox) != 4:
        return False

    vertical_gap = float(current_bbox[1]) - float(previous_bbox[3])
    if vertical_gap < -8.0 or vertical_gap > 64.0:
        return False
    if _horizontal_overlap_ratio(tuple(previous_bbox), tuple(current_bbox)) < 0.72:
        return False

    if _has_same_page_toc_outline_parent_link(previous_block, current_block):
        return True
    return _has_forward_toc_outline_progression(previous_block, current_block)


def _combine_same_page_toc_blocks(previous_block: dict[str, Any], current_block: dict[str, Any]) -> dict[str, Any]:
    merged_block = dict(previous_block)
    merged_raw_entries = list(previous_block.get("_raw_entries", []) or []) + list(current_block.get("_raw_entries", []) or [])
    merged_promoted_ids = list(previous_block.get("promoted_text_block_ids", []) or []) + list(
        current_block.get("promoted_text_block_ids", []) or []
    )
    merged_block["title"] = previous_block.get("title") or current_block.get("title")
    merged_block["toc_title"] = previous_block.get("toc_title") or current_block.get("toc_title")
    merged_block["title_inferred"] = bool(
        previous_block.get("title_inferred") or current_block.get("title_inferred")
    )
    merged_block["source_candidate_id"] = previous_block.get("source_candidate_id") or current_block.get("source_candidate_id")
    merged_block["merged_source_candidate_ids"] = list(
        dict.fromkeys(
            [previous_block.get("source_candidate_id"), current_block.get("source_candidate_id")]
            + list(previous_block.get("merged_source_candidate_ids", []) or [])
            + list(current_block.get("merged_source_candidate_ids", []) or [])
        )
    )
    _populate_toc_block_from_raw_entries(merged_block, merged_raw_entries)
    merged_block["promoted_text_block_ids"] = list(dict.fromkeys(item for item in merged_promoted_ids if item))
    return merged_block


def _has_same_page_toc_outline_parent_link(previous_block: dict[str, Any], current_block: dict[str, Any]) -> bool:
    previous_outline_indices = {
        str(entry.get("outline_index") or "").strip()
        for entry in previous_block.get("entries", []) or []
        if str(entry.get("outline_index") or "").strip()
    }
    for entry in current_block.get("entries", []) or []:
        outline_index = str(entry.get("outline_index") or "").strip()
        outline_parent_index = str(entry.get("outline_parent_index") or _parent_outline_index(outline_index) or "").strip()
        if outline_parent_index and outline_parent_index in previous_outline_indices:
            return True
    return False


def _has_forward_toc_outline_progression(previous_block: dict[str, Any], current_block: dict[str, Any]) -> bool:
    previous_outline_key = _last_toc_entry_outline_key(list(previous_block.get("entries", []) or []))
    current_outline_key = _first_toc_entry_outline_key(list(current_block.get("entries", []) or []))
    if previous_outline_key and current_outline_key and previous_outline_key[0] == current_outline_key[0]:
        return current_outline_key[1] > previous_outline_key[1]

    previous_locator_key = _last_toc_page_locator_key(list(previous_block.get("entries", []) or []))
    current_locator_key = _first_toc_page_locator_key(list(current_block.get("entries", []) or []))
    if previous_locator_key and current_locator_key and previous_locator_key[0] == current_locator_key[0]:
        return current_locator_key[1] >= previous_locator_key[1]
    return False


def _first_toc_entry_outline_key(
    entries: list[dict[str, Any]],
) -> tuple[str, tuple[int, ...] | str] | None:
    for entry in entries:
        outline_key = _toc_outline_sort_key(str(entry.get("outline_index") or "").strip())
        if outline_key is not None:
            return outline_key
    return None


def _last_toc_entry_outline_key(
    entries: list[dict[str, Any]],
) -> tuple[str, tuple[int, ...] | str] | None:
    for entry in reversed(entries):
        outline_key = _toc_outline_sort_key(str(entry.get("outline_index") or "").strip())
        if outline_key is not None:
            return outline_key
    return None


def _first_toc_page_locator_key(entries: list[dict[str, Any]]) -> tuple[str, int] | None:
    for entry in entries:
        locator_kind = str(entry.get("page_locator_kind") or "").strip()
        locator_value = entry.get("page_locator_value")
        if locator_kind in {"arabic", "roman"} and locator_value is not None:
            return locator_kind, int(locator_value)
    return None


def _last_toc_page_locator_key(entries: list[dict[str, Any]]) -> tuple[str, int] | None:
    for entry in reversed(entries):
        locator_kind = str(entry.get("page_locator_kind") or "").strip()
        locator_value = entry.get("page_locator_value")
        if locator_kind in {"arabic", "roman"} and locator_value is not None:
            return locator_kind, int(locator_value)
    return None


def _toc_outline_sort_key(outline_index: str) -> tuple[str, tuple[int, ...] | str] | None:
    candidate = str(outline_index or "").strip()
    if not candidate:
        return None
    if candidate.startswith("APPENDIX "):
        return ("appendix", candidate)
    if all(part.isdigit() for part in candidate.split(".") if part):
        return ("numeric", tuple(int(part) for part in candidate.split(".") if part))
    roman_value = _roman_to_int(candidate)
    if roman_value is not None:
        return ("roman", (roman_value,))
    if len(candidate) == 1 and candidate.isalpha():
        return ("alpha", (ord(candidate.upper()) - ord("A") + 1,))
    return None


def _roman_to_int(text: str) -> int | None:
    candidate = str(text or "").strip().upper()
    if not candidate:
        return None
    if any(char not in {"I", "V", "X", "L", "C", "D", "M"} for char in candidate):
        return None
    roman_values = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }
    total = 0
    previous_value = 0
    for char in reversed(candidate):
        value = roman_values[char]
        if value < previous_value:
            total -= value
        else:
            total += value
            previous_value = value
    return total


def _normalize_toc_entries(raw_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_entries: list[dict[str, Any]] = []

    for raw_entry in raw_entries:
        current_entry = dict(raw_entry)
        current_entry["source_row_indices"] = list(raw_entry.get("source_row_indices", []))
        current_entry["source_block_ids"] = _normalize_toc_source_block_ids(raw_entry)
        current_entry["has_wrapped_rows"] = False

        if merged_entries and _should_merge_toc_continuation(merged_entries[-1], current_entry):
            parent_entry = merged_entries[-1]
            parent_entry["text"] = _merge_toc_entry_text(parent_entry.get("text", ""), current_entry.get("text", ""))
            parent_entry["source_row_indices"].extend(current_entry.get("source_row_indices", []))
            parent_entry["source_block_ids"] = _merge_toc_source_block_ids(
                parent_entry.get("source_block_ids"),
                current_entry.get("source_block_ids"),
            )
            merged_bbox = _merge_toc_entry_bboxes(parent_entry.get("bbox"), current_entry.get("bbox"))
            if merged_bbox:
                parent_entry["bbox"] = merged_bbox
            parent_entry["has_wrapped_rows"] = True
            continue

        merged_entries.append(current_entry)

    outline_to_entry_index: dict[str, int] = {}
    level_stack: dict[int, int] = {}
    normalized_entries: list[dict[str, Any]] = []

    for entry_index, entry in enumerate(merged_entries, start=1):
        normalized_entry = dict(entry)
        normalized_entry["entry_index"] = entry_index
        source_row_indices = list(entry.get("source_row_indices", []))
        normalized_entry["source_row_indices"] = source_row_indices
        normalized_entry["source_row_span"] = [
            min(source_row_indices) if source_row_indices else None,
            max(source_row_indices) if source_row_indices else None,
        ]
        normalized_entry["source_row_count"] = len(source_row_indices)
        normalized_entry["outline_parent_index"] = _parent_outline_index(entry.get("outline_index"))
        normalized_entry["source_block_ids"] = list(entry.get("source_block_ids", []) or [])
        if normalized_entry["source_block_ids"] and not normalized_entry.get("source_block_id"):
            normalized_entry["source_block_id"] = normalized_entry["source_block_ids"][0]

        parent_entry_index = None
        outline_parent_index = normalized_entry["outline_parent_index"]
        if outline_parent_index:
            parent_entry_index = outline_to_entry_index.get(outline_parent_index)
        if parent_entry_index is None and not outline_parent_index:
            for level in sorted(level_stack.keys(), reverse=True):
                if level < int(entry.get("level", 1) or 1):
                    parent_entry_index = level_stack[level]
                    break
        normalized_entry["parent_entry_index"] = parent_entry_index

        level = int(entry.get("level", 1) or 1)
        for stacked_level in [item for item in level_stack if item >= level]:
            del level_stack[stacked_level]
        level_stack[level] = entry_index

        outline_index = str(entry.get("outline_index") or "").strip()
        if outline_index:
            outline_to_entry_index[outline_index] = entry_index

        normalized_entries.append(normalized_entry)

    return normalized_entries


def _normalize_toc_source_block_ids(entry: dict[str, Any]) -> list[str]:
    source_block_ids: list[str] = []
    for raw_value in entry.get("source_block_ids", []) or []:
        normalized = str(raw_value or "").strip()
        if normalized and normalized not in source_block_ids:
            source_block_ids.append(normalized)

    single_source_id = str(entry.get("source_block_id") or "").strip()
    if single_source_id and single_source_id not in source_block_ids:
        source_block_ids.append(single_source_id)
    return source_block_ids


def _merge_toc_source_block_ids(
    left_ids: list[str] | None,
    right_ids: list[str] | None,
) -> list[str]:
    merged: list[str] = []
    for raw_value in (left_ids or []) + (right_ids or []):
        normalized = str(raw_value or "").strip()
        if normalized and normalized not in merged:
            merged.append(normalized)
    return merged


def _merge_toc_entry_bboxes(
    left_bbox: list[Any] | tuple[Any, ...] | None,
    right_bbox: list[Any] | tuple[Any, ...] | None,
) -> list[float] | None:
    bboxes = [
        bbox
        for bbox in (_validated_toc_bbox(left_bbox), _validated_toc_bbox(right_bbox))
        if bbox is not None
    ]
    if not bboxes:
        return None
    return _bbox_to_list(_bbox_union(bboxes))


def _validated_toc_bbox(
    bbox: list[Any] | tuple[Any, ...] | None,
) -> tuple[float, float, float, float] | None:
    if not bbox or len(bbox) != 4:
        return None
    try:
        normalized = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        return None
    return normalized


def _tighten_toc_block_bbox(toc_block: dict[str, Any]) -> None:
    entry_bboxes = [
        bbox
        for bbox in (
            _validated_toc_bbox(entry.get("bbox"))
            for entry in toc_block.get("entries", []) or []
        )
        if bbox is not None
    ]
    if not entry_bboxes:
        entry_bboxes = [
            bbox
            for bbox in (
                _validated_toc_bbox(entry.get("bbox"))
                for entry in toc_block.get("_raw_entries", []) or []
            )
            if bbox is not None
        ]
    if not entry_bboxes:
        return

    evidence_bbox = _bbox_union(entry_bboxes)
    explicit_title = str(toc_block.get("title") or toc_block.get("toc_title") or "").strip()
    title_is_inferred = bool(toc_block.get("title_inferred"))
    original_bbox = _validated_toc_bbox(toc_block.get("bbox"))

    if explicit_title and not title_is_inferred and original_bbox is not None:
        evidence_bbox = (
            min(original_bbox[0], evidence_bbox[0]),
            min(original_bbox[1], evidence_bbox[1]),
            max(original_bbox[2], evidence_bbox[2]),
            evidence_bbox[3],
        )

    toc_block["bbox"] = _bbox_to_list(evidence_bbox)


def _summarize_toc_entries(
    raw_entries: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    page_locator_kinds = {
        "arabic": 0,
        "roman": 0,
        "unknown": 0,
    }
    missing_page_locator_count = 0
    wrapped_entry_count = 0
    max_entry_level = 0
    max_outline_depth = 0

    for entry in entries:
        if entry.get("page_locator"):
            kind = str(entry.get("page_locator_kind") or "unknown")
            page_locator_kinds[kind] = page_locator_kinds.get(kind, 0) + 1
        else:
            missing_page_locator_count += 1
        if entry.get("has_wrapped_rows"):
            wrapped_entry_count += 1
        max_entry_level = max(max_entry_level, int(entry.get("level", 0) or 0))
        max_outline_depth = max(max_outline_depth, int(entry.get("outline_depth", 0) or 0))

    return {
        "raw_entry_row_count": len(raw_entries),
        "wrapped_entry_count": wrapped_entry_count,
        "max_entry_level": max_entry_level,
        "max_outline_depth": max_outline_depth,
        "page_locator_kinds": page_locator_kinds,
        "missing_page_locator_count": missing_page_locator_count,
    }


def _annotate_toc_diagnostics(
    entries: list[dict[str, Any]],
    external_outline_parent_map: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotated_entries: list[dict[str, Any]] = [dict(entry) for entry in entries]
    entry_index_map = {
        int(entry.get("entry_index", 0) or 0): entry
        for entry in annotated_entries
        if int(entry.get("entry_index", 0) or 0) > 0
    }
    external_parent_lookup = {
        str(outline_index or "").strip(): dict(entry)
        for outline_index, entry in (external_outline_parent_map or {}).items()
        if str(outline_index or "").strip()
    }
    previous_locator_by_kind: dict[str, dict[str, Any]] = {}
    review_items: list[dict[str, Any]] = []
    issue_counts = {
        "missing_page_locator": 0,
        "unresolved_outline_parent": 0,
        "parent_locator_regression": 0,
        "reading_order_page_regression": 0,
        "empty_heading_text": 0,
    }
    known_locator_count = 0

    for entry in annotated_entries:
        audit_flags: list[str] = []
        entry_index = int(entry.get("entry_index", 0) or 0)
        entry_text = str(entry.get("text") or "").strip()
        locator_kind = str(entry.get("page_locator_kind") or "unknown")
        locator_value = entry.get("page_locator_value")
        outline_index = str(entry.get("outline_index") or "").strip()
        outline_parent_index = str(entry.get("outline_parent_index") or "").strip()
        parent_entry_index = entry.get("parent_entry_index")
        parent_entry = None
        if parent_entry_index:
            parent_entry = entry_index_map.get(int(parent_entry_index))
        if not parent_entry and outline_parent_index:
            parent_entry = external_parent_lookup.get(outline_parent_index)

        entry["sequence_parent_entry_index"] = None
        entry["sequence_parent_outline_index"] = None
        entry["sequence_parent_page"] = None
        entry["sequence_parent_toc_id"] = None
        if parent_entry and parent_entry not in entry_index_map.values():
            entry["sequence_parent_entry_index"] = parent_entry.get("sequence_entry_index")
            entry["sequence_parent_outline_index"] = parent_entry.get("outline_index")
            entry["sequence_parent_page"] = parent_entry.get("page")
            entry["sequence_parent_toc_id"] = parent_entry.get("toc_id")

        section_anchor = _resolve_toc_section_anchor(
            entry,
            entry_index_map,
            external_outline_parent_map=external_parent_lookup,
        )
        entry["section_anchor_entry_index"] = section_anchor["entry_index"]
        entry["section_anchor_outline_index"] = section_anchor["outline_index"]
        entry["section_anchor_text"] = section_anchor["text"]
        entry["section_anchor_page"] = section_anchor["page"]
        entry["section_anchor_toc_id"] = section_anchor["toc_id"]

        if not entry_text:
            audit_flags.append("empty_heading_text")
            issue_counts["empty_heading_text"] += 1
            review_items.append(
                _build_toc_review_item(
                    code="empty_heading_text",
                    severity="medium",
                    entry=entry,
                    message="TOC entry has no normalized heading text.",
                )
            )

        if not entry.get("page_locator"):
            audit_flags.append("missing_page_locator")
            issue_counts["missing_page_locator"] += 1
            review_items.append(
                _build_toc_review_item(
                    code="missing_page_locator",
                    severity="low",
                    entry=entry,
                    message="TOC entry has no visible page locator after normalization.",
                )
            )
        else:
            known_locator_count += 1

        if outline_parent_index and not parent_entry:
            audit_flags.append("unresolved_outline_parent")
            issue_counts["unresolved_outline_parent"] += 1
            severity = "medium" if int(entry.get("outline_depth", 0) or 0) >= 3 else "low"
            review_items.append(
                _build_toc_review_item(
                    code="unresolved_outline_parent",
                    severity=severity,
                    entry=entry,
                    message=f"TOC outline parent '{outline_parent_index}' is not present in the parsed outline.",
                )
            )

        if (
            parent_entry
            and locator_kind in {"arabic", "roman"}
            and locator_kind == str(parent_entry.get("page_locator_kind") or "")
            and locator_value is not None
            and parent_entry.get("page_locator_value") is not None
            and int(locator_value) < int(parent_entry.get("page_locator_value"))
        ):
            audit_flags.append("parent_locator_regression")
            issue_counts["parent_locator_regression"] += 1
            review_items.append(
                _build_toc_review_item(
                    code="parent_locator_regression",
                    severity="medium",
                    entry=entry,
                    message=(
                        f"TOC child page locator {locator_value} is earlier than parent "
                        f"locator {parent_entry.get('page_locator_value')}."
                    ),
                )
            )

        if locator_kind in {"arabic", "roman"} and locator_value is not None:
            previous_entry = previous_locator_by_kind.get(locator_kind)
            if (
                previous_entry
                and previous_entry.get("page_locator_value") is not None
                and int(locator_value) < int(previous_entry["page_locator_value"])
                and int(entry.get("level", 1) or 1) <= int(previous_entry.get("level", 1) or 1)
            ):
                audit_flags.append("reading_order_page_regression")
                issue_counts["reading_order_page_regression"] += 1
                review_items.append(
                    _build_toc_review_item(
                        code="reading_order_page_regression",
                        severity="medium",
                        entry=entry,
                        message=(
                            f"TOC page locator regresses from {previous_entry['page_locator_value']} "
                            f"to {locator_value} in reading order."
                        ),
                    )
                )
            previous_locator_by_kind[locator_kind] = entry

        entry["has_page_locator"] = bool(entry.get("page_locator"))
        entry["audit_flags"] = audit_flags
        entry["review_required"] = any(
            item["severity"] in {"medium", "high"} and int(item.get("entry_index", 0) or 0) == entry_index
            for item in review_items
        )

    mixed_page_numbering = sum(1 for kind, entry in previous_locator_by_kind.items() if entry) >= 2
    aggregated_review_items, aggregated_issue_counts = _aggregate_toc_review_items(review_items)
    diagnostics = {
        "review_required": any(item["severity"] in {"medium", "high"} for item in review_items),
        "review_item_count": len(review_items),
        "aggregated_review_item_count": len(aggregated_review_items),
        "issue_counts": issue_counts,
        "aggregated_issue_counts": aggregated_issue_counts,
        "locator_coverage_ratio": round(known_locator_count / max(1, len(annotated_entries)), 3),
        "mixed_page_numbering": mixed_page_numbering,
        "review_items": review_items,
        "aggregated_review_items": aggregated_review_items,
    }
    return annotated_entries, diagnostics


def _build_toc_review_item(
    code: str,
    severity: str,
    entry: dict[str, Any],
    message: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "entry_index": int(entry.get("entry_index", 0) or 0),
        "outline_index": entry.get("outline_index"),
        "page_locator": entry.get("page_locator"),
        "section_anchor_entry_index": entry.get("section_anchor_entry_index"),
        "section_anchor_outline_index": entry.get("section_anchor_outline_index"),
        "section_anchor_text": entry.get("section_anchor_text"),
        "message": message,
    }


def _resolve_toc_section_anchor(
    entry: dict[str, Any],
    entry_index_map: dict[int, dict[str, Any]],
    external_outline_parent_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    anchor = entry
    parent_entry_index = entry.get("parent_entry_index")
    if parent_entry_index:
        parent_entry = entry_index_map.get(int(parent_entry_index))
        if parent_entry:
            anchor = parent_entry
    if anchor is entry:
        outline_parent_index = str(entry.get("outline_parent_index") or "").strip()
        if outline_parent_index and external_outline_parent_map:
            external_parent = external_outline_parent_map.get(outline_parent_index)
            if external_parent:
                anchor = external_parent

    current = anchor
    visited: set[tuple[int, str, int]] = set()
    while current:
        current_index = int(current.get("entry_index", 0) or 0)
        current_outline = str(current.get("outline_index") or "").strip()
        current_page = int(current.get("page", 0) or 0)
        visit_key = (current_index, current_outline, current_page)
        if current_index > 0 or current_outline:
            if visit_key in visited:
                break
            visited.add(visit_key)
        if str(current.get("outline_index") or "").strip() or str(current.get("text") or "").strip():
            break
        next_parent = current.get("parent_entry_index")
        if next_parent:
            current = entry_index_map.get(int(next_parent))
            continue
        next_outline_parent = str(current.get("outline_parent_index") or "").strip()
        if next_outline_parent and external_outline_parent_map:
            current = external_outline_parent_map.get(next_outline_parent)
            continue
        break

    return {
        "entry_index": int(current.get("entry_index", 0) or 0) if current else None,
        "outline_index": current.get("outline_index") if current else None,
        "text": current.get("text") if current else None,
        "page": int(current.get("page", 0) or 0) if current else None,
        "toc_id": current.get("toc_id") if current else None,
    }


def _aggregate_toc_review_items(
    review_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[tuple[str, int | None, str], list[dict[str, Any]]] = {}
    for item in review_items:
        code = str(item.get("code") or "")
        if code != "missing_page_locator":
            continue
        section_anchor_entry_index = item.get("section_anchor_entry_index")
        section_anchor_outline_index = str(item.get("section_anchor_outline_index") or "")
        group_key = (code, section_anchor_entry_index, section_anchor_outline_index)
        grouped.setdefault(group_key, []).append(item)

    aggregated_review_items: list[dict[str, Any]] = []
    aggregated_issue_counts: dict[str, int] = {}

    for _, items in grouped.items():
        if len(items) < 2:
            continue
        sorted_items = sorted(items, key=lambda item: int(item.get("entry_index", 0) or 0))
        first_item = sorted_items[0]
        outline_indices = [item.get("outline_index") for item in sorted_items if item.get("outline_index")]
        entry_indices = [int(item.get("entry_index", 0) or 0) for item in sorted_items]
        section_outline_index = first_item.get("section_anchor_outline_index")
        section_text = str(first_item.get("section_anchor_text") or "").strip()
        section_label_parts = [part for part in (section_outline_index, section_text) if part]
        section_label = " ".join(section_label_parts).strip() or "current TOC section"

        aggregated_review_items.append(
            {
                "code": first_item.get("code"),
                "severity": first_item.get("severity"),
                "section_anchor_entry_index": first_item.get("section_anchor_entry_index"),
                "section_anchor_outline_index": section_outline_index,
                "section_anchor_text": section_text,
                "item_count": len(sorted_items),
                "entry_indices": entry_indices,
                "entry_index_range": [min(entry_indices), max(entry_indices)],
                "outline_indices": outline_indices,
                "message": (
                    f"Section '{section_label}' contains {len(sorted_items)} child TOC entries "
                    "without visible page locators."
                ),
            }
        )
        code = str(first_item.get("code") or "")
        aggregated_issue_counts[code] = aggregated_issue_counts.get(code, 0) + 1

    aggregated_review_items.sort(
        key=lambda item: (
            int((item.get("entry_index_range") or [0])[0] or 0),
            str(item.get("code") or ""),
        )
    )
    return aggregated_review_items, aggregated_issue_counts


def _derive_toc_title(table_ast: dict[str, Any]) -> str | None:
    explicit_title = str(table_ast.get("title") or "").strip()
    if explicit_title:
        return explicit_title

    for row in table_ast.get("display_grid", []) or table_ast.get("raw_grid", []) or []:
        filled_cells = _row_non_empty_cells(row)
        if len(filled_cells) != 1:
            continue
        text = filled_cells[0][1]
        if _is_toc_title_row(text):
            return text

    header_texts = [
        str(cell.get("text", "")).strip()
        for cell in table_ast.get("header", [])
        if str(cell.get("text", "")).strip()
    ]
    if len(header_texts) == 1 and _is_toc_title_row(header_texts[0]):
        return header_texts[0]
    return explicit_title or None


def _should_merge_toc_continuation(
    previous_entry: dict[str, Any],
    current_entry: dict[str, Any],
) -> bool:
    current_text = str(current_entry.get("text") or "").strip()
    if not current_text:
        return False
    if current_entry.get("outline_index") or current_entry.get("page_locator"):
        return False
    if not previous_entry:
        return False

    previous_level = int(previous_entry.get("level", 1) or 1)
    current_level = int(current_entry.get("level", 1) or 1)
    if current_level + 1 < previous_level:
        return False

    return bool(previous_entry.get("text"))


def _merge_toc_entry_text(previous_text: str, continuation_text: str) -> str:
    left = str(previous_text or "").strip()
    right = str(continuation_text or "").strip()
    if not left:
        return right
    if not right:
        return left
    if left.endswith("-"):
        return f"{left[:-1]}{right}"
    return f"{left.rstrip()} {right.lstrip()}".strip()


def _parent_outline_index(outline_index: str | None) -> str | None:
    candidate = str(outline_index or "").strip()
    if not candidate:
        return None
    segments = [segment for segment in candidate.split(".") if segment]
    if len(segments) <= 1:
        return None
    if len(segments) == 2:
        if segments[1] == "0":
            return None
        return f"{segments[0]}.0"
    return ".".join(segments[:-1])


def _row_non_empty_cells(row: list[Any]) -> list[tuple[int, str]]:
    filled_cells: list[tuple[int, str]] = []
    for cell_index, cell in enumerate(row or []):
        text = str(cell or "").replace("\n", " ").strip()
        if text:
            filled_cells.append((cell_index, text))
    return filled_cells


def _is_toc_title_row(text: str) -> bool:
    candidate = str(text or "").strip()
    return bool(candidate) and bool(TOC_TITLE_ROW_PATTERN.fullmatch(candidate))


def _build_toc_row_hints(
    raw_rows: list[list[Any]],
    raw_row_objects: list[RawRow] | None = None,
) -> list[dict[str, Any]]:
    provisional_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(raw_rows, start=1):
        filled_cells = _row_non_empty_cells(row)
        if not filled_cells:
            continue
        texts = [text for _, text in filled_cells]
        if len(texts) == 1 and _is_toc_title_row(texts[0]):
            continue

        content_texts = list(texts)
        page_locator: str | None = None
        if _looks_like_page_locator(texts[-1]) and not _looks_like_inventory_path(texts[-1]):
            page_locator = texts[-1]
            content_texts = texts[:-1]
        elif content_texts:
            inline_text, inline_page_locator = _split_inline_toc_page_locator(content_texts[-1])
            if inline_page_locator:
                page_locator = inline_page_locator
                content_texts = content_texts[:-1]
                if inline_text:
                    content_texts.append(inline_text)

        content_texts = [str(text).strip() for text in content_texts if str(text).strip()]
        if not content_texts:
            continue

        outline_index, outline_depth, leading_remainder, outline_kind = _extract_outline_prefix(content_texts[0])
        entry_texts = ([leading_remainder] if leading_remainder else []) + content_texts[1:] if outline_index else content_texts
        entry_text = " ".join(text for text in entry_texts if text).strip()
        if page_locator and _looks_like_numeric_toc_content_stub(entry_text):
            continue
        sort_y0 = float(row_index)
        sort_x0 = float(_toc_row_leading_anchor(row_index, filled_cells, raw_row_objects) or 0.0)
        if raw_row_objects and 0 < row_index <= len(raw_row_objects):
            raw_row = raw_row_objects[row_index - 1]
            if raw_row.bbox:
                sort_y0 = float(raw_row.bbox[1])
        row_bbox = None
        if raw_row_objects and 0 < row_index <= len(raw_row_objects):
            raw_row = raw_row_objects[row_index - 1]
            if raw_row.bbox:
                row_bbox = list(raw_row.bbox)
        provisional_rows.append(
            {
                "row_index": row_index,
                "leading_anchor": sort_x0,
                "outline_index": outline_index,
                "outline_depth": outline_depth,
                "outline_kind": outline_kind,
                "text": entry_text,
                "page_locator": page_locator,
                "has_inventory_path": any(_looks_like_inventory_path(text) for text in texts),
                "sort_y0": sort_y0,
                "sort_x0": sort_x0,
                "source_kind": "table_row",
                "bbox": row_bbox,
            }
        )

    leading_anchors = sorted(
        {
            float(row["leading_anchor"])
            for row in provisional_rows
            if row.get("leading_anchor") is not None
        }
    )
    anchor_to_level = {anchor: index + 1 for index, anchor in enumerate(leading_anchors)}

    row_hints: list[dict[str, Any]] = []
    for row in provisional_rows:
        row_hint = dict(row)
        leading_anchor = row_hint.get("leading_anchor")
        row_hint["level"] = anchor_to_level.get(float(leading_anchor), 1) if leading_anchor is not None else 1
        row_hints.append(row_hint)
    return row_hints


def _toc_row_leading_anchor(
    row_index: int,
    filled_cells: list[tuple[int, str]],
    raw_row_objects: list[RawRow] | None = None,
) -> float | None:
    if raw_row_objects and 0 < row_index <= len(raw_row_objects):
        raw_row = raw_row_objects[row_index - 1]
        for cell in raw_row.cells:
            if not str(cell.text or "").strip():
                continue
            if cell.bbox:
                return round(float(cell.bbox[0]), 1)
    if filled_cells:
        return float(filled_cells[0][0])
    return None


def _build_toc_text_block_entries(
    text_blocks: list[dict[str, Any]],
    occupied_bboxes: list[tuple[float, float, float, float]] | None = None,
    allow_embedded_rows: bool = False,
) -> list[dict[str, Any]]:
    raw_entries: list[dict[str, Any]] = []
    occupied = occupied_bboxes or []
    synthetic_row_index = 0

    candidate_blocks: list[dict[str, Any]] = []
    for text_block in sorted(text_blocks, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        bbox = _validated_toc_bbox(text_block.get("bbox"))
        if bbox is None:
            continue
        if (
            not allow_embedded_rows
            and any(_bbox_overlap_ratio(bbox, other_bbox) >= 0.5 for other_bbox in occupied)
        ):
            continue

        original_text = str(text_block.get("text", "")).replace("\n", " ").strip()
        if not original_text:
            continue
        candidate_blocks.append(text_block)

    for row_blocks in _group_toc_text_blocks_by_visual_rows(candidate_blocks):
        built_entry = _build_toc_text_row_entry(row_blocks)
        if not built_entry:
            continue
        synthetic_row_index += 1
        built_entry["row_index"] = synthetic_row_index
        raw_entries.append(built_entry)

    return raw_entries


def _group_toc_text_blocks_by_visual_rows(
    text_blocks: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    if not text_blocks:
        return []

    row_tolerance = _toc_text_row_tolerance(text_blocks)
    rows: list[dict[str, Any]] = []
    for text_block in sorted(text_blocks, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        bbox = _validated_toc_bbox(text_block.get("bbox"))
        if bbox is None:
            continue
        block_center_y = (bbox[1] + bbox[3]) / 2

        best_row = None
        best_distance = float("inf")
        for row in rows:
            row_bbox = row["bbox"]
            row_center_y = (row_bbox[1] + row_bbox[3]) / 2
            center_distance = abs(block_center_y - row_center_y)
            if center_distance > row_tolerance and _toc_vertical_overlap_ratio(row_bbox, bbox) < 0.55:
                continue
            if center_distance < best_distance:
                best_distance = center_distance
                best_row = row

        if best_row is None:
            rows.append({"bbox": bbox, "blocks": [text_block]})
            continue

        best_row["blocks"].append(text_block)
        best_row["bbox"] = _bbox_union([best_row["bbox"], bbox])

    return [
        sorted(row["blocks"], key=lambda item: item["bbox"][0])
        for row in rows
    ]


def _toc_text_row_tolerance(text_blocks: list[dict[str, Any]]) -> float:
    heights = sorted(
        max(1.0, float(block["bbox"][3]) - float(block["bbox"][1]))
        for block in text_blocks
        if len(block.get("bbox", []) or []) == 4
    )
    if not heights:
        return 6.0
    median_height = heights[len(heights) // 2]
    return max(3.0, min(12.0, median_height * 0.7))


def _toc_vertical_overlap_ratio(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    overlap = max(0.0, min(bbox_a[3], bbox_b[3]) - max(bbox_a[1], bbox_b[1]))
    min_height = min(max(1.0, bbox_a[3] - bbox_a[1]), max(1.0, bbox_b[3] - bbox_b[1]))
    return overlap / min_height


def _build_toc_text_row_entry(
    row_blocks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    ordered_blocks = sorted(row_blocks, key=lambda item: item["bbox"][0])
    source_block_ids: list[str] = []
    row_bboxes: list[tuple[float, float, float, float]] = []
    content_segments: list[str] = []
    page_locator: str | None = None
    locator_is_standalone = False
    leading_anchor: float | None = None

    for text_block in ordered_blocks:
        bbox = _validated_toc_bbox(text_block.get("bbox"))
        if bbox is None:
            continue
        original_text = str(text_block.get("text", "")).replace("\n", " ").strip()
        if not original_text:
            continue
        if _is_toc_title_row(original_text):
            continue

        source_block_id = str(text_block.get("block_id") or "").strip()
        if source_block_id and source_block_id not in source_block_ids:
            source_block_ids.append(source_block_id)
        row_bboxes.append(bbox)

        inline_text, inline_page_locator = _split_inline_toc_page_locator(original_text)
        if inline_page_locator:
            if inline_text:
                content_segments.append(inline_text)
                if leading_anchor is None:
                    leading_anchor = round(float(bbox[0]), 1)
            page_locator = inline_page_locator
            continue

        if _looks_like_page_locator(original_text) and not _looks_like_inventory_path(original_text):
            page_locator = original_text
            locator_is_standalone = True
            continue

        content_segments.append(original_text)
        if leading_anchor is None:
            leading_anchor = round(float(bbox[0]), 1)

    if not row_bboxes:
        return None

    content_text = re.sub(r"\s+", " ", " ".join(content_segments)).strip()
    if not content_text or not page_locator:
        return None

    outline_index, outline_depth, remainder_text, _ = _extract_outline_prefix(content_text)
    entry_text = (remainder_text if outline_index else content_text).strip()
    if not entry_text:
        return None
    if not _is_toc_text_row_candidate(
        entry_text,
        outline_index=outline_index,
        locator_is_standalone=locator_is_standalone,
        block_count=len(row_bboxes),
    ):
        return None

    row_bbox = _bbox_union(row_bboxes)
    page_locator_kind = _classify_page_locator_kind(page_locator)
    return {
        "source_row_indices": [],
        "source_block_id": source_block_ids[0] if source_block_ids else None,
        "source_block_ids": source_block_ids,
        "source_kind": "text_block",
        "leading_anchor": leading_anchor if leading_anchor is not None else round(float(row_bbox[0]), 1),
        "outline_index": outline_index,
        "outline_depth": outline_depth,
        "text": entry_text,
        "page_locator": page_locator,
        "page_locator_kind": page_locator_kind,
        "page_locator_value": _page_locator_value(page_locator, page_locator_kind),
        "sort_y0": float(row_bbox[1]),
        "sort_x0": float(row_bbox[0]),
        "bbox": _bbox_to_list(row_bbox),
    }


def _is_toc_text_row_candidate(
    text: str,
    *,
    outline_index: str | None,
    locator_is_standalone: bool,
    block_count: int,
) -> bool:
    candidate = re.sub(r"\s+", " ", str(text or "")).strip()
    if not candidate:
        return False
    if _looks_like_inventory_path(candidate):
        return False
    if len(candidate) > 220:
        return False
    if re.match(r"^(?:figure|table)\b", candidate, re.IGNORECASE):
        return False
    if outline_index:
        return True
    if locator_is_standalone and block_count >= 2:
        return True
    return len(candidate.split()) <= 10 and len(candidate) <= 96


def _split_inline_toc_page_locator(text: str) -> tuple[str, str | None]:
    candidate = str(text or "").replace("\n", " ").strip()
    if not candidate:
        return "", None
    match = INLINE_TOC_PAGE_LOCATOR_PATTERN.match(candidate)
    if not match:
        return candidate, None
    locator = match.group("locator").strip()
    if not _looks_like_page_locator(locator):
        return candidate, None
    body = match.group("body").rstrip(" .\u2026·•_")
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return candidate, None
    return body, locator


def _extract_outline_prefix(text: str) -> tuple[str | None, int, str, str | None]:
    candidate = str(text or "").strip()
    if not candidate:
        return None, 0, "", None

    match = OUTLINE_INDEX_TOKEN_PATTERN.match(candidate)
    if match:
        outline_index = match.group(1)
        remainder = candidate[match.end():].lstrip(" .:-")
        return outline_index, len([segment for segment in outline_index.split(".") if segment]), remainder, "numeric"

    match = SINGLE_NUMERIC_OUTLINE_PREFIX_PATTERN.match(candidate)
    if match:
        outline_index = f"{match.group(1)}.0"
        remainder = candidate[match.end():].strip()
        return outline_index, 2, remainder, "numeric"

    match = APPENDIX_OUTLINE_PREFIX_PATTERN.match(candidate)
    if match:
        outline_index = re.sub(r"\s+", " ", match.group(1).upper()).strip()
        remainder = candidate[match.end():].strip()
        return outline_index, 1, remainder, "appendix"

    match = ROMAN_OUTLINE_PREFIX_PATTERN.match(candidate)
    if match:
        outline_index = match.group(1).upper()
        remainder = candidate[match.end():].strip()
        return outline_index, 1, remainder, "roman"

    match = ALPHA_OUTLINE_PREFIX_PATTERN.match(candidate)
    if match:
        outline_index = match.group(1).upper()
        remainder = candidate[match.end():].strip()
        return outline_index, 1, remainder, "alpha"

    return None, 0, candidate, None


def _looks_like_page_locator(text: str) -> bool:
    candidate = str(text or "").strip()
    return bool(candidate) and bool(PAGE_LOCATOR_PATTERN.fullmatch(candidate))


def _classify_page_locator_kind(text: str | None) -> str:
    candidate = str(text or "").strip()
    if not candidate:
        return "unknown"
    if candidate.isdigit():
        return "arabic"
    if re.fullmatch(r"[ivxlcdm]+", candidate, re.IGNORECASE):
        return "roman"
    return "unknown"


def _page_locator_value(text: str | None, locator_kind: str | None = None) -> int | None:
    candidate = str(text or "").strip()
    if not candidate:
        return None
    kind = locator_kind or _classify_page_locator_kind(candidate)
    if kind == "arabic" and candidate.isdigit():
        return int(candidate)
    if kind == "roman":
        return _roman_to_int(candidate)
    return None


def _looks_like_inventory_path(text: str) -> bool:
    candidate = str(text or "").strip()
    return bool(candidate) and bool(INVENTORY_PATH_PATTERN.search(candidate))


def _is_placeholder_header_text(text: str) -> bool:
    candidate = str(text or "").strip()
    return bool(candidate) and bool(re.fullmatch(r"column\s+\d+", candidate, re.IGNORECASE))


def _outline_token_depth(text: str) -> int:
    candidate = str(text or "").strip()
    if not candidate:
        return 0
    _, outline_depth, _, _ = _extract_outline_prefix(candidate)
    return outline_depth


def _roman_to_int(text: str) -> int | None:
    candidate = str(text or "").strip().upper()
    if not candidate or not re.fullmatch(r"[IVXLCDM]+", candidate):
        return None

    roman_values = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }
    total = 0
    previous_value = 0
    for char in reversed(candidate):
        value = roman_values[char]
        if value < previous_value:
            total -= value
        else:
            total += value
            previous_value = value
    return total


def _build_context(
    instance: TableInstance,
    text_blocks: list[dict[str, Any]] | None,
    page_drawings: list[dict[str, Any]] | None,
    page_height: float,
    words: list[tuple[float, float, float, float, str]] | None,
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
    cfg = get_pdf_parser_settings().cross_page_stitching
    header_candidates = _extract_header_candidates(words, bbox)
    preceding_block = _find_preceding_text_block(text_blocks, bbox, cfg)
    title_block = _find_title_block(text_blocks, bbox)
    instance_title_text = str(instance.title or "").strip()
    if title_block is not None and instance_title_text:
        title_identity = _normalize_title_identity(str(title_block.get("text", "")).strip())
        instance_identity = _normalize_title_identity(instance_title_text)
        if title_identity and instance_identity and title_identity != instance_identity:
            title_block = None
    if title_block is None:
        title_block = _promote_preceding_heading_to_structural_title(
            instance=instance,
            preceding_block=preceding_block,
            header_candidates=header_candidates,
        )

    context["title_block"] = title_block
    if context["title_block"] is None and instance.title:
        context["title_block"] = {
            "text": instance.title,
            "bbox": list(bbox),
            "source": "internal_title_row",
        }
    context["section_hint"] = _find_section_hint_block(text_blocks, bbox)
    context["preceding_text_block"] = preceding_block
    context["header_candidates"] = header_candidates

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


def _extract_header_candidates(
    words: list[tuple[float, float, float, float, str]] | None,
    bbox: tuple[float, float, float, float],
) -> list[str]:
    if not words or not bbox:
        return []

    top = float(bbox[1])

    def _word_coords(word):
        if isinstance(word, tuple):
            x0, y0, x1, y1, text = word[:5]
        else:
            x0, y0, x1, y1 = word.x0, word.y0, word.x1, word.y1
            text = getattr(word, "text", "")
        return x0, y0, x1, y1, text

    row_buckets: dict[float, list[tuple[float, float, str, float]]] = {}
    for word in words:
        x0, y0, x1, y1, text = _word_coords(word)
        if x1 < bbox[0] or x0 > bbox[2]:
            continue
        if y0 < top - 28 or y0 > top + 8:
            continue
        key = round(float(y0) / 3.0) * 3.0
        row_buckets.setdefault(key, []).append((float(x0), float(x1), str(text), float(y0)))

    def _group_row_tokens(row_words: list[tuple[float, float, str, float]]) -> list[str]:
        row_words = sorted(row_words, key=lambda item: item[0])
        grouped: list[list[str]] = []
        current: list[str] = []
        last_x1: float | None = None
        header_gap_threshold = 15.0
        for x0, x1, text, _ in row_words:
            if last_x1 is not None and x0 - last_x1 > header_gap_threshold:
                if current:
                    grouped.append(current)
                current = []
            current.append(text)
            last_x1 = x1
        if current:
            grouped.append(current)
        return [" ".join(group) for group in grouped if group]

    def _row_score(header_candidates: list[str], row_y0: float) -> tuple[float, float]:
        alpha_count = sum(len(re.findall(r"[A-Za-z]+", text)) for text in header_candidates)
        numeric_count = sum(1 for text in header_candidates if re.search(r"\d", text))
        score = alpha_count - (numeric_count * 0.8) - abs(top - row_y0) * 0.05
        return score, -abs(top - row_y0)

    direct_row_candidates: list[str] = []
    direct_row_rank: tuple[float, float] | None = None
    best_candidates: list[str] = []
    best_rank: tuple[float, float] | None = None
    for row_y_key, row_words in row_buckets.items():
        candidates = _group_row_tokens(row_words)
        if len(candidates) < 2:
            continue
        row_y0 = min(item[3] for item in row_words)
        rank = _row_score(candidates, row_y0)
        if abs(row_y0 - top) <= 8:
            if direct_row_rank is None or rank > direct_row_rank:
                direct_row_rank = rank
                direct_row_candidates = candidates
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_candidates = candidates

    if direct_row_candidates:
        direct_numeric_like = sum(1 for text in direct_row_candidates if re.search(r"\d", text))
        if direct_numeric_like <= max(1, len(direct_row_candidates) // 2):
            return direct_row_candidates

    return best_candidates


def _horizontal_overlap_ratio(bbox_a: tuple, bbox_b: tuple) -> float:
    """Calculate horizontal overlap ratio."""
    x_overlap = max(0, min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0]))
    width_a = bbox_a[2] - bbox_a[0]
    width_b = bbox_b[2] - bbox_b[0]
    if width_a <= 0 or width_b <= 0:
        return 0.0
    return x_overlap / min(width_a, width_b)


def _find_preceding_text_block(
    text_blocks: list[dict[str, Any]] | None,
    bbox: tuple[float, float, float, float],
    cfg: CrossPageStitchingThresholds,
) -> dict[str, Any] | None:
    """Locate a text block immediately above a table within configurable bounds."""
    if not text_blocks:
        return None
    best_block: dict[str, Any] | None = None
    best_gap = float("inf")
    for block in text_blocks:
        block_bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        gap = bbox[1] - block_bbox[3]
        if gap <= 0 or gap > cfg.preceding_text_gap_max:
            continue
        if block_bbox[1] <= cfg.preceding_text_ignore_top_margin:
            continue
        overlap = _horizontal_overlap_ratio(bbox, block_bbox)
        if overlap < cfg.preceding_text_overlap_min:
            continue
        text = str(block.get("text", "")).strip()
        if len(text) < cfg.preceding_text_min_length:
            continue
        if gap < best_gap:
            best_block = block
            best_gap = gap
    return best_block


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
            if cell.get("logical_row", cell.get("row", 0)) == source_row + 1:
                cell["text"] = None

    # 移除空行并更新 row_texts
    if rows_to_merge:
        old_data_row_count = len(ast.data_grid) if getattr(ast, "data_grid", None) else len(ast.grid)
        display_prefix_len = 0
        if getattr(ast, "display_grid", None):
            display_prefix_len = max(0, len(ast.display_grid) - old_data_row_count)
        # 过滤空行
        new_grid = [row for row in ast.grid if any(c for c in row if c and c.strip())]
        ast.grid = new_grid
        ast.row_count = len(new_grid)
        ast.data_grid = [list(row) for row in new_grid]
        ast.data_row_count = len(new_grid)
        ast.logical_row_count = len(new_grid)

        # 重建 row_texts
        ast.row_texts = []
        for row in new_grid:
            parts = []
            for cell in row:
                text = cell.replace('\n', ' ').strip() if cell else "null"
                parts.append(text if text else "null")
            ast.row_texts.append(" | ".join(parts))
        ast.data_row_texts = list(ast.row_texts)

        if getattr(ast, "display_grid", None):
            ast.display_grid = ast.display_grid[:display_prefix_len] + [list(row) for row in new_grid]
            ast.display_row_count = len(ast.display_grid)
            ast.display_row_texts = []
            for row in ast.display_grid:
                parts = []
                for cell in row:
                    text = cell.replace('\n', ' ').strip() if cell else "null"
                    parts.append(text if text else "null")
                ast.display_row_texts.append(" | ".join(parts))


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
        page_words = _extract_words_from_page(page)

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
                    text=w[4],
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


def _collect_item_bboxes(
    items: list[dict[str, Any]],
) -> list[tuple[float, float, float, float]]:
    bboxes: list[tuple[float, float, float, float]] = []
    for item in items:
        bbox_raw = item.get("bbox")
        if not bbox_raw or len(bbox_raw) != 4:
            continue
        bboxes.append(tuple(float(value) for value in bbox_raw))
    return bboxes


def _exclude_words_in_occupied_regions(
    words: list[_Word],
    occupied_bboxes: list[tuple[float, float, float, float]],
) -> list[_Word]:
    if not words or not occupied_bboxes:
        return list(words)

    remaining: list[_Word] = []
    for word in words:
        word_bbox = (word.x0, word.y0, word.x1, word.y1)
        occupied = False
        for bbox in occupied_bboxes:
            if _bbox_overlap_ratio(word_bbox, bbox) >= 0.5:
                occupied = True
                break
            if bbox[0] - 1.0 <= word.xc <= bbox[2] + 1.0 and bbox[1] - 1.0 <= word.yc <= bbox[3] + 1.0:
                occupied = True
                break
        if not occupied:
            remaining.append(word)
    return remaining


def _word_identity(word: Any) -> tuple[str, float, float, float, float]:
    return (
        str(getattr(word, "text", "")),
        round(float(getattr(word, "x0", 0.0)), 3),
        round(float(getattr(word, "y0", 0.0)), 3),
        round(float(getattr(word, "x1", 0.0)), 3),
        round(float(getattr(word, "y1", 0.0)), 3),
    )


def _exclude_words_by_identity(
    words: list[_Word],
    consumed_words: list[Any],
) -> list[_Word]:
    if not words or not consumed_words:
        return list(words)

    consumed = {_word_identity(word) for word in consumed_words}
    return [word for word in words if _word_identity(word) not in consumed]


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
    current_context: dict[str, Any] | None = None,
    page_drawings: list[dict[str, Any]] | None = None,
    layout_profile: dict[str, Any] | None = None,
    assessment: ContinuationAssessment | None = None,
) -> bool:
    """Guard supplemental word-clustering candidates with config-driven rules."""
    policy = get_pdf_parser_settings().table_detection_policy
    if raw_evidence.physical_row_count < policy.min_rows_for_supplemental_candidate:
        return False
    if raw_evidence.physical_col_count < policy.min_cols_for_supplemental_candidate:
        return False
    if _looks_like_two_column_narrative_false_positive(
        raw_evidence=raw_evidence,
        words=words,
        page_width=page_width,
        current_context=current_context,
        page_drawings=page_drawings,
        layout_profile=layout_profile,
        assessment=assessment,
    ):
        return False

    score = _tabular_strength_score(raw_evidence)
    layout_mode = str((layout_profile or {}).get("mode", "") or "")
    is_two_column_page = layout_mode in {"two_column", "mixed"} or is_two_column_layout(words, page_width)
    if is_two_column_page:
        if not policy.enable_two_column_guard:
            return score >= 0.55
        return score >= policy.two_column_min_tabular_score

    return score >= 0.45


def _word_group_bbox(words: list[Any]) -> tuple[float, float, float, float] | None:
    if not words:
        return None
    return (
        min(float(getattr(word, "x0", 0.0)) for word in words),
        min(float(getattr(word, "y0", 0.0)) for word in words),
        max(float(getattr(word, "x1", 0.0)) for word in words),
        max(float(getattr(word, "y1", 0.0)) for word in words),
    )


def _candidate_has_two_column_gutter_separation(
    raw_evidence: RawTableEvidence,
    layout_profile: dict[str, Any] | None,
    page_width: float,
    page_words: list[_Word] | None = None,
) -> bool:
    split = _resolve_candidate_two_column_word_split(
        raw_evidence=raw_evidence,
        layout_profile=layout_profile,
        page_width=page_width,
        page_words=page_words,
    )
    return split is not None


def _resolve_candidate_two_column_word_split(
    raw_evidence: RawTableEvidence,
    layout_profile: dict[str, Any] | None,
    page_width: float,
    page_words: list[_Word] | None = None,
) -> dict[str, Any] | None:
    if raw_evidence.source != "word_clustering":
        return None

    bbox = tuple(float(value) for value in raw_evidence.bbox)
    width_ratio = (bbox[2] - bbox[0]) / max(1.0, page_width)
    if width_ratio < 0.75:
        return None

    candidate_words = list(raw_evidence.words or [])
    if len(candidate_words) < 20:
        return None

    layout_mode = str((layout_profile or {}).get("mode", "") or "")
    page_has_two_column_signal = layout_mode in {"two_column", "mixed"}
    if not page_has_two_column_signal and page_words:
        page_has_two_column_signal = is_two_column_layout(page_words, page_width)
    if not page_has_two_column_signal:
        return None

    column_mid = float((layout_profile or {}).get("column_mid", page_width / 2.0) or (page_width / 2.0))
    lane_tolerance = float(
        (layout_profile or {}).get("lane_tolerance", max(12.0, page_width * 0.04))
        or max(12.0, page_width * 0.04)
    )
    gutter_guard = max(8.0, min(18.0, lane_tolerance * 0.6))

    left_words = [word for word in candidate_words if float(getattr(word, "x_center", 0.0)) <= column_mid - gutter_guard]
    right_words = [word for word in candidate_words if float(getattr(word, "x_center", 0.0)) >= column_mid + gutter_guard]
    gutter_words = [
        word
        for word in candidate_words
        if column_mid - gutter_guard < float(getattr(word, "x_center", 0.0)) < column_mid + gutter_guard
    ]
    unresolved_gutter_words: list[Any] = []
    for word in gutter_words:
        word_x0 = float(getattr(word, "x0", 0.0))
        word_x1 = float(getattr(word, "x1", 0.0))
        # Keep narrow edge tokens that sit wholly on one side of the true
        # column midline inside that lane instead of discarding them as gutter.
        # This protects compact first columns such as page-7 Table 5 in the
        # two-column literature regression sample.
        if word_x1 <= column_mid + 1.0:
            left_words.append(word)
            continue
        if word_x0 >= column_mid - 1.0:
            right_words.append(word)
            continue
        unresolved_gutter_words.append(word)
    gutter_words = unresolved_gutter_words
    if len(left_words) < 8 or len(right_words) < 8:
        return None

    left_bbox = _word_group_bbox(left_words)
    right_bbox = _word_group_bbox(right_words)
    if left_bbox is None or right_bbox is None:
        return None
    left_width = float(left_bbox[2]) - float(left_bbox[0])
    right_width = float(right_bbox[2]) - float(right_bbox[0])
    if left_width < page_width * 0.18 or right_width < page_width * 0.18:
        return None

    gutter_ratio = len(gutter_words) / max(1, len(candidate_words))
    if gutter_ratio > 0.08:
        return None

    return {
        "column_mid": column_mid,
        "gutter_guard": gutter_guard,
        "left_words": left_words,
        "right_words": right_words,
        "left_bbox": left_bbox,
        "right_bbox": right_bbox,
    }


def _candidate_table_title_mention_count(raw_evidence: RawTableEvidence) -> int:
    title_identities: set[str] = set()
    for row in raw_evidence.raw_data or []:
        for cell in row:
            text = str(cell or "").strip()
            if not text or not _looks_like_table_title(text):
                continue
            title_identities.add(_normalize_title_identity(text) or _compact_context_text(text))
    return len(title_identities)


def _looks_like_composite_two_column_table_candidate(
    raw_evidence: RawTableEvidence,
    current_context: dict[str, Any] | None,
    layout_profile: dict[str, Any] | None,
    page_width: float,
    page_words: list[_Word] | None = None,
) -> bool:
    split = _resolve_candidate_two_column_word_split(
        raw_evidence=raw_evidence,
        layout_profile=layout_profile,
        page_width=page_width,
        page_words=page_words,
    )
    if split is None:
        return False

    narrative_ratio = _narrative_cell_ratio(raw_evidence)
    title_mentions = _candidate_table_title_mention_count(raw_evidence)
    title_text = str((current_context or {}).get("title_text", "") or "").strip()

    if _looks_like_lane_local_titled_table_with_cross_column_pollution(
        raw_evidence=raw_evidence,
        current_context=current_context,
        split=split,
    ):
        return True
    if title_mentions >= 2:
        return True
    if raw_evidence.physical_row_count >= 6 and narrative_ratio >= 0.45:
        return True
    if title_mentions >= 1 and narrative_ratio >= 0.20:
        return True
    if title_text and raw_evidence.physical_col_count >= 5 and narrative_ratio >= 0.15:
        return True
    return False


def _split_two_column_composite_candidate(
    raw_evidence: RawTableEvidence,
    *,
    current_context: dict[str, Any] | None,
    text_blocks: list[dict[str, Any]] | None = None,
    layout_profile: dict[str, Any] | None,
    page_width: float,
    page_words: list[_Word] | None = None,
) -> list[RawTableEvidence]:
    split = _resolve_candidate_two_column_word_split(
        raw_evidence=raw_evidence,
        layout_profile=layout_profile,
        page_width=page_width,
        page_words=page_words,
    )
    if split is None:
        return []

    should_split = _looks_like_composite_two_column_table_candidate(
        raw_evidence,
        current_context,
        layout_profile,
        page_width,
        page_words,
    )

    lane_payloads: list[dict[str, Any]] = []
    for lane_index, lane_words in enumerate((split["left_words"], split["right_words"])):
        lane_bbox = split["left_bbox"] if lane_index == 0 else split["right_bbox"]
        payload: dict[str, Any] = {
            "lane_index": lane_index,
            "lane_words": lane_words,
            "lane_bbox": lane_bbox,
            "other_bbox": split["right_bbox"] if lane_index == 0 else split["left_bbox"],
            "evidence": None,
            "context": None,
        }
        if len(lane_words) >= 8:
            lane_evidence = extract_raw_evidence_from_words(
                words=lane_words,
                page_number=raw_evidence.page_number,
                page_height=raw_evidence.page_height,
                page_width=raw_evidence.page_width,
            )
            payload["evidence"] = lane_evidence
            if lane_evidence is not None and text_blocks:
                payload["context"] = _build_current_table_context(
                    raw_evidence=lane_evidence,
                    text_blocks=text_blocks,
                )
        lane_payloads.append(payload)

    if not should_split and text_blocks:
        structured_payloads = [
            payload for payload in lane_payloads if payload.get("evidence") is not None
        ]
        if len(structured_payloads) == 1:
            surviving_payload = structured_payloads[0]
            surviving_evidence = surviving_payload["evidence"]
            surviving_context = surviving_payload.get("context") or {}
            surviving_title_text = str(surviving_context.get("title_text", "") or "").strip()
            surviving_title_block = surviving_context.get("title_block")
            if (
                surviving_evidence is not None
                and _looks_like_table_title(surviving_title_text)
                and surviving_title_block is not None
            ):
                title_bbox_raw = surviving_title_block.get("bbox")
                if isinstance(title_bbox_raw, (list, tuple)) and len(title_bbox_raw) == 4:
                    title_bbox = tuple(float(value) for value in title_bbox_raw)
                    title_width_ratio = (
                        (title_bbox[2] - title_bbox[0]) / max(1.0, raw_evidence.page_width)
                    )
                    surviving_title_overlap = _horizontal_overlap_ratio(
                        title_bbox,
                        surviving_payload["lane_bbox"],
                    )
                    other_title_overlap = _horizontal_overlap_ratio(
                        title_bbox,
                        surviving_payload["other_bbox"],
                    )
                    min_surviving_rows = max(5, int(round(float(raw_evidence.physical_row_count) * 0.55)))
                    if (
                        title_width_ratio <= 0.58
                        and surviving_title_overlap >= 0.65
                        and other_title_overlap <= 0.12
                        and surviving_evidence.physical_col_count >= 4
                        and surviving_evidence.physical_row_count >= min_surviving_rows
                    ):
                        should_split = True

    if not should_split:
        return []

    split_candidates: list[RawTableEvidence] = []
    for payload in lane_payloads:
        lane_words = payload["lane_words"]
        split_evidence = payload.get("evidence")
        if len(lane_words) < 8 or split_evidence is None:
            continue
        split_candidates.append(split_evidence)
    return split_candidates


def _looks_like_lane_local_titled_table_with_cross_column_pollution(
    raw_evidence: RawTableEvidence,
    current_context: dict[str, Any] | None,
    split: dict[str, Any],
) -> bool:
    title_block = (current_context or {}).get("title_block")
    if not title_block:
        return False
    if raw_evidence.physical_row_count < 3 or raw_evidence.physical_col_count < 5:
        return False

    title_bbox_raw = title_block.get("bbox")
    if not isinstance(title_bbox_raw, (list, tuple)) or len(title_bbox_raw) != 4:
        return False
    title_bbox = tuple(float(value) for value in title_bbox_raw)
    title_width_ratio = (title_bbox[2] - title_bbox[0]) / max(1.0, raw_evidence.page_width)
    if title_width_ratio > 0.58:
        return False

    left_bbox = split["left_bbox"]
    right_bbox = split["right_bbox"]
    left_title_overlap = _horizontal_overlap_ratio(title_bbox, left_bbox)
    right_title_overlap = _horizontal_overlap_ratio(title_bbox, right_bbox)
    if max(left_title_overlap, right_title_overlap) < 0.65:
        return False
    if min(left_title_overlap, right_title_overlap) > 0.12:
        return False

    lane_structures: list[RawTableEvidence | None] = []
    for lane_words in (split["left_words"], split["right_words"]):
        if len(lane_words) < 8:
            lane_structures.append(None)
            continue
        lane_structures.append(
            extract_raw_evidence_from_words(
                words=lane_words,
                page_number=raw_evidence.page_number,
                page_height=raw_evidence.page_height,
                page_width=raw_evidence.page_width,
            )
        )

    structured_lane_count = sum(1 for evidence in lane_structures if evidence is not None)
    if structured_lane_count != 1:
        return False

    surviving_lane = next(evidence for evidence in lane_structures if evidence is not None)
    if surviving_lane.physical_row_count < max(3, raw_evidence.physical_row_count - 1):
        return False
    return surviving_lane.physical_col_count >= max(4, raw_evidence.physical_col_count - 1)


def _expand_candidate_with_nearby_title_words(
    raw_evidence: RawTableEvidence,
    *,
    page_words: list[_Word],
    current_context: dict[str, Any] | None,
) -> RawTableEvidence:
    if raw_evidence.source != "word_clustering":
        return raw_evidence
    if not page_words or not current_context:
        return raw_evidence

    title_block = current_context.get("title_block")
    if not title_block:
        return raw_evidence

    title_bbox_raw = title_block.get("bbox")
    if not isinstance(title_bbox_raw, (list, tuple)) or len(title_bbox_raw) != 4:
        return raw_evidence
    title_bbox = tuple(float(value) for value in title_bbox_raw)
    candidate_bbox = tuple(float(value) for value in raw_evidence.bbox)
    title_gap = candidate_bbox[1] - title_bbox[3]
    if title_gap < 0 or title_gap > 42:
        return raw_evidence

    current_title_text = str((current_context or {}).get("title_text", "") or "").strip()
    first_row_text = " ".join(
        str(cell or "").strip()
        for cell in (raw_evidence.raw_data[0] if raw_evidence.raw_data else [])
        if str(cell or "").strip()
    ).strip()
    if _looks_like_table_title(first_row_text) and (
        not current_title_text
        or _normalize_title_identity(first_row_text) == _normalize_title_identity(current_title_text)
    ):
        return raw_evidence

    expand_x0 = min(candidate_bbox[0], title_bbox[0]) - 6.0
    expand_y0 = min(candidate_bbox[1], title_bbox[1]) - 2.0
    expand_x1 = max(candidate_bbox[2], title_bbox[2]) + 6.0
    expand_y1 = candidate_bbox[3] + 1.5
    expanded_words = [
        word
        for word in page_words
        if expand_x0 <= float(getattr(word, "x0", 0.0)) <= expand_x1
        and float(getattr(word, "x1", 0.0)) <= expand_x1 + 1.0
        and expand_y0 <= float(getattr(word, "y0", 0.0))
        and float(getattr(word, "y1", 0.0)) <= expand_y1
    ]
    if len(expanded_words) <= len(raw_evidence.words or []):
        return raw_evidence

    expanded_evidence = extract_raw_evidence_from_words(
        words=expanded_words,
        page_number=raw_evidence.page_number,
        page_height=raw_evidence.page_height,
        page_width=raw_evidence.page_width,
    )
    if expanded_evidence is None:
        return raw_evidence
    if float(expanded_evidence.bbox[1]) > float(raw_evidence.bbox[1]) - 6.0:
        return raw_evidence
    return expanded_evidence


def _candidate_has_explicit_table_title_support(
    raw_evidence: RawTableEvidence,
    current_context: dict[str, Any] | None,
) -> bool:
    title_mentions = _candidate_table_title_mention_count(raw_evidence)
    if title_mentions >= 1:
        return True

    title_candidates = [
        str((current_context or {}).get("title_text", "") or "").strip(),
        str((current_context or {}).get("preceding_text", "") or "").strip(),
    ]
    return any(
        candidate
        and _looks_like_table_title(candidate)
        and not TOC_TITLE_PATTERN.search(candidate)
        for candidate in title_candidates
    )


def _normalize_candidate_grid_text(text: str | None) -> str:
    return " ".join(str(text or "").split()).strip()


def _looks_like_same_column_paragraph_continuation(
    previous_text: str,
    current_text: str,
) -> bool:
    previous = _normalize_candidate_grid_text(previous_text)
    current = _normalize_candidate_grid_text(current_text)
    if not previous or not current:
        return False

    previous_tail = previous[-1]
    current_lower = current.lower()
    if previous.endswith("-") and re.match(r"^[a-z0-9\[]", current_lower):
        return True
    if previous_tail in {'"', "“", "(", "["}:
        return True
    if previous_tail not in ".!?;:" and re.match(r"^[a-z\[]", current_lower):
        return True
    return bool(
        previous_tail not in ".!?;:"
        and re.match(
            (
                r"^(and|or|of|to|the|in|for|with|where|which|that|is|are|was|were|"
                r"be|been|being|from|on|at|by|as|it|this|these|those|their|our|its|"
                r"such|moreover|therefore|meanwhile|subsequently|resulting|regardless|"
                r"researchers|character|entity|subject|object|row|column|because|when|"
                r"while|then)\b"
            ),
            current_lower,
        )
    )


def _prose_continuation_metrics(raw_evidence: RawTableEvidence) -> dict[str, float]:
    raw_rows = list(raw_evidence.raw_data or [])
    if not raw_rows:
        return {
            "filled_cells": 0.0,
            "long_cells": 0.0,
            "one_cell_rows": 0.0,
            "multi_cell_rows": 0.0,
            "continuation_pairs": 0.0,
            "comparable_pairs": 0.0,
            "continuation_ratio": 0.0,
        }

    filled_cells = 0
    long_cells = 0
    one_cell_rows = 0
    multi_cell_rows = 0
    max_cols = max((len(row) for row in raw_rows), default=0)

    for row in raw_rows:
        filled_row = [cell for cell in row if _normalize_candidate_grid_text(cell)]
        filled_count = len(filled_row)
        filled_cells += filled_count
        if filled_count == 1:
            one_cell_rows += 1
        if filled_count >= 2:
            multi_cell_rows += 1
        for cell in filled_row:
            normalized = _normalize_candidate_grid_text(cell)
            if len(normalized) >= 28:
                long_cells += 1

    comparable_pairs = 0
    continuation_pairs = 0
    for row_index in range(len(raw_rows) - 1):
        previous_row = raw_rows[row_index]
        current_row = raw_rows[row_index + 1]
        for col_index in range(max_cols):
            previous_text = _normalize_candidate_grid_text(
                previous_row[col_index] if col_index < len(previous_row) else ""
            )
            current_text = _normalize_candidate_grid_text(
                current_row[col_index] if col_index < len(current_row) else ""
            )
            if not previous_text or not current_text:
                continue
            comparable_pairs += 1
            if _looks_like_same_column_paragraph_continuation(previous_text, current_text):
                continuation_pairs += 1

    return {
        "filled_cells": float(filled_cells),
        "long_cells": float(long_cells),
        "one_cell_rows": float(one_cell_rows),
        "multi_cell_rows": float(multi_cell_rows),
        "continuation_pairs": float(continuation_pairs),
        "comparable_pairs": float(comparable_pairs),
        "continuation_ratio": (
            float(continuation_pairs) / float(comparable_pairs)
            if comparable_pairs
            else 0.0
        ),
    }


def _looks_like_multiline_paragraph_grid_false_positive(
    raw_evidence: RawTableEvidence,
    current_context: dict[str, Any] | None,
    width_ratio: float,
    narrative_ratio: float,
    assessment: ContinuationAssessment | None,
) -> bool:
    if raw_evidence.source != "word_clustering":
        return False
    if _candidate_has_explicit_table_title_support(raw_evidence, current_context):
        return False
    if assessment and assessment.is_continuation and assessment.overall_confidence >= 0.75:
        return False

    local_signal = str((current_context or {}).get("local_signal", "") or "")
    metrics = _prose_continuation_metrics(raw_evidence)
    comparable_pairs = int(metrics["comparable_pairs"])
    if comparable_pairs < 4:
        return False

    continuation_ratio = float(metrics["continuation_ratio"])
    long_cells = int(metrics["long_cells"])
    one_cell_rows = int(metrics["one_cell_rows"])
    multi_cell_rows = int(metrics["multi_cell_rows"])
    row_count = int(raw_evidence.physical_row_count or 0)

    if continuation_ratio >= 0.72 and narrative_ratio >= 0.45 and long_cells >= 4:
        return True

    if (
        local_signal == "narrative_barrier"
        and width_ratio <= 0.62
        and continuation_ratio >= 0.75
        and long_cells >= 4
        and one_cell_rows >= 2
    ):
        return True

    if (
        local_signal == "narrative_barrier"
        and continuation_ratio >= 0.58
        and narrative_ratio >= 0.50
        and long_cells >= 4
        and multi_cell_rows >= max(3, row_count - 2)
    ):
        return True

    return (
        width_ratio >= 0.72
        and continuation_ratio >= 0.50
        and narrative_ratio >= 0.55
        and long_cells >= 6
        and multi_cell_rows >= max(3, row_count - 1)
    )


def _looks_like_two_column_narrative_false_positive(
    raw_evidence: RawTableEvidence,
    words: list[_Word],
    page_width: float,
    current_context: dict[str, Any] | None,
    page_drawings: list[dict[str, Any]] | None,
    layout_profile: dict[str, Any] | None,
    assessment: ContinuationAssessment | None,
) -> bool:
    if raw_evidence.source != "word_clustering":
        return False

    layout_mode = str((layout_profile or {}).get("mode", "") or "")
    two_column_page = layout_mode in {"two_column", "mixed"} or is_two_column_layout(words, page_width)
    if not two_column_page:
        return False

    bbox = tuple(float(value) for value in raw_evidence.bbox)
    width_ratio = (bbox[2] - bbox[0]) / max(1.0, page_width)

    title_text = str((current_context or {}).get("title_text", "") or "").strip()
    preceding_text = str((current_context or {}).get("preceding_text", "") or "").strip()

    center_y = (bbox[1] + bbox[3]) / 2.0
    body_top = float((layout_profile or {}).get("body_top", 0.0) or 0.0)
    body_bottom = float((layout_profile or {}).get("body_bottom", raw_evidence.page_height) or raw_evidence.page_height)
    in_body_zone = body_bottom <= body_top or (body_top <= center_y <= body_bottom)
    if not in_body_zone:
        return False

    wide_horizontal_lines = _count_wide_horizontal_rules_in_bbox(
        bbox=bbox,
        page_drawings=page_drawings or [],
        page_width=page_width,
    )
    if wide_horizontal_lines >= 2:
        return False

    narrative_ratio = _narrative_cell_ratio(raw_evidence)
    first_row_text = " ".join(
        str(cell or "").strip()
        for cell in (raw_evidence.raw_data[0] if raw_evidence.raw_data else [])
        if str(cell or "").strip()
    ).strip()
    second_row_text = " ".join(
        str(cell or "").strip()
        for cell in (raw_evidence.raw_data[1] if len(raw_evidence.raw_data or []) > 1 else [])
        if str(cell or "").strip()
    ).strip()
    has_sentence_like_opening = sum(
        1
        for candidate in (first_row_text, second_row_text)
        if candidate
        and len(candidate.split()) >= 6
        and re.search(r"[a-z]{4,}", candidate)
        and ("," in candidate or ";" in candidate or len(candidate) >= 48)
        and not _looks_like_table_title(candidate)
    ) >= 1

    if _looks_like_multiline_paragraph_grid_false_positive(
        raw_evidence=raw_evidence,
        current_context=current_context,
        width_ratio=width_ratio,
        narrative_ratio=narrative_ratio,
        assessment=assessment,
    ):
        return True

    if width_ratio >= 0.75:
        if _looks_like_composite_two_column_table_candidate(
            raw_evidence,
            current_context,
            layout_profile,
            page_width,
            words,
        ):
            return True
        if title_text or _looks_like_table_title(preceding_text):
            return False
        if assessment and assessment.is_continuation and narrative_ratio < 0.55:
            return False
        return raw_evidence.physical_row_count >= 8 and narrative_ratio >= 0.55

    if title_text or _looks_like_table_title(preceding_text):
        return False
    if assessment and assessment.is_continuation and narrative_ratio < 0.55:
        return False
    return (
        width_ratio <= 0.62
        and raw_evidence.physical_row_count >= 6
        and raw_evidence.physical_col_count <= 3
        and narrative_ratio >= 0.45
        and has_sentence_like_opening
    )


def _supplemental_rejection_peel_words(
    raw_evidence: RawTableEvidence,
) -> list[Any]:
    candidate_words = list(raw_evidence.words or [])
    raw_rows = list(raw_evidence.rows or [])
    if not candidate_words or not raw_rows:
        return candidate_words

    row_count = len(raw_rows)
    peel_rows = 1
    if row_count >= 24:
        peel_rows = 8
    elif row_count >= 16:
        peel_rows = 6
    elif row_count >= 10:
        peel_rows = 4
    elif row_count >= 6:
        peel_rows = 2

    peelable_rows = [row for row in raw_rows[:peel_rows] if row.bbox]
    if not peelable_rows:
        return candidate_words

    peel_y1 = max(float(row.bbox[3]) for row in peelable_rows)
    peeled_words = [
        word
        for word in candidate_words
        if float(getattr(word, "y1", 0.0)) <= peel_y1 + 1.5
    ]
    return peeled_words or candidate_words


def _count_wide_horizontal_rules_in_bbox(
    bbox: tuple[float, float, float, float],
    page_drawings: list[dict[str, Any]],
    page_width: float,
) -> int:
    count = 0
    for draw in page_drawings:
        rect = draw.get("rect")
        if not rect:
            continue
        draw_bbox = tuple(float(value) for value in rect)
        width = draw_bbox[2] - draw_bbox[0]
        height = draw_bbox[3] - draw_bbox[1]
        if width < page_width * 0.35 or height > 3.0:
            continue
        if _bbox_overlap_ratio(draw_bbox, bbox) < 0.65:
            continue
        count += 1
    return count


def _narrative_cell_ratio(raw_evidence: RawTableEvidence) -> float:
    raw_data = raw_evidence.raw_data or []
    filled_cells: list[str] = []
    narrative_like = 0
    for row in raw_data:
        for cell in row:
            text = str(cell or "").strip()
            if not text:
                continue
            filled_cells.append(text)
            token_count = len(text.split())
            if len(text) >= 36 and (token_count >= 6 or any(mark in text for mark in ".;,!?")):
                narrative_like += 1
    if not filled_cells:
        return 0.0
    return narrative_like / len(filled_cells)


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
        header = None
        if table.get("header"):
            header = TableHeader(
                cells=table.get("header", []),
                row_index=0,
                inherited=table.get("header_inherited", False),
            )
        instance = TableInstance(
            table_id=table.get("table_id", ""),
            page_number=table.get("page", 0),
            bbox=tuple(table.get("bbox", (0, 0, 0, 0))),
            col_count=table.get("col_count", 0),
            row_count=table.get("display_row_count", table.get("row_count", 0)),
            header=header,
            cells=table.get("cells", []),
            grid=table.get("display_grid", table.get("grid", [])),
            raw_grid=table.get("raw_grid", table.get("grid", [])),
            data_grid=table.get("data_grid", table.get("grid", [])),
            row_texts=table.get("display_row_texts", table.get("row_texts", [])),
            raw_row_texts=table.get("raw_row_texts", table.get("row_texts", [])),
            data_row_texts=table.get("data_row_texts", table.get("row_texts", [])),
            raw_row_count=table.get("raw_row_count", table.get("row_count", 0)),
            data_row_count=table.get("data_row_count", table.get("row_count", 0)),
            structural_empty_rows=table.get("structural_empty_rows", []),
            physical_col_count=table.get("physical_col_count", table.get("col_count", 0)),
            physical_row_count=table.get("physical_row_count", table.get("row_count", 0)),
            near_page_top=table.get("near_page_top", False),
            near_page_bottom=table.get("near_page_bottom", False),
            is_continuation=table.get("is_continuation", False),
            column_signature=table.get("column_signature", []),
            column_hash=table.get("column_hash", ""),
            detection_method=table.get("detection_method", "pymupdf_builtin"),
            title=table.get("title"),
            section_hint=table.get("section_hint"),
            toc_context=table.get("toc_context", False),
            grid_line_score=table.get("grid_line_score", 0.0),
            title_row_index=table.get("title_row_index"),
            header_row_index=table.get("header_row_index"),
            data_start_row=table.get("data_start_row", 0),
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
    "split_internal_table_segments",
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
