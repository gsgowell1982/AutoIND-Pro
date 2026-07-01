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
    extract_caption_anchored_horizontal_rule_tables,
    extract_visual_structure_grid_tables,
    extract_text_aligned_borderless_grid_tables,
    extract_structured_text_region_tables,
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
    project_simple_schema_header_data_views,
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
from .ocr_policy import PageOcrContext, should_scan_embedded_image_table_regions
from .table_vector_ocr import extract_embedded_image_ocr_table_candidates, extract_vector_ocr_table_candidates


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
CTD_NUMBERED_SECTION_HEADING_PATTERN = re.compile(r"^\s*\d+(?:\.\d+){1,6}\s+\S+")
STUDY_METADATA_BOUNDARY_PATTERN = re.compile(
    r"(试验编号|研究系统|靶向实体、试验系统和方法|种属/品系|物种|给药方法|溶媒/剂型|"
    r"放射性核素|分析物/分析方法|采样时间|CTD\s*中?的位置|GLP\s*依从性|供试品)\s*[:：]",
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
    "\uff08\u7eed",
    "(\u7eed",
    "\u7eed\uff09",
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
    image_blocks: list[dict[str, Any]] | None = None,
    layout_profile: dict[str, Any] | None = None,
    prev_tables: list[dict[str, Any]] | None = None,
    table_counter: int = 0,
    out_stats: dict[str, int] | None = None,
    out_toc_blocks: list[dict[str, Any]] | None = None,
    ocr_context: PageOcrContext | None = None,
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
            if table_ast.get("semantic_role") == "toc_outline":
                table_counter += 1
                toc_outline_count += 1
                if out_toc_blocks is not None:
                    out_toc_blocks.append(_build_toc_block(table_ast))
            else:
                ownership = _arbitrate_table_candidate_ownership(raw_evidence, table_ast, current_context)
                if ownership.get("primary_type") == "data_table":
                    table_counter += 1
                    tables.append(table_ast)
                else:
                    rejected_count += 1
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

    # Strategy 2b: Embedded raster-image tables without text-layer table objects
    # or explicit captions. Discovery is image-region based, but candidates
    # still flow through the shared AST and ownership arbitration path.
    occupied_bboxes = _collect_item_bboxes(tables + (out_toc_blocks or []))
    image_ocr_decision = should_scan_embedded_image_table_regions(ocr_context)
    if image_ocr_decision.enabled:
        image_ocr_candidates = extract_embedded_image_ocr_table_candidates(
            page=page,
            page_number=page_number,
            page_height=page_height,
            page_width=page_width,
            image_blocks=image_blocks or [],
            page_words=page_words,
            occupied_bboxes=occupied_bboxes,
            ocr_context=ocr_context,
        )
    else:
        image_ocr_candidates = []
        if out_stats is not None and image_blocks:
            out_stats["embedded_image_table_ocr_skipped"] = out_stats.get("embedded_image_table_ocr_skipped", 0) + 1
    for raw_evidence_image in image_ocr_candidates:
        raw_candidate_count += 1
        current_context = _build_current_table_context(
            raw_evidence=raw_evidence_image,
            text_blocks=text_blocks,
        )
        table_ast = _process_raw_evidence(
            raw_evidence=raw_evidence_image,
            page=page,
            page_number=page_number,
            page_height=page_height,
            text_blocks=text_blocks,
            page_drawings=page_drawings,
            prev_tables=prev_tables,
            table_counter=table_counter,
            parent_col_count=None,
            parent_header=None,
            current_context=current_context,
            words=page_words,
        )
        ownership = _arbitrate_table_candidate_ownership(raw_evidence_image, table_ast, current_context)
        if table_ast and table_ast.get("semantic_role") == "toc_outline":
            rejected_count += 1
            continue
        if table_ast and ownership.get("primary_type") != "data_table":
            rejected_count += 1
            continue
        if table_ast:
            replaced_tables, replaced_fragment = _replace_weaker_overlapping_table_fragments(tables, table_ast)
            if replaced_fragment:
                tables = replaced_tables
                occupied_bboxes = _collect_item_bboxes(tables + (out_toc_blocks or []))
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

    # Strategy 3: Caption-anchored horizontal-rule candidates. This covers
    # booktabs/three-line tables where PyMuPDF exposes the rules as drawings but
    # does not create a table object and the text-only fallback lacks boundary
    # evidence. It runs before the text-aligned fallback because explicit
    # caption/rule evidence is stronger than pure word-alignment evidence.
    occupied_bboxes = _collect_item_bboxes(tables + (out_toc_blocks or []))
    rule_candidates = extract_caption_anchored_horizontal_rule_tables(
        words=_exclude_words_in_occupied_regions(page_words, occupied_bboxes),
        drawings=page_drawings or [],
        page_number=page_number,
        page_height=page_height,
        page_width=page_width,
        occupied_bboxes=occupied_bboxes,
    )
    for raw_evidence_rule in rule_candidates:
        raw_candidate_count += 1
        current_context = _build_current_table_context(
            raw_evidence=raw_evidence_rule,
            text_blocks=text_blocks,
        )
        candidates = _resolve_parent_context(
            raw_evidence=raw_evidence_rule,
            page_number=page_number,
            prev_tables=prev_tables,
            current_context=current_context,
        )
        assessment = _assess_continuation_candidates_v2(
            candidates=candidates,
            raw_evidence=raw_evidence_rule,
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
            raw_evidence=raw_evidence_rule,
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

    # Strategy 4: Text-aligned borderless grid candidates. These are pages with
    # no explicit table object/rules, but stable text-layer column anchors.
    occupied_bboxes = _collect_occupied_bboxes_for_late_table_reconstruction(
        tables=tables,
        out_toc_blocks=out_toc_blocks or [],
    )
    aligned_candidates = extract_text_aligned_borderless_grid_tables(
        words=page_words,
        page_number=page_number,
        page_height=page_height,
        page_width=page_width,
        occupied_bboxes=occupied_bboxes,
    )
    for raw_evidence_aligned in aligned_candidates:
        if _text_aligned_candidate_should_defer_to_word_clustering(raw_evidence_aligned):
            rejected_count += 1
            continue
        raw_candidate_count += 1
        current_context = _build_current_table_context(
            raw_evidence=raw_evidence_aligned,
            text_blocks=text_blocks,
        )
        candidates = _resolve_parent_context(
            raw_evidence=raw_evidence_aligned,
            page_number=page_number,
            prev_tables=prev_tables,
            current_context=current_context,
        )
        assessment = _assess_continuation_candidates_v2(
            candidates=candidates,
            raw_evidence=raw_evidence_aligned,
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
            raw_evidence=raw_evidence_aligned,
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
        if table_ast:
            if _has_preferred_existing_owner_for_text_aligned_candidate(tables, table_ast):
                rejected_count += 1
                continue
            replaced_tables, replaced_fragment = _replace_weaker_overlapping_table_fragments(tables, table_ast)
            if replaced_fragment:
                tables = replaced_tables
                occupied_bboxes = _collect_item_bboxes(tables + (out_toc_blocks or []))
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

    # Strategy 4b: Low-column structured text regions. These are boxed or
    # rule-bounded one/two-column table-like regions that the wider text-aligned
    # lattice intentionally ignores.
    occupied_bboxes = _collect_occupied_bboxes_for_late_table_reconstruction(
        tables=tables,
        out_toc_blocks=out_toc_blocks or [],
    )
    structured_region_candidates = extract_structured_text_region_tables(
        words=_exclude_words_in_occupied_regions(page_words, occupied_bboxes),
        drawings=page_drawings or [],
        page_number=page_number,
        page_height=page_height,
        page_width=page_width,
        occupied_bboxes=occupied_bboxes,
    )
    for raw_evidence_structured in structured_region_candidates:
        raw_candidate_count += 1
        current_context = _build_current_table_context(
            raw_evidence=raw_evidence_structured,
            text_blocks=text_blocks,
        )
        table_ast = _process_raw_evidence(
            raw_evidence=raw_evidence_structured,
            page=page,
            page_number=page_number,
            page_height=page_height,
            text_blocks=text_blocks,
            page_drawings=page_drawings,
            prev_tables=prev_tables,
            table_counter=table_counter,
            parent_col_count=None,
            parent_header=None,
            current_context=current_context,
            words=page_words,
        )
        ownership = _arbitrate_table_candidate_ownership(raw_evidence_structured, table_ast, current_context)
        if table_ast and table_ast.get("semantic_role") == "toc_outline":
            rejected_count += 1
            continue
        if table_ast and ownership.get("primary_type") != "data_table":
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

    # Strategy 5: Caption-less visual structure grids. Some tables have visible
    # horizontal/vertical separators or shaded bands but no explicit "Table N"
    # caption. This runs after text-aligned recovery so it does not steal mature
    # IND/eCTD borderless table detections.
    occupied_bboxes = _collect_occupied_bboxes_for_visual_structure_reconstruction(
        tables=tables,
        out_toc_blocks=out_toc_blocks or [],
    )
    visual_grid_candidates = extract_visual_structure_grid_tables(
        words=page_words,
        drawings=page_drawings or [],
        page_number=page_number,
        page_height=page_height,
        page_width=page_width,
        occupied_bboxes=occupied_bboxes,
    )
    for raw_evidence_visual in visual_grid_candidates:
        raw_candidate_count += 1
        ownership = _arbitrate_visual_structure_candidate_ownership(raw_evidence_visual)
        if ownership["primary_type"] != "data_table":
            rejected_count += 1
            continue
        current_context = _build_current_table_context(
            raw_evidence=raw_evidence_visual,
            text_blocks=text_blocks,
        )
        candidates = _resolve_parent_context(
            raw_evidence=raw_evidence_visual,
            page_number=page_number,
            prev_tables=prev_tables,
            current_context=current_context,
        )
        assessment = _assess_continuation_candidates_v2(
            candidates=candidates,
            raw_evidence=raw_evidence_visual,
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
            raw_evidence=raw_evidence_visual,
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
        ownership = _arbitrate_table_candidate_ownership(raw_evidence_visual, table_ast, current_context)
        if table_ast and _looks_like_visual_structure_non_table_false_positive(table_ast):
            rejected_count += 1
            continue
        if table_ast and table_ast.get("semantic_role") == "toc_outline":
            rejected_count += 1
            continue
        if table_ast and ownership.get("primary_type") != "data_table":
            rejected_count += 1
            continue
        if table_ast:
            replaced_tables, replaced_fragment = _replace_weaker_overlapping_table_fragments(tables, table_ast)
            if replaced_fragment:
                tables = replaced_tables
                occupied_bboxes = _collect_item_bboxes(tables + (out_toc_blocks or []))
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

    # Strategy 6: Word-clustering supplemental candidate (borderless-table support)
    # Run on residual page words after excluding already accepted regions, then keep
    # iterating over remaining structured groups under the same unified path.
    occupied_bboxes = _collect_occupied_bboxes_for_late_table_reconstruction(
        tables=tables,
        out_toc_blocks=out_toc_blocks or [],
    )
    if len(occupied_bboxes) < len(_collect_item_bboxes(tables + (out_toc_blocks or []))) or _has_text_aligned_owner_for_word_cluster_challenge(tables):
        challenged_table, challenged_consumed_words = _build_supplemental_word_cluster_candidate(
            page_words=page_words,
            page=page,
            page_number=page_number,
            page_height=page_height,
            page_width=page_width,
            text_blocks=text_blocks,
            page_drawings=page_drawings,
            prev_tables=prev_tables,
            table_counter=table_counter,
            layout_profile=layout_profile,
        )
        if challenged_table:
            replaced_tables, replaced_fragment = _replace_weaker_overlapping_table_fragments(tables, challenged_table)
            if replaced_fragment:
                tables = replaced_tables
            if _is_distinct_from_existing_tables(
                challenged_table,
                tables,
                overlap_threshold=detection_policy.supplemental_dedup_overlap_threshold,
            ):
                table_counter += 1
                tables.append(challenged_table)
                occupied_bboxes = _collect_occupied_bboxes_for_late_table_reconstruction(
                    tables=tables,
                    out_toc_blocks=out_toc_blocks or [],
                )
            elif challenged_consumed_words:
                rejected_count += 1
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
                split_ownership = _arbitrate_table_candidate_ownership(split_evidence, split_table_ast, split_context)
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
                if split_table_ast and split_ownership.get("primary_type") != "data_table":
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
        ownership = _arbitrate_table_candidate_ownership(raw_evidence_fallback, table_ast, current_context)
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
        elif table_ast and ownership.get("primary_type") != "data_table":
            rejected_count += 1
        elif table_ast and accept_supplemental:
            replaced_tables, replaced_fragment = _replace_weaker_overlapping_table_fragments(tables, table_ast)
            if replaced_fragment:
                tables = replaced_tables
            if _is_distinct_from_existing_tables(
                table_ast,
                tables,
                overlap_threshold=detection_policy.supplemental_dedup_overlap_threshold,
            ):
                table_counter += 1
                tables.append(table_ast)
            else:
                rejected_count += 1
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

    if tables:
        _repair_pymupdf_stacked_header_body_gap_rows(tables, page_words)

    if tables:
        reviewed_tables, final_review_rejected = _finalize_page_table_ownership(tables)
        if final_review_rejected:
            tables = reviewed_tables
            rejected_count += final_review_rejected
            table_counter = max(0, table_counter - final_review_rejected)

    if out_stats is not None:
        out_stats["raw_candidates"] = out_stats.get("raw_candidates", 0) + raw_candidate_count
        out_stats["accepted"] = out_stats.get("accepted", 0) + len(tables)
        out_stats["toc_outlines"] = out_stats.get("toc_outlines", 0) + toc_outline_count
        out_stats["rejected"] = out_stats.get("rejected", 0) + rejected_count

    return tables, table_counter


def _repair_pymupdf_stacked_header_body_gap_rows(tables: list[dict[str, Any]], page_words: list[Any]) -> int:
    """Recover a data row that sits between PyMuPDF header/body fragments.

    PyMuPDF can split a ruled-looking table into a header fragment and a body
    fragment when the first body row has no detected cell borders. The row still
    exists in the page text layer. This repair only promotes a gap row when the
    two fragments share the same table band and the gap words form a compact,
    column-aligned data record.
    """

    if len(tables) < 2 or not page_words:
        return 0
    ordered = sorted(
        tables,
        key=lambda table: (
            float((table.get("bbox") or [0.0, 0.0, 0.0, 0.0])[1]),
            float((table.get("bbox") or [0.0, 0.0, 0.0, 0.0])[0]),
        ),
    )
    repaired = 0
    for index in range(len(ordered) - 1):
        top = ordered[index]
        bottom = ordered[index + 1]
        if _repair_single_pymupdf_stacked_gap_row(top, bottom, page_words):
            repaired += 1
    return repaired


def _repair_single_pymupdf_stacked_gap_row(top: dict[str, Any], bottom: dict[str, Any], page_words: list[Any]) -> bool:
    if str(top.get("detection_source") or top.get("detection_method") or "") != "pymupdf_builtin":
        return False
    if str(bottom.get("detection_source") or bottom.get("detection_method") or "") != "pymupdf_builtin":
        return False
    top_bbox = _table_bbox_tuple(top)
    bottom_bbox = _table_bbox_tuple(bottom)
    if not top_bbox or not bottom_bbox:
        return False
    if top_bbox[1] > bottom_bbox[1]:
        return False
    gap = bottom_bbox[1] - top_bbox[3]
    if gap < 3.0 or gap > 36.0:
        return False
    if _horizontal_overlap_ratio(top_bbox, bottom_bbox) < 0.78:
        return False

    top_grid = _table_grid_rows(top)
    bottom_grid = _table_grid_rows(bottom)
    if not top_grid or not bottom_grid or len(top_grid) > 4 or len(bottom_grid) < 2:
        return False
    col_count = max((len(row) for row in bottom_grid if isinstance(row, list)), default=0)
    if col_count < 3:
        return False

    gap_rows = _candidate_word_rows_between_table_fragments(page_words, top_bbox, bottom_bbox)
    for gap_row_words in gap_rows:
        projected_row, row_bbox = _project_gap_word_row_to_table_columns(gap_row_words, bottom_bbox, col_count)
        if not projected_row or not row_bbox:
            continue
        if not _looks_like_missing_stacked_table_data_row(projected_row, col_count):
            continue
        first_existing = _row_signature_for_gap_repair(bottom_grid[0])
        if first_existing and first_existing == _row_signature_for_gap_repair(projected_row):
            continue
        _prepend_gap_row_to_table_fragment(bottom, projected_row, row_bbox)
        return True
    return False


def _table_bbox_tuple(table: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = table.get("bbox") or []
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None


def _table_grid_rows(table: dict[str, Any]) -> list[list[Any]]:
    for key in ("display_grid", "raw_grid", "grid"):
        grid = table.get(key)
        if isinstance(grid, list) and grid:
            return [list(row) for row in grid if isinstance(row, list)]
    return []


def _candidate_word_rows_between_table_fragments(
    page_words: list[Any],
    top_bbox: tuple[float, float, float, float],
    bottom_bbox: tuple[float, float, float, float],
) -> list[list[tuple[float, float, float, float, str]]]:
    words: list[tuple[float, float, float, float, str]] = []
    x0 = min(top_bbox[0], bottom_bbox[0]) - 6.0
    x1 = max(top_bbox[2], bottom_bbox[2]) + 6.0
    y0 = top_bbox[3] - 2.0
    y1 = bottom_bbox[1] + 2.0
    for word in page_words:
        item = _coerce_page_word_tuple(word)
        if item is None:
            continue
        wx0, wy0, wx1, wy1, text = item
        if not text:
            continue
        center_y = (wy0 + wy1) / 2.0
        if wx1 < x0 or wx0 > x1 or center_y < y0 or center_y > y1:
            continue
        words.append(item)
    return _cluster_caption_words_by_line(words)


def _coerce_page_word_tuple(word: Any) -> tuple[float, float, float, float, str] | None:
    if isinstance(word, dict):
        bbox = word.get("bbox") or []
        text = str(word.get("text") or "").strip()
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), text)
            except (TypeError, ValueError):
                return None
    if isinstance(word, (list, tuple)) and len(word) >= 5:
        try:
            return (float(word[0]), float(word[1]), float(word[2]), float(word[3]), str(word[4] or "").strip())
        except (TypeError, ValueError):
            return None
    if all(hasattr(word, attr) for attr in ("x0", "y0", "x1", "y1")):
        try:
            return (
                float(getattr(word, "x0")),
                float(getattr(word, "y0")),
                float(getattr(word, "x1")),
                float(getattr(word, "y1")),
                str(getattr(word, "text", "") or "").strip(),
            )
        except (TypeError, ValueError):
            return None
    return None


def _project_gap_word_row_to_table_columns(
    row_words: list[tuple[float, float, float, float, str]],
    table_bbox: tuple[float, float, float, float],
    col_count: int,
) -> tuple[list[str | None] | None, tuple[float, float, float, float] | None]:
    if len(row_words) < max(3, min(col_count, 5)):
        return None, None
    ordered = sorted(row_words, key=lambda item: item[0])
    row_bbox = (
        min(item[0] for item in ordered),
        min(item[1] for item in ordered),
        max(item[2] for item in ordered),
        max(item[3] for item in ordered),
    )
    if len(ordered) == col_count:
        return [item[4] or None for item in ordered], row_bbox

    width = max(1.0, table_bbox[2] - table_bbox[0])
    projected: list[list[str]] = [[] for _ in range(col_count)]
    for wx0, _wy0, wx1, _wy1, text in ordered:
        center_x = (wx0 + wx1) / 2.0
        relative = min(0.999, max(0.0, (center_x - table_bbox[0]) / width))
        col_idx = min(col_count - 1, max(0, int(relative * col_count)))
        projected[col_idx].append(text)
    row = [" ".join(parts).strip() or None for parts in projected]
    if sum(1 for cell in row if cell) < max(3, min(col_count, 5)):
        return None, None
    return row, row_bbox


def _looks_like_missing_stacked_table_data_row(row: list[str | None], col_count: int) -> bool:
    values = [str(cell or "").strip() for cell in row[:col_count]]
    filled = [text for text in values if text]
    if len(filled) < max(3, min(col_count, 5)):
        return False
    if not re.search(r"[A-Za-z\u4e00-\u9fff]", filled[0]):
        return False
    value_like = sum(1 for text in filled[1:] if _looks_like_table_value_atom(text))
    return value_like >= max(2, min(len(filled) - 1, 4))


def _prepend_gap_row_to_table_fragment(
    table: dict[str, Any],
    row: list[str | None],
    row_bbox: tuple[float, float, float, float],
) -> None:
    for key in ("display_grid", "raw_grid", "grid"):
        grid = table.get(key)
        if isinstance(grid, list):
            table[key] = [list(row), *[list(item) for item in grid if isinstance(item, list)]]
    display_grid = _table_grid_rows(table)
    table["data_start_row"] = 0
    table["header"] = []
    table["data_grid"] = [list(item) for item in display_grid]
    table["row_count"] = len(display_grid)
    table["display_row_count"] = len(display_grid)
    table["raw_row_count"] = len(display_grid)
    table["data_row_count"] = len(display_grid)
    table["logical_row_count"] = len(display_grid)
    row_texts = [" | ".join(str(cell or "").strip() for cell in item) for item in display_grid]
    table["row_texts"] = list(row_texts)
    table["raw_row_texts"] = list(row_texts)
    table["display_row_texts"] = list(row_texts)
    table["data_row_texts"] = list(row_texts)
    bbox = _table_bbox_tuple(table)
    if bbox:
        table["bbox"] = [
            min(bbox[0], row_bbox[0]),
            min(bbox[1], row_bbox[1]),
            max(bbox[2], row_bbox[2]),
            max(bbox[3], row_bbox[3]),
        ]
    table["gap_row_repair"] = {
        "source": "pymupdf_stacked_header_body_gap_row",
        "inserted_row_count": 1,
    }


def _row_signature_for_gap_repair(row: list[Any]) -> tuple[str, ...]:
    return tuple(re.sub(r"\W+", "", str(cell or "").lower()) for cell in row if str(cell or "").strip())


def _finalize_page_table_ownership(tables: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Apply final ownership gates before page tables leave extraction."""
    reviewed: list[dict[str, Any]] = []
    rejected = 0
    ordered_tables = sorted(
        list(tables),
        key=lambda table: (
            float((table.get("bbox") or [0.0, 0.0, 0.0, 0.0])[1]),
            float((table.get("bbox") or [0.0, 0.0, 0.0, 0.0])[0]),
        ),
    )
    for index, table in enumerate(ordered_tables):
        previous_table = ordered_tables[index - 1] if index > 0 else None
        next_table = ordered_tables[index + 1] if index + 1 < len(ordered_tables) else None
        if _looks_like_visual_structure_non_table_false_positive(table) or _looks_like_visual_bridge_between_table_boundaries(
            table,
            previous_table=previous_table,
            next_table=next_table,
        ) or _looks_like_visual_intro_bridge_duplicate(
            table,
            previous_table=previous_table,
            next_table=next_table,
        ) or _looks_like_remote_caption_rule_fragment_after_bottom_caption_owner(
            table,
            previous_table=previous_table,
        ) or _looks_like_text_aligned_tail_bridge_after_bottom_caption_owner(
            table,
            previous_table=previous_table,
        ):
            rejected += 1
            continue
        reviewed.append(table)
    return reviewed, rejected


def _looks_like_text_aligned_tail_bridge_after_bottom_caption_owner(
    table: dict[str, Any],
    *,
    previous_table: dict[str, Any] | None,
) -> bool:
    if previous_table is None:
        return False
    source = str(table.get("detection_source") or table.get("detection_method") or "")
    if source != "text_aligned_borderless_grid":
        return False
    previous_title_block = previous_table.get("title_block") if isinstance(previous_table.get("title_block"), dict) else {}
    if str(previous_title_block.get("source") or "") != "bottom_caption_text_aligned_grid":
        return False
    title_block = table.get("title_block") if isinstance(table.get("title_block"), dict) else {}
    if str(title_block.get("source") or "") == "bottom_caption_text_aligned_grid":
        return False
    bbox_raw = table.get("bbox") or []
    prev_bbox_raw = previous_table.get("bbox") or []
    if len(bbox_raw) != 4 or len(prev_bbox_raw) != 4:
        return False
    bbox = tuple(float(value) for value in bbox_raw)
    prev_bbox = tuple(float(value) for value in prev_bbox_raw)
    vertical_overlap = max(0.0, min(bbox[3], prev_bbox[3]) - max(bbox[1], prev_bbox[1]))
    vertical_gap = bbox[1] - prev_bbox[3]
    if vertical_overlap <= 0.0 and vertical_gap > 10.0:
        return False
    if _horizontal_overlap_ratio(bbox, prev_bbox) < 0.35:
        return False
    grid = table.get("semantic_grid") or table.get("display_grid") or table.get("raw_grid") or []
    if not isinstance(grid, list) or len(grid) < 2:
        return False
    row_texts = [
        " ".join(str(cell or "").strip() for cell in row if str(cell or "").strip()).strip()
        for row in grid
        if isinstance(row, list)
    ]
    caption_rows = sum(1 for text in row_texts if _looks_like_table_title(text))
    prose_rows = sum(1 for text in row_texts if len(text.split()) >= 8 and re.search(r"[.。！？!?；;]?\s*$", text))
    return caption_rows >= 1 or prose_rows >= max(1, len(row_texts) - 1)


def _looks_like_remote_caption_rule_fragment_after_bottom_caption_owner(
    table: dict[str, Any],
    *,
    previous_table: dict[str, Any] | None,
) -> bool:
    if previous_table is None:
        return False
    if not _caption_anchored_rule_has_remote_upstream_caption_for_late_reconstruction(table):
        return False
    previous_title_block = previous_table.get("title_block") if isinstance(previous_table.get("title_block"), dict) else {}
    if str(previous_title_block.get("source") or "") != "bottom_caption_text_aligned_grid":
        return False
    current_title = _compact_context_text(str(table.get("title") or ""))
    previous_title = _compact_context_text(str(previous_table.get("title") or ""))
    if not current_title or not previous_title:
        return False
    if not (
        current_title in previous_title
        or previous_title in current_title
        or current_title.startswith(previous_title[: min(len(previous_title), 48)])
        or previous_title.startswith(current_title[: min(len(current_title), 48)])
    ):
        return False
    grid = table.get("semantic_grid") or table.get("display_grid") or table.get("raw_grid") or []
    col_count = max((len(row) for row in grid if isinstance(row, list)), default=0)
    return col_count <= 4


def _looks_like_visual_intro_bridge_duplicate(
    table: dict[str, Any],
    *,
    previous_table: dict[str, Any] | None,
    next_table: dict[str, Any] | None,
) -> bool:
    """Reject visual candidates that blend prose intro with a stronger table.

    The ownership decision is based on structure, not document identity: a
    captionless visual grid with leading sentence fragments and an overlapping
    later table is a bridge/duplicate candidate. The later table owns the
    tabular region; the prefix remains body text.
    """
    source = str(table.get("detection_source") or table.get("detection_method") or "")
    if source != "visual_structure_grid":
        return False
    if str(table.get("title") or table.get("caption_text") or "").strip():
        return False

    stronger_neighbor = _overlapping_stronger_table_neighbor(table, previous_table, next_table)
    if stronger_neighbor is None:
        return False

    grid = table.get("display_grid") or table.get("raw_grid") or table.get("grid") or []
    if not isinstance(grid, list) or len(grid) < 4:
        return False
    rows = [row for row in grid if isinstance(row, list)]
    if len(rows) < 4:
        return False

    neighbor_grid = stronger_neighbor.get("display_grid") or stronger_neighbor.get("raw_grid") or stronger_neighbor.get("grid") or []
    first_shared_index, shared_rows = _first_shared_visual_row_signature_index(rows, neighbor_grid)
    if first_shared_index is None or shared_rows < 1:
        return False
    if not _leading_visual_rows_form_prose_intro(rows[:first_shared_index]):
        return False

    table_rows_after_intro = sum(1 for row in rows[first_shared_index:] if _visual_row_has_data_body_shape(row) or _visual_row_has_table_header_shape(row))
    return table_rows_after_intro >= 1


def _overlapping_stronger_table_neighbor(
    table: dict[str, Any],
    previous_table: dict[str, Any] | None,
    next_table: dict[str, Any] | None,
) -> dict[str, Any] | None:
    bbox_raw = table.get("bbox")
    if not bbox_raw or len(bbox_raw) != 4:
        return None
    bbox = tuple(float(value) for value in bbox_raw)
    candidates = [item for item in (previous_table, next_table) if item is not None]
    for candidate in candidates:
        candidate_bbox_raw = candidate.get("bbox")
        if not candidate_bbox_raw or len(candidate_bbox_raw) != 4:
            continue
        candidate_bbox = tuple(float(value) for value in candidate_bbox_raw)
        if _bbox_overlap_ratio(bbox, candidate_bbox) < 0.24:
            continue
        source = str(candidate.get("detection_source") or candidate.get("detection_method") or "")
        if source in {"structured_text_region", "pymupdf_builtin", "caption_anchored_horizontal_rules", "embedded_image_ocr"}:
            return candidate
        if source != "visual_structure_grid" and str(candidate.get("title") or candidate.get("caption_text") or "").strip():
            return candidate
    return None


def _shared_visual_row_signature_count(
    rows: list[list[Any]],
    neighbor_grid: Any,
) -> int:
    _, count = _first_shared_visual_row_signature_index(rows, neighbor_grid)
    return count


def _first_shared_visual_row_signature_index(
    rows: list[list[Any]],
    neighbor_grid: Any,
) -> tuple[int | None, int]:
    if not isinstance(neighbor_grid, list) or not neighbor_grid:
        return None, 0
    neighbor_signatures = {
        _visual_row_token_signature(row)
        for row in neighbor_grid
        if isinstance(row, list)
    }
    neighbor_signatures.discard("")
    if not neighbor_signatures:
        return None, 0
    count = 0
    first_index: int | None = None
    for index, row in enumerate(rows):
        signature = _visual_row_token_signature(row)
        if signature and signature in neighbor_signatures:
            if first_index is None:
                first_index = index
            count += 1
    return first_index, count


def _leading_visual_rows_form_prose_intro(rows: list[list[Any]]) -> bool:
    if len(rows) < 2:
        return False
    row_texts = [
        " ".join(text for _, text in _row_non_empty_cells(row)).strip()
        for row in rows
        if isinstance(row, list) and _row_non_empty_cells(row)
    ]
    if len(row_texts) < 2:
        return False
    combined = " ".join(row_texts)
    words = re.findall(r"[A-Za-z]{2,}", combined)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", combined)
    sentence_cues = re.findall(
        r"\b(?:a|an|the|of|for|with|where|when|if|then|only|however|because|therefore|and|or|in|to|from|by|as|that|both|single)\b",
        combined,
        re.IGNORECASE,
    )
    punctuation = re.findall(r"[.;:!?,\u3002\uff0c\uff1b\uff1a\uff01\uff1f]", combined)
    has_sentence_evidence = (
        (len(words) >= 10 or len(cjk_chars) >= 24)
        and len(sentence_cues) >= 3
        and (punctuation or len(row_texts) >= 3)
    )
    if not has_sentence_evidence:
        compact_label_rows = 0
        for row in rows:
            filled = [text for _, text in _row_non_empty_cells(row)]
            if filled and all(_visual_cell_is_compact_label(text) for text in filled):
                compact_label_rows += 1
        if compact_label_rows >= len(row_texts):
            return False
    return bool(
        has_sentence_evidence
    )


def _visual_row_token_signature(row: list[Any]) -> str:
    tokens = sorted(_visual_row_normalized_tokens(row))
    return "|".join(tokens)


def _build_current_table_context(
    raw_evidence: RawTableEvidence,
    text_blocks: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Collect current-table local context before continuation assessment."""
    projected_raw_data = _caption_rule_rows_with_external_header_continuations(raw_evidence, text_blocks)
    opening_structure = analyze_table_opening_structure(
        projected_raw_data or raw_evidence.raw_data or [],
        raw_evidence.physical_col_count,
    )
    internal_title_text = str(opening_structure.title_text or "").strip()
    header_cells = list(opening_structure.header_cells)
    header_texts = list(opening_structure.header_texts)
    raw_caption_text = str(getattr(raw_evidence, "caption_text", "") or "").strip()
    raw_caption_bbox = getattr(raw_evidence, "caption_bbox", None)
    raw_caption_source = str(getattr(raw_evidence, "caption_source", "") or "").strip() or raw_evidence.source
    raw_caption_block = None
    if raw_caption_text:
        raw_caption_block = {
            "text": raw_caption_text,
            "bbox": list(raw_caption_bbox) if raw_caption_bbox else list(raw_evidence.bbox),
            "source": raw_caption_source,
        }

    if not text_blocks:
        return {
            "title_block": raw_caption_block,
            "preceding_text_block": None,
            "section_hint": None,
            "title_text": internal_title_text or raw_caption_text,
            "preceding_text": "",
            "local_signal": "new_table_title" if (internal_title_text or raw_caption_text) else "none",
            "header_cells": header_cells,
            "header_texts": header_texts,
        }

    bbox = raw_evidence.bbox
    cfg = get_pdf_parser_settings().cross_page_stitching
    title_block = _find_title_block(text_blocks, bbox)
    if title_block is None:
        title_block = _find_descriptive_micro_table_title_block(
            text_blocks=text_blocks,
            words=getattr(raw_evidence, "words", None),
            bbox=bbox,
            row_count=int(raw_evidence.physical_row_count or 0),
            col_count=int(raw_evidence.physical_col_count or 0),
            raw_rows=raw_evidence.raw_data or [],
            page_height=float(raw_evidence.page_height or 0.0),
            page_width=float(raw_evidence.page_width or 0.0),
        )
    raw_caption_forces_title = False
    if raw_caption_block is not None and str(raw_caption_source or "") == "bottom_caption_text_aligned_grid":
        title_block = raw_caption_block
        raw_caption_forces_title = True
    if title_block is not None and raw_caption_text:
        raw_caption_text = _prefer_text_layer_caption_spacing(
            raw_caption_text,
            str(title_block.get("text", "") or ""),
        )
        raw_caption_block["text"] = raw_caption_text if raw_caption_block else raw_caption_text
    if (
        not raw_caption_forces_title
        and
        raw_caption_block is not None
        and title_block is not None
        and _raw_caption_is_better_title(raw_caption_text, str(title_block.get("text", "") or ""))
    ):
        title_block = raw_caption_block
    elif raw_caption_block is not None and title_block is not None:
        raw_caption_block = None
    if title_block is None and raw_caption_block is not None:
        title_block = raw_caption_block
    preceding_block = _find_preceding_text_block(text_blocks, bbox, cfg)
    section_hint = _find_section_hint_block(text_blocks, bbox)
    section_boundary_block = _find_table_boundary_text_block(text_blocks, bbox, include_study_metadata=False)
    if section_hint is None and section_boundary_block is not None:
        section_hint = section_boundary_block
    barrier_block = section_boundary_block or _find_table_boundary_text_block(text_blocks, bbox)
    if preceding_block is None and barrier_block is not None:
        preceding_block = barrier_block
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
    if _looks_like_study_metadata_boundary_text(title_text):
        title_text = ""
        internal_title_text = ""
    if title_block and _looks_like_study_metadata_boundary_text(str(title_block.get("text", "") or "")):
        title_block = None
    if (
        title_block
        and _looks_like_table_title(str(title_block.get("text", "") or "").strip())
        and (not title_text or not _looks_like_table_title(title_text))
    ):
        title_text = str(title_block.get("text", "")).strip()
    if not title_text and title_block:
        title_text = str(title_block.get("text", "")).strip()
    elif not title_text and raw_caption_text:
        title_text = raw_caption_text
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


def _caption_rule_rows_with_external_header_continuations(
    raw_evidence: RawTableEvidence,
    text_blocks: list[dict[str, Any]] | None,
) -> list[list[str | None]] | None:
    if str(raw_evidence.source or "") != "caption_anchored_horizontal_rules":
        return None
    if not text_blocks or not raw_evidence.raw_data or raw_evidence.physical_col_count < 2:
        return None

    bbox = tuple(float(value) for value in raw_evidence.bbox)
    first_row = list(raw_evidence.raw_data[0])
    first_non_empty = [str(cell or "").strip() for cell in first_row if str(cell or "").strip()]
    if len(first_non_empty) < 2:
        return None

    header_row_block_projection = _caption_rule_project_header_from_text_block(raw_evidence, text_blocks)
    if header_row_block_projection:
        return header_row_block_projection

    candidates: list[tuple[float, dict[str, Any]]] = []
    for block in text_blocks:
        text = str(block.get("text", "") or "").strip()
        if not text or _looks_like_table_title(text):
            continue
        block_bbox_raw = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        if len(block_bbox_raw) != 4:
            continue
        block_bbox = tuple(float(value) for value in block_bbox_raw)
        if block_bbox[1] < bbox[1] - 1.0 or block_bbox[0] < bbox[0] - 8.0 or block_bbox[2] > bbox[2] + 12.0:
            continue
        vertical_gap = block_bbox[1] - bbox[1]
        if vertical_gap <= 0.0 or vertical_gap > 18.0:
            continue
        if _horizontal_overlap_ratio(bbox, block_bbox) < 0.08:
            continue
        candidates.append((vertical_gap, block))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    continuation = str(candidates[0][1].get("text", "") or "").strip()
    if not continuation or len(continuation.split()) > 3:
        return None
    if _caption_rule_external_header_continuation_is_body_value(continuation, raw_evidence.raw_data):
        return None
    if any(
        re.sub(r"\s+", "", continuation).lower() == re.sub(r"\s+", "", cell).lower()
        for cell in first_non_empty
    ):
        return None

    continuation_row = [None] * int(raw_evidence.physical_col_count)
    continuation_row[0] = continuation
    return [first_row, continuation_row] + [list(row) for row in raw_evidence.raw_data[1:]]


def _caption_rule_external_header_continuation_is_body_value(
    continuation: str,
    raw_data: list[list[str | None]] | None,
) -> bool:
    cleaned = str(continuation or "").strip()
    if not cleaned:
        return False
    compact = re.sub(r"\s+", "", cleaned).lower()
    for row in (raw_data or [])[1:]:
        for cell in row:
            if compact and compact == re.sub(r"\s+", "", str(cell or "")).lower():
                return True
    return _looks_like_body_identifier_text(cleaned)


def _caption_rule_project_header_from_text_block(
    raw_evidence: RawTableEvidence,
    text_blocks: list[dict[str, Any]] | None,
) -> list[list[str | None]] | None:
    projection = _caption_rule_header_projection_metadata(raw_evidence, text_blocks)
    if not projection:
        return None
    projected_raw_data = projection.get("projected_raw_data")
    return [list(row) for row in projected_raw_data] if isinstance(projected_raw_data, list) else None


def _caption_rule_header_projection_metadata(
    raw_evidence: RawTableEvidence,
    text_blocks: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not text_blocks or not raw_evidence.raw_data:
        return None
    header_texts = [str(cell or "").strip() for cell in raw_evidence.raw_data[0]]
    if len(header_texts) < 2 or not header_texts[0]:
        return None
    bbox = tuple(float(value) for value in raw_evidence.bbox)
    compact_header = re.sub(r"\s+", "", " ".join(header_texts))
    candidates: list[tuple[float, str]] = []
    for block in text_blocks:
        text = str(block.get("text", "") or "").strip()
        if not text or _looks_like_table_title(text):
            continue
        block_bbox_raw = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        if len(block_bbox_raw) != 4:
            continue
        block_bbox = tuple(float(value) for value in block_bbox_raw)
        if block_bbox[1] < bbox[1] - 3.0 or block_bbox[3] > bbox[1] + 28.0:
            continue
        if _horizontal_overlap_ratio(bbox, block_bbox) < 0.35:
            continue
        compact_text = re.sub(r"\s+", "", text)
        if not compact_text or not compact_header:
            continue
        if compact_header == compact_text or not _header_text_block_contains_all_header_tokens(text, header_texts):
            continue
        candidates.append((abs(block_bbox[1] - bbox[1]), text))

    if not candidates:
        return _caption_rule_header_projection_from_adjacent_blocks(raw_evidence, text_blocks, header_texts)
    candidates.sort(key=lambda item: item[0])
    source_text = candidates[0][1]
    expanded_header = _merge_header_tokens_from_text_block(source_text, header_texts)
    if expanded_header == header_texts:
        return None
    continuation_row = _build_header_projection_continuation_row(header_texts, expanded_header)
    return {
        "base_header": list(header_texts),
        "projected_header": list(expanded_header),
        "continuation_row": continuation_row,
        "projected_raw_data": [expanded_header] + [list(row) for row in raw_evidence.raw_data[1:]],
        "audit_grid": [list(header_texts), continuation_row] + [list(row) for row in raw_evidence.raw_data[1:]],
        "source": "text_block_header_projection",
        "source_text": source_text,
    }


def _caption_rule_header_projection_from_adjacent_blocks(
    raw_evidence: RawTableEvidence,
    text_blocks: list[dict[str, Any]] | None,
    header_texts: list[str],
) -> dict[str, Any] | None:
    if not text_blocks or not raw_evidence.raw_data:
        return None
    bbox = tuple(float(value) for value in raw_evidence.bbox)
    header_blocks: list[dict[str, Any]] = []
    for block in text_blocks:
        text = str(block.get("text", "") or "").strip()
        if not text or _looks_like_table_title(text):
            continue
        block_bbox_raw = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        if len(block_bbox_raw) != 4:
            continue
        block_bbox = tuple(float(value) for value in block_bbox_raw)
        if block_bbox[1] < bbox[1] - 3.0 or block_bbox[1] > bbox[1] + 24.0:
            continue
        if block_bbox[2] < bbox[0] - 4.0 or block_bbox[0] > bbox[2] + 4.0:
            continue
        header_blocks.append({"text": text, "bbox": block_bbox})

    if len(header_blocks) < len([cell for cell in header_texts if cell]):
        return None

    projected = list(header_texts)
    continuation_row: list[str | None] = [None] * len(header_texts)
    changed = False
    consumed_continuations: set[tuple[str, tuple[float, float, float, float]]] = set()
    for idx, base_text in enumerate(header_texts):
        base = str(base_text or "").strip()
        if not base:
            continue
        base_block = _find_matching_header_block(base, header_blocks)
        if not base_block:
            continue
        exact_base_block = (
            re.sub(r"\s+", "", str(base_block.get("text", "") or "")).lower()
            == re.sub(r"\s+", "", base).lower()
        )
        if not exact_base_block and idx != 0:
            continue
        base_bbox = tuple(base_block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        continuation_parts: list[tuple[float, str, tuple[str, tuple[float, float, float, float]]]] = []
        for block in header_blocks:
            text = str(block.get("text", "") or "").strip()
            if not text or _text_matches_any_header_cell(text, header_texts):
                continue
            block_bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
            continuation_key = (text, block_bbox)
            if continuation_key in consumed_continuations:
                continue
            if block_bbox[1] <= base_bbox[1] + 1.0:
                continue
            if block_bbox[1] - base_bbox[1] > 18.0:
                continue
            if not _header_block_same_column(base_bbox, block_bbox):
                continue
            if _cell_value_profile_for_header_projection(text) in {"numeric", "statistical"}:
                continue
            continuation_parts.append((block_bbox[1], text, continuation_key))
        if not continuation_parts:
            continue
        continuation_parts.sort(key=lambda item: item[0])
        tail = " ".join(part for _y, part, _key in continuation_parts).strip()
        if not tail:
            continue
        if _caption_rule_external_header_continuation_is_body_value(tail, raw_evidence.raw_data):
            continue
        projected[idx] = f"{base} {tail}".strip()
        continuation_row[idx] = tail
        consumed_continuations.update(key for _y, _part, key in continuation_parts)
        changed = True

    if not changed:
        return None
    source_text = " ".join(
        str(block.get("text", "") or "").strip()
        for block in sorted(header_blocks, key=lambda item: (tuple(item.get("bbox", (0, 0, 0, 0)))[1], tuple(item.get("bbox", (0, 0, 0, 0)))[0]))
        if str(block.get("text", "") or "").strip()
    )
    return {
        "base_header": list(header_texts),
        "projected_header": list(projected),
        "continuation_row": continuation_row,
        "projected_raw_data": [projected] + [list(row) for row in raw_evidence.raw_data[1:]],
        "audit_grid": [list(header_texts), continuation_row] + [list(row) for row in raw_evidence.raw_data[1:]],
        "source": "adjacent_text_block_header_projection",
        "source_text": source_text,
    }


def _find_matching_header_block(base_text: str, header_blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    base_compact = re.sub(r"\s+", "", str(base_text or "")).lower()
    if not base_compact:
        return None
    matches = [
        block
        for block in header_blocks
        if re.sub(r"\s+", "", str(block.get("text", "") or "")).lower() == base_compact
    ]
    if matches:
        return sorted(matches, key=lambda item: (tuple(item.get("bbox", (0, 0, 0, 0)))[1], tuple(item.get("bbox", (0, 0, 0, 0)))[0]))[0]
    for block in header_blocks:
        text_compact = re.sub(r"\s+", "", str(block.get("text", "") or "")).lower()
        if base_compact in text_compact:
            return block
    return None


def _text_matches_any_header_cell(text: str, header_texts: list[str]) -> bool:
    compact = re.sub(r"\s+", "", str(text or "")).lower()
    return bool(compact) and any(compact == re.sub(r"\s+", "", cell).lower() for cell in header_texts if cell)


def _header_block_same_column(
    base_bbox: tuple[float, float, float, float],
    block_bbox: tuple[float, float, float, float],
) -> bool:
    base_width = max(1.0, base_bbox[2] - base_bbox[0])
    block_width = max(1.0, block_bbox[2] - block_bbox[0])
    x_overlap = max(0.0, min(base_bbox[2], block_bbox[2]) - max(base_bbox[0], block_bbox[0]))
    if x_overlap / min(base_width, block_width) >= 0.45:
        return True
    return abs(block_bbox[0] - base_bbox[0]) <= max(4.0, base_width * 0.18)


def _cell_value_profile_for_header_projection(text: str) -> str:
    cleaned = str(text or "").strip()
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:%|[A-Za-z]+)?", cleaned):
        return "numeric"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?\s*[±+/-]\s*\d+(?:\.\d+)?", cleaned):
        return "statistical"
    return "text"


def _header_text_block_contains_all_header_tokens(text: str, header_texts: list[str]) -> bool:
    compact_text = re.sub(r"\s+", "", text).lower()
    return all(re.sub(r"\s+", "", token).lower() in compact_text for token in header_texts if token)


def _merge_header_tokens_from_text_block(text: str, header_texts: list[str]) -> list[str | None]:
    merged = list(header_texts)
    for idx, token in enumerate(header_texts):
        if not token:
            continue
        next_token = next((item for item in header_texts[idx + 1 :] if item), "")
        if not next_token:
            continue
        pattern = re.compile(
            rf"{re.escape(token)}\s+(?P<tail>[A-Za-z][A-Za-z0-9\-]*)\s+{re.escape(next_token)}",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            continue
        tail = str(match.group("tail") or "").strip()
        if _looks_like_body_identifier_text(tail):
            continue
        if tail and tail.lower() not in {token.lower(), next_token.lower()}:
            merged[idx] = f"{token} {tail}".strip()
            break
    return merged


def _looks_like_body_identifier_text(text: str | None) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    if len(cleaned) <= 48 and len(cleaned.split()) <= 5 and not re.search(r"[.;:!?。；：！？]\s*$", cleaned):
        if re.search(r"[+\-#*$]", cleaned) and re.search(r"[A-Za-z0-9]", cleaned):
            return True
    if re.search(r"\d", cleaned) and re.search(r"[A-Za-z]", cleaned):
        return True
    if re.fullmatch(r"[A-Z]{2,}(?:[-_/][A-Z0-9]+)+", cleaned):
        return True
    return False


def _build_header_projection_continuation_row(
    base_header: list[str],
    projected_header: list[str | None],
) -> list[str | None]:
    continuation: list[str | None] = [None] * max(len(base_header), len(projected_header))
    for idx, projected in enumerate(projected_header):
        if idx >= len(base_header):
            continue
        base = str(base_header[idx] or "").strip()
        expanded = str(projected or "").strip()
        if not base or not expanded or expanded == base:
            continue
        if not expanded.lower().startswith(base.lower()):
            continue
        tail = expanded[len(base) :].strip()
        if tail:
            continuation[idx] = tail
            break
    return continuation


def _prefer_text_layer_caption_spacing(raw_caption_text: str, title_text: str) -> str:
    raw_text = str(raw_caption_text or "").strip()
    candidate = str(title_text or "").strip()
    if not raw_text:
        return _normalize_caption_prefix_spacing(candidate)
    if not candidate:
        return _normalize_caption_prefix_spacing(raw_text)
    if re.sub(r"\s+", "", raw_text) != re.sub(r"\s+", "", candidate):
        return _normalize_caption_prefix_spacing(raw_text)
    raw_score = _caption_spacing_quality_score(raw_text)
    candidate_score = _caption_spacing_quality_score(candidate)
    if candidate_score > raw_score:
        return _normalize_caption_prefix_spacing(candidate)
    if raw_score > candidate_score:
        return _normalize_caption_prefix_spacing(raw_text)
    if _caption_has_rich_prefix_gap(raw_text) and not _caption_has_rich_prefix_gap(candidate):
        return _normalize_caption_prefix_spacing(raw_text)
    if raw_text == candidate:
        return _normalize_caption_prefix_spacing(raw_text)
    return _normalize_caption_prefix_spacing(candidate)


def _caption_spacing_quality_score(text: str) -> int:
    cleaned = str(text or "").strip()
    if not cleaned:
        return 0
    score = 0
    words = re.findall(r"[A-Za-z\ufb01\ufb02]{2,}|\d+|[\u4e00-\u9fff]+", cleaned)
    score += len(words)
    score += len(re.findall(r"\s+", cleaned))
    compact_tail = _caption_tail_without_prefix(cleaned)
    if compact_tail and len(compact_tail) >= 16:
        long_alpha_runs = re.findall(r"[A-Za-z\ufb01\ufb02]{16,}", compact_tail)
        score -= sum(len(run) for run in long_alpha_runs)
    return score


def _caption_tail_without_prefix(text: str) -> str:
    return re.sub(r"(?i)^\s*(?:table|fig\.?|figure)\s*[0-9A-Za-z]+\s*", "", str(text or "").strip(), count=1)


def _caption_has_rich_prefix_gap(text: str) -> bool:
    return bool(re.match(r"(?i)^\s*(?:table|fig\.?|figure)[\u00a0\u2000-\u200a\u202f\u205f\u3000]+[0-9A-Za-z]", str(text or "")))


def _normalize_caption_prefix_spacing(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"(?i)^((?:table|fig\.?|figure))(?=[0-9A-Za-z])", r"\1 ", cleaned, count=1)
    return _repair_caption_cjk_variable_spacing(cleaned)


def _repair_caption_cjk_variable_spacing(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"([\u4e00-\u9fff])\s+([A-Za-z]\d*)", r"\1\2", cleaned)
    cleaned = re.sub(r"([A-Za-z]\d*)\s+([\u4e00-\u9fff])", r"\1\2", cleaned)
    cleaned = re.sub(r"([A-Za-z]\d*)\s+(?=及|和|与|或|及其|及び)", r"\1", cleaned)
    return cleaned


def _normalize_caption_prefix_to_ascii_space(text: str) -> str:
    cleaned = _normalize_caption_prefix_spacing(text)
    return re.sub(
        r"(?i)^((?:table|fig\.?|figure))[\u00a0\u2000-\u200a\u202f\u205f\u3000]+([0-9A-Za-z])",
        r"\1 \2",
        cleaned,
        count=1,
    )


def _recover_rich_table_title_from_page(
    *,
    page: Any,
    table_bbox: tuple[float, float, float, float],
    current_title: str,
) -> str:
    current = str(current_title or "").strip()
    if not current or not _looks_like_table_title(current):
        return current

    candidates: list[tuple[float, str]] = []
    try:
        blocks = page.get_text("dict", flags=11).get("blocks", [])
    except Exception:
        blocks = []

    compact_current = re.sub(r"\s+", "", current)
    for block in blocks:
        if "lines" not in block:
            continue
        rich_text = " ".join(
            str(span.get("text", "") or "")
            for line in block.get("lines", []) or []
            for span in line.get("spans", []) or []
        ).strip()
        if not rich_text or not _looks_like_table_title(rich_text):
            continue
        if re.sub(r"\s+", "", rich_text) != compact_current:
            continue
        block_bbox_raw = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        if len(block_bbox_raw) != 4:
            continue
        block_bbox = tuple(float(value) for value in block_bbox_raw)
        if not _caption_block_is_near_table(table_bbox, block_bbox):
            continue
        candidates.append((max(0.0, table_bbox[1] - block_bbox[3]), _normalize_rich_caption_spacing(rich_text)))

    word_title = _recover_table_title_from_words(page=page, table_bbox=table_bbox, current_title=current)
    if word_title and re.sub(r"\s+", "", word_title) == compact_current:
        candidates.append((0.0, word_title))

    if not candidates:
        return _normalize_caption_prefix_spacing(current)
    candidates.sort(key=lambda item: (item[0], -_caption_spacing_quality_score(item[1])))
    best = current
    best_score = _caption_spacing_quality_score(current)
    for _gap, candidate in candidates:
        candidate = _prefer_text_layer_caption_spacing(candidate, current)
        candidate_score = _caption_spacing_quality_score(candidate)
        if candidate_score > best_score or (_caption_has_rich_prefix_gap(candidate) and not _caption_has_rich_prefix_gap(best)):
            best = candidate
            best_score = candidate_score
    return _normalize_caption_prefix_spacing(best)


def _caption_block_is_near_table(
    table_bbox: tuple[float, float, float, float],
    block_bbox: tuple[float, float, float, float],
) -> bool:
    gap = table_bbox[1] - block_bbox[3]
    if gap < -12.0 or gap > 180.0:
        return False
    return _horizontal_overlap_ratio(table_bbox, block_bbox) >= 0.10


def _recover_table_title_from_words(
    *,
    page: Any,
    table_bbox: tuple[float, float, float, float],
    current_title: str,
) -> str | None:
    try:
        words = page.get_text("words")
    except Exception:
        return None
    current_compact = re.sub(r"\s+", "", str(current_title or ""))
    if not current_compact:
        return None

    word_items: list[tuple[float, float, float, float, str]] = []
    for word in words or []:
        if len(word) < 5:
            continue
        x0, y0, x1, y1, value = word[:5]
        text = str(value or "").strip()
        if not text:
            continue
        bbox = (float(x0), float(y0), float(x1), float(y1))
        if not _candidate_word_near_table_title(table_bbox, bbox):
            continue
        word_items.append((bbox[0], bbox[1], bbox[2], bbox[3], text))

    if not word_items:
        return None
    rows = _cluster_caption_words_by_line(word_items)
    for start_idx in range(len(rows)):
        combined: list[str] = []
        row_bbox: tuple[float, float, float, float] | None = None
        for row in rows[start_idx:start_idx + 4]:
            row_text = _join_caption_word_row(row)
            if not row_text:
                continue
            combined.append(row_text)
            row_bbox = _union_word_bbox(row_bbox, row)
            candidate = _normalize_rich_caption_spacing(" ".join(combined))
            if not _looks_like_table_title(candidate):
                continue
            if re.sub(r"\s+", "", candidate) == current_compact and row_bbox and _caption_block_is_near_table(table_bbox, row_bbox):
                return candidate
    return None


def _candidate_word_near_table_title(
    table_bbox: tuple[float, float, float, float],
    word_bbox: tuple[float, float, float, float],
) -> bool:
    vertical_gap = table_bbox[1] - word_bbox[3]
    if vertical_gap < -18.0 or vertical_gap > 190.0:
        return False
    expanded_table = (table_bbox[0] - 12.0, table_bbox[1], table_bbox[2] + 12.0, table_bbox[3])
    return _horizontal_overlap_ratio(expanded_table, word_bbox) > 0.0 or (
        table_bbox[0] - 18.0 <= word_bbox[0] <= table_bbox[2] + 18.0
    )


def _cluster_caption_words_by_line(
    words: list[tuple[float, float, float, float, str]],
) -> list[list[tuple[float, float, float, float, str]]]:
    ordered = sorted(words, key=lambda item: (item[1], item[0]))
    rows: list[list[tuple[float, float, float, float, str]]] = []
    for word in ordered:
        center_y = (word[1] + word[3]) / 2.0
        for row in rows:
            row_center_y = sum((item[1] + item[3]) / 2.0 for item in row) / len(row)
            if abs(center_y - row_center_y) <= 4.0:
                row.append(word)
                break
        else:
            rows.append([word])
    return [sorted(row, key=lambda item: item[0]) for row in rows]


def _join_caption_word_row(row: list[tuple[float, float, float, float, str]]) -> str:
    return " ".join(str(item[4] or "").strip() for item in row if str(item[4] or "").strip()).strip()


def _union_word_bbox(
    current: tuple[float, float, float, float] | None,
    row: list[tuple[float, float, float, float, str]],
) -> tuple[float, float, float, float] | None:
    if not row:
        return current
    row_bbox = (
        min(item[0] for item in row),
        min(item[1] for item in row),
        max(item[2] for item in row),
        max(item[3] for item in row),
    )
    if current is None:
        return row_bbox
    return (
        min(current[0], row_bbox[0]),
        min(current[1], row_bbox[1]),
        max(current[2], row_bbox[2]),
        max(current[3], row_bbox[3]),
    )


def _normalize_rich_caption_spacing(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    rich_prefix_gap = _caption_has_rich_prefix_gap(raw)
    cleaned = " ".join(raw.split())
    if rich_prefix_gap:
        cleaned = re.sub(
            r"(?i)^((?:table|fig\.?|figure))\s+([0-9A-Za-z])",
            lambda match: f"{match.group(1)}\u202f{match.group(2)}",
            cleaned,
            count=1,
        )
    return cleaned



def _apply_external_header_projection_to_ast(
    table: dict[str, Any],
    projection: dict[str, Any],
) -> None:
    projected_header = [
        str(cell or "").strip()
        for cell in projection.get("projected_header", []) or []
    ]
    base_header = [
        str(cell or "").strip()
        for cell in projection.get("base_header", []) or []
    ]
    if not projected_header or len(projected_header) != len(base_header):
        return
    current_header = [
        str(cell.get("text", "") if isinstance(cell, dict) else cell or "").strip()
        for cell in table.get("header", []) or []
    ]
    if current_header and len(current_header) != len(projected_header):
        return
    compact_current = re.sub(r"\s+", "", " ".join(current_header or base_header)).lower()
    compact_base = re.sub(r"\s+", "", " ".join(base_header)).lower()
    if compact_current and compact_base and compact_current != compact_base:
        if not _header_projection_matches_with_duplicate_continuation(current_header, base_header, projected_header, projection):
            return

    table["header"] = [
        {"col": idx + 1, "text": text or f"Column {idx + 1}"}
        for idx, text in enumerate(projected_header)
    ]
    table["header_rebuilt_by_text_block_projection"] = True
    table.setdefault("header_row_groups", [])
    table["header_row_groups"].append(
        {
            "start_row": _external_title_audit_row_count(table),
            "end_row": _external_title_audit_row_count(table) + 1,
            "source": str(projection.get("source") or "text_block_header_projection"),
        }
    )
    table["data_start_row"] = _external_title_audit_row_count(table) + 2
    _refresh_external_header_projection_audit_rows(table, projection)


def _header_projection_matches_with_duplicate_continuation(
    current_header: list[str],
    base_header: list[str],
    projected_header: list[str],
    projection: dict[str, Any],
) -> bool:
    if not current_header or len(current_header) != len(projected_header):
        return False
    continuation_tokens = {
        re.sub(r"\s+", "", str(token or "")).lower()
        for token in projection.get("continuation_row", []) or []
        if str(token or "").strip()
    }
    if not continuation_tokens:
        return False
    for idx, current in enumerate(current_header):
        current_compact = re.sub(r"\s+", "", current).lower()
        projected_compact = re.sub(r"\s+", "", projected_header[idx]).lower()
        base_compact = re.sub(r"\s+", "", base_header[idx]).lower()
        if current_compact in {projected_compact, base_compact}:
            continue
        duplicate_tail_ok = any(current_compact == f"{base_compact}{token}" for token in continuation_tokens)
        if duplicate_tail_ok:
            continue
        return False
    return True


def _refresh_external_header_projection_audit_rows(
    table: dict[str, Any],
    projection: dict[str, Any],
) -> None:
    audit_grid = projection.get("audit_grid")
    if not isinstance(audit_grid, list) or not audit_grid:
        return
    col_count = max(
        int(table.get("col_count", 0) or 0),
        max((len(row) for row in audit_grid if isinstance(row, list)), default=0),
        len(table.get("header", []) or []),
    )
    title_rows = _external_title_audit_rows(table, col_count)
    data_rows = table.get("data_grid") or table.get("grid") or table.get("raw_grid") or []
    data_rows = [list(row) for row in data_rows if isinstance(row, list)]
    audit_rows = title_rows + [list(row) for row in audit_grid[:2]] + data_rows
    table["raw_row_texts"] = _render_table_audit_rows(audit_rows, col_count)
    table["raw_row_count"] = len(table["raw_row_texts"])
    table["external_header_projection"] = {
        "source": str(projection.get("source") or "text_block_header_projection"),
        "source_text": str(projection.get("source_text") or ""),
    }


def _external_title_audit_row_count(table: dict[str, Any]) -> int:
    title = str(table.get("title") or "").strip()
    if not title:
        return 0
    title_block = table.get("title_block") or {}
    bbox = title_block.get("bbox") or table.get("title_bbox") or []
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        height = float(bbox[3]) - float(bbox[1])
        if height > 17.0 and len(title.split()) >= 8:
            return 2
    return 1


def _external_title_audit_rows(table: dict[str, Any], col_count: int) -> list[list[str | None]]:
    title = str(table.get("title") or "").strip()
    if not title:
        return []
    row_count = _external_title_audit_row_count(table)
    if row_count <= 1:
        row = [None] * max(1, col_count)
        row[0] = title
        return [row]
    first, second = _split_title_for_audit_rows(title)
    first_row = [None] * max(1, col_count)
    second_row = [None] * max(1, col_count)
    first_row[0] = first
    second_row[0] = second
    return [first_row, second_row]


def _split_title_for_audit_rows(title: str) -> tuple[str, str]:
    words = str(title or "").split()
    if len(words) < 2:
        return title, ""
    midpoint = len(title) / 2.0
    best_idx = 1
    best_distance = float("inf")
    running = 0
    for idx, word in enumerate(words[:-1], start=1):
        running += len(word) + (1 if idx > 1 else 0)
        distance = abs(running - midpoint)
        if distance < best_distance:
            best_idx = idx
            best_distance = distance
    return " ".join(words[:best_idx]).strip(), " ".join(words[best_idx:]).strip()


def _render_table_audit_rows(rows: list[list[str | None]], col_count: int) -> list[str]:
    rendered: list[str] = []
    width = max(1, col_count)
    for row in rows:
        cells = list(row) + [None] * max(0, width - len(row))
        rendered.append(" | ".join(str(cell).strip() if str(cell or "").strip() else "null" for cell in cells[:width]))
    return rendered


def _raw_caption_is_better_title(raw_caption_text: str, title_text: str) -> bool:
    raw_clean = re.sub(r"\s+", "", str(raw_caption_text or ""))
    title_clean = re.sub(r"\s+", "", str(title_text or ""))
    if not raw_clean or not title_clean:
        return bool(raw_clean)
    if raw_clean == title_clean:
        return False
    if title_clean in raw_clean and len(raw_clean) > len(title_clean):
        return True
    raw_open = raw_clean.count("[") + raw_clean.count("【") + raw_clean.count("(") + raw_clean.count("（")
    raw_close = raw_clean.count("]") + raw_clean.count("】") + raw_clean.count(")") + raw_clean.count("）")
    title_open = title_clean.count("[") + title_clean.count("【") + title_clean.count("(") + title_clean.count("（")
    title_close = title_clean.count("]") + title_clean.count("】") + title_clean.count(")") + title_clean.count("）")
    return title_open > title_close and raw_open <= raw_close and raw_clean.startswith(title_clean.rstrip("[【(（"))


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


def _find_descriptive_micro_table_title_block(
    *,
    text_blocks: list[dict[str, Any]] | None,
    words: list[tuple[float, float, float, float, str]] | None,
    bbox: tuple[float, float, float, float],
    row_count: int,
    col_count: int,
    raw_rows: list[list[Any]],
    page_height: float,
    page_width: float,
) -> dict[str, Any] | None:
    """Promote centered short descriptors above record-like one-row tables.

    IND/ICH revision-history pages often use a centered descriptive phrase as
    the title for a single-row record table without a "Table X" caption.
    This profile is intentionally limited to micro tables so ordinary section
    headings above larger tables keep their existing ownership path.
    """
    if not _is_record_like_micro_table(row_count=row_count, col_count=col_count, raw_rows=raw_rows):
        return None

    candidates: list[tuple[float, float, dict[str, Any]]] = []
    table_x0, table_y0, table_x1, _table_y1 = [float(value) for value in bbox]
    table_width = max(1.0, table_x1 - table_x0)
    table_center = (table_x0 + table_x1) / 2.0

    for block in text_blocks or []:
        text = str(block.get("text") or "").strip()
        block_bbox = _coerce_context_bbox(block.get("bbox"))
        if not text or block_bbox is None:
            continue
        if _looks_like_study_metadata_boundary_text(text):
            continue
        candidate = _score_descriptive_micro_table_title_candidate(
            text=text,
            candidate_bbox=block_bbox,
            table_bbox=bbox,
            table_center=table_center,
            table_width=table_width,
            page_height=page_height,
            page_width=page_width,
        )
        if candidate is None:
            continue
        gap, center_delta = candidate
        candidates.append((gap, center_delta, {**block, "source": "descriptive_title_micro_table"}))

    word_candidate = _descriptive_micro_table_title_from_words(
        words=words,
        table_bbox=bbox,
        table_center=table_center,
        table_width=table_width,
        page_height=page_height,
        page_width=page_width,
    )
    if word_candidate is not None:
        if _looks_like_study_metadata_boundary_text(str(word_candidate.get("text") or "")):
            word_candidate = None
    if word_candidate is not None:
        word_bbox = _coerce_context_bbox(word_candidate.get("bbox"))
        if word_bbox is not None:
            gap = max(0.0, table_y0 - word_bbox[3])
            center_delta = abs(((word_bbox[0] + word_bbox[2]) / 2.0) - table_center)
            candidates.append((gap, center_delta, word_candidate))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], -len(str(item[2].get("text") or ""))))
    selected = dict(candidates[0][2])
    selected["source"] = "descriptive_title_micro_table"
    selected["ownership_profile"] = "descriptive_title_micro_table"
    selected["ownership_evidence"] = {
        "micro_table": True,
        "row_count": row_count,
        "col_count": col_count,
        "vertical_gap": round(float(candidates[0][0]), 3),
        "center_delta": round(float(candidates[0][1]), 3),
        "alignment": "centered_over_table",
    }
    return selected


def _is_record_like_micro_table(
    *,
    row_count: int,
    col_count: int,
    raw_rows: list[list[Any]],
) -> bool:
    if row_count < 1 or row_count > 2:
        return False
    if col_count < 2:
        return False
    rows = [row for row in raw_rows or [] if isinstance(row, list) and any(str(cell or "").strip() for cell in row)]
    if not rows or len(rows) > 2:
        return False
    filled = [str(cell or "").strip() for row in rows for cell in row if str(cell or "").strip()]
    if len(filled) < 2:
        return False
    value_like = sum(1 for text in filled if _micro_table_cell_is_record_value(text))
    descriptive = sum(1 for text in filled if _micro_table_cell_is_descriptive_text(text))
    return value_like >= 2 or (value_like >= 1 and descriptive >= 1 and len(filled) >= 3)


def _micro_table_cell_is_record_value(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    if re.search(r"\d{4}\s*年|\d{1,2}\s*月|\d{1,2}\s*日", cleaned):
        return True
    if re.search(r"\b\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?\b", cleaned):
        return True
    if re.fullmatch(r"[A-Za-z]{1,8}(?:[-_/]?[A-Za-z0-9]+)*(?:\([A-Za-z0-9]+\))?", cleaned):
        return True
    if re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9 ]{1,18}(?:\([A-Za-z0-9]+\))?", cleaned) and re.search(r"[A-Za-z0-9]", cleaned):
        return True
    return False


def _micro_table_cell_is_descriptive_text(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    compact = _compact_context_text(cleaned)
    return 6 <= len(compact) <= 90 and bool(re.search(r"[\u4e00-\u9fffA-Za-z]", cleaned))


def _score_descriptive_micro_table_title_candidate(
    *,
    text: str,
    candidate_bbox: tuple[float, float, float, float],
    table_bbox: tuple[float, float, float, float],
    table_center: float,
    table_width: float,
    page_height: float,
    page_width: float,
) -> tuple[float, float] | None:
    cleaned = str(text or "").strip()
    if not _looks_like_descriptive_micro_table_title_text(cleaned):
        return None
    gap = float(table_bbox[1]) - float(candidate_bbox[3])
    if gap < -2.0 or gap > 32.0:
        return None
    if _horizontal_overlap_ratio(table_bbox, candidate_bbox) < 0.10:
        return None
    center = (float(candidate_bbox[0]) + float(candidate_bbox[2])) / 2.0
    center_delta = abs(center - table_center)
    if center_delta > max(24.0, table_width * 0.12):
        return None
    candidate_width = max(1.0, float(candidate_bbox[2]) - float(candidate_bbox[0]))
    if candidate_width > table_width * 0.86:
        return None
    if page_height > 0 and page_width > 0 and _is_running_header_block(cleaned, candidate_bbox, page_height, page_width):
        return None
    return max(0.0, gap), center_delta


def _looks_like_descriptive_micro_table_title_text(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    compact = _compact_context_text(cleaned)
    if len(compact) < 4 or len(compact) > 72:
        return False
    if len(cleaned.split()) > 10:
        return False
    if re.search(r"[。！？!?；;]\s*$", cleaned):
        return False
    if TOC_HEADING_PATTERN.search(cleaned) or TOC_TITLE_PATTERN.search(cleaned):
        return False
    if _looks_like_table_title(cleaned):
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", cleaned))


def _descriptive_micro_table_title_from_words(
    *,
    words: list[tuple[float, float, float, float, str]] | None,
    table_bbox: tuple[float, float, float, float],
    table_center: float,
    table_width: float,
    page_height: float,
    page_width: float,
) -> dict[str, Any] | None:
    if not words:
        return None

    def coords(word: Any) -> tuple[float, float, float, float, str]:
        if isinstance(word, tuple):
            x0, y0, x1, y1, text = word[:5]
        else:
            x0, y0, x1, y1 = word.x0, word.y0, word.x1, word.y1
            text = getattr(word, "text", "")
        return float(x0), float(y0), float(x1), float(y1), str(text)

    table_top = float(table_bbox[1])
    row_buckets: dict[float, list[tuple[float, float, float, float, str]]] = {}
    for word in words:
        x0, y0, x1, y1, text = coords(word)
        if not text.strip():
            continue
        if y1 > table_top + 2.0:
            continue
        if table_top - y1 > 34.0:
            continue
        if x1 < float(table_bbox[0]) - 8.0 or x0 > float(table_bbox[2]) + 8.0:
            continue
        key = round(y0 / 3.0) * 3.0
        row_buckets.setdefault(key, []).append((x0, y0, x1, y1, text))

    candidates: list[tuple[float, float, dict[str, Any]]] = []
    for row_words in row_buckets.values():
        row_words = sorted(row_words, key=lambda item: item[0])
        text = " ".join(item[4] for item in row_words).strip()
        bbox = (
            min(item[0] for item in row_words),
            min(item[1] for item in row_words),
            max(item[2] for item in row_words),
            max(item[3] for item in row_words),
        )
        score = _score_descriptive_micro_table_title_candidate(
            text=text,
            candidate_bbox=bbox,
            table_bbox=table_bbox,
            table_center=table_center,
            table_width=table_width,
            page_height=page_height,
            page_width=page_width,
        )
        if score is None:
            continue
        gap, center_delta = score
        candidates.append(
            (
                gap,
                center_delta,
                {
                    "text": text,
                    "bbox": [float(value) for value in bbox],
                    "source": "descriptive_title_micro_table",
                },
            )
        )

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], -len(str(item[2].get("text") or ""))))
    return candidates[0][2]


def _coerce_context_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


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
    if _looks_like_study_metadata_boundary_text(text):
        return None
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


def _looks_like_numbered_section_heading_text(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    if _contains_continuation_hint(cleaned):
        return False
    if not CTD_NUMBERED_SECTION_HEADING_PATTERN.match(cleaned):
        return False
    body = CTD_NUMBERED_SECTION_HEADING_PATTERN.sub("", cleaned, count=1).strip()
    if len(body) < 2:
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", body))


def _looks_like_study_metadata_boundary_text(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    if _contains_continuation_hint(cleaned):
        return False
    return bool(STUDY_METADATA_BOUNDARY_PATTERN.search(cleaned))


def _find_table_boundary_text_block(
    text_blocks: list[dict[str, Any]] | None,
    bbox: tuple[float, float, float, float],
    *,
    max_gap: float = 240.0,
    include_study_metadata: bool = True,
) -> dict[str, Any] | None:
    """Find a strong section/study metadata boundary above a table."""
    if not text_blocks:
        return None

    candidates: list[tuple[int, float, dict[str, Any]]] = []
    for block in text_blocks:
        text = str(block.get("text", "")).strip()
        if not text:
            continue
        block_bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        gap = bbox[1] - block_bbox[3]
        if gap < 0 or gap > max_gap:
            continue
        overlap = _horizontal_overlap_ratio(bbox, block_bbox)
        if overlap < 0.05 and block_bbox[2] < bbox[0]:
            continue
        if _looks_like_numbered_section_heading_text(text):
            candidates.append((0, gap, block))
        elif include_study_metadata and _looks_like_study_metadata_boundary_text(text):
            candidates.append((1, gap, block))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


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
        if _looks_like_study_metadata_boundary_text(title_text):
            title_text = ""
        else:
            if _contains_continuation_hint(title_text):
                return "continuation_title"
            return "new_table_title"

    if preceding_text and _looks_like_study_metadata_boundary_text(preceding_text):
        if section_text and _compact_context_text(section_text) == _compact_context_text(preceding_text):
            return "section_heading"
        return "study_metadata_context"

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
        if _looks_like_numbered_section_heading_text(preceding_text):
            return "section_heading"
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
    header_projection = _caption_rule_header_projection_metadata(raw_evidence, text_blocks)
    projected_raw_data = None if header_projection else _caption_rule_rows_with_external_header_continuations(raw_evidence, text_blocks)
    if projected_raw_data:
        raw_evidence = RawTableEvidence(
            page_number=raw_evidence.page_number,
            bbox=raw_evidence.bbox,
            physical_col_count=raw_evidence.physical_col_count,
            physical_row_count=len(projected_raw_data),
            rows=raw_evidence.rows,
            chars=raw_evidence.chars,
            spans=raw_evidence.spans,
            words=raw_evidence.words,
            drawings=raw_evidence.drawings,
            raw_data=projected_raw_data,
            page_height=raw_evidence.page_height,
            page_width=raw_evidence.page_width,
            near_page_top=raw_evidence.near_page_top,
            near_page_bottom=raw_evidence.near_page_bottom,
            source=raw_evidence.source,
            caption_text=getattr(raw_evidence, "caption_text", ""),
            caption_bbox=getattr(raw_evidence, "caption_bbox", None),
            caption_source=getattr(raw_evidence, "caption_source", ""),
        )

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
    context["raw_source"] = str(raw_evidence.source or "").strip()
    context["raw_physical_col_count"] = int(raw_evidence.physical_col_count or 0)
    context["raw_physical_row_count"] = int(raw_evidence.physical_row_count or 0)
    context["raw_rule_or_box_support_count"] = len(raw_evidence.drawings or [])
    context["raw_has_rule_or_box_support"] = bool(raw_evidence.drawings)
    if current_context:
        current_title_block = current_context.get("title_block")
        if (
            current_title_block
            and _looks_like_table_title(str(current_title_block.get("text", "") or "").strip())
            and (
                not context.get("title_block")
                or not _looks_like_table_title(str(context.get("title_block", {}).get("text", "") or "").strip())
            )
        ):
            context["title_block"] = current_title_block
        if current_context.get("section_hint") and not context.get("section_hint"):
            context["section_hint"] = current_context.get("section_hint")
    if (
        current_context
        and str(getattr(raw_evidence, "caption_text", "") or "").strip()
        and str((current_context.get("title_block") or {}).get("source", "") or "") != "text-layer"
    ):
        current_title_block = current_context.get("title_block")
        if (
            current_title_block
            and context.get("title_block")
            and _raw_caption_is_better_title(
                str(current_title_block.get("text", "") or ""),
                str(context.get("title_block", {}).get("text", "") or ""),
            )
        ):
            context["title_block"] = current_title_block
        if current_title_block and not context.get("title_block"):
            context["title_block"] = current_title_block
        if current_context.get("section_hint") and not context.get("section_hint"):
            context["section_hint"] = current_context.get("section_hint")
    current_context_title_text = str((current_context or {}).get("title_text", "") or "").strip()
    if (
        current_context_title_text
        and instance.title
        and _normalize_title_identity(current_context_title_text) == _normalize_title_identity(str(instance.title or ""))
        and not context.get("title_block")
    ):
        context["title_block"] = {
            "text": current_context_title_text,
            "bbox": list(raw_evidence.bbox),
            "source": "internal_title_row",
        }

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
    strong_visual_structure_invalid = _should_accept_strong_visual_structure_candidate(
        raw_evidence=raw_evidence,
        confidence=confidence,
    )
    if not confidence.is_valid_table and not toc_only_invalid and not strong_visual_structure_invalid:
        return None

    if confidence.overall_confidence < 0.3:
        return None

    # Phase 6: Build AST
    ast = build_logical_ast(instance, continuum_result)

    # Enrich with context
    if context.get("title_block"):
        ast.title = context["title_block"].get("text", "").strip()
    if instance.title and (
        not str(ast.title or "").strip()
        or (
            str((context.get("title_block") or {}).get("source", "") or "") == "internal_title_row"
            and _normalize_title_identity(str(ast.title or "")) == _normalize_title_identity(str(instance.title or ""))
        )
    ):
        ast.title = str(instance.title or "").strip()
    title_block_source = str((context.get("title_block") or {}).get("source", "") or "")
    if title_block_source == "bottom_caption_text_aligned_grid":
        ast.title = _normalize_caption_prefix_spacing(str(ast.title or "").strip())
    else:
        ast.title = _normalize_caption_prefix_spacing(_recover_rich_table_title_from_page(
            page=page,
            table_bbox=tuple(ast.bbox),
            current_title=str(ast.title or "").strip(),
        ))
    if str(raw_evidence.caption_source or "") == "bottom_caption_text_aligned_grid" and str(raw_evidence.caption_text or "").strip():
        ast.title = _normalize_caption_prefix_spacing(str(raw_evidence.caption_text or "").strip())
    if str(raw_evidence.source or "") == "vector_ocr":
        ast.title = _normalize_caption_prefix_to_ascii_space(str(ast.title or ""))
    if context.get("section_hint"):
        ast.section_hint = context["section_hint"].get("text", "").strip()
    ast.toc_context = context.get("toc_context", False)
    ast.grid_line_score = context.get("grid_line_score", 0.0)

    # Post-process cells
    _postprocess_cells(ast)

    table_bbox = tuple(ast.bbox)
    result = ast.to_dict()
    if ast.title:
        result["title"] = ast.title
    dense_header_groups = (
        (normalized.semantic_rule_engine or {})
        .get("dense_leaf_header_groups", {})
        .get("header_column_groups", [])
    )
    if dense_header_groups:
        result["header_column_groups"] = [dict(group) for group in dense_header_groups]
    if context.get("title_block"):
        title_block = context["title_block"]
        title_block_text = _prefer_text_layer_caption_spacing(
            str(title_block.get("text", "") or "").strip(),
            str(ast.title or "").strip(),
        )
        title_bbox = title_block.get("bbox")
        if isinstance(title_bbox, (list, tuple)) and len(title_bbox) == 4:
            result["title_bbox"] = [float(value) for value in title_bbox]
            result["title_block"] = {
                "text": title_block_text,
                "bbox": list(result["title_bbox"]),
                "source": str(title_block.get("source", "text-layer") or "text-layer"),
            }
            if str(title_block.get("source") or "") == "descriptive_title_micro_table":
                result["context_profile"] = "descriptive_title_micro_table"
                result["title_block"]["ownership_profile"] = "descriptive_title_micro_table"
                if isinstance(title_block.get("ownership_evidence"), dict):
                    result["title_block"]["ownership_evidence"] = dict(title_block["ownership_evidence"])
    if str(raw_evidence.caption_source or "") == "bottom_caption_text_aligned_grid" and str(raw_evidence.caption_text or "").strip():
        caption_text = _normalize_caption_prefix_spacing(str(raw_evidence.caption_text or "").strip())
        result["title"] = caption_text
        if raw_evidence.caption_bbox and len(raw_evidence.caption_bbox) == 4:
            result["title_bbox"] = [float(value) for value in raw_evidence.caption_bbox]
            result["title_block"] = {
                "text": caption_text,
                "bbox": list(result["title_bbox"]),
                "source": str(raw_evidence.caption_source or "bottom_caption_text_aligned_grid"),
            }
    result["detection_source"] = raw_evidence.source
    if str(raw_evidence.source or "").strip() and not result.get("detection_method"):
        result["detection_method"] = str(raw_evidence.source or "").strip()
    if raw_evidence.words:
        result["word_evidence"] = [
            word.to_dict() if hasattr(word, "to_dict") else {
                "text": str(getattr(word, "text", "") or ""),
                "bbox": [
                    float(getattr(word, "x0", 0.0) or 0.0),
                    float(getattr(word, "y0", 0.0) or 0.0),
                    float(getattr(word, "x1", 0.0) or 0.0),
                    float(getattr(word, "y1", 0.0) or 0.0),
                ],
            }
            for word in raw_evidence.words
            if str(getattr(word, "text", "") or "").strip()
        ]
    result["header_candidates"] = context.get("header_candidates", [])
    if _should_override_header_with_candidates(result):
        result["header"] = [
            {"col": idx + 1, "text": str(text).strip()}
            for idx, text in enumerate(result.get("header_candidates") or [])
            if str(text).strip()
        ]
        result["header_rebuilt_by_context"] = True
    if header_projection:
        _apply_external_header_projection_to_ast(result, header_projection)
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
    if semantic_role == "business_table" and _looks_like_visual_structure_non_table_false_positive(result):
        return None
    if semantic_role == "business_table" and _looks_like_numbered_guidance_false_positive(result):
        return None
    if semantic_role == "business_table" and _looks_like_two_column_narrative_table_ast_false_positive(result):
        return None
    if semantic_role == "business_table" and _looks_like_cross_column_algorithm_formula_false_positive(result):
        return None
    if semantic_role == "business_table" and _looks_like_text_aligned_narrative_formula_projection_false_positive(result):
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
    else:
        project_simple_schema_header_data_views(result)
        if header_projection:
            _apply_external_header_projection_to_ast(result, header_projection)

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
    if marker_count >= 2 and page_marker_count >= 2:
        return True

    source = str(table_ast.get("detection_source") or table_ast.get("detection_method") or "")
    if source in {"structured_text_region", "visual_structure_grid", "text_aligned_borderless_grid"}:
        masthead_markers = [
            "contentslistsavailable",
            "journalhomepage",
            "wwwelseviercom",
            "sciencedirect",
            "theoreticalbiology",
        ]
        affiliation_markers = [
            "departmentof",
            "university",
            "institute",
            "schoolof",
            "china",
        ]
        title_author_markers = [
            "classification",
            "author",
            "journalof",
        ]
        masthead_count = sum(1 for marker in masthead_markers if marker in row_text_compact or marker in page_context_compact)
        affiliation_count = sum(1 for marker in affiliation_markers if marker in row_text_compact)
        title_author_count = sum(1 for marker in title_author_markers if marker in row_text_compact)
        near_top = bool(table_ast.get("near_page_top", False)) or float((table_ast.get("bbox") or [0, 999])[1]) <= 180.0
        if near_top and masthead_count >= 2 and (affiliation_count >= 1 or title_author_count >= 1):
            return True

    return False


def _looks_like_visual_structure_non_table_false_positive(table_ast: dict[str, Any]) -> bool:
    source = str(table_ast.get("detection_source") or table_ast.get("detection_method") or "")
    if source != "visual_structure_grid":
        return False

    row_texts = [
        str(text or "").strip()
        for text in (table_ast.get("display_row_texts") or table_ast.get("row_texts") or [])
        if str(text or "").strip()
    ]
    blob = "\n".join(row_texts)
    compact = _compact_context_text(blob)

    publication_markers = [
        "contentslistsavailableatsciencedirect",
        "journalhomepage",
        "articleinfo",
        "abstract",
        "keywords",
        "openaccess",
    ]
    if sum(1 for marker in publication_markers if marker in compact) >= 2:
        return True

    algorithm_markers = [
        "algorithm",
        "initialization",
        "output",
        "compute",
        "regularizationparameter",
    ]
    math_symbol_rows = sum(
        1
        for text in row_texts
        if re.search(r"[=∑≤≥{}]|\\sum|\\min|\\max|\bs\.?\s*t\.?\b", text, re.IGNORECASE)
    )
    if "algorithm" in compact and (
        sum(1 for marker in algorithm_markers if marker in compact) >= 2
        or math_symbol_rows >= 2
    ):
        return True

    if _visual_structure_grid_lacks_data_body_evidence(table_ast):
        return True

    if _visual_structure_grid_is_two_column_narrative(table_ast, row_texts):
        return True

    if _looks_like_table_title(str(table_ast.get("title", "") or "").strip()):
        return False

    if _visual_structure_grid_is_fragmented_prose_clause_grid(table_ast):
        return True

    if _visual_structure_grid_is_multicolumn_body_prose(table_ast, row_texts):
        return True

    if _visual_structure_grid_is_section_prose(table_ast, row_texts):
        return True

    return False


def _visual_structure_grid_lacks_data_body_evidence(table_ast: dict[str, Any]) -> bool:
    """Reject visual-grid candidates that only contain prose/list/title layout.

    Visual separators prove that a region has two-dimensional alignment; they
    do not prove it owns a data table. A candidate must contain a repeatable
    body: rows with several meaningful cells, compact labels/values, numeric or
    symbolic measures, or flowchart-style multi-column relations. This keeps
    real visual tables/flowcharts while releasing page headers, captions,
    section prose, and references back to text/list ownership.
    """
    grid = table_ast.get("display_grid") or table_ast.get("raw_grid") or table_ast.get("grid") or []
    if not isinstance(grid, list) or not grid:
        return True

    row_count = len([row for row in grid if isinstance(row, list)])
    col_count = max((len(row) for row in grid if isinstance(row, list)), default=0)
    if row_count < 4 or col_count < 2:
        return True

    has_title = _looks_like_table_title(str(table_ast.get("title", "") or "").strip())
    placeholder_cells = 0
    data_body_rows = 0
    header_like_rows = 0
    prose_rows = 0
    sparse_rows = 0
    single_cell_rows = 0
    list_or_outline_rows = 0
    title_or_caption_rows = 0
    leading_outline_rows = 0

    for row_index, row in enumerate(grid):
        if not isinstance(row, list):
            continue
        filled = _row_non_empty_cells(row)
        if not filled:
            sparse_rows += 1
            continue

        texts = [text for _, text in filled]
        placeholder_cells += sum(1 for text in texts if _is_placeholder_header_text(text))
        joined = " ".join(texts).strip()

        if len(filled) <= 1:
            sparse_rows += 1
            single_cell_rows += 1

        if _visual_row_is_title_or_caption_fragment(texts):
            title_or_caption_rows += 1
            continue
        if _visual_row_is_list_or_outline_fragment(texts):
            list_or_outline_rows += 1
            if row_index <= 2:
                leading_outline_rows += 1
            continue
        if _visual_row_is_prose_continuation(texts):
            prose_rows += 1
            continue
        if _visual_row_has_table_header_shape(row):
            header_like_rows += 1
        if _visual_row_has_data_body_shape(row):
            data_body_rows += 1

    structural_rows = max(1, row_count - sparse_rows)
    non_data_rows = prose_rows + list_or_outline_rows + title_or_caption_rows

    if row_count <= 5 and data_body_rows < 2:
        return True

    if placeholder_cells and data_body_rows < max(2, structural_rows // 3):
        return True

    if (
        not has_title
        and placeholder_cells
        and leading_outline_rows >= 1
        and (single_cell_rows + prose_rows + list_or_outline_rows) >= max(3, structural_rows // 2)
    ):
        return True

    if (
        not has_title
        and placeholder_cells
        and data_body_rows <= max(3, header_like_rows + 1)
        and (prose_rows + list_or_outline_rows) >= max(2, structural_rows // 3)
    ):
        return True

    if (
        not has_title
        and leading_outline_rows >= 1
        and single_cell_rows >= max(4, structural_rows // 2)
        and prose_rows >= max(3, structural_rows // 3)
        and title_or_caption_rows >= 1
    ):
        return True

    if not has_title and non_data_rows >= max(3, (structural_rows + 1) // 2) and data_body_rows < max(2, structural_rows // 3):
        return True

    if list_or_outline_rows >= 2 and data_body_rows <= header_like_rows + 1:
        return True

    if prose_rows >= max(3, structural_rows // 2) and data_body_rows < max(2, structural_rows // 3):
        return True

    if data_body_rows == 0:
        return True

    return False


def _looks_like_visual_bridge_between_table_boundaries(
    table_ast: dict[str, Any],
    *,
    previous_table: dict[str, Any] | None,
    next_table: dict[str, Any] | None,
) -> bool:
    """Reject visual-grid candidates that bridge adjacent table boundaries.

    A common failure mode is a residual visual grid starting on the last row of
    an accepted table and extending into the note/caption/prose gap before the
    next content unit. That region has aligned text, but ownership belongs to
    the neighboring table's boundary metadata and body flow, not to a new table.
    """
    source = str(table_ast.get("detection_source") or table_ast.get("detection_method") or "")
    if source != "visual_structure_grid":
        return False

    grid = table_ast.get("display_grid") or table_ast.get("raw_grid") or table_ast.get("grid") or []
    if not isinstance(grid, list) or len(grid) < 3:
        return False

    bbox_raw = table_ast.get("bbox")
    if not bbox_raw or len(bbox_raw) != 4:
        return False
    bbox = tuple(float(value) for value in bbox_raw)

    overlaps_previous = _visual_candidate_overlaps_table_boundary(table_ast, previous_table, edge="bottom")
    overlaps_next = _visual_candidate_overlaps_table_boundary(table_ast, next_table, edge="top")
    if not overlaps_previous and not overlaps_next:
        return False

    row_profiles = [_visual_bridge_row_profile(row) for row in grid if isinstance(row, list)]
    if not row_profiles:
        return False

    repeated_boundary_rows = _visual_candidate_repeated_neighbor_boundary_row_count(
        table_ast,
        previous_table=previous_table,
        next_table=next_table,
    )
    data_rows = max(0, sum(1 for item in row_profiles if item == "data") - repeated_boundary_rows)
    boundary_rows = sum(1 for item in row_profiles if item == "boundary") + repeated_boundary_rows
    caption_rows = sum(1 for item in row_profiles if item == "caption")
    note_rows = sum(1 for item in row_profiles if item == "note")
    prose_rows = sum(1 for item in row_profiles if item == "prose")
    sparse_rows = sum(1 for item in row_profiles if item == "sparse")
    non_table_rows = caption_rows + note_rows + prose_rows + sparse_rows
    structural_rows = len(row_profiles)

    if boundary_rows < 1:
        return False

    if overlaps_previous and overlaps_next and non_table_rows >= 1:
        return True

    if overlaps_previous and non_table_rows >= max(2, structural_rows - boundary_rows - data_rows):
        return True

    return bool(boundary_rows >= 1 and data_rows <= 1 and non_table_rows >= max(2, structural_rows // 2))


def _visual_candidate_repeated_neighbor_boundary_row_count(
    candidate: dict[str, Any],
    *,
    previous_table: dict[str, Any] | None,
    next_table: dict[str, Any] | None,
) -> int:
    grid = candidate.get("display_grid") or candidate.get("raw_grid") or candidate.get("grid") or []
    if not isinstance(grid, list) or not grid:
        return 0
    count = 0
    first_row = next((row for row in grid if isinstance(row, list) and _row_non_empty_cells(row)), None)
    last_row = next((row for row in reversed(grid) if isinstance(row, list) and _row_non_empty_cells(row)), None)
    if first_row is not None and _visual_candidate_row_matches_neighbor_boundary(first_row, previous_table, edge="bottom"):
        count += 1
    if last_row is not None and _visual_candidate_row_matches_neighbor_boundary(last_row, next_table, edge="top"):
        count += 1
    return count


def _visual_candidate_row_matches_neighbor_boundary(
    row: list[Any],
    neighbor: dict[str, Any] | None,
    *,
    edge: str,
) -> bool:
    if neighbor is None:
        return False
    neighbor_grid = neighbor.get("display_grid") or neighbor.get("raw_grid") or neighbor.get("grid") or []
    if not isinstance(neighbor_grid, list) or not neighbor_grid:
        return False
    neighbor_rows = [item for item in neighbor_grid if isinstance(item, list) and _row_non_empty_cells(item)]
    if not neighbor_rows:
        return False
    boundary_row = neighbor_rows[-1] if edge == "bottom" else neighbor_rows[0]
    candidate_tokens = _visual_row_normalized_tokens(row)
    boundary_tokens = _visual_row_normalized_tokens(boundary_row)
    if not candidate_tokens or not boundary_tokens:
        return False
    shared = candidate_tokens.intersection(boundary_tokens)
    return len(shared) >= min(2, len(boundary_tokens)) and len(shared) / max(1, len(boundary_tokens)) >= 0.67


def _visual_row_normalized_tokens(row: list[Any]) -> set[str]:
    tokens: set[str] = set()
    for _, text in _row_non_empty_cells(row):
        compact = re.sub(r"\s+", "", str(text or "").strip()).lower()
        if compact:
            tokens.add(compact)
    return tokens


def _visual_candidate_overlaps_table_boundary(
    candidate: dict[str, Any],
    neighbor: dict[str, Any] | None,
    *,
    edge: str,
) -> bool:
    if neighbor is None:
        return False
    cand_bbox_raw = candidate.get("bbox")
    other_bbox_raw = neighbor.get("bbox")
    if not cand_bbox_raw or len(cand_bbox_raw) != 4 or not other_bbox_raw or len(other_bbox_raw) != 4:
        return False
    cand_bbox = tuple(float(value) for value in cand_bbox_raw)
    other_bbox = tuple(float(value) for value in other_bbox_raw)
    cand_height = max(1.0, cand_bbox[3] - cand_bbox[1])
    other_height = max(1.0, other_bbox[3] - other_bbox[1])
    vertical_overlap = max(0.0, min(cand_bbox[3], other_bbox[3]) - max(cand_bbox[1], other_bbox[1]))
    horizontal_overlap = max(0.0, min(cand_bbox[2], other_bbox[2]) - max(cand_bbox[0], other_bbox[0]))
    horizontal_ratio = horizontal_overlap / max(1.0, min(cand_bbox[2] - cand_bbox[0], other_bbox[2] - other_bbox[0]))
    if horizontal_ratio < 0.45:
        return False
    if vertical_overlap / min(cand_height, other_height) >= 0.12:
        return True
    if edge == "bottom":
        return 0.0 <= cand_bbox[1] - other_bbox[3] <= 8.0
    if edge == "top":
        return 0.0 <= other_bbox[1] - cand_bbox[3] <= 8.0
    return False


def _visual_bridge_row_profile(row: list[Any]) -> str:
    filled = [text for _, text in _row_non_empty_cells(row)]
    if not filled:
        return "sparse"
    if _visual_row_is_title_or_caption_fragment(filled):
        return "caption"
    if _visual_row_is_explicit_note_or_source(filled):
        return "note"
    if _visual_row_is_fragmented_sentence_bridge(filled):
        return "prose"
    if _visual_row_is_prose_continuation(filled):
        return "prose"
    if _visual_row_has_data_body_shape(row):
        return "data"
    if len(filled) <= 1:
        return "boundary"
    numeric_or_value_cells = sum(1 for text in filled if _looks_like_tabular_value_cell(text))
    compact_label_cells = sum(1 for text in filled if _visual_cell_is_compact_label(text))
    if numeric_or_value_cells >= 1 and compact_label_cells >= 1:
        return "data"
    return "boundary"


def _visual_row_is_fragmented_sentence_bridge(texts: list[str]) -> bool:
    if len(texts) < 4:
        return False
    joined = " ".join(str(text or "").strip() for text in texts if str(text or "").strip())
    if len(joined) < 70:
        return False
    latin_words = re.findall(r"[A-Za-z]{2,}", joined)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", joined)
    if len(latin_words) < 10 and len(cjk_chars) < 24:
        return False
    sentence_cues = len(
        re.findall(
            r"\b(?:a|an|the|of|for|with|where|when|if|then|only|however|because|therefore|and|or|in|to|from|by|as)\b",
            joined,
            re.IGNORECASE,
        )
    )
    punctuation = len(re.findall(r"[.;:!?,\u3002\uff0c\uff1b\uff1a\uff01\uff1f]", joined))
    numeric_or_value_cells = sum(1 for text in texts if _looks_like_tabular_value_cell(text))
    return sentence_cues >= 3 and (punctuation >= 1 or sentence_cues >= 5) and numeric_or_value_cells <= 1


def _visual_row_is_explicit_note_or_source(texts: list[str]) -> bool:
    joined = " ".join(str(text or "").strip() for text in texts if str(text or "").strip())
    if not joined:
        return False
    return bool(
        re.match(
            r"^\s*(?:source|sources|note|notes|remark|remarks|说明|注|备注|数据为|显示的数据为)\s*[:：]",
            joined,
            re.IGNORECASE,
        )
        or re.match(r"^\s*[*#$+\u2020\u2021a-zA-Z]\s*[-:：]", joined)
    )


def _visual_row_is_title_or_caption_fragment(texts: list[str]) -> bool:
    if not texts:
        return False
    joined = " ".join(str(text or "").strip() for text in texts if str(text or "").strip())
    if not joined:
        return False
    if _looks_like_table_title(joined):
        return True
    return any(_looks_like_table_title(text) for text in texts)


def _visual_row_is_list_or_outline_fragment(texts: list[str]) -> bool:
    if not texts:
        return False
    first = str(texts[0] or "").strip()
    if not first:
        return False
    if _outline_token_depth(first) >= 1:
        return True
    if re.fullmatch(r"(?:\d+|[A-Za-z]|[ivxlcdmIVXLCDM]+)[.)、]", first):
        return True
    return bool(re.match(r"^\d+(?:\.\d+)*\.?$", first) and len(texts) >= 2)


def _visual_row_is_prose_continuation(texts: list[str]) -> bool:
    joined = " ".join(str(text or "").strip() for text in texts if str(text or "").strip())
    if not joined:
        return False
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", joined))
    latin_words = len(re.findall(r"[A-Za-z]{3,}", joined))
    punctuation = len(re.findall(r"[.;:!?,\u3002\uff0c\uff1b\uff1a\uff01\uff1f]", joined))
    sentence_markers = bool(
        re.search(
            r"\b(?:and|or|of|in|the|that|for|with|because|compared|according|should|must|may|used|using)\b",
            joined,
            re.IGNORECASE,
        )
    )
    if len(texts) <= 2 and (cjk_chars >= 16 or latin_words >= 10 or punctuation >= 2):
        return True
    return bool((cjk_chars >= 24 or latin_words >= 14) and (punctuation >= 1 or sentence_markers))


def _visual_row_has_data_body_shape(row: list[Any]) -> bool:
    filled = [text for _, text in _row_non_empty_cells(row)]
    if len(filled) < 2:
        return False
    if any(_is_placeholder_header_text(text) for text in filled):
        return False
    if _visual_row_is_title_or_caption_fragment(filled):
        return False
    if _visual_row_is_list_or_outline_fragment(filled):
        return False
    if _visual_row_is_prose_continuation(filled):
        return False

    numeric_or_value_cells = sum(1 for text in filled if _looks_like_tabular_value_cell(text))
    compact_label_cells = sum(1 for text in filled if _visual_cell_is_compact_label(text))
    relation_cells = sum(1 for text in filled if re.search(r"[-=+*/%<>≤≥→←↔↑↓]|\\to|\\rightarrow", text))

    if numeric_or_value_cells >= 1 and compact_label_cells >= 1:
        return True
    if relation_cells >= 1 and compact_label_cells >= 2:
        return True
    if len(filled) >= 3 and compact_label_cells >= 2:
        return True
    return bool(_row_non_empty_ratio(row) >= 0.55 and compact_label_cells >= 2)


def _visual_cell_is_compact_label(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    if _is_placeholder_header_text(candidate):
        return False
    if len(candidate) > 56:
        return False
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", candidate))
    latin_words = len(re.findall(r"[A-Za-z]{2,}", candidate))
    return cjk_chars <= 14 and latin_words <= 6


def _visual_structure_grid_is_two_column_narrative(
    table_ast: dict[str, Any],
    row_texts: list[str],
) -> bool:
    if int(table_ast.get("col_count", 0) or 0) != 2:
        return False
    if int(table_ast.get("row_count", 0) or 0) < 8:
        return False
    narrative_rows = 0
    for row in table_ast.get("display_grid") or table_ast.get("grid") or []:
        if not isinstance(row, list) or len(row) < 2:
            continue
        values = [str(cell or "").strip() for cell in row[:2]]
        if all(len(re.findall(r"[A-Za-z\u4e00-\u9fff]+", value)) >= 4 for value in values):
            narrative_rows += 1
    heading_like_rows = sum(
        1
        for text in row_texts[:8]
        if text and len(text) <= 80 and not re.search(r"\d", text) and not re.search(r"[|].*[|]", text)
    )
    return narrative_rows >= 4 or (narrative_rows >= 3 and heading_like_rows >= 2)


def _visual_structure_grid_is_section_prose(
    table_ast: dict[str, Any],
    row_texts: list[str],
) -> bool:
    if int(table_ast.get("col_count", 0) or 0) > 2:
        return False
    if int(table_ast.get("row_count", 0) or 0) < 5:
        return False
    section_like_rows = sum(
        1
        for text in row_texts[:6]
        if re.match(r"^\s*\d+(?:\.\d+)+\s+", text)
        or re.match(r"^\s*\d+(?:\.\d+)+\s*\|", text)
    )
    prose_rows = 0
    for text in row_texts:
        cleaned = text.replace("|", " ")
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
        latin_words = len(re.findall(r"[A-Za-z]{3,}", cleaned))
        punctuation = len(re.findall(r"[，。；：,.]", cleaned))
        if cjk_count >= 12 or latin_words >= 10 or punctuation >= 2:
            prose_rows += 1
    return section_like_rows >= 1 and prose_rows >= max(3, len(row_texts) // 2)


def _visual_structure_grid_is_multicolumn_body_prose(
    table_ast: dict[str, Any],
    row_texts: list[str],
) -> bool:
    if int(table_ast.get("col_count", 0) or 0) < 3:
        return False
    if int(table_ast.get("row_count", 0) or 0) < 6:
        return False
    if str(table_ast.get("title", "") or "").strip():
        return False

    grid = table_ast.get("display_grid") or table_ast.get("grid") or []
    first_filled_row = next(
        (row for row in grid if isinstance(row, list) and len(_row_non_empty_cells(row)) >= 2),
        None,
    )
    if first_filled_row is not None:
        first_texts = [text for _, text in _row_non_empty_cells(first_filled_row)]
        first_word_count = sum(len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", text)) for text in first_texts)
        if (
            len(first_texts) >= min(3, int(table_ast.get("col_count", 0) or 0))
            and all(len(text) <= 46 for text in first_texts)
            and first_word_count <= 9
            and _visual_row_has_table_header_shape(first_filled_row)
        ):
            return False

    prose_rows = 0
    sentence_continuation_rows = 0
    schema_like_rows = 0
    numeric_or_symbol_rows = 0
    for row in grid:
        if not isinstance(row, list):
            continue
        filled = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
        if len(filled) < 2:
            continue
        joined = " ".join(filled)
        latin_words = len(re.findall(r"[A-Za-z]{3,}", joined))
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", joined))
        punctuation = len(re.findall(r"[.;:!?。；：！？]", joined))
        if latin_words >= 8 or cjk_chars >= 16 or punctuation >= 1:
            prose_rows += 1
        if re.search(r"\b(?:and|or|of|in|the|that|for|with|because|compared|results?)\b", joined, re.IGNORECASE):
            sentence_continuation_rows += 1
        if _visual_row_has_table_header_shape(row) and latin_words <= 8 and cjk_chars < 12:
            schema_like_rows += 1
        if sum(1 for text in filled if re.fullmatch(r"[-+→←=•\d\s.,%]+", text)) >= 1:
            numeric_or_symbol_rows += 1

    if prose_rows < max(4, len(row_texts) // 3):
        return False
    if sentence_continuation_rows < max(3, prose_rows // 2):
        return False
    if schema_like_rows >= max(2, prose_rows // 2) and numeric_or_symbol_rows >= 2:
        return False
    return True


def _visual_structure_grid_is_fragmented_prose_clause_grid(table_ast: dict[str, Any]) -> bool:
    """Reject visual grids formed by one paragraph sliced into short clauses.

    The visual-structure detector intentionally runs on captionless regions, so
    it needs an evidence gate that distinguishes repeated table records from a
    paragraph whose lines happen to align into columns. Real visual tables keep
    schema/value evidence or repeatable compact records; prose-clause grids have
    many short phrase cells, strong sentence glue across cells, and almost no
    numeric or symbolic value evidence.
    """
    if int(table_ast.get("col_count", 0) or 0) < 3:
        return False
    if str(table_ast.get("title") or table_ast.get("caption_text") or "").strip():
        return False

    grid = table_ast.get("display_grid") or table_ast.get("raw_grid") or table_ast.get("grid") or []
    rows = [row for row in grid if isinstance(row, list)]
    if len(rows) < 4:
        return False

    multi_cell_rows = 0
    fragmented_rows = 0
    sentence_glue_rows = 0
    value_rows = 0
    compact_phrase_cells = 0
    filled_cell_count = 0

    for row in rows:
        filled = [text for _, text in _row_non_empty_cells(row)]
        if len(filled) >= 3:
            multi_cell_rows += 1
        if not filled:
            continue

        filled_cell_count += len(filled)
        compact_phrase_cells += sum(
            1
            for text in filled
            if len(str(text or "").strip()) <= 32
            and 1 <= len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", str(text or ""))) <= 4
        )
        if sum(1 for text in filled if _looks_like_tabular_value_cell(text)) >= 1:
            value_rows += 1

        joined = " ".join(str(text or "").strip() for text in filled if str(text or "").strip())
        if len(filled) >= 3 and (
            _visual_row_is_fragmented_sentence_bridge(filled)
            or re.search(
                r"\b(?:the|this|that|it|does|not|of|for|or|and|on|in|to|with|if|such|an|any|may|be)\b",
                joined,
                re.IGNORECASE,
            )
        ):
            sentence_glue_rows += 1
        if len(filled) >= 3 and len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", joined)) >= 7:
            fragmented_rows += 1

    if multi_cell_rows < 3 or fragmented_rows < 3:
        return False
    if filled_cell_count == 0:
        return False
    if value_rows > max(1, len(rows) // 4):
        return False

    compact_phrase_ratio = compact_phrase_cells / max(1, filled_cell_count)
    return sentence_glue_rows >= max(2, fragmented_rows // 2) and compact_phrase_ratio >= 0.60


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


def _looks_like_cross_column_algorithm_formula_false_positive(table_ast: dict[str, Any]) -> bool:
    """Reject mixed prose/formula/algorithm regions mis-promoted as tables.

    Borderless word clustering can occasionally join a literature page's left
    article column with a right algorithm box because both have stable x anchors.
    A real table should preserve one ownership domain; this profile demotes only
    candidates that simultaneously carry narrative prose, display-equation
    labels, and algorithm-pseudocode signals without a real data-table schema.
    """

    source = str(table_ast.get("detection_source") or table_ast.get("detection_method") or "").strip()
    if source not in {"word_clustering", "text_aligned_borderless_grid"}:
        return False
    if table_ast.get("is_continuation", False):
        return False
    if _has_wide_schema_data_table_evidence(table_ast):
        return False

    row_count = int(table_ast.get("row_count", 0) or 0)
    col_count = int(table_ast.get("col_count", 0) or 0)
    if row_count < 4 or col_count < 4:
        return False

    title = str(table_ast.get("title") or "").strip()
    if _looks_like_table_title(title):
        return False
    title_is_narrative = bool(
        title
        and len(re.findall(r"[A-Za-z]{2,}", title)) >= 5
        and re.search(r"\b(?:the|this|that|where|with|method|problem|function|algorithm|constraint|solution|feature|gradient|optimization)\b", title, re.IGNORECASE)
    )

    grid = table_ast.get("display_grid") or table_ast.get("raw_grid") or table_ast.get("grid") or []
    if not isinstance(grid, list) or not grid:
        return False

    filled_cells: list[str] = []
    row_texts: list[str] = []
    algorithm_rows = 0
    numbered_step_rows = 0
    formula_label_rows = 0
    formula_symbol_rows = 0
    narrative_rows = 0
    schema_like_rows = 0
    sparse_rows = 0

    for row in grid:
        if not isinstance(row, list):
            continue
        cells = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
        if not cells:
            continue
        if len(cells) <= max(1, len(row) // 2):
            sparse_rows += 1
        row_text = " ".join(cells)
        row_texts.append(row_text)
        filled_cells.extend(cells)

        if re.search(r"\bAlgorithm\s+\d+\b|\bInitialization\b|\bOutput\b|\bEnd\b", row_text, re.IGNORECASE):
            algorithm_rows += 1
        if re.search(r"^\s*\d+\s*[\.)]\s+\b(?:Compute|Update|Repeat|Until|For|if|end|Calculate|Set)\b", row_text, re.IGNORECASE):
            numbered_step_rows += 1
        if re.search(r"\(\s*\d{1,3}\s*\)", row_text):
            formula_label_rows += 1
        if re.search(r"[=∑Σλσθαβγ]|\b(?:min|max|log|exp)\b", row_text, re.IGNORECASE):
            formula_symbol_rows += 1
        words = re.findall(r"[A-Za-z]{2,}", row_text)
        if len(words) >= 7 and re.search(
            r"\b(?:the|this|that|where|with|method|problem|function|algorithm|constraint|solution|feature|gradient)\b",
            row_text,
            re.IGNORECASE,
        ):
            narrative_rows += 1
        if len(cells) >= 3 and sum(1 for text in cells if _looks_like_tabular_value_cell(text)) >= 2:
            schema_like_rows += 1

    if len(filled_cells) < 8:
        return False

    joined = " ".join(row_texts + ([title] if title else []))
    has_algorithm_domain = algorithm_rows >= 1 or numbered_step_rows >= 2
    has_formula_domain = formula_label_rows >= 1 and formula_symbol_rows >= 2
    has_narrative_domain = narrative_rows >= 2 or title_is_narrative
    has_mixed_column_ownership = bool(
        has_algorithm_domain
        and re.search(r"\(\s*\d{1,3}\s*\)", joined)
        and re.search(r"\b(?:where|constraint|solution|problem|method)\b", joined, re.IGNORECASE)
    )
    if schema_like_rows >= max(3, row_count // 3):
        return False
    return (
        has_algorithm_domain
        and has_formula_domain
        and has_narrative_domain
        and has_mixed_column_ownership
        and sparse_rows >= 1
    )


def _looks_like_text_aligned_narrative_formula_projection_false_positive(table_ast: dict[str, Any]) -> bool:
    source = str(table_ast.get("detection_source") or table_ast.get("detection_method") or "")
    if source != "text_aligned_borderless_grid":
        return False
    if table_ast.get("is_continuation", False):
        return False
    if str(table_ast.get("title") or "").strip():
        return False
    if str(table_ast.get("local_context_signal") or "") != "narrative_barrier":
        return False

    row_count = int(table_ast.get("row_count") or table_ast.get("display_row_count") or 0)
    col_count = int(table_ast.get("col_count") or 0)
    if row_count > 5 or col_count < 6:
        return False

    grid = table_ast.get("display_grid") or table_ast.get("raw_grid") or table_ast.get("grid") or []
    if not isinstance(grid, list) or len(grid) < 3:
        return False

    header_texts = [
        str(cell.get("text", "")).strip()
        for cell in table_ast.get("header", []) or []
        if str(cell.get("text", "")).strip()
    ]
    placeholder_headers = sum(1 for text in header_texts if _is_placeholder_header_text(text))
    prose_header_tokens = sum(
        1
        for text in header_texts
        if re.search(r"\b(?:this|method|uses?|class|as|the|where|function|kernel|different|than)\b", text, re.IGNORECASE)
    )

    filled_cells: list[str] = []
    formula_like_cells = 0
    prose_fragment_cells = 0
    short_word_cells = 0
    for row in grid:
        if not isinstance(row, list):
            continue
        for cell in row:
            text = str(cell or "").strip()
            if not text:
                continue
            filled_cells.append(text)
            if re.search(r"[=∑∈≤≥√∫]|\b[a-zA-Z]\s*\(", text):
                formula_like_cells += 1
            words = re.findall(r"[A-Za-z]{2,}", text)
            if len(words) >= 2 and re.search(
                r"\b(?:the|this|method|class|feature|selection|where|kernel|function|many|different|than|been|used)\b",
                text,
                re.IGNORECASE,
            ):
                prose_fragment_cells += 1
            if len(words) <= 3 and len(text) <= 24:
                short_word_cells += 1
    if len(filled_cells) < 8:
        return False

    compact_text = " ".join(filled_cells)
    has_formula_label_fragment = bool(re.search(r"\(\s*\d+\s*\)", compact_text))
    short_fragment_ratio = short_word_cells / max(1, len(filled_cells))
    return (
        placeholder_headers >= 1
        or prose_header_tokens >= 3
        or (
            short_fragment_ratio >= 0.55
            and prose_fragment_cells >= 4
            and (formula_like_cells >= 1 or has_formula_label_fragment)
        )
    )


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
    if str(table_ast.get("detection_source") or table_ast.get("detection_method") or "") in {
        "embedded_image_ocr",
        "vector_ocr",
    }:
        return False
    if _has_wide_schema_data_table_evidence(table_ast):
        return False
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
    if _has_wide_schema_data_table_evidence(table_ast):
        return False
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


def _has_wide_schema_data_table_evidence(table_ast: dict[str, Any]) -> bool:
    """Protect real wide data tables from formula/layout false-positive filters."""

    source = str(table_ast.get("detection_source") or table_ast.get("detection_method") or "").strip()
    if source not in {"text_aligned_borderless_grid", "word_clustering", "structured_text_region", "visual_structure_grid"}:
        return False

    row_count = int(table_ast.get("row_count", 0) or 0)
    col_count = int(table_ast.get("col_count", 0) or 0)
    if row_count < 3 or col_count < 5:
        return False

    grid = table_ast.get("display_grid") or table_ast.get("raw_grid") or table_ast.get("grid") or []
    if not isinstance(grid, list) or len(grid) < 3:
        return False

    header_texts = [
        str(cell.get("text", "")).strip()
        for cell in (table_ast.get("header") or [])
        if isinstance(cell, dict) and str(cell.get("text", "")).strip()
    ]
    if not header_texts and isinstance(grid[0], list):
        header_texts = [str(cell or "").strip() for cell in grid[0] if str(cell or "").strip()]

    semantic_header_texts = [text for text in header_texts if not _is_placeholder_header_text(text)]
    compact_schema_headers = [
        text
        for text in semantic_header_texts
        if len(text) <= 80
        and len(text.split()) <= 8
        and re.search(r"[A-Za-z\u4e00-\u9fff]", text)
    ]
    first_row_cells = [str(cell or "").strip() for cell in (grid[0] if isinstance(grid[0], list) else [])]
    first_row_filled = [text for text in first_row_cells if text]
    first_row_numeric_values = sum(1 for text in first_row_filled[1:] if _looks_like_table_value_atom(text))
    has_schema_header = len(compact_schema_headers) >= 3
    has_stub_value_header = (
        len(first_row_filled) >= 4
        and bool(re.search(r"[A-Za-z\u4e00-\u9fff]", first_row_filled[0]))
        and first_row_numeric_values >= 3
    )
    if not has_schema_header and not has_stub_value_header:
        return False

    body_rows = [row for row in grid[1:] if isinstance(row, list)]
    dense_body_rows = 0
    identifier_or_value_rows = 0
    for row in body_rows:
        values = [str(cell or "").strip() for cell in row[:col_count]]
        filled = [text for text in values if text]
        if len(filled) >= max(3, min(5, col_count - 1)):
            dense_body_rows += 1
        value_atoms = sum(1 for text in filled if _looks_like_table_value_atom(text))
        text_atoms = sum(1 for text in filled if re.search(r"[A-Za-z\u4e00-\u9fff]", text))
        if value_atoms >= 2 or (value_atoms >= 1 and text_atoms >= 1):
            identifier_or_value_rows += 1

    return dense_body_rows >= 2 and identifier_or_value_rows >= 2


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


def _arbitrate_visual_structure_candidate_ownership(
    raw_evidence: RawTableEvidence,
) -> dict[str, Any]:
    """Classify visual grid candidates before table projection.

    A stable visual grid only proves two-dimensional layout. It does not prove
    the region is a data table: charts also expose axes, tick labels, plotted
    numeric values, source rows, and figure captions. This arbitration is the
    first page-ownership boundary for caption-less visual candidates.
    """
    if str(raw_evidence.source or "") != "visual_structure_grid":
        return {
            "primary_type": "data_table",
            "confidence": 1.0,
            "reasons": ["non_visual_structure_candidate"],
        }

    raw_rows = raw_evidence.raw_data or []
    if not raw_rows:
        return {
            "primary_type": "layout_matrix",
            "confidence": 0.3,
            "reasons": ["empty_visual_grid"],
        }

    row_texts = [
        " ".join(text for _, text in _row_non_empty_cells(row)).strip()
        for row in raw_rows
        if _row_non_empty_cells(row)
    ]
    flat_texts = [
        text
        for row in raw_rows
        for _, text in _row_non_empty_cells(row)
    ]
    total_cells = sum(len(row) for row in raw_rows if isinstance(row, list))
    non_empty_count = len(flat_texts)
    empty_ratio = 1.0 - (non_empty_count / max(1, total_cells))

    figure_rows = sum(1 for text in row_texts if _looks_like_figure_caption_or_title_text(text))
    source_rows = sum(1 for text in row_texts if _looks_like_chart_source_text(text))
    tick_rows = sum(1 for text in row_texts if _looks_like_axis_tick_row_text(text))
    plotted_numeric_rows = sum(1 for row in raw_rows if _looks_like_sparse_plotted_numeric_row(row))
    axis_unit_rows = sum(1 for text in row_texts if _looks_like_chart_axis_or_unit_text(text))
    repeated_year_rows = sum(1 for text in row_texts if len(re.findall(r"\b(?:19|20)\d{2}\\b", text)) >= 3)

    chart_score = 0.0
    chart_reasons: list[str] = []
    if figure_rows:
        chart_score += min(3.0, figure_rows * 1.2)
        chart_reasons.append("figure_caption_or_title_rows")
    if source_rows:
        chart_score += min(2.0, source_rows * 0.8)
        chart_reasons.append("source_rows")
    if tick_rows:
        chart_score += min(2.2, tick_rows * 0.8)
        chart_reasons.append("axis_tick_rows")
    if plotted_numeric_rows >= 2:
        chart_score += min(2.5, plotted_numeric_rows * 0.45)
        chart_reasons.append("sparse_plotted_numeric_rows")
    if axis_unit_rows:
        chart_score += min(1.4, axis_unit_rows * 0.7)
        chart_reasons.append("axis_or_unit_rows")
    if repeated_year_rows:
        chart_score += min(1.2, repeated_year_rows * 0.6)
        chart_reasons.append("repeated_year_axis_rows")
    if empty_ratio >= 0.35 and plotted_numeric_rows >= 2:
        chart_score += 0.8
        chart_reasons.append("sparse_numeric_layout")

    table_score = 0.0
    table_reasons: list[str] = []
    header_like_rows = sum(1 for row in raw_rows[:3] if _visual_row_has_table_header_shape(row))
    fullish_rows = sum(1 for row in raw_rows if _row_non_empty_ratio(row) >= 0.55)
    body_text_rows = sum(1 for row in raw_rows if _visual_row_has_multi_column_text_body(row))
    if header_like_rows:
        table_score += min(2.0, header_like_rows * 1.0)
        table_reasons.append("header_like_rows")
    if fullish_rows >= max(3, len(raw_rows) // 3):
        table_score += 1.8
        table_reasons.append("stable_fullish_rows")
    if body_text_rows >= max(2, len(raw_rows) // 3):
        table_score += 1.5
        table_reasons.append("multi_column_text_body")
    if raw_evidence.horizontal_lines or raw_evidence.vertical_lines:
        table_score += 0.8
        table_reasons.append("visual_grid_lines")

    primary_type = "data_table"
    if chart_score >= max(3.0, table_score + 1.25):
        primary_type = "chart_figure"
    elif table_score < 2.0 and chart_score >= 1.5:
        primary_type = "layout_matrix"

    confidence = 0.5
    if primary_type == "chart_figure":
        confidence = min(0.98, 0.55 + (chart_score - table_score) / 10.0)
    elif primary_type == "data_table":
        confidence = min(0.98, 0.55 + (table_score - chart_score) / 10.0)

    return {
        "primary_type": primary_type,
        "confidence": round(confidence, 3),
        "chart_score": round(chart_score, 3),
        "table_score": round(table_score, 3),
        "empty_ratio": round(empty_ratio, 3),
        "reasons": chart_reasons if primary_type != "data_table" else table_reasons,
    }


def _arbitrate_table_candidate_ownership(
    raw_evidence: RawTableEvidence,
    table_ast: dict[str, Any] | None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify any table candidate before it is accepted as data table.

    Detector source is only provenance. Ownership is decided from candidate
    geometry/content: sparse plotted values, axis/category rows, figure/source
    context, title support, and narrative-continuation pressure.
    """
    if not raw_evidence or not table_ast:
        return {"primary_type": "layout_matrix", "confidence": 0.3, "reasons": ["empty_candidate"]}

    source = str(raw_evidence.source or "")
    if source == "visual_structure_grid":
        visual_ownership = _arbitrate_visual_structure_candidate_ownership(raw_evidence)
        if visual_ownership.get("primary_type") != "data_table":
            return visual_ownership

    flat_locator_ownership = _arbitrate_flat_page_locator_list_candidate_ownership(
        raw_evidence,
        table_ast,
        context or {},
    )
    if flat_locator_ownership.get("primary_type") != "data_table":
        return flat_locator_ownership

    cross_boundary_ownership = _arbitrate_cross_boundary_chart_body_candidate_ownership(
        raw_evidence,
        table_ast,
        context or {},
    )
    if cross_boundary_ownership.get("primary_type") != "data_table":
        return cross_boundary_ownership

    grid = table_ast.get("display_grid") or table_ast.get("raw_grid") or raw_evidence.raw_data or []
    if not isinstance(grid, list) or not grid:
        return {"primary_type": "layout_matrix", "confidence": 0.4, "reasons": ["empty_grid"]}

    row_texts = [
        " ".join(text for _, text in _row_non_empty_cells(row)).strip()
        for row in grid
        if isinstance(row, list) and _row_non_empty_cells(row)
    ]
    flat_texts = [
        text
        for row in grid
        if isinstance(row, list)
        for _, text in _row_non_empty_cells(row)
    ]
    total_cells = sum(len(row) for row in grid if isinstance(row, list))
    non_empty_count = len(flat_texts)
    empty_ratio = 1.0 - (non_empty_count / max(1, total_cells))
    title_text = str((context or {}).get("title_text", "") or table_ast.get("title", "") or "").strip()
    preceding_text = str((context or {}).get("preceding_text", "") or "").strip()
    context_text = " ".join(text for text in (title_text, preceding_text) if text)
    has_explicit_table_support = _has_explicit_table_title_signal(table_ast, context or {})
    has_figure_context = any(
        _looks_like_figure_caption_or_title_text(text)
        for text in [context_text, *row_texts[:3], *row_texts[-3:]]
        if text
    )

    sparse_numeric_rows = sum(1 for row in grid if isinstance(row, list) and _looks_like_sparse_plotted_numeric_row(row))
    axis_tick_rows = sum(1 for text in row_texts if _looks_like_axis_tick_row_text(text))
    source_rows = sum(1 for text in row_texts if _looks_like_chart_source_text(text))
    axis_unit_rows = sum(1 for text in row_texts if _looks_like_chart_axis_or_unit_text(text))
    numeric_tokens = sum(1 for text in flat_texts if _looks_like_chart_numeric_token(text))
    numeric_ratio = numeric_tokens / max(1, non_empty_count)
    narrative_ratio = _narrative_cell_ratio(raw_evidence)
    body_text_rows = sum(1 for row in grid if isinstance(row, list) and _visual_row_has_multi_column_text_body(row))

    table_score = 0.0
    table_reasons: list[str] = []
    semantic_header_texts = [
        str(cell.get("text", "")).strip()
        for cell in (table_ast.get("header") or [])
        if str(cell.get("text", "")).strip() and not _is_placeholder_header_text(str(cell.get("text", "")).strip())
    ]
    if has_explicit_table_support:
        table_score += 3.0
        table_reasons.append("explicit_table_title_support")
    if len(semantic_header_texts) >= 2:
        table_score += 1.2
        table_reasons.append("semantic_header")
    if sum(1 for row in grid if isinstance(row, list) and _row_non_empty_ratio(row) >= 0.55) >= max(3, len(grid) // 3):
        table_score += 1.0
        table_reasons.append("stable_filled_rows")

    chart_score = 0.0
    chart_reasons: list[str] = []
    if has_figure_context:
        chart_score += 2.4
        chart_reasons.append("figure_context")
    if sparse_numeric_rows >= 3:
        chart_score += min(2.6, sparse_numeric_rows * 0.45)
        chart_reasons.append("sparse_plotted_numeric_rows")
    if axis_tick_rows:
        chart_score += min(1.8, axis_tick_rows * 0.8)
        chart_reasons.append("axis_tick_rows")
    if source_rows:
        chart_score += min(1.2, source_rows * 0.6)
        chart_reasons.append("source_rows")
    if axis_unit_rows:
        chart_score += min(1.0, axis_unit_rows * 0.5)
        chart_reasons.append("axis_or_unit_rows")
    if empty_ratio >= 0.45 and numeric_ratio >= 0.55:
        chart_score += 1.0
        chart_reasons.append("sparse_numeric_grid")
    has_semantic_header = len(semantic_header_texts) >= 2
    has_chart_or_axis_context = has_figure_context or axis_tick_rows > 0 or sparse_numeric_rows > 0 or source_rows > 0
    if (
        body_text_rows >= max(3, len(grid) // 3)
        and not has_explicit_table_support
        and not has_semantic_header
        and has_chart_or_axis_context
    ):
        chart_score += 2.5
        chart_reasons.append("narrative_body_projection")

    if chart_score >= max(3.0, table_score + 1.0):
        return {
            "primary_type": "chart_figure",
            "confidence": round(min(0.98, 0.55 + (chart_score - table_score) / 10.0), 3),
            "chart_score": round(chart_score, 3),
            "table_score": round(table_score, 3),
            "empty_ratio": round(empty_ratio, 3),
            "reasons": chart_reasons,
        }
    return {
        "primary_type": "data_table",
        "confidence": round(min(0.98, 0.55 + max(0.0, table_score - chart_score) / 10.0), 3),
        "chart_score": round(chart_score, 3),
        "table_score": round(table_score, 3),
        "empty_ratio": round(empty_ratio, 3),
        "reasons": table_reasons,
    }


def _arbitrate_flat_page_locator_list_candidate_ownership(
    raw_evidence: RawTableEvidence,
    table_ast: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Classify flat contents/index lists before weak table projection.

    A contents page can be visually indistinguishable from a borderless
    two/three-column table: left title text and a right locator column. The
    owner is navigation when nearly every row ends in a page locator, there is
    no explicit table title/schema, and the non-locator cells look like short
    section titles rather than measured data.
    """
    source = str(raw_evidence.source or table_ast.get("detection_source") or table_ast.get("detection_method") or "")
    if source not in {"structured_text_region", "text_aligned_borderless_grid", "visual_structure_grid", "word_clustering"}:
        return {"primary_type": "data_table", "confidence": 0.55, "reasons": []}
    if _has_explicit_table_title_signal(table_ast, context):
        return {"primary_type": "data_table", "confidence": 0.72, "reasons": ["explicit_table_title_support"]}

    semantic_signals = dict(table_ast.get("semantic_signals") or {})
    if bool(semantic_signals.get("page_schema_header")):
        return {"primary_type": "data_table", "confidence": 0.70, "reasons": ["page_schema_header"]}

    grid = table_ast.get("display_grid") or table_ast.get("raw_grid") or raw_evidence.raw_data or []
    rows = [row for row in grid if isinstance(row, list) and _row_non_empty_cells(row)]
    if len(rows) < 3:
        return {"primary_type": "data_table", "confidence": 0.55, "reasons": []}

    locator_rows = 0
    title_like_rows = 0
    non_locator_numeric_rows = 0
    for row in rows:
        filled = _row_non_empty_cells(row)
        if not filled:
            continue
        locator_index, locator_text = filled[-1]
        if not _looks_like_page_locator(locator_text):
            continue
        locator_rows += 1
        body_cells = [text for index, text in filled if index != locator_index and text.strip()]
        body_text = " ".join(body_cells).strip()
        if _looks_like_flat_toc_entry_title_text(body_text):
            title_like_rows += 1
        if any(re.search(r"\d", text) for text in body_cells):
            non_locator_numeric_rows += 1

    locator_ratio = locator_rows / max(1, len(rows))
    title_like_ratio = title_like_rows / max(1, locator_rows)
    has_toc_context = bool(semantic_signals.get("toc_title_support")) or bool(context.get("toc_context"))
    if locator_ratio >= 0.75 and title_like_ratio >= 0.70 and non_locator_numeric_rows == 0:
        confidence = 0.88 if has_toc_context else 0.80
        return {
            "primary_type": "toc_outline",
            "confidence": confidence,
            "reasons": ["flat_page_locator_list", "short_title_rows"],
        }

    return {"primary_type": "data_table", "confidence": 0.55, "reasons": []}


def _looks_like_flat_toc_entry_title_text(text: str) -> bool:
    candidate = " ".join(str(text or "").strip().split())
    if not candidate or len(candidate) > 140:
        return False
    if re.search(r"\b(?:mean|average|total|rate|ratio|amount|score|value|count|percentage|percent)\b", candidate, re.IGNORECASE):
        return False
    if re.search(r"[=<>±×÷]|(?:\d+\s*%)", candidate):
        return False
    words = re.findall(r"[A-Za-z\u4e00-\u9fff][A-Za-z\u4e00-\u9fff,-]*", candidate)
    return 1 <= len(words) <= 12


def _arbitrate_cross_boundary_chart_body_candidate_ownership(
    raw_evidence: RawTableEvidence,
    table_ast: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Release weak table candidates that join chart evidence with body flow.

    The signal is structural rather than document-specific: captionless
    text-derived candidates may be two-dimensional, but if their rows contain
    chart legends/axis-style labels and then cross into a section heading or
    prose continuation, the region has multiple owners and must not be emitted
    as one data table.
    """
    source = str(raw_evidence.source or table_ast.get("detection_source") or table_ast.get("detection_method") or "")
    if source not in {"structured_text_region", "text_aligned_borderless_grid", "visual_structure_grid"}:
        return {"primary_type": "data_table", "confidence": 0.55, "reasons": []}
    if _has_explicit_table_title_signal(table_ast, context):
        return {"primary_type": "data_table", "confidence": 0.72, "reasons": ["explicit_table_title_support"]}
    if str(table_ast.get("title") or table_ast.get("caption_text") or "").strip():
        return {"primary_type": "data_table", "confidence": 0.70, "reasons": ["table_title"]}
    if _has_wide_schema_data_table_evidence(table_ast):
        return {"primary_type": "data_table", "confidence": 0.74, "reasons": ["wide_schema_value_matrix"]}

    grid = table_ast.get("display_grid") or table_ast.get("raw_grid") or raw_evidence.raw_data or []
    rows = [row for row in grid if isinstance(row, list)]
    if len(rows) < 3:
        return {"primary_type": "data_table", "confidence": 0.55, "reasons": []}

    row_texts = [" ".join(text for _, text in _row_non_empty_cells(row)).strip() for row in rows]
    if not any(row_texts):
        return {"primary_type": "layout_matrix", "confidence": 0.4, "reasons": ["empty_grid"]}

    chart_like_rows = [
        text
        for text in row_texts
        if text
        and (
            _looks_like_axis_tick_row_text(text)
            or _looks_like_chart_axis_or_unit_text(text)
            or _looks_like_chart_legend_or_series_row_text(text)
            or _looks_like_chart_source_text(text)
        )
    ]
    body_boundary_rows = [
        text
        for text in row_texts
        if text and _looks_like_region_boundary_body_or_heading_row_text(text)
    ]
    local_context_signal = str(table_ast.get("local_context_signal") or "").strip()
    has_boundary_signal = bool(body_boundary_rows) or local_context_signal in {"narrative_barrier", "section_heading"}
    if chart_like_rows and has_boundary_signal:
        return {
            "primary_type": "chart_figure",
            "confidence": 0.86,
            "reasons": ["chart_or_legend_rows", "body_or_heading_boundary"],
        }

    return {"primary_type": "data_table", "confidence": 0.55, "reasons": []}


def _looks_like_chart_legend_or_series_row_text(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    lower = candidate.lower()
    legend_terms = (
        "will not terminate",
        "will terminate",
        "don't know",
        "don＊t know",
        "dont know",
        "tourism",
        "agriculture",
        "handicraft",
        "legend",
    )
    if any(term in lower for term in legend_terms):
        return True
    compact_cells = [part.strip() for part in re.split(r"\s{2,}|\s*\|\s*", candidate) if part.strip()]
    if len(compact_cells) >= 3 and all(len(part.split()) <= 4 for part in compact_cells):
        alpha_cells = sum(1 for part in compact_cells if re.search(r"[A-Za-z\u4e00-\u9fff]", part))
        return alpha_cells >= 3
    return False


def _looks_like_region_boundary_body_or_heading_row_text(text: str) -> bool:
    candidate = " ".join(str(text or "").strip().split())
    if not candidate:
        return False
    if re.match(r"^\d+(?:\.\d+)+\.?\s+[A-Z][A-Za-z]", candidate):
        return True
    words = re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", candidate)
    if len(words) >= 8 and re.search(r"\b(?:the|they|their|would|whether|another|respondents|employees)\b", candidate, re.IGNORECASE):
        return True
    return False


def _should_accept_strong_visual_structure_candidate(
    *,
    raw_evidence: RawTableEvidence,
    confidence: Any,
) -> bool:
    if str(raw_evidence.source or "") != "visual_structure_grid":
        return False
    try:
        overall_confidence = float(getattr(confidence, "overall_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        overall_confidence = 0.0
    if overall_confidence < 0.55:
        return False
    risk_flags = set(getattr(confidence, "risk_flags", []) or [])
    blocking_risks = {
        "toc_like_table",
        "formula_layout",
        "two_column_narrative",
        "publication_front_matter",
    }
    if risk_flags & blocking_risks:
        return False

    ownership = _arbitrate_visual_structure_candidate_ownership(raw_evidence)
    if ownership.get("primary_type") != "data_table":
        return False
    try:
        ownership_confidence = float(ownership.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        ownership_confidence = 0.0
    if ownership_confidence < 0.78:
        return False

    raw_data = raw_evidence.raw_data or []
    if len(raw_data) < 4 or int(raw_evidence.physical_col_count or 0) < 2:
        return False
    return bool(raw_evidence.drawings)


def _looks_like_figure_caption_or_title_text(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    return bool(re.search(r"\b(?:figure|fig\.?)\s*\d+(?:\.\d+)*\b", candidate, re.IGNORECASE))


def _looks_like_chart_source_text(text: str) -> bool:
    return bool(re.search(r"\bsource\s*:", str(text or ""), re.IGNORECASE))


def _looks_like_chart_axis_or_unit_text(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    unit_markers = [
        r"\(in\s+(?:thousands|millions|percent|us\$|billion)\)",
        r"\bin\s+thousands\b",
        r"\bus\$\b",
        r"\bpercent\b",
    ]
    return any(re.search(pattern, candidate, re.IGNORECASE) for pattern in unit_markers)


def _looks_like_axis_tick_row_text(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    year_tokens = re.findall(r"\b(?:19|20)\d{2}(?:\s*\([A-Za-z]+\))?\b", candidate)
    if len(year_tokens) >= 3:
        return True
    month_year_tokens = re.findall(r"\b\d{1,2}/(?:19|20)\d{2}\b", candidate)
    if len(month_year_tokens) >= 3:
        return True
    numeric_tokens = re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{4,6}\b", candidate)
    return len(numeric_tokens) >= 4 and len(re.findall(r"[A-Za-z]{3,}", candidate)) <= 3


def _looks_like_sparse_plotted_numeric_row(row: list[Any]) -> bool:
    filled = _row_non_empty_cells(row)
    if not filled:
        return False
    if len(filled) > max(3, len(row) // 2):
        return False
    numeric_like = sum(1 for _, text in filled if _looks_like_chart_numeric_token(text))
    return numeric_like >= 1 and numeric_like == len(filled)


def _looks_like_chart_numeric_token(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    return bool(re.fullmatch(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d{4,6}", candidate))


def _row_non_empty_ratio(row: list[Any]) -> float:
    if not isinstance(row, list) or not row:
        return 0.0
    return len(_row_non_empty_cells(row)) / len(row)


def _visual_row_has_table_header_shape(row: list[Any]) -> bool:
    filled = [text for _, text in _row_non_empty_cells(row)]
    if len(filled) < 2:
        return False
    alpha_cells = sum(1 for text in filled if re.search(r"[A-Za-z\u4e00-\u9fff]", text))
    numeric_only = sum(1 for text in filled if _looks_like_chart_numeric_token(text))
    return alpha_cells >= 2 and numeric_only == 0


def _visual_row_has_multi_column_text_body(row: list[Any]) -> bool:
    filled = [text for _, text in _row_non_empty_cells(row)]
    if len(filled) < 2:
        return False
    textish = 0
    for text in filled:
        words = re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", text)
        if len(words) >= 2 or len(str(text).strip()) >= 12:
            textish += 1
    return textish >= 2


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
    chart_numeric_false_positive = _looks_like_chart_numeric_toc_false_positive(
        table_ast=table_ast,
        raw_evidence=raw_evidence,
        context=context,
        meaningful_rows=meaningful_rows,
    )

    page_locator_ratio = (page_locator_rows / meaningful_rows) if meaningful_rows else 0.0
    inventory_path_ratio = (inventory_path_rows / meaningful_rows) if meaningful_rows else 0.0
    hierarchical_ratio = (hierarchical_rows / meaningful_rows) if meaningful_rows else 0.0
    locator_indent_level_count = len(set(locator_indent_positions))
    max_outline_depth = max(outline_depths, default=0)
    outline_ladder = locator_indent_level_count >= 2 or max_outline_depth >= 3 or hierarchical_ratio >= 0.18
    flat_titled_toc_list = (
        toc_title_support
        and meaningful_rows >= 3
        and page_locator_ratio >= 0.70
        and inventory_path_ratio <= 0.15
    )
    schema_header_blocks_toc = (
        has_schema_header
        and not page_schema_header
        and not toc_title_support
        and (
            page_locator_ratio < 0.8
            or (
                max_outline_depth == 0
                and hierarchical_ratio < 0.5
                and not page_schema_header
            )
        )
    )
    toc_outline = (
        not table_ast.get("is_continuation", False)
        and meaningful_rows >= 4
        and page_locator_ratio >= 0.45
        and inventory_path_ratio <= 0.15
        and (outline_ladder or flat_titled_toc_list)
        and not schema_header_blocks_toc
        and not explicit_table_title_support
        and not chart_numeric_false_positive
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
        "flat_titled_toc_list": flat_titled_toc_list,
        "chart_numeric_false_positive": chart_numeric_false_positive,
        "semantic_role_reason": (
            "outline_navigation"
            if toc_outline
            else "business_table_default"
        ),
    }
    return role, signals


def _looks_like_chart_numeric_toc_false_positive(
    *,
    table_ast: dict[str, Any],
    raw_evidence: RawTableEvidence,
    context: dict[str, Any],
    meaningful_rows: int,
) -> bool:
    if str(table_ast.get("title", "") or "").strip():
        return False
    if meaningful_rows > 8:
        return False
    if str(raw_evidence.source or "") not in {"pymupdf_builtin", "word_clustering", "visual_structure_grid"}:
        return False
    preceding_text = str((context.get("preceding_text_block") or {}).get("text", "") or "").strip()
    has_figure_context = bool(re.match(r"^(?:figure|fig\.?|图)\b", preceding_text, re.IGNORECASE))
    grid = table_ast.get("raw_grid") or table_ast.get("display_grid") or table_ast.get("grid") or []
    non_empty: list[str] = []
    empty_cell_count = 0
    total_cell_count = 0
    for row in grid:
        if not isinstance(row, list):
            continue
        for cell in row:
            total_cell_count += 1
            text = str(cell or "").strip()
            if text:
                non_empty.append(text)
            else:
                empty_cell_count += 1
    if not non_empty or total_cell_count == 0:
        return False
    numeric_like_count = sum(1 for text in non_empty if re.fullmatch(r"\d{1,4}(?:\s+\d{1,4})*", text))
    sparse_ratio = empty_cell_count / max(1, total_cell_count)
    has_outline_tokens = any(_outline_token_depth(text) > 0 for text in non_empty)
    if has_outline_tokens:
        return False
    mostly_numeric = numeric_like_count / len(non_empty) >= 0.85
    return has_figure_context and mostly_numeric and sparse_ratio >= 0.45


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
                "outline_kind": row_hint.get("outline_kind"),
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
    _annotate_toc_list_structure_evidence(toc_block)
    toc_block["_raw_entries"] = prepared_raw_entries
    _tighten_toc_block_bbox(toc_block)
    return toc_block


def _annotate_toc_list_structure_evidence(toc_block: dict[str, Any]) -> None:
    entries = [dict(entry) for entry in (toc_block.get("entries") or [])]
    entry_count = len(entries)
    if not entry_count:
        toc_block["list_structure_evidence"] = {
            "entry_count": 0,
            "locator_entry_count": 0,
            "locator_coverage_ratio": 0.0,
            "indent_level_count": 0,
            "max_outline_depth": 0,
            "parent_link_count": 0,
            "root_entry_count": 0,
            "leaf_entry_count": 0,
            "outline_kinds": {},
            "support": False,
            "score": 0.0,
        }
        signals = dict(toc_block.get("semantic_signals") or {})
        signals["list_hierarchy_support"] = False
        signals["list_hierarchy_score"] = 0.0
        toc_block["semantic_signals"] = signals
        return

    locator_entry_count = sum(1 for entry in entries if str(entry.get("page_locator") or "").strip())
    anchors = {
        round(float(entry.get("leading_anchor")), 1)
        for entry in entries
        if entry.get("leading_anchor") is not None
    }
    outline_depths = [int(entry.get("outline_depth", 0) or 0) for entry in entries]
    outline_kinds: dict[str, int] = {}
    for entry in entries:
        outline_kind = str(entry.get("outline_kind") or _toc_entry_outline_kind(entry) or "").strip()
        if outline_kind:
            outline_kinds[outline_kind] = outline_kinds.get(outline_kind, 0) + 1

    parent_entry_indices = {
        int(entry.get("parent_entry_index"))
        for entry in entries
        if entry.get("parent_entry_index") is not None
    }
    entry_indices = {
        int(entry.get("entry_index"))
        for entry in entries
        if entry.get("entry_index") is not None
    }
    parent_link_count = sum(1 for entry in entries if entry.get("parent_entry_index") is not None)
    root_entry_count = max(0, entry_count - parent_link_count)
    leaf_entry_count = len(entry_indices - parent_entry_indices) if entry_indices else 0
    locator_coverage_ratio = locator_entry_count / entry_count
    parent_link_ratio = parent_link_count / max(1, entry_count - 1)
    indent_level_count = len(anchors)
    max_outline_depth = max(outline_depths, default=0)
    outline_kind_count = len(outline_kinds)
    hierarchical_shape = (
        indent_level_count >= 2
        or max_outline_depth >= 2
        or parent_link_ratio >= 0.2
        or outline_kind_count >= 2
    )
    score_parts = [
        min(1.0, locator_coverage_ratio),
        min(1.0, indent_level_count / 3.0),
        min(1.0, max_outline_depth / 4.0),
        min(1.0, parent_link_ratio),
        min(1.0, outline_kind_count / 3.0),
    ]
    score = round(sum(score_parts) / len(score_parts), 3)
    support = bool(entry_count >= 4 and locator_coverage_ratio >= 0.45 and hierarchical_shape and score >= 0.45)
    evidence = {
        "entry_count": entry_count,
        "locator_entry_count": locator_entry_count,
        "locator_coverage_ratio": round(locator_coverage_ratio, 3),
        "indent_level_count": indent_level_count,
        "max_outline_depth": max_outline_depth,
        "parent_link_count": parent_link_count,
        "parent_link_ratio": round(parent_link_ratio, 3),
        "root_entry_count": root_entry_count,
        "leaf_entry_count": leaf_entry_count,
        "outline_kinds": dict(sorted(outline_kinds.items())),
        "support": support,
        "score": score,
    }
    toc_block["list_structure_evidence"] = evidence
    signals = dict(toc_block.get("semantic_signals") or {})
    signals["list_hierarchy_support"] = support
    signals["list_hierarchy_score"] = score
    signals["list_hierarchy_parent_link_count"] = parent_link_count
    toc_block["semantic_signals"] = signals


def _toc_entry_outline_kind(entry: dict[str, Any]) -> str | None:
    outline_index = str(entry.get("outline_index") or "").strip()
    if not outline_index:
        return None
    if outline_index.startswith("APPENDIX "):
        return "appendix"
    if all(segment.isdigit() for segment in outline_index.split(".") if segment):
        return "numeric"
    if re.fullmatch(r"[IVXLCDM]+", outline_index, re.IGNORECASE):
        return "roman"
    if re.fullmatch(r"[A-Z]", outline_index):
        return "alpha"
    return "other"


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
    section_hint = _find_section_hint_block(text_blocks, bbox)
    section_boundary_block = _find_table_boundary_text_block(text_blocks, bbox, include_study_metadata=False)
    if section_hint is None and section_boundary_block is not None:
        section_hint = section_boundary_block
    barrier_block = section_boundary_block or _find_table_boundary_text_block(text_blocks, bbox)
    if preceding_block is None and barrier_block is not None:
        preceding_block = barrier_block
    title_block = _find_title_block(text_blocks, bbox)
    if title_block is None:
        title_block = _find_descriptive_micro_table_title_block(
            text_blocks=text_blocks,
            words=words,
            bbox=bbox,
            row_count=int(instance.row_count or 0),
            col_count=int(instance.col_count or 0),
            raw_rows=instance.raw_grid or [],
            page_height=float(page_height or 0.0),
            page_width=0.0,
        )
    instance_title_text = str(instance.title or "").strip()
    if _looks_like_study_metadata_boundary_text(instance_title_text):
        instance_title_text = ""
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
    if context["title_block"] is None and instance_title_text:
        context["title_block"] = {
            "text": instance_title_text,
            "bbox": list(bbox),
            "source": "internal_title_row",
        }
    context["section_hint"] = section_hint
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


def _collect_occupied_bboxes_for_late_table_reconstruction(
    *,
    tables: list[dict[str, Any]],
    out_toc_blocks: list[dict[str, Any]],
) -> list[tuple[float, float, float, float]]:
    """Return occupied regions that should block late table reconstruction.

    Late text-aligned reconstruction is an ownership arbitration pass, not a
    residual-word pass. Early detectors can emit isolated physical rows from a
    larger borderless table; those weak fragments should be replaceable by a
    later coherent grid candidate, so they must not reserve the page region.
    Stronger existing tables and non-table regions still block reconstruction.
    """
    blocking_items: list[dict[str, Any]] = []
    for table in tables:
        if _is_weak_table_fragment_for_late_reconstruction(table):
            continue
        if _is_extendable_word_cluster_for_late_text_aligned_reconstruction(table):
            continue
        if _caption_anchored_rule_has_remote_upstream_caption_for_late_reconstruction(table):
            continue
        blocking_items.append(table)
    blocking_items.extend(out_toc_blocks or [])
    return _collect_item_bboxes(blocking_items)


def _caption_anchored_rule_has_remote_upstream_caption_for_late_reconstruction(table: dict[str, Any]) -> bool:
    """Allow later text-aligned evidence to challenge a weak caption-rule owner.

    A three-line/rule detector can pair a ruled region with a text-layer caption
    that actually belongs to the previous borderless table, especially when that
    caption sits well above the ruled region and has continuation text between
    the caption start and the ruled table. Such a candidate should not reserve
    the region against later text-aligned reconstruction; if no stronger later
    candidate appears, the already accepted table remains intact.
    """
    source = str(table.get("detection_source") or table.get("detection_method") or "")
    if source != "caption_anchored_horizontal_rules":
        return False
    bbox_raw = table.get("bbox") or []
    title_block = table.get("title_block") if isinstance(table.get("title_block"), dict) else None
    if not title_block or len(bbox_raw) != 4:
        return False
    title_bbox_raw = title_block.get("bbox") or []
    if len(title_bbox_raw) != 4:
        return False
    bbox = tuple(float(value) for value in bbox_raw)
    title_bbox = tuple(float(value) for value in title_bbox_raw)
    if title_bbox[3] >= bbox[1]:
        return False
    title_height = max(1.0, title_bbox[3] - title_bbox[1])
    vertical_gap = bbox[1] - title_bbox[3]
    if vertical_gap <= max(22.0, title_height * 2.0):
        return False
    preceding = table.get("preceding_text_block") if isinstance(table.get("preceding_text_block"), dict) else None
    preceding_bbox_raw = preceding.get("bbox") if preceding else None
    if not preceding_bbox_raw or len(preceding_bbox_raw) != 4:
        return False
    preceding_bbox = tuple(float(value) for value in preceding_bbox_raw)
    return title_bbox[3] <= preceding_bbox[1] <= preceding_bbox[3] <= bbox[1]


def _collect_occupied_bboxes_for_visual_structure_reconstruction(
    *,
    tables: list[dict[str, Any]],
    out_toc_blocks: list[dict[str, Any]],
) -> list[tuple[float, float, float, float]]:
    """Return occupied regions that should block visual grid reconstruction.

    Explicit visual grid evidence is a stronger ownership signal than a weak
    text-only projection. A pure text-aligned candidate can over-split a visual
    flowchart/table into many placeholder columns; such candidates should be
    allowed to compete with the later visual-structure detector instead of
    permanently consuming all words in the region.
    """
    blocking_items: list[dict[str, Any]] = []
    for table in tables:
        if _is_weak_text_projection_for_visual_reconstruction(table):
            continue
        blocking_items.append(table)
    blocking_items.extend(out_toc_blocks or [])
    return _collect_item_bboxes(blocking_items)


def _is_weak_table_fragment_for_late_reconstruction(table: dict[str, Any]) -> bool:
    grid = table.get("display_grid") or table.get("raw_grid") or []
    if not isinstance(grid, list) or not grid:
        return False
    row_count = len(grid)
    col_count = max((len(row) for row in grid if isinstance(row, list)), default=0)
    source = str(table.get("detection_source") or table.get("detection_method") or "")
    if source not in {"pymupdf_builtin", "word_clustering"}:
        if source == "structured_text_region":
            return _is_single_column_structured_region_fragment(table, grid)
        return False
    if row_count > 2 or col_count < 3:
        return False
    semantic_role = str(table.get("semantic_role") or "")
    if semantic_role in {"toc_outline", "chart_figure", "layout_matrix"}:
        return False
    if str(table.get("title") or table.get("caption") or "").strip():
        return False
    confidence = table.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    return confidence_value <= 0.72 or row_count == 1


def _is_extendable_word_cluster_for_late_text_aligned_reconstruction(table: dict[str, Any]) -> bool:
    source = str(table.get("detection_source") or table.get("detection_method") or "")
    if source != "word_clustering":
        return False
    grid = table.get("display_grid") or table.get("raw_grid") or []
    if not isinstance(grid, list):
        return False
    row_count = len(grid)
    col_count = max((len(row) for row in grid if isinstance(row, list)), default=0)
    if row_count < 4 or col_count < 4:
        return False
    bbox_raw = table.get("bbox") or []
    page_height = float(table.get("page_height") or 0.0)
    if len(bbox_raw) != 4 or page_height <= 0:
        return False
    bbox = tuple(float(value) for value in bbox_raw)
    near_page_bottom = bbox[3] >= page_height * 0.72
    note_blocks = table.get("note_blocks") if isinstance(table.get("note_blocks"), list) else []
    has_below_note = any(
        isinstance(note, dict)
        and str(note.get("relation") or "") == "below"
        for note in note_blocks
    )
    return not near_page_bottom and not has_below_note


def _is_single_column_structured_region_fragment(table: dict[str, Any], grid: list[Any]) -> bool:
    row_count = len(grid)
    col_count = max((len(row) for row in grid if isinstance(row, list)), default=0)
    if col_count > 1 or row_count < 4:
        return False
    bbox = table.get("bbox") or []
    if len(bbox) != 4:
        return False
    width = float(bbox[2]) - float(bbox[0])
    page_width = float(table.get("page_width") or 0.0)
    if page_width > 0 and width >= page_width * 0.72:
        return False
    cells = [
        str(cell or "").strip()
        for row in grid
        if isinstance(row, list)
        for cell in row
        if str(cell or "").strip()
    ]
    if len(cells) < 4:
        return False
    long_cells = sum(1 for text in cells if len(text) >= 24 and len(text.split()) >= 3)
    compact_cells = sum(1 for text in cells if len(text) <= 24 and len(text.split()) <= 4)
    title_text = str(table.get("title") or table.get("caption_text") or "").strip()
    title_is_first_cell = bool(title_text) and bool(cells) and _compact_context_text(title_text) == _compact_context_text(cells[0])
    if title_text and not title_is_first_cell:
        return False
    return long_cells >= 3 and compact_cells <= max(3, len(cells) // 3)


def _is_weak_text_projection_for_visual_reconstruction(table: dict[str, Any]) -> bool:
    source = str(table.get("detection_source") or table.get("detection_method") or "")
    if source != "text_aligned_borderless_grid":
        return False
    if str(table.get("title") or table.get("caption") or "").strip():
        return False

    grid = table.get("display_grid") or table.get("raw_grid") or []
    if not isinstance(grid, list) or not grid:
        return False
    row_count = len(grid)
    col_count = max((len(row) for row in grid if isinstance(row, list)), default=0)
    if row_count < 3 or col_count < 5:
        return False

    confidence = table.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    if confidence_value > 0.76:
        return False

    total_cells = 0
    filled_cells = 0
    placeholder_cells = 0
    short_fragment_cells = 0
    for row in grid:
        if not isinstance(row, list):
            continue
        for cell in row:
            total_cells += 1
            text = str(cell or "").strip()
            if not text:
                continue
            filled_cells += 1
            if re.fullmatch(r"Column\s+\d+", text, re.IGNORECASE):
                placeholder_cells += 1
            if len(text) <= 18 and len(text.split()) <= 3:
                short_fragment_cells += 1
    if total_cells <= 0 or filled_cells <= 0:
        return False

    empty_ratio = 1.0 - (filled_cells / total_cells)
    placeholder_ratio = placeholder_cells / max(1, filled_cells)
    short_fragment_ratio = short_fragment_cells / max(1, filled_cells)
    return (
        empty_ratio >= 0.30
        or placeholder_ratio >= 0.12
        or (col_count >= 8 and short_fragment_ratio >= 0.60)
    )


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


def _replace_weaker_overlapping_table_fragments(
    tables: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Replace weak single-row fragments with a stronger encompassing table.

    Built-in detectors sometimes emit isolated physical rows from a borderless
    table. Once a later candidate reconstructs the same region as a coherent
    multi-row grid, the ownership graph should keep the stronger table object
    and release the fragments instead of letting early detector order win.
    """
    cand_bbox_raw = candidate.get("bbox")
    if not cand_bbox_raw or len(cand_bbox_raw) != 4:
        return tables, False
    cand_bbox = tuple(float(v) for v in cand_bbox_raw)
    cand_grid = candidate.get("display_grid") or candidate.get("raw_grid") or []
    cand_rows = len(cand_grid)
    cand_cols = max((len(row) for row in cand_grid), default=0)
    cand_source = str(candidate.get("detection_source") or candidate.get("detection_method") or "")
    if cand_source not in {"text_aligned_borderless_grid", "visual_structure_grid", "word_clustering"}:
        return tables, False
    min_candidate_cols = 2 if cand_source == "visual_structure_grid" else 4
    if cand_rows < 4 or cand_cols < min_candidate_cols:
        return tables, False
    if cand_source == "word_clustering" and _word_cluster_spans_multiple_mature_text_aligned_owners(candidate, tables):
        return tables, False

    remove_indices: set[int] = set()
    for index, existing in enumerate(tables):
        ex_bbox_raw = existing.get("bbox")
        if not ex_bbox_raw or len(ex_bbox_raw) != 4:
            continue
        ex_bbox = tuple(float(v) for v in ex_bbox_raw)
        overlap_ratio = _bbox_overlap_ratio(cand_bbox, ex_bbox)
        vertical_overlap = max(0.0, min(cand_bbox[3], ex_bbox[3]) - max(cand_bbox[1], ex_bbox[1]))
        ex_height = max(1.0, ex_bbox[3] - ex_bbox[1])
        ex_inside_candidate = (
            vertical_overlap / ex_height >= 0.80
            and ex_bbox[0] >= cand_bbox[0] - 8.0
            and ex_bbox[2] <= cand_bbox[2] + 8.0
        )
        if overlap_ratio < 0.55 and not ex_inside_candidate:
            continue
        ex_grid = existing.get("display_grid") or existing.get("raw_grid") or []
        ex_rows = len(ex_grid)
        ex_cols = max((len(row) for row in ex_grid), default=0)
        ex_source = str(existing.get("detection_source") or existing.get("detection_method") or "")
        weak_fragment = ex_rows <= 2 and ex_cols <= cand_cols and ex_source in {"pymupdf_builtin", "word_clustering"}
        weak_single_column_region = (
            cand_source == "word_clustering"
            and ex_source == "structured_text_region"
            and ex_cols <= 1
            and cand_cols >= 4
            and cand_rows >= max(8, ex_rows + 4)
            and ex_inside_candidate
            and _structured_single_column_region_title_is_internal(existing)
        )
        weak_text_projection = (
            cand_source == "visual_structure_grid"
            and _is_weak_text_projection_for_visual_reconstruction(existing)
        )
        weak_partial_text_aligned_owner = _candidate_word_cluster_replaces_partial_text_aligned_owner(
            candidate=candidate,
            existing=existing,
            overlap_ratio=overlap_ratio,
            ex_inside_candidate=ex_inside_candidate,
            cand_rows=cand_rows,
            cand_cols=cand_cols,
            ex_rows=ex_rows,
            ex_cols=ex_cols,
        )
        bottom_caption_replaces_rule_owner = _candidate_bottom_caption_text_grid_replaces_rule_owner(
            candidate=candidate,
            existing=existing,
            overlap_ratio=overlap_ratio,
            ex_inside_candidate=ex_inside_candidate,
            cand_rows=cand_rows,
            cand_cols=cand_cols,
            ex_rows=ex_rows,
            ex_cols=ex_cols,
        )
        if (
            weak_fragment
            or weak_single_column_region
            or weak_text_projection
            or weak_partial_text_aligned_owner
            or bottom_caption_replaces_rule_owner
        ):
            remove_indices.add(index)

    if not remove_indices:
        return tables, False
    return [table for index, table in enumerate(tables) if index not in remove_indices], True


def _word_cluster_spans_multiple_mature_text_aligned_owners(
    candidate: dict[str, Any],
    tables: list[dict[str, Any]],
) -> bool:
    """Prevent coarse word clusters from merging independent text-layer tables."""

    cand_bbox_raw = candidate.get("bbox")
    if not cand_bbox_raw or len(cand_bbox_raw) != 4:
        return False
    cand_bbox = tuple(float(value) for value in cand_bbox_raw)

    mature_owner_count = 0
    for existing in tables:
        source = str(existing.get("detection_source") or existing.get("detection_method") or "")
        if source != "text_aligned_borderless_grid":
            continue
        ex_bbox_raw = existing.get("bbox")
        if not ex_bbox_raw or len(ex_bbox_raw) != 4:
            continue
        grid = existing.get("display_grid") or existing.get("raw_grid") or []
        rows = len(grid)
        cols = max((len(row) for row in grid if isinstance(row, list)), default=0)
        if rows < 4 or cols < 4:
            continue
        ex_bbox = tuple(float(value) for value in ex_bbox_raw)
        vertical_overlap = max(0.0, min(cand_bbox[3], ex_bbox[3]) - max(cand_bbox[1], ex_bbox[1]))
        ex_height = max(1.0, ex_bbox[3] - ex_bbox[1])
        horizontally_aligned = _horizontal_overlap_ratio(cand_bbox, ex_bbox) >= 0.55
        contained_or_strong_overlap = (
            vertical_overlap / ex_height >= 0.70
            and horizontally_aligned
        ) or _bbox_overlap_ratio(cand_bbox, ex_bbox) >= 0.35
        if contained_or_strong_overlap:
            mature_owner_count += 1
            if mature_owner_count >= 2:
                return True
    return False


def _has_text_aligned_owner_for_word_cluster_challenge(tables: list[dict[str, Any]]) -> bool:
    """Run a source-arbitration challenge when a borderless text grid owns a page.

    Text-aligned reconstruction is high precision for many borderless tables,
    but it can stop early when a body row resembles a caption. A full-page
    word-clustering candidate is only allowed to replace that owner later if it
    proves a larger coherent table over the same region.
    """
    for table in tables:
        source = str(table.get("detection_source") or table.get("detection_method") or "")
        if source != "text_aligned_borderless_grid":
            continue
        grid = table.get("display_grid") or table.get("raw_grid") or []
        rows = len(grid)
        cols = max((len(row) for row in grid if isinstance(row, list)), default=0)
        if rows >= 4 and cols >= 4:
            return True
    return False


def _candidate_word_cluster_replaces_partial_text_aligned_owner(
    *,
    candidate: dict[str, Any],
    existing: dict[str, Any],
    overlap_ratio: float,
    ex_inside_candidate: bool,
    cand_rows: int,
    cand_cols: int,
    ex_rows: int,
    ex_cols: int,
) -> bool:
    cand_source = str(candidate.get("detection_source") or candidate.get("detection_method") or "")
    ex_source = str(existing.get("detection_source") or existing.get("detection_method") or "")
    if cand_source != "word_clustering" or ex_source != "text_aligned_borderless_grid":
        return False
    if overlap_ratio < 0.35 and not ex_inside_candidate:
        return False
    if cand_rows < max(ex_rows + 4, int(ex_rows * 1.45)):
        return False
    if cand_cols < max(3, ex_cols - 2):
        return False
    if cand_cols > ex_cols + 1:
        return False
    cand_grid = candidate.get("display_grid") or candidate.get("raw_grid") or []
    ex_grid = existing.get("display_grid") or existing.get("raw_grid") or []
    if not cand_grid or not ex_grid:
        return False
    cand_header_tokens = _normalized_grid_token_set(cand_grid[:2])
    ex_header_tokens = _normalized_grid_token_set(ex_grid[:2])
    if cand_header_tokens and ex_header_tokens:
        shared = len(cand_header_tokens & ex_header_tokens)
        if shared < max(2, min(len(cand_header_tokens), len(ex_header_tokens)) // 2):
            return False
    candidate_title = str(candidate.get("title") or "").strip()
    existing_title = str(existing.get("title") or "").strip()
    candidate_has_clean_title = bool(candidate_title and not TOC_TITLE_PATTERN.search(candidate_title))
    existing_has_boundary_toc_title = bool(existing_title and TOC_TITLE_PATTERN.search(existing_title))
    if candidate_has_clean_title and existing_has_boundary_toc_title:
        return True
    if candidate_has_clean_title and cand_rows >= max(ex_rows + 6, int(ex_rows * 1.75)):
        return True
    cand_score = _table_ast_structure_strength(candidate)
    ex_score = _table_ast_structure_strength(existing)
    return cand_score >= ex_score and cand_rows >= ex_rows + 6


def _normalized_grid_token_set(rows: list[Any]) -> set[str]:
    tokens: set[str] = set()
    for row in rows:
        if not isinstance(row, list):
            continue
        for cell in row:
            text = str(cell or "").strip().lower()
            if not text:
                continue
            tokens.update(token for token in re.findall(r"[a-z0-9]{2,}", text) if token)
            tokens.update(token for token in re.findall(r"[\u4e00-\u9fff]+", text) if token)
    return tokens


def _table_ast_structure_strength(table: dict[str, Any]) -> float:
    grid = table.get("display_grid") or table.get("raw_grid") or []
    rows = len(grid)
    cols = max((len(row) for row in grid if isinstance(row, list)), default=0)
    if rows <= 0 or cols <= 0:
        return 0.0
    filled = 0
    multi_cell_rows = 0
    for row in grid:
        if not isinstance(row, list):
            continue
        row_fill = sum(1 for cell in row if str(cell or "").strip())
        filled += row_fill
        if row_fill >= 2:
            multi_cell_rows += 1
    density = filled / max(1, rows * cols)
    return (
        0.35 * min(1.0, rows / 12.0)
        + 0.25 * min(1.0, cols / 5.0)
        + 0.25 * density
        + 0.15 * (multi_cell_rows / max(1, rows))
    )


def _has_preferred_existing_owner_for_text_aligned_candidate(
    tables: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> bool:
    cand_source = str(candidate.get("detection_source") or candidate.get("detection_method") or "")
    if cand_source != "text_aligned_borderless_grid":
        return False
    cand_bbox_raw = candidate.get("bbox")
    if not cand_bbox_raw or len(cand_bbox_raw) != 4:
        return False
    cand_bbox = tuple(float(v) for v in cand_bbox_raw)
    cand_grid = candidate.get("display_grid") or candidate.get("raw_grid") or []
    cand_rows = len(cand_grid)
    cand_cols = max((len(row) for row in cand_grid if isinstance(row, list)), default=0)
    if cand_rows < 4 or cand_cols < 4:
        return False
    cand_cells = [
        str(cell or "").strip()
        for row in cand_grid
        if isinstance(row, list)
        for cell in row
        if str(cell or "").strip()
    ]
    cand_has_section_group_rows = any(
        re.match(r"^\d+(?:\.\d+)+\s+\S+", text)
        for text in cand_cells
    )

    for existing in tables:
        ex_source = str(existing.get("detection_source") or existing.get("detection_method") or "")
        if ex_source not in {"word_clustering", "pymupdf_builtin"}:
            continue
        ex_bbox_raw = existing.get("bbox")
        if not ex_bbox_raw or len(ex_bbox_raw) != 4:
            continue
        ex_bbox = tuple(float(v) for v in ex_bbox_raw)
        overlap_ratio = _bbox_overlap_ratio(cand_bbox, ex_bbox)
        vertical_overlap = max(0.0, min(cand_bbox[3], ex_bbox[3]) - max(cand_bbox[1], ex_bbox[1]))
        ex_height = max(1.0, ex_bbox[3] - ex_bbox[1])
        cand_height = max(1.0, cand_bbox[3] - cand_bbox[1])
        same_region = overlap_ratio >= 0.55 or (
            vertical_overlap / ex_height >= 0.80
            and vertical_overlap / cand_height >= 0.72
        )
        if not same_region:
            continue
        ex_grid = existing.get("display_grid") or existing.get("raw_grid") or []
        ex_rows = len(ex_grid)
        ex_cols = max((len(row) for row in ex_grid if isinstance(row, list)), default=0)
        ex_cells = [
            str(cell or "").strip()
            for row in ex_grid
            if isinstance(row, list)
            for cell in row
            if str(cell or "").strip()
        ]
        ex_has_section_group_rows = any(
            re.match(r"^\d+(?:\.\d+)+\s+\S+", text)
            for text in ex_cells
        )
        has_below_note = any(
            isinstance(note, dict)
            and str(note.get("relation") or "") == "below"
            for note in (existing.get("note_blocks") or [])
        )
        if (
            ex_rows >= max(4, int(cand_rows * 0.60))
            and ex_cols >= max(4, cand_cols - 2)
            and (has_below_note or ex_has_section_group_rows or cand_has_section_group_rows)
        ):
            return True
    return False


def _text_aligned_candidate_should_defer_to_word_clustering(raw_evidence: RawTableEvidence) -> bool:
    if str(raw_evidence.source or "") != "text_aligned_borderless_grid":
        return False
    raw_rows = raw_evidence.raw_data or []
    if len(raw_rows) < 4:
        return False
    col_count = int(raw_evidence.physical_col_count or 0)
    if col_count < 4:
        return False
    section_group_rows = 0
    body_rows_after_group = 0
    for row_index, row in enumerate(raw_rows):
        filled = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
        if not filled:
            continue
        if any(re.match(r"^\d+(?:\.\d+)+\s+\S+", text) for text in filled):
            section_group_rows += 1
            continue
        if section_group_rows and row_index > 0 and len(filled) >= max(3, min(col_count - 1, 6)):
            body_rows_after_group += 1
    return section_group_rows >= 1 and body_rows_after_group >= 3


def _candidate_bottom_caption_text_grid_replaces_rule_owner(
    *,
    candidate: dict[str, Any],
    existing: dict[str, Any],
    overlap_ratio: float,
    ex_inside_candidate: bool,
    cand_rows: int,
    cand_cols: int,
    ex_rows: int,
    ex_cols: int,
) -> bool:
    """Let a bottom-caption text grid replace an earlier rule candidate.

    Booktabs-style tables can have captions below the table. A rule-bounded
    detector that runs earlier may attach the previous table's bottom caption
    to the next ruled region. When a later text-aligned candidate owns the same
    physical region and carries an explicit bottom-caption source, the ownership
    graph should keep that candidate rather than preserve detector order.
    """
    cand_source = str(candidate.get("detection_source") or candidate.get("detection_method") or "")
    ex_source = str(existing.get("detection_source") or existing.get("detection_method") or "")
    if cand_source != "text_aligned_borderless_grid" or ex_source != "caption_anchored_horizontal_rules":
        return False
    title_block = candidate.get("title_block") if isinstance(candidate.get("title_block"), dict) else {}
    title_source = str(title_block.get("source") or "")
    if title_source != "bottom_caption_text_aligned_grid":
        return False
    if overlap_ratio < 0.55 and not ex_inside_candidate:
        return False
    if cand_rows < max(2, ex_rows) or cand_cols < max(3, ex_cols):
        return False
    candidate_title = str(candidate.get("title") or candidate.get("caption_text") or "").strip()
    if candidate_title and not _looks_like_table_title(candidate_title):
        return False
    return True


def _structured_single_column_region_title_is_internal(table: dict[str, Any]) -> bool:
    title = str(table.get("caption_text") or table.get("title") or "").strip()
    if not title:
        return True
    grid = table.get("display_grid") or table.get("raw_grid") or table.get("grid") or []
    first_text = ""
    for row in grid if isinstance(grid, list) else []:
        if not isinstance(row, list):
            continue
        for cell in row:
            first_text = str(cell or "").strip()
            if first_text:
                break
        if first_text:
            break
    if not first_text:
        return False
    return _compact_context_text(title) == _compact_context_text(first_text)


def _build_supplemental_word_cluster_candidate(
    *,
    page_words: list[_Word],
    page: Any,
    page_number: int,
    page_height: float,
    page_width: float,
    text_blocks: list[dict[str, Any]] | None,
    page_drawings: list[dict[str, Any]] | None,
    prev_tables: list[dict[str, Any]] | None,
    table_counter: int,
    layout_profile: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[Any]]:
    raw_evidence = extract_raw_evidence_from_words(
        words=page_words,
        page_number=page_number,
        page_height=page_height,
        page_width=page_width,
    )
    if raw_evidence is None:
        return None, []
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
    assessment = _assess_continuation_candidates_v2(
        candidates=candidates,
        raw_evidence=raw_evidence,
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
        parent_col_count=assessment.selected_parent_col_count if assessment.is_continuation else None,
        parent_header=assessment.selected_parent_header if assessment.is_continuation else None,
        parent_bbox=assessment.selected_parent_bbox if assessment.is_continuation else None,
        parent_column_boundaries=assessment.selected_parent_column_boundaries if assessment.is_continuation else None,
        assessment=assessment,
        current_context=current_context,
        words=page_words,
    )
    if not table_ast:
        return None, list(raw_evidence.words or [])
    if table_ast.get("semantic_role") == "toc_outline":
        return None, list(raw_evidence.words or [])
    ownership = _arbitrate_table_candidate_ownership(raw_evidence, table_ast, current_context)
    if ownership.get("primary_type") != "data_table":
        return None, list(raw_evidence.words or [])
    if not _can_accept_supplemental_candidate(
        raw_evidence=raw_evidence,
        words=page_words,
        page_width=page_width,
        current_context=current_context,
        page_drawings=page_drawings,
        layout_profile=layout_profile,
        assessment=assessment,
    ):
        return None, list(raw_evidence.words or [])
    return table_ast, list(raw_evidence.words or [])


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
    if _has_schema_complete_wide_table_ownership(raw_evidence=raw_evidence, page_width=page_width):
        return True
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
    if _looks_like_dense_narrative_grid_without_tabular_values(raw_evidence):
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
    if _has_schema_complete_wide_table_ownership(raw_evidence=raw_evidence, page_width=page_width):
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

    if _has_schema_complete_wide_table_ownership(raw_evidence=raw_evidence, page_width=page_width):
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


def _has_schema_complete_wide_table_ownership(
    *,
    raw_evidence: RawTableEvidence,
    page_width: float,
) -> bool:
    """Prefer a complete parent table lattice over lane-local two-column splits.

    Composite splitting is useful when two independent lane tables or narrative
    fragments were accidentally merged. A true wide table has a different
    invariant: the parent candidate already owns a stable multi-column schema,
    including populated cells on both sides of the would-be split. In that case
    splitting loses structure and should not preempt the parent candidate.
    """
    raw_data = raw_evidence.raw_data or []
    col_count = int(raw_evidence.physical_col_count or 0)
    if col_count < 4 or len(raw_data) < 3:
        return False

    bbox = tuple(float(value) for value in raw_evidence.bbox)
    width_ratio = (bbox[2] - bbox[0]) / max(1.0, float(page_width or raw_evidence.page_width or 1.0))
    if width_ratio < 0.68:
        return False

    schema_row_count = 0
    populated_right_edge_rows = 0
    multi_column_data_rows = 0
    max_filled = 0
    for row in raw_data:
        if not isinstance(row, list):
            continue
        cells = [str(cell or "").strip() for cell in row[:col_count]]
        filled_indexes = [idx for idx, text in enumerate(cells) if text]
        filled_count = len(filled_indexes)
        max_filled = max(max_filled, filled_count)
        if filled_count >= min(col_count, 4):
            schema_row_count += 1
        if filled_count >= 3:
            multi_column_data_rows += 1
        if cells and str(cells[-1]).strip():
            populated_right_edge_rows += 1

    if max_filled < min(col_count, 4) or schema_row_count < 1:
        return False
    if multi_column_data_rows < max(3, min(6, len(raw_data) // 4)):
        return False
    if populated_right_edge_rows < max(2, min(5, len(raw_data) // 5)):
        return False

    words = list(raw_evidence.words or [])
    if words:
        midpoint = (bbox[0] + bbox[2]) / 2.0
        left_count = sum(1 for word in words if float(getattr(word, "x_center", 0.0)) < midpoint)
        right_count = sum(1 for word in words if float(getattr(word, "x_center", 0.0)) >= midpoint)
        if min(left_count, right_count) < max(4, len(words) // 8):
            return False

    if _narrative_cell_ratio(raw_evidence) >= 0.86:
        return False

    return True


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


def _looks_like_dense_narrative_grid_without_tabular_values(raw_evidence: RawTableEvidence) -> bool:
    """Reject word-clustered prose columns projected as a dense grid.

    Dense two-column body text can look table-like geometrically: every visual
    row has two filled cells. A real data table normally contains at least some
    short labels, enumerators, numeric/statistical values, or a semantic header.
    When nearly all cells are long sentence fragments, the ownership belongs to
    body flow, even if an earlier caption exists nearby.
    """
    if raw_evidence.source != "word_clustering":
        return False
    raw_data = raw_evidence.raw_data or []
    if len(raw_data) < 3:
        return False
    col_count = int(raw_evidence.physical_col_count or 0)
    if col_count < 2 or col_count > 3:
        return False

    filled_cells: list[str] = []
    for row in raw_data:
        for cell in row:
            text = _normalize_candidate_grid_text(cell)
            if text:
                filled_cells.append(text)
    if len(filled_cells) < 6:
        return False

    narrative_cells = sum(
        1
        for text in filled_cells
        if len(text) >= 32 and (len(text.split()) >= 5 or any(mark in text for mark in ".;,!?"))
    )
    numeric_or_formula_cells = sum(
        1
        for text in filled_cells
        if _looks_like_tabular_value_cell(text)
    )
    short_label_cells = sum(1 for text in filled_cells if len(text) <= 18 and len(text.split()) <= 3)
    return (
        narrative_cells / len(filled_cells) >= 0.78
        and numeric_or_formula_cells <= max(1, len(filled_cells) // 6)
        and short_label_cells <= max(1, len(filled_cells) // 5)
    )


def _looks_like_tabular_value_cell(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    if re.fullmatch(
        r"[+\-−]?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:%|mg/kg|ng/ml|mmhg|kg|ml|g|h|d|min|s|[A-Za-z]{1,4}))?",
        candidate,
        re.IGNORECASE,
    ):
        return True
    if re.fullmatch(r"[<>≤≥=]\s*\d+(?:\.\d+)?(?:\s*%)?", candidate):
        return True
    if re.search(r"[=±≤≥∑√∫]", candidate) and len(candidate.split()) <= 5:
        return True
    return False


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
