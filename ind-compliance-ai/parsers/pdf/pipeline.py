"""PDF extraction pipeline using the unified table parsing architecture v1.0.0.

统一架构:
    PDF
    └─ PyMuPDF (物理层) - raw text extraction
        └─ Raw Objects Layer - 原始证据提取
            └─ Normalization Layer - 物理证据规范化 (含 parent_bbox 继承)
                └─ Assembly Layer - 表格实例组装
                    └─ Continuum Engine - 6 Phase 处理
                        └─ AST Layer - 逻辑表格 AST

6 Phase 处理流程:
    Phase 1: Table Identity Resolution - 表格身份判定
    Phase 2: Logical Grid Stabilization - 逻辑网格稳定
    Phase 3: Cross-Page Continuity - 跨页连续性
    Phase 4: Cell Semantics & State Machine - 单元格语义状态机
    Phase 5: Nested Structure Detection - 嵌套结构识别
    Phase 6: Confidence, Risk & Review Policy - 置信度评估

The table parsing module handles:
- Table detection (PyMuPDF + word-clustering fallback)
- Continuation detection with parent_bbox column inheritance
- Cross-page stitching
- Fragment merging
- ID renumbering
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import pymupdf
except ImportError:  # pragma: no cover - optional runtime dependency
    pymupdf = None  # type: ignore[assignment]

from .image_blocks import (
    _assign_figure_titles,
    _deduplicate_page_images,
    _demote_textual_image_blocks,
)
from .layout import _extract_words
from .tables import (
    # Main entry points
    extract_tables_from_page,
    extract_tables_from_document,
    # Postprocess functions
    stitch_cross_page_tables,
    renumber_table_ids,
    merge_same_page_table_fragments,
)
from .text_blocks import (
    _extract_page_text_and_images,
    _filter_header_footer_text_blocks,
    _merge_semantic_text_blocks,
)
from .types import PdfPipelineState


def _get_text_blocks(page: Any) -> list[dict[str, Any]]:
    """Extract text blocks from page for context detection."""
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
    """Extract drawings from page for grid detection."""
    try:
        return page.get_drawings()
    except Exception:
        return []


def run_pdf_extraction_pipeline(path: Path) -> PdfPipelineState:
    """Run the PDF extraction pipeline.

    This function:
    1. Extracts text blocks, images, and words from each page
    2. Demotes textual image blocks (OCR recovery)
    3. Merges over-segmented text blocks
    4. Extracts tables using the unified architecture
    5. Assigns figure titles
    6. Filters header/footer text blocks
    7. Stitches cross-page tables and renumbers IDs

    Args:
        path: Path to the PDF file

    Returns:
        PdfPipelineState containing all extracted data
    """
    if pymupdf is None:
        raise RuntimeError("PyMuPDF is required for .pdf parsing. Install with: pip install pymupdf")

    state = PdfPipelineState()
    document = pymupdf.open(path)
    figure_index = 1
    table_counter = 0

    try:
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            page_rect = page.rect
            state.page_heights[page_number] = float(page_rect.height)

            # Extract raw content from page
            text_blocks, page_images = _extract_page_text_and_images(page, page_number)
            page_words = _extract_words(page)
            try:
                page_drawings = page.get_drawings()
            except Exception:
                page_drawings = []

            # Step-1: Correct image-vs-text confusion using text-layer/OCR/path signals.
            text_blocks, page_images, recovered_image_text = _demote_textual_image_blocks(
                page=page,
                page_number=page_number,
                page_rect=page_rect,
                image_blocks=page_images,
                text_blocks=text_blocks,
                page_words=page_words,
                page_drawings=page_drawings,
            )
            state.counters.image_text_recovered_count += recovered_image_text
            page_images, duplicate_image_blocks_removed = _deduplicate_page_images(page_images, page_number)
            state.counters.duplicate_image_blocks_removed += duplicate_image_blocks_removed

            # Step-2: Semantic + layout-aware merge for over-segmented text blocks.
            text_blocks, semantic_merge_count = _merge_semantic_text_blocks(
                text_blocks=text_blocks,
                page_number=page_number,
                page_width=float(page_rect.width),
            )
            state.counters.semantic_merge_count += semantic_merge_count

            # Step-3: Extract tables using unified architecture
            # Get additional context for table detection
            context_text_blocks = _get_text_blocks(page)
            context_drawings = _get_drawings(page)

            page_tables, table_counter = extract_tables_from_page(
                page=page,
                page_number=page_number,
                page_height=float(page_rect.height),
                text_blocks=context_text_blocks,
                page_drawings=context_drawings,
                prev_tables=state.table_asts,
                table_counter=table_counter,
            )

            # Merge same-page fragments if multiple tables found
            if len(page_tables) > 1:
                page_tables, merge_count = merge_same_page_table_fragments(page_tables)
                state.counters.table_fragment_merge_count += merge_count

            # Add tables to state
            state.table_asts.extend(page_tables)

            # Step-4: Assign figure titles
            page_figures, figure_index = _assign_figure_titles(page_images, text_blocks, figure_index)
            state.figure_nodes.extend(page_figures)

            # Store page payload
            state.page_payloads.append(
                {
                    "page_number": page_number,
                    "width": float(page_rect.width),
                    "height": float(page_rect.height),
                    "text_blocks": text_blocks,
                    "images": page_images,
                    "tables": page_tables,
                    "semantic_merge_count": semantic_merge_count,
                    "image_text_recovered_count": recovered_image_text,
                    "table_fragment_merge_count": merge_count if len(page_tables) > 1 else 0,
                }
            )
    finally:
        document.close()

    # Post-processing: filter header/footer, stitch cross-page tables, renumber
    state.counters.header_footer_filtered_count = _filter_header_footer_text_blocks(state.page_payloads)
    state.counters.cross_page_table_links = stitch_cross_page_tables(state.table_asts, state.page_heights)
    renumber_table_ids(state.table_asts)

    return state
