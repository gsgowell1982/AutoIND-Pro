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
    _build_single_table_ast,
    _build_table_ast_from_pymupdf,
    _cluster_rows,
    _find_continuation_hint,
    _find_section_table_hint,
    _find_table_title_block,
    _group_table_rows,
    _is_glossary_hint_text,
    _is_toc_context,
    _is_valid_table_candidate,
    _merge_same_page_table_fragments,
    _merge_same_title_tables,
    _renumber_table_ids,
    _stitch_cross_page_tables,
    _table_grid_line_score,
    _table_reference_like_ratio,
    _table_toc_row_ratio,
)
from .text_blocks import (
    _extract_page_text_and_images,
    _filter_header_footer_text_blocks,
    _merge_semantic_text_blocks,
)
from .types import PdfPipelineState


def _resolve_continuation_context(
    candidate_table: dict[str, Any],
    continuation_hint: dict[str, Any] | None,
    accepted_tables: list[dict[str, Any]],
) -> None:
    if continuation_hint is None:
        return
    candidate_table["continuation_hint"] = continuation_hint
    parent_table = next(
        (
            table
            for table in reversed(accepted_tables)
            if str(table.get("table_id", "")) == str(continuation_hint.get("table_id", ""))
        ),
        None,
    )
    if parent_table is not None and str(parent_table.get("continuation_context", "")).strip():
        candidate_table["continuation_context"] = parent_table["continuation_context"]


def _build_page_tables(
    page_number: int,
    page_rect: "pymupdf.Rect",
    text_blocks: list[dict[str, Any]],
    page_words: list[Any],
    page_drawings: list[dict[str, Any]],
    accepted_tables: list[dict[str, Any]],
    table_index: int,
    state: PdfPipelineState,
    page: Any = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Build table ASTs for a page.

    Strategy:
    1. Try PyMuPDF's built-in table detection first (more accurate for multi-line cells)
    2. Fall back to custom word-clustering method if PyMuPDF fails
    """
    raw_page_tables: list[dict[str, Any]] = []
    next_table_index = table_index

    # Strategy 1: Try PyMuPDF's built-in table detection first
    if page is not None:
        try:
            pymupdf_tables = page.find_tables()
            if pymupdf_tables and hasattr(pymupdf_tables, 'tables') and pymupdf_tables.tables:
                for pymupdf_table in pymupdf_tables.tables:
                    candidate_table = _build_table_ast_from_pymupdf(
                        page=page,
                        page_number=page_number,
                        page_height=float(page_rect.height),
                        table_id=f"tbl_{next_table_index:03d}",
                    )
                    if candidate_table is not None:
                        table_bbox = tuple(candidate_table["bbox"])
                        title_block = _find_table_title_block(table_bbox, text_blocks)
                        section_hint_block = _find_section_table_hint(table_bbox, text_blocks)

                        if title_block is not None:
                            candidate_table["title"] = title_block["text"]
                            candidate_table["title_block_id"] = title_block["block_id"]
                        if section_hint_block is not None:
                            candidate_table["section_hint"] = section_hint_block["text"]
                            candidate_table["section_hint_block_id"] = section_hint_block["block_id"]
                            if _is_glossary_hint_text(section_hint_block["text"]):
                                candidate_table["continuation_context"] = "glossary"

                        raw_page_tables.append(candidate_table)
                        next_table_index += 1

                # If PyMuPDF found tables, use them and skip custom detection
                if raw_page_tables:
                    page_tables, page_fragment_merge_count = _merge_same_page_table_fragments(raw_page_tables)
                    page_tables, page_title_merge_count = _merge_same_title_tables(page_tables)
                    page_total_fragment_merges = page_fragment_merge_count + page_title_merge_count
                    state.counters.table_fragment_merge_count += page_total_fragment_merges
                    return page_tables, next_table_index, page_total_fragment_merges
        except Exception:
            pass  # Fall through to custom detection

    # Strategy 2: Fall back to custom word-clustering table detection
    table_row_groups = _group_table_rows(_cluster_rows(page_words))
    for row_group in table_row_groups:
        candidate_table = _build_single_table_ast(
            rows=row_group,
            page_number=page_number,
            page_height=float(page_rect.height),
            table_id=f"tbl_{next_table_index:03d}",
        )
        if candidate_table is None:
            continue
        table_bbox = tuple(candidate_table["bbox"])
        title_block = _find_table_title_block(table_bbox, text_blocks)
        section_hint_block = _find_section_table_hint(table_bbox, text_blocks)
        toc_context = _is_toc_context(table_bbox, text_blocks)
        grid_line_score = _table_grid_line_score(table_bbox, page_drawings)
        toc_row_ratio = _table_toc_row_ratio(candidate_table)
        reference_like_ratio = _table_reference_like_ratio(candidate_table)

        candidate_table["grid_line_score"] = round(grid_line_score, 3)
        candidate_table["toc_row_ratio"] = round(toc_row_ratio, 3)
        candidate_table["reference_like_ratio"] = round(reference_like_ratio, 3)
        if title_block is not None:
            candidate_table["title"] = title_block["text"]
            candidate_table["title_block_id"] = title_block["block_id"]
        if section_hint_block is not None:
            candidate_table["section_hint"] = section_hint_block["text"]
            candidate_table["section_hint_block_id"] = section_hint_block["block_id"]
            if _is_glossary_hint_text(section_hint_block["text"]):
                candidate_table["continuation_context"] = "glossary"

        continuation_hint: dict[str, Any] | None = None
        if title_block is None:
            continuation_hint = _find_continuation_hint(candidate_table, accepted_tables)
            _resolve_continuation_context(candidate_table, continuation_hint, accepted_tables)
        if not _is_valid_table_candidate(
            candidate_table,
            title_block=title_block,
            section_hint_block=section_hint_block,
            toc_context=toc_context,
            grid_line_score=grid_line_score,
            continuation_hint=continuation_hint,
        ):
            state.counters.rejected_table_candidates += 1
            continue
        if title_block is None:
            candidate_table["title_missing"] = True
        raw_page_tables.append(candidate_table)
        next_table_index += 1

    page_tables, page_fragment_merge_count = _merge_same_page_table_fragments(raw_page_tables)
    page_tables, page_title_merge_count = _merge_same_title_tables(page_tables)
    page_total_fragment_merges = page_fragment_merge_count + page_title_merge_count
    state.counters.table_fragment_merge_count += page_total_fragment_merges
    return page_tables, next_table_index, page_total_fragment_merges


def run_pdf_extraction_pipeline(path: Path) -> PdfPipelineState:
    if pymupdf is None:
        raise RuntimeError("PyMuPDF is required for .pdf parsing. Install with: pip install pymupdf")

    state = PdfPipelineState()
    document = pymupdf.open(path)
    figure_index = 1
    table_index = 1

    try:
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            page_rect = page.rect
            state.page_heights[page_number] = float(page_rect.height)
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

            page_tables, table_index, page_total_fragment_merges = _build_page_tables(
                page_number=page_number,
                page_rect=page_rect,
                text_blocks=text_blocks,
                page_words=page_words,
                page_drawings=page_drawings,
                accepted_tables=state.table_asts,
                table_index=table_index,
                state=state,
                page=page,
            )
            state.table_asts.extend(page_tables)

            page_figures, figure_index = _assign_figure_titles(page_images, text_blocks, figure_index)
            state.figure_nodes.extend(page_figures)
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
                    "table_fragment_merge_count": page_total_fragment_merges,
                }
            )
    finally:
        document.close()

    state.counters.header_footer_filtered_count = _filter_header_footer_text_blocks(state.page_payloads)
    state.counters.cross_page_table_links = _stitch_cross_page_tables(state.table_asts, state.page_heights)
    _renumber_table_ids(state.table_asts)
    return state
