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

# Version: v1.0.5
# Updates:
# - Pass page height and page words into semantic text merging so footer-artifact
#   merge barriers can use unified words-layer continuity evidence.
# - Keep figure title assignment, table-context extraction, and text output on
#   the same reconstructed-text path while preventing footer contamination.

from pathlib import Path
import re
import time
from typing import Any

try:
    import pymupdf
except ImportError:  # pragma: no cover - optional runtime dependency
    pymupdf = None  # type: ignore[assignment]

from .image_blocks import (
    _assign_figure_titles,
    _deduplicate_page_images,
    _demote_textual_image_blocks,
    _enrich_image_content,
    _filter_non_content_images,
    _synthesize_vector_figure_blocks,
)
from .layout import (
    _extract_words,
    annotate_text_blocks_with_layout,
    infer_page_text_layout_profile,
)
from .ocr_policy import build_page_ocr_context, should_attempt_full_page_ocr
from .region_ownership import arbitrate_region_ownership, collect_region_candidates
from .tables import (
    # Main entry points
    extract_tables_from_page,
    extract_tables_from_document,
    # Postprocess functions
    stitch_cross_page_tables,
    renumber_table_ids,
    merge_same_page_table_fragments,
    split_internal_table_segments,
)
from .text_blocks import (
    _extract_page_text_and_images,
    _filter_header_footer_text_blocks,
    _merge_semantic_text_blocks,
    annotate_table_presentation_bboxes,
    recover_sparse_text_layer_page_with_full_page_ocr,
    repair_suspicious_body_text_blocks_with_local_ocr,
    reconstruct_visual_text_lines,
    _suppress_table_text_blocks,
)
from .types import PdfPipelineState


def _record_stage_time(state: PdfPipelineState, stage: str, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    state.stage_timings[stage] = round(float(state.stage_timings.get(stage, 0.0) or 0.0) + elapsed, 6)


def _document_has_dominant_text_layer(document: Any) -> bool:
    total_pages = 0
    healthy_pages = 0
    token_total = 0
    for page in document:
        total_pages += 1
        try:
            text = str(page.get_text("text") or "")
        except Exception:
            text = ""
        token_count = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", text))
        token_total += token_count
        if token_count >= 16:
            healthy_pages += 1
    if total_pages <= 0:
        return False
    return healthy_pages >= max(2, int(total_pages * 0.55)) and token_total >= total_pages * 20


def _attach_region_ownership_observations(state: PdfPipelineState) -> None:
    candidates = collect_region_candidates(
        page_payloads=state.page_payloads,
        table_asts=state.table_asts,
        toc_nodes=state.toc_nodes,
        figure_nodes=state.figure_nodes,
        external_candidates=state.external_region_candidates,
    )
    result = arbitrate_region_ownership(candidates)
    payload = result.to_dict()
    state.region_candidates = payload["candidates"]
    state.ownership_decisions = payload["decisions"]
    state.region_nodes = payload["nodes"]
    state.counters.region_candidate_count = payload["summary"]["candidate_count"]
    state.counters.region_ownership_decision_count = payload["summary"]["decision_count"]
    state.counters.region_node_count = payload["summary"]["node_count"]


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


def _extract_page_link_annotation_counts(page: Any) -> dict[str, int]:
    """Collect bounded link-annotation evidence for downstream navigation checks."""
    counts = {
        "link_annotation_count": 0,
        "navigational_link_count": 0,
        "internal_link_count": 0,
        "external_file_link_count": 0,
        "external_uri_link_count": 0,
        "external_file_link_targets": [],
        "link_annotation_xrefs": [],
    }
    try:
        links = list(page.get_links() or [])
    except Exception:
        return counts

    for link in links:
        kind = int(link.get("kind", 0) or 0)
        if kind <= 0:
            continue
        counts["link_annotation_count"] += 1
        xref = int(link.get("xref", 0) or 0)
        if xref > 0:
            counts["link_annotation_xrefs"].append(xref)
        if kind == pymupdf.LINK_GOTO:
            counts["internal_link_count"] += 1
        elif kind in (pymupdf.LINK_GOTOR, pymupdf.LINK_LAUNCH):
            counts["external_file_link_count"] += 1
            target = str(link.get("file") or "").strip()
            if target:
                counts["external_file_link_targets"].append(target)
        elif kind == pymupdf.LINK_URI:
            counts["external_uri_link_count"] += 1

    counts["navigational_link_count"] = (
        counts["internal_link_count"] + counts["external_file_link_count"]
    )
    return counts


def _extract_page_non_link_annotation_counts(page: Any) -> dict[str, Any]:
    """Collect non-link PDF annotation evidence for downstream validation checks."""
    counts = {
        "non_link_annotation_count": 0,
        "non_link_annotation_types": [],
    }
    try:
        annots = list(page.annots() or [])
    except Exception:
        return counts

    link_annot_type = int(getattr(pymupdf, "PDF_ANNOT_LINK", 1) or 1)
    for annot in annots:
        annot_type = getattr(annot, "type", None)
        if not isinstance(annot_type, tuple) or not annot_type:
            continue
        annot_kind = int(annot_type[0] or 0)
        if annot_kind == link_annot_type:
            continue
        counts["non_link_annotation_count"] += 1
        annot_name = str(annot_type[1] or "").strip()
        if annot_name and annot_name not in counts["non_link_annotation_types"]:
            counts["non_link_annotation_types"].append(annot_name)
    return counts


def _extract_document_embedded_file_metadata(document: Any) -> dict[str, Any]:
    """Collect embedded-file metadata for bounded PDF attachment checks."""
    metadata = {
        "embedded_file_count": 0,
        "embedded_file_names": [],
    }
    try:
        metadata["embedded_file_count"] = int(document.embfile_count() or 0)
    except Exception:
        metadata["embedded_file_count"] = 0
    if metadata["embedded_file_count"] <= 0:
        return metadata
    try:
        metadata["embedded_file_names"] = [
            str(item).strip()
            for item in list(document.embfile_names() or [])
            if str(item).strip()
        ]
    except Exception:
        metadata["embedded_file_names"] = []
    return metadata


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
    document_text_layer_dominant = _document_has_dominant_text_layer(document)
    figure_index = 1
    table_counter = 0
    toc_counter = 0
    try:
        embedded_toc = document.get_toc()
    except Exception:
        embedded_toc = []
    state.embedded_outline_count = len(embedded_toc)
    state.embedded_outline_depth = max((int(item[0] or 0) for item in embedded_toc), default=0)
    embedded_file_metadata = _extract_document_embedded_file_metadata(document)
    state.embedded_file_count = int(embedded_file_metadata["embedded_file_count"] or 0)
    state.embedded_file_names = list(embedded_file_metadata["embedded_file_names"] or [])

    try:
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            page_rect = page.rect
            state.page_heights[page_number] = float(page_rect.height)
            page_link_counts = _extract_page_link_annotation_counts(page)
            page_annotation_counts = _extract_page_non_link_annotation_counts(page)
            state.link_annotation_count += page_link_counts["link_annotation_count"]
            state.navigational_link_count += page_link_counts["navigational_link_count"]
            state.internal_link_count += page_link_counts["internal_link_count"]
            state.external_file_link_count += page_link_counts["external_file_link_count"]
            state.external_uri_link_count += page_link_counts["external_uri_link_count"]
            state.external_file_link_targets.extend(page_link_counts["external_file_link_targets"])
            for xref in page_link_counts["link_annotation_xrefs"]:
                state.link_annotation_xref_page_map[int(xref)] = page_number
            state.non_link_annotation_count += page_annotation_counts["non_link_annotation_count"]
            for annot_type in page_annotation_counts["non_link_annotation_types"]:
                if annot_type not in state.non_link_annotation_types:
                    state.non_link_annotation_types.append(annot_type)

            # Extract raw content from page
            stage_start = time.perf_counter()
            text_blocks, page_images = _extract_page_text_and_images(page, page_number)
            _record_stage_time(state, "extract_page_text_and_images", stage_start)

            stage_start = time.perf_counter()
            page_words = _extract_words(page)
            _record_stage_time(state, "extract_words", stage_start)
            try:
                stage_start = time.perf_counter()
                page_drawings = page.get_drawings()
                _record_stage_time(state, "get_drawings", stage_start)
            except Exception:
                page_drawings = []

            ocr_context = build_page_ocr_context(
                text_blocks=text_blocks,
                page_words=page_words,
                page_drawings=page_drawings,
                page_width=float(page_rect.width),
                page_height=float(page_rect.height),
                document_text_layer_dominant=document_text_layer_dominant,
            )
            full_page_ocr_decision = should_attempt_full_page_ocr(ocr_context)
            if full_page_ocr_decision.enabled:
                stage_start = time.perf_counter()
                text_blocks, full_page_ocr_recovered = recover_sparse_text_layer_page_with_full_page_ocr(
                    page=page,
                    page_number=page_number,
                    text_blocks=text_blocks,
                    page_width=float(page_rect.width),
                    page_height=float(page_rect.height),
                    page_words=page_words,
                    page_drawings=page_drawings,
                )
                _record_stage_time(state, "full_page_ocr", stage_start)
                if full_page_ocr_recovered:
                    ocr_context = build_page_ocr_context(
                        text_blocks=text_blocks,
                        page_words=page_words,
                        page_drawings=page_drawings,
                        page_width=float(page_rect.width),
                        page_height=float(page_rect.height),
                        document_text_layer_dominant=document_text_layer_dominant,
                    )
            else:
                full_page_ocr_recovered = 0
            state.counters.full_page_ocr_recovered_count += full_page_ocr_recovered

            stage_start = time.perf_counter()
            page_images = _synthesize_vector_figure_blocks(
                image_blocks=page_images,
                text_blocks=text_blocks,
                page_drawings=page_drawings,
                page_number=page_number,
                page_width=float(page_rect.width),
            )
            _record_stage_time(state, "synthesize_vector_figure_blocks", stage_start)

            # Step-1: Correct image-vs-text confusion using text-layer/OCR/path signals.
            stage_start = time.perf_counter()
            text_blocks, page_images, recovered_image_text = _demote_textual_image_blocks(
                page=page,
                page_number=page_number,
                page_rect=page_rect,
                image_blocks=page_images,
                text_blocks=text_blocks,
                page_words=page_words,
                page_drawings=page_drawings,
                ocr_context=ocr_context,
            )
            _record_stage_time(state, "demote_textual_image_blocks", stage_start)
            state.counters.image_text_recovered_count += recovered_image_text
            state.counters.image_text_ocr_skipped_count += sum(
                1
                for image in page_images
                if str((image.get("text_recovery") or {}).get("source", "") or "") == "ocr-skipped"
            )
            stage_start = time.perf_counter()
            page_images, duplicate_image_blocks_removed = _deduplicate_page_images(page_images, page_number)
            _record_stage_time(state, "deduplicate_page_images", stage_start)
            state.counters.duplicate_image_blocks_removed += duplicate_image_blocks_removed

            stage_start = time.perf_counter()
            layout_profile = infer_page_text_layout_profile(
                text_blocks=text_blocks,
                page_words=page_words,
                page_width=float(page_rect.width),
                page_height=float(page_rect.height),
            )
            _record_stage_time(state, "infer_page_text_layout_profile", stage_start)
            annotate_text_blocks_with_layout(text_blocks, layout_profile)

            # Step-2: Repair suspicious local body-text gaps with OCR only when
            # the text layer appears corrupted inside a narrow body region.
            stage_start = time.perf_counter()
            text_blocks, _ = repair_suspicious_body_text_blocks_with_local_ocr(
                page=page,
                page_number=page_number,
                text_blocks=text_blocks,
                page_width=float(page_rect.width),
                page_height=float(page_rect.height),
                layout_profile=layout_profile,
            )
            _record_stage_time(state, "local_body_text_ocr_repair", stage_start)

            # Step-3: Reconstruct visual text lines using words-layer continuity.
            stage_start = time.perf_counter()
            text_blocks, visual_line_merge_count = reconstruct_visual_text_lines(
                text_blocks=text_blocks,
                page_words=page_words,
                page_number=page_number,
                page_width=float(page_rect.width),
                layout_profile=layout_profile,
            )
            _record_stage_time(state, "reconstruct_visual_text_lines", stage_start)

            # Step-4: Semantic + layout-aware merge for over-segmented text blocks.
            stage_start = time.perf_counter()
            text_blocks, semantic_merge_count = _merge_semantic_text_blocks(
                text_blocks=text_blocks,
                page_number=page_number,
                page_width=float(page_rect.width),
                page_height=float(page_rect.height),
                page_words=page_words,
                layout_profile=layout_profile,
            )
            _record_stage_time(state, "merge_semantic_text_blocks", stage_start)
            semantic_merge_count += visual_line_merge_count
            state.counters.semantic_merge_count += semantic_merge_count

            # Step-5: Extract tables using unified architecture.
            # Clone reconstructed text blocks so downstream suppression can mutate
            # the page payload without changing the table-context snapshot.
            context_text_blocks = [
                {
                    **block,
                    "bbox": list(block.get("bbox", [])),
                }
                for block in text_blocks
                if str(block.get("text", "")).strip()
            ]
            stage_start = time.perf_counter()
            context_drawings = _get_drawings(page)
            _record_stage_time(state, "get_context_drawings", stage_start)

            page_table_stats: dict[str, int] = {}
            page_toc_blocks: list[dict[str, Any]] = []
            stage_start = time.perf_counter()
            page_tables, table_counter = extract_tables_from_page(
                page=page,
                page_number=page_number,
                page_height=float(page_rect.height),
                text_blocks=context_text_blocks,
                page_drawings=context_drawings,
                image_blocks=page_images,
                layout_profile=layout_profile,
                prev_tables=state.table_asts,
                table_counter=table_counter,
                out_stats=page_table_stats,
                out_toc_blocks=page_toc_blocks,
                ocr_context=ocr_context,
            )
            _record_stage_time(state, "extract_tables_from_page", stage_start)
            state.counters.raw_table_candidates += page_table_stats.get("raw_candidates", 0)
            state.counters.accepted_table_candidates += page_table_stats.get("accepted", 0)
            state.counters.toc_block_count += page_table_stats.get("toc_outlines", 0)
            state.counters.rejected_table_candidates += page_table_stats.get("rejected", 0)
            state.counters.embedded_image_table_ocr_skipped_count += page_table_stats.get("embedded_image_table_ocr_skipped", 0)

            # Merge same-page fragments if multiple tables found
            if len(page_tables) > 1:
                page_tables, merge_count = merge_same_page_table_fragments(page_tables)
                state.counters.table_fragment_merge_count += merge_count
            else:
                merge_count = 0
            split_internal_table_segments(page_tables)

            if page_toc_blocks:
                for toc_block in page_toc_blocks:
                    toc_counter += 1
                    toc_block["toc_id"] = f"toc_{toc_counter:03d}"
                state.toc_nodes.extend(page_toc_blocks)

            # Add tables to state
            state.table_asts.extend(page_tables)

            # Step-6: Assign figure titles
            stage_start = time.perf_counter()
            page_figures, figure_index = _assign_figure_titles(
                page_images, text_blocks, figure_index, float(page_rect.height)
            )
            _record_stage_time(state, "assign_figure_titles", stage_start)
            stage_start = time.perf_counter()
            page_images, page_figures = _enrich_image_content(
                image_blocks=page_images,
                figure_nodes=page_figures,
                text_blocks=text_blocks,
                page_height=float(page_rect.height),
                page=page,
            )
            _record_stage_time(state, "enrich_image_content", stage_start)
            stage_start = time.perf_counter()
            page_images, page_figures, _ = _filter_non_content_images(page_images, page_figures)
            _record_stage_time(state, "filter_non_content_images", stage_start)
            state.figure_nodes.extend(page_figures)

            suppressed_regions = page_tables + page_toc_blocks
            annotate_table_presentation_bboxes(text_blocks, page_tables)
            stage_start = time.perf_counter()
            text_blocks, suppressed_count = _suppress_table_text_blocks(text_blocks, suppressed_regions)
            _record_stage_time(state, "suppress_table_text_blocks", stage_start)
            state.counters.table_text_suppressed_count += suppressed_count

            # Store page payload
            state.page_payloads.append(
                {
                    "page_number": page_number,
                    "width": float(page_rect.width),
                    "height": float(page_rect.height),
                    "page_words": page_words,
                    "page_drawings": page_drawings,
                    "text_blocks": text_blocks,
                    "images": page_images,
                    "tables": page_tables,
                    "toc_blocks": page_toc_blocks,
                    "layout_profile": layout_profile,
                    "semantic_merge_count": semantic_merge_count,
                    "image_text_recovered_count": recovered_image_text,
                    "full_page_ocr_recovered_count": full_page_ocr_recovered,
                    "table_fragment_merge_count": merge_count,
                }
            )
    finally:
        document.close()

    # Post-processing: filter header/footer, stitch cross-page tables, renumber
    state.counters.header_footer_filtered_count = _filter_header_footer_text_blocks(state.page_payloads)
    state.counters.cross_page_table_links = stitch_cross_page_tables(state.table_asts, state.page_heights)
    renumber_table_ids(state.table_asts)
    _attach_region_ownership_observations(state)

    return state
