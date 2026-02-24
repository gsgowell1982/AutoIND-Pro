from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import pymupdf
except ImportError:  # pragma: no cover - optional runtime dependency
    pymupdf = None  # type: ignore[assignment]

from parsers.common.atomic_fact_extractor import extract_atomic_facts
from parsers.pdf.image_blocks import (
    _assign_figure_titles,
    _deduplicate_page_images,
    _demote_textual_image_blocks,
)
from parsers.pdf.layout import _extract_words
from parsers.pdf.tables import (
    _build_single_table_ast,
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
from parsers.pdf.text_blocks import (
    _extract_page_text_and_images,
    _filter_header_footer_text_blocks,
    _merge_semantic_text_blocks,
    _suppress_table_text_blocks,
)


def parse_pdf(path: Path) -> dict[str, Any]:
    """Parse PDF into text/image/table AST while preserving BBox anchors."""
    if pymupdf is None:
        raise RuntimeError("PyMuPDF is required for .pdf parsing. Install with: pip install pymupdf")

    document = pymupdf.open(path)
    pages: list[dict[str, Any]] = []
    text_bounding_boxes: list[dict[str, Any]] = []
    image_blocks: list[dict[str, Any]] = []
    figure_nodes: list[dict[str, Any]] = []
    table_asts: list[dict[str, Any]] = []
    document_ast_pages: list[dict[str, Any]] = []
    all_page_text: list[str] = []
    page_heights: dict[int, float] = {}
    page_payloads: list[dict[str, Any]] = []

    total_semantic_merges = 0
    total_recovered_image_text = 0
    rejected_table_candidates = 0
    total_table_fragment_merges = 0
    total_header_footer_filtered = 0
    total_table_text_suppressed = 0
    total_duplicate_image_blocks_removed = 0
    figure_index = 1
    table_index = 1

    for page_index, page in enumerate(document):
        page_number = page_index + 1
        page_rect = page.rect
        page_heights[page_number] = float(page_rect.height)
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
        total_recovered_image_text += recovered_image_text
        page_images, duplicate_image_blocks_removed = _deduplicate_page_images(page_images, page_number)
        total_duplicate_image_blocks_removed += duplicate_image_blocks_removed

        # Step-2: Semantic + layout-aware merge for over-segmented text blocks.
        text_blocks, semantic_merge_count = _merge_semantic_text_blocks(
            text_blocks=text_blocks,
            page_number=page_number,
            page_width=float(page_rect.width),
        )
        total_semantic_merges += semantic_merge_count

        table_row_groups = _group_table_rows(_cluster_rows(page_words))
        raw_page_tables: list[dict[str, Any]] = []
        for row_group in table_row_groups:
            candidate_table = _build_single_table_ast(
                rows=row_group,
                page_number=page_number,
                page_height=float(page_rect.height),
                table_id=f"tbl_{table_index:03d}",
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
                continuation_hint = _find_continuation_hint(candidate_table, table_asts)
                if continuation_hint is not None:
                    candidate_table["continuation_hint"] = continuation_hint
                    parent_table = next(
                        (
                            table
                            for table in reversed(table_asts)
                            if str(table.get("table_id", "")) == str(continuation_hint.get("table_id", ""))
                        ),
                        None,
                    )
                    if parent_table is not None and str(parent_table.get("continuation_context", "")).strip():
                        candidate_table["continuation_context"] = parent_table["continuation_context"]
            if not _is_valid_table_candidate(
                candidate_table,
                title_block=title_block,
                section_hint_block=section_hint_block,
                toc_context=toc_context,
                grid_line_score=grid_line_score,
                continuation_hint=continuation_hint,
            ):
                rejected_table_candidates += 1
                continue
            if title_block is None:
                candidate_table["title_missing"] = True
            raw_page_tables.append(candidate_table)
            table_index += 1

        page_tables, page_fragment_merge_count = _merge_same_page_table_fragments(raw_page_tables)
        page_tables, page_title_merge_count = _merge_same_title_tables(page_tables)
        page_total_fragment_merges = page_fragment_merge_count + page_title_merge_count
        total_table_fragment_merges += page_total_fragment_merges
        table_asts.extend(page_tables)

        page_figures, figure_index = _assign_figure_titles(page_images, text_blocks, figure_index)
        figure_nodes.extend(page_figures)
        page_payloads.append(
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

    document.close()
    total_header_footer_filtered = _filter_header_footer_text_blocks(page_payloads)
    stitched_table_count = _stitch_cross_page_tables(table_asts, page_heights)
    _renumber_table_ids(table_asts)

    analysis_page_texts: list[str] = []
    for page_payload in page_payloads:
        page_number = int(page_payload["page_number"])
        ordered_analysis_blocks = sorted(page_payload["text_blocks"], key=lambda item: (item["bbox"][1], item["bbox"][0]))
        analysis_text = "\n".join(item["text"] for item in ordered_analysis_blocks).strip()
        if analysis_text:
            analysis_page_texts.append(analysis_text)

        visible_text_blocks, table_text_suppressed_count = _suppress_table_text_blocks(
            ordered_analysis_blocks,
            page_payload["tables"],
        )
        total_table_text_suppressed += table_text_suppressed_count

        page_text = "\n".join(item["text"] for item in visible_text_blocks).strip()
        all_page_text.append(page_text)
        page_bbox_nodes: list[dict[str, Any]] = []

        for text_block in visible_text_blocks:
            bbox = text_block["bbox"]
            text_bounding_boxes.append(
                {
                    "id": text_block["block_id"],
                    "page_number": page_number,
                    "text": text_block["text"],
                    "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                }
            )
            page_bbox_nodes.append(
                {
                    "block_type": "text",
                    "block_id": text_block["block_id"],
                    "page": page_number,
                    "bbox": bbox,
                    "text": text_block["text"],
                    "source": text_block.get("source", "text-layer"),
                }
            )

        for image_block in page_payload["images"]:
            bbox = image_block["bbox"]
            image_blocks.append(image_block)
            text_bounding_boxes.append(
                {
                    "id": image_block["image_id"],
                    "page_number": page_number,
                    "text": f"[IMAGE] {image_block.get('title', '')}".strip(),
                    "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                }
            )
            page_bbox_nodes.append(
                {
                    "block_type": "image",
                    "block_id": image_block["image_id"],
                    "page": page_number,
                    "bbox": bbox,
                    "image_id": image_block["image_id"],
                    "figure_ref": image_block.get("figure_ref"),
                    "title": image_block.get("title", ""),
                    "text_recovery": image_block.get("text_recovery", {}),
                }
            )

        for table_ast in page_payload["tables"]:
            bbox = table_ast["bbox"]
            text_bounding_boxes.append(
                {
                    "id": table_ast["table_id"],
                    "page_number": page_number,
                    "text": f"[TABLE] {table_ast['table_id']}",
                    "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                }
            )
            page_bbox_nodes.append(
                {
                    "block_type": "table",
                    "block_id": table_ast["table_id"],
                    "page": page_number,
                    "bbox": bbox,
                    "table_id": table_ast["table_id"],
                    "column_hash": table_ast.get("column_hash"),
                    "title": table_ast.get("title", ""),
                }
            )

        page_bbox_nodes.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        document_ast_pages.append({"page": page_number, "blocks": page_bbox_nodes})
        pages.append(
            {
                "page_number": page_number,
                "width": float(page_payload["width"]),
                "height": float(page_payload["height"]),
                "text": page_text,
                "block_count": len(visible_text_blocks),
                "image_count": len(page_payload["images"]),
                "table_count": len(page_payload["tables"]),
                "semantic_merge_count": int(page_payload["semantic_merge_count"]),
                "image_text_recovered_count": int(page_payload["image_text_recovered_count"]),
                "header_footer_filtered_count": int(page_payload.get("header_footer_filtered", 0)),
                "table_text_suppressed_count": table_text_suppressed_count,
                "table_fragment_merge_count": int(page_payload.get("table_fragment_merge_count", 0)),
            }
        )

    merged_text = "\n\n".join(analysis_page_texts)

    return {
        "source_path": str(path),
        "source_type": "pdf",
        "pages": pages,
        "bounding_boxes": text_bounding_boxes,
        "image_blocks": image_blocks,
        "figures": figure_nodes,
        "table_asts": table_asts,
        "tables": table_asts,
        "document_ast": {
            "source_type": "pdf",
            "pages": document_ast_pages,
            "table_refs": [table["table_id"] for table in table_asts],
            "image_refs": [image["image_id"] for image in image_blocks],
        },
        "text": merged_text,
        "atomic_facts": extract_atomic_facts(merged_text),
        "metadata": {
            "page_count": len(pages),
            "bounding_box_count": len(text_bounding_boxes),
            "image_count": len(image_blocks),
            "table_count": len(table_asts),
            "figure_count": len(figure_nodes),
            "cross_page_table_links": stitched_table_count,
            "semantic_merge_count": total_semantic_merges,
            "image_text_recovered_count": total_recovered_image_text,
            "rejected_table_candidates": rejected_table_candidates,
            "table_fragment_merge_count": total_table_fragment_merges,
            "duplicate_image_blocks_removed": total_duplicate_image_blocks_removed,
            "header_footer_filtered_count": total_header_footer_filtered,
            "table_text_suppressed_count": total_table_text_suppressed,
            "parser_hint": "pdf-ast-v5",
        },
    }

