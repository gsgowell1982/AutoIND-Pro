from __future__ import annotations

from pathlib import Path
from typing import Any

from parsers.common.atomic_fact_extractor import extract_atomic_facts

from .text_blocks import _suppress_table_text_blocks
from .types import PDF_PARSER_HINT, PdfPipelineState


def build_pdf_parse_result(path: Path, state: PdfPipelineState) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    text_bounding_boxes: list[dict[str, Any]] = []
    image_blocks: list[dict[str, Any]] = []
    document_ast_pages: list[dict[str, Any]] = []
    analysis_page_texts: list[str] = []
    total_table_text_suppressed = 0

    for page_payload in state.page_payloads:
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
        "figures": state.figure_nodes,
        "table_asts": state.table_asts,
        "tables": state.table_asts,
        "document_ast": {
            "source_type": "pdf",
            "pages": document_ast_pages,
            "table_refs": [table["table_id"] for table in state.table_asts],
            "image_refs": [image["image_id"] for image in image_blocks],
        },
        "text": merged_text,
        "atomic_facts": extract_atomic_facts(merged_text),
        "metadata": {
            "page_count": len(pages),
            "bounding_box_count": len(text_bounding_boxes),
            "image_count": len(image_blocks),
            "table_count": len(state.table_asts),
            "figure_count": len(state.figure_nodes),
            "cross_page_table_links": state.counters.cross_page_table_links,
            "semantic_merge_count": state.counters.semantic_merge_count,
            "image_text_recovered_count": state.counters.image_text_recovered_count,
            "rejected_table_candidates": state.counters.rejected_table_candidates,
            "table_fragment_merge_count": state.counters.table_fragment_merge_count,
            "duplicate_image_blocks_removed": state.counters.duplicate_image_blocks_removed,
            "header_footer_filtered_count": state.counters.header_footer_filtered_count,
            "table_text_suppressed_count": total_table_text_suppressed,
            "parser_hint": PDF_PARSER_HINT,
        },
    }

