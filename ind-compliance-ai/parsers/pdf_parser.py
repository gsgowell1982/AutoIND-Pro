from pathlib import Path
from typing import Any

import pymupdf

from parsers.common.atomic_fact_extractor import extract_atomic_facts


def parse_pdf(path: Path) -> dict[str, Any]:
    """Parse PDF content into normalized material representation."""
    document = pymupdf.open(path)
    pages: list[dict[str, Any]] = []
    bounding_boxes: list[dict[str, Any]] = []
    all_page_text: list[str] = []

    for page_index, page in enumerate(document):
        page_number = page_index + 1
        page_blocks = page.get_text("blocks")
        text_fragments: list[str] = []

        for block_index, block in enumerate(page_blocks):
            x0, y0, x1, y1, raw_text = block[:5]
            text = " ".join(str(raw_text).split())
            if not text:
                continue
            text_fragments.append(text)
            bounding_boxes.append(
                {
                    "id": f"p{page_number}-b{block_index}",
                    "page_number": page_number,
                    "text": text,
                    "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                }
            )

        page_text = "\n".join(text_fragments)
        all_page_text.append(page_text)
        page_rect = page.rect
        pages.append(
            {
                "page_number": page_number,
                "width": page_rect.width,
                "height": page_rect.height,
                "text": page_text,
                "block_count": len(text_fragments),
            }
        )

    document.close()
    merged_text = "\n\n".join(all_page_text)

    return {
        "source_path": str(path),
        "source_type": "pdf",
        "pages": pages,
        "bounding_boxes": bounding_boxes,
        "text": merged_text,
        "atomic_facts": extract_atomic_facts(merged_text),
        "metadata": {
            "page_count": len(pages),
            "bounding_box_count": len(bounding_boxes),
        },
    }
