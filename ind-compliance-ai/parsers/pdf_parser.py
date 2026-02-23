from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import statistics
from typing import Any

try:
    import pymupdf
except ImportError:  # pragma: no cover - optional runtime dependency
    pymupdf = None  # type: ignore[assignment]

from parsers.common.atomic_fact_extractor import extract_atomic_facts


def _clean_text(value: str) -> str:
    return " ".join(str(value).split())


def _bbox_to_list(bbox: tuple[float, float, float, float]) -> list[float]:
    return [round(float(item), 2) for item in bbox]


def _bbox_union(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    x0 = min(item[0] for item in bboxes)
    y0 = min(item[1] for item in bboxes)
    x1 = max(item[2] for item in bboxes)
    y1 = max(item[3] for item in bboxes)
    return (x0, y0, x1, y1)


def _horizontal_overlap_ratio(
    a_bbox: tuple[float, float, float, float],
    b_bbox: tuple[float, float, float, float],
) -> float:
    left = max(a_bbox[0], b_bbox[0])
    right = min(a_bbox[2], b_bbox[2])
    overlap = max(0.0, right - left)
    min_width = max(1.0, min(a_bbox[2] - a_bbox[0], b_bbox[2] - b_bbox[0]))
    return overlap / min_width


def _extract_page_text_and_images(
    page: "pymupdf.Page",
    page_number: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    page_dict = page.get_text("dict")
    text_blocks: list[dict[str, Any]] = []
    image_blocks: list[dict[str, Any]] = []

    for block_index, block in enumerate(page_dict.get("blocks", [])):
        block_type = int(block.get("type", 0))
        bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        if block_type == 0:
            line_texts: list[str] = []
            for line in block.get("lines", []):
                span_text = "".join(str(span.get("text", "")) for span in line.get("spans", []))
                cleaned = _clean_text(span_text)
                if cleaned:
                    line_texts.append(cleaned)
            text = "\n".join(line_texts).strip()
            if not text:
                continue
            text_blocks.append(
                {
                    "block_type": "text",
                    "block_id": f"txt_p{page_number}_{len(text_blocks) + 1:03d}",
                    "page": page_number,
                    "bbox": _bbox_to_list(bbox),
                    "text": text,
                    "source_block_index": block_index,
                }
            )
        elif block_type == 1:
            image_blocks.append(
                {
                    "block_type": "image",
                    "image_id": f"img_p{page_number}_{len(image_blocks) + 1:03d}",
                    "page": page_number,
                    "bbox": _bbox_to_list(bbox),
                    "width": round(max(0.0, bbox[2] - bbox[0]), 2),
                    "height": round(max(0.0, bbox[3] - bbox[1]), 2),
                    "source_block_index": block_index,
                }
            )
    return text_blocks, image_blocks


def _assign_figure_titles(
    image_blocks: list[dict[str, Any]],
    text_blocks: list[dict[str, Any]],
    figure_start_index: int,
) -> tuple[list[dict[str, Any]], int]:
    figure_nodes: list[dict[str, Any]] = []
    figure_index = figure_start_index

    for image_block in image_blocks:
        image_bbox = tuple(image_block["bbox"])
        below_candidates: list[tuple[float, str]] = []
        nearby_candidates: list[tuple[float, str]] = []
        for text_block in text_blocks:
            text_bbox = tuple(text_block["bbox"])
            overlap_ratio = _horizontal_overlap_ratio(image_bbox, text_bbox)
            if overlap_ratio < 0.15:
                continue
            text = _clean_text(text_block["text"])
            if not text:
                continue
            distance = text_bbox[1] - image_bbox[3]
            if distance >= 0:
                below_candidates.append((distance, text))
            else:
                nearby_candidates.append((abs(distance), text))

        title_text = ""
        if below_candidates:
            title_text = sorted(below_candidates, key=lambda item: item[0])[0][1]
        elif nearby_candidates:
            title_text = sorted(nearby_candidates, key=lambda item: item[0])[0][1]
        if title_text:
            title_text = title_text[:160]

        figure_ref = f"Figure {figure_index}"
        image_block["title"] = title_text
        image_block["figure_ref"] = figure_ref
        figure_nodes.append(
            {
                "figure_ref": figure_ref,
                "image_id": image_block["image_id"],
                "title": title_text,
                "page": image_block["page"],
                "bbox": image_block["bbox"],
            }
        )
        figure_index += 1

    return figure_nodes, figure_index


@dataclass(slots=True)
class _Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


def _extract_words(page: "pymupdf.Page") -> list[_Word]:
    words: list[_Word] = []
    for item in page.get_text("words", sort=True):
        if len(item) < 5:
            continue
        x0, y0, x1, y1, raw_text = item[:5]
        text = _clean_text(str(raw_text))
        if not text:
            continue
        words.append(_Word(float(x0), float(y0), float(x1), float(y1), text))
    return words


def _cluster_rows(words: list[_Word]) -> list[dict[str, Any]]:
    if not words:
        return []
    median_height = statistics.median([word.height for word in words]) if words else 8.0
    row_tol = max(2.5, median_height * 0.55)
    rows: list[dict[str, Any]] = []

    for word in sorted(words, key=lambda item: (item.yc, item.x0)):
        placed = False
        for row in rows:
            if abs(word.yc - row["yc"]) <= row_tol:
                row["words"].append(word)
                row["yc"] = (row["yc"] * row["n"] + word.yc) / (row["n"] + 1)
                row["n"] += 1
                placed = True
                break
        if not placed:
            rows.append({"yc": word.yc, "n": 1, "words": [word]})

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        row_words = sorted(row["words"], key=lambda item: item.x0)
        row_bbox = _bbox_union([(item.x0, item.y0, item.x1, item.y1) for item in row_words])
        normalized_rows.append(
            {
                "words": row_words,
                "bbox": row_bbox,
                "y0": row_bbox[1],
                "y1": row_bbox[3],
                "height": max(0.1, row_bbox[3] - row_bbox[1]),
            }
        )

    normalized_rows.sort(key=lambda item: item["y0"])
    return normalized_rows


def _group_table_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for row in rows:
        if len(row["words"]) < 2:
            if len(current) >= 2:
                groups.append(current)
            current = []
            continue

        if not current:
            current = [row]
            continue

        previous = current[-1]
        vertical_gap = row["y0"] - previous["y1"]
        avg_height = (row["height"] + previous["height"]) / 2
        if vertical_gap <= max(6.0, avg_height * 1.25):
            current.append(row)
        else:
            if len(current) >= 2:
                groups.append(current)
            current = [row]

    if len(current) >= 2:
        groups.append(current)

    return [group for group in groups if max(len(row["words"]) for row in group) >= 2]


def _cluster_columns(rows: list[dict[str, Any]]) -> list[float]:
    words = [word for row in rows for word in row["words"]]
    if not words:
        return []
    median_width = statistics.median([word.width for word in words]) if words else 12.0
    col_tol = max(8.0, median_width * 0.9)
    centers = sorted(word.xc for word in words)
    if not centers:
        return []

    clusters: list[list[float]] = [[centers[0]]]
    for center in centers[1:]:
        if abs(center - clusters[-1][-1]) <= col_tol:
            clusters[-1].append(center)
        else:
            clusters.append([center])

    return [sum(cluster) / len(cluster) for cluster in clusters]


def _closest_column_index(x_center: float, column_centers: list[float]) -> int:
    return min(range(len(column_centers)), key=lambda index: abs(column_centers[index] - x_center))


def _column_hash(column_signature: list[float]) -> str:
    signature = ",".join(f"{value:.3f}" for value in column_signature)
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]


def _build_single_table_ast(
    rows: list[dict[str, Any]],
    page_number: int,
    page_height: float,
    table_id: str,
) -> dict[str, Any] | None:
    column_centers = _cluster_columns(rows)
    if len(column_centers) < 2 or len(column_centers) > 24:
        return None

    table_bbox = _bbox_union([row["bbox"] for row in rows])
    table_width = max(1.0, table_bbox[2] - table_bbox[0])
    column_signature = [round((center - table_bbox[0]) / table_width, 3) for center in column_centers]

    row_cell_maps: list[dict[int, list[_Word]]] = []
    row_bboxes: list[tuple[float, float, float, float]] = []
    for row in rows:
        cell_map: dict[int, list[_Word]] = {}
        for word in row["words"]:
            col_index = _closest_column_index(word.xc, column_centers)
            cell_map.setdefault(col_index, []).append(word)
        row_cell_maps.append(cell_map)
        row_bboxes.append(row["bbox"])

    header_row_index = 0
    for index, cell_map in enumerate(row_cell_maps):
        non_empty = sum(1 for words in cell_map.values() if words)
        if non_empty >= 2:
            header_row_index = index
            break

    header: list[dict[str, Any]] = []
    header_cells = row_cell_maps[header_row_index] if row_cell_maps else {}
    for col_index in range(len(column_centers)):
        words = sorted(header_cells.get(col_index, []), key=lambda item: item.x0)
        text = _clean_text(" ".join(word.text for word in words))
        header.append({"text": text or f"Column {col_index + 1}", "col": col_index + 1})

    cells: list[dict[str, Any]] = []
    for raw_row_index, cell_map in enumerate(row_cell_maps[header_row_index + 1 :], start=1):
        for col_index in range(len(column_centers)):
            words = sorted(cell_map.get(col_index, []), key=lambda item: item.x0)
            if not words:
                continue
            cell_text = _clean_text(" ".join(word.text for word in words))
            if not cell_text:
                continue
            bbox = _bbox_union([(word.x0, word.y0, word.x1, word.y1) for word in words])
            covered_columns = [
                index
                for index, center in enumerate(column_centers)
                if bbox[0] - 2 <= center <= bbox[2] + 2
            ]
            colspan = max(1, len(covered_columns))
            cells.append(
                {
                    "row": raw_row_index,
                    "col": col_index + 1,
                    "text": cell_text,
                    "bbox": _bbox_to_list(bbox),
                    "rowspan": 1,
                    "colspan": colspan,
                }
            )

    if not cells:
        return None

    return {
        "block_type": "table",
        "table_id": table_id,
        "page": page_number,
        "bbox": _bbox_to_list(table_bbox),
        "header": header,
        "cells": cells,
        "row_count": len(row_cell_maps),
        "col_count": len(column_centers),
        "column_signature": column_signature,
        "column_hash": _column_hash(column_signature),
        "header_row_index": header_row_index + 1,
        "near_page_bottom": table_bbox[3] >= page_height * 0.72,
        "near_page_top": table_bbox[1] <= page_height * 0.28,
    }


def _column_similarity(signature_a: list[float], signature_b: list[float]) -> float:
    if not signature_a or not signature_b:
        return 0.0
    tolerance = 0.07
    matched = 0
    for value in signature_a:
        if any(abs(value - other) <= tolerance for other in signature_b):
            matched += 1
    return matched / max(len(signature_a), len(signature_b))


def _header_similarity(header_a: list[dict[str, Any]], header_b: list[dict[str, Any]]) -> float:
    texts_a = [str(item.get("text", "")).strip().lower() for item in header_a if str(item.get("text", "")).strip()]
    texts_b = [str(item.get("text", "")).strip().lower() for item in header_b if str(item.get("text", "")).strip()]
    if not texts_a or not texts_b:
        return 0.0
    overlap = len(set(texts_a) & set(texts_b))
    return overlap / max(len(set(texts_a)), len(set(texts_b)))


def _stitch_cross_page_tables(
    table_asts: list[dict[str, Any]],
    page_heights: dict[int, float],
) -> int:
    if not table_asts:
        return 0
    stitched_count = 0
    sorted_tables = sorted(table_asts, key=lambda item: (item["page"], item["bbox"][1]))

    for previous, current in zip(sorted_tables, sorted_tables[1:]):
        if current["page"] != previous["page"] + 1:
            continue
        prev_page_height = max(1.0, page_heights.get(previous["page"], 1.0))
        curr_page_height = max(1.0, page_heights.get(current["page"], 1.0))
        prev_near_bottom = previous["bbox"][3] >= prev_page_height * 0.72
        curr_near_top = current["bbox"][1] <= curr_page_height * 0.28
        similarity = _column_similarity(previous["column_signature"], current["column_signature"])
        is_likely_continuation = (prev_near_bottom and curr_near_top) or similarity >= 0.9
        if similarity < 0.78 or not is_likely_continuation:
            continue

        previous.setdefault("continued_to", [])
        previous["continued_to"].append(current["table_id"])
        current["continued_from"] = previous["table_id"]
        current["cross_page_similarity"] = round(similarity, 3)

        header_similarity = _header_similarity(previous.get("header", []), current.get("header", []))
        if header_similarity < 0.45:
            current["header"] = previous.get("header", [])
            current["header_inherited"] = True
        stitched_count += 1

    return stitched_count


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

    figure_index = 1
    table_index = 1

    for page_index, page in enumerate(document):
        page_number = page_index + 1
        text_blocks, page_images = _extract_page_text_and_images(page, page_number)
        page_words = _extract_words(page)
        table_row_groups = _group_table_rows(_cluster_rows(page_words))
        page_tables: list[dict[str, Any]] = []
        page_rect = page.rect
        page_heights[page_number] = float(page_rect.height)

        for row_group in table_row_groups:
            table_id = f"tbl_{table_index:03d}"
            table_ast = _build_single_table_ast(
                rows=row_group,
                page_number=page_number,
                page_height=float(page_rect.height),
                table_id=table_id,
            )
            if table_ast is None:
                continue
            table_asts.append(table_ast)
            page_tables.append(table_ast)
            table_index += 1

        page_figures, figure_index = _assign_figure_titles(page_images, text_blocks, figure_index)
        figure_nodes.extend(page_figures)
        image_blocks.extend(page_images)

        page_text = "\n".join(item["text"] for item in text_blocks)
        all_page_text.append(page_text)

        page_bbox_nodes: list[dict[str, Any]] = []
        for text_block in text_blocks:
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
                }
            )

        for image_block in page_images:
            bbox = image_block["bbox"]
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
                }
            )

        for table_ast in page_tables:
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
                }
            )

        page_bbox_nodes.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        document_ast_pages.append({"page": page_number, "blocks": page_bbox_nodes})
        pages.append(
            {
                "page_number": page_number,
                "width": float(page_rect.width),
                "height": float(page_rect.height),
                "text": page_text,
                "block_count": len(text_blocks),
                "image_count": len(page_images),
                "table_count": len(page_tables),
            }
        )

    document.close()
    stitched_table_count = _stitch_cross_page_tables(table_asts, page_heights)
    merged_text = "\n\n".join(all_page_text)

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
            "parser_hint": "pdf-ast-v2",
        },
    }
