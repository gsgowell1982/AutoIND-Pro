from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
import statistics
from typing import Any

try:
    import pymupdf
except ImportError:  # pragma: no cover - optional runtime dependency
    pymupdf = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional runtime dependency
    Image = None  # type: ignore[assignment]

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional runtime dependency
    pytesseract = None  # type: ignore[assignment]

from parsers.common.atomic_fact_extractor import extract_atomic_facts

TABLE_TITLE_PATTERN = re.compile(
    r"^\s*(?:附?表\s*[0-9一二三四五六七八九十]+|table\s*\d+)\s*(?:[.:：、\-]|\s+)",
    re.IGNORECASE,
)
TABLE_SECTION_HINT_PATTERN = re.compile(r"(术语表|词汇表|名词表|名词解释|参数表|清单|附录表|glossary)", re.IGNORECASE)
TOC_LINE_PATTERN = re.compile(r"[\.·…]{4,}\s*\d{1,3}\s*$")
TOC_HEADING_PATTERN = re.compile(r"^\s*(目录|contents)\s*$", re.IGNORECASE)


def _clean_text(value: str) -> str:
    return " ".join(str(value).replace("\x00", " ").split())


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _compact_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", _clean_text(text).lower())


def _text_contains_text(source: str, target: str) -> bool:
    source_compact = _compact_text(source)
    target_compact = _compact_text(target)
    if not source_compact or not target_compact:
        return False
    return target_compact in source_compact


def _bbox_to_list(bbox: tuple[float, float, float, float]) -> list[float]:
    return [round(float(item), 2) for item in bbox]


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_union(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    x0 = min(item[0] for item in bboxes)
    y0 = min(item[1] for item in bboxes)
    x1 = max(item[2] for item in bboxes)
    y1 = max(item[3] for item in bboxes)
    return (x0, y0, x1, y1)


def _bbox_intersection_ratio(
    a_bbox: tuple[float, float, float, float],
    b_bbox: tuple[float, float, float, float],
) -> float:
    x0 = max(a_bbox[0], b_bbox[0])
    y0 = max(a_bbox[1], b_bbox[1])
    x1 = min(a_bbox[2], b_bbox[2])
    y1 = min(a_bbox[3], b_bbox[3])
    intersection = _bbox_area((x0, y0, x1, y1))
    min_area = max(1.0, min(_bbox_area(a_bbox), _bbox_area(b_bbox)))
    return intersection / min_area


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
        bbox = tuple(float(item) for item in block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        if block_type == 0:
            line_texts: list[str] = []
            font_sizes: list[float] = []
            for line in block.get("lines", []):
                span_text = ""
                for span in line.get("spans", []):
                    span_text += str(span.get("text", ""))
                    try:
                        font_sizes.append(float(span.get("size", 0.0)))
                    except (TypeError, ValueError):
                        continue
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
                    "font_size": statistics.median(font_sizes) if font_sizes else 0.0,
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


def _text_starts_with_break_marker(text: str) -> bool:
    return bool(re.match(r"^[•\-·●◆\(\)（）\[\]]", text.strip()))


def _text_ends_with_break_marker(text: str) -> bool:
    return bool(re.search(r"[。！？；:：]\s*$", text.strip()))


def _join_text_fragments(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if _has_cjk(left[-1]) and _has_cjk(right[0]):
        return left + right
    if left.endswith("-") or right.startswith((")", "）", "]", "】", "%", "％")):
        return left + right
    if left.endswith(("(", "（", "/", "／")):
        return left + right
    return f"{left} {right}"


def _should_merge_text_blocks(
    left: dict[str, Any],
    right: dict[str, Any],
    page_width: float,
    row_tolerance: float,
) -> bool:
    left_bbox = tuple(left["bbox"])
    right_bbox = tuple(right["bbox"])
    left_height = max(1.0, left_bbox[3] - left_bbox[1])
    right_height = max(1.0, right_bbox[3] - right_bbox[1])
    left_center = (left_bbox[1] + left_bbox[3]) / 2
    right_center = (right_bbox[1] + right_bbox[3]) / 2
    if abs(left_center - right_center) > row_tolerance:
        return False

    gap = right_bbox[0] - left_bbox[2]
    if gap < -2.5:
        return False
    max_gap = max(24.0, page_width * 0.045, min(left_height, right_height) * 2.4)
    if gap > max_gap:
        return False

    left_text = _clean_text(left.get("text", ""))
    right_text = _clean_text(right.get("text", ""))
    if not left_text or not right_text:
        return False
    if _text_ends_with_break_marker(left_text):
        return False
    if _text_starts_with_break_marker(right_text):
        return False

    left_font = float(left.get("font_size", 0.0) or 0.0)
    right_font = float(right.get("font_size", 0.0) or 0.0)
    if left_font > 0 and right_font > 0:
        ratio = max(left_font, right_font) / max(0.1, min(left_font, right_font))
        if ratio > 1.45:
            return False

    if _has_cjk(left_text) or _has_cjk(right_text):
        return True
    return len(left_text) <= 64 and len(right_text) <= 64


def _merge_semantic_text_blocks(
    text_blocks: list[dict[str, Any]],
    page_number: int,
    page_width: float,
) -> tuple[list[dict[str, Any]], int]:
    if not text_blocks:
        return [], 0
    ordered = sorted(text_blocks, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    heights = [max(1.0, block["bbox"][3] - block["bbox"][1]) for block in ordered]
    row_tolerance = max(2.5, statistics.median(heights) * 0.45)

    merged: list[dict[str, Any]] = []
    merge_count = 0
    for block in ordered:
        normalized = {
            "block_type": "text",
            "page": page_number,
            "bbox": [float(item) for item in block["bbox"]],
            "text": _clean_text(block.get("text", "")),
            "font_size": float(block.get("font_size", 0.0) or 0.0),
            "source_block_indices": [block.get("source_block_index")],
            "source": block.get("source", "text-layer"),
        }
        if not normalized["text"]:
            continue
        if not merged:
            merged.append(normalized)
            continue
        previous = merged[-1]
        if _should_merge_text_blocks(previous, normalized, page_width, row_tolerance):
            previous_bbox = tuple(previous["bbox"])
            normalized_bbox = tuple(normalized["bbox"])
            previous["bbox"] = _bbox_to_list(_bbox_union([previous_bbox, normalized_bbox]))
            previous["text"] = _join_text_fragments(previous["text"], normalized["text"])
            if normalized["source"] == "image-text-recovery":
                previous["source"] = "semantic-merged-image-text"
            previous["source_block_indices"].extend(normalized["source_block_indices"])
            font_sizes = [float(previous.get("font_size", 0.0) or 0.0), float(normalized.get("font_size", 0.0) or 0.0)]
            font_sizes = [item for item in font_sizes if item > 0]
            previous["font_size"] = statistics.median(font_sizes) if font_sizes else 0.0
            merge_count += 1
            continue
        merged.append(normalized)

    deduped, dedup_count = _deduplicate_text_blocks(merged)
    for index, block in enumerate(deduped, start=1):
        block["block_id"] = f"txt_p{page_number}_{index:03d}"
    return deduped, merge_count + dedup_count


def _is_duplicate_text_block(candidate: dict[str, Any], existing: dict[str, Any]) -> bool:
    candidate_bbox = tuple(candidate.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    existing_bbox = tuple(existing.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    overlap = _bbox_intersection_ratio(candidate_bbox, existing_bbox)
    if overlap < 0.72:
        return False
    candidate_text = _clean_text(candidate.get("text", ""))
    existing_text = _clean_text(existing.get("text", ""))
    if not candidate_text or not existing_text:
        return False
    return _text_contains_text(candidate_text, existing_text) or _text_contains_text(existing_text, candidate_text)


def _deduplicate_text_blocks(text_blocks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if not text_blocks:
        return [], 0
    deduped: list[dict[str, Any]] = []
    duplicate_count = 0
    for block in sorted(text_blocks, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        duplicate_index = next(
            (
                index
                for index, kept in enumerate(deduped)
                if _is_duplicate_text_block(block, kept)
            ),
            -1,
        )
        if duplicate_index < 0:
            deduped.append(block)
            continue
        duplicate_count += 1
        kept_text = _clean_text(deduped[duplicate_index].get("text", ""))
        candidate_text = _clean_text(block.get("text", ""))
        # Prefer longer candidate text if two blocks are duplicates.
        if len(_compact_text(candidate_text)) > len(_compact_text(kept_text)):
            deduped[duplicate_index] = block
    return deduped, duplicate_count


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


def _words_in_bbox(
    page_words: list[_Word],
    bbox: tuple[float, float, float, float],
    margin: float = 1.6,
) -> list[_Word]:
    expanded = (bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin)
    hits: list[_Word] = []
    for word in page_words:
        overlap = _bbox_intersection_ratio((word.x0, word.y0, word.x1, word.y1), expanded)
        if overlap >= 0.25:
            hits.append(word)
    return hits


def _words_to_text(words: list[_Word]) -> str:
    if not words:
        return ""
    ordered = sorted(words, key=lambda item: (item.yc, item.x0))
    median_height = statistics.median([word.height for word in words]) if words else 8.0
    row_tol = max(2.2, median_height * 0.48)
    lines: list[list[_Word]] = []
    for word in ordered:
        if not lines:
            lines.append([word])
            continue
        previous = lines[-1][-1]
        if abs(word.yc - previous.yc) <= row_tol:
            lines[-1].append(word)
        else:
            lines.append([word])
    joined_lines: list[str] = []
    for line_words in lines:
        line_text = _clean_text(" ".join(item.text for item in sorted(line_words, key=lambda item: item.x0)))
        if line_text:
            joined_lines.append(line_text)
    return "\n".join(joined_lines).strip()


def _drawing_rect_to_tuple(drawing: dict[str, Any]) -> tuple[float, float, float, float] | None:
    rect = drawing.get("rect")
    if rect is None:
        return None
    if hasattr(rect, "x0"):
        return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
    if isinstance(rect, (list, tuple)) and len(rect) >= 4:
        return (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
    return None


def _bbox_contains_path(
    drawings: list[dict[str, Any]],
    bbox: tuple[float, float, float, float],
) -> bool:
    for drawing in drawings:
        rect = _drawing_rect_to_tuple(drawing)
        if rect is None:
            continue
        if _bbox_intersection_ratio(rect, bbox) >= 0.2:
            return True
    return False


def _ocr_text_from_clip(
    page: "pymupdf.Page",
    bbox: tuple[float, float, float, float],
) -> tuple[str, float]:
    if pytesseract is None or Image is None:
        return "", 0.0
    rect = pymupdf.Rect(*bbox)
    if rect.width < 8 or rect.height < 8:
        return "", 0.0
    try:
        pixmap = page.get_pixmap(clip=rect, dpi=220, alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        try:
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                lang="chi_sim+eng",
            )
        except Exception:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception:
        return "", 0.0

    words: list[str] = []
    confidences: list[float] = []
    total = len(data.get("text", []))
    for index in range(total):
        text = _clean_text(str(data["text"][index]))
        if not text:
            continue
        conf_value = str(data.get("conf", ["-1"] * total)[index]).strip()
        try:
            confidence = float(conf_value)
        except ValueError:
            confidence = -1.0
        if confidence >= 0:
            confidences.append(confidence / 100.0)
        words.append(text)
    if not words:
        return "", 0.0
    joined = _clean_text(" ".join(words))
    average_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return joined, average_conf


def _recover_text_from_image_region(
    page: "pymupdf.Page",
    image_bbox: tuple[float, float, float, float],
    page_words: list[_Word],
    page_drawings: list[dict[str, Any]],
) -> dict[str, Any]:
    text_layer_candidate = _clean_text(page.get_textbox(pymupdf.Rect(*image_bbox)))
    has_path = _bbox_contains_path(page_drawings, image_bbox)
    if len(text_layer_candidate) >= 2:
        return {
            "text": text_layer_candidate,
            "confidence": 0.98,
            "source": "text-layer",
            "has_path": has_path,
        }

    in_bbox_words = _words_in_bbox(page_words, image_bbox)
    word_text = _words_to_text(in_bbox_words)
    if len(word_text) >= 2:
        return {
            "text": word_text,
            "confidence": 0.9,
            "source": "word-cluster",
            "has_path": has_path,
        }

    ocr_text, ocr_confidence = _ocr_text_from_clip(page, image_bbox)
    if len(ocr_text) >= 2:
        return {
            "text": ocr_text,
            "confidence": max(ocr_confidence, 0.7 if has_path else 0.6),
            "source": "ocr",
            "has_path": has_path,
        }
    return {"text": "", "confidence": 0.0, "source": "none", "has_path": has_path}


def _demote_textual_image_blocks(
    page: "pymupdf.Page",
    page_number: int,
    page_rect: "pymupdf.Rect",
    image_blocks: list[dict[str, Any]],
    text_blocks: list[dict[str, Any]],
    page_words: list[_Word],
    page_drawings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    page_area = max(1.0, float(page_rect.width) * float(page_rect.height))
    merged_text_blocks = list(text_blocks)
    kept_images: list[dict[str, Any]] = []
    converted_count = 0

    for image_block in image_blocks:
        bbox = tuple(float(item) for item in image_block["bbox"])
        recovered = _recover_text_from_image_region(page, bbox, page_words, page_drawings)
        recovered_text = _clean_text(recovered["text"])
        confidence = float(recovered["confidence"])
        has_path = bool(recovered.get("has_path", False))
        recovered_source = str(recovered.get("source", "none"))
        image_area = _bbox_area(bbox)
        area_ratio = image_area / page_area

        image_block["text_recovery"] = {
            "text": recovered_text,
            "confidence": round(confidence, 3),
            "source": recovered_source,
            "has_path": has_path,
        }

        has_overlapping_text = any(
            _bbox_intersection_ratio(tuple(block["bbox"]), bbox) >= 0.55
            and (
                _text_contains_text(_clean_text(block.get("text", "")), recovered_text)
                or _text_contains_text(recovered_text, _clean_text(block.get("text", "")))
            )
            for block in merged_text_blocks
        )

        if len(recovered_text) >= 2 and has_overlapping_text:
            # If this image region is already represented by text, drop duplicate image block.
            converted_count += 1
            continue

        if recovered_source in {"text-layer", "word-cluster"}:
            confidence_gate = 0.86
        elif recovered_source == "ocr":
            confidence_gate = 0.9 if has_path else 0.93
        else:
            confidence_gate = 0.95

        short_text_label = len(_compact_text(recovered_text)) <= 20 and len(_compact_text(recovered_text)) >= 2
        should_convert = (
            len(recovered_text) >= 2
            and (
                (confidence >= confidence_gate and area_ratio <= 0.24)
                or (has_path and confidence >= 0.78 and area_ratio <= 0.28)
                or (short_text_label and confidence >= 0.84 and area_ratio <= 0.08)
            )
        )
        if should_convert:
            merged_text_blocks.append(
                {
                    "block_type": "text",
                    "block_id": f"txt_img_recover_p{page_number}_{converted_count + 1:03d}",
                    "page": page_number,
                    "bbox": _bbox_to_list(bbox),
                    "text": recovered_text,
                    "font_size": 0.0,
                    "source": "image-text-recovery",
                    "source_image_id": image_block["image_id"],
                }
            )
            converted_count += 1
            continue
        kept_images.append(image_block)

    return merged_text_blocks, kept_images, converted_count


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
    row_texts: list[str] = []
    for row in rows:
        cell_map: dict[int, list[_Word]] = {}
        for word in row["words"]:
            col_index = _closest_column_index(word.xc, column_centers)
            cell_map.setdefault(col_index, []).append(word)
        row_cell_maps.append(cell_map)
        row_texts.append(_clean_text(" ".join(word.text for word in sorted(row["words"], key=lambda item: item.x0))))

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
    row_non_empty_counts: list[int] = []
    for raw_row_index, cell_map in enumerate(row_cell_maps[header_row_index + 1 :], start=1):
        non_empty_this_row = 0
        for col_index in range(len(column_centers)):
            words = sorted(cell_map.get(col_index, []), key=lambda item: item.x0)
            if not words:
                continue
            cell_text = _clean_text(" ".join(word.text for word in words))
            if not cell_text:
                continue
            non_empty_this_row += 1
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
        row_non_empty_counts.append(non_empty_this_row)

    if not cells:
        return None

    data_row_count = max(1, len(row_cell_maps) - (header_row_index + 1))
    coverage_ratio = len(cells) / max(1.0, data_row_count * len(column_centers))
    aligned_ratio = sum(1 for count in row_non_empty_counts if count >= 2) / max(1, len(row_non_empty_counts))
    numeric_ratio = sum(1 for cell in cells if re.search(r"\d", str(cell.get("text", "")))) / max(1, len(cells))
    structure_score = min(1.0, 0.45 * aligned_ratio + 0.35 * min(1.0, coverage_ratio) + 0.2 * numeric_ratio)

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
        "structure_score": round(structure_score, 3),
        "row_texts": row_texts,
        "near_page_bottom": table_bbox[3] >= page_height * 0.72,
        "near_page_top": table_bbox[1] <= page_height * 0.28,
    }


def _find_table_title_block(
    table_bbox: tuple[float, float, float, float],
    text_blocks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for block in text_blocks:
        text = _clean_text(block.get("text", ""))
        if not text or not TABLE_TITLE_PATTERN.search(text):
            continue
        bbox = tuple(block["bbox"])
        vertical_gap = table_bbox[1] - bbox[3]
        if vertical_gap < -8 or vertical_gap > 160:
            continue
        overlap = _horizontal_overlap_ratio(table_bbox, bbox)
        if overlap < 0.12:
            continue
        distance = vertical_gap if vertical_gap >= 0 else abs(vertical_gap) + 25
        candidates.append((distance, block))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _find_section_table_hint(
    table_bbox: tuple[float, float, float, float],
    text_blocks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for block in text_blocks:
        text = _clean_text(block.get("text", ""))
        if not text or not TABLE_SECTION_HINT_PATTERN.search(text):
            continue
        bbox = tuple(block["bbox"])
        vertical_gap = table_bbox[1] - bbox[3]
        if vertical_gap < -8 or vertical_gap > 180:
            continue
        overlap = _horizontal_overlap_ratio(table_bbox, bbox)
        if overlap < 0.1:
            continue
        candidates.append((max(0.0, vertical_gap), block))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _is_toc_context(
    table_bbox: tuple[float, float, float, float],
    text_blocks: list[dict[str, Any]],
) -> bool:
    table_top = table_bbox[1]
    for block in text_blocks:
        text = _clean_text(block.get("text", ""))
        if not text:
            continue
        bbox = tuple(block["bbox"])
        if bbox[3] > table_top + 15:
            continue
        vertical_gap = table_top - bbox[3]
        if vertical_gap < 0 or vertical_gap > 220:
            continue
        if TOC_HEADING_PATTERN.search(text):
            return True
    return False


def _table_toc_row_ratio(table_ast: dict[str, Any]) -> float:
    row_texts = [str(item) for item in table_ast.get("row_texts", [])]
    if not row_texts:
        return 0.0
    toc_rows = 0.0
    for row_text in row_texts:
        cleaned = _clean_text(row_text)
        if not cleaned:
            continue
        if TOC_LINE_PATTERN.search(cleaned):
            toc_rows += 1.0
            continue
        tokens = cleaned.split()
        if len(tokens) >= 3 and re.fullmatch(r"\d+(?:\.\d+){1,4}", tokens[0]) and re.fullmatch(r"\d{1,3}", tokens[-1]):
            middle = " ".join(tokens[1:-1])
            has_textual_middle = bool(re.search(r"[A-Za-z\u4e00-\u9fff]", middle))
            has_dot_leader = bool(re.search(r"[\.·…]{2,}", middle))
            if has_textual_middle and (has_dot_leader or len(middle) >= 3):
                toc_rows += 0.75
                continue
        if re.search(r"[\.·…]{2,}", cleaned) and re.search(r"\d{1,3}\s*$", cleaned):
            toc_rows += 0.65
    return min(1.0, toc_rows / max(1, len(row_texts)))


def _table_grid_line_score(
    table_bbox: tuple[float, float, float, float],
    page_drawings: list[dict[str, Any]],
) -> float:
    overlap_count = 0
    for drawing in page_drawings:
        rect = _drawing_rect_to_tuple(drawing)
        if rect is None:
            continue
        if _bbox_intersection_ratio(rect, table_bbox) >= 0.18:
            overlap_count += 1
    return min(1.0, overlap_count / 8.0)


def _find_continuation_hint(
    candidate_table: dict[str, Any],
    accepted_tables: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not bool(candidate_table.get("near_page_top")):
        return None
    best_parent: dict[str, Any] | None = None
    best_similarity = 0.0
    current_page = int(candidate_table.get("page", 0))
    for previous in reversed(accepted_tables[-12:]):
        if int(previous.get("page", 0)) != current_page - 1:
            continue
        if not bool(previous.get("near_page_bottom")):
            continue
        similarity = _column_similarity(
            list(previous.get("column_signature", [])),
            list(candidate_table.get("column_signature", [])),
        )
        if similarity > best_similarity:
            best_similarity = similarity
            best_parent = previous
    if best_parent is None or best_similarity < 0.62:
        return None
    return {
        "table_id": str(best_parent.get("table_id", "")),
        "similarity": round(best_similarity, 3),
    }


def _is_valid_table_candidate(
    table_ast: dict[str, Any],
    title_block: dict[str, Any] | None,
    section_hint_block: dict[str, Any] | None,
    toc_context: bool,
    grid_line_score: float,
    continuation_hint: dict[str, Any] | None,
) -> bool:
    score = float(table_ast.get("structure_score", 0.0))
    row_count = int(table_ast.get("row_count", 0))
    col_count = int(table_ast.get("col_count", 0))
    toc_row_ratio = float(table_ast.get("toc_row_ratio", 0.0))
    if toc_context or toc_row_ratio >= 0.38:
        return False

    if title_block is not None:
        return score >= 0.32 and row_count >= 2 and col_count >= 2

    if continuation_hint is not None:
        return row_count >= 2 and col_count >= 2 and score >= 0.42

    if section_hint_block is not None:
        return row_count >= 4 and col_count >= 2 and score >= 0.55

    if grid_line_score >= 0.45:
        return row_count >= 4 and col_count >= 2 and score >= 0.56

    # Without caption/hint/grid, keep only very strong grids.
    strong_multi_col = row_count >= 5 and col_count >= 3 and score >= 0.9 and toc_row_ratio <= 0.15
    strong_two_col = row_count >= 8 and col_count >= 2 and score >= 0.93 and toc_row_ratio <= 0.1
    return strong_multi_col or strong_two_col


def _merge_two_table_fragments(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    primary_bbox = tuple(primary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    secondary_bbox = tuple(secondary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    primary["bbox"] = _bbox_to_list(_bbox_union([primary_bbox, secondary_bbox]))

    current_max_row = max((int(cell.get("row", 0)) for cell in primary.get("cells", [])), default=0)
    secondary_cells: list[dict[str, Any]] = []
    for cell in secondary.get("cells", []):
        cell_copy = dict(cell)
        cell_copy["row"] = int(cell_copy.get("row", 0)) + current_max_row
        secondary_cells.append(cell_copy)
    primary.setdefault("cells", [])
    primary["cells"].extend(secondary_cells)

    primary["row_count"] = int(primary.get("row_count", 0)) + int(secondary.get("row_count", 0))
    primary["col_count"] = max(int(primary.get("col_count", 0)), int(secondary.get("col_count", 0)))
    primary["structure_score"] = round(
        max(float(primary.get("structure_score", 0.0)), float(secondary.get("structure_score", 0.0))),
        3,
    )
    primary["grid_line_score"] = round(
        max(float(primary.get("grid_line_score", 0.0)), float(secondary.get("grid_line_score", 0.0))),
        3,
    )
    primary["near_page_bottom"] = bool(primary.get("near_page_bottom")) or bool(secondary.get("near_page_bottom"))
    primary["near_page_top"] = bool(primary.get("near_page_top")) and bool(secondary.get("near_page_top"))
    if not primary.get("title") and secondary.get("title"):
        primary["title"] = secondary.get("title")
        primary["title_block_id"] = secondary.get("title_block_id")
    if not primary.get("section_hint") and secondary.get("section_hint"):
        primary["section_hint"] = secondary.get("section_hint")
        primary["section_hint_block_id"] = secondary.get("section_hint_block_id")
    primary.setdefault("merged_from", [])
    primary["merged_from"].append(str(secondary.get("table_id", "")))
    primary.setdefault("row_texts", [])
    primary["row_texts"].extend([str(item) for item in secondary.get("row_texts", [])])
    primary["toc_row_ratio"] = round(_table_toc_row_ratio(primary), 3)
    return primary


def _can_merge_table_fragments_on_same_page(primary: dict[str, Any], secondary: dict[str, Any]) -> bool:
    if int(primary.get("page", 0)) != int(secondary.get("page", 0)):
        return False
    primary_bbox = tuple(primary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    secondary_bbox = tuple(secondary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    vertical_gap = secondary_bbox[1] - primary_bbox[3]
    if vertical_gap < -10 or vertical_gap > 52:
        return False
    overlap = _horizontal_overlap_ratio(primary_bbox, secondary_bbox)
    if overlap < 0.5:
        return False
    similarity = _column_similarity(
        list(primary.get("column_signature", [])),
        list(secondary.get("column_signature", [])),
    )
    if similarity < 0.68:
        return False
    if float(primary.get("toc_row_ratio", 0.0)) > 0.25 or float(secondary.get("toc_row_ratio", 0.0)) > 0.25:
        return False
    return True


def _merge_same_page_table_fragments(page_tables: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if not page_tables:
        return [], 0
    sorted_tables = sorted(page_tables, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    merged_tables: list[dict[str, Any]] = [dict(sorted_tables[0])]
    merge_count = 0
    for table in sorted_tables[1:]:
        candidate = dict(table)
        previous = merged_tables[-1]
        if _can_merge_table_fragments_on_same_page(previous, candidate):
            merged_tables[-1] = _merge_two_table_fragments(previous, candidate)
            merge_count += 1
            continue
        merged_tables.append(candidate)
    return merged_tables, merge_count


def _is_header_footer_candidate(
    block: dict[str, Any],
    page_height: float,
) -> bool:
    bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    top_limit = max(72.0, page_height * 0.11)
    bottom_limit = page_height - max(72.0, page_height * 0.11)
    return bbox[3] <= top_limit or bbox[1] >= bottom_limit


def _header_footer_signature(text: str) -> str:
    normalized = _compact_text(text)
    normalized = re.sub(r"\d+", "#", normalized)
    normalized = normalized.replace("#", "")
    return normalized


def _looks_like_page_number(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    return bool(
        re.fullmatch(r"(?:第\s*)?\d{1,4}(?:\s*/\s*\d{1,4})?(?:\s*页)?", cleaned)
        or re.fullmatch(r"\d{1,4}\s*[-–—]\s*\d{1,4}", cleaned)
    )


def _filter_header_footer_text_blocks(
    page_payloads: list[dict[str, Any]],
) -> int:
    signature_counts: dict[str, int] = {}
    for payload in page_payloads:
        page_height = float(payload["height"])
        for block in payload["text_blocks"]:
            if not _is_header_footer_candidate(block, page_height):
                continue
            signature = _header_footer_signature(str(block.get("text", "")))
            if len(signature) < 3:
                continue
            signature_counts[signature] = signature_counts.get(signature, 0) + 1

    removed_total = 0
    for payload in page_payloads:
        page_height = float(payload["height"])
        kept: list[dict[str, Any]] = []
        removed = 0
        for block in payload["text_blocks"]:
            text = _clean_text(block.get("text", ""))
            if _is_header_footer_candidate(block, page_height):
                signature = _header_footer_signature(text)
                repeated_signature = len(signature) >= 3 and signature_counts.get(signature, 0) >= 2
                if repeated_signature or _looks_like_page_number(text):
                    removed += 1
                    continue
            kept.append(block)
        payload["text_blocks"] = kept
        payload["header_footer_filtered"] = removed
        removed_total += removed
    return removed_total


def _suppress_table_text_blocks(
    text_blocks: list[dict[str, Any]],
    page_tables: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    if not text_blocks or not page_tables:
        return text_blocks, 0
    table_bboxes = [tuple(item.get("bbox", (0.0, 0.0, 0.0, 0.0))) for item in page_tables]
    kept: list[dict[str, Any]] = []
    removed = 0
    for block in text_blocks:
        text_bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        text_center = ((text_bbox[0] + text_bbox[2]) / 2, (text_bbox[1] + text_bbox[3]) / 2)
        inside_table = False
        for table_bbox in table_bboxes:
            overlap = _bbox_intersection_ratio(text_bbox, table_bbox)
            center_inside = (
                table_bbox[0] <= text_center[0] <= table_bbox[2]
                and table_bbox[1] <= text_center[1] <= table_bbox[3]
            )
            if overlap >= 0.42 or (center_inside and overlap >= 0.2):
                inside_table = True
                break
        if inside_table:
            removed += 1
            continue
        kept.append(block)
    return kept, removed


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
        is_likely_continuation = (prev_near_bottom and curr_near_top) or similarity >= 0.82
        if similarity < 0.62 or not is_likely_continuation:
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
    page_payloads: list[dict[str, Any]] = []

    total_semantic_merges = 0
    total_recovered_image_text = 0
    rejected_table_candidates = 0
    total_table_fragment_merges = 0
    total_header_footer_filtered = 0
    total_table_text_suppressed = 0
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
            continuation_hint = _find_continuation_hint(candidate_table, table_asts)
            toc_row_ratio = _table_toc_row_ratio(candidate_table)

            candidate_table["grid_line_score"] = round(grid_line_score, 3)
            candidate_table["toc_row_ratio"] = round(toc_row_ratio, 3)
            if continuation_hint is not None:
                candidate_table["continuation_hint"] = continuation_hint
            if title_block is not None:
                candidate_table["title"] = title_block["text"]
                candidate_table["title_block_id"] = title_block["block_id"]
            if section_hint_block is not None:
                candidate_table["section_hint"] = section_hint_block["text"]
                candidate_table["section_hint_block_id"] = section_hint_block["block_id"]
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
        total_table_fragment_merges += page_fragment_merge_count
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
                "table_fragment_merge_count": page_fragment_merge_count,
            }
        )

    document.close()
    total_header_footer_filtered = _filter_header_footer_text_blocks(page_payloads)
    stitched_table_count = _stitch_cross_page_tables(table_asts, page_heights)

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
            "header_footer_filtered_count": total_header_footer_filtered,
            "table_text_suppressed_count": total_table_text_suppressed,
            "parser_hint": "pdf-ast-v4",
        },
    }
