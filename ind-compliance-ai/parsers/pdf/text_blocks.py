from __future__ import annotations

import re
import statistics
from typing import Any

from .shared import (
    _bbox_intersection_ratio,
    _bbox_to_list,
    _bbox_union,
    _clean_text,
    _compact_text,
    _has_cjk,
    _horizontal_overlap_ratio,
    _text_contains_text,
)


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


def _vertical_overlap_ratio(
    a_bbox: tuple[float, float, float, float],
    b_bbox: tuple[float, float, float, float],
) -> float:
    top = max(a_bbox[1], b_bbox[1])
    bottom = min(a_bbox[3], b_bbox[3])
    overlap = max(0.0, bottom - top)
    min_height = max(1.0, min(a_bbox[3] - a_bbox[1], b_bbox[3] - b_bbox[1]))
    return overlap / min_height


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
    vertical_overlap = _vertical_overlap_ratio(left_bbox, right_bbox)
    horizontal_overlap = _horizontal_overlap_ratio(left_bbox, right_bbox)
    if abs(left_center - right_center) > row_tolerance and vertical_overlap < 0.6:
        return False

    gap = right_bbox[0] - left_bbox[2]
    if gap < -2.5 and not (vertical_overlap >= 0.6 and horizontal_overlap >= 0.18):
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
        if ratio > 1.45 and not (vertical_overlap >= 0.6 and horizontal_overlap >= 0.18):
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

