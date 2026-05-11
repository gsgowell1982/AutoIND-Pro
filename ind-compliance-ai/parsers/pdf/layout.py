from __future__ import annotations

import statistics
from typing import Any

from .settings import get_pdf_parser_settings

from .shared import (
    _Word,
    _bbox_intersection_ratio,
    _bbox_union,
    _clean_text,
)


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


def _row_bbox_from_words(words: list[_Word]) -> tuple[float, float, float, float]:
    return _bbox_union([(item.x0, item.y0, item.x1, item.y1) for item in words])


def _vertical_overlap_ratio(
    a_bbox: tuple[float, float, float, float],
    b_bbox: tuple[float, float, float, float],
) -> float:
    top = max(a_bbox[1], b_bbox[1])
    bottom = min(a_bbox[3], b_bbox[3])
    overlap = max(0.0, bottom - top)
    min_height = max(1.0, min(a_bbox[3] - a_bbox[1], b_bbox[3] - b_bbox[1]))
    return overlap / min_height


def is_two_column_layout(words: list[_Word], page_width: float) -> bool:
    """Heuristic two-column detector for literature-style pages."""
    policy = get_pdf_parser_settings().table_detection_policy
    if not policy.enable_two_column_guard:
        return False
    if not words or page_width <= 0:
        return False
    mid = page_width / 2.0
    gutter_half = page_width * max(0.0, policy.two_column_gutter_ratio_min) / 2.0
    left_count = 0
    right_count = 0
    gutter_count = 0
    for word in words:
        xc = (word.x0 + word.x1) / 2.0
        dist = abs(xc - mid)
        if dist <= gutter_half:
            gutter_count += 1
        elif xc < mid:
            left_count += 1
        else:
            right_count += 1

    side_total = left_count + right_count
    if side_total <= 0:
        return False
    balance = abs(left_count - right_count) / side_total
    gutter_ratio = gutter_count / max(1, len(words))
    if len(words) < policy.two_column_min_words:
        sparse_min_words = max(60, int(policy.two_column_min_words * 0.65))
        sparse_side_min = max(18, int(sparse_min_words * 0.22))
        return (
            len(words) >= sparse_min_words
            and left_count >= sparse_side_min
            and right_count >= sparse_side_min
            and balance <= policy.two_column_balance_tolerance
            and gutter_ratio <= policy.two_column_gutter_ratio_max * 0.6
        )
    return balance <= policy.two_column_balance_tolerance and gutter_ratio <= policy.two_column_gutter_ratio_max


def _body_top_limit(page_height: float) -> float:
    return max(72.0, page_height * 0.11)


def _body_bottom_limit(page_height: float) -> float:
    return page_height - max(72.0, page_height * 0.11)


def _profile_detects_columns(layout_profile: dict[str, Any] | None) -> bool:
    if not layout_profile:
        return False
    return str(layout_profile.get("mode", "single_column")) in {"two_column", "mixed"}


def infer_page_text_layout_profile(
    text_blocks: list[dict[str, Any]],
    page_words: list[_Word],
    page_width: float,
    page_height: float,
) -> dict[str, Any]:
    """Infer whether a page should use single-column or column-aware reading order."""
    profile = {
        "mode": "single_column",
        "confidence": 0.0,
        "page_width": float(page_width or 0.0),
        "page_height": float(page_height or 0.0),
        "column_mid": page_width / 2.0 if page_width > 0 else 0.0,
        "left_column_center": 0.0,
        "right_column_center": 0.0,
        "body_top": _body_top_limit(page_height),
        "body_bottom": _body_bottom_limit(page_height),
        "column_row_count": 0,
        "two_sided_row_count": 0,
        "full_width_row_count": 0,
    }
    if not text_blocks or not page_words or page_width <= 0 or page_height <= 0:
        return profile
    if not is_two_column_layout(page_words, page_width):
        return profile

    body_top = float(profile["body_top"])
    body_bottom = float(profile["body_bottom"])
    mid = page_width / 2.0
    column_gap_min = page_width * 0.06
    width_limit = page_width * 0.58
    center_sep = page_width * 0.08
    narrow_blocks: list[dict[str, Any]] = []
    left_candidates: list[dict[str, Any]] = []
    right_candidates: list[dict[str, Any]] = []

    for block in text_blocks:
        text = _clean_text(str(block.get("text", "")))
        bbox = tuple(float(item) for item in block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        if not text or len(bbox) != 4:
            continue
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        if width <= 0 or height <= 0 or height > page_height * 0.08:
            continue
        if width > width_limit:
            continue
        center_y = (bbox[1] + bbox[3]) / 2.0
        if bbox[1] < body_top:
            top_overflow = body_top - center_y
            allowed_top_overflow = max(18.0, height * 2.0, page_width * 0.06)
            if top_overflow > allowed_top_overflow:
                continue
        if bbox[3] > body_bottom:
            bottom_overflow = center_y - body_bottom
            allowed_bottom_overflow = max(18.0, height * 2.0, page_width * 0.08)
            if bottom_overflow > allowed_bottom_overflow:
                continue
        center_x = (bbox[0] + bbox[2]) / 2.0
        narrow_blocks.append(block)
        if center_x <= mid - center_sep:
            left_candidates.append(block)
        elif center_x >= mid + center_sep:
            right_candidates.append(block)

    if len(left_candidates) < 3 or len(right_candidates) < 3:
        return profile

    left_centers = [((float(block["bbox"][0]) + float(block["bbox"][2])) / 2.0) for block in left_candidates]
    right_centers = [((float(block["bbox"][0]) + float(block["bbox"][2])) / 2.0) for block in right_candidates]
    left_center = statistics.median(left_centers)
    right_center = statistics.median(right_centers)
    if right_center - left_center < column_gap_min:
        return profile

    profile["column_mid"] = round((left_center + right_center) / 2.0, 2)
    profile["left_column_center"] = round(left_center, 2)
    profile["right_column_center"] = round(right_center, 2)
    lane_tolerance = max(12.0, page_width * 0.04)
    profile["lane_tolerance"] = round(lane_tolerance, 2)
    profile["narrow_block_count"] = len(narrow_blocks)

    row_height_basis = [
        max(1.0, float(block["bbox"][3]) - float(block["bbox"][1]))
        for block in narrow_blocks
    ]
    row_tolerance = max(2.5, statistics.median(row_height_basis) * 0.45) if row_height_basis else 2.5
    ordered = sorted(narrow_blocks, key=lambda item: (((item["bbox"][1] + item["bbox"][3]) / 2.0), item["bbox"][0]))
    rows: list[dict[str, Any]] = []
    for block in ordered:
        bbox = tuple(float(item) for item in block["bbox"])
        center_y = (bbox[1] + bbox[3]) / 2.0
        best_row: dict[str, Any] | None = None
        best_score: tuple[float, float] | None = None
        for row in rows:
            row_bbox = tuple(row["bbox"])
            overlap = _vertical_overlap_ratio(bbox, row_bbox)
            center_diff = abs(center_y - float(row["center_y"]))
            if center_diff > row_tolerance and overlap < 0.6:
                continue
            score = (overlap, -center_diff)
            if best_score is None or score > best_score:
                best_score = score
                best_row = row
        if best_row is None:
            rows.append(
                {
                    "bbox": list(bbox),
                    "center_y": center_y,
                    "blocks": [block],
                }
            )
            continue
        best_row["blocks"].append(block)
        best_row["bbox"] = list(_bbox_union([tuple(best_row["bbox"]), bbox]))
        best_row["center_y"] = statistics.median(
            [((item["bbox"][1] + item["bbox"][3]) / 2.0) for item in best_row["blocks"]]
        )

    column_rows = 0
    two_sided_rows = 0
    full_width_rows = 0
    for row in rows:
        has_left = False
        has_right = False
        row_is_full_width = False
        for block in row["blocks"]:
            bbox = tuple(float(item) for item in block["bbox"])
            width = max(0.0, bbox[2] - bbox[0])
            center_x = (bbox[0] + bbox[2]) / 2.0
            if width >= page_width * 0.7:
                row_is_full_width = True
                break
            if center_x <= profile["column_mid"] - lane_tolerance:
                has_left = True
            elif center_x >= profile["column_mid"] + lane_tolerance:
                has_right = True
            else:
                row_is_full_width = True
                break
        if row_is_full_width:
            full_width_rows += 1
            continue
        if has_left or has_right:
            column_rows += 1
        if has_left and has_right:
            two_sided_rows += 1

    if column_rows < 3 or (two_sided_rows < 2 and min(len(left_candidates), len(right_candidates)) < 8):
        return profile

    profile["column_row_count"] = column_rows
    profile["two_sided_row_count"] = two_sided_rows
    profile["full_width_row_count"] = full_width_rows
    profile["mode"] = "mixed" if full_width_rows > 0 else "two_column"
    confidence = min(
        0.99,
        0.45
        + min(column_rows, 12) * 0.03
        + min(two_sided_rows, 8) * 0.03
        + min(len(narrow_blocks), 40) * 0.004,
    )
    profile["confidence"] = round(confidence, 3)
    return profile


def classify_text_block_layout_lane(
    block: dict[str, Any],
    layout_profile: dict[str, Any] | None,
) -> str:
    if not _profile_detects_columns(layout_profile):
        return "full_width"

    bbox = tuple(float(item) for item in block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(bbox) != 4:
        return "full_width"
    page_width = float(layout_profile.get("page_width", 0.0) or 0.0)
    if page_width <= 0:
        page_width = max(float(bbox[2]), 1.0)
    width = max(0.0, bbox[2] - bbox[0])
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    column_mid = float(layout_profile.get("column_mid", page_width / 2.0))
    lane_tolerance = float(layout_profile.get("lane_tolerance", max(12.0, page_width * 0.04)))
    body_top = float(layout_profile.get("body_top", 0.0))
    body_bottom = float(layout_profile.get("body_bottom", 0.0))

    def _classify_horizontal_lane() -> str:
        if center_x <= column_mid - lane_tolerance:
            return "left"
        if center_x >= column_mid + lane_tolerance:
            return "right"
        if bbox[2] <= column_mid + lane_tolerance:
            return "left"
        if bbox[0] >= column_mid - lane_tolerance:
            return "right"
        return "full_width"

    if width >= page_width * 0.7:
        return "full_width"
    if body_top and center_y < body_top:
        block_height = max(1.0, bbox[3] - bbox[1])
        top_overflow = body_top - center_y
        allowed_top_overflow = max(18.0, block_height * 2.0, page_width * 0.06)
        if top_overflow > allowed_top_overflow:
            return "full_width"
        return _classify_horizontal_lane()
    if body_bottom and center_y > body_bottom:
        block_height = max(1.0, bbox[3] - bbox[1])
        bottom_overflow = center_y - body_bottom
        allowed_bottom_overflow = max(18.0, block_height * 2.0, page_width * 0.08)
        if bottom_overflow > allowed_bottom_overflow:
            return "full_width"
        if abs(center_x - column_mid) <= max(8.0, lane_tolerance * 0.8):
            return "full_width"
        return _classify_horizontal_lane()
    return _classify_horizontal_lane()


def annotate_text_blocks_with_layout(
    text_blocks: list[dict[str, Any]],
    layout_profile: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not text_blocks:
        return text_blocks
    mode = str((layout_profile or {}).get("mode", "single_column"))
    confidence = float((layout_profile or {}).get("confidence", 0.0) or 0.0)
    for block in text_blocks:
        lane = classify_text_block_layout_lane(block, layout_profile)
        block["layout_mode"] = mode
        block["layout_lane"] = lane
        block["layout_confidence"] = confidence
    return text_blocks

