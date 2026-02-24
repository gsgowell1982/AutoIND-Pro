from __future__ import annotations

import statistics
from typing import Any

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

