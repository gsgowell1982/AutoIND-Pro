from __future__ import annotations

import re
from typing import Any

from .shared import _clean_text


_FOOTNOTE_START_RE = re.compile(r"^(?P<marker>\d{1,3}|[*\u2020\u2021])(?:\s+|(?=[A-Za-z\u4e00-\u9fff]))(?P<body>\S.*)$")
_FOOTNOTE_MARKER_RE = re.compile(r"^(?:\d{1,3}|[*\u2020\u2021])$")


def _page_body_font_size(text_blocks: list[dict[str, Any]], page_height: float) -> float:
    sizes: list[float] = []
    for block in text_blocks:
        if str(block.get("block_type", "text") or "text") != "text":
            continue
        bbox = list(block.get("bbox", []))
        if len(bbox) >= 4 and float(bbox[1]) >= float(page_height) * 0.80:
            continue
        try:
            font_size = float(block.get("font_size", 0.0) or 0.0)
        except (TypeError, ValueError):
            font_size = 0.0
        if font_size > 0:
            sizes.append(font_size)
    if not sizes:
        return 0.0
    sizes.sort()
    return sizes[len(sizes) // 2]


def _text_block_font_size(text_block: dict[str, Any]) -> float:
    try:
        return float(text_block.get("font_size", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _leading_marker_span(text_block: dict[str, Any], marker: str) -> dict[str, Any] | None:
    for span in text_block.get("spans", []) or []:
        text = _clean_text(str(span.get("text", "")))
        if text == marker:
            return span
        if text.startswith(marker) and len(text) <= len(marker) + 1:
            return span
        break
    return None


def _looks_like_footnote_start_block(
    text_block: dict[str, Any],
    *,
    page_height: float,
    body_font_size: float,
) -> tuple[bool, str, str]:
    if str(text_block.get("block_type", "text") or "text") != "text":
        return False, "", ""
    text = _clean_text(str(text_block.get("text", "")))
    match = _FOOTNOTE_START_RE.match(text)
    if not match:
        return False, "", ""
    marker = str(match.group("marker") or "").strip()
    body_text = _clean_text(str(match.group("body") or ""))
    if not marker or not body_text:
        return False, "", ""
    bbox = list(text_block.get("bbox", []))
    if len(bbox) < 4:
        return False, "", ""
    y0 = float(bbox[1])
    if y0 < max(float(page_height) * 0.78, float(page_height) - 150.0):
        return False, "", ""
    font_size = _text_block_font_size(text_block)
    if body_font_size > 0 and font_size > body_font_size * 0.86:
        return False, "", ""
    marker_span = _leading_marker_span(text_block, marker)
    if marker_span is not None:
        marker_size = float(marker_span.get("font_size", 0.0) or 0.0)
        if marker_size > 0 and font_size > 0 and marker_size <= font_size * 0.78:
            return True, marker, body_text
    if body_font_size > 0 and font_size <= body_font_size * 0.72:
        return True, marker, body_text
    return False, "", ""


def _looks_like_footnote_continuation_block(
    text_block: dict[str, Any],
    previous_block: dict[str, Any],
    *,
    page_height: float,
    body_font_size: float,
) -> bool:
    if str(text_block.get("block_type", "text") or "text") != "text":
        return False
    text = _clean_text(str(text_block.get("text", "")))
    if not text or _FOOTNOTE_START_RE.match(text):
        return False
    bbox = list(text_block.get("bbox", []))
    previous_bbox = list(previous_block.get("bbox", []))
    if len(bbox) < 4 or len(previous_bbox) < 4:
        return False
    if float(bbox[1]) < max(float(page_height) * 0.78, float(page_height) - 150.0):
        return False
    vertical_gap = float(bbox[1]) - float(previous_bbox[3])
    if vertical_gap < -1.5 or vertical_gap > 18.0:
        return False
    font_size = _text_block_font_size(text_block)
    previous_font_size = _text_block_font_size(previous_block)
    if body_font_size > 0 and font_size > body_font_size * 0.86:
        return False
    if previous_font_size > 0 and font_size > 0:
        ratio = max(font_size, previous_font_size) / max(0.1, min(font_size, previous_font_size))
        if ratio > 1.25:
            return False
    x0 = float(bbox[0])
    previous_x0 = float(previous_bbox[0])
    previous_x1 = float(previous_bbox[2])
    return previous_x0 - 6.0 <= x0 <= previous_x1


def _looks_like_inline_footnote_ref_span(
    text_block: dict[str, Any],
    span: dict[str, Any],
    marker: str,
    *,
    block_font_size: float,
) -> bool:
    if not marker or not _FOOTNOTE_MARKER_RE.fullmatch(marker):
        return False
    text = _clean_text(str(span.get("text", "")))
    if text != marker:
        return False
    try:
        span_font_size = float(span.get("font_size", 0.0) or 0.0)
    except (TypeError, ValueError):
        span_font_size = 0.0
    if block_font_size <= 0 or span_font_size <= 0 or span_font_size > block_font_size * 0.82:
        return False
    spans = list(text_block.get("spans", []) or [])
    marker_index = next((idx for idx, candidate in enumerate(spans) if candidate is span), -1)
    if marker_index <= 0:
        return False
    bbox = list(span.get("bbox", []) or [])
    block_bbox = list(text_block.get("bbox", []) or [])
    if len(bbox) >= 4 and len(block_bbox) >= 4:
        span_center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
        block_center_y = (float(block_bbox[1]) + float(block_bbox[3])) / 2.0
        if span_center_y > block_center_y + 2.5:
            return False
    return True


def _drawing_bbox(drawing: dict[str, Any]) -> list[float]:
    rect = drawing.get("rect")
    if rect is not None:
        try:
            return [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
        except AttributeError:
            pass
        try:
            values = list(rect)
            if len(values) >= 4:
                return [float(values[0]), float(values[1]), float(values[2]), float(values[3])]
        except (TypeError, ValueError):
            pass
    bbox = drawing.get("bbox")
    if bbox is not None:
        try:
            values = list(bbox)
            if len(values) >= 4:
                return [float(values[0]), float(values[1]), float(values[2]), float(values[3])]
        except (TypeError, ValueError):
            pass
    return []


def _find_footnote_separator_line(
    page_drawings: list[dict[str, Any]],
    *,
    page_width: float,
    page_height: float,
    footnote_bbox: list[float],
) -> dict[str, Any] | None:
    if not page_drawings or page_width <= 0.0 or page_height <= 0.0 or len(footnote_bbox) < 4:
        return None
    footnote_top = float(footnote_bbox[1])
    candidates: list[tuple[float, dict[str, Any]]] = []
    for drawing in page_drawings:
        bbox = _drawing_bbox(drawing)
        if len(bbox) < 4:
            continue
        x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
        width = max(0.0, x1 - x0)
        height = max(0.0, y1 - y0)
        if width < 80.0 or width < page_width * 0.12 or width > page_width * 0.55:
            continue
        if height > 3.0:
            continue
        if y0 < page_height * 0.60 or y0 >= footnote_top:
            continue
        gap_to_footnote = footnote_top - y1
        if gap_to_footnote < 4.0 or gap_to_footnote > 45.0:
            continue
        left_margin_ratio = x0 / page_width
        if left_margin_ratio < 0.08 or left_margin_ratio > 0.42:
            continue
        score = gap_to_footnote + abs(width - page_width * 0.22) * 0.02
        candidates.append(
            (
                score,
                {
                    "present": True,
                    "bbox": [x0, y0, x1, y1],
                    "gap_to_footnote": round(gap_to_footnote, 2),
                    "source": "page_drawing",
                },
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


_LOCAL_NOTE_START_RE = re.compile(
    r"^(?P<marker>[a-zA-Z]|[ivxlcdmIVXLCDM]{1,6}|[*#+$+\u2020\u2021])\s*[-\u2010-\u2015\uff0d:：.)）]\s*(?P<body>\S.*)$"
)


def _local_note_start(text: str) -> tuple[str, str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return "", ""
    match = _LOCAL_NOTE_START_RE.match(cleaned)
    if not match:
        return "", ""
    marker = _clean_text(str(match.group("marker") or ""))
    body = _clean_text(str(match.group("body") or ""))
    if not marker or not body:
        return "", ""
    return marker, body


def _template_numbered_note_start(text: str) -> tuple[int | None, str, str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return None, "", ""
    match = re.match(
        r"^(?:(?:备注|注释|说明)[:：]\s*)?"
        r"(?:\((?P<ascii>\d{1,3})\)|（(?P<fullwidth>\d{1,3})）)"
        r"\s*(?P<body>\S.*)$",
        cleaned,
    )
    if not match:
        return None, "", ""
    marker = str(match.group("ascii") or match.group("fullwidth") or "").strip()
    body = _clean_text(str(match.group("body") or ""))
    if not marker or not body:
        return None, "", ""
    try:
        return int(marker), marker, body
    except ValueError:
        return None, "", ""


def _normalize_structure_template_numbered_note_runs(template: dict[str, Any]) -> None:
    notes = [dict(note) for note in template.get("note_blocks", []) or [] if isinstance(note, dict)]
    if not notes:
        return

    numbered_notes: list[tuple[int, int, dict[str, Any]]] = []
    for original_index, note in enumerate(notes):
        note_number, marker, body = _template_numbered_note_start(str(note.get("text") or ""))
        if note_number is None:
            continue
        note["note_number"] = note_number
        note["marker"] = marker
        note["body"] = body
        note["note_run_id"] = str(template.get("structure_template_id") or template.get("block_id") or "template_note_run")
        note["note_run_order"] = note_number
        numbered_notes.append((note_number, original_index, note))
        notes[original_index] = note

    if len(numbered_notes) < 2:
        if numbered_notes:
            template["note_blocks"] = notes
        return

    numbers = {number for number, _index, _note in numbered_notes}
    has_contiguous_prefix = numbers == set(range(1, len(numbers) + 1))
    if not has_contiguous_prefix:
        template["note_blocks"] = notes
        return
    numbered_sequence = [number for number, _index, _note in numbered_notes]
    if numbered_sequence == sorted(numbered_sequence):
        template["note_blocks"] = notes
        return

    numbered_indices = {original_index for _number, original_index, _note in numbered_notes}
    ordered_numbered = [
        note
        for _number, _index, note in sorted(numbered_notes, key=lambda item: (item[0], item[1]))
    ]
    reordered: list[dict[str, Any]] = []
    inserted = False
    for index, note in enumerate(notes):
        if index in numbered_indices:
            if not inserted:
                reordered.extend(ordered_numbered)
                inserted = True
            continue
        reordered.append(note)
    if not inserted:
        reordered.extend(ordered_numbered)
    for index, note in enumerate(reordered, start=1):
        note.setdefault("note_index", index)
    template["note_blocks"] = reordered
    signals = dict(template.get("semantic_signals", {}) or {})
    signals["numbered_note_run_order_normalized"] = True
    signals["numbered_note_run_count"] = len(ordered_numbered)
    template["semantic_signals"] = signals


def _local_note_marker_anchor_refs(
    rows: list[str],
    note_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    notes_by_marker = {
        str(note.get("marker") or "").strip(): note
        for note in note_blocks
        if str(note.get("marker") or "").strip()
    }
    if not notes_by_marker:
        return []
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row_index, row in enumerate(rows, start=1):
        text = _clean_text(row)
        if not text:
            continue
        for marker, note in notes_by_marker.items():
            if not marker:
                continue
            escaped = re.escape(marker)
            has_anchor = bool(
                re.search(
                    rf"(?<=[A-Za-z\u4e00-\u9fff\)\]）]){escaped}(?=$|[\s,;，；:/\)\]）])",
                    text,
                )
            )
            if marker.isdigit():
                has_anchor = bool(re.search(rf"(?:\({escaped}\)|\uff08{escaped}\uff09)", text))
            if not has_anchor:
                continue
            key = (marker, text, str(note.get("text") or ""))
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                {
                    "marker": marker,
                    "anchor_text": text,
                    "anchor_row_index": row_index,
                    "note_text": str(note.get("text") or ""),
                    "note_block_id": note.get("source_block_id"),
                    "relation": "local_template_note_marker",
                    "confidence": 0.86,
                }
            )
    return refs
