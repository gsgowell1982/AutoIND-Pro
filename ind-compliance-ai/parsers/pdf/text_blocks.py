from __future__ import annotations

from functools import lru_cache
import re
import statistics
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional runtime dependency
    np = None  # type: ignore[assignment]

try:
    import pymupdf
except ImportError:  # pragma: no cover - optional runtime dependency
    pymupdf = None  # type: ignore[assignment]

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # pragma: no cover - optional runtime dependency
    RapidOCR = None  # type: ignore[assignment]

from .layout import (
    _body_bottom_limit,
    _body_top_limit,
    _words_in_bbox,
    _words_to_text,
    classify_text_block_layout_lane,
)
from .shared import (
    _Word,
    _bbox_intersection_ratio,
    _bbox_to_list,
    _bbox_union,
    _clean_text,
    _compact_text,
    _has_cjk,
    _horizontal_overlap_ratio,
    _text_contains_text,
)

# Update Notes:
# - v1.0.7 (2026-03-16): Classify page-edge text into unified structural roles
#   so numbered top-of-page section headings are preserved while footer/page
#   artifacts remain blocked from body/caption merges and header/footer removal.
# - v1.0.6 (2026-03-16): Add footer-artifact merge barriers in semantic text
#   merging so page numbers and tiny footer tokens do not pollute body/caption
#   text without words-layer continuity evidence.
#
# Version: v1.0.5
# Updates:
# - 在图像图注附近增加近似重复块的归并，规整 Figure caption 重复/页脚混入问题。
# - 优化 near-duplicate 合并 bbox 选择，优先保留更紧凑区域以避免跨越页脚或紧邻的图像。
# - 按行构建文本块，确保表格首行不会与前面的标题合并，从而让 table suppression 更可靠。

# - Add a visual-line reconstruction stage that uses words-layer continuity to
#   repair fragmented same-line text objects before semantic merging.
# - Sort text blocks by visual rows instead of raw bbox top values so oversized
#   glyph boxes do not scramble left-to-right reading order.
_ROMAN_PAGE_NUMBER_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
_HEADING_PREFIX_RE = re.compile(r"^(?:\d{1,3}|[A-Z]|[IVXLCDM]{1,8})[.)]$", re.IGNORECASE)
_NUMBERED_HEADING_RE = re.compile(r"^(?:\d{1,3}|[A-Z]|[IVXLCDM]{1,8})[.)]\s+\S", re.IGNORECASE)
_DOTTED_NUMBERED_HEADING_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3})+\s+\S")
_STRUCTURAL_NUMBERED_HEADING_RE = re.compile(
    r"^(?:\d{1,3}(?:\.\d{1,3})*|[IVXLCDM]{1,8})[.)]?\s+\S",
    re.IGNORECASE,
)
_QUOTE_CHAR_RE = re.compile(r"[\"'“”‘’]")
_DOUBLE_QUOTE_CHAR_RE = re.compile(r"[\"\u201c\u201d]")
_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]{3,}")
_SHORT_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]{1,2}")
_MULTILINGUAL_DAMAGE_HARD_CHAR_RE = re.compile(r"[\u00d7\u22c6\u2208\u2212\u2265\u2264\uff08\uff09\uff0c\u3002\uff1b\u223c\u02c6\u00af]")
_MULTILINGUAL_DAMAGE_SOFT_CHAR_RE = re.compile(r"[\u00a0\u2002\u2003\u2009\u200a\u202f]")
_BODY_OCR_RENDER_SCALE = 4.5
_INLINE_EQUATION_MARKER_RE = re.compile(r"^\(\s*\d{1,3}\s*\)$")


def _extract_visible_span_metrics(
    spans: list[dict[str, Any]],
    fallback_bbox: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], list[float]]:
    visible_bboxes: list[tuple[float, float, float, float]] = []
    visible_font_sizes: list[float] = []

    for span in spans:
        if not _clean_text(str(span.get("text", ""))):
            continue
        span_bbox = tuple(float(item) for item in span.get("bbox", fallback_bbox))
        if len(span_bbox) != 4:
            continue
        visible_bboxes.append(span_bbox)
        try:
            span_font_size = float(span.get("size", 0.0) or 0.0)
        except (TypeError, ValueError):
            span_font_size = 0.0
        if span_font_size > 0:
            visible_font_sizes.append(span_font_size)

    if visible_bboxes:
        return _bbox_union(visible_bboxes), visible_font_sizes
    return fallback_bbox, visible_font_sizes


def _extract_visible_span_records(
    spans: list[dict[str, Any]],
    fallback_bbox: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for span in spans:
        text = _clean_text(str(span.get("text", "")))
        if not text:
            continue
        raw_bbox = span.get("bbox", fallback_bbox)
        try:
            bbox = tuple(float(item) for item in raw_bbox)
        except (TypeError, ValueError):
            continue
        if len(bbox) != 4:
            continue
        try:
            font_size = float(span.get("size", 0.0) or 0.0)
        except (TypeError, ValueError):
            font_size = 0.0
        records.append(
            {
                "text": text,
                "bbox": _bbox_to_list(bbox),
                "font_size": font_size,
                "font": str(span.get("font", "") or ""),
                "flags": int(span.get("flags", 0) or 0),
            }
        )
    return records


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
            for line_index, line in enumerate(block.get("lines", [])):
                span_text = ""
                font_sizes: list[float] = []
                spans = list(line.get("spans", []) or [])
                for span in spans:
                    span_text += str(span.get("text", ""))
                    try:
                        font_sizes.append(float(span.get("size", 0.0)))
                    except (TypeError, ValueError):
                        continue
                text = _clean_text(span_text)
                if not text:
                    continue
                line_bbox = tuple(float(item) for item in line.get("bbox", bbox))
                visible_line_bbox, visible_font_sizes = _extract_visible_span_metrics(spans, line_bbox)
                visible_spans = _extract_visible_span_records(spans, line_bbox)
                text_blocks.append(
                    {
                        "block_type": "text",
                        "block_id": f"txt_p{page_number}_{len(text_blocks) + 1:03d}",
                        "page": page_number,
                        "bbox": _bbox_to_list(visible_line_bbox),
                        "text": text,
                        "font_size": statistics.median(visible_font_sizes or font_sizes) if (visible_font_sizes or font_sizes) else 0.0,
                        "spans": visible_spans,
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


@lru_cache(maxsize=1)
def _get_body_text_ocr_engine() -> Any:
    if RapidOCR is None:
        return None
    try:
        return RapidOCR()
    except Exception:
        return None


def _bbox_width(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0]))


def _bbox_height(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, float(bbox[3]) - float(bbox[1]))


def _compute_body_ocr_render_scale(clip_width: float, clip_height: float) -> float:
    return _BODY_OCR_RENDER_SCALE


def _row_bbox(row: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    return _bbox_union([tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0))) for block in row])


def _row_text(row: list[dict[str, Any]]) -> str:
    return _clean_text(" ".join(str(block.get("text", "")) for block in row))


def _text_contains_quote_marker(text: str) -> bool:
    normalized = _clean_text(text)
    return any(char in normalized for char in ('"', "'", "\u201c", "\u201d", "\u2018", "\u2019"))


def _text_contains_double_quote_marker(text: str) -> bool:
    return bool(_DOUBLE_QUOTE_CHAR_RE.search(_clean_text(text)))


def _cjk_char_count(text: str) -> int:
    normalized = _clean_text(text)
    return sum(1 for char in normalized if "\u4e00" <= char <= "\u9fff")


def _page_has_damaged_multilingual_text_signal(text_blocks: list[dict[str, Any]]) -> bool:
    page_text = "\n".join(_clean_text(str(block.get("text", ""))) for block in text_blocks)
    if not page_text:
        return False
    hard_count = len(_MULTILINGUAL_DAMAGE_HARD_CHAR_RE.findall(page_text))
    soft_count = len(_MULTILINGUAL_DAMAGE_SOFT_CHAR_RE.findall(page_text))
    return hard_count >= 2 or (hard_count >= 1 and soft_count >= 2)


def _row_has_large_internal_gap(row: list[dict[str, Any]], page_width: float) -> bool:
    if len(row) < 2:
        return False
    gap_threshold = max(18.0, page_width * 0.045)
    ordered = sorted(row, key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return any(
        (float(right["bbox"][0]) - float(left["bbox"][2])) >= gap_threshold
        for left, right in zip(ordered, ordered[1:])
    )


def _row_is_narrow_body_candidate(
    row: list[dict[str, Any]],
    page_width: float,
    page_height: float,
) -> bool:
    bbox = _row_bbox(row)
    if _bbox_width(bbox) > page_width * 0.62:
        return False
    if _bbox_height(bbox) > page_height * 0.06:
        return False
    body_top = _body_top_limit(page_height)
    body_bottom = _body_bottom_limit(page_height)
    center_y = (bbox[1] + bbox[3]) / 2.0
    return body_top <= center_y <= body_bottom


def _is_suspicious_quote_gap_seed_row(
    row: list[dict[str, Any]],
    page_width: float,
    page_height: float,
) -> bool:
    row_text = _row_text(row)
    if not row_text or _has_cjk(row_text):
        return False
    if not _row_is_narrow_body_candidate(row, page_width, page_height):
        return False
    if not _text_contains_quote_marker(row_text):
        return False
    if not _row_has_large_internal_gap(row, page_width):
        return False
    return any(str(block.get("source", "text-layer")) == "text-layer" for block in row)


def _is_multilingual_quote_spot_check_row(
    row: list[dict[str, Any]],
    page_width: float,
    page_height: float,
    *,
    page_has_damaged_multilingual_signal: bool,
) -> bool:
    row_text = _row_text(row)
    if not row_text or _has_cjk(row_text):
        return False
    if not page_has_damaged_multilingual_signal:
        return False
    if not _row_is_narrow_body_candidate(row, page_width, page_height):
        return False
    if not _text_contains_double_quote_marker(row_text):
        return False
    if len(row_text) < 12 or len(row_text) > 180:
        return False
    return any(str(block.get("source", "text-layer")) == "text-layer" for block in row)


def _row_is_cluster_continuation(
    row: list[dict[str, Any]],
    cluster_bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
) -> bool:
    row_text = _row_text(row)
    if not row_text or _has_cjk(row_text):
        return False
    if not _row_is_narrow_body_candidate(row, page_width, page_height):
        return False
    row_bbox = _row_bbox(row)
    row_center_x = (row_bbox[0] + row_bbox[2]) / 2.0
    cluster_center_x = (cluster_bbox[0] + cluster_bbox[2]) / 2.0
    if abs(row_center_x - cluster_center_x) > page_width * 0.1:
        return False
    if _horizontal_overlap_ratio(row_bbox, cluster_bbox) < 0.3:
        return False
    vertical_gap = max(0.0, row_bbox[1] - cluster_bbox[3])
    return vertical_gap <= max(8.0, _bbox_height(cluster_bbox) * 0.32)


def _collect_quote_gap_cluster_rows(
    rows: list[list[dict[str, Any]]],
    start_index: int,
    page_width: float,
    page_height: float,
) -> tuple[list[list[dict[str, Any]]], int]:
    cluster_rows = [rows[start_index]]
    cluster_bbox = _row_bbox(rows[start_index])
    max_rows = 5
    index = start_index + 1

    while index < len(rows) and len(cluster_rows) < max_rows:
        row = rows[index]
        if not _row_is_cluster_continuation(row, cluster_bbox, page_width, page_height):
            break
        row_text = _row_text(row)
        if not _text_contains_quote_marker(row_text):
            break
        cluster_rows.append(row)
        cluster_bbox = _bbox_union([cluster_bbox, _row_bbox(row)])
        index += 1

    return cluster_rows, index - 1


def _extract_alpha_tokens(text: str) -> set[str]:
    return {token.lower() for token in _ALPHA_TOKEN_RE.findall(_clean_text(text))}


def _alpha_token_overlap_ratio(source_text: str, candidate_text: str) -> tuple[int, float]:
    source_tokens = _extract_alpha_tokens(source_text)
    candidate_tokens = _extract_alpha_tokens(candidate_text)
    if not source_tokens or not candidate_tokens:
        return 0, 0.0
    shared = source_tokens & candidate_tokens
    ratio = len(shared) / max(1, min(len(source_tokens), len(candidate_tokens)))
    return len(shared), ratio


def _cluster_source_block_indices(cluster_rows: list[list[dict[str, Any]]]) -> list[int]:
    indices: list[int] = []
    for row in cluster_rows:
        for block in row:
            indices.extend(_normalized_source_block_indices(block))
    return sorted(set(indices))


def _clip_bbox_with_margin(
    bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    x_margin: float = 6.0,
    y_margin: float = 2.0,
) -> tuple[float, float, float, float]:
    return (
        max(0.0, bbox[0] - x_margin),
        max(0.0, bbox[1] - y_margin),
        min(page_width, bbox[2] + x_margin),
        min(page_height, bbox[3] + y_margin),
    )


def _infer_cluster_layout_lane(
    cluster_blocks: list[dict[str, Any]],
    layout_profile: dict[str, Any] | None,
) -> str:
    explicit_lane = next(
        (
            str(block.get("layout_lane", "") or "").strip()
            for block in cluster_blocks
            if str(block.get("layout_lane", "") or "").strip()
        ),
        "",
    )
    if explicit_lane:
        return explicit_lane
    if not cluster_blocks or not layout_profile:
        return ""
    cluster_bbox = _bbox_union(
        [tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0))) for block in cluster_blocks]
    )
    return classify_text_block_layout_lane(
        {"bbox": _bbox_to_list(cluster_bbox)},
        layout_profile,
    )


def _clip_bbox_to_layout_lane(
    clip_bbox: tuple[float, float, float, float],
    cluster_blocks: list[dict[str, Any]],
    layout_profile: dict[str, Any] | None,
    page_width: float,
) -> tuple[float, float, float, float]:
    if not layout_profile:
        return clip_bbox
    mode = str(layout_profile.get("mode", "") or "")
    if mode not in {"two_column", "mixed"}:
        return clip_bbox

    lane = _infer_cluster_layout_lane(cluster_blocks, layout_profile)
    if lane not in {"left", "right"}:
        return clip_bbox

    x0, y0, x1, y1 = clip_bbox
    column_mid = float(layout_profile.get("column_mid", page_width / 2.0) or (page_width / 2.0))
    lane_tolerance = float(layout_profile.get("lane_tolerance", max(12.0, page_width * 0.04)) or max(12.0, page_width * 0.04))
    gutter_guard = max(4.0, min(10.0, lane_tolerance * 0.25))

    if lane == "left":
        x1 = min(x1, column_mid - gutter_guard)
    else:
        x0 = max(x0, column_mid + gutter_guard)

    min_clip_width = 24.0
    if x1 - x0 < min_clip_width:
        return clip_bbox
    return (x0, y0, x1, y1)


def _ocr_block_stays_within_layout_lane(
    bbox: tuple[float, float, float, float],
    layout_lane: str,
    layout_profile: dict[str, Any] | None,
    page_width: float,
) -> bool:
    if not layout_profile or layout_lane not in {"left", "right"}:
        return True
    mode = str(layout_profile.get("mode", "") or "")
    if mode not in {"two_column", "mixed"}:
        return True

    column_mid = float(layout_profile.get("column_mid", page_width / 2.0) or (page_width / 2.0))
    lane_tolerance = float(layout_profile.get("lane_tolerance", max(12.0, page_width * 0.04)) or max(12.0, page_width * 0.04))
    gutter_guard = max(4.0, min(10.0, lane_tolerance * 0.25))
    if layout_lane == "left":
        return float(bbox[2]) <= column_mid + gutter_guard
    return float(bbox[0]) >= column_mid - gutter_guard


def _run_local_body_text_ocr(
    page: "pymupdf.Page",
    clip_bbox: tuple[float, float, float, float],
) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    ocr_engine = _get_body_text_ocr_engine()
    if ocr_engine is None or np is None or pymupdf is None:
        return []

    clip = pymupdf.Rect(*clip_bbox)
    if clip.width < 24 or clip.height < 20:
        return []
    render_scale = _compute_body_ocr_render_scale(float(clip.width), float(clip.height))

    try:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(render_scale, render_scale), clip=clip, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        result, _ = ocr_engine(image)
    except Exception:
        return []

    entries: list[tuple[str, float, tuple[float, float, float, float]]] = []
    if not result:
        return entries

    scale_x = pix.width / max(1.0, float(clip.width))
    scale_y = pix.height / max(1.0, float(clip.height))
    for item in result:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        box = item[0]
        text = _clean_text(str(item[1]))
        confidence = float(item[2] or 0.0)
        if not text or confidence < 0.4:
            continue
        try:
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
        except Exception:
            continue
        bbox = (
            float(clip.x0) + min(xs) / max(1.0, scale_x),
            float(clip.y0) + min(ys) / max(1.0, scale_y),
            float(clip.x0) + max(xs) / max(1.0, scale_x),
            float(clip.y0) + max(ys) / max(1.0, scale_y),
        )
        entries.append((text, confidence, bbox))

    entries.sort(key=lambda item: (((item[2][1] + item[2][3]) / 2.0), item[2][0]))
    return entries


def _group_local_ocr_rows(
    entries: list[tuple[str, float, tuple[float, float, float, float]]],
) -> list[list[tuple[str, float, tuple[float, float, float, float]]]]:
    if not entries:
        return []

    heights = [_bbox_height(item[2]) for item in entries if _bbox_height(item[2]) > 0]
    row_tolerance = max(4.0, statistics.median(heights) * 0.65) if heights else 4.0
    rows: list[dict[str, Any]] = []
    for entry in entries:
        bbox = entry[2]
        center_y = (bbox[1] + bbox[3]) / 2.0
        target_row: dict[str, Any] | None = None
        for row in rows:
            if abs(center_y - float(row["center_y"])) <= row_tolerance:
                target_row = row
                break
        if target_row is None:
            rows.append({"center_y": center_y, "entries": [entry]})
            continue
        target_row["entries"].append(entry)
        target_row["center_y"] = statistics.mean(
            [((item[2][1] + item[2][3]) / 2.0) for item in target_row["entries"]]
        )
    grouped_rows = [
        sorted(list(row["entries"]), key=lambda item: (item[2][0], item[2][1]))
        for row in rows
    ]
    row_heights = [
        _bbox_height(_bbox_union([entry[2] for entry in row_entries]))
        for row_entries in grouped_rows
        if row_entries
    ]
    if not row_heights:
        return grouped_rows
    min_height = max(4.5, statistics.median(row_heights) * 0.55)
    return [
        row_entries
        for row_entries in grouped_rows
        if _bbox_height(_bbox_union([entry[2] for entry in row_entries])) >= min_height
    ]


def _clean_local_ocr_row_text(text: str) -> str:
    # Remove isolated lowercase OCR artifacts while preserving common prose tokens like "a".
    cleaned = _clean_text(text)
    cleaned = re.sub(r"(\b[a-z]{3,}\s)i(\s[a-z]{3,}\b)", r"\1 \2", cleaned)
    cleaned = re.sub(r"\b(?![ai]\b)[a-z]\b", " ", cleaned)
    cleaned = re.sub(r"([\"'\u201c\u201d\u2018\u2019])\s+(?=[\"'\u201c\u201d\u2018\u2019])", r"\1", cleaned)
    return _clean_text(cleaned)


def _join_ocr_row_text(entries: list[tuple[str, float, tuple[float, float, float, float]]]) -> str:
    merged = ""
    for text, _, _ in entries:
        merged = text if not merged else _join_text_fragments(merged, text)
    return _clean_local_ocr_row_text(merged)


def _build_body_ocr_repair_blocks(
    cluster_rows: list[list[dict[str, Any]]],
    ocr_rows: list[list[tuple[str, float, tuple[float, float, float, float]]]],
    page_number: int,
    *,
    reason: str = "suspicious_quote_gap",
    layout_profile: dict[str, Any] | None = None,
    page_width: float = 0.0,
) -> list[dict[str, Any]]:
    if not ocr_rows:
        return []
    source_indices = _cluster_source_block_indices(cluster_rows)
    cluster_blocks = [block for row in cluster_rows for block in row]
    font_sizes = [
        float(block.get("font_size", 0.0) or 0.0)
        for block in cluster_blocks
        if float(block.get("font_size", 0.0) or 0.0) > 0
    ]
    font_size = statistics.median(font_sizes) if font_sizes else 0.0
    layout_mode = next((block.get("layout_mode") for block in cluster_blocks if block.get("layout_mode")), "")
    layout_lane = _infer_cluster_layout_lane(cluster_blocks, layout_profile)
    layout_confidence = next(
        (
            float(block.get("layout_confidence", 0.0) or 0.0)
            for block in cluster_blocks
            if block.get("layout_confidence") is not None
        ),
        0.0,
    )

    repaired_blocks: list[dict[str, Any]] = []
    for row_index, entries in enumerate(ocr_rows, start=1):
        text = _join_ocr_row_text(entries)
        if not text:
            continue
        bbox = _bbox_union([entry[2] for entry in entries])
        if not _ocr_block_stays_within_layout_lane(
            bbox,
            layout_lane,
            layout_profile,
            page_width,
        ):
            continue
        block = {
            "block_type": "text",
            "block_id": f"txt_p{page_number}_ocr_{row_index:03d}",
            "page": page_number,
            "bbox": _bbox_to_list(bbox),
            "text": text,
            "font_size": font_size,
            "source": "body-ocr-repair",
            "source_block_indices": list(source_indices),
            "ocr_repair_reason": reason,
        }
        if layout_mode:
            block["layout_mode"] = layout_mode
        if layout_lane:
            block["layout_lane"] = layout_lane
        block["layout_confidence"] = layout_confidence
        repaired_blocks.append(block)
    return repaired_blocks


def _repair_quote_gap_cluster_with_local_ocr(
    page: "pymupdf.Page",
    cluster_rows: list[list[dict[str, Any]]],
    page_number: int,
    page_width: float,
    page_height: float,
    *,
    reason: str = "suspicious_quote_gap",
    layout_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cluster_blocks = [block for row in cluster_rows for block in row]
    cluster_text = _clean_text(" ".join(str(block.get("text", "")) for block in cluster_blocks))
    if not cluster_text or _has_cjk(cluster_text):
        return []

    cluster_bbox = _row_bbox(cluster_blocks)
    x_margin = 24.0 if len(cluster_rows) == 1 else 6.0
    y_margin = 8.0 if len(cluster_rows) == 1 else 2.0
    clip_bbox = _clip_bbox_with_margin(
        cluster_bbox,
        page_width,
        page_height,
        x_margin=x_margin,
        y_margin=y_margin,
    )
    clip_bbox = _clip_bbox_to_layout_lane(
        clip_bbox,
        cluster_blocks,
        layout_profile,
        page_width,
    )
    ocr_entries = _run_local_body_text_ocr(page, clip_bbox)
    if not ocr_entries:
        return []

    ocr_rows = _group_local_ocr_rows(ocr_entries)
    if not ocr_rows:
        return []

    ocr_text = "\n".join(_join_ocr_row_text(row) for row in ocr_rows).strip()
    if not ocr_text or _cjk_char_count(ocr_text) < 2:
        return []

    shared_token_count, overlap_ratio = _alpha_token_overlap_ratio(cluster_text, ocr_text)
    if shared_token_count < 4 or overlap_ratio < 0.45:
        return []

    repaired_blocks = _build_body_ocr_repair_blocks(
        cluster_rows,
        ocr_rows,
        page_number,
        reason=reason,
        layout_profile=layout_profile,
        page_width=page_width,
    )
    if not repaired_blocks:
        return []
    if not any(_cjk_char_count(str(block.get("text", ""))) >= 2 for block in repaired_blocks):
        return []
    return repaired_blocks


def _repair_lane_suspicious_body_text_blocks(
    page: "pymupdf.Page",
    page_number: int,
    lane_blocks: list[dict[str, Any]],
    page_width: float,
    page_height: float,
    layout_profile: dict[str, Any] | None = None,
    *,
    page_has_damaged_multilingual_signal: bool,
) -> tuple[list[dict[str, Any]], int]:
    rows = _group_text_blocks_by_visual_rows(lane_blocks)
    if not rows:
        return lane_blocks, 0

    repaired_blocks: list[dict[str, Any]] = []
    repair_count = 0
    row_index = 0
    while row_index < len(rows):
        row = rows[row_index]
        if _is_suspicious_quote_gap_seed_row(row, page_width, page_height):
            cluster_rows, cluster_end_index = _collect_quote_gap_cluster_rows(
                rows,
                row_index,
                page_width,
                page_height,
            )
            repaired_cluster = _repair_quote_gap_cluster_with_local_ocr(
                page,
                cluster_rows,
                page_number,
                page_width,
                page_height,
                layout_profile=layout_profile,
            )
            if repaired_cluster:
                repaired_blocks.extend(repaired_cluster)
                repair_count += 1
            else:
                for cluster_row in cluster_rows:
                    repaired_blocks.extend(cluster_row)
            row_index = cluster_end_index + 1
            continue

        if _is_multilingual_quote_spot_check_row(
            row,
            page_width,
            page_height,
            page_has_damaged_multilingual_signal=page_has_damaged_multilingual_signal,
        ):
            repaired_row = _repair_quote_gap_cluster_with_local_ocr(
                page,
                [row],
                page_number,
                page_width,
                page_height,
                reason="multilingual_quote_spot_check",
                layout_profile=layout_profile,
            )
            if repaired_row:
                repaired_blocks.extend(repaired_row)
                repair_count += 1
                row_index += 1
                continue
        repaired_blocks.extend(row)
        row_index += 1

    return repaired_blocks, repair_count


def repair_suspicious_body_text_blocks_with_local_ocr(
    page: "pymupdf.Page",
    page_number: int,
    text_blocks: list[dict[str, Any]],
    page_width: float,
    page_height: float,
    layout_profile: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if not text_blocks or RapidOCR is None or np is None or pymupdf is None:
        return text_blocks, 0

    page_has_damaged_multilingual_signal = _page_has_damaged_multilingual_text_signal(text_blocks)
    lane_blocks: dict[str, list[dict[str, Any]]] = {}
    for block in text_blocks:
        lane = _layout_lane(block, layout_profile)
        lane_blocks.setdefault(lane or "full_width", []).append(block)

    repaired: list[dict[str, Any]] = []
    repair_count = 0
    for blocks in lane_blocks.values():
        lane_repaired, lane_count = _repair_lane_suspicious_body_text_blocks(
            page,
            page_number,
            blocks,
            page_width,
            page_height,
            layout_profile=layout_profile,
            page_has_damaged_multilingual_signal=page_has_damaged_multilingual_signal,
        )
        repaired.extend(lane_repaired)
        repair_count += lane_count

    return _sort_text_blocks_by_visual_rows(repaired), repair_count


def _text_starts_with_break_marker(text: str) -> bool:
    return bool(re.match(r"^[•\-·●◆\(\)（）\[\]]", text.strip()))


def _text_ends_with_break_marker(text: str) -> bool:
    return bool(re.search(r"[。！？；:：]\s*$", text.strip()))


def _looks_like_inline_equation_marker_text(text: str) -> bool:
    return bool(_INLINE_EQUATION_MARKER_RE.fullmatch(_clean_text(text)))


def _looks_like_short_mathish_fragment(text: str) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) > 18:
        return False
    if _looks_like_inline_equation_marker_text(compact):
        return True
    short_tokens = _SHORT_ALPHA_TOKEN_RE.findall(compact)
    if not short_tokens:
        return False
    has_single_letter_token = any(len(token) == 1 for token in short_tokens)
    has_math_signal = bool(re.search(r"[\(\)\[\]\{\}=+\-*/αβγλσθ]", compact, re.IGNORECASE))
    if has_single_letter_token and has_math_signal:
        return True
    return has_single_letter_token and len(short_tokens) >= 2 and len(compact) <= 8


def _looks_like_inline_math_fragment(text: str, *, allow_atom: bool = True) -> bool:
    compact = _clean_text(text)
    if not compact or len(compact) > 120:
        return False
    if _looks_like_inline_equation_marker_text(compact):
        return False
    lowered = compact.lower()
    if "http" in lowered or "doi" in lowered:
        return False
    if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z ]{3,}$", compact):
        return False
    if allow_atom and re.fullmatch(r"(?:[A-Za-z0-9]{1,3}|[∑∫√=+\-*/≤≥<>]{1,4})", compact):
        return True
    has_math_operator = bool(re.search(r"[=+\-*/∑∫√≤≥<>伪尾纬位蟽胃]", compact, re.IGNORECASE))
    has_grouping = bool(re.search(r"[\(\)\[\]\{\}]", compact))
    short_tokens = _SHORT_ALPHA_TOKEN_RE.findall(compact)
    has_variable_tokens = sum(1 for token in short_tokens if len(token) <= 2) >= 1
    if has_math_operator and (has_variable_tokens or has_grouping):
        return True
    if has_grouping and sum(1 for token in short_tokens if len(token) == 1) >= 2:
        return True
    return _looks_like_short_mathish_fragment(compact)


def _has_inline_math_context(text: str) -> bool:
    compact = _clean_text(text)
    if not compact:
        return False
    lowered = compact.lower()
    if re.search(r"(?:i\.e\.|e\.g\.|respectively|where|denotes|defined as|given by)", lowered):
        return bool(re.search(r"[=∑∫√≤≥<>+\-*/\(\)\[\]]", compact))
    if re.search(r"[A-Za-z]\s*[=≤≥<>]", compact):
        return True
    return compact.endswith(("=", "+", "-", "*", "/", "(", "[", "{", ","))


def _can_merge_inline_math_row_fragments(
    left: dict[str, Any],
    right: dict[str, Any],
    page_width: float,
) -> bool:
    left_text = _clean_text(left.get("text", ""))
    right_text = _clean_text(right.get("text", ""))
    if not left_text or not right_text:
        return False
    if _text_starts_with_break_marker(right_text):
        return False
    left_has_context = _has_inline_math_context(left_text)
    if not (left_has_context or _looks_like_inline_math_fragment(left_text, allow_atom=False)):
        return False
    if not (_looks_like_inline_math_fragment(right_text, allow_atom=left_has_context) or _has_inline_math_context(right_text)):
        return False

    left_bbox = tuple(left["bbox"])
    right_bbox = tuple(right["bbox"])
    if right_bbox[0] < left_bbox[0] - 1.5:
        return False
    left_height = max(1.0, left_bbox[3] - left_bbox[1])
    right_height = max(1.0, right_bbox[3] - right_bbox[1])
    vertical_overlap = _vertical_overlap_ratio(left_bbox, right_bbox)
    center_gap = abs(_block_center_y(left) - _block_center_y(right))
    max_center_gap = max(7.5, min(left_height, right_height) * 1.45)
    if vertical_overlap < 0.25 and center_gap > max_center_gap:
        return False

    gap = right_bbox[0] - left_bbox[2]
    max_gap = max(18.0, page_width * 0.035, min(left_height, right_height) * 2.6)
    min_gap = -10.0 if left_has_context else -4.0
    return min_gap <= gap <= max_gap


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


def _block_height(block: dict[str, Any]) -> float:
    bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    return max(1.0, bbox[3] - bbox[1])


def _shares_source_block_lineage(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_indices = set(_normalized_source_block_indices(left))
    right_indices = set(_normalized_source_block_indices(right))
    if not left_indices or not right_indices:
        return False
    return bool(left_indices & right_indices)


def _block_center_y(block: dict[str, Any]) -> float:
    bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    return (bbox[1] + bbox[3]) / 2


def _normalized_source_block_indices(block: dict[str, Any]) -> list[int]:
    indices = block.get("source_block_indices")
    if isinstance(indices, list):
        normalized = [int(item) for item in indices if isinstance(item, (int, float))]
        if normalized:
            return sorted(set(normalized))
    source_index = block.get("source_block_index")
    if isinstance(source_index, (int, float)):
        return [int(source_index)]
    return []


def _visual_row_tolerance(text_blocks: list[dict[str, Any]]) -> float:
    heights = [_block_height(block) for block in text_blocks if block.get("bbox")]
    if not heights:
        return 2.5
    return max(2.5, statistics.median(heights) * 0.45)


def _group_text_blocks_by_visual_rows(
    text_blocks: list[dict[str, Any]],
    row_tolerance: float | None = None,
) -> list[list[dict[str, Any]]]:
    if not text_blocks:
        return []

    effective_tolerance = row_tolerance if row_tolerance is not None else _visual_row_tolerance(text_blocks)
    ordered = sorted(text_blocks, key=lambda item: (_block_center_y(item), item["bbox"][0], item["bbox"][1]))
    rows: list[dict[str, Any]] = []

    for block in ordered:
        bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        center_y = _block_center_y(block)
        best_row: dict[str, Any] | None = None
        best_score: tuple[float, float] | None = None

        for row in rows:
            row_bbox = tuple(row["bbox"])
            overlap = _vertical_overlap_ratio(bbox, row_bbox)
            center_diff = abs(center_y - float(row["center_y"]))
            if center_diff > effective_tolerance and overlap < 0.6:
                continue
            score = (overlap, -center_diff)
            if best_score is None or score > best_score:
                best_score = score
                best_row = row

        if best_row is None:
            rows.append(
                {
                    "bbox": _bbox_to_list(bbox),
                    "center_y": center_y,
                    "blocks": [block],
                }
            )
            continue

        best_row["blocks"].append(block)
        best_row["bbox"] = _bbox_to_list(_bbox_union([tuple(best_row["bbox"]), bbox]))
        best_row["center_y"] = statistics.median([_block_center_y(item) for item in best_row["blocks"]])

    rows.sort(key=lambda row: (float(row["center_y"]), float(row["bbox"][0])))
    grouped_rows: list[list[dict[str, Any]]] = []
    for row in rows:
        grouped_rows.append(sorted(row["blocks"], key=lambda item: (item["bbox"][0], _block_center_y(item))))
    return grouped_rows


def _sort_text_blocks_by_visual_rows(text_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not text_blocks:
        return []
    ordered: list[dict[str, Any]] = []
    for row in _group_text_blocks_by_visual_rows(text_blocks):
        ordered.extend(row)
    return ordered


def _layout_lane(
    block: dict[str, Any],
    layout_profile: dict[str, Any] | None = None,
) -> str:
    lane = str(block.get("layout_lane", "") or "").strip()
    if lane:
        return lane
    return classify_text_block_layout_lane(block, layout_profile)


def _same_layout_lane(
    left: dict[str, Any],
    right: dict[str, Any],
    layout_profile: dict[str, Any] | None = None,
) -> bool:
    left_lane = _layout_lane(left, layout_profile)
    right_lane = _layout_lane(right, layout_profile)
    if not left_lane or not right_lane:
        return True
    return left_lane == right_lane


def _row_layout_mode(
    row: list[dict[str, Any]],
    layout_profile: dict[str, Any] | None = None,
) -> str:
    lanes = {_layout_lane(block, layout_profile) for block in row if _clean_text(block.get("text", ""))}
    if not lanes:
        return "full_width"
    if "full_width" in lanes:
        return "full_width"
    return "columns"


def _reading_order_zones(
    text_blocks: list[dict[str, Any]],
    layout_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ordered_rows = _group_text_blocks_by_visual_rows(text_blocks)
    zones: list[dict[str, Any]] = []
    for row in ordered_rows:
        row_mode = _row_layout_mode(row, layout_profile)
        if not zones or zones[-1]["mode"] != row_mode:
            zones.append({"mode": row_mode, "rows": [row]})
            continue
        zones[-1]["rows"].append(row)
    return zones


def _compact_publication_heading_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean_text(text).lower())


def _looks_like_article_info_heading(text: str) -> bool:
    compact = _compact_publication_heading_text(text)
    return compact in {
        "articleinfo",
        "articleinformation",
        "articlehistory",
    }


def _looks_like_abstract_heading(text: str) -> bool:
    return _compact_publication_heading_text(text) == "abstract"


def _looks_like_publication_body_start(text: str) -> bool:
    normalized = _clean_text(text)
    return bool(
        re.match(
            r"^(?:\d{1,2}(?:\.\d+)*\.?\s+)?(?:introduction|background)\b",
            normalized,
            re.IGNORECASE,
        )
    )


def _looks_like_publication_footer_block(block: dict[str, Any], page_height: float) -> bool:
    text = _clean_text(str(block.get("text", "")))
    if not text:
        return False
    bbox = tuple(float(item) for item in block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(bbox) != 4:
        return False
    lower_page = bbox[1] >= page_height * 0.84 if page_height > 0 else False
    compact = text.lower()
    footer_signal = (
        "doi.org" in compact
        or compact.startswith("doi:")
        or "corresponding author" in compact
        or "e-mail address" in compact
        or "email address" in compact
        or "all rights reserved" in compact
        or bool(re.search(r"\bissn\b|^\d{4}-\d{3}[\dx]/", compact))
    )
    return lower_page and footer_signal


def _flatten_visual_rows(rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        flattened.extend(sorted(row, key=lambda item: (item["bbox"][0], _block_center_y(item))))
    return flattened


def _order_publication_front_matter_zone(
    rows: list[list[dict[str, Any]]],
    layout_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    """Order article metadata and abstract panels before the body on journal pages."""
    flattened = _flatten_visual_rows(rows)
    if not flattened:
        return None

    article_index = next(
        (
            index
            for index, block in enumerate(flattened)
            if _looks_like_article_info_heading(str(block.get("text", "")))
        ),
        -1,
    )
    abstract_index = next(
        (
            index
            for index, block in enumerate(flattened)
            if _looks_like_abstract_heading(str(block.get("text", "")))
        ),
        -1,
    )
    if article_index < 0 or abstract_index < 0:
        return None

    article_heading = flattened[article_index]
    abstract_heading = flattened[abstract_index]
    article_bbox = tuple(float(item) for item in article_heading.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    abstract_bbox = tuple(float(item) for item in abstract_heading.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(article_bbox) != 4 or len(abstract_bbox) != 4:
        return None
    if abs(_block_center_y(article_heading) - _block_center_y(abstract_heading)) > max(
        8.0,
        _block_height(article_heading) * 1.4,
        _block_height(abstract_heading) * 1.4,
    ):
        return None
    if abstract_bbox[0] <= article_bbox[0]:
        return None

    body_index = next(
        (
            index
            for index, block in enumerate(flattened)
            if index > max(article_index, abstract_index)
            and _looks_like_publication_body_start(str(block.get("text", "")))
        ),
        -1,
    )
    if body_index < 0:
        return None

    page_width = float((layout_profile or {}).get("page_width", 0.0) or 0.0)
    page_height = float((layout_profile or {}).get("page_height", 0.0) or 0.0)
    abstract_left = abstract_bbox[0]
    panel_margin = max(10.0, page_width * 0.02) if page_width > 0 else 10.0
    abstract_y = min(article_bbox[1], abstract_bbox[1])

    prefix = flattened[: min(article_index, abstract_index)]
    panel_blocks = [
        block
        for index, block in enumerate(flattened)
        if min(article_index, abstract_index) <= index < body_index
        and tuple(float(item) for item in block.get("bbox", (0.0, 0.0, 0.0, 0.0)))[1] >= abstract_y - panel_margin
    ]

    article_blocks: list[dict[str, Any]] = []
    abstract_blocks: list[dict[str, Any]] = []
    for block in panel_blocks:
        text = str(block.get("text", ""))
        bbox = tuple(float(item) for item in block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        if len(bbox) != 4:
            continue
        if block is article_heading or _looks_like_article_info_heading(text):
            article_blocks.append(block)
        elif block is abstract_heading or _looks_like_abstract_heading(text):
            abstract_blocks.append(block)
        elif bbox[0] >= abstract_left - panel_margin:
            abstract_blocks.append(block)
        else:
            article_blocks.append(block)

    if len(article_blocks) < 2 or len(abstract_blocks) < 2:
        return None

    body_blocks = flattened[body_index:]
    body_footer_blocks = [
        block for block in body_blocks
        if _looks_like_publication_footer_block(block, page_height)
    ]
    body_content_blocks = [
        block for block in body_blocks
        if block not in body_footer_blocks
    ]
    left_body_blocks = [
        block for block in body_content_blocks
        if _layout_lane(block, layout_profile) == "left"
    ]
    right_body_blocks = [
        block for block in body_content_blocks
        if _layout_lane(block, layout_profile) == "right"
    ]
    other_body_blocks = [
        block for block in body_content_blocks
        if _layout_lane(block, layout_profile) not in {"left", "right"}
    ]

    return (
        prefix
        + article_blocks
        + abstract_blocks
        + left_body_blocks
        + right_body_blocks
        + other_body_blocks
        + body_footer_blocks
    )


def build_reading_order_diagnostics(
    text_blocks: list[dict[str, Any]],
    layout_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode = str((layout_profile or {}).get("mode", "single_column") or "single_column")
    confidence = float((layout_profile or {}).get("confidence", 0.0) or 0.0)
    diagnostics: dict[str, Any] = {
        "layout_mode": mode,
        "layout_confidence": round(confidence, 3),
        "strategy": "visual_rows",
        "zone_count": 0,
        "column_zone_count": 0,
        "full_width_zone_count": 0,
        "left_block_count": 0,
        "right_block_count": 0,
        "full_width_block_count": 0,
        "ambiguous_lane_block_count": 0,
        "review_required": False,
        "signals": [],
    }
    if not text_blocks:
        return diagnostics

    lane_counts = {"left": 0, "right": 0, "full_width": 0, "ambiguous": 0}
    for block in text_blocks:
        lane = _layout_lane(block, layout_profile)
        if lane in {"left", "right", "full_width"}:
            lane_counts[lane] += 1
        else:
            lane_counts["ambiguous"] += 1
    diagnostics["left_block_count"] = lane_counts["left"]
    diagnostics["right_block_count"] = lane_counts["right"]
    diagnostics["full_width_block_count"] = lane_counts["full_width"]
    diagnostics["ambiguous_lane_block_count"] = lane_counts["ambiguous"]

    has_column_lanes = lane_counts["left"] > 0 or lane_counts["right"] > 0
    if not has_column_lanes:
        return diagnostics

    zones = _reading_order_zones(text_blocks, layout_profile)
    column_zone_count = sum(1 for zone in zones if zone["mode"] == "columns")
    full_width_zone_count = sum(1 for zone in zones if zone["mode"] == "full_width")
    diagnostics["strategy"] = "zone_columns_left_then_right"
    diagnostics["zone_count"] = len(zones)
    diagnostics["column_zone_count"] = column_zone_count
    diagnostics["full_width_zone_count"] = full_width_zone_count
    diagnostics["signals"] = [
        "column_lanes_detected",
        "full_width_zone_boundaries" if full_width_zone_count else "column_only_page",
    ]

    if min(lane_counts["left"], lane_counts["right"]) == 0:
        diagnostics["review_required"] = True
        diagnostics["signals"].append("unbalanced_column_lanes")
    if mode in {"two_column", "mixed"} and confidence < 0.6:
        diagnostics["review_required"] = True
        diagnostics["signals"].append("low_layout_confidence")
    if lane_counts["ambiguous"] > 0:
        diagnostics["review_required"] = True
        diagnostics["signals"].append("ambiguous_layout_lanes")
    return diagnostics


def order_text_blocks_for_reading(
    text_blocks: list[dict[str, Any]],
    layout_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not text_blocks:
        return []

    ordered_rows = _group_text_blocks_by_visual_rows(text_blocks)
    if not ordered_rows:
        return []

    has_column_lanes = any(
        _layout_lane(block, layout_profile) in {"left", "right"}
        for block in text_blocks
    )
    if not has_column_lanes:
        ordered: list[dict[str, Any]] = []
        for row in ordered_rows:
            ordered.extend(row)
        return ordered

    zones = _reading_order_zones(text_blocks, layout_profile)

    ordered: list[dict[str, Any]] = []
    for zone in zones:
        if zone["mode"] == "full_width":
            for row in zone["rows"]:
                ordered.extend(sorted(row, key=lambda item: (item["bbox"][0], _block_center_y(item))))
            continue

        publication_front_matter_order = _order_publication_front_matter_zone(
            zone["rows"],
            layout_profile,
        )
        if publication_front_matter_order is not None:
            ordered.extend(publication_front_matter_order)
            continue

        left_blocks: list[dict[str, Any]] = []
        right_blocks: list[dict[str, Any]] = []
        full_width_blocks: list[dict[str, Any]] = []
        for row in zone["rows"]:
            left_blocks.extend(
                block for block in row
                if _layout_lane(block, layout_profile) == "left"
            )
            right_blocks.extend(
                block for block in row
                if _layout_lane(block, layout_profile) == "right"
            )
            full_width_blocks.extend(
                block for block in row
                if _layout_lane(block, layout_profile) not in {"left", "right"}
            )

        ordered.extend(left_blocks)
        ordered.extend(right_blocks)
        ordered.extend(full_width_blocks)
    return ordered


def _word_identity(word: _Word) -> tuple[str, float, float, float, float]:
    return (
        _compact_text(word.text),
        round(word.x0, 2),
        round(word.y0, 2),
        round(word.x1, 2),
        round(word.y1, 2),
    )


def _resolve_union_word_text(
    left: dict[str, Any],
    right: dict[str, Any],
    page_words: list[_Word],
) -> tuple[str, bool]:
    left_bbox = tuple(left["bbox"])
    right_bbox = tuple(right["bbox"])
    left_words = _words_in_bbox(page_words, left_bbox, margin=0.8)
    right_words = _words_in_bbox(page_words, right_bbox, margin=0.8)
    union_bbox = _bbox_union([left_bbox, right_bbox])
    union_words = _words_in_bbox(page_words, union_bbox, margin=0.8)

    union_text = _words_to_text(union_words)
    if not union_text:
        return "", False

    union_compact = _compact_text(union_text)
    expected_compact = _compact_text(f"{left.get('text', '')}{right.get('text', '')}")
    if not union_compact or union_compact != expected_compact:
        return "", False

    left_word_ids = {_word_identity(word) for word in left_words}
    right_word_ids = {_word_identity(word) for word in right_words}
    shared_word_bridge = bool(left_word_ids & right_word_ids)
    return union_text, shared_word_bridge


def _merge_visual_line_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    merged_text: str,
) -> dict[str, Any]:
    font_sizes = [
        float(left.get("font_size", 0.0) or 0.0),
        float(right.get("font_size", 0.0) or 0.0),
    ]
    font_sizes = [value for value in font_sizes if value > 0]
    merged = {
        "block_type": "text",
        "page": left.get("page", right.get("page")),
        "bbox": _bbox_to_list(_bbox_union([tuple(left["bbox"]), tuple(right["bbox"])])),
        "text": _clean_text(merged_text),
        "font_size": statistics.median(font_sizes) if font_sizes else 0.0,
        "spans": list(left.get("spans", []) or []) + list(right.get("spans", []) or []),
        "source_block_indices": sorted(
            set(_normalized_source_block_indices(left) + _normalized_source_block_indices(right))
        ),
        "source": left.get("source", right.get("source", "text-layer")),
        "visual_line_reconstructed": True,
        "reconstruction_evidence": "words_continuity",
    }
    layout_lane = left.get("layout_lane") or right.get("layout_lane")
    if layout_lane:
        merged["layout_lane"] = layout_lane
    layout_mode = left.get("layout_mode") or right.get("layout_mode")
    if layout_mode:
        merged["layout_mode"] = layout_mode
    layout_confidence = left.get("layout_confidence", right.get("layout_confidence"))
    if layout_confidence is not None:
        merged["layout_confidence"] = float(layout_confidence or 0.0)
    return merged


def _should_reconstruct_visual_line_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    page_words: list[_Word],
    page_width: float,
    row_tolerance: float,
) -> tuple[bool, str]:
    left_bbox = tuple(left["bbox"])
    right_bbox = tuple(right["bbox"])
    same_source_lineage = _shares_source_block_lineage(left, right)
    if right_bbox[0] < left_bbox[0]:
        return False, ""

    vertical_overlap = _vertical_overlap_ratio(left_bbox, right_bbox)
    if abs(_block_center_y(left) - _block_center_y(right)) > row_tolerance and vertical_overlap < 0.6:
        return False, ""

    gap = right_bbox[0] - left_bbox[2]
    max_gap = max(24.0, page_width * 0.045, min(_block_height(left), _block_height(right)) * 2.4)
    min_gap = -8.0 if same_source_lineage else -2.5
    if gap < min_gap or gap > max_gap:
        return False, ""

    left_text = _clean_text(left.get("text", ""))
    right_text = _clean_text(right.get("text", ""))
    if not left_text or not right_text:
        return False, ""
    if (not same_source_lineage) and (
        _text_ends_with_break_marker(left_text) or _text_starts_with_break_marker(right_text)
    ):
        return False, ""

    union_word_text, shared_word_bridge = _resolve_union_word_text(left, right, page_words)
    if not union_word_text:
        if same_source_lineage and vertical_overlap >= 0.6:
            return True, _join_text_fragments(left_text, right_text)
        return False, ""

    if shared_word_bridge:
        return True, union_word_text

    if same_source_lineage and vertical_overlap >= 0.6:
        return True, union_word_text

    if gap <= max(1.0, min(_block_height(left), _block_height(right)) * 0.08):
        return True, union_word_text
    return False, ""


def reconstruct_visual_text_lines(
    text_blocks: list[dict[str, Any]],
    page_words: list[_Word],
    page_number: int,
    page_width: float,
    layout_profile: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if not text_blocks or not page_words:
        return text_blocks, 0

    row_tolerance = _visual_row_tolerance(text_blocks)
    reconstructed: list[dict[str, Any]] = []
    merge_count = 0

    for row in _group_text_blocks_by_visual_rows(text_blocks, row_tolerance):
        if not row:
            continue
        row_blocks: list[dict[str, Any]] = []
        for block in row:
            normalized = dict(block)
            if "source_block_indices" not in normalized:
                normalized["source_block_indices"] = _normalized_source_block_indices(block)
            row_blocks.append(normalized)

        compacted: list[dict[str, Any]] = []
        idx = 0
        while idx < len(row_blocks):
            current = row_blocks[idx]
            while idx + 1 < len(row_blocks):
                if not _same_layout_lane(current, row_blocks[idx + 1], layout_profile):
                    break
                should_merge, merged_text = _should_reconstruct_visual_line_pair(
                    current,
                    row_blocks[idx + 1],
                    page_words,
                    page_width,
                    row_tolerance,
                )
                if not should_merge:
                    break
                current = _merge_visual_line_pair(current, row_blocks[idx + 1], merged_text)
                merge_count += 1
                idx += 1
            compacted.append(current)
            idx += 1

        reconstructed.extend(compacted)

    for index, block in enumerate(reconstructed, start=1):
        block["block_id"] = f"txt_p{page_number}_{index:03d}"
    return reconstructed, merge_count


def _should_merge_text_blocks(
    left: dict[str, Any],
    right: dict[str, Any],
    page_width: float,
    row_tolerance: float,
    page_height: float | None = None,
    page_words: list[_Word] | None = None,
    layout_profile: dict[str, Any] | None = None,
) -> bool:
    if not _same_layout_lane(left, right, layout_profile):
        return False

    left_text = _clean_text(left.get("text", ""))
    right_text = _clean_text(right.get("text", ""))
    left_is_marker = _looks_like_inline_equation_marker_text(left_text)
    right_is_marker = _looks_like_inline_equation_marker_text(right_text)
    if left_is_marker != right_is_marker:
        return False
    inline_math_row_merge = _can_merge_inline_math_row_fragments(left, right, page_width)
    if not _shares_source_block_lineage(left, right):
        left_mathish = _looks_like_short_mathish_fragment(left_text)
        right_mathish = _looks_like_short_mathish_fragment(right_text)
        if left_mathish != right_mathish and not inline_math_row_merge:
            return False

    left_bbox = tuple(left["bbox"])
    right_bbox = tuple(right["bbox"])
    left_height = max(1.0, left_bbox[3] - left_bbox[1])
    right_height = max(1.0, right_bbox[3] - right_bbox[1])
    left_center = (left_bbox[1] + left_bbox[3]) / 2
    right_center = (right_bbox[1] + right_bbox[3]) / 2
    vertical_overlap = _vertical_overlap_ratio(left_bbox, right_bbox)
    horizontal_overlap = _horizontal_overlap_ratio(left_bbox, right_bbox)
    if abs(left_center - right_center) > row_tolerance and vertical_overlap < 0.6 and not inline_math_row_merge:
        return False

    gap = right_bbox[0] - left_bbox[2]
    if gap < -2.5 and not (vertical_overlap >= 0.6 and horizontal_overlap >= 0.18):
        return False
    max_gap = max(24.0, page_width * 0.045, min(left_height, right_height) * 2.4)
    if gap > max_gap:
        return False

    if not left_text or not right_text:
        return False
    if _text_ends_with_break_marker(left_text):
        return False
    if _text_starts_with_break_marker(right_text):
        return False

    if page_height is not None:
        left_is_footer_artifact = _is_footer_artifact_block(left, page_height)
        right_is_footer_artifact = _is_footer_artifact_block(right, page_height)
        if left_is_footer_artifact != right_is_footer_artifact:
            has_same_line_continuity = bool(page_words) and _should_reconstruct_visual_line_pair(
                left,
                right,
                page_words,
                page_width,
                row_tolerance,
            )[0]
            if not has_same_line_continuity:
                return False

    left_font = float(left.get("font_size", 0.0) or 0.0)
    right_font = float(right.get("font_size", 0.0) or 0.0)
    if left_font > 0 and right_font > 0:
        ratio = max(left_font, right_font) / max(0.1, min(left_font, right_font))
        if ratio > 1.45 and not (vertical_overlap >= 0.6 and horizontal_overlap >= 0.18) and not inline_math_row_merge:
            return False

    if inline_math_row_merge:
        return True
    if _has_cjk(left_text) or _has_cjk(right_text):
        return True
    return len(left_text) <= 64 and len(right_text) <= 64


def _merge_semantic_text_blocks(
    text_blocks: list[dict[str, Any]],
    page_number: int,
    page_width: float,
    page_height: float | None = None,
    page_words: list[_Word] | None = None,
    layout_profile: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if not text_blocks:
        return [], 0
    ordered = order_text_blocks_for_reading(text_blocks, layout_profile)
    row_tolerance = _visual_row_tolerance(ordered)

    merged: list[dict[str, Any]] = []
    merge_count = 0
    for block in ordered:
        normalized = {
            "block_type": "text",
            "page": page_number,
            "bbox": [float(item) for item in block["bbox"]],
            "text": _clean_text(block.get("text", "")),
            "font_size": float(block.get("font_size", 0.0) or 0.0),
            "spans": list(block.get("spans", []) or []),
            "source_block_indices": _normalized_source_block_indices(block),
            "source": block.get("source", "text-layer"),
        }
        layout_lane = block.get("layout_lane")
        if layout_lane:
            normalized["layout_lane"] = layout_lane
        layout_mode = block.get("layout_mode")
        if layout_mode:
            normalized["layout_mode"] = layout_mode
        if block.get("layout_confidence") is not None:
            normalized["layout_confidence"] = float(block.get("layout_confidence", 0.0) or 0.0)
        if block.get("visual_line_reconstructed"):
            normalized["visual_line_reconstructed"] = True
            normalized["reconstruction_evidence"] = block.get("reconstruction_evidence", "words_continuity")
        if not normalized["text"]:
            continue
        if not merged:
            merged.append(normalized)
            continue
        previous = merged[-1]
        if _should_merge_text_blocks(
            previous,
            normalized,
            page_width,
            row_tolerance,
            page_height=page_height,
            page_words=page_words,
            layout_profile=layout_profile,
        ):
            previous_bbox = tuple(previous["bbox"])
            normalized_bbox = tuple(normalized["bbox"])
            previous["bbox"] = _bbox_to_list(_bbox_union([previous_bbox, normalized_bbox]))
            previous["text"] = _join_text_fragments(previous["text"], normalized["text"])
            if normalized["source"] == "image-text-recovery":
                previous["source"] = "semantic-merged-image-text"
            previous["source_block_indices"].extend(normalized["source_block_indices"])
            previous["spans"].extend(list(normalized.get("spans", []) or []))
            font_sizes = [float(previous.get("font_size", 0.0) or 0.0), float(normalized.get("font_size", 0.0) or 0.0)]
            font_sizes = [item for item in font_sizes if item > 0]
            previous["font_size"] = statistics.median(font_sizes) if font_sizes else 0.0
            merge_count += 1
            continue
        merged.append(normalized)

    deduped, dedup_count = _deduplicate_text_blocks(merged)
    deduped, near_dup_count = _merge_near_duplicate_text_blocks(deduped)
    dedup_count += near_dup_count
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
    has_column_lanes = any(_layout_lane(block) in {"left", "right"} for block in text_blocks)
    iterable = text_blocks if has_column_lanes else sorted(text_blocks, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for block in iterable:
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


def _text_blocks_are_close(
    a_bbox: tuple[float, float, float, float],
    b_bbox: tuple[float, float, float, float],
) -> bool:
    horizontal_overlap = _horizontal_overlap_ratio(a_bbox, b_bbox)
    left_gap = max(0.0, b_bbox[0] - a_bbox[2])
    right_gap = max(0.0, a_bbox[0] - b_bbox[2])
    touching = left_gap <= 4.0 or right_gap <= 4.0
    return horizontal_overlap >= 0.05 or touching


def _prefer_compact_bbox(
    existing_bbox: tuple[float, float, float, float],
    candidate_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    existing_height = existing_bbox[3] - existing_bbox[1]
    candidate_height = candidate_bbox[3] - candidate_bbox[1]
    if candidate_height < existing_height * 0.85:
        return candidate_bbox
    if existing_height < candidate_height * 0.85:
        return existing_bbox
    return existing_bbox


def _merge_near_duplicate_text_blocks(
    text_blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    if not text_blocks:
        return [], 0
    merged: list[dict[str, Any]] = []
    duplicates = 0
    for block in text_blocks:
        normalized = _compact_text(block.get("text", ""))
        if not normalized:
            merged.append(dict(block))
            continue
        block_bbox = tuple(block["bbox"])
        candidate = next(
            (
                existing
                for existing in merged
                if _compact_text(existing["text"]) == normalized
                and _vertical_overlap_ratio(tuple(existing["bbox"]), block_bbox) >= 0.6
                and _text_blocks_are_close(tuple(existing["bbox"]), block_bbox)
            ),
            None,
        )
        if candidate:
            candidate_bbox = tuple(candidate["bbox"])
            preferred_bbox = _prefer_compact_bbox(candidate_bbox, block_bbox)
            candidate["bbox"] = _bbox_to_list(preferred_bbox)
            if len(block.get("text", "")) > len(candidate.get("text", "")):
                candidate["text"] = block["text"]
            duplicates += 1
            continue
        merged.append(dict(block))
    return merged, duplicates


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


def _is_small_footer_block(block: dict[str, Any], page_height: float) -> bool:
    bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    height = max(0.0, bbox[3] - bbox[1])
    return height <= max(12.0, page_height * 0.035)


def _top_margin_limit(page_height: float) -> float:
    return max(72.0, page_height * 0.11)


def _bottom_margin_limit(page_height: float) -> float:
    return page_height - max(72.0, page_height * 0.11)


def _is_top_margin_block(block: dict[str, Any], page_height: float) -> bool:
    bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    return bbox[3] <= _top_margin_limit(page_height)


def _is_bottom_margin_block(block: dict[str, Any], page_height: float) -> bool:
    bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    return bbox[1] >= _bottom_margin_limit(page_height)


def _looks_like_heading_prefix(text: str) -> bool:
    return bool(_HEADING_PREFIX_RE.fullmatch(_clean_text(text)))


def _starts_with_numbered_heading(text: str) -> bool:
    cleaned = _clean_text(text)
    return bool(_NUMBERED_HEADING_RE.match(cleaned) or _DOTTED_NUMBERED_HEADING_RE.match(cleaned))


def _starts_with_structural_numbered_heading(text: str) -> bool:
    return bool(_STRUCTURAL_NUMBERED_HEADING_RE.match(_clean_text(text)))


def _blocks_share_visual_row(
    left: dict[str, Any],
    right: dict[str, Any],
    row_tolerance: float,
) -> bool:
    left_bbox = tuple(left.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    right_bbox = tuple(right.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    vertical_overlap = _vertical_overlap_ratio(left_bbox, right_bbox)
    center_gap = abs(_block_center_y(left) - _block_center_y(right))
    return center_gap <= row_tolerance or vertical_overlap >= 0.6


def _has_heading_prefix_peer(
    block: dict[str, Any],
    page_height: float,
    peer_blocks: list[dict[str, Any]] | None = None,
) -> bool:
    if not peer_blocks or not _is_top_margin_block(block, page_height):
        return False
    row_tolerance = _visual_row_tolerance(peer_blocks)
    block_bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))

    for peer in peer_blocks:
        if peer is block:
            continue
        if not _is_top_margin_block(peer, page_height):
            continue
        if not _looks_like_heading_prefix(str(peer.get("text", ""))):
            continue
        if not _blocks_share_visual_row(block, peer, row_tolerance):
            continue
        peer_bbox = tuple(peer.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        gap = block_bbox[0] - peer_bbox[2]
        reverse_gap = peer_bbox[0] - block_bbox[2]
        if gap <= 36.0 or reverse_gap <= 36.0:
            return True
    return False


def _has_same_row_running_header_peer(
    block: dict[str, Any],
    page_height: float,
    peer_blocks: list[dict[str, Any]] | None = None,
) -> bool:
    if not peer_blocks or not _is_top_margin_block(block, page_height):
        return False
    row_tolerance = _visual_row_tolerance(peer_blocks)
    for peer in peer_blocks:
        if peer is block:
            continue
        if not _is_top_margin_block(peer, page_height):
            continue
        if not _blocks_share_visual_row(block, peer, row_tolerance):
            continue
        peer_text = _clean_text(str(peer.get("text", "")))
        peer_compact = _compact_text(peer_text)
        if not peer_compact or _looks_like_page_number(peer_text):
            continue
        if _looks_like_heading_prefix(peer_text):
            continue
        if _starts_with_structural_numbered_heading(peer_text):
            continue
        if len(peer_compact) >= 8:
            return True
    return False


def _margin_text_role(
    block: dict[str, Any],
    page_height: float,
    peer_blocks: list[dict[str, Any]] | None = None,
) -> str:
    text = _clean_text(block.get("text", ""))
    if page_height <= 0 or not text:
        return "none"
    if not _is_header_footer_candidate(block, page_height):
        return "none"

    if _is_bottom_margin_block(block, page_height):
        if _looks_like_page_number(text):
            return "footer_artifact"
        if _is_small_footer_block(block, page_height) and len(_compact_text(text)) <= 3:
            return "footer_artifact"
        if text.isdigit():
            bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
            height = max(0.0, bbox[3] - bbox[1])
            if height <= max(16.0, page_height * 0.04):
                return "footer_artifact"
        return "running_header"

    if _is_top_margin_block(block, page_height):
        if _starts_with_numbered_heading(text):
            return "section_heading"
        if _looks_like_heading_prefix(text):
            return "heading_prefix"
        if _has_heading_prefix_peer(block, page_height, peer_blocks):
            return "section_heading"
        if _looks_like_page_number(text) and _has_same_row_running_header_peer(block, page_height, peer_blocks):
            return "footer_artifact"
        return "running_header"

    return "none"


def _is_footer_artifact_block(
    block: dict[str, Any],
    page_height: float,
) -> bool:
    return _margin_text_role(block, page_height) == "footer_artifact"


def _looks_like_page_number(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    return bool(
        re.fullmatch(r"(?:菴\s*)?\d{1,4}(?:\s*/\s*\d{1,4})?(?:\s*珜)?", cleaned)
        or re.fullmatch(r"\d{1,4}\s*[-每〞]\s*\d{1,4}", cleaned)
        or _ROMAN_PAGE_NUMBER_RE.fullmatch(cleaned)
        or (len(cleaned) == 1 and cleaned.isalpha())
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
            role = _margin_text_role(block, page_height, payload["text_blocks"])
            if role not in {"running_header", "footer_artifact"}:
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
                role = _margin_text_role(block, page_height, payload["text_blocks"])
                signature = _header_footer_signature(text)
                repeated_signature = len(signature) >= 3 and signature_counts.get(signature, 0) >= 2
                if role == "footer_artifact":
                    removed += 1
                    continue
                if role == "running_header" and repeated_signature:
                    removed += 1
                    continue
                if role == "running_header" and _is_small_footer_block(block, page_height) and len(text) <= 2 and text.isalpha():
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
    table_bboxes = _table_text_suppression_bboxes(text_blocks, page_tables)
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


def annotate_table_presentation_bboxes(
    text_blocks: list[dict[str, Any]],
    page_tables: list[dict[str, Any]],
) -> int:
    if not text_blocks or not page_tables:
        return 0
    annotated = 0
    for table in page_tables:
        table_bbox = _valid_bbox_tuple(table.get("bbox"))
        if table_bbox is None:
            continue
        owned_bboxes = _table_owned_text_bboxes(text_blocks, table, table_bbox)
        if not owned_bboxes:
            continue
        presentation_bbox = _bbox_union([table_bbox, *owned_bboxes])
        table["presentation_bbox"] = _bbox_to_list(presentation_bbox)
        table["owned_text_bboxes"] = [_bbox_to_list(bbox) for bbox in owned_bboxes]
        annotated += 1
    return annotated


def _table_text_suppression_bboxes(
    text_blocks: list[dict[str, Any]],
    page_tables: list[dict[str, Any]],
) -> list[tuple[float, float, float, float]]:
    bboxes: list[tuple[float, float, float, float]] = []
    for table in page_tables:
        bbox = _valid_bbox_tuple(table.get("bbox"))
        if bbox is None:
            continue
        owned_bboxes = [bbox]
        for key in ("title_bbox", "caption_bbox", "header_bbox", "presentation_bbox"):
            extra_bbox = _valid_bbox_tuple(table.get(key))
            if extra_bbox is not None:
                owned_bboxes.append(extra_bbox)

        title_block = table.get("title_block")
        if isinstance(title_block, dict):
            title_bbox = _valid_bbox_tuple(title_block.get("bbox"))
            if title_bbox is not None:
                owned_bboxes.append(title_bbox)

        owned_bboxes.extend(_table_owned_text_bboxes(text_blocks, table, bbox))
        presentation_bbox = _bbox_union(owned_bboxes)
        bboxes.append(tuple(float(item) for item in presentation_bbox))
    return bboxes


def _table_owned_text_bboxes(
    text_blocks: list[dict[str, Any]],
    table: dict[str, Any],
    table_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    owned_bboxes: list[tuple[float, float, float, float]] = []
    for key in ("title_bbox", "caption_bbox", "header_bbox"):
        extra_bbox = _valid_bbox_tuple(table.get(key))
        if extra_bbox is not None:
            owned_bboxes.append(extra_bbox)
    title_block = table.get("title_block")
    if isinstance(title_block, dict):
        title_bbox = _valid_bbox_tuple(title_block.get("bbox"))
        if title_bbox is not None:
            owned_bboxes.append(title_bbox)
    owned_bboxes.extend(_find_table_caption_and_header_text_bboxes(text_blocks, table, table_bbox))
    deduped: list[tuple[float, float, float, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for bbox in owned_bboxes:
        key = tuple(round(float(item), 2) for item in bbox)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(bbox)
    return deduped


def _valid_bbox_tuple(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def _find_table_caption_and_header_text_bboxes(
    text_blocks: list[dict[str, Any]],
    table: dict[str, Any],
    table_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    title_text = _clean_text(str(table.get("title", "")))
    header_texts = [
        _clean_text(str(cell.get("text", "")))
        for cell in table.get("header", []) or []
        if isinstance(cell, dict) and _clean_text(str(cell.get("text", "")))
    ]
    header_norms = {_compact_text(text) for text in header_texts if _compact_text(text)}
    if not title_text and not header_norms:
        return []

    table_width = max(1.0, table_bbox[2] - table_bbox[0])
    max_gap_above = max(34.0, min(96.0, table_width * 0.18))
    candidates: list[tuple[float, float, float, float]] = []
    for block in text_blocks:
        block_bbox = _valid_bbox_tuple(block.get("bbox"))
        if block_bbox is None:
            continue
        if block_bbox[3] > table_bbox[1] + 10.0:
            continue
        gap = table_bbox[1] - block_bbox[3]
        if gap < -10.0 or gap > max_gap_above:
            continue
        if _horizontal_overlap_ratio(block_bbox, table_bbox) < 0.08:
            continue

        text = _clean_text(str(block.get("text", "")))
        text_norm = _compact_text(text)
        if not text_norm:
            continue
        if _text_matches_table_title_fragment(text, title_text) or text_norm in header_norms:
            candidates.append(block_bbox)

    return candidates


def _text_matches_table_title_fragment(text: str, title_text: str) -> bool:
    if not text or not title_text:
        return False
    text_norm = _compact_text(text)
    title_norm = _compact_text(title_text)
    if not text_norm or not title_norm:
        return False
    if text_norm == title_norm:
        return True
    if text_norm.startswith("table") and text_norm in title_norm:
        return True
    if len(text_norm) >= 12 and text_norm in title_norm:
        return True
    return False

