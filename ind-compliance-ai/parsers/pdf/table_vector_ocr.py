from __future__ import annotations

from dataclasses import dataclass
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

from .shared import _Word, _bbox_union, _clean_text
from .ocr_policy import PageOcrContext, should_scan_embedded_image_table_regions
from .table_modules.raw_objects import (
    RawCell,
    RawDrawing,
    RawRow,
    RawSpan,
    RawTableEvidence,
    RawWord,
    _extract_drawings_from_page,
)

_VECTOR_OCR_SHORT_TABLE_EDGE_THRESHOLD = 90.0
_VECTOR_OCR_SHORT_TABLE_SCALE = 4.0
_VECTOR_OCR_STANDARD_TABLE_SCALE = 3.5


@dataclass(slots=True)
class _OcrEntry:
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]

    @property
    def x0(self) -> float:
        return float(self.bbox[0])

    @property
    def y0(self) -> float:
        return float(self.bbox[1])

    @property
    def x1(self) -> float:
        return float(self.bbox[2])

    @property
    def y1(self) -> float:
        return float(self.bbox[3])

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


def _compute_vector_ocr_render_scale(clip_width: float, clip_height: float) -> float:
    if clip_width <= 0 or clip_height <= 0:
        return _VECTOR_OCR_SHORT_TABLE_SCALE

    short_edge = min(clip_width, clip_height)
    # Short, wide borderless tables lose token separation quickly below 4.0x.
    if short_edge < _VECTOR_OCR_SHORT_TABLE_EDGE_THRESHOLD:
        return _VECTOR_OCR_SHORT_TABLE_SCALE
    return _VECTOR_OCR_STANDARD_TABLE_SCALE


def extract_vector_ocr_table_candidates(
    page: Any,
    page_number: int,
    page_height: float,
    page_width: float,
    title_blocks: list[dict[str, Any]],
    page_drawings: list[dict[str, Any]],
    page_words: list[_Word],
    occupied_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> list[RawTableEvidence]:
    if (
        RapidOCR is None
        or np is None
        or pymupdf is None
        or not title_blocks
        or not page_drawings
    ):
        return []

    results: list[RawTableEvidence] = []
    claimed_bboxes = list(occupied_bboxes or [])
    for title_block in sorted(title_blocks, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        table_region = _find_vector_table_region_below_title(
            title_block=title_block,
            page_drawings=page_drawings,
            page_words=page_words,
            page_width=page_width,
            page_height=page_height,
            occupied_bboxes=claimed_bboxes,
        )
        if table_region is None:
            continue
        raw_evidence = _build_vector_ocr_raw_evidence(
            page=page,
            table_bbox=table_region,
            page_number=page_number,
            page_height=page_height,
            page_width=page_width,
        )
        if raw_evidence is None:
            continue
        results.append(raw_evidence)
        claimed_bboxes.append(raw_evidence.bbox)
    return results


def extract_embedded_image_ocr_table_candidates(
    page: Any,
    page_number: int,
    page_height: float,
    page_width: float,
    image_blocks: list[dict[str, Any]],
    page_words: list[_Word],
    occupied_bboxes: list[tuple[float, float, float, float]] | None = None,
    ocr_context: PageOcrContext | None = None,
) -> list[RawTableEvidence]:
    """Recover table candidates from embedded raster images.

    This is a discovery layer only. It admits image regions that look like
    standalone tabular data after OCR and then returns normal RawTableEvidence
    so normalization, AST projection, ownership arbitration, and export remain
    shared with all other table sources.
    """
    decision = should_scan_embedded_image_table_regions(ocr_context)
    if not decision.enabled:
        return []
    if RapidOCR is None or np is None or pymupdf is None or not image_blocks:
        return []

    results: list[RawTableEvidence] = []
    claimed_bboxes = list(occupied_bboxes or [])
    for image_block in sorted(image_blocks, key=lambda item: tuple(item.get("bbox", (0.0, 0.0, 0.0, 0.0))[:2])):
        bbox = _image_block_bbox(image_block)
        if bbox is None:
            continue
        if _image_block_is_chart_owned_region(image_block):
            continue
        if not _image_region_is_table_candidate(
            bbox=bbox,
            image_block=image_block,
            page_words=page_words,
            page_width=page_width,
            page_height=page_height,
            occupied_bboxes=claimed_bboxes,
        ):
            continue
        raw_evidence = _build_vector_ocr_raw_evidence(
            page=page,
            table_bbox=bbox,
            page_number=page_number,
            page_height=page_height,
            page_width=page_width,
            prefer_stable_row_anchors=True,
        )
        if raw_evidence is None:
            continue
        if not _ocr_raw_evidence_has_table_lattice(raw_evidence):
            continue
        raw_evidence.source = "embedded_image_ocr"
        results.append(raw_evidence)
        claimed_bboxes.append(raw_evidence.bbox)
    return results


@lru_cache(maxsize=1)
def _get_vector_ocr_engine() -> Any:
    if RapidOCR is None:
        return None
    try:
        return RapidOCR()
    except Exception:
        return None


def _find_vector_table_region_below_title(
    title_block: dict[str, Any],
    page_drawings: list[dict[str, Any]],
    page_words: list[_Word],
    page_width: float,
    page_height: float,
    occupied_bboxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    title_bbox = tuple(float(item) for item in title_block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(title_bbox) != 4:
        return None

    scan_bottom = min(page_height - 6.0, title_bbox[3] + min(380.0, page_height * 0.48))
    horizontal_lines = [
        bbox
        for bbox in (
            _drawing_bbox(draw)
            for draw in page_drawings
        )
        if bbox
        and bbox[1] >= title_bbox[3] - 4.0
        and bbox[1] <= scan_bottom
        and _is_wide_horizontal_rule_bbox(bbox, page_width)
        and _horizontal_overlap_ratio(bbox, title_bbox) >= 0.25
    ]
    if len(horizontal_lines) < 2:
        return None

    horizontal_lines.sort(key=lambda bbox: (bbox[1], bbox[0]))
    seed = horizontal_lines[0]
    aligned_lines = [
        bbox
        for bbox in horizontal_lines
        if abs(bbox[0] - seed[0]) <= 24.0 and abs(bbox[2] - seed[2]) <= 24.0
    ]
    if len(aligned_lines) < 2:
        return None

    table_bbox = (
        min(bbox[0] for bbox in aligned_lines),
        min(bbox[1] for bbox in aligned_lines),
        max(bbox[2] for bbox in aligned_lines),
        max(bbox[3] for bbox in aligned_lines),
    )
    if _bbox_height(table_bbox) < 48.0 or _bbox_width(table_bbox) < page_width * 0.35:
        return None
    if any(_bbox_overlap_ratio(table_bbox, occupied) >= 0.35 for occupied in occupied_bboxes):
        return None

    glyph_like_drawings = [
        bbox
        for bbox in (
            _drawing_bbox(draw)
            for draw in page_drawings
        )
        if bbox and _bbox_overlap_ratio(bbox, table_bbox) >= 0.7 and _is_dense_dark_glyph_rect(bbox)
    ]
    if len(glyph_like_drawings) < 25:
        return None

    text_layer_words = [
        word
        for word in page_words
        if _bbox_contains_word(table_bbox, word)
    ]
    if len(text_layer_words) > 6:
        return None

    return table_bbox


def _image_block_bbox(image_block: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = image_block.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        values = tuple(float(value) for value in bbox[:4])
    except Exception:
        return None
    if _bbox_width(values) <= 0.0 or _bbox_height(values) <= 0.0:
        return None
    return values


def _image_block_is_chart_owned_region(image_block: dict[str, Any]) -> bool:
    """Return true when the image region is already owned by chart/figure semantics.

    Embedded-image OCR can turn bar charts and line charts into plausible
    row/column grids. That OCR is still figure evidence, not a competing table
    owner. Parent/child figure-with-table extraction should be modeled
    explicitly before this gate is relaxed.
    """
    semantics = image_block.get("figure_semantics")
    if isinstance(semantics, dict) and str(semantics.get("semantic_type") or "") == "chart_figure":
        return True
    if str(image_block.get("image_kind_guess") or "") == "chart_figure":
        return True
    return False


def _image_region_is_table_candidate(
    *,
    bbox: tuple[float, float, float, float],
    image_block: dict[str, Any],
    page_words: list[_Word],
    page_width: float,
    page_height: float,
    occupied_bboxes: list[tuple[float, float, float, float]],
) -> bool:
    if _bbox_width(bbox) < page_width * 0.30 or _bbox_height(bbox) < max(42.0, page_height * 0.045):
        return False
    if _bbox_height(bbox) > page_height * 0.82:
        return False
    if any(_bbox_overlap_ratio(bbox, occupied) >= 0.35 for occupied in occupied_bboxes):
        return False
    text_layer_words = [word for word in page_words if _bbox_contains_word(bbox, word)]
    if len(text_layer_words) > 8:
        return False
    caption_text = str(image_block.get("caption_text") or "")
    if re.search(r"\b(?:figure|fig\.?)\s*\d+", caption_text, re.IGNORECASE):
        return False
    if not _raster_clip_has_table_like_structure(bbox=bbox, page_width=page_width, page_height=page_height, image_block=image_block):
        return False
    return True


def _raster_clip_has_table_like_structure(
    *,
    bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    image_block: dict[str, Any],
) -> bool:
    """Cheap pre-OCR gate for embedded image table candidates.

    This prevents the OCR table path from scanning ordinary photographs,
    diagrams, and decorative figures. It intentionally uses only geometry and
    image-block metadata so expensive OCR remains a last-mile confirmation.
    """
    width = _bbox_width(bbox)
    height = _bbox_height(bbox)
    if width <= 0.0 or height <= 0.0:
        return False
    aspect = width / max(1.0, height)
    if aspect < 1.15 and height > page_height * 0.18:
        return False
    if width < page_width * 0.55 and height < page_height * 0.11:
        return False
    pixel_width = _safe_float(image_block.get("width"))
    pixel_height = _safe_float(image_block.get("height"))
    if pixel_width and pixel_height:
        pixel_aspect = pixel_width / max(1.0, pixel_height)
        if pixel_aspect < 1.10 and aspect < 1.25:
            return False
    return True


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if result <= 0.0:
        return None
    return result


def _ocr_raw_evidence_has_table_lattice(raw_evidence: RawTableEvidence) -> bool:
    grid = raw_evidence.raw_data or []
    if len(grid) < 3 or raw_evidence.physical_col_count < 2:
        return False
    if _ocr_raw_evidence_looks_like_chart_axis_projection(raw_evidence):
        return False
    if _ocr_raw_evidence_looks_like_bar_chart_projection(raw_evidence):
        return False
    non_empty_by_row = [
        sum(1 for cell in row if str(cell or "").strip())
        for row in grid
        if isinstance(row, list)
    ]
    if len(non_empty_by_row) < 3:
        return False
    multi_cell_rows = sum(1 for count in non_empty_by_row if count >= 2)
    if multi_cell_rows < max(3, len(non_empty_by_row) // 2):
        return False
    header_text = " ".join(str(cell or "") for cell in (grid[0] if isinstance(grid[0], list) else []))
    numeric_cells = sum(
        1
        for row in grid
        if isinstance(row, list)
        for cell in row
        if _looks_like_numeric_table_value(str(cell or ""))
    )
    text_cells = sum(
        1
        for row in grid
        if isinstance(row, list)
        for cell in row
        if str(cell or "").strip()
    )
    has_repeated_numeric_body = numeric_cells >= max(4, text_cells // 4)
    has_header_words = len(re.findall(r"[A-Za-z\u4e00-\u9fff]{3,}", header_text)) >= 2
    return has_repeated_numeric_body or has_header_words


def _ocr_raw_evidence_looks_like_bar_chart_projection(raw_evidence: RawTableEvidence) -> bool:
    """Reject OCR grids produced by horizontal/vertical bar charts.

    Bar charts commonly OCR as rows of category labels plus one plotted value,
    followed by an axis tick row such as ``0% 5% 10% ...``. That is chart
    evidence. A real table can contain percent values, but it should also have
    authored table structure: header/schema evidence, ruling evidence, or stable
    rectangular value columns beyond a chart tick axis.
    """
    if re.search(r"\b(?:table|tab\.?)\s*\d+|表\s*[\d一二三四五六七八九十]+", raw_evidence.caption_text, re.IGNORECASE):
        return False
    if raw_evidence.vertical_lines or raw_evidence.rectangles or len(raw_evidence.horizontal_lines) >= 2:
        return False

    grid = [row for row in (raw_evidence.raw_data or []) if isinstance(row, list)]
    if len(grid) < 5:
        return False

    non_empty_rows = [
        [str(cell or "").strip() for cell in row if str(cell or "").strip()]
        for row in grid
    ]
    non_empty_rows = [row for row in non_empty_rows if row]
    if len(non_empty_rows) < 5:
        return False

    axis_tick_rows = [
        row
        for row in non_empty_rows
        if _percent_tokens_form_axis_ticks(" ".join(row))
    ]
    if not axis_tick_rows:
        return False

    label_value_rows = 0
    percent_value_rows = 0
    header_like_rows = 0
    for row in non_empty_rows:
        text = " ".join(row)
        if _percent_tokens_form_axis_ticks(text):
            continue
        percent_tokens = re.findall(r"[-+]?\d+(?:\.\d+)?\s*%", text)
        word_tokens = re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", text)
        if percent_tokens:
            percent_value_rows += 1
        if percent_tokens and word_tokens and len(percent_tokens) <= 2:
            label_value_rows += 1
            continue
        if len(word_tokens) >= 2 and not percent_tokens:
            header_like_rows += 1

    if percent_value_rows < max(4, len(non_empty_rows) // 3):
        return False
    if label_value_rows < max(3, int((len(non_empty_rows) - len(axis_tick_rows)) * 0.45)):
        return False
    if header_like_rows >= max(3, len(non_empty_rows) // 2):
        return False

    percent_cells = 0
    populated_cells = 0
    for row in grid:
        for cell in row:
            text = str(cell or "").strip()
            if not text:
                continue
            populated_cells += 1
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?\s*%", text):
                percent_cells += 1

    return percent_cells >= 4 and percent_cells / max(1, populated_cells) >= 0.20


def _percent_tokens_form_axis_ticks(text: str) -> bool:
    values: list[float] = []
    for match in re.finditer(r"[-+]?\d+(?:\.\d+)?\s*%", str(text or "")):
        try:
            values.append(float(match.group(0).replace("%", "").strip()))
        except Exception:
            continue
    if _numbers_form_tick_sequence(values):
        return True
    unique_sorted = sorted(set(values))
    if len(unique_sorted) < 5:
        return False
    if max(unique_sorted) - min(unique_sorted) < 20.0:
        return False
    return _numbers_form_tick_sequence(unique_sorted)


def _ocr_raw_evidence_looks_like_chart_axis_projection(raw_evidence: RawTableEvidence) -> bool:
    """Reject OCR grids made from chart axes, data labels, and legends.

    Embedded charts often OCR into two rough columns: y-axis ticks on the left
    and scattered plotted value labels on the right, with legend labels near the
    bottom. That is visual chart evidence, not a rectangular data table.
    """
    grid = [row for row in (raw_evidence.raw_data or []) if isinstance(row, list)]
    if raw_evidence.physical_col_count > 2 or len(grid) < 6:
        return False

    non_empty_rows = [
        [str(cell or "").strip() for cell in row if str(cell or "").strip()]
        for row in grid
    ]
    non_empty_rows = [row for row in non_empty_rows if row]
    if len(non_empty_rows) < 6:
        return False

    text_word_rows = [
        idx
        for idx, row in enumerate(non_empty_rows)
        if len(re.findall(r"[A-Za-z\u4e00-\u9fff]{3,}", " ".join(row))) >= 1
    ]
    if text_word_rows and min(text_word_rows) <= 1:
        return False

    first_col_numbers: list[float] = []
    first_col_numeric_rows = 0
    for row in grid:
        first = str(row[0] if row else "").strip()
        numbers = _extract_plain_numeric_tokens(first)
        if numbers and _cell_is_mostly_numeric(first):
            first_col_numeric_rows += 1
            first_col_numbers.append(numbers[0])

    if first_col_numeric_rows < 5:
        return False
    numeric_prefix_len = 0
    for row in non_empty_rows:
        text = " ".join(row)
        if _cell_is_mostly_numeric(text):
            numeric_prefix_len += 1
            continue
        break
    if numeric_prefix_len < max(5, int(len(non_empty_rows) * 0.55)):
        return False

    total_non_empty = sum(len(row) for row in non_empty_rows)
    numeric_like_cells = sum(1 for row in non_empty_rows for cell in row if _cell_is_mostly_numeric(cell))
    if numeric_like_cells / max(1, total_non_empty) < 0.68:
        return False

    empty_second_rows = 0
    second_col_multi_numeric_rows = 0
    for row in grid:
        if len(row) < 2 or not str(row[1] or "").strip():
            empty_second_rows += 1
            continue
        if len(_extract_plain_numeric_tokens(str(row[1] or ""))) >= 2:
            second_col_multi_numeric_rows += 1

    sparse_or_scattered_values = (
        empty_second_rows >= max(2, len(grid) // 3)
        or second_col_multi_numeric_rows >= 2
    )
    return sparse_or_scattered_values and _numbers_form_tick_sequence(first_col_numbers)


def _extract_plain_numeric_tokens(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])", str(text or "")):
        try:
            values.append(float(match.group(0)))
        except Exception:
            continue
    return values


def _cell_is_mostly_numeric(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    cleaned = re.sub(r"[-+]?\d+(?:\.\d+)?", "", candidate)
    cleaned = re.sub(r"[\s,.;:%()/-]+", "", cleaned)
    return not cleaned


def _numbers_form_tick_sequence(values: list[float]) -> bool:
    if len(values) < 5:
        return False
    seq = values[:]
    direction = 0
    diffs: list[float] = []
    for left, right in zip(seq, seq[1:]):
        diff = right - left
        if abs(diff) < 1e-6:
            continue
        if direction == 0:
            direction = 1 if diff > 0 else -1
        elif (diff > 0 and direction < 0) or (diff < 0 and direction > 0):
            return False
        diffs.append(abs(diff))
    if len(diffs) < 4:
        return False
    median_diff = statistics.median(diffs)
    if median_diff <= 0.0:
        return False
    regular = sum(1 for diff in diffs if abs(diff - median_diff) <= max(0.75, median_diff * 0.18))
    return regular >= max(4, int(len(diffs) * 0.7))


def _looks_like_numeric_table_value(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    return bool(re.fullmatch(r"[-+]?[\d,]+(?:\.\d+)?(?:e[-+]?\d+|%|[a-zA-Z/\u00b2\u00b3-]*)?", candidate, re.IGNORECASE))


def _build_vector_ocr_raw_evidence(
    page: Any,
    table_bbox: tuple[float, float, float, float],
    page_number: int,
    page_height: float,
    page_width: float,
    prefer_stable_row_anchors: bool = False,
) -> RawTableEvidence | None:
    ocr_engine = _get_vector_ocr_engine()
    if ocr_engine is None or np is None or pymupdf is None:
        return None

    clip = pymupdf.Rect(*table_bbox)
    if clip.width < 24 or clip.height < 24:
        return None
    render_scale = _compute_vector_ocr_render_scale(float(clip.width), float(clip.height))

    try:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(render_scale, render_scale), clip=clip, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        result, _ = ocr_engine(image)
    except Exception:
        return None

    entries = _normalize_ocr_result_entries(result, clip, pix.width, pix.height)
    if len(entries) < 8:
        return None

    row_groups = _group_ocr_entries_into_rows(entries)
    if len(row_groups) < 4:
        return None

    if prefer_stable_row_anchors:
        column_anchors = _detect_ocr_column_anchors_from_rows(row_groups, table_bbox)
        if len(column_anchors) < 2:
            column_anchors = _detect_ocr_column_anchors(entries, table_bbox)
    else:
        column_anchors = _detect_ocr_column_anchors(entries, table_bbox)
    if len(column_anchors) < 2 or len(column_anchors) > 8:
        return None

    raw_rows, raw_data, spans, words = _build_raw_rows_from_ocr_entries(row_groups, column_anchors)
    if len(raw_rows) < 4:
        return None

    drawings = _extract_drawings_from_page(page, table_bbox)
    return RawTableEvidence(
        page_number=page_number,
        bbox=table_bbox,
        physical_col_count=len(column_anchors),
        physical_row_count=len(raw_rows),
        rows=raw_rows,
        chars=[],
        spans=spans,
        words=words,
        drawings=drawings,
        raw_data=raw_data,
        page_height=page_height,
        page_width=page_width,
        near_page_top=table_bbox[1] <= page_height * 0.28,
        near_page_bottom=table_bbox[3] >= page_height * 0.72,
        source="vector_ocr",
    )


def _merge_two_column_ocr_continuation_rows(
    raw_rows: list[RawRow],
    raw_data: list[list[str | None]],
    spans: list[RawSpan],
    words: list[RawWord],
) -> tuple[list[RawRow], list[list[str | None]], list[RawSpan], list[RawWord]]:
    if len(raw_data) < 2:
        return raw_rows, raw_data, spans, words

    merged_data: list[list[str | None]] = []
    skip_next = False
    for idx, row in enumerate(raw_data):
        if skip_next:
            skip_next = False
            continue
        current = list(row)
        if (
            idx + 1 < len(raw_data)
            and len(current) >= 2
            and len(raw_data[idx + 1]) >= 2
            and not str(raw_data[idx + 1][0] or "").strip()
            and str(raw_data[idx + 1][1] or "").strip()
            and _two_column_ocr_row_accepts_value_continuation(current, raw_data[idx + 1])
        ):
            current[1] = _clean_text(" ".join(part for part in [str(current[1] or ""), str(raw_data[idx + 1][1] or "")] if part.strip()))
            skip_next = True
        merged_data.append(current)

    if len(merged_data) == len(raw_data):
        return raw_rows, raw_data, spans, words

    rebuilt_rows: list[RawRow] = []
    for row_index, row_values in enumerate(merged_data):
        cells = [
            RawCell(
                physical_col=col_index,
                physical_row=row_index,
                text=str(value).strip() if str(value or "").strip() else None,
                spans=[],
                bbox=None,
            )
            for col_index, value in enumerate(row_values)
        ]
        rebuilt_rows.append(
            RawRow(
                physical_row=row_index,
                cells=cells,
                bbox=None,
                y0=0.0,
                y1=0.0,
            )
        )
    return rebuilt_rows, merged_data, spans, words


def _two_column_ocr_row_accepts_value_continuation(
    current: list[str | None],
    following: list[str | None],
) -> bool:
    left = str(current[0] or "").strip()
    value = str(current[1] or "").strip()
    continuation = str(following[1] or "").strip()
    if not value or not continuation:
        return False
    if not left:
        return True
    if _ocr_entry_looks_like_left_column_label(left) and re.search(r"\[MASK\]", continuation, re.IGNORECASE):
        return True
    return not _ocr_entry_looks_like_left_column_label(left)


def _normalize_ocr_result_entries(
    result: Any,
    clip: Any,
    image_width: int,
    image_height: int,
) -> list[_OcrEntry]:
    entries: list[_OcrEntry] = []
    if not result:
        return entries

    scale_x = image_width / max(1.0, float(clip.width))
    scale_y = image_height / max(1.0, float(clip.height))
    for item in result:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        box, text, confidence = item[0], _clean_text(str(item[1])), float(item[2] or 0.0)
        if not text or confidence < 0.35:
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
        entries.append(_OcrEntry(text=text, confidence=confidence, bbox=bbox))

    entries.sort(key=lambda entry: (entry.y_center, entry.x0))
    return entries


def _detect_ocr_column_anchors(
    entries: list[_OcrEntry],
    table_bbox: tuple[float, float, float, float],
) -> list[float]:
    if not entries:
        return []
    tolerance = max(12.0, _bbox_width(table_bbox) * 0.035)
    clusters: list[list[float]] = []
    for x0 in sorted(entry.x0 for entry in entries):
        if not clusters or abs(x0 - statistics.mean(clusters[-1])) > tolerance:
            clusters.append([x0])
            continue
        clusters[-1].append(x0)
    return [statistics.mean(cluster) for cluster in clusters if cluster]


def _detect_ocr_column_anchors_from_rows(
    row_groups: list[list[_OcrEntry]],
    table_bbox: tuple[float, float, float, float],
) -> list[float]:
    """Infer OCR table columns from repeated row starts, not header fragments."""

    if not row_groups:
        return []
    row_groups = [row for row in row_groups if row]
    tolerance = max(14.0, _bbox_width(table_bbox) * 0.045)
    candidate_rows = [
        row
        for row in row_groups
        if len(row) >= 3
        and not _ocr_row_looks_like_spanning_caption(row, table_bbox)
    ]
    if len(candidate_rows) < 2:
        candidate_rows = [row for row in row_groups if len(row) >= 2]
    if len(candidate_rows) < 2:
        return []

    label_value_anchors = _detect_two_column_label_value_ocr_anchors(row_groups, table_bbox)
    if label_value_anchors:
        return label_value_anchors

    best_count, support = _most_supported_row_entry_count(candidate_rows)
    if best_count < 2 or support < 2:
        return _detect_ocr_column_anchors_by_support(candidate_rows, table_bbox)
    stable_rows = [row for row in candidate_rows if len(row) == best_count]
    if len(stable_rows) < 2:
        return _detect_ocr_column_anchors_by_support(candidate_rows, table_bbox)

    anchors: list[float] = []
    for idx in range(best_count):
        values = [float(row[idx].x0) for row in stable_rows]
        if max(values) - min(values) > tolerance * 2.8:
            return _detect_ocr_column_anchors_by_support(candidate_rows, table_bbox)
        anchors.append(float(statistics.median(values)))
    if any(right <= left + 8.0 for left, right in zip(anchors, anchors[1:])):
        return _detect_ocr_column_anchors_by_support(candidate_rows, table_bbox)
    return anchors


def _detect_ocr_column_anchors_by_support(
    row_groups: list[list[_OcrEntry]],
    table_bbox: tuple[float, float, float, float],
) -> list[float]:
    tolerance = max(16.0, _bbox_width(table_bbox) * 0.05)
    clusters: list[list[float]] = []
    for x0 in sorted(float(entry.x0) for row in row_groups for entry in row):
        if not clusters or abs(x0 - statistics.median(clusters[-1])) > tolerance:
            clusters.append([x0])
        else:
            clusters[-1].append(x0)
    supported = [
        float(statistics.median(cluster))
        for cluster in clusters
        if len(cluster) >= 2
    ]
    if len(supported) < 2 or len(supported) > 8:
        return []
    return supported


def _most_supported_row_entry_count(row_groups: list[list[_OcrEntry]]) -> tuple[int, int]:
    counts: dict[int, int] = {}
    for row in row_groups:
        counts[len(row)] = counts.get(len(row), 0) + 1
    return max(counts.items(), key=lambda item: (item[1], item[0])) if counts else (0, 0)


def _detect_two_column_label_value_ocr_anchors(
    row_groups: list[list[_OcrEntry]],
    table_bbox: tuple[float, float, float, float],
) -> list[float]:
    """Detect OCR tables whose right value cell is split into several words."""

    if len(row_groups) < 3:
        return []
    width = _bbox_width(table_bbox)
    left_entries = [
        row[0]
        for row in row_groups
        if len(row) >= 2 and row[0].x0 <= table_bbox[0] + width * 0.32
    ]
    if len(left_entries) < 2:
        return []
    left_anchor = float(statistics.median(entry.x0 for entry in left_entries))
    right_entries = [
        entry
        for row in row_groups
        for entry in row
        if entry.x0 >= left_anchor + max(42.0, width * 0.18)
    ]
    if len(right_entries) < max(4, len(row_groups)):
        return []
    right_anchor = float(statistics.median(min(entry.x0 for entry in row if entry.x0 >= left_anchor + max(42.0, width * 0.18)) for row in row_groups if any(entry.x0 >= left_anchor + max(42.0, width * 0.18) for entry in row)))
    if right_anchor <= left_anchor + max(30.0, width * 0.12):
        return []
    left_textual = sum(1 for entry in left_entries if re.search(r"[A-Za-z\u4e00-\u9fff]", entry.text or ""))
    right_textual = sum(1 for entry in right_entries if re.search(r"[A-Za-z\u4e00-\u9fff]", entry.text or ""))
    if left_textual < 2 or right_textual < 3:
        return []
    return [left_anchor, right_anchor]


def _ocr_row_looks_like_spanning_caption(
    row: list[_OcrEntry],
    table_bbox: tuple[float, float, float, float],
) -> bool:
    if not row:
        return False
    if len(row) == 1 and _bbox_width(row[0].bbox) >= _bbox_width(table_bbox) * 0.45:
        return True
    text = " ".join(entry.text for entry in row).strip()
    if len(row) <= 2 and len(text) >= 24 and re.search(r"[:：]\s*$", text):
        return True
    return False


def _split_ocr_single_cell_stub_rows(
    row_groups: list[list[_OcrEntry]],
    column_anchors: list[float],
) -> list[list[_OcrEntry]]:
    if len(column_anchors) < 2 or not row_groups:
        return row_groups
    result: list[list[_OcrEntry]] = []
    for idx, row in enumerate(row_groups):
        if not row:
            continue
        if (
            result
            and len(row) >= 2
            and len(result[-1]) == 2
            and (
                not _ocr_entry_looks_like_left_column_label(result[-1][0].text)
            )
            and (
                not _ocr_entry_looks_like_left_column_label(row[0].text)
                or _ocr_value_cell_continuation_has_mask(row)
            )
            and row[0].x_center >= column_anchors[1]
            and row[0].y0 - result[-1][1].y1 <= max(14.0, row[0].height * 1.6)
        ):
            result[-1][1] = _merge_ocr_entries([result[-1][1], *row])
            continue
        if (
            len(row) >= 3
            and idx + 1 < len(row_groups)
            and len(row_groups[idx + 1]) >= 2
            and _ocr_entries_are_left_header_phrase(row[:2], column_anchors)
            and row[2].x_center >= column_anchors[0]
            and row_groups[idx + 1][0].x_center >= column_anchors[0]
            and row_groups[idx + 1][0].y0 - row[2].y1 <= max(14.0, row_groups[idx + 1][0].height * 1.6)
        ):
            result.append([_merge_ocr_entries(row[:2]), _merge_ocr_entries([row[2], *row_groups[idx + 1]])])
            row_groups[idx + 1] = []
            continue
        if (
            len(row) == 2
            and idx + 1 < len(row_groups)
            and len(row_groups[idx + 1]) == 2
            and _ocr_row_has_first_column_header_stub(row, column_anchors)
            and not _ocr_row_has_first_column_header_stub(row_groups[idx + 1], column_anchors)
            and row_groups[idx + 1][0].x_center >= column_anchors[1]
            and row_groups[idx + 1][0].y0 - row[1].y1 <= max(14.0, row_groups[idx + 1][0].height * 1.6)
        ):
            result.append([row[0], row[1], *row_groups[idx + 1]])
            row_groups[idx + 1] = []
            continue
        if (
            result
            and len(row) == 1
            and len(result[-1]) == 2
            and result[-1][1].x_center >= column_anchors[1]
            and row[0].x_center >= column_anchors[1]
            and not _ocr_entry_looks_like_left_column_label(row[0].text)
            and row[0].y0 - result[-1][1].y1 <= max(14.0, row[0].height * 1.6)
        ):
            result[-1].append(row[0])
            continue
        if (
            len(row) == 1
            and result
            and idx + 1 < len(row_groups)
            and len(row_groups[idx + 1]) == 1
            and row[0].x_center < column_anchors[1]
            and row_groups[idx + 1][0].x_center >= column_anchors[1]
        ):
            result.append([row[0], row_groups[idx + 1][0]])
            row_groups[idx + 1] = []
            continue
        if row:
            result.append(row)
    return result


def _ocr_row_has_first_column_header_stub(
    row: list[_OcrEntry],
    column_anchors: list[float],
) -> bool:
    if len(column_anchors) < 2 or not row:
        return False
    first = row[0]
    text = re.sub(r"\s+", "", first.text.lower())
    return first.x_center < column_anchors[1] and bool(text) and not re.search(r"\[mask\]|\d", text)


def _ocr_entry_looks_like_left_column_label(text: str | None) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    compact = re.sub(r"\s+", "", candidate)
    if re.fullmatch(r"(?:BERT|RoBERTa(?:-wwm)?|GPT|T5|XLNet|ERNIE|ALBERT|ELECTRA)", compact, re.IGNORECASE):
        return True
    if re.search(r"\[MASK\]|\d", compact, re.IGNORECASE):
        return True
    if re.fullmatch(r"[A-Z][A-Za-z0-9_-]{1,24}", compact) and not re.fullmatch(r"[A-Z][a-z]{3,}", compact):
        return True
    return False


def _ocr_value_cell_continuation_has_mask(row: list[_OcrEntry]) -> bool:
    text = " ".join(entry.text for entry in row)
    return bool(re.search(r"\[MASK\]", text, re.IGNORECASE))


def _ocr_entries_are_left_header_phrase(
    entries: list[_OcrEntry],
    column_anchors: list[float],
) -> bool:
    if len(entries) < 2 or len(column_anchors) < 2:
        return False
    if entries[1].x_center >= column_anchors[0]:
        return False
    texts = [re.sub(r"\s+", "", entry.text.lower()) for entry in entries]
    if any(not text or re.search(r"\[mask\]|\d", text) for text in texts):
        return False
    return entries[1].x0 - entries[0].x1 <= max(18.0, entries[0].height * 2.5)


def _merge_ocr_entries(entries: list[_OcrEntry]) -> _OcrEntry:
    text = " ".join(entry.text for entry in entries if entry.text).strip()
    confidence = min((entry.confidence for entry in entries), default=0.0)
    bbox = (
        min(entry.x0 for entry in entries),
        min(entry.y0 for entry in entries),
        max(entry.x1 for entry in entries),
        max(entry.y1 for entry in entries),
    )
    return _OcrEntry(text=text, confidence=confidence, bbox=bbox)


def _group_ocr_entries_into_rows(entries: list[_OcrEntry]) -> list[list[_OcrEntry]]:
    if not entries:
        return []

    row_height_basis = [entry.height for entry in entries if entry.height > 0]
    row_tolerance = max(4.0, statistics.median(row_height_basis) * 0.65) if row_height_basis else 4.0
    rows: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda item: (item.y_center, item.x0)):
        target_row: dict[str, Any] | None = None
        for row in rows:
            if abs(entry.y_center - float(row["center_y"])) <= row_tolerance:
                target_row = row
                break
        if target_row is None:
            rows.append({"center_y": entry.y_center, "entries": [entry]})
            continue
        target_row["entries"].append(entry)
        target_row["center_y"] = statistics.mean(item.y_center for item in target_row["entries"])

    grouped_rows = [
        sorted(list(row["entries"]), key=lambda item: item.x0)
        for row in rows
    ]
    grouped_rows.sort(key=lambda row: min(item.y0 for item in row))
    return grouped_rows


def _build_raw_rows_from_ocr_entries(
    row_groups: list[list[_OcrEntry]],
    column_anchors: list[float],
) -> tuple[list[RawRow], list[list[str | None]], list[RawSpan], list[RawWord]]:
    raw_rows: list[RawRow] = []
    raw_data: list[list[str | None]] = []
    spans: list[RawSpan] = []
    words: list[RawWord] = []

    for row_index, entries in enumerate(row_groups):
        grouped_by_column: list[list[_OcrEntry]] = [[] for _ in column_anchors]
        for entry in entries:
            column_index = min(
                range(len(column_anchors)),
                key=lambda idx: abs(entry.x0 - column_anchors[idx]),
            )
            grouped_by_column[column_index].append(entry)

        row_cells: list[RawCell] = []
        row_bbox_components: list[tuple[float, float, float, float]] = []
        row_texts: list[str | None] = []
        for column_index, column_entries in enumerate(grouped_by_column):
            column_entries.sort(key=lambda item: (item.y0, item.x0))
            cell_text = _join_ocr_cell_text(column_entries)
            cell_bbox = _bbox_union([entry.bbox for entry in column_entries]) if column_entries else None
            cell_spans = [
                RawSpan(
                    text=entry.text,
                    x0=entry.x0,
                    y0=entry.y0,
                    x1=entry.x1,
                    y1=entry.y1,
                    size=max(0.0, entry.height),
                    font="ocr-vector",
                    flags=0,
                    chars=[],
                    origin=(entry.x0, entry.y0),
                )
                for entry in column_entries
            ]
            spans.extend(cell_spans)
            words.extend(
                RawWord(
                    text=entry.text,
                    x0=entry.x0,
                    y0=entry.y0,
                    x1=entry.x1,
                    y1=entry.y1,
                )
                for entry in column_entries
            )
            if cell_bbox is not None:
                row_bbox_components.append(cell_bbox)
            row_cells.append(
                RawCell(
                    physical_col=column_index,
                    physical_row=row_index,
                    text=cell_text or None,
                    spans=cell_spans,
                    bbox=cell_bbox,
                )
            )
            row_texts.append(cell_text or None)

        if not any(text for text in row_texts):
            continue
        row_bbox = _bbox_union(row_bbox_components) if row_bbox_components else None
        raw_rows.append(
            RawRow(
                physical_row=len(raw_rows),
                cells=row_cells,
                bbox=row_bbox,
                y0=float(row_bbox[1]) if row_bbox else 0.0,
                y1=float(row_bbox[3]) if row_bbox else 0.0,
            )
        )
        raw_data.append(row_texts)

    return raw_rows, raw_data, spans, words


def _join_ocr_cell_text(entries: list[_OcrEntry]) -> str:
    if not entries:
        return ""
    parts: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        text = _clean_text(entry.text)
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    return " ".join(parts).strip()


def _drawing_bbox(draw: dict[str, Any]) -> tuple[float, float, float, float] | None:
    rect = draw.get("rect")
    if rect is None:
        return None
    try:
        return (
            float(rect[0]),
            float(rect[1]),
            float(rect[2]),
            float(rect[3]),
        )
    except Exception:
        return None


def _is_wide_horizontal_rule_bbox(
    bbox: tuple[float, float, float, float],
    page_width: float,
) -> bool:
    return _bbox_width(bbox) >= page_width * 0.35 and _bbox_height(bbox) <= 2.5


def _is_dense_dark_glyph_rect(bbox: tuple[float, float, float, float]) -> bool:
    width = _bbox_width(bbox)
    height = _bbox_height(bbox)
    area = width * height
    return width <= 180.0 and height <= 20.0 and area <= 900.0


def _bbox_contains_word(
    bbox: tuple[float, float, float, float],
    word: _Word,
) -> bool:
    word_bbox = (float(word.x0), float(word.y0), float(word.x1), float(word.y1))
    return _bbox_overlap_ratio(bbox, word_bbox) >= 0.75


def _horizontal_overlap_ratio(
    a_bbox: tuple[float, float, float, float],
    b_bbox: tuple[float, float, float, float],
) -> float:
    overlap = max(0.0, min(a_bbox[2], b_bbox[2]) - max(a_bbox[0], b_bbox[0]))
    min_width = max(1.0, min(_bbox_width(a_bbox), _bbox_width(b_bbox)))
    return overlap / min_width


def _bbox_overlap_ratio(
    a_bbox: tuple[float, float, float, float],
    b_bbox: tuple[float, float, float, float],
) -> float:
    x0 = max(a_bbox[0], b_bbox[0])
    y0 = max(a_bbox[1], b_bbox[1])
    x1 = min(a_bbox[2], b_bbox[2])
    y1 = min(a_bbox[3], b_bbox[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    min_area = max(
        1.0,
        min(_bbox_width(a_bbox) * _bbox_height(a_bbox), _bbox_width(b_bbox) * _bbox_height(b_bbox)),
    )
    return intersection / min_area


def _bbox_width(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0]))


def _bbox_height(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, float(bbox[3]) - float(bbox[1]))
