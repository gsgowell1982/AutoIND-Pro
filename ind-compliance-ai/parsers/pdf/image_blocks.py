from __future__ import annotations

import io
import re
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

from .layout import _bbox_contains_path, _drawing_rect_to_tuple, _words_in_bbox, _words_to_text
from .shared import (
    _Word,
    _bbox_area,
    _bbox_intersection_ratio,
    _bbox_to_list,
    _bbox_union,
    _clean_text,
    _compact_text,
    _horizontal_overlap_ratio,
    _text_contains_text,
)
from .text_blocks import _is_footer_artifact_block

# Version: v1.0.2
# Updates:
# - Reuse the unified footer-artifact rule from text_blocks so figure-title
#   selection and semantic text merging share the same footer/page-number guard.
# - Keep figure caption candidate filtering aligned with the new merge barrier.

_FIGURE_LABEL_KEYWORDS = ("figure", "fig", "图", "图表", "表", "chart", "illustration")
_PUBLICATION_ARTIFACT_KEYWORDS = (
    "openaccess",
    "creativecommons",
    "copyright",
    "licence",
    "license",
    "fulllistofauthorinformation",
    "correspondence",
    "authorinformation",
    "sciencedirect",
    "contentslistsavailable",
    "journalhomepage",
)
_EXPLICIT_FIGURE_CAPTION_RE = re.compile(r"^(?:fig(?:ure)?\.?\s*\d+)\b", re.IGNORECASE)


def _is_explicit_figure_caption(text: str) -> bool:
    return bool(_EXPLICIT_FIGURE_CAPTION_RE.match(_clean_text(text)))


def _expanded_bbox(
    bbox: tuple[float, float, float, float],
    x_pad: float,
    y_pad: float,
) -> tuple[float, float, float, float]:
    return (
        float(bbox[0]) - x_pad,
        float(bbox[1]) - y_pad,
        float(bbox[2]) + x_pad,
        float(bbox[3]) + y_pad,
    )


def _expanded_bboxes_intersect(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    *,
    x_pad: float = 4.0,
    y_pad: float = 4.0,
) -> bool:
    left_expanded = _expanded_bbox(left, x_pad, y_pad)
    right_expanded = _expanded_bbox(right, x_pad, y_pad)
    return not (
        left_expanded[2] < right_expanded[0]
        or right_expanded[2] < left_expanded[0]
        or left_expanded[3] < right_expanded[1]
        or right_expanded[3] < left_expanded[1]
    )


def _build_drawing_components(
    page_drawings: list[dict[str, Any]],
    page_width: float,
) -> list[dict[str, Any]]:
    drawing_rects: list[tuple[float, float, float, float]] = []
    for drawing in page_drawings:
        rect = _drawing_rect_to_tuple(drawing)
        if rect is None:
            continue
        width = max(0.0, rect[2] - rect[0])
        height = max(0.0, rect[3] - rect[1])
        if width <= 0.0 and height <= 0.0:
            continue
        # Ignore separator rules that would otherwise connect unrelated vector regions.
        if min(width, height) <= 1.2 and max(width, height) >= page_width * 0.55:
            continue
        drawing_rects.append(rect)

    if not drawing_rects:
        return []

    parents = list(range(len(drawing_rects)))

    def _find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def _union(left_index: int, right_index: int) -> None:
        left_root = _find(left_index)
        right_root = _find(right_index)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left_rect in enumerate(drawing_rects):
        for right_index in range(left_index + 1, len(drawing_rects)):
            right_rect = drawing_rects[right_index]
            if _expanded_bboxes_intersect(left_rect, right_rect, x_pad=4.0, y_pad=4.0):
                _union(left_index, right_index)

    grouped_rects: dict[int, list[tuple[float, float, float, float]]] = {}
    for index, rect in enumerate(drawing_rects):
        grouped_rects.setdefault(_find(index), []).append(rect)

    components: list[dict[str, Any]] = []
    for rects in grouped_rects.values():
        union_bbox = _bbox_union(rects)
        width = max(0.0, union_bbox[2] - union_bbox[0])
        height = max(0.0, union_bbox[3] - union_bbox[1])
        components.append(
            {
                "bbox": union_bbox,
                "width": width,
                "height": height,
                "count": len(rects),
            }
        )
    return components


def _panel_component_dimension_compatible(
    candidate: dict[str, Any],
    reference_width: float,
    reference_height: float,
) -> bool:
    candidate_width = float(candidate.get("width", 0.0) or 0.0)
    candidate_height = float(candidate.get("height", 0.0) or 0.0)
    if reference_width <= 0.0 or reference_height <= 0.0:
        return False
    width_ratio = candidate_width / reference_width
    height_ratio = candidate_height / reference_height
    return 0.72 <= width_ratio <= 1.45 and 0.72 <= height_ratio <= 1.45


def _bbox_vertical_overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    overlap = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    min_height = max(1.0, min(max(0.0, left[3] - left[1]), max(0.0, right[3] - right[1])))
    return overlap / min_height


def _synthesize_vector_figure_blocks(
    image_blocks: list[dict[str, Any]],
    text_blocks: list[dict[str, Any]],
    page_drawings: list[dict[str, Any]],
    page_number: int,
    page_width: float,
) -> list[dict[str, Any]]:
    if not text_blocks or not page_drawings or page_width <= 0.0:
        return image_blocks

    drawing_components = _build_drawing_components(page_drawings, page_width)
    if not drawing_components:
        return image_blocks

    augmented_images = list(image_blocks)
    min_anchor_width = max(72.0, page_width * 0.16)
    min_anchor_height = 60.0

    for caption_block in text_blocks:
        caption_text = _clean_text(str(caption_block.get("text", "")))
        if not _is_explicit_figure_caption(caption_text):
            continue

        caption_bbox = tuple(float(item) for item in caption_block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        caption_top = float(caption_bbox[1])
        anchor_components: list[dict[str, Any]] = []
        min_gap: float | None = None

        for component in drawing_components:
            component_bbox = tuple(float(item) for item in component["bbox"])
            gap = caption_top - component_bbox[3]
            if gap < 0.0 or gap > 20.0:
                continue
            if float(component["width"]) < min_anchor_width:
                continue
            if float(component["height"]) < min_anchor_height:
                continue
            if int(component["count"]) < 12:
                continue
            anchor_components.append({**component, "caption_gap": gap})
            min_gap = gap if min_gap is None else min(min_gap, gap)

        if not anchor_components or min_gap is None:
            continue

        selected_components = [
            component
            for component in anchor_components
            if abs(float(component["caption_gap"]) - min_gap) <= 6.0
        ]
        reference_width = sum(float(component["width"]) for component in selected_components) / max(1, len(selected_components))
        reference_height = sum(float(component["height"]) for component in selected_components) / max(1, len(selected_components))

        for component in drawing_components:
            if component in selected_components:
                continue
            component_bbox = tuple(float(item) for item in component["bbox"])
            if component_bbox[3] > caption_top:
                continue
            if int(component.get("count", 0) or 0) < 12:
                continue
            if not _panel_component_dimension_compatible(component, reference_width, reference_height):
                continue
            if _bbox_vertical_overlap_ratio(component_bbox, tuple(selected_components[0]["bbox"])) < 0.72:
                continue
            selected_components.append(component)

        candidate_bbox = _bbox_union([tuple(component["bbox"]) for component in selected_components])

        expanded = True
        while expanded:
            expanded = False
            current_top = min(float(component["bbox"][1]) for component in selected_components)
            current_left = min(float(component["bbox"][0]) for component in selected_components)
            current_right = max(float(component["bbox"][2]) for component in selected_components)
            max_upward_gap = max(96.0, reference_height * 1.55)
            for component in drawing_components:
                if component in selected_components:
                    continue
                component_bbox = tuple(float(item) for item in component["bbox"])
                if component_bbox[3] > caption_top:
                    continue
                if int(component.get("count", 0) or 0) < 12:
                    continue
                if not _panel_component_dimension_compatible(component, reference_width, reference_height):
                    continue
                if current_top - component_bbox[3] > max_upward_gap:
                    continue
                center_x = (component_bbox[0] + component_bbox[2]) / 2.0
                if center_x < current_left - reference_width * 0.9 or center_x > current_right + reference_width * 0.9:
                    continue
                selected_components.append(component)
                candidate_bbox = _bbox_union([candidate_bbox, component_bbox])
                expanded = True

        for component in drawing_components:
            if component in selected_components:
                continue
            component_bbox = tuple(float(item) for item in component["bbox"])
            if component_bbox[3] > caption_top:
                continue
            vertical_overlap = max(
                0.0,
                min(candidate_bbox[3], component_bbox[3]) - max(candidate_bbox[1], component_bbox[1]),
            )
            min_height = max(
                1.0,
                min(
                    max(0.0, candidate_bbox[3] - candidate_bbox[1]),
                    max(0.0, component_bbox[3] - component_bbox[1]),
                ),
            )
            if vertical_overlap / min_height < 0.3:
                continue
            horizontal_gap = max(
                0.0,
                max(candidate_bbox[0], component_bbox[0]) - min(candidate_bbox[2], component_bbox[2]),
            )
            if horizontal_gap > max(24.0, page_width * 0.05):
                continue
            selected_components.append(component)
            candidate_bbox = _bbox_union([candidate_bbox, component_bbox])

        if any(
            _bbox_intersection_ratio(
                tuple(float(item) for item in image.get("bbox", (0.0, 0.0, 0.0, 0.0))),
                candidate_bbox,
            )
            >= 0.55
            for image in augmented_images
        ):
            continue

        candidate_gap = caption_top - candidate_bbox[3]
        synthetic_image = {
            "block_type": "image",
            "image_id": f"img_p{page_number}_{len(augmented_images) + 1:03d}",
            "page": page_number,
            "bbox": _bbox_to_list(candidate_bbox),
            "width": round(max(0.0, candidate_bbox[2] - candidate_bbox[0]), 2),
            "height": round(max(0.0, candidate_bbox[3] - candidate_bbox[1]), 2),
            "source_block_index": caption_block.get("source_block_index", -1),
            "synthetic_source": "vector_figure_from_caption",
            "caption_text": caption_text,
            "caption_source_block_id": caption_block.get("block_id"),
            "caption_bbox": _bbox_to_list(caption_bbox),
            "caption_gap": round(candidate_gap, 3),
        }
        augmented_images.append(synthetic_image)

    return augmented_images


def _ocr_text_from_clip(
    page: "pymupdf.Page",
    bbox: tuple[float, float, float, float],
) -> tuple[str, float]:
    if pytesseract is None or Image is None or pymupdf is None:
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


def _looks_like_figure_caption(text: str) -> bool:
    normalized = _compact_text(text).lower()
    return any(keyword in normalized for keyword in _FIGURE_LABEL_KEYWORDS)


def _recover_text_from_image_region(
    page: "pymupdf.Page",
    image_bbox: tuple[float, float, float, float],
    page_words: list[_Word],
    page_drawings: list[dict[str, Any]],
) -> dict[str, Any]:
    if pymupdf is None:
        return {"text": "", "confidence": 0.0, "source": "none", "has_path": False}
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
        preserve_as_vector_figure = str(image_block.get("synthetic_source", "") or "") == "vector_figure_from_caption"
        recovered = _recover_text_from_image_region(page, bbox, page_words, page_drawings)
        recovered_text = _clean_text(recovered["text"])
        confidence = float(recovered["confidence"])
        has_path = bool(recovered.get("has_path", False))
        recovered_source = str(recovered.get("source", "none"))
        image_area = _bbox_area(bbox)
        area_ratio = image_area / page_area
        is_figure_caption = _looks_like_figure_caption(recovered_text)

        image_block["text_recovery"] = {
            "text": recovered_text,
            "confidence": round(confidence, 3),
            "source": recovered_source,
            "has_path": has_path,
        }

        if preserve_as_vector_figure:
            kept_images.append(image_block)
            continue

        has_overlapping_text = any(
            _bbox_intersection_ratio(tuple(block["bbox"]), bbox) >= 0.55
            and (
                _text_contains_text(_clean_text(block.get("text", "")), recovered_text)
                or _text_contains_text(recovered_text, _clean_text(block.get("text", "")))
            )
            for block in merged_text_blocks
        )

        if len(recovered_text) >= 2 and has_overlapping_text and not is_figure_caption:
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
        if should_convert and not is_figure_caption:
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


def _are_duplicate_image_blocks(primary: dict[str, Any], secondary: dict[str, Any]) -> bool:
    primary_bbox = tuple(float(item) for item in primary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    secondary_bbox = tuple(float(item) for item in secondary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    overlap = _bbox_intersection_ratio(primary_bbox, secondary_bbox)
    if overlap < 0.9:
        return False
    primary_area = max(1.0, _bbox_area(primary_bbox))
    secondary_area = max(1.0, _bbox_area(secondary_bbox))
    area_ratio = max(primary_area, secondary_area) / min(primary_area, secondary_area)
    if area_ratio > 1.4:
        return False
    primary_center = ((primary_bbox[0] + primary_bbox[2]) / 2, (primary_bbox[1] + primary_bbox[3]) / 2)
    secondary_center = ((secondary_bbox[0] + secondary_bbox[2]) / 2, (secondary_bbox[1] + secondary_bbox[3]) / 2)
    center_distance = abs(primary_center[0] - secondary_center[0]) + abs(primary_center[1] - secondary_center[1])
    return center_distance <= 12.0


def _deduplicate_page_images(
    image_blocks: list[dict[str, Any]],
    page_number: int,
) -> tuple[list[dict[str, Any]], int]:
    if not image_blocks:
        return [], 0
    ordered = sorted(
        image_blocks,
        key=lambda item: (_bbox_area(tuple(float(v) for v in item.get("bbox", (0.0, 0.0, 0.0, 0.0)))), item.get("image_id")),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    removed_count = 0
    for image in ordered:
        duplicate = any(_are_duplicate_image_blocks(image, existing) for existing in kept)
        if duplicate:
            removed_count += 1
            continue
        kept.append(dict(image))

    kept = sorted(kept, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for index, image in enumerate(kept, start=1):
        image["image_id"] = f"img_p{page_number}_{index:03d}"
    return kept, removed_count


def _is_footer_like_text_block(
    block: dict[str, Any],
    page_height: float,
) -> bool:
    return _is_footer_artifact_block(block, page_height)


def _is_publication_artifact_image(
    image_bbox: tuple[float, float, float, float],
    page_height: float,
    caption_text: str,
    embedded_text: str,
    text_recovery: dict[str, Any],
    nearby_context_blocks: list[dict[str, Any]],
) -> bool:
    caption_norm = _compact_text(caption_text)
    embedded_norm = _compact_text(embedded_text)
    context_norm = " ".join(_compact_text(str(block.get("text", ""))) for block in nearby_context_blocks)
    width = max(0.0, image_bbox[2] - image_bbox[0])
    height = max(0.0, image_bbox[3] - image_bbox[1])
    area = max(1.0, width * height)
    small_artifact = area <= 1800.0 and max(width, height) <= 42.0
    margin_bound = image_bbox[3] <= max(96.0, page_height * 0.16) or image_bbox[1] >= page_height - max(96.0, page_height * 0.16)
    has_path = bool(text_recovery.get("has_path", False))
    publication_context = any(keyword in context_norm for keyword in _PUBLICATION_ARTIFACT_KEYWORDS)
    top_logo_like = margin_bound and max(width, height) <= 92.0 and area <= 6800.0
    affiliation_marker_count = sum(
        1
        for marker in ("departmentof", "schoolof", "university", "institute", "china")
        if marker in context_norm
    )

    if _looks_like_figure_caption(caption_text):
        return False
    if any(keyword in caption_norm for keyword in _PUBLICATION_ARTIFACT_KEYWORDS):
        return True
    if any(keyword in embedded_norm for keyword in _PUBLICATION_ARTIFACT_KEYWORDS):
        return True
    if publication_context and top_logo_like:
        return True
    if top_logo_like and affiliation_marker_count >= 2:
        return True
    if (
        top_logo_like
        and not nearby_context_blocks
        and not embedded_norm
        and not _looks_like_figure_caption(caption_text)
    ):
        return True
    if "@" in context_norm and small_artifact:
        return True
    if has_path and small_artifact and margin_bound and not caption_norm and embedded_norm in {"", "openaccess"}:
        return True
    if has_path and publication_context and top_logo_like and not _looks_like_figure_caption(caption_text):
        return True
    return False


def _assign_figure_titles(
    image_blocks: list[dict[str, Any]],
    text_blocks: list[dict[str, Any]],
    figure_start_index: int,
    page_height: float,
) -> tuple[list[dict[str, Any]], int]:
    figure_nodes: list[dict[str, Any]] = []
    figure_index = figure_start_index

    for image_block in image_blocks:
        image_bbox = tuple(image_block["bbox"])
        candidate_queue: list[tuple[float, int, str, dict[str, Any], float]] = []
        for text_block in text_blocks:
            text_bbox = tuple(text_block["bbox"])
            overlap_ratio = _horizontal_overlap_ratio(image_bbox, text_bbox)
            if overlap_ratio < 0.15:
                continue
            if _is_footer_like_text_block(text_block, page_height):
                continue
            text = _clean_text(text_block["text"])
            if not text:
                continue
            distance = text_bbox[1] - image_bbox[3]
            is_below = 0 if distance >= 0 else 1
            candidate_queue.append((abs(distance), is_below, text, text_block, distance))

        title_text = ""
        selected_caption_block: dict[str, Any] | None = None
        selected_caption_gap: float | None = None
        seen_captions: set[str] = set()
        for _, _, candidate_text, source_block, source_distance in sorted(candidate_queue, key=lambda item: (item[0], item[1])):
            normalized = _compact_text(candidate_text)
            if not normalized or normalized in seen_captions:
                continue
            seen_captions.add(normalized)
            if not title_text or _looks_like_figure_caption(candidate_text):
                title_text = candidate_text[:160]
                selected_caption_block = source_block
                selected_caption_gap = source_distance
            if _looks_like_figure_caption(candidate_text):
                break

        figure_ref = f"Figure {figure_index}"
        image_block["title"] = title_text
        image_block["caption_text"] = title_text
        image_block["figure_ref"] = figure_ref
        if selected_caption_block is not None:
            image_block["caption_source_block_id"] = selected_caption_block.get("block_id")
            image_block["caption_bbox"] = list(selected_caption_block.get("bbox", []))
            image_block["caption_gap"] = round(abs(float(selected_caption_gap or 0.0)), 3)
        figure_nodes.append(
            {
                "figure_ref": figure_ref,
                "image_id": image_block["image_id"],
                "title": title_text,
                "caption_text": title_text,
                "page": image_block["page"],
                "bbox": image_block["bbox"],
            }
        )
        figure_index += 1
    return figure_nodes, figure_index


def _enrich_image_content(
    image_blocks: list[dict[str, Any]],
    figure_nodes: list[dict[str, Any]],
    text_blocks: list[dict[str, Any]],
    page_height: float,
    page: "pymupdf.Page | None" = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    figure_by_image_id = {
        str(figure.get("image_id", "")): figure
        for figure in figure_nodes
        if str(figure.get("image_id", "")).strip()
    }

    for image_block in image_blocks:
        _refine_suspicious_full_page_figure_bbox(image_block, page_height, page=page)
        text_recovery = image_block.get("text_recovery") or {}
        caption_text = _clean_text(str(image_block.get("caption_text") or image_block.get("title") or ""))
        embedded_text = _clean_text(str(text_recovery.get("text", "")))
        nearby_context_blocks = _collect_nearby_image_context_blocks(
            image_bbox=tuple(float(item) for item in image_block.get("bbox", (0.0, 0.0, 0.0, 0.0))),
            text_blocks=text_blocks,
            page_height=page_height,
            caption_text=caption_text,
            embedded_text=embedded_text,
        )
        nearby_context_text = " ".join(block["text"] for block in nearby_context_blocks if block.get("text")).strip()
        content_segments = _build_image_content_segments(
            caption_text=caption_text,
            embedded_text=embedded_text,
            text_recovery=text_recovery,
            nearby_context_blocks=nearby_context_blocks,
        )
        content_text = _join_unique_segment_texts(content_segments)
        image_kind_guess = _guess_image_kind(
            image_bbox=tuple(float(item) for item in image_block.get("bbox", (0.0, 0.0, 0.0, 0.0))),
            page_height=page_height,
            caption_text=caption_text,
            embedded_text=embedded_text,
            text_recovery=text_recovery,
            nearby_context_blocks=nearby_context_blocks,
        )
        content_signals = {
            "has_caption": bool(caption_text),
            "has_embedded_text": bool(embedded_text),
            "embedded_text_source": str(text_recovery.get("source", "none")),
            "embedded_text_confidence": round(float(text_recovery.get("confidence", 0.0) or 0.0), 3),
            "has_nearby_context": bool(nearby_context_blocks),
            "nearby_context_block_count": len(nearby_context_blocks),
            "has_path": bool(text_recovery.get("has_path", False)),
        }

        image_block["caption_text"] = caption_text
        image_block["embedded_text"] = embedded_text
        image_block["embedded_text_source"] = str(text_recovery.get("source", "none"))
        image_block["embedded_text_confidence"] = round(float(text_recovery.get("confidence", 0.0) or 0.0), 3)
        image_block["nearby_context_blocks"] = nearby_context_blocks
        image_block["nearby_context_text"] = nearby_context_text
        image_block["content_segments"] = content_segments
        image_block["content_text"] = content_text
        image_block["image_kind_guess"] = image_kind_guess
        image_block["content_signals"] = content_signals

        figure_node = figure_by_image_id.get(str(image_block.get("image_id", "")))
        if figure_node is not None:
            figure_node["bbox"] = list(image_block.get("bbox", []))
            figure_node["caption_text"] = caption_text
            figure_node["embedded_text"] = embedded_text
            figure_node["embedded_text_source"] = image_block["embedded_text_source"]
            figure_node["embedded_text_confidence"] = image_block["embedded_text_confidence"]
            figure_node["nearby_context_text"] = nearby_context_text
            figure_node["content_segments"] = content_segments
            figure_node["content_text"] = content_text
            figure_node["image_kind_guess"] = image_kind_guess
            figure_node["content_signals"] = dict(content_signals)
            if image_block.get("caption_bbox"):
                figure_node["caption_bbox"] = list(image_block["caption_bbox"])
            if image_block.get("caption_source_block_id"):
                figure_node["caption_source_block_id"] = image_block["caption_source_block_id"]
            if image_block.get("caption_gap") is not None:
                figure_node["caption_gap"] = image_block["caption_gap"]

    return image_blocks, figure_nodes


def _refine_suspicious_full_page_figure_bbox(
    image_block: dict[str, Any],
    page_height: float,
    page: "pymupdf.Page | None" = None,
) -> None:
    bbox = tuple(float(item) for item in image_block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    caption_bbox = tuple(float(item) for item in image_block.get("caption_bbox", ()) or ())
    if len(bbox) != 4 or len(caption_bbox) != 4:
        return
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    if width <= 0 or height <= 0:
        return
    full_page_like = bbox[0] <= 1.0 and bbox[1] <= 1.0 and bbox[3] >= page_height - 1.0
    if not full_page_like:
        return
    caption_gap = float(image_block.get("caption_gap", 0.0) or 0.0)
    if caption_gap <= 40.0:
        return
    refined_top = max(72.0, page_height * 0.11)
    if page is not None and pymupdf is not None and Image is not None:
        try:
            clip_bottom = min(float(caption_bbox[1]) - 6.0, page_height - 24.0)
            clip_rect = pymupdf.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), clip_bottom)
            if clip_rect.height > 24 and clip_rect.width > 24:
                pixmap = page.get_pixmap(clip=clip_rect, dpi=96, alpha=False)
                image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                width_px, height_px = image.size
                if width_px > 0 and height_px > 0:
                    pixels = image.load()
                    first_content_row: int | None = None
                    for row_index in range(height_px):
                        dark_pixel_count = 0
                        for col_index in range(width_px):
                            r, g, b = pixels[col_index, row_index]
                            if min(r, g, b) < 245:
                                dark_pixel_count += 1
                        if dark_pixel_count >= max(6, int(width_px * 0.01)):
                            first_content_row = row_index
                            break
                    if first_content_row is not None:
                        refined_top = max(
                            0.0,
                            float(clip_rect.y0) + (first_content_row / max(1, height_px)) * float(clip_rect.height) - 6.0,
                        )
        except Exception:
            refined_top = max(72.0, page_height * 0.11)
    refined_bottom = min(float(caption_bbox[1]) - 6.0, page_height - 72.0)
    if refined_bottom <= refined_top + 40.0:
        return

    image_block["bbox"] = [round(float(bbox[0]), 2), round(refined_top, 2), round(float(bbox[2]), 2), round(refined_bottom, 2)]
    text_recovery = dict(image_block.get("text_recovery") or {})
    if str(text_recovery.get("source", "")) == "text-layer":
        text_recovery["text"] = ""
        text_recovery["confidence"] = 0.0
        image_block["text_recovery"] = text_recovery


def _filter_non_content_images(
    image_blocks: list[dict[str, Any]],
    figure_nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if not image_blocks:
        return [], figure_nodes, 0

    filtered_images: list[dict[str, Any]] = []
    removed_image_ids: set[str] = set()
    for image_block in image_blocks:
        bbox = tuple(float(value) for value in image_block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        caption_gap = image_block.get("caption_gap")
        no_nearby_context = not bool(image_block.get("nearby_context_blocks"))
        no_embedded_text = not _compact_text(str(image_block.get("embedded_text", "") or ""))
        top_margin_small_icon = bbox[3] <= 220.0 and max(width, height) <= 32.0
        if str(image_block.get("image_kind_guess", "") or "") == "publication_artifact":
            image_block["analysis_excluded"] = True
            image_block["analysis_excluded_reason"] = "publication_artifact"
            removed_image_ids.add(str(image_block.get("image_id", "")))
            continue
        if (
            top_margin_small_icon
            and no_nearby_context
            and no_embedded_text
            and isinstance(caption_gap, (int, float))
            and float(caption_gap) >= 120.0
            and not _looks_like_figure_caption(str(image_block.get("caption_text", "") or ""))
        ):
            image_block["analysis_excluded"] = True
            image_block["analysis_excluded_reason"] = "publication_artifact"
            removed_image_ids.add(str(image_block.get("image_id", "")))
            continue
        filtered_images.append(image_block)

    if not removed_image_ids:
        return image_blocks, figure_nodes, 0

    filtered_figures: list[dict[str, Any]] = []
    for figure in figure_nodes:
        if str(figure.get("image_id", "")) in removed_image_ids:
            figure["analysis_excluded"] = True
            figure["analysis_excluded_reason"] = "publication_artifact"
            continue
        filtered_figures.append(figure)
    return filtered_images, filtered_figures, len(removed_image_ids)


def _collect_nearby_image_context_blocks(
    image_bbox: tuple[float, float, float, float],
    text_blocks: list[dict[str, Any]],
    page_height: float,
    caption_text: str,
    embedded_text: str,
) -> list[dict[str, Any]]:
    caption_norm = _compact_text(caption_text)
    embedded_norm = _compact_text(embedded_text)
    candidates: list[tuple[float, int, dict[str, Any]]] = []

    for text_block in text_blocks:
        text = _clean_text(str(text_block.get("text", "")))
        if not text:
            continue
        if _is_footer_like_text_block(text_block, page_height):
            continue
        text_norm = _compact_text(text)
        if not text_norm or text_norm == caption_norm or text_norm == embedded_norm:
            continue

        text_bbox = tuple(float(item) for item in text_block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        overlap_ratio = _horizontal_overlap_ratio(image_bbox, text_bbox)
        if overlap_ratio < 0.12:
            continue

        relation = ""
        gap = 0.0
        if text_bbox[3] <= image_bbox[1]:
            relation = "above"
            gap = image_bbox[1] - text_bbox[3]
        elif text_bbox[1] >= image_bbox[3]:
            relation = "below"
            gap = text_bbox[1] - image_bbox[3]
        else:
            continue

        if gap > 160.0:
            continue

        context_block = {
            "text": text,
            "bbox": list(text_bbox),
            "block_id": text_block.get("block_id"),
            "relation": relation,
            "gap": round(gap, 3),
        }
        relation_priority = 0 if relation == "above" else 1
        candidates.append((gap, relation_priority, context_block))

    deduped_blocks: list[dict[str, Any]] = []
    seen_norms: set[str] = set()
    for _, _, block in sorted(candidates, key=lambda item: (item[0], item[1], item[2]["bbox"][1], item[2]["bbox"][0])):
        normalized = _compact_text(block.get("text", ""))
        if not normalized or normalized in seen_norms:
            continue
        seen_norms.add(normalized)
        deduped_blocks.append(block)
        if len(deduped_blocks) >= 3:
            break
    return deduped_blocks


def _build_image_content_segments(
    caption_text: str,
    embedded_text: str,
    text_recovery: dict[str, Any],
    nearby_context_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if caption_text:
        segments.append({"role": "caption", "text": caption_text})

    embedded_norm = _compact_text(embedded_text)
    caption_norm = _compact_text(caption_text)
    if embedded_text and embedded_norm != caption_norm:
        segments.append(
            {
                "role": "embedded_text",
                "text": embedded_text,
                "source": str(text_recovery.get("source", "none")),
                "confidence": round(float(text_recovery.get("confidence", 0.0) or 0.0), 3),
            }
        )

    for block in nearby_context_blocks:
        text = _clean_text(str(block.get("text", "")))
        if not text:
            continue
        segments.append(
            {
                "role": "nearby_context",
                "text": text,
                "relation": block.get("relation"),
                "gap": block.get("gap"),
                "block_id": block.get("block_id"),
            }
        )
    return segments


def _join_unique_segment_texts(segments: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    seen_norms: set[str] = set()
    for segment in segments:
        text = _clean_text(str(segment.get("text", "")))
        normalized = _compact_text(text)
        if not text or not normalized or normalized in seen_norms:
            continue
        seen_norms.add(normalized)
        parts.append(text)
    return "\n".join(parts).strip()


def _guess_image_kind(
    image_bbox: tuple[float, float, float, float],
    page_height: float,
    caption_text: str,
    embedded_text: str,
    text_recovery: dict[str, Any],
    nearby_context_blocks: list[dict[str, Any]],
) -> str:
    has_path = bool(text_recovery.get("has_path", False))
    embedded_length = len(_compact_text(embedded_text))
    has_caption = bool(_clean_text(caption_text))
    has_context = bool(nearby_context_blocks)

    if _is_publication_artifact_image(
        image_bbox=image_bbox,
        page_height=page_height,
        caption_text=caption_text,
        embedded_text=embedded_text,
        text_recovery=text_recovery,
        nearby_context_blocks=nearby_context_blocks,
    ):
        return "publication_artifact"

    if has_path:
        return "path_screenshot"
    if has_caption and embedded_length >= 20:
        return "captioned_textual_figure"
    if has_caption:
        return "captioned_figure"
    if embedded_length >= 20 and has_context:
        return "contextual_textual_image"
    if embedded_length >= 20:
        return "textual_image"
    return "graphic_image"

