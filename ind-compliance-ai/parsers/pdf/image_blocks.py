from __future__ import annotations

import io
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

from .layout import _bbox_contains_path, _words_in_bbox, _words_to_text
from .shared import (
    _Word,
    _bbox_area,
    _bbox_intersection_ratio,
    _bbox_to_list,
    _clean_text,
    _compact_text,
    _horizontal_overlap_ratio,
    _text_contains_text,
)


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

