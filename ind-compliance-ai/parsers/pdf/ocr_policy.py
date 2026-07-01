from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any

from .settings import get_pdf_parser_settings
from .shared import _Word, _clean_text


@dataclass(frozen=True, slots=True)
class OcrPolicyDecision:
    enabled: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PageOcrContext:
    runtime_profile: str
    text_token_count: int
    word_token_count: int
    drawing_count: int
    drawing_area_ratio: float
    has_healthy_text_layer: bool
    has_sparse_text_layer: bool
    document_text_layer_dominant: bool = False


def current_ocr_runtime_profile() -> str:
    env_value = str(os.environ.get("PDF_OCR_RUNTIME_PROFILE", "") or "").strip().lower()
    if env_value:
        return env_value
    return str(get_pdf_parser_settings().ocr_runtime_policy.runtime_profile or "text_layer_ind_fast").strip().lower()


def build_page_ocr_context(
    *,
    text_blocks: list[dict[str, Any]],
    page_words: list[_Word],
    page_drawings: list[dict[str, Any]],
    page_width: float,
    page_height: float,
    runtime_profile: str | None = None,
    document_text_layer_dominant: bool = False,
) -> PageOcrContext:
    profile = str(runtime_profile or current_ocr_runtime_profile() or "text_layer_ind_fast").strip().lower()
    visible_text = _clean_text(" ".join(str(block.get("text", "")) for block in text_blocks))
    visible_word_text = _clean_text(" ".join(str(getattr(word, "text", "")) for word in page_words))
    text_token_count = _searchable_token_count(visible_text)
    word_token_count = _searchable_token_count(visible_word_text)
    drawing_area_ratio = _drawing_area_ratio(page_drawings, page_width, page_height)
    drawing_count = len(page_drawings or [])
    has_healthy_text_layer = text_token_count >= 16 or word_token_count >= 16
    has_sparse_text_layer = (text_token_count < 8 and word_token_count < 8) and (
        drawing_count >= 80 or drawing_area_ratio >= 0.08 or bool(visible_text)
    )
    return PageOcrContext(
        runtime_profile=profile,
        text_token_count=text_token_count,
        word_token_count=word_token_count,
        drawing_count=drawing_count,
        drawing_area_ratio=round(drawing_area_ratio, 4),
        has_healthy_text_layer=has_healthy_text_layer,
        has_sparse_text_layer=has_sparse_text_layer,
        document_text_layer_dominant=bool(document_text_layer_dominant),
    )


def should_attempt_full_page_ocr(context: PageOcrContext) -> OcrPolicyDecision:
    if context.runtime_profile in {"disabled", "text_layer_only"}:
        return OcrPolicyDecision(False, "ocr_disabled_profile")
    if context.runtime_profile in {"scan_high_recall", "benchmark_ocr"}:
        return OcrPolicyDecision(True, "high_recall_profile")
    if context.document_text_layer_dominant and context.runtime_profile in {"text_layer_ind_fast", "auto", "ind_review"}:
        return OcrPolicyDecision(False, "text_layer_dominant_document_fast_profile")
    if context.has_sparse_text_layer:
        return OcrPolicyDecision(True, "sparse_text_layer_visual_content")
    return OcrPolicyDecision(False, "text_layer_available")


def should_attempt_image_text_ocr(context: PageOcrContext | None) -> OcrPolicyDecision:
    if context is None:
        return OcrPolicyDecision(True, "legacy_no_context")
    if context.runtime_profile in {"disabled", "text_layer_only"}:
        return OcrPolicyDecision(False, "ocr_disabled_profile")
    if context.runtime_profile in {"scan_high_recall", "benchmark_ocr"}:
        return OcrPolicyDecision(True, "high_recall_profile")
    if context.document_text_layer_dominant and context.runtime_profile in {"text_layer_ind_fast", "auto", "ind_review"}:
        return OcrPolicyDecision(False, "text_layer_dominant_document_fast_profile")
    if context.has_sparse_text_layer:
        return OcrPolicyDecision(True, "sparse_text_layer_image_text_fallback")
    if context.has_healthy_text_layer:
        return OcrPolicyDecision(False, "healthy_text_layer_text_first_profile")
    return OcrPolicyDecision(True, "uncertain_text_layer")


def should_attempt_image_evidence_ocr(context: PageOcrContext | None) -> OcrPolicyDecision:
    """Decide whether OCR may be attached only as image-owned evidence.

    This branch is intentionally separate from generic image-text recovery.
    Generic recovery may demote a textual image into body text; evidence OCR
    stays owned by the figure/image node and is only consumed by explicit
    evidence/projection layers.
    """
    if context is None:
        return OcrPolicyDecision(False, "legacy_no_context")
    if context.runtime_profile in {"disabled", "text_layer_only"}:
        return OcrPolicyDecision(False, "ocr_disabled_profile")
    if context.runtime_profile in {"benchmark_image_evidence", "benchmark_neutral"}:
        return OcrPolicyDecision(True, "benchmark_image_evidence_profile")
    if context.runtime_profile in {"scan_high_recall", "benchmark_ocr"}:
        return OcrPolicyDecision(True, "high_recall_profile")
    return OcrPolicyDecision(False, "image_evidence_ocr_not_requested")


def should_attempt_embedded_image_table_ocr(context: PageOcrContext | None) -> OcrPolicyDecision:
    if context is None:
        return OcrPolicyDecision(True, "legacy_no_context")
    if context.runtime_profile in {"disabled", "text_layer_only"}:
        return OcrPolicyDecision(False, "ocr_disabled_profile")
    if context.runtime_profile in {"scan_high_recall", "benchmark_ocr"}:
        return OcrPolicyDecision(True, "high_recall_profile")
    if context.runtime_profile in {"benchmark_image_evidence", "benchmark_neutral"} and context.has_healthy_text_layer:
        return OcrPolicyDecision(False, "healthy_text_layer_text_first_profile")
    if context.document_text_layer_dominant and context.runtime_profile in {"text_layer_ind_fast", "auto", "ind_review"}:
        return OcrPolicyDecision(False, "text_layer_dominant_document_fast_profile")
    if context.has_sparse_text_layer:
        return OcrPolicyDecision(True, "sparse_text_layer_raster_table_fallback")
    if context.has_healthy_text_layer:
        return OcrPolicyDecision(False, "healthy_text_layer_text_first_profile")
    return OcrPolicyDecision(True, "uncertain_text_layer")


def should_scan_embedded_image_table_regions(context: PageOcrContext | None) -> OcrPolicyDecision:
    """Decide whether to run region-gated raster table discovery.

    A page can have a healthy searchable text layer while still containing a
    table as an embedded raster image. In that case generic image text OCR
    should remain disabled so figure internals do not pollute body flow, but
    the table evidence path may scan image regions with cheap visual gates and
    only OCR regions that look table-like.
    """
    if context is None:
        return OcrPolicyDecision(True, "legacy_no_context")
    if context.runtime_profile in {"disabled", "text_layer_only"}:
        return OcrPolicyDecision(False, "ocr_disabled_profile")
    if context.runtime_profile in {"scan_high_recall", "benchmark_ocr"}:
        return OcrPolicyDecision(True, "high_recall_profile")
    if context.runtime_profile in {"text_layer_ind_fast", "auto", "ind_review", "benchmark_image_evidence", "benchmark_neutral"}:
        return OcrPolicyDecision(True, "region_gated_embedded_table_scan")
    return should_attempt_embedded_image_table_ocr(context)


def _searchable_token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", _clean_text(text)))


def _drawing_area_ratio(page_drawings: list[dict[str, Any]], page_width: float, page_height: float) -> float:
    page_area = max(1.0, float(page_width or 0.0) * float(page_height or 0.0))
    drawing_area = 0.0
    for drawing in page_drawings or []:
        rect = drawing.get("rect")
        if rect is None:
            continue
        try:
            x0 = float(rect.x0)
            y0 = float(rect.y0)
            x1 = float(rect.x1)
            y1 = float(rect.y1)
        except Exception:
            continue
        drawing_area += min(page_area, max(0.0, x1 - x0) * max(0.0, y1 - y0))
    return drawing_area / page_area
