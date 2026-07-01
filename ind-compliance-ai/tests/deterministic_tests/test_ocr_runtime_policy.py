from __future__ import annotations

import unittest
from unittest.mock import patch

from parsers.pdf.image_blocks import _demote_textual_image_blocks, _recover_text_from_image_region
from parsers.pdf.ocr_policy import (
    build_page_ocr_context,
    should_attempt_embedded_image_table_ocr,
    should_attempt_image_evidence_ocr,
    should_scan_embedded_image_table_regions,
)


class OcrRuntimePolicyTests(unittest.TestCase):
    def test_default_policy_skips_image_text_ocr_on_healthy_text_layer_page(self) -> None:
        context = build_page_ocr_context(
            text_blocks=[
                {"text": "This is a normal text-layer IND page with enough searchable body tokens."},
                {"text": "Tables and figures should use text layer evidence before any image OCR."},
            ],
            page_words=[],
            page_drawings=[],
            page_width=612.0,
            page_height=792.0,
        )

        with patch("parsers.pdf.image_blocks._ocr_text_from_clip", side_effect=AssertionError("OCR should be gated")):
            recovered = _recover_text_from_image_region(
                page=_PageWithoutText(),
                image_bbox=(100.0, 140.0, 460.0, 320.0),
                page_words=[],
                page_drawings=[],
                ocr_context=context,
            )

        self.assertEqual(recovered["source"], "ocr-skipped")
        self.assertEqual(recovered["text"], "")

    def test_benchmark_image_evidence_profile_recovers_ocr_as_image_owned_evidence_only(self) -> None:
        context = build_page_ocr_context(
            text_blocks=[
                {"text": "This page has enough searchable body tokens to remain text-layer dominant."},
                {"text": "A chart image may still need OCR evidence for evaluation projection."},
            ],
            page_words=[],
            page_drawings=[],
            page_width=612.0,
            page_height=792.0,
            runtime_profile="benchmark_image_evidence",
        )

        decision = should_attempt_image_evidence_ocr(context)

        self.assertTrue(decision.enabled)
        self.assertEqual(decision.reason, "benchmark_image_evidence_profile")
        with patch(
            "parsers.pdf.image_blocks._ocr_text_from_clip",
            return_value=("Q11 What factors influence your choice of print 0% 10% 20%", 0.82),
        ):
            recovered = _recover_text_from_image_region(
                page=_PageWithoutText(),
                image_bbox=(100.0, 140.0, 460.0, 320.0),
                page_words=[],
                page_drawings=[],
                ocr_context=context,
            )

        self.assertEqual(recovered["source"], "ocr-image-evidence")
        self.assertTrue(recovered["evidence_only"])
        self.assertIn("Q11", recovered["text"])

    def test_benchmark_image_evidence_ocr_is_not_demoted_into_body_text(self) -> None:
        context = build_page_ocr_context(
            text_blocks=[
                {"text": "This page has enough searchable body tokens to remain text-layer dominant."},
                {"text": "The chart image should preserve OCR as image evidence only."},
            ],
            page_words=[],
            page_drawings=[],
            page_width=612.0,
            page_height=792.0,
            runtime_profile="benchmark_image_evidence",
        )
        image_blocks = [
            {
                "block_type": "image",
                "image_id": "img_p1_001",
                "page": 1,
                "bbox": [100.0, 140.0, 460.0, 320.0],
            }
        ]

        with patch(
            "parsers.pdf.image_blocks._ocr_text_from_clip",
            return_value=("Q11 What factors influence your choice of print 0% 10% 20%", 0.82),
        ):
            text_blocks, kept_images, converted_count = _demote_textual_image_blocks(
                page=_PageWithoutText(),
                page_number=1,
                page_rect=_Rect(0.0, 0.0, 612.0, 792.0),
                image_blocks=image_blocks,
                text_blocks=[],
                page_words=[],
                page_drawings=[],
                ocr_context=context,
            )

        self.assertEqual(converted_count, 0)
        self.assertEqual(text_blocks, [])
        self.assertEqual(len(kept_images), 1)
        self.assertEqual(kept_images[0]["demote_guard"], "ocr_image_evidence_only")
        self.assertTrue((kept_images[0]["text_recovery"] or {}).get("evidence_only"))

    def test_default_policy_scans_embedded_image_table_regions_on_healthy_text_layer_page(self) -> None:
        context = build_page_ocr_context(
            text_blocks=[
                {"text": "This text layer has enough content to classify the page as searchable."},
                {"text": "A raster table image may still need table-only evidence recovery."},
            ],
            page_words=[],
            page_drawings=[],
            page_width=612.0,
            page_height=792.0,
        )

        full_table_ocr = should_attempt_embedded_image_table_ocr(context)
        region_scan = should_scan_embedded_image_table_regions(context)

        self.assertFalse(full_table_ocr.enabled)
        self.assertEqual(full_table_ocr.reason, "healthy_text_layer_text_first_profile")
        self.assertTrue(region_scan.enabled)
        self.assertEqual(region_scan.reason, "region_gated_embedded_table_scan")

    def test_benchmark_image_evidence_profile_preserves_region_gated_table_scan(self) -> None:
        context = build_page_ocr_context(
            text_blocks=[
                {"text": "This text layer has enough content to classify the page as searchable."},
                {"text": "Benchmark image evidence must not disable raster table region recovery."},
            ],
            page_words=[],
            page_drawings=[],
            page_width=612.0,
            page_height=792.0,
            runtime_profile="benchmark_image_evidence",
        )

        full_table_ocr = should_attempt_embedded_image_table_ocr(context)
        region_scan = should_scan_embedded_image_table_regions(context)

        self.assertFalse(full_table_ocr.enabled)
        self.assertEqual(full_table_ocr.reason, "healthy_text_layer_text_first_profile")
        self.assertTrue(region_scan.enabled)
        self.assertEqual(region_scan.reason, "region_gated_embedded_table_scan")

    def test_text_layer_only_profile_blocks_embedded_image_table_region_scan(self) -> None:
        context = build_page_ocr_context(
            text_blocks=[
                {"text": "This text layer has enough content to classify the page as searchable."},
                {"text": "The explicit text-layer-only profile disables OCR-backed evidence paths."},
            ],
            page_words=[],
            page_drawings=[],
            page_width=612.0,
            page_height=792.0,
            runtime_profile="text_layer_only",
        )

        decision = should_scan_embedded_image_table_regions(context)

        self.assertFalse(decision.enabled)
        self.assertEqual(decision.reason, "ocr_disabled_profile")

    def test_high_recall_policy_keeps_embedded_image_table_ocr_available(self) -> None:
        context = build_page_ocr_context(
            text_blocks=[
                {"text": "This text layer has enough content to classify the page as searchable."},
                {"text": "The high recall profile should keep raster table OCR available."},
            ],
            page_words=[],
            page_drawings=[],
            page_width=612.0,
            page_height=792.0,
            runtime_profile="scan_high_recall",
        )

        decision = should_attempt_embedded_image_table_ocr(context)

        self.assertTrue(decision.enabled)
        self.assertEqual(decision.reason, "high_recall_profile")

    def test_document_text_layer_profile_blocks_sparse_page_ocr_in_default_fast_profile(self) -> None:
        context = build_page_ocr_context(
            text_blocks=[],
            page_words=[],
            page_drawings=[{"rect": _Rect(0.0, 0.0, 500.0, 500.0)} for _ in range(100)],
            page_width=612.0,
            page_height=792.0,
            document_text_layer_dominant=True,
        )

        decision = should_attempt_embedded_image_table_ocr(context)

        self.assertFalse(decision.enabled)
        self.assertEqual(decision.reason, "text_layer_dominant_document_fast_profile")


class _PageWithoutText:
    def get_textbox(self, rect) -> str:  # noqa: ANN001
        return ""


class _Rect:
    def __init__(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


if __name__ == "__main__":
    unittest.main()
