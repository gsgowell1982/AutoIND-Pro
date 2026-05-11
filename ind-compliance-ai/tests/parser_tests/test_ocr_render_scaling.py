from __future__ import annotations

import unittest

from parsers.pdf.table_vector_ocr import _compute_vector_ocr_render_scale
from parsers.pdf.text_blocks import _compute_body_ocr_render_scale


class OcrRenderScalingTests(unittest.TestCase):
    def test_body_ocr_keeps_high_resolution_for_typical_multilingual_row(self) -> None:
        scale = _compute_body_ocr_render_scale(255.0, 32.0)
        self.assertEqual(scale, 4.5)

    def test_body_ocr_keeps_high_resolution_for_taller_rows(self) -> None:
        scale = _compute_body_ocr_render_scale(260.0, 42.0)
        self.assertEqual(scale, 4.5)

    def test_body_ocr_caps_at_maximum_scale_for_small_rows(self) -> None:
        scale = _compute_body_ocr_render_scale(140.0, 20.0)
        self.assertEqual(scale, 4.5)

    def test_vector_ocr_uses_lower_scale_for_standard_multiline_region(self) -> None:
        scale = _compute_vector_ocr_render_scale(361.85, 226.49)
        self.assertEqual(scale, 3.5)

    def test_vector_ocr_keeps_short_wide_tables_at_full_resolution(self) -> None:
        scale = _compute_vector_ocr_render_scale(349.89, 56.18)
        self.assertEqual(scale, 4.0)


if __name__ == "__main__":
    unittest.main()
