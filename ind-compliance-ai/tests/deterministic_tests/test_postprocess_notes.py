from __future__ import annotations

import unittest

from parsers.pdf.postprocess_notes import (
    _drawing_bbox,
    _find_footnote_separator_line,
    _leading_marker_span,
    _local_note_marker_anchor_refs,
    _local_note_start,
    _looks_like_footnote_continuation_block,
    _looks_like_footnote_start_block,
    _looks_like_inline_footnote_ref_span,
    _normalize_structure_template_numbered_note_runs,
    _page_body_font_size,
    _template_numbered_note_start,
    _text_block_font_size,
)


class PostprocessNotesTests(unittest.TestCase):
    def test_local_symbol_note_start_extracts_marker_and_body(self) -> None:
        self.assertEqual(_local_note_start("a-未见不良反应剂量"), ("a", "未见不良反应剂量"))

    def test_numbered_template_note_start_accepts_ascii_and_fullwidth_markers(self) -> None:
        self.assertEqual(
            _template_numbered_note_start("备注：(1)国际非专利药品名称(INN)"),
            (1, "1", "国际非专利药品名称(INN)"),
        )
        self.assertEqual(
            _template_numbered_note_start("（2） 国际非专利药品名称(INN)"),
            (2, "2", "国际非专利药品名称(INN)"),
        )

    def test_numbered_note_normalization_preserves_continuation_after_ordered_note(self) -> None:
        template = {
            "structure_template_id": "synthetic_template",
            "note_blocks": [
                {"text": "备注：(1)第一条说明", "bbox": [1, 10, 100, 20]},
                {"text": "第一条说明的续行", "bbox": [1, 22, 100, 32]},
                {"text": "(2)第二条说明", "bbox": [1, 40, 100, 50]},
            ],
        }

        _normalize_structure_template_numbered_note_runs(template)

        self.assertEqual(
            [note.get("text") for note in template["note_blocks"]],
            ["备注：(1)第一条说明", "第一条说明的续行", "(2)第二条说明"],
        )

    def test_local_note_marker_anchor_refs_links_marker_to_row(self) -> None:
        refs = _local_note_marker_anchor_refs(
            ["NOAELa"],
            [{"marker": "a", "text": "a-未见不良反应剂量", "source_block_id": "note-1"}],
        )

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["marker"], "a")
        self.assertEqual(refs[0]["anchor_text"], "NOAELa")
        self.assertEqual(refs[0]["note_block_id"], "note-1")

    def test_footnote_start_and_continuation_use_position_and_small_font(self) -> None:
        body_blocks = [
            {"block_type": "text", "text": "Body text", "bbox": [40, 120, 500, 140], "font_size": 10.0},
            {"block_type": "text", "text": "More body", "bbox": [40, 220, 500, 240], "font_size": 10.0},
            {"block_type": "text", "text": "1 footnote body", "bbox": [40, 720, 260, 732], "font_size": 7.0},
        ]
        body_font_size = _page_body_font_size(body_blocks, 800.0)
        footnote = {
            "block_type": "text",
            "text": "1 footnote body",
            "bbox": [40, 720, 260, 732],
            "font_size": 7.0,
            "spans": [{"text": "1", "font_size": 5.0, "bbox": [40, 719, 44, 725]}],
        }
        continuation = {
            "block_type": "text",
            "text": "continued footnote text",
            "bbox": [46, 735, 300, 747],
            "font_size": 7.1,
        }

        self.assertEqual(body_font_size, 10.0)
        self.assertEqual(_text_block_font_size({"font_size": "7.5"}), 7.5)
        self.assertEqual(_leading_marker_span(footnote, "1"), footnote["spans"][0])
        self.assertEqual(
            _looks_like_footnote_start_block(
                footnote,
                page_height=800.0,
                body_font_size=body_font_size,
            ),
            (True, "1", "footnote body"),
        )
        self.assertTrue(
            _looks_like_footnote_continuation_block(
                continuation,
                footnote,
                page_height=800.0,
                body_font_size=body_font_size,
            )
        )

    def test_inline_footnote_reference_span_requires_superscript_marker(self) -> None:
        marker_span = {"text": "1", "font_size": 6.0, "bbox": [110, 98, 114, 104]}
        text_block = {
            "text": "Compound1 text",
            "bbox": [40, 100, 200, 116],
            "font_size": 10.0,
            "spans": [
                {"text": "Compound", "font_size": 10.0, "bbox": [40, 101, 108, 115]},
                marker_span,
                {"text": " text", "font_size": 10.0, "bbox": [116, 101, 170, 115]},
            ],
        }

        self.assertTrue(
            _looks_like_inline_footnote_ref_span(
                text_block,
                marker_span,
                "1",
                block_font_size=10.0,
            )
        )
        self.assertFalse(
            _looks_like_inline_footnote_ref_span(
                text_block,
                text_block["spans"][0],
                "Compound",
                block_font_size=10.0,
            )
        )

    def test_drawing_bbox_and_footnote_separator_line_contract(self) -> None:
        drawing = {"bbox": [60, 680, 190, 681]}
        separator = _find_footnote_separator_line(
            [drawing, {"bbox": [10, 200, 550, 205]}],
            page_width=600.0,
            page_height=800.0,
            footnote_bbox=[60, 705, 300, 730],
        )

        self.assertEqual(_drawing_bbox(drawing), [60.0, 680.0, 190.0, 681.0])
        self.assertEqual(separator["bbox"], [60.0, 680.0, 190.0, 681.0])
        self.assertEqual(separator["source"], "page_drawing")


if __name__ == "__main__":
    unittest.main()
