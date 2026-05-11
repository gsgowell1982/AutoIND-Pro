# Version: v1.0.2
# Optimization Summary:
# - Cover words-anchored visual-line reconstruction for fragmented same-line text.
# - Ensure reading order follows visual rows instead of raw bbox top values.
# - Reject reconstruction when fragments are too far apart despite token compatibility.
# - Guard semantic merges so footer artifacts do not contaminate caption/body
#   text without words-layer continuity.
# - Preserve numbered top-of-page section headings through semantic merging and
#   header/footer filtering under the unified margin-role rules.

from __future__ import annotations

import unittest

from parsers.pdf.layout import classify_text_block_layout_lane
from parsers.pdf.shared import _Word
from parsers.pdf.text_blocks import (
    _extract_page_text_and_images,
    _filter_header_footer_text_blocks,
    _merge_semantic_text_blocks,
    _sort_text_blocks_by_visual_rows,
    order_text_blocks_for_reading,
    reconstruct_visual_text_lines,
)


def _block(
    text: str,
    bbox: tuple[float, float, float, float],
    *,
    font_size: float,
    source_block_index: int,
) -> dict:
    return {
        "block_type": "text",
        "block_id": f"txt_p1_{source_block_index:03d}",
        "page": 1,
        "bbox": list(bbox),
        "text": text,
        "font_size": font_size,
        "source_block_index": source_block_index,
        "source": "text-layer",
    }


class TextBlockVisualReconstructionTests(unittest.TestCase):
    def test_extraction_uses_visible_spans_for_text_bbox_and_font_size(self) -> None:
        class _FakePage:
            def get_text(self, mode: str) -> dict:
                assert mode == "dict"
                return {
                    "blocks": [
                        {
                            "type": 0,
                            "bbox": [100.0, 100.0, 420.0, 150.0],
                            "lines": [
                                {
                                    "bbox": [132.75, 404.06, 482.25, 417.34],
                                    "spans": [
                                        {
                                            "text": "Office of Communication, Training and Manufacturers Assistance (HFM-40) ",
                                            "bbox": [132.75, 404.06, 482.25, 417.34],
                                            "size": 12.0,
                                        }
                                    ],
                                },
                                {
                                    "bbox": [191.25, 405.02, 427.69, 435.74],
                                    "spans": [
                                        {
                                            "text": "1401 Rockville Pike, Rockville, MD  20852-1448",
                                            "bbox": [191.25, 419.06, 420.75, 432.34],
                                            "size": 12.0,
                                        },
                                        {
                                            "text": " ",
                                            "bbox": [420.75, 405.02, 427.69, 435.74],
                                            "size": 27.75,
                                        },
                                    ],
                                },
                            ],
                        }
                    ]
                }

        text_blocks, image_blocks = _extract_page_text_and_images(_FakePage(), 1)

        self.assertEqual(image_blocks, [])
        self.assertEqual(len(text_blocks), 2)
        self.assertEqual(text_blocks[1]["text"], "1401 Rockville Pike, Rockville, MD 20852-1448")
        self.assertEqual(text_blocks[1]["bbox"], [191.25, 419.06, 420.75, 432.34])
        self.assertEqual(text_blocks[1]["font_size"], 12.0)
        self.assertGreater(text_blocks[1]["bbox"][1], text_blocks[0]["bbox"][3])

    def test_reconstruct_same_visual_line_from_words_continuity(self) -> None:
        blocks = [
            _block("eCTD", (154.34, 177.52, 261.65, 224.01), font_size=42.0, source_block_index=4),
            _block("技术规范", (261.65, 172.07, 472.58, 225.33), font_size=71.37, source_block_index=6),
        ]
        words = [_Word(154.34, 172.07, 472.58, 225.33, "eCTD技术规范")]

        reconstructed, merge_count = reconstruct_visual_text_lines(
            text_blocks=blocks,
            page_words=words,
            page_number=1,
            page_width=595.0,
        )

        self.assertEqual(merge_count, 1)
        self.assertEqual(len(reconstructed), 1)
        self.assertEqual(reconstructed[0]["text"], "eCTD技术规范")
        self.assertEqual(reconstructed[0]["source_block_indices"], [4, 6])
        self.assertTrue(reconstructed[0]["visual_line_reconstructed"])

    def test_visual_row_sorting_ignores_oversized_bbox_top_bias(self) -> None:
        blocks = [
            _block("技术规范", (261.65, 172.07, 472.58, 225.33), font_size=71.37, source_block_index=6),
            _block("eCTD", (154.34, 177.52, 261.65, 224.01), font_size=42.0, source_block_index=4),
            _block("国家药品监督管理局", (225.41, 601.57, 374.02, 619.24), font_size=15.96, source_block_index=18),
        ]

        ordered = _sort_text_blocks_by_visual_rows(blocks)

        self.assertEqual([block["text"] for block in ordered], ["eCTD", "技术规范", "国家药品监督管理局"])

    def test_do_not_reconstruct_far_apart_fragments(self) -> None:
        blocks = [
            _block("eCTD", (50.0, 180.0, 150.0, 220.0), font_size=42.0, source_block_index=4),
            _block("技术规范", (340.0, 176.0, 520.0, 224.0), font_size=71.37, source_block_index=6),
        ]
        words = [
            _Word(50.0, 176.0, 150.0, 224.0, "eCTD"),
            _Word(340.0, 176.0, 520.0, 224.0, "技术规范"),
        ]

        reconstructed, merge_count = reconstruct_visual_text_lines(
            text_blocks=blocks,
            page_words=words,
            page_number=1,
            page_width=595.0,
        )

        self.assertEqual(merge_count, 0)
        self.assertEqual(len(reconstructed), 2)

    def test_semantic_merge_preserves_visual_row_order_after_reconstruction(self) -> None:
        blocks = [
            _block("技术规范", (261.65, 172.07, 472.58, 225.33), font_size=71.37, source_block_index=6),
            _block("eCTD", (154.34, 177.52, 261.65, 224.01), font_size=42.0, source_block_index=4),
        ]
        words = [_Word(154.34, 172.07, 472.58, 225.33, "eCTD技术规范")]

        reconstructed, reconstruction_count = reconstruct_visual_text_lines(
            text_blocks=blocks,
            page_words=words,
            page_number=1,
            page_width=595.0,
        )
        merged, merge_count = _merge_semantic_text_blocks(reconstructed, page_number=1, page_width=595.0)

        self.assertEqual(reconstruction_count, 1)
        self.assertEqual(merge_count, 0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "eCTD技术规范")


    def test_semantic_merge_keeps_footer_page_number_out_of_caption_text(self) -> None:
        blocks = [
            _block("Figure 1: IND main folder", (224.25, 525.26, 591.75, 741.12), font_size=267.7, source_block_index=7),
            _block("Figure 1; IND main folder", (103.5, 688.31, 224.25, 701.59), font_size=12.0, source_block_index=5),
            _block("6", (294.0, 727.31, 302.44, 740.59), font_size=10.88, source_block_index=2),
        ]
        words = [
            _Word(103.5, 688.31, 132.71, 701.59, "Figure"),
            _Word(135.29, 688.31, 144.2, 701.59, "1;"),
            _Word(146.78, 688.31, 167.25, 701.59, "IND"),
            _Word(169.83, 688.31, 192.56, 701.59, "main"),
            _Word(195.14, 688.31, 221.67, 701.59, "folder"),
            _Word(224.25, 525.26, 312.45, 741.12, "Figure"),
            _Word(327.15, 525.26, 356.55, 741.12, "1:"),
            _Word(371.25, 525.26, 415.35, 741.12, "IND"),
            _Word(430.05, 525.26, 488.85, 741.12, "main"),
            _Word(503.55, 525.26, 591.75, 741.12, "folder"),
            _Word(294.0, 727.31, 300.0, 740.59, "6"),
        ]

        merged, merge_count = _merge_semantic_text_blocks(
            blocks,
            page_number=8,
            page_width=612.0,
            page_height=792.0,
            page_words=words,
        )

        self.assertEqual(merge_count, 1)
        self.assertEqual(len(merged), 2)
        self.assertFalse(any(block["text"] == "Figure 1: IND main folder 6" for block in merged))

        figure_blocks = [block for block in merged if "Figure 1" in block["text"]]
        self.assertEqual(len(figure_blocks), 1)
        self.assertEqual(figure_blocks[0]["text"], "Figure 1: IND main folder")
        self.assertGreater(figure_blocks[0]["bbox"][1], 680.0)
        self.assertLess(figure_blocks[0]["bbox"][3], 705.0)

        footer_blocks = [block for block in merged if block["text"] == "6"]
        self.assertEqual(len(footer_blocks), 1)

    def test_semantic_merge_allows_footer_number_with_true_same_line_continuity(self) -> None:
        blocks = [
            _block("1", (294.0, 727.31, 300.0, 740.59), font_size=10.88, source_block_index=2),
            _block("of 24", (300.5, 727.31, 330.0, 740.59), font_size=10.88, source_block_index=3),
        ]
        words = [
            _Word(294.0, 727.31, 300.0, 740.59, "1"),
            _Word(300.5, 727.31, 311.0, 740.59, "of"),
            _Word(314.0, 727.31, 330.0, 740.59, "24"),
        ]

        merged, merge_count = _merge_semantic_text_blocks(
            blocks,
            page_number=8,
            page_width=612.0,
            page_height=792.0,
            page_words=words,
        )

        self.assertEqual(merge_count, 1)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "1 of 24")

    def test_semantic_merge_keeps_numbered_top_heading_together(self) -> None:
        blocks = [
            _block("5.", (144.0, 73.31, 156.0, 86.59), font_size=12.0, source_block_index=2),
            _block("Publications", (180.0, 73.31, 237.0, 86.59), font_size=12.0, source_block_index=2),
        ]
        words = [
            _Word(144.0, 73.31, 153.0, 86.59, "5."),
            _Word(180.0, 73.31, 233.76, 86.59, "Publications"),
        ]

        merged, merge_count = _merge_semantic_text_blocks(
            blocks,
            page_number=13,
            page_width=612.0,
            page_height=792.0,
            page_words=words,
        )

        self.assertEqual(merge_count, 1)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "5. Publications")

    def test_filter_preserves_repeated_numbered_top_headings(self) -> None:
        page_payloads = [
            {
                "page_number": 13,
                "width": 612.0,
                "height": 792.0,
                "text_blocks": [
                    {
                        **_block("5. Publications", (144.0, 73.31, 237.0, 86.59), font_size=12.0, source_block_index=2),
                        "block_id": "txt_p13_001",
                    }
                ],
                "images": [],
                "tables": [],
            },
            {
                "page_number": 14,
                "width": 612.0,
                "height": 792.0,
                "text_blocks": [
                    {
                        **_block("2. Publications", (144.0, 73.31, 240.0, 86.59), font_size=12.0, source_block_index=2),
                        "block_id": "txt_p14_001",
                    }
                ],
                "images": [],
                "tables": [],
            },
        ]

        removed = _filter_header_footer_text_blocks(page_payloads)

        self.assertEqual(removed, 0)
        self.assertEqual(page_payloads[0]["text_blocks"][0]["text"], "5. Publications")
        self.assertEqual(page_payloads[1]["text_blocks"][0]["text"], "2. Publications")

    def test_reconstruct_does_not_merge_across_two_column_gutter(self) -> None:
        blocks = [
            _block("left column sentence", (56.7, 314.7, 292.8, 329.2), font_size=9.5, source_block_index=1),
            _block("right column sentence", (304.7, 314.6, 540.9, 329.2), font_size=9.5, source_block_index=2),
        ]
        words = [
            _Word(56.7, 314.7, 131.0, 329.2, "left"),
            _Word(134.0, 314.7, 192.0, 329.2, "column"),
            _Word(195.0, 314.7, 250.0, 329.2, "sentence"),
            _Word(304.7, 314.6, 391.0, 329.2, "right"),
            _Word(394.0, 314.6, 454.0, 329.2, "column"),
            _Word(457.0, 314.6, 540.9, 329.2, "sentence"),
        ]

        reconstructed, merge_count = reconstruct_visual_text_lines(
            text_blocks=blocks,
            page_words=words,
            page_number=1,
            page_width=595.0,
            layout_profile={
                "mode": "two_column",
                "page_width": 595.0,
                "column_mid": 298.0,
                "lane_tolerance": 20.0,
            },
        )

        self.assertEqual(merge_count, 0)
        self.assertEqual(len(reconstructed), 2)

    def test_reading_order_prefers_left_column_before_right_column_within_zone(self) -> None:
        heading = _block("Figure caption", (62.7, 278.2, 308.7, 288.1), font_size=9.0, source_block_index=1)
        heading["layout_lane"] = "left"
        heading["layout_mode"] = "two_column"
        left_1 = _block("Left row 1", (56.7, 314.7, 292.8, 329.2), font_size=9.5, source_block_index=2)
        left_1["layout_lane"] = "left"
        left_1["layout_mode"] = "two_column"
        right_1 = _block("Right row 1", (304.7, 314.6, 540.9, 329.2), font_size=9.5, source_block_index=3)
        right_1["layout_lane"] = "right"
        right_1["layout_mode"] = "two_column"
        left_2 = _block("Left row 2", (56.7, 326.7, 292.9, 341.2), font_size=9.5, source_block_index=4)
        left_2["layout_lane"] = "left"
        left_2["layout_mode"] = "two_column"
        right_2 = _block("Right row 2", (304.7, 326.6, 540.9, 341.2), font_size=9.5, source_block_index=5)
        right_2["layout_lane"] = "right"
        right_2["layout_mode"] = "two_column"
        footer = _block("Bottom note", (56.7, 700.0, 240.0, 712.0), font_size=9.0, source_block_index=6)
        footer["layout_lane"] = "full_width"
        footer["layout_mode"] = "mixed"

        ordered = order_text_blocks_for_reading(
            [right_1, left_2, heading, right_2, footer, left_1],
            {"mode": "mixed", "column_mid": 298.0, "lane_tolerance": 20.0},
        )

        self.assertEqual(
            [block["text"] for block in ordered],
            ["Figure caption", "Left row 1", "Left row 2", "Right row 1", "Right row 2", "Bottom note"],
        )

    def test_layout_lane_keeps_bottom_column_rows_out_of_full_width_merge_path(self) -> None:
        profile = {
            "mode": "two_column",
            "page_width": 595.276,
            "column_mid": 298.79,
            "lane_tolerance": 23.81,
            "body_top": 86.99,
            "body_bottom": 703.87,
        }
        left_bottom = _block(
            "et al. [13] proposed a dependency-driven relation extrac-",
            (56.69, 698.97, 290.55, 713.51),
            font_size=9.8,
            source_block_index=3,
        )
        right_bottom = _block(
            "model for overlapping relation extraction. Subsequently,",
            (304.72, 698.47, 540.9, 713.0),
            font_size=9.8,
            source_block_index=5,
        )
        centered_footer = _block(
            "6",
            (294.0, 727.31, 302.44, 740.59),
            font_size=10.88,
            source_block_index=7,
        )

        self.assertEqual(classify_text_block_layout_lane(left_bottom, profile), "left")
        self.assertEqual(classify_text_block_layout_lane(right_bottom, profile), "right")
        self.assertEqual(classify_text_block_layout_lane(centered_footer, profile), "full_width")


if __name__ == "__main__":
    unittest.main()
