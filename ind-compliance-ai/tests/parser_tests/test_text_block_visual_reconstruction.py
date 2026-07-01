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

from parsers.pdf.layout import (
    annotate_text_blocks_with_layout,
    classify_text_block_layout_lane,
    infer_page_text_layout_profile,
)
from parsers.pdf.shared import _Word
from parsers.pdf.text_blocks import (
    _extract_page_text_and_images,
    _filter_header_footer_text_blocks,
    _merge_semantic_text_blocks,
    _sort_text_blocks_by_visual_rows,
    build_reading_order_diagnostics,
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

    def test_mixed_reading_order_restarts_columns_after_full_width_separator(self) -> None:
        top_heading = _block("Full width article heading", (54.0, 88.0, 540.0, 104.0), font_size=12.0, source_block_index=1)
        top_heading["layout_lane"] = "full_width"
        left_a1 = _block("Left A1", (56.0, 120.0, 280.0, 132.0), font_size=9.0, source_block_index=2)
        left_a1["layout_lane"] = "left"
        right_a1 = _block("Right A1", (314.0, 120.0, 540.0, 132.0), font_size=9.0, source_block_index=3)
        right_a1["layout_lane"] = "right"
        left_a2 = _block("Left A2", (56.0, 136.0, 280.0, 148.0), font_size=9.0, source_block_index=4)
        left_a2["layout_lane"] = "left"
        right_a2 = _block("Right A2", (314.0, 136.0, 540.0, 148.0), font_size=9.0, source_block_index=5)
        right_a2["layout_lane"] = "right"
        table_caption = _block("Full width table caption", (54.0, 180.0, 540.0, 194.0), font_size=9.0, source_block_index=6)
        table_caption["layout_lane"] = "full_width"
        left_b1 = _block("Left B1", (56.0, 214.0, 280.0, 226.0), font_size=9.0, source_block_index=7)
        left_b1["layout_lane"] = "left"
        right_b1 = _block("Right B1", (314.0, 214.0, 540.0, 226.0), font_size=9.0, source_block_index=8)
        right_b1["layout_lane"] = "right"

        ordered = order_text_blocks_for_reading(
            [right_b1, right_a2, left_a1, table_caption, right_a1, top_heading, left_b1, left_a2],
            {"mode": "mixed", "page_width": 595.0, "column_mid": 298.0, "lane_tolerance": 20.0},
        )

        self.assertEqual(
            [block["text"] for block in ordered],
            [
                "Full width article heading",
                "Left A1",
                "Left A2",
                "Right A1",
                "Right A2",
                "Full width table caption",
                "Left B1",
                "Right B1",
            ],
        )

    def test_reading_order_groups_local_three_panel_visual_text_before_rowwise_flattening(self) -> None:
        intro = _block("Summary", (48.0, 80.0, 760.0, 100.0), font_size=16.0, source_block_index=1)
        intro["layout_lane"] = "full_width"
        left_title = _block("Higher Return", (95.0, 270.0, 250.0, 286.0), font_size=14.0, source_block_index=2)
        left_title["layout_lane"] = "left"
        middle_title = _block("Reduced Time", (370.0, 270.0, 560.0, 286.0), font_size=14.0, source_block_index=3)
        middle_title["layout_lane"] = "full_width"
        right_title = _block("Current Best", (660.0, 270.0, 820.0, 286.0), font_size=14.0, source_block_index=4)
        right_title["layout_lane"] = "right"
        left_body_1 = _block("Left explanation line one", (95.0, 315.0, 300.0, 327.0), font_size=10.0, source_block_index=5)
        left_body_1["layout_lane"] = "left"
        middle_body_1 = _block("Middle explanation line one", (370.0, 315.0, 580.0, 327.0), font_size=10.0, source_block_index=6)
        middle_body_1["layout_lane"] = "full_width"
        right_body_1 = _block("Right explanation line one", (660.0, 315.0, 860.0, 327.0), font_size=10.0, source_block_index=7)
        right_body_1["layout_lane"] = "right"
        left_body_2 = _block("Left explanation line two", (95.0, 332.0, 290.0, 344.0), font_size=10.0, source_block_index=8)
        left_body_2["layout_lane"] = "left"
        middle_body_2 = _block("Middle explanation line two", (370.0, 332.0, 575.0, 344.0), font_size=10.0, source_block_index=9)
        middle_body_2["layout_lane"] = "full_width"
        right_body_2 = _block("Right explanation line two", (660.0, 332.0, 850.0, 344.0), font_size=10.0, source_block_index=10)
        right_body_2["layout_lane"] = "right"
        note = _block("1 Footnote for all panels", (48.0, 470.0, 620.0, 486.0), font_size=8.0, source_block_index=11)
        note["layout_lane"] = "left"

        ordered = order_text_blocks_for_reading(
            [
                right_body_1,
                middle_body_2,
                intro,
                left_body_2,
                right_title,
                middle_title,
                left_title,
                note,
                right_body_2,
                middle_body_1,
                left_body_1,
            ],
            {
                "mode": "mixed",
                "confidence": 0.9,
                "page_width": 960.0,
                "column_mid": 440.0,
                "lane_tolerance": 38.0,
            },
        )

        self.assertEqual(
            [block["text"] for block in ordered],
            [
                "Summary",
                "Higher Return",
                "Left explanation line one",
                "Left explanation line two",
                "Reduced Time",
                "Middle explanation line one",
                "Middle explanation line two",
                "Current Best",
                "Right explanation line one",
                "Right explanation line two",
                "1 Footnote for all panels",
            ],
        )

    def test_reading_order_diagnostics_describe_two_column_zones_without_reordering_text(self) -> None:
        heading = _block("Full width heading", (54.0, 88.0, 540.0, 104.0), font_size=12.0, source_block_index=1)
        heading["layout_lane"] = "full_width"
        left_1 = _block("Left 1", (56.0, 120.0, 280.0, 132.0), font_size=9.0, source_block_index=2)
        left_1["layout_lane"] = "left"
        right_1 = _block("Right 1", (314.0, 120.0, 540.0, 132.0), font_size=9.0, source_block_index=3)
        right_1["layout_lane"] = "right"
        left_2 = _block("Left 2", (56.0, 136.0, 280.0, 148.0), font_size=9.0, source_block_index=4)
        left_2["layout_lane"] = "left"
        right_2 = _block("Right 2", (314.0, 136.0, 540.0, 148.0), font_size=9.0, source_block_index=5)
        right_2["layout_lane"] = "right"

        diagnostics = build_reading_order_diagnostics(
            [right_2, heading, right_1, left_2, left_1],
            {"mode": "mixed", "confidence": 0.91, "page_width": 595.0, "column_mid": 298.0, "lane_tolerance": 20.0},
        )

        self.assertEqual(diagnostics["strategy"], "zone_columns_left_then_right")
        self.assertEqual(diagnostics["zone_count"], 2)
        self.assertEqual(diagnostics["column_zone_count"], 1)
        self.assertEqual(diagnostics["full_width_zone_count"], 1)
        self.assertEqual(diagnostics["left_block_count"], 2)
        self.assertEqual(diagnostics["right_block_count"], 2)
        self.assertFalse(diagnostics["review_required"])

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

    def test_unbalanced_reference_columns_still_infer_column_lanes(self) -> None:
        left_blocks = [
            _block(f"Left reference line {idx}", (70.0, 80.0 + idx * 20.0, 289.0, 90.0 + idx * 20.0), font_size=8.0, source_block_index=idx)
            for idx in range(10)
        ]
        right_blocks = [
            _block(f"Right reference line {idx}", (316.0, 80.0 + idx * 20.0, 524.0, 90.0 + idx * 20.0), font_size=8.0, source_block_index=20 + idx)
            for idx in range(4)
        ]
        # Deliberately unbalanced word counts reproduce a common references-page
        # shape: one column continues much farther than the other, but the
        # geometry still has two stable separated text lanes.
        words = []
        for block in left_blocks:
            bbox = block["bbox"]
            for token_index in range(5):
                x0 = bbox[0] + token_index * 34.0
                words.append(_Word(x0, bbox[1], x0 + 20.0, bbox[3], f"L{token_index}"))
        for block in right_blocks:
            bbox = block["bbox"]
            for token_index in range(3):
                x0 = bbox[0] + token_index * 42.0
                words.append(_Word(x0, bbox[1], x0 + 24.0, bbox[3], f"R{token_index}"))

        blocks = [*left_blocks, *right_blocks]
        profile = infer_page_text_layout_profile(
            text_blocks=blocks,
            page_words=words,
            page_width=595.0,
            page_height=842.0,
        )
        annotate_text_blocks_with_layout(blocks, profile)

        self.assertIn(profile["mode"], {"two_column", "mixed"})
        self.assertEqual(classify_text_block_layout_lane(left_blocks[0], profile), "left")
        self.assertEqual(classify_text_block_layout_lane(right_blocks[0], profile), "right")

        reconstructed, reconstruction_count = reconstruct_visual_text_lines(
            text_blocks=[left_blocks[0], right_blocks[0]],
            page_words=words,
            page_number=1,
            page_width=595.0,
            layout_profile=profile,
        )

        self.assertEqual(reconstruction_count, 0)
        self.assertEqual(len(reconstructed), 2)

    def test_asymmetric_table_matrix_does_not_trigger_text_column_fallback(self) -> None:
        blocks = [
            _block("Jurisdiction GATS XVII", (77.0, 72.0, 210.0, 85.0), font_size=8.0, source_block_index=1),
            _block("Foreign", (219.0, 72.0, 261.0, 85.0), font_size=8.0, source_block_index=2),
            _block("Reservation Ownership", (148.0, 86.0, 279.0, 99.0), font_size=8.0, source_block_index=3),
            _block("Restrictions on Foreign", (288.0, 72.0, 408.0, 85.0), font_size=8.0, source_block_index=4),
            _block("Reporting", (453.0, 99.0, 507.0, 112.0), font_size=8.0, source_block_index=5),
            _block("Requirements", (453.0, 113.0, 530.0, 126.0), font_size=8.0, source_block_index=13),
            _block("Finland N", (77.0, 195.0, 161.0, 208.0), font_size=8.0, source_block_index=6),
            _block("Y", (219.0, 195.0, 230.0, 208.0), font_size=8.0, source_block_index=7),
            _block("Prior approval for a foreigner purchase", (356.0, 195.0, 512.0, 208.0), font_size=8.0, source_block_index=8),
            _block("France", (77.0, 346.0, 113.0, 359.0), font_size=8.0, source_block_index=9),
            _block("N", (148.0, 346.0, 161.0, 359.0), font_size=8.0, source_block_index=10),
            _block("Y", (219.0, 346.0, 230.0, 359.0), font_size=8.0, source_block_index=11),
            _block("None.", (356.0, 346.0, 388.0, 359.0), font_size=8.0, source_block_index=12),
            _block("long table cell continuation line", (356.0, 374.0, 512.0, 387.0), font_size=8.0, source_block_index=14),
            _block("another wrapped explanation line", (356.0, 388.0, 512.0, 401.0), font_size=8.0, source_block_index=15),
            _block("more wrapped explanation line", (356.0, 402.0, 512.0, 415.0), font_size=8.0, source_block_index=16),
        ]
        words = []
        for block in blocks:
            bbox = block["bbox"]
            token_count = max(1, len(block["text"].split()))
            for token_index in range(token_count):
                x0 = bbox[0] + token_index * 28.0
                words.append(_Word(x0, bbox[1], min(x0 + 20.0, bbox[2]), bbox[3], f"T{token_index}"))

        profile = infer_page_text_layout_profile(
            text_blocks=blocks,
            page_words=words,
            page_width=612.0,
            page_height=792.0,
        )

        self.assertEqual(profile["mode"], "single_column")


if __name__ == "__main__":
    unittest.main()
