from __future__ import annotations

import unittest

from parsers.common.atomic_fact_extractor import extract_atomic_facts
from parsers.pdf.postprocess import (
    _build_heading_section_anchor,
    _build_content_units,
    _build_fact_extraction_corpus,
    _build_text_content_evidence,
    _merge_page_nodes_with_text_reading_order,
)


class ContentUnitProjectionTests(unittest.TestCase):
    def test_heading_anchor_accepts_numbered_heading_without_trailing_dot(self) -> None:
        heading_anchor = _build_heading_section_anchor(
            {
                "block_type": "text",
                "text": "1.1 目的",
                "semantic_role": "section_heading",
                "page": 7,
                "bbox": [111.14, 121.43, 185.66, 137.39],
            },
            module_context=None,
        )

        self.assertIsNotNone(heading_anchor)
        self.assertEqual(heading_anchor["outline_index"], "1.1")
        self.assertEqual(heading_anchor["section_title"], "目的")
        self.assertEqual(heading_anchor["section_level"], 2)

    def test_content_units_flatten_segments_and_build_fact_corpus(self) -> None:
        content_evidence = [
            {
                "evidence_id": "ce_text_txt001",
                "source_type": "text",
                "source_id": "txt001",
                "page": 1,
                "bbox": [0.0, 0.0, 100.0, 20.0],
                "semantic_role": "text_block",
                "section_context": {
                    "module_label": "M3",
                    "outline_index": "3.2.S.4.1",
                    "section_title": "Specifications",
                    "anchor_source": "heading",
                    "anchor_confidence": 0.95,
                },
                "segments": [
                    {"role": "body", "text": "Drug Name: ExampleDrug"},
                ],
            },
            {
                "evidence_id": "ce_table_tbl001",
                "source_type": "table",
                "source_id": "tbl_001",
                "page": 1,
                "bbox": [0.0, 50.0, 300.0, 140.0],
                "semantic_role": "business_table",
                "segments": [
                    {"role": "title", "text": "Batch Inventory"},
                    {"role": "header", "text": "Batch Number | Strength"},
                    {"role": "row", "row_index": 1, "text": "Batch Number: B-001 | Strength: 100 mg"},
                ],
            },
            {
                "evidence_id": "ce_image_img001",
                "source_type": "image",
                "source_id": "img_001",
                "page": 1,
                "bbox": [0.0, 160.0, 200.0, 240.0],
                "semantic_role": "captioned_figure",
                "segments": [
                    {"role": "caption", "text": "Figure 1: Manufacturing overview"},
                    {"role": "embedded_text", "text": "Manufacturing Site: Site A", "source": "ocr", "confidence": 0.91},
                    {"role": "nearby_context", "text": "Dosage Form: Injection", "relation": "above", "gap": 12.0},
                ],
            },
            {
                "evidence_id": "ce_toc_toc001",
                "source_type": "toc",
                "source_id": "toc_001",
                "page": 1,
                "bbox": [0.0, 250.0, 200.0, 340.0],
                "semantic_role": "toc_outline",
                "segments": [
                    {"role": "title", "text": "TABLE OF CONTENTS"},
                    {"role": "entry", "entry_index": 1, "text": "Drug Name | 3"},
                ],
            },
            {
                "evidence_id": "ce_structure_template_st001",
                "source_type": "structure_template",
                "source_id": "structure_template_001",
                "page": 1,
                "bbox": [0.0, 342.0, 200.0, 348.0],
                "semantic_role": "structure_template",
                "segments": [
                    {"role": "title", "text": "Nonclinical outline template"},
                    {
                        "role": "entry",
                        "entry_index": 1,
                        "text": "2.6.3 Pharmacology",
                        "outline_index": "2.6.3",
                        "page_target": None,
                        "mapping_status": "template_no_page_target",
                    },
                ],
            },
            {
                "evidence_id": "ce_text_eq001",
                "source_type": "text",
                "source_id": "eq_001",
                "page": 1,
                "bbox": [0.0, 350.0, 200.0, 380.0],
                "semantic_role": "display_equation",
                "segments": [
                    {"role": "equation", "text": "fr(h, t) = rT(h 鈰唗) (1)", "equation_label": "(1)"},
                ],
            },
            {
                "evidence_id": "ce_algorithm_alg001",
                "source_type": "algorithm",
                "source_id": "alg_001",
                "page": 1,
                "bbox": [0.0, 390.0, 240.0, 470.0],
                "semantic_role": "algorithm_pseudocode",
                "segments": [
                    {"role": "title", "text": "Algorithm 1. Example procedure"},
                    {"role": "step", "text": "1. Initialize W"},
                    {"role": "step", "text": "2. Update W"},
                ],
            },
        ]

        units = _build_content_units(content_evidence)

        self.assertEqual(len(units), 15)
        self.assertEqual(units[0]["source_type"], "text")
        self.assertEqual(units[0]["unit_role"], "body")
        self.assertTrue(units[0]["fact_extraction_eligible"])
        self.assertEqual(units[0]["section_context"]["module_label"], "M3")
        self.assertEqual(units[0]["section_context"]["outline_index"], "3.2.S.4.1")

        toc_entry_unit = next(unit for unit in units if unit["source_type"] == "toc" and unit["unit_role"] == "entry")
        self.assertFalse(toc_entry_unit["fact_extraction_eligible"])
        self.assertEqual(toc_entry_unit["text"], "Drug Name | 3")

        template_entry_unit = next(
            unit
            for unit in units
            if unit["source_type"] == "structure_template" and unit["unit_role"] == "entry"
        )
        self.assertFalse(template_entry_unit["fact_extraction_eligible"])
        self.assertEqual(template_entry_unit["attributes"]["mapping_status"], "template_no_page_target")

        image_embedded_unit = next(
            unit
            for unit in units
            if unit["source_type"] == "image" and unit["unit_role"] == "embedded_text"
        )
        self.assertTrue(image_embedded_unit["fact_extraction_eligible"])
        self.assertEqual(image_embedded_unit["attributes"]["source"], "ocr")

        equation_unit = next(
            unit
            for unit in units
            if unit["semantic_role"] == "display_equation" and unit["unit_role"] == "equation"
        )
        self.assertFalse(equation_unit["fact_extraction_eligible"])
        self.assertEqual(equation_unit["attributes"]["equation_label"], "(1)")

        algorithm_step_unit = next(
            unit
            for unit in units
            if unit["source_type"] == "algorithm" and unit["unit_role"] == "step"
        )
        self.assertFalse(algorithm_step_unit["fact_extraction_eligible"])
        self.assertEqual(algorithm_step_unit["text"], "1. Initialize W")

        corpus, used_unit_count = _build_fact_extraction_corpus(units)
        self.assertEqual(used_unit_count, 7)
        self.assertIn("Drug Name: ExampleDrug", corpus)
        self.assertIn("Batch Inventory", corpus)
        self.assertIn("Batch Number: B-001 | Strength: 100 mg", corpus)
        self.assertIn("Manufacturing Site: Site A", corpus)
        self.assertIn("Dosage Form: Injection", corpus)
        self.assertNotIn("Drug Name | 3", corpus)
        self.assertNotIn("2.6.3 Pharmacology", corpus)
        self.assertNotIn("fr(h, t) = rT(h 鈰唗) (1)", corpus)
        self.assertNotIn("Algorithm 1. Example procedure", corpus)
        self.assertNotIn("1. Initialize W", corpus)

        facts = extract_atomic_facts(corpus)
        self.assertEqual(
            facts,
            {
                "drug_name": "ExampleDrug",
                "dosage_form": "Injection",
                "batch_number": "B-001",
                "manufacturing_site": "Site A",
                "strength": "100 mg",
            },
        )

    def test_numbered_body_points_keep_body_evidence_with_list_metadata(self) -> None:
        evidence = _build_text_content_evidence(
            {
                "block_type": "text",
                "block_id": "txt_list_001",
                "text": "1. The applicant should provide a description of the sequence and related submission files.",
                "page": 14,
                "bbox": [72.0, 180.0, 480.0, 196.0],
                "section_context": {
                    "outline_index": "4.1",
                    "section_title": "Administrative Information",
                    "anchor_source": "heading",
                },
            }
        )

        self.assertEqual(evidence["semantic_role"], "body_list_item")
        self.assertEqual(evidence["segments"][0]["role"], "body")
        self.assertEqual(evidence["segments"][0]["body_list_marker"], "1")
        self.assertEqual(evidence["segments"][0]["body_list_parent_outline_index"], "4.1")
        self.assertFalse(evidence["segments"][0]["outline_heading_candidate"])

        units = _build_content_units([evidence])
        self.assertEqual(units[0]["unit_role"], "body")
        self.assertEqual(units[0]["semantic_role"], "body_list_item")
        self.assertTrue(units[0]["fact_extraction_eligible"])
        self.assertEqual(units[0]["attributes"]["body_list_marker"], "1")
        self.assertEqual(units[0]["section_context"]["outline_index"], "4.1")

    def test_content_units_preserve_evidence_reading_order_before_physical_coordinates(self) -> None:
        content_evidence = [
            {
                "evidence_id": "ce_text_left_title",
                "source_type": "text",
                "source_id": "txt_left_title",
                "page": 1,
                "bbox": [90.0, 300.0, 250.0, 320.0],
                "semantic_role": "text_block",
                "segments": [{"role": "body", "text": "Left panel title"}],
            },
            {
                "evidence_id": "ce_text_left_body",
                "source_type": "text",
                "source_id": "txt_left_body",
                "page": 1,
                "bbox": [90.0, 350.0, 280.0, 365.0],
                "semantic_role": "text_block",
                "segments": [{"role": "body", "text": "Left panel body"}],
            },
            {
                "evidence_id": "ce_text_middle_title",
                "source_type": "text",
                "source_id": "txt_middle_title",
                "page": 1,
                "bbox": [370.0, 300.0, 550.0, 320.0],
                "semantic_role": "text_block",
                "segments": [{"role": "body", "text": "Middle panel title"}],
            },
            {
                "evidence_id": "ce_text_middle_body",
                "source_type": "text",
                "source_id": "txt_middle_body",
                "page": 1,
                "bbox": [370.0, 350.0, 570.0, 365.0],
                "semantic_role": "text_block",
                "segments": [{"role": "body", "text": "Middle panel body"}],
            },
        ]

        units = _build_content_units(content_evidence)

        self.assertEqual(
            [unit["text"] for unit in units],
            [
                "Left panel title",
                "Left panel body",
                "Middle panel title",
                "Middle panel body",
            ],
        )

    def test_document_ast_merge_preserves_text_reading_order_before_physical_coordinates(self) -> None:
        text_nodes = [
            {
                "block_type": "text",
                "block_id": "txt_left_title",
                "page": 1,
                "bbox": [90.0, 300.0, 250.0, 320.0],
                "text": "Left panel title",
                "layout_lane": "left",
                "reading_order_strategy": "local_visual_panels",
            },
            {
                "block_type": "text",
                "block_id": "txt_left_body",
                "page": 1,
                "bbox": [90.0, 350.0, 280.0, 365.0],
                "text": "Left panel body",
                "layout_lane": "left",
                "reading_order_strategy": "local_visual_panels",
            },
            {
                "block_type": "text",
                "block_id": "txt_middle_title",
                "page": 1,
                "bbox": [370.0, 300.0, 550.0, 320.0],
                "text": "Middle panel title",
                "layout_lane": "full_width",
                "reading_order_strategy": "local_visual_panels",
            },
            {
                "block_type": "text",
                "block_id": "txt_middle_body",
                "page": 1,
                "bbox": [370.0, 350.0, 570.0, 365.0],
                "text": "Middle panel body",
                "layout_lane": "full_width",
                "reading_order_strategy": "local_visual_panels",
            },
        ]

        merged = _merge_page_nodes_with_text_reading_order(
            text_nodes,
            [],
            {
                "mode": "mixed",
                "confidence": 0.9,
                "page_width": 960.0,
                "column_mid": 440.0,
                "lane_tolerance": 38.0,
            },
        )

        self.assertEqual(
            [node["text"] for node in merged],
            [
                "Left panel title",
                "Left panel body",
                "Middle panel title",
                "Middle panel body",
            ],
        )

    def test_document_ast_merge_keeps_text_reading_order_when_float_nodes_exist(self) -> None:
        text_nodes = [
            {
                "block_type": "text",
                "block_id": "txt_left_title",
                "page": 1,
                "bbox": [90.0, 300.0, 250.0, 320.0],
                "text": "Left panel title",
                "layout_lane": "left",
                "reading_order_strategy": "local_visual_panels",
            },
            {
                "block_type": "text",
                "block_id": "txt_left_body",
                "page": 1,
                "bbox": [90.0, 350.0, 280.0, 365.0],
                "text": "Left panel body",
                "layout_lane": "left",
                "reading_order_strategy": "local_visual_panels",
            },
            {
                "block_type": "text",
                "block_id": "txt_middle_title",
                "page": 1,
                "bbox": [370.0, 300.0, 550.0, 320.0],
                "text": "Middle panel title",
                "layout_lane": "full_width",
                "reading_order_strategy": "local_visual_panels",
            },
            {
                "block_type": "text",
                "block_id": "txt_middle_body",
                "page": 1,
                "bbox": [370.0, 350.0, 570.0, 365.0],
                "text": "Middle panel body",
                "layout_lane": "full_width",
                "reading_order_strategy": "local_visual_panels",
            },
        ]
        other_nodes = [
            {
                "block_type": "image",
                "block_id": "img_footer",
                "page": 1,
                "bbox": [500.0, 430.0, 700.0, 460.0],
                "image_id": "img_footer",
            }
        ]

        merged = _merge_page_nodes_with_text_reading_order(
            text_nodes,
            other_nodes,
            {
                "mode": "mixed",
                "confidence": 0.9,
                "page_width": 960.0,
                "column_mid": 440.0,
                "lane_tolerance": 38.0,
            },
        )

        self.assertEqual(
            [node.get("text") or node.get("block_id") for node in merged],
            [
                "Left panel title",
                "Left panel body",
                "Middle panel title",
                "Middle panel body",
                "img_footer",
            ],
        )

    def test_document_ast_merge_does_not_treat_two_column_y_reset_as_visual_panel(self) -> None:
        text_nodes = [
            {
                "block_type": "text",
                "block_id": "txt_left_title",
                "page": 1,
                "bbox": [72.0, 300.0, 260.0, 320.0],
                "text": "Left column heading above float",
                "layout_lane": "left",
            },
            {
                "block_type": "text",
                "block_id": "txt_left_body",
                "page": 1,
                "bbox": [72.0, 350.0, 270.0, 365.0],
                "text": "Left column body below float",
                "layout_lane": "left",
            },
            {
                "block_type": "text",
                "block_id": "txt_right_title",
                "page": 1,
                "bbox": [370.0, 300.0, 560.0, 320.0],
                "text": "Right column heading above float",
                "layout_lane": "right",
            },
            {
                "block_type": "text",
                "block_id": "txt_right_body",
                "page": 1,
                "bbox": [370.0, 350.0, 570.0, 365.0],
                "text": "Right column body below float",
                "layout_lane": "right",
            },
        ]
        other_nodes = [
            {
                "block_type": "image",
                "block_id": "img_full_width_between_rows",
                "page": 1,
                "bbox": [80.0, 330.0, 850.0, 340.0],
                "layout_lane": "full_width",
                "image_id": "img_full_width_between_rows",
            }
        ]

        merged = _merge_page_nodes_with_text_reading_order(
            text_nodes,
            other_nodes,
            {
                "mode": "mixed",
                "confidence": 0.9,
                "page_width": 960.0,
                "column_mid": 320.0,
                "lane_tolerance": 38.0,
            },
        )

        self.assertEqual(
            [node.get("text") or node.get("block_id") for node in merged],
            [
                "Left column heading above float",
                "Right column heading above float",
                "img_full_width_between_rows",
                "Left column body below float",
                "Right column body below float",
            ],
        )


if __name__ == "__main__":
    unittest.main()
