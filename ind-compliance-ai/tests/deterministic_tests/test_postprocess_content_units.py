from __future__ import annotations

import unittest

from parsers.pdf.postprocess_content_units import (
    _annotate_content_evidence_order_from_document_ast,
    _build_content_units,
    _build_fact_extraction_corpus,
    _count_content_evidence_types,
    _count_content_unit_types,
)


class PostprocessContentUnitsTests(unittest.TestCase):
    def test_annotate_content_evidence_order_uses_document_ast_order_and_fallback_ids(self) -> None:
        evidence = [
            {"source_type": "table", "source_id": "tbl001"},
            {"source_type": "text", "source_id": "eq001"},
            {"source_type": "image", "source_id": "img001"},
            {"source_type": "text", "source_id": "txt_fallback"},
            {"source_type": "text", "source_id": "missing"},
        ]
        document_ast_pages = [
            {
                "page": 1,
                "blocks": [
                    {"block_type": "text", "block_id": "txt_fallback"},
                    {"block_type": "table", "table_id": "tbl001", "block_id": "tbl_block"},
                    {"block_type": "equation", "equation_id": "eq001", "block_id": "eq_block"},
                ],
            },
            {
                "page": 2,
                "blocks": [
                    {"block_type": "image", "image_id": "img001", "block_id": "img_block"},
                ],
            },
        ]

        _annotate_content_evidence_order_from_document_ast(evidence, document_ast_pages)

        self.assertEqual(evidence[0]["content_order_index"], 1)
        self.assertEqual(evidence[1]["content_order_index"], 2)
        self.assertEqual(evidence[2]["content_order_index"], 3)
        self.assertEqual(evidence[3]["content_order_index"], 0)
        self.assertNotIn("content_order_index", evidence[4])

    def test_build_content_units_flattens_segments_and_preserves_metadata(self) -> None:
        section_context = {"outline_index": "2.6.7", "section_title": "Toxicology"}
        units = _build_content_units(
            [
                {
                    "evidence_id": "ce_table_tbl001",
                    "source_type": "table",
                    "source_id": "tbl001",
                    "page": 2,
                    "bbox": [20.0, 80.0, 300.0, 180.0],
                    "semantic_role": "business_table",
                    "content_order_index": 2,
                    "segments": [
                        {"role": "title", "text": " Study Table ", "kind": "caption"},
                        {"role": "row", "text": "Dose: 10 mg/kg", "row_index": 1},
                    ],
                },
                {
                    "evidence_id": "ce_text_txt001",
                    "source_type": "text",
                    "source_id": "txt001",
                    "page": 2,
                    "bbox": [20.0, 40.0, 300.0, 60.0],
                    "semantic_role": "text_block",
                    "content_order_index": 1,
                    "section_context": section_context,
                    "segments": [{"role": "body", "text": " Summary paragraph "}],
                },
            ]
        )

        self.assertEqual([unit["text"] for unit in units], ["Summary paragraph", "Study Table", "Dose: 10 mg/kg"])
        self.assertEqual(units[0]["unit_id"], "cu_ce_text_txt001_001")
        self.assertEqual(units[0]["section_context"], section_context)
        self.assertIsNot(units[0]["section_context"], section_context)
        self.assertTrue(units[0]["fact_extraction_eligible"])
        self.assertTrue(units[1]["fact_extraction_eligible"])
        self.assertEqual(units[1]["attributes"], {"kind": "caption"})
        self.assertTrue(units[2]["fact_extraction_eligible"])

    def test_count_content_evidence_types_groups_empty_source_type_as_unknown(self) -> None:
        self.assertEqual(
            _count_content_evidence_types(
                [
                    {"source_type": "text"},
                    {"source_type": "table"},
                    {"source_type": "text"},
                    {"source_type": ""},
                    {},
                ]
            ),
            {"text": 2, "table": 1, "unknown": 2},
        )

    def test_count_content_unit_types_groups_empty_source_type_as_unknown(self) -> None:
        self.assertEqual(
            _count_content_unit_types(
                [
                    {"source_type": "toc"},
                    {"source_type": "text"},
                    {"source_type": "toc"},
                    {"source_type": None},
                ]
            ),
            {"toc": 2, "text": 1, "unknown": 1},
        )

    def test_build_fact_extraction_corpus_uses_eligible_units_and_compact_deduplication(self) -> None:
        corpus, used_count = _build_fact_extraction_corpus(
            [
                {"text": " Alpha  Beta ", "fact_extraction_eligible": True},
                {"text": "Alpha Beta", "fact_extraction_eligible": True},
                {"text": "Gamma", "fact_extraction_eligible": False},
                {"text": "Delta", "fact_extraction_eligible": True},
            ]
        )

        self.assertEqual(corpus, "Alpha Beta\nDelta")
        self.assertEqual(used_count, 2)


if __name__ == "__main__":
    unittest.main()
