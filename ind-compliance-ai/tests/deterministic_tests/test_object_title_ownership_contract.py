from __future__ import annotations

import unittest

from parsers.pdf.postprocess import (
    _build_image_composite_object,
    _build_structure_template_composite_object,
    _build_table_composite_object,
)


class ObjectTitleOwnershipContractTests(unittest.TestCase):
    def test_composite_objects_separate_visible_title_owner_from_metadata_references(self) -> None:
        objects = [
            _build_table_composite_object(
                {
                    "table_id": "tbl_001",
                    "semantic_role": "business_table",
                    "title": "2.6.7.17 Other toxicity summary",
                    "semantic_grid": [["Species", "Result"], ["Rat", "No finding"]],
                }
            ),
            _build_structure_template_composite_object(
                {
                    "block_type": "structure_template",
                    "structure_template_id": "structure_template_001",
                    "title": "2.6.7 Study summary",
                    "row_texts": ["Study", "Result"],
                }
            ),
            _build_image_composite_object(
                {
                    "image_id": "img_001",
                    "title": "Figure 1 Study design",
                    "content_segments": [{"role": "embedded_text", "text": "Dose groups"}],
                }
            ),
        ]

        for composite in objects:
            with self.subTest(object_family=composite.get("object_family")):
                self.assertEqual(composite.get("title_policy"), "owned_object_title")
                self.assertEqual(composite.get("visible_title_owner"), composite.get("ownership_domain"))
                self.assertEqual(
                    composite.get("metadata_title_reference_policy"),
                    "may_reference_without_visible_rendering",
                )
                self.assertIn("title", composite.get("presentation_order") or [])


if __name__ == "__main__":
    unittest.main()
