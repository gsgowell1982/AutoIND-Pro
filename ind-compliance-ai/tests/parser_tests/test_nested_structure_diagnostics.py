from __future__ import annotations

import unittest
from types import SimpleNamespace

from parsers.pdf.table_modules.continuum.nested import detect_nested_structure


class NestedStructureDiagnosticsTests(unittest.TestCase):
    def test_detects_high_confidence_suspected_nested_subtable(self) -> None:
        instance = SimpleNamespace(
            cells=[
                {
                    "row": 2,
                    "col": 2,
                    "rowspan": 2,
                    "colspan": 2,
                    "bbox": [120.0, 220.0, 280.0, 320.0],
                    "text": (
                        "Dose  Visit 1  Visit 2\n"
                        "10mg  pass     fail\n"
                        "20mg  fail     pass"
                    ),
                }
            ]
        )

        nested = detect_nested_structure(instance, {})

        self.assertTrue(nested.has_nested)
        self.assertEqual(nested.nesting_type, "suspected_subtable")
        self.assertEqual(len(nested.nested_regions), 1)
        region = nested.nested_regions[0]
        self.assertEqual(region["type"], "suspected_nested_subtable")
        self.assertEqual(region["row"], 2)
        self.assertEqual(region["col"], 2)
        self.assertEqual(region["bbox"], [120.0, 220.0, 280.0, 320.0])
        self.assertIn("merged_rows_and_cols", region["signals"])
        self.assertIn("line_internal_alignment", region["signals"])

    def test_does_not_flag_single_line_merged_title_cell_as_nested(self) -> None:
        instance = SimpleNamespace(
            cells=[
                {
                    "row": 1,
                    "col": 1,
                    "rowspan": 1,
                    "colspan": 4,
                    "bbox": [80.0, 100.0, 420.0, 118.0],
                    "text": "Table 5 Statistics of different triples in a sentence",
                }
            ]
        )

        nested = detect_nested_structure(instance, {})

        self.assertFalse(nested.has_nested)
        self.assertEqual(nested.nested_regions, [])
        self.assertIsNone(nested.nesting_type)


if __name__ == "__main__":
    unittest.main()
