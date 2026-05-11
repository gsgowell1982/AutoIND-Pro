from __future__ import annotations

import unittest
from types import SimpleNamespace

from parsers.pdf.table_modules.continuum.semantic_repairs import repair_directory_listing_structure


class DirectoryListingSemanticRepairTests(unittest.TestCase):
    def test_folder_row_does_not_keep_duplicate_child_filename_list(self) -> None:
        grid = [
            ["app", None, "application folder"],
            ["seq", None, "sequence folder"],
            ["m1", None, "module folder"],
            ["dtd", "dtd\ncn-regional-1-0.xsd\nich-ectd-3-2.dtd\nxlink.xsd", "DTD and Schema folder"],
            [None, "cn-regional-1-0.xsd", "schema file"],
            [None, "ich-ectd-3-2.dtd", "dtd file"],
            [None, "xlink.xsd", "xlink schema"],
            ["style", None, "stylesheet folder"],
            [None, "style.xsl", "stylesheet file"],
            ["util", None, "utility folder"],
        ]
        rows = [
            SimpleNamespace(cells=[
                SimpleNamespace(logical_col=col_idx, text=value, supplemented=False, supplement_reason=None)
                for col_idx, value in enumerate(row)
            ])
            for row in grid
        ]
        raw_evidence = SimpleNamespace(
            physical_col_count=5,
            bbox=(0.0, 0.0, 300.0, 200.0),
            rows=[],
            spans=[],
            words=[],
        )

        changed = repair_directory_listing_structure(rows, grid, raw_evidence, logical_col_count=3)

        self.assertGreaterEqual(changed, 1)
        self.assertIsNone(grid[3][1])
        self.assertEqual(grid[4][1], "cn-regional-1-0.xsd")
        self.assertEqual(grid[5][1], "ich-ectd-3-2.dtd")
        self.assertEqual(grid[6][1], "xlink.xsd")
        self.assertIsNone(rows[3].cells[1].text)
        self.assertEqual(rows[3].cells[1].supplement_reason, "directory_embedded_child_filename_list_removed")


if __name__ == "__main__":
    unittest.main()
