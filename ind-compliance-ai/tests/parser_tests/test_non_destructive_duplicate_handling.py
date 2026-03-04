# Version: v1.0.11
# Optimization Summary:
# - Regression test for non-destructive duplicate-column handling.
# - Ensures first-column values are preserved when col0 == col1 patterns appear.

from __future__ import annotations

import unittest

from parsers.pdf.table_modules.normalization import (
    NormalizedCell,
    NormalizedRow,
    _detect_and_fix_duplicate_values,
)


class NonDestructiveDuplicateHandlingTests(unittest.TestCase):
    def test_preserve_first_column_values(self) -> None:
        grid = [
            ["序列", "相关序列"],
            ["0002", "0002"],
            ["0004", "0004"],
        ]
        rows = [
            NormalizedRow(logical_row=0, physical_row=0, cells=[]),
            NormalizedRow(
                logical_row=1,
                physical_row=1,
                cells=[
                    NormalizedCell(1, 0, 1, 0, 0, 1, text="0002"),
                    NormalizedCell(1, 1, 1, 1, 1, 1, text="0002"),
                ],
            ),
            NormalizedRow(
                logical_row=2,
                physical_row=2,
                cells=[
                    NormalizedCell(2, 0, 2, 0, 0, 1, text="0004"),
                    NormalizedCell(2, 1, 2, 1, 1, 1, text="0004"),
                ],
            ),
        ]

        _detect_and_fix_duplicate_values(rows, grid, logical_col_count=2)

        self.assertEqual(grid[1][0], "0002")
        self.assertEqual(grid[2][0], "0004")
        self.assertEqual(rows[1].cells[0].text, "0002")
        self.assertEqual(rows[2].cells[0].text, "0004")


if __name__ == "__main__":
    unittest.main()

