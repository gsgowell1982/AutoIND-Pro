# Version: v1.0.11
# Optimization Summary:
# - Verify supplement logic is non-destructive by default.
# - Keep optional writeback path available for experimental mode.

from __future__ import annotations

import unittest

from parsers.pdf.table_modules.normalization import (
    NormalizedCell,
    NormalizedRow,
    _supplement_missing_content,
)
from parsers.pdf.table_modules.raw_objects import RawSpan, RawTableEvidence


class SupplementPolicyTests(unittest.TestCase):
    def _base_rows_grid(self) -> tuple[list[NormalizedRow], list[list[str | None]]]:
        rows = [
            NormalizedRow(
                logical_row=0,
                physical_row=0,
                cells=[
                    NormalizedCell(0, 0, 0, 0, 0, 1, text=None),
                    NormalizedCell(0, 1, 0, 1, 1, 1, text="B"),
                ],
            ),
            NormalizedRow(
                logical_row=1,
                physical_row=1,
                cells=[
                    NormalizedCell(1, 0, 1, 0, 0, 1, text="C"),
                    NormalizedCell(1, 1, 1, 1, 1, 1, text="D"),
                ],
            ),
        ]
        grid: list[list[str | None]] = [[None, "B"], ["C", "D"]]
        return rows, grid

    def _raw_evidence(self) -> RawTableEvidence:
        span = RawSpan(text="A", x0=10, y0=10, x1=20, y1=20, size=10)
        return RawTableEvidence(
            page_number=1,
            bbox=(0, 0, 100, 100),
            physical_col_count=2,
            physical_row_count=2,
            spans=[span],
        )

    def test_default_non_destructive_mode(self) -> None:
        rows, grid = self._base_rows_grid()
        raw = self._raw_evidence()

        supplemented, candidates = _supplement_missing_content(
            rows=rows,
            grid=grid,
            raw_evidence=raw,
            column_mapping=[[0], [1]],
            logical_col_count=2,
            apply_writeback=False,
            enable_diagnostics=True,
            diagnostic_min_candidates_per_table=1,
            diagnostic_max_candidate_ratio=1.0,
            diagnostic_min_text_length=1,
            suppress_col0_diagnostics_for_continuation=False,
        )

        self.assertEqual(supplemented, 0)
        self.assertIsNone(grid[0][0])
        self.assertGreaterEqual(len(candidates), 1)

    def test_experimental_writeback_mode(self) -> None:
        rows, grid = self._base_rows_grid()
        raw = self._raw_evidence()

        supplemented, candidates = _supplement_missing_content(
            rows=rows,
            grid=grid,
            raw_evidence=raw,
            column_mapping=[[0], [1]],
            logical_col_count=2,
            apply_writeback=True,
            enable_diagnostics=False,
            diagnostic_min_candidates_per_table=1,
            diagnostic_max_candidate_ratio=1.0,
            diagnostic_min_text_length=1,
            suppress_col0_diagnostics_for_continuation=False,
        )

        self.assertEqual(supplemented, 1)
        self.assertEqual(grid[0][0], "A")
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()

