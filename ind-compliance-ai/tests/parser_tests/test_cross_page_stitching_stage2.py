# Version: v1.0.8
# Optimization Summary:
# - Add stage-2 regression tests for cross-page table stitching policy.
# - Cover title conflict rejection, repeated-title continuation acceptance,
#   and low-similarity + low-overlap rejection.

from __future__ import annotations

import unittest

from parsers.pdf.table_modules.postprocess import stitch_cross_page_tables


def _table(
    table_id: str,
    page: int,
    bbox: tuple[float, float, float, float],
    col_sig: list[float],
    col_count: int,
    title: str = "",
) -> dict:
    return {
        "table_id": table_id,
        "page": page,
        "bbox": list(bbox),
        "column_signature": col_sig,
        "col_count": col_count,
        "title": title,
        "header": [],
    }


class CrossPageStitchingStage2Tests(unittest.TestCase):
    def test_reject_when_titles_conflict(self) -> None:
        prev_tbl = _table("tbl_001", 1, (50, 600, 560, 780), [0.25, 0.75], 2, title="Table A")
        curr_tbl = _table("tbl_002", 2, (52, 40, 558, 230), [0.25, 0.75], 2, title="Table B")
        tables = [prev_tbl, curr_tbl]
        links = stitch_cross_page_tables(tables, {1: 800.0, 2: 800.0})
        self.assertEqual(links, 0)
        self.assertNotIn("continued_from", curr_tbl)

    def test_allow_when_current_repeats_same_title(self) -> None:
        prev_tbl = _table("tbl_001", 1, (50, 600, 560, 780), [0.25, 0.75], 2, title="Table A")
        curr_tbl = _table("tbl_002", 2, (52, 40, 558, 230), [0.25, 0.75], 2, title="Table A")
        tables = [prev_tbl, curr_tbl]
        links = stitch_cross_page_tables(tables, {1: 800.0, 2: 800.0})
        self.assertEqual(links, 1)
        self.assertEqual(curr_tbl.get("continued_from"), "tbl_001")
        self.assertTrue(curr_tbl.get("is_continuation"))

    def test_reject_low_similarity_and_low_overlap(self) -> None:
        prev_tbl = _table("tbl_001", 1, (50, 600, 300, 780), [0.2, 0.8], 2, title="Table A")
        # Very small horizontal overlap with previous bbox.
        curr_tbl = _table("tbl_002", 2, (290, 40, 560, 230), [0.45, 0.9], 2, title="")
        tables = [prev_tbl, curr_tbl]
        links = stitch_cross_page_tables(tables, {1: 800.0, 2: 800.0})
        self.assertEqual(links, 0)
        self.assertNotIn("continued_from", curr_tbl)


if __name__ == "__main__":
    unittest.main()

