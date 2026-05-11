# Version: v1.0.0
# Optimization Summary:
# - Cover contradiction-first same-page fragment merge decisions.
# - Reject local semantic barriers and conflicting titles before same-page merge.
# - Keep same-title merge permissive only when no local contradiction exists.

from __future__ import annotations

import unittest

from parsers.pdf.table_modules.postprocess import (
    can_merge_table_fragments_on_same_page,
    can_merge_tables_with_same_title,
    merge_same_page_table_fragments,
)


def _table(
    table_id: str,
    bbox: tuple[float, float, float, float],
    *,
    title: str = "",
    local_context_signal: str = "none",
    preceding_text_block: dict | None = None,
) -> dict:
    return {
        "table_id": table_id,
        "page": 1,
        "bbox": list(bbox),
        "column_signature": [0.25, 0.75],
        "col_count": 2,
        "row_count": 6,
        "header": [{"col": 1, "text": "Name"}, {"col": 2, "text": "Value"}],
        "cells": [{"row": 1, "col": 1, "logical_row": 1, "text": "A"}],
        "structure_score": 0.9,
        "row_texts": ["Name | Value"],
        "near_page_top": False,
        "near_page_bottom": False,
        "toc_row_ratio": 0.0,
        "title": title,
        "local_context_signal": local_context_signal,
        "preceding_text_block": preceding_text_block,
    }


class SamePageMergeSemanticTests(unittest.TestCase):
    def test_merge_same_page_fragments_when_secondary_has_running_header(self) -> None:
        primary = _table("tbl_001", (50.0, 100.0, 560.0, 220.0), title="Table A")
        secondary = _table(
            "tbl_002",
            (52.0, 223.0, 558.0, 340.0),
            title="Table A",
            local_context_signal="running_header",
            preceding_text_block={"text": "Template Header V1.0", "bbox": [40.0, 70.0, 560.0, 84.0]},
        )

        merged_tables, merge_count = merge_same_page_table_fragments([primary, secondary])

        self.assertEqual(merge_count, 1)
        self.assertEqual(len(merged_tables), 1)
        self.assertIn("tbl_002", merged_tables[0].get("merged_from", []))

    def test_reject_same_page_merge_when_secondary_has_narrative_barrier(self) -> None:
        primary = _table("tbl_001", (50.0, 100.0, 560.0, 220.0), title="Table A")
        secondary = _table(
            "tbl_002",
            (52.0, 223.0, 558.0, 340.0),
            title="Table A",
            local_context_signal="narrative_barrier",
            preceding_text_block={
                "text": "The following table summarizes a different submission case.",
                "bbox": [48.0, 222.0, 560.0, 230.0],
            },
        )

        self.assertFalse(can_merge_table_fragments_on_same_page(primary, secondary))

        merged_tables, merge_count = merge_same_page_table_fragments([primary, secondary])
        self.assertEqual(merge_count, 0)
        self.assertEqual(len(merged_tables), 2)

    def test_reject_same_page_merge_when_titles_conflict(self) -> None:
        primary = _table("tbl_001", (50.0, 100.0, 560.0, 220.0), title="Table A")
        secondary = _table("tbl_002", (52.0, 223.0, 558.0, 340.0), title="Table B")

        self.assertFalse(can_merge_table_fragments_on_same_page(primary, secondary))

    def test_reject_same_title_merge_when_local_barrier_exists(self) -> None:
        primary = _table("tbl_001", (50.0, 100.0, 560.0, 220.0), title="Table A")
        secondary = _table(
            "tbl_002",
            (52.0, 228.0, 558.0, 345.0),
            title="Table A",
            local_context_signal="new_table_title",
            preceding_text_block={"text": "Table A. Stability Results", "bbox": [50.0, 222.0, 260.0, 226.0]},
        )

        self.assertFalse(can_merge_tables_with_same_title(primary, secondary))

    def test_allow_same_title_merge_when_signal_is_continuation_title(self) -> None:
        primary = _table("tbl_001", (50.0, 100.0, 560.0, 220.0), title="Table A")
        secondary = _table(
            "tbl_002",
            (52.0, 228.0, 558.0, 345.0),
            title="Table A",
            local_context_signal="continuation_title",
            preceding_text_block={"text": "Table A (continued)", "bbox": [50.0, 222.0, 240.0, 226.0]},
        )

        self.assertTrue(can_merge_tables_with_same_title(primary, secondary))


if __name__ == "__main__":
    unittest.main()
