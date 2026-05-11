# Version: v1.0.0
# Optimization Summary:
# - Verify fully empty logical rows are removed from business-facing table rows.
# - Preserve raw audit rows and raw row_texts for traceability.
# - Ensure postprocess rebuild keeps semantic and raw row views in sync.

from __future__ import annotations

import unittest
from types import SimpleNamespace

from parsers.pdf.table_modules.assembly import assemble_table_instance
from parsers.pdf.table_modules.ast import build_logical_ast
from parsers.pdf.table_modules.normalization import (
    NormalizedCell,
    NormalizedRow,
    NormalizedTable,
)
from parsers.pdf.table_modules.postprocess import _refresh_row_texts_from_grid


def _cell(
    logical_row: int,
    logical_col: int,
    physical_row: int,
    physical_col: int,
    text: str | None,
) -> NormalizedCell:
    return NormalizedCell(
        logical_row=logical_row,
        logical_col=logical_col,
        physical_row=physical_row,
        physical_col_start=physical_col,
        physical_col_end=physical_col,
        physical_colspan=1,
        text=text,
    )


def _row(
    logical_row: int,
    physical_row: int,
    values: list[str | None],
) -> NormalizedRow:
    return NormalizedRow(
        logical_row=logical_row,
        physical_row=physical_row,
        cells=[
            _cell(logical_row, col_idx, physical_row, col_idx, value)
            for col_idx, value in enumerate(values)
        ],
    )


def _normalized_table_with_empty_middle_row() -> NormalizedTable:
    grid = [
        ["m1", None, "desc-1"],
        [None, None, None],
        ["m2", None, "desc-2"],
    ]
    rows = [_row(idx, idx, values) for idx, values in enumerate(grid)]
    return NormalizedTable(
        page_number=21,
        bbox=(0.0, 0.0, 300.0, 120.0),
        logical_col_count=3,
        logical_row_count=3,
        physical_col_count=3,
        physical_row_count=3,
        rows=rows,
        grid=grid,
        near_page_top=True,
        near_page_bottom=False,
    )


class SemanticRowFilteringTests(unittest.TestCase):
    def test_assembly_filters_empty_rows_but_preserves_raw_view(self) -> None:
        instance = assemble_table_instance(
            normalized=_normalized_table_with_empty_middle_row(),
            table_id="tbl_001",
            parent_header=[
                {"col": 1, "text": "Folder"},
                {"col": 2, "text": "File"},
                {"col": 3, "text": "Description"},
            ],
            assessment=SimpleNamespace(is_continuation=True),
        )

        self.assertEqual(instance.raw_row_count, 3)
        self.assertEqual(instance.row_count, 2)
        self.assertEqual(instance.data_row_count, 2)
        self.assertEqual(instance.structural_empty_rows, [2])
        self.assertEqual(instance.raw_row_texts[1], "null | null | null")
        self.assertEqual(instance.row_texts, ["m1 | null | desc-1", "m2 | null | desc-2"])
        self.assertEqual(instance.grid, [["m1", None, "desc-1"], ["m2", None, "desc-2"]])
        self.assertEqual(instance.data_row_texts, ["m1 | null | desc-1", "m2 | null | desc-2"])
        self.assertEqual(instance.data_grid, [["m1", None, "desc-1"], ["m2", None, "desc-2"]])
        self.assertEqual(sorted({cell["row"] for cell in instance.cells if cell.get("text")}), [1, 2])

    def test_postprocess_refresh_rebuilds_semantic_and_raw_rows(self) -> None:
        table = {
            "raw_grid": [
                ["m1", None, "desc-1"],
                [None, None, None],
                ["m2", None, "desc-2"],
            ],
            "grid": [
                ["m1", None, "desc-1"],
                [None, None, None],
                ["m2", None, "desc-2"],
            ],
            "row_count": 3,
        }

        _refresh_row_texts_from_grid(table)

        self.assertEqual(table["raw_row_count"], 3)
        self.assertEqual(table["row_count"], 2)
        self.assertEqual(table["data_row_count"], 2)
        self.assertEqual(table["display_row_count"], 2)
        self.assertEqual(table["structural_empty_rows"], [2])
        self.assertEqual(table["raw_row_texts"][1], "null | null | null")
        self.assertEqual(table["row_texts"], ["m1 | null | desc-1", "m2 | null | desc-2"])
        self.assertEqual(table["grid"], [["m1", None, "desc-1"], ["m2", None, "desc-2"]])
        self.assertEqual(table["data_row_texts"], ["m1 | null | desc-1", "m2 | null | desc-2"])
        self.assertEqual(table["display_row_texts"], ["m1 | null | desc-1", "m2 | null | desc-2"])

    def test_ast_export_keeps_semantic_and_raw_row_views(self) -> None:
        instance = assemble_table_instance(
            normalized=_normalized_table_with_empty_middle_row(),
            table_id="tbl_001",
            parent_header=[
                {"col": 1, "text": "Folder"},
                {"col": 2, "text": "File"},
                {"col": 3, "text": "Description"},
            ],
            assessment=SimpleNamespace(is_continuation=True),
        )

        continuum_result = SimpleNamespace(
            continuity=SimpleNamespace(
                is_continuation=True,
                parent_table_id="tbl_000",
                inherited_fields=[],
                similarity_score=0.95,
            ),
            nested=SimpleNamespace(
                has_nested=True,
                nesting_type="suspected_subtable",
                nested_regions=[
                    {
                        "row": 1,
                        "col": 1,
                        "rowspan": 2,
                        "colspan": 2,
                        "type": "suspected_nested_subtable",
                        "confidence": 0.86,
                        "signals": ["merged_rows_and_cols", "line_internal_alignment"],
                    }
                ],
            ),
            confidence=SimpleNamespace(
                overall_confidence=0.9,
                risk_flags=["suspected_nested_structure"],
                review_required=True,
                review_reasons=["疑似存在单元格内嵌套表格"],
            ),
        )

        ast = build_logical_ast(instance, continuum_result)
        payload = ast.to_dict()

        self.assertEqual(payload["row_count"], 2)
        self.assertEqual(payload["data_row_count"], 2)
        self.assertEqual(payload["display_row_count"], 2)
        self.assertEqual(payload["raw_row_count"], 3)
        self.assertEqual(payload["row_texts"], ["m1 | null | desc-1", "m2 | null | desc-2"])
        self.assertEqual(payload["data_row_texts"], ["m1 | null | desc-1", "m2 | null | desc-2"])
        self.assertEqual(payload["display_row_texts"], ["m1 | null | desc-1", "m2 | null | desc-2"])
        self.assertEqual(payload["raw_row_texts"][1], "null | null | null")
        self.assertEqual(payload["structural_empty_rows"], [2])
        self.assertEqual(payload["raw_grid"][1], [None, None, None])
        self.assertEqual(payload["grid"], [["m1", None, "desc-1"], ["m2", None, "desc-2"]])
        self.assertEqual(payload["data_grid"], [["m1", None, "desc-1"], ["m2", None, "desc-2"]])
        self.assertEqual(payload["display_grid"], [["m1", None, "desc-1"], ["m2", None, "desc-2"]])
        self.assertEqual(payload["risk_flags"], ["suspected_nested_structure"])
        self.assertTrue(payload["review_required"])
        self.assertEqual(payload["review_reasons"], ["疑似存在单元格内嵌套表格"])
        self.assertEqual(payload["nested_structure"]["nesting_type"], "suspected_subtable")
        self.assertEqual(payload["nested_structure"]["region_count"], 1)
        self.assertEqual(payload["nested_structure"]["regions"][0]["type"], "suspected_nested_subtable")


if __name__ == "__main__":
    unittest.main()
