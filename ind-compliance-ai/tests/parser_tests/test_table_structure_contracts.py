from __future__ import annotations

import os
from pathlib import Path
import unittest

from parsers.pdf.table_contracts import (
    diff_table_contracts,
    summarize_document_tables,
    summarize_table,
)
from parsers.pdf_parser import parse_pdf


def _resolve_ectd_technical_spec_pdf() -> Path:
    override = os.environ.get("IND_ECTD_REGRESSION_PDF", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "eCTD技术规范.pdf"


class TableStructureContractTests(unittest.TestCase):
    maxDiff = None

    def test_summarize_table_records_all_view_shapes_without_cell_text_leakage(self) -> None:
        table = {
            "table_id": "tbl_001",
            "page": 3,
            "title": "Example table",
            "col_count": 3,
            "physical_col_count": 5,
            "logical_col_count": 3,
            "raw_grid": [["A", None, "B"], ["1", None, "2"]],
            "display_grid": [["A", None, "B"], ["1", None, "2"]],
            "data_grid": [["1", None, "2"]],
            "continued_from": "tbl_000",
            "continued_to": ["tbl_002"],
            "review_required": True,
            "review_reasons": ["ambiguous boundary"],
        }

        summary = summarize_table(table)

        self.assertEqual(summary["table_id"], "tbl_001")
        self.assertEqual(summary["page"], 3)
        self.assertEqual(summary["col_count"], 3)
        self.assertEqual(summary["physical_col_count"], 5)
        self.assertEqual(summary["logical_col_count"], 3)
        self.assertEqual(summary["views"]["raw_grid"]["row_count"], 2)
        self.assertEqual(summary["views"]["raw_grid"]["max_col_count"], 3)
        self.assertEqual(summary["views"]["raw_grid"]["row_widths"], [3, 3])
        self.assertEqual(summary["views"]["data_grid"]["non_empty_cell_count"], 2)
        self.assertEqual(summary["continued_from"], "tbl_000")
        self.assertEqual(summary["continued_to"], ["tbl_002"])
        self.assertTrue(summary["review_required"])
        self.assertEqual(summary["review_reasons"], ["ambiguous boundary"])
        self.assertNotIn("Example table", str(summary.get("fingerprints", {})))
        self.assertNotIn("ambiguous boundary", str(summary.get("fingerprints", {})))

    def test_diff_table_contracts_reports_shape_and_fingerprint_drift(self) -> None:
        before = {
            "table_count": 1,
            "tables": [
                summarize_table(
                    {
                        "table_id": "tbl_001",
                        "page": 1,
                        "col_count": 3,
                        "raw_grid": [["A", "B", "C"]],
                        "display_grid": [["A", "B", "C"]],
                        "data_grid": [["1", "2", "3"]],
                    }
                )
            ],
        }
        after = {
            "table_count": 1,
            "tables": [
                summarize_table(
                    {
                        "table_id": "tbl_001",
                        "page": 1,
                        "col_count": 2,
                        "raw_grid": [["A", "B"]],
                        "display_grid": [["A", "B"]],
                        "data_grid": [["1", "2"]],
                    }
                )
            ],
        }

        diff = diff_table_contracts(before, after)

        self.assertFalse(diff["matches"])
        self.assertEqual(diff["table_count_before"], 1)
        self.assertEqual(diff["table_count_after"], 1)
        self.assertEqual(diff["added_tables"], [])
        self.assertEqual(diff["removed_tables"], [])
        self.assertEqual(diff["changed_tables"][0]["table_id"], "tbl_001")
        self.assertIn("col_count", diff["changed_tables"][0]["changed_fields"])
        self.assertIn("views.data_grid.row_widths", diff["changed_tables"][0]["changed_fields"])
        self.assertIn("fingerprints.data_grid", diff["changed_tables"][0]["changed_fields"])

    def test_ectd_technical_spec_known_table_contracts_are_protected(self) -> None:
        sample_path = _resolve_ectd_technical_spec_pdf()
        if not sample_path.exists():
            raise unittest.SkipTest("eCTD technical specification sample not found")

        result = parse_pdf(sample_path)
        contract = summarize_document_tables(result)
        by_id = {table["table_id"]: table for table in contract["tables"]}

        self.assertEqual(contract["table_count"], 12)
        self.assertEqual(contract["continued_table_count"], 5)
        self.assertEqual(by_id["tbl_007"]["col_count"], 3)
        self.assertEqual(by_id["tbl_007"]["continued_from"], "tbl_006")
        self.assertEqual(by_id["tbl_007"]["views"]["raw_grid"]["row_count"], 28)
        self.assertEqual(by_id["tbl_007"]["views"]["display_grid"]["row_count"], 25)
        self.assertEqual(by_id["tbl_007"]["views"]["data_grid"]["row_count"], 25)
        self.assertEqual(by_id["tbl_007"]["views"]["raw_grid"]["row_widths"][:12], [3] * 12)
        self.assertEqual(by_id["tbl_007"]["views"]["raw_grid"]["empty_cells_by_column"][1], 22)
        self.assertEqual(by_id["tbl_007"]["views"]["display_grid"]["empty_cells_by_column"][1], 19)
        self.assertEqual(by_id["tbl_007"]["views"]["data_grid"]["empty_cells_by_column"][1], 19)

        self.assertEqual(by_id["tbl_010"]["continued_to"], ["tbl_011"])
        self.assertEqual(by_id["tbl_011"]["continued_from"], "tbl_010")
        self.assertEqual(by_id["tbl_011"]["continued_to"], ["tbl_012"])
        self.assertEqual(by_id["tbl_012"]["continued_from"], "tbl_011")


if __name__ == "__main__":
    unittest.main()
