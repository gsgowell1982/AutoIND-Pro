from __future__ import annotations

import os
from pathlib import Path
import unittest

from parsers.pdf.table_contracts import summarize_document_tables
from parsers.pdf_parser import parse_pdf


def _sample_root() -> Path:
    override = os.environ.get("IND_PDF_SAMPLE_REGRESSION_ROOT", "").strip()
    if override:
        return Path(override)
    return Path(r"D:\AutoIND-Pro")


SAMPLE_NAMES = [
    "eCTD\u6280\u672f\u89c4\u8303.pdf",
    "eCTD\u5b9e\u65bd\u6307\u5357.pdf",
    "eCTD\u9a8c\u8bc1\u6807\u51c6.pdf",
    "test-ind.pdf",
    "A-tst.pdf",
    "2-column-tst.pdf",
    "\u9644\u4ef61-4\uff1aCTD\u6a21\u5757\u4e00\u6587\u4ef6\u7ec4\u7ec7\u7ed3\u6784.pdf",
]


class PdfSampleBatchRegressionTests(unittest.TestCase):
    maxDiff = None

    def test_representative_pdf_samples_satisfy_parser_contract_invariants(self) -> None:
        sample_root = _sample_root()
        samples = [sample_root / name for name in SAMPLE_NAMES]
        existing_samples = [sample for sample in samples if sample.exists()]
        if not existing_samples:
            raise unittest.SkipTest(
                "No PDF sample regression files found. Set IND_PDF_SAMPLE_REGRESSION_ROOT."
            )

        missing = [sample.name for sample in samples if not sample.exists()]
        self.assertEqual(missing, [], msg=f"missing representative PDF samples: {missing}")

        for sample_path in existing_samples:
            with self.subTest(pdf=sample_path.name):
                result = parse_pdf(sample_path)
                metadata = result.get("metadata", {})
                pages = list(result.get("pages") or [])
                table_asts = list(result.get("table_asts") or [])
                toc_blocks = list(result.get("toc_blocks") or [])
                image_blocks = list(result.get("image_blocks") or [])

                self.assertGreater(int(metadata.get("page_count", 0) or 0), 0)
                self.assertEqual(int(metadata.get("page_count", 0) or 0), len(pages))
                self.assertEqual(int(metadata.get("table_count", 0) or 0), len(table_asts))
                self.assertEqual(int(metadata.get("toc_count", 0) or 0), len(toc_blocks))
                self.assertEqual(int(metadata.get("image_count", 0) or 0), len(image_blocks))

                table_ids = [str(table.get("table_id") or "") for table in table_asts]
                self.assertEqual(len(table_ids), len(set(table_ids)))

                table_contract = summarize_document_tables(result)
                self.assertEqual(table_contract["table_count"], len(table_asts))
                self.assertEqual(
                    table_contract["continued_table_count"],
                    sum(1 for table in table_asts if table.get("continued_from")),
                )
                self.assertEqual(
                    table_contract["review_required_table_count"],
                    sum(1 for table in table_asts if table.get("review_required")),
                )
                for contract_table in table_contract["tables"]:
                    self._assert_table_contract_summary(contract_table)

                for table in table_asts:
                    self._assert_table_grid_contract(table)

                for toc in toc_blocks:
                    self._assert_toc_contract(toc)

                for image in image_blocks:
                    self._assert_bbox_contract(image.get("bbox"), "image bbox")

    def _assert_table_grid_contract(self, table: dict) -> None:
        table_id = str(table.get("table_id") or "<missing>")
        col_count = int(table.get("col_count", 0) or 0)
        row_count = int(table.get("row_count", 0) or 0)
        self.assertGreater(col_count, 0, msg=f"{table_id} has no columns")
        self.assertGreaterEqual(row_count, 0, msg=f"{table_id} has invalid row_count")
        self._assert_bbox_contract(table.get("bbox"), f"{table_id} bbox")

        raw_upper_bound = max(
            col_count,
            int(table.get("logical_col_count", 0) or 0),
            int(table.get("physical_col_count", 0) or 0),
        )
        if isinstance(table.get("raw_grid"), list):
            for row_index, row in enumerate(table["raw_grid"]):
                self.assertIsInstance(row, list, msg=f"{table_id}.raw_grid[{row_index}]")
                self.assertLessEqual(
                    len(row),
                    raw_upper_bound,
                    msg=f"{table_id}.raw_grid[{row_index}] exceeds raw/logical column bound",
                )

        for grid_key in ("display_grid", "data_grid", "grid"):
            grid = table.get(grid_key)
            if not isinstance(grid, list):
                continue
            for row_index, row in enumerate(grid):
                self.assertIsInstance(row, list, msg=f"{table_id}.{grid_key}[{row_index}]")
                self.assertLessEqual(
                    len(row),
                    col_count,
                    msg=f"{table_id}.{grid_key}[{row_index}] exceeds col_count",
                )

        header_cols = [
            int(header_cell.get("col", 0) or 0)
            for header_cell in table.get("header") or []
        ]
        if header_cols:
            if min(header_cols) == 0:
                self.assertLess(
                    max(header_cols),
                    col_count,
                    msg=f"{table_id} zero-based header col exceeds col_count",
                )
            else:
                self.assertGreaterEqual(
                    min(header_cols),
                    1,
                    msg=f"{table_id} one-based header col is below 1",
                )
                self.assertLessEqual(
                    max(header_cols),
                    col_count,
                    msg=f"{table_id} one-based header col exceeds col_count",
                )

    def _assert_table_contract_summary(self, table: dict) -> None:
        table_id = str(table.get("table_id") or "<missing>")
        self.assertTrue(table_id.startswith("tbl_"), msg=f"bad table id in contract: {table_id}")
        self.assertGreater(int(table.get("page", 0) or 0), 0, msg=f"{table_id} page missing")
        self.assertGreater(int(table.get("col_count", 0) or 0), 0, msg=f"{table_id} col_count missing")
        views = table.get("views") or {}
        fingerprints = table.get("fingerprints") or {}
        self.assertIn("raw_grid", views, msg=f"{table_id} contract missing raw_grid summary")
        self.assertIn("raw_grid", fingerprints, msg=f"{table_id} contract missing raw_grid fingerprint")
        for grid_key, view in views.items():
            row_count = int(view.get("row_count", 0) or 0)
            row_widths = list(view.get("row_widths") or [])
            self.assertEqual(
                row_count,
                len(row_widths),
                msg=f"{table_id}.{grid_key} row_count/row_widths mismatch",
            )
            max_col_count = int(view.get("max_col_count", 0) or 0)
            if row_widths:
                self.assertEqual(
                    max_col_count,
                    max(row_widths),
                    msg=f"{table_id}.{grid_key} max_col_count mismatch",
                )
            non_empty_by_col = list(view.get("non_empty_cells_by_column") or [])
            empty_by_col = list(view.get("empty_cells_by_column") or [])
            self.assertEqual(
                len(non_empty_by_col),
                max_col_count,
                msg=f"{table_id}.{grid_key} non-empty column summary width mismatch",
            )
            self.assertEqual(
                len(empty_by_col),
                max_col_count,
                msg=f"{table_id}.{grid_key} empty column summary width mismatch",
            )
            self.assertEqual(
                int(view.get("non_empty_cell_count", 0) or 0),
                sum(non_empty_by_col),
                msg=f"{table_id}.{grid_key} non-empty count mismatch",
            )

    def _assert_toc_contract(self, toc: dict) -> None:
        toc_id = str(toc.get("toc_id") or "<missing>")
        self._assert_bbox_contract(toc.get("bbox"), f"{toc_id} bbox")
        entry_count = int(toc.get("entry_count", 0) or 0)
        entries = list(toc.get("entries") or [])
        self.assertEqual(entry_count, len(entries), msg=f"{toc_id} entry_count mismatch")
        for entry in entries:
            self.assertTrue(
                str(entry.get("text") or "").strip(),
                msg=f"{toc_id} contains empty TOC entry text",
            )

    def _assert_bbox_contract(self, bbox: object, label: str) -> None:
        self.assertIsInstance(bbox, (list, tuple), msg=f"{label} missing")
        self.assertEqual(len(bbox), 4, msg=f"{label} must have four coordinates")
        x0, y0, x1, y1 = [float(value) for value in bbox]
        self.assertLess(x0, x1, msg=f"{label} has non-positive width")
        self.assertLess(y0, y1, msg=f"{label} has non-positive height")


if __name__ == "__main__":
    unittest.main()
