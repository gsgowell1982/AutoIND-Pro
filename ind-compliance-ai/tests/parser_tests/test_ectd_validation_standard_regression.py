from __future__ import annotations

import os
from pathlib import Path
import unittest

from parsers.pdf_parser import parse_pdf


def _resolve_ectd_validation_standard_pdf() -> Path:
    override = os.environ.get("IND_ECTD_VALIDATION_STANDARD_REGRESSION_PDF", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data" / "regulations" / "eCTD验证标准.pdf"


class EctdValidationStandardRegressionTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_path = _resolve_ectd_validation_standard_pdf()
        if not cls.sample_path.exists():
            raise unittest.SkipTest(
                "eCTD validation standard sample not found. "
                "Set IND_ECTD_VALIDATION_STANDARD_REGRESSION_PDF or place "
                "eCTD验证标准.pdf under data/regulations."
            )
        cls.result = parse_pdf(cls.sample_path)
        cls.tables = list(cls.result.get("table_asts", []) or [])

    def test_continued_rule_rows_split_sequence_number_and_description_columns(self) -> None:
        page3_table = next(
            table
            for table in self.tables
            if int(table.get("page", 0) or 0) == 3
            and any("2.9" in str(cell or "") for row in table.get("data_grid", []) for cell in row)
        )
        rows_by_number = {
            str(row[0]).strip(): row
            for row in page3_table.get("data_grid", []) or []
            if row and str(row[0] or "").strip()
        }

        self.assertEqual(
            rows_by_number["2.9"][:4],
            ["2.9", "序列文件夹要求", "序列文件夹名称必须仅包含4个数字。", "错误"],
        )
        self.assertEqual(rows_by_number["2.10"][0], "2.10")
        self.assertEqual(rows_by_number["2.10"][1], "序列编号")
        self.assertEqual(rows_by_number["3.1"][0], "3.1")
        self.assertEqual(rows_by_number["3.1"][1], "index.xml文件必须存在")

    def test_continued_section_group_rows_are_not_split_as_rule_items(self) -> None:
        page3_table = next(
            table
            for table in self.tables
            if int(table.get("page", 0) or 0) == 3
            and any("3 - ICH 骨架文件" in str(cell or "") for row in table.get("data_grid", []) for cell in row)
        )
        group_rows = [
            row
            for row in page3_table.get("data_grid", []) or []
            if row and str(row[0] or "").strip() == "3 - ICH 骨架文件"
        ]
        self.assertEqual(len(group_rows), 1)
        self.assertEqual(group_rows[0][1:], [None, None, None])

    def test_severity_explanation_table_is_not_folded_into_validation_rule_continuation(self) -> None:
        page11_tables = [
            table
            for table in self.tables
            if int(table.get("page", 0) or 0) == 11
        ]
        self.assertGreaterEqual(len(page11_tables), 2)

        continuation_table = next(
            table
            for table in page11_tables
            if any(str(row[0] or "").strip() == "6.26" for row in table.get("data_grid", []) or [])
        )
        continuation_blob = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in continuation_table.get("data_grid", []) or []
        )
        self.assertIn("6.26", continuation_blob)
        self.assertNotIn("说明:", continuation_blob)
        self.assertNotIn("错误 必须遵守的关键验证标准", continuation_blob)

        explanation_table = next(
            table
            for table in page11_tables
            if any(str(row[0] or "").strip() == "说明:" for row in table.get("display_grid", []) or [])
        )
        explanation_blob = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in explanation_table.get("display_grid", []) or []
        )
        self.assertIn("说明:", explanation_blob)
        self.assertIn("错误 | 必须遵守的关键验证标准", explanation_blob)
        self.assertFalse(explanation_table.get("continued_from"))

    def test_section_group_rows_do_not_duplicate_title_into_next_column(self) -> None:
        section_rows: dict[str, list[object | None]] = {}
        raw_section_rows: dict[str, list[object | None]] = {}
        for table in self.tables:
            for row in table.get("raw_grid", []) or []:
                first = str(row[0] or "").strip() if row else ""
                if first.startswith("4.1 -") or first.startswith("4.2 -"):
                    raw_section_rows[first[:3]] = row
            for row in table.get("display_grid", []) or []:
                first = str(row[0] or "").strip() if row else ""
                if first.startswith("4.1 -") or first.startswith("4.2 -"):
                    section_rows[first[:3]] = row

        self.assertIn("4.1", raw_section_rows)
        self.assertIn("4.2", raw_section_rows)
        self.assertEqual(raw_section_rows["4.1"][1:], [None, None, None])
        self.assertEqual(raw_section_rows["4.2"][1:], [None, None, None])
        self.assertIn("4.1", section_rows)
        self.assertIn("4.2", section_rows)
        self.assertEqual(section_rows["4.1"][1:], [None, None, None])
        self.assertEqual(section_rows["4.2"][1:], [None, None, None])

    def test_section_group_rows_expose_full_width_merge_semantics(self) -> None:
        table = next(
            table
            for table in self.tables
            if any(str(row[0] or "").strip().startswith("4.1 -") for row in table.get("display_grid", []) or [])
        )
        display_rows = table.get("display_grid", []) or []
        row_index = next(
            index
            for index, row in enumerate(display_rows, start=1)
            if row and str(row[0] or "").strip().startswith("4.1 -")
        )
        row = display_rows[row_index - 1]
        merged_rows = table.get("merged_rows") or []

        matching = [
            item
            for item in merged_rows
            if int(item.get("row", 0) or 0) == row_index
            and str(item.get("kind") or "") == "section_group"
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].get("text"), row[0])
        self.assertEqual(matching[0].get("colspan"), len(row))

    def test_internal_explanation_segment_projects_to_own_three_column_schema(self) -> None:
        explanation_table = next(
            table
            for table in self.tables
            if int(table.get("page", 0) or 0) == 11
            and not table.get("continued_from")
            and any(str(row[0] or "").strip().endswith(":") for row in table.get("display_grid", []) or [])
        )

        self.assertTrue(all(len(row) == 3 for row in explanation_table.get("display_grid", []) or []))
        self.assertTrue(all(len(row) == 3 for row in explanation_table.get("data_grid", []) or []))
        first_row = explanation_table.get("display_grid", [])[0]
        self.assertTrue(str(first_row[0] or "").strip().endswith(":"))
        self.assertEqual(first_row[1:], [None, None])

        merged_rows = explanation_table.get("merged_rows") or []
        matching = [
            item
            for item in merged_rows
            if int(item.get("row", 0) or 0) == 1
            and str(item.get("kind") or "") == "table_note_title"
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].get("text"), first_row[0])
        self.assertEqual(matching[0].get("colspan"), len(first_row))


if __name__ == "__main__":
    unittest.main()
