from __future__ import annotations

import unittest

from parsers.pdf.table_modules.postprocess import (
    project_table_header_grammar,
    project_rowspan_body_groups,
)


class TableHeaderGrammarTests(unittest.TestCase):
    def test_projects_generic_spanning_headers_without_slash_or_pdf_specific_text(self) -> None:
        table = {
            "col_count": 5,
            "header_row_index": 0,
            "data_start_row": 2,
            "raw_grid": [
                ["Group A", None, None, "Group B", None],
                ["A1", "A2", "A3", "B1", "B2"],
                ["10", "20", "30", "low", "high"],
                ["11", "21", "31", "medium", "higher"],
            ],
            "grid": [
                ["Group A", None, None, "Group B", None],
                ["A1", "A2", "A3", "B1", "B2"],
                ["10", "20", "30", "low", "high"],
                ["11", "21", "31", "medium", "higher"],
            ],
        }

        changed = project_table_header_grammar(table)

        self.assertTrue(changed)
        self.assertEqual(
            [cell["text"] for cell in table.get("header", [])],
            ["A1", "A2", "A3", "B1", "B2"],
        )
        self.assertEqual(table.get("display_grid", [])[0], ["Group A", None, None, "Group B", None])
        self.assertEqual(table.get("display_grid", [])[1], ["A1", "A2", "A3", "B1", "B2"])
        self.assertEqual(table.get("data_grid", [])[0], ["10", "20", "30", "low", "high"])

        groups = table.get("header_column_groups", [])
        self.assertTrue(
            any(
                group.get("text") == "Group A"
                and group.get("start_col") == 0
                and group.get("end_col") == 2
                and group.get("colspan") == 3
                and group.get("source") == "table_header_grammar"
                for group in groups
            ),
            msg=groups,
        )
        self.assertTrue(
            any(
                group.get("text") == "Group B"
                and group.get("start_col") == 3
                and group.get("end_col") == 4
                and group.get("colspan") == 2
                and group.get("source") == "table_header_grammar"
                for group in groups
            ),
            msg=groups,
        )

    def test_keeps_slash_adjacent_leaf_projection_as_part_of_same_header_grammar(self) -> None:
        table = {
            "col_count": 5,
            "header_row_index": 0,
            "data_start_row": 2,
            "raw_grid": [
                ["Dose(mg/kg)/Route", "Column 2", "Percent of dose", "Column 4", "Column 5"],
                [None, "Route", "Urine*", "Feces", "Total+"],
                ["2.8", "i.v.", "88.1", "5.5", "93.6"],
                ["8.8", "p.o.", "89.4", "6.9", "95.3"],
            ],
            "grid": [
                ["Dose(mg/kg)/Route", "Column 2", "Percent of dose", "Column 4", "Column 5"],
                [None, "Route", "Urine*", "Feces", "Total+"],
                ["2.8", "i.v.", "88.1", "5.5", "93.6"],
                ["8.8", "p.o.", "89.4", "6.9", "95.3"],
            ],
        }

        changed = project_table_header_grammar(table)

        self.assertTrue(changed)
        self.assertEqual(
            [cell["text"] for cell in table.get("header", [])],
            ["Dose(mg/kg)", "Route", "Urine*", "Feces", "Total+"],
        )
        groups = table.get("header_column_groups", [])
        self.assertTrue(
            any(
                group.get("text") == "Percent of dose"
                and group.get("start_col") == 2
                and group.get("end_col") == 4
                and group.get("source") == "table_header_grammar"
                for group in groups
            ),
            msg=groups,
        )

    def test_projects_rowspan_stub_and_fragmented_unit_group_headers(self) -> None:
        table = {
            "col_count": 6,
            "header_row_index": 0,
            "data_start_row": 2,
            "raw_grid": [
                ["Tissue", None, None, "Concentration(ng", "equiv*", "g) 72 h"],
                [None, "1 h", "6 h", "24 h", "48 h", None],
                ["Blood", "105", "96.6", "2.34", "2.34", "3.65"],
                ["Plasma", "142", "175", "3.12", "ND", "ND"],
            ],
            "grid": [
                ["Tissue", None, None, "Concentration(ng", "equiv*", "g) 72 h"],
                [None, "1 h", "6 h", "24 h", "48 h", None],
                ["Blood", "105", "96.6", "2.34", "2.34", "3.65"],
                ["Plasma", "142", "175", "3.12", "ND", "ND"],
            ],
        }

        changed = project_table_header_grammar(table)

        self.assertTrue(changed)
        self.assertEqual(
            [cell["text"] for cell in table.get("header", [])],
            ["Tissue", "1 h", "6 h", "24 h", "48 h", "72 h"],
        )
        self.assertEqual(table.get("display_grid", [])[0], ["Tissue", "Concentration(ng equiv*/g)", None, None, None, None])
        self.assertEqual(table.get("display_grid", [])[1], [None, "1 h", "6 h", "24 h", "48 h", "72 h"])

        groups = table.get("header_column_groups", [])
        self.assertTrue(
            any(
                group.get("text") == "Concentration(ng equiv*/g)"
                and group.get("start_col") == 1
                and group.get("end_col") == 5
                and group.get("colspan") == 5
                for group in groups
            ),
            msg=groups,
        )
        row_groups = table.get("header_row_groups", [])
        self.assertTrue(
            any(
                group.get("text") == "Tissue"
                and group.get("col") == 0
                and group.get("start_row") == 0
                and group.get("end_row") == 1
                and group.get("rowspan") == 2
                for group in row_groups
            ),
            msg=row_groups,
        )

    def test_projects_slash_header_split_across_two_visual_rows(self) -> None:
        table = {
            "col_count": 6,
            "header_row_index": 0,
            "data_start_row": 2,
            "raw_grid": [
                ["Dose(mg/kg)/", None, "Percent of dose", None, None, None],
                ["Route", None, "Urine", "Feces", "Bile", "Total"],
                ["1.75", "i.v.", "61.3", "30.3", "-", "95.2"],
                ["1.75", "p.o.", "57.4", "37.0", "-", "95.2"],
            ],
            "grid": [
                ["Dose(mg/kg)/", None, "Percent of dose", None, None, None],
                ["Route", None, "Urine", "Feces", "Bile", "Total"],
                ["1.75", "i.v.", "61.3", "30.3", "-", "95.2"],
                ["1.75", "p.o.", "57.4", "37.0", "-", "95.2"],
            ],
        }

        changed = project_table_header_grammar(table)

        self.assertTrue(changed)
        self.assertEqual(
            [cell["text"] for cell in table.get("header", [])],
            ["Dose(mg/kg)", "Route", "Urine", "Feces", "Bile", "Total"],
        )
        self.assertEqual(table.get("display_grid", [])[0], ["Dose(mg/kg)", "Route", "Percent of dose", None, None, None])
        self.assertEqual(table.get("display_grid", [])[1], [None, None, "Urine", "Feces", "Bile", "Total"])
        groups = table.get("header_column_groups", [])
        self.assertTrue(
            any(
                group.get("text") == "Percent of dose"
                and group.get("start_col") == 2
                and group.get("end_col") == 5
                and group.get("colspan") == 4
                for group in groups
            ),
            msg=groups,
        )

    def test_projects_group_header_with_leaf_headers_spanning_two_visual_rows(self) -> None:
        table = {
            "col_count": 5,
            "header_row_index": 0,
            "data_start_row": 2,
            "raw_grid": [
                ["Species(Form)", "Dose(mg/kg/day)", None, "System exposure", "Reference"],
                [None, None, "Cmax(ng/ml)", "AUC(ng h/ml)#", None],
                ["Human(Tablet)", "0.48$", "36.7", "557", "X"],
                ["Mouse(Solution)", "8.8", "68.9(1.9)*", "72.7(0.2)*", "Y"],
                [None, "21.9", "267(7.3)*", "207(0.5)*", None],
            ],
            "grid": [
                ["Species(Form)", "Dose(mg/kg/day)", None, "System exposure", "Reference"],
                [None, None, "Cmax(ng/ml)", "AUC(ng h/ml)#", None],
                ["Human(Tablet)", "0.48$", "36.7", "557", "X"],
                ["Mouse(Solution)", "8.8", "68.9(1.9)*", "72.7(0.2)*", "Y"],
                [None, "21.9", "267(7.3)*", "207(0.5)*", None],
            ],
        }

        changed = project_table_header_grammar(table)

        self.assertTrue(changed)
        self.assertEqual(
            [cell["text"] for cell in table.get("header", [])],
            ["Species(Form)", "Dose(mg/kg/day)", "Cmax(ng/ml)", "AUC(ng h/ml)#", "Reference"],
        )
        self.assertEqual(
            table.get("display_grid", [])[0],
            ["Species(Form)", "Dose(mg/kg/day)", "System exposure", None, "Reference"],
        )
        self.assertEqual(table.get("display_grid", [])[1], [None, None, "Cmax(ng/ml)", "AUC(ng h/ml)#", None])
        groups = table.get("header_column_groups", [])
        self.assertTrue(
            any(
                group.get("text") == "System exposure"
                and group.get("start_col") == 2
                and group.get("end_col") == 3
                and group.get("colspan") == 2
                for group in groups
            ),
            msg=groups,
        )
        row_groups = table.get("header_row_groups", [])
        for expected_col, expected_text in ((0, "Species(Form)"), (1, "Dose(mg/kg/day)"), (4, "Reference")):
            self.assertTrue(
                any(
                    group.get("text") == expected_text
                    and group.get("col") == expected_col
                    and group.get("start_row") == 0
                    and group.get("end_row") == 1
                    and group.get("rowspan") == 2
                    for group in row_groups
                ),
                msg=row_groups,
            )

    def test_records_sparse_body_rowspan_groups_without_collapsing_audit_grid(self) -> None:
        table = {
            "col_count": 5,
            "header": [
                {"col": 1, "text": "Species(Form)"},
                {"col": 2, "text": "Dose"},
                {"col": 3, "text": "Cmax"},
                {"col": 4, "text": "AUC"},
                {"col": 5, "text": "Reference"},
            ],
            "data_grid": [
                ["Human(Tablet)", "0.48", "36.7", "557", "X"],
                ["Mouse(Solution)", "8.8", "68.9", "72.7", "Y"],
                [None, "21.9", "267", "207", None],
                [None, "43.8", "430", "325", None],
                ["Rat(Solution)", "50", "479", "1580", "Z"],
                ["Dog(Solution)", "1.5", "5.58", "15.9", "V"],
                [None, "5", "24.8", "69.3", None],
                [None, "15", "184", "511", None],
            ],
            "grid": [
                ["Human(Tablet)", "0.48", "36.7", "557", "X"],
                ["Mouse(Solution)", "8.8", "68.9", "72.7", "Y"],
                [None, "21.9", "267", "207", None],
                [None, "43.8", "430", "325", None],
                ["Rat(Solution)", "50", "479", "1580", "Z"],
                ["Dog(Solution)", "1.5", "5.58", "15.9", "V"],
                [None, "5", "24.8", "69.3", None],
                [None, "15", "184", "511", None],
            ],
        }

        changed = project_rowspan_body_groups(table)

        self.assertTrue(changed)
        groups = table.get("row_groups", [])
        self.assertTrue(
            any(
                group.get("text") == "Mouse(Solution)"
                and group.get("col") == 0
                and group.get("start_data_row") == 2
                and group.get("end_data_row") == 4
                and group.get("rowspan") == 3
                for group in groups
            ),
            msg=groups,
        )
        self.assertTrue(
            any(
                group.get("text") == "Dog(Solution)"
                and group.get("col") == 0
                and group.get("start_data_row") == 6
                and group.get("end_data_row") == 8
                and group.get("rowspan") == 3
                for group in groups
            ),
            msg=groups,
        )
        self.assertIsNone(table.get("data_grid", [])[2][0])


if __name__ == "__main__":
    unittest.main()
