from __future__ import annotations

import unittest

from parsers.pdf import postprocess


class IndLateProjectionLosslessnessTests(unittest.TestCase):
    def test_study_condition_result_matrix_infers_non_three_leaf_group_spans(self) -> None:
        def cell(row: int, col: int, text: str) -> dict:
            return {
                "row": row,
                "col": col,
                "text": text,
                "bbox": [float(col * 60), float(row * 20), float(col * 60 + 50), float(row * 20 + 12)],
            }

        template = {
            "structure_template_id": "synthetic_study_context",
            "template_kind": "study_metadata",
            "template_profile": "populated_study_metadata",
            "page": 1,
            "bbox": [0.0, 0.0, 360.0, 30.0],
            "fields": [
                {"label": "Species", "value": "Rat Dog"},
            ],
            "row_texts": ["Species Rat Dog"],
        }
        table = {
            "table_id": "synthetic_two_leaf_result_matrix",
            "semantic_role": "business_table",
            "page": 1,
            "bbox": [0.0, 40.0, 360.0, 140.0],
            "display_grid": [
                ["Time", "Urine", "Total", "Urine", "Total"],
                ["0-24 h", "11", "22", "33", "44"],
                ["24-48 h", "12", "23", "34", "45"],
            ],
            "raw_grid": [
                ["Time", "Urine", "Total", "Urine", "Total"],
                ["0-24 h", "11", "22", "33", "44"],
                ["24-48 h", "12", "23", "34", "45"],
            ],
            "cells": [
                cell(0, 0, "Time"),
                cell(0, 1, "Urine"),
                cell(0, 2, "Total"),
                cell(0, 3, "Urine"),
                cell(0, 4, "Total"),
                cell(1, 0, "0-24 h"),
                cell(1, 1, "11"),
                cell(1, 2, "22"),
                cell(1, 3, "33"),
                cell(1, 4, "44"),
                cell(2, 0, "24-48 h"),
                cell(2, 1, "12"),
                cell(2, 2, "23"),
                cell(2, 3, "34"),
                cell(2, 4, "45"),
            ],
        }

        postprocess._bind_study_context_result_matrices(
            structure_templates=[template],
            table_nodes=[table],
        )

        binding = table.get("semantic_context_binding") or {}
        self.assertEqual(binding.get("group_count"), 2)
        self.assertEqual(binding.get("leaf_count_per_group"), 2)
        self.assertEqual(binding.get("span_inference_source"), "repeated_leaf_header_period")
        self.assertEqual(
            [
                (group.get("label"), group.get("start_leaf_col"), group.get("end_leaf_col"), group.get("colspan"))
                for group in binding.get("header_group_rows", [])
            ],
            [
                ("Rat", 1, 2, 2),
                ("Dog", 3, 4, 2),
            ],
        )

        projection = (
            table.get("semantic_projection_v2", {}) or {}
        ).get("study_condition_grouped_result_matrix_projection", {})
        self.assertEqual(projection.get("leaf_count_per_group"), 2)
        self.assertEqual(projection.get("span_inference_source"), "repeated_leaf_header_period")

    def test_semantic_projection_common_keeps_observed_grids_as_evidence_layer(self) -> None:
        table = {
            "table_id": "synthetic_late_projection",
            "display_grid": [
                ["Dose", "Column 2", "0", "10"],
                ["Result", "", "-", "+"],
                ["Dunnett", "*-p<0.05", "", ""],
            ],
            "raw_grid": [
                ["Dose", "Column 2", "0", "10"],
                ["Result", "", "-", "+"],
                ["Dunnett", "*-p<0.05", "", ""],
            ],
            "data_grid": [
                ["Result", "", "-", "+"],
                ["Dunnett", "*-p<0.05", "", ""],
            ],
        }
        semantic_grid = [
            ["Dose", "0", "10"],
            ["Result", "-", "+"],
        ]

        postprocess._apply_semantic_grid_projection_common(
            table,
            semantic_grid=semantic_grid,
            source="dose_response_result_panel_projection",
        )

        self.assertEqual(
            table["display_grid"],
            [
                ["Dose", "Column 2", "0", "10"],
                ["Result", "", "-", "+"],
                ["Dunnett", "*-p<0.05", "", ""],
            ],
        )
        self.assertEqual(
            table["data_grid"],
            [
                ["Result", "", "-", "+"],
                ["Dunnett", "*-p<0.05", "", ""],
            ],
        )
        self.assertEqual(table["semantic_grid"], semantic_grid)
        self.assertEqual(table["semantic_display_grid"], semantic_grid)
        self.assertEqual(
            table["evidence_grid_before_semantic_projection"],
            [
                ["Dose", "Column 2", "0", "10"],
                ["Result", "", "-", "+"],
                ["Dunnett", "*-p<0.05", "", ""],
            ],
        )

    def test_dose_response_text_projection_accepts_schema_derived_leaf_count(self) -> None:
        projected = postprocess._dose_response_project_text_row(
            "Body weight 1 2 3 4 5 6",
            expected_col_count=7,
        )

        self.assertEqual(
            projected,
            ["Body weight", "1", "2", "3", "4", "5", "6"],
        )

    def test_dose_response_grid_projection_accepts_non_5_or_9_header_width(self) -> None:
        table = {
            "display_grid": [
                ["Dose", "0 M", "0 F", "10 M", "10 F", "30 M", "30 F"],
                ["Body weight", "1", "2", "3", "4", "5", "6"],
            ],
            "raw_grid": [
                ["Dose", "0 M", "0 F", "10 M", "10 F", "30 M", "30 F"],
                ["Body weight", "1", "2", "3", "4", "5", "6"],
            ],
        }
        projected = postprocess._dose_response_result_panel_grid_from_words(
            table,
            page_words=[],
            inherited_header=["Dose", "0 M", "0 F", "10 M", "10 F", "30 M", "30 F"],
        )

        self.assertEqual(
            projected,
            [
                ["Dose", "0 M", "0 F", "10 M", "10 F", "30 M", "30 F"],
                ["Body weight", "1", "2", "3", "4", "5", "6"],
            ],
        )

    def test_dose_response_schema_helpers_support_dynamic_mf_leaf_groups(self) -> None:
        header = ["Dose", "0 M", "0 F", "10 M", "10 F", "30 M", "30 F"]

        self.assertTrue(postprocess._dose_response_header_is_supported(header))
        self.assertTrue(postprocess._dose_response_header_has_sex_leaf_columns(header))
        self.assertEqual(
            postprocess._dose_response_panel_logical_header_spans(header),
            [
                {
                    "row": 0,
                    "col": 0,
                    "rowspan": 2,
                    "colspan": 1,
                    "text": "Dose",
                    "source": "dose_response_multilevel_header",
                },
                {
                    "row": 0,
                    "col": 1,
                    "rowspan": 1,
                    "colspan": 2,
                    "text": "0",
                    "source": "dose_response_multilevel_header",
                },
                {
                    "row": 0,
                    "col": 3,
                    "rowspan": 1,
                    "colspan": 2,
                    "text": "10",
                    "source": "dose_response_multilevel_header",
                },
                {
                    "row": 0,
                    "col": 5,
                    "rowspan": 1,
                    "colspan": 2,
                    "text": "30",
                    "source": "dose_response_multilevel_header",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
